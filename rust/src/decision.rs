//! Decision models: D in the paper.
//!
//! A decision model answers two yes/no questions about a candidate call and returns the
//! probability of *yes* for each: the function question D_f (is this the right function, or is
//! making no call right?) and the argument question D_a (are the arguments present, supported
//! by the request and in the right parameters?). The argument question is not asked for a no
//! call, and neither question is asked for an unreadable candidate; such a score is `None`.
//!
//! Both decision models of the paper speak the same HTTP protocol (`POST /v1/systemone`):
//!
//! * [`Jev`], TypeSafe's hosted decision model (`jev-1.13.0`), and
//! * [`PPLXDecider`], the open-weight Perplexity Decider v1.1 27B served on your own machine.
//!
//! Only the request, the function schemas and the candidate are sent; never a reference answer.
//! To plug in another decision model, implement [`DecisionModel::ask`] (and the thresholds),
//! or point a [`SystemOne`] at any server that speaks the protocol.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

use crate::error::{DecisionModelError, Error, Result};
use crate::message::{calls_value, Call};
use crate::prompts::{ARGUMENT_QUESTION, FUNCTION_QUESTION, REPAIR_INSTRUCTION};
use crate::providers::{env_first, hostname, normalize_base_url, token_like, wait, Usage};
use crate::pyjson;
use crate::schema::{check_schema, errors_value, function_definitions, SchemaError};
use crate::transport::{default_sleep, default_transport, HttpRequest, HttpResponse, Method, SharedTransport, Sleep};

/// HTTP statuses a decision-model request retries.
pub const RETRY_STATUSES: [u16; 7] = [408, 429, 500, 502, 503, 504, 529];
const OBSERVABLE_KEYS: [&str; 5] = ["role", "content", "name", "tool_calls", "tool_call_id"];

/// What a decision model answered about one state.
#[derive(Clone, Debug, Default)]
pub struct Answer {
    /// The probability of yes for each question key (`function_correct`, `arguments_correct`).
    pub probabilities: HashMap<String, f64>,
    /// The model that answered.
    pub model: String,
    /// `input_tokens` and `output_tokens`; `null` when unknown.
    pub usage: Usage,
    /// HTTP attempts.
    pub api_calls: u32,
    /// Seconds spent waiting between retries.
    pub retry_wait_s: f64,
}

/// The judgment of one candidate: the schema check S, both questions, the verdicts and the
/// feedback record the generator reads.
#[derive(Clone, Debug)]
pub struct Judgment {
    /// V_f: the schema check's function part passed and p_f reached its threshold.
    pub function_pass: bool,
    /// V_a; `None` when the argument question was skipped (no call).
    pub arguments_pass: Option<bool>,
    /// The candidate is a no call, so the argument question was not asked.
    pub arguments_skipped: bool,
    /// p_f, or `None` when not asked (an unreadable candidate).
    pub function_probability: Option<f64>,
    /// p_a, or `None` when not asked.
    pub arguments_probability: Option<f64>,
    /// Both applicable verdicts passed.
    pub accepted: bool,
    /// The feedback record (`source`, `checks`, `failed_checks`, `schema_errors`, `instruction`).
    pub feedback: Value,
    /// The schema check's errors.
    pub schema_errors: Vec<SchemaError>,
    /// Token counts of the decision-model request.
    pub usage: Usage,
    /// The model that answered, if asked.
    pub model: Option<String>,
    /// The model that was requested.
    pub requested_model: String,
    /// Seconds spent judging, without retry waits.
    pub latency_s: f64,
    /// Seconds spent waiting between retries.
    pub retry_wait_s: f64,
    /// HTTP attempts.
    pub api_calls: u32,
}

/// A decision model: implement `ask` and `model`; `judge` builds the questions and verdicts.
pub trait DecisionModel: Send + Sync {
    /// Ask `questions` about `state` (`messages`, `tools`, `proposed_calls`) and return the
    /// probability of yes for each question key.
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> std::result::Result<Answer, DecisionModelError>;

    /// The requested model name.
    fn model(&self) -> &str;

    /// The pass mark of the function question (0.5 in the paper).
    fn function_threshold(&self) -> f64 {
        0.5
    }

    /// The pass mark of the argument question (0.5 in the paper).
    fn arguments_threshold(&self) -> f64 {
        0.5
    }

    /// The `source` field of the feedback records the generator reads.
    fn feedback_source(&self) -> &str {
        "decision_model_noul_and_deterministic_schema_validation"
    }

    /// Judge one candidate. `messages` is the conversation (objects with `role` and `content`,
    /// plus `tool_calls` or `tool_call_id` for earlier tool turns), `tools` the function
    /// definitions and `calls` the candidate (`Some(&[])` for no call, `None` if unreadable).
    fn judge(&self, messages: &[Value], tools: &[Value], calls: Option<&[Call]>) -> Result<Judgment> {
        judge_candidate(self, messages, tools, calls)
    }
}

impl<D: DecisionModel + ?Sized> DecisionModel for Box<D> {
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> std::result::Result<Answer, DecisionModelError> {
        (**self).ask(state, questions)
    }
    fn model(&self) -> &str {
        (**self).model()
    }
    fn function_threshold(&self) -> f64 {
        (**self).function_threshold()
    }
    fn arguments_threshold(&self) -> f64 {
        (**self).arguments_threshold()
    }
    fn feedback_source(&self) -> &str {
        (**self).feedback_source()
    }
    fn judge(&self, messages: &[Value], tools: &[Value], calls: Option<&[Call]>) -> Result<Judgment> {
        (**self).judge(messages, tools, calls)
    }
}

impl<D: DecisionModel + ?Sized> DecisionModel for Arc<D> {
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> std::result::Result<Answer, DecisionModelError> {
        (**self).ask(state, questions)
    }
    fn model(&self) -> &str {
        (**self).model()
    }
    fn function_threshold(&self) -> f64 {
        (**self).function_threshold()
    }
    fn arguments_threshold(&self) -> f64 {
        (**self).arguments_threshold()
    }
    fn feedback_source(&self) -> &str {
        (**self).feedback_source()
    }
    fn judge(&self, messages: &[Value], tools: &[Value], calls: Option<&[Call]>) -> Result<Judgment> {
        (**self).judge(messages, tools, calls)
    }
}

fn probability(answer: &Answer, key: &str) -> std::result::Result<f64, DecisionModelError> {
    match answer.probabilities.get(key) {
        Some(p) if p.is_finite() && (0.0..=1.0).contains(p) => Ok(*p),
        Some(_) => Err(DecisionModelError::new(
            format!("Decision model returned an invalid probability for {}.", pyjson::repr_str(key)),
            "response_invalid",
        )),
        None => Err(DecisionModelError::new(
            format!("Decision model response is missing the answer {}.", pyjson::repr_str(key)),
            "response_invalid",
        )),
    }
}

/// The default [`DecisionModel::judge`]: the schema check, both questions, the verdicts
/// V(p) = p is known and p >= threshold, and the feedback record.
pub fn judge_candidate<D: DecisionModel + ?Sized>(
    model: &D,
    messages: &[Value],
    tools: &[Value],
    calls: Option<&[Call]>,
) -> Result<Judgment> {
    let started = Instant::now();
    let local = check_schema(tools, &calls_value(calls))?;
    let arguments_skipped = calls.is_some_and(|calls| calls.is_empty());
    let (mut function_probability, mut arguments_probability) = (None, None);
    let mut usage = Usage::new();
    usage.insert("input_tokens".into(), json!(0));
    usage.insert("output_tokens".into(), json!(0));
    let (mut resolved, mut api_calls, mut waited) = (None, 0, 0.0);
    if let (true, Some(calls)) = (local.well_formed, calls) {
        // An unreadable candidate is not sent to the decision model.
        let mut observable = Vec::with_capacity(messages.len());
        for message in messages {
            let Value::Object(message) = message else {
                return Err(Error::invalid("Each message must be an object"));
            };
            let mut kept = Map::new();
            for key in OBSERVABLE_KEYS {
                if let Some(value) = message.get(key) {
                    kept.insert(key.to_string(), value.clone());
                }
            }
            observable.push(Value::Object(kept));
        }
        let state = pyjson::object(vec![
            ("messages", Value::Array(observable)),
            ("tools", Value::Array(function_definitions(tools)?)),
            ("proposed_calls", Value::Array(calls.iter().map(Call::to_value).collect())),
        ]);
        let mut questions = Map::new();
        questions.insert("function_correct".into(), FUNCTION_QUESTION.clone());
        if !arguments_skipped {
            questions.insert("arguments_correct".into(), ARGUMENT_QUESTION.clone());
        }
        let answer = model.ask(&state, &questions)?;
        function_probability = Some(probability(&answer, "function_correct")?);
        if !arguments_skipped {
            arguments_probability = Some(probability(&answer, "arguments_correct")?);
        }
        resolved = Some(answer.model);
        usage = answer.usage;
        api_calls = answer.api_calls;
        waited = answer.retry_wait_s;
    }

    // V(p) is true iff p is known and p >= threshold; S must pass too.
    let (function_threshold, arguments_threshold) = (model.function_threshold(), model.arguments_threshold());
    let function_pass = local.function_pass && function_probability.is_some_and(|p| p >= function_threshold);
    let arguments_pass = (!arguments_skipped)
        .then(|| local.arguments_pass && arguments_probability.is_some_and(|p| p >= arguments_threshold));
    let optional = |p: Option<f64>| p.map_or(Value::Null, pyjson::float);
    let checks = pyjson::object(vec![
        (
            "function_selection",
            pyjson::object(vec![
                ("pass", Value::Bool(function_pass)),
                ("probability_yes", optional(function_probability)),
                ("threshold", pyjson::float(function_threshold)),
                ("schema_pass", Value::Bool(local.function_pass)),
            ]),
        ),
        (
            "arguments",
            pyjson::object(vec![
                ("skipped", Value::Bool(arguments_skipped)),
                ("pass", arguments_pass.map_or(Value::Null, Value::Bool)),
                ("probability_yes", optional(arguments_probability)),
                ("threshold", pyjson::float(arguments_threshold)),
                ("schema_pass", Value::Bool(local.arguments_pass)),
            ]),
        ),
    ]);
    let mut failed = Vec::new();
    if !function_pass {
        failed.push(json!("function_selection"));
    }
    if !arguments_skipped && arguments_pass != Some(true) {
        failed.push(json!("arguments"));
    }
    let feedback = pyjson::object(vec![
        ("source", Value::String(model.feedback_source().to_string())),
        ("checks", checks),
        ("failed_checks", Value::Array(failed)),
        ("schema_errors", errors_value(&local.errors)),
        ("instruction", Value::String(REPAIR_INSTRUCTION.to_string())),
    ]);
    Ok(Judgment {
        function_pass,
        arguments_pass,
        arguments_skipped,
        function_probability,
        arguments_probability,
        accepted: function_pass && (arguments_skipped || arguments_pass == Some(true)),
        feedback,
        schema_errors: local.errors,
        usage,
        model: resolved,
        requested_model: model.model().to_string(),
        latency_s: (started.elapsed().as_secs_f64() - waited).max(0.0),
        retry_wait_s: waited,
        api_calls,
    })
}

// HTTP retries ----------------------------------------------------------------------------------

/// `Retry-After` as seconds or an HTTP date; otherwise 2^attempt, at most 30 seconds.
fn retry_delay(response: Option<&HttpResponse>, attempt: u32) -> f64 {
    if let Some(value) = response.and_then(|r| r.header("retry-after")) {
        let delay = value.trim().parse::<f64>().ok().or_else(|| http_date_delay(value)).unwrap_or(-1.0);
        if delay.is_finite() && delay >= 0.0 {
            return delay.min(30.0);
        }
    }
    2f64.powi(attempt as i32).min(30.0)
}

/// Seconds from now until an RFC 2822 / HTTP date such as `Wed, 21 Oct 2015 07:28:00 GMT`.
fn http_date_delay(value: &str) -> Option<f64> {
    let text = value.trim();
    let text = text.split_once(',').map_or(text, |(_, rest)| rest).trim();
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() < 4 {
        return None;
    }
    let day: i64 = parts[0].parse().ok()?;
    let months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
    let month = months.iter().position(|m| parts[1].to_ascii_lowercase().starts_with(m))? as i64 + 1;
    let mut year: i64 = parts[2].parse().ok()?;
    if year < 100 {
        year += if year > 68 { 1900 } else { 2000 };
    }
    let clock: Vec<i64> = parts[3].split(':').map(|p| p.parse().ok()).collect::<Option<_>>()?;
    let (hour, minute, second) = (clock.first().copied()?, clock.get(1).copied()?, clock.get(2).copied().unwrap_or(0));
    let offset = match parts.get(4) {
        Some(zone) if zone.starts_with(['+', '-']) && zone.len() == 5 => {
            let sign = if zone.starts_with('-') { -1 } else { 1 };
            let hours: i64 = zone[1..3].parse().ok()?;
            let minutes: i64 = zone[3..5].parse().ok()?;
            sign * (hours * 3600 + minutes * 60)
        }
        _ => 0,
    };
    // Days from the civil date (Howard Hinnant's algorithm).
    let (y, m) = if month <= 2 { (year - 1, month + 9) } else { (year, month - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let stamp = days * 86400 + hour * 3600 + minute * 60 + second - offset;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs_f64();
    Some(stamp as f64 - now)
}

// The /v1/systemone protocol ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flavor {
    Generic,
    Jev,
    Pplx,
}

/// A decision model reached over the `/v1/systemone` protocol.
///
/// [`Jev`] and [`PPLXDecider`] are configured instances of it; build one directly for another
/// server that speaks the protocol.
pub struct SystemOne {
    flavor: Flavor,
    feedback_source: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
    function_threshold: f64,
    arguments_threshold: f64,
    timeout: Duration,
    max_retries: u32,
    transport: SharedTransport,
    sleep: Sleep,
    check_provenance: bool,
    provenance: Mutex<Option<Value>>,
}

impl SystemOne {
    /// A builder for a server at `base_url` (the origin before `/v1/systemone`) and a model name.
    pub fn builder(base_url: impl Into<String>, model: impl Into<String>) -> SystemOneBuilder {
        SystemOneBuilder {
            base_url: base_url.into(),
            model: model.into(),
            common: Common::new(60, 3),
            feedback_source: None,
        }
    }

    fn display_name(&self) -> &'static str {
        match self.flavor {
            Flavor::Generic => "Decision model",
            Flavor::Jev => "Jev",
            Flavor::Pplx => "PPLX Decider",
        }
    }

    /// The origin requests go to.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Some(key) = &self.api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        headers.push(("Content-Type".to_string(), "application/json".to_string()));
        headers
    }

    fn request_id(&self, response: &HttpResponse) -> Option<String> {
        for header in ["x-request-id", "request-id", "apim-request-id"] {
            if let Some(value) = response.header(header) {
                let leaks = self.api_key.as_deref().is_some_and(|key| value.contains(key));
                if token_like(value, 128, &['.', '_', ':', '-']) && !leaks {
                    return Some(value.to_string());
                }
            }
        }
        None
    }

    fn http_error(&self, response: &HttpResponse, attempts: u32) -> DecisionModelError {
        let mut error = DecisionModelError::new(
            format!("{} HTTP {}; no judgment recorded.", self.display_name(), response.status),
            "http_error",
        );
        error.status_code = Some(response.status);
        error.request_id = self.request_id(response);
        error.attempts = Some(attempts);
        if self.flavor == Flavor::Pplx && response.status == 422 {
            if let Some(Value::Object(body)) = response.parse_json() {
                let detail = match body.get("detail") {
                    Some(Value::String(text)) => text.clone(),
                    Some(other) => pyjson::repr(other),
                    None => String::new(),
                };
                if detail.contains("8192-token limit") {
                    error.code = "context_limit_exceeded".to_string();
                }
            }
        }
        error
    }

    /// POST with bounded retries. Returns the body, the attempts and the time spent waiting.
    fn post(&self, payload: &Value) -> std::result::Result<(Map<String, Value>, u32, f64), DecisionModelError> {
        let name = self.display_name();
        let request = HttpRequest {
            method: Method::Post,
            url: format!("{}/v1/systemone", self.base_url),
            headers: self.headers(),
            body: serde_json::to_vec(payload).unwrap_or_default(),
            timeout: self.timeout,
        };
        let mut waited = 0.0;
        for attempt in 0..=self.max_retries {
            let response = match self.transport.send(&request) {
                Ok(response) => response,
                Err(_) if attempt < self.max_retries => {
                    waited += wait(&self.sleep, retry_delay(None, attempt));
                    continue;
                }
                Err(_) => {
                    let mut error = DecisionModelError::new(
                        format!("{name} connection failed after {} attempts.", attempt + 1),
                        "connection_failed",
                    );
                    error.attempts = Some(attempt + 1);
                    return Err(error);
                }
            };
            if RETRY_STATUSES.contains(&response.status) && attempt < self.max_retries {
                waited += wait(&self.sleep, retry_delay(Some(&response), attempt));
                continue;
            }
            if !response.is_success() {
                return Err(self.http_error(&response, attempt + 1));
            }
            let body = match response.parse_json() {
                Some(Value::Object(body)) => body,
                Some(_) => {
                    return Err(DecisionModelError::new(
                        format!("{name} response must be a JSON object."),
                        "response_invalid",
                    ))
                }
                None => {
                    return Err(DecisionModelError::new(
                        format!("{name} response was not valid JSON."),
                        "response_invalid",
                    ))
                }
            };
            return Ok((self.checked(body)?, attempt + 1, waited));
        }
        unreachable!("the loop returns on its last attempt")
    }

    /// The PPLX Decider's response must come from the pinned checkpoint; missing usage is unknown.
    fn checked(&self, mut body: Map<String, Value>) -> std::result::Result<Map<String, Value>, DecisionModelError> {
        if self.flavor != Flavor::Pplx {
            return Ok(body);
        }
        let model = body.get("model").and_then(Value::as_str);
        if model != Some(PPLXDecider::MODEL_ID) && model != Some(PPLXDecider::SERVER_MODEL) {
            return Err(DecisionModelError::new("PPLX Decider response has an unexpected model ID.", "model_mismatch"));
        }
        if body.get("revision").is_some_and(|r| r.as_str() != Some(PPLXDecider::MODEL_REVISION)) {
            return Err(DecisionModelError::new(
                "PPLX Decider response has an unexpected model revision.",
                "model_mismatch",
            ));
        }
        body.insert("model".into(), json!(PPLXDecider::MODEL_ID));
        if !body.contains_key("usage") {
            // Missing usage is unknown, never zero.
            body.insert("usage".into(), json!({"input_tokens": null, "output_tokens": null}));
        }
        Ok(body)
    }

    /// The PPLX Decider server's reported checkpoint, verified against the pinned one.
    fn read_provenance(&self) -> std::result::Result<Value, DecisionModelError> {
        let request = HttpRequest {
            method: Method::Get,
            url: format!("{}/provenance", self.base_url),
            headers: self.headers(),
            body: Vec::new(),
            timeout: self.timeout,
        };
        let response = self.transport.send(&request).map_err(|_| {
            DecisionModelError::new("PPLX Decider provenance could not be reached.", "connection_failed")
        })?;
        if response.status != 200 {
            let mut error =
                DecisionModelError::new("PPLX Decider provenance is unavailable.", "provenance_unavailable");
            error.status_code = Some(response.status);
            return Err(error);
        }
        let body = match response.parse_json() {
            Some(Value::Object(body)) => body,
            Some(_) => {
                return Err(DecisionModelError::new("PPLX Decider provenance must be an object.", "provenance_invalid"))
            }
            None => {
                return Err(DecisionModelError::new(
                    "PPLX Decider provenance was not valid JSON.",
                    "provenance_invalid",
                ))
            }
        };
        let expected = json!({"model_id": PPLXDecider::MODEL_ID, "revision": PPLXDecider::MODEL_REVISION,
                              "native_attention": "noncausal_full_attention", "max_input_tokens": PPLXDecider::MAX_INPUT_TOKENS});
        let matches = expected.as_object().is_some_and(|expected| {
            expected.iter().all(|(key, value)| body.get(key).is_some_and(|found| pyjson::equal(found, value)))
        });
        let temperature = body.get("temperature").and_then(|t| if t.is_number() { pyjson::as_number(t) } else { None });
        let calibrated = temperature.is_some_and(|t| (t - PPLXDecider::CALIBRATION_TEMPERATURE).abs() <= 1e-12);
        if !matches || !calibrated {
            return Err(DecisionModelError::new(
                "The server does not run the pinned PPLX Decider checkpoint.",
                "provenance_mismatch",
            ));
        }
        let mut result = expected;
        result["temperature"] = pyjson::float(temperature.unwrap_or_default());
        Ok(result)
    }

    fn ask_http(
        &self,
        state: &Value,
        questions: &Map<String, Value>,
    ) -> std::result::Result<Answer, DecisionModelError> {
        if self.flavor == Flavor::Pplx && self.check_provenance {
            let mut cached = self.provenance.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if cached.is_none() {
                *cached = Some(self.read_provenance()?);
            }
        }
        let name = self.display_name();
        let payload = pyjson::object(vec![
            ("model", json!(self.model)),
            ("state", state.clone()),
            ("questions", Value::Object(questions.clone())),
        ]);
        let (body, api_calls, waited) = self.post(&payload)?;
        let invalid = |message: String| DecisionModelError::new(message, "response_invalid");
        let answers = body.get("answers").and_then(Value::as_object);
        let mut probabilities = HashMap::new();
        for key in questions.keys() {
            let shown = pyjson::repr_str(key);
            let answer = answers.and_then(|answers| answers.get(key)).and_then(Value::as_object);
            let Some(answer) = answer.filter(|a| a.get("type").and_then(Value::as_str) == Some("noul")) else {
                return Err(invalid(format!("{name} response is missing the answer {shown}.")));
            };
            let value = answer.get("noul").filter(|v| v.is_number()).and_then(pyjson::as_number);
            match value {
                Some(p) if p.is_finite() && (0.0..=1.0).contains(&p) => {
                    probabilities.insert(key.clone(), p);
                }
                _ => return Err(invalid(format!("{name} returned an invalid probability for {shown}."))),
            }
        }
        let model = match body.get("model") {
            Some(Value::String(model)) if !model.is_empty() => model.clone(),
            _ => return Err(invalid(format!("{name} response is missing the resolved model ID."))),
        };
        let Some(Value::Object(raw)) = body.get("usage") else {
            return Err(invalid(format!("{name} response is missing token usage.")));
        };
        let mut usage = Usage::new();
        for kind in ["input_tokens", "output_tokens"] {
            let count = match raw.get(kind) {
                None | Some(Value::Null) => Value::Null,
                Some(value @ Value::Number(_)) if pyjson::is_int(value) && value.as_u64().is_some() => value.clone(),
                Some(_) => return Err(invalid(format!("{name} response contains invalid token usage."))),
            };
            usage.insert(kind.to_string(), count);
        }
        Ok(Answer { probabilities, model, usage, api_calls, retry_wait_s: waited })
    }

    fn describe(&self, f: &mut fmt::Formatter<'_>, class: &str) -> fmt::Result {
        write!(
            f,
            "{class}(model={}, base_url={}, function_threshold={}, arguments_threshold={})",
            pyjson::repr_str(&self.model),
            pyjson::repr_str(&self.base_url),
            pyjson::float_repr(self.function_threshold),
            pyjson::float_repr(self.arguments_threshold)
        )
    }
}

impl DecisionModel for SystemOne {
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> std::result::Result<Answer, DecisionModelError> {
        self.ask_http(state, questions)
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn function_threshold(&self) -> f64 {
        self.function_threshold
    }
    fn arguments_threshold(&self) -> f64 {
        self.arguments_threshold
    }
    fn feedback_source(&self) -> &str {
        &self.feedback_source
    }
}

impl fmt::Debug for SystemOne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.describe(f, "SystemOne")
    }
}

// Builders ------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Common {
    api_key: Option<String>,
    threshold: f64,
    function_threshold: Option<f64>,
    arguments_threshold: Option<f64>,
    timeout: Duration,
    max_retries: u32,
    transport: Option<SharedTransport>,
    sleep: Option<Sleep>,
}

impl Common {
    fn new(timeout_s: u64, max_retries: u32) -> Self {
        Common {
            api_key: None,
            threshold: 0.5,
            function_threshold: None,
            arguments_threshold: None,
            timeout: Duration::from_secs(timeout_s),
            max_retries,
            transport: None,
            sleep: None,
        }
    }

    #[allow(clippy::too_many_arguments)] // a private constructor shared by three builders
    fn build(
        self,
        flavor: Flavor,
        class: &str,
        base_url: &str,
        model: String,
        feedback_source: String,
        use_env_proxy: bool,
        check_provenance: bool,
    ) -> Result<SystemOne> {
        let threshold = |value: f64| {
            if value.is_finite() && (0.0..=1.0).contains(&value) {
                Ok(value)
            } else {
                Err(Error::invalid("Thresholds must be numbers between 0 and 1."))
            }
        };
        let function_threshold = threshold(self.function_threshold.unwrap_or(self.threshold))?;
        let arguments_threshold = threshold(self.arguments_threshold.unwrap_or(self.threshold))?;
        if self.timeout.is_zero() {
            return Err(Error::invalid("timeout must be a positive number of seconds."));
        }
        if let Some(key) = &self.api_key {
            if key.trim().is_empty() || key.contains(['\n', '\r']) {
                return Err(Error::invalid("API keys must be nonempty single-line strings."));
            }
        }
        if model.trim().is_empty() {
            return Err(Error::invalid(format!("{class} needs a model name.")));
        }
        let Some(base_url) = normalize_base_url(base_url) else {
            return Err(Error::invalid(format!(
                "{class} needs an http(s) base_url without credentials, query or fragment."
            )));
        };
        let transport = match self.transport {
            Some(transport) => transport,
            None => default_transport(use_env_proxy)?,
        };
        Ok(SystemOne {
            flavor,
            feedback_source,
            base_url,
            model,
            api_key: self.api_key,
            function_threshold,
            arguments_threshold,
            timeout: self.timeout,
            max_retries: self.max_retries,
            transport,
            sleep: self.sleep.unwrap_or_else(default_sleep),
            check_provenance,
            provenance: Mutex::new(None),
        })
    }
}

macro_rules! common_setters {
    () => {
        /// The pass mark of both questions (0.5 in the paper).
        pub fn threshold(mut self, threshold: f64) -> Self {
            self.common.threshold = threshold;
            self
        }

        /// Override the function question's pass mark.
        pub fn function_threshold(mut self, threshold: f64) -> Self {
            self.common.function_threshold = Some(threshold);
            self
        }

        /// Override the argument question's pass mark.
        pub fn arguments_threshold(mut self, threshold: f64) -> Self {
            self.common.arguments_threshold = Some(threshold);
            self
        }

        /// The time limit of one HTTP request.
        pub fn timeout(mut self, timeout: Duration) -> Self {
            self.common.timeout = timeout;
            self
        }

        /// HTTP retries after the first attempt; unrelated to DM-Call's repair limit.
        pub fn max_retries(mut self, retries: u32) -> Self {
            self.common.max_retries = retries;
            self
        }

        /// The HTTP transport (for tests, proxies or another client).
        pub fn transport(mut self, transport: SharedTransport) -> Self {
            self.common.transport = Some(transport);
            self
        }

        /// How to wait between retries.
        pub fn sleep(mut self, sleep: Sleep) -> Self {
            self.common.sleep = Some(sleep);
            self
        }

        /// A bearer key.
        pub fn api_key(mut self, key: impl Into<String>) -> Self {
            self.common.api_key = Some(key.into());
            self
        }
    };
}

/// Builds a [`SystemOne`] decision model.
pub struct SystemOneBuilder {
    base_url: String,
    model: String,
    common: Common,
    feedback_source: Option<String>,
}

impl SystemOneBuilder {
    common_setters!();

    /// The `source` field of the feedback records.
    pub fn feedback_source(mut self, source: impl Into<String>) -> Self {
        self.feedback_source = Some(source.into());
        self
    }

    /// Check the options and build the decision model.
    pub fn build(self) -> Result<SystemOne> {
        let source = self
            .feedback_source
            .unwrap_or_else(|| "decision_model_noul_and_deterministic_schema_validation".to_string());
        self.common.build(Flavor::Generic, "SystemOne", &self.base_url, self.model, source, true, false)
    }
}

/// TypeSafe's hosted decision model. Key: `TYPESAFE_API_KEY`.
///
/// The default model is `jev-1.13.0`, the one used in the paper (`TYPESAFE_MODEL` changes it),
/// and the default endpoint is `https://api.typesafe.ai` (`TYPESAFE_BASE_URL`).
pub struct Jev(SystemOne);

impl Jev {
    /// The paper's model.
    pub const DEFAULT_MODEL: &'static str = "jev-1.13.0";
    /// TypeSafe's endpoint.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.typesafe.ai";
    /// The `source` field of Jev's feedback records.
    pub const FEEDBACK_SOURCE: &'static str = "jev_noul_and_deterministic_schema_validation";

    /// A builder; every option falls back to the environment or the paper's settings.
    pub fn builder() -> JevBuilder {
        JevBuilder { common: Common::new(60, 3), model: None, base_url: None }
    }

    /// Jev with the key, model and endpoint from the environment and threshold 0.5.
    pub fn from_env() -> Result<Jev> {
        Jev::builder().build()
    }

    /// The origin requests go to.
    pub fn base_url(&self) -> &str {
        self.0.base_url()
    }
}

/// Builds [`Jev`].
pub struct JevBuilder {
    common: Common,
    model: Option<String>,
    base_url: Option<String>,
}

impl JevBuilder {
    common_setters!();

    /// The model (otherwise `TYPESAFE_MODEL`, then `jev-1.13.0`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The endpoint (otherwise `TYPESAFE_BASE_URL`, then `https://api.typesafe.ai`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Check the options and build Jev.
    pub fn build(mut self) -> Result<Jev> {
        let key = self.common.api_key.take().filter(|key| !key.is_empty()).or_else(|| env_first(&["TYPESAFE_API_KEY"]));
        let Some(key) = key else {
            return Err(Error::invalid("Jev needs an API key: pass .api_key(...) or set TYPESAFE_API_KEY."));
        };
        self.common.api_key = Some(key);
        let nonempty = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let base_url = self.base_url.filter(|u| !u.is_empty()).or_else(|| nonempty("TYPESAFE_BASE_URL"));
        let model = self.model.filter(|m| !m.is_empty()).or_else(|| nonempty("TYPESAFE_MODEL"));
        self.common
            .build(
                Flavor::Jev,
                "Jev",
                base_url.as_deref().unwrap_or(Jev::DEFAULT_BASE_URL),
                model.unwrap_or_else(|| Jev::DEFAULT_MODEL.to_string()),
                Jev::FEEDBACK_SOURCE.to_string(),
                true,
                false,
            )
            .map(Jev)
    }
}

impl DecisionModel for Jev {
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> std::result::Result<Answer, DecisionModelError> {
        self.0.ask_http(state, questions)
    }
    fn model(&self) -> &str {
        &self.0.model
    }
    fn function_threshold(&self) -> f64 {
        self.0.function_threshold
    }
    fn arguments_threshold(&self) -> f64 {
        self.0.arguments_threshold
    }
    fn feedback_source(&self) -> &str {
        &self.0.feedback_source
    }
}

impl fmt::Debug for Jev {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.describe(f, "Jev")
    }
}

/// Perplexity Decider v1.1 27B, an open-weight decision model, served on your own machine.
///
/// `base_url` is the origin of a server running the checkpoint
/// (`perplexity-ai/pplx-decider-v1.1-27b` at revision `14c9d901...`) with its own native
/// inference code, which serves `/v1/systemone`. A generic causal chat server is not this
/// model. Plain HTTP is accepted only for `localhost`; use an SSH tunnel or HTTPS otherwise.
/// Proxy environment variables are ignored.
///
/// With `check_provenance(true)` the client first reads `GET /provenance` and stops unless
/// the server reports the pinned checkpoint, revision, attention mode and calibration
/// temperature. The checkpoint's own server does not provide the route, so it is off by default.
pub struct PPLXDecider(SystemOne);

impl PPLXDecider {
    /// The checkpoint.
    pub const MODEL_ID: &'static str = "perplexity-ai/pplx-decider-v1.1-27b";
    /// The pinned revision.
    pub const MODEL_REVISION: &'static str = "14c9d90137962d300f3de95633f7b6535e80c5ce";
    /// The calibration temperature the server must report.
    pub const CALIBRATION_TEMPERATURE: f64 = 1.0087417621345625;
    /// The checkpoint's input limit.
    pub const MAX_INPUT_TOKENS: u64 = 8192;
    /// The model name the checkpoint's server reports.
    pub const SERVER_MODEL: &'static str = "autojev-qwen3.8-27b";
    /// The model name sent in requests.
    pub const WIRE_MODEL: &'static str = "jev-latest";
    /// The `source` field of its feedback records.
    pub const FEEDBACK_SOURCE: &'static str = "perplexity_decider_noul_and_deterministic_schema_validation";

    /// A builder for a server at `base_url` (an origin without a path).
    pub fn builder(base_url: impl Into<String>) -> PPLXDeciderBuilder {
        PPLXDeciderBuilder { base_url: base_url.into(), common: Common::new(180, 0), check_provenance: false }
    }

    /// The decider at `base_url` with the default options.
    pub fn new(base_url: impl Into<String>) -> Result<PPLXDecider> {
        PPLXDecider::builder(base_url).build()
    }

    /// The origin requests go to.
    pub fn base_url(&self) -> &str {
        self.0.base_url()
    }

    /// Read and verify the server's provenance now.
    pub fn provenance(&self) -> std::result::Result<Value, DecisionModelError> {
        self.0.read_provenance()
    }
}

/// Builds [`PPLXDecider`].
pub struct PPLXDeciderBuilder {
    base_url: String,
    common: Common,
    check_provenance: bool,
}

impl PPLXDeciderBuilder {
    common_setters!();

    /// Verify `GET /provenance` before the first judgment.
    pub fn check_provenance(mut self, check: bool) -> Self {
        self.check_provenance = check;
        self
    }

    /// Check the options and build the decider.
    pub fn build(self) -> Result<PPLXDecider> {
        let url = self.base_url.trim();
        let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
        let rest = if scheme.is_empty() { url } else { rest };
        let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let tail = if scheme.is_empty() { url } else { &rest[end..] };
        let path = tail.split(['?', '#']).next().unwrap_or_default();
        if !path.is_empty() && path != "/" {
            return Err(Error::invalid("PPLXDecider base_url must be an origin without a path."));
        }
        let local = hostname(url).is_some_and(|host| matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1"));
        if scheme.eq_ignore_ascii_case("http") && !local {
            return Err(Error::invalid("Use HTTPS, or an SSH tunnel to localhost, for a remote PPLX Decider."));
        }
        self.common
            .build(
                Flavor::Pplx,
                "PPLXDecider",
                url,
                PPLXDecider::WIRE_MODEL.to_string(),
                PPLXDecider::FEEDBACK_SOURCE.to_string(),
                false,
                self.check_provenance,
            )
            .map(PPLXDecider)
    }
}

impl DecisionModel for PPLXDecider {
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> std::result::Result<Answer, DecisionModelError> {
        self.0.ask_http(state, questions)
    }
    fn model(&self) -> &str {
        &self.0.model
    }
    fn function_threshold(&self) -> f64 {
        self.0.function_threshold
    }
    fn arguments_threshold(&self) -> f64 {
        self.0.arguments_threshold
    }
    fn feedback_source(&self) -> &str {
        &self.0.feedback_source
    }
}

impl fmt::Debug for PPLXDecider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PPLXDecider(model={}, revision={}, base_url={}, function_threshold={}, arguments_threshold={})",
            pyjson::repr_str(PPLXDecider::MODEL_ID),
            pyjson::repr_str(&PPLXDecider::MODEL_REVISION[..8]),
            pyjson::repr_str(&self.0.base_url),
            pyjson::float_repr(self.0.function_threshold),
            pyjson::float_repr(self.0.arguments_threshold)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_seconds_and_dates() {
        let response = HttpResponse::new(429, "").with_header("Retry-After", "2");
        assert_eq!(retry_delay(Some(&response), 0), 2.0);
        let past = HttpResponse::new(429, "").with_header("Retry-After", "Wed, 21 Oct 2015 07:28:00 GMT");
        assert_eq!(retry_delay(Some(&past), 3), 8.0);
        assert_eq!(retry_delay(None, 10), 30.0);
        let delay = http_date_delay("Thu, 01 Jan 1970 00:00:10 GMT").unwrap();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        assert!((delay + now - 10.0).abs() < 1.0);
    }
}
