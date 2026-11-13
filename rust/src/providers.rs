//! Function-calling models (generators) over plain HTTP, without vendor SDKs.
//!
//! Each provider turns (messages, tools, repair history) into one request and the response
//! into a [`Proposal`]: a list of calls (`Some(vec![])` for no call), or `None` when the output
//! holds no complete proposal (cut off at the token limit, filtered, refused or malformed).
//! Keys come from the builder or environment variables and never appear in errors, `Debug`
//! output or results.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::error::{Error, ProviderError, Result};
use crate::message::{Call, Message};
use crate::prompts::{functions_of, with_history, SYSTEM_PROMPT};
use crate::pyjson::{self, Num};
use crate::transport::{default_sleep, default_transport, HttpRequest, HttpResponse, Method, SharedTransport, Sleep};

/// Why an output holds no complete proposal. Such a proposal is rejected and repaired.
pub const INCOMPLETE: [(&str, &str); 6] = [
    ("length", "The response reached the output token limit before a complete proposal."),
    ("content_filter", "The provider's content filter stopped the response before a complete proposal."),
    ("refusal", "The response was a refusal and contains no proposal."),
    ("prompt_filter", "The provider's content filter rejected the request; no response was generated."),
    ("malformed_call", "The model produced a malformed function call."),
    ("abnormal_stop", "The response stopped abnormally before a complete proposal."),
];

fn incomplete_message(reason: &str) -> &'static str {
    INCOMPLETE.iter().find(|(key, _)| *key == reason).map(|(_, text)| *text).unwrap_or("Incomplete response.")
}

/// Token counts by kind (`input_tokens`, `cached_input_tokens`, `output_tokens`), in the order
/// they were reported. A decision model's unknown count is `null`.
pub type Usage = Map<String, Value>;

/// One generator output.
#[derive(Clone, Debug, Default)]
pub struct Proposal {
    /// The calls; `Some(vec![])` is a decision to make no call and `None` an unreadable output.
    pub calls: Option<Vec<Call>>,
    /// The text the model wrote, if any.
    pub content: Option<String>,
    /// Why some or all of the output could not be read.
    pub parse_errors: Vec<String>,
    /// `length`, `content_filter`, `refusal`, `prompt_filter`, `malformed_call` or
    /// `abnormal_stop` for an unreadable output.
    pub incomplete: Option<String>,
    /// The provider's short name (`openai`, `anthropic`, ...).
    pub provider: String,
    /// The model that was requested.
    pub requested_model: String,
    /// The model the provider reported.
    pub model: Option<String>,
    /// Token counts of this request.
    pub usage: Usage,
    /// Seconds spent in HTTP requests, without retry waits.
    pub latency_s: f64,
    /// Seconds spent waiting between retries.
    pub retry_wait_s: f64,
    /// HTTP attempts.
    pub api_calls: u32,
}

impl Proposal {
    /// A complete proposal with the given calls (use an empty list for no call).
    pub fn with_calls(calls: Vec<Call>) -> Self {
        Proposal { calls: Some(calls), ..Proposal::default() }
    }

    /// A readable decision to make no call: an empty call list without parse errors.
    pub(crate) fn readable_no_call(&self) -> bool {
        matches!(&self.calls, Some(calls) if calls.is_empty())
            && self.incomplete.is_none()
            && self.parse_errors.is_empty()
    }
}

/// Anything that proposes calls: the DM-Call loop calls `generate` once per attempt.
///
/// `history` holds the earlier rejected candidates and their feedback, already in the chosen
/// representation, each as `{"round"?, "calls", "feedback"}`. Built-in providers put them in
/// one user message before the original messages ([`crate::prompts::history_message`]).
pub trait Generator: Send + Sync {
    /// One proposal for the messages and tools after the given repair history.
    fn generate(&self, messages: &[Message], tools: &[Value], history: &[Value]) -> Result<Proposal>;
}

impl<G: Generator + ?Sized> Generator for Box<G> {
    fn generate(&self, messages: &[Message], tools: &[Value], history: &[Value]) -> Result<Proposal> {
        (**self).generate(messages, tools, history)
    }
}

impl<G: Generator + ?Sized> Generator for std::sync::Arc<G> {
    fn generate(&self, messages: &[Message], tools: &[Value], history: &[Value]) -> Result<Proposal> {
        (**self).generate(messages, tools, history)
    }
}

/// Which wire protocol and defaults a [`Provider`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    /// Any Chat Completions endpoint (vLLM, Ollama, LM Studio, ...).
    OpenAICompatible,
    /// OpenAI Chat Completions.
    OpenAI,
    /// xAI Grok.
    XAI,
    /// OpenRouter.
    OpenRouter,
    /// The Meta Model API (Muse models).
    Meta,
    /// An Azure OpenAI deployment.
    AzureOpenAI,
    /// Anthropic Messages.
    Anthropic,
    /// Google Gemini `generateContent`.
    Gemini,
}

impl ProviderKind {
    /// The short name used in proposals and errors.
    pub fn name(self) -> &'static str {
        match self {
            ProviderKind::OpenAICompatible => "openai-compatible",
            ProviderKind::OpenAI => "openai",
            ProviderKind::XAI => "xai",
            ProviderKind::OpenRouter => "openrouter",
            ProviderKind::Meta => "meta",
            ProviderKind::AzureOpenAI => "azure",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Gemini => "gemini",
        }
    }

    fn class(self) -> &'static str {
        match self {
            ProviderKind::OpenAICompatible => "OpenAICompatible",
            ProviderKind::OpenAI => "OpenAI",
            ProviderKind::XAI => "XAI",
            ProviderKind::OpenRouter => "OpenRouter",
            ProviderKind::Meta => "Meta",
            ProviderKind::AzureOpenAI => "AzureOpenAI",
            ProviderKind::Anthropic => "Anthropic",
            ProviderKind::Gemini => "Gemini",
        }
    }

    fn is_chat(self) -> bool {
        !matches!(self, ProviderKind::Anthropic | ProviderKind::Gemini)
    }

    fn default_base_url(self) -> Option<&'static str> {
        match self {
            ProviderKind::OpenAI => Some("https://api.openai.com/v1"),
            ProviderKind::XAI => Some("https://api.x.ai/v1"),
            ProviderKind::OpenRouter => Some("https://openrouter.ai/api/v1"),
            ProviderKind::Meta => Some("https://api.meta.ai/v1"),
            ProviderKind::Anthropic => Some("https://api.anthropic.com/v1"),
            ProviderKind::Gemini => Some("https://generativelanguage.googleapis.com/v1beta"),
            ProviderKind::OpenAICompatible | ProviderKind::AzureOpenAI => None,
        }
    }

    fn base_url_env(self) -> &'static [&'static str] {
        match self {
            ProviderKind::OpenAI => &["OPENAI_BASE_URL"],
            ProviderKind::Anthropic => &["ANTHROPIC_BASE_URL"],
            _ => &[],
        }
    }

    fn key_env(self) -> &'static [&'static str] {
        match self {
            ProviderKind::OpenAICompatible => &[],
            ProviderKind::OpenAI => &["OPENAI_API_KEY"],
            ProviderKind::XAI => &["XAI_API_KEY"],
            ProviderKind::OpenRouter => &["OPENROUTER_API_KEY"],
            ProviderKind::Meta => &["MODEL_API_KEY"],
            ProviderKind::AzureOpenAI => &["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"],
            ProviderKind::Anthropic => &["ANTHROPIC_API_KEY"],
            ProviderKind::Gemini => &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        }
    }

    fn max_tokens_field(self) -> &'static str {
        match self {
            ProviderKind::XAI | ProviderKind::OpenRouter => "max_tokens",
            _ => "max_completion_tokens",
        }
    }
}

/// Options shared by every provider builder. Fields a provider does not use must stay unset.
#[derive(Clone)]
pub struct ProviderOptions {
    /// The API key; otherwise the provider's environment variable.
    pub api_key: Option<String>,
    /// The base URL; otherwise the environment variable or the provider's default.
    pub base_url: Option<String>,
    /// The output token limit (4096 by default); `None` sends no limit where the API allows it.
    pub max_output_tokens: Option<u32>,
    /// The sampling temperature; `None` sends none.
    pub temperature: Option<f64>,
    /// The system prompt; [`SYSTEM_PROMPT`] by default.
    pub system_prompt: Option<String>,
    /// The time limit of one HTTP request (120 seconds).
    pub timeout: Duration,
    /// HTTP retries after the first attempt (3), on 408, 429 and 5xx responses and transport failures.
    pub max_retries: u32,
    /// Extra request headers.
    pub headers: Vec<(String, String)>,
    /// The HTTP transport; the default `ureq` client when unset.
    pub transport: Option<SharedTransport>,
    /// How to wait between retries; `std::thread::sleep` when unset.
    pub sleep: Option<Sleep>,
    /// Chat Completions: `reasoning_effort`.
    pub reasoning_effort: Option<String>,
    /// Chat Completions: `parallel_tool_calls`. `None` (the default) uses the provider's
    /// default, `Some(false)` except for Meta, which sends nothing; `Some(None)` omits it.
    pub parallel_tool_calls: Option<Option<bool>>,
    /// Chat Completions: the name of the token limit field.
    pub max_tokens_field: Option<String>,
    /// Chat Completions: the path after the base URL (`/chat/completions`).
    pub path: String,
    /// Azure: the resource endpoint; otherwise `AZURE_OPENAI_ENDPOINT`.
    pub endpoint: Option<String>,
    /// Azure: the API version; otherwise `AZURE_OPENAI_API_VERSION` or `2024-12-01-preview`.
    pub api_version: Option<String>,
    /// Gemini: `thinkingConfig.thinkingBudget`.
    pub thinking_budget: Option<i64>,
}

impl Default for ProviderOptions {
    fn default() -> Self {
        ProviderOptions {
            api_key: None,
            base_url: None,
            max_output_tokens: Some(4096),
            temperature: None,
            system_prompt: None,
            timeout: Duration::from_secs(120),
            max_retries: 3,
            headers: Vec::new(),
            transport: None,
            sleep: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            max_tokens_field: None,
            path: "/chat/completions".to_string(),
            endpoint: None,
            api_version: None,
            thinking_budget: None,
        }
    }
}

impl fmt::Debug for ProviderOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderOptions").field("base_url", &self.base_url).finish_non_exhaustive()
    }
}

pub(crate) fn env_first(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()))
}

/// Python's `urlsplit` checks: http(s), a host, no credentials, query or fragment.
pub(crate) fn normalize_base_url(value: &str) -> Option<String> {
    let url = value.trim().trim_end_matches('/');
    let (scheme, rest) = url.split_once("://")?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let netloc = &rest[..end];
    if netloc.is_empty() || netloc.contains('@') {
        return None;
    }
    let tail = &rest[end..];
    let (before_fragment, fragment) = tail.split_once('#').unwrap_or((tail, ""));
    let query = before_fragment.split_once('?').map_or("", |(_, query)| query);
    if !query.is_empty() || !fragment.is_empty() {
        return None;
    }
    Some(url.to_string())
}

/// The lower-case host of an `http(s)://host[:port]/...` URL (brackets removed for IPv6).
pub(crate) fn hostname(url: &str) -> Option<String> {
    let (_, rest) = url.trim().split_once("://")?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let netloc = rest[..end].rsplit('@').next()?;
    let host = if let Some(stripped) = netloc.strip_prefix('[') {
        stripped.split(']').next()?
    } else {
        netloc.split(':').next()?
    };
    Some(host.to_ascii_lowercase())
}

/// Python's `urllib.parse.quote(text, safe='')`.
fn quote(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

pub(crate) fn token_like(value: &str, max: usize, extra: &[char]) -> bool {
    !value.is_empty()
        && value.chars().count() <= max
        && value.chars().all(|c| c.is_ascii_alphanumeric() || extra.contains(&c))
}

pub(crate) fn wait(sleep: &Sleep, seconds: f64) -> f64 {
    let started = Instant::now();
    sleep(Duration::from_secs_f64(seconds.max(0.0)));
    started.elapsed().as_secs_f64()
}

enum Fail {
    Provider(String),
    Protocol,
}

type Decoded = (Option<Vec<Call>>, Vec<String>, Option<String>, Option<String>);

fn unreadable(reason: &str) -> Decoded {
    (None, vec![incomplete_message(reason).to_string()], None, Some(reason.to_string()))
}

fn complete(calls: Vec<Call>, errors: Vec<String>, text: Vec<String>) -> Decoded {
    let content = if text.is_empty() { None } else { Some(text.join("\n")) };
    (Some(calls), errors, content, None)
}

/// Malformed arguments stay a rejected, repairable call, never a no-call.
fn arguments(value: Option<&Value>, encoded: bool) -> std::result::Result<(Value, Option<&'static str>), Fail> {
    let parsed;
    let mut value = value;
    if let (true, Some(Value::String(text))) = (encoded, value) {
        parsed = if text.trim().is_empty() {
            json!({})
        } else {
            match pyjson::loads(text) {
                Ok(v) => v,
                Err(_) => return Ok((Value::Null, Some("Function arguments are not valid JSON."))),
            }
        };
        value = Some(&parsed);
    }
    match value {
        None | Some(Value::Null) => Ok((json!({}), None)),
        Some(object @ Value::Object(_)) if pyjson::all_finite(object) => Ok((object.clone(), None)),
        Some(Value::Object(_)) => Ok((Value::Null, Some("Function arguments must contain finite JSON values."))),
        Some(_) => Ok((Value::Null, Some("Function arguments must be a JSON object."))),
    }
}

fn append_call(
    calls: &mut Vec<Call>,
    errors: &mut Vec<String>,
    name: Option<&Value>,
    raw: Option<&Value>,
    encoded: bool,
) -> std::result::Result<(), Fail> {
    let name = match name {
        Some(Value::String(name)) if !name.is_empty() => name.clone(),
        _ => return Err(Fail::Provider("The response contained an invalid function name.".into())),
    };
    let (value, error) = arguments(raw, encoded)?;
    calls.push(Call::new(name, value));
    if let Some(error) = error {
        errors.push(error.to_string());
    }
    Ok(())
}

fn count(value: Option<&Value>) -> std::result::Result<Option<u64>, Fail> {
    let invalid = || Fail::Provider("The response contained invalid token usage.".into());
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => match pyjson::num(n) {
            Num::Int(text) => text.parse::<u64>().map(Some).map_err(|_| invalid()),
            Num::Float(_) => Err(invalid()),
        },
        Some(_) => Err(invalid()),
    }
}

fn usage_map(entries: Vec<(&str, Option<u64>)>) -> Usage {
    let mut usage = Usage::new();
    for (key, value) in entries {
        if let Some(value) = value {
            usage.insert(key.to_string(), Value::from(value));
        }
    }
    usage
}

fn hashable(value: Option<&Value>) -> std::result::Result<Option<&Value>, Fail> {
    match value {
        Some(Value::Array(_) | Value::Object(_)) => Err(Fail::Protocol),
        other => Ok(other),
    }
}

/// An HTTP function-calling model. The typed wrappers ([`OpenAI`], [`Anthropic`], ...) build it.
pub struct Provider {
    kind: ProviderKind,
    model: String,
    api_key: Option<String>,
    base_url: String,
    max_output_tokens: Option<u32>,
    temperature: Option<f64>,
    system_prompt: String,
    timeout: Duration,
    max_retries: u32,
    extra_headers: Vec<(String, String)>,
    transport: SharedTransport,
    sleep: Sleep,
    reasoning_effort: Option<String>,
    parallel_tool_calls: Option<bool>,
    max_tokens_field: String,
    path: String,
    api_version: String,
    thinking_budget: Option<i64>,
}

impl fmt::Debug for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(model={}, base_url={})",
            self.kind.class(),
            pyjson::repr_str(&self.model),
            pyjson::repr_str(&self.base_url)
        )
    }
}

impl Provider {
    /// Build a provider of the given kind; prefer the typed builders such as [`OpenAI::builder`].
    pub fn new(kind: ProviderKind, model: &str, options: ProviderOptions) -> Result<Provider> {
        let class = kind.class();
        let mut model = model.to_string();
        if kind == ProviderKind::Gemini {
            model = model.strip_prefix("models/").map(str::to_string).unwrap_or(model);
        }
        if !kind.is_chat()
            && (options.reasoning_effort.is_some()
                || options.max_tokens_field.is_some()
                || options.parallel_tool_calls.is_some()
                || options.path != "/chat/completions")
        {
            return Err(Error::invalid(format!("{class} does not take Chat Completions options.")));
        }
        if kind != ProviderKind::AzureOpenAI && (options.endpoint.is_some() || options.api_version.is_some()) {
            return Err(Error::invalid(format!("{class} does not take Azure options (endpoint, api_version).")));
        }
        if kind != ProviderKind::Gemini && options.thinking_budget.is_some() {
            return Err(Error::invalid(format!("{class} does not take thinking_budget.")));
        }
        let mut base_url = options.base_url.clone().filter(|url| !url.is_empty());
        let mut api_version = String::new();
        if kind == ProviderKind::AzureOpenAI {
            let endpoint = options
                .endpoint
                .clone()
                .filter(|url| !url.is_empty())
                .or(base_url.take())
                .or_else(|| env_first(&["AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"]));
            let Some(endpoint) = endpoint else {
                return Err(Error::invalid(
                    "AzureOpenAI needs an endpoint: pass .endpoint(...) or set AZURE_OPENAI_ENDPOINT.",
                ));
            };
            base_url = Some(endpoint);
            api_version = options
                .api_version
                .clone()
                .filter(|v| !v.is_empty())
                .or_else(|| env_first(&["AZURE_OPENAI_API_VERSION"]))
                .unwrap_or_else(|| "2024-12-01-preview".to_string());
            let b = api_version.as_bytes();
            let shaped = (b.len() == 10 || (b.len() == 18 && &api_version[10..] == "-preview"))
                && b[..10]
                    .iter()
                    .enumerate()
                    .all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() });
            if !shaped {
                return Err(Error::invalid("api_version must look like 2024-12-01-preview."));
            }
            if model == "." || model == ".." || model.chars().any(|c| (c as u32) < 32) {
                return Err(Error::invalid("The deployment name contains invalid characters."));
            }
        }
        if model.trim().is_empty() {
            return Err(Error::invalid(format!("{class} needs a model name.")));
        }
        if options.max_output_tokens == Some(0) {
            return Err(Error::invalid("max_output_tokens must be a positive integer or None."));
        }
        if options.timeout.is_zero() {
            return Err(Error::invalid("timeout must be a positive number of seconds."));
        }
        if options.temperature.is_some_and(|t| !t.is_finite()) {
            return Err(Error::invalid("temperature must be a finite number or None."));
        }
        let system_prompt = options.system_prompt.clone().unwrap_or_else(|| SYSTEM_PROMPT.to_string());
        if system_prompt.trim().is_empty() {
            return Err(Error::invalid("system_prompt must be nonempty text."));
        }
        let api_key = options.api_key.clone().filter(|key| !key.is_empty()).or_else(|| env_first(kind.key_env()));
        if api_key.is_none() && kind != ProviderKind::OpenAICompatible {
            let names = kind.key_env().join(" or ");
            return Err(Error::invalid(format!("{class} needs an API key: pass .api_key(...) or set {names}.")));
        }
        if api_key.as_deref().is_some_and(|key| key.contains(['\n', '\r'])) {
            return Err(Error::invalid("API keys must be single-line strings."));
        }
        let base = base_url
            .or_else(|| env_first(kind.base_url_env()))
            .or_else(|| kind.default_base_url().map(str::to_string))
            .and_then(|url| normalize_base_url(&url));
        let Some(base_url) = base else {
            return Err(Error::invalid(format!(
                "{class} needs an http(s) base_url without credentials, query or fragment."
            )));
        };
        if kind.is_chat() && !options.path.starts_with('/') {
            return Err(Error::invalid("path must start with '/'."));
        }
        let transport = match options.transport.clone() {
            Some(transport) => transport,
            None => default_transport(true)?,
        };
        Ok(Provider {
            kind,
            model,
            api_key,
            base_url,
            max_output_tokens: options.max_output_tokens,
            temperature: options.temperature,
            system_prompt,
            timeout: options.timeout,
            max_retries: options.max_retries,
            extra_headers: options.headers.clone(),
            transport,
            sleep: options.sleep.clone().unwrap_or_else(default_sleep),
            reasoning_effort: options.reasoning_effort.clone(),
            // Meta rejects some tool settings for Muse models, so it sends none unless asked.
            parallel_tool_calls: options.parallel_tool_calls.unwrap_or((kind != ProviderKind::Meta).then_some(false)),
            max_tokens_field: options.max_tokens_field.clone().unwrap_or_else(|| kind.max_tokens_field().to_string()),
            path: options.path.clone(),
            api_version,
            thinking_budget: options.thinking_budget,
        })
    }

    /// The protocol and defaults in use.
    pub fn kind(&self) -> ProviderKind {
        self.kind
    }

    /// The requested model (the deployment name for Azure).
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The base URL without a trailing slash.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The request URL.
    pub fn url(&self) -> String {
        match self.kind {
            ProviderKind::AzureOpenAI => format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                self.base_url,
                quote(&self.model),
                self.api_version
            ),
            ProviderKind::Anthropic => format!("{}/messages", self.base_url),
            ProviderKind::Gemini => format!("{}/models/{}:generateContent", self.base_url, quote(&self.model)),
            _ => format!("{}{}", self.base_url, self.path),
        }
    }

    fn auth_headers(&self) -> Vec<(String, String)> {
        let key = self.api_key.clone().unwrap_or_default();
        match self.kind {
            ProviderKind::AzureOpenAI => vec![("api-key".into(), key)],
            ProviderKind::Anthropic => {
                vec![("x-api-key".into(), key), ("anthropic-version".into(), "2023-06-01".into())]
            }
            ProviderKind::Gemini => vec![("x-goog-api-key".into(), key)],
            _ if self.api_key.is_some() => vec![("Authorization".into(), format!("Bearer {key}"))],
            _ => Vec::new(),
        }
    }

    fn headers(&self) -> Vec<(String, String)> {
        let mut fixed = self.auth_headers();
        fixed.push(("Content-Type".into(), "application/json".into()));
        let mut headers: Vec<(String, String)> = self
            .extra_headers
            .iter()
            .filter(|(name, _)| !fixed.iter().any(|(known, _)| known.eq_ignore_ascii_case(name)))
            .cloned()
            .collect();
        headers.extend(fixed);
        headers
    }

    /// The JSON body of one request.
    pub fn payload(&self, messages: &[Message], tools: &[Value], history: &[Value]) -> Result<Value> {
        let conversation = with_history(messages, history)?;
        let functions = functions_of(tools)?;
        let mut payload = Map::new();
        match self.kind {
            ProviderKind::Anthropic => {
                let mut system = vec![self.system_prompt.clone()];
                let mut turns = Vec::new();
                for message in conversation {
                    match message.role.as_str() {
                        "system" | "developer" => system.push(message.content),
                        "user" | "assistant" => turns.push(message.to_value()),
                        _ => {
                            return Err(Error::invalid(
                                "Anthropic accepts only user and assistant conversation messages.",
                            ))
                        }
                    }
                }
                payload.insert("model".into(), json!(self.model));
                payload.insert("system".into(), json!(system.join("\n\n")));
                payload.insert("messages".into(), Value::Array(turns));
                payload.insert("max_tokens".into(), json!(self.max_output_tokens.unwrap_or(4096)));
                if !functions.is_empty() {
                    let tools: Vec<Value> = functions
                        .iter()
                        .map(|f| {
                            pyjson::object(vec![
                                ("name", f["name"].clone()),
                                ("description", f.get("description").cloned().unwrap_or_else(|| json!(""))),
                                (
                                    "input_schema",
                                    f.get("parameters")
                                        .cloned()
                                        .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                                ),
                            ])
                        })
                        .collect();
                    payload.insert("tools".into(), Value::Array(tools));
                    payload.insert("tool_choice".into(), json!({"type": "auto", "disable_parallel_tool_use": true}));
                }
                if let Some(temperature) = self.temperature {
                    payload.insert("temperature".into(), pyjson::float(temperature));
                }
            }
            ProviderKind::Gemini => {
                let mut system = vec![self.system_prompt.clone()];
                let mut contents = Vec::new();
                for message in conversation {
                    match message.role.as_str() {
                        "system" | "developer" => system.push(message.content),
                        "user" | "assistant" => {
                            let role = if message.role == "assistant" { "model" } else { "user" };
                            contents.push(json!({"role": role, "parts": [{"text": message.content}]}));
                        }
                        _ => {
                            return Err(Error::invalid("Gemini accepts only user and assistant conversation messages."))
                        }
                    }
                }
                payload.insert("systemInstruction".into(), json!({"parts": [{"text": system.join("\n\n")}]}));
                payload.insert("contents".into(), Value::Array(contents));
                if !functions.is_empty() {
                    let declarations: Vec<Value> = functions
                        .iter()
                        .map(|f| {
                            pyjson::object(vec![
                                ("name", f["name"].clone()),
                                ("description", f.get("description").cloned().unwrap_or_else(|| json!(""))),
                                (
                                    "parametersJsonSchema",
                                    f.get("parameters")
                                        .cloned()
                                        .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                                ),
                            ])
                        })
                        .collect();
                    payload.insert("tools".into(), json!([{"functionDeclarations": declarations}]));
                    payload.insert("toolConfig".into(), json!({"functionCallingConfig": {"mode": "AUTO"}}));
                }
                let mut config = Map::new();
                if let Some(tokens) = self.max_output_tokens {
                    config.insert("maxOutputTokens".into(), json!(tokens));
                }
                if let Some(temperature) = self.temperature {
                    config.insert("temperature".into(), pyjson::float(temperature));
                }
                if let Some(budget) = self.thinking_budget {
                    config.insert("thinkingConfig".into(), json!({"thinkingBudget": budget}));
                }
                if !config.is_empty() {
                    payload.insert("generationConfig".into(), Value::Object(config));
                }
            }
            _ => {
                let mut sent = vec![Message::system(self.system_prompt.clone()).to_value()];
                sent.extend(conversation.iter().map(Message::to_value));
                payload.insert("model".into(), json!(self.model));
                payload.insert("messages".into(), Value::Array(sent));
                if !functions.is_empty() {
                    // A request without functions carries no tool settings.
                    let tools: Vec<Value> =
                        functions.into_iter().map(|f| json!({"type": "function", "function": f})).collect();
                    payload.insert("tools".into(), Value::Array(tools));
                    payload.insert("tool_choice".into(), json!("auto"));
                    if let Some(parallel) = self.parallel_tool_calls {
                        payload.insert("parallel_tool_calls".into(), json!(parallel));
                    }
                }
                if let Some(tokens) = self.max_output_tokens {
                    payload.insert(self.max_tokens_field.clone(), json!(tokens));
                }
                if let Some(effort) = self.reasoning_effort.as_ref().filter(|effort| !effort.is_empty()) {
                    payload.insert("reasoning_effort".into(), json!(effort));
                }
                if let Some(temperature) = self.temperature {
                    payload.insert("temperature".into(), pyjson::float(temperature));
                }
            }
        }
        Ok(Value::Object(payload))
    }

    fn decode(&self, data: &Map<String, Value>) -> std::result::Result<Decoded, Fail> {
        match self.kind {
            ProviderKind::Anthropic => decode_anthropic(data),
            ProviderKind::Gemini => decode_gemini(data),
            _ => decode_chat(data),
        }
    }

    fn usage(&self, data: &Map<String, Value>) -> std::result::Result<Usage, Fail> {
        match self.kind {
            ProviderKind::Anthropic => {
                let Some(Value::Object(raw)) = data.get("usage") else { return Ok(Usage::new()) };
                let read = count(raw.get("cache_read_input_tokens"))?.unwrap_or(0);
                let written = count(raw.get("cache_creation_input_tokens"))?.unwrap_or(0);
                let base = count(raw.get("input_tokens"))?;
                let output = count(raw.get("output_tokens"))?;
                Ok(usage_map(vec![
                    ("cached_input_tokens", Some(read)),
                    ("output_tokens", output),
                    ("input_tokens", base.map(|base| base + read + written)),
                ]))
            }
            ProviderKind::Gemini => {
                let Some(Value::Object(raw)) = data.get("usageMetadata") else { return Ok(Usage::new()) };
                let output = count(raw.get("candidatesTokenCount"))?;
                let thoughts = count(raw.get("thoughtsTokenCount"))?.unwrap_or(0);
                let total = if output.is_none() && thoughts == 0 { None } else { Some(output.unwrap_or(0) + thoughts) };
                Ok(usage_map(vec![
                    ("input_tokens", count(raw.get("promptTokenCount"))?),
                    ("cached_input_tokens", count(raw.get("cachedContentTokenCount"))?),
                    ("output_tokens", total),
                ]))
            }
            _ => {
                let Some(Value::Object(raw)) = data.get("usage") else { return Ok(Usage::new()) };
                let cached = match raw.get("prompt_tokens_details") {
                    Some(Value::Object(details)) => count(details.get("cached_tokens"))?,
                    _ => None,
                };
                Ok(usage_map(vec![
                    ("input_tokens", count(raw.get("prompt_tokens"))?),
                    ("cached_input_tokens", cached),
                    ("output_tokens", count(raw.get("completion_tokens"))?),
                ]))
            }
        }
    }

    fn filtered_request(&self, status: u16, code: Option<&str>) -> bool {
        self.kind.is_chat() && status == 400 && matches!(code, Some("content_filter" | "invalid_prompt"))
    }

    fn request_id(&self, response: &HttpResponse) -> Option<String> {
        for header in ["x-request-id", "request-id", "apim-request-id", "x-goog-request-id"] {
            let value = response.header(header).unwrap_or_default();
            let leaks = self.api_key.as_deref().is_some_and(|key| !key.is_empty() && value.contains(key));
            if token_like(value, 128, &['.', '_', ':', '-']) && !leaks {
                return Some(value.to_string());
            }
        }
        None
    }

    fn error_code(response: &HttpResponse) -> Option<String> {
        let body = response.parse_json()?;
        let Value::Object(error) = body.get("error")? else { return None };
        let code = match error.get("code") {
            Some(Value::String(code)) => Some(Value::String(code.clone())),
            _ => [error.get("type"), error.get("status")].into_iter().flatten().find(|v| pyjson::truthy(v)).cloned(),
        };
        match code {
            Some(Value::String(code)) if token_like(&code, 64, &['_', '.', '-']) => Some(code),
            _ => None,
        }
    }

    fn fail(&self, message: String, response: &HttpResponse, attempts: u32) -> Error {
        Error::Provider(ProviderError {
            message,
            status_code: Some(response.status),
            request_id: self.request_id(response),
            attempts: Some(attempts),
            provider_code: None,
        })
    }
}

fn retry_delay(response: &HttpResponse, attempt: u32) -> f64 {
    let fallback = 2f64.powi(attempt as i32);
    let delay =
        response.header("retry-after").map_or(fallback, |value| value.trim().parse::<f64>().unwrap_or(fallback));
    if delay.is_finite() {
        delay.clamp(0.0, 30.0)
    } else {
        fallback.min(30.0)
    }
}

impl Generator for Provider {
    fn generate(&self, messages: &[Message], tools: &[Value], history: &[Value]) -> Result<Proposal> {
        let payload = self.payload(messages, tools, history)?;
        let request = HttpRequest {
            method: Method::Post,
            url: self.url(),
            headers: self.headers(),
            body: serde_json::to_vec(&payload)
                .map_err(|_| Error::invalid("The request could not be encoded as JSON."))?,
            timeout: self.timeout,
        };
        let name = self.kind.name();
        let started = Instant::now();
        let mut waited = 0.0;
        let mut attempts = 0u32;
        let mut last = None;
        for attempt in 0..=self.max_retries {
            attempts += 1;
            match self.transport.send(&request) {
                Err(error) => {
                    if attempt == self.max_retries {
                        let mut failure = ProviderError::new(format!(
                            "{name} transport failed after {attempts} attempts ({}).",
                            error.kind
                        ));
                        failure.attempts = Some(attempts);
                        return Err(Error::Provider(failure));
                    }
                    waited += wait(&self.sleep, 2f64.powi(attempt as i32).min(8.0));
                }
                Ok(response) => {
                    let retryable = matches!(response.status, 408 | 429) || response.status >= 500;
                    if retryable && attempt < self.max_retries {
                        waited += wait(&self.sleep, retry_delay(&response, attempt));
                        continue;
                    }
                    last = Some(response);
                    break;
                }
            }
        }
        let response = last.expect("the loop ends with a response or returns");
        let latency = (started.elapsed().as_secs_f64() - waited).max(0.0);
        let mut data = Map::new();
        let mut usage = Usage::new();
        let decoded = if !response.is_success() {
            let code = Self::error_code(&response);
            if !self.filtered_request(response.status, code.as_deref()) {
                let detail = code.as_ref().map(|c| format!(" ({c})")).unwrap_or_default();
                return Err(Error::Provider(ProviderError {
                    message: format!("{name} HTTP {}{detail}; check the model, key and endpoint.", response.status),
                    status_code: Some(response.status),
                    request_id: self.request_id(&response),
                    attempts: Some(attempts),
                    provider_code: code,
                }));
            }
            unreadable("prompt_filter")
        } else {
            let protocol = || format!("The {name} response did not match its protocol.");
            data = match response.parse_json() {
                Some(Value::Object(map)) => map,
                Some(_) => return Err(self.fail("The response was not a JSON object.".into(), &response, attempts)),
                None => return Err(self.fail(protocol(), &response, attempts)),
            };
            let outcome = self.decode(&data).and_then(|decoded| Ok((decoded, self.usage(&data)?)));
            match outcome {
                Ok((decoded, counted)) => {
                    usage = counted;
                    decoded
                }
                Err(Fail::Provider(message)) => return Err(self.fail(message, &response, attempts)),
                Err(Fail::Protocol) => return Err(self.fail(protocol(), &response, attempts)),
            }
        };
        let (calls, parse_errors, content, incomplete) = decoded;
        let resolved = match data.get("model") {
            Some(value) if pyjson::truthy(value) => Some(value),
            _ => data.get("modelVersion"),
        };
        Ok(Proposal {
            calls,
            content,
            parse_errors,
            incomplete,
            provider: name.to_string(),
            requested_model: self.model.clone(),
            model: resolved.and_then(Value::as_str).map(str::to_string),
            usage,
            latency_s: latency,
            retry_wait_s: waited,
            api_calls: attempts,
        })
    }
}

fn decode_chat(data: &Map<String, Value>) -> std::result::Result<Decoded, Fail> {
    if data.get("error").is_some_and(|e| !e.is_null()) {
        return Err(Fail::Provider("The response contained a provider error.".into()));
    }
    let choice = match data.get("choices") {
        Some(Value::Array(choices)) => match choices.first() {
            Some(Value::Object(choice)) => choice,
            _ => return Err(Fail::Provider("The response did not contain a choices list.".into())),
        },
        _ => return Err(Fail::Provider("The response did not contain a choices list.".into())),
    };
    let finish = hashable(choice.get("finish_reason"))?.and_then(Value::as_str);
    if let Some(reason @ ("length" | "content_filter")) = finish {
        return Ok(unreadable(reason));
    }
    let message = match choice.get("message") {
        Some(Value::Object(message)) if message.get("role").is_none_or(|role| role.as_str() == Some("assistant")) => {
            message
        }
        _ => return Err(Fail::Provider("The response did not contain an assistant message.".into())),
    };
    if message.get("refusal").is_some_and(pyjson::truthy) {
        return Ok(unreadable("refusal"));
    }
    let (mut calls, mut errors, mut text) = (Vec::new(), Vec::new(), Vec::new());
    let native = match message.get("tool_calls") {
        Some(value) if pyjson::truthy(value) => match value {
            Value::Array(items) => items.as_slice(),
            _ => return Err(Fail::Provider("The response tool_calls must be a list.".into())),
        },
        _ => &[],
    };
    for item in native {
        let Some(Value::Object(function)) = item.get("function") else {
            return Err(Fail::Provider("The response contained an invalid function-call object.".into()));
        };
        let default = json!("{}");
        let raw = function.get("arguments").unwrap_or(&default);
        append_call(&mut calls, &mut errors, function.get("name"), Some(raw), true)?;
    }
    if native.is_empty() {
        if let Some(legacy) = message.get("function_call").filter(|value| !value.is_null()) {
            let Some(function) = legacy.as_object() else {
                return Err(Fail::Provider("The response contained an invalid function-call object.".into()));
            };
            let default = json!("{}");
            append_call(
                &mut calls,
                &mut errors,
                function.get("name"),
                Some(function.get("arguments").unwrap_or(&default)),
                true,
            )?;
        }
    }
    match message.get("content") {
        Some(Value::String(content)) if !content.is_empty() => text.push(content.clone()),
        None | Some(Value::Null) | Some(Value::String(_)) => {}
        Some(_) => return Err(Fail::Provider("The message content was not text.".into())),
    }
    Ok(complete(calls, errors, text))
}

fn decode_anthropic(data: &Map<String, Value>) -> std::result::Result<Decoded, Fail> {
    if data.get("type").and_then(Value::as_str) == Some("error") || data.get("error").is_some_and(|e| !e.is_null()) {
        return Err(Fail::Provider("The response contained a provider error.".into()));
    }
    let blocks = match (data.get("role").and_then(Value::as_str), data.get("content")) {
        (Some("assistant"), Some(Value::Array(blocks))) => blocks,
        _ => return Err(Fail::Provider("The Anthropic response was not an assistant message.".into())),
    };
    let reason = hashable(data.get("stop_reason"))?.and_then(Value::as_str);
    if matches!(reason, Some("max_tokens" | "model_context_window_exceeded")) {
        return Ok(unreadable("length"));
    }
    let refused_details = matches!(data.get("stop_details"), Some(Value::Object(d)) if d.get("type").and_then(Value::as_str) == Some("refusal"));
    if reason == Some("refusal") || refused_details {
        return Ok(unreadable("refusal"));
    }
    let (mut calls, mut errors, mut text) = (Vec::new(), Vec::new(), Vec::new());
    for block in blocks {
        let Value::Object(block) = block else {
            return Err(Fail::Provider("An Anthropic content block was malformed.".into()));
        };
        match block.get("type").and_then(Value::as_str) {
            Some("tool_use") => append_call(&mut calls, &mut errors, block.get("name"), block.get("input"), false)?,
            Some("text") => {
                if let Some(Value::String(content)) = block.get("text") {
                    if !content.is_empty() {
                        text.push(content.clone());
                    }
                }
            }
            _ => {}
        }
    }
    Ok(complete(calls, errors, text))
}

const GEMINI_FILTERED: [&str; 7] =
    ["SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "LANGUAGE"];

fn decode_gemini(data: &Map<String, Value>) -> std::result::Result<Decoded, Fail> {
    if data.get("error").is_some_and(|e| !e.is_null()) {
        return Err(Fail::Provider("The response contained a provider error.".into()));
    }
    let candidates = data.get("candidates");
    if !candidates.is_some_and(pyjson::truthy) {
        let blocked = matches!(data.get("promptFeedback"), Some(Value::Object(f)) if f.get("blockReason").is_some_and(pyjson::truthy));
        if blocked {
            return Ok(unreadable("prompt_filter"));
        }
        return Err(Fail::Provider("The Gemini response had no candidates.".into()));
    }
    let Some(Value::Array(candidates)) = candidates else { return Err(Fail::Protocol) };
    let Some(Value::Object(candidate)) = candidates.first() else { return Err(Fail::Protocol) };
    let reason = hashable(candidate.get("finishReason"))?.and_then(Value::as_str);
    if candidate.get("finishReason").is_some_and(|value| !value.is_null() && !value.is_string()) {
        return Ok(unreadable("abnormal_stop"));
    }
    match reason {
        Some("MAX_TOKENS") => return Ok(unreadable("length")),
        Some(reason) if GEMINI_FILTERED.contains(&reason) => return Ok(unreadable("content_filter")),
        Some("MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL") => return Ok(unreadable("malformed_call")),
        Some(reason) if reason != "STOP" && reason != "FINISH_REASON_UNSPECIFIED" => {
            return Ok(unreadable("abnormal_stop"))
        }
        _ => {}
    }
    if reason != Some("STOP") && !candidate.get("content").is_some_and(Value::is_object) {
        return Ok(unreadable("abnormal_stop"));
    }
    let content = match candidate.get("content") {
        Some(value) if pyjson::truthy(value) => match value {
            Value::Object(content) => Some(content),
            _ => return Err(Fail::Protocol),
        },
        _ => None,
    };
    let parts: &[Value] = match content.and_then(|c| c.get("parts")) {
        Some(Value::Array(parts)) => parts,
        // Python iterates a string's characters or an object's keys; neither is a part.
        Some(value @ (Value::String(_) | Value::Object(_))) if pyjson::truthy(value) => {
            return Err(Fail::Provider("A Gemini content part was malformed.".into()));
        }
        Some(value) if pyjson::truthy(value) => return Err(Fail::Protocol),
        _ => &[],
    };
    let (mut calls, mut errors, mut text) = (Vec::new(), Vec::new(), Vec::new());
    for part in parts {
        let Value::Object(part) = part else {
            return Err(Fail::Provider("A Gemini content part was malformed.".into()));
        };
        if part.get("functionCall").is_some_and(|value| !value.is_object()) {
            return Ok(unreadable("malformed_call"));
        }
        if let Some(Value::Object(call)) = part.get("functionCall") {
            append_call(&mut calls, &mut errors, call.get("name"), call.get("args"), false)?;
        } else if let Some(Value::String(content)) = part.get("text") {
            if !content.is_empty() && !part.get("thought").is_some_and(pyjson::truthy) {
                text.push(content.clone());
            }
        }
    }
    Ok(complete(calls, errors, text))
}

// Typed providers and their builders -----------------------------------------------------------

mod sealed {
    pub trait Sealed {}
}

/// A provider type that [`ProviderBuilder`] can build.
pub trait ProviderType: sealed::Sealed + Sized {
    /// The protocol and defaults of this type.
    const KIND: ProviderKind;
    #[doc(hidden)]
    fn wrap(provider: Provider) -> Self;
}

/// A provider that speaks Chat Completions (OpenAI, OpenAI-compatible, xAI, OpenRouter, Meta, Azure).
pub trait ChatProvider: ProviderType {}

/// Builds a provider. Every setter is optional; `build` checks the options.
pub struct ProviderBuilder<P> {
    model: String,
    options: ProviderOptions,
    marker: PhantomData<fn() -> P>,
}

impl<P: ProviderType> ProviderBuilder<P> {
    fn new(model: impl Into<String>) -> Self {
        ProviderBuilder { model: model.into(), options: ProviderOptions::default(), marker: PhantomData }
    }

    /// The API key (otherwise the provider's environment variable).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.options.api_key = Some(key.into());
        self
    }

    /// The base URL (otherwise the environment variable or the provider's default).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.options.base_url = Some(url.into());
        self
    }

    /// The output token limit (4096 by default); `None` sends none where the API allows it.
    pub fn max_output_tokens(mut self, tokens: Option<u32>) -> Self {
        self.options.max_output_tokens = tokens;
        self
    }

    /// The sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.options.temperature = Some(temperature);
        self
    }

    /// Replace the system prompt (the paper's [`SYSTEM_PROMPT`] by default).
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.options.system_prompt = Some(prompt.into());
        self
    }

    /// The time limit of one HTTP request (120 seconds).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.options.timeout = timeout;
        self
    }

    /// HTTP retries after the first attempt (3).
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.options.max_retries = retries;
        self
    }

    /// One more request header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.headers.push((name.into(), value.into()));
        self
    }

    /// The HTTP transport (for tests, proxies or another client).
    pub fn transport(mut self, transport: SharedTransport) -> Self {
        self.options.transport = Some(transport);
        self
    }

    /// How to wait between retries.
    pub fn sleep(mut self, sleep: Sleep) -> Self {
        self.options.sleep = Some(sleep);
        self
    }

    /// Check the options and build the provider.
    pub fn build(self) -> Result<P> {
        Provider::new(P::KIND, &self.model, self.options).map(P::wrap)
    }
}

impl<P: ChatProvider> ProviderBuilder<P> {
    /// `reasoning_effort`, such as `"none"` or `"low"`.
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.options.reasoning_effort = Some(effort.into());
        self
    }

    /// `parallel_tool_calls` (`Some(false)` by default, nothing for Meta; `None` leaves it out).
    pub fn parallel_tool_calls(mut self, parallel: Option<bool>) -> Self {
        self.options.parallel_tool_calls = Some(parallel);
        self
    }

    /// The name of the output token limit field (`max_completion_tokens` or `max_tokens`).
    pub fn max_tokens_field(mut self, field: impl Into<String>) -> Self {
        self.options.max_tokens_field = Some(field.into());
        self
    }

    /// The path after the base URL (`/chat/completions`).
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.options.path = path.into();
        self
    }
}

impl ProviderBuilder<AzureOpenAI> {
    /// The resource endpoint, such as `https://my-resource.openai.azure.com`.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.options.endpoint = Some(endpoint.into());
        self
    }

    /// The API version, such as `2024-12-01-preview`.
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.options.api_version = Some(version.into());
        self
    }
}

impl ProviderBuilder<Gemini> {
    /// `thinkingConfig.thinkingBudget`.
    pub fn thinking_budget(mut self, budget: i64) -> Self {
        self.options.thinking_budget = Some(budget);
        self
    }
}

macro_rules! provider_type {
    ($(#[$doc:meta])* $name:ident, $kind:ident, chat = $chat:tt) => {
        $(#[$doc])*
        pub struct $name(Provider);

        impl $name {
            /// A builder for this provider and model.
            pub fn builder(model: impl Into<String>) -> ProviderBuilder<$name> {
                ProviderBuilder::new(model)
            }

            /// This provider with every other option from the environment or the defaults.
            pub fn new(model: impl Into<String>) -> Result<$name> {
                Self::builder(model).build()
            }

            /// The underlying HTTP provider.
            pub fn provider(&self) -> &Provider {
                &self.0
            }
        }

        impl sealed::Sealed for $name {}

        impl ProviderType for $name {
            const KIND: ProviderKind = ProviderKind::$kind;
            fn wrap(provider: Provider) -> Self {
                $name(provider)
            }
        }

        provider_type!(@chat $name, $chat);

        impl Deref for $name {
            type Target = Provider;
            fn deref(&self) -> &Provider {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }

        impl Generator for $name {
            fn generate(&self, messages: &[Message], tools: &[Value], history: &[Value]) -> Result<Proposal> {
                self.0.generate(messages, tools, history)
            }
        }
    };
    (@chat $name:ident, true) => {
        impl ChatProvider for $name {}
    };
    (@chat $name:ident, false) => {};
}

provider_type! {
    /// OpenAI Chat Completions. Key: `OPENAI_API_KEY` (base URL: `OPENAI_BASE_URL`, optional).
    OpenAI, OpenAI, chat = true
}
provider_type! {
    /// Any Chat Completions endpoint with function tools: vLLM, Ollama, LM Studio, Together, Groq, ...
    /// It needs a base URL and no key.
    OpenAICompatible, OpenAICompatible, chat = true
}
provider_type! {
    /// xAI Grok, OpenAI-compatible. Key: `XAI_API_KEY`.
    XAI, XAI, chat = true
}
provider_type! {
    /// OpenRouter, one OpenAI-compatible endpoint for many models. Key: `OPENROUTER_API_KEY`.
    OpenRouter, OpenRouter, chat = true
}
provider_type! {
    /// The Meta Model API (Muse models such as `muse-spark-1.3`), OpenAI-compatible. Key: `MODEL_API_KEY`.
    ///
    /// Muse models are reasoning models: Meta rejects `reasoning_effort: "none"` for them, and
    /// `tool_choice` other than `"auto"`. This provider therefore sends neither `reasoning_effort`
    /// nor `parallel_tool_calls` unless you set them.
    Meta, Meta, chat = true
}
provider_type! {
    /// An Azure OpenAI deployment, directly or behind an API Management gateway.
    /// Key: `AZURE_OPENAI_API_KEY`, endpoint: `AZURE_OPENAI_ENDPOINT`, version: `AZURE_OPENAI_API_VERSION`.
    /// The model name is the deployment name.
    AzureOpenAI, AzureOpenAI, chat = true
}
provider_type! {
    /// Anthropic Messages. Key: `ANTHROPIC_API_KEY` (base URL: `ANTHROPIC_BASE_URL`, optional).
    Anthropic, Anthropic, chat = false
}
provider_type! {
    /// Google Gemini (`generateContent` with function declarations). Key: `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
    Gemini, Gemini, chat = false
}

/// The provider names `provider_from_spec` accepts and the kind each builds.
pub const PROVIDERS: [(&str, ProviderKind); 10] = [
    ("anthropic", ProviderKind::Anthropic),
    ("azure", ProviderKind::AzureOpenAI),
    ("claude", ProviderKind::Anthropic),
    ("gemini", ProviderKind::Gemini),
    ("google", ProviderKind::Gemini),
    ("grok", ProviderKind::XAI),
    ("meta", ProviderKind::Meta),
    ("openai", ProviderKind::OpenAI),
    ("openrouter", ProviderKind::OpenRouter),
    ("xai", ProviderKind::XAI),
];

/// A provider from `"provider:model"`, such as `"openai:gpt-5-nano"` or `"azure:<deployment>"`,
/// with keys and endpoints from the environment.
pub fn provider_from_spec(spec: &str) -> Result<Provider> {
    provider_from_spec_with(spec, ProviderOptions::default())
}

/// [`provider_from_spec`] with options.
pub fn provider_from_spec_with(spec: &str, options: ProviderOptions) -> Result<Provider> {
    let (name, model) = match spec.split_once(':') {
        Some((name, model)) if !model.is_empty() => (name, model),
        _ => {
            return Err(Error::invalid(
                "Use \"provider:model\", e.g. \"openai:gpt-5-nano\", \"anthropic:claude-sonnet-4-5\", \
                 \"gemini:gemini-2.5-flash\", \"xai:grok-4\", \"meta:muse-spark-1.3\" or \"azure:<deployment>\".",
            ))
        }
    };
    let lowered = name.to_lowercase();
    let Some((_, kind)) = PROVIDERS.iter().find(|(known, _)| *known == lowered) else {
        let names = PROVIDERS.iter().map(|(known, _)| *known).collect::<Vec<_>>().join(", ");
        return Err(Error::invalid(format!(
            "Unknown provider {}; choose from {names}, or build OpenAICompatible with a base_url for any other endpoint.",
            pyjson::repr_str(name)
        )));
    };
    Provider::new(*kind, model, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_are_checked_like_python() {
        assert_eq!(
            normalize_base_url(" https://proxy.example.com/v1/ ").as_deref(),
            Some("https://proxy.example.com/v1")
        );
        for bad in ["https://user:pw@example.com", "ftp://x", "https://x/?a=1", "https://x/#f", "https:/x", ""] {
            assert_eq!(normalize_base_url(bad), None, "{bad}");
        }
        assert_eq!(hostname("http://[::1]:8000").as_deref(), Some("::1"));
        assert_eq!(hostname("http://LocalHost:8000/").as_deref(), Some("localhost"));
        assert_eq!(quote("my deployment/é"), "my%20deployment%2F%C3%A9");
    }
}
