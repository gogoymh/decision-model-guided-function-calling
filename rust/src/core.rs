//! DM-Call: propose a function call, verify it with a decision model, repair it with cumulative feedback.
//!
//! At step t the generator proposes a candidate c_t (one call, or no call). The candidate is
//! accepted when it passes the schema check S and every decision-model question that applies
//! to it (Eq. 2 of the paper). A rejected candidate and its feedback d_t join the history H,
//! which the generator reads, oldest first and marked as not executed, before the unchanged
//! request. The loop returns the first accepted candidate; if none of c_0..c_r is accepted it
//! returns c_0. Nothing is executed while the loop runs.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::decision::{DecisionModel, Jev, Judgment};
use crate::error::{BoxError, Error, Result};
use crate::feedback::{represent_history, single_check, Feedback};
use crate::message::{calls_value, format_call, Call, Message};
use crate::providers::{Generator, Proposal, Usage};
use crate::pyjson;
use crate::schema::{check_schema, SchemaCheck, SchemaError};
use crate::tools::{normalize_calls, toolset, Handler, Tool};

/// The paper's pass mark of both questions.
pub const DEFAULT_THRESHOLD: f64 = 0.5;
/// The paper's repair limit r*.
pub const DEFAULT_MAX_REPAIRS: u32 = 3;

const ROLES: [&str; 4] = ["system", "developer", "user", "assistant"];
const HISTORY_ROLES: [&str; 6] = ["system", "developer", "user", "assistant", "tool", "function"];
const HISTORY_KEYS: [&str; 5] = ["role", "content", "name", "tool_calls", "tool_call_id"];
const TEXT_PARTS: [&str; 3] = ["text", "input_text", "output_text"];

macro_rules! choice_enum {
    ($(#[$doc:meta])* $name:ident, $option:literal, { $($(#[$vdoc:meta])* $variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vdoc])* $variant),+
        }

        impl $name {
            /// The option's text value.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $($name::$variant => $text),+
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = Error;

            fn from_str(value: &str) -> Result<Self> {
                match value {
                    $($text => Ok($name::$variant),)+
                    _ => {
                        let choices = [$(pyjson::repr_str($text)),+].join(", ");
                        Err(Error::invalid(format!("{} must be one of {}.", $option, choices)))
                    }
                }
            }
        }
    };
}

choice_enum!(
    /// Which decision-model questions decide acceptance.
    Checks, "checks", {
        /// Both questions: DM-Call (the default).
        #[default]
        Both => "both",
        /// The function question only: DM-Call-Function.
        Function => "function",
        /// The argument question only: DM-Call-Argument.
        Arguments => "arguments",
    }
);

choice_enum!(
    /// Which earlier candidates the generator sees during repair.
    History, "history", {
        /// Every earlier candidate and its feedback: DM-Call (the default).
        #[default]
        Cumulative => "cumulative",
        /// Only the latest rejected candidate, with its real round number: DM-Call-Latest.
        Latest => "latest",
    }
);

choice_enum!(
    /// The final candidate when none is accepted within the repair limit.
    OnExhaustion, "on_exhaustion", {
        /// The first candidate c_0 (the paper's rule).
        #[default]
        First => "first",
        /// The last candidate.
        Last => "last",
    }
);

// Inputs ---------------------------------------------------------------------------------------------

/// Anything that can be a conversation: a prompt string, [`Message`]s or JSON messages.
pub trait IntoMessages {
    /// The messages as JSON objects, before validation.
    fn message_values(&self) -> Result<Vec<Value>>;
}

impl IntoMessages for str {
    fn message_values(&self) -> Result<Vec<Value>> {
        Ok(vec![Message::user(self).to_value()])
    }
}

impl IntoMessages for String {
    fn message_values(&self) -> Result<Vec<Value>> {
        self.as_str().message_values()
    }
}

impl IntoMessages for Message {
    fn message_values(&self) -> Result<Vec<Value>> {
        Ok(vec![self.to_value()])
    }
}

impl IntoMessages for [Message] {
    fn message_values(&self) -> Result<Vec<Value>> {
        Ok(self.iter().map(Message::to_value).collect())
    }
}

impl IntoMessages for Vec<Message> {
    fn message_values(&self) -> Result<Vec<Value>> {
        self.as_slice().message_values()
    }
}

impl IntoMessages for [Value] {
    fn message_values(&self) -> Result<Vec<Value>> {
        Ok(self.to_vec())
    }
}

impl IntoMessages for Vec<Value> {
    fn message_values(&self) -> Result<Vec<Value>> {
        Ok(self.clone())
    }
}

impl IntoMessages for Value {
    /// A string is one user message, an object one message and an array a list of messages.
    fn message_values(&self) -> Result<Vec<Value>> {
        match self {
            Value::String(text) => text.message_values(),
            Value::Object(_) => Ok(vec![self.clone()]),
            Value::Array(items) => Ok(items.clone()),
            _ => Err(Error::invalid("messages must be a string or a nonempty list of {'role', 'content'} messages.")),
        }
    }
}

/// Anything that can be proposed calls for [`DMCall::verify`]: [`Call`]s or JSON in any SDK shape.
pub trait IntoCalls {
    /// The calls as JSON (`null` for no call).
    fn calls_json(&self) -> Value;
}

impl IntoCalls for Value {
    fn calls_json(&self) -> Value {
        self.clone()
    }
}

impl IntoCalls for [Value] {
    fn calls_json(&self) -> Value {
        Value::Array(self.to_vec())
    }
}

impl IntoCalls for Vec<Value> {
    fn calls_json(&self) -> Value {
        Value::Array(self.clone())
    }
}

impl IntoCalls for Call {
    fn calls_json(&self) -> Value {
        Value::Array(vec![self.to_value()])
    }
}

impl IntoCalls for [Call] {
    fn calls_json(&self) -> Value {
        calls_value(Some(self))
    }
}

impl IntoCalls for Vec<Call> {
    fn calls_json(&self) -> Value {
        calls_value(Some(self))
    }
}

/// A prompt or chat messages as text [`Message`]s with roles system, developer, user or assistant.
fn text_messages(values: Vec<Value>) -> Result<Vec<Message>> {
    if values.is_empty() {
        return Err(Error::invalid("messages must be a string or a nonempty list of {'role', 'content'} messages."));
    }
    let mut result = Vec::with_capacity(values.len());
    for value in values {
        let role = value.get("role").and_then(Value::as_str).filter(|role| ROLES.contains(role));
        let (Some(role), Value::Object(message)) = (role, &value) else {
            return Err(Error::invalid(
                "Each message needs a role (system, developer, user or assistant) and text content; \
                 give earlier tool results to the model as text.",
            ));
        };
        let content = match message.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => {
                let mut texts = Vec::with_capacity(parts.len());
                for part in parts {
                    let kind = part.get("type").and_then(Value::as_str);
                    match (kind, part.get("text")) {
                        (Some(kind), Some(Value::String(text))) if TEXT_PARTS.contains(&kind) => {
                            texts.push(text.clone())
                        }
                        _ => return Err(Error::invalid("Only text message content is supported.")),
                    }
                }
                texts.join("\n")
            }
            _ => return Err(Error::invalid("Message content must be text.")),
        };
        result.push(Message::new(role, content));
    }
    Ok(result)
}

/// A conversation as the decision model sees it: text turns, earlier tool calls and tool results.
fn conversation(values: Vec<Value>) -> Result<Vec<Value>> {
    if values.is_empty() {
        return Err(Error::invalid("messages must not be empty."));
    }
    let mut result = Vec::with_capacity(values.len());
    for value in values {
        let role = value.get("role").and_then(Value::as_str);
        let Value::Object(message) = &value else {
            return Err(Error::invalid("Each message needs a role: system, developer, user, assistant or tool."));
        };
        if !role.is_some_and(|role| HISTORY_ROLES.contains(&role)) {
            return Err(Error::invalid("Each message needs a role: system, developer, user, assistant or tool."));
        }
        let mut kept = Map::new();
        for key in HISTORY_KEYS {
            if let Some(item) = message.get(key).filter(|item| !item.is_null()) {
                kept.insert(key.to_string(), item.clone());
            }
        }
        result.push(Value::Object(kept));
    }
    Ok(result)
}

// Results ---------------------------------------------------------------------------------------------

/// One step of the loop: the candidate, its judgment, the schema check and the feedback record.
#[derive(Clone, Debug)]
pub struct TraceStep {
    /// The attempt index (0 is the initial proposal).
    pub attempt: usize,
    /// The generator's output.
    pub proposal: Proposal,
    /// The decision model's judgment; `None` for an initial no call returned without verification.
    pub verification: Option<Judgment>,
    /// The schema check of the candidate.
    pub schema: SchemaCheck,
    /// The feedback record for the configured questions (with numeric scores).
    pub feedback: Value,
    /// Whether the candidate was accepted; `None` for a preserved initial no call.
    pub gate_pass: Option<bool>,
}

/// One candidate with its verdict, scores and schema errors.
#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    /// The candidate's calls (`None` if unreadable).
    pub calls: Option<Vec<Call>>,
    /// Whether it was accepted (`None` for a preserved initial no call).
    pub accepted: Option<bool>,
    /// p_f, if asked.
    pub function_probability: Option<f64>,
    /// p_a, if asked.
    pub arguments_probability: Option<f64>,
    /// The schema check's errors.
    pub schema_errors: Vec<SchemaError>,
}

/// The decision model's probabilities of yes for the final candidate (`None` if not asked).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Probabilities {
    /// p_f.
    pub function: Option<f64>,
    /// p_a.
    pub arguments: Option<f64>,
}

/// Token counts summed over every generator request and every decision-model request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsageSummary {
    /// The generator's counts.
    pub generator: Usage,
    /// The decision model's counts.
    pub decision_model: Usage,
}

/// The outcome of one DM-Call request.
///
/// `calls` is the final candidate: `Some(vec![])` is a decision to make no call and `None` an
/// unreadable candidate (cut off, filtered or refused). `accepted` says whether it passed every
/// check; it is `None` for an initial no call returned without verification
/// (`initial_no_call_preserved`). When no candidate passed within the repair limit
/// (`exhausted`), the final candidate is the first one ([`OnExhaustion::First`], the paper's
/// rule) or the last one; `final_attempt` is its index in `trace`, which keeps every candidate,
/// judgment and feedback record in order.
#[derive(Clone, Debug)]
pub struct CallResult {
    /// The final candidate.
    pub calls: Option<Vec<Call>>,
    /// Whether it passed every check (`None` for a preserved initial no call).
    pub accepted: Option<bool>,
    /// No candidate was accepted within the repair limit.
    pub exhausted: bool,
    /// Repairs made (proposals after the first).
    pub repairs: usize,
    /// The final candidate's text, if any.
    pub text: Option<String>,
    /// Whether `run` executed the final call.
    pub executed: bool,
    /// What the executed handlers returned, in call order.
    pub outputs: Vec<Value>,
    /// Seconds from start to finish.
    pub latency_s: f64,
    /// Every step, oldest first.
    pub trace: Vec<TraceStep>,
    /// The index of the final candidate in `trace`.
    pub final_attempt: usize,
    /// The initial no call was returned without verification.
    pub initial_no_call_preserved: bool,
}

fn add_counts(total: &mut Usage, counts: &Usage) {
    for (key, value) in counts {
        if pyjson::is_int(value) {
            if let Some(count) = value.as_u64() {
                let sum = total.get(key).and_then(Value::as_u64).unwrap_or(0) + count;
                total.insert(key.clone(), json!(sum));
            }
        }
    }
}

impl CallResult {
    /// The first call of the final candidate, or `None` for no call or an unreadable candidate.
    pub fn call(&self) -> Option<&Call> {
        self.calls.as_ref().and_then(|calls| calls.first())
    }

    /// The final candidate is a decision to make no call.
    pub fn no_call(&self) -> bool {
        matches!(&self.calls, Some(calls) if calls.is_empty())
    }

    /// What the executed handler returned (an array if several calls ran).
    pub fn output(&self) -> Option<Value> {
        match self.outputs.len() {
            0 => None,
            1 => Some(self.outputs[0].clone()),
            _ => Some(Value::Array(self.outputs.clone())),
        }
    }

    /// Each candidate with its verdict, scores and schema errors, oldest first.
    pub fn attempts(&self) -> Vec<Attempt> {
        self.trace
            .iter()
            .map(|step| Attempt {
                calls: step.proposal.calls.clone(),
                accepted: step.gate_pass,
                function_probability: step.verification.as_ref().and_then(|v| v.function_probability),
                arguments_probability: step.verification.as_ref().and_then(|v| v.arguments_probability),
                schema_errors: step.schema.errors.clone(),
            })
            .collect()
    }

    /// The decision model's probabilities of yes for the final candidate.
    pub fn probabilities(&self) -> Probabilities {
        match self.trace.get(self.final_attempt).and_then(|step| step.verification.as_ref()) {
            Some(check) => {
                Probabilities { function: check.function_probability, arguments: check.arguments_probability }
            }
            None => Probabilities::default(),
        }
    }

    /// Token counts summed over every generator request and every decision-model request.
    pub fn usage(&self) -> UsageSummary {
        let mut summary = UsageSummary::default();
        for step in &self.trace {
            add_counts(&mut summary.generator, &step.proposal.usage);
            if let Some(check) = &step.verification {
                add_counts(&mut summary.decision_model, &check.usage);
            }
        }
        summary
    }
}

impl fmt::Display for CallResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decision = match &self.calls {
            None => "unreadable candidate".to_string(),
            Some(calls) if calls.is_empty() => "no call".to_string(),
            Some(calls) => format!("call={}", calls.iter().map(format_call).collect::<Vec<_>>().join("; ")),
        };
        let accepted = self.accepted.map_or("none".to_string(), |a| a.to_string());
        write!(f, "CallResult({decision}, accepted={accepted}, repairs={}", self.repairs)?;
        if self.executed {
            write!(f, ", output={}", self.output().map_or("null".to_string(), |o| pyjson::dumps(&o)))?;
        }
        f.write_str(")")
    }
}

/// The schema check's and the decision model's judgment of calls from any source.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// The calls pass the schema check and every applicable question.
    pub accepted: bool,
    /// The calls, normalized.
    pub calls: Vec<Call>,
    /// p_f, if asked.
    pub function_probability: Option<f64>,
    /// p_a, if asked.
    pub arguments_probability: Option<f64>,
    /// The schema check's errors.
    pub schema_errors: Vec<SchemaError>,
    /// The feedback record.
    pub feedback: Value,
    /// The full judgment.
    pub raw: Judgment,
}

fn verdict<M, C>(decision_model: &dyn DecisionModel, messages: &M, calls: &C, tools: &[Tool]) -> Result<Verdict>
where
    M: IntoMessages + ?Sized,
    C: IntoCalls + ?Sized,
{
    let (schemas, _) = toolset(tools, &[])?;
    let conversation = conversation(messages.message_values()?)?;
    let proposed = normalize_calls(&calls.calls_json())?;
    let local = check_schema(&schemas, &calls_value(Some(&proposed)))?;
    let check = decision_model.judge(&conversation, &schemas, Some(&proposed))?;
    let accepted =
        local.accepted && check.function_pass && (check.arguments_skipped || check.arguments_pass == Some(true));
    Ok(Verdict {
        accepted,
        calls: proposed,
        function_probability: check.function_probability,
        arguments_probability: check.arguments_probability,
        schema_errors: local.errors,
        feedback: check.feedback.clone(),
        raw: check,
    })
}

/// Check calls from any source before running them; nothing is generated or executed.
///
/// `messages` is the conversation so far, including earlier tool calls and tool results.
/// `calls` may be [`Call`]s or JSON in the OpenAI, Anthropic or Gemini shapes. The calls pass
/// if they pass the schema check and the decision model's function question and, unless there
/// is no call, its argument question.
pub fn verify<M, C>(messages: &M, calls: &C, tools: &[Tool], decision_model: &dyn DecisionModel) -> Result<Verdict>
where
    M: IntoMessages + ?Sized,
    C: IntoCalls + ?Sized,
{
    verdict(decision_model, messages, calls, tools)
}

// The client --------------------------------------------------------------------------------------------

/// Per-request overrides of a [`DMCall`]'s settings.
#[derive(Clone, Default)]
pub struct CallOptions {
    /// Use these tools instead of the client's; retain handlers of the same names unless replaced.
    pub tools: Option<Vec<Tool>>,
    /// Extra handlers by tool name.
    pub handlers: Vec<(String, Handler)>,
    /// Override the repair limit.
    pub max_repairs: Option<u32>,
    /// Override the exhaustion rule.
    pub on_exhaustion: Option<OnExhaustion>,
    /// `run` only: execute the final candidate even if none was accepted.
    pub execute_unverified: Option<bool>,
}

/// Decision-model-guided function calling.
///
/// The defaults are the paper's settings: at most 3 repairs, both questions, the whole
/// history, Hybrid feedback, the first candidate when none passes, and a readable initial no
/// call returned without verification. The paper's ablations are [`Checks::Function`]
/// (DM-Call-Function), [`Checks::Arguments`] (DM-Call-Argument) and [`History::Latest`]
/// (DM-Call-Latest).
pub struct DMCall {
    generator: Box<dyn Generator>,
    decision_model: Box<dyn DecisionModel>,
    tools: Vec<Tool>,
    handlers: Vec<(String, Handler)>,
    max_repairs: u32,
    checks: Checks,
    history: History,
    feedback: Feedback,
    on_exhaustion: OnExhaustion,
    preserve_initial_no_call: bool,
    execute_unverified: bool,
}

impl fmt::Debug for DMCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DMCall")
            .field("tools", &self.tools.iter().map(Tool::name).collect::<Vec<_>>())
            .field("checks", &self.checks)
            .field("history", &self.history)
            .field("feedback", &self.feedback)
            .field("max_repairs", &self.max_repairs)
            .field("on_exhaustion", &self.on_exhaustion)
            .field("preserve_initial_no_call", &self.preserve_initial_no_call)
            .finish_non_exhaustive()
    }
}

/// Builds a [`DMCall`].
pub struct DMCallBuilder {
    generator: Box<dyn Generator>,
    decision_model: Option<Box<dyn DecisionModel>>,
    threshold: Option<f64>,
    jev_api_key: Option<String>,
    tools: Vec<Tool>,
    handlers: Vec<(String, Handler)>,
    max_repairs: u32,
    checks: Checks,
    history: History,
    feedback: Feedback,
    on_exhaustion: OnExhaustion,
    preserve_initial_no_call: bool,
    execute_unverified: bool,
}

impl DMCallBuilder {
    /// Add one tool.
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set or replace the handler of a tool by name.
    pub fn handler<F>(mut self, name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(&Value) -> std::result::Result<Value, BoxError> + Send + Sync + 'static,
    {
        self.handlers.push((name.into(), Arc::new(handler)));
        self
    }

    /// The decision model ([`Jev`] from the environment by default).
    pub fn decision_model(mut self, decision_model: impl DecisionModel + 'static) -> Self {
        self.decision_model = Some(Box::new(decision_model));
        self
    }

    /// The pass mark of both questions for the default Jev (0.5). Configure a decision model
    /// you pass yourself on that model instead.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// The key of the default Jev (otherwise `TYPESAFE_API_KEY`).
    pub fn jev_api_key(mut self, key: impl Into<String>) -> Self {
        self.jev_api_key = Some(key.into());
        self
    }

    /// At most this many repairs after the first proposal (3).
    pub fn max_repairs(mut self, repairs: u32) -> Self {
        self.max_repairs = repairs;
        self
    }

    /// Which questions decide acceptance ([`Checks::Both`]).
    pub fn checks(mut self, checks: Checks) -> Self {
        self.checks = checks;
        self
    }

    /// Which earlier candidates the generator sees ([`History::Cumulative`]).
    pub fn history(mut self, history: History) -> Self {
        self.history = history;
        self
    }

    /// How earlier feedback is shown ([`Feedback::Hybrid`]).
    pub fn feedback(mut self, feedback: Feedback) -> Self {
        self.feedback = feedback;
        self
    }

    /// The final candidate when none is accepted ([`OnExhaustion::First`]).
    pub fn on_exhaustion(mut self, rule: OnExhaustion) -> Self {
        self.on_exhaustion = rule;
        self
    }

    /// Return a readable initial no call without verification or repair (true).
    pub fn preserve_initial_no_call(mut self, preserve: bool) -> Self {
        self.preserve_initial_no_call = preserve;
        self
    }

    /// Let `run` execute the final candidate even when none was accepted (false).
    pub fn execute_unverified(mut self, execute: bool) -> Self {
        self.execute_unverified = execute;
        self
    }

    /// Check the options and build the client.
    pub fn build(self) -> Result<DMCall> {
        toolset(&self.tools, &self.handlers)?; // fail fast on a bad tool or handler name
        let decision_model = match self.decision_model {
            Some(model) => {
                if self.threshold.is_some() || self.jev_api_key.is_some() {
                    return Err(Error::invalid(
                        "Configure the decision model itself (api_key, threshold), not DMCall.",
                    ));
                }
                model
            }
            None => {
                let mut jev = Jev::builder().threshold(self.threshold.unwrap_or(DEFAULT_THRESHOLD));
                if let Some(key) = self.jev_api_key {
                    jev = jev.api_key(key);
                }
                Box::new(jev.build()?)
            }
        };
        Ok(DMCall {
            generator: self.generator,
            decision_model,
            tools: self.tools,
            handlers: self.handlers,
            max_repairs: self.max_repairs,
            checks: self.checks,
            history: self.history,
            feedback: self.feedback,
            on_exhaustion: self.on_exhaustion,
            preserve_initial_no_call: self.preserve_initial_no_call,
            execute_unverified: self.execute_unverified,
        })
    }
}

impl DMCall {
    /// A builder around a generator, such as `OpenAI::new("gpt-5-nano")?` or
    /// `provider_from_spec("anthropic:claude-sonnet-4-5")?`.
    pub fn builder(generator: impl Generator + 'static) -> DMCallBuilder {
        DMCallBuilder {
            generator: Box::new(generator),
            decision_model: None,
            threshold: None,
            jev_api_key: None,
            tools: Vec::new(),
            handlers: Vec::new(),
            max_repairs: DEFAULT_MAX_REPAIRS,
            checks: Checks::Both,
            history: History::Cumulative,
            feedback: Feedback::Hybrid,
            on_exhaustion: OnExhaustion::First,
            preserve_initial_no_call: true,
            execute_unverified: false,
        }
    }

    /// The repair limit.
    pub fn max_repairs(&self) -> u32 {
        self.max_repairs
    }

    /// The questions that decide acceptance.
    pub fn checks(&self) -> Checks {
        self.checks
    }

    /// Which earlier candidates the generator sees.
    pub fn history(&self) -> History {
        self.history
    }

    /// How earlier feedback is shown.
    pub fn feedback(&self) -> Feedback {
        self.feedback
    }

    /// The exhaustion rule.
    pub fn on_exhaustion(&self) -> OnExhaustion {
        self.on_exhaustion
    }

    /// Whether a readable initial no call is returned without verification.
    pub fn preserve_initial_no_call(&self) -> bool {
        self.preserve_initial_no_call
    }

    /// The decision model.
    pub fn decision_model(&self) -> &dyn DecisionModel {
        self.decision_model.as_ref()
    }

    /// Propose, verify and repair a call. Nothing is executed.
    pub fn call<M: IntoMessages + ?Sized>(&self, messages: &M) -> Result<CallResult> {
        self.call_with(messages, CallOptions::default())
    }

    /// [`DMCall::call`] with per-request overrides.
    pub fn call_with<M: IntoMessages + ?Sized>(&self, messages: &M, options: CallOptions) -> Result<CallResult> {
        self.solve(messages, options, false)
    }

    /// `call`, then execute the final call with its handler if it was accepted.
    ///
    /// With `execute_unverified` the final candidate also runs when none was accepted: the
    /// first one, or the last with [`OnExhaustion::Last`].
    pub fn run<M: IntoMessages + ?Sized>(&self, messages: &M) -> Result<CallResult> {
        self.run_with(messages, CallOptions::default())
    }

    /// [`DMCall::run`] with per-request overrides.
    pub fn run_with<M: IntoMessages + ?Sized>(&self, messages: &M, options: CallOptions) -> Result<CallResult> {
        self.solve(messages, options, true)
    }

    /// Check proposed calls ([`Call`]s or JSON in any SDK shape) with the schema check and the questions.
    pub fn verify<M, C>(&self, messages: &M, calls: &C) -> Result<Verdict>
    where
        M: IntoMessages + ?Sized,
        C: IntoCalls + ?Sized,
    {
        verdict(self.decision_model.as_ref(), messages, calls, &self.tools)
    }

    /// [`DMCall::verify`] against other tools.
    pub fn verify_with_tools<M, C>(&self, messages: &M, calls: &C, tools: &[Tool]) -> Result<Verdict>
    where
        M: IntoMessages + ?Sized,
        C: IntoCalls + ?Sized,
    {
        verdict(self.decision_model.as_ref(), messages, calls, tools)
    }

    /// A_t (Eq. 2) for the configured questions, and the feedback record the generator gets.
    fn accept(&self, local: &SchemaCheck, check: &Judgment) -> Result<(bool, Value)> {
        let arguments = check.arguments_skipped || check.arguments_pass == Some(true);
        Ok(match self.checks {
            Checks::Function => (local.accepted && check.function_pass, single_check(&check.feedback, "function")?),
            Checks::Arguments => (local.accepted && arguments, single_check(&check.feedback, "arguments")?),
            Checks::Both => (local.accepted && check.function_pass && arguments, check.feedback.clone()),
        })
    }

    fn solve<M: IntoMessages + ?Sized>(&self, messages: &M, options: CallOptions, execute: bool) -> Result<CallResult> {
        let limit = options.max_repairs.unwrap_or(self.max_repairs);
        let rule = options.on_exhaustion.unwrap_or(self.on_exhaustion);
        let unverified = options.execute_unverified.unwrap_or(self.execute_unverified);
        let (schemas, registry) = match &options.tools {
            None => {
                let mut handlers = self.handlers.clone();
                handlers.extend(options.handlers.iter().cloned());
                toolset(&self.tools, &handlers)?
            }
            Some(tools) => {
                let (schemas, mut registry) = toolset(tools, &options.handlers)?;
                let known = toolset(&self.tools, &self.handlers)?.1;
                for tool in tools {
                    if let Some(handler) = known.get(tool.name()) {
                        registry.entry(tool.name().to_string()).or_insert_with(|| handler.clone());
                    }
                }
                (schemas, registry)
            }
        };
        if execute {
            for schema in &schemas {
                let name = schema["function"]["name"].as_str().unwrap_or_default();
                if !registry.contains_key(name) {
                    return Err(Error::MissingHandler(name.to_string()));
                }
            }
        }
        let messages = text_messages(messages.message_values()?)?;
        let message_values: Vec<Value> = messages.iter().map(Message::to_value).collect();
        let started = Instant::now();
        let mut history: Vec<Value> = Vec::new();
        let mut trace: Vec<TraceStep> = Vec::new();
        let mut accepted = false;
        for attempt in 0..=limit as usize {
            let shown: Vec<Value> = match (self.history, history.last()) {
                // Only the latest rejection, with its real round number.
                (History::Latest, Some(last)) => {
                    let mut last = last.clone();
                    if let Value::Object(record) = &mut last {
                        record.insert("round".into(), json!(attempt));
                    }
                    vec![last]
                }
                _ => history.clone(),
            };
            let proposal = self.generator.generate(&messages, &schemas, &represent_history(&shown, self.feedback)?)?;
            let calls = calls_value(proposal.calls.as_deref());
            let local = check_schema(&schemas, &calls)?;
            if attempt == 0 && self.preserve_initial_no_call && proposal.readable_no_call() && local.accepted {
                // A readable initial no call is returned without verification or repair.
                let text = proposal.content.clone();
                trace.push(TraceStep {
                    attempt: 0,
                    proposal,
                    verification: None,
                    schema: local,
                    feedback: json!({}),
                    gate_pass: None,
                });
                return Ok(CallResult {
                    calls: Some(Vec::new()),
                    accepted: None,
                    exhausted: false,
                    repairs: 0,
                    text,
                    executed: false,
                    outputs: Vec::new(),
                    latency_s: started.elapsed().as_secs_f64(),
                    trace,
                    final_attempt: 0,
                    initial_no_call_preserved: true,
                });
            }
            let check = self.decision_model.judge(&message_values, &schemas, proposal.calls.as_deref())?;
            let (passed, feedback) = self.accept(&local, &check)?;
            accepted = passed;
            trace.push(TraceStep {
                attempt,
                proposal,
                verification: Some(check),
                schema: local,
                feedback: feedback.clone(),
                gate_pass: Some(passed),
            });
            if passed {
                break;
            }
            history.push(pyjson::object(vec![("calls", calls), ("feedback", feedback)]));
        }

        let exhausted = !accepted;
        let final_attempt = if exhausted && rule == OnExhaustion::First { 0 } else { trace.len() - 1 };
        let calls = trace[final_attempt].proposal.calls.clone();
        let mut outputs = Vec::new();
        let mut executed = false;
        if execute && (accepted || unverified) {
            executed = execute_calls(calls.as_deref(), &registry, &mut outputs)?;
        }
        Ok(CallResult {
            calls,
            accepted: Some(accepted),
            exhausted,
            repairs: trace.len() - 1,
            text: trace[final_attempt].proposal.content.clone(),
            executed,
            outputs,
            latency_s: started.elapsed().as_secs_f64(),
            trace,
            final_attempt,
            initial_no_call_preserved: false,
        })
    }
}

/// Run the final calls with their handlers. An unreadable candidate never runs.
fn execute_calls(
    calls: Option<&[Call]>,
    registry: &HashMap<String, Handler>,
    outputs: &mut Vec<Value>,
) -> Result<bool> {
    let Some(calls) = calls else { return Ok(false) };
    if !calls.iter().all(|call| call.arguments.is_object()) {
        return Ok(false);
    }
    if let Some(missing) = calls.iter().find(|call| !registry.contains_key(&call.name)) {
        return Err(Error::MissingHandler(missing.name.clone()));
    }
    for call in calls {
        let output =
            registry[&call.name](&call.arguments).map_err(|source| Error::Tool { name: call.name.clone(), source })?;
        outputs.push(output);
    }
    Ok(!calls.is_empty())
}
