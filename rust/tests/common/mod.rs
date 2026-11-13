//! Shared test helpers. Every test is offline: HTTP goes through a mock transport.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

use dmcall::pyjson;
use dmcall::{CallResult, HttpRequest, HttpResponse, HttpTransport, Message, Sleep, TransportError};
use serde_json::{json, Map, Value};

pub const KEY_NAMES: [&str; 17] = [
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MODEL_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "TYPESAFE_API_KEY",
    "TYPESAFE_BASE_URL",
    "TYPESAFE_MODEL",
];

/// Remove every key and endpoint variable once per test process, so the developer's shell
/// cannot change what the tests send.
pub fn isolate_env() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        for name in KEY_NAMES {
            std::env::remove_var(name);
        }
    });
}

pub fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("spec")
}

/// A spec file parsed with key order and number kinds kept.
pub fn load(relative: &str) -> Value {
    let text = std::fs::read_to_string(spec_dir().join(relative)).expect("spec file");
    pyjson::loads(&text).expect("valid JSON")
}

/// Python `==`, with a readable failure.
#[track_caller]
pub fn assert_py_eq(actual: &Value, expected: &Value, what: &str) {
    assert!(
        pyjson::equal(actual, expected),
        "{what} differs\n  actual:   {}\n  expected: {}",
        pyjson::dumps(actual),
        pyjson::dumps(expected)
    );
}

pub fn no_sleep() -> Sleep {
    Arc::new(|_: Duration| {})
}

type Responder = Box<dyn Fn(&HttpRequest) -> Result<HttpResponse, TransportError> + Send + Sync>;

/// A transport that records every request and answers from a queue or a function.
pub struct Mock {
    pub requests: Mutex<Vec<HttpRequest>>,
    responder: Responder,
}

impl Mock {
    pub fn with<F>(responder: F) -> Arc<Mock>
    where
        F: Fn(&HttpRequest) -> Result<HttpResponse, TransportError> + Send + Sync + 'static,
    {
        Arc::new(Mock { requests: Mutex::new(Vec::new()), responder: Box::new(responder) })
    }

    /// Serve these JSON bodies with status 200, in order.
    pub fn queue(bodies: Vec<Value>) -> Arc<Mock> {
        let queue = Mutex::new(VecDeque::from(bodies));
        Mock::with(move |_| {
            let body = queue.lock().unwrap().pop_front().expect("an unexpected extra request");
            Ok(HttpResponse::json(200, &body))
        })
    }

    /// Serve these responses in order.
    pub fn responses(responses: Vec<HttpResponse>) -> Arc<Mock> {
        let queue = Mutex::new(VecDeque::from(responses));
        Mock::with(move |_| Ok(queue.lock().unwrap().pop_front().expect("an unexpected extra request")))
    }

    pub fn sent(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub fn bodies(&self) -> Vec<Value> {
        self.sent().iter().map(body_of).collect()
    }
}

impl HttpTransport for Mock {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        self.requests.lock().unwrap().push(request.clone());
        (self.responder)(request)
    }
}

pub fn body_of(request: &HttpRequest) -> Value {
    pyjson::loads(std::str::from_utf8(&request.body).unwrap()).unwrap()
}

/// The headers the spec records, with lower-case names.
pub fn listed_headers(request: &HttpRequest) -> Value {
    const LISTED: [&str; 6] =
        ["authorization", "x-api-key", "api-key", "x-goog-api-key", "anthropic-version", "content-type"];
    let mut map = Map::new();
    for (name, value) in &request.headers {
        let lower = name.to_ascii_lowercase();
        if LISTED.contains(&lower.as_str()) {
            map.insert(lower, Value::String(value.clone()));
        }
    }
    Value::Object(map)
}

pub fn messages_of(value: &Value) -> Vec<Message> {
    value.as_array().unwrap().iter().map(|m| serde_json::from_value(m.clone()).unwrap()).collect()
}

fn optional(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |p| serde_json::Number::from_f64(p).map_or(Value::Null, Value::Number))
}

pub fn calls_json(calls: &Option<Vec<dmcall::Call>>) -> Value {
    match calls {
        None => Value::Null,
        Some(calls) => Value::Array(calls.iter().map(dmcall::Call::to_value).collect()),
    }
}

pub fn attempts_json(result: &CallResult) -> Value {
    Value::Array(
        result
            .attempts()
            .iter()
            .map(|attempt| {
                json!({
                    "calls": calls_json(&attempt.calls),
                    "accepted": attempt.accepted,
                    "function_probability": optional(attempt.function_probability),
                    "arguments_probability": optional(attempt.arguments_probability),
                    "schema_errors": attempt.schema_errors.iter().map(|e| e.to_value()).collect::<Vec<_>>(),
                })
            })
            .collect(),
    )
}

/// The Python reference's result summary.
pub fn summary(result: &CallResult) -> Value {
    let probabilities = result.probabilities();
    let usage = result.usage();
    json!({
        "calls": calls_json(&result.calls),
        "accepted": result.accepted,
        "exhausted": result.exhausted,
        "repairs": result.repairs,
        "final_attempt": result.final_attempt,
        "text": result.text,
        "initial_no_call_preserved": result.initial_no_call_preserved,
        "attempts": attempts_json(result),
        "probabilities": {"function": optional(probabilities.function), "arguments": optional(probabilities.arguments)},
        "usage": {"generator": Value::Object(usage.generator), "decision_model": Value::Object(usage.decision_model)},
    })
}

/// Only the keys present in `expected`.
pub fn subset(actual: &Value, expected: &Value) -> Value {
    let mut map = Map::new();
    for key in expected.as_object().unwrap().keys() {
        map.insert(key.clone(), actual.get(key).cloned().unwrap_or(Value::Null));
    }
    Value::Object(map)
}
