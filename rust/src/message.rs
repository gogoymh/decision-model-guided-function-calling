//! Chat messages and function calls.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One text chat message. `role` is `system`, `developer`, `user` or `assistant`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// `system`, `developer`, `user` or `assistant`.
    pub role: String,
    /// The text.
    pub content: String,
}

impl Message {
    /// A message with any role.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Message { role: role.into(), content: content.into() }
    }

    /// A system message.
    pub fn system(content: impl Into<String>) -> Self {
        Message::new("system", content)
    }

    /// A developer message.
    pub fn developer(content: impl Into<String>) -> Self {
        Message::new("developer", content)
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Message::new("user", content)
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::new("assistant", content)
    }

    /// The message as `{"role", "content"}`.
    pub fn to_value(&self) -> Value {
        let mut map = Map::new();
        map.insert("role".into(), Value::String(self.role.clone()));
        map.insert("content".into(), Value::String(self.content.clone()));
        Value::Object(map)
    }
}

/// One proposed function call. `arguments` is a JSON object, or `null` when the model's
/// arguments were not valid JSON (such a call is rejected and repaired, never run).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Call {
    /// The function's name.
    pub name: String,
    /// The keyword arguments as a JSON object (`null` when unreadable).
    pub arguments: Value,
}

impl Call {
    /// A call with the given arguments.
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Call { name: name.into(), arguments }
    }

    /// The call as `{"name", "arguments"}`.
    pub fn to_value(&self) -> Value {
        let mut map = Map::new();
        map.insert("name".into(), Value::String(self.name.clone()));
        map.insert("arguments".into(), self.arguments.clone());
        Value::Object(map)
    }
}

/// Calls as a JSON value: `null` for an unreadable candidate, otherwise a list.
pub(crate) fn calls_value(calls: Option<&[Call]>) -> Value {
    match calls {
        None => Value::Null,
        Some(calls) => Value::Array(calls.iter().map(Call::to_value).collect()),
    }
}

/// `name(key=value, ...)` with JSON values, as in the Python reference's `repr`.
pub(crate) fn format_call(call: &Call) -> String {
    match &call.arguments {
        Value::Object(arguments) => format!(
            "{}({})",
            call.name,
            arguments
                .iter()
                .map(|(key, value)| format!("{}={}", key, crate::pyjson::dumps(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => format!("{}(<malformed arguments>)", call.name),
    }
}
