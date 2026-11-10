//! Tools as JSON Schema definitions, each optionally paired with a handler, and calls from any SDK.
//!
//! A tool is a function the model may call: a name, a description and a JSON Schema for its
//! keyword arguments. [`Tool::new`] accepts OpenAI (Chat Completions or Responses), Anthropic
//! (`input_schema`) and Gemini (`parameters` with upper-case types, or `parametersJsonSchema`)
//! definitions and stores them in the OpenAI Chat Completions shape.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::error::{BoxError, Error, Result};
use crate::message::Call;
use crate::pyjson;
use crate::schema::check_parameters;

/// What runs an accepted call: it receives the call's arguments object and returns the output.
pub type Handler = Arc<dyn Fn(&Value) -> std::result::Result<Value, BoxError> + Send + Sync>;

/// A tool definition with an optional handler.
#[derive(Clone)]
pub struct Tool {
    definition: Value,
    handler: Option<Handler>,
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool").field("name", &self.name()).field("has_handler", &self.handler.is_some()).finish()
    }
}

impl Tool {
    /// A tool from an OpenAI, Anthropic or Gemini function definition.
    ///
    /// ```
    /// let tool = dmcall::Tool::new(serde_json::json!({
    ///     "name": "get_weather",
    ///     "description": "Current weather for a city.",
    ///     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    /// })).unwrap();
    /// assert_eq!(tool.name(), "get_weather");
    /// ```
    pub fn new(definition: Value) -> Result<Tool> {
        Ok(Tool { definition: schema_of(&definition)?, handler: None })
    }

    /// A tool from its name, description and parameter schema.
    pub fn function(name: &str, description: &str, parameters: Value) -> Result<Tool> {
        Tool::new(json!({"name": name, "description": description, "parameters": parameters}))
    }

    /// Attach the function that runs an accepted call.
    pub fn with_handler<F>(mut self, handler: F) -> Tool
    where
        F: Fn(&Value) -> std::result::Result<Value, BoxError> + Send + Sync + 'static,
    {
        self.handler = Some(Arc::new(handler));
        self
    }

    /// The tool's name.
    pub fn name(&self) -> &str {
        self.definition["function"]["name"].as_str().unwrap_or_default()
    }

    /// The definition as `{"type": "function", "function": {"name", "description", "parameters"}}`.
    pub fn definition(&self) -> &Value {
        &self.definition
    }

    /// The handler, if any.
    pub fn handler(&self) -> Option<&Handler> {
        self.handler.as_ref()
    }
}

/// Gemini's OpenAPI-style schemas spell types in capitals (OBJECT, STRING); JSON Schema does not.
fn lower_types(schema: &Value) -> Value {
    fn upper(text: &str) -> bool {
        text.chars().any(char::is_alphabetic) && !text.chars().any(char::is_lowercase)
    }
    match schema {
        Value::Array(items) => Value::Array(items.iter().map(lower_types).collect()),
        Value::Object(map) => {
            let mut result = Map::new();
            for (key, value) in map {
                let lowered = match (key.as_str(), value) {
                    ("type", Value::String(kind)) if upper(kind) => Value::String(kind.to_lowercase()),
                    ("type", Value::Array(kinds)) => Value::Array(
                        kinds
                            .iter()
                            .map(|kind| match kind {
                                Value::String(text) if upper(text) => Value::String(text.to_lowercase()),
                                other => other.clone(),
                            })
                            .collect(),
                    ),
                    (
                        "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas"
                        | "dependencies",
                        Value::Object(named),
                    ) => Value::Object(named.iter().map(|(name, item)| (name.clone(), lower_types(item))).collect()),
                    (
                        "items"
                        | "prefixItems"
                        | "additionalItems"
                        | "additionalProperties"
                        | "unevaluatedItems"
                        | "unevaluatedProperties"
                        | "contains"
                        | "propertyNames"
                        | "allOf"
                        | "anyOf"
                        | "oneOf"
                        | "not"
                        | "if"
                        | "then"
                        | "else"
                        | "contentSchema",
                        _,
                    ) => lower_types(value),
                    _ => value.clone(),
                };
                result.insert(key.clone(), lowered);
            }
            Value::Object(result)
        }
        other => other.clone(),
    }
}

/// An OpenAI-style tool definition `{"type": "function", "function": {...}}` from an OpenAI,
/// Anthropic or Gemini function definition, checked: a nonempty name, a text description and
/// a valid JSON Schema object for the parameters.
pub fn schema_of(tool: &Value) -> Result<Value> {
    let Value::Object(outer) = tool else {
        return Err(Error::invalid("A tool is a function definition object."));
    };
    let definition = match outer.get("function") {
        Some(Value::Object(inner)) => inner,
        _ => outer,
    };
    if definition.contains_key("functionDeclarations") || definition.contains_key("function_declarations") {
        return Err(Error::invalid("Pass each Gemini function declaration as its own tool."));
    }
    let parameters = ["parameters", "input_schema", "parametersJsonSchema", "parameters_json_schema"]
        .iter()
        .find_map(|key| definition.get(*key).filter(|value| !value.is_null()))
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    let name = definition.get("name").cloned().unwrap_or(Value::Null);
    let description = match definition.get("description") {
        Some(value) if pyjson::truthy(value) => value.clone(),
        _ => Value::String(String::new()),
    };
    let parameters = lower_types(&parameters);

    let name = match &name {
        Value::String(text) if !text.trim().is_empty() => text.clone(),
        _ => return Err(Error::invalid("Every tool needs a nonempty name.")),
    };
    let shown = pyjson::repr_str(&name);
    if !description.is_string() {
        return Err(Error::invalid(format!("Tool {shown}: the description must be text.")));
    }
    let object_type = match &parameters {
        Value::Object(map) => map.get("type").is_none_or(|kind| kind.as_str() == Some("object")),
        _ => false,
    };
    if !object_type {
        return Err(Error::invalid(format!(
            "Tool {shown}: parameters must be a JSON Schema object with type 'object'."
        )));
    }
    if !pyjson::all_finite(&parameters) || check_parameters(&parameters).is_err() {
        return Err(Error::invalid(format!("Tool {shown}: parameters are not a valid JSON Schema (SchemaError).")));
    }
    Ok(json!({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}))
}

/// Tool definitions for the model, and the handlers that implement them by name.
pub(crate) fn toolset(
    tools: &[Tool],
    handlers: &[(String, Handler)],
) -> Result<(Vec<Value>, HashMap<String, Handler>)> {
    let mut schemas = Vec::with_capacity(tools.len());
    let mut registry = HashMap::new();
    let mut names: Vec<&str> = Vec::new();
    for tool in tools {
        let name = tool.name();
        if names.contains(&name) {
            return Err(Error::invalid(format!("Two tools are named {}.", pyjson::repr_str(name))));
        }
        names.push(name);
        schemas.push(tool.definition.clone());
        if let Some(handler) = &tool.handler {
            registry.insert(name.to_string(), handler.clone());
        }
    }
    for (name, handler) in handlers {
        if !names.contains(&name.as_str()) {
            return Err(Error::invalid(format!("handler {} is not one of the tools.", pyjson::repr_str(name))));
        }
        registry.insert(name.clone(), handler.clone());
    }
    Ok((schemas, registry))
}

/// Calls as [`Call`]s from DM-Call, OpenAI (Chat Completions or Responses), Anthropic or
/// Gemini shapes.
///
/// `null` means no call (OpenAI reports `tool_calls: null`). Arguments that are not valid JSON
/// become `null`, which the schema check rejects. A missing or non-text name becomes `""`.
pub fn normalize_calls(calls: &Value) -> Result<Vec<Call>> {
    let items: Vec<&Value> = match calls {
        Value::Null => return Ok(Vec::new()),
        Value::Object(_) => vec![calls],
        Value::Array(items) => items.iter().collect(),
        _ => return Err(Error::invalid("calls must be a call or a list of calls.")),
    };
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        let Value::Object(item) = item else {
            return Err(Error::invalid("Each call must be an object."));
        };
        let item = ["functionCall", "function_call", "function"]
            .iter()
            .find_map(|key| item.get(*key).and_then(Value::as_object))
            .unwrap_or(item);
        let arguments = ["arguments", "input", "args"].iter().find_map(|key| item.get(*key)).cloned();
        let arguments = match arguments {
            None | Some(Value::Null) => json!({}),
            Some(Value::String(text)) if text.trim().is_empty() => json!({}),
            Some(Value::String(text)) => pyjson::loads(&text).unwrap_or(Value::Null),
            Some(other) => other,
        };
        let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
        result.push(Call::new(name, arguments));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather() -> Value {
        json!({"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}})
    }

    #[test]
    fn definition_formats() {
        let parameters = weather()["function"]["parameters"].clone();
        let mut flat = weather()["function"].clone();
        flat["type"] = json!("function");
        let forms = [
            weather(),
            weather()["function"].clone(),
            flat,
            json!({"name": "get_weather", "description": "Current weather for a city.", "input_schema": parameters}),
            json!({"name": "get_weather", "description": "Current weather for a city.", "parameters": {
                "type": "OBJECT", "properties": {"city": {"type": "STRING"},
                    "unit": {"type": "STRING", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}),
        ];
        for form in forms {
            assert!(pyjson::equal(&schema_of(&form).unwrap(), &weather()), "{form}");
        }
        let ping = schema_of(&json!({"name": "ping", "description": "Ping."})).unwrap();
        assert_eq!(pyjson::dumps(&ping["function"]["parameters"]), r#"{"type": "object", "properties": {}}"#);
    }

    #[test]
    fn bad_tools() {
        for bad in [
            json!("get_weather"),
            json!({"name": "", "parameters": {"type": "object"}}),
            json!({"name": "x", "parameters": {"type": "array"}}),
            json!({"name": "x", "parameters": {"type": "object", "properties": {"a": {"type": "nonsense"}}}}),
            json!({"name": "x", "description": 5}),
            json!({"functionDeclarations": []}),
        ] {
            assert!(Tool::new(bad.clone()).is_err(), "{bad}");
        }
        let tool = Tool::new(weather()).unwrap();
        assert!(toolset(&[tool.clone(), tool.clone()], &[]).is_err());
        let noop: Handler = Arc::new(|_| Ok(Value::Null));
        assert!(toolset(&[tool], &[("nope".into(), noop)]).is_err());
    }

    #[test]
    fn calls_from_every_sdk_shape() {
        let seoul = [Call::new("get_weather", json!({"city": "Seoul"}))];
        for shape in [
            json!({"name": "get_weather", "arguments": {"city": "Seoul"}}),
            json!({"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}),
            json!({"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}),
            json!({"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}),
            json!({"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}),
        ] {
            let calls = normalize_calls(&Value::Array(vec![shape])).unwrap();
            assert_eq!(calls.len(), 1);
            assert!(calls[0].name == seoul[0].name && pyjson::equal(&calls[0].arguments, &seoul[0].arguments));
        }
        assert!(normalize_calls(&Value::Null).unwrap().is_empty());
        assert!(normalize_calls(&json!({"name": "f", "arguments": "{oops"})).unwrap()[0].arguments.is_null());
        assert!(normalize_calls(&json!(3)).is_err());
    }
}
