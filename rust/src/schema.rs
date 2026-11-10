//! The local schema check S(c) of the paper.
//!
//! S(c) is true when the candidate is readable, names an available function and supplies every
//! required argument with its declared JSON type. Value constraints (enum, ranges, lengths,
//! patterns, formats) are not checked here: whether a value fits the request and the parameter
//! description is the decision model's question. A value equal to a parameter's declared
//! default is admissible even when its JSON type differs (for example `null`), and, as in a
//! Python call, keyword names outside a closed parameter list are rejected. An empty
//! collection (no call) passes.
//!
//! This is a type-only JSON Schema validator written to give the same errors, in the same
//! words and order, as Python's `jsonschema` package on the keywords `type`, `properties`,
//! `required`, `items`, `prefixItems`, `additionalProperties`, `patternProperties`, `anyOf`,
//! `allOf`, `oneOf`, `$ref` (to `#...` JSON pointers), `$defs` and `definitions`.
//!
//! A tool's parameters are checked and reduced to their types once and kept by their JSON
//! text, so every later request and attempt reuses the result. A `$ref` must resolve inside
//! the parameters. Nothing is fetched, and a reference that only leads back to its own
//! schema is refused, when the tool is defined and by every check, even of an empty call list.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};

use regex::Regex;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::pyjson::{self, Num};

/// One schema-check error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SchemaError {
    /// `"function"` or `"arguments"`: which part of the candidate failed.
    pub check: String,
    /// Where, for example `calls[0].arguments["city"]`.
    pub path: String,
    /// What is wrong, in Python `jsonschema` wording.
    pub message: String,
}

impl SchemaError {
    fn new(check: &str, path: impl Into<String>, message: impl Into<String>) -> Self {
        SchemaError { check: check.to_string(), path: path.into(), message: message.into() }
    }

    /// The error as `{"check", "path", "message"}`.
    pub fn to_value(&self) -> Value {
        pyjson::object(vec![
            ("check", Value::String(self.check.clone())),
            ("path", Value::String(self.path.clone())),
            ("message", Value::String(self.message.clone())),
        ])
    }
}

pub(crate) fn errors_value(errors: &[SchemaError]) -> Value {
    Value::Array(errors.iter().map(SchemaError::to_value).collect())
}

/// The outcome of S(c).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SchemaCheck {
    /// No function-selection error.
    pub function_pass: bool,
    /// No argument error.
    pub arguments_pass: bool,
    /// S(c): both passed.
    pub accepted: bool,
    /// The candidate is readable, so the decision model can judge it.
    pub well_formed: bool,
    /// Every error, in order.
    pub errors: Vec<SchemaError>,
    /// The feedback record of the check.
    pub feedback: Value,
}

impl SchemaCheck {
    /// The check as the Python reference's dict.
    pub fn to_value(&self) -> Value {
        pyjson::object(vec![
            ("function_pass", Value::Bool(self.function_pass)),
            ("arguments_pass", Value::Bool(self.arguments_pass)),
            ("accepted", Value::Bool(self.accepted)),
            ("well_formed", Value::Bool(self.well_formed)),
            ("errors", errors_value(&self.errors)),
            ("feedback", self.feedback.clone()),
        ])
    }
}

/// The parameters of a tool that declares none.
static NO_PARAMETERS: LazyLock<Value> = LazyLock::new(|| json!({"type": "object", "properties": {}}));

/// Each tool's name and definition object, in order. Names must be nonempty and unique.
fn named_definitions(tools: &[Value]) -> Result<Vec<(&str, &Map<String, Value>)>> {
    let mut definitions: Vec<(&str, &Map<String, Value>)> = Vec::with_capacity(tools.len());
    for tool in tools {
        let Value::Object(map) = tool else {
            return Err(Error::invalid("Every tool must be a function definition object"));
        };
        let Value::Object(definition) = map.get("function").unwrap_or(tool) else {
            return Err(Error::invalid("Every tool.function must be an object"));
        };
        let name = match definition.get("name") {
            Some(Value::String(name)) if !name.is_empty() && !definitions.iter().any(|(known, _)| known == name) => {
                name
            }
            _ => return Err(Error::invalid("Available function names must be nonempty and unique")),
        };
        definitions.push((name, definition));
    }
    Ok(definitions)
}

/// The observable part of each tool: name, description and parameters.
///
/// Any other field of a tool definition stays out of what the decision model sees.
pub fn function_definitions(tools: &[Value]) -> Result<Vec<Value>> {
    let definitions = named_definitions(tools)?
        .into_iter()
        .map(|(name, definition)| {
            pyjson::object(vec![
                ("name", Value::String(name.to_string())),
                ("description", definition.get("description").cloned().unwrap_or_else(|| Value::String(String::new()))),
                ("parameters", definition.get("parameters").unwrap_or(&NO_PARAMETERS).clone()),
            ])
        })
        .collect();
    Ok(definitions)
}

// Drafts -------------------------------------------------------------------------------------

/// The JSON Schema draft named by `$schema` (2020-12 when absent or unknown).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Draft {
    D4,
    D6,
    D7,
    D2019,
    D2020,
}

pub(crate) fn draft_of(schema: &Map<String, Value>) -> Result<Draft> {
    let Some(uri) = schema.get("$schema") else { return Ok(Draft::D2020) };
    let Value::String(uri) = uri else {
        return Err(Error::invalid("An available function has an invalid JSON Schema"));
    };
    let uri = uri.strip_suffix('#').unwrap_or(uri);
    let lowered = match uri.split_once("://") {
        Some((scheme, rest)) => format!("{}://{}", scheme.to_ascii_lowercase(), rest),
        None => uri.to_string(),
    };
    Ok(match lowered.as_str() {
        "http://json-schema.org/draft-03/schema" | "http://json-schema.org/draft-04/schema" => Draft::D4,
        "http://json-schema.org/draft-06/schema" => Draft::D6,
        "http://json-schema.org/draft-07/schema" => Draft::D7,
        "https://json-schema.org/draft/2019-09/schema" => Draft::D2019,
        _ => Draft::D2020,
    })
}

const SIMPLE_TYPES: [&str; 7] = ["array", "boolean", "integer", "null", "number", "object", "string"];

fn is_integer(value: &Value, draft: Draft) -> bool {
    match value {
        Value::Number(n) => match pyjson::num(n) {
            Num::Int(_) => true,
            Num::Float(f) => draft >= Draft::D6 && f.is_finite() && f.fract() == 0.0,
        },
        _ => false,
    }
}

fn is_type(value: &Value, kind: &str, draft: Draft) -> Option<bool> {
    Some(match kind {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => is_integer(value, draft),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        "any" if draft == Draft::D4 => true,
        _ => return None,
    })
}

// The meta-schema check ------------------------------------------------------------------------

/// Whether `schema` is a valid JSON Schema of `draft`, for the keywords that matter here.
///
/// This follows the official meta-schemas for the shape of every standard keyword; unknown
/// keywords are allowed, as in the meta-schemas. `pattern` strings are not compiled.
pub(crate) fn valid_schema(schema: &Value, draft: Draft) -> bool {
    meta(schema, draft, 0)
}

fn meta(schema: &Value, draft: Draft, depth: usize) -> bool {
    if depth > 256 {
        return false;
    }
    let object = match schema {
        Value::Object(object) => object,
        Value::Bool(_) => return draft >= Draft::D6,
        _ => return false,
    };
    let sub = |value: &Value| meta(value, draft, depth + 1);
    let schema_map = |value: &Value| value.as_object().is_some_and(|map| map.values().all(sub));
    let schema_list = |value: &Value| value.as_array().is_some_and(|items| !items.is_empty() && items.iter().all(sub));
    let nonnegative = |value: &Value| is_integer(value, draft) && pyjson::as_number(value).is_some_and(|n| n >= 0.0);
    let strings = |value: &Value, nonempty: bool| {
        value.as_array().is_some_and(|items| {
            (!nonempty || !items.is_empty())
                && items.iter().all(Value::is_string)
                && items.iter().enumerate().all(|(i, item)| !items[..i].contains(item))
        })
    };
    for (key, value) in object {
        let ok = match key.as_str() {
            "$schema" | "$ref" | "$dynamicRef" | "$recursiveRef" | "$comment" | "title" | "description" | "format"
            | "contentEncoding" | "contentMediaType" | "pattern" => value.is_string(),
            "$id" => value
                .as_str()
                .is_some_and(|id| draft < Draft::D2019 || id.find('#').is_none_or(|at| at == id.len() - 1)),
            "id" if draft == Draft::D4 => value.is_string(),
            "$anchor" | "$dynamicAnchor" => value.as_str().is_some_and(|anchor| {
                let mut chars = anchor.chars();
                chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | ':'))
            }),
            "$recursiveAnchor" => value.is_boolean(),
            "$vocabulary" => value.as_object().is_some_and(|map| map.values().all(Value::is_boolean)),
            "$defs" | "definitions" | "properties" | "dependentSchemas" => schema_map(value),
            "patternProperties" => {
                schema_map(value)
                    && value.as_object().is_some_and(|map| map.keys().all(|pattern| Regex::new(pattern).is_ok()))
            }
            "additionalProperties"
            | "additionalItems"
            | "contains"
            | "propertyNames"
            | "not"
            | "if"
            | "then"
            | "else"
            | "unevaluatedItems"
            | "unevaluatedProperties"
            | "contentSchema" => sub(value),
            "items" => sub(value) || (draft < Draft::D2020 && schema_list(value)),
            "prefixItems" | "allOf" | "anyOf" | "oneOf" => schema_list(value),
            "type" => match value {
                Value::String(kind) => SIMPLE_TYPES.contains(&kind.as_str()),
                Value::Array(kinds) => {
                    strings(value, true)
                        && kinds.iter().all(|kind| kind.as_str().is_some_and(|k| SIMPLE_TYPES.contains(&k)))
                }
                _ => false,
            },
            "enum" => value.as_array().is_some_and(|items| draft != Draft::D4 || !items.is_empty()),
            "required" => strings(value, draft == Draft::D4),
            "multipleOf" => pyjson::as_number(value).is_some_and(|n| n > 0.0),
            "maximum" | "minimum" => value.is_number(),
            "exclusiveMaximum" | "exclusiveMinimum" => {
                if draft == Draft::D4 {
                    value.is_boolean()
                } else {
                    value.is_number()
                }
            }
            "maxLength" | "minLength" | "maxItems" | "minItems" | "maxProperties" | "minProperties" | "maxContains"
            | "minContains" => nonnegative(value),
            "uniqueItems" | "readOnly" | "writeOnly" | "deprecated" => value.is_boolean(),
            "examples" => draft == Draft::D4 || value.is_array(),
            "dependentRequired" => value.as_object().is_some_and(|map| map.values().all(|v| strings(v, false))),
            "dependencies" => value.as_object().is_some_and(|map| map.values().all(|v| sub(v) || strings(v, false))),
            _ => true,
        };
        if !ok {
            return false;
        }
    }
    true
}

// The type-only schema ----------------------------------------------------------------------------

const TYPE_KEYWORDS: [&str; 16] = [
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "definitions",
    "type",
    "properties",
    "required",
    "items",
    "prefixItems",
    "additionalProperties",
    "patternProperties",
    "anyOf",
    "allOf",
    "oneOf",
    "default",
];

/// A schema's types, required names and nesting without value constraints.
fn type_schema(schema: &Value) -> Value {
    match schema {
        Value::Array(items) => Value::Array(items.iter().map(type_schema).collect()),
        Value::Object(map) => {
            let mut result = Map::new();
            for (key, value) in map {
                if !TYPE_KEYWORDS.contains(&key.as_str()) {
                    continue;
                }
                match (key.as_str(), value) {
                    ("properties" | "patternProperties" | "$defs" | "definitions", Value::Object(entries)) => {
                        let mapped = entries.iter().map(|(name, item)| (name.clone(), type_schema(item))).collect();
                        result.insert(key.clone(), Value::Object(mapped));
                    }
                    ("items" | "prefixItems" | "additionalProperties", _) => {
                        result.insert(key.clone(), type_schema(value));
                    }
                    ("anyOf" | "allOf" | "oneOf", Value::Array(items)) => {
                        let branches: Vec<Value> = items.iter().map(type_schema).collect();
                        let all = result.entry("allOf").or_insert_with(|| Value::Array(Vec::new()));
                        if let Value::Array(all) = all {
                            if key == "allOf" {
                                all.extend(branches);
                            } else {
                                // Without value constraints, oneOf branches may overlap; any branch suffices.
                                all.push(json!({"anyOf": branches}));
                            }
                        }
                    }
                    _ => {
                        result.insert(key.clone(), value.clone());
                    }
                }
            }
            Value::Object(result)
        }
        other => other.clone(),
    }
}

// The validator --------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Step {
    Key(String),
    Index(usize),
}

struct Issue<'a> {
    path: Vec<Step>,
    message: String,
    keyword: &'static str,
    schema: &'a Value,
    instance: &'a Value,
}

struct Validator<'a> {
    root: &'a Value,
    draft: Draft,
    regexes: &'a HashMap<String, Regex>,
}

impl<'a> Validator<'a> {
    fn new(root: &'a Value, draft: Draft, regexes: &'a HashMap<String, Regex>) -> Self {
        Validator { root, draft, regexes }
    }

    fn matches(&self, pattern: &str, text: &str) -> bool {
        self.regexes.get(pattern).is_some_and(|regex| regex.is_match(text))
    }

    fn issue(
        path: &[Step],
        message: String,
        keyword: &'static str,
        schema: &'a Value,
        instance: &'a Value,
    ) -> Issue<'a> {
        Issue { path: path.to_vec(), message, keyword, schema, instance }
    }

    /// `stepped` says the caller added this instance's key or index to `path`, as Python
    /// jsonschema's `descend(path=...)` does.
    fn validate(
        &self,
        instance: &'a Value,
        schema: &'a Value,
        path: &mut Vec<Step>,
        out: &mut Vec<Issue<'a>>,
        depth: usize,
        stepped: bool,
    ) -> Result<()> {
        if depth > 512 {
            return Err(Error::invalid("An available function's JSON Schema nests or refers too deeply"));
        }
        let object = match schema {
            Value::Bool(true) => return Ok(()),
            Value::Bool(false) => {
                // jsonschema reports a false subschema at its parent's path.
                let parent = if stepped { &path[..path.len() - 1] } else { &path[..] };
                let message = format!("False schema does not allow {}", pyjson::repr(instance));
                out.push(Self::issue(parent, message, "false", schema, instance));
                return Ok(());
            }
            Value::Object(object) => object,
            _ => return Ok(()),
        };
        let keywords: Vec<(&String, &'a Value)> = match object.get_key_value("$ref") {
            Some(reference) if self.draft <= Draft::D7 => vec![reference],
            _ => object.iter().collect(),
        };
        for (key, value) in keywords {
            match key.as_str() {
                "type" => {
                    let kinds: Vec<&Value> = match value {
                        Value::Array(kinds) => kinds.iter().collect(),
                        other => vec![other],
                    };
                    let mut matched = false;
                    for kind in &kinds {
                        let known = kind.as_str().and_then(|k| is_type(instance, k, self.draft));
                        match known {
                            Some(true) => matched = true,
                            Some(false) => {}
                            None => return Err(Error::invalid("An available function has an invalid JSON Schema")),
                        }
                    }
                    if !matched {
                        let names = kinds.iter().map(|kind| pyjson::repr(kind)).collect::<Vec<_>>().join(", ");
                        let message = format!("{} is not of type {}", pyjson::repr(instance), names);
                        out.push(Self::issue(path, message, "type", schema, instance));
                    }
                }
                "properties" => {
                    if let (Value::Object(target), Value::Object(properties)) = (instance, value) {
                        for (name, subschema) in properties {
                            if let Some(item) = target.get(name) {
                                path.push(Step::Key(name.clone()));
                                self.validate(item, subschema, path, out, depth + 1, true)?;
                                path.pop();
                            }
                        }
                    }
                }
                "required" => {
                    if let (Value::Object(target), Value::Array(required)) = (instance, value) {
                        for name in required {
                            let present = name.as_str().is_some_and(|n| target.contains_key(n));
                            if !present {
                                let message = format!("{} is a required property", pyjson::repr(name));
                                out.push(Self::issue(path, message, "required", schema, instance));
                            }
                        }
                    }
                }
                "items" => {
                    let Value::Array(items) = instance else { continue };
                    if self.draft == Draft::D2020 {
                        let prefix = object.get("prefixItems").and_then(Value::as_array).map_or(0, Vec::len);
                        if items.len() <= prefix {
                            continue;
                        }
                        let extra = items.len() - prefix;
                        if value == &Value::Bool(false) {
                            let rest =
                                if extra != 1 { Value::Array(items[prefix..].to_vec()) } else { items[prefix].clone() };
                            let noun = if prefix != 1 { "items" } else { "item" };
                            let message = format!(
                                "Expected at most {prefix} {noun} but found {extra} extra: {}",
                                pyjson::repr(&rest)
                            );
                            out.push(Self::issue(path, message, "items", schema, instance));
                        } else {
                            for (index, item) in items.iter().enumerate().skip(prefix) {
                                path.push(Step::Index(index));
                                self.validate(item, value, path, out, depth + 1, true)?;
                                path.pop();
                            }
                        }
                    } else if let Value::Array(schemas) = value {
                        for (index, (item, subschema)) in items.iter().zip(schemas).enumerate() {
                            path.push(Step::Index(index));
                            self.validate(item, subschema, path, out, depth + 1, true)?;
                            path.pop();
                        }
                    } else {
                        for (index, item) in items.iter().enumerate() {
                            path.push(Step::Index(index));
                            self.validate(item, value, path, out, depth + 1, true)?;
                            path.pop();
                        }
                    }
                }
                "prefixItems" if self.draft == Draft::D2020 => {
                    if let (Value::Array(items), Value::Array(schemas)) = (instance, value) {
                        for (index, (item, subschema)) in items.iter().zip(schemas).enumerate() {
                            path.push(Step::Index(index));
                            self.validate(item, subschema, path, out, depth + 1, true)?;
                            path.pop();
                        }
                    }
                }
                "additionalProperties" => {
                    let Value::Object(target) = instance else { continue };
                    let properties = object.get("properties").and_then(Value::as_object);
                    let patterns: Vec<&String> = object
                        .get("patternProperties")
                        .and_then(Value::as_object)
                        .map_or(Vec::new(), |p| p.keys().collect());
                    let extras: Vec<&String> = target
                        .keys()
                        .filter(|name| !properties.is_some_and(|p| p.contains_key(*name)))
                        .filter(|name| !patterns.iter().any(|pattern| self.matches(pattern, name)))
                        .collect();
                    if value.is_object() {
                        for name in extras {
                            path.push(Step::Key(name.clone()));
                            self.validate(&target[name.as_str()], value, path, out, depth + 1, true)?;
                            path.pop();
                        }
                    } else if !pyjson::truthy(value) && !extras.is_empty() {
                        let mut sorted = extras.clone();
                        sorted.sort();
                        let joined = sorted.iter().map(|name| pyjson::repr_str(name)).collect::<Vec<_>>().join(", ");
                        let message = if object.contains_key("patternProperties") {
                            let mut names = patterns.clone();
                            names.sort();
                            let regexes = names.iter().map(|p| pyjson::repr_str(p)).collect::<Vec<_>>().join(", ");
                            let verb = if extras.len() == 1 { "does" } else { "do" };
                            format!("{joined} {verb} not match any of the regexes: {regexes}")
                        } else {
                            let verb = if extras.len() == 1 { "was" } else { "were" };
                            format!("Additional properties are not allowed ({joined} {verb} unexpected)")
                        };
                        out.push(Self::issue(path, message, "additionalProperties", schema, instance));
                    }
                }
                "patternProperties" => {
                    if let (Value::Object(target), Value::Object(patterns)) = (instance, value) {
                        for (pattern, subschema) in patterns {
                            for (name, item) in target {
                                if self.matches(pattern, name) {
                                    path.push(Step::Key(name.clone()));
                                    self.validate(item, subschema, path, out, depth + 1, true)?;
                                    path.pop();
                                }
                            }
                        }
                    }
                }
                "allOf" => {
                    if let Value::Array(branches) = value {
                        for branch in branches {
                            self.validate(instance, branch, path, out, depth + 1, false)?;
                        }
                    }
                }
                "anyOf" | "oneOf" => {
                    // The type-only schema turns oneOf into anyOf; any branch suffices.
                    if let Value::Array(branches) = value {
                        let mut valid = false;
                        for branch in branches {
                            let mut errors = Vec::new();
                            self.validate(instance, branch, &mut path.clone(), &mut errors, depth + 1, false)?;
                            if errors.is_empty() {
                                valid = true;
                                break;
                            }
                        }
                        if !valid {
                            let message =
                                format!("{} is not valid under any of the given schemas", pyjson::repr(instance));
                            out.push(Self::issue(path, message, "anyOf", schema, instance));
                        }
                    }
                }
                "$ref" => {
                    let Value::String(reference) = value else { continue };
                    let target = self.resolve(reference)?;
                    self.validate(instance, target, path, out, depth + 1, false)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// A `$ref` within the same document: `#`, a JSON pointer `#/...`, optionally after the
    /// root's own `$id`.
    fn resolve(&self, reference: &str) -> Result<&'a Value> {
        let unresolvable = || Error::invalid(UNRESOLVABLE);
        let (base, fragment) = reference.split_once('#').unwrap_or((reference, ""));
        if !base.is_empty() {
            let own = self.root.get("$id").and_then(Value::as_str).map(|id| id.trim_end_matches('#'));
            if own != Some(base) {
                return Err(unresolvable());
            }
        }
        let fragment = percent_decode(fragment).ok_or_else(unresolvable)?;
        if fragment.is_empty() {
            return Ok(self.root);
        }
        let Some(pointer) = fragment.strip_prefix('/') else { return Err(unresolvable()) };
        let mut target = self.root;
        for token in pointer.split('/') {
            let token = token.replace("~1", "/").replace("~0", "~");
            target = match target {
                Value::Object(map) => map.get(&token),
                Value::Array(items) => token.parse::<usize>().ok().and_then(|index| items.get(index)),
                _ => None,
            }
            .ok_or_else(unresolvable)?;
        }
        if !target.is_object() && !target.is_boolean() {
            return Err(unresolvable());
        }
        Ok(target)
    }
}

fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let decoded = (bytes[index] == b'%' && index + 2 < bytes.len())
            .then(|| std::str::from_utf8(&bytes[index + 1..index + 3]).ok())
            .flatten()
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        if let Some(byte) = decoded {
            out.push(byte);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn collect_patterns(schema: &Value, regexes: &mut HashMap<String, Regex>) -> Result<()> {
    match schema {
        Value::Object(map) => {
            for (key, value) in map {
                match key.as_str() {
                    "properties" | "$defs" | "definitions" | "patternProperties" => {
                        if let Value::Object(named) = value {
                            for (name, child) in named {
                                if key == "patternProperties" && !regexes.contains_key(name) {
                                    let regex = Regex::new(name).map_err(|_| {
                                        Error::invalid("An available function has an invalid JSON Schema")
                                    })?;
                                    regexes.insert(name.clone(), regex);
                                }
                                collect_patterns(child, regexes)?;
                            }
                        }
                    }
                    "items" | "prefixItems" | "additionalProperties" | "anyOf" | "allOf" => {
                        collect_patterns(value, regexes)?
                    }
                    _ => {}
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_patterns(item, regexes)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// `str(list(path))` of a Python `jsonschema` error, the sort key of the errors.
fn path_key(path: &[Step]) -> String {
    let parts: Vec<String> = path
        .iter()
        .map(|step| match step {
            Step::Key(name) => pyjson::repr_str(name),
            Step::Index(index) => index.to_string(),
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// A value equal to the parameter's own declared default is admissible.
fn declared_default(issue: &Issue) -> bool {
    if issue.keyword != "type" {
        return false;
    }
    match issue.schema.get("default") {
        Some(default) => pyjson::dumps_sorted(issue.instance) == pyjson::dumps_sorted(default),
        None => false,
    }
}

const UNRESOLVABLE: &str = "An available function's JSON Schema has a $ref that cannot be resolved";

struct Prepared {
    original: Map<String, Value>,
    typed: Value,
    draft: Draft,
    regexes: HashMap<String, Regex>,
}

static PREPARED: LazyLock<Mutex<HashMap<String, Arc<Prepared>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Check every schema reference, even if no current candidate reaches it. A cycle that
/// reapplies schemas to the same value cannot terminate; nested properties/items can recurse.
fn check_references(validator: &Validator<'_>) -> Result<()> {
    fn follow(
        node: &Value,
        validator: &Validator<'_>,
        schemas: &HashSet<usize>,
        state: &mut HashMap<usize, bool>,
        depth: usize,
    ) -> Result<()> {
        if depth > 512 {
            return Err(Error::invalid(UNRESOLVABLE));
        }
        let Some(map) = node.as_object() else { return Ok(()) };
        let key = node as *const Value as usize;
        match state.get(&key) {
            Some(true) => return Err(Error::invalid(UNRESOLVABLE)),
            Some(false) => return Ok(()),
            None => {}
        }
        state.insert(key, true);
        if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
            let target = validator.resolve(reference)?;
            if target.is_object() && !schemas.contains(&(target as *const Value as usize)) {
                return Err(Error::invalid(UNRESOLVABLE));
            }
            if target.is_boolean() && !valid_schema(target, validator.draft) {
                return Err(Error::invalid(UNRESOLVABLE));
            }
            follow(target, validator, schemas, state, depth + 1)?;
            if validator.draft <= Draft::D7 {
                state.insert(key, false);
                return Ok(());
            }
        }
        for keyword in ["allOf", "anyOf"] {
            if let Some(items) = map.get(keyword).and_then(Value::as_array) {
                for item in items {
                    follow(item, validator, schemas, state, depth + 1)?;
                }
            }
        }
        state.insert(key, false);
        Ok(())
    }
    fn visit<'a>(node: &'a Value, schemas: &mut Vec<&'a Value>, depth: usize) -> Result<()> {
        if depth > 512 {
            return Err(Error::invalid(UNRESOLVABLE));
        }
        match node {
            Value::Array(items) => {
                for item in items {
                    visit(item, schemas, depth + 1)?;
                }
            }
            Value::Object(map) => {
                schemas.push(node);
                for (key, value) in map {
                    match key.as_str() {
                        "properties" | "patternProperties" | "$defs" | "definitions" => {
                            if let Some(named) = value.as_object() {
                                for child in named.values() {
                                    visit(child, schemas, depth + 1)?;
                                }
                            }
                        }
                        "items" | "prefixItems" | "additionalProperties" | "allOf" | "anyOf" => {
                            visit(value, schemas, depth + 1)?
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    let mut schemas = Vec::new();
    visit(validator.root, &mut schemas, 0)?;
    let known = schemas.iter().map(|node| *node as *const Value as usize).collect();
    let mut state = HashMap::new();
    for node in schemas {
        follow(node, validator, &known, &mut state, 0)?;
    }
    Ok(())
}

fn prepare(parameters: &Value) -> Result<Arc<Prepared>> {
    let Value::Object(map) = parameters else {
        return Err(Error::invalid("Function parameters must be a JSON Schema object"));
    };
    let text = serde_json::to_string(parameters).map_err(|_| Error::invalid("Invalid JSON Schema"))?;
    if let Some(prepared) = PREPARED.lock().unwrap().get(&text) {
        return Ok(prepared.clone());
    }
    let draft = draft_of(map)?;
    if !valid_schema(parameters, draft) {
        return Err(Error::invalid("An available function has an invalid JSON Schema"));
    }
    let typed = type_schema(parameters);
    let mut regexes = HashMap::new();
    collect_patterns(&typed, &mut regexes)?;
    check_references(&Validator::new(&typed, draft, &regexes))?;
    let prepared = Arc::new(Prepared { original: map.clone(), typed, draft, regexes });
    let mut cache = PREPARED.lock().unwrap();
    if cache.len() >= 1024 {
        cache.clear();
    }
    cache.insert(text, prepared.clone());
    Ok(prepared)
}

/// Validate and compile one tool's parameters once; errors are configuration errors.
pub(crate) fn check_parameters(parameters: &Value) -> Result<Draft> {
    Ok(prepare(parameters)?.draft)
}

/// Run S(c) on proposed `calls` (`[{"name", "arguments"}]`, or `null` for an unreadable
/// candidate) against the available `tools`.
///
/// Invalid tool definitions are an error: they are configuration errors, not a property of
/// the candidate.
pub fn check_schema(tools: &[Value], calls: &Value) -> Result<SchemaCheck> {
    let definitions = named_definitions(tools)?;
    let mut prepared: Vec<(&str, Arc<Prepared>)> = Vec::with_capacity(definitions.len());
    for (name, definition) in definitions {
        let parameters = definition.get("parameters").unwrap_or(&NO_PARAMETERS);
        prepared.push((name, prepare(parameters)?));
    }
    let find = |name: &str| prepared.iter().find(|(known, _)| *known == name).map(|(_, p)| p);

    let mut errors: Vec<SchemaError> = Vec::new();
    let mut well_formed = calls.is_array();
    match calls {
        Value::Array(calls) => {
            for (index, call) in calls.iter().enumerate() {
                let path = format!("calls[{index}]");
                let Value::Object(call) = call else {
                    well_formed = false;
                    errors.push(SchemaError::new("function", &path, "Each call must be an object."));
                    errors.push(SchemaError::new("arguments", &path, "No argument object to inspect."));
                    continue;
                };
                let name = match call.get("name") {
                    Some(Value::String(name)) if !name.is_empty() => Some(name.as_str()),
                    _ => None,
                };
                match name {
                    None => {
                        well_formed = false;
                        errors.push(SchemaError::new(
                            "function",
                            format!("{path}.name"),
                            "Name must be a nonempty string.",
                        ));
                    }
                    Some(name) if find(name).is_none() => {
                        errors.push(SchemaError::new(
                            "function",
                            format!("{path}.name"),
                            "Selected function is not available.",
                        ));
                    }
                    _ => {}
                }
                let arguments_path = format!("{path}.arguments");
                let Some(arguments @ Value::Object(argument_map)) = call.get("arguments") else {
                    well_formed = false;
                    errors.push(SchemaError::new("arguments", &arguments_path, "Arguments must be a JSON object."));
                    continue;
                };
                if !pyjson::all_finite(arguments) {
                    well_formed = false;
                    errors.push(SchemaError::new(
                        "arguments",
                        &arguments_path,
                        "Arguments must be finite JSON values.",
                    ));
                    continue;
                }
                let Some(tool) = name.and_then(find) else {
                    errors.push(SchemaError::new(
                        "arguments",
                        &arguments_path,
                        "Cannot validate arguments for an unavailable function.",
                    ));
                    continue;
                };
                let validator = Validator::new(&tool.typed, tool.draft, &tool.regexes);
                let mut issues = Vec::new();
                validator.validate(arguments, &tool.typed, &mut Vec::new(), &mut issues, 0, false)?;
                let mut keyed: Vec<(String, Issue)> =
                    issues.into_iter().map(|issue| (path_key(&issue.path), issue)).collect();
                keyed.sort_by(|a, b| a.0.cmp(&b.0));
                for (_, issue) in keyed {
                    if declared_default(&issue) {
                        continue;
                    }
                    let suffix: String = issue
                        .path
                        .iter()
                        .map(|step| match step {
                            Step::Key(name) => format!("[{}]", pyjson::dumps_str(name, true)),
                            Step::Index(index) => format!("[{index}]"),
                        })
                        .collect();
                    errors.push(SchemaError::new("arguments", format!("{arguments_path}{suffix}"), issue.message));
                }
                // An ordinary function signature has closed top-level keyword names.
                // Respect an explicit additionalProperties or patternProperties rule.
                let schema = &tool.original;
                if let Some(Value::Object(properties)) = schema.get("properties") {
                    if !schema.contains_key("additionalProperties") && !schema.contains_key("patternProperties") {
                        let mut unknown: Vec<&String> =
                            argument_map.keys().filter(|key| !properties.contains_key(*key)).collect();
                        unknown.sort();
                        for name in unknown {
                            errors.push(SchemaError::new(
                                "arguments",
                                &arguments_path,
                                format!("Unknown parameter: {name}."),
                            ));
                        }
                    }
                }
            }
        }
        _ => {
            errors.push(SchemaError::new("function", "calls", "Proposed calls must be a list."));
            errors.push(SchemaError::new("arguments", "calls", "No valid call collection to inspect."));
        }
    }

    let function_pass = !errors.iter().any(|error| error.check == "function");
    let arguments_pass = !errors.iter().any(|error| error.check == "arguments");
    let feedback = pyjson::object(vec![
        ("source", Value::String("deterministic_schema_validation".to_string())),
        ("function_pass", Value::Bool(function_pass)),
        ("arguments_pass", Value::Bool(arguments_pass)),
        ("schema_errors", errors_value(&errors)),
    ]);
    Ok(SchemaCheck {
        function_pass,
        arguments_pass,
        accepted: function_pass && arguments_pass,
        well_formed,
        errors,
        feedback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_must_target_a_schema_not_a_data_value() {
        for schema in [
            json!({"$schema": "http://json-schema.org/draft-04/schema#", "type": "object", "properties": {"x": {"$ref": "#/default"}}, "default": true}),
            json!({"type": "object", "properties": {"x": {"$ref": "#/type"}}}),
            json!({"type": "object", "properties": {"x": {"$ref": "#/default"}}, "default": {"type": "PREMIUM"}}),
            json!({"type": "object", "properties": {"x": {"$ref": "#/default"}}, "default": {"type": "string"}}),
            json!({"type": "object", "properties": {"x": {"$ref": "#/properties"}, "type": {"type": "integer"}}}),
            json!({"type": "object", "properties": {"x": {"$ref": "#/default"}}, "default": {"type": "object", "properties": {"child": {"$ref": "#/missing"}}}}),
        ] {
            assert!(check_parameters(&schema).is_err());
            assert!(crate::Tool::function("f", "", schema.clone()).is_err());
            assert!(check_schema(&[json!({"name": "f", "parameters": schema})], &json!([])).is_err());
        }
    }

    #[test]
    fn compiled_parameters_are_reused_without_inspecting_default_data() {
        let schema = json!({"type": "object", "properties": {
            "payload": {"type": "object", "default": {"patternProperties": {"[": {}}}}
        }});
        let first = prepare(&schema).unwrap();
        let second = prepare(&schema).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    fn weather() -> Value {
        json!({"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}})
    }

    fn check(arguments: Value, tools: &[Value], name: &str) -> SchemaCheck {
        check_schema(tools, &json!([{"name": name, "arguments": arguments}])).unwrap()
    }

    #[test]
    fn valid_call_and_no_call_pass() {
        assert!(check(json!({"city": "Seoul"}), &[weather()], "get_weather").accepted);
        let empty = check_schema(&[weather()], &json!([])).unwrap();
        assert!(empty.accepted && empty.well_formed && empty.errors.is_empty());
    }

    #[test]
    fn argument_structure_errors_and_unavailable_functions() {
        for arguments in [json!({}), json!({"city": "Seoul", "invented": 1}), json!({"city": 3})] {
            let result = check(arguments, &[weather()], "get_weather");
            assert!(result.function_pass && !result.arguments_pass && result.well_formed);
        }
        let result = check(json!({}), &[weather()], "invented");
        assert!(!result.function_pass && !result.arguments_pass && result.well_formed);
        let unreadable = [
            json!(null),
            json!(["get_weather"]),
            json!([{"name": "get_weather", "arguments": "{}"}]),
            json!([{"name": "", "arguments": {}}]),
        ];
        for calls in unreadable {
            let result = check_schema(&[weather()], &calls).unwrap();
            assert!(!result.accepted && !result.well_formed);
        }
    }

    #[test]
    fn alternatives_and_open_keywords() {
        let tools = [json!({"type": "function", "function": {"name": "f", "description": "", "parameters": {
            "type": "object", "additionalProperties": false, "required": ["x"], "properties": {
                "x": {"oneOf": [{"type": "string", "enum": ["a"]}, {"type": "string", "maxLength": 1}]}}}}})];
        assert!(check(json!({"x": "abc"}), &tools, "f").accepted);
        for arguments in [json!({"x": 3}), json!({}), json!({"x": "a", "y": 1})] {
            assert!(!check(arguments, &tools, "f").accepted);
        }
    }

    #[test]
    fn pattern_properties_drafts_and_refs() {
        let tools = [json!({"type": "function", "function": {"name": "f", "description": "", "parameters": {
            "type": "object", "patternProperties": {"^x-": {"type": "integer"}}, "additionalProperties": false}}})];
        let result = check(json!({"x-a": 1, "x-b": "no", "other": 1}), &tools, "f");
        let messages: Vec<&str> = result.errors.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(messages, ["'no' is not of type 'integer'", "'other' does not match any of the regexes: '^x-'"]);
        let draft4 = [json!({"type": "function", "function": {"name": "f", "description": "", "parameters": {
            "$schema": "http://json-schema.org/draft-04/schema#", "type": "object",
            "properties": {"n": {"type": "integer"}, "pair": {"type": "array", "items": [{"type": "string"}]}}}}})];
        let result = check(json!({"n": 2.0, "pair": [1]}), &draft4, "f");
        let messages: Vec<&str> = result.errors.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(messages, ["2.0 is not of type 'integer'", "1 is not of type 'string'"]);
        let bad_ref = [json!({"type": "function", "function": {"name": "f", "description": "",
            "parameters": {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}}}})];
        assert!(check_schema(&bad_ref, &json!([{"name": "f", "arguments": {"a": 1}}])).is_err());
        let nonsense = [json!({"type": "function", "function": {"name": "f",
            "parameters": {"type": "object", "properties": {"a": {"type": "nonsense"}}}}})];
        assert!(check_schema(&nonsense, &json!([])).is_err());
    }

    #[test]
    fn unicode_paths_are_ascii_escaped() {
        let tools = [json!({"type": "function", "function": {"name": "f", "description": "",
            "parameters": {"type": "object", "properties": {"도시": {"type": "string"}}}}})];
        let result = check(json!({"도시": 1}), &tools, "f");
        assert_eq!(result.errors[0].path, "calls[0].arguments[\"\\ub3c4\\uc2dc\"]");
    }
}
