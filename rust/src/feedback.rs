//! Feedback representations: how the decision model's scores reach the generator (Φ in the paper).
//!
//! Only the generator's copy of earlier feedback changes. Acceptance always uses the numeric
//! probabilities, and [`crate::CallResult::trace`] keeps them unchanged.
//!
//! * [`Feedback::Hybrid`] (the default): each probability keeps its value and threshold and
//!   gains a verbal label, Φ(p) = (p, ℓ(p)).
//! * [`Feedback::Numeric`]: probabilities and thresholds only, Φ(p) = p.
//! * [`Feedback::Text`]: labels only, Φ(p) = ℓ(p); probabilities and thresholds are removed.
//!
//! ℓ uses five bands of equal width, lower bound inclusive; the last band includes 1. A
//! question that was not asked (the argument question on a no call) keeps a null score and a
//! null label.

use std::fmt;
use std::str::FromStr;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::pyjson;

/// The cut points between the five label bands.
pub const CUT_POINTS: [f64; 4] = [0.2, 0.4, 0.6, 0.8];

/// The five verbal labels, lowest band first.
pub const LABELS: [&str; 5] =
    ["very likely incorrect", "likely incorrect", "uncertain", "likely correct", "very likely correct"];

const LABEL_FIELDS: [(&str, &str); 5] = [
    ("probability_yes", "feedback_label"),
    ("function_probability", "function_label"),
    ("arguments_probability", "arguments_label"),
    ("p_f", "function_label"),
    ("p_a", "arguments_label"),
];
const THRESHOLDS: [&str; 3] = ["threshold", "function_threshold", "arguments_threshold"];

/// How earlier feedback is shown to the generator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Feedback {
    /// Probabilities, thresholds and labels (the paper's setting).
    #[default]
    Hybrid,
    /// Probabilities and thresholds only.
    Numeric,
    /// Labels only.
    Text,
}

impl Feedback {
    /// `"hybrid"`, `"numeric"` or `"text"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Feedback::Hybrid => "hybrid",
            Feedback::Numeric => "numeric",
            Feedback::Text => "text",
        }
    }
}

impl fmt::Display for Feedback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Feedback {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "hybrid" => Ok(Feedback::Hybrid),
            "numeric" => Ok(Feedback::Numeric),
            "text" => Ok(Feedback::Text),
            _ => Err(Error::invalid("feedback must be one of 'hybrid', 'numeric', 'text'.")),
        }
    }
}

/// ℓ(p): the verbal band of a probability; `None` for a question that was not asked.
pub fn label(probability: Option<f64>) -> Result<Option<&'static str>> {
    let Some(p) = probability else { return Ok(None) };
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(Error::invalid("Feedback probabilities must be finite numbers between 0 and 1, or null."));
    }
    Ok(Some(LABELS[CUT_POINTS.iter().filter(|cut| **cut <= p).count()]))
}

/// ℓ of a JSON value: `null` stays `null`; anything but a number in [0, 1] is an error.
pub fn label_value(value: &Value) -> Result<Value> {
    let probability = match value {
        Value::Null => None,
        Value::Number(_) => pyjson::as_number(value),
        _ => return Err(Error::invalid("Feedback probabilities must be finite numbers between 0 and 1, or null.")),
    };
    Ok(label(probability)?.map_or(Value::Null, |text| Value::String(text.to_string())))
}

fn label_field(key: &str) -> Option<&'static str> {
    LABEL_FIELDS.iter().find(|(name, _)| *name == key).map(|(_, field)| *field)
}

fn hybrid(value: &Value) -> Result<Value> {
    match value {
        Value::Array(items) => Ok(Value::Array(items.iter().map(hybrid).collect::<Result<_>>()?)),
        Value::Object(map) => {
            let mut result = Map::new();
            for (key, item) in map {
                let copied = if key == "schema_errors" { item.clone() } else { hybrid(item)? };
                result.insert(key.clone(), copied);
                if let Some(field) = label_field(key) {
                    if map.contains_key(field) || result.contains_key(field) {
                        return Err(Error::invalid("Repair feedback has conflicting added judgment fields."));
                    }
                    result.insert(field.to_string(), label_value(item)?);
                }
            }
            Ok(Value::Object(result))
        }
        other => Ok(other.clone()),
    }
}

fn text(value: &Value) -> Result<Value> {
    match value {
        Value::Array(items) => Ok(Value::Array(items.iter().map(text).collect::<Result<_>>()?)),
        Value::Object(map) => {
            let mut result = Map::new();
            for (key, item) in map {
                if THRESHOLDS.contains(&key.as_str()) {
                    continue;
                }
                if let Some(field) = label_field(key) {
                    if map.contains_key(field) || result.contains_key(field) {
                        return Err(Error::invalid("Repair feedback has conflicting judgment fields."));
                    }
                    result.insert(field.to_string(), label_value(item)?);
                } else if let (true, Value::String(instruction)) = (key == "instruction", item) {
                    result.insert(
                        key.clone(),
                        Value::String(instruction.replace("The probabilities", "The feedback labels")),
                    );
                } else if key == "schema_errors" {
                    result.insert(key.clone(), item.clone());
                } else {
                    result.insert(key.clone(), text(item)?);
                }
            }
            Ok(Value::Object(result))
        }
        other => Ok(other.clone()),
    }
}

/// The generator's copy of one feedback record under a representation. The input is not changed.
pub fn represent(feedback: &Value, representation: Feedback) -> Result<Value> {
    if !feedback.is_object() {
        return Err(Error::invalid("Repair feedback must be an object."));
    }
    match representation {
        Feedback::Numeric => Ok(feedback.clone()),
        Feedback::Text => text(feedback),
        Feedback::Hybrid => hybrid(feedback),
    }
}

/// Every record of a repair history under a representation, in the same order.
pub fn represent_history(history: &[Value], representation: Feedback) -> Result<Vec<Value>> {
    history
        .iter()
        .map(|item| {
            let mut item = item.clone();
            let record = item
                .as_object_mut()
                .filter(|record| record.get("feedback").is_some_and(Value::is_object))
                .ok_or_else(|| Error::invalid("Each repair history record needs a feedback object."))?;
            let shown = represent(&record["feedback"], representation)?;
            record.insert("feedback".to_string(), shown);
            Ok(item)
        })
        .collect()
}

/// A feedback record without the other question, for DM-Call-Function (`keep = "function"`)
/// and DM-Call-Argument (`keep = "arguments"`).
///
/// The source, schema errors, thresholds and instruction stay; only the hidden question's
/// verdict and score are removed.
pub fn single_check(feedback: &Value, keep: &str) -> Result<Value> {
    let hidden = match keep {
        "function" => "arguments",
        "arguments" => "function",
        _ => return Err(Error::invalid("keep must be 'function' or 'arguments'.")),
    };
    let hidden_record = if hidden == "function" { "function_selection" } else { "arguments" };
    let mut result = feedback.clone();
    if let Value::Object(map) = &mut result {
        if let Some(Value::Object(checks)) = map.get_mut("checks") {
            checks.shift_remove(hidden_record);
        }
        if let Some(Value::Array(failed)) = map.get_mut("failed_checks") {
            failed.retain(|name| name.as_str() != Some(hidden_record));
        }
        let prefix = format!("{hidden}_");
        map.retain(|key, _| !key.starts_with(&prefix));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn label_bands() {
        let cases = [
            (0.0, "very likely incorrect"),
            (0.19, "very likely incorrect"),
            (0.2, "likely incorrect"),
            (0.48, "uncertain"),
            (0.52, "uncertain"),
            (0.6, "likely correct"),
            (0.8, "very likely correct"),
            (1.0, "very likely correct"),
        ];
        for (p, expected) in cases {
            assert_eq!(label(Some(p)).unwrap(), Some(expected));
        }
        assert_eq!(label(None).unwrap(), None);
        for bad in [-0.1, 1.1, f64::NAN] {
            assert!(label(Some(bad)).is_err());
        }
        assert!(label_value(&json!(true)).is_err() && label_value(&json!("0.5")).is_err());
    }

    fn sample() -> Value {
        json!({
            "source": "jev_noul_and_deterministic_schema_validation",
            "checks": {"function_selection": {"pass": true, "probability_yes": 0.95, "threshold": 0.5, "schema_pass": true},
                       "arguments": {"skipped": false, "pass": false, "probability_yes": 0.12, "threshold": 0.5,
                                     "schema_pass": true}},
            "failed_checks": ["arguments"],
            "schema_errors": [{"check": "arguments", "path": "calls[0].arguments", "message": "'probability_yes' 0.3"}],
            "instruction": "Reconsider ... The probabilities are verifier judgments, not ground truth or explanations.",
        })
    }

    #[test]
    fn hybrid_keeps_numbers_and_adds_labels() {
        let out = represent(&sample(), Feedback::Hybrid).unwrap();
        let keys: Vec<&String> = out["checks"]["arguments"].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["skipped", "pass", "probability_yes", "feedback_label", "threshold", "schema_pass"]);
        assert_eq!(out["checks"]["arguments"]["feedback_label"], "very likely incorrect");
        assert!(pyjson::equal(&out["schema_errors"], &sample()["schema_errors"]));
    }

    #[test]
    fn text_replaces_numbers_and_thresholds() {
        let out = represent(&sample(), Feedback::Text).unwrap();
        assert!(pyjson::equal(
            &out["checks"]["arguments"],
            &json!({"skipped": false, "pass": false, "feedback_label": "very likely incorrect", "schema_pass": true})
        ));
        assert!(out["instruction"]
            .as_str()
            .unwrap()
            .ends_with("The feedback labels are verifier judgments, not ground truth or explanations."));
        let skipped = json!({"checks": {"arguments": {"skipped": true, "pass": null, "probability_yes": null, "threshold": 0.5}}});
        assert!(represent(&skipped, Feedback::Hybrid).unwrap()["checks"]["arguments"]["feedback_label"].is_null());
        assert!(represent_history(&[json!({"calls": []})], Feedback::Hybrid).is_err());
        assert!("words".parse::<Feedback>().is_err());
    }

    #[test]
    fn single_check_removes_only_the_other_question() {
        let function_only = single_check(&sample(), "function").unwrap();
        assert_eq!(function_only["checks"].as_object().unwrap().keys().collect::<Vec<_>>(), ["function_selection"]);
        assert_eq!(function_only["failed_checks"], json!([]));
        let arguments_only = single_check(&sample(), "arguments").unwrap();
        assert_eq!(arguments_only["failed_checks"], json!(["arguments"]));
        assert!(single_check(&sample(), "both").is_err());
    }
}
