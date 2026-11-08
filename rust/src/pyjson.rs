//! JSON text and value helpers that behave like Python's `json` module and `repr`.
//!
//! The history message DM-Call sends to the generator is JSON text inside a string, and it must
//! equal Python's `json.dumps(value, ensure_ascii=False)` character for character: `", "` and
//! `": "` separators, insertion key order, floats in Python `repr` form (`1.0`, `1e+16`,
//! `1e-05`), integers without a decimal point, non-ASCII kept, and control characters escaped.
//! The schema check's messages use Python `repr` of values (`'Seoul'`, `True`, `None`).
//!
//! Numbers keep whether they were written as an integer or a float: `serde_json` stores the
//! number as parsed, and a number whose text has a fraction or an exponent is a float, as in
//! Python's `json.loads`. With the default `big-integers` feature, integers beyond 64 bits keep
//! every digit.

use serde_json::{Map, Number, Value};

/// A JSON number as Python's `json.loads` would read it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Num {
    /// An integer, as canonical decimal text (`-0` becomes `0`).
    Int(String),
    /// A float (may be infinite when the text overflows, like Python's `float`).
    Float(f64),
}

pub(crate) fn num(n: &Number) -> Num {
    let text = n.to_string();
    if text.contains(['.', 'e', 'E']) {
        Num::Float(text.parse::<f64>().unwrap_or(f64::NAN))
    } else {
        let digits = text.strip_prefix('-').unwrap_or(&text);
        if digits.bytes().all(|b| b == b'0') {
            Num::Int("0".to_string())
        } else {
            Num::Int(text)
        }
    }
}

impl Num {
    pub(crate) fn as_f64(&self) -> f64 {
        match self {
            Num::Int(text) => text.parse::<f64>().unwrap_or(f64::NAN),
            Num::Float(value) => *value,
        }
    }
}

/// Parse JSON text, keeping key order and whether each number was an integer or a float.
pub fn loads(text: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str(text)
}

/// Python's `json.dumps(value, ensure_ascii=False)`.
pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, false, false);
    out
}

/// Python's `json.dumps(value)` (non-ASCII escaped as `\uXXXX`).
pub fn dumps_ascii(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, true, false);
    out
}

/// Python's `json.dumps(value, sort_keys=True)`.
pub(crate) fn dumps_sorted(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, true, true);
    out
}

/// Python's `json.dumps` of one string.
pub(crate) fn dumps_str(text: &str, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_string(&mut out, text, ensure_ascii);
    out
}

fn write_value(out: &mut String, value: &Value, ensure_ascii: bool, sort_keys: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => match num(n) {
            Num::Int(text) => out.push_str(&text),
            Num::Float(f) if f.is_nan() => out.push_str("NaN"),
            Num::Float(f) if f.is_infinite() => out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" }),
            Num::Float(f) => out.push_str(&float_repr(f)),
        },
        Value::String(s) => write_string(out, s, ensure_ascii),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(out, item, ensure_ascii, sort_keys);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(out, key, ensure_ascii);
                out.push_str(": ");
                write_value(out, item, ensure_ascii, sort_keys);
            }
            out.push('}');
        }
    }
}

fn write_string(out: &mut String, text: &str, ensure_ascii: bool) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest digits that round-trip, in exponent form when the
/// exponent is below -4 or at least 16 (`1e+16`, `1e-05`), otherwise a decimal with `.0` for
/// integral values.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // The shortest digits that round-trip. When two strings of that length round-trip, Python
    // takes the one nearest the exact value, rounding half to even; Rust's fixed-precision
    // formatting rounds the exact value the same way.
    let shortest = format!("{:e}", value);
    let length = shortest.split_once('e').map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count());
    let nearest = format!("{:.*e}", length.saturating_sub(1), value);
    let formatted = if nearest.parse::<f64>() == Ok(value) { nearest } else { shortest };
    let (mantissa, exponent) = formatted.split_once('e').expect("exponent form");
    let exponent: i32 = exponent.parse().expect("exponent digits");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let point = exponent as usize + 1;
            if digits.len() <= point {
                out.push_str(&digits);
                out.push_str(&"0".repeat(point - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exponent - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exponent.abs()));
    }
    out
}

/// Python's `repr` of a JSON value: `'Seoul'`, `True`, `None`, `['a']`, `{'a': 1}`, `2.0`.
pub fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => match num(n) {
            Num::Int(text) => text,
            Num::Float(f) => float_repr(f),
        },
        Value::String(s) => repr_str(s),
        Value::Array(items) => format!("[{}]", items.iter().map(repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter().map(|(k, v)| format!("{}: {}", repr_str(k), repr(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Python's `repr(str)`.
pub fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for ch in text.chars() {
        let code = ch as u32;
        match ch {
            c if c == quote || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if code < 0x20 || code == 0x7f => out.push_str(&format!("\\x{:02x}", code)),
            c if code < 0x7f => out.push(c),
            c if printable(c) => out.push(c),
            _ if code <= 0xff => out.push_str(&format!("\\x{:02x}", code)),
            _ if code <= 0xffff => out.push_str(&format!("\\u{:04x}", code)),
            _ => out.push_str(&format!("\\U{:08x}", code)),
        }
    }
    out.push(quote);
    out
}

/// An approximation of Python's `str.isprintable` for one non-ASCII character: separators,
/// format and private-use characters and noncharacters are not printable.
fn printable(ch: char) -> bool {
    let c = ch as u32;
    let not_printable = matches!(c,
        0x80..=0xa0 | 0xad | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x890..=0x891 | 0x8e2 | 0x1680 | 0x180e
        | 0x2000..=0x200f | 0x2028..=0x202f | 0x205f..=0x2064 | 0x2066..=0x206f | 0x3000 | 0xd800..=0xf8ff
        | 0xfdd0..=0xfdef | 0xfeff | 0xfff9..=0xfffb | 0x110bd | 0x110cd | 0x13430..=0x1343f
        | 0x1bca0..=0x1bca3 | 0x1d173..=0x1d17a | 0xe0001 | 0xe0020..=0xe007f | 0xf0000..=0x10ffff)
        || (c & 0xfffe) == 0xfffe;
    !not_printable
}

/// Python's `==` on JSON values: objects compare without order and `1 == 1.0`.
/// Booleans never equal numbers here (Python's `True == 1` is not mirrored).
pub fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => match (num(x), num(y)) {
            (Num::Int(p), Num::Int(q)) => p == q,
            (p, q) => p.as_f64() == q.as_f64(),
        },
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| equal(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(key, p)| y.get(key).is_some_and(|q| equal(p, q)))
        }
        _ => false,
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => match num(n) {
            Num::Int(text) => text != "0",
            Num::Float(f) => f != 0.0,
        },
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// `isinstance(value, int) and not isinstance(value, bool)`.
pub(crate) fn is_int(value: &Value) -> bool {
    matches!(value, Value::Number(n) if matches!(num(n), Num::Int(_)))
}

/// A JSON number as `f64` (integer or float), or `None` for any other value.
pub(crate) fn as_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => Some(num(n).as_f64()),
        _ => None,
    }
}

/// Whether every number in the value is finite (`json.dumps(value, allow_nan=False)` succeeds).
pub(crate) fn all_finite(value: &Value) -> bool {
    match value {
        Value::Number(n) => match num(n) {
            Num::Float(f) => f.is_finite(),
            Num::Int(_) => true,
        },
        Value::Array(items) => items.iter().all(all_finite),
        Value::Object(map) => map.values().all(all_finite),
        _ => true,
    }
}

/// A float JSON value (`1.0` stays a float in the history text).
pub(crate) fn float(value: f64) -> Value {
    Number::from_f64(value).map(Value::Number).unwrap_or(Value::Null)
}

pub(crate) fn object(entries: Vec<(&str, Value)>) -> Value {
    let mut map = Map::new();
    for (key, value) in entries {
        map.insert(key.to_string(), value);
    }
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (-0.0, "-0.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1.234e-5, "1.234e-05"),
            (123456789.125, "123456789.125"),
            (1e22, "1e+22"),
            (300.0, "300.0"),
            (0.30000000000000004, "0.30000000000000004"),
            (9999999999999998.0, "9999999999999998.0"),
            (834400034008635.0 + 0.25, "834400034008635.2"), // an exact tie: Python rounds half to even
            (5e-324, "5e-324"),
            (1e100, "1e+100"),
            (-2.5e-300, "-2.5e-300"),
            (f64::INFINITY, "inf"),
        ];
        for (value, expected) in cases {
            assert_eq!(float_repr(value), expected, "{value}");
        }
    }

    #[test]
    fn repr_matches_python() {
        let value: Value = loads(r#"{"a": [1, 2.0, "it's", "q\"x", null, true, "\u00a0\u2028tab\t"]}"#).unwrap();
        assert_eq!(repr(&value), r#"{'a': [1, 2.0, "it's", 'q"x', None, True, '\xa0\u2028tab\t']}"#);
        assert_eq!(repr_str("서울 🌤"), "'서울 🌤'");
        assert_eq!(repr_str("both ' and \""), "'both \\' and \"'");
        assert_eq!(repr_str("\u{7f}\u{1}"), "'\\x7f\\x01'");
    }

    #[test]
    fn ascii_and_sorted_dumps() {
        let value: Value = loads(r#"{"b": "\uc11c\uc6b8", "a": [1e-7, "\u007f"]}"#).unwrap();
        assert_eq!(dumps_ascii(&value), r#"{"b": "\uc11c\uc6b8", "a": [1e-07, "\u007f"]}"#);
        assert_eq!(dumps_sorted(&value), r#"{"a": [1e-07, "\u007f"], "b": "\uc11c\uc6b8"}"#);
        assert_eq!(dumps_str("\u{1f324}", true), r#""\ud83c\udf24""#);
    }

    #[test]
    fn numbers_keep_their_kind() {
        #[cfg(feature = "big-integers")]
        assert_eq!(dumps(&loads("-0").unwrap()), "0");
        #[cfg(feature = "big-integers")]
        assert_eq!(dumps(&loads("100000000000000000000").unwrap()), "100000000000000000000");
        assert_eq!(dumps(&loads("[4.0, 4, 1E2, 2e-1]").unwrap()), "[4.0, 4, 100.0, 0.2]");
        assert!(equal(&loads("1").unwrap(), &loads("1.0").unwrap()));
        assert!(!equal(&loads("true").unwrap(), &loads("1").unwrap()));
        assert!(equal(&loads(r#"{"a": 1, "b": 2}"#).unwrap(), &loads(r#"{"b": 2, "a": 1.0}"#).unwrap()));
        assert_eq!(dumps(&float(1.0)), "1.0");
        assert!(is_int(&loads("3").unwrap()) && !is_int(&loads("3.0").unwrap()));
    }
}
