//! Jev and the PPLX Decider: the wire format, verdicts, thresholds and failures.
//! Ported from the Python reference's `tests/test_decision_models.py`.

mod common;

use std::sync::{Arc, Mutex};

use common::*;
use dmcall::{
    Answer, Call, DMCall, DecisionModel, DecisionModelError, Error, HttpResponse, Jev, JevBuilder, PPLXDecider,
    SystemOne,
};
use serde_json::{json, Map, Value};

const SECRET: &str = "sk-test-SECRET-123";

fn weather() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}})
}

fn messages() -> Vec<Value> {
    vec![json!({"role": "system", "content": "Be brief."}), json!({"role": "user", "content": "Weather in Seoul?"})]
}

fn seoul() -> Vec<Call> {
    vec![Call::new("get_weather", json!({"city": "Seoul"}))]
}

fn answer(function: f64, arguments: Option<f64>, model: &str) -> Value {
    let mut answers = json!({"function_correct": {"type": "noul", "noul": function}});
    if let Some(arguments) = arguments {
        answers["arguments_correct"] = json!({"type": "noul", "noul": arguments});
    }
    json!({"model": model, "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 0}})
}

fn jev_with(mock: &Arc<Mock>) -> JevBuilder {
    Jev::builder().api_key("jev-key").transport(mock.clone()).sleep(no_sleep())
}

fn replying(body: Value) -> Arc<Mock> {
    Mock::with(move |_| Ok(HttpResponse::json(200, &body)))
}

#[test]
fn wire_format_sends_only_observable_inputs() {
    isolate_env();
    let mock = replying(answer(0.9, Some(0.8), "jev-1.13.0"));
    let mut tool = weather();
    tool["metadata"] = json!({"reference": "get_weather(city='Seoul')"});
    let mut conversation = messages();
    conversation.push(json!({"role": "user", "content": "hi", "ground_truth": "nope"}));
    let record = jev_with(&mock).build().unwrap().judge(&conversation, &[tool], Some(&seoul())).unwrap();
    let request = &mock.sent()[0];
    let body = body_of(request);
    assert_eq!(request.url, "https://api.typesafe.ai/v1/systemone");
    assert_eq!(body["model"], "jev-1.13.0");
    assert_py_eq(&body["state"]["messages"][2], &json!({"role": "user", "content": "hi"}), "observable message");
    assert_py_eq(&body["state"]["tools"], &json!([weather()["function"]]), "tools");
    assert_py_eq(
        &body["state"]["proposed_calls"],
        &json!([{"name": "get_weather", "arguments": {"city": "Seoul"}}]),
        "calls",
    );
    let questions =
        json!({"function_correct": *dmcall::FUNCTION_QUESTION, "arguments_correct": *dmcall::ARGUMENT_QUESTION});
    assert_py_eq(&body["questions"], &questions, "questions");
    assert!(record.accepted && record.function_probability == Some(0.9) && record.arguments_probability == Some(0.8));
    assert_eq!(record.feedback["failed_checks"], json!([]));
    assert_eq!(record.api_calls, 1);
}

#[test]
fn no_call_asks_only_the_function_question() {
    isolate_env();
    let mock = replying(answer(0.97, None, "jev-1.13.0"));
    let record = jev_with(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&[])).unwrap();
    let body = body_of(&mock.sent()[0]);
    assert_eq!(body["questions"].as_object().unwrap().keys().collect::<Vec<_>>(), ["function_correct"]);
    assert!(record.accepted && record.arguments_skipped && record.arguments_probability.is_none());
    assert!(record.feedback["checks"]["arguments"]["pass"].is_null());
}

#[test]
fn both_questions_must_pass_at_the_threshold() {
    isolate_env();
    for (function, arguments, failed) in
        [(0.4, 0.9, json!(["function_selection"])), (0.9, 0.49, json!(["arguments"])), (0.5, 0.5, json!([]))]
    {
        let mock = replying(answer(function, Some(arguments), "jev-1.13.0"));
        let record = jev_with(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&seoul())).unwrap();
        assert_eq!(record.feedback["failed_checks"], failed);
        assert_eq!(record.accepted, failed == json!([]));
    }
}

#[test]
fn schema_failure_is_judged_but_cannot_be_overruled() {
    isolate_env();
    let mock = replying(answer(0.99, Some(0.99), "jev-1.13.0"));
    let calls = [Call::new("get_weather", json!({}))];
    let record = jev_with(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&calls)).unwrap();
    assert_eq!(mock.sent().len(), 1); // a readable candidate is still judged
    assert_eq!(record.arguments_probability, Some(0.99));
    assert!(record.arguments_pass == Some(false) && !record.accepted && !record.schema_errors.is_empty());
}

#[test]
fn unreadable_candidate_is_not_sent() {
    isolate_env();
    let malformed = [Call::new("get_weather", Value::Null)];
    for calls in [None, Some(&malformed[..])] {
        let mock = Mock::with(|_| panic!("An unreadable candidate must not reach the decision model"));
        let record = jev_with(&mock).build().unwrap().judge(&messages(), &[weather()], calls).unwrap();
        assert!(record.api_calls == 0 && record.function_probability.is_none() && !record.accepted);
    }
}

#[test]
fn thresholds_and_key_hygiene() {
    isolate_env();
    let mock = replying(answer(0.75, Some(0.85), "jev-1.13.0"));
    let model = Jev::builder()
        .api_key(SECRET)
        .model("jev-2")
        .threshold(0.7)
        .arguments_threshold(0.9)
        .transport(mock.clone())
        .build()
        .unwrap();
    let record = model.judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    let request = &mock.sent()[0];
    assert_eq!(request.header("authorization"), Some(format!("Bearer {SECRET}").as_str()));
    assert!(!format!("{model:?}").contains(SECRET) && !format!("{request:?}").contains(SECRET));
    assert_eq!(body_of(request)["model"], "jev-2");
    assert!(record.function_pass && record.arguments_pass == Some(false));
    for bad in [1.5, -0.1, f64::NAN] {
        let error = Jev::builder().api_key("k").threshold(bad).transport(mock.clone()).build().unwrap_err();
        assert!(error.to_string().contains("Thresholds"));
    }
    assert!(Jev::builder().api_key("a\nb").transport(mock.clone()).build().is_err());
}

#[test]
fn retries_are_bounded_and_errors_sanitized() {
    isolate_env();
    let mock = Mock::responses(vec![
        HttpResponse::new(429, "").with_header("Retry-After", "0"),
        HttpResponse::json(200, &answer(0.95, Some(0.95), "jev-1.13.0")),
    ]);
    let record = jev_with(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    assert_eq!(record.api_calls, 2);
    let failing = Mock::with(|_| Ok(HttpResponse::new(401, SECRET).with_header("x-request-id", "req-7")));
    let error = jev_with(&failing).build().unwrap().judge(&messages(), &[weather()], Some(&seoul())).unwrap_err();
    let Error::DecisionModel(error) = error else { panic!("expected a decision-model error") };
    assert_eq!(failing.sent().len(), 1);
    assert_eq!(
        (error.status_code, error.request_id.as_deref(), error.code.as_str()),
        (Some(401), Some("req-7"), "http_error")
    );
    assert!(!error.to_string().contains(SECRET));
    let overloaded = Mock::with(|_| Ok(HttpResponse::new(503, "")));
    let error = jev_with(&overloaded)
        .max_retries(2)
        .build()
        .unwrap()
        .judge(&messages(), &[weather()], Some(&seoul()))
        .unwrap_err();
    let Error::DecisionModel(error) = error else { panic!("expected a decision-model error") };
    assert_eq!(error.attempts, Some(3));
    let unreachable = Mock::with(|_| Err(dmcall::TransportError::new("ConnectionFailed")));
    let error = jev_with(&unreachable)
        .max_retries(1)
        .build()
        .unwrap()
        .judge(&messages(), &[weather()], Some(&seoul()))
        .unwrap_err();
    let Error::DecisionModel(error) = error else { panic!("expected a decision-model error") };
    assert_eq!((error.code.as_str(), error.attempts), ("connection_failed", Some(2)));
}

#[test]
fn invalid_responses_are_errors_not_rejections() {
    isolate_env();
    let mut no_usage = answer(0.95, Some(0.95), "jev-1.13.0");
    no_usage["usage"] = Value::Null;
    let mut no_model = answer(0.95, Some(0.95), "jev-1.13.0");
    no_model["model"] = json!("");
    for body in [
        json!({"answers": {}}),
        answer(1.5, Some(0.95), "jev-1.13.0"),
        no_usage,
        no_model,
        json!(["not", "an", "object"]),
    ] {
        let mock = replying(body.clone());
        let result = jev_with(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&seoul()));
        assert!(matches!(result, Err(Error::DecisionModel(ref e)) if e.code == "response_invalid"), "{body}");
    }
    let mock = Mock::with(|_| Ok(HttpResponse::new(200, "not json")));
    assert!(jev_with(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&seoul())).is_err());
}

// PPLX Decider --------------------------------------------------------------------------------------------

fn provenance() -> Value {
    json!({"model_id": "perplexity-ai/pplx-decider-v1.1-27b", "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce",
           "native_attention": "noncausal_full_attention", "max_input_tokens": 8192, "temperature": 1.0087417621345625})
}

fn pplx(mock: &Arc<Mock>) -> dmcall::PPLXDeciderBuilder {
    PPLXDecider::builder("http://127.0.0.1:8000").transport(mock.clone())
}

#[test]
fn pplx_uses_the_same_questions_and_reports_the_checkpoint() {
    isolate_env();
    let mock = replying(answer(0.9, Some(0.2), "autojev-qwen3.8-27b"));
    let record = pplx(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    let request = &mock.sent()[0];
    let body = body_of(request);
    assert_eq!(request.url, "http://127.0.0.1:8000/v1/systemone");
    assert_eq!(body["model"], "jev-latest");
    assert_py_eq(&body["questions"]["function_correct"], &dmcall::FUNCTION_QUESTION, "question");
    assert_eq!(record.model.as_deref(), Some("perplexity-ai/pplx-decider-v1.1-27b"));
    assert!(!record.accepted);
    assert_eq!(record.feedback["source"], "perplexity_decider_noul_and_deterministic_schema_validation");
    assert!(request.header("authorization").is_none());
}

#[test]
fn pplx_missing_usage_stays_unknown() {
    isolate_env();
    let mut body = answer(0.95, Some(0.95), "autojev-qwen3.8-27b");
    body.as_object_mut().unwrap().remove("usage");
    let record = pplx(&replying(body)).build().unwrap().judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    assert_py_eq(&Value::Object(record.usage), &json!({"input_tokens": null, "output_tokens": null}), "usage");
}

#[test]
fn pplx_rejects_another_model() {
    isolate_env();
    let mut wrong_revision = answer(0.95, Some(0.95), "perplexity-ai/pplx-decider-v1.1-27b");
    wrong_revision["revision"] = json!("0".repeat(40));
    for body in [answer(0.95, Some(0.95), "some-other-model"), wrong_revision] {
        let result = pplx(&replying(body)).build().unwrap().judge(&messages(), &[weather()], Some(&seoul()));
        assert!(matches!(result, Err(Error::DecisionModel(ref e)) if e.code == "model_mismatch"));
    }
}

#[test]
fn pplx_provenance_check_is_opt_in() {
    isolate_env();
    let handler = |request: &dmcall::HttpRequest| {
        if request.url.ends_with("/provenance") {
            Ok(HttpResponse::json(200, &provenance()))
        } else {
            Ok(HttpResponse::json(200, &answer(0.95, Some(0.95), "autojev-qwen3.8-27b")))
        }
    };
    let mock = Mock::with(handler);
    let decider = pplx(&mock).check_provenance(true).build().unwrap();
    decider.judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    decider.judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    let paths: Vec<String> =
        mock.sent().iter().map(|r| r.url.trim_start_matches("http://127.0.0.1:8000").to_string()).collect();
    assert_eq!(paths, ["/provenance", "/v1/systemone", "/v1/systemone"]);
    assert_eq!(mock.sent()[0].method, dmcall::Method::Get);
    let unchecked = Mock::with(handler);
    pplx(&unchecked).build().unwrap().judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    assert_eq!(unchecked.sent().len(), 1);
    let mut wrong = provenance();
    wrong["revision"] = json!("main");
    let decider = pplx(&replying(wrong)).check_provenance(true).build().unwrap();
    let result = decider.judge(&messages(), &[weather()], Some(&seoul()));
    assert!(matches!(result, Err(Error::DecisionModel(ref e)) if e.code == "provenance_mismatch"));
}

#[test]
fn pplx_context_limit_is_an_error() {
    isolate_env();
    let mock =
        Mock::with(|_| Ok(HttpResponse::json(422, &json!({"detail": "Prepared input exceeds the 8192-token limit."}))));
    let result = pplx(&mock).build().unwrap().judge(&messages(), &[weather()], Some(&seoul()));
    assert!(matches!(result, Err(Error::DecisionModel(ref e)) if e.code == "context_limit_exceeded"));
}

#[test]
fn pplx_rejects_unsafe_urls() {
    isolate_env();
    let mock = replying(json!({}));
    for url in ["http://10.0.0.4:8000", "https://user:pw@example.com", "http://localhost:8000/v1", "ftp://localhost"] {
        assert!(PPLXDecider::builder(url).transport(mock.clone()).build().is_err(), "{url}");
    }
    let decider = PPLXDecider::builder("https://decider.example.com").transport(mock).build().unwrap();
    assert_eq!(decider.base_url(), "https://decider.example.com");
}

// A decision model of your own ----------------------------------------------------------------------------

/// Says yes to every function question and to arguments that name Seoul.
struct Rules {
    asked: Mutex<Vec<Vec<String>>>,
}

impl DecisionModel for Rules {
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> Result<Answer, DecisionModelError> {
        self.asked.lock().unwrap().push(questions.keys().cloned().collect());
        let seoul = state["proposed_calls"][0]["arguments"]["city"] == "Seoul";
        let mut probabilities = std::collections::HashMap::new();
        probabilities.insert("function_correct".to_string(), 1.0);
        if questions.contains_key("arguments_correct") {
            probabilities.insert("arguments_correct".to_string(), if seoul { 1.0 } else { 0.0 });
        }
        Ok(Answer { probabilities, model: "rules-1".into(), ..Answer::default() })
    }

    fn model(&self) -> &str {
        "rules-1"
    }
}

#[test]
fn a_custom_decision_model_plugs_into_the_loop() {
    isolate_env();
    let rules = Arc::new(Rules { asked: Mutex::new(Vec::new()) });
    let record =
        rules.judge(&messages(), &[weather()], Some(&[Call::new("get_weather", json!({"city": "Busan"}))])).unwrap();
    assert!(!record.accepted && record.function_pass && record.model.as_deref() == Some("rules-1"));
    assert_eq!(record.feedback["source"], "decision_model_noul_and_deterministic_schema_validation");
    let tool = dmcall::Tool::new(weather()).unwrap();
    let verdict = dmcall::verify("Weather in Seoul?", &seoul(), &[tool], rules.as_ref()).unwrap();
    assert!(verdict.accepted);
    assert_eq!(rules.asked.lock().unwrap()[0], ["function_correct", "arguments_correct"]);
    let agent = DMCall::builder(
        dmcall::OpenAICompatible::builder("m")
            .base_url("http://localhost:1/v1")
            .transport(replying(json!({})))
            .build()
            .unwrap(),
    )
    .decision_model(rules.clone())
    .build()
    .unwrap();
    assert_eq!(agent.decision_model().model(), "rules-1");
}

#[test]
fn system_one_speaks_the_same_protocol() {
    isolate_env();
    let mock = replying(answer(0.9, Some(0.9), "my-decider"));
    let model =
        SystemOne::builder("https://decider.example.com/", "my-decider").transport(mock.clone()).build().unwrap();
    let record = model.judge(&messages(), &[weather()], Some(&seoul())).unwrap();
    assert!(record.accepted);
    assert_eq!(mock.sent()[0].url, "https://decider.example.com/v1/systemone");
    assert!(mock.sent()[0].header("authorization").is_none());
}
