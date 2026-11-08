//! The DM-Call loop: acceptance, cumulative repair, the paper's variants, fallback and execution.
//! Ported from the Python reference's `tests/test_loop.py`.

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use common::*;
use dmcall::{
    Call, CallOptions, Checks, DMCall, DMCallBuilder, Error, Feedback, Generator, History, HttpResponse, Jev, Message,
    OnExhaustion, Proposal, Result, Tool,
};
use serde_json::{json, Value};

fn weather() -> Tool {
    Tool::new(json!({"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}))
    .unwrap()
}

fn seoul() -> Option<Vec<Call>> {
    Some(vec![Call::new("get_weather", json!({"city": "Seoul"}))])
}

fn busan() -> Option<Vec<Call>> {
    Some(vec![Call::new("get_weather", json!({"city": "Busan"}))])
}

fn none() -> Option<Vec<Call>> {
    Some(Vec::new())
}

/// A generator that returns scripted candidates and records the history it saw.
struct Scripted {
    proposals: Mutex<VecDeque<Option<Vec<Call>>>>,
    histories: Mutex<Vec<Vec<Value>>>,
}

impl Scripted {
    fn new(proposals: Vec<Option<Vec<Call>>>) -> Arc<Scripted> {
        Arc::new(Scripted { proposals: Mutex::new(proposals.into()), histories: Mutex::new(Vec::new()) })
    }

    fn histories(&self) -> Vec<Vec<Value>> {
        self.histories.lock().unwrap().clone()
    }
}

impl Generator for Scripted {
    fn generate(&self, _messages: &[Message], _tools: &[Value], history: &[Value]) -> Result<Proposal> {
        self.histories.lock().unwrap().push(history.to_vec());
        let calls = self.proposals.lock().unwrap().pop_front().expect("no scripted proposal left");
        let mut usage = dmcall::Usage::new();
        usage.insert("input_tokens".into(), json!(10));
        usage.insert("output_tokens".into(), json!(2));
        Ok(Proposal {
            content: if calls.as_ref().is_some_and(|c| !c.is_empty()) { None } else { Some("No tool fits.".into()) },
            incomplete: calls.is_none().then(|| "length".to_string()),
            parse_errors: if calls.is_none() { vec!["cut off".into()] } else { Vec::new() },
            calls,
            usage,
            latency_s: 0.01,
            ..Proposal::default()
        })
    }
}

fn answer(function: f64, arguments: Option<f64>) -> Value {
    let mut answers = json!({"function_correct": {"type": "noul", "noul": function}});
    if let Some(arguments) = arguments {
        answers["arguments_correct"] = json!({"type": "noul", "noul": arguments});
    }
    json!({"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 0}})
}

/// Jev over a mock: arguments are right only for Seoul; a no call scores `no_call`.
fn scripted_jev(no_call: f64) -> (Jev, Arc<Mock>) {
    let mock = Mock::with(move |request| {
        let body = body_of(request);
        let calls = body["state"]["proposed_calls"].as_array().unwrap().clone();
        let reply = if calls.is_empty() {
            answer(no_call, None)
        } else {
            let right = calls[0]["arguments"]["city"] == "Seoul";
            answer(0.95, Some(if right { 0.95 } else { 0.05 }))
        };
        Ok(HttpResponse::json(200, &reply))
    });
    (Jev::builder().api_key("jev-key").transport(mock.clone()).sleep(no_sleep()).build().unwrap(), mock)
}

fn agent(proposals: Vec<Option<Vec<Call>>>) -> (DMCallBuilder, Arc<Scripted>, Arc<Mock>) {
    let model = Scripted::new(proposals);
    let (jev, mock) = scripted_jev(0.05);
    (DMCall::builder(model.clone()).decision_model(jev).tool(weather()), model, mock)
}

fn recording_weather(ran: Arc<Mutex<Vec<String>>>) -> Tool {
    weather().with_handler(move |arguments| {
        let city = arguments["city"].as_str().unwrap_or_default().to_string();
        ran.lock().unwrap().push(city.clone());
        Ok(json!({"city": city, "temp": 21}))
    })
}

#[test]
fn repairs_with_cumulative_feedback_then_runs() {
    isolate_env();
    let ran = Arc::new(Mutex::new(Vec::new()));
    let model = Scripted::new(vec![busan(), none(), seoul()]);
    let (jev, _) = scripted_jev(0.05);
    let loop_ =
        DMCall::builder(model.clone()).decision_model(jev).tool(recording_weather(ran.clone())).build().unwrap();
    let result = loop_.run("Weather in Seoul?").unwrap();
    assert_eq!(result.calls, seoul());
    assert_eq!(result.accepted, Some(true));
    assert!(result.repairs == 2 && !result.exhausted && result.executed);
    assert_eq!(*ran.lock().unwrap(), ["Seoul"]);
    assert_py_eq(&result.output().unwrap(), &json!({"city": "Seoul", "temp": 21}), "output");
    let histories = model.histories();
    assert_eq!(histories.iter().map(Vec::len).collect::<Vec<_>>(), [0, 1, 2]);
    assert_py_eq(&histories[2][0]["calls"], &calls_json(&busan()), "first call");
    assert_eq!(histories[2][1]["calls"], json!([]));
    assert_eq!(histories[2][0]["feedback"]["failed_checks"], json!(["arguments"]));
    // Hybrid feedback: the generator reads the score and its label; the trace keeps the score.
    let shown = &histories[1][0]["feedback"]["checks"]["arguments"];
    assert_eq!(shown["probability_yes"].as_f64(), Some(0.05));
    assert_eq!(shown["feedback_label"], "very likely incorrect");
    assert!(result.trace[0].feedback["checks"]["arguments"].get("feedback_label").is_none());
    let attempts = result.attempts();
    assert_eq!(attempts.iter().map(|a| a.accepted).collect::<Vec<_>>(), [Some(false), Some(false), Some(true)]);
    assert_eq!(attempts[1].arguments_probability, None); // a no call skips the argument question
    let usage = result.usage();
    assert_py_eq(&Value::Object(usage.generator), &json!({"input_tokens": 30, "output_tokens": 6}), "generator usage");
    assert_py_eq(
        &Value::Object(usage.decision_model),
        &json!({"input_tokens": 900, "output_tokens": 0}),
        "decision usage",
    );
    assert_eq!((result.probabilities().function, result.probabilities().arguments), (Some(0.95), Some(0.95)));
    assert!(result.to_string().starts_with("CallResult(call=get_weather(city=\"Seoul\"), accepted=true, repairs=2"));
}

#[test]
fn rejected_calls_do_not_run_unless_asked() {
    isolate_env();
    let ran = Arc::new(Mutex::new(Vec::new()));
    let (jev, _) = scripted_jev(0.05);
    let model = Scripted::new(vec![busan(), busan(), busan(), busan()]);
    let loop_ = DMCall::builder(model).decision_model(jev).tool(recording_weather(ran.clone())).build().unwrap();
    let result = loop_.run("Weather in Seoul?").unwrap();
    assert!(result.accepted == Some(false) && result.exhausted && result.repairs == 3 && !result.executed);
    assert!(ran.lock().unwrap().is_empty());
    let (jev, _) = scripted_jev(0.05);
    let again = DMCall::builder(Scripted::new(vec![busan(), busan()]))
        .decision_model(jev)
        .tool(recording_weather(ran.clone()))
        .max_repairs(1)
        .build()
        .unwrap();
    let result = again
        .run_with("Weather in Seoul?", CallOptions { execute_unverified: Some(true), ..Default::default() })
        .unwrap();
    assert!(result.repairs == 1 && result.executed);
    assert_eq!(*ran.lock().unwrap(), ["Busan"]);
}

#[test]
fn without_an_accepted_candidate_the_first_one_is_final() {
    isolate_env();
    let ran = Arc::new(Mutex::new(Vec::new()));
    let (jev, _) = scripted_jev(0.05);
    let first = DMCall::builder(Scripted::new(vec![none(), busan()]))
        .decision_model(jev)
        .tool(recording_weather(ran.clone()))
        .max_repairs(1)
        .preserve_initial_no_call(false)
        .build()
        .unwrap();
    let unverified = || CallOptions { execute_unverified: Some(true), ..Default::default() };
    let result = first.run_with("Weather in Seoul?", unverified()).unwrap();
    assert!(result.exhausted && result.accepted == Some(false) && result.repairs == 1);
    assert!(result.no_call() && result.final_attempt == 0 && !result.executed && ran.lock().unwrap().is_empty());
    assert_eq!(result.text.as_deref(), Some("No tool fits."));
    assert_eq!((result.probabilities().function, result.probabilities().arguments), (Some(0.05), None));
    let (jev, _) = scripted_jev(0.05);
    let last = DMCall::builder(Scripted::new(vec![none(), busan()]))
        .decision_model(jev)
        .tool(recording_weather(ran.clone()))
        .max_repairs(1)
        .on_exhaustion(OnExhaustion::Last)
        .preserve_initial_no_call(false)
        .build()
        .unwrap();
    let result = last.run_with("Weather in Seoul?", unverified()).unwrap();
    assert!(result.calls == busan() && result.final_attempt == 1 && result.executed);
    assert_eq!(*ran.lock().unwrap(), ["Busan"]);
    let (builder, _, _) = agent(vec![busan(), none(), busan(), seoul()]);
    let per_call = builder.max_repairs(1).build().unwrap();
    let options = CallOptions { on_exhaustion: Some(OnExhaustion::Last), ..Default::default() };
    assert_eq!(per_call.call_with("Weather in Seoul?", options).unwrap().calls, none());
    let passed = per_call.call("Weather in Seoul?").unwrap();
    assert!(passed.calls == seoul() && passed.accepted == Some(true) && passed.final_attempt == 1);
    assert!("best".parse::<OnExhaustion>().is_err());
}

#[test]
fn unreadable_candidate_is_repaired_and_never_run() {
    isolate_env();
    let (builder, _, mock) = agent(vec![None, seoul()]);
    let loop_ = builder.handler("get_weather", |arguments| Ok(arguments["city"].clone())).build().unwrap();
    let result = loop_.run("Weather in Seoul?").unwrap();
    let attempts = result.attempts();
    assert!(attempts[0].calls.is_none() && !attempts[0].schema_errors.is_empty());
    assert_eq!(result.accepted, Some(true));
    assert_eq!(result.output(), Some(json!("Seoul")));
    assert_eq!(mock.sent().len(), 1);
    let (builder, _, _) = agent(vec![None]);
    let stuck =
        builder.max_repairs(0).handler("get_weather", |arguments| Ok(arguments["city"].clone())).build().unwrap();
    let result =
        stuck.run_with("Weather?", CallOptions { execute_unverified: Some(true), ..Default::default() }).unwrap();
    assert!(result.calls.is_none() && !result.executed);
    assert!(result.to_string().starts_with("CallResult(unreadable candidate"));
}

#[test]
fn initial_no_call_is_returned_without_verification() {
    isolate_env();
    let model = Scripted::new(vec![none()]);
    let (jev, mock) = scripted_jev(0.97);
    let loop_ = DMCall::builder(model.clone()).decision_model(jev).tool(weather()).build().unwrap();
    let result = loop_.run("Tell me a joke.").unwrap();
    assert!(result.no_call() && result.accepted.is_none() && result.initial_no_call_preserved);
    assert!(!result.executed && result.text.as_deref() == Some("No tool fits.") && mock.sent().is_empty());
    assert_eq!(result.attempts()[0].accepted, None);
    assert_eq!(model.histories().len(), 1);
    let (jev, mock) = scripted_jev(0.97);
    let checked = DMCall::builder(Scripted::new(vec![none()]))
        .decision_model(jev)
        .tool(weather())
        .preserve_initial_no_call(false)
        .build()
        .unwrap();
    let result = checked.call("Tell me a joke.").unwrap();
    assert!(result.accepted == Some(true) && !result.initial_no_call_preserved && mock.sent().len() == 1);
}

#[test]
fn latest_history_keeps_the_real_round() {
    isolate_env();
    let (builder, model, _) = agent(vec![busan(), none(), seoul()]);
    builder.history(History::Latest).build().unwrap().call("Weather in Seoul?").unwrap();
    let histories = model.histories();
    assert_eq!(histories.iter().map(Vec::len).collect::<Vec<_>>(), [0, 1, 1]);
    assert_eq!(histories[2][0]["round"], json!(2));
    assert_eq!(histories[2][0]["calls"], json!([]));
}

#[test]
fn single_question_variants() {
    isolate_env();
    let (builder, _, _) = agent(vec![busan()]);
    let result = builder.checks(Checks::Function).build().unwrap().call("Weather in Seoul?").unwrap();
    // The wrong city passes when only the function is checked.
    assert!(result.accepted == Some(true) && result.repairs == 0);
    let keys: Vec<&String> = result.trace[0].feedback["checks"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["function_selection"]);
    let (builder, _, _) = agent(vec![none(), seoul()]);
    let arguments_only = builder.checks(Checks::Arguments).preserve_initial_no_call(false).build().unwrap();
    let result = arguments_only.call("Weather in Seoul?").unwrap();
    // A no call has no arguments to reject.
    assert!(result.accepted == Some(true) && result.no_call() && result.repairs == 0);
}

#[test]
fn feedback_representations_reach_only_the_generator() {
    isolate_env();
    for (representation, field) in
        [(Feedback::Numeric, "threshold"), (Feedback::Text, "feedback_label"), (Feedback::Hybrid, "feedback_label")]
    {
        let (builder, model, _) = agent(vec![busan(), seoul()]);
        let result = builder.feedback(representation).build().unwrap().call("Weather in Seoul?").unwrap();
        let histories = model.histories();
        let shown = &histories[1][0]["feedback"]["checks"]["arguments"];
        assert!(shown.get(field).is_some());
        assert_eq!(result.trace[0].feedback["checks"]["arguments"]["probability_yes"].as_f64(), Some(0.05));
        assert_eq!(shown.get("probability_yes").is_some(), representation != Feedback::Text);
    }
}

#[test]
fn options_and_defaults() {
    isolate_env();
    let (builder, _, _) = agent(vec![]);
    let loop_ = builder.build().unwrap();
    assert_eq!(loop_.max_repairs(), 3);
    assert_eq!(
        (loop_.checks(), loop_.history(), loop_.feedback()),
        (Checks::Both, History::Cumulative, Feedback::Hybrid)
    );
    assert_eq!(loop_.on_exhaustion(), OnExhaustion::First);
    assert!(loop_.preserve_initial_no_call());
    for (name, value) in [("checks", "neither"), ("history", "none"), ("feedback", "words"), ("on_exhaustion", "best")]
    {
        let failed = match name {
            "checks" => value.parse::<Checks>().is_err(),
            "history" => value.parse::<History>().is_err(),
            "feedback" => value.parse::<Feedback>().is_err(),
            _ => value.parse::<OnExhaustion>().is_err(),
        };
        assert!(failed, "{name}");
    }
    let (jev, _) = scripted_jev(0.05);
    let error = DMCall::builder(Scripted::new(vec![])).decision_model(jev).threshold(0.9).build().unwrap_err();
    assert!(error.to_string().contains("decision model itself"));
    let error = DMCall::builder(Scripted::new(vec![])).build().unwrap_err();
    assert!(error.to_string().contains("TYPESAFE_API_KEY"));
}

#[test]
fn per_call_overrides_and_missing_handlers() {
    isolate_env();
    let (builder, _, _) = agent(vec![busan(), seoul()]);
    let loop_ = builder.build().unwrap();
    let result =
        loop_.call_with("Weather in Seoul?", CallOptions { max_repairs: Some(0), ..Default::default() }).unwrap();
    assert!(result.repairs == 0 && result.accepted == Some(false) && result.exhausted);
    let error = loop_
        .run_with("Weather in Seoul?", CallOptions { tools: Some(vec![weather()]), ..Default::default() })
        .unwrap_err();
    assert!(matches!(error, Error::MissingHandler(ref name) if name == "get_weather"));
    let handler: dmcall::Handler = Arc::new(|_| Ok(Value::Null));
    let options = CallOptions { handlers: vec![("nope".into(), handler)], ..Default::default() };
    assert!(loop_.call_with("x", options).unwrap_err().to_string().contains("not one of the tools"));
}

#[test]
fn tool_errors_propagate() {
    isolate_env();
    let (builder, _, _) = agent(vec![seoul()]);
    let loop_ = builder.handler("get_weather", |_| Err("station offline".into())).build().unwrap();
    let error = loop_.run("Weather in Seoul?").unwrap_err();
    assert!(matches!(error, Error::Tool { ref name, .. } if name == "get_weather"));
    assert!(error.to_string().contains("station offline"));
}

#[test]
fn messages_are_validated() {
    isolate_env();
    let (builder, _, _) = agent(vec![seoul(), seoul()]);
    let loop_ = builder.build().unwrap();
    let parts = json!([{"role": "user", "content": [{"type": "text", "text": "Weather"}, {"type": "text", "text": "in Seoul?"}]}]);
    loop_.call(&parts).unwrap();
    for bad in [
        json!([]),
        json!([{"role": "tool", "content": "{}"}]),
        json!([{"role": "user", "content": {"x": 1}}]),
        json!(3),
    ] {
        assert!(matches!(loop_.call(&bad), Err(Error::Invalid(_))), "{bad}");
    }
    let typed = vec![Message::system("Be brief."), Message::user("Weather in Seoul?")];
    assert_eq!(loop_.call(&typed).unwrap().accepted, Some(true));
}

#[test]
fn verify_checks_calls_from_any_sdk() {
    isolate_env();
    let (jev, mock) = scripted_jev(0.05);
    let tools = [weather()];
    let shapes = [
        json!({"name": "get_weather", "arguments": {"city": "Seoul"}}),
        json!({"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}),
        json!({"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}),
        json!({"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}),
    ];
    for shape in shapes {
        let verdict = dmcall::verify("Weather in Seoul?", &json!([shape]), &tools, &jev).unwrap();
        assert!(verdict.accepted && verdict.calls == seoul().unwrap() && verdict.arguments_probability == Some(0.95));
    }
    assert!(!dmcall::verify("Weather in Seoul?", &busan().unwrap(), &tools, &jev).unwrap().accepted);
    let count = mock.sent().len();
    let malformed =
        dmcall::verify("Weather?", &json!({"name": "get_weather", "arguments": "{oops"}), &tools, &jev).unwrap();
    assert!(!malformed.accepted && mock.sent().len() == count); // rejected locally; nothing is sent
    assert!(dmcall::verify("Tell me a joke.", &Value::Null, &tools, &jev).unwrap().calls.is_empty());
    let (yes, _) = scripted_jev(0.9);
    assert!(dmcall::verify("Tell me a joke.", &json!([]), &tools, &yes).unwrap().accepted);

    let history = json!([
        {"role": "user", "content": "Weather in Busan, then Seoul?"},
        {"role": "assistant", "content": null, "refusal": null, "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Busan\"}"}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "{\"temp\": 18}"},
    ]);
    assert!(dmcall::verify(&history, &seoul().unwrap(), &tools, &jev).unwrap().accepted);
    let sent = body_of(mock.sent().last().unwrap());
    let roles: Vec<&str> =
        sent["state"]["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "tool"]);
    assert!(
        sent["state"]["messages"][1].get("refusal").is_none() && sent["state"]["messages"][1].get("content").is_none()
    );
    assert!(dmcall::verify(&json!([{"role": "robot", "content": "x"}]), &seoul().unwrap(), &tools, &jev).is_err());
    let (builder, _, _) = agent(vec![]);
    assert!(builder.build().unwrap().verify("Weather in Seoul?", &seoul().unwrap()).unwrap().accepted);
}

#[test]
fn clients_can_be_shared_across_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<DMCall>();
    shareable::<Jev>();
    shareable::<dmcall::PPLXDecider>();
    shareable::<dmcall::OpenAI>();
    shareable::<dmcall::CallResult>();
    shareable::<Tool>();
}
