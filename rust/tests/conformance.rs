//! The language-neutral conformance files in `../spec` that every port must pass.

mod common;

use std::sync::Arc;

use common::*;
use dmcall::feedback::{represent, single_check, CUT_POINTS, LABELS};
use dmcall::providers::INCOMPLETE;
use dmcall::pyjson;
use dmcall::{
    Anthropic, AzureOpenAI, DMCall, DMCallBuilder, Feedback, Gemini, Generator, Jev, Meta, OpenAI, OpenAICompatible,
    OpenRouter, PPLXDecider, Tool, XAI,
};
use serde_json::{json, Value};

fn tools_of(value: &Value) -> Vec<Tool> {
    value.as_array().unwrap().iter().map(|tool| Tool::new(tool.clone()).unwrap()).collect()
}

/// Map a scenario's Python keyword options onto the builder.
fn with_options(mut builder: DMCallBuilder, options: &Value) -> DMCallBuilder {
    for (key, value) in options.as_object().unwrap() {
        let text = value.as_str().unwrap_or_default();
        builder = match key.as_str() {
            "feedback" => builder.feedback(text.parse().unwrap()),
            "history" => builder.history(text.parse().unwrap()),
            "checks" => builder.checks(text.parse().unwrap()),
            "on_exhaustion" => builder.on_exhaustion(text.parse().unwrap()),
            "max_repairs" => builder.max_repairs(value.as_u64().unwrap() as u32),
            "preserve_initial_no_call" => builder.preserve_initial_no_call(value.as_bool().unwrap()),
            other => panic!("unknown option {other}"),
        };
    }
    builder
}

// prompts.json -----------------------------------------------------------------------------------------

#[test]
fn prompts_file_matches_the_library() {
    let data = load("prompts.json");
    assert_eq!(data["system_prompt"], dmcall::SYSTEM_PROMPT);
    assert_eq!(data["history_header"], dmcall::prompts::HISTORY_HEADER);
    assert_eq!(data["history_footer"], dmcall::prompts::HISTORY_FOOTER);
    assert_eq!(data["repair_instruction"], dmcall::prompts::REPAIR_INSTRUCTION);
    assert_py_eq(&dmcall::FUNCTION_QUESTION, &data["function_question"], "function question");
    assert_py_eq(&dmcall::ARGUMENT_QUESTION, &data["argument_question"], "argument question");
    // The question objects are sent as JSON, so their key order is part of the request too.
    assert_eq!(pyjson::dumps(&dmcall::FUNCTION_QUESTION), pyjson::dumps(&data["function_question"]));
    assert_eq!(pyjson::dumps(&dmcall::ARGUMENT_QUESTION), pyjson::dumps(&data["argument_question"]));
    assert_eq!(data["labels"], json!(LABELS));
    let cuts: Vec<f64> = data["cut_points"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(cuts, CUT_POINTS);
    let defaults = &data["defaults"];
    assert_eq!(defaults["threshold"].as_f64(), Some(dmcall::DEFAULT_THRESHOLD));
    assert_eq!(defaults["max_repairs"].as_u64(), Some(dmcall::DEFAULT_MAX_REPAIRS as u64));
    assert_eq!(defaults["checks"], dmcall::Checks::default().as_str());
    assert_eq!(defaults["history"], dmcall::History::default().as_str());
    assert_eq!(defaults["feedback"], Feedback::default().as_str());
    assert_eq!(defaults["on_exhaustion"], dmcall::OnExhaustion::default().as_str());
    assert_eq!(
        defaults["max_output_tokens"].as_u64(),
        dmcall::ProviderOptions::default().max_output_tokens.map(u64::from)
    );
    let jev = &data["jev"];
    assert_eq!(jev["model"], Jev::DEFAULT_MODEL);
    assert_eq!(jev["base_url"], Jev::DEFAULT_BASE_URL);
    assert_eq!(jev["feedback_source"], Jev::FEEDBACK_SOURCE);
    let pplx = &data["pplx_decider"];
    assert_eq!(pplx["model_id"], PPLXDecider::MODEL_ID);
    assert_eq!(pplx["revision"], PPLXDecider::MODEL_REVISION);
    assert_eq!(pplx["calibration_temperature"].as_f64(), Some(PPLXDecider::CALIBRATION_TEMPERATURE));
    assert_eq!(pplx["max_input_tokens"].as_u64(), Some(PPLXDecider::MAX_INPUT_TOKENS));
    assert_eq!(pplx["server_model"], PPLXDecider::SERVER_MODEL);
    assert_eq!(pplx["wire_model"], PPLXDecider::WIRE_MODEL);
    assert_eq!(pplx["feedback_source"], PPLXDecider::FEEDBACK_SOURCE);
    let statuses: Vec<u64> = data["retry_statuses"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(statuses, dmcall::decision::RETRY_STATUSES.map(u64::from));
    for (reason, text) in INCOMPLETE {
        assert_eq!(data["incomplete_reasons"][reason], text);
    }
    assert_eq!(data["incomplete_reasons"].as_object().unwrap().len(), INCOMPLETE.len());
}

// json_text.json -----------------------------------------------------------------------------------------

#[test]
fn json_text() {
    let data = load("conformance/json_text.json");
    let cases = data["cases"].as_array().unwrap();
    for case in cases {
        let input = case["input"].as_str().unwrap();
        if !cfg!(feature = "big-integers") && input == "100000000000000000000" {
            continue; // integers beyond 64 bits need the default `big-integers` feature
        }
        let value = pyjson::loads(input).unwrap();
        assert_eq!(pyjson::dumps(&value), case["output"].as_str().unwrap(), "input {input}");
    }
    assert!(!cases.is_empty());
}

// schema_check.json --------------------------------------------------------------------------------------

#[test]
fn schema_check() {
    let data = load("conformance/schema_check.json");
    let cases = data["cases"].as_array().unwrap();
    for case in cases {
        let tools = case["tools"].as_array().unwrap();
        let result = dmcall::check_schema(tools, &case["calls"]).unwrap();
        assert_py_eq(&result.to_value(), &case["expected"], case["name"].as_str().unwrap());
        // Error order is part of the contract.
        assert_eq!(pyjson::dumps(&result.to_value()["errors"]), pyjson::dumps(&case["expected"]["errors"]));
    }
    assert!(!cases.is_empty());
    let invalid = data["configuration_errors"].as_array().unwrap();
    for case in invalid {
        let tools = case["tools"].as_array().unwrap();
        assert!(dmcall::check_schema(tools, &json!([])).is_err(), "{}", case["name"]);
    }
    assert_eq!(invalid.len(), 3);
}

// feedback.json -------------------------------------------------------------------------------------------

#[test]
fn feedback() {
    let data = load("conformance/feedback.json");
    for case in data["labels"].as_array().unwrap() {
        assert_eq!(dmcall::feedback::label_value(&case["p"]).unwrap(), case["label"], "p = {}", case["p"]);
    }
    for bad in data["invalid_probabilities"].as_array().unwrap() {
        assert!(dmcall::feedback::label_value(bad).is_err(), "{bad}");
    }
    for case in data["represent"].as_array().unwrap() {
        let representation: Feedback = case["representation"].as_str().unwrap().parse().unwrap();
        let shown = represent(&case["feedback"], representation).unwrap();
        assert_py_eq(&shown, &case["expected"], "represent");
        // The generator reads this record as text, so key order matters too.
        assert_eq!(pyjson::dumps(&shown), pyjson::dumps(&case["expected"]));
    }
    for case in data["single_check"].as_array().unwrap() {
        let kept = single_check(&case["feedback"], case["keep"].as_str().unwrap()).unwrap();
        assert_eq!(pyjson::dumps(&kept), pyjson::dumps(&case["expected"]));
    }
    for conflict in data["conflicts"]["feedback"].as_array().unwrap() {
        for representation in data["conflicts"]["representations"].as_array().unwrap() {
            let representation: Feedback = representation.as_str().unwrap().parse().unwrap();
            assert!(represent(conflict, representation).is_err(), "{conflict}");
        }
    }
}

// normalize_calls.json ------------------------------------------------------------------------------------

#[test]
fn normalize_calls() {
    let data = load("conformance/normalize_calls.json");
    for case in data["cases"].as_array().unwrap() {
        let calls = dmcall::normalize_calls(&case["input"]).unwrap();
        let actual = Value::Array(calls.iter().map(dmcall::Call::to_value).collect());
        assert_py_eq(&actual, &case["expected"], "normalize_calls");
    }
}

// providers.json -------------------------------------------------------------------------------------------

fn capture(generator: &dyn Generator, mock: &Mock, example: &Value, history: &[Value]) -> Value {
    let messages = messages_of(&example["messages"]);
    let tools = example["tools"].as_array().unwrap();
    // An empty response body is not a valid answer; only the request is recorded here.
    assert!(generator.generate(&messages, tools, history).is_err());
    let request = mock.sent().pop().unwrap();
    json!({"url": request.url, "headers": listed_headers(&request), "body": body_of(&request)})
}

#[test]
fn provider_requests() {
    isolate_env();
    let data = load("conformance/providers.json");
    let example = &data["example"];
    let history = data["history"].as_array().unwrap().clone();
    let no_tools = json!({"messages": [{"role": "user", "content": "hi"}], "tools": []});
    let mock = Mock::queue(vec![json!({}); 9]);
    let transport = || mock.clone() as dmcall::SharedTransport;
    let captured = [
        capture(
            &OpenAI::builder("gpt-5-nano")
                .api_key("KEY")
                .reasoning_effort("none")
                .transport(transport())
                .build()
                .unwrap(),
            &mock,
            example,
            &history,
        ),
        capture(
            &OpenAI::builder("gpt-5-nano").api_key("KEY").transport(transport()).build().unwrap(),
            &mock,
            &no_tools,
            &[],
        ),
        capture(
            &OpenAICompatible::builder("gemma-4-31b-it")
                .base_url("http://localhost:8000/v1")
                .transport(transport())
                .build()
                .unwrap(),
            &mock,
            example,
            &[],
        ),
        capture(&XAI::builder("grok-4").api_key("KEY").transport(transport()).build().unwrap(), &mock, example, &[]),
        capture(
            &OpenRouter::builder("meta/llama").api_key("KEY").temperature(0.2).transport(transport()).build().unwrap(),
            &mock,
            example,
            &[],
        ),
        capture(
            &Meta::builder("muse-spark-1.3").api_key("KEY").transport(transport()).build().unwrap(),
            &mock,
            example,
            &history,
        ),
        capture(
            &AzureOpenAI::builder("my deployment")
                .endpoint("https://res.openai.azure.com/")
                .api_key("KEY")
                .transport(transport())
                .build()
                .unwrap(),
            &mock,
            example,
            &history,
        ),
        capture(
            &Anthropic::builder("claude-sonnet-5-5").api_key("KEY").transport(transport()).build().unwrap(),
            &mock,
            example,
            &history,
        ),
        capture(
            &Gemini::builder("models/gemini-2.5-flash")
                .api_key("KEY")
                .temperature(0.0)
                .thinking_budget(0)
                .transport(transport())
                .build()
                .unwrap(),
            &mock,
            example,
            &history,
        ),
    ];
    let expected = data["requests"].as_array().unwrap();
    assert_eq!(captured.len(), expected.len());
    for (ours, theirs) in captured.iter().zip(expected) {
        let name = theirs["name"].as_str().unwrap();
        assert_eq!(ours["url"], theirs["url"], "{name}");
        assert_py_eq(&ours["headers"], &theirs["headers"], name);
        assert_py_eq(&ours["body"], &theirs["body"], name);
        // The history message is text: compare it character for character.
        let text = |body: &Value| match body.get("messages").or_else(|| body.get("contents")) {
            Some(Value::Array(items)) => pyjson::dumps(&Value::Array(items.clone())),
            _ => String::new(),
        };
        assert_eq!(text(&ours["body"]), text(&theirs["body"]), "{name}");
    }
}

#[test]
fn provider_decoding() {
    isolate_env();
    let data = load("conformance/providers.json");
    let example = &data["example"];
    let messages = messages_of(&example["messages"]);
    let tools = example["tools"].as_array().unwrap();
    let cases = data["decode"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mock = Mock::queue(vec![case["response"].clone()]);
        let generator: Box<dyn Generator> = match case["provider"].as_str().unwrap() {
            "openai" => Box::new(OpenAI::builder("m").api_key("KEY").transport(mock.clone()).build().unwrap()),
            "anthropic" => Box::new(Anthropic::builder("m").api_key("KEY").transport(mock.clone()).build().unwrap()),
            "gemini" => Box::new(Gemini::builder("m").api_key("KEY").transport(mock.clone()).build().unwrap()),
            other => panic!("unknown provider {other}"),
        };
        let proposal = generator.generate(&messages, tools, &[]).unwrap();
        let actual = json!({
            "calls": calls_json(&proposal.calls),
            "content": proposal.content,
            "parse_errors": proposal.parse_errors,
            "incomplete": proposal.incomplete,
            "usage": Value::Object(proposal.usage.clone()),
            "model": proposal.model,
        });
        assert_py_eq(&actual, &case["expected"], name);
    }
    assert_eq!(cases.len(), 18);

    let filtered = &data["filtered_prompt"];
    let status = filtered["status"].as_u64().unwrap() as u16;
    let mock = Mock::responses(vec![dmcall::HttpResponse::json(status, &filtered["response"])]);
    let openai = OpenAI::builder("m").api_key("KEY").transport(mock).build().unwrap();
    let proposal = openai.generate(&messages, tools, &[]).unwrap();
    assert!(proposal.calls.is_none() && proposal.usage.is_empty());
    assert_eq!(proposal.incomplete.as_deref(), filtered["expected_incomplete"].as_str());
}

// paper_parity.json ----------------------------------------------------------------------------------------

#[test]
fn paper_parity_loop_scenarios() {
    isolate_env();
    let golden = load("conformance/paper_parity.json");
    let tools = tools_of(&golden["tools"]);
    let generator_key = golden["generator_api_key"].as_str().unwrap();
    let jev_key = golden["jev_api_key"].as_str().unwrap();
    let scenarios = golden["scenarios"].as_object().unwrap();
    for (name, scenario) in scenarios {
        let served_generator = Mock::queue(scenario["generator_responses"].as_array().unwrap().clone());
        let served_decision = Mock::queue(scenario["decision_responses"].as_array().unwrap().clone());
        let jev = Jev::builder().api_key(jev_key).transport(served_decision.clone()).build().unwrap();
        let builder = if scenario["provider"] == "anthropic" {
            DMCall::builder(
                Anthropic::builder("claude-sonnet-5-5")
                    .api_key(generator_key)
                    .transport(served_generator.clone())
                    .build()
                    .unwrap(),
            )
        } else {
            DMCall::builder(
                OpenAI::builder("gpt-5-nano")
                    .api_key(generator_key)
                    .reasoning_effort("none")
                    .transport(served_generator.clone())
                    .build()
                    .unwrap(),
            )
        };
        let agent =
            with_options(builder.decision_model(jev).tools(tools.clone()), &scenario["options"]).build().unwrap();
        let result = agent.call(&golden["messages"]).unwrap();

        let sent = served_generator.bodies();
        let expected = scenario["generator_requests"].as_array().unwrap();
        assert_eq!(sent.len(), expected.len(), "{name}: generator requests");
        for (ours, theirs) in sent.iter().zip(expected) {
            assert_py_eq(ours, theirs, name);
            // The history message is sent as text, so its exact characters matter, not only its JSON.
            assert_eq!(pyjson::dumps(ours), pyjson::dumps(theirs), "{name}");
        }
        let decided = served_decision.bodies();
        let expected = scenario["decision_requests"].as_array().unwrap();
        assert_eq!(decided.len(), expected.len(), "{name}: decision requests");
        for (ours, theirs) in decided.iter().zip(expected) {
            assert_py_eq(ours, theirs, name);
            assert_eq!(pyjson::dumps(ours), pyjson::dumps(theirs), "{name}");
        }
        assert_py_eq(&subset(&summary(&result), &scenario["result"]), &scenario["result"], name);
    }
    assert_eq!(scenarios.len(), 14);
}

#[test]
fn paper_parity_verify() {
    isolate_env();
    let golden = load("conformance/paper_parity.json");
    let recorded = &golden["verify"];
    let tools = tools_of(&golden["tools"]);
    let served = Mock::queue(recorded["decision_responses"].as_array().unwrap().clone());
    let jev =
        Jev::builder().api_key(golden["jev_api_key"].as_str().unwrap()).transport(served.clone()).build().unwrap();
    let calls = recorded["calls"].as_array().unwrap();
    let verdicts = [
        dmcall::verify(&recorded["messages"], &calls[0], &tools, &jev).unwrap(),
        dmcall::verify(recorded["second_messages"].as_str().unwrap(), &calls[1], &tools, &jev).unwrap(),
    ];
    let sent = served.bodies();
    let expected = recorded["decision_requests"].as_array().unwrap();
    assert_eq!(sent.len(), expected.len());
    for (ours, theirs) in sent.iter().zip(expected) {
        assert_eq!(pyjson::dumps(ours), pyjson::dumps(theirs));
    }
    let actual: Vec<Value> = verdicts
        .iter()
        .map(|v| {
            json!({
                "accepted": v.accepted,
                "calls": v.calls.iter().map(dmcall::Call::to_value).collect::<Vec<_>>(),
                "function_probability": v.function_probability,
                "arguments_probability": v.arguments_probability,
                "schema_errors": v.schema_errors.iter().map(|e| e.to_value()).collect::<Vec<_>>(),
                "feedback": v.feedback,
            })
        })
        .collect();
    assert_py_eq(&Value::Array(actual), &recorded["verdicts"], "verdicts");
}

// loop_edges.json -------------------------------------------------------------------------------------------

fn edge_generator(spec: &Value, transport: Arc<Mock>) -> Box<dyn Generator> {
    let model = spec["model"].as_str().unwrap();
    let options = spec.get("options").cloned().unwrap_or(json!({}));
    let text = |key: &str| options.get(key).and_then(Value::as_str).map(str::to_string);
    let number = |key: &str| options.get(key).and_then(Value::as_f64);
    macro_rules! common {
        ($builder:expr) => {{
            let mut builder = $builder.transport(transport.clone());
            if let Some(key) = text("api_key") {
                builder = builder.api_key(key);
            }
            if let Some(url) = text("base_url") {
                builder = builder.base_url(url);
            }
            if let Some(temperature) = number("temperature") {
                builder = builder.temperature(temperature);
            }
            builder
        }};
    }
    match spec["kind"].as_str().unwrap() {
        "openai" => {
            let mut builder = common!(OpenAI::builder(model));
            if let Some(effort) = text("reasoning_effort") {
                builder = builder.reasoning_effort(effort);
            }
            Box::new(builder.build().unwrap())
        }
        "openai-compatible" => Box::new(common!(OpenAICompatible::builder(model)).build().unwrap()),
        "azure" => Box::new(common!(AzureOpenAI::builder(model)).endpoint(text("endpoint").unwrap()).build().unwrap()),
        "xai" => Box::new(common!(XAI::builder(model)).build().unwrap()),
        "anthropic" => Box::new(common!(Anthropic::builder(model)).build().unwrap()),
        "gemini" => Box::new(common!(Gemini::builder(model)).build().unwrap()),
        other => panic!("unknown generator {other}"),
    }
}

#[test]
fn loop_edges() {
    isolate_env();
    let data = load("conformance/loop_edges.json");
    let tools = tools_of(&data["tools"]);
    let scenarios = data["scenarios"].as_object().unwrap();
    for (name, scenario) in scenarios {
        let served_generator = Mock::queue(scenario["generator_responses"].as_array().unwrap().clone());
        let served_decision = Mock::queue(scenario["decision_responses"].as_array().unwrap().clone());
        let generator = edge_generator(&scenario["generator"], served_generator.clone());
        let builder = DMCall::builder(generator).tools(tools.clone());
        let builder = if scenario["decision_model"]["kind"] == "pplx" {
            let url = scenario["decision_model"]["base_url"].as_str().unwrap();
            builder.decision_model(PPLXDecider::builder(url).transport(served_decision.clone()).build().unwrap())
        } else {
            builder
                .decision_model(Jev::builder().api_key("JEV-KEY").transport(served_decision.clone()).build().unwrap())
        };
        let agent = with_options(builder, &scenario["options"]).build().unwrap();
        let result = agent.call(&data["messages"]).unwrap();

        for (mock, key) in [(&served_generator, "generator_requests"), (&served_decision, "decision_requests")] {
            let sent = mock.sent();
            let expected = scenario[key].as_array().unwrap();
            assert_eq!(sent.len(), expected.len(), "{name}: {key}");
            for (ours, theirs) in sent.iter().zip(expected) {
                assert_eq!(ours.url, theirs["url"].as_str().unwrap(), "{name}: {key}");
                assert_py_eq(&listed_headers(ours), &theirs["headers"], &format!("{name}: {key} headers"));
                assert_py_eq(&body_of(ours), &theirs["body"], &format!("{name}: {key}"));
                if key == "generator_requests" {
                    let text =
                        |body: &Value| pyjson::dumps(body.get("messages").or_else(|| body.get("contents")).unwrap());
                    assert_eq!(text(&body_of(ours)), text(&theirs["body"]), "{name}: history text");
                }
            }
        }
        assert_py_eq(&summary(&result), &scenario["result"], name);
    }
    assert_eq!(scenarios.len(), 7);
}
