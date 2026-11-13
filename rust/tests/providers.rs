//! Generators over plain HTTP: requests, decoding, unreadable outputs and errors.
//! Ported from the Python reference's `tests/test_providers.py` (environment cases are in `env.rs`).

mod common;

use std::sync::Arc;

use common::*;
use dmcall::{
    Anthropic, AzureOpenAI, Error, Gemini, Generator, HttpResponse, Message, OpenAI, OpenAICompatible, Proposal,
};
use serde_json::{json, Value};

const SECRET: &str = "sk-test-SECRET-123";

fn weather() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}})
}

fn messages() -> Vec<Message> {
    vec![Message::system("Be brief."), Message::user("Weather in Seoul?")]
}

fn history() -> Vec<Value> {
    vec![
        json!({"calls": [{"name": "get_weather", "arguments": {"city": "Busan"}}],
               "feedback": {"failed_checks": ["arguments"], "instruction": "Reconsider."}}),
        json!({"round": 2, "calls": [], "feedback": {"failed_checks": ["function_selection"]}}),
    ]
}

/// A Chat Completions response body.
fn chat(calls: &[(&str, &str)], content: Value, finish: Option<&str>, extra: Value) -> Value {
    let tool_calls: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (name, arguments))| json!({"id": format!("call_{i}"), "type": "function", "function": {"name": name, "arguments": arguments}}))
        .collect();
    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls.clone());
    }
    for (key, value) in extra.as_object().unwrap() {
        message[key] = value.clone();
    }
    let finish = finish.unwrap_or(if tool_calls.is_empty() { "stop" } else { "tool_calls" });
    json!({"model": "resolved-model", "choices": [{"finish_reason": finish, "message": message}]})
}

fn seoul_args() -> &'static str {
    "{\"city\": \"Seoul\"}"
}

fn generate(generator: &dyn Generator, history: &[Value]) -> dmcall::Result<Proposal> {
    generator.generate(&messages(), &[weather()], history)
}

#[test]
fn openai_request_and_decoding() {
    isolate_env();
    let mut body = chat(&[("get_weather", seoul_args())], Value::Null, None, json!({}));
    body["usage"] =
        json!({"prompt_tokens": 120, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 64}});
    let mock = Mock::queue(vec![body]);
    let model =
        OpenAI::builder("gpt-5-nano").api_key(SECRET).reasoning_effort("none").transport(mock.clone()).build().unwrap();
    let proposal = generate(&model, &history()).unwrap();
    let request = &mock.sent()[0];
    let sent = body_of(request);
    assert_eq!(request.url, "https://api.openai.com/v1/chat/completions");
    assert_eq!(request.header("authorization"), Some(format!("Bearer {SECRET}").as_str()));
    assert_py_eq(&sent["messages"][0], &json!({"role": "system", "content": dmcall::SYSTEM_PROMPT}), "system");
    assert!(sent["messages"][1]["content"]
        .as_str()
        .unwrap()
        .starts_with("Accumulated revision history (oldest first; proposals NOT executed)"));
    let original: Vec<Value> = messages().iter().map(Message::to_value).collect();
    assert_py_eq(
        &Value::Array(sent["messages"].as_array().unwrap()[2..].to_vec()),
        &Value::Array(original),
        "original request",
    );
    assert_py_eq(&sent["tools"], &json!([weather()]), "tools");
    assert_eq!((&sent["tool_choice"], &sent["parallel_tool_calls"]), (&json!("auto"), &json!(false)));
    assert_eq!((&sent["max_completion_tokens"], &sent["reasoning_effort"]), (&json!(4096), &json!("none")));
    assert_py_eq(
        &calls_json(&proposal.calls),
        &json!([{"name": "get_weather", "arguments": {"city": "Seoul"}}]),
        "calls",
    );
    assert!(proposal.incomplete.is_none());
    assert_py_eq(
        &Value::Object(proposal.usage),
        &json!({"input_tokens": 120, "cached_input_tokens": 64, "output_tokens": 9}),
        "usage",
    );
    assert_eq!((proposal.model.as_deref(), proposal.provider.as_str()), (Some("resolved-model"), "openai"));
    assert!(!format!("{model:?}").contains(SECRET));
}

#[test]
fn history_lists_rounds_oldest_first() {
    isolate_env();
    let mock = Mock::queue(vec![chat(&[], Value::Null, None, json!({}))]);
    let model = OpenAI::builder("m").api_key("k").transport(mock.clone()).build().unwrap();
    generate(&model, &history()).unwrap();
    let text = body_of(&mock.sent()[0])["messages"][1]["content"].as_str().unwrap().to_string();
    let inner = text.split_once('\n').unwrap().1.rsplit_once("\nUse these previous judgments").unwrap().0;
    let records = dmcall::pyjson::loads(inner).unwrap();
    assert_eq!(records[0]["round"], json!(1));
    assert_eq!(records[1]["round"], json!(2));
    assert_py_eq(&records[0]["calls"], &history()[0]["calls"], "calls");
    assert_eq!(records[1]["calls"], json!([]));
    assert!(text.ends_with("when answering the original request below."));
}

#[test]
fn openai_outcomes() {
    isolate_env();
    let cases = [
        (chat(&[], json!("No function applies."), None, json!({})), json!([]), None),
        (chat(&[], Value::Null, None, json!({})), json!([]), None),
        (chat(&[], Value::Null, Some("length"), json!({})), Value::Null, Some("length")),
        (chat(&[], Value::Null, Some("content_filter"), json!({})), Value::Null, Some("content_filter")),
        (chat(&[], Value::Null, None, json!({"refusal": "I can't help with that."})), Value::Null, Some("refusal")),
        (
            chat(&[("get_weather", "{not json")], Value::Null, None, json!({})),
            json!([{"name": "get_weather", "arguments": null}]),
            None,
        ),
    ];
    for (body, calls, incomplete) in cases {
        let model = OpenAI::builder("m").api_key("k").transport(Mock::queue(vec![body.clone()])).build().unwrap();
        let proposal = generate(&model, &[]).unwrap();
        assert_py_eq(&calls_json(&proposal.calls), &calls, "calls");
        assert_eq!(proposal.incomplete.as_deref(), incomplete);
        let malformed = proposal.calls.as_ref().is_some_and(|c| c.first().is_some_and(|c| c.arguments.is_null()));
        if incomplete.is_some() || malformed {
            assert!(!proposal.parse_errors.is_empty());
        }
    }
}

#[test]
fn filtered_prompt_is_an_unreadable_candidate_not_an_error() {
    isolate_env();
    let body = json!({"error": {"code": "content_filter", "message": "filtered"}});
    let mock = Mock::responses(vec![HttpResponse::json(400, &body)]);
    let proposal = generate(&OpenAI::builder("m").api_key("k").transport(mock).build().unwrap(), &[]).unwrap();
    assert!(
        proposal.calls.is_none()
            && proposal.incomplete.as_deref() == Some("prompt_filter")
            && proposal.usage.is_empty()
    );
}

#[test]
fn http_errors_are_sanitized_and_retries_bounded() {
    isolate_env();
    let body = json!({"error": {"code": "invalid_api_key", "message": SECRET}});
    let mock = Mock::with(move |_| Ok(HttpResponse::json(401, &body).with_header("x-request-id", "req-1")));
    let model =
        OpenAI::builder("m").api_key(SECRET).max_retries(2).sleep(no_sleep()).transport(mock.clone()).build().unwrap();
    let Err(Error::Provider(error)) = generate(&model, &[]) else { panic!("expected a provider error") };
    assert_eq!(mock.sent().len(), 1);
    assert_eq!(
        (error.status_code, error.provider_code.as_deref(), error.request_id.as_deref()),
        (Some(401), Some("invalid_api_key"), Some("req-1"))
    );
    assert!(!error.to_string().contains(SECRET));
    let answers = Mock::responses(vec![
        HttpResponse::new(429, "").with_header("retry-after", "1"),
        HttpResponse::new(503, ""),
        HttpResponse::json(200, &chat(&[("get_weather", seoul_args())], Value::Null, None, json!({}))),
    ]);
    let retried =
        OpenAI::builder("m").api_key("k").max_retries(2).sleep(no_sleep()).transport(answers).build().unwrap();
    let proposal = generate(&retried, &[]).unwrap();
    assert_eq!(proposal.api_calls, 3);
    assert_py_eq(
        &calls_json(&proposal.calls),
        &json!([{"name": "get_weather", "arguments": {"city": "Seoul"}}]),
        "calls",
    );
    let exhausted = OpenAI::builder("m")
        .api_key("k")
        .max_retries(1)
        .sleep(no_sleep())
        .transport(Mock::with(|_| Ok(HttpResponse::new(500, ""))))
        .build()
        .unwrap();
    let error = generate(&exhausted, &[]).unwrap_err();
    assert!(error.to_string().contains("HTTP 500"));
    let unreachable = OpenAI::builder("m")
        .api_key("k")
        .max_retries(1)
        .sleep(no_sleep())
        .transport(Mock::with(|_| Err(dmcall::TransportError::new("Timeout"))))
        .build()
        .unwrap();
    let Err(Error::Provider(error)) = generate(&unreachable, &[]) else { panic!("expected a provider error") };
    assert_eq!(error.to_string(), "openai transport failed after 2 attempts (Timeout).");
    let garbage = OpenAI::builder("m")
        .api_key("k")
        .transport(Mock::with(|_| Ok(HttpResponse::new(200, "<html>"))))
        .build()
        .unwrap();
    assert!(generate(&garbage, &[]).unwrap_err().to_string().contains("did not match its protocol"));
}

#[test]
fn bad_options_are_rejected() {
    isolate_env();
    let mock = Mock::queue(vec![]);
    assert!(OpenAI::builder("m")
        .api_key("k")
        .base_url("https://user:pass@example.com/v1")
        .transport(mock.clone())
        .build()
        .is_err());
    let error = OpenAI::builder("m").api_key("a\nb").transport(mock.clone()).build().unwrap_err();
    assert!(error.to_string().contains("single-line"));
    assert!(OpenAI::builder(" ").api_key("k").transport(mock.clone()).build().is_err());
    assert!(OpenAI::builder("m").api_key("k").max_output_tokens(Some(0)).transport(mock.clone()).build().is_err());
    assert!(OpenAICompatible::builder("m")
        .path("chat")
        .base_url("http://localhost:1")
        .transport(mock.clone())
        .build()
        .is_err());
    let error = OpenAICompatible::builder("m").transport(mock).build().unwrap_err();
    assert!(error.to_string().contains("base_url"));
}

#[test]
fn any_openai_compatible_server_needs_no_key() {
    isolate_env();
    let mock = Mock::queue(vec![chat(&[], Value::Null, None, json!({}))]);
    let model = OpenAICompatible::builder("gemma-4-31b-it")
        .base_url("http://localhost:8000/v1")
        .transport(mock.clone())
        .build()
        .unwrap();
    model.generate(&[Message::user("hi")], &[], &[]).unwrap();
    let request = &mock.sent()[0];
    let body = body_of(request);
    assert_eq!(request.url, "http://localhost:8000/v1/chat/completions");
    assert!(request.header("authorization").is_none());
    for key in ["tools", "tool_choice", "parallel_tool_calls"] {
        assert!(body.get(key).is_none(), "{key}"); // no functions, no tool settings
    }
}

#[test]
fn azure_deployment_url_and_header() {
    isolate_env();
    let mock = Mock::queue(vec![chat(&[], Value::Null, None, json!({}))]);
    let model = AzureOpenAI::builder("my deployment")
        .endpoint("https://res.openai.azure.com/")
        .api_key(SECRET)
        .transport(mock.clone())
        .build()
        .unwrap();
    generate(&model, &[]).unwrap();
    let request = &mock.sent()[0];
    assert_eq!(request.url, "https://res.openai.azure.com/openai/deployments/my%20deployment/chat/completions?api-version=2024-12-01-preview");
    assert_eq!(request.header("api-key"), Some(SECRET));
    assert!(request.header("authorization").is_none());
    let error = AzureOpenAI::builder("d")
        .endpoint("https://r")
        .api_key("k")
        .api_version("latest")
        .transport(mock.clone())
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("api_version"));
    assert!(AzureOpenAI::builder("..").endpoint("https://r").api_key("k").transport(mock).build().is_err());
}

#[test]
fn anthropic_request_and_decoding() {
    isolate_env();
    let body = json!({"type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "tool_use",
        "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}],
        "usage": {"input_tokens": 50, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 20, "output_tokens": 7}});
    let mock = Mock::queue(vec![body.clone()]);
    let model = Anthropic::builder("claude-x").api_key(SECRET).transport(mock.clone()).build().unwrap();
    let proposal = generate(&model, &history()).unwrap();
    let request = &mock.sent()[0];
    let sent = body_of(request);
    assert_eq!(request.url, "https://api.anthropic.com/v1/messages");
    assert_eq!((request.header("x-api-key"), request.header("anthropic-version")), (Some(SECRET), Some("2023-06-01")));
    assert_eq!(sent["system"].as_str().unwrap(), format!("{}\n\nBe brief.", dmcall::SYSTEM_PROMPT));
    let roles: Vec<&str> = sent["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "user"]);
    assert_py_eq(&sent["tools"][0]["input_schema"], &weather()["function"]["parameters"], "input_schema");
    assert_py_eq(&sent["tool_choice"], &json!({"type": "auto", "disable_parallel_tool_use": true}), "tool_choice");
    assert_eq!(proposal.content.as_deref(), Some("Checking."));
    assert_py_eq(
        &Value::Object(proposal.usage),
        &json!({"input_tokens": 100, "cached_input_tokens": 30, "output_tokens": 7}),
        "usage",
    );
    for (stop, incomplete) in [("max_tokens", "length"), ("refusal", "refusal")] {
        let mut cut = body.clone();
        cut["stop_reason"] = json!(stop);
        cut["content"] = json!([]);
        let model = Anthropic::builder("c").api_key("k").transport(Mock::queue(vec![cut])).build().unwrap();
        assert_eq!(generate(&model, &[]).unwrap().incomplete.as_deref(), Some(incomplete));
    }
}

#[test]
fn gemini_request_and_decoding() {
    isolate_env();
    let body = json!({"modelVersion": "gemini-x-001", "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
        {"text": "thinking", "thought": true}, {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]}}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 6, "thoughtsTokenCount": 40}});
    let mock = Mock::queue(vec![body]);
    let model = Gemini::builder("models/gemini-x")
        .api_key(SECRET)
        .temperature(0.0)
        .thinking_budget(0)
        .transport(mock.clone())
        .build()
        .unwrap();
    let mut conversation = messages();
    conversation.extend([Message::assistant("Which unit?"), Message::user("Celsius.")]);
    let proposal = model.generate(&conversation, &[weather()], &[]).unwrap();
    let request = &mock.sent()[0];
    let sent = body_of(request);
    assert_eq!(request.url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent");
    assert_eq!(request.header("x-goog-api-key"), Some(SECRET));
    assert_eq!(
        sent["systemInstruction"]["parts"][0]["text"].as_str().unwrap(),
        format!("{}\n\nBe brief.", dmcall::SYSTEM_PROMPT)
    );
    let roles: Vec<&str> = sent["contents"].as_array().unwrap().iter().map(|c| c["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "model", "user"]);
    let declaration = &sent["tools"][0]["functionDeclarations"][0];
    assert_py_eq(&declaration["parametersJsonSchema"], &weather()["function"]["parameters"], "declaration");
    assert!(declaration.get("parameters").is_none());
    assert_py_eq(&sent["toolConfig"], &json!({"functionCallingConfig": {"mode": "AUTO"}}), "toolConfig");
    assert_py_eq(
        &sent["generationConfig"],
        &json!({"maxOutputTokens": 4096, "temperature": 0, "thinkingConfig": {"thinkingBudget": 0}}),
        "config",
    );
    assert!(proposal.content.is_none() && proposal.model.as_deref() == Some("gemini-x-001"));
    assert_py_eq(&Value::Object(proposal.usage), &json!({"input_tokens": 80, "output_tokens": 46}), "usage");
}

#[test]
fn gemini_malformed_native_calls_and_invalid_finish_types_are_unreadable() {
    for call in [Value::Null, json!("not-an-object"), json!([]), json!(true)] {
        let model = Gemini::builder("m")
            .api_key("KEY")
            .transport(Mock::queue(vec![
                json!({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"functionCall": call}]}}]}),
            ]))
            .build()
            .unwrap();
        let result = generate(&model, &[]).unwrap();
        assert!(result.calls.is_none());
        assert_eq!(result.incomplete.as_deref(), Some("malformed_call"));
    }
    for reason in [json!(123), json!(true)] {
        let model = Gemini::builder("m")
            .api_key("KEY")
            .transport(Mock::queue(vec![json!({"candidates": [{"finishReason": reason, "content": {"parts": []}}]})]))
            .build()
            .unwrap();
        let result = generate(&model, &[]).unwrap();
        assert!(result.calls.is_none());
        assert_eq!(result.incomplete.as_deref(), Some("abnormal_stop"));
    }
}

#[test]
fn gemini_unreadable_outputs() {
    isolate_env();
    let cases = [
        (json!({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}), "length"),
        (json!({"candidates": [{"finishReason": "SAFETY"}]}), "content_filter"),
        (json!({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]}), "malformed_call"),
        (json!({"promptFeedback": {"blockReason": "SAFETY"}}), "prompt_filter"),
    ];
    for (body, incomplete) in cases {
        let model = Gemini::builder("g").api_key("k").transport(Mock::queue(vec![body])).build().unwrap();
        let proposal = generate(&model, &[]).unwrap();
        assert!(proposal.calls.is_none());
        assert_eq!(proposal.incomplete.as_deref(), Some(incomplete));
    }
    let empty =
        Gemini::builder("g").api_key("k").transport(Mock::queue(vec![json!({"candidates": []})])).build().unwrap();
    assert!(generate(&empty, &[]).unwrap_err().to_string().contains("no candidates"));
}

#[test]
fn custom_generators_plug_in() {
    isolate_env();
    struct Fixed;
    impl Generator for Fixed {
        fn generate(&self, _: &[Message], _: &[Value], _: &[Value]) -> dmcall::Result<Proposal> {
            Ok(Proposal::with_calls(vec![dmcall::Call::new("get_weather", json!({"city": "Seoul"}))]))
        }
    }
    let jev = dmcall::Jev::builder()
        .api_key("k")
        .transport(Mock::queue(vec![json!({"model": "jev-1.13.0", "answers": {
            "function_correct": {"type": "noul", "noul": 0.9}, "arguments_correct": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 1, "output_tokens": 0}})]))
        .build()
        .unwrap();
    let agent = dmcall::DMCall::builder(Arc::new(Fixed))
        .decision_model(jev)
        .tool(dmcall::Tool::new(weather()).unwrap())
        .build()
        .unwrap();
    assert_eq!(agent.call("Weather in Seoul?").unwrap().accepted, Some(true));
}
