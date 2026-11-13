//! Keys, endpoints and models from environment variables. These tests change the process
//! environment, so they run one at a time in their own test binary.

mod common;

use std::sync::{Mutex, MutexGuard};

use common::*;
#[cfg(feature = "ureq")]
use dmcall::{provider_from_spec, provider_from_spec_with, DMCall, ProviderKind};
use dmcall::{AzureOpenAI, DecisionModel, Generator, Jev, Message, Meta, OpenAI, OpenRouter, ProviderOptions, XAI};
use serde_json::json;

static ENV: Mutex<()> = Mutex::new(());

fn clean_env() -> MutexGuard<'static, ()> {
    let guard = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    for name in KEY_NAMES {
        std::env::remove_var(name);
    }
    guard
}

fn empty_chat() -> serde_json::Value {
    json!({"model": "m", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": null}}]})
}

#[test]
fn keys_come_from_arguments_or_environment() {
    let _guard = clean_env();
    assert!(OpenAI::new("m").unwrap_err().to_string().contains("OPENAI_API_KEY"));
    std::env::set_var("OPENAI_API_KEY", "from-env");
    std::env::set_var("OPENAI_BASE_URL", "https://proxy.example.com/v1/");
    let mock = Mock::queue(vec![empty_chat()]);
    let model = OpenAI::builder("m").transport(mock.clone()).build().unwrap();
    model.generate(&[Message::user("hi")], &[], &[]).unwrap();
    let request = &mock.sent()[0];
    assert_eq!(request.header("authorization"), Some("Bearer from-env"));
    assert_eq!(request.url, "https://proxy.example.com/v1/chat/completions");
    // An explicit key wins over the environment.
    let explicit = OpenAI::builder("m").api_key("explicit").transport(mock.clone()).build().unwrap();
    assert_eq!(explicit.model(), "m");
}

#[test]
fn openai_compatible_hosts_read_their_own_keys() {
    let _guard = clean_env();
    for (name, url, field) in [
        ("XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "max_tokens"),
        ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", "max_tokens"),
    ] {
        std::env::set_var(name, "k");
        let mock = Mock::queue(vec![empty_chat()]);
        let generator: Box<dyn Generator> = if name == "XAI_API_KEY" {
            Box::new(XAI::builder("m").transport(mock.clone()).build().unwrap())
        } else {
            Box::new(OpenRouter::builder("m").transport(mock.clone()).build().unwrap())
        };
        generator.generate(&[Message::user("hi")], &[], &[]).unwrap();
        assert_eq!(mock.sent()[0].url, url);
        assert_eq!(body_of(&mock.sent()[0])[field], json!(4096));
    }
}

#[test]
fn azure_reads_endpoint_and_key() {
    let _guard = clean_env();
    std::env::set_var("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/");
    std::env::set_var("AZURE_OPENAI_API_KEY", "secret");
    let mock = Mock::queue(vec![empty_chat()]);
    AzureOpenAI::builder("my deployment")
        .transport(mock.clone())
        .build()
        .unwrap()
        .generate(&[Message::user("hi")], &[], &[])
        .unwrap();
    assert_eq!(mock.sent()[0].url, "https://res.openai.azure.com/openai/deployments/my%20deployment/chat/completions?api-version=2024-12-01-preview");
    assert_eq!(mock.sent()[0].header("api-key"), Some("secret"));
    std::env::remove_var("AZURE_OPENAI_ENDPOINT");
    let error = AzureOpenAI::builder("d").transport(mock).build().unwrap_err();
    assert!(error.to_string().contains("AZURE_OPENAI_ENDPOINT"));
}

#[test]
#[cfg(feature = "ureq")] // these build the default HTTP transport
fn provider_specs() {
    let _guard = clean_env();
    for name in
        ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY", "MODEL_API_KEY"]
    {
        std::env::set_var(name, "k");
    }
    let expected = [
        ("openai:gpt-5-nano", ProviderKind::OpenAI),
        ("anthropic:claude-x", ProviderKind::Anthropic),
        ("claude:c", ProviderKind::Anthropic),
        ("gemini:g", ProviderKind::Gemini),
        ("google:g", ProviderKind::Gemini),
        ("xai:grok-4", ProviderKind::XAI),
        ("grok:grok-4", ProviderKind::XAI),
        ("openrouter:meta/llama", ProviderKind::OpenRouter),
        ("meta:muse-spark-1.3", ProviderKind::Meta),
    ];
    for (spec, kind) in expected {
        let built = provider_from_spec(spec).unwrap();
        assert_eq!(built.kind(), kind, "{spec}");
        assert_eq!(built.model(), spec.split_once(':').unwrap().1);
    }
    let options = ProviderOptions {
        endpoint: Some("https://r.openai.azure.com".into()),
        api_key: Some("k".into()),
        ..ProviderOptions::default()
    };
    assert_eq!(provider_from_spec_with("azure:dep", options).unwrap().kind(), ProviderKind::AzureOpenAI);
    for bad in ["gpt-5-nano", "openai:", "mystery:model"] {
        assert!(provider_from_spec(bad).is_err(), "{bad}");
    }
    let wrong = ProviderOptions { thinking_budget: Some(0), ..ProviderOptions::default() };
    assert!(provider_from_spec_with("openai:m", wrong).is_err());
}

#[test]
fn jev_reads_key_and_model_from_the_environment() {
    let _guard = clean_env();
    assert!(Jev::from_env().unwrap_err().to_string().contains("TYPESAFE_API_KEY"));
    std::env::set_var("TYPESAFE_API_KEY", "sk-env");
    std::env::set_var("TYPESAFE_MODEL", "jev-2");
    let mock =
        Mock::queue(vec![json!({"model": "jev-2", "answers": {"function_correct": {"type": "noul", "noul": 0.9}},
                                        "usage": {"input_tokens": 1, "output_tokens": 0}})]);
    let jev = Jev::builder().transport(mock.clone()).build().unwrap();
    assert_eq!((jev.model(), jev.base_url()), ("jev-2", "https://api.typesafe.ai"));
    jev.judge(&[json!({"role": "user", "content": "hi"})], &[], Some(&[])).unwrap();
    assert_eq!(mock.sent()[0].header("authorization"), Some("Bearer sk-env"));
    assert!(!format!("{jev:?}").contains("sk-env"));
}

#[test]
#[cfg(feature = "ureq")] // these build the default HTTP transport
fn dmcall_builds_the_default_jev() {
    let _guard = clean_env();
    std::env::set_var("TYPESAFE_API_KEY", "jev-env");
    std::env::set_var("OPENAI_API_KEY", "openai-env");
    let built = DMCall::builder(provider_from_spec("openai:gpt-5-nano").unwrap()).threshold(0.9).build().unwrap();
    let model = built.decision_model();
    assert_eq!((model.function_threshold(), model.arguments_threshold()), (0.9, 0.9));
    let built = DMCall::builder(provider_from_spec("openai:gpt-5-nano").unwrap()).build().unwrap();
    let model = built.decision_model();
    assert_eq!(model.model(), "jev-1.13.0"); // the paper's settings by default
    assert_eq!((model.function_threshold(), model.arguments_threshold()), (0.5, 0.5));
    let built = DMCall::builder(provider_from_spec("openai:gpt-5-nano").unwrap()).jev_api_key("explicit").build();
    assert!(built.is_ok());
}

#[test]
fn meta_muse_request() {
    let _guard = clean_env();
    std::env::set_var("MODEL_API_KEY", "sk-test-SECRET-123");
    let reply = json!({"model": "muse-spark-1.3", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": null, "tool_calls": [{"id": "call_0", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]}}]});
    let mock = Mock::queue(vec![reply]);
    let model = Meta::builder("muse-spark-1.3").transport(mock.clone()).build().unwrap();
    let weather = json!({"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}});
    let history = [
        json!({"calls": [{"name": "get_weather", "arguments": {"city": "Busan"}}],
               "feedback": {"failed_checks": ["arguments"], "instruction": "Reconsider."}}),
        json!({"round": 2, "calls": [], "feedback": {"failed_checks": ["function_selection"]}}),
    ];
    let messages = [Message::system("Be brief."), Message::user("Weather in Seoul?")];
    let proposal = model.generate(&messages, &[weather], &history).unwrap();
    let request = &mock.sent()[0];
    let body = body_of(request);
    assert_eq!(request.url, "https://api.meta.ai/v1/chat/completions");
    assert_eq!(request.header("authorization"), Some("Bearer sk-test-SECRET-123"));
    assert_eq!((&body["model"], &body["tool_choice"]), (&json!("muse-spark-1.3"), &json!("auto")));
    assert_eq!(body["max_completion_tokens"], json!(4096));
    // Meta rejects reasoning_effort "none" for Muse models; neither option is sent unless set.
    assert!(body.get("parallel_tool_calls").is_none() && body.get("reasoning_effort").is_none());
    assert_py_eq(
        &calls_json(&proposal.calls),
        &json!([{"name": "get_weather", "arguments": {"city": "Seoul"}}]),
        "calls",
    );
    assert_eq!(proposal.provider, "meta");
    std::env::remove_var("MODEL_API_KEY");
    assert!(Meta::new("muse-spark-1.3").unwrap_err().to_string().contains("MODEL_API_KEY"));
    let options = ProviderOptions {
        api_key: Some("k".into()),
        reasoning_effort: Some("low".into()),
        transport: Some(mock.clone()),
        ..ProviderOptions::default()
    };
    let built = dmcall::provider_from_spec_with("meta:muse-spark-1.3", options).unwrap();
    assert_eq!(built.kind(), dmcall::ProviderKind::Meta);
    assert_eq!(built.payload(&[], &[], &[]).unwrap()["reasoning_effort"], json!("low"));
    // Explicit options are still sent.
    let explicit =
        Meta::builder("muse-spark-1.3").api_key("k").parallel_tool_calls(Some(false)).transport(mock).build().unwrap();
    let tool = json!({"name": "f", "parameters": {"type": "object", "properties": {}}});
    assert_eq!(explicit.payload(&[], &[tool], &[]).unwrap()["parallel_tool_calls"], json!(false));
}
