"""Generators over plain HTTP: requests, decoding, unreadable outputs, keys and errors."""

import json

import httpx
import pytest

import dmcall
from conftest import EXAMPLE, HISTORY, SECRET, WEATHER, ScriptedJev, ScriptedModel, chat, client, seoul


def test_openai_request_and_decoding():
    seen = []
    model = dmcall.OpenAI("gpt-5-nano", api_key=SECRET, reasoning_effort="none", client=client(
        lambda r: httpx.Response(200, json=chat([("get_weather", {"city": "Seoul"})],
                                                usage={"prompt_tokens": 120, "completion_tokens": 9,
                                                       "prompt_tokens_details": {"cached_tokens": 64}})), seen))
    proposal = model.generate(EXAMPLE, HISTORY)
    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert body["messages"][0] == {"role": "system", "content": dmcall.SYSTEM_PROMPT}
    assert body["messages"][1]["content"].startswith("Accumulated revision history (oldest first; proposals NOT executed)")
    assert body["messages"][2:] == EXAMPLE["messages"]  # the original request follows, unchanged
    assert body["tools"] == [WEATHER] and body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert body["max_completion_tokens"] == 4096 and body["reasoning_effort"] == "none"
    assert proposal["calls"] == seoul() and proposal["incomplete"] is None
    assert proposal["usage"] == {"input_tokens": 120, "cached_input_tokens": 64, "output_tokens": 9}
    assert proposal["model"] == "resolved-model" and proposal["provider"] == "openai"
    assert SECRET not in repr(model)


def test_history_lists_rounds_oldest_first():
    seen = []
    model = dmcall.OpenAI("m", api_key="k", client=client(lambda r: httpx.Response(200, json=chat()), seen))
    model.generate(EXAMPLE, HISTORY)
    text = json.loads(seen[0].content)["messages"][1]["content"]
    records = json.loads(text.split("\n", 1)[1].rsplit("\nUse these previous judgments", 1)[0])
    assert [record["round"] for record in records] == [1, 2]
    assert records[0]["calls"] == HISTORY[0]["calls"] and records[1]["calls"] == []
    assert text.endswith("when answering the original request below.")


@pytest.mark.parametrize("body, calls, incomplete", [
    (chat(content="No function applies."), [], None),
    (chat(), [], None),
    (chat(finish="length"), None, "length"),
    (chat(finish="content_filter"), None, "content_filter"),
    (chat(refusal="I can't help with that."), None, "refusal"),
    (chat([("get_weather", "{not json")]), [{"name": "get_weather", "arguments": None}], None),
    (chat([("get_weather", "3")]), [{"name": "get_weather", "arguments": None}], None),
], ids=["text", "empty", "length", "content-filter", "refusal", "bad-json", "not-an-object"])
def test_openai_outcomes(body, calls, incomplete):
    model = dmcall.OpenAI("m", api_key="k", client=client(lambda r: httpx.Response(200, json=body)))
    proposal = model.generate(EXAMPLE)
    assert proposal["calls"] == calls and proposal["incomplete"] == incomplete
    if incomplete or calls and calls[0]["arguments"] is None:
        assert proposal["parse_errors"]


@pytest.mark.parametrize("call, calls", [
    ({"tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Seoul"}}}]},
     seoul()),
    ({"tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": None}}]},
     [{"name": "get_weather", "arguments": {}}]),
    ({"tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": 3}}]},
     [{"name": "get_weather", "arguments": None}]),
    ({"function_call": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}}, seoul()),
], ids=["object-arguments", "null-arguments", "number-arguments", "legacy-function-call"])
def test_openai_compatible_servers_that_bend_the_protocol(call, calls):
    """Arguments sent as an object or null, and the older function_call field, are still a call."""
    body = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, **call}}]}
    model = dmcall.OpenAICompatible("m", base_url="http://localhost:8000/v1",
                                    client=client(lambda r: httpx.Response(200, json=body)))
    proposal = model.generate(EXAMPLE)
    assert proposal["calls"] == calls and proposal["incomplete"] is None
    assert bool(proposal["parse_errors"]) == (calls[0]["arguments"] is None)


def test_redirects_are_not_followed():
    seen = []
    model = dmcall.Anthropic("m", api_key=SECRET, client=client(
        lambda r: httpx.Response(307, headers={"location": "https://elsewhere.example/v1/messages"}), seen))
    with pytest.raises(dmcall.ProviderError, match="HTTP 307"):
        model.generate(EXAMPLE)
    assert len(seen) == 1 and seen[0].url.host == "api.anthropic.com"  # the key never leaves for another host


def test_malformed_gemini_native_call_is_repaired_instead_of_preserved_as_no_call():
    responses = iter([
        {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"functionCall": "invalid"}]}}]},
        {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]}}]},
    ])
    seen = []
    model = dmcall.Gemini("m", api_key="KEY", client=client(lambda r: httpx.Response(200, json=next(responses)), seen))
    decision = ScriptedJev()
    agent = dmcall.DMCall(model, tools=[WEATHER], decision_model=decision.jev)
    result = agent.call("Weather in Seoul?")
    assert result.repairs == 1 and result.accepted and not result.initial_no_call_preserved
    assert result.attempts[0]["calls"] is None and result.calls == seoul()
    assert len(seen) == 2 and len(decision.requests) == 1


def test_filtered_prompt_is_an_unreadable_candidate_not_an_error():
    body = {"error": {"code": "content_filter", "message": "filtered"}}
    model = dmcall.OpenAI("m", api_key="k", client=client(lambda r: httpx.Response(400, json=body)))
    proposal = model.generate(EXAMPLE)
    assert proposal["calls"] is None and proposal["incomplete"] == "prompt_filter" and proposal["usage"] == {}


def test_http_errors_are_sanitized_and_retries_bounded():
    seen = []
    model = dmcall.OpenAI("m", api_key=SECRET, max_retries=2, client=client(
        lambda r: httpx.Response(401, json={"error": {"code": "invalid_api_key", "message": SECRET}},
                                 headers={"x-request-id": "req-1"}), seen))
    with pytest.raises(dmcall.ProviderError) as caught:
        model.generate(EXAMPLE)
    assert len(seen) == 1 and caught.value.status_code == 401 and caught.value.provider_code == "invalid_api_key"
    assert caught.value.request_id == "req-1" and SECRET not in str(caught.value)
    answers = iter([httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(503),
                    httpx.Response(200, json=chat([("get_weather", {"city": "Seoul"})]))])
    retried = dmcall.OpenAI("m", api_key="k", max_retries=2, client=client(lambda r: next(answers)))
    proposal = retried.generate(EXAMPLE)
    assert proposal["api_calls"] == 3 and proposal["calls"] == seoul()
    exhausted = dmcall.OpenAI("m", api_key="k", max_retries=1, client=client(lambda r: httpx.Response(500)))
    with pytest.raises(dmcall.ProviderError, match="HTTP 500"):
        exhausted.generate(EXAMPLE)


def test_keys_come_from_arguments_or_environment(monkeypatch):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        dmcall.OpenAI("m")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1/")
    seen = []
    model = dmcall.OpenAI("m", client=client(lambda r: httpx.Response(200, json=chat()), seen))
    model.generate(EXAMPLE)
    assert seen[0].headers["authorization"] == "Bearer from-env"
    assert str(seen[0].url) == "https://proxy.example.com/v1/chat/completions"
    with pytest.raises(ValueError, match="base_url"):
        dmcall.OpenAI("m", base_url="https://user:pass@example.com/v1")
    with pytest.raises(ValueError, match="single-line"):
        dmcall.OpenAI("m", api_key="a\nb")


@pytest.mark.parametrize("cls, env, url, tokens_field", [
    (dmcall.XAI, "XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "max_tokens"),
    (dmcall.OpenRouter, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", "max_tokens"),
], ids=["xai", "openrouter"])
def test_openai_compatible_hosts(monkeypatch, cls, env, url, tokens_field):
    monkeypatch.setenv(env, "k")
    seen = []
    cls("m", client=client(lambda r: httpx.Response(200, json=chat()), seen)).generate(EXAMPLE)
    assert str(seen[0].url) == url and json.loads(seen[0].content)[tokens_field] == 4096


def test_meta_muse_request(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", SECRET)
    seen = []
    model = dmcall.Meta("muse-spark-1.3", client=client(lambda r: httpx.Response(200, json=chat(
        [("get_weather", {"city": "Seoul"})])), seen))
    proposal = model.generate(EXAMPLE, HISTORY)
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.meta.ai/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"
    assert body["model"] == "muse-spark-1.3" and body["tool_choice"] == "auto" and body["max_completion_tokens"] == 4096
    # Meta rejects reasoning_effort "none" for Muse models; neither option is sent unless set.
    assert "parallel_tool_calls" not in body and "reasoning_effort" not in body
    assert proposal["calls"] == seoul() and proposal["provider"] == "meta"
    monkeypatch.delenv("MODEL_API_KEY")
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        dmcall.Meta("muse-spark-1.3")
    built = dmcall.provider("meta:muse-spark-1.3", api_key="k", reasoning_effort="low")
    assert isinstance(built, dmcall.Meta) and built.payload([], [], None)["reasoning_effort"] == "low"


def test_any_openai_compatible_server_needs_no_key():
    seen = []
    model = dmcall.OpenAICompatible("gemma-4-31b-it", base_url="http://localhost:8000/v1",
                                    client=client(lambda r: httpx.Response(200, json=chat()), seen))
    model.generate({"messages": [{"role": "user", "content": "hi"}], "tools": []})
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://localhost:8000/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & set(body)  # no functions, no tool settings
    with pytest.raises(ValueError, match="base_url"):
        dmcall.OpenAICompatible("m")


def test_azure_deployment_url_and_header(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", SECRET)
    seen = []
    dmcall.AzureOpenAI("my deployment", client=client(lambda r: httpx.Response(200, json=chat()), seen)).generate(EXAMPLE)
    assert str(seen[0].url) == ("https://res.openai.azure.com/openai/deployments/my%20deployment/chat/completions"
                                "?api-version=2024-12-01-preview")
    assert seen[0].headers["api-key"] == SECRET and "authorization" not in seen[0].headers
    with pytest.raises(ValueError, match="api_version"):
        dmcall.AzureOpenAI("d", api_version="latest")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT")
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        dmcall.AzureOpenAI("d")


def test_anthropic_request_and_decoding():
    seen = []
    body = {"type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}],
            "usage": {"input_tokens": 50, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 20,
                      "output_tokens": 7}}
    model = dmcall.Anthropic("claude-x", api_key=SECRET, client=client(lambda r: httpx.Response(200, json=body), seen))
    proposal = model.generate(EXAMPLE, HISTORY)
    request = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.anthropic.com/v1/messages"
    assert seen[0].headers["x-api-key"] == SECRET and seen[0].headers["anthropic-version"] == "2023-06-01"
    assert request["system"] == dmcall.SYSTEM_PROMPT + "\n\nBe brief."
    assert [m["role"] for m in request["messages"]] == ["user", "user"]
    assert request["tools"][0]["input_schema"] == WEATHER["function"]["parameters"]
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert proposal["calls"] == seoul() and proposal["content"] == "Checking."
    assert proposal["usage"] == {"input_tokens": 100, "cached_input_tokens": 30, "output_tokens": 7}
    for stop, incomplete in (("max_tokens", "length"), ("refusal", "refusal")):
        cut = {**body, "stop_reason": stop, "content": []}
        model = dmcall.Anthropic("c", api_key="k", client=client(lambda r, cut=cut: httpx.Response(200, json=cut)))
        assert model.generate(EXAMPLE)["incomplete"] == incomplete


def test_gemini_request_and_decoding():
    seen = []
    body = {"modelVersion": "gemini-x-001", "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
        {"text": "thinking", "thought": True}, {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]}}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 6, "thoughtsTokenCount": 40}}
    model = dmcall.Gemini("models/gemini-x", api_key=SECRET, temperature=0, thinking_budget=0,
                          client=client(lambda r: httpx.Response(200, json=body), seen))
    proposal = model.generate({**EXAMPLE, "messages": EXAMPLE["messages"] + [
        {"role": "assistant", "content": "Which unit?"}, {"role": "user", "content": "Celsius."}]})
    request = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
    assert seen[0].headers["x-goog-api-key"] == SECRET
    assert request["systemInstruction"]["parts"][0]["text"] == dmcall.SYSTEM_PROMPT + "\n\nBe brief."
    assert [c["role"] for c in request["contents"]] == ["user", "model", "user"]
    declaration = request["tools"][0]["functionDeclarations"][0]
    assert declaration["parametersJsonSchema"] == WEATHER["function"]["parameters"] and "parameters" not in declaration
    assert request["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    assert request["generationConfig"] == {"maxOutputTokens": 4096, "temperature": 0,
                                           "thinkingConfig": {"thinkingBudget": 0}}
    assert proposal["calls"] == seoul() and proposal["content"] is None and proposal["model"] == "gemini-x-001"
    assert proposal["usage"] == {"input_tokens": 80, "output_tokens": 46}


@pytest.mark.parametrize("body, incomplete", [
    ({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}, "length"),
    ({"candidates": [{"finishReason": "SAFETY"}]}, "content_filter"),
    ({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]}, "malformed_call"),
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "prompt_filter"),
    ({"candidates": [{"finishReason": "OTHER"}]}, "abnormal_stop"),
    ({"candidates": [{"finishReason": "MALFORMED_RESPONSE", "content": {"parts": [{"text": "x"}]}}]}, "abnormal_stop"),
    ({"candidates": [{"finishReason": "TOO_MANY_TOOL_CALLS"}]}, "abnormal_stop"),
    ({"candidates": [{"finishReason": "A_REASON_ADDED_LATER"}]}, "abnormal_stop"),
    ({"candidates": [{"finishReason": "FINISH_REASON_UNSPECIFIED"}]}, "abnormal_stop"),
    ({"candidates": [{}]}, "abnormal_stop"),
], ids=["length", "safety", "malformed", "blocked-prompt", "other", "malformed-response", "too-many-calls",
        "unknown-reason", "unspecified-without-content", "empty-candidate"])
def test_gemini_unreadable_outputs(body, incomplete):
    model = dmcall.Gemini("g", api_key="k", client=client(lambda r: httpx.Response(200, json=body)))
    proposal = model.generate(EXAMPLE)
    assert proposal["calls"] is None and proposal["incomplete"] == incomplete and proposal["parse_errors"]


def test_gemini_finished_answers_without_a_call_are_no_calls():
    call = {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}
    for body, calls in (({"candidates": [{"finishReason": "STOP"}]}, []),
                        ({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "No."}]}}]}, []),
                        ({"candidates": [{"content": {"parts": [call]}}]}, seoul())):
        model = dmcall.Gemini("g", api_key="k", client=client(lambda r, body=body: httpx.Response(200, json=body)))
        proposal = model.generate(EXAMPLE)
        assert proposal["calls"] == calls and proposal["incomplete"] is None


def test_provider_specs(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY",
                 "MODEL_API_KEY"):
        monkeypatch.setenv(name, "k")
    expected = {"openai:gpt-5-nano": dmcall.OpenAI, "anthropic:claude-x": dmcall.Anthropic, "claude:c": dmcall.Anthropic,
                "gemini:g": dmcall.Gemini, "google:g": dmcall.Gemini, "xai:grok-4": dmcall.XAI, "grok:grok-4": dmcall.XAI,
                "openrouter:meta/llama": dmcall.OpenRouter, "meta:muse-spark-1.3": dmcall.Meta}
    for spec, cls in expected.items():
        built = dmcall.provider(spec)
        assert type(built) is cls and built.model == spec.split(":", 1)[1]
        built.close()
    built = dmcall.provider("azure:dep", endpoint="https://r.openai.azure.com", api_key="k")
    assert isinstance(built, dmcall.AzureOpenAI)
    for bad in ("gpt-5-nano", "openai:", "mystery:model"):
        with pytest.raises(ValueError):
            dmcall.provider(bad)
    stand_in = ScriptedModel()
    assert dmcall.provider(stand_in) is stand_in
    with pytest.raises(ValueError):
        dmcall.provider(stand_in, api_key="k")
    with pytest.raises(TypeError):
        dmcall.provider(42)
