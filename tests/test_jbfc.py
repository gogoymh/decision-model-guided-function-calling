"""The jbfc library: providers, Jev, tools, the repair loop and parity with the paper's code."""

import asyncio
import enum
import json
from typing import Annotated, Literal, Optional, TypedDict

import httpx
import pytest

import jbfc
from jbfc import providers as providers_module
from jev_fc import llm as paper_llm
from jev_fc.llm import FunctionCallingLLM

KEY_NAMES = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GEMINI_API_KEY",
             "GOOGLE_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY",
             "AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT", "AZURE_OPENAI_API_VERSION", "TYPESAFE_API_KEY",
             "JEV_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_MODEL", "API_GATEWAY", "APIM_SUBSCRIPTION_KEY",
             "APIM_BASE_URL", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_STYLE", "LLM_REASONING_EFFORT",
             "LLM_TEMPERATURE", "LLM_MAX_COMPLETION_TOKENS", "AZURE_MODEL")
SECRET = "sk-test-SECRET-123"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(providers_module.Provider, "_sleep", lambda self, delay: 0.0)


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}
EXAMPLE = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Seoul?"}],
           "tools": [WEATHER]}
HISTORY = [{"calls": [{"name": "get_weather", "arguments": {"city": "Busan"}}],
            "feedback": {"failed_checks": ["arguments"], "instruction": "Reconsider."}},
           {"round": 2, "calls": [], "feedback": {"failed_checks": ["function_selection"]}}]


def client(handler, seen=None):
    def wrapped(request):
        if seen is not None:
            seen.append(request)
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(wrapped))


def chat(calls=(), content=None, finish=None, usage=None, **message):
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}}
                  for i, (name, arguments) in enumerate(calls)]
    body = {"model": "resolved-model", "choices": [{
        "finish_reason": finish or ("tool_calls" if tool_calls else "stop"),
        "message": {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {}),
                    **message}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def jev_answer(function=0.95, arguments=0.95):
    answers = {"function_correct": {"type": "noul", "noul": function}}
    if arguments is not None:
        answers["arguments_correct"] = {"type": "noul", "noul": arguments}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 0}}


class ScriptedModel:
    """A provider stand-in that returns scripted proposals and records the history it saw."""

    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.histories = []

    def generate(self, example, history=None):
        self.histories.append(json.loads(json.dumps(history or [])))
        calls = self.proposals.pop(0)
        incomplete = "length" if calls is None else None
        return {"calls": calls, "content": None if calls else "No tool fits.", "incomplete": incomplete,
                "usage": {"input_tokens": 10, "output_tokens": 2}, "latency_s": 0.01}


class ScriptedJev:
    """A Jev stand-in over mocked HTTP: arguments judged by city, no-call judged by ``no_call``."""

    def __init__(self, good_city="Seoul", function=0.95, no_call=0.05):
        self.inner = jbfc.Jev("jev-key", client=client(self.answer))
        self.good_city, self.function, self.no_call, self.requests = good_city, function, no_call, []

    def answer(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        calls = body["state"]["proposed_calls"]
        if not calls:
            return httpx.Response(200, json=jev_answer(self.no_call, None))
        arguments = 0.95 if calls[0]["arguments"].get("city") == self.good_city else 0.05
        return httpx.Response(200, json=jev_answer(self.function, arguments))

    def verify(self, example, calls):
        return self.inner.verify(example, calls)


def seoul():
    return [{"name": "get_weather", "arguments": {"city": "Seoul"}}]


def busan():
    return [{"name": "get_weather", "arguments": {"city": "Busan"}}]


# Providers -------------------------------------------------------------------------
def test_openai_request_and_decoding():
    seen = []
    model = jbfc.OpenAI("gpt-5-nano", api_key=SECRET, reasoning_effort="none", client=client(
        lambda r: httpx.Response(200, json=chat([("get_weather", {"city": "Seoul"})],
                                                usage={"prompt_tokens": 120, "completion_tokens": 9,
                                                       "prompt_tokens_details": {"cached_tokens": 64}})), seen))
    proposal = model.generate(EXAMPLE, HISTORY)
    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert body["messages"][0] == {"role": "system", "content": jbfc.SYSTEM_PROMPT}
    assert body["messages"][1]["content"].startswith("Accumulated revision history")
    assert body["messages"][2:] == EXAMPLE["messages"]
    assert body["tools"] == [WEATHER] and body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert body["max_completion_tokens"] == 4096 and body["reasoning_effort"] == "none"
    assert proposal["calls"] == seoul() and proposal["incomplete"] is None
    assert proposal["usage"] == {"input_tokens": 120, "cached_input_tokens": 64, "output_tokens": 9}
    assert proposal["model"] == "resolved-model" and proposal["provider"] == "openai"
    assert SECRET not in repr(model) and SECRET not in str(vars(model).get("api_key", ""))


@pytest.mark.parametrize("body, calls, incomplete", [
    (chat(content="No function applies."), [], None),
    (chat(), [], None),
    (chat(finish="length"), None, "length"),
    (chat(finish="content_filter"), None, "content_filter"),
    (chat(refusal="I can't help with that."), None, "refusal"),
    (chat([("get_weather", "{not json")]), [{"name": "get_weather", "arguments": None}], None),
], ids=["text", "empty", "length", "content-filter", "refusal", "bad-json"])
def test_openai_outcomes(body, calls, incomplete):
    model = jbfc.OpenAI("m", api_key="k", client=client(lambda r: httpx.Response(200, json=body)))
    proposal = model.generate(EXAMPLE)
    assert proposal["calls"] == calls and proposal["incomplete"] == incomplete
    if incomplete or calls and calls[0]["arguments"] is None:
        assert proposal["parse_errors"]


def test_filtered_prompt_is_an_unreadable_proposal_not_an_error():
    body = {"error": {"code": "content_filter", "message": "filtered"}}
    model = jbfc.OpenAI("m", api_key="k", client=client(lambda r: httpx.Response(400, json=body)))
    proposal = model.generate(EXAMPLE)
    assert proposal["calls"] is None and proposal["incomplete"] == "prompt_filter" and proposal["usage"] == {}


def test_http_errors_are_sanitized_and_retries_bounded():
    seen = []
    model = jbfc.OpenAI("m", api_key=SECRET, max_retries=2, client=client(
        lambda r: httpx.Response(401, json={"error": {"code": "invalid_api_key", "message": SECRET}},
                                 headers={"x-request-id": "req-1"}), seen))
    with pytest.raises(jbfc.ProviderError) as caught:
        model.generate(EXAMPLE)
    assert len(seen) == 1 and caught.value.status_code == 401 and caught.value.provider_code == "invalid_api_key"
    assert caught.value.request_id == "req-1" and SECRET not in str(caught.value)
    answers = iter([httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(503),
                    httpx.Response(200, json=chat([("get_weather", {"city": "Seoul"})]))])
    retried = jbfc.OpenAI("m", api_key="k", max_retries=2, client=client(lambda r: next(answers)))
    proposal = retried.generate(EXAMPLE)
    assert proposal["api_calls"] == 3 and proposal["calls"] == seoul()
    exhausted = jbfc.OpenAI("m", api_key="k", max_retries=1, client=client(lambda r: httpx.Response(500)))
    with pytest.raises(jbfc.ProviderError, match="HTTP 500"):
        exhausted.generate(EXAMPLE)


def test_keys_come_from_arguments_or_environment(monkeypatch):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        jbfc.OpenAI("m")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1/")
    seen = []
    model = jbfc.OpenAI("m", client=client(lambda r: httpx.Response(200, json=chat()), seen))
    model.generate(EXAMPLE)
    assert seen[0].headers["authorization"] == "Bearer from-env"
    assert str(seen[0].url) == "https://proxy.example.com/v1/chat/completions"
    with pytest.raises(ValueError, match="base_url"):
        jbfc.OpenAI("m", base_url="https://user:pass@example.com/v1")
    with pytest.raises(ValueError, match="single-line"):
        jbfc.OpenAI("m", api_key="a\nb")


@pytest.mark.parametrize("cls, env, url, tokens_field", [
    (jbfc.XAI, "XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "max_tokens"),
    (jbfc.OpenRouter, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", "max_tokens"),
], ids=["xai", "openrouter"])
def test_openai_compatible_hosts(monkeypatch, cls, env, url, tokens_field):
    monkeypatch.setenv(env, "k")
    seen = []
    cls("m", client=client(lambda r: httpx.Response(200, json=chat()), seen)).generate(EXAMPLE)
    assert str(seen[0].url) == url and json.loads(seen[0].content)[tokens_field] == 4096


def test_any_openai_compatible_server_needs_no_key():
    seen = []
    model = jbfc.OpenAICompatible("llama3.1", base_url="http://localhost:11434/v1", parallel_tool_calls=None,
                                  client=client(lambda r: httpx.Response(200, json=chat()), seen))
    model.generate({"messages": [{"role": "user", "content": "hi"}], "tools": []})
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & set(body)
    with pytest.raises(ValueError, match="base_url"):
        jbfc.OpenAICompatible("m")


def test_azure_deployment_url_and_header(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", SECRET)
    seen = []
    jbfc.AzureOpenAI("my deployment", client=client(lambda r: httpx.Response(200, json=chat()), seen)).generate(EXAMPLE)
    assert str(seen[0].url) == ("https://res.openai.azure.com/openai/deployments/my%20deployment/chat/completions"
                                "?api-version=2024-12-01-preview")
    assert seen[0].headers["api-key"] == SECRET and "authorization" not in seen[0].headers
    with pytest.raises(ValueError, match="api_version"):
        jbfc.AzureOpenAI("d", api_version="latest")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT")
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        jbfc.AzureOpenAI("d")


def test_anthropic_request_and_decoding():
    seen = []
    body = {"type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}],
            "usage": {"input_tokens": 50, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 20,
                      "output_tokens": 7}}
    model = jbfc.Anthropic("claude-x", api_key=SECRET, client=client(lambda r: httpx.Response(200, json=body), seen))
    proposal = model.generate(EXAMPLE, HISTORY)
    request = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.anthropic.com/v1/messages"
    assert seen[0].headers["x-api-key"] == SECRET and seen[0].headers["anthropic-version"] == "2023-06-01"
    assert request["system"] == jbfc.SYSTEM_PROMPT + "\n\nBe brief."
    assert [m["role"] for m in request["messages"]] == ["user", "user"]
    assert request["tools"][0]["input_schema"] == WEATHER["function"]["parameters"]
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert proposal["calls"] == seoul() and proposal["content"] == "Checking."
    assert proposal["usage"] == {"input_tokens": 100, "cached_input_tokens": 30, "output_tokens": 7}
    for stop, incomplete in (("max_tokens", "length"), ("refusal", "refusal")):
        cut = {**body, "stop_reason": stop, "content": []}
        model = jbfc.Anthropic("c", api_key="k", client=client(lambda r, cut=cut: httpx.Response(200, json=cut)))
        assert model.generate(EXAMPLE)["incomplete"] == incomplete


def test_gemini_request_and_decoding():
    seen = []
    body = {"modelVersion": "gemini-x-001", "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
        {"text": "thinking", "thought": True}, {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]}}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 6, "thoughtsTokenCount": 40}}
    model = jbfc.Gemini("models/gemini-x", api_key=SECRET, temperature=0, thinking_budget=0,
                        client=client(lambda r: httpx.Response(200, json=body), seen))
    proposal = model.generate({**EXAMPLE, "messages": EXAMPLE["messages"] + [
        {"role": "assistant", "content": "Which unit?"}, {"role": "user", "content": "Celsius."}]})
    request = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
    assert seen[0].headers["x-goog-api-key"] == SECRET
    assert request["systemInstruction"]["parts"][0]["text"] == jbfc.SYSTEM_PROMPT + "\n\nBe brief."
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
], ids=["length", "safety", "malformed", "blocked-prompt"])
def test_gemini_unreadable_outputs(body, incomplete):
    model = jbfc.Gemini("g", api_key="k", client=client(lambda r: httpx.Response(200, json=body)))
    proposal = model.generate(EXAMPLE)
    assert proposal["calls"] is None and proposal["incomplete"] == incomplete


def test_provider_specs(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "k")
    expected = {"openai:gpt-5-nano": jbfc.OpenAI, "anthropic:claude-x": jbfc.Anthropic, "claude:c": jbfc.Anthropic,
                "gemini:g": jbfc.Gemini, "google:g": jbfc.Gemini, "xai:grok-4": jbfc.XAI, "grok:grok-4": jbfc.XAI,
                "openrouter:meta/llama": jbfc.OpenRouter}
    for spec, cls in expected.items():
        built = jbfc.provider(spec)
        assert type(built) is cls and built.model == spec.split(":", 1)[1]
        built.close()
    built = jbfc.provider("azure:dep", endpoint="https://r.openai.azure.com", api_key="k")
    assert isinstance(built, jbfc.AzureOpenAI)
    for bad in ("gpt-5-nano", "openai:", "mystery:model"):
        with pytest.raises(ValueError):
            jbfc.provider(bad)
    stand_in = ScriptedModel()
    assert jbfc.provider(stand_in) is stand_in
    with pytest.raises(ValueError):
        jbfc.provider(stand_in, api_key="k")


# Parity with the experiments -----------------------------------------------------------
def test_prompts_match_the_papers_generator(monkeypatch):
    assert jbfc.SYSTEM_PROMPT is paper_llm.SYSTEM_PROMPT
    paper = FunctionCallingLLM("k", model="gpt-5-nano", gateway="direct", api_style="chat_completions",
                               reasoning_effort="none")
    ours = jbfc.OpenAI("gpt-5-nano", api_key="k", reasoning_effort="none")
    for history in (None, HISTORY):
        assert ours.payload(EXAMPLE["messages"], EXAMPLE["tools"], history) == paper._payload(EXAMPLE, history)
    no_tools = {"messages": EXAMPLE["messages"], "tools": []}
    assert ours.payload(no_tools["messages"], [], HISTORY) == paper._payload(no_tools, HISTORY)
    claude = FunctionCallingLLM("k", model="claude-x", gateway="direct", api_style="anthropic_messages")
    assert (jbfc.Anthropic("claude-x", api_key="k").payload(EXAMPLE["messages"], EXAMPLE["tools"], HISTORY)
            == claude._payload(EXAMPLE, HISTORY))
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "apim-key")
    azure = FunctionCallingLLM(model="gpt-5-nano", gateway="apim", api_style="azure_chat_completions")
    ours = jbfc.AzureOpenAI("gpt-5-nano", endpoint=azure.base_url, api_key="apim-key", reasoning_effort="none")
    assert ours.url == azure.endpoint
    assert ours.payload(EXAMPLE["messages"], EXAMPLE["tools"], HISTORY) == azure._payload(EXAMPLE, HISTORY)
    for paper_model in (paper, claude, azure):
        paper_model.close()


def test_jev_keys_endpoint_and_thresholds(monkeypatch):
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        jbfc.Jev()
    seen = []
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    jev = jbfc.Jev(threshold=0.7, arguments_threshold=0.9,
                   client=client(lambda r: httpx.Response(200, json=jev_answer(0.75, 0.85)), seen))
    check = jev.verify(EXAMPLE, seoul())
    assert str(seen[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}" and SECRET not in repr(jev)
    assert check["function_pass"] is True and check["arguments_pass"] is False
    assert json.loads(seen[0].content)["model"] == "jev-1.13.0"


# The loop ---------------------------------------------------------------------------
def test_jbfc_repairs_with_cumulative_feedback_then_runs():
    model, jev = ScriptedModel(busan(), [], seoul()), ScriptedJev()
    ran = []

    def get_weather(city: str, unit: str = "celsius"):
        ran.append(city)
        return {"city": city, "temp": 21}

    result = jbfc.JBFC(model, jev=jev, tools=[get_weather]).run("Weather in Seoul?")
    assert result.calls == seoul() and result.accepted is True and result.repairs == 2 and not result.exhausted
    assert ran == ["Seoul"] and result.executed and result.output == {"city": "Seoul", "temp": 21}
    assert [len(history) for history in model.histories] == [0, 1, 2]
    assert model.histories[2][0]["calls"] == busan() and model.histories[2][1]["calls"] == []
    assert model.histories[2][0]["feedback"]["failed_checks"] == ["arguments"]
    assert [step["accepted"] for step in result.attempts] == [False, False, True]
    assert result.attempts[1]["arguments_probability"] is None  # no-call skips the argument question
    assert result.usage == {"model": {"input_tokens": 30, "output_tokens": 6},
                            "jev": {"input_tokens": 900, "output_tokens": 0}}
    assert result.probabilities == {"function": 0.95, "arguments": 0.95}
    assert repr(result).startswith('Result(call=get_weather(city="Seoul"), accepted=True, repairs=2')


def test_rejected_calls_do_not_run_unless_asked():
    ran = []

    def get_weather(city: str):
        ran.append(city)

    agent = jbfc.JBFC(ScriptedModel(busan(), busan(), busan()), jev=ScriptedJev(), tools=[get_weather])
    result = agent.run("Weather in Seoul?")
    assert result.accepted is False and result.exhausted and result.repairs == 2 and not result.executed and ran == []
    again = jbfc.JBFC(ScriptedModel(busan(), busan()), jev=ScriptedJev(), tools=[get_weather], max_repairs=1)
    result = again.run("Weather in Seoul?", execute_unverified=True)
    assert result.repairs == 1 and result.executed and ran == ["Busan"]


def test_unreadable_proposal_is_repaired_and_never_run():
    agent = jbfc.JBFC(ScriptedModel(None, seoul()), jev=ScriptedJev(), tools=[WEATHER],
                      functions={"get_weather": lambda city, unit="celsius": city})
    result = agent.run("Weather in Seoul?")
    assert result.attempts[0]["calls"] is None and result.attempts[0]["schema_errors"]
    assert result.accepted and result.output == "Seoul"
    stuck = jbfc.JBFC(ScriptedModel(None), jev=ScriptedJev(), tools=[WEATHER], max_repairs=0,
                      functions={"get_weather": lambda city: city})
    result = stuck.run("Weather?", execute_unverified=True)
    assert result.calls is None and not result.executed and repr(result).startswith("Result(unreadable proposal")


def test_no_call_decision():
    jev = ScriptedJev(no_call=0.97)
    result = jbfc.JBFC(ScriptedModel([]), jev=jev, tools=[WEATHER]).run("Tell me a joke.")
    assert result.no_call and result.accepted and not result.executed and result.text == "No tool fits."
    assert "arguments_correct" not in jev.requests[0]["questions"]


def test_methods(monkeypatch):
    direct = jbfc.JBFC(ScriptedModel(busan()), method="direct", tools=[WEATHER])
    assert direct.jev is None
    result = direct.call("Weather in Seoul?")
    assert result.accepted is None and result.repairs == 0 and result.method == "direct"
    schema = jbfc.JBFC(ScriptedModel([{"name": "get_weather", "arguments": {}}], seoul()), method="schema",
                       tools=[WEATHER]).call("Weather in Seoul?")
    assert schema.repairs == 1 and schema.accepted
    refine = jbfc.JBFC(ScriptedModel(busan(), seoul(), seoul()), method="self_refine", tools=[WEATHER]).call("x")
    assert refine.repairs == 2 and refine.accepted is None
    latest_model = ScriptedModel(busan(), [], seoul())
    jbfc.JBFC(latest_model, jev=ScriptedJev(), method="jbfc-latest", tools=[WEATHER]).call("Weather in Seoul?")
    assert [len(h) for h in latest_model.histories] == [0, 1, 1] and latest_model.histories[2][0]["round"] == 2
    for name, internal in jbfc.METHODS.items():
        assert jbfc.JBFC(ScriptedModel(), jev=ScriptedJev(), method=internal).method == internal
        assert jbfc.JBFC(ScriptedModel(), jev=ScriptedJev(), method=name.upper()).method == internal
    with pytest.raises(ValueError, match="Unknown method"):
        jbfc.JBFC(ScriptedModel(), method="magic")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        jbfc.JBFC(ScriptedModel())
    with pytest.raises(ValueError, match="threshold"):
        jbfc.JBFC(ScriptedModel(), method="direct", threshold=0.9)
    with pytest.raises(ValueError, match="Jev instance"):
        jbfc.JBFC(ScriptedModel(), jev=ScriptedJev(), threshold=0.9)
    with pytest.raises(ValueError, match="max_repairs"):
        jbfc.JBFC(ScriptedModel(), jev=ScriptedJev(), max_repairs=-1)
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-env")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    with jbfc.JBFC("openai:gpt-5-nano", threshold=0.9, reasoning_effort="low") as agent:
        assert isinstance(agent.model, jbfc.OpenAI) and agent.model.reasoning_effort == "low"
        assert agent.jev.function_threshold == agent.jev.arguments_threshold == 0.9


def test_per_call_overrides_and_missing_functions():
    agent = jbfc.JBFC(ScriptedModel(busan(), seoul()), jev=ScriptedJev(), tools=[WEATHER])
    result = agent.call("Weather in Seoul?", max_repairs=0)
    assert result.repairs == 0 and result.accepted is False and result.exhausted
    with pytest.raises(LookupError, match="No Python function"):
        agent.run("Weather in Seoul?", tools=[WEATHER])
    with pytest.raises(ValueError, match="not one of the tools"):
        agent.call("x", functions={"nope": print})


def test_async_wrappers():
    agent = jbfc.JBFC(ScriptedModel(seoul(), seoul()), jev=ScriptedJev(),
                      tools=[WEATHER], functions={"get_weather": lambda city, unit="celsius": city.upper()})

    async def main():
        called = await agent.acall("Weather in Seoul?")
        ran = await agent.arun("Weather in Seoul?")
        verdict = await agent.averify("Weather in Seoul?", seoul())
        return called, ran, verdict

    called, ran, verdict = asyncio.run(main())
    assert called.accepted and not called.executed and ran.output == "SEOUL" and verdict


def test_messages_are_validated():
    model = ScriptedModel(seoul(), seoul())
    agent = jbfc.JBFC(model, method="direct", tools=[WEATHER])
    agent.call([{"role": "user", "content": [{"type": "text", "text": "Weather"}, {"type": "text", "text": "in Seoul?"}]}])
    for bad in ([], [{"role": "tool", "content": "{}"}], [{"role": "user", "content": {"x": 1}}], 3):
        with pytest.raises(ValueError):
            agent.call(bad)


def test_verify_accepts_calls_from_any_sdk():
    jev = ScriptedJev()
    shapes = [
        {"name": "get_weather", "arguments": {"city": "Seoul"}},
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Seoul"}'},
        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}},
        {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}},
        {"function_call": {"name": "get_weather", "args": {"city": "Seoul"}, "id": None}},
    ]
    for shape in shapes:
        verdict = jbfc.verify("Weather in Seoul?", [shape], [WEATHER], jev=jev.inner)
        assert verdict and verdict.calls == seoul() and verdict.arguments_probability == 0.95
    assert not jbfc.verify("Weather in Seoul?", busan(), [WEATHER], jev=jev.inner)
    count = len(jev.requests)
    malformed = jbfc.verify("Weather?", {"name": "get_weather", "arguments": "{oops"}, [WEATHER], jev=jev.inner)
    assert not malformed and len(jev.requests) == count  # rejected locally, Jev is not asked
    assert jbfc.verify("Tell me a joke.", None, [WEATHER], jev=jev.inner).calls == []
    assert jbfc.verify("Tell me a joke.", [], [WEATHER], jev=ScriptedJev(no_call=0.9).inner)
    agent = jbfc.JBFC(ScriptedModel(), jev=jev.inner, tools=[WEATHER])
    assert agent.verify("Weather in Seoul?", seoul()).accepted

    class SDKMessage:  # what an SDK returns, e.g. an assistant turn with a tool call
        def model_dump(self, mode="json"):
            return {"role": "assistant", "content": None, "refusal": None, "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Busan"}'}}]}

    history = [{"role": "user", "content": "Weather in Busan, then Seoul?"}, SDKMessage(),
               {"role": "tool", "tool_call_id": "c0", "content": '{"temp": 18}'}]
    assert jbfc.verify(history, seoul(), [WEATHER], jev=jev.inner)
    sent = jev.requests[-1]["state"]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool"] and "refusal" not in sent[1]
    assert sent[1]["tool_calls"][0]["id"] == "c0" and sent[2]["tool_call_id"] == "c0"
    with pytest.raises(ValueError):
        jbfc.verify([{"role": "robot", "content": "x"}], seoul(), [WEATHER], jev=jev.inner)
    with pytest.raises(ValueError, match="needs Jev"):
        jbfc.JBFC(ScriptedModel(), method="direct").verify("x", seoul())


# Tools ----------------------------------------------------------------------------------
class Unit(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class Address(TypedDict):
    street: str
    zip_code: Annotated[str, "Postal code"]


def book(city: str, guests: int, budget: float, refundable: bool, tags: list[str], extras: dict[str, int],
         room: Literal["single", "double"], unit: Unit = Unit.CELSIUS, note: Optional[str] = None,
         when: tuple[int, int] = (9, 0), address: Address | None = None,
         label: Annotated[str, "A short label"] = "trip", **more) -> dict:
    """Book a stay.

    Longer description.

    Args:
        city: Where to stay.
        guests (int): How many people,
            including children.
        room: Room type.

    Returns:
        A confirmation.
    """
    return {"city": city, "unit": unit}


def test_schema_from_python_function():
    schema = jbfc.schema_of(book)["function"]
    properties = schema["parameters"]["properties"]
    assert schema["name"] == "book" and schema["description"] == "Book a stay.\n\nLonger description."
    assert schema["parameters"]["required"] == ["city", "guests", "budget", "refundable", "tags", "extras", "room"]
    assert schema["parameters"]["additionalProperties"] is True
    assert properties["city"] == {"type": "string", "description": "Where to stay."}
    assert properties["guests"] == {"type": "integer", "description": "How many people, including children."}
    assert properties["budget"] == {"type": "number"} and properties["refundable"] == {"type": "boolean"}
    assert properties["tags"] == {"type": "array", "items": {"type": "string"}}
    assert properties["extras"] == {"type": "object", "additionalProperties": {"type": "integer"}}
    assert properties["room"] == {"type": "string", "enum": ["single", "double"], "description": "Room type."}
    assert properties["unit"] == {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
    assert properties["note"] == {"type": "string", "default": None}
    assert properties["when"] == {"type": "array", "prefixItems": [{"type": "integer"}] * 2, "minItems": 2,
                                  "maxItems": 2, "default": [9, 0]}
    assert properties["address"]["properties"]["zip_code"] == {"type": "string", "description": "Postal code"}
    assert properties["address"]["required"] == ["street", "zip_code"] and properties["address"]["default"] is None
    assert properties["label"] == {"type": "string", "description": "A short label", "default": "trip"}
    arguments = {"city": "Seoul", "guests": 2, "budget": 1.5, "refundable": True, "tags": [], "extras": {},
                 "room": "single", "unit": "fahrenheit"}
    assert jbfc.invoke(book, arguments)["unit"] is Unit.FAHRENHEIT


def test_docstring_styles():
    def numpy_style(city, days=1):
        """Forecast.

        Parameters
        ----------
        city : str
            City name.
        days : int
            Number of days.
        """

    def sphinx_style(city, days=1):
        """Forecast.

        :param city: City name.
        :param int days: Number of
            days.
        :returns: A forecast.
        """

    for function in (numpy_style, sphinx_style):
        properties = jbfc.schema_of(function)["function"]["parameters"]["properties"]
        assert jbfc.schema_of(function)["function"]["description"] == "Forecast."
        assert properties["city"]["description"] == "City name." and properties["days"]["description"] == "Number of days."
        assert properties["days"]["default"] == 1 and "type" not in properties["city"]


def test_tool_decorator_and_definition_formats():
    @jbfc.tool(name="weather", description="Weather now.")
    def get_weather(city: str):
        return city

    assert jbfc.schema_of(get_weather)["function"]["name"] == "weather"
    assert jbfc.schema_of(get_weather)["function"]["description"] == "Weather now."
    assert get_weather("Seoul") == "Seoul"
    custom = jbfc.tool(lambda **kw: kw, name="raw", parameters={"type": "object", "properties": {"q": {}}})
    assert jbfc.schema_of(custom)["function"]["parameters"] == {"type": "object", "properties": {"q": {}}}
    parameters = WEATHER["function"]["parameters"]
    for definition in (WEATHER, WEATHER["function"], {"type": "function", **WEATHER["function"]},
                       {"name": "get_weather", "description": "Current weather for a city.", "input_schema": parameters},
                       {"name": "get_weather", "description": "Current weather for a city.", "parameters": {
                           "type": "OBJECT", "properties": {"city": {"type": "STRING"},
                                                            "unit": {"type": "STRING", "enum": ["celsius", "fahrenheit"]}},
                           "required": ["city"]}}):
        assert jbfc.schema_of(definition) == WEATHER
    schemas, registry = jbfc.toolset([get_weather, WEATHER], functions={"get_weather": print})
    assert [s["function"]["name"] for s in schemas] == ["weather", "get_weather"] and set(registry) == {"weather", "get_weather"}


@pytest.mark.parametrize("bad, error", [
    (lambda city: city, ValueError),
    ("get_weather", TypeError),
    ({"name": "", "parameters": {"type": "object"}}, ValueError),
    ({"name": "x", "parameters": {"type": "array"}}, ValueError),
    ({"name": "x", "parameters": {"type": "object", "properties": {"a": {"type": "nonsense"}}}}, ValueError),
], ids=["unnamed-lambda", "string", "no-name", "not-object", "invalid-schema"])
def test_bad_tools(bad, error):
    with pytest.raises(error):
        jbfc.toolset([bad])


def test_unsupported_signatures():
    def positional(city, /):
        pass

    def odd(value: complex):
        pass

    with pytest.raises(TypeError, match="positional-only"):
        jbfc.schema_of(positional)
    with pytest.raises(TypeError, match="parameters="):
        jbfc.schema_of(odd)
    with pytest.raises(ValueError, match="Two tools"):
        jbfc.toolset([WEATHER, WEATHER])

    async def remote(city: str):
        return city

    with pytest.raises(TypeError, match="Async"):
        jbfc.invoke(remote, {"city": "Seoul"})
