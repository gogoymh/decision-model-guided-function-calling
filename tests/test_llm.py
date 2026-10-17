import json
from copy import deepcopy

import httpx
import pytest

from jev_fc.llm import FunctionCallingLLM, LLMAPIError


@pytest.fixture(autouse=True)
def isolated_api_environment(monkeypatch):
    for name in ("API_GATEWAY", "LLM_API_STYLE", "LLM_MODEL", "LLM_REASONING_EFFORT", "LLM_TEMPERATURE",
                 "LLM_BASE_URL", "LLM_MAX_COMPLETION_TOKENS", "APIM_BASE_URL", "APIM_SUBSCRIPTION_KEY",
                 "LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def example():
    return {
        "messages": [{"role": "user", "content": "Weather in Seoul", "private_label": "ORACLE"}],
        "tools": [{"type": "function", "function": {"name": "weather", "description": "Current weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "ground_truth": "ORACLE", "name_map": "ORACLE",
    }


def response(arguments='{"city":"Seoul"}'):
    return {"model": "pinned-test-model", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": arguments}}],
    }}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}}}


def test_native_calls_and_feedback_without_fake_execution_or_oracles(example):
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert "ORACLE" not in request.content.decode()
        return httpx.Response(200, json=response())

    history = [{"calls": [{"name": "weather", "arguments": {"city": "Paris"}}],
                "feedback": {"arguments_probability": .1, "instruction": "Recheck the city"}}]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = FunctionCallingLLM("test-key", model="pinned-test-model", client=client).generate(example, history)
    assert result["calls"] == [{"name": "weather", "arguments": {"city": "Seoul"}}]
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 40
    assert observed[0]["tool_choice"] == "auto"
    assert observed[0]["parallel_tool_calls"] is False
    assert all(message["role"] != "tool" for message in observed[0]["messages"])
    assert "NOT executed" in observed[0]["messages"][-2]["content"]
    assert observed[0]["messages"][-1]["content"] == example["messages"][0]["content"]
    assert result["api_calls"] == 1


def test_malformed_argument_json_is_repairable_proposal(example):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response("{broken")))) as client:
        result = FunctionCallingLLM("test-key", client=client).generate(example)
    assert result["calls"][0]["arguments"] is None
    assert result["parse_errors"] == ["Function arguments are not valid JSON."]


def test_valid_no_call_response(example):
    data = response()
    data["choices"][0] = {"finish_reason": "stop", "message": {"role": "assistant", "content": "No matching tool."}}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))) as client:
        result = FunctionCallingLLM("test-key", client=client).generate(example)
    assert result["calls"] == []


@pytest.mark.parametrize("body", [None, {}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": None}]},
                                 {"choices": [{"message": {"tool_calls": "wrong"}}]}])
def test_malformed_protocol_is_sanitized_error(example, body):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        with pytest.raises(LLMAPIError):
            FunctionCallingLLM("test-key", client=client).generate(example)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_incomplete_generation_not_treated_as_complete_proposal(example, finish_reason):
    data = response()
    data["choices"][0]["finish_reason"] = finish_reason
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))) as client:
        with pytest.raises(LLMAPIError, match="truncated or filtered"):
            FunctionCallingLLM("test-key", client=client).generate(example)


def test_rate_limit_retries_and_preserves_attempt_count(example, monkeypatch):
    observed, delays = [], []
    monkeypatch.setattr("jev_fc.llm.time.sleep", delays.append)

    def handler(request):
        observed.append(request)
        return httpx.Response(429, headers={"Retry-After": "0.1"}) if len(observed) == 1 else httpx.Response(200, json=response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = FunctionCallingLLM("test-key", client=client).generate(example)
    assert len(observed) == result["api_calls"] == 2
    assert delays == [.1]


def test_auth_error_does_not_disclose_secret_or_response(example):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401, text="test-key SECRET"))) as client:
        with pytest.raises(LLMAPIError, match="HTTP 401") as error:
            FunctionCallingLLM("test-key", client=client).generate(example)
    assert "test-key" not in str(error.value)
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("retries", [-1, True, 1.5])
def test_invalid_http_retry_budget(retries):
    with pytest.raises(ValueError, match="max_retries"):
        FunctionCallingLLM("test-key", max_retries=retries)


def test_missing_usage_remains_unknown_for_accounting(example):
    data = response()
    data.pop("usage")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))) as client:
        result = FunctionCallingLLM("test-key", client=client).generate(example)
    assert "prompt_tokens" not in result["usage"]


def responses_body(arguments='{"city":"Seoul"}'):
    return {"model": "pinned-test-model", "status": "completed", "error": None, "incomplete_details": None,
            "output": [{"id": "fc_1", "call_id": "call_1", "type": "function_call", "status": "completed",
                        "name": "weather", "arguments": arguments}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 40}}}


def anthropic_body():
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test-model",
            "stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Seoul"}}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 40,
                      "cache_creation_input_tokens": 10,
                      "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 0}}}


def generate_body(example, style, body):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        return FunctionCallingLLM("test-key", model="test-model", client=client, gateway="direct", api_style=style).generate(example)


def test_apim_defaults_to_responses_and_never_sends_provider_authorization(example, monkeypatch):
    monkeypatch.setenv("API_GATEWAY", "apim")
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "subscription-fixture-key")
    monkeypatch.setenv("LLM_API_KEY", "unused-provider-key")
    observed = []
    original = deepcopy(example)
    example["tools"][0]["function"]["parameters"]["required"] = ["city"]
    example["tools"][0]["function"]["parameters"]["properties"]["unit"] = {"type": "string", "default": "celsius"}
    example["tools"][0]["function"]["private_label"] = "ORACLE"

    def handler(request):
        assert str(request.url) == "https://enhans-aoai-prod-apim.azure-api.net/llm/openai/v1/responses"
        assert request.headers["api-key"] == "subscription-fixture-key"
        assert "authorization" not in request.headers and "x-api-key" not in request.headers
        assert "ORACLE" not in request.content.decode()
        observed.append(json.loads(request.content))
        return httpx.Response(200, json=responses_body())

    history = [{"calls": [{"name": "weather", "arguments": {"city": "Paris"}}], "feedback": {"arguments_pass": False}}]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = FunctionCallingLLM(client=client, max_completion_tokens=123, timeout=35)
        result = llm.generate(example, history)
    assert (llm.gateway, llm.api_style, llm.provider, llm.auth_header) == ("apim", "responses", "openai", "api-key")
    assert llm.timeout == 35 and llm.endpoint.endswith("/responses")
    payload = observed[0]
    assert payload["store"] is False and payload["max_output_tokens"] == 123
    assert payload["reasoning"] == {"effort": "none"} and "reasoning_effort" not in payload
    assert payload["tools"][0]["strict"] is False and "function" not in payload["tools"][0]
    assert payload["tools"][0]["parameters"]["required"] == ["city"]
    assert "additionalProperties" not in payload["tools"][0]["parameters"]
    assert "unit" not in payload["tools"][0]["parameters"]["required"]
    assert "NOT executed" in payload["input"][0]["content"]
    assert payload["input"][-1]["content"] == example["messages"][0]["content"]
    assert payload["instructions"] and all(item["role"] != "tool" for item in payload["input"])
    assert result["calls"] == [{"name": "weather", "arguments": {"city": "Seoul"}}]
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}}
    assert result["raw_usage"] == responses_body()["usage"]
    assert original["messages"] == example["messages"]


def test_direct_responses_uses_bearer_and_responses_path(example):
    def handler(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        assert "api-key" not in request.headers
        return httpx.Response(200, json=responses_body())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        FunctionCallingLLM("test-key", api_style="responses", client=client).generate(example)


def test_apim_chat_completions_is_rejected_before_any_request():
    with pytest.raises(ValueError, match="no documented Chat Completions route"):
        FunctionCallingLLM("test-key", gateway="apim", api_style="chat_completions")


@pytest.mark.parametrize("status", [None, "queued", "in_progress", "failed", "incomplete", "cancelled"])
def test_responses_requires_completed_status(example, status):
    data = responses_body()
    data["status"] = status
    with pytest.raises(LLMAPIError, match="incomplete, failed, or not terminal"):
        generate_body(example, "responses", data)


@pytest.mark.parametrize("output", [
    [], [{"type": "reasoning", "summary": []}], [None],
    [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "SECRET"}]}],
    [{"type": "message", "role": "assistant", "content": []}],
    [{"type": "function_call", "name": "weather"}],
    [{"type": "function_call", "name": "", "arguments": "{}"}],
    [{"type": "function_call", "name": "weather", "arguments": {}, "status": "completed"}],
    [{"type": "function_call", "name": "weather", "arguments": "{}", "status": "incomplete"}],
    [{"type": "web_search_call", "status": "completed"}],
])
def test_responses_refusal_malformed_or_nonproposal_output_is_not_no_call(example, output):
    data = responses_body()
    data["output"] = output
    with pytest.raises(LLMAPIError) as caught:
        generate_body(example, "responses", data)
    assert "SECRET" not in str(caught.value)


def test_responses_completed_text_is_valid_no_call(example):
    data = responses_body()
    data["output"] = [{"type": "reasoning", "summary": []}, {"type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "No available function applies."}]}]
    result = generate_body(example, "responses", data)
    assert result["calls"] == [] and result["content"] == "No available function applies."


@pytest.mark.parametrize("arguments", ["{broken", "null", "[]", '{"city":NaN}'])
def test_responses_bad_arguments_are_retained_for_repair(example, arguments):
    result = generate_body(example, "responses", responses_body(arguments))
    assert result["calls"] == [{"name": "weather", "arguments": None}]
    assert len(result["parse_errors"]) == 1


@pytest.mark.parametrize("gateway", ["direct", "apim"])
def test_anthropic_auth_message_tools_optional_schema_and_usage(example, gateway):
    seen = []
    example = deepcopy(example)
    example["messages"].insert(0, {"role": "system", "content": "Use metric defaults.", "private_label": "ORACLE"})
    example["tools"][0]["function"]["parameters"]["properties"]["unit"] = {"type": "string"}
    example["tools"][0]["function"]["parameters"]["required"] = ["city"]

    def handler(request):
        if gateway == "apim":
            assert str(request.url) == "https://enhans-aoai-prod-apim.azure-api.net/llm/anthropic/v1/messages"
            assert request.headers["api-key"] == "test-key"
            for header in ("authorization", "x-api-key", "anthropic-version"):
                assert header not in request.headers
        else:
            assert str(request.url) == "https://api.anthropic.com/v1/messages"
            assert request.headers["x-api-key"] == "test-key"
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert "authorization" not in request.headers and "api-key" not in request.headers
        assert "ORACLE" not in request.content.decode()
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=anthropic_body())

    history = [{"calls": [], "feedback": {"function_pass": False}}]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = FunctionCallingLLM("test-key", model="claude-test-model", api_style="anthropic_messages", gateway=gateway, client=client, max_completion_tokens=512)
        result = llm.generate(example, history)
    payload = seen[0]
    assert payload["max_tokens"] == 512 and "Use metric defaults." in payload["system"]
    assert "NOT executed" in payload["messages"][0]["content"]
    assert payload["messages"][-1]["content"] == example["messages"][-1]["content"]
    assert all(message["role"] in {"user", "assistant"} for message in payload["messages"])
    assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert payload["tools"][0]["input_schema"]["required"] == ["city"]
    assert "parameters" not in payload["tools"][0] and "function" not in payload["tools"][0]
    assert not {"reasoning", "reasoning_effort", "temperature", "max_output_tokens"}.intersection(payload)
    assert llm.reasoning_effort is None and llm.provider == "anthropic"
    assert result["usage"]["prompt_tokens"] == 150
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 40
    assert result["usage"]["cache_creation_input_tokens"] == 10
    assert result["usage"]["cache_creation"] == {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 0}
    assert result["raw_usage"] == anthropic_body()["usage"]


def test_anthropic_requires_explicit_model_and_rejects_openai_reasoning():
    with pytest.raises(ValueError, match="explicit Anthropic"):
        FunctionCallingLLM("test-key", api_style="anthropic_messages")
    with pytest.raises(ValueError, match="reasoning_effort"):
        FunctionCallingLLM("test-key", model="claude-test-model", api_style="anthropic_messages", reasoning_effort="high")


@pytest.mark.parametrize("reason", [None, "max_tokens", "refusal", "pause_turn", "model_context_window_exceeded", "unknown"])
def test_anthropic_nonterminal_or_refused_output_is_not_no_call(example, reason):
    data = anthropic_body()
    data["stop_reason"] = reason
    with pytest.raises(LLMAPIError, match="incomplete, refused, or not terminal"):
        generate_body(example, "anthropic_messages", data)


def test_anthropic_stop_details_refusal_not_mistaken_for_end_turn(example):
    data = anthropic_body()
    data.update(stop_reason="end_turn", content=[{"type": "text", "text": "Refused"}],
                stop_details={"type": "refusal", "explanation": "SECRET"})
    with pytest.raises(LLMAPIError) as caught:
        generate_body(example, "anthropic_messages", data)
    assert "SECRET" not in str(caught.value)


def test_anthropic_completed_text_is_valid_no_call(example):
    data = anthropic_body()
    data.update(stop_reason="end_turn", content=[{"type": "text", "text": "No available function applies."}])
    assert generate_body(example, "anthropic_messages", data)["calls"] == []


@pytest.mark.parametrize("content", [[], [None], [{"type": "thinking", "thinking": "internal"}],
                                    [{"type": "tool_use", "name": "weather"}],
                                    [{"type": "server_tool_use", "name": "web_search", "input": {}}]])
def test_anthropic_malformed_or_unsupported_output_fails(example, content):
    data = anthropic_body()
    data["content"] = content
    with pytest.raises(LLMAPIError):
        generate_body(example, "anthropic_messages", data)


def test_anthropic_nonobject_arguments_remain_repairable_call(example):
    data = anthropic_body()
    data["content"][0]["input"] = "bad-object"
    result = generate_body(example, "anthropic_messages", data)
    assert result["calls"] == [{"name": "weather", "arguments": None}] and result["parse_errors"]


def test_anthropic_optional_null_cache_creation_details_do_not_invalidate_call(example):
    data = anthropic_body()
    data["usage"].update(cache_creation_input_tokens=0, cache_creation=None)
    result = generate_body(example, "anthropic_messages", data)
    assert result["usage"]["prompt_tokens"] == 140
    assert result["raw_usage"]["cache_creation"] is None
    assert result["calls"] == [{"name": "weather", "arguments": {"city": "Seoul"}}]


def test_responses_cached_input_cannot_exceed_total_input(example):
    data = responses_body()
    data["usage"]["input_tokens_details"]["cached_tokens"] = 101
    with pytest.raises(LLMAPIError, match="exceeded total"):
        generate_body(example, "responses", data)


@pytest.mark.parametrize("style,body_factory", [("chat_completions", response), ("responses", responses_body), ("anthropic_messages", anthropic_body)])
def test_missing_observed_model_and_usage_are_not_invented(example, style, body_factory):
    data = body_factory()
    data.pop("usage")
    data.pop("model")
    result = generate_body(example, style, data)
    assert result["model"] is None and result["requested_model"] == "test-model"
    assert result["usage"] == {} and result["raw_usage"] is None


@pytest.mark.parametrize("style,body_factory,key", [("chat_completions", response, "prompt_tokens"), ("responses", responses_body, "input_tokens"), ("anthropic_messages", anthropic_body, "cache_creation_input_tokens")])
@pytest.mark.parametrize("bad_count", [-1, True, 1.5])
def test_provider_usage_counts_must_be_nonnegative_integers(example, style, body_factory, key, bad_count):
    data = body_factory()
    data["usage"][key] = bad_count
    with pytest.raises(LLMAPIError, match="token usage"):
        generate_body(example, style, data)


@pytest.mark.parametrize("style,body_factory", [("responses", responses_body), ("anthropic_messages", anthropic_body)])
def test_provider_error_envelopes_never_become_no_call(example, style, body_factory):
    data = body_factory()
    data["error"] = {"message": "SECRET fixture failure"}
    with pytest.raises(LLMAPIError) as caught:
        generate_body(example, style, data)
    assert "SECRET" not in str(caught.value)


def test_apim_failure_exposes_only_safe_status_request_id_and_attempts(example):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(403, headers={"apim-request-id": "req-123:abc"}, text="SECRET"))) as client:
        with pytest.raises(LLMAPIError) as caught:
            FunctionCallingLLM("subscription-fixture-key", gateway="apim", client=client).generate(example)
    assert caught.value.status_code == 403 and caught.value.request_id == "req-123:abc" and caught.value.attempts == 1
    assert "SECRET" not in str(caught.value) and "subscription-fixture-key" not in str(caught.value)


@pytest.mark.parametrize("request_id", ["subscription-fixture-key", "prefix-subscription-fixture-key-suffix", "unsafe id!", "r" * 129])
def test_apim_request_id_cannot_echo_credentials_or_arbitrary_text(example, request_id):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401, headers={"apim-request-id": request_id}))) as client:
        with pytest.raises(LLMAPIError) as caught:
            FunctionCallingLLM("subscription-fixture-key", gateway="apim", client=client).generate(example)
    assert caught.value.request_id is None


def test_chat_refusal_and_malformed_empty_tool_calls_are_not_valid_no_call(example):
    for update in ({"refusal": "SECRET"}, {"tool_calls": {}}):
        data = response()
        data["choices"][0] = {"finish_reason": "stop", "message": {"role": "assistant", "content": "No call", **update}}
        with pytest.raises(LLMAPIError):
            generate_body(example, "chat_completions", data)


def test_responses_http_retry_is_bounded_and_reports_final_attempts(example, monkeypatch):
    attempts, delays = [], []
    monkeypatch.setattr("jev_fc.llm.time.sleep", delays.append)

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, headers={"retry-after": "1000", "x-request-id": "req-final"}, text="SECRET")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMAPIError) as caught:
            FunctionCallingLLM("test-key", api_style="responses", client=client, max_retries=1).generate(example)
    assert len(attempts) == caught.value.attempts == 2 and delays == [30]
    assert caught.value.status_code == 503 and caught.value.request_id == "req-final"


def test_responses_transport_errors_are_bounded_without_disclosing_exception_text(example, monkeypatch):
    attempts, delays = [], []
    monkeypatch.setattr("jev_fc.llm.time.sleep", delays.append)

    def handler(request):
        attempts.append(request)
        raise httpx.ConnectError("SECRET fixture credentials", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMAPIError) as caught:
            FunctionCallingLLM("test-key", api_style="responses", client=client, max_retries=1).generate(example)
    assert caught.value.attempts == len(attempts) == 2 and delays == [1]
    assert caught.value.status_code is None and caught.value.request_id is None
    assert "SECRET" not in str(caught.value)


@pytest.mark.parametrize("style", ["chat_completions", "responses", "anthropic_messages"])
def test_two_repairs_prepend_all_verdicts_before_unchanged_request_and_allow_function_change(example, style):
    from jev_fc.jev import JevVerifier
    from jev_fc.pipeline import run_example

    original_request = "Weather in Seoul\nUse Celsius, please. "
    example["messages"][0]["content"] = original_request
    other = deepcopy(example["tools"][0])
    other["function"]["name"] = "other"
    example["tools"].append(other)
    before = deepcopy(example)
    planned = [("other", "Paris"), ("weather", "Paris"), ("weather", "Seoul")]
    requests, checks, executions = [], [], []

    def generator_http(request):
        assert not executions, "A repair proposal must not run a function"
        assert "ORACLE" not in request.content.decode()
        payload = json.loads(request.content)
        requests.append(payload)
        name, city = planned[len(requests) - 1]
        if style == "chat_completions":
            body = response(json.dumps({"city": city}))
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = name
        elif style == "responses":
            body = responses_body(json.dumps({"city": city}))
            body["output"][0]["name"] = name
        else:
            body = anthropic_body()
            body["content"][0].update(name=name, input={"city": city})
        return httpx.Response(200, json=body)

    def verifier_http(request):
        assert not executions
        assert "ORACLE" not in request.content.decode()
        body = json.loads(request.content)
        assert set(body["questions"]) == {"function_correct", "arguments_correct"}
        assert body["state"]["messages"] == [{"role": "user", "content": original_request}]
        assert [tool["name"] for tool in body["state"]["tools"]] == ["weather", "other"]
        call, = body["state"]["proposed_calls"]
        checks.append(call)
        function_p = .95 if call["name"] == "weather" else .1
        arguments_p = .95 if call["name"] == "other" or call["arguments"]["city"] == "Seoul" else .1
        return httpx.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 50, "output_tokens": 0},
            "answers": {"function_correct": {"type": "noul", "noul": function_p},
                        "arguments_correct": {"type": "noul", "noul": arguments_p}}})

    with (httpx.Client(transport=httpx.MockTransport(generator_http)) as generator_client,
          httpx.Client(transport=httpx.MockTransport(verifier_http)) as verifier_client):
        generator = FunctionCallingLLM("test-key", model="test-model", api_style=style,
                                       gateway="direct", client=generator_client)
        verifier = JevVerifier("test-key", model="jev-1.13.0", gateway="direct", client=verifier_client)
        result = run_example(example, generator, verifier, max_repairs=3,
                             executor=lambda calls: executions.append(deepcopy(calls)))

    assert result["repairs"] == 2 and result["accepted"]
    assert len(requests) == len(checks) == 3 and len(executions) == 1
    assert [call["name"] for call in checks] == ["other", "weather", "weather"]
    for attempt, payload in enumerate(requests):
        messages = payload["input" if style == "responses" else "messages"]
        conversation = [message for message in messages if message["role"] != "system"]
        assert conversation[-1] == {"role": "user", "content": original_request}
        assert all(message["role"] != "tool" for message in messages)
        assert payload["tools"] == requests[0]["tools"]
        assert payload["tool_choice"] == ({"type": "auto", "disable_parallel_tool_use": True}
                                          if style == "anthropic_messages" else "auto")
        if attempt == 0:
            assert conversation == [{"role": "user", "content": original_request}]
        else:
            assert len(conversation) == 2
            header, encoded, instruction = conversation[0]["content"].split("\n", 2)
            assert "oldest first" in header and "NOT executed" in header
            accumulated = json.loads(encoded)
            assert accumulated == [{"round": i + 1, "calls": result["trace"][i]["proposal"]["calls"],
                                    "feedback": result["trace"][i]["feedback"]} for i in range(attempt)]
            assert "original request below" in instruction
    assert example == before
