import json

import httpx
import pytest

from jev_fc.llm import FunctionCallingLLM, LLMAPIError


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
    assert "NOT executed" in observed[0]["messages"][-1]["content"]
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
