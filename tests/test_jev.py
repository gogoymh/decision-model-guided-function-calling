"""Wire-contract and acceptance-policy tests; all HTTP requests are mocked."""

import json

import httpx
import pytest

from jev_fc.jev import JevAPIError, JevVerifier, validate_calls


@pytest.fixture
def example():
    return {
        "id": "fixture",
        "messages": [{"role": "user", "content": "Find the weather in Seoul", "private_label": "DO_NOT_SEND"}],
        "tools": [{"type": "function", "function": {
            "name": "weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1}},
                "required": ["city"],
            },
        }}],
        "ground_truth": "DO_NOT_SEND",
        "name_map": {"weather": "DO_NOT_SEND"},
    }


def wire_reply(function=0.95, arguments=0.92):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "function_correct": {"type": "noul", "noul": function},
            "arguments_correct": {"type": "noul", "noul": arguments},
        },
        "usage": {"input_tokens": 1200, "output_tokens": 40},
    }


def test_wire_contract_excludes_reference_labels_and_checks_together(example):
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert "DO_NOT_SEND" not in request.content.decode()
        assert set(observed[-1]["questions"]) == {"function_correct", "arguments_correct"}
        assert all(q["type"] == "noul" for q in observed[-1]["questions"].values())
        return httpx.Response(200, json=wire_reply())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        verifier = JevVerifier("test-key", client=client)
        result = verifier.verify(example, [{"name": "weather", "arguments": {"city": "Seoul"}, "reference": "DO_NOT_SEND"}])
    assert len(observed) == 1
    assert result["accepted"] is True
    assert result["usage"]["input_tokens"] == 1200
    assert result["api_calls"] == 1
    assert result["model"] == "jev-1.13.0"
    assert result["latency_s"] >= 0
    assert result["feedback"]["failed_checks"] == []


@pytest.mark.parametrize("p_function,p_arguments,accepted", [(0.8, 0.8, True), (0.95, 0.79, False), (0.79, 0.95, False), (0.1, 0.1, False)])
def test_both_dimensions_must_pass(example, p_function, p_arguments, accepted):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=wire_reply(p_function, p_arguments)))) as client:
        result = JevVerifier("test-key", client=client).verify(example, [{"name": "weather", "arguments": {"city": "Seoul"}}])
    assert result["accepted"] is accepted
    assert result["function_probability"] == p_function
    assert result["arguments_probability"] == p_arguments


def test_schema_failure_overrides_high_probability(example):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=wire_reply(1.0, 1.0)))) as client:
        result = JevVerifier("test-key", client=client).verify(example, [{"name": "weather", "arguments": {"city": 7}}])
    assert result["function_pass"] is True
    assert result["arguments_pass"] is False
    assert result["accepted"] is False
    assert result["schema_errors"]


def test_empty_call_collection_valid_structurally_and_judged_semantically(example):
    assert validate_calls(example, [])["accepted"] is True
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=wire_reply(0.1, 1.0)))) as client:
        result = JevVerifier("test-key", client=client).verify(example, [])
    assert result["function_pass"] is False
    assert result["arguments_pass"] is True


@pytest.mark.parametrize("calls", [None, ["weather"], [{"name": "weather", "arguments": "{}"}], [{"name": "weather", "arguments": {"city": float("nan")}}]])
def test_malformed_calls_reject_locally_without_network(example, calls):
    def handler(_):
        pytest.fail("Malformed calls must not trigger a paid API request")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = JevVerifier("test-key", client=client).verify(example, calls)
    assert result["accepted"] is False
    assert result["api_calls"] == 0
    assert result["function_probability"] is None


@pytest.mark.parametrize("arguments", [{}, {"city": "Seoul", "invented": 1}, {"city": "Seoul", "days": True}, {"city": "Seoul", "days": 0}])
def test_structural_argument_errors(example, arguments):
    result = validate_calls(example, [{"name": "weather", "arguments": arguments}])
    assert result["function_pass"] is True
    assert result["arguments_pass"] is False


def test_unknown_function_is_not_available(example):
    result = validate_calls(example, [{"name": "invented", "arguments": {}}])
    assert result["function_pass"] is False
    assert result["arguments_pass"] is False


def test_explicit_open_keyword_schema_respected(example):
    example["tools"][0]["function"]["parameters"]["additionalProperties"] = True
    assert validate_calls(example, [{"name": "weather", "arguments": {"city": "Seoul", "extra": 1}}])["accepted"] is True


def test_retry_after_then_success(example):
    attempts = []
    delays = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json=wire_reply())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = JevVerifier("test-key", client=client, sleep=delays.append).verify(example, [])
    assert delays == [0.25]
    assert result["api_calls"] == 2


def test_invalid_api_response_is_error_not_pass(example):
    body = wire_reply()
    body["answers"]["arguments_correct"]["noul"] = "0.99"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        with pytest.raises(JevAPIError, match="invalid Noul probability"):
            JevVerifier("test-key", client=client).verify(example, [])


def test_auth_failure_does_not_retry_or_disclose_response(example):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(401, text="sensitive body test-key")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JevAPIError, match="HTTP 401") as exc:
            JevVerifier("test-key", client=client).verify(example, [])
    assert len(attempts) == 1
    assert "test-key" not in str(exc.value)
    assert "sensitive" not in str(exc.value)


def test_overload_retry_budget_is_bounded(example):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(529)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JevAPIError, match="HTTP 529"):
            JevVerifier("test-key", client=client, max_retries=2, sleep=lambda _: None).verify(example, [])
    assert len(attempts) == 3


def test_invalid_tool_schema_is_configuration_error(example):
    example["tools"][0]["function"]["parameters"]["type"] = "tuple"
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        validate_calls(example, [])


def test_env_model_and_explicit_override(monkeypatch):
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-1.14.0")
    with JevVerifier("test-key") as verifier:
        assert verifier.model == "jev-1.14.0"
    with JevVerifier("test-key", model="jev-1.13.0") as verifier:
        assert verifier.model == "jev-1.13.0"
