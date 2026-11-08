"""Jev and the PPLX Decider: the wire format, verdicts, keys, thresholds and failures."""

import json

import httpx
import pytest

import dmcall
from conftest import EXAMPLE, SECRET, WEATHER, answer, client, seoul

MESSAGES, TOOLS = EXAMPLE["messages"], EXAMPLE["tools"]
REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce"
PROVENANCE = {"model_id": "perplexity-ai/pplx-decider-v1.1-27b", "revision": REVISION,
              "native_attention": "noncausal_full_attention", "max_input_tokens": 8192,
              "temperature": 1.0087417621345625}


def jev(handler, seen=None, **options):
    return dmcall.Jev("jev-key", client=client(handler, seen), sleep=lambda _: None, **options)


def test_wire_format_sends_only_observable_inputs():
    seen = []
    tools = [{**WEATHER, "metadata": {"reference": "get_weather(city='Seoul')"}}]
    messages = MESSAGES + [{"role": "user", "content": "hi", "ground_truth": "nope"}]
    record = jev(lambda r: httpx.Response(200, json=answer(0.9, 0.8)), seen).judge(messages, tools, seoul())
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.typesafe.ai/v1/systemone" and body["model"] == "jev-1.13.0"
    assert body["state"]["messages"][-1] == {"role": "user", "content": "hi"}
    assert body["state"]["tools"] == [WEATHER["function"]]
    assert body["state"]["proposed_calls"] == seoul()
    assert body["questions"] == {"function_correct": dmcall.FUNCTION_QUESTION,
                                 "arguments_correct": dmcall.ARGUMENT_QUESTION}
    assert record["accepted"] and record["function_probability"] == 0.9 and record["arguments_probability"] == 0.8
    assert record["feedback"]["failed_checks"] == [] and record["api_calls"] == 1


def test_no_call_asks_only_the_function_question():
    seen = []
    record = jev(lambda r: httpx.Response(200, json=answer(0.97, None)), seen).judge(MESSAGES, TOOLS, [])
    assert list(json.loads(seen[0].content)["questions"]) == ["function_correct"]
    assert record["accepted"] and record["arguments_skipped"] and record["arguments_probability"] is None
    assert record["feedback"]["checks"]["arguments"]["pass"] is None


@pytest.mark.parametrize("function, arguments, failed", [(0.4, 0.9, ["function_selection"]),
                                                         (0.9, 0.49, ["arguments"]),
                                                         (0.5, 0.5, [])])
def test_both_questions_must_pass_at_the_threshold(function, arguments, failed):
    record = jev(lambda r: httpx.Response(200, json=answer(function, arguments))).judge(MESSAGES, TOOLS, seoul())
    assert record["feedback"]["failed_checks"] == failed and record["accepted"] is (failed == [])


def test_schema_failure_is_judged_but_cannot_be_overruled():
    seen = []
    record = jev(lambda r: httpx.Response(200, json=answer(0.99, 0.99)), seen).judge(
        MESSAGES, TOOLS, [{"name": "get_weather", "arguments": {}}])
    assert len(seen) == 1 and record["arguments_probability"] == 0.99  # a readable candidate is still judged
    assert record["arguments_pass"] is False and not record["accepted"] and record["schema_errors"]


@pytest.mark.parametrize("calls", [None, [{"name": "get_weather", "arguments": None}]])
def test_unreadable_candidate_is_not_sent(calls):
    def handler(_):
        pytest.fail("An unreadable candidate must not reach the decision model")

    record = jev(handler).judge(MESSAGES, TOOLS, calls)
    assert record["api_calls"] == 0 and record["function_probability"] is None and not record["accepted"]


def test_thresholds_keys_and_model(monkeypatch):
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        dmcall.Jev()
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-2")
    seen = []
    model = dmcall.Jev(threshold=0.7, arguments_threshold=0.9,
                       client=client(lambda r: httpx.Response(200, json=answer(0.75, 0.85)), seen))
    record = model.judge(MESSAGES, TOOLS, seoul())
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}" and SECRET not in repr(model)
    assert json.loads(seen[0].content)["model"] == "jev-2"
    assert record["function_pass"] is True and record["arguments_pass"] is False
    for bad in (1.5, -0.1, True):
        with pytest.raises(ValueError, match="Thresholds"):
            dmcall.Jev("k", threshold=bad)


def test_retries_are_bounded_and_errors_sanitized():
    replies = iter([httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json=answer())])
    record = jev(lambda r: next(replies)).judge(MESSAGES, TOOLS, seoul())
    assert record["api_calls"] == 2
    seen = []
    failing = jev(lambda r: httpx.Response(401, text=SECRET, headers={"x-request-id": "req-7"}), seen)
    with pytest.raises(dmcall.DecisionModelError) as caught:
        failing.judge(MESSAGES, TOOLS, seoul())
    assert len(seen) == 1 and caught.value.status_code == 401 and caught.value.request_id == "req-7"
    assert SECRET not in str(caught.value)
    overloaded = jev(lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(dmcall.DecisionModelError) as caught:
        overloaded.judge(MESSAGES, TOOLS, seoul())
    assert caught.value.attempts == 3


@pytest.mark.parametrize("body", [{"answers": {}}, answer(1.5), {**answer(), "usage": None},
                                  {**answer(), "model": ""}, ["not", "an", "object"]])
def test_invalid_responses_are_errors_not_rejections(body):
    with pytest.raises(dmcall.DecisionModelError):
        jev(lambda r: httpx.Response(200, json=body)).judge(MESSAGES, TOOLS, seoul())


# PPLX Decider ------------------------------------------------------------------------------------
def pplx(handler, seen=None, **options):
    return dmcall.PPLXDecider("http://127.0.0.1:8000", client=client(handler, seen), **options)


def test_pplx_uses_the_same_questions_and_reports_the_checkpoint():
    seen = []
    decider = pplx(lambda r: httpx.Response(200, json=answer(0.9, 0.2, model="autojev-qwen3.8-27b")), seen)
    record = decider.judge(MESSAGES, TOOLS, seoul())
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://127.0.0.1:8000/v1/systemone" and body["model"] == "jev-latest"
    assert body["questions"]["function_correct"] == dmcall.FUNCTION_QUESTION
    assert record["model"] == "perplexity-ai/pplx-decider-v1.1-27b" and not record["accepted"]
    assert record["feedback"]["source"] == "perplexity_decider_noul_and_deterministic_schema_validation"
    assert "authorization" not in seen[0].headers


def test_pplx_missing_usage_stays_unknown():
    body = {"model": "autojev-qwen3.8-27b", "answers": answer()["answers"]}
    record = pplx(lambda r: httpx.Response(200, json=body)).judge(MESSAGES, TOOLS, seoul())
    assert record["usage"] == {"input_tokens": None, "output_tokens": None}


@pytest.mark.parametrize("body", [answer(model="some-other-model"),
                                  {**answer(model="perplexity-ai/pplx-decider-v1.1-27b"), "revision": "0" * 40}])
def test_pplx_rejects_another_model(body):
    with pytest.raises(dmcall.DecisionModelError) as caught:
        pplx(lambda r: httpx.Response(200, json=body)).judge(MESSAGES, TOOLS, seoul())
    assert caught.value.code == "model_mismatch"


def test_pplx_provenance_check_is_opt_in():
    seen = []

    def handler(request):
        if request.url.path == "/provenance":
            return httpx.Response(200, json=PROVENANCE)
        return httpx.Response(200, json=answer(model="autojev-qwen3.8-27b"))

    decider = pplx(handler, seen, check_provenance=True)
    decider.judge(MESSAGES, TOOLS, seoul())
    decider.judge(MESSAGES, TOOLS, seoul())
    assert [request.url.path for request in seen] == ["/provenance", "/v1/systemone", "/v1/systemone"]
    unchecked = []
    pplx(handler, unchecked).judge(MESSAGES, TOOLS, seoul())
    assert [request.url.path for request in unchecked] == ["/v1/systemone"]
    wrong = pplx(lambda r: httpx.Response(200, json={**PROVENANCE, "revision": "main"}), check_provenance=True)
    with pytest.raises(dmcall.DecisionModelError) as caught:
        wrong.judge(MESSAGES, TOOLS, seoul())
    assert caught.value.code == "provenance_mismatch"


def test_pplx_context_limit_is_an_error_and_no_call_skips_the_request():
    over = pplx(lambda r: httpx.Response(422, json={"detail": "Prepared input exceeds the 8192-token limit."}))
    with pytest.raises(dmcall.DecisionModelError) as caught:
        over.judge(MESSAGES, TOOLS, seoul())
    assert caught.value.code == "context_limit_exceeded"


@pytest.mark.parametrize("url", ["http://10.0.0.4:8000", "https://user:pw@example.com", "http://localhost:8000/v1",
                                 "ftp://localhost"])
def test_pplx_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        dmcall.PPLXDecider(url)
    assert dmcall.PPLXDecider("https://decider.example.com").base_url == "https://decider.example.com"
