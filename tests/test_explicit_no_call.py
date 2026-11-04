"""Explicit candidate state, unchanged gates and provenance; HTTP is mocked."""

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from dmgfc import DMGFC, Jev
from dmgfc.experimental.explicit_no_call import (
    NO_CALL_QUESTION,
    PROMPT_VERSION,
    ExplicitNoCallJev,
    ExplicitNoCallPerplexityDecider,
    conditional_question_bank,
    no_call_selection,
)
from dmgfc.perplexity import (
    CALIBRATION_TEMPERATURE,
    INHERITED_SERVER_MODEL,
    MODEL_ID,
    MODEL_REVISION,
    PerplexityDecider,
    fingerprint,
)
from jev_fc.jev import ARGUMENT_QUESTION, FUNCTION_QUESTION

TOOL = {"type": "function", "function": {
    "name": "weather", "description": "Weather for a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
    },
}}
CALL = [{"name": "weather", "arguments": {"city": "Seoul"}}]
EXAMPLE = {
    "messages": [{"role": "user", "content": "Weather in Seoul?", "ground_truth": "SECRET_LABEL"}],
    "tools": [TOOL], "id": "SECRET_ID", "category": "SECRET_CATEGORY", "ground_truth": CALL,
}
PROVENANCE = {
    "model_id": MODEL_ID, "revision": MODEL_REVISION,
    "native_attention": "noncausal_full_attention", "temperature": CALIBRATION_TEMPERATURE,
    "max_input_tokens": 8192, "inference_batch_size": 8, "one_branch_per_question": True,
}
BACKENDS = [(ExplicitNoCallJev, Jev), (ExplicitNoCallPerplexityDecider, PerplexityDecider)]


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for name in ("API_GATEWAY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_MODEL",
                 "APIM_SUBSCRIPTION_KEY", "APIM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def make_verifier(cls, seen, *, probability=0.9, handler=None, **kwargs):
    native = issubclass(cls, PerplexityDecider)

    def respond(request):
        seen.append(request)
        if handler is not None:
            return handler(request)
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {key: {"type": "noul", "noul": probability} for key in payload["questions"]},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    client = httpx.Client(transport=httpx.MockTransport(respond))
    return cls(client=client, **kwargs) if native else cls("offline-key", client=client, **kwargs)


def posts(seen):
    return [json.loads(request.content) for request in seen if request.method == "POST"]


@pytest.mark.parametrize("cls,_base", BACKENDS)
@pytest.mark.parametrize("probability,accepted", [(0.2, False), (0.499, False), (0.5, True), (0.8, True)])
def test_explicit_no_call_keeps_function_threshold_polarity_and_one_question(cls, _base, probability, accepted):
    seen = []
    verifier = make_verifier(cls, seen, probability=probability, max_retries=0)
    result = verifier.verify(EXAMPLE, [])
    [payload] = posts(seen)
    assert payload["state"]["proposed_calls"] == []
    assert payload["state"]["proposed_selection"] == no_call_selection()
    assert payload["questions"] == {"function_correct": NO_CALL_QUESTION}
    assert result["accepted"] is accepted
    assert result["function_probability"] == probability
    assert result["arguments_skipped"] is True
    assert result["arguments_probability"] is result["arguments_pass"] is None
    assert result["feedback"]["failed_checks"] == ([] if accepted else ["function_selection"])
    assert result["api_calls"] == 1
    assert result["proposed_selection"] == result["feedback"]["proposed_selection"] == no_call_selection()
    metadata = result["prompt_provenance"]
    assert metadata["explicit_no_call_applied"] is True
    assert metadata["applied_payload_sha256"] == fingerprint(payload)
    assert metadata["applied_questions_sha256"] == fingerprint(payload["questions"])
    assert metadata["applied_function_question_sha256"] == fingerprint(NO_CALL_QUESTION)
    assert metadata["question_count"] == 1
    assert "SECRET_" not in json.dumps(payload)


@pytest.mark.parametrize("cls,base", BACKENDS)
@pytest.mark.parametrize("calls", [CALL, [{"name": "weather", "arguments": {}}],
                                  [{"name": "unavailable", "arguments": {}}]])
def test_nonempty_payloads_are_byte_identical_to_default_and_keep_schema_gate(cls, base, calls):
    changed_seen, original_seen = [], []
    changed = make_verifier(cls, changed_seen, probability=0.9, max_retries=0).verify(EXAMPLE, calls)
    original = make_verifier(base, original_seen, probability=0.9, max_retries=0).verify(EXAMPLE, calls)
    assert [r.content for r in changed_seen if r.method == "POST"] == [
        r.content for r in original_seen if r.method == "POST"]
    assert posts(changed_seen)[0]["questions"] == {
        "function_correct": FUNCTION_QUESTION, "arguments_correct": ARGUMENT_QUESTION,
    }
    assert "proposed_selection" not in changed and "proposed_selection" not in changed["feedback"]
    assert changed["feedback"] == original["feedback"]
    for key in ("accepted", "function_probability", "arguments_probability", "function_pass",
                "arguments_pass", "arguments_skipped", "schema_errors", "usage", "api_calls"):
        assert changed[key] == original[key]
    assert changed["prompt_provenance"]["explicit_no_call_applied"] is False
    assert changed["prompt_provenance"]["applied_payload_sha256"] == fingerprint(posts(changed_seen)[0])


@pytest.mark.parametrize("cls,base", BACKENDS)
@pytest.mark.parametrize("calls", [None, {}, "NO_FUNCTION_CALL", [{"name": "weather", "arguments": None}]])
def test_unreadable_and_malformed_canonical_candidates_make_no_requests(cls, base, calls):
    changed_seen, original_seen = [], []
    changed = make_verifier(cls, changed_seen).verify(EXAMPLE, calls)
    original = make_verifier(base, original_seen).verify(EXAMPLE, calls)
    assert changed_seen == original_seen == []
    assert changed["accepted"] is original["accepted"] is False
    assert changed["api_calls"] == 0
    assert changed["feedback"] == original["feedback"]
    assert "proposed_selection" not in changed
    metadata = changed["prompt_provenance"]
    assert metadata["explicit_no_call_applied"] is False
    assert metadata["applied_questions_sha256"] is metadata["applied_function_question_sha256"] is None
    assert metadata["applied_payload_sha256"] is None
    assert metadata["question_count"] == 0


@pytest.mark.parametrize("cls,_base", BACKENDS)
def test_input_and_question_constants_are_not_mutated_and_selection_is_fresh(cls, _base):
    snapshot = copy.deepcopy((EXAMPLE, CALL, FUNCTION_QUESTION, ARGUMENT_QUESTION, NO_CALL_QUESTION))
    seen = []
    verifier = make_verifier(cls, seen)
    first = verifier.verify(EXAMPLE, [])
    first["feedback"]["proposed_selection"]["description"] = "Mutated saved feedback"
    verifier.verify(EXAMPLE, CALL)
    second = verifier.verify(EXAMPLE, [])
    assert (EXAMPLE, CALL, FUNCTION_QUESTION, ARGUMENT_QUESTION, NO_CALL_QUESTION) == snapshot
    assert second["feedback"]["proposed_selection"] == no_call_selection()
    bank = conditional_question_bank()
    bank["function_correct_when_no_function_call"]["instructions"] = "Mutated returned bank"
    assert NO_CALL_QUESTION == snapshot[-1]
    assert posts(seen)[0]["state"]["proposed_selection"] == no_call_selection()


@pytest.mark.parametrize("cls,base", BACKENDS)
def test_configuration_hashes_real_conditional_protocol_and_credentials_are_absent(cls, base):
    seen = []
    verifier = make_verifier(cls, seen)
    original = make_verifier(base, [])
    config = verifier.configuration()
    assert seen == []
    assert config["questions_sha256"] == fingerprint(conditional_question_bank())
    assert config["inherited_questions_sha256"] == fingerprint({
        "function_correct": FUNCTION_QUESTION, "arguments_correct": ARGUMENT_QUESTION,
    })
    assert config["questions_sha256"] != config["inherited_questions_sha256"]
    assert config["prompt_version"] == PROMPT_VERSION
    assert config["protocol_source_sha256"]["experimental/explicit_no_call.py"] == hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "src/dmgfc/experimental/explicit_no_call.py").read_bytes()
    ).hexdigest()
    assert "offline-key" not in json.dumps(config) and "Authorization" not in json.dumps(config)
    if issubclass(cls, PerplexityDecider):
        before = original.configuration()
        for key in ("model_id", "revision", "native_attention", "calibration_temperature",
                    "max_input_tokens", "base_url", "wire_model", "timeout_s", "max_retries"):
            assert config[key] == before[key]
        for key, value in before["protocol_source_sha256"].items():
            assert config["protocol_source_sha256"][key] == value
        assert fingerprint(config) != fingerprint(before)


def test_native_runtime_provenance_and_accounting_are_preserved():
    seen = []
    verifier = make_verifier(ExplicitNoCallPerplexityDecider, seen)
    result = verifier.verify(EXAMPLE, [])
    assert [r.method for r in seen] == ["GET", "POST"]
    assert result["model_provenance"] == PROVENANCE
    assert result["reported_server_model"] == INHERITED_SERVER_MODEL
    assert result["model"] == MODEL_ID
    assert result["api_calls"] == result["control_api_calls"] == result["question_evaluations"] == 1
    assert result["forward_calls"] == result["successful_forward_batches"] == 1
    assert result["configuration_fingerprint"] == fingerprint(verifier.configuration())
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 0}
    assert result["estimated_usd"] is None


@pytest.mark.parametrize("cls,_base", BACKENDS)
def test_retry_count_and_payload_identity_remain_bounded(cls, _base):
    native = issubclass(cls, PerplexityDecider)
    seen, attempts = [], []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        attempts.append(request.content)
        if len(attempts) == 1:
            return httpx.Response(529, headers={"Retry-After": "0"})
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {"function_correct": {"type": "noul", "noul": 0.8}},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    verifier = make_verifier(cls, seen, handler=handler, max_retries=1)
    result = verifier.verify(EXAMPLE, [])
    assert result["api_calls"] == 2 and attempts[0] == attempts[1]
    assert result["accepted"] is True
    assert result["prompt_provenance"]["applied_questions_sha256"] == fingerprint(
        json.loads(attempts[0])["questions"])
    if native:
        assert result["control_api_calls"] == 1
        assert result["usage"] == {"input_tokens": None, "output_tokens": None}
        assert result["successful_response_usage"] == {"input_tokens": 120, "output_tokens": 0}


@pytest.mark.parametrize("cls,_base", BACKENDS)
def test_parallel_requests_keep_actual_prompt_metadata_separate(cls, _base):
    native = issubclass(cls, PerplexityDecider)
    barrier = threading.Barrier(2)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        payload = json.loads(request.content)
        barrier.wait(timeout=5)
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {key: {"type": "noul", "noul": 0.9} for key in payload["questions"]},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    verifier = make_verifier(cls, [], handler=handler)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda calls: verifier.verify(EXAMPLE, calls), [[], CALL]))
    no_call, call = [r["prompt_provenance"] for r in results]
    assert no_call["explicit_no_call_applied"] is True and no_call["question_count"] == 1
    assert call["explicit_no_call_applied"] is False and call["question_count"] == 2
    assert no_call["applied_function_question_sha256"] == fingerprint(NO_CALL_QUESTION)
    assert call["applied_function_question_sha256"] == fingerprint(FUNCTION_QUESTION)


@pytest.mark.parametrize("cls,_base", BACKENDS)
def test_canonical_no_call_survives_repair_history_and_keep_first_without_dispatch(cls, _base):
    seen = []
    verifier = make_verifier(cls, seen, probability=0.2)

    class Generator:
        def __init__(self):
            self.histories = []

        def generate(self, example, history=None):
            self.histories.append(copy.deepcopy(history or []))
            return {"calls": [] if not history else CALL, "usage": {}, "latency_s": 0.0}

    generator = Generator()
    result = DMGFC(generator, tools=[TOOL], jev=verifier, max_repairs=1,
                   preserve_initial_no_call=False).call("Weather in Seoul?")
    assert result.exhausted is True and result.final_attempt == 0
    assert result.calls == [] and result.executed is False
    assert generator.histories[1][0]["calls"] == []
    assert generator.histories[1][0]["feedback"]["proposed_selection"] == no_call_selection()
    assert result.trace[0]["verification"]["prompt_provenance"]["explicit_no_call_applied"] is True
    assert result.trace[1]["proposal"]["calls"] == CALL
    assert "proposed_selection" not in posts(seen)[1]["state"]
    assert all("NO_FUNCTION_CALL" not in json.dumps(p["state"]["tools"]) for p in posts(seen))
