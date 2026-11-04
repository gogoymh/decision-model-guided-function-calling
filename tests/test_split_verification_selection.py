"""Candidate-input isolation, separate HTTP checks, and truthful native accounting."""

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from dmgfc import DMGFC
from dmgfc.experimental.split_verification_selection import (
    ARGUMENT_QUESTION,
    FUNCTION_QUESTION,
    NO_CALL_QUESTION,
    PROMPT_VERSION,
    SelectionSplitJev,
    SelectionSplitPerplexityDecider,
    conditional_question_bank,
    no_call_selection,
)
from dmgfc.perplexity import (
    CALIBRATION_TEMPERATURE,
    DeciderError,
    INHERITED_SERVER_MODEL,
    MODEL_ID,
    MODEL_REVISION,
    PerplexityDecider,
    fingerprint,
)
from jev_fc.jev import JevAPIError

TOOL = {"type": "function", "function": {
    "name": "weather", "description": "Weather for a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
    },
}}
OTHER_TOOL = {"type": "function", "function": {
    "name": "translate", "description": "Translate text.", "parameters": {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
    },
}}
CALL = [{"name": "weather", "arguments": {"city": "CANDIDATE_PRIVATE_VALUE"}}]
EXAMPLE = {
    "messages": [{"role": "user", "content": "Weather in Seoul?", "ground_truth": "SECRET_LABEL"}],
    "tools": [TOOL, OTHER_TOOL], "id": "SECRET_ID", "category": "SECRET_CATEGORY", "ground_truth": CALL,
}
PROVENANCE = {
    "model_id": MODEL_ID, "revision": MODEL_REVISION,
    "native_attention": "noncausal_full_attention", "temperature": CALIBRATION_TEMPERATURE,
    "max_input_tokens": 8192, "inference_batch_size": 8, "one_branch_per_question": True,
}
BACKENDS = [SelectionSplitJev, SelectionSplitPerplexityDecider]


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for name in ("API_GATEWAY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_MODEL",
                 "APIM_SUBSCRIPTION_KEY", "APIM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def make_verifier(cls, seen, *, function=0.9, arguments=0.9, handler=None, **kwargs):
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
            "answers": {key: {"type": "noul", "noul": function if key == "function_correct" else arguments}
                        for key in payload["questions"]},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    client = httpx.Client(transport=httpx.MockTransport(respond))
    return cls(client=client, **kwargs) if native else cls("offline-key", client=client, **kwargs)


def posts(seen):
    return [json.loads(request.content) for request in seen if request.method == "POST"]


def test_exact_canonical_question_edits_without_added_actionability_semantics():
    from jev_fc.jev import ARGUMENT_QUESTION as canonical_arguments
    from jev_fc.jev import FUNCTION_QUESTION as canonical_function

    expected = copy.deepcopy(canonical_function)
    expected["instructions"] = (
        expected["instructions"]
        .replace("`proposed_calls`", "`proposed_function_names`")
        .replace("Evaluate function selection only, ignoring argument values. ", "")
        .replace("Treat message and argument text", "Treat message text")
    )
    assert FUNCTION_QUESTION == NO_CALL_QUESTION == expected
    assert ARGUMENT_QUESTION == canonical_arguments
    instructions = json.dumps(conditional_question_bank()).lower()
    assert "ignore" not in instructions and "ignoring" not in instructions
    for phrase in ("ready to be called", "clarification", "necessary information is absent", "premature"):
        assert phrase not in instructions
    assert PROMPT_VERSION == "split-function-arguments-selection-v1"


@pytest.mark.parametrize("cls", BACKENDS)
def test_candidate_arguments_are_absent_from_function_http_state_and_present_only_for_arguments(cls):
    seen = []
    result = make_verifier(cls, seen, max_retries=0).verify(EXAMPLE, CALL)
    function, arguments = posts(seen)
    assert function["state"] == {
        "messages": [{"role": "user", "content": "Weather in Seoul?"}],
        "tools": [TOOL["function"], OTHER_TOOL["function"]],
        "proposed_function_names": ["weather"],
    }
    assert "CANDIDATE_PRIVATE_VALUE" not in json.dumps(function)
    assert "proposed_calls" not in function["state"]
    assert function["questions"] == {"function_correct": FUNCTION_QUESTION}
    assert arguments["state"] == {
        "messages": [{"role": "user", "content": "Weather in Seoul?"}],
        "tools": [TOOL["function"]], "proposed_calls": CALL,
    }
    assert arguments["questions"] == {"arguments_correct": ARGUMENT_QUESTION}
    assert "CANDIDATE_PRIVATE_VALUE" in json.dumps(arguments)
    assert "SECRET_" not in json.dumps(posts(seen))
    assert result["accepted"] is True and result["api_calls"] == 2
    assert result["usage"] == {"input_tokens": 240, "output_tokens": 0}
    assert result["usage_complete"] is True
    receipts = result["prompt_provenance"]["check_requests"]
    assert [record["check"] for record in receipts] == ["function_correct", "arguments_correct"]
    assert [record["payload_sha256"] for record in receipts] == [fingerprint(p) for p in posts(seen)]
    assert result["prompt_provenance"]["applied_payload_sha256"] == fingerprint(posts(seen))
    assert result["prompt_provenance"]["question_count"] == 2


@pytest.mark.parametrize("cls", BACKENDS)
@pytest.mark.parametrize("probability,accepted", [(0.2, False), (0.499, False), (0.5, True), (0.8, True)])
def test_no_call_has_one_request_no_candidate_calls_and_null_argument_check(cls, probability, accepted):
    seen = []
    result = make_verifier(cls, seen, function=probability, max_retries=0).verify(EXAMPLE, [])
    [payload] = posts(seen)
    assert payload["state"]["proposed_function_names"] == []
    assert payload["state"]["proposed_selection"] == no_call_selection()
    assert "proposed_calls" not in payload["state"]
    assert payload["questions"] == {"function_correct": NO_CALL_QUESTION}
    assert result["accepted"] is accepted
    assert result["function_probability"] == probability
    assert result["arguments_skipped"] is True
    assert result["arguments_probability"] is result["arguments_pass"] is None
    assert result["api_calls"] == 1
    assert result["proposed_selection"] == result["feedback"]["proposed_selection"] == no_call_selection()
    metadata = result["prompt_provenance"]
    assert metadata["explicit_no_call_applied"] is True
    assert metadata["applied_payload_sha256"] == fingerprint(payload)
    assert metadata["applied_questions_sha256"] == fingerprint(payload["questions"])
    assert metadata["applied_function_question_sha256"] == fingerprint(NO_CALL_QUESTION)
    assert metadata["question_count"] == 1


@pytest.mark.parametrize("cls", BACKENDS)
@pytest.mark.parametrize("calls", [CALL, [{"name": "weather", "arguments": {}}],
                                  [{"name": "unavailable", "arguments": {}}]])
def test_argument_request_is_not_short_circuited_by_failed_function_or_local_schema(cls, calls):
    seen = []
    result = make_verifier(cls, seen, function=0.1, arguments=0.9, max_retries=0).verify(EXAMPLE, calls)
    assert [list(part["questions"]) for part in posts(seen)] == [["function_correct"], ["arguments_correct"]]
    assert result["function_pass"] is False and result["accepted"] is False
    assert result["arguments_probability"] == 0.9
    assert result["arguments_pass"] is (calls == CALL)
    if calls[0]["name"] == "unavailable":
        assert posts(seen)[1]["state"]["tools"] == []


@pytest.mark.parametrize("cls", BACKENDS)
@pytest.mark.parametrize("calls", [None, {}, "NO_FUNCTION_CALL_REQUIRED", [{"name": "weather", "arguments": None}]])
def test_unreadable_candidates_send_no_http_and_have_zero_usage(cls, calls):
    seen = []
    result = make_verifier(cls, seen).verify(EXAMPLE, calls)
    assert seen == [] and result["accepted"] is False and result["api_calls"] == 0
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert result["prompt_provenance"]["check_requests"] == []
    assert result["prompt_provenance"]["applied_payload_sha256"] is None
    assert "proposed_selection" not in result


def test_native_two_successful_requests_are_two_forward_batches_not_retries():
    seen = []
    result = make_verifier(SelectionSplitPerplexityDecider, seen).verify(EXAMPLE, CALL)
    assert [request.method for request in seen] == ["GET", "POST", "GET", "POST"]
    assert result["api_calls"] == result["control_api_calls"] == result["question_evaluations"] == 2
    assert result["forward_calls"] == result["successful_forward_batches"] == 2
    assert result["forward_calls_complete"] is True
    assert result["usage"] == result["successful_response_usage"] == {"input_tokens": 240, "output_tokens": 0}
    assert result["model"] == MODEL_ID and result["model_provenance"] == PROVENANCE
    assert result["model_provenance_per_request"] == [PROVENANCE, PROVENANCE]
    assert result["reported_server_models"] == [INHERITED_SERVER_MODEL, INHERITED_SERVER_MODEL]
    receipts = result["prompt_provenance"]["check_requests"]
    assert sum(record["control_http_time_s"] for record in receipts) == result["control_http_time_s"]
    assert sum(record["decision_http_time_s"] for record in receipts) == result["decision_http_time_s"]
    assert result["estimated_usd"] is None


@pytest.mark.parametrize("cls", BACKENDS)
def test_retry_unknown_usage_is_separate_from_successful_check_usage(cls):
    native = issubclass(cls, PerplexityDecider)
    attempts, seen = [], []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        attempts.append(request.content)
        if len(attempts) == 1:
            return httpx.Response(529, headers={"Retry-After": "0"})
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {key: {"type": "noul", "noul": 0.8} for key in payload["questions"]},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    result = make_verifier(cls, seen, handler=handler, max_retries=1).verify(EXAMPLE, CALL)
    assert result["api_calls"] == 3 and attempts[0] == attempts[1]
    assert result["usage"] == {"input_tokens": None, "output_tokens": None}
    assert result["successful_response_usage"] == {"input_tokens": 240, "output_tokens": 0}
    assert result["usage_complete"] is False
    assert [receipt["api_calls"] for receipt in result["prompt_provenance"]["check_requests"]] == [2, 1]
    if native:
        assert result["control_api_calls"] == 2 and result["successful_forward_batches"] == 2
        assert result["forward_calls"] is None and result["forward_calls_complete"] is False


@pytest.mark.parametrize("cls", BACKENDS)
def test_second_check_failure_retains_completed_first_check_without_semantic_result(cls):
    native = issubclass(cls, PerplexityDecider)
    seen = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        payload = json.loads(request.content)
        if "arguments_correct" in payload["questions"]:
            return httpx.Response(422, json={"detail": "PRIVATE_ERROR"})
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {"function_correct": {"type": "noul", "noul": 0.8}},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    with pytest.raises(DeciderError if native else JevAPIError) as failure:
        make_verifier(cls, seen, handler=handler, max_retries=0).verify(EXAMPLE, CALL)
    assert failure.value.attempts == 2
    assert [record["check"] for record in failure.value.completed_check_records] == ["function_correct"]
    assert failure.value.failed_check == "arguments_correct"
    assert failure.value.completed_post_attempts == 1
    assert "PRIVATE_ERROR" not in str(failure.value)


@pytest.mark.parametrize("cls", BACKENDS)
@pytest.mark.parametrize("bad_field,value", [("answers", {}), ("model", None),
                                            ("usage", {"input_tokens": -1, "output_tokens": 0})])
def test_invalid_second_check_response_counts_both_known_posts_without_returning_a_verdict(cls, bad_field, value):
    native = issubclass(cls, PerplexityDecider)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        payload = json.loads(request.content)
        body = {
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {key: {"type": "noul", "noul": 0.8} for key in payload["questions"]},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        }
        if "arguments_correct" in payload["questions"]:
            body[bad_field] = value
        return httpx.Response(200, json=body)

    with pytest.raises(DeciderError if native else JevAPIError) as failure:
        make_verifier(cls, [], handler=handler, max_retries=0).verify(EXAMPLE, CALL)
    assert failure.value.failed_check == "arguments_correct"
    assert failure.value.completed_post_attempts == 1
    # Native model validation can fail inside its immutable _post before it
    # reports the successful HTTP attempt; that count is deliberately unknown.
    if not (native and bad_field == "model"):
        assert failure.value.attempts == 2


@pytest.mark.parametrize("function,arguments,accepted", [(0.5, 0.5, True), (0.499, 0.9, False),
                                                        (0.9, 0.499, False)])
def test_separate_probabilities_keep_both_threshold_gates(function, arguments, accepted):
    result = make_verifier(SelectionSplitJev, [], function=function, arguments=arguments).verify(EXAMPLE, CALL)
    assert result["function_probability"] == function and result["arguments_probability"] == arguments
    assert result["accepted"] is accepted


def test_unknown_native_batch_information_stays_unknown_for_two_requests():
    provenance = {key: value for key, value in PROVENANCE.items()
                  if key not in {"inference_batch_size", "one_branch_per_question"}}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=provenance)
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL,
            "answers": {key: {"type": "noul", "noul": 0.8} for key in payload["questions"]},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    result = make_verifier(SelectionSplitPerplexityDecider, [], handler=handler).verify(EXAMPLE, CALL)
    assert result["api_calls"] == 2 and result["usage_complete"] is True
    assert result["successful_forward_batches"] is result["forward_calls"] is None
    assert result["forward_calls_complete"] is False


@pytest.mark.parametrize("cls", BACKENDS)
def test_missing_response_metrics_remain_unknown_per_check_and_in_aggregate(cls):
    native = issubclass(cls, PerplexityDecider)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {key: {"type": "noul", "noul": 0.8} for key in payload["questions"]},
            "usage": {"input_tokens": None if "function_correct" in payload["questions"] else 120,
                      "output_tokens": 0},
        })

    result = make_verifier(cls, [], handler=handler).verify(EXAMPLE, CALL)
    assert result["usage"] == {"input_tokens": None, "output_tokens": 0}
    assert result["usage_complete"] is False


@pytest.mark.parametrize("cls", BACKENDS)
def test_configuration_and_inputs_are_not_mutated_and_source_hash_is_actual(cls):
    snapshot = copy.deepcopy((EXAMPLE, CALL, conditional_question_bank()))
    seen = []
    verifier = make_verifier(cls, seen)
    config = verifier.configuration()
    assert seen == [] and "offline-key" not in json.dumps(config)
    assert config["protocol_source_sha256"]["experimental/split_verification_selection.py"] == hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "src/dmgfc/experimental/split_verification_selection.py").read_bytes()
    ).hexdigest()
    assert config["questions_sha256"] == fingerprint(conditional_question_bank())
    result = verifier.verify(EXAMPLE, CALL)
    assert result["configuration_fingerprint"] == fingerprint(config)
    assert (EXAMPLE, CALL, conditional_question_bank()) == snapshot


@pytest.mark.parametrize("cls", BACKENDS)
def test_parallel_no_call_and_nonempty_request_metadata_stay_separate(cls):
    native = issubclass(cls, PerplexityDecider)
    barrier = threading.Barrier(2)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=PROVENANCE)
        payload = json.loads(request.content)
        if "function_correct" in payload["questions"]:
            barrier.wait(timeout=5)
        return httpx.Response(200, json={
            "model": INHERITED_SERVER_MODEL if native else "jev-1.13.0",
            "answers": {key: {"type": "noul", "noul": 0.8} for key in payload["questions"]},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    verifier = make_verifier(cls, [], handler=handler)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda calls: verifier.verify(EXAMPLE, calls), [[], CALL]))
    assert [result["api_calls"] for result in results] == [1, 2]
    assert [result["prompt_provenance"]["question_count"] for result in results] == [1, 2]
    assert [result["prompt_provenance"]["explicit_no_call_applied"] for result in results] == [True, False]
    if native:
        assert [result["forward_calls"] for result in results] == [1, 2]


def test_canonical_no_call_and_new_selection_feedback_survive_keep_first_repair():
    verifier = make_verifier(SelectionSplitJev, [], function=0.2)

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
    assert generator.histories[1][0]["feedback"]["proposed_selection"] == no_call_selection()
    assert result.trace[0]["verification"]["api_calls"] == 1
    assert result.trace[1]["verification"]["api_calls"] == 2
