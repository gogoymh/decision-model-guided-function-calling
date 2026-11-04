"""Generator-only text feedback cannot alter frozen Jev gates or trace values."""

import hashlib
import json
import math
from copy import deepcopy

import httpx
import pytest

from dmgfc.feedback import FeedbackGenerator, FeedbackScheme, SCHEME_IDS
from dmgfc.providers import OpenAI
from jev_fc.jev import JevVerifier
from jev_fc.llm import FunctionCallingLLM
from jev_fc.pipeline import run_example


@pytest.fixture
def example():
    return {
        "id": "private-benchmark-id", "category": "simple_python", "ground_truth": "ORACLE",
        "messages": [{"role": "user", "content": "Find the weather in Seoul at 0.5 degrees."}],
        "tools": [{"type": "function", "function": {
            "name": "weather", "description": "Weather.", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
            },
        }}],
    }


def feedback(function=.1, arguments=.8, *, skipped=False):
    return {
        "source": "jev_noul_and_deterministic_schema_validation",
        "checks": {
            "function_selection": {"pass": function is not None and function >= .5,
                                   "probability_yes": function, "threshold": .5, "schema_pass": True},
            "arguments": {"skipped": skipped, "pass": None if skipped else arguments >= .5,
                          "probability_yes": None if skipped else arguments,
                          "threshold": .5, "schema_pass": True},
        },
        "failed_checks": ["function_selection"],
        "schema_errors": [{"check": "arguments", "path": "calls[0].arguments",
                           "message": "Missing the 3 required arguments."}],
        "instruction": ("Reconsider the failed dimensions using the original request and "
                        "available function definitions, then submit a complete replacement "
                        "call collection. The probabilities are verifier judgments, not "
                        "ground truth or explanations."),
    }


@pytest.mark.parametrize("scheme_id", SCHEME_IDS[1:])
def test_every_text_scheme_has_exact_nonoverlapping_boundaries(scheme_id):
    scheme = FeedbackScheme.from_id(scheme_id)
    assert scheme.label(0) == scheme.labels[0]
    assert scheme.label(1) == scheme.labels[-1]
    assert scheme.label(None) is None
    for index, cut in enumerate(scheme.cut_points):
        assert scheme.label(math.nextafter(cut, 0)) == scheme.labels[index]
        assert scheme.label(cut) == scheme.labels[index + 1]
        assert scheme.label(math.nextafter(cut, 1)) == scheme.labels[index + 1]
    assert len(scheme.labels) == scheme.bins
    assert len(scheme.cut_points) == scheme.bins - 1


def test_user_example_uses_decimal_cuts_rather_than_thirds():
    scheme = FeedbackScheme.from_id("solved_3")
    assert scheme.cut_points == (.33, .66)
    assert [scheme.label(p) for p in (.329, .33, .659, .66)] == [
        "unsolved", "partially solved", "partially solved", "solved",
    ]


@pytest.mark.parametrize("scheme_id, labels", [
    ("solved_3", ("unsolved", "partially solved", "solved")),
    ("solved_4", ("unsolved", "weakly solved", "mostly solved", "solved")),
    ("solved_5", ("unsolved", "slightly solved", "partially solved", "mostly solved", "solved")),
    ("correctness_3", ("likely incorrect", "uncertain", "likely correct")),
    ("correctness_4", ("very likely incorrect", "likely incorrect", "likely correct", "very likely correct")),
    ("correctness_5", ("very likely incorrect", "likely incorrect", "uncertain", "likely correct",
                       "very likely correct")),
    ("action_3", ("replace", "reconsider", "keep")),
    ("action_4", ("replace", "revise", "recheck", "keep")),
    ("action_5", ("replace", "substantially revise", "revise", "recheck", "keep")),
])
def test_label_families_are_prespecified(scheme_id, labels):
    assert FeedbackScheme.from_id(scheme_id).labels == labels


@pytest.mark.parametrize("invalid", [True, False, "0.5", [], {}, -.001, 1.001, float("nan"),
                                    float("inf"), -float("inf"), 10 ** 400])
def test_invalid_probabilities_fail_closed(invalid):
    with pytest.raises(ValueError, match="Feedback probabilities"):
        FeedbackScheme.from_id("solved_3").label(invalid)


@pytest.mark.parametrize("family,bins", [
    ("solved", 3.0), ("solved", True), ("solved", 2), ("solved", 6), ("solved", None),
    ("numeric", 3), ("unknown", 3), ([], 3), ("solved", []),
])
def test_invalid_scheme_configuration_fails_before_generation(family, bins):
    with pytest.raises(ValueError):
        FeedbackScheme(family, bins)


@pytest.mark.parametrize("invalid", ["status_3", "SOLVED_3", "solved_03", "solved_3 ", 3, None, []])
def test_scheme_ids_are_exact(invalid):
    with pytest.raises(ValueError, match="Choose a feedback scheme"):
        FeedbackScheme.from_id(invalid)


def test_scheme_manifest_and_hash_are_detached_and_deterministic():
    schemes = [FeedbackScheme.from_id(scheme_id) for scheme_id in SCHEME_IDS]
    assert len({scheme.config_sha256 for scheme in schemes}) == 10
    scheme = schemes[1]
    manifest = scheme.config()
    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert digest == scheme.config_sha256 == FeedbackScheme("solved", 3).config_sha256
    manifest["cut_points"][0] = .8
    manifest["labels"][0] = "invented"
    assert scheme.config()["cut_points"] == [.33, .66]
    assert scheme.labels[0] == "unsolved"
    with pytest.raises(AttributeError):
        scheme.bins = 5


@pytest.mark.parametrize("scheme_id", SCHEME_IDS[1:])
def test_transform_preserves_schema_pass_skip_rounds_and_task_numbers_without_judgment_numbers(scheme_id):
    original = [{"round": 7, "calls": [{"name": "weather", "arguments": {"threshold": .5}}],
                 "feedback": feedback(.55, None, skipped=True)}]
    before = deepcopy(original)
    transformed = FeedbackScheme.from_id(scheme_id).transform_history(original)
    assert original == before
    assert transformed[0]["round"] == 7
    assert transformed[0]["calls"] == original[0]["calls"]
    checks = transformed[0]["feedback"]["checks"]
    assert checks["function_selection"]["pass"] is True
    assert checks["function_selection"]["schema_pass"] is True
    assert checks["arguments"] == {"skipped": True, "pass": None, "feedback_label": None, "schema_pass": True}
    assert transformed[0]["feedback"]["schema_errors"] == original[0]["feedback"]["schema_errors"]
    assert transformed[0]["feedback"]["failed_checks"] == original[0]["feedback"]["failed_checks"]
    assert checks["function_selection"]["feedback_label"] == FeedbackScheme.from_id(scheme_id).label(.55)
    semantic = json.dumps(transformed[0]["feedback"])
    assert "probability_yes" not in semantic and "threshold" not in semantic and "0.55" not in semantic
    assert "feedback labels are verifier judgments, not ground truth or explanations" in semantic


def test_scalar_legacy_probability_fields_and_nested_thresholds_cannot_leak():
    original = {"checks": {"function_selection": {"probability_yes": .66, "threshold": .5}},
                "function_probability": .66, "arguments_probability": None,
                "nested": {"function_threshold": .5, "arguments_threshold": .5, "p_f": .66, "p_a": None}}
    transformed = FeedbackScheme.from_id("solved_3").transform_feedback(original)
    assert transformed == {"checks": {"function_selection": {"feedback_label": "solved"}},
                           "function_label": "solved", "arguments_label": None,
                           "nested": {"function_label": "solved", "arguments_label": None}}


def test_conflicting_probability_fields_are_rejected_instead_of_silently_overwritten():
    scheme = FeedbackScheme.from_id("solved_3")
    for original in ({"probability_yes": .2, "feedback_label": "solved"},
                     {"p_f": .2, "function_probability": .8}):
        with pytest.raises(ValueError, match="conflicting judgment"):
            scheme.transform_feedback(original)


def test_numeric_scheme_is_deepcopy_with_no_feedback_or_prompt_changes():
    scheme = FeedbackScheme()
    assert scheme.scheme_id == "numeric" and scheme.cut_points == scheme.labels == ()
    history = [{"calls": [], "feedback": feedback()}]
    assert scheme.transform_history(None) is None
    assert scheme.transform_history([]) == []
    transformed = scheme.transform_history(history)
    assert transformed == history and transformed is not history
    transformed[0]["feedback"]["checks"]["function_selection"]["probability_yes"] = .9
    assert history[0]["feedback"]["checks"]["function_selection"]["probability_yes"] == .1
    assert scheme.label(.33) == .33


def _response(calls):
    native = [{"id": f"call_{index}", "type": "function", "function": {
        "name": call["name"], "arguments": json.dumps(call["arguments"]),
    }} for index, call in enumerate(calls)]
    return {"model": "gpt-5-nano", "choices": [{"finish_reason": "tool_calls" if native else "stop",
            "message": {"role": "assistant", "content": None if native else "No call is required.",
                        "tool_calls": native}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


@pytest.mark.parametrize("provider", ["frozen", "canonical"])
@pytest.mark.parametrize("scheme_id", ["numeric", "solved_3", "correctness_4", "action_5"])
def test_request_audit_matches_actual_wire_payload_and_precedes_send(example, provider, scheme_id):
    history = [{"round": 3, "calls": [], "feedback": feedback(.55, None, skipped=True)}]
    before = deepcopy(history)
    events, requests, audits = [], [], []

    def respond(request):
        events.append("send")
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_response([]))

    def audit(request):
        events.append("audit")
        audits.append(deepcopy(request))
        # Audit consumers cannot modify the body that is about to be sent.
        request["history"][0]["feedback"]["source"] = "mutated"
        request["payload"]["model"] = "mutated"

    client = httpx.Client(transport=httpx.MockTransport(respond))
    if provider == "frozen":
        base = FunctionCallingLLM("offline-key", model="gpt-5-nano", client=client,
                                  reasoning_effort="none", max_completion_tokens=4096)
        baseline = base._payload(example, history)
    else:
        base = OpenAI("gpt-5-nano", api_key="offline-key", client=client,
                      reasoning_effort="none", max_output_tokens=4096)
        baseline = base.payload(example["messages"], example["tools"], history)
    wrapper = FeedbackGenerator(base, scheme_id, before_generate=audit)
    proposal = wrapper.generate(example, history)
    assert proposal["calls"] == []
    assert events == ["audit", "send"]
    assert wrapper.last_request == audits[0]
    assert audits[0]["payload"] == requests[0]
    assert audits[0]["payload_sha256"] == hashlib.sha256(json.dumps(
        requests[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    wire = json.dumps(requests[0])
    assert "offline-key" not in wire and "ORACLE" not in wire and "private-benchmark-id" not in wire
    assert "simple_python" not in wire
    assert history == before
    if scheme_id == "numeric":
        assert requests[0] == baseline
    else:
        assert requests[0] != baseline
        assert "probability_yes" not in wire and '"threshold"' not in wire
        assert "0.55" not in wire
        # Task numbers in the unchanged original request must remain visible.
        assert "at 0.5 degrees" in wire
        assert "proposals NOT executed" in wire


def test_initial_request_is_identical_for_all_ten_arms(example):
    requests = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: (
        requests.append(json.loads(request.content)) or httpx.Response(200, json=_response([])))))
    base = FunctionCallingLLM("offline-key", model="gpt-5-nano", client=client)
    for scheme_id in SCHEME_IDS:
        FeedbackGenerator(base, scheme_id).generate(example, [])
    assert len(requests) == 10
    assert all(request == requests[0] for request in requests)


@pytest.mark.parametrize("method", ["jev_joint", "jev_joint_latest", "jev_function", "jev_arguments"])
def test_frozen_pipeline_retains_raw_verdicts_and_unchanged_gates_while_only_generator_sees_text(example, method):
    calls = [{"name": "weather", "arguments": {"city": "Paris"}}]
    generated, verdicts = [], []

    def generate_response(request):
        generated.append(json.loads(request.content))
        return httpx.Response(200, json=_response(calls))

    def verify_response(request):
        verdicts.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            key: {"type": "noul", "noul": .2} for key in verdicts[-1]["questions"]},
            "usage": {"input_tokens": 10, "output_tokens": 0}})

    base = FunctionCallingLLM("offline-key", model="gpt-5-nano",
                              client=httpx.Client(transport=httpx.MockTransport(generate_response)))
    verifier = JevVerifier("offline-key", function_threshold=.5, arguments_threshold=.5,
                           client=httpx.Client(transport=httpx.MockTransport(verify_response)))
    wrapper = FeedbackGenerator(base, "action_4")
    result = run_example(example, wrapper, verifier, method=method, max_repairs=3,
                         exhaustion_policy="abstain", initial_proposal={"calls": calls})
    assert result["repairs"] == 3 and result["exhausted"] is True and result["dispatched"] is False
    assert len(generated) == 3 and len(verdicts) == 4
    assert [step["gate_pass"] for step in result["trace"]] == [False] * 4
    for step in result["trace"]:
        assert step["verification"]["function_probability"] == .2
        assert step["verification"]["arguments_probability"] == .2
        raw_check = step["verification"]["feedback"]["checks"]["function_selection"]
        assert raw_check["probability_yes"] == .2 and raw_check["threshold"] == .5
        assert "feedback_label" not in json.dumps(step["feedback"])
    for body in generated:
        assert "probability_yes" not in json.dumps(body)
        assert '"threshold"' not in json.dumps(body)
    history = wrapper.last_request["history"]
    assert len(history) == (1 if method == "jev_joint_latest" else 3)
    if method == "jev_joint_latest":
        assert history[0]["round"] == 3
    if method == "jev_function":
        assert "arguments" not in history[-1]["feedback"]["checks"]
    elif method == "jev_arguments":
        assert "function_selection" not in history[-1]["feedback"]["checks"]


def test_invalid_feedback_never_reaches_underlying_generator_or_audit():
    class NeverGenerator:
        def generate(self, example, history):
            pytest.fail("Invalid judgment cannot be sent.")

    wrapper = FeedbackGenerator(NeverGenerator(), "solved_3",
                                before_generate=lambda audit: pytest.fail("No request should be audited."))
    with pytest.raises(ValueError, match="Feedback probabilities"):
        wrapper.generate({}, [{"calls": [], "feedback": feedback(float("nan"))}])
    assert wrapper.last_request is None


def test_rejected_no_call_keeps_null_argument_label_and_only_function_question(example):
    requests, audits = [], []
    calls = [{"name": "weather", "arguments": {"city": "Seoul"}}]

    def verify_response(request):
        body = json.loads(request.content)
        requests.append(body)
        probability = .2 if len(requests) == 1 else .9
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            key: {"type": "noul", "noul": probability} for key in body["questions"]},
            "usage": {"input_tokens": 10, "output_tokens": 0}})

    base = FunctionCallingLLM("offline-key", model="gpt-5-nano", client=httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_response(calls)))))
    verifier = JevVerifier("offline-key", function_threshold=.5, arguments_threshold=.5,
                           client=httpx.Client(transport=httpx.MockTransport(verify_response)))
    result = run_example(example, FeedbackGenerator(base, "solved_3", before_generate=audits.append),
                         verifier, max_repairs=1, initial_proposal={"calls": []}, exhaustion_policy="abstain")
    assert result["accepted"] is True and result["repairs"] == 1 and result["calls"] == calls
    assert set(requests[0]["questions"]) == {"function_correct"}
    assert set(requests[1]["questions"]) == {"function_correct", "arguments_correct"}
    raw = result["trace"][0]["verification"]
    assert raw["arguments_probability"] is None and raw["arguments_pass"] is None and raw["arguments_skipped"]
    transformed = audits[0]["history"][0]["feedback"]["checks"]
    assert transformed["function_selection"]["feedback_label"] == "unsolved"
    assert transformed["arguments"] == {"skipped": True, "pass": None, "feedback_label": None, "schema_pass": True}


def test_wrapper_never_hands_ids_categories_or_oracle_metadata_to_underlying_generator(example):
    class Generator:
        def generate(self, public, history):
            assert set(public) == {"messages", "tools"}
            assert public == {key: example[key] for key in ("messages", "tools")}
            public["messages"][0]["content"] = "changed by a custom generator"
            return {"calls": []}

    original = deepcopy(example)
    FeedbackGenerator(Generator()).generate(example, [])
    assert example == original


def test_simple_decorated_generator_has_explicit_unknown_payload_and_no_proposal_mutation():
    expected = {"calls": [], "usage": {"prompt_tokens": 12}, "custom": "unchanged"}

    class Generator:
        def generate(self, example, history):
            assert history[0]["feedback"]["checks"]["function_selection"]["feedback_label"] == "unsolved"
            return expected

    wrapper = FeedbackGenerator(Generator(), "solved_3")
    assert wrapper.generate({}, [{"calls": [], "feedback": feedback()}]) is expected
    assert wrapper.last_request["payload"] is None and wrapper.last_request["payload_sha256"] is None
