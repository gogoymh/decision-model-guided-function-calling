"""Default initial no-call policy and parity for every unaffected trajectory."""

import asyncio
from copy import deepcopy
import inspect
import json

import httpx
import pytest

import dmgfc
from dmgfc.initial_no_call import POLICY_VERSION, run_example
from jev_fc.pipeline import run_example as historical_run_example

EXAMPLE = {"messages": [{"role": "user", "content": "Weather in Seoul?"}], "tools": [{
    "type": "function", "function": {"name": "weather", "description": "Weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}},
}]}
CALL = [{"name": "weather", "arguments": {"city": "Seoul"}}]
MISSING = [{"name": "weather", "arguments": {}}]
VERIFIED_METHODS = ("jev_joint", "jev_function", "jev_arguments", "jev_joint_latest")


def proposal(calls, **options):
    return {"calls": deepcopy(calls), "content": "Please give more detail." if calls == [] else None,
            "usage": {"input_tokens": 12, "output_tokens": 3}, "latency_s": .1,
            "incomplete": None, "parse_errors": [], **options}


class Model:
    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.histories = []
        self.examples = []

    def generate(self, example, history=None):
        self.examples.append(deepcopy(example))
        self.histories.append(deepcopy(history or []))
        return deepcopy(self.proposals.pop(0))


class Verifier:
    def __init__(self, accept_empty=False):
        self.calls = []
        self.accept_empty = accept_empty

    def verify(self, example, calls):
        self.calls.append(deepcopy(calls))
        empty = calls == []
        good = self.accept_empty if empty else calls == CALL
        feedback = {"failed_checks": [] if good else ["function_selection", "arguments"],
                    "function_probability": .9 if good else .1,
                    "arguments_probability": None if empty else (.9 if good else .1),
                    "instruction": "Use these checks to repair the proposal."}
        return {"function_pass": good, "arguments_pass": None if empty else good,
                "arguments_skipped": empty,
                "function_probability": .9 if good else .1,
                "arguments_probability": None if empty else (.9 if good else .1),
                "feedback": feedback, "latency_s": .02,
                "usage": {"input_tokens": 30, "output_tokens": 0}}


@pytest.mark.parametrize("method", VERIFIED_METHODS)
def test_initial_empty_terminates_without_verifier_repair_or_callback(method):
    model, verifier = Model(proposal([])), Verifier()
    callbacks = []
    record = run_example(EXAMPLE, model, verifier, method=method,
                         executor=lambda calls: callbacks.append(calls))
    assert model.histories == [[]] and model.examples == [EXAMPLE]
    assert verifier.calls == [] and callbacks == []
    assert record["calls"] == [] and record["accepted"] is None
    assert record["repairs"] == 0 and not record["exhausted"]
    assert record["initial_no_call_preserved"] is True
    assert record["initial_no_call_policy"] == POLICY_VERSION
    assert record["execution_result"] is None and record["executor_invoked"] is False
    assert record["logical_latency_s"] == .1
    step = record["trace"][0]
    assert step["gate_pass"] is None and step["policy_terminated"] is True
    assert step["verification"]["api_calls"] == 0
    assert step["verification"]["function_probability"] is None
    assert step["verification"]["arguments_probability"] is None
    assert step["feedback"] == {}


def test_public_default_result_exposes_bypass_instead_of_verifier_acceptance():
    model, verifier = Model(proposal([])), Verifier()
    agent = dmgfc.DMGFC(model, jev=verifier, tools=EXAMPLE["tools"])
    result = agent.run("Weather in Seoul?", execute_unverified=True)
    assert inspect.signature(dmgfc.DMGFC).parameters["preserve_initial_no_call"].default is True
    assert result.calls == [] and result.text == "Please give more detail."
    assert result.initial_no_call_preserved and result.accepted is None
    assert result.repairs == 0 and result.final_attempt == 0 and not result.exhausted
    assert result.output is None and not result.executed
    assert result.attempts[0]["accepted"] is None
    assert result.probabilities == {"function": None, "arguments": None}
    assert result.usage == {"model": {"input_tokens": 12, "output_tokens": 3},
                            "jev": {"input_tokens": 0, "output_tokens": 0}}
    assert not verifier.calls and len(model.histories) == 1


@pytest.mark.parametrize("method", VERIFIED_METHODS)
def test_nonempty_and_later_empty_trace_matches_the_historical_engine(method):
    candidates = (proposal(MISSING), proposal([]), proposal(CALL))
    old_model, new_model = Model(*candidates), Model(*candidates)
    old_verifier, new_verifier = Verifier(), Verifier()
    old = historical_run_example(EXAMPLE, old_model, old_verifier, method=method)
    new = run_example(EXAMPLE, new_model, new_verifier, method=method)
    assert new["trace"] == old["trace"]
    assert new["calls"] == old["calls"] and new["accepted"] == old["accepted"]
    assert new["repairs"] == old["repairs"]
    assert new["logical_latency_s"] == old["logical_latency_s"]
    assert new["initial_proposal_shared"] == old["initial_proposal_shared"] is False
    assert new_model.histories == old_model.histories and new_model.examples == old_model.examples
    assert new_verifier.calls == old_verifier.calls and [] in new_verifier.calls
    assert not new["initial_no_call_preserved"]


@pytest.mark.parametrize("bad", [
    proposal(None, incomplete="length", parse_errors=["truncated"]),
    proposal([{"name": "weather", "arguments": None}], parse_errors=["invalid JSON"]),
    proposal([], incomplete="refusal"),
    proposal([], parse_errors=["invalid tool response"]),
    proposal({}),
    proposal(()),
    proposal(""),
])
def test_unreadable_malformed_and_invalid_empty_candidates_are_not_preserved(bad):
    model, verifier = Model(bad, proposal(CALL)), Verifier()
    record = run_example(EXAMPLE, model, verifier)
    assert not record["initial_no_call_preserved"]
    assert len(model.histories) == 2 and len(verifier.calls) == 2
    assert record["trace"][0]["proposal"] == bad
    assert record["calls"] == CALL and record["accepted"] is True


@pytest.mark.parametrize("method", VERIFIED_METHODS)
def test_optout_reuses_exact_historical_path_and_recovers_initial_empty(method):
    model, verifier = Model(proposal([]), proposal(CALL)), Verifier()
    record = run_example(EXAMPLE, model, verifier, method=method, preserve_initial_no_call=False)
    assert "initial_no_call_preserved" not in record
    assert verifier.calls[0] == []
    if method == "jev_arguments":
        assert record["calls"] == [] and record["repairs"] == 0
    else:
        assert record["calls"] == CALL and record["repairs"] == 1
    public_model, public_verifier = Model(proposal([]), proposal(CALL)), Verifier()
    result = dmgfc.DMGFC(public_model, jev=public_verifier, tools=EXAMPLE["tools"], method=method,
                         preserve_initial_no_call=False).call("Weather in Seoul?")
    assert result.calls == record["calls"] and result.accepted == record["accepted"]
    assert not result.initial_no_call_preserved


@pytest.mark.parametrize("method", ["direct", "schema_only", "self_refine"])
def test_no_verifier_controls_retain_their_historical_behavior(method):
    candidates = [proposal([]), proposal(CALL), proposal(CALL), proposal(CALL)]
    old_model, new_model = Model(*candidates), Model(*candidates)
    old = historical_run_example(EXAMPLE, old_model, method=method)
    new = run_example(EXAMPLE, new_model, method=method)
    assert new["trace"] == old["trace"] and new["calls"] == old["calls"]
    assert old_model.histories == new_model.histories
    assert "initial_no_call_preserved" not in new


def test_explicit_verify_still_judges_empty_calls():
    verifier = Verifier(accept_empty=True)
    agent = dmgfc.DMGFC(Model(), jev=verifier, tools=EXAMPLE["tools"])
    verdict = agent.verify("Tell me a joke.", [])
    assert verdict.accepted and verifier.calls == [[]]


def test_async_default_preserves_initial_empty_without_verifier():
    model, verifier = Model(proposal([]), proposal([])), Verifier()
    agent = dmgfc.DMGFC(model, jev=verifier, tools=EXAMPLE["tools"])

    async def run_both():
        return await agent.acall("Tell me a joke."), await agent.arun("Tell me a joke.")

    called, ran = asyncio.run(run_both())
    assert called.initial_no_call_preserved and ran.initial_no_call_preserved
    assert not ran.executed and verifier.calls == [] and len(model.histories) == 2


def test_cached_empty_candidate_is_not_regenerated_or_mutated():
    initial = proposal([])
    original = deepcopy(initial)
    model, verifier = Model(), Verifier()
    record = run_example(EXAMPLE, model, verifier, initial_proposal=initial)
    assert not model.histories and not verifier.calls
    assert record["initial_proposal_shared"] is True and initial == original
    record["trace"][0]["proposal"]["content"] = "changed"
    assert initial == original


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_policy_option_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="preserve_initial_no_call"):
        dmgfc.DMGFC(Model(), jev=Verifier(), preserve_initial_no_call=value)


def test_wall_time_includes_initial_generation(monkeypatch):
    ticks = iter([10.0, 12.0])
    monkeypatch.setattr("dmgfc.initial_no_call.time.perf_counter", lambda: next(ticks))
    record = run_example(EXAMPLE, Model(proposal([])), Verifier())
    assert record["wall_time_s"] == 2.0


def test_real_provider_empty_output_makes_one_generator_post_and_zero_jev_posts():
    generated, checked = [], []

    def generate(request):
        generated.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "offline-model", "choices": [{
            "message": {"role": "assistant", "content": "Please specify a location."},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 4}})

    def verify(request):
        checked.append(request)
        pytest.fail("The initial no-call must not send a decision-model request.")

    with httpx.Client(transport=httpx.MockTransport(generate)) as gen_client:
        with httpx.Client(transport=httpx.MockTransport(verify)) as jev_client:
            model = dmgfc.OpenAI("offline-model", api_key="offline-test-key", client=gen_client)
            verifier = dmgfc.Jev("offline-test-key", client=jev_client)
            result = dmgfc.DMGFC(model, jev=verifier, tools=EXAMPLE["tools"]).call("Weather?")
    assert len(generated) == 1 and checked == []
    assert result.calls == [] and result.initial_no_call_preserved
    assert result.text == "Please specify a location."
    assert result.usage["model"] == {"input_tokens": 20, "output_tokens": 4}


def test_native_decider_initial_empty_makes_no_post_or_provenance_get():
    checked = []

    def verify(request):
        checked.append(request)
        pytest.fail("No native control or inference request is needed for initial no-call.")

    with httpx.Client(transport=httpx.MockTransport(verify)) as native_client:
        verifier = dmgfc.PerplexityDecider(client=native_client)
        result = dmgfc.DMGFC(Model(proposal([])), jev=verifier, tools=EXAMPLE["tools"]).call("Weather?")
    assert checked == [] and result.initial_no_call_preserved
    assert result.accepted is None and result.probabilities == {"function": None, "arguments": None}


def test_nonempty_real_provider_and_jev_wire_payloads_match_optout_exactly():
    def execute(preserve):
        generations, judgments = [], []
        candidates = iter([MISSING, [], CALL])

        def generate(request):
            generations.append(json.loads(request.content))
            calls = next(candidates)
            message = {"role": "assistant", "content": "Checking."}
            if calls:
                message["tool_calls"] = [{"id": "call-1", "type": "function", "function": {
                    "name": calls[0]["name"], "arguments": json.dumps(calls[0]["arguments"])}}]
            return httpx.Response(200, json={"model": "offline-model", "choices": [{
                "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4}})

        def verify(request):
            body = json.loads(request.content)
            judgments.append(body)
            good = body["state"]["proposed_calls"] == CALL
            return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
                key: {"type": "noul", "noul": .9 if good else .1} for key in body["questions"]},
                "usage": {"input_tokens": 30, "output_tokens": 0}})

        with httpx.Client(transport=httpx.MockTransport(generate)) as gen_client:
            with httpx.Client(transport=httpx.MockTransport(verify)) as jev_client:
                model = dmgfc.OpenAI("offline-model", api_key="offline-test-key", client=gen_client)
                verifier = dmgfc.Jev("offline-test-key", client=jev_client)
                result = dmgfc.DMGFC(model, jev=verifier, tools=EXAMPLE["tools"],
                                     preserve_initial_no_call=preserve).call("Weather in Seoul?")
        return result, generations, judgments

    old, old_generations, old_judgments = execute(False)
    new, new_generations, new_judgments = execute(True)
    assert new_generations == old_generations and new_judgments == old_judgments
    assert len(new_generations) == len(new_judgments) == 3
    assert new_judgments[1]["state"]["proposed_calls"] == []
    assert new.calls == old.calls == CALL and new.repairs == old.repairs == 2
    assert new.usage == old.usage and new.probabilities == old.probabilities
    assert not new.initial_no_call_preserved
