"""DMCall uses the paper's settings by default: Hybrid repair feedback and initial no-call preservation."""

from copy import deepcopy
import inspect

import pytest

import dmcall
import dmgfc
from dmcall.feedback import HybridFeedbackGenerator, HybridFeedbackScheme

TOOLS = [{"type": "function", "function": {"name": "weather", "description": "Weather.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}}]
CALL = [{"name": "weather", "arguments": {"city": "Seoul"}}]
WRONG = [{"name": "weather", "arguments": {"city": "Busan"}}]


def proposal(calls):
    return {"calls": deepcopy(calls), "content": None, "usage": {"input_tokens": 12, "output_tokens": 3},
            "latency_s": .1, "incomplete": None, "parse_errors": []}


class Model:
    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.histories = []
        self.closed = False

    def generate(self, example, history=None):
        self.histories.append(deepcopy(history or []))
        return deepcopy(self.proposals.pop(0))

    def close(self):
        self.closed = True


class Verifier:
    """Accepts only CALL; reports probabilities 0.12 (reject) or 0.95 (accept)."""

    def __init__(self):
        self.calls = []

    def verify(self, example, calls):
        self.calls.append(deepcopy(calls))
        good = calls == CALL
        p = .95 if good else .12
        feedback = {"failed_checks": [] if good else ["arguments"], "function_probability": .95,
                    "arguments_probability": p, "instruction": "Use these checks to repair the proposal."}
        return {"function_pass": True, "arguments_pass": good, "arguments_skipped": False,
                "function_probability": .95, "arguments_probability": p, "feedback": feedback,
                "latency_s": .02, "usage": {"input_tokens": 30, "output_tokens": 0}}


@pytest.mark.parametrize("p,label", [(0, "very likely incorrect"), (.19, "very likely incorrect"),
                                     (.2, "likely incorrect"), (.5, "uncertain"), (.6, "likely correct"),
                                     (.8, "very likely correct"), (1, "very likely correct"), (None, None)])
def test_hybrid_bands(p, label):
    assert HybridFeedbackScheme().label(p) == label


def test_hybrid_keeps_numbers_and_adds_labels():
    feedback = {"function_probability": .95, "arguments_probability": .12, "schema_errors": ["x"]}
    out = HybridFeedbackScheme().transform_feedback(feedback)
    assert out["function_probability"] == .95 and out["arguments_probability"] == .12
    assert out["function_label"] == "very likely correct" and out["arguments_label"] == "very likely incorrect"
    assert out["schema_errors"] == ["x"] and feedback == {"function_probability": .95,
                                                          "arguments_probability": .12, "schema_errors": ["x"]}


def test_dmcall_defaults_are_the_paper_settings():
    params = inspect.signature(dmcall.DMCall).parameters
    assert params["feedback"].default == "hybrid" and params["method"].default == "dmcall"
    assert inspect.signature(dmgfc.DMGFC).parameters["preserve_initial_no_call"].default is True
    with pytest.raises(ValueError):
        dmcall.DMCall(Model(), jev=Verifier(), tools=TOOLS, feedback="text")


def test_dmcall_repair_sees_hybrid_feedback_and_trace_keeps_numbers():
    model, verifier = Model(proposal(WRONG), proposal(CALL)), Verifier()
    agent = dmcall.DMCall(model, jev=verifier, tools=TOOLS)
    assert isinstance(agent.model, HybridFeedbackGenerator) and repr(agent.model) == repr(model)
    result = agent.call("Weather in Seoul?")
    assert result.calls == CALL and result.accepted and result.repairs == 1
    seen = model.histories[1][0]["feedback"]
    assert seen["arguments_probability"] == .12 and seen["arguments_label"] == "very likely incorrect"
    assert seen["function_label"] == "very likely correct"
    assert result.attempts[0]["arguments_probability"] == .12


def test_numeric_option_matches_dmgfc():
    histories = []
    for agent_cls, options in ((dmcall.DMCall, {"feedback": "numeric"}), (dmgfc.DMGFC, {})):
        model = Model(proposal(WRONG), proposal(CALL))
        agent_cls(model, jev=Verifier(), tools=TOOLS, **options).call("Weather in Seoul?")
        histories.append(model.histories)
    assert histories[0] == histories[1]
    assert "arguments_label" not in histories[0][1][0]["feedback"]


def test_initial_no_call_is_returned_without_a_check():
    model, verifier = Model(proposal([])), Verifier()
    result = dmcall.DMCall(model, jev=verifier, tools=TOOLS).call("Say hello.")
    assert result.calls == [] and result.initial_no_call_preserved and result.accepted is None
    assert verifier.calls == []


@pytest.mark.parametrize("method", ["direct", "schema", "self-refine"])
def test_methods_without_a_decision_model_keep_their_generator(method):
    model = Model(proposal(CALL))
    agent = dmcall.DMCall(model, jev=Verifier(), tools=TOOLS, method=method)
    assert agent.model is model


def test_close_reaches_the_wrapped_generator():
    model = Model()
    agent = dmcall.DMCall(model, jev=Verifier(), tools=TOOLS)
    agent.model.close()
    assert model.closed
