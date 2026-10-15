from copy import deepcopy

import pytest

from jev_fc.jev import JevAPIError
from jev_fc.pipeline import run_example


@pytest.fixture
def example():
    return {
        "id": "one", "category": "simple_python",
        "messages": [{"role": "user", "content": "Weather in Seoul"}],
        "tools": [{"type": "function", "function": {
            "name": "weather", "description": "Current weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }}],
        "ground_truth": "ORACLE", "original_function": "ORACLE", "name_map": "ORACLE",
    }


def proposal(city="Seoul"):
    return {"calls": [{"name": "weather", "arguments": {"city": city}}], "latency_s": 2.0,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


class Generator:
    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.seen = []

    def generate(self, example, history=None):
        self.seen.append((deepcopy(example), deepcopy(history or [])))
        assert "ground_truth" not in example
        assert "original_function" not in example
        assert "name_map" not in example
        return deepcopy(next(self.proposals))


class Verifier:
    def __init__(self, verdicts, execution_events=None):
        self.verdicts = iter(verdicts)
        self.seen = []
        self.execution_events = execution_events

    def verify(self, example, calls):
        self.seen.append((deepcopy(example), deepcopy(calls)))
        assert "ground_truth" not in example
        assert "name_map" not in example
        if self.execution_events is not None:
            assert self.execution_events == []
        function_pass, arguments_pass = next(self.verdicts)
        return {"function_pass": function_pass, "arguments_pass": arguments_pass,
                "function_probability": .95 if function_pass else .2,
                "arguments_probability": .95 if arguments_pass else .2,
                "feedback": {"function_pass": function_pass, "arguments_pass": arguments_pass},
                "latency_s": .5, "api_calls": 1, "usage": {"input_tokens": 300, "output_tokens": 0}}


def test_three_repairs_means_four_proposals_then_one_execution(example):
    generated = [proposal("Paris"), proposal("London"), proposal("Tokyo"), proposal("Seoul")]
    generator = Generator(generated)
    executions = []
    verifier = Verifier([(False, False)] * 4, executions)
    result = run_example(example, generator, verifier, max_repairs=3,
                         executor=lambda calls: executions.append(deepcopy(calls)))
    assert len(generator.seen) == len(verifier.seen) == 4
    assert result["repairs"] == 3
    assert result["exhausted"] is True
    assert result["accepted"] is False
    assert result["dispatched"] is True
    assert result["calls"] == generated[-1]["calls"]
    assert executions == [generated[-1]["calls"]]
    assert len(generator.seen[-1][1]) == 3
    assert result["logical_latency_s"] == 10


def test_joint_gate_must_pass_on_same_candidate(example):
    generator = Generator([proposal("Paris"), proposal("London"), proposal("Seoul")])
    verifier = Verifier([(True, False), (False, True), (True, True)])
    result = run_example(example, generator, verifier)
    assert result["repairs"] == 2
    assert [step["gate_pass"] for step in result["trace"]] == [False, False, True]
    assert result["accepted"] is True
    assert result["exhausted"] is False


def test_abstention_does_not_execute_failed_final_proposal(example):
    generator = Generator([proposal()])
    verifier = Verifier([(False, True)])
    executions = []
    result = run_example(example, generator, verifier, max_repairs=0, exhaustion_policy="abstain",
                         executor=executions.append)
    assert result["abstained"] is True
    assert result["dispatched"] is False
    assert executions == []


def test_shared_initial_avoids_new_generation_and_is_not_mutated(example):
    initial = proposal()
    before = deepcopy(initial)
    generator = Generator([])
    result = run_example(example, generator, Verifier([(True, True)]), initial_proposal=initial)
    assert generator.seen == []
    assert result["initial_proposal_shared"] is True
    assert result["logical_latency_s"] == 2.5
    result["calls"][0]["arguments"]["city"] = "mutated"
    assert initial == before


def test_schema_only_repairs_without_a_semantic_verifier(example):
    generator = Generator([proposal(42), proposal("Seoul")])
    result = run_example(example, generator, method="schema_only")
    assert result["accepted"] is True
    assert result["repairs"] == 1
    assert generator.seen[-1][1][0]["feedback"]["schema_errors"]


def test_direct_uses_one_proposal_despite_repair_budget(example):
    result = run_example(example, Generator([proposal()]), method="direct", max_repairs=3)
    assert result["repairs"] == 0
    assert result["accepted"] is None


def test_self_refine_uses_full_budget_without_verifier(example):
    generator = Generator([proposal()] * 4)
    result = run_example(example, generator, method="self_refine", max_repairs=3)
    assert result["repairs"] == 3
    assert result["accepted"] is None
    assert result["dispatched"] is True


def test_verifier_api_failure_never_turns_into_execution(example):
    class BrokenVerifier:
        def verify(self, *_):
            raise JevAPIError("Jev HTTP 401")

    executed = []
    with pytest.raises(JevAPIError):
        run_example(example, Generator([proposal()]), BrokenVerifier(), executor=executed.append)
    assert executed == []


@pytest.mark.parametrize("method", ["jev_function", "jev_arguments", "jev_joint"])
def test_semantic_ablations_share_structural_floor(example, method):
    result = run_example(example, Generator([proposal(42)]), Verifier([(True, True)]),
                         method=method, max_repairs=0, exhaustion_policy="abstain")
    assert result["accepted"] is False
    assert result["dispatched"] is False


def test_function_only_feedback_does_not_leak_other_semantic_check(example):
    generator = Generator([proposal(), proposal()])
    run_example(example, generator, Verifier([(False, True), (True, False)]),
                method="jev_function", max_repairs=1)
    feedback = generator.seen[-1][1][0]["feedback"]
    assert "arguments_probability" not in feedback
    assert "arguments_pass" not in feedback
    assert feedback["function_pass"] is False
