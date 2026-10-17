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
        arguments_skipped = calls == []
        arguments_pass = None if arguments_skipped else arguments_pass
        return {"function_pass": function_pass, "arguments_pass": arguments_pass,
                "arguments_skipped": arguments_skipped,
                "function_probability": .95 if function_pass else .2,
                "arguments_probability": None if arguments_skipped else .95 if arguments_pass else .2,
                "feedback": {"function_pass": function_pass, "arguments_pass": arguments_pass,
                             "arguments_skipped": arguments_skipped},
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


@pytest.mark.parametrize("method", ["jev_joint", "jev_joint_latest"])
def test_joint_gate_must_pass_on_same_candidate(example, method):
    generator = Generator([proposal("Paris"), proposal("London"), proposal("Seoul")])
    verifier = Verifier([(True, False), (False, True), (True, True)])
    result = run_example(example, generator, verifier, method=method)
    assert result["repairs"] == 2
    assert [step["gate_pass"] for step in result["trace"]] == [False, False, True]
    assert result["accepted"] is True
    assert result["exhausted"] is False


@pytest.mark.parametrize("method", ["jev_function", "jev_arguments", "jev_joint", "jev_joint_latest"])
@pytest.mark.parametrize("function_pass", [True, False])
def test_no_call_skips_argument_gate_without_changing_ablation_meaning(example, method, function_pass):
    result = run_example(example, Generator([{"calls": [], "latency_s": 2}]),
                         Verifier([(function_pass, None)]), method=method, max_repairs=0,
                         exhaustion_policy="abstain")
    expected = True if method == "jev_arguments" else function_pass
    assert result["accepted"] is expected
    assert result["dispatched"] is expected
    check = result["trace"][0]["verification"]
    assert check["arguments_skipped"] is True
    assert check["arguments_pass"] is None and check["arguments_probability"] is None
    if method == "jev_arguments":
        assert "function_pass" not in result["trace"][0]["feedback"]
        assert result["trace"][0]["feedback"]["arguments_skipped"] is True


def test_rejected_no_call_can_be_repaired_to_call_with_argument_check(example):
    generator = Generator([{"calls": [], "latency_s": 2}, proposal()])
    result = run_example(example, generator, Verifier([(False, None), (True, True)]), max_repairs=1)
    assert result["accepted"] is True and result["repairs"] == 1
    assert [step["verification"]["arguments_skipped"] for step in result["trace"]] == [True, False]
    assert generator.seen[1][1][0]["feedback"]["arguments_pass"] is None


def test_repair_to_no_call_accepts_without_a_fictional_argument_pass(example):
    result = run_example(example, Generator([proposal(), {"calls": [], "latency_s": 2}]),
                         Verifier([(False, True), (True, None)]), max_repairs=1)
    assert result["accepted"] is True and result["calls"] == []
    assert result["trace"][-1]["verification"]["arguments_pass"] is None


@pytest.mark.parametrize("method", ["jev_function", "jev_arguments", "jev_joint", "jev_joint_latest"])
def test_empty_argument_object_is_not_an_empty_call_collection(example, method):
    candidate = {"calls": [{"name": "weather", "arguments": {}}]}
    result = run_example(example, Generator([candidate]), Verifier([(True, True)]),
                         method=method, max_repairs=0)
    assert result["accepted"] is False
    assert result["trace"][0]["verification"]["arguments_skipped"] is False
    assert result["trace"][0]["schema"]["arguments_pass"] is False


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


@pytest.mark.parametrize("method", ["jev_function", "jev_arguments", "jev_joint", "jev_joint_latest"])
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


@pytest.mark.parametrize("budget", [0, 4, 5, 8])
def test_arbitrary_nonnegative_budget_keeps_one_final_dispatch(example, budget):
    generator = Generator([proposal()] * (budget + 1))
    executions = []
    verifier = Verifier([(False, False)] * (budget + 1), executions)
    result = run_example(example, generator, verifier, max_repairs=budget, executor=executions.append)
    assert len(generator.seen) == len(verifier.seen) == budget + 1
    assert result["repairs"] == budget and result["exhausted"]
    assert executions == [result["calls"]]


def test_large_budget_still_stops_at_first_joint_pass(example):
    generator = Generator([proposal(), proposal()])
    executions = []
    result = run_example(example, generator, Verifier([(False, True), (True, True)], executions),
                         max_repairs=20, executor=executions.append)
    assert result["repairs"] == 1 and not result["exhausted"]
    assert len(generator.seen) == 2 and executions == [result["calls"]]


@pytest.mark.parametrize("budget", [-1, True, False, 1.0, 2.5, None, "3", float("inf"), float("nan")])
def test_invalid_budget_is_rejected_before_generation(example, budget):
    generator = Generator([])
    with pytest.raises(ValueError, match="nonnegative integer"):
        run_example(example, generator, Verifier([]), max_repairs=budget)
    assert generator.seen == []


@pytest.mark.parametrize("method", ["jev_joint", "jev_joint_latest"])
@pytest.mark.parametrize("policy", ["execute_last", "abstain"])
def test_latest_history_limits_prompt_only_and_keeps_shared_initial_full_trace_and_policy(example, method, policy):
    initial = proposal("Paris")
    before = deepcopy(initial)
    generator = Generator([proposal("London"), proposal("Seoul")])
    executions = []
    verifier = Verifier([(True, False), (False, True), (True, False)], executions)
    result = run_example(example, generator, verifier, method=method, initial_proposal=initial,
                         max_repairs=2, exhaustion_policy=policy, executor=executions.append)
    assert result["initial_proposal_shared"] is True and initial == before
    assert len(generator.seen) == 2 and len(verifier.seen) == len(result["trace"]) == 3
    assert result["repairs"] == 2 and result["exhausted"] and not result["accepted"]
    assert result["dispatched"] is (policy == "execute_last")
    assert executions == ([result["calls"]] if policy == "execute_last" else [])
    assert [step["attempt"] for step in result["trace"]] == [0, 1, 2]
    assert result["logical_latency_s"] == 7.5
    for attempt, (public, history) in enumerate(generator.seen, 1):
        assert public["messages"] == example["messages"] and public["tools"] == example["tools"]
        expected = [{"calls": result["trace"][i]["proposal"]["calls"],
                     "feedback": result["trace"][i]["feedback"]} for i in range(attempt)]
        if method == "jev_joint_latest":
            expected = [{**expected[-1], "round": attempt}]
        assert history == expected
