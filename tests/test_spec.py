"""The Python reference passes the language-neutral conformance files in ``spec/`` that every port must pass."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

import dmcall
from dmcall import feedback as fb
from dmcall import prompts

SPEC = Path(__file__).resolve().parents[1] / "spec"


def load(name):
    return json.loads((SPEC / "conformance" / name).read_text())


def test_prompts_file_matches_the_library():
    data = json.loads((SPEC / "prompts.json").read_text())
    assert data["incomplete_reasons"] == dmcall.providers.INCOMPLETE
    assert data["system_prompt"] == dmcall.SYSTEM_PROMPT
    assert data["history_header"] == prompts.HISTORY_HEADER and data["history_footer"] == prompts.HISTORY_FOOTER
    assert data["function_question"] == dmcall.FUNCTION_QUESTION and data["argument_question"] == dmcall.ARGUMENT_QUESTION
    assert data["repair_instruction"] == prompts.REPAIR_INSTRUCTION
    assert tuple(data["labels"]) == fb.LABELS and tuple(data["cut_points"]) == fb.CUT_POINTS


@pytest.mark.parametrize("case", load("json_text.json")["cases"], ids=lambda case: case["input"][:30])
def test_json_text(case):
    assert json.dumps(json.loads(case["input"]), ensure_ascii=False) == case["output"]


@pytest.mark.parametrize("case", load("schema_check.json")["cases"], ids=lambda case: case["name"])
def test_schema_check(case):
    assert dmcall.check_schema(case["tools"], case["calls"]) == case["expected"]


@pytest.mark.parametrize("case", load("schema_check.json")["configuration_errors"], ids=lambda case: case["name"])
def test_schema_configuration_errors(case):
    with pytest.raises(ValueError):
        dmcall.check_schema(case["tools"], [])


def test_feedback():
    data = load("feedback.json")
    for case in data["labels"]:
        assert dmcall.label(case["p"]) == case["label"]
    for bad in data["invalid_probabilities"]:
        with pytest.raises(ValueError):
            dmcall.label(bad)
    for case in data["represent"]:
        assert fb.represent(case["feedback"], case["representation"]) == case["expected"]
    for case in data["single_check"]:
        assert fb.single_check(case["feedback"], case["keep"]) == case["expected"]
    for conflict in data["conflicts"]["feedback"]:
        for representation in data["conflicts"]["representations"]:
            with pytest.raises(ValueError):
                fb.represent(conflict, representation)


def test_normalize_calls():
    for case in load("normalize_calls.json")["cases"]:
        assert dmcall.normalize_calls(deepcopy(case["input"])) == case["expected"]


@pytest.mark.parametrize("case", load("tools.json")["cases"], ids=lambda case: case["name"])
def test_tool_definitions(case):
    assert dmcall.schema_of(deepcopy(case["input"])) == {"type": "function", "function": case["expected"]}


@pytest.mark.parametrize("case", load("tools.json")["configuration_errors"], ids=lambda case: case["name"])
def test_tool_configuration_errors(case):
    with pytest.raises(ValueError):
        dmcall.schema_of(deepcopy(case["input"]))


PROVIDERS = {"openai": dmcall.OpenAI, "anthropic": dmcall.Anthropic, "gemini": dmcall.Gemini}


def test_provider_decoding():
    data = load("providers.json")
    for case in data["decode"]:
        provider = PROVIDERS[case["provider"]]("m", api_key="KEY",
                                               client=httpx.Client(transport=httpx.MockTransport(
                                                   lambda r, body=case["response"]: httpx.Response(200, json=body))))
        proposal = provider.generate(data["example"])
        assert {key: proposal[key] for key in case["expected"]} == case["expected"], case["name"]


@pytest.mark.parametrize("name", sorted(load("loop_edges.json")["scenarios"]))
def test_loop_edges(name):
    import sys
    sys.path.insert(0, str(SPEC))
    from tests_helpers import make_decision, make_generator, summary

    data = load("loop_edges.json")
    scenario = data["scenarios"][name]
    sent_g, sent_d = [], []
    served_g, served_d = list(scenario["generator_responses"]), list(scenario["decision_responses"])

    def g(request):
        sent_g.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json=served_g.pop(0))

    def d(request):
        sent_d.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json=served_d.pop(0))

    generator = make_generator(scenario["generator"], httpx.Client(transport=httpx.MockTransport(g)))
    decision = make_decision(scenario["decision_model"], httpx.Client(transport=httpx.MockTransport(d)))
    result = dmcall.DMCall(generator, decision_model=decision, tools=data["tools"],
                           **scenario["options"]).call(deepcopy(data["messages"]))
    assert sent_g == [{"url": r["url"], "body": r["body"]} for r in scenario["generator_requests"]]
    assert sent_d == [{"url": r["url"], "body": r["body"]} for r in scenario["decision_requests"]]
    assert summary(result) == scenario["result"]
