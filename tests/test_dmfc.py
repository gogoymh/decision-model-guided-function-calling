"""Compatibility of the former DMFC and JBFC public APIs with canonical DMGFC."""

import importlib
import json
from pathlib import Path
import tomllib

import pytest

import dmgfc
import dmfc
import jbfc
from jev_fc.llm import SYSTEM_PROMPT
from jev_fc.pipeline import run_example


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}
WRONG = [{"name": "get_weather", "arguments": {"city": "Busan"}}]
RIGHT = [{"name": "get_weather", "arguments": {"city": "Seoul"}}]


class Model:
    def __init__(self, *calls):
        self.calls = list(calls)
        self.histories = []

    def generate(self, example, history=None):
        self.histories.append(json.loads(json.dumps(history or [])))
        return {"calls": self.calls.pop(0), "content": None, "usage": {}, "latency_s": 0.0}


class DecisionModel:
    """A local scripted verifier; no additional production adapter or API calls."""

    def verify(self, example, calls, **kwargs):
        correct = calls == RIGHT
        skipped = not calls
        return {
            "function_pass": bool(calls), "arguments_pass": correct if not skipped else None,
            "function_probability": 0.9 if calls else 0.1,
            "arguments_probability": (0.9 if correct else 0.1) if not skipped else None,
            "arguments_skipped": skipped,
            "feedback": {"failed_checks": [] if correct else ["arguments" if calls else "function_selection"]},
        }


def test_public_types_and_prompts_are_shared_with_legacy_imports():
    assert dmfc.DMFC.__name__ == "DMGFC"
    assert dmfc.DMFC.__module__ == "dmgfc.core"
    assert jbfc.JBFC is jbfc.DMFC is dmfc.DMFC is dmgfc.DMGFC
    assert dmfc.SYSTEM_PROMPT is jbfc.SYSTEM_PROMPT is SYSTEM_PROMPT
    for name in ("Jev", "Result", "Verdict", "OpenAI", "Anthropic", "Gemini", "Provider", "tool", "verify"):
        assert getattr(dmfc, name) is getattr(jbfc, name)
    for module in ("core", "providers", "tools", "jev", "_prompt"):
        assert importlib.import_module(f"jbfc.{module}") is importlib.import_module(f"dmfc.{module}")
    assert importlib.import_module("jbfc.core").JBFC is dmfc.DMFC


@pytest.mark.parametrize("canonical, legacy, internal", [
    ("dmgfc", "jbfc", "jev_joint"),
    ("dmgfc-latest", "jbfc-latest", "jev_joint_latest"),
    ("dmgfc-f", "jev-function", "jev_function"),
    ("dmgfc-a", "jev-arguments", "jev_arguments"),
])
def test_method_aliases_keep_frozen_ids_and_return_canonical_names(canonical, legacy, internal):
    for name in (canonical, canonical.upper(), canonical.replace("-", "_"),
                 canonical.replace("dmgfc", "dmfc"), legacy, internal):
        agent = dmfc.DMFC(Model(RIGHT), jev=DecisionModel(), tools=[WEATHER], method=name)
        assert agent.method == internal
        assert agent.call("Weather in Seoul?").method == canonical
        assert repr(agent).startswith("DMGFC(") and f"method={canonical!r}" in repr(agent)


def test_short_latest_alias_and_legacy_class_use_the_same_history_rules():
    cumulative_model = Model(WRONG, [], RIGHT)
    cumulative = dmfc.DMFC(cumulative_model, jev=DecisionModel(), tools=[WEATHER]).call("Weather in Seoul?")
    latest_model = Model(WRONG, [], RIGHT)
    latest = jbfc.JBFC(latest_model, jev=DecisionModel(), tools=[WEATHER], method="dmfc-l").call("Weather in Seoul?")
    assert cumulative.calls == latest.calls == RIGHT
    assert cumulative.repairs == latest.repairs == 2
    assert [len(history) for history in cumulative_model.histories] == [0, 1, 2]
    assert [len(history) for history in latest_model.histories] == [0, 1, 1]
    assert latest_model.histories[-1][0]["round"] == 2
    assert cumulative.method == "dmgfc" and latest.method == "dmgfc-latest"


def test_canonical_default_preserves_the_frozen_loop_and_first_proposal_rule():
    model, verifier = Model(WRONG, RIGHT), DecisionModel()
    agent = dmfc.DMFC(model, jev=verifier, tools=[WEATHER])
    result = agent.call("Weather in Seoul?")
    example = {"messages": [{"role": "user", "content": "Weather in Seoul?"}], "tools": [WEATHER]}
    frozen = run_example(example, Model(WRONG, RIGHT), verifier, method="jev_joint", max_repairs=3,
                         exhaustion_policy="abstain")
    assert result.calls == frozen["calls"] == RIGHT and result.accepted == frozen["accepted"]
    assert result.repairs == frozen["repairs"] == 1
    for field in ("proposal", "verification", "gate_pass", "schema"):
        assert [step.get(field) for step in result.trace] == [step.get(field) for step in frozen["trace"]]
    assert agent.max_repairs == 3 and agent.on_exhaustion == "first"
    exhausted = dmfc.DMFC(Model(WRONG, []), jev=verifier, tools=[WEATHER], max_repairs=1).call("Weather?")
    assert exhausted.exhausted and exhausted.final_attempt == 0 and exhausted.calls == WRONG


def test_default_adapter_remains_jev_with_the_paper_threshold(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    with dmfc.DMFC(Model()) as agent:
        assert isinstance(agent.jev, dmfc.Jev)
        assert agent.jev.function_threshold == agent.jev.arguments_threshold == 0.5
        assert agent.jev.model == "jev-1.13.0"


def test_new_and_old_tool_decorators_preserve_metadata_in_both_directions():
    @dmfc.tool(name="new_weather")
    def new(city: str):
        return city

    @jbfc.tool(name="old_weather")
    def old(city: str):
        return city

    def legacy_only(city: str):
        return city

    legacy_only.__jbfc_tool__ = {
        "name": "legacy_weather", "description": "Old decorated function.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    for function, name in ((new, "new_weather"), (old, "old_weather"), (legacy_only, "legacy_weather")):
        assert dmfc.schema_of(function) == jbfc.schema_of(function)
        assert dmfc.schema_of(function)["function"]["name"] == name
        assert dmfc.invoke(function, {"city": "Seoul"}) == "Seoul"
    assert new.__dmfc_tool__ == new.__jbfc_tool__
    assert old.__dmfc_tool__ == old.__jbfc_tool__


def test_metadata_builds_both_imports_and_keeps_the_old_cli_alias():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "dmgfc"
    assert metadata["project"]["scripts"]["dmfc-eval"] == metadata["project"]["scripts"]["jev-fc"]
    assert set(metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]) == {
        "src/dmgfc", "src/dmfc", "src/jbfc", "src/jev_fc",
    }
    locked = tomllib.loads((root / "uv.lock").read_text())
    editable = [package for package in locked["package"] if package.get("source") == {"editable": "."}]
    assert len(editable) == 1 and editable[0]["name"] == "dmgfc"
