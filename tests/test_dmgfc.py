"""Canonical DMGFC names share the tested providers and frozen verification protocol."""

import importlib
import inspect
import json
import pickle
import sys
import tomllib
from pathlib import Path

import httpx
import pytest

import dmgfc
import dmfc
import jbfc
from dmgfc import cli
from jev_fc.jev import ARGUMENT_QUESTION, FUNCTION_QUESTION


def test_canonical_class_and_all_compatibility_modules_share_identity():
    assert dmgfc.DMGFC.__name__ == "DMGFC"
    assert dmgfc.DMGFC.__module__ == "dmgfc.core"
    assert inspect.signature(dmgfc.DMGFC).parameters["method"].default == "dmgfc"
    assert dmgfc.DMGFC is dmgfc.DMFC is dmgfc.JBFC is dmfc.DMFC is jbfc.JBFC
    for module in ("core", "providers", "tools", "jev", "_prompt", "perplexity"):
        implementation = importlib.import_module(f"dmgfc.{module}")
        for package in ("dmfc", "jbfc"):
            assert importlib.import_module(f"{package}.{module}") is implementation
    for name in dmgfc.__all__:
        assert getattr(dmgfc, name) is getattr(dmfc, name) is getattr(jbfc, name)


def test_legacy_provider_patching_and_errors_reach_canonical_code(monkeypatch):
    canonical = importlib.import_module("dmgfc.providers")
    old = importlib.import_module("jbfc.providers")
    captured = []
    monkeypatch.setattr(old.time, "sleep", captured.append)
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    with dmgfc.OpenAI("offline-model", api_key="offline-test-key", client=client, max_retries=1) as model:
        with pytest.raises(dmfc.ProviderError) as error:
            model.generate({"messages": [{"role": "user", "content": "Test."}], "tools": []})
    assert isinstance(error.value, jbfc.ProviderError)
    assert canonical.time.sleep is old.time.sleep and len(captured) == 1


@pytest.mark.parametrize("package,name,canonical", [
    ("dmfc", "DMFC", "DMGFC"), ("jbfc", "JBFC", "DMGFC"),
    ("dmfc", "Result", "Result"), ("jbfc", "Verdict", "Verdict"),
])
def test_historical_pickle_global_names_remain_loadable(package, name, canonical):
    # Protocol 0 GLOBAL reproduces class references saved by older releases.
    historical_reference = f"c{package}.core\n{name}\n.".encode()
    assert pickle.loads(historical_reference) is getattr(dmgfc, canonical)
    new_reference = pickle.dumps(getattr(dmgfc, canonical))
    assert b"dmgfc.core" in new_reference
    assert pickle.loads(new_reference) is getattr(dmgfc, canonical)


@pytest.mark.parametrize("attribute", ["__dmgfc_tool__", "__dmfc_tool__", "__jbfc_tool__"])
def test_all_tool_metadata_generations_still_work(attribute):
    def weather(city: str):
        return city

    definition = {"name": "legacy_weather", "description": "Existing tool.",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
    setattr(weather, attribute, definition)
    assert dmgfc.schema_of(weather)["function"] == definition
    assert dmfc.invoke(weather, {"city": "Seoul"}) == "Seoul"
    decorated = dmgfc.tool(weather, name="canonical_weather")
    assert decorated.__dmgfc_tool__ is decorated.__dmfc_tool__ is decorated.__jbfc_tool__


def test_real_jev_adapter_keeps_wire_questions_no_call_skip_and_model_identity():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {key: {"type": "noul", "noul": 0.9} for key in body["questions"]},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })

    client = httpx.Client(transport=httpx.MockTransport(respond))
    verifier = dmgfc.Jev("offline-test-key", client=client)
    example = {"messages": [{"role": "user", "content": "Weather in Seoul?"}], "tools": [{
        "type": "function", "function": {"name": "weather", "description": "Weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}},
    }]}
    verifier.verify(example, [{"name": "weather", "arguments": {"city": "Seoul"}}])
    no_call = verifier.verify(example, [])
    assert requests[0]["model"] == "jev-1.13.0"
    assert requests[0]["questions"]["function_correct"] == FUNCTION_QUESTION
    assert requests[0]["questions"]["arguments_correct"] == ARGUMENT_QUESTION
    assert set(requests[1]["questions"]) == {"function_correct"}
    assert no_call["arguments_skipped"] and no_call["arguments_probability"] is None


def test_canonical_cli_translates_only_explicit_methods_preserving_engine_defaults(monkeypatch):
    captured = []
    original = sys.argv
    monkeypatch.setattr(cli, "_engine_main", lambda: captured.append(list(sys.argv)) or 0)
    assert cli.main(["run", "--methods", "dmgfc,DMGFC_F,dmfc-a,jbfc-latest,direct"]) == 0
    assert captured[-1][1:] == ["run", "--methods", "jev_joint,jev_function,jev_arguments,jev_joint_latest,direct"]
    assert cli.main(["run", "--methods=dmgfc-latest,schema,self-refine"]) == 0
    assert captured[-1][1:] == ["run", "--methods=jev_joint_latest,schema_only,self_refine"]
    assert cli.main(["run"]) == 0 and captured[-1][1:] == ["run"]
    assert sys.argv is original


def test_unknown_canonical_cli_method_fails_before_engine_or_network(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_engine_main", lambda: pytest.fail("Engine must not run."))
    assert cli.main(["run", "--methods=dmgfc-unknown"]) == 1
    assert "choose from dmgfc, dmgfc-latest, dmgfc-f, dmgfc-a" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [
    ["report", "--", "--methods=dmgfc"],
    ["report", "--", "--methods", "dmgfc"],
    ["run", "--", "--methods=dmgfc"],
    ["run", "--methods", "dmgfc", "--", "--methods=dmgfc-f"],
])
def test_cli_preserves_method_like_paths_and_values_after_option_separator(arguments, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "_engine_main", lambda: captured.append(list(sys.argv)) or 0)
    assert cli.main(arguments) == 0
    expected = list(arguments)
    if expected[:3] == ["run", "--methods", "dmgfc"]:
        expected[2] = "jev_joint"
    assert captured == [["dmgfc-eval", *expected]]


def test_real_report_parser_receives_literal_method_named_directory(monkeypatch, capsys):
    frozen_cli = importlib.import_module("jev_fc.cli")
    directories = []
    monkeypatch.setattr(frozen_cli, "build_summary", lambda path: directories.append(path) or {
        "status": "isolated-test-report", "methods": {}})
    assert cli.main(["report", "--", "--methods=dmgfc"]) == 0
    assert directories == ["--methods=dmgfc"]
    assert json.loads(capsys.readouterr().out)["status"] == "isolated-test-report"


def test_help_after_option_separator_remains_literal(monkeypatch, capsys):
    captured = []

    def engine():
        captured.append(list(sys.argv))
        print("JBFC literal content")
        return 0

    monkeypatch.setattr(cli, "_engine_main", engine)
    assert cli.main(["report", "--", "--help"]) == 0
    assert captured == [["dmgfc-eval", "report", "--", "--help"]]
    assert capsys.readouterr().out == "JBFC literal content\n"


@pytest.mark.parametrize("arguments", [
    ["--help"], ["run", "--help"], ["run", "--methods", "unknown", "--help"],
    ["run", "--methods=unknown", "--help"],
])
def test_canonical_cli_help_discloses_public_names_without_network(arguments, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dmgfc-eval"])
    original = sys.argv
    monkeypatch.setattr(httpx.Client, "post", lambda *_args, **_kwargs: pytest.fail("Help cannot POST."))
    monkeypatch.setattr(httpx.Client, "get", lambda *_args, **_kwargs: pytest.fail("Help cannot GET."))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 0 and sys.argv is original
    output = capsys.readouterr().out
    assert "usage: dmgfc-eval" in output and "JBFC" not in output
    if arguments[0] == "run":
        assert "Public --methods names: dmgfc, dmgfc-latest, dmgfc-f, dmgfc-a" in output
        assert "saved records retain those IDs" in output


@pytest.mark.parametrize("arguments", [["run", "--methods"], ["run", "--methods", "--help"]])
def test_missing_method_value_defers_to_argparse(arguments, monkeypatch, capsys):
    monkeypatch.setattr(httpx.Client, "post", lambda *_args, **_kwargs: pytest.fail("Parser error cannot POST."))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 2
    output = capsys.readouterr()
    assert "argument --methods: expected one argument" in output.err
    assert "Unknown method" not in output.err


def test_module_entry_point_help_uses_canonical_command_name(monkeypatch, capsys):
    original = ["/package/dmgfc/__main__.py"]
    monkeypatch.setattr(sys, "argv", original)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0 and sys.argv is original
    output = capsys.readouterr().out
    assert "usage: dmgfc-eval" in output and "__main__.py" not in output


def test_new_distribution_packages_canonical_api_and_keeps_original_entry_points():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "dmgfc"
    assert metadata["project"]["scripts"] == {
        "dmgfc-eval": "dmgfc.cli:main", "dmfc-eval": "jev_fc.cli:main", "jev-fc": "jev_fc.cli:main",
    }
    assert set(metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]) == {
        "src/dmgfc", "src/dmfc", "src/jbfc", "src/jev_fc",
    }
