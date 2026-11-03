"""DM-Call public names share the dmgfc implementation and keep every earlier name working."""

import importlib
import inspect
import pickle
import sys

import pytest

import dmcall
import dmgfc
import dmfc
import jbfc
from dmcall import cli


def test_dmcall_exports_every_shared_name_and_the_dmcall_class():
    assert dmcall.DMCall.__name__ == "DMCall" and dmcall.DMCall.__module__ == "dmcall.core"
    assert issubclass(dmcall.DMCall, dmgfc.DMGFC)
    assert inspect.signature(dmcall.DMCall).parameters["method"].default == "dmcall"
    for name in dmgfc.__all__:
        if name != "METHODS":
            assert getattr(dmcall, name) is getattr(dmgfc, name)
    assert set(dmgfc.__all__) <= set(dmcall.__all__) and "DMCall" in dmcall.__all__


@pytest.mark.parametrize("module", ["providers", "tools", "jev", "_prompt", "perplexity"])
def test_dmcall_submodules_are_the_shared_modules(module):
    assert importlib.import_module(f"dmcall.{module}") is importlib.import_module(f"dmgfc.{module}")


@pytest.mark.parametrize("name,shared", [
    ("dmcall", "dmgfc"), ("DM-Call", "dmgfc"), ("dmcall-latest", "dmgfc-latest"), ("dmcall_l", "dmgfc-l"),
    ("dmcall-f", "dmgfc-f"), ("dmcall-a", "dmgfc-a"),
])
def test_dmcall_method_names_select_the_same_loops(name, shared):
    assert dmcall.core.public_method(name) == shared
    assert dmcall.METHODS[name.lower().replace("_", "-")] == dmgfc.METHODS[shared]
    agent = dmcall.DMCall(dmgfc.OpenAI("offline-model", api_key="offline-test-key"), method=name,
                          jev=dmgfc.Jev(api_key="offline-test-key"))
    assert agent.method == dmgfc.DMGFC(dmgfc.OpenAI("offline-model", api_key="offline-test-key"), method=shared,
                                       jev=dmgfc.Jev(api_key="offline-test-key")).method


@pytest.mark.parametrize("name", ["dmgfc", "dmgfc-f", "direct", "schema", "self-refine", "jbfc", "dmfc-latest"])
def test_earlier_method_names_still_work_on_dmcall(name):
    assert dmcall.METHODS[name] == dmgfc.METHODS[name]
    dmcall.DMCall(dmgfc.OpenAI("offline-model", api_key="offline-test-key"), method=name,
                  jev=dmgfc.Jev(api_key="offline-test-key"))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        dmcall.DMCall(dmgfc.OpenAI("offline-model", api_key="offline-test-key"), method="dmcall-x",
                      jev=dmgfc.Jev(api_key="offline-test-key"))


def test_dmcall_class_pickles_by_its_public_name():
    reference = pickle.dumps(dmcall.DMCall)
    assert b"dmcall.core" in reference and pickle.loads(reference) is dmcall.DMCall
    assert dmfc.DMFC is jbfc.JBFC is dmgfc.DMGFC


def test_cli_translates_dmcall_method_names_before_the_shared_entry_point(monkeypatch):
    seen = []
    monkeypatch.setattr(cli._shared, "main", lambda arguments: seen.append(list(arguments)) or 0)
    assert cli.main(["run", "--methods", "direct,dmcall,dmcall-f", "--", "--methods=dmcall-a"]) == 0
    assert cli.main(["run", "--methods=dmcall-latest"]) == 0
    assert cli.main(["report", "--methods", "dmcall"]) == 0
    assert seen == [["run", "--methods", "direct,dmgfc,dmgfc-f", "--", "--methods=dmcall-a"],
                    ["run", "--methods=dmgfc-latest"], ["report", "--methods", "dmcall"]]


def test_cli_help_uses_dmcall_names_without_model_requests(capsys):
    original = sys.argv
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--help"])
    assert exit_info.value.code == 0 and sys.argv is original
    output = capsys.readouterr().out
    assert "usage: dmcall-eval" in output and "DMGFC" not in output and "dmgfc-eval" not in output
    assert "dmcall" in output
