"""Offline suite orchestration: development selection must precede test use."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from jev_fc.pipeline import METHODS


_spec = importlib.util.spec_from_file_location("run_suite", Path(__file__).parents[1] / "scripts/run_suite.py")
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)


def option(command, name):
    return command[command.index(name) + 1]


@pytest.fixture
def suite_run(tmp_path, monkeypatch):
    manifest = {
        "split": "test", "limit": None, "split_seed": 1729,
        "methods": list(METHODS), "replicates": 3, "max_repairs": 2,
        "threshold": .8, "exhaustion_policy": "execute_last",
        "llm_model": "gpt-5.4-mini-2026-03-17", "jev_model": "jev-1.13.0",
        "reasoning_effort": "none", "temperature": None, "max_completion_tokens": 4096,
        "llm_provider": "openai", "llm_api_style": "responses", "llm_gateway": "apim",
        "jev_gateway": "apim", "synthetic": False,
    }
    selection = {"kind": "repair_budget_selection", "selection_id": "mock-dev-selection",
                 "selected_max_repairs": 2, "grid": [0, 1, 2, 3, 4, 5],
                 "selection_method": "jev_joint",
                 "selection_rule": "max_dev_accuracy_then_smallest_budget",
                 "provenance": {"split": "dev"}, "candidates": []}
    observed = {"commands": [], "replays": [], "events": [], "paper_eligible": True}
    paper = tmp_path / "paper/main.tex"
    paper.parent.mkdir()
    paper.write_text("existing manuscript\n")
    monkeypatch.setattr(suite, "__file__", str(tmp_path / "scripts/run_suite.py"))

    def run(command, *, cwd, check):
        assert cwd == tmp_path and check is True
        observed["commands"].append(command)
        if command[3] == "run":
            split = option(command, "--split")
            observed["events"].append("run_" + split)
            directory = Path(option(command, "--output"))
            directory.mkdir(parents=True, exist_ok=True)
            saved = {**manifest, "split": split, "methods": option(command, "--methods").split(","),
                     "max_repairs": int(option(command, "--max-repairs"))}
            if split == "test":
                assert "select_dev" in observed["events"]
                artifact = json.loads(Path(option(command, "--repair-budget-selection")).read_text())
                assert artifact == selection
                saved["repair_budget_selection"] = artifact
            (directory / "manifest.json").write_text(json.dumps(saved))
        else:
            assert command[3] == "report"
            observed["events"].append("report")
            if "--paper" in command:
                # The real report command checks artifact/canonical-data validity;
                # these tests replace subprocesses, never bypass production guards.
                paper.write_text("updated manuscript tables\n")
        return subprocess.CompletedProcess(command, 0)

    def select(source, grid):
        assert observed["events"] == ["run_dev"]
        assert Path(source).name == "dev_budget"
        observed["events"].append("select_dev")
        observed["selection_source"] = source
        observed["grid"] = tuple(grid)
        selection["grid"] = list(grid)
        return selection

    def derive(source, target, **kwargs):
        assert "run_test" in observed["events"]
        observed["events"].append("test_diagnostic")
        observed["replays"].append((source, target, kwargs))

    def eligibility(saved):
        assert saved["repair_budget_selection"] == selection
        observed["eligibility_manifest"] = saved
        return observed["paper_eligible"]

    monkeypatch.setattr(suite.subprocess, "run", run)
    monkeypatch.setattr(suite, "select_repair_budget", select)
    monkeypatch.setattr(suite, "derive_policy_run", derive)
    monkeypatch.setattr(suite, "paper_protocol_matches", eligibility)
    return manifest, selection, observed, paper


def test_default_suite_selects_on_dev_before_all_test_methods_and_paper(suite_run, capsys):
    _, selection, observed, paper = suite_run
    assert suite.main([]) == 0
    assert observed["events"][:3] == ["run_dev", "select_dev", "run_test"]
    development, test, report = observed["commands"]
    assert option(development, "--methods") == "jev_joint"
    assert option(development, "--max-repairs") == "5"
    assert "--limit" not in development
    assert option(test, "--methods").split(",") == list(METHODS)
    assert len(METHODS) == 6
    assert 303 * int(option(development, "--replicates")) == 909
    assert 2748 * int(option(test, "--replicates")) * len(METHODS) == 49464
    assert int(option(test, "--max-repairs")) == selection["selected_max_repairs"]
    assert Path(option(development, "--output")).parts[-3:] == ("results", "tuned", "dev_budget")
    assert Path(option(test, "--output")).name == "test_selected"
    assert [entry[2] for entry in observed["replays"]] == [{"max_repairs": 0}, {"max_repairs": 1}]
    assert all(entry[1].parent.name == "test_prefix_diagnostics" for entry in observed["replays"])
    assert report[-2:] == ["--paper", "paper/main.tex"]
    assert paper.read_text() == "updated manuscript tables\n"
    assert "Development selection chose r=2" in capsys.readouterr().out


@pytest.mark.parametrize("selected_budget", [0, 4, 5])
def test_selected_budget_including_zero_and_above_three_controls_all_test_work(suite_run, selected_budget):
    _, selection, observed, _ = suite_run
    selection["selected_max_repairs"] = selected_budget
    assert suite.main(["--include-abstain"]) == 0
    test = observed["commands"][1]
    assert int(option(test, "--max-repairs")) == selected_budget
    prefixes = observed["replays"][:-1]
    assert [item[2]["max_repairs"] for item in prefixes] == list(range(selected_budget))
    assert all(item[2]["max_repairs"] < selected_budget for item in prefixes)
    assert observed["replays"][-1][2] == {"max_repairs": selected_budget, "exhaustion_policy": "abstain"}


@pytest.mark.parametrize("flag", ["--main-only", "--dev-only"])
def test_development_selection_is_never_skipped(suite_run, flag):
    _, _, observed, _ = suite_run
    assert suite.main([flag, "--skip-paper"]) == 0
    assert observed["events"][:2] == ["run_dev", "select_dev"]
    if flag == "--dev-only":
        assert observed["events"] == ["run_dev", "select_dev"]
        assert len(observed["commands"]) == 1
    else:
        assert observed["events"] == ["run_dev", "select_dev", "run_test", "report"]
    assert observed["replays"] == []


def test_custom_grid_output_root_and_runtime_options_propagate(suite_run):
    _, selection, observed, _ = suite_run
    selection["selected_max_repairs"] = 7
    assert suite.main(["--repair-grid", "0,2,7", "--results-dir", "custom/suite",
                       "--replicates", "2", "--workers", "3", "--threshold", ".9", "--skip-paper"]) == 0
    assert observed["grid"] == (0, 2, 7)
    for command in observed["commands"][:2]:
        assert option(command, "--replicates") == "2"
        assert option(command, "--workers") == "3"
        assert option(command, "--threshold") == "0.9"
        assert Path(option(command, "--output")).parent.parts[-2:] == ("custom", "suite")
        assert option(command, "--max-repairs") == "7"
    assert [row[2]["max_repairs"] for row in observed["replays"]] == [0, 2]


def test_alternative_protocol_still_finishes_reports_without_changing_paper(suite_run, capsys):
    manifest, _, observed, paper = suite_run
    manifest.update(llm_provider="anthropic", llm_api_style="anthropic_messages", llm_model="alternative-model")
    observed["paper_eligible"] = False
    before = paper.read_bytes()
    assert suite.main([]) == 0
    assert "--paper" not in observed["commands"][-1]
    assert paper.read_bytes() == before
    assert "declared protocol" in capsys.readouterr().out


def test_explicit_skip_preserves_paper_and_still_reports(suite_run, monkeypatch, capsys):
    _, _, observed, paper = suite_run
    before = paper.read_bytes()

    def unexpected_check(_manifest):
        raise AssertionError("Explicit --skip-paper should not need eligibility checking")

    monkeypatch.setattr(suite, "paper_protocol_matches", unexpected_check)
    assert suite.main(["--skip-paper", "--main-only"]) == 0
    assert observed["events"] == ["run_dev", "select_dev", "run_test", "report"]
    assert "--paper" not in observed["commands"][-1]
    assert paper.read_bytes() == before
    assert "--skip-paper was requested" in capsys.readouterr().out


def test_invalid_selection_stops_before_any_test_request(suite_run, monkeypatch):
    _, _, observed, _ = suite_run

    def reject_selection(*_, **__):
        raise ValueError("Development source is incomplete")

    monkeypatch.setattr(suite, "select_repair_budget", reject_selection)
    with pytest.raises(ValueError, match="incomplete"):
        suite.main([])
    assert observed["events"] == ["run_dev"]
    assert len(observed["commands"]) == 1


@pytest.mark.parametrize("selection_method", [None, "schema_only"])
def test_suite_requires_jbfc_development_source(suite_run, selection_method):
    _, selection, observed, _ = suite_run
    selection["selection_method"] = selection_method
    with pytest.raises(ValueError, match="JBFC development method"):
        suite.main([])
    assert observed["events"] == ["run_dev", "select_dev"]
    assert len(observed["commands"]) == 1


def test_real_report_failure_is_not_hidden_by_success_message(suite_run, monkeypatch, capsys):
    run = suite.subprocess.run

    def failing_report(command, **kwargs):
        if command[3] == "report":
            raise subprocess.CalledProcessError(1, command)
        return run(command, **kwargs)

    monkeypatch.setattr(suite.subprocess, "run", failing_report)
    with pytest.raises(subprocess.CalledProcessError):
        suite.main([])
    assert "Experiment suite complete" not in capsys.readouterr().out


@pytest.mark.parametrize("grid", ["", "-1,2", "1.5,2", "x,2", "1,,2", "0,1,1"])
def test_invalid_grid_cannot_start_network_work(suite_run, grid):
    _, _, observed, _ = suite_run
    with pytest.raises(SystemExit):
        suite.main(["--repair-grid", grid])
    assert observed["commands"] == []


def test_changed_selection_does_not_overwrite_saved_audit_artifact(suite_run, tmp_path):
    _, selection, observed, _ = suite_run
    artifact_path = tmp_path / "results/tuned/repair_budget_selection.json"
    artifact_path.parent.mkdir(parents=True)
    prior = {**selection, "grid": [0, 2, 5], "selection_id": "older-development-grid"}
    original = json.dumps(prior, indent=2) + "\n"
    artifact_path.write_text(original)
    with pytest.raises(ValueError, match="new --results-dir"):
        suite.main([])
    assert observed["events"] == ["run_dev", "select_dev"]
    assert artifact_path.read_text() == original


def test_unchanged_selection_reuses_existing_artifact_without_rewriting(suite_run, tmp_path):
    _, selection, _, _ = suite_run
    artifact_path = tmp_path / "results/tuned/repair_budget_selection.json"
    artifact_path.parent.mkdir(parents=True)
    original = json.dumps(selection, separators=(",", ":"))
    artifact_path.write_text(original)
    assert suite.main(["--main-only", "--skip-paper"]) == 0
    assert artifact_path.read_text() == original


def test_repeated_suite_reuses_selection_and_same_derived_paths(suite_run, tmp_path):
    _, _, observed, _ = suite_run
    assert suite.main(["--skip-paper", "--include-abstain"]) == 0
    artifact_path = tmp_path / "results/tuned/repair_budget_selection.json"
    saved_selection = artifact_path.read_bytes()
    first_commands = list(observed["commands"])
    first_replays = list(observed["replays"])
    observed["commands"].clear()
    observed["events"].clear()
    observed["replays"].clear()
    assert suite.main(["--skip-paper", "--include-abstain"]) == 0
    assert artifact_path.read_bytes() == saved_selection
    assert observed["commands"] == first_commands
    assert observed["replays"] == first_replays
    assert all("--resume" in command for command in observed["commands"] if command[3] == "run")
