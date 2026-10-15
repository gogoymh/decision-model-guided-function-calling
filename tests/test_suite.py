"""Suite orchestration stays offline here, including manuscript eligibility."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from jev_fc.pipeline import METHODS


_spec = importlib.util.spec_from_file_location("run_suite", Path(__file__).parents[1] / "scripts/run_suite.py")
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)


@pytest.fixture
def suite_run(tmp_path, monkeypatch):
    manifest = {
        "split": "test", "limit": None, "split_seed": 1729,
        "methods": list(METHODS), "replicates": 3, "max_repairs": 3,
        "threshold": .8, "exhaustion_policy": "execute_last",
        "llm_model": "gpt-5.4-mini-2026-03-17", "jev_model": "jev-1.13.0",
        "reasoning_effort": "none", "temperature": None, "max_completion_tokens": 4096,
        "llm_provider": "openai", "llm_api_style": "responses", "llm_gateway": "apim",
        "jev_gateway": "apim", "synthetic": False,
        "prices_usd_per_million": {"llm_input": .75, "llm_cached_input": .075,
                                   "llm_output": 4.5, "jev_input": .042},
    }
    observed = {"commands": [], "replays": []}
    paper = tmp_path / "paper/main.tex"
    paper.parent.mkdir()
    paper.write_text("existing manuscript\n")
    monkeypatch.setattr(suite, "__file__", str(tmp_path / "scripts/run_suite.py"))

    def run(command, *, cwd, check):
        assert cwd == tmp_path and check is True
        observed["commands"].append(command)
        if command[3] == "run":
            directory = tmp_path / "results/main"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(json.dumps(manifest))
        elif "--paper" in command:
            # Simulate the separate report command's effect, without running
            # generation, editing the real manuscript or weakening its guards.
            paper.write_text("updated manuscript tables\n")
        return subprocess.CompletedProcess(command, 0)

    def derive(source, target, **kwargs):
        observed["replays"].append((source, target, kwargs))

    monkeypatch.setattr(suite.subprocess, "run", run)
    monkeypatch.setattr(suite, "derive_policy_run", derive)
    return manifest, observed, paper


def test_canonical_suite_reports_and_populates_paper(suite_run, capsys):
    _, observed, paper = suite_run
    assert suite.main([]) == 0
    assert [entry[2] for entry in observed["replays"]] == [{"max_repairs": n} for n in range(3)]
    assert observed["commands"][-1][3:] == ["report", "results/main", "--paper", "paper/main.tex"]
    assert paper.read_text() == "updated manuscript tables\n"
    assert "Experiment suite complete" in capsys.readouterr().out


@pytest.mark.parametrize("changes", [
    {"llm_provider": "anthropic", "llm_api_style": "messages", "llm_model": "alternative-model",
     "reasoning_effort": None},
    {"llm_api_style": "chat_completions"},
    {"llm_gateway": "direct", "jev_gateway": "direct"},
    {"replicates": 1},
    {"prices_usd_per_million": {"llm_input": None, "llm_cached_input": None,
                                "llm_output": None, "jev_input": .042}},
])
def test_alternative_configuration_completes_replays_and_report_without_paper(suite_run, capsys, changes):
    manifest, observed, paper = suite_run
    manifest.update(changes)
    original = paper.read_bytes()
    assert suite.main(["--include-abstain"]) == 0
    assert len(observed["replays"]) == 4
    assert observed["replays"][-1][2] == {"exhaustion_policy": "abstain"}
    assert observed["commands"][-1][3:] == ["report", "results/main"]
    assert paper.read_bytes() == original
    output = capsys.readouterr().out
    assert "Experiment suite complete; reports saved" in output
    assert "Paper update skipped" in output and "declared protocol" in output


def test_explicit_skip_preserves_canonical_paper_and_still_reports(suite_run, monkeypatch, capsys):
    _, observed, paper = suite_run
    original = paper.read_bytes()

    def unexpected_check(_manifest):
        raise AssertionError("Explicit --skip-paper should not need eligibility checking")

    monkeypatch.setattr(suite, "paper_protocol_matches", unexpected_check)
    assert suite.main(["--skip-paper", "--main-only"]) == 0
    assert observed["replays"] == []
    assert observed["commands"][-1][3:] == ["report", "results/main"]
    assert paper.read_bytes() == original
    assert "--skip-paper was requested" in capsys.readouterr().out


def test_real_report_failure_is_not_hidden_by_successful_completion(suite_run, monkeypatch, capsys):
    run = suite.subprocess.run

    def failing_report(command, **kwargs):
        if command[3] == "report":
            raise subprocess.CalledProcessError(1, command)
        return run(command, **kwargs)

    monkeypatch.setattr(suite.subprocess, "run", failing_report)
    with pytest.raises(subprocess.CalledProcessError):
        suite.main([])
    assert "Experiment suite complete" not in capsys.readouterr().out
