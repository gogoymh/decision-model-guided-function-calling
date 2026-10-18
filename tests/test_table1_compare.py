"""Table 1's third column: Direct with a larger generator, next to gpt-5-nano's Direct and JBFC."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_fc.reporting import PAPER_PRICES, PAPER_SETTINGS, TABLE1_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("table1_compare", ROOT / "scripts/table1_compare.py")
table1_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table1_compare)


def by_category(method, value):
    return {f"{category}/{method}": {"accuracy": value, "relevance_call_rate": value}
            for category, _, _ in TABLE1_CATEGORIES}


def main_summary():
    manifest = {**PAPER_SETTINGS, "methods": ["direct", "jev_joint"], "prices_usd_per_million": dict(PAPER_PRICES),
                "source_sha256": "code", "dataset_sha256": "data", "ids": ["a", "b"], "split_seed": 1729,
                "llm_provider": "openai"}
    return {"status": "complete", "manifest": manifest,
            "methods": {"direct": {"accuracy": .638, "mean_cost_usd": 5.6e-5},
                        "jev_joint": {"accuracy": .735, "mean_cost_usd": 2.21e-4}},
            "by_category": {**by_category("direct", .6), **by_category("jev_joint", .7)}}


def compare_summary(accuracy=.7, **changes):
    manifest = {**main_summary()["manifest"], "methods": ["direct"], "llm_model": "gpt-5.4-mini",
                "jev_model": None, "jev_gateway": None,
                "prices_usd_per_million": {"llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5,
                                           "jev_input": .042}, **changes}
    return {"status": "complete", "manifest": manifest,
            "methods": {"direct": {"accuracy": accuracy, "mean_cost_usd": 7.1e-4}},
            "by_category": by_category("direct", accuracy)}


def test_unrun_column_is_empty_and_named_for_the_planned_model():
    table = table1_compare.latex_table1(main_summary())
    assert r"\multicolumn{2}{c}{\texttt{gpt-5-nano}} & \texttt{gpt-5.4-mini}" in table
    assert r"Category & Examples & Direct & \textbf{JBFC} & Direct\\" in table
    assert r"Simple Python & 400 & 60.0 & \textbf{70.0} & --\\" in table
    assert r"Total & 3051 & 63.8 & \textbf{73.5} & --\\" in table and r"USD/1k & & 0.056 & 0.221 & --\\" in table
    assert "-- means the experiment has not been run." in table and r"\label{tab:categories}" in table


def test_filled_column_bolds_the_highest_accuracy_per_row_and_shows_cost():
    table = table1_compare.latex_table1(main_summary(), compare_summary(.75))
    assert r"Irrelevance & 240 & 60.0 & 70.0 & \textbf{75.0}\\" in table
    assert r"Total & 3051 & 63.8 & 73.5 & \textbf{75.0}\\" in table and r"USD/1k & & 0.056 & 0.221 & 0.710\\" in table
    assert "has not been run" not in table and "reasoning effort" not in table
    tied = table1_compare.latex_table1(main_summary(), compare_summary(.7))
    assert r"Irrelevance & 240 & 60.0 & \textbf{70.0} & \textbf{70.0}\\" in tied
    low = table1_compare.latex_table1(main_summary(), compare_summary(.7, reasoning_effort="low"))
    assert r"\texttt{gpt-5.4-mini} uses reasoning effort \texttt{low}." in low


@pytest.mark.parametrize("change", [
    {"methods": ["direct", "jev_joint"]}, {"source_sha256": "other code"}, {"dataset_sha256": "other"},
    {"ids": ["a"]}, {"llm_api_style": "responses"}, {"max_completion_tokens": 1024}, {"max_repairs": 0},
    {"replicates": 3}, {"llm_model": "gpt-5-nano"}, {"jev_model": "jev-1.13.0"},
    {"prices_usd_per_million": {"llm_input": None, "llm_cached_input": None, "llm_output": None}},
])
def test_comparison_must_follow_the_main_run_except_the_generator(monkeypatch, change):
    monkeypatch.setattr(table1_compare, "_check_all_examples", lambda manifest: None)
    table1_compare.check_compare(main_summary(), compare_summary())
    with pytest.raises(ValueError):
        table1_compare.check_compare(main_summary(), compare_summary(**change))


def write_run(path, summary, records):
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text("".join(json.dumps({"n": n}) + "\n" for n in range(records)))
    (path / "initial.jsonl").write_text(json.dumps({"proposal": {"model": "gpt-5.4-mini-2026-03-17"}}) + "\n")
    (path / "summary.json").write_text(json.dumps({**summary, "expected_records": records,
                                                   "completed_records": records}))


def test_cli_fills_only_table_1_and_waits_for_the_comparison(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(table1_compare, "_check_all_examples", lambda manifest: None)
    main_run, compare_run, paper = tmp_path / "main", tmp_path / "compare", tmp_path / "main.tex"
    write_run(main_run, main_summary(), 4)
    paper.write_text("% BEGIN AUTO CATEGORIES\nOLD TABLE\n% END AUTO CATEGORIES\n% BEGIN AUTO RESULTS\nkeep\n% END AUTO RESULTS\n")
    table1_compare.main([str(main_run), "--compare", str(compare_run), "--paper", str(paper)])
    assert "has not been run" in paper.read_text() and "keep" in paper.read_text()
    write_run(compare_run, compare_summary(.75), 2)
    report = table1_compare.main([str(main_run), "--compare", str(compare_run), "--paper", str(paper)])
    source = paper.read_text()
    assert r"\textbf{75.0}" in source and "has not been run" not in source and "OLD TABLE" not in source
    assert report["compare_returned_models"] == ["gpt-5.4-mini-2026-03-17"] and report["model"] == "gpt-5.4-mini"


def test_partial_or_stale_runs_cannot_fill_the_table(tmp_path):
    run = tmp_path / "run"
    write_run(run, {**compare_summary(), "status": "partial"}, 2)
    with pytest.raises(ValueError):
        table1_compare.load_summary(run)
    stale = tmp_path / "stale"
    write_run(stale, compare_summary(), 2)
    (stale / "records.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        table1_compare.load_summary(stale)


def test_preflight_accepts_generator_only_flag():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/preflight.py"), "--generator-only"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "usage: preflight.py [--generator-only] ENV_FILE" in result.stderr
