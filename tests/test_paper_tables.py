"""Tables 1 and 2 at r=3: gpt-5-nano's seven methods, and Direct/JBFC for gpt-5-nano and gpt-5.4-mini."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_fc.pipeline import METHODS
from jev_fc.reporting import PAPER_PRICES, PAPER_SETTINGS, TABLE1_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_tables", ROOT / "scripts/paper_tables.py")
paper_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_tables)


def by_category(method, value):
    return {f"{category}/{method}": {"accuracy": value, "relevance_call_rate": value}
            for category, _, _ in TABLE1_CATEGORIES}


def method_values(accuracy, cost):
    return {"accuracy": accuracy, "function_accuracy": .8, "arguments_accuracy_conditional": .8,
            "mean_latency_s": 2.0, "mean_cost_usd": cost}


def main_summary():
    manifest = {**PAPER_SETTINGS, "max_repairs": 3, "methods": list(METHODS),
                "prices_usd_per_million": dict(PAPER_PRICES), "source_sha256": "code", "dataset_sha256": "data",
                "ids": ["a", "b"], "split_seed": 1729,
                "jev_gateway": "direct", "jev_routes": [{"first_attempt": 0, "jev_gateway": "apim"},
                                                       {"first_attempt": 3, "jev_gateway": "direct"}]}
    methods = {method: method_values(.70, 2e-4) for method in METHODS}
    methods["direct"] = method_values(.638, 5.6e-5)
    methods["jev_joint"] = method_values(.753, 2.73e-4)
    return {"status": "complete", "manifest": manifest, "methods": methods,
            "by_category": {**by_category("direct", .6), **by_category("jev_joint", .7)}}


def compare_summary(direct=.8, jbfc=.82, **changes):
    manifest = {**main_summary()["manifest"], "methods": ["direct", "jev_joint"], "llm_model": "gpt-5.4-mini",
                "jev_gateway": "direct", "jev_routes": None,
                "prices_usd_per_million": {"llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5,
                                           "jev_input": .042}, **changes}
    return {"status": "complete", "manifest": manifest,
            "methods": {"direct": method_values(direct, 7.1e-4), "jev_joint": method_values(jbfc, 9e-4)},
            "by_category": {**by_category("direct", direct), **by_category("jev_joint", jbfc)}}


def test_unrun_comparison_leaves_two_empty_columns_named_for_the_planned_model():
    table = paper_tables.latex_table1(main_summary())
    assert r"\multicolumn{2}{c}{\texttt{gpt-5-nano}} & \multicolumn{2}{c}{\texttt{gpt-5.4-mini}}" in table
    assert r"Category & Examples & Direct & \textbf{JBFC} & Direct & \textbf{JBFC}\\" in table
    assert r"Simple Python & 400 & 60.0 & \textbf{70.0} & -- & --\\" in table
    assert r"Total & 3051 & 63.8 & \textbf{75.3} & -- & --\\" in table
    assert r"USD/1k & & 0.056 & 0.273 & -- & --\\" in table
    assert "repair limit $r=3$" in table and "-- means the experiment has not been run." in table


def test_filled_table_bolds_the_highest_of_four_and_shows_each_cost():
    table = paper_tables.latex_table1(main_summary(), compare_summary(.80, .82))
    assert r"Irrelevance & 240 & 60.0 & 70.0 & 80.0 & \textbf{82.0}\\" in table
    assert r"Total & 3051 & 63.8 & 75.3 & 80.0 & \textbf{82.0}\\" in table
    assert r"USD/1k & & 0.056 & 0.273 & 0.710 & 0.900\\" in table and "has not been run" not in table
    tied = paper_tables.latex_table1(main_summary(), compare_summary(.70, .70))
    assert r"Irrelevance & 240 & 60.0 & \textbf{70.0} & \textbf{70.0} & \textbf{70.0}\\" in tied
    low = paper_tables.latex_table1(main_summary(), compare_summary(reasoning_effort="low"))
    assert r"\texttt{gpt-5.4-mini} uses reasoning effort \texttt{low}." in low


def test_table2_states_the_raised_limit():
    table = paper_tables.latex_table2(main_summary())
    assert "from one run with repair limit $r=3$" in table and r"\textbf{JBFC} & \textbf{75.3}" in table
    assert "$r=2$" not in paper_tables.latex_table2(main_summary())


@pytest.mark.parametrize("change", [
    {"methods": ["direct"]}, {"source_sha256": "other code"}, {"dataset_sha256": "other"}, {"ids": ["a"]},
    {"llm_api_style": "responses"}, {"max_completion_tokens": 1024}, {"max_repairs": 2}, {"replicates": 3},
    {"llm_model": "gpt-5-nano"}, {"jev_model": "jev-2"}, {"jev_gateway": "elsewhere"},
    {"prices_usd_per_million": {"llm_input": None, "llm_cached_input": None, "llm_output": None}},
])
def test_comparison_must_follow_the_main_run_except_the_generator(monkeypatch, change):
    monkeypatch.setattr(paper_tables, "_check_all_examples", lambda manifest: None)
    paper_tables.check_compare(main_summary(), compare_summary())
    with pytest.raises(ValueError):
        paper_tables.check_compare(main_summary(), compare_summary(**change))


@pytest.mark.parametrize("change", [{"max_repairs": 2}, {"methods": ["direct", "jev_joint"]}, {"llm_model": "x"},
                                    {"jev_gateway": "apim"}])
def test_main_run_must_follow_the_r3_protocol(monkeypatch, change):
    monkeypatch.setattr(paper_tables, "_check_all_examples", lambda manifest: None)
    paper_tables.check_main(main_summary())
    summary = main_summary()
    summary["manifest"].update(change)
    with pytest.raises(ValueError):
        paper_tables.check_main(summary)


def write_run(path, summary, records):
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text("".join(json.dumps({"n": n}) + "\n" for n in range(records)))
    (path / "summary.json").write_text(json.dumps({**summary, "expected_records": records,
                                                   "completed_records": records}))


def test_cli_fills_both_tables_and_waits_for_the_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_tables, "_check_all_examples", lambda manifest: None)
    main_run, compare_run, paper = tmp_path / "main", tmp_path / "compare", tmp_path / "main.tex"
    write_run(main_run, main_summary(), 14)
    paper.write_text("% BEGIN AUTO CATEGORIES\nOLD TABLE 1\n% END AUTO CATEGORIES\n"
                     "% BEGIN AUTO RESULTS\nOLD TABLE 2\n% END AUTO RESULTS\n")
    paper_tables.main([str(main_run), "--compare", str(compare_run), "--paper", str(paper)])
    source = paper.read_text()
    assert "has not been run" in source and "OLD TABLE" not in source and "$r=3$" in source
    write_run(compare_run, compare_summary(.8, .82), 4)
    report = paper_tables.main([str(main_run), "--compare", str(compare_run), "--paper", str(paper)])
    source = paper.read_text()
    assert r"\textbf{82.0}" in source and "has not been run" not in source
    assert report["model"] == "gpt-5.4-mini" and report["compare_jbfc"]["accuracy"] == .82


def test_partial_or_stale_runs_cannot_fill_the_tables(tmp_path):
    run = tmp_path / "run"
    write_run(run, {**compare_summary(), "status": "partial"}, 2)
    with pytest.raises(ValueError):
        paper_tables.load_summary(run)
    stale = tmp_path / "stale"
    write_run(stale, compare_summary(), 2)
    (stale / "records.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        paper_tables.load_summary(stale)


def test_preflight_accepts_generator_only_flag():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/preflight.py"), "--generator-only"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "usage: preflight.py [--generator-only] ENV_FILE" in result.stderr
