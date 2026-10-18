"""Table 1 as the mean of three runs per generator (make repeat)."""

import importlib.util
import json
from pathlib import Path

import pytest

from jev_fc.pipeline import METHODS
from jev_fc.reporting import PAPER_PRICES, PAPER_SETTINGS, TABLE1_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("table1_repeat", ROOT / "scripts/table1_repeat.py")
table1_repeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table1_repeat)

MINI_PRICES = {"llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5, "jev_input": .042}


@pytest.fixture(autouse=True)
def any_examples(monkeypatch):
    """The synthetic runs below have two examples instead of BFCL's 3,051."""
    monkeypatch.setattr(table1_repeat.paper_tables, "_check_all_examples", lambda manifest: None)


def by_category(method, value):
    return {f"{category}/{method}": {"accuracy": value, "relevance_call_rate": value}
            for category, _, _ in TABLE1_CATEGORIES}


def summary(manifest, direct, jbfc, costs=(5.6e-5, 2.73e-4)):
    methods = {method: {"accuracy": .7, "mean_cost_usd": 2e-4} for method in manifest["methods"]}
    methods["direct"] = {"accuracy": direct, "mean_cost_usd": costs[0], "mean_latency_s": 1.3}
    methods["jev_joint"] = {"accuracy": jbfc, "mean_cost_usd": costs[1], "mean_latency_s": 3.5, "mean_repairs": 1.0}
    return {"status": "complete", "manifest": manifest, "methods": methods,
            "by_category": {**by_category("direct", direct), **by_category("jev_joint", jbfc)}}


def nano_manifest(**changes):
    return {**PAPER_SETTINGS, "max_repairs": 3, "methods": list(METHODS), "prices_usd_per_million": dict(PAPER_PRICES),
            "source_sha256": "code", "dataset_sha256": "data", "ids": ["a", "b"], "split_seed": 1729,
            "jev_gateway": "direct", "jev_routes": [{"first_attempt": 0, "jev_gateway": "apim"},
                                                   {"first_attempt": 3, "jev_gateway": "direct"}], **changes}


def mini_manifest(**changes):
    return nano_manifest(methods=["direct", "jev_joint"], llm_model="gpt-5.4-mini", jev_routes=None,
                         prices_usd_per_million=dict(MINI_PRICES), added=["jev_joint"],
                         extends={"run_dir": "results/direct_gpt-5.4-mini", "max_repairs": 2}, **changes)


def repeat_manifest(first, run, **changes):
    model = first["llm_model"]
    return {**first, "methods": ["direct", "jev_joint"], "jev_routes": None, "jev_gateway": "direct",
            "added": ["jev_joint"], "extends": {"run_dir": f"results/direct_{model}_run{run}", "max_repairs": 2},
            **changes}


def runs():
    nano, mini = nano_manifest(), mini_manifest()
    small = [summary(nano, .638, .753), summary(repeat_manifest(nano, 2), .640, .749),
             summary(repeat_manifest(nano, 3), .636, .760)]
    large = [summary(mini, .813, .796, (7.06e-4, 1.718e-3)), summary(repeat_manifest(mini, 2), .810, .800, (7e-4, 1.7e-3)),
             summary(repeat_manifest(mini, 3), .816, .792, (7.12e-4, 1.736e-3))]
    return small, large


def test_table_shows_each_cells_mean_the_totals_spread_and_the_mean_cost():
    table = table1_repeat.latex_table1(*runs())
    assert r"\multicolumn{2}{c}{\texttt{gpt-5-nano}} & \multicolumn{2}{c}{\texttt{gpt-5.4-mini}}" in table
    assert r"Irrelevance & 240 & 63.8 & 75.4 & \textbf{81.3} & 79.6\\" in table
    assert (r"Total & 3051 & 63.8{\scriptsize$\pm$0.2} & 75.4{\scriptsize$\pm$0.6} & "
            r"\textbf{81.3}{\scriptsize$\pm$0.3} & 79.6{\scriptsize$\pm$0.4}\\") in table
    assert "s.d. &" not in table
    assert r"USD/1k & & 0.056 & 0.273 & 0.706 & 1.718\\" in table
    assert r"Accuracy (\%) on all BFCL examples, mean of three runs, $r=3$ (Direct makes no repairs)." in table
    assert r"Bold: highest per row. Total: mean $\pm$ s.d. across runs. USD/1k: mean cost per 1,000 requests." in table
    assert r"$^\dagger$Call presence only; not in the total.}" in table
    assert r"\label{tab:categories}" in table and "one run" not in table


@pytest.mark.parametrize("change", [
    {"methods": ["direct"]}, {"methods": list(METHODS)}, {"source_sha256": "other code"}, {"dataset_sha256": "other"},
    {"max_repairs": 2}, {"replicates": 3}, {"temperature": 1.0}, {"llm_model": "gpt-5.4-mini"},
    {"reasoning_effort": "low"}, {"prices_usd_per_million": dict(MINI_PRICES)}, {"jev_model": "jev-2"},
    {"llm_gateway": "direct"}, {"added": []}, {"extends": None}, {"jev_gateway": "elsewhere"},
])
def test_a_repeat_must_follow_run_one_except_for_new_samples(change):
    first = summary(nano_manifest(), .638, .753)
    table1_repeat.check_repeat(first, summary(repeat_manifest(first["manifest"], 2), .64, .75), "run 2")
    with pytest.raises(ValueError):
        table1_repeat.check_repeat(first, summary(repeat_manifest(first["manifest"], 2, **change), .64, .75), "run 2")


def row(example, method, correct, initial, calls, category="simple_python"):
    return {"id": example, "replicate": 0, "method": method, "category": category, "score": {"correct": correct},
            "trace": [{"proposal": {"calls": calls, "content": None}, "score": {"correct": initial}}]}


# (Direct, JBFC) correctness of examples a and b in runs 1-3; Direct is also JBFC's first proposal.
SMALL = [{"a": (0, 1), "b": (1, 1)}, {"a": (0, 1), "b": (1, 0)}, {"a": (1, 1), "b": (0, 1)}]
LARGE = [{"a": (1, 1), "b": (1, 0)}, {"a": (1, 0), "b": (1, 1)}, {"a": (1, 1), "b": (0, 0)}]


def records(outcomes, run, extra_methods=()):
    lines = []
    for example, (direct, jbfc) in outcomes.items():
        calls = [{"name": f"f{run}", "arguments": {"example": example}}]
        lines.append(row(example, "direct", bool(direct), bool(direct), calls))
        lines.append(row(example, "jev_joint", bool(jbfc), bool(direct), calls))
        lines += [row(example, method, True, bool(direct), calls) for method in extra_methods]
    return lines


def write_run(path, run_summary, lines):
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (path / "summary.json").write_text(json.dumps({**run_summary, "expected_records": len(lines),
                                                   "completed_records": len(lines)}))


def write_runs(tmp_path):
    small, large = runs()
    others = [m for m in METHODS if m not in ("direct", "jev_joint")]
    small_dirs, large_dirs = [], []
    for index in range(3):
        small_dirs.append(tmp_path / f"nano{index + 1}")
        large_dirs.append(tmp_path / f"mini{index + 1}")
        write_run(small_dirs[-1], small[index], records(SMALL[index], index + 1, others if index == 0 else ()))
        write_run(large_dirs[-1], large[index], records(LARGE[index], index + 1))
    return [str(d) for d in small_dirs], [str(d) for d in large_dirs]


def test_cli_writes_the_mean_table_and_reports_paired_differences_over_runs(tmp_path):
    small_dirs, large_dirs = write_runs(tmp_path)
    paper, report_path = tmp_path / "main.tex", tmp_path / "repeat.json"
    paper.write_text("% BEGIN AUTO CATEGORIES\nOLD TABLE 1\n% END AUTO CATEGORIES\n"
                     "% BEGIN AUTO RESULTS\nOLD TABLE 2\n% END AUTO RESULTS\n")
    report = table1_repeat.main(["--small", *small_dirs, "--large", *large_dirs, "--paper", str(paper),
                                 "--json", str(report_path)])
    source = paper.read_text()
    assert "mean of three runs" in source and "OLD TABLE 1" not in source and "OLD TABLE 2" in source
    assert json.loads(report_path.read_text()) == json.loads(json.dumps(report))
    nano = report["paired"]["gpt-5-nano jbfc - direct"]
    assert nano["delta"] == pytest.approx(1 / 3) and nano["n_paired_ids"] == 2 and nano["n_paired_records"] == 6
    # Large Direct minus small JBFC, per example averaged over runs: a (1 - 1), b (2/3 - 2/3).
    assert report["paired"]["gpt-5.4-mini direct - gpt-5-nano jbfc"]["delta"] == pytest.approx(0)
    per_run = report["per_run"]["gpt-5-nano"]
    assert [(r["fixes"], r["breaks"]) for r in per_run] == [(1, 0), (1, 1), (1, 0)]
    assert [r["same_first_calls_as_run_1"] for r in per_run] == [1.0, 0.0, 0.0]
    assert per_run[1]["jbfc_minus_direct"]["delta"] == pytest.approx(0)
    total = report["columns"]["gpt-5-nano/jbfc"]["accuracy"]
    assert total["runs"] == [.753, .749, .760] and total["mean"] == pytest.approx(.754)


def test_a_copied_run_is_rejected(tmp_path):
    small_dirs, large_dirs = write_runs(tmp_path)
    copy = Path(small_dirs[2])
    lines = [json.loads(line) for line in (Path(small_dirs[1]) / "records.jsonl").read_text().splitlines()]
    (copy / "records.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError, match="same first proposals"):
        table1_repeat.main(["--small", *small_dirs, "--large", *large_dirs])


def test_if_complete_waits_for_missing_runs_and_changes_nothing(tmp_path, capsys):
    small_dirs, large_dirs = write_runs(tmp_path)
    paper = tmp_path / "main.tex"
    paper.write_text("% BEGIN AUTO CATEGORIES\nOLD TABLE 1\n% END AUTO CATEGORIES\n")
    missing = str(tmp_path / "not_yet")
    assert table1_repeat.main(["--small", *small_dirs[:2], missing, "--large", *large_dirs, "--paper", str(paper),
                               "--if-complete"]) is None
    assert "OLD TABLE 1" in paper.read_text() and "not_yet" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError):
        table1_repeat.main(["--small", *small_dirs[:2], missing, "--large", *large_dirs])


@pytest.mark.parametrize("argv", [["--small", "x", "y", "--large", "z"], ["--small", "x", "--large", "y"],
                                  ["--small", "x", "x", "--large", "y", "z"]])
def test_runs_must_pair_up_and_differ(argv):
    with pytest.raises(ValueError):
        table1_repeat.main(argv)
