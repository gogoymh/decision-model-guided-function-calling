"""JBFC at another Jev threshold against Direct and JBFC at the run's threshold (make threshold)."""

import importlib.util
import json
from pathlib import Path

import pytest

from jev_fc.reporting import TABLE1_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("threshold_compare", ROOT / "scripts/threshold_compare.py")
threshold_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold_compare)

BASE = {"source_sha256": "code", "dataset_sha256": "data", "ids": ["a", "b"], "replicates": 1, "max_repairs": 3,
        "llm_model": "gpt-5-nano", "reasoning_effort": "none", "max_completion_tokens": 4096, "jev_model": "jev-1.13.0",
        "prices_usd_per_million": {"llm_input": .05}, "exhaustion_policy": "execute_last", "threshold": .8,
        "methods": ["direct", "jev_joint"]}


def row(example, method, correct, initial, run, accepted=True):
    return {"id": example, "replicate": 0, "method": method, "category": "irrelevance", "accepted": accepted,
            "score": {"correct": correct},
            "trace": [{"proposal": {"calls": [{"name": f"f{run}"}], "content": None}, "score": {"correct": initial}}]}


def summary(manifest, outcomes):
    """outcomes: method -> accuracy."""
    return {"status": "complete", "manifest": manifest,
            "methods": {m: {"accuracy": a, "mean_cost_usd": 1e-4, "mean_latency_s": 1.0} for m, a in outcomes.items()},
            "by_category": {f"{c}/{m}": {"accuracy": a, "relevance_call_rate": a}
                            for m, a in outcomes.items() for c, _, _ in TABLE1_CATEGORIES}}


def write(path, run_summary, rows):
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "summary.json").write_text(json.dumps({**run_summary, "expected_records": len(rows),
                                                   "completed_records": len(rows)}))


# (Direct, JBFC at 0.8, JBFC at 0.7) correctness of examples a and b in two runs.
OUTCOMES = [{"a": (1, 0, 1), "b": (0, 1, 1)}, {"a": (1, 1, 1), "b": (0, 0, 1)}]


def make_runs(tmp_path, change_rerun=None, other_first=False):
    bases, reruns = [], []
    for index, outcomes in enumerate(OUTCOMES, start=1):
        base, rerun = tmp_path / f"base{index}", tmp_path / f"t07_{index}"
        base_rows, rerun_rows = [], []
        for example, (direct, jbfc, new) in outcomes.items():
            base_rows += [row(example, "direct", bool(direct), bool(direct), index),
                          row(example, "jev_joint", bool(jbfc), bool(direct), index)]
            first_run = index + 10 if other_first else index
            rerun_rows += [row(example, "direct", bool(direct), bool(direct), first_run),
                           row(example, "jev_joint", bool(new), bool(direct), first_run)]
        accuracy = {m: sum(o[i] for o in outcomes.values()) / 2 for i, m in enumerate(("direct", "jev_joint"))}
        write(base, summary(BASE, accuracy), base_rows)
        manifest = {**BASE, "threshold": .7, "rethreshold": {"method": "jev_joint", "from": .8, "to": .7},
                    "extends": {"run_dir": str(base)}, **(change_rerun or {})}
        write(rerun, summary(manifest, {"direct": accuracy["direct"],
                                        "jev_joint": sum(o[2] for o in outcomes.values()) / 2}), rerun_rows)
        bases.append(str(base))
        reruns.append(str(rerun))
    return bases, reruns


def test_report_pairs_each_rerun_with_its_run(tmp_path):
    bases, reruns = make_runs(tmp_path)
    report = threshold_compare.compare_model("gpt-5-nano", bases, .7, reruns)
    assert report["runs"] == [1, 2] and report["threshold"] == .7 and report["base_threshold"] == .8
    assert report["columns"]["jbfc_t0.7"]["accuracy"]["runs"] == [1.0, 1.0]
    assert report["columns"]["jbfc"]["accuracy"]["mean"] == pytest.approx(.5)
    # Per example over the two runs: a (1-1, 1-1) -> 0, b (1-0, 1-0) -> 1; mean 0.5.
    assert report["paired"]["jbfc_t0.7 - direct"]["delta"] == pytest.approx(.5)
    # Against JBFC at 0.8: a (1-0, 1-1) -> .5, b (1-1, 1-0) -> .5.
    assert report["paired"]["jbfc_t0.7 - jbfc"]["delta"] == pytest.approx(.5)
    assert report["paired"]["jbfc - direct"]["delta"] == pytest.approx(0)
    assert report["per_run"][0]["jbfc"] == {"fixes": 1, "breaks": 1}
    assert report["per_run"][0]["jbfc_t0.7"] == {"fixes": 1, "breaks": 0}


def test_runs_without_a_finished_rerun_are_left_out(tmp_path):
    bases, reruns = make_runs(tmp_path)
    report = threshold_compare.compare_model("gpt-5-nano", bases, .7, [reruns[0], str(tmp_path / "missing")])
    assert report["runs"] == [1]
    assert threshold_compare.compare_model("gpt-5-nano", bases, .7, [str(tmp_path / "x"), str(tmp_path / "y")]) is None


@pytest.mark.parametrize("change", [{"threshold": .6}, {"rethreshold": {"to": .6}}, {"source_sha256": "other"},
                                    {"llm_model": "gpt-5.4-mini"}, {"max_repairs": 2}, {"extends": {"run_dir": "elsewhere"}}])
def test_a_rerun_must_come_from_its_run_and_change_only_the_threshold(tmp_path, change):
    bases, reruns = make_runs(tmp_path, change_rerun=change)
    with pytest.raises(ValueError):
        threshold_compare.compare_model("gpt-5-nano", bases, .7, reruns)


def test_a_rerun_must_share_the_runs_first_proposals(tmp_path):
    bases, reruns = make_runs(tmp_path, other_first=True)
    with pytest.raises(ValueError, match="first proposals"):
        threshold_compare.compare_model("gpt-5-nano", bases, .7, reruns)
