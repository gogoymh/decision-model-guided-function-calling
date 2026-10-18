"""Table of the Jev threshold experiments (docs/threshold.md)."""

import importlib.util
import json
from pathlib import Path

import pytest

from jev_fc.reporting import TABLE1_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("threshold_table", ROOT / "scripts/threshold_table.py")
threshold_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold_table)

MANIFEST = {"source_sha256": "code", "dataset_sha256": "data", "ids": ["a", "b"], "replicates": 1, "max_repairs": 3,
            "llm_model": "gpt-5-nano", "reasoning_effort": "none", "max_completion_tokens": 4096,
            "jev_model": "jev-1.13.0", "prices_usd_per_million": {"llm_input": .05}, "exhaustion_policy": "execute_last",
            "threshold": .8, "methods": ["direct", "jev_joint"]}


def row(example, method, final, first, accepted, p_function, run):
    gate = accepted and method == "jev_joint" and final == first  # passed at once
    return {"id": example, "replicate": 0, "method": method, "category": "irrelevance", "accepted": accepted,
            "score": {"correct": final},
            "trace": [{"proposal": {"calls": [{"name": f"f{run}"}], "content": None}, "score": {"correct": first},
                       "gate_pass": gate, "schema": {"accepted": True},
                       "verification": {"function_probability": p_function, "arguments_probability": .9,
                                        "arguments_skipped": False}}]}


def summary(manifest, direct, jbfc):
    values = {"direct": direct, "jev_joint": jbfc}
    return {"status": "complete", "manifest": manifest,
            "methods": {m: {"accuracy": a, "function_accuracy": a, "mean_cost_usd": 1e-4, "mean_latency_s": 1.0,
                            "mean_repairs": 1.0, "exhaustion_rate": .5, "coverage": .5, "accepted_precision": 1.0}
                        for m, a in values.items()},
            "by_category": {f"{c}/{m}": {"accuracy": a, "relevance_call_rate": a}
                            for m, a in values.items() for c, _, _ in TABLE1_CATEGORIES}}


def write(path, run_summary, rows):
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "summary.json").write_text(json.dumps({**run_summary, "expected_records": len(rows),
                                                   "completed_records": len(rows)}))


def build(tmp_path):
    """Two runs. Run k: a's first proposal is right; b's is wrong. JBFC at 0.8 breaks a at the limit and
    fixes b; JBFC at 0.7 accepts a at once and leaves b wrong at the limit."""
    results, base_dirs = tmp_path / "results", []
    for run in (1, 2):
        base = results / f"base{run}"
        write(base, summary(MANIFEST, .5, .5),
              [row("a", "direct", True, True, True, .75, run), row("b", "direct", False, False, True, .4, run),
               row("a", "jev_joint", False, True, False, .75, run), row("b", "jev_joint", True, False, True, .4, run)])
        rerun = results / f"jbfc_t0.7_gpt-5-nano_run{run}"
        manifest = {**MANIFEST, "threshold": .7, "rethreshold": {"method": "jev_joint", "from": .8, "to": .7},
                    "extends": {"run_dir": str(base)}}
        write(rerun, summary(manifest, .5, .5),
              [row("a", "direct", True, True, True, .75, run), row("b", "direct", False, False, True, .4, run),
               row("a", "jev_joint", True, True, True, .75 if run == 1 else .76, run),
               row("b", "jev_joint", False, False, False, .4, run)])
        base_dirs.append(f"results/base{run}")
    return base_dirs


def test_report_rows_and_trajectory_counts(tmp_path):
    base_dirs = build(tmp_path)
    assert threshold_table.thresholds_found("gpt-5-nano", tmp_path / "results") == [.7]
    report = threshold_table.model_report("gpt-5-nano", base_dirs, root=tmp_path)
    assert list(report["jbfc"]) == ["0.8", "0.7"]
    base, rerun = report["jbfc"]["0.8"]["trajectories"], report["jbfc"]["0.7"]["trajectories"]
    assert (base["fixes"], base["breaks"], base["breaks_at_limit"]) == (1, 1, 1)
    assert (base["right_rejected"], base["wrong_rejected"]) == (1, 1)
    assert (rerun["fixes"], rerun["breaks"], rerun["right_rejected"]) == (0, 0, 0)
    # Keeping the first proposal when nothing passes: a stays right at 0.8, b keeps its accepted fix.
    assert base["keep_first_correct"] / base["n"] == pytest.approx(1.0)
    assert report["jbfc"]["0.7"]["vs_base"]["delta"] == pytest.approx(0)
    repeat = report["jbfc"]["0.7"]["jev_repeat"]
    assert repeat["first_proposals"] == 4 and repeat["same_probabilities"] == pytest.approx(.75)
    assert repeat["verdict_changed"] == pytest.approx(0)


def test_markdown_lists_every_threshold_with_signed_intervals(tmp_path):
    base_dirs = build(tmp_path)
    text = threshold_table.markdown({"gpt-5-nano": threshold_table.model_report("gpt-5-nano", base_dirs, root=tmp_path)})
    assert "| | **JBFC** | 0.8 | 50.0 ± 0.0 | 50.0 / 50.0 |" in text and "| | **JBFC** | 0.7 |" in text
    assert "| | 0.8 | 1 / 1 | 1 / 1 | 1 | 1 (1) | 100.0 |" in text
    assert "자동 생성 파일" in text and "-" not in text.split("## 정확도")[1].split("##")[0].replace("|---", "").replace("-5-nano", "")


@pytest.mark.parametrize("value, text", [(.0004, "0.0"), (-.0004, "0.0"), (.109, "+10.9"), (-.016, "−1.6")])
def test_signed_points(value, text):
    assert threshold_table._signed(value) == text
