"""Metric denominators and counterfactual stopping policies are research-critical."""

import hashlib
import json
from pathlib import Path

import pytest

from jev_fc.benchmarks import load_examples, stratified_split
from jev_fc.reporting import build_summary, derive_policy_run, summarize_group, token_cost, update_paper


def record(name="a", category="simple_python", correct=False, accepted=True, initial_correct=None):
    if initial_correct is None:
        initial_correct = correct
    score = {"correct": correct, "function_correct": correct, "arguments_correct": correct if correct else None}
    trace = [{"proposal": {"calls": [], "latency_s": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
              "verification": {"function_probability": .7, "arguments_probability": .9,
                               "latency_s": .1, "api_calls": 1, "usage": {"input_tokens": 5}},
              "score": {**score, "correct": initial_correct}, "gate_pass": accepted}]
    return {"id": name, "category": category, "method": "jev_joint", "replicate": 0, "score": score,
            "trace": trace, "accepted": accepted, "repairs": 0, "dispatched": True,
            "exhausted": not accepted, "logical_latency_s": 1.1, "synthetic": False, "config_id": "test"}


def test_presence_only_category_does_not_inflate_semantic_precision():
    rows = [record("wrong", correct=False), record("presence", "live_relevance", correct=True)]
    metrics = summarize_group(rows)
    assert metrics["n"] == 2 and metrics["gate_diagnostic_n"] == 1
    assert metrics["accepted_precision"] == 0
    assert metrics["false_accept_rate"] == 1
    assert metrics["relevance_call_rate"] == 1
    assert metrics["accuracy"] == 0


def test_false_rates_use_their_actual_conditioning_denominators():
    rows = [record("wa", correct=False), record("wr", correct=False, accepted=False),
            record("ra", correct=True), record("rr", correct=True, accepted=False),
            record("ra2", correct=True)]
    metrics = summarize_group(rows)
    assert metrics["false_accept_rate"] == .5
    assert metrics["false_reject_rate"] == pytest.approx(1 / 3)
    assert metrics["selective_risk"] == pytest.approx(1 / 3)
    assert metrics["accepted_precision"] == pytest.approx(2 / 3)


def test_no_accepted_predictions_yields_unavailable_precision():
    result = summarize_group([record(accepted=False)])
    assert result["accepted_precision"] is None
    assert result["coverage"] == 0


PRICES = {"llm_input": .75, "llm_output": 4.5, "llm_cached_input": .075, "jev_input": .042}


def test_missing_usage_never_means_free_inference():
    row = record()
    row["trace"][0]["verification"]["usage"]["input_tokens"] = None
    accounting = token_cost(row, PRICES)
    assert accounting["estimated_usd"] is None
    assert not accounting["complete"]


def write_run(directory, rows):
    manifest = {"ids": sorted({r["id"] for r in rows}), "replicates": 1, "methods": ["jev_joint"],
                "expected_records": len({r["id"] for r in rows}), "config_id": "test", "synthetic": False,
                "prices_usd_per_million": PRICES, "max_repairs": 3, "split": "test"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_duplicate_records_cannot_falsely_complete_a_run(tmp_path):
    row = record()
    write_run(tmp_path, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        build_summary(tmp_path)


def test_foreign_config_cannot_enter_metrics(tmp_path):
    row = record()
    row["config_id"] = "foreign"
    write_run(tmp_path, [row])
    with pytest.raises(ValueError, match="mismatched-config"):
        build_summary(tmp_path)


def test_initial_diagnostics_do_not_include_final_cost_and_repairs(tmp_path):
    row = record()
    row["repairs"] = 3
    write_run(tmp_path, [row])
    initial = build_summary(tmp_path)["initial_candidate_diagnostics"]["jev_joint"]
    assert "mean_repairs" not in initial
    assert "mean_cost_usd" not in initial
    assert "mean_latency_s" not in initial


def test_budget_replay_uses_prefix_and_preserves_final_dispatch_policy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    row = record(correct=True, accepted=True, initial_correct=False)
    row["trace"][0]["gate_pass"] = False
    last = record(correct=True, accepted=True)["trace"][0]
    row["trace"].append(last)
    row["repairs"] = 1
    write_run(source, [row])
    zero = derive_policy_run(source, tmp_path / "zero", max_repairs=0)
    one = derive_policy_run(source, tmp_path / "one", max_repairs=1)
    abstain = derive_policy_run(source, tmp_path / "abstain", max_repairs=0, exhaustion_policy="abstain")
    assert zero["methods"]["jev_joint"]["accuracy"] == 0
    assert zero["methods"]["jev_joint"]["dispatch_rate"] == 1
    assert zero["methods"]["jev_joint"]["mean_llm_proposals"] == 1
    assert one["methods"]["jev_joint"]["accuracy"] == 1
    assert one["methods"]["jev_joint"]["mean_llm_proposals"] == 2
    assert abstain["methods"]["jev_joint"]["dispatch_rate"] == 0


def test_manuscript_rejects_same_ids_but_modified_data_digest(tmp_path):
    path = Path("data/bfcl/examples.jsonl")
    if not path.exists():
        pytest.skip("Requires prepared BFCL files")
    _, test = stratified_split(load_examples(path), seed=1729)
    summary = {"status": "complete", "manifest": {"ids": [r["id"] for r in test], "split": "test",
                                                    "dataset_sha256": "tampered", "limit": None}}
    with pytest.raises(ValueError, match="digest"):
        update_paper(tmp_path / "paper.tex", summary)


def test_manuscript_rejects_different_reasoning_setting(tmp_path):
    path = Path("data/bfcl/examples.jsonl")
    if not path.exists():
        pytest.skip("Requires prepared BFCL files")
    _, test = stratified_split(load_examples(path), seed=1729)
    from jev_fc.pipeline import METHODS
    manifest = {"ids": [r["id"] for r in test], "split": "test", "limit": None,
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "methods": METHODS,
                "replicates": 3, "max_repairs": 3, "threshold": .8, "exhaustion_policy": "execute_last",
                "split_seed": 1729, "llm_model": "gpt-5.4-mini-2026-03-17", "jev_model": "jev-1.13.0",
                "reasoning_effort": "high", "temperature": None, "max_completion_tokens": 4096}
    with pytest.raises(ValueError, match="declared protocol"):
        update_paper(tmp_path / "paper.tex", {"status": "complete", "manifest": manifest})
