"""Metric denominators and counterfactual stopping policies are research-critical."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from jev_fc.benchmarks import load_examples
from jev_fc.reporting import (
    PAPER_PRICES,
    PAPER_SETTINGS,
    TABLE1_CATEGORIES,
    build_summary,
    derive_policy_run,
    latex_category_table,
    latex_table,
    paired_delta,
    paper_protocol_matches,
    summarize_group,
    token_cost,
    update_paper,
    update_table1,
)


def record(name="a", category="simple_python", correct=False, accepted=True, initial_correct=None, max_repairs=3):
    if initial_correct is None:
        initial_correct = correct
    score = {"correct": correct, "function_correct": correct, "arguments_correct": correct if correct else None}
    trace = [{"attempt": 0, "proposal": {"calls": [{"name": "fixture_function", "arguments": {}}],
                                         "latency_s": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
              "verification": {"function_probability": .7, "arguments_probability": .9,
                               "arguments_skipped": False,
                               "latency_s": .1, "api_calls": 1, "usage": {"input_tokens": 5}},
              "score": {**score, "correct": initial_correct}, "gate_pass": accepted}]
    return {"id": name, "category": category, "method": "jev_joint", "replicate": 0, "score": score,
            "trace": trace, "accepted": accepted, "repairs": 0, "dispatched": True,
            "exhausted": not accepted, "logical_latency_s": 1.1, "synthetic": False, "config_id": "test",
            "max_repairs": max_repairs}


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


def test_skipped_no_call_arguments_do_not_become_brier_observations():
    called = record("called", correct=True)
    empty = record("empty", category="irrelevance", correct=True)
    empty["trace"][0]["proposal"]["calls"] = []
    empty["trace"][0]["verification"].update(arguments_skipped=True, arguments_probability=None,
                                              arguments_pass=None)
    empty["score"]["arguments_correct"] = None
    empty["trace"][0]["score"]["arguments_correct"] = None
    metrics = summarize_group([called, empty])
    assert metrics["verified_candidates_n"] == 2
    assert metrics["brier_function_n"] == 2
    assert metrics["brier_arguments_n"] == 1
    assert metrics["brier_arguments"] == pytest.approx(.01)
    assert metrics["arguments_scored_n"] == 1


PRICES = {"llm_input": .75, "llm_output": 4.5, "llm_cached_input": .075, "jev_input": .042}


def repaired(name, initial_calls, final_calls, initial_score, final_score):
    row = record(name, correct=final_score["correct"])
    first = deepcopy(row["trace"][0])
    first["proposal"]["calls"] = initial_calls
    first["score"] = initial_score
    last = deepcopy(first)
    last.update(attempt=1, score=final_score)
    last["proposal"]["calls"] = final_calls
    row.update(trace=[first, last], score=final_score, repairs=1)
    row["accounting"] = token_cost(row, PRICES)
    return row


def test_initial_to_final_transitions_tokens_and_gate_denominators_are_reported():
    right = {"correct": True, "function_correct": True, "arguments_correct": True}
    wrong_function = {"correct": False, "function_correct": False, "arguments_correct": None}
    wrong_arguments = {"correct": False, "function_correct": True, "arguments_correct": False}
    calls = [{"name": "fixture_function", "arguments": {}}]
    other = [{"name": "other_function", "arguments": {}}]
    rows = [repaired("function_fix", other, calls, wrong_function, right),
            repaired("argument_fix", calls, calls, wrong_arguments, right),
            repaired("regression", calls, other, right, wrong_function)]
    metrics = summarize_group(rows)
    assert metrics["function_change_rate"] == pytest.approx(2 / 3)
    assert metrics["function_transition_n"] == 3
    assert metrics["function_fix_rate"] == pytest.approx(1 / 3)
    assert metrics["function_regression_rate"] == pytest.approx(1 / 3)
    assert metrics["argument_transition_n"] == 1 and metrics["argument_fix_rate"] == 1
    assert metrics["argument_regression_rate"] == 0
    assert (metrics["accepted_n"], metrics["gate_correct_n"], metrics["gate_incorrect_n"]) == (3, 2, 1)
    assert metrics["token_observed_n"] == 3
    assert metrics["mean_llm_input_tokens"] == 20 and metrics["mean_llm_output_tokens"] == 4
    assert metrics["mean_jev_input_tokens"] == 10

    unknown = deepcopy(rows[0])
    unknown["trace"][1]["proposal"]["usage"] = {}
    unknown["accounting"] = token_cost(unknown, PRICES)
    assert unknown["accounting"]["tokens_complete"] is False
    partial = summarize_group([rows[1], unknown])
    assert partial["token_observed_n"] == 1 and partial["mean_llm_input_tokens"] is None


def test_missing_usage_never_means_free_inference():
    row = record()
    row["trace"][0]["verification"]["usage"]["input_tokens"] = None
    accounting = token_cost(row, PRICES)
    assert accounting["estimated_usd"] is None
    assert not accounting["complete"]


def test_responses_cache_hit_is_discounted_once():
    row = record()
    row["trace"][0]["proposal"]["usage"] = {
        "input_tokens": 100, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 20},
    }
    result = token_cost(row, PRICES)
    assert result["llm_input"] == 100 and result["llm_cached_input"] == 20
    assert result["estimated_usd"] == pytest.approx((80 * .75 + 20 * .075 + 2 * 4.5 + 5 * .042) / 1e6)


def test_anthropic_reads_and_two_write_lifetimes_are_billed_once():
    row = record()
    row["trace"][0]["proposal"]["usage"] = {
        "prompt_tokens": 135, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 20},
        "cache_creation_input_tokens": 15,
        "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 5},
    }
    result = token_cost(row, {**PRICES, "llm_cache_write_5m": 1.25, "llm_cache_write_1h": 2})
    assert result["complete"] and result["llm_input"] == 135
    assert result["estimated_usd"] == pytest.approx((100 * .75 + 20 * .075 + 10 * 1.25 + 5 * 2 + 2 * 4.5 + 5 * .042) / 1e6)
    missing_rate = token_cost(row, PRICES)
    assert missing_rate["estimated_usd"] is None and not missing_rate["complete"]
    del row["trace"][0]["proposal"]["usage"]["cache_creation"]
    unknown_lifetime = token_cost(row, {**PRICES, "llm_cache_write_5m": 1.25, "llm_cache_write_1h": 2})
    assert unknown_lifetime["estimated_usd"] is None
    assert unknown_lifetime["llm_cache_write_unknown"] == 15


def test_unknown_consumed_rate_is_not_zero_cost():
    result = token_cost(record(), {**PRICES, "llm_input": None})
    assert result["estimated_usd"] is None and not result["complete"]


def write_run(directory, rows, **overrides):
    manifest = {"ids": sorted({r["id"] for r in rows}), "replicates": 1, "methods": ["jev_joint"],
                "expected_records": len({r["id"] for r in rows}), "config_id": "test", "synthetic": False,
                "prices_usd_per_million": PRICES, "max_repairs": 3, "split": "test"}
    manifest.update(overrides)
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


@pytest.mark.parametrize("methods", [["unknown_method"], ["jev_joint", "jev_joint"], [], [42]])
def test_report_rejects_unsupported_or_duplicate_methods(tmp_path, methods):
    write_run(tmp_path, [record()], methods=methods)
    with pytest.raises(ValueError, match="unique supported methods"):
        build_summary(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_initial_diagnostics_do_not_include_final_cost_and_repairs(tmp_path):
    row = record()
    row["repairs"] = 3
    write_run(tmp_path, [row])
    initial = build_summary(tmp_path)["initial_candidate_diagnostics"]["jev_joint"]
    assert "mean_repairs" not in initial
    assert "mean_cost_usd" not in initial
    assert "mean_latency_s" not in initial


def test_paired_delta_uses_requested_baseline_and_clusters_repeated_runs():
    rows = []
    for name, replicate, main_correct, latest_correct in [
        ("a", 0, True, False), ("a", 1, True, False),
        ("b", 0, False, True), ("c", 0, False, False),
    ]:
        for method, correct in (("jev_joint", main_correct), ("jev_joint_latest", latest_correct), ("direct", True)):
            rows.append({**record(name, correct=correct), "method": method, "replicate": replicate})
    # Different repeats cannot be paired, and presence-only labels are excluded.
    rows += [{**record("unpaired", correct=True), "method": "jev_joint", "replicate": 0},
             {**record("unpaired", correct=False), "method": "jev_joint_latest", "replicate": 1},
             {**record("presence", category="live_relevance", correct=True), "method": "jev_joint"},
             {**record("presence", category="live_relevance", correct=False), "method": "jev_joint_latest"}]
    paired = paired_delta(rows, "jev_joint", samples=100, baseline="jev_joint_latest")
    assert paired["n_paired_ids"] == 3
    assert paired["n_paired_records"] == 4
    # Per-ID differences are +1, -1, and 0, not a +0.25 unclustered average.
    assert paired["delta"] == 0
    assert len(paired["ci95"]) == 2
    assert paired == paired_delta(rows, "jev_joint", samples=100, baseline="jev_joint_latest")


def test_paired_delta_default_baseline_and_positional_seed_remain_compatible():
    rows = [{**record(correct=False), "method": "direct"}, record(correct=True)]
    default = paired_delta(rows, "jev_joint", 17, 25)
    assert default == paired_delta(rows, "jev_joint", 17, 25, baseline="direct")
    assert default["delta"] == 1 and default["ci95"] == [1, 1]
    rows[0]["score"]["correct"] = None
    assert paired_delta(rows, "jev_joint")["delta"] is None


def history_pair(name="a", main_correct=True, latest_correct=False, replicate=0):
    rows = []
    initial = record(name, correct=False, accepted=False)["trace"][0]
    for method, correct in (("jev_joint", main_correct), ("jev_joint_latest", latest_correct)):
        row = record(name, correct=correct, max_repairs=2)
        row.update(method=method, replicate=replicate, initial_proposal_shared=True,
                   exhaustion_policy="execute_last", repairs=2)
        final = deepcopy(row["trace"][0])
        final["attempt"] = 2
        middle = deepcopy(initial)
        middle["attempt"] = 1
        row["trace"] = [deepcopy(initial), middle, final]
        rows.append(row)
    return rows


def test_summary_reports_observed_history_ablation_with_explicit_direction(tmp_path):
    rows = history_pair("a") + history_pair("b", main_correct=True, latest_correct=True)
    write_run(tmp_path, rows, methods=["jev_joint", "jev_joint_latest"], expected_records=4,
              max_repairs=2, exhaustion_policy="execute_last")
    summary = build_summary(tmp_path)
    contrast = summary["history_ablation"]
    assert summary["status"] == "complete"
    assert contrast["method"] == "jev_joint" and contrast["baseline"] == "jev_joint_latest"
    assert contrast["contrast"] == "JBFC minus JBFC-L"
    assert contrast["outcome"] == "final_call_accuracy"
    assert contrast["n_paired_ids"] == contrast["n_paired_records"] == 2
    assert contrast["delta"] == .5
    assert contrast["pairing"] == "example_id_and_replicate"
    assert contrast["excluded_category"] == "live_relevance"
    assert "jev_joint_latest" in summary["initial_candidate_diagnostics"]
    assert summary["paired_vs_direct"]["jev_joint_latest"]["delta"] is None
    assert json.loads((tmp_path / "summary.json").read_text())["history_ablation"] == contrast


@pytest.mark.parametrize("fault", ["not_shared", "initial_proposal", "initial_score", "repair_limit", "dispatch_policy"])
def test_history_contrast_rejects_unmatched_initial_or_run_conditions(tmp_path, fault):
    rows = history_pair()
    latest = rows[1]
    if fault == "not_shared":
        latest["initial_proposal_shared"] = False
    elif fault == "initial_proposal":
        latest["trace"][0]["proposal"]["calls"] = []
    elif fault == "initial_score":
        latest["trace"][0]["score"]["correct"] = True
    elif fault == "repair_limit":
        latest["max_repairs"] = 3
    else:
        latest["exhaustion_policy"] = "abstain"
    write_run(tmp_path, rows, methods=["jev_joint", "jev_joint_latest"], expected_records=2,
              max_repairs=2, exhaustion_policy="execute_last")
    with pytest.raises(ValueError, match="History ablation requires"):
        build_summary(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_partial_history_report_does_not_invent_unobserved_pairs(tmp_path):
    rows = history_pair()[:1]
    write_run(tmp_path, rows, methods=["jev_joint", "jev_joint_latest"], expected_records=2,
              max_repairs=2, exhaustion_policy="execute_last")
    summary = build_summary(tmp_path)
    assert summary["status"] == "partial"
    assert summary["history_ablation"]["n_paired_records"] == 0
    assert summary["history_ablation"]["delta"] is None
    assert summary["history_ablation"]["ci95"] is None


def test_budget_replay_uses_prefix_and_preserves_final_dispatch_policy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    row = record(correct=True, accepted=True, initial_correct=False)
    row["trace"][0]["gate_pass"] = False
    last = record(correct=True, accepted=True)["trace"][0]
    last["attempt"] = 1
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


def long_record(cap=5):
    row = record(correct=True, accepted=True, max_repairs=cap)
    row["trace"] = [deepcopy(row["trace"][0]) for _ in range(cap + 1)]
    for attempt, step in enumerate(row["trace"]):
        step["attempt"] = attempt
        step["gate_pass"] = attempt == cap
        step["score"]["correct"] = attempt >= 4
    row["repairs"] = cap
    return row


def test_replay_accepts_larger_limits_and_development_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    row = long_record()
    write_run(source, [row], max_repairs=5, split="dev", repair_budget_selection={"retained": "metadata"})
    before = (source / "records.jsonl").read_bytes()
    four = derive_policy_run(source, tmp_path / "four", max_repairs=4)
    full = derive_policy_run(source, tmp_path / "full")
    assert four["manifest"]["split"] == "dev"
    assert four["manifest"]["max_repairs"] == 4
    assert four["methods"]["jev_joint"]["mean_llm_proposals"] == 5
    assert four["methods"]["jev_joint"]["accuracy"] == 1
    assert four["methods"]["jev_joint"]["coverage"] == 0
    assert full["manifest"]["max_repairs"] == 5
    assert full["methods"]["jev_joint"]["coverage"] == 1
    assert full["manifest"]["repair_budget_selection"] == {"retained": "metadata"}
    assert (source / "records.jsonl").read_bytes() == before


def test_replay_keeps_only_actual_cumulative_jbfc_trajectories(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    main, latest = long_record(), long_record()
    latest["method"] = "jev_joint_latest"
    latest["trace"][0]["score"]["correct"] = True
    write_run(source, [main, latest], methods=["jev_joint", "jev_joint_latest"], expected_records=2,
              max_repairs=5)
    summary = derive_policy_run(source, tmp_path / "prefix", max_repairs=0)
    assert summary["manifest"]["methods"] == ["jev_joint"]
    assert set(summary["methods"]) == {"jev_joint"}
    assert summary["methods"]["jev_joint"]["accuracy"] == 0
    assert summary["history_ablation"] is None


def test_latest_only_run_cannot_supply_a_cumulative_jbfc_replay(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    latest = long_record()
    latest["method"] = "jev_joint_latest"
    write_run(source, [latest], methods=["jev_joint_latest"], max_repairs=5)
    with pytest.raises(ValueError, match="complete empirical joint run"):
        derive_policy_run(source, tmp_path / "prefix", max_repairs=0)
    assert not (tmp_path / "prefix").exists()


@pytest.mark.parametrize("limit", [-1, True, False, 2.5, "2", 6])
def test_replay_rejects_invalid_or_unobserved_limits(tmp_path, limit):
    source = tmp_path / "source"
    source.mkdir()
    write_run(source, [long_record()], max_repairs=5)
    with pytest.raises(ValueError):
        derive_policy_run(source, tmp_path / "output", max_repairs=limit)
    assert not (tmp_path / "output").exists()


def test_replay_never_overwrites_source_or_existing_results(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_run(source, [long_record()], max_repairs=5)
    before = (source / "records.jsonl").read_bytes()
    with pytest.raises(ValueError, match="differ"):
        derive_policy_run(source, source)
    output = tmp_path / "output"
    derive_policy_run(source, output)
    saved = (output / "records.jsonl").read_bytes()
    with pytest.raises(ValueError, match="not overwritten"):
        derive_policy_run(source, output, max_repairs=0)
    assert (source / "records.jsonl").read_bytes() == before
    assert (output / "records.jsonl").read_bytes() == saved


def test_repeated_suite_replays_reuse_identical_prefix_and_abstain_evidence(tmp_path):
    source = tmp_path / "test_selected"
    source.mkdir()
    write_run(source, [long_record()], max_repairs=5)
    jobs = [(tmp_path / "test_prefix_diagnostics" / f"budget_{limit}", {"max_repairs": limit})
            for limit in range(6)]
    jobs.append((tmp_path / "abstain", {"exhaustion_policy": "abstain"}))
    first = {str(path): derive_policy_run(source, path, **options) for path, options in jobs}
    saved = {str(path): {name: ((path / name).read_bytes(), (path / name).stat().st_mtime_ns)
                        for name in ("manifest.json", "records.jsonl")} for path, _ in jobs}
    # The second suite pass refreshes summaries but cannot rewrite evidence.
    for path, options in jobs:
        assert derive_policy_run(source, path, **options) == first[str(path)]
        for name, (data, timestamp) in saved[str(path)].items():
            assert (path / name).read_bytes() == data
            assert (path / name).stat().st_mtime_ns == timestamp


@pytest.mark.parametrize("changed", ["source", "derived_records", "derived_manifest"])
def test_replay_resume_rejects_changed_or_corrupt_evidence(tmp_path, changed):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    write_run(source, [long_record()], max_repairs=5)
    derive_policy_run(source, output, max_repairs=2)
    if changed == "source":
        path = source / "records.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    elif changed == "derived_records":
        path = output / "records.jsonl"
        row = json.loads(path.read_text())
        row["score"]["correct"] = True
        path.write_text(json.dumps(row) + "\n")
    else:
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["max_repairs"] = 1
        path.write_text(json.dumps(manifest))
    before = {name: (output / name).read_bytes() for name in ("manifest.json", "records.jsonl")}
    with pytest.raises(ValueError, match="not overwritten"):
        derive_policy_run(source, output, max_repairs=2)
    assert all((output / name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize("fault", ["duplicate", "foreign_config", "missing_method", "missing_tail", "early_pass"])
def test_replay_rejects_corrupt_or_incomplete_source_before_writing(tmp_path, fault):
    source = tmp_path / "source"
    source.mkdir()
    row = long_record()
    rows, overrides = [row], {"max_repairs": 5}
    if fault == "duplicate":
        rows.append(deepcopy(row))
    elif fault == "foreign_config":
        row["config_id"] = "other"
    elif fault == "missing_method":
        overrides.update(methods=["direct", "jev_joint"], expected_records=2)
    elif fault == "missing_tail":
        row["trace"].pop()
        row["repairs"] -= 1
        row["accepted"] = False
    else:
        row["trace"][0]["gate_pass"] = True
    write_run(source, rows, **overrides)
    with pytest.raises(ValueError):
        derive_policy_run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_main_table_lists_jbfc_last_below_its_variants():
    from jev_fc.pipeline import METHODS
    from jev_fc.reporting import TABLE_METHOD_ORDER

    assert sorted(TABLE_METHOD_ORDER) == sorted(METHODS)
    rows = [line.split(" & ")[0] for line in latex_table().splitlines() if " & " in line][1:]
    assert rows == ["Direct", "Schema repair", "Self-refine", r"\textbf{JBFC-F}", r"\textbf{JBFC-A}",
                    r"\textbf{JBFC-L}", r"\textbf{JBFC}"]


def paper_template(path):
    path.write_text("\n".join(
        f"% BEGIN AUTO {marker}\nold {marker}\n% END AUTO {marker}"
        for marker in ("RESULTS", "CATEGORIES")
    ))


def table1_summary(**changes):
    path = Path("data/bfcl/examples.jsonl")
    if not path.exists():
        pytest.skip("Requires prepared BFCL files")
    manifest = {**PAPER_SETTINGS, "ids": [row["id"] for row in load_examples(path)],
                "methods": ["direct", "jev_joint"], "prices_usd_per_million": dict(PAPER_PRICES),
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **changes}
    return {"status": "complete", "manifest": manifest,
            "methods": {"direct": {"accuracy": .9}, "jev_joint": {"accuracy": .95}},
            "by_category": {"simple_python/direct": {"accuracy": .908}, "simple_python/jev_joint": {"accuracy": .925},
                            "live_relevance/direct": {"relevance_call_rate": .5}}}


def test_filled_table1_drops_the_empty_cell_note():
    summary = table1_summary()
    summary["by_category"] = {f"{category}/{method}": {"accuracy": .5, "relevance_call_rate": .5}
                              for category, _, _ in TABLE1_CATEGORIES for method in ("direct", "jev_joint")}
    table = latex_category_table(summary)
    assert "--" not in table and "has not been run" not in table
    # Tied rows stay plain; only the strictly higher total is bold.
    assert table.count("50.0") == 14 and r"\textbf{50.0}" not in table
    assert r"Total & 3051 & 90.0 & \textbf{95.0}\\" in table
    assert "Bold marks the higher accuracy when the two differ." in table
    assert r"total accuracy excludes these 16 cases.}" in table


@pytest.mark.parametrize("direct,jbfc,row", [
    (.95, .9, r"Simple Python & 400 & \textbf{95.0} & 90.0\\"),
    (.9, .95, r"Simple Python & 400 & 90.0 & \textbf{95.0}\\"),
    (.9124, .9121, r"Simple Python & 400 & 91.2 & 91.2\\"),  # equal as shown: no bold
    (.9, None, r"Simple Python & 400 & 90.0 & --\\"),
], ids=["direct_higher", "jbfc_higher", "equal_as_shown", "missing"])
def test_table1_bolds_only_the_strictly_higher_shown_accuracy(direct, jbfc, row):
    summary = table1_summary()
    summary["methods"] = {"direct": {"accuracy": .5}, "jev_joint": {"accuracy": .5}}
    summary["by_category"] = {"simple_python/direct": {"accuracy": direct}, "simple_python/jev_joint": {"accuracy": jbfc}}
    table = latex_category_table(summary)
    assert row in table and r"Total & 3051 & 50.0 & 50.0\\" in table
    assert ("Bold marks" in table) == (r"\textbf{9" in table)


def test_unrun_table1_lists_all_examples_without_a_test_split():
    table = latex_category_table()
    assert r"Category & Examples & Direct & \textbf{JBFC}\\" in table
    assert r"Simple Python & 400 & -- & --\\" in table and r"Total & 3051 & -- & --\\" in table
    assert "Test" not in table and "2748" not in table
    assert "$r=2$" in table and "these 16 cases" in table and "Generator" not in table
    assert "-- means the experiment has not been run." in table and "Bold" not in table


def test_table1_fills_only_categories_from_a_complete_all_example_run(tmp_path):
    paper = tmp_path / "paper.tex"
    paper_template(paper)
    update_table1(paper, table1_summary())
    source = paper.read_text()
    table = source.split("% BEGIN AUTO CATEGORIES")[1].split("% END AUTO CATEGORIES")[0]
    assert r"Simple Python & 400 & 90.8 & \textbf{92.5}\\" in table
    assert r"Live relevance$^\dagger$ & 16 & 50.0 & --\\" in table
    assert r"Total & 3051 & 90.0 & \textbf{95.0}\\" in table and r"Generator: \texttt{gpt-5-nano}." in table
    assert "-- means the experiment has not been run." in table  # some cells are still empty
    assert "old RESULTS" in source  # Table 2 needs all seven methods
    with pytest.raises(ValueError, match="complete run"):
        update_table1(paper, {**table1_summary(), "status": "partial"})


OTHER_PROTOCOLS = [{"split": "test"}, {"split": "dev"}, {"limit": 5}, {"max_repairs": 3}, {"max_repairs": True},
                   {"replicates": 3}, {"replicates": True}, {"threshold": .9}, {"exhaustion_policy": "abstain"},
                   {"synthetic": True}, {"llm_model": "gpt-5.4-mini-2026-03-17"}, {"reasoning_effort": "low"},
                   {"llm_api_style": "responses"}, {"max_completion_tokens": 8192},
                   {"prices_usd_per_million": {"llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5,
                                               "jev_input": .042}},
                   {"ids": ["simple_python_0"]}, {"dataset_sha256": "tampered"}]


@pytest.mark.parametrize("change", OTHER_PROTOCOLS + [{"methods": ["direct"]}])
def test_table1_rejects_other_protocols_without_writing(tmp_path, change):
    paper = tmp_path / "paper.tex"
    paper_template(paper)
    original = paper.read_bytes()
    with pytest.raises(ValueError):
        update_table1(paper, table1_summary(**change))
    assert paper.read_bytes() == original


def test_table_separates_call_function_and_conditional_argument_accuracy():
    table = latex_table({"methods": {"jev_joint": {"accuracy": .6, "function_accuracy": .8,
                                                   "arguments_accuracy_conditional": .75},
                                     "jev_joint_latest": {"accuracy": .5, "function_accuracy": .7,
                                                          "arguments_accuracy_conditional": .65}}})
    assert r"\textbf{JBFC} & \textbf{60.0} & \textbf{80.0} & \textbf{75.0}" in table
    assert r"\textbf{JBFC-L} & 50.0 & 70.0 & 65.0" in table
    assert "Argument accuracy" in table and "correct function and argument labels" in table



def test_unrun_main_table_states_the_fixed_protocol():
    table = latex_table()
    assert "Methods on all BFCL examples, from one run with repair limit $r=2$." in table
    assert "development" not in table and "test set" not in table
    assert "Generator" not in table and "Bold" not in table and "-- means unavailable or unrun." in table
    assert r"\textbf{JBFC-L} & -- & -- & -- & -- & --" in table


def test_main_table_bolds_the_highest_accuracy_in_each_column():
    methods = {"direct": {"accuracy": .5, "function_accuracy": .9, "mean_latency_s": 1.0, "mean_cost_usd": 1e-4},
               "jev_joint": {"accuracy": .7, "function_accuracy": .9, "mean_latency_s": 3.0, "mean_cost_usd": 2e-4}}
    table = latex_table({"methods": methods})
    assert r"Direct & 50.0 & \textbf{90.0} & -- & 1.00 & 0.100 \\" in table  # tied highest: both bold
    assert r"\textbf{JBFC} & \textbf{70.0} & \textbf{90.0} & -- & 3.00 & 0.200 \\" in table
    assert "Bold marks the highest accuracy in each column." in table


def test_old_test_split_protocol_cannot_match_paper():
    from jev_fc.pipeline import METHODS
    old = {**PAPER_SETTINGS, "split": "test", "replicates": 3, "methods": list(METHODS),
           "prices_usd_per_million": dict(PAPER_PRICES)}
    assert not paper_protocol_matches(old)
    assert paper_protocol_matches({**old, "split": "all", "replicates": 1})


def paper_summary(**changes):
    from jev_fc.pipeline import METHODS
    summary = table1_summary(**{"methods": list(METHODS), **changes})
    summary["methods"] = {method: {"accuracy": .5, "function_accuracy": .6, "arguments_accuracy_conditional": .7,
                                   "mean_latency_s": 2.0, "mean_cost_usd": 1e-4} for method in METHODS}
    summary["methods"]["direct"]["accuracy"] = .9
    summary["methods"]["jev_joint"]["accuracy"] = .95
    return summary


def test_seven_method_run_fills_tables_1_and_2_from_the_same_run(tmp_path):
    paper = tmp_path / "paper.tex"
    paper_template(paper)
    update_paper(paper, paper_summary())
    source = paper.read_text()
    main = source.split("% BEGIN AUTO RESULTS")[1].split("% END AUTO RESULTS")[0]
    categories = source.split("% BEGIN AUTO CATEGORIES")[1].split("% END AUTO CATEGORIES")[0]
    assert r"Direct & 90.0 &" in main and r"\textbf{JBFC} & \textbf{95.0} &" in main
    assert r"Generator: \texttt{gpt-5-nano}." in main and "$r=2$" in main
    assert r"Total & 3051 & 90.0 & \textbf{95.0}\\" in categories
    update_paper(paper)  # without a summary both tables return to their unrun form
    source = paper.read_text()
    assert r"Direct & -- & -- & -- & -- & -- \\" in source and r"Total & 3051 & -- & --\\" in source


@pytest.mark.parametrize("change", OTHER_PROTOCOLS + [{"methods": ["direct", "jev_joint"]}])
def test_paper_rejects_other_protocols_without_writing(tmp_path, change):
    paper = tmp_path / "paper.tex"
    paper_template(paper)
    original = paper.read_bytes()
    with pytest.raises(ValueError):
        update_paper(paper, paper_summary(**change))
    with pytest.raises(ValueError, match="complete run"):
        update_paper(paper, {**paper_summary(), "status": "partial"})
    assert paper.read_bytes() == original


def test_report_command_routes_seven_method_runs_to_both_tables(monkeypatch):
    from jev_fc import cli
    from jev_fc.pipeline import METHODS

    calls = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "update_table1", lambda paper, summary: calls.append(("table1", len(summary["manifest"]["methods"]))))
    monkeypatch.setattr(cli, "update_paper", lambda paper, summary: calls.append(("paper", len(summary["manifest"]["methods"]))))
    for methods in (["direct", "jev_joint"], list(METHODS)):
        monkeypatch.setattr(cli, "build_summary", lambda _, methods=methods: {"status": "complete", "methods": {},
                                                                              "manifest": {"methods": methods}})
        monkeypatch.setattr("sys.argv", ["jev-fc", "report", "run-dir", "--paper", "paper.tex"])
        assert cli.main() == 0
    assert calls == [("table1", 2), ("paper", 7)]
