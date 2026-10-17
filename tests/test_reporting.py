"""Metric denominators and counterfactual stopping policies are research-critical."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from jev_fc.benchmarks import load_examples, stratified_split
from jev_fc.reporting import (
    PAPER_REPAIR_GRID,
    PAPER_SETTINGS,
    build_summary,
    derive_policy_run,
    latex_budget_table,
    latex_table,
    paper_protocol_matches,
    summarize_group,
    token_cost,
    update_paper,
)


def record(name="a", category="simple_python", correct=False, accepted=True, initial_correct=None, max_repairs=3):
    if initial_correct is None:
        initial_correct = correct
    score = {"correct": correct, "function_correct": correct, "arguments_correct": correct if correct else None}
    trace = [{"attempt": 0, "proposal": {"calls": [], "latency_s": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
              "verification": {"function_probability": .7, "arguments_probability": .9,
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


PRICES = {"llm_input": .75, "llm_output": 4.5, "llm_cached_input": .075, "jev_input": .042}


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


def test_unrun_development_table_has_no_selected_limit_or_imputed_values():
    table = latex_budget_table()
    assert r"$r$ & Dev Acc. (\%) & Repairs & Sec. & USD/1k" in table
    for limit in PAPER_REPAIR_GRID:
        assert f"${limit}$ & -- & -- & -- & --" in table
    assert "marks the selected limit" not in table
    assert "^{*}" not in table
    assert r"\label{tab:diagnostics}" in table
    assert "Development data choose" in table
    assert "highest call accuracy" in table and "ties choose the smaller limit" in table


def test_main_table_does_not_invent_a_selected_limit():
    table = latex_table()
    assert "chosen on development data" in table
    assert "$r=" not in table


def test_raw_test_manifest_without_selection_cannot_match_paper():
    from jev_fc.pipeline import METHODS
    manifest = {**PAPER_SETTINGS, "split": "test", "limit": None,
                "methods": list(METHODS), "max_repairs": 5, "prices_usd_per_million": PRICES}
    assert not paper_protocol_matches(manifest)


@pytest.fixture
def selected_development_run(tmp_path):
    # Reuse the tuning tests' real canonical-ID fixture and official scoring.
    # Its mock proposals are confined to tmp_path and never become experiments.
    from test_tuning import make_test_config, write_dev_fixture
    from jev_fc.tuning import select_repair_budget

    source = tmp_path / "development"
    manifest, _, test = write_dev_fixture(source, replicates=3)
    selection = select_repair_budget(source)
    test_manifest = make_test_config(manifest, selection, test)
    test_manifest["repair_budget_selection"] = selection
    return source, selection, test_manifest


def paper_template(path):
    path.write_text("\n".join(
        f"% BEGIN AUTO {marker}\nold {marker}\n% END AUTO {marker}"
        for marker in ("RESULTS", "CATEGORIES", "DIAGNOSTICS")
    ))


def test_valid_selection_fills_dev_table_without_using_dev_accuracy_as_test_result(selected_development_run, tmp_path):
    _, selection, manifest = selected_development_run
    assert selection["selected_max_repairs"] == 2
    assert paper_protocol_matches(manifest)
    paper = tmp_path / "paper.tex"
    paper_template(paper)
    summary = {"status": "complete", "manifest": manifest,
               "methods": {"direct": {"accuracy": .125}, "jev_joint": {"accuracy": .75}}}
    update_paper(paper, summary)
    source = paper.read_text()
    main = source.split("% BEGIN AUTO RESULTS")[1].split("% END AUTO RESULTS")[0]
    dev = source.split("% BEGIN AUTO DIAGNOSTICS")[1].split("% END AUTO DIAGNOSTICS")[0]
    dev_accuracy = next(item["accuracy"] for item in selection["candidates"] if item["max_repairs"] == 2)
    assert r"\textbf{JBFC} & 75.0" in main
    assert "$r=2$" in main
    assert f"$2^{{*}}$ & {dev_accuracy * 100:.1f}" in dev
    assert "$3^{*}$" not in dev
    assert "Dev Acc." in dev and "Accuracy excludes live relevance" in dev


def test_dev_table_uses_actual_valid_grid_but_custom_grid_cannot_fill_default_paper(selected_development_run):
    from jev_fc.tuning import select_repair_budget

    source, _, manifest = selected_development_run
    selection = select_repair_budget(source, [0, 2, 5])
    table = latex_budget_table(selection)
    assert "$0$ &" in table and "$2^{*}$ &" in table and "$5$ &" in table
    assert "$1$ &" not in table and "$3$ &" not in table and "$4$ &" not in table
    manifest["repair_budget_selection"] = selection
    assert not paper_protocol_matches(manifest)


def test_paper_rejects_manual_incompatible_forged_and_stale_selections(selected_development_run, tmp_path):
    from jev_fc.tuning import _fingerprint

    source, selection, manifest = selected_development_run
    paper = tmp_path / "paper.tex"
    paper_template(paper)
    original = paper.read_bytes()

    def rejects(candidate):
        assert not paper_protocol_matches(candidate)
        with pytest.raises(ValueError, match="declared protocol"):
            update_paper(paper, {"status": "complete", "manifest": candidate})
        assert paper.read_bytes() == original

    missing = deepcopy(manifest)
    missing.pop("repair_budget_selection")
    rejects(missing)
    manual = deepcopy(manifest)
    manual["max_repairs"] = 3
    rejects(manual)
    incompatible = deepcopy(manifest)
    incompatible["workers"] += 1
    rejects(incompatible)
    forged = deepcopy(manifest)
    forged_selection = forged["repair_budget_selection"]
    forged_selection["candidates"][0]["accuracy"] = .999
    # Even a newly computed outer hash cannot replace source verification.
    forged_selection["selection_id"] = _fingerprint({k: v for k, v in forged_selection.items() if k != "selection_id"})
    rejects(forged)

    records = source / "records.jsonl"
    records.write_bytes(records.read_bytes() + b"\n")
    rejects(manifest)
    with pytest.raises(ValueError, match="stale|source evidence"):
        latex_budget_table(selection)


def test_development_summary_cannot_populate_test_results(selected_development_run, tmp_path):
    source, _, _ = selected_development_run
    development = build_summary(source)
    paper = tmp_path / "paper.tex"
    paper_template(paper)
    original = paper.read_bytes()
    with pytest.raises(ValueError, match="complete test runs"):
        update_paper(paper, development)
    assert paper.read_bytes() == original


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


def test_table_separates_call_function_and_conditional_argument_accuracy():
    table = latex_table({"methods": {"jev_joint": {"accuracy": .6, "function_accuracy": .8,
                                                   "arguments_accuracy_conditional": .75}}})
    assert r"\textbf{JBFC} & 60.0 & 80.0 & 75.0" in table
    assert "Argument accuracy" in table and "correct function and argument labels" in table
