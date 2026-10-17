"""Offline fixture traces exercise dev selection; no API experiment is run."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from jev_fc import __version__
from jev_fc.benchmarks import BFCL_REVISION, load_examples, stratified_split
from jev_fc.jev import validate_calls
from jev_fc.pipeline import METHODS, run_example
from jev_fc.scoring import score_example
from jev_fc.tuning import (
    BUDGET_SELECTION_METHOD,
    DEFAULT_REPAIR_GRID,
    _fingerprint,
    _source_hash,
    load_repair_budget_selection,
    main,
    select_repair_budget,
    validate_repair_budget_selection,
    validate_repair_grid,
)

DATA = Path(__file__).resolve().parents[1] / "data/bfcl/examples.jsonl"
PRICES = {"llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5,
          "jev_input": .042, "llm_cache_write_5m": None, "llm_cache_write_1h": None}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_rows(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def refresh_manifest(directory, manifest):
    manifest["config_id"] = _fingerprint({key: value for key, value in manifest.items()
                                          if key not in {"config_id", "created_at"}})
    _write_json(directory / "manifest.json", manifest)
    rows = _read_rows(directory / "records.jsonl")
    for row in rows:
        row["config_id"] = manifest["config_id"]
    _write_rows(directory / "records.jsonl", rows)


def write_dev_fixture(directory, *, max_repairs=5, replicates=1):
    """Write offline mock traces over all real canonical dev IDs.

    No-call categories become correct at r=2 and regress at r=4; one stops early
    at r=1. Presence-only labels favor r=0 and must be excluded. This helper is
    shared with report guard tests, never with experiment code.
    """
    if not DATA.exists():
        pytest.skip("Requires prepared, pinned BFCL data (no network in tests)")
    directory.mkdir(parents=True, exist_ok=True)
    development, test = stratified_split(load_examples(DATA), seed=1729)
    manifest = {
        "version": __version__, "source_sha256": _source_hash(),
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(), "dataset_revisions": [BFCL_REVISION],
        "ids": [e["id"] for e in development], "split": "dev", "split_seed": 1729, "limit": None,
        "methods": ["jev_joint"], "replicates": replicates, "max_repairs": max_repairs,
        "threshold": .8, "exhaustion_policy": "execute_last",
        "llm_model": "gpt-5.4-mini-2026-03-17", "llm_base_url": "https://fixture.invalid/llm/openai/v1",
        "llm_provider": "openai", "llm_api_style": "responses", "llm_gateway": "apim",
        "llm_endpoint": "https://fixture.invalid/llm/openai/v1/responses", "llm_auth_header": "api-key",
        "llm_timeout_s": 120, "llm_max_retries": 2, "reasoning_effort": "none", "temperature": None,
        "max_completion_tokens": 4096, "jev_model": "jev-1.13.0",
        "jev_base_url": "https://fixture.invalid/llm/typesafe", "jev_gateway": "apim",
        "jev_endpoint": "https://fixture.invalid/llm/typesafe/v1/systemone", "jev_timeout_s": 60,
        "jev_max_retries": 2, "workers": 4, "prices_usd_per_million": PRICES,
        "expected_records": len(development) * replicates, "synthetic": False,
    }
    manifest["config_id"] = _fingerprint(manifest)
    manifest["created_at"] = "offline-test-fixture"
    early = next(e["id"] for e in development if e["category"] == "irrelevance")
    rows, initials = [], []
    for replicate in range(replicates):
        for example in development:
            proposals = []
            for attempt in range(max_repairs + 1):
                calls = [] if attempt in {2, 3} or (example["id"] == early and attempt >= 1) else [
                    {"name": "fixture_unknown_function", "arguments": {}}]
                proposals.append({"calls": calls, "latency_s": 1.0,
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "api_calls": 1})

            class Generator:
                def __init__(self):
                    self.proposals = iter(proposals[1:])

                def generate(self, *_):
                    return deepcopy(next(self.proposals))

            class Verifier:
                def verify(self, public, calls):
                    local = validate_calls(public, calls)
                    p = .9 if example["id"] == early and not calls else .1
                    skipped = calls == []
                    return {"function_probability": p, "arguments_probability": None if skipped else p,
                            "arguments_skipped": skipped,
                            "function_pass": local["function_pass"] and p >= .8,
                            "arguments_pass": None if skipped else local["arguments_pass"] and p >= .8,
                            "feedback": {"source": "offline-fixture"}, "api_calls": 1,
                            "usage": {"input_tokens": 5}, "latency_s": .25}

            row = run_example(example, Generator(), Verifier(), max_repairs=max_repairs,
                              initial_proposal=proposals[0])
            for step in row["trace"]:
                step["score"] = score_example(example, step["proposal"]["calls"], checker_path=DATA.parent / "upstream/ast_checker.py")
            row.update(replicate=replicate, synthetic=False, config_id=manifest["config_id"],
                       score=row["trace"][-1]["score"])
            rows.append(row)
            initials.append({"id": example["id"], "replicate": replicate, "proposal": proposals[0]})
    _write_json(directory / "manifest.json", manifest)
    _write_rows(directory / "records.jsonl", rows)
    _write_rows(directory / "initial.jsonl", initials)
    return manifest, rows, test


@pytest.fixture
def dev_run(tmp_path):
    directory = tmp_path / "dev"
    manifest, rows, test = write_dev_fixture(directory)
    return directory, manifest, rows, test


def test_full_dev_selection_exact_ties_early_stop_and_presence_exclusion(dev_run):
    directory, manifest, _, _ = dev_run
    before = {name: (directory / name).read_bytes() for name in ("manifest.json", "records.jsonl", "initial.jsonl")}
    result = select_repair_budget(directory)
    assert result["grid"] == list(DEFAULT_REPAIR_GRID)
    assert result["selected_max_repairs"] == 2
    assert result["selection_rule"] == "max_dev_accuracy_then_smallest_budget"
    assert result["selection_method"] == BUDGET_SELECTION_METHOD == "jev_joint"
    candidates = {item["max_repairs"]: item for item in result["candidates"]}
    assert [candidates[r]["correct"] for r in range(6)] == [0, 1, 112, 112, 1, 1]
    assert all(item["n"] == 303 and item["exact_n"] == 302 for item in candidates.values())
    assert candidates[2]["accuracy"] == pytest.approx(112 / 302)
    assert candidates[5]["mean_llm_proposals"] == pytest.approx((302 * 6 + 2) / 303)
    assert candidates[2]["mean_cost_usd"] > candidates[1]["mean_cost_usd"] > 0
    assert candidates[2]["mean_latency_s"] > candidates[1]["mean_latency_s"] > 0
    assert result["provenance"]["settings"]["threshold"] == manifest["threshold"]
    assert result == select_repair_budget(directory)
    assert all((directory / name).read_bytes() == value for name, value in before.items())
    assert not (directory / "summary.json").exists()


def test_arbitrary_grid_and_replicate_denominators(tmp_path):
    directory = tmp_path / "dev"
    write_dev_fixture(directory, max_repairs=7, replicates=2)
    result = select_repair_budget(directory, [7, 3, 0, 2])
    assert result["grid"] == [0, 2, 3, 7] and result["selected_max_repairs"] == 2
    assert result["candidates"][1]["exact_n"] == 604
    assert result["candidates"][1]["correct"] == 224
    assert set(result["candidates"][1]["by_replicate"]) == {"0", "1"}


@pytest.mark.parametrize("grid", [[], [1, 1], [-1], [True], [False], [1.0], [float("inf")],
                                  [float("nan")], ["2"], "0,1", None])
def test_invalid_grids_are_rejected(grid):
    with pytest.raises(ValueError):
        validate_repair_grid(grid)


@pytest.mark.parametrize("change", [
    {"split": "test"}, {"split": "all"}, {"limit": 2}, {"synthetic": True}, {"derived_from": "elsewhere"},
    {"exhaustion_policy": "abstain"}, {"source_sha256": "changed"}, {"dataset_sha256": "changed"},
    {"split_seed": 17}, {"expected_records": 302}, {"methods": ["direct"]}, {"methods": ["schema_only"]},
])
def test_selection_rejects_wrong_source_protocol(dev_run, change):
    directory, manifest, _, _ = dev_run
    manifest.update(change)
    refresh_manifest(directory, manifest)
    with pytest.raises(ValueError):
        select_repair_budget(directory)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "synthetic", "bad_config", "wrong_category"])
def test_selection_rejects_incomplete_or_foreign_record_grid(dev_run, mutation):
    directory, _, rows, _ = dev_run
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "foreign":
        rows[0]["id"] = "foreign_test_id"
    elif mutation == "synthetic":
        rows[0]["synthetic"] = True
    elif mutation == "bad_config":
        rows[0]["config_id"] = "other"
    else:
        rows[0]["category"] = "live_relevance"
    _write_rows(directory / "records.jsonl", rows)
    with pytest.raises(ValueError):
        select_repair_budget(directory)


@pytest.mark.parametrize("mutation", ["score", "short", "gate", "threshold", "attempt", "initial"])
def test_selection_rechecks_scores_and_trajectory_integrity(dev_run, mutation):
    directory, _, rows, _ = dev_run
    row = next(row for row in rows if row["repairs"] == 5)
    if mutation == "score":
        row["trace"][0]["score"]["correct"] = True
    elif mutation == "short":
        row["trace"].pop()
    elif mutation == "gate":
        row["trace"][0]["gate_pass"] = True
    elif mutation == "threshold":
        row["trace"][0]["verification"]["function_pass"] = True
    elif mutation == "attempt":
        row["trace"][1]["attempt"] = 4
    else:
        row["trace"][0]["proposal"]["content"] = "changed"
    _write_rows(directory / "records.jsonl", rows)
    with pytest.raises(ValueError):
        select_repair_budget(directory)


@pytest.mark.parametrize("empty,mutation", [
    (True, {"arguments_skipped": False}), (True, {"arguments_skipped": 1}),
    (True, {"arguments_probability": .95}), (True, {"arguments_pass": True}),
    (True, {"arguments_pass": False}), (False, {"arguments_skipped": True}),
    (False, {"arguments_skipped": 0}), (False, {"arguments_pass": None}),
])
def test_selection_rejects_forged_skipped_argument_checks(dev_run, empty, mutation):
    directory, _, rows, _ = dev_run
    step = next(step for row in rows for step in row["trace"]
                if (step["proposal"]["calls"] == []) is empty)
    step["verification"].update(mutation)
    _write_rows(directory / "records.jsonl", rows)
    with pytest.raises(ValueError, match="skip status|Verifier decisions"):
        select_repair_budget(directory)


def test_selected_prefix_preserves_no_call_skip_and_function_rejection(dev_run):
    directory, _, rows, _ = dev_run
    result = select_repair_budget(directory)
    no_calls = [step for row in rows for step in row["trace"] if step["proposal"]["calls"] == []]
    assert no_calls
    assert all(step["verification"]["arguments_skipped"] is True for step in no_calls)
    assert all(step["verification"]["arguments_pass"] is None for step in no_calls)
    assert all(step["verification"]["arguments_probability"] is None for step in no_calls)
    assert any(step["gate_pass"] is True for step in no_calls)
    assert any(step["gate_pass"] is False for step in no_calls)
    assert result["selected_max_repairs"] == 2


def test_missing_optional_provider_fields_produce_unknown_cost_or_latency(dev_run):
    directory, _, rows, _ = dev_run
    # Missing model/usage/timing must not invalidate an otherwise complete trace.
    rows[0]["trace"][1]["proposal"].pop("usage")
    rows[0]["trace"][1]["proposal"].pop("latency_s")
    _write_rows(directory / "records.jsonl", rows)
    result = select_repair_budget(directory)
    assert result["candidates"][1]["mean_cost_usd"] is None
    assert result["candidates"][1]["mean_latency_s"] is None
    assert result["selected_max_repairs"] == 2


def make_test_config(manifest, selection, test):
    config = {key: deepcopy(value) for key, value in manifest.items() if key not in {"config_id", "created_at"}}
    config.update(split="test", ids=[row["id"] for row in test], max_repairs=selection["selected_max_repairs"],
                  methods=list(METHODS),
                  expected_records=len(test) * manifest["replicates"] * len(METHODS), repair_budget_selection=selection)
    return config


def test_saved_artifact_validation_and_compatible_test_config(dev_run):
    directory, manifest, _, test = dev_run
    selection = select_repair_budget(directory)
    path = directory / "repair_budget_selection.json"
    _write_json(path, selection)
    config = make_test_config(manifest, selection, test)
    assert config["methods"] == list(METHODS) and len(METHODS) == 7
    assert "jev_joint_latest" in config["methods"]
    assert config["expected_records"] == 2748 * manifest["replicates"] * 7
    assert load_repair_budget_selection(path, test_config=config) == selection
    assert validate_repair_budget_selection(selection) == selection


@pytest.mark.parametrize("method", [None, "schema_only", "self_refine", "jev_joint_latest"])
def test_selection_must_explicitly_use_jbfc_before_source_read(method):
    selection = {"kind": "repair_budget_selection", "selection_method": method,
                 "source_dir": "/nonexistent-source-must-not-be-read"}
    with pytest.raises(ValueError, match="based on jev_joint"):
        validate_repair_budget_selection(selection)


@pytest.mark.parametrize("field,value", [("threshold", .9), ("workers", 1), ("llm_api_style", "anthropic_messages"),
                                        ("replicates", 2), ("max_repairs", 3), ("split", "dev"), ("limit", 1)])
def test_selected_budget_cannot_transfer_to_different_test_settings(dev_run, field, value):
    directory, manifest, _, test = dev_run
    selection = select_repair_budget(directory)
    config = make_test_config(manifest, selection, test)
    config[field] = value
    with pytest.raises(ValueError):
        validate_repair_budget_selection(selection, test_config=config)


def test_forged_artifact_or_changed_source_is_rejected(dev_run):
    directory, manifest, rows, _ = dev_run
    selection = select_repair_budget(directory)
    forged = deepcopy(selection)
    forged["selected_max_repairs"] = 5
    with pytest.raises(ValueError, match="fingerprint"):
        validate_repair_budget_selection(forged)
    forged["selection_id"] = _fingerprint({k: v for k, v in forged.items() if k != "selection_id"})
    with pytest.raises(ValueError, match="stale|source"):
        validate_repair_budget_selection(forged)
    rows[0]["trace"][-1]["proposal"]["usage"]["completion_tokens"] += 1
    _write_rows(directory / "records.jsonl", rows)
    with pytest.raises(ValueError, match="stale|source"):
        validate_repair_budget_selection(selection)
    manifest["created_at"] = "changed provenance"
    _write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        validate_repair_budget_selection(selection)


def test_no_case_subset_or_longer_grid_can_be_selected(dev_run):
    directory, manifest, _, _ = dev_run
    with pytest.raises(ValueError, match="exceeds"):
        select_repair_budget(directory, [0, 6])
    manifest["ids"] = manifest["ids"][:-1]
    refresh_manifest(directory, manifest)
    with pytest.raises(ValueError, match="303"):
        select_repair_budget(directory)


def test_budget_tuning_cli_writes_artifact_and_protects_evidence(dev_run):
    directory, _, _, _ = dev_run
    assert main([str(directory), "--grid", "5,2,0"]) == 0
    path = directory / "repair_budget_selection.json"
    assert load_repair_budget_selection(path)["selected_max_repairs"] == 2
    for name in ("manifest.json", "records.jsonl", "initial.jsonl"):
        original = (directory / name).read_bytes()
        with pytest.raises(SystemExit):
            main([str(directory), "--output", str(directory / name)])
        assert (directory / name).read_bytes() == original
