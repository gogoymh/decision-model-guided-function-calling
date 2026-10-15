import importlib.util
import json
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location("calibrate_threshold", Path(__file__).parents[1] / "scripts/calibrate_threshold.py")
calibration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibration)


def write_run(directory, candidates, **overrides):
    """Write tiny wire-format fixtures. They are never real reported experiments."""
    records = []
    for index, candidate in enumerate(candidates):
        function_probability, arguments_probability, correct, schema_pass = candidate[:4]
        category = candidate[4] if len(candidate) > 4 else "simple_python"
        score = {"correct": correct}
        records.append({
            "id": str(index), "replicate": 0, "method": "jev_joint", "category": category,
            "config_id": "config", "synthetic": False, "max_repairs": 0, "repairs": 0, "score": score,
            "trace": [{"attempt": 0, "proposal": {"model": "generator-v1"},
                       "verification": {"function_probability": function_probability,
                                        "arguments_probability": arguments_probability, "model": "jev-1.13.0"},
                       "schema": {"accepted": schema_pass}, "score": score}],
        })
    manifest = {"split": "dev", "limit": None, "max_repairs": 0, "synthetic": False,
                "ids": [row["id"] for row in records], "replicates": 1, "methods": ["jev_joint"],
                "expected_records": len(records), "config_id": "config", "split_seed": 1729,
                "llm_model": "generator-v1", "jev_model": "jev-1.13.0", "threshold": .8,
                "dataset_sha256": "dataset-sha", "dataset_revisions": ["revision"], "source_sha256": "source-sha"}
    manifest.update(overrides)
    directory.mkdir(exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return records, manifest


def test_threshold_sweep_uses_both_probabilities_schema_floor_and_denominators(tmp_path):
    write_run(tmp_path, [(.95, .95, True, True), (.9, .6, False, True),
                         (.8, .9, True, True), (.99, .99, False, False)])
    result = calibration.calibrate(tmp_path)
    by_threshold = {row["threshold"]: row for row in result["metrics"]}
    at_08 = by_threshold[.8]
    assert at_08["counts"] == {"evaluable": 4, "correct": 2, "incorrect": 2,
                               "accepted": 2, "rejected": 2, "true_accept": 2,
                               "false_accept": 0, "true_reject": 2, "false_reject": 0}
    assert at_08["coverage"] == .5
    assert at_08["accepted_precision"] == 1
    assert at_08["false_accept_rate"] == 0
    assert at_08["false_reject_rate"] == 0
    assert at_08["balanced_gate_accuracy"] == 1
    assert at_08["denominators"]["accepted_precision"] == 2
    assert at_08["denominators"]["false_accept_rate"] == 2
    # 0.7 and 0.8 make identical decisions: final tie favors lower threshold.
    assert result["recommendation"]["threshold"] == .7
    assert result["provenance"]["ids"] == ["0", "1", "2", "3"]
    assert result["provenance"]["observed_jev_models"] == ["jev-1.13.0"]
    assert len(result["provenance"]["records_sha256"]) == 64


def test_equal_balanced_accuracy_prefers_higher_coverage(tmp_path):
    write_run(tmp_path, [(.8, .8, True, True), (.8, .8, False, True)])
    result = calibration.calibrate(tmp_path, [.7, .9])
    assert [row["balanced_gate_accuracy"] for row in result["metrics"]] == [.5, .5]
    assert result["recommendation"]["threshold"] == .7


@pytest.mark.parametrize("correct", [True, False])
def test_missing_correctness_class_yields_no_recommendation(tmp_path, correct):
    write_run(tmp_path, [(.9, .9, correct, True)])
    result = calibration.calibrate(tmp_path)
    assert result["status"] == "no_recommendation"
    assert result["recommendation"] is None
    assert all(row["balanced_gate_accuracy"] is None for row in result["metrics"])
    missing_rate = "false_accept_rate" if correct else "false_reject_rate"
    assert all(row[missing_rate] is None for row in result["metrics"])


def test_no_probabilities_does_not_manufacture_zeros_or_recommendation(tmp_path):
    write_run(tmp_path, [(None, None, True, True), (None, None, False, False)])
    result = calibration.calibrate(tmp_path)
    assert result["recommendation"] is None
    assert result["observations"]["both_probabilities_observed"] == 0
    assert result["observations"]["excluded"]["missing_probabilities"] == 1
    assert result["observations"]["schema_rejected_without_probabilities"] == 1
    assert result["observations"]["evaluable_records"] == 1
    assert result["metrics"][0]["accepted_precision"] is None


def test_live_relevance_is_excluded_and_does_not_supply_correctness_class(tmp_path):
    write_run(tmp_path, [(.9, .9, True, True), (.2, .2, False, True, "live_relevance")])
    result = calibration.calibrate(tmp_path)
    assert result["recommendation"] is None
    assert result["observations"]["excluded"]["live_relevance"] == 1


@pytest.mark.parametrize("overrides", [{"split": "test"}, {"split": "all"}, {"limit": 1},
                                       {"max_repairs": 1}, {"synthetic": True}, {"derived_from": "another-run"}])
def test_rejects_test_smoke_repair_synthetic_and_replayed_runs(tmp_path, overrides):
    write_run(tmp_path, [(.9, .9, True, True), (.2, .2, False, True)], **overrides)
    with pytest.raises(ValueError):
        calibration.calibrate(tmp_path)


def test_incomplete_duplicate_and_mismatched_records_are_rejected(tmp_path):
    records, _ = write_run(tmp_path, [(.9, .9, True, True), (.2, .2, False, True)])
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="complete"):
        calibration.calibrate(tmp_path)
    path.write_text("".join(json.dumps(row) + "\n" for row in [records[0], records[0]]))
    with pytest.raises(ValueError, match="Duplicate"):
        calibration.calibrate(tmp_path)
    records[1]["config_id"] = "different"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="configuration"):
        calibration.calibrate(tmp_path)


def test_partial_missing_observations_are_explicitly_counted(tmp_path):
    write_run(tmp_path, [(.95, .95, True, True), (.2, .2, False, True),
                         (.9, None, False, True), (.9, .9, None, True)])
    result = calibration.calibrate(tmp_path)
    assert result["recommendation"] is not None
    assert result["observations"]["evaluable_records"] == 2
    assert result["observations"]["excluded"]["missing_probabilities"] == 1
    assert result["observations"]["excluded"]["missing_correctness_label"] == 1


def test_cli_writes_default_json_without_changing_source_run(tmp_path):
    write_run(tmp_path, [(.95, .95, True, True), (.2, .2, False, True)])
    before = (tmp_path / "records.jsonl").read_bytes()
    assert calibration.main([str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["status"] == "recommended"
    assert (tmp_path / "records.jsonl").read_bytes() == before
    assert calibration.main([str(tmp_path), "--output", str(tmp_path / "manifest.json")]) == 1


@pytest.mark.parametrize("bad_grid", [[], [-.1], [1.1], [float("nan")], [True]])
def test_invalid_grid_rejected(tmp_path, bad_grid):
    write_run(tmp_path, [(.95, .95, True, True), (.2, .2, False, True)])
    with pytest.raises(ValueError, match="grid"):
        calibration.calibrate(tmp_path, bad_grid)
