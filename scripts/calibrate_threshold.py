#!/usr/bin/env python3
"""Select an optional joint gate threshold from a complete, empirical dev run.

This is an offline initial-candidate analysis. It makes no API calls and cannot
estimate how changing a threshold would change later repair prompts or outputs.
The preregistered, untuned main default remains 0.8 unless a dev recommendation
is deliberately frozen and passed to a new experiment with --threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _valid_probability(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _check_run(manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    if manifest.get("split") != "dev":
        raise ValueError("Calibration requires a dev split; test/all splits are never accepted.")
    if manifest.get("limit") is not None:
        raise ValueError("Calibration requires the complete dev split without --limit.")
    if manifest.get("max_repairs") != 0 or isinstance(manifest.get("max_repairs"), bool):
        raise ValueError("Calibration requires max_repairs=0 so only initial candidates are inspected.")
    if manifest.get("synthetic") is not False or any(row.get("synthetic") is not False for row in records):
        raise ValueError("Calibration requires explicitly empirical records; synthetic/unknown provenance is rejected.")
    if manifest.get("derived_from"):
        raise ValueError("Use an original zero-repair dev run, not a replay-derived run.")
    methods = manifest.get("methods")
    if not isinstance(methods, list) or "jev_joint" not in methods or len(set(methods)) != len(methods):
        raise ValueError("The dev run must include the jev_joint method with unique method names.")
    ids, replicates = manifest.get("ids"), manifest.get("replicates")
    if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("The dev manifest must contain a nonempty list of unique example IDs.")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("The dev manifest must have a positive replicate count.")
    config_id = manifest.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("The dev manifest must identify the run configuration.")
    expected = {(example_id, replicate, method) for example_id in ids for replicate in range(replicates) for method in methods}
    observed = [(row.get("id"), row.get("replicate"), row.get("method")) for row in records]
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate method records are not valid calibration observations.")
    if set(observed) != expected or manifest.get("expected_records") != len(expected):
        raise ValueError("Calibration requires complete ID/replicate/method coverage matching the manifest.")
    if any(row.get("config_id") != config_id for row in records):
        raise ValueError("Every calibration record must match the manifest's configuration ID.")
    for row in records:
        if row.get("method") != "jev_joint":
            continue
        trace = row.get("trace")
        if row.get("max_repairs") != 0 or row.get("repairs") != 0 or not isinstance(trace, list) or len(trace) != 1:
            raise ValueError("Every joint record must contain exactly one proposal and zero repairs.")
        if trace[0].get("attempt") != 0:
            raise ValueError("Calibration can inspect only attempt 0.")


def calibrate(run_dir: str | Path, grid: tuple[float, ...] | list[float] = DEFAULT_GRID) -> dict[str, Any]:
    """Read saved dev traces and return an auditable threshold recommendation."""
    thresholds = sorted(set(grid))
    if not thresholds or any(not _valid_probability(value) for value in thresholds):
        raise ValueError("Threshold grid must contain finite numbers between 0 and 1.")
    directory = Path(run_dir)
    manifest_path, records_path = directory / "manifest.json", directory / "records.jsonl"
    manifest = json.loads(manifest_path.read_text())
    records = _load_records(records_path)
    _check_run(manifest, records)
    joint = [row for row in records if row["method"] == "jev_joint"]
    excluded = Counter(live_relevance=0, missing_correctness_label=0, missing_probabilities=0)
    candidates = []
    probabilities_observed = 0
    schema_rejected_without_probabilities = 0
    for row in joint:
        if row["category"] == "live_relevance":
            excluded["live_relevance"] += 1
            continue
        step = row["trace"][0]
        correct = step.get("score", {}).get("correct")
        if correct is None:
            excluded["missing_correctness_label"] += 1
            continue
        if not isinstance(correct, bool):
            raise ValueError("Candidate correctness labels must be booleans or null.")
        schema_pass = step.get("schema", {}).get("accepted")
        if not isinstance(schema_pass, bool):
            raise ValueError("Every candidate needs a boolean schema.accepted verdict.")
        check = step.get("verification", {})
        probabilities = [check.get("function_probability"), check.get("arguments_probability")]
        for value in probabilities:
            if value is not None and not _valid_probability(value):
                raise ValueError("A recorded Jev probability is outside its numeric 0–1 contract.")
        both_observed = all(value is not None for value in probabilities)
        if both_observed:
            probabilities_observed += 1
        elif schema_pass:
            # A passing schema cannot determine the semantic gate without both
            # observations. Exclude it explicitly rather than manufacture zeros.
            excluded["missing_probabilities"] += 1
            continue
        else:
            # An AND gate always rejects structural failures, independently of
            # unobserved semantic probabilities (e.g. malformed argument JSON).
            schema_rejected_without_probabilities += 1
        candidates.append({"id": row["id"], "replicate": row["replicate"], "correct": correct,
                           "schema_pass": schema_pass, "probabilities": probabilities})

    evaluations = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for candidate in candidates:
            accepted = candidate["schema_pass"] and all(value is not None and value >= threshold for value in candidate["probabilities"])
            if candidate["correct"]:
                tp += int(accepted)
                fn += int(not accepted)
            else:
                fp += int(accepted)
                tn += int(not accepted)
        correct_n, incorrect_n = tp + fn, fp + tn
        accepted_n, total_n = tp + fp, tp + fp + tn + fn
        true_accept_rate, true_reject_rate = _ratio(tp, correct_n), _ratio(tn, incorrect_n)
        balanced = ((true_accept_rate + true_reject_rate) / 2
                    if true_accept_rate is not None and true_reject_rate is not None else None)
        evaluations.append({
            "threshold": threshold,
            "coverage": _ratio(accepted_n, total_n),
            "accepted_precision": _ratio(tp, accepted_n),
            "false_accept_rate": _ratio(fp, incorrect_n),
            "false_reject_rate": _ratio(fn, correct_n),
            "balanced_gate_accuracy": balanced,
            "counts": {"evaluable": total_n, "correct": correct_n, "incorrect": incorrect_n,
                       "accepted": accepted_n, "rejected": tn + fn,
                       "true_accept": tp, "false_accept": fp, "true_reject": tn, "false_reject": fn},
            "denominators": {"coverage": total_n, "accepted_precision": accepted_n,
                             "false_accept_rate": incorrect_n, "false_reject_rate": correct_n,
                             "balanced_gate_accuracy": {"correct": correct_n, "incorrect": incorrect_n}},
        })

    reasons = []
    if probabilities_observed == 0:
        reasons.append("No eligible candidate has both Jev probabilities; no threshold recommendation is identifiable.")
    if not candidates:
        reasons.append("No eligible candidates remain after exclusions.")
    elif not any(candidate["correct"] for candidate in candidates) or not any(not candidate["correct"] for candidate in candidates):
        reasons.append("Both correct and incorrect candidate classes are required to compare balanced gate accuracy.")
    recommendation = None
    if not reasons:
        best = max(evaluations, key=lambda row: (row["balanced_gate_accuracy"], row["coverage"], -row["threshold"]))
        recommendation = {"threshold": best["threshold"], "balanced_gate_accuracy": best["balanced_gate_accuracy"],
                          "coverage": best["coverage"], "accepted_precision": best["accepted_precision"],
                          "criterion": "Highest balanced gate accuracy; ties prefer higher coverage, then lower threshold."}

    return {
        "status": "recommended" if recommendation is not None else "no_recommendation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline initial-candidate gate selection; this does not replay or predict repaired trajectories.",
        "untuned_default_threshold": 0.8,
        "recommendation": recommendation,
        "reasons": reasons,
        "threshold_grid": thresholds,
        "observations": {
            "joint_records": len(joint), "evaluable_records": len(candidates),
            "evaluable_unique_ids": len({candidate["id"] for candidate in candidates}),
            "both_probabilities_observed": probabilities_observed,
            "schema_rejected_without_probabilities": schema_rejected_without_probabilities,
            "excluded": dict(excluded),
        },
        "metrics": evaluations,
        "provenance": {
            "run_dir": str(directory.resolve()), "config_id": manifest["config_id"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            "source_sha256": manifest.get("source_sha256"), "dataset_sha256": manifest.get("dataset_sha256"),
            "dataset_revisions": manifest.get("dataset_revisions"),
            "split": manifest["split"], "split_seed": manifest.get("split_seed"), "ids": manifest["ids"],
            "replicates": manifest["replicates"], "max_repairs": 0, "synthetic": False,
            "requested_llm_model": manifest.get("llm_model"), "requested_jev_model": manifest.get("jev_model"),
            "observed_llm_models": sorted({row["trace"][0]["proposal"].get("model") for row in joint
                                           if row["trace"][0]["proposal"].get("model")}),
            "observed_jev_models": sorted({row["trace"][0]["verification"].get("model") for row in joint
                                           if row["trace"][0]["verification"].get("model")}),
            "run_threshold": manifest.get("threshold"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Complete empirical dev run with jev_joint and --max-repairs 0")
    parser.add_argument("--output", help="JSON destination; defaults to RUN_DIR/calibration.json")
    parser.add_argument("--grid", default=",".join(str(value) for value in DEFAULT_GRID), help="Comma-separated threshold values")
    args = parser.parse_args(argv)
    try:
        grid = tuple(float(value) for value in args.grid.split(","))
        result = calibrate(args.run_dir, grid)
        output = Path(args.output) if args.output else Path(args.run_dir) / "calibration.json"
        if output.resolve() in {(Path(args.run_dir) / name).resolve() for name in ("manifest.json", "records.jsonl")}:
            raise ValueError("Calibration output must not overwrite the source run files.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"status": result["status"], "recommendation": result["recommendation"], "output": str(output)}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
