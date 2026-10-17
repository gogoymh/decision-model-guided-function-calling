"""Select a repair limit from complete development traces, without API calls.

Selection is read-only. An artifact binds the choice to its complete source
evidence; loading it repeats selection before it can configure a test run.
The selected limit is based only on jev_joint (JBFC). It is shared by the test
repair methods; they are not tuned separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .benchmarks import (
    ANSWER_HASHES,
    BFCL_REVISION,
    EXPECTED_COUNTS,
    OFFICIAL_CHECKER_SHA256,
    QUESTION_HASHES,
    load_examples,
    stratified_split,
)
from .pipeline import METHODS

DEFAULT_REPAIR_GRID = (0, 1, 2, 3, 4, 5)
SELECTION_RULE = "max_dev_accuracy_then_smallest_budget"
BUDGET_SELECTION_METHOD = "jev_joint"
COMPATIBILITY_FIELDS = (
    "version", "source_sha256", "dataset_sha256", "dataset_revisions", "split_seed",
    "replicates", "threshold", "exhaustion_policy", "llm_model", "llm_base_url",
    "llm_provider", "llm_api_style", "llm_gateway", "llm_endpoint", "llm_auth_header",
    "llm_timeout_s", "llm_max_retries", "reasoning_effort", "temperature",
    "max_completion_tokens", "jev_model", "jev_base_url", "jev_gateway", "jev_endpoint",
    "jev_timeout_s", "jev_max_retries", "workers", "prices_usd_per_million",
)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def _fingerprint(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hash():
    return hashlib.sha256(b"".join(
        path.name.encode() + path.read_bytes()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    )).hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_repair_grid(grid=DEFAULT_REPAIR_GRID):
    """Return sorted, distinct nonnegative integers; reject strings and booleans."""
    if isinstance(grid, (str, bytes, dict)):
        raise ValueError("Repair grid must contain distinct nonnegative integers")
    try:
        values = list(grid)
    except TypeError:
        raise ValueError("Repair grid must contain distinct nonnegative integers") from None
    if not values:
        raise ValueError("Repair grid cannot be empty")
    for value in values:
        _integer(value, "Repair grid entry")
    if len(set(values)) != len(values):
        raise ValueError("Repair grid entries must be distinct")
    return sorted(values)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        raise ValueError("Missing or invalid tuning evidence file") from None


def _read_jsonl(path):
    try:
        values = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        raise ValueError("Missing or invalid tuning evidence file") from None
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Tuning evidence records must be objects")
    return values


def _canonical_benchmark():
    """Reuse the prepared-data checksum check and verify its audited source pins."""
    path = Path(__file__).resolve().parents[2] / "data/bfcl/examples.jsonl"
    examples = load_examples(path)
    prepared = _read_json(path.parent / "manifest.json")
    digest = _file_hash(path)
    pins = {f"raw/BFCL_v4_{cat}.json": value for cat, value in QUESTION_HASHES.items()}
    pins.update({f"raw/possible_answer/BFCL_v4_{cat}.json": value for cat, value in ANSWER_HASHES.items()})
    pins["upstream/ast_checker.py"] = OFFICIAL_CHECKER_SHA256
    if (prepared.get("revision") != BFCL_REVISION
            or prepared.get("normalized_sha256", {}).get("examples.jsonl") != digest
            or dict(Counter(e["category"] for e in examples)) != EXPECTED_COUNTS
            or any(e.get("dataset_revision") != BFCL_REVISION for e in examples)
            or any(prepared.get("sources", {}).get(name, {}).get("sha256") != value for name, value in pins.items())):
        raise ValueError("Tuning requires the complete pinned canonical BFCL dataset")
    checker = path.parent / "upstream/ast_checker.py"
    if _file_hash(checker) != OFFICIAL_CHECKER_SHA256:
        raise ValueError("Tuning requires the pinned official checker")
    development, test = stratified_split(examples, seed=1729)
    if len(development) != 303 or sum(e["category"] != "live_relevance" for e in development) != 302:
        raise ValueError("Canonical development split must contain 303 examples (302 scored)")
    return development, test, digest, checker, pins


def _settings(manifest):
    if any(field not in manifest for field in COMPATIBILITY_FIELDS):
        raise ValueError("Run is missing model/API/runtime/configuration provenance")
    # Only declared settings are copied: never serialize credentials or arbitrary
    # extra manifest fields into a selection artifact.
    settings = {field: deepcopy(manifest[field]) for field in COMPATIBILITY_FIELDS}
    _json_bytes(settings)  # Reject non-finite values before fingerprinting.
    return settings


def _validate_manifest(manifest, development, digest):
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be an object")
    if (manifest.get("synthetic") is not False or manifest.get("split") != "dev"
            or manifest.get("limit") is not None or manifest.get("derived_from") is not None
            or manifest.get("repair_budget_selection") is not None or manifest.get("smoke")):
        raise ValueError("Selection requires a complete original dev run, not synthetic/smoke/test/all/derived data")
    settings = _settings(manifest)
    if (manifest.get("dataset_sha256") != digest or manifest.get("dataset_revisions") != [BFCL_REVISION]
            or manifest.get("split_seed") != 1729 or manifest.get("source_sha256") != _source_hash()):
        raise ValueError("Development data, split, or code fingerprint does not match the canonical setup")
    if manifest.get("exhaustion_policy") != "execute_last":
        raise ValueError("Repair budget selection requires execute_last final-call accuracy")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or len(ids) != 303 or set(ids) != {e["id"] for e in development}:
        raise ValueError("Selection requires all 303 canonical development IDs exactly once")
    methods = manifest.get("methods")
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
            or not set(methods).issubset(METHODS) or "jev_joint" not in methods):
        raise ValueError("Development methods must be distinct supported methods including jev_joint")
    replicates = _integer(manifest.get("replicates"), "Replicates", 1)
    _integer(manifest.get("max_repairs"), "Source max_repairs")
    threshold = manifest.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Development threshold must be fixed and between zero and one")
    expected = len(ids) * replicates * len(methods)
    if manifest.get("expected_records") != expected:
        raise ValueError("Development ID/replicate/method grid has inconsistent expected_records")
    body = {key: value for key, value in manifest.items() if key not in {"config_id", "created_at"}}
    if manifest.get("config_id") != _fingerprint(body):
        raise ValueError("Development manifest configuration fingerprint is invalid")
    return settings


def _validate_joint_trace(row, example, initial, manifest, checker):
    from .jev import validate_calls
    from .scoring import score_example

    trace = row.get("trace")
    cap = manifest["max_repairs"]
    if (not isinstance(trace, list) or not 1 <= len(trace) <= cap + 1
            or row.get("max_repairs") != cap or row.get("exhaustion_policy") != "execute_last"
            or row.get("dispatch_mode") != "simulated" or row.get("execution_result") is not None):
        raise ValueError("Invalid development trajectory or execution policy")
    if trace[0].get("proposal") != initial:
        raise ValueError("Development trajectory differs from its shared initial proposal")
    for index, step in enumerate(trace):
        if step.get("attempt") != index:
            raise ValueError("Development trajectory attempts are not consecutive")
        proposal, verification = step.get("proposal", {}), step.get("verification", {})
        if "calls" not in proposal:
            raise ValueError("Development proposal has no call collection")
        local = validate_calls(example, proposal["calls"])
        if step.get("schema") != local:
            raise ValueError("Saved schema validation differs from the candidate")
        arguments_skipped = proposal["calls"] == []
        if (verification.get("arguments_skipped") is not arguments_skipped
                or "arguments_probability" not in verification or "arguments_pass" not in verification
                or (arguments_skipped and (verification["arguments_probability"] is not None
                                          or verification["arguments_pass"] is not None))):
            raise ValueError("Argument-check skip status differs from the proposed call collection")
        probabilities = [verification.get(f"{name}_probability") for name in ("function", "arguments")]
        for probability in probabilities:
            if probability is not None and (isinstance(probability, bool)
                    or not isinstance(probability, (int, float)) or not math.isfinite(probability)
                    or not 0 <= probability <= 1):
                raise ValueError("Invalid verifier probability in development trace")
        expected_passes = [bool(local[f"{name}_pass"] and probability is not None
                                and probability >= manifest["threshold"])
                           for name, probability in zip(("function", "arguments"), probabilities)]
        if arguments_skipped:
            expected_passes[1] = None
        if any(verification.get(f"{name}_pass") is not passed
               for name, passed in zip(("function", "arguments"), expected_passes)):
            raise ValueError("Verifier decisions differ from the fixed threshold/schema checks")
        passed = bool(local["accepted"] and expected_passes[0]
                      and (arguments_skipped or expected_passes[1]))
        if step.get("gate_pass") is not passed or (index < len(trace) - 1 and passed):
            raise ValueError("Development trajectory violates its acceptance/early-stop rule")
        # Re-score saved calls offline; reference labels never enter model inputs.
        if step.get("score") != score_example(example, proposal["calls"], checker_path=checker):
            raise ValueError("Saved development score differs from official offline scoring")
        for payload in (proposal, verification):
            latency = payload.get("latency_s")
            if latency is not None and (isinstance(latency, bool) or not isinstance(latency, (int, float))
                                        or not math.isfinite(latency) or latency < 0):
                raise ValueError("Invalid latency in development trace")
    terminal = trace[-1]
    if not terminal["gate_pass"] and len(trace) != cap + 1:
        raise ValueError("Development trajectory stopped before acceptance or its repair limit")
    if (row.get("score") != terminal["score"] or row.get("calls") != terminal["proposal"]["calls"]
            or row.get("accepted") is not terminal["gate_pass"] or row.get("repairs") != len(trace) - 1
            or row.get("exhausted") is not (not terminal["gate_pass"])
            or row.get("dispatched") is not True or row.get("abstained") is not False):
        raise ValueError("Development terminal record does not match its trajectory")


def _evidence(run_dir):
    directory = Path(run_dir).resolve()
    manifest = _read_json(directory / "manifest.json")
    development, test, digest, checker, pins = _canonical_benchmark()
    settings = _validate_manifest(manifest, development, digest)
    examples = {example["id"]: example for example in development}
    expected = {(example_id, replicate, method) for example_id in manifest["ids"]
                for replicate in range(manifest["replicates"]) for method in manifest["methods"]}
    rows = _read_jsonl(directory / "records.jsonl")
    seen = set()
    for row in rows:
        key = (row.get("id"), row.get("replicate"), row.get("method"))
        _integer(row.get("replicate"), "Record replicate")
        if (key not in expected or key in seen or row.get("config_id") != manifest["config_id"]
                or row.get("synthetic") is not False or row.get("derived_from") is not None
                or row.get("category") != examples.get(row.get("id"), {}).get("category")):
            raise ValueError("Duplicate, foreign, synthetic, derived, or mismatched development record")
        seen.add(key)
    if seen != expected:
        raise ValueError("Selection requires a complete development ID/replicate/method grid")
    initials = {}
    expected_initials = {(example_id, replicate) for example_id in manifest["ids"]
                         for replicate in range(manifest["replicates"])}
    for row in _read_jsonl(directory / "initial.jsonl"):
        key = (row.get("id"), row.get("replicate"))
        _integer(row.get("replicate"), "Initial replicate")
        if key not in expected_initials or key in initials or not isinstance(row.get("proposal"), dict):
            raise ValueError("Duplicate, foreign, or malformed initial proposal")
        initials[key] = row["proposal"]
    if set(initials) != expected_initials:
        raise ValueError("Selection requires the complete shared-initial proposal grid")
    joint = [row for row in rows if row["method"] == BUDGET_SELECTION_METHOD]
    for row in joint:
        _validate_joint_trace(row, examples[row["id"]], initials[(row["id"], row["replicate"])], manifest, checker)
    return directory, manifest, settings, joint, test, pins


def _prefix(row, budget, prices):
    from .reporting import token_cost

    result = deepcopy(row)
    result["trace"] = result["trace"][:budget + 1]
    last = result["trace"][-1]
    result.update(score=last["score"], calls=last["proposal"]["calls"], accepted=last["gate_pass"],
                  exhausted=not last["gate_pass"], repairs=len(result["trace"]) - 1,
                  max_repairs=budget, dispatched=True, abstained=False)
    latencies = [part.get("latency_s") for step in result["trace"]
                 for part in (step["proposal"], step["verification"])]
    result["logical_latency_s"] = sum(latencies) if all(value is not None for value in latencies) else None
    result["accounting"] = token_cost(result, prices)
    return result


def _metrics(rows):
    from .reporting import summarize_group

    # summarize_group expects numeric latencies. Never turn missing timings into
    # reported zero latency: calculate other metrics, then mark timings unknown.
    missing_latency = any(row["logical_latency_s"] is None for row in rows)
    numeric = [{**row, "logical_latency_s": row["logical_latency_s"] or 0.0} for row in rows]
    values = summarize_group(numeric)
    keys = ("n", "exact_n", "accuracy", "mean_repairs", "mean_llm_proposals", "mean_latency_s",
            "p95_latency_s", "mean_cost_usd", "cost_observed_n", "coverage", "exhaustion_rate")
    metrics = {key: values[key] for key in keys}
    metrics["correct"] = sum(row["score"]["correct"] for row in rows if row["category"] != "live_relevance")
    if missing_latency:
        metrics["mean_latency_s"] = metrics["p95_latency_s"] = None
    return metrics


def select_repair_budget(run_dir, grid=DEFAULT_REPAIR_GRID) -> dict:
    """Select maximum dev accuracy, breaking exact ties with the smallest limit.

    All candidates have the same denominator, so integer correct-counts give an
    exact tie rule without floating-point tolerances or test-set information.
    """
    grid = validate_repair_grid(grid)
    directory, manifest, settings, joint, _, pins = _evidence(run_dir)
    if max(grid) > manifest["max_repairs"]:
        raise ValueError("Repair grid exceeds the recorded source repair limit")
    candidates = []
    for budget in grid:
        rows = [_prefix(row, budget, manifest["prices_usd_per_million"]) for row in joint]
        candidates.append({"max_repairs": budget, **_metrics(rows), "by_replicate": {
            str(rep): _metrics([row for row in rows if row["replicate"] == rep])
            for rep in range(manifest["replicates"])
        }})
    selected = min(candidates, key=lambda item: (-item["correct"], item["max_repairs"]))
    artifact = {
        "kind": "repair_budget_selection", "artifact_version": 1,
        "selection_rule": SELECTION_RULE, "selection_method": BUDGET_SELECTION_METHOD, "grid": grid,
        "selected_max_repairs": selected["max_repairs"], "source_dir": str(directory),
        "candidates": candidates,
        "provenance": {
            "settings": settings, "config_id": manifest["config_id"],
            "source_sha256": manifest["source_sha256"], "dataset_sha256": manifest["dataset_sha256"],
            "source_manifest_sha256": _file_hash(directory / "manifest.json"),
            "source_records_sha256": _file_hash(directory / "records.jsonl"),
            "source_initial_sha256": _file_hash(directory / "initial.jsonl"),
            "dataset_source_sha256": pins, "source_max_repairs": manifest["max_repairs"],
            "dev_examples": 303, "primary_dev_examples": 302, "replicates": manifest["replicates"],
        },
    }
    return {**artifact, "selection_id": _fingerprint(artifact)}


def validate_repair_budget_selection(selection, test_config=None) -> dict:
    """Recompute a selection and optionally require compatible canonical test settings."""
    if not isinstance(selection, dict) or selection.get("kind") != "repair_budget_selection":
        raise ValueError("Expected a repair_budget_selection artifact")
    if selection.get("selection_method") != BUDGET_SELECTION_METHOD:
        raise ValueError("Repair-budget selection must be based on jev_joint (JBFC)")
    claimed = selection.get("selection_id")
    body = {key: value for key, value in selection.items() if key != "selection_id"}
    if claimed != _fingerprint(body):
        raise ValueError("Repair-budget selection fingerprint is invalid")
    if not isinstance(selection.get("source_dir"), str):
        raise ValueError("Repair-budget selection has no source directory")
    recomputed = select_repair_budget(selection["source_dir"], selection.get("grid"))
    if _json_bytes(recomputed) != _json_bytes(selection):
        raise ValueError("Repair-budget selection is stale or differs from its source evidence")
    if test_config is not None:
        if (not isinstance(test_config, dict) or test_config.get("split") != "test"
                or test_config.get("limit") is not None or test_config.get("synthetic") is not False
                or test_config.get("derived_from") is not None):
            raise ValueError("A selected budget requires a complete original canonical test run")
        _, test, digest, _, _ = _canonical_benchmark()
        ids = test_config.get("ids")
        if (not isinstance(ids, list) or len(ids) != 2748 or set(ids) != {e["id"] for e in test}
                or test_config.get("dataset_sha256") != digest):
            raise ValueError("A selected budget requires all 2,748 canonical test IDs and matching data")
        if (_integer(test_config.get("max_repairs"), "Test max_repairs") != selection["selected_max_repairs"]
                or _settings(test_config) != selection["provenance"]["settings"]):
            raise ValueError("Test settings are incompatible with the development repair-budget selection")
    return deepcopy(recomputed)


def load_repair_budget_selection(path, test_config=None) -> dict:
    return validate_repair_budget_selection(_read_json(path), test_config=test_config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Complete canonical development run")
    parser.add_argument("--grid", default=",".join(map(str, DEFAULT_REPAIR_GRID)))
    parser.add_argument("--output", help="Default: RUN_DIR/repair_budget_selection.json")
    args = parser.parse_args(argv)
    try:
        grid = validate_repair_grid([int(part.strip()) for part in args.grid.split(",")])
        source = Path(args.run_dir).resolve()
        output = Path(args.output).resolve() if args.output else source / "repair_budget_selection.json"
        # A selection must never replace the evidence that justifies it.
        if output.exists() and _read_json(output).get("kind") != "repair_budget_selection":
            raise ValueError("Refusing to overwrite a source-evidence or unrelated file")
        if output in {source / name for name in ("manifest.json", "records.jsonl", "initial.jsonl", "errors.jsonl", "summary.json", "summary.csv")}:
            raise ValueError("Refusing to overwrite source evidence")
        artifact = select_repair_budget(source, grid)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"selection_id": artifact["selection_id"], "selected_max_repairs": artifact["selected_max_repairs"],
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
