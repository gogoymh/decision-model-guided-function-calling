#!/usr/bin/env python3
"""Hash a real dataset and cached first proposals; prepare a plan without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dmgfc.perplexity import PerplexityDecider, fingerprint  # noqa: E402
from dgx_decider_smoke import write_new_json  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Dataset/cache must be a nonempty JSON array or JSONL of objects.")
    return rows


def prepare(dataset: Path, initials: Path, generator_manifest: Path, *, base_url: str,
            source_records: Path | None = None) -> dict:
    examples, proposals = read_rows(dataset), read_rows(initials)
    manifest = json.loads(generator_manifest.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Generator manifest must be an object.")
    ids = [example.get("id") for example in examples]
    if (not all(isinstance(value, str) and value for value in ids) or len(set(ids)) != len(ids)
            or any(not isinstance(example.get("messages"), list) or not isinstance(example.get("tools"), list)
                   for example in examples)):
        raise ValueError("Prepared examples need unique string IDs, messages and tools.")
    allowed, seen, hashes, replicate_ids = set(ids), set(), {}, {}
    for row in proposals:
        identifier, replicate = row.get("id"), row.get("replicate", 0)
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("Cached replicate must be a nonnegative integer.")
        key = (identifier, replicate)
        if not isinstance(identifier, str) or identifier not in allowed or key in seen:
            raise ValueError("Cached initials contain duplicate or foreign (ID, replicate) pairs.")
        proposal = row.get("proposal")
        if not isinstance(proposal, dict) or "calls" not in proposal:
            raise ValueError("Every cached initial needs a proposal with calls; unreadable calls may be null.")
        if proposal["calls"] is not None and not isinstance(proposal["calls"], list):
            raise ValueError("Cached proposal calls must be a list or null.")
        if row.get("config_id") is not None and row["config_id"] != manifest.get("config_id"):
            raise ValueError("Cached initial fingerprint differs from its supplied generator manifest.")
        if row.get("config_id") is None and source_records is None:
            raise ValueError("Legacy initials without config IDs require matching immutable source records.")
        seen.add(key)
        replicate_ids.setdefault(replicate, set()).add(identifier)
        hashes[f"{replicate}:{identifier}"] = fingerprint(proposal)
    if any(values != allowed for values in replicate_ids.values()):
        raise ValueError("Each cached replicate must cover every selected dataset ID exactly once.")
    source_record_proof = None
    if source_records is not None:
        records, proven, record_keys = read_rows(source_records), set(), set()
        initial_by_key = {(row["id"], row.get("replicate", 0)): row["proposal"] for row in proposals}
        for row in records:
            key = (row.get("id"), row.get("replicate", 0))
            method = row.get("method")
            if key not in initial_by_key or not isinstance(method, str) or (key, method) in record_keys:
                raise ValueError("Source records contain foreign or duplicate case/replicate/method rows.")
            if row.get("config_id") is not None and row["config_id"] != manifest.get("config_id"):
                raise ValueError("Source record fingerprint differs from the source manifest.")
            trace = row.get("trace")
            if (not isinstance(trace, list) or not trace or not isinstance(trace[0], dict)
                    or trace[0].get("proposal") != initial_by_key[key]):
                raise ValueError("Source attempt-zero proposal differs from its cached initial.")
            proven.add(key)
            record_keys.add((key, method))
        if proven != seen:
            raise ValueError("Source records do not prove every cached initial.")
        source_record_proof = {"path": str(source_records.resolve()), "sha256": sha256_file(source_records),
                               "rows": len(records), "proven_initial_pairs": len(proven)}
    generator_fields = ("llm_model", "llm_provider", "llm_api_style", "llm_gateway", "llm_endpoint",
                        "reasoning_effort", "temperature", "max_completion_tokens", "source_sha256",
                        "config_id", "dataset_sha256", "dataset_revisions")
    generator = {key: manifest[key] for key in generator_fields if key in manifest}
    if not isinstance(generator.get("llm_model"), str) or not generator["llm_model"]:
        raise ValueError("The source generator manifest must identify llm_model.")
    # For manifests that pin the dataset file, require it; do not silently relabel
    # an unrelated initial cache as shared proposals for the new comparison.
    digest = sha256_file(dataset)
    if manifest.get("dataset_sha256") is not None and manifest["dataset_sha256"] != digest:
        raise ValueError("Dataset file differs from the source generator manifest.")
    with PerplexityDecider(base_url=base_url) as verifier:
        decision_configuration = verifier.configuration()
    plan = {"format_version": "dgx-decider-experiment-preparation-v1", "plan_only": True,
            "ready_for_benchmark": False, "status": "inputs_fingerprinted; full experiment runner not executed",
            "dataset": {"path": str(dataset.resolve()), "sha256": digest, "examples": len(examples), "ids": ids},
            "initial_cache": {"path": str(initials.resolve()), "sha256": sha256_file(initials),
                              "rows": len(proposals), "replicates": sorted(replicate_ids),
                              "proposal_sha256_by_key": hashes},
            "source_generator_manifest_sha256": sha256_file(generator_manifest), "generator": generator,
            "source_record_proof": source_record_proof,
            "decision_model": decision_configuration,
            "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(
                list((ROOT / "src/dmgfc").glob("*.py")) + list((ROOT / "src/dmfc").glob("*.py"))
                + list((ROOT / "src/jev_fc").glob("*.py")))},
            "proposed_comparison": {"repair_limits": [0, 1, 2, 3, 5, 7, 10],
                "threshold": 0.5, "final_rule": "keep_first", "shared_initial_proposal": True,
                "execute_tools": False, "workers": 1,
                "arms": ["hosted_jev", "perplexity_decider_native"],
                "random_selection_and_multi_agent": "separate baselines require implementation and prespecification",
                "historical_selection": "r=3, threshold=0.5 and keep_first were previously selected on observed examples"},
            "execution": {"model_api_calls": 0, "generator_calls": 0, "tool_executions": 0},
            "required_before_running": ["Verify actual native model provenance and measured DGX smoke",
                "Pin the offline evaluator and dataset revision", "Approve generator model/settings and request budget",
                "Prespecify all arms, repeats, concurrency and failure accounting before running"],
            "metric_labels": {"latency": "measured DGX Spark client wall time and API duration in seconds",
                "input_tokens": "sum of native prepared question branches; missing counters remain null",
                "output_tokens": "reported generated-token count; native decisions report 0",
                "api_calls": "POST attempts including retries; provenance GETs separate",
                "forward_calls": "actual successful batched model invocations when native batch marker is verified; retries may be unknown",
                "estimated_usd": None}}
    return {**plan, "configuration_fingerprint": fingerprint(plan)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--initials", required=True, type=Path)
    parser.add_argument("--generator-manifest", required=True, type=Path)
    parser.add_argument("--source-records", type=Path, help="Required for legacy initials without config IDs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        plan = prepare(args.dataset, args.initials, args.generator_manifest, base_url=args.base_url,
                       source_records=args.source_records)
        write_new_json(args.output, plan)
    except (ValueError, OSError) as error:
        # Do not print JSON parser source excerpts or file contents.
        print(json.dumps({"status": "preparation_failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "inputs_fingerprinted", "output": str(args.output),
                      "configuration_fingerprint": plan["configuration_fingerprint"],
                      "model_api_calls": 0, "ready_for_benchmark": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
