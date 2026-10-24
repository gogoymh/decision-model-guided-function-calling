#!/usr/bin/env python3
"""Explicitly opt-in, bounded BFCL Direct/native-Decider experiment from cached initials.

Offline preflight is the default. Only --execute permits paid repair generations
and native inference. Existing records and caches are never changed or replaced.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dmfc.perplexity import PerplexityDecider, fingerprint  # noqa: E402
from jev_fc.benchmarks import BFCL_REVISION, DEFAULT_CATEGORIES  # noqa: E402
from jev_fc.cli import _source_hash  # noqa: E402
from jev_fc.jev import validate_calls  # noqa: E402
from jev_fc.llm import FunctionCallingLLM  # noqa: E402
from jev_fc.pipeline import run_example  # noqa: E402
from jev_fc.scoring import load_official_checker, score_example  # noqa: E402
from dgx_decider_smoke import write_new_json  # noqa: E402
from prepare_dgx_decider_experiment import prepare, read_rows, sha256_file  # noqa: E402

METHODS = (("direct", "direct"), ("perplexity_decider_joint", "jev_joint"))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True, type=Path)
    result.add_argument("--dataset-manifest", required=True, type=Path)
    result.add_argument("--initials", required=True, type=Path)
    result.add_argument("--generator-manifest", required=True, type=Path)
    result.add_argument("--source-records", required=True, type=Path)
    result.add_argument("--checker", required=True, type=Path)
    result.add_argument("--base-url", default="http://127.0.0.1:18081")
    result.add_argument("--limit", type=int, default=5, help="Prespecified first N cases; default 5, recorded in identity.")
    result.add_argument("--replicate", type=int, default=0, help="Cached replicate to reuse; no new first proposals.")
    result.add_argument("--max-repairs", type=int, default=3)
    result.add_argument("--max-new-generator-calls", type=int, default=15,
                        help="Hard cap on logical repair requests; HTTP retries are disabled.")
    result.add_argument("--output", required=True, type=Path, help="New directory; no overwrite or resume.")
    result.add_argument("--execute", action="store_true", help="Explicitly authorize live model requests after preflight.")
    return result


def source_generator_settings(manifest: dict) -> dict:
    required = ("llm_model", "llm_api_style", "llm_gateway", "llm_base_url", "llm_endpoint",
                "reasoning_effort", "temperature", "max_completion_tokens")
    if any(key not in manifest for key in required):
        raise ValueError("Source generator manifest is missing model/interface/sampling settings.")
    settings = {key: manifest[key] for key in required}
    settings["llm_timeout_s"] = manifest.get("llm_timeout_s", 120)
    if not all(isinstance(settings[key], str) and settings[key] for key in required[:5]):
        raise ValueError("Source generator model and interface settings must be strings.")
    if (isinstance(settings["max_completion_tokens"], bool)
            or not isinstance(settings["max_completion_tokens"], int) or settings["max_completion_tokens"] < 1):
        raise ValueError("Source generator output-token cap must be a positive integer.")
    return settings


def make_generator(settings: dict, *, preflight: bool):
    generator = FunctionCallingLLM(
        api_key="offline-preflight-placeholder" if preflight else None,
        model=settings["llm_model"], api_style=settings["llm_api_style"],
        gateway=settings["llm_gateway"], base_url=settings["llm_base_url"],
        reasoning_effort=settings["reasoning_effort"], temperature=settings["temperature"],
        max_completion_tokens=settings["max_completion_tokens"], timeout=settings["llm_timeout_s"],
        max_retries=0,
    )
    # None is an explicit original setting, not permission for an environment
    # variable to silently introduce sampling/effort changes.
    generator.temperature = settings["temperature"]
    generator.reasoning_effort = settings["reasoning_effort"] if generator.provider == "openai" else None
    if generator.endpoint != settings["llm_endpoint"]:
        generator.close()
        raise ValueError("Resolved generator endpoint differs from its original recorded interface.")
    return generator


def preflight(args) -> tuple[dict, list[dict], dict]:
    for value, name in ((args.limit, "limit"), (args.max_new_generator_calls, "max-new-generator-calls")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be positive.")
    if not 0 <= args.max_repairs <= 10 or args.replicate < 0:
        raise ValueError("max-repairs must be 0..10 and replicate must be nonnegative.")
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new directory.")
    proof = prepare(args.dataset, args.initials, args.generator_manifest, base_url=args.base_url,
                    source_records=args.source_records)
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    if (not isinstance(dataset_manifest, dict) or dataset_manifest.get("revision") != BFCL_REVISION
            or dataset_manifest.get("normalized_sha256", {}).get(args.dataset.name) != sha256_file(args.dataset)):
        raise ValueError("Dataset revision/checksum does not match its audited BFCL manifest.")
    all_examples = read_rows(args.dataset)
    if any(example.get("dataset_revision") != BFCL_REVISION or example.get("category") not in DEFAULT_CATEGORIES
           for example in all_examples):
        raise ValueError("Only the pinned, prepared single-turn BFCL categories are supported.")
    for example in all_examples:
        if (not example["messages"] or any(not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str) for message in example["messages"])):
            raise ValueError("BFCL model inputs must contain valid text messages.")
        validate_calls({key: deepcopy(example[key]) for key in ("messages", "tools")}, [])
    load_official_checker(args.checker)  # existing loader verifies the official file hash before import
    selected = all_examples[:args.limit]
    initials = {row["id"]: row for row in read_rows(args.initials) if row.get("replicate", 0) == args.replicate}
    if not selected or any(example["id"] not in initials for example in selected):
        raise ValueError("Selected replicate does not contain every selected cached initial.")
    source_manifest = json.loads(args.generator_manifest.read_text())
    settings = source_generator_settings(source_manifest)
    generator = make_generator(settings, preflight=True)
    generator.close()
    protocol = {"format_version": "native-decider-bfcl-v1", "synthetic": False,
        "purpose": "new native-verifier experiment; historical paper runs remain unchanged",
        "input_proof": proof, "prepared_dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "checker_sha256": sha256_file(args.checker), "checker_revision": BFCL_REVISION,
        "runner_sha256": sha256_file(Path(__file__)), "frozen_source_sha256": _source_hash(),
        "initial_source_sha256": source_manifest.get("source_sha256"),
        "source_differs_from_initial_run": source_manifest.get("source_sha256") != _source_hash(),
        "generator": {**settings, "max_retries": 0},
        "methods": [public for public, _internal in METHODS], "selected_ids": [row["id"] for row in selected],
        "cached_replicate": args.replicate, "limit": args.limit, "max_repairs": args.max_repairs,
        "max_new_generator_calls": args.max_new_generator_calls, "workers": 1,
        "threshold": 0.5, "final_rule": "keep_first", "tools_execute": False,
        "same_initial_proposal_for_each_arm": True, "new_first_proposals": False,
        "expected_records": len(selected) * len(METHODS),
        "selection_note": "First N IDs in the immutable dataset; original threshold/repair policy was selected on observed examples",
        "pricing": {"local_hardware_usd": None, "generator_usd": None},
        "required_generator_credential_env": ("APIM_SUBSCRIPTION_KEY" if settings["llm_gateway"] == "apim"
            else "ANTHROPIC_API_KEY or LLM_API_KEY" if settings["llm_api_style"] == "anthropic_messages"
            else "OPENAI_API_KEY or LLM_API_KEY")}
    protocol["configuration_fingerprint"] = fingerprint(protocol)
    return protocol, selected, initials


class RequestBudget:
    """Cap logical repair requests before calling a generator with zero HTTP retries."""

    def __init__(self, generator, maximum: int, on_response=None):
        self.generator, self.maximum, self.attempted = generator, maximum, 0
        self.on_response = on_response

    def generate(self, example, history):
        if self.attempted >= self.maximum:
            raise RuntimeError("Prespecified new-generator request cap reached.")
        self.attempted += 1
        proposal = self.generator.generate(example, history)
        if self.on_response is not None:
            self.on_response(proposal, self.attempted)
        return proposal


def _known_sum(values):
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           or value < 0 for value in values):
        return None
    return sum(values)


def accounting(record: dict) -> dict:
    proposals = [step["proposal"] for step in record["trace"]]
    verifications = [step["verification"] for step in record["trace"] if "model_provenance" in step["verification"]]
    def tokens(rows, kind):
        values = []
        for row in rows:
            usage = row.get("usage", {})
            usage = usage if isinstance(usage, dict) else {}
            if kind == "cached_input_tokens":
                details = usage.get("prompt_tokens_details")
                value = usage.get(kind, details.get("cached_tokens") if isinstance(details, dict) else None)
            else:
                value = usage.get(kind, usage.get("prompt_tokens" if kind == "input_tokens" else "completion_tokens"))
            values.append(value)
        return _integer_sum(values)
    def counters(rows, field):
        return _integer_sum([row.get(field) for row in rows])
    return {"standalone_generator_input_tokens": tokens(proposals, "input_tokens"),
            "standalone_generator_cached_input_tokens": tokens(proposals, "cached_input_tokens"),
            "standalone_generator_output_tokens": tokens(proposals, "output_tokens"),
            "new_generator_input_tokens": tokens(proposals[1:], "input_tokens"),
            "new_generator_cached_input_tokens": tokens(proposals[1:], "cached_input_tokens"),
            "new_generator_output_tokens": tokens(proposals[1:], "output_tokens"),
            "native_input_tokens": tokens(verifications, "input_tokens"),
            "native_output_tokens": tokens(verifications, "output_tokens"),
            "standalone_generator_api_calls": counters(proposals, "api_calls"),
            "new_generator_api_calls": counters(proposals[1:], "api_calls"),
            "native_api_calls": counters(verifications, "api_calls"),
            "native_control_api_calls": counters(verifications, "control_api_calls"),
            "native_forward_calls": counters(verifications, "forward_calls"),
            "native_question_evaluations": counters(verifications, "question_evaluations"),
            "standalone_generator_latency_s": _known_sum([row.get("latency_s") for row in proposals]),
            "native_decision_http_time_s": _known_sum([row.get("decision_http_time_s") for row in verifications]),
            "initial_generation_charged_once_per_arm": True,
            "estimated_usd": None, "generator_estimated_usd": None, "local_hardware_estimated_usd": None}


def _integer_sum(values):
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return None
    return sum(values)


def safe_error(error: Exception) -> dict:
    result = {"error_type": type(error).__name__, "at": datetime.now(timezone.utc).isoformat()}
    for name in ("status_code", "attempts"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            result[name] = value
    code = getattr(error, "provider_code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code):
        result["provider_code"] = code
    return result


def append_jsonl(path: Path, value: dict):
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute(args, protocol: dict, examples: list[dict], initials: dict, *, generator=None,
            verifier=None, scorer=None) -> dict:
    if not args.execute:
        raise ValueError("Only --execute enables model requests.")
    owned = []
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_new_json(args.output / "manifest.json", {**protocol, "created_at": datetime.now(timezone.utc).isoformat()})
    records, budget, phase, current_id, current_method = [], None, "client_setup", None, None
    try:
        if verifier is None:
            verifier = PerplexityDecider(base_url=args.base_url, max_retries=0,
                                         api_key=os.getenv("DGX_DECIDER_API_KEY"))
            owned.append(verifier)
        # Strict checkpoint readiness must succeed before any paid repair request.
        provenance = verifier.provenance()
        write_new_json(args.output / "native-model-provenance.json", provenance)
        if generator is None:
            generator = make_generator(protocol["generator"], preflight=False)
            owned.append(generator)
        budget = RequestBudget(generator, args.max_new_generator_calls,
            on_response=lambda proposal, attempt: append_jsonl(args.output / "repair-generations.jsonl", {
                "id": current_id, "method": current_method, "new_generator_request": attempt,
                "proposal": deepcopy(proposal), "configuration_fingerprint": protocol["configuration_fingerprint"]}))
        native = verifier
        class ObservedVerifier:
            def verify(self, public, calls):
                check = native.verify(public, calls)
                append_jsonl(args.output / "native-verifications.jsonl", {
                    "id": current_id, "method": current_method, "verification": deepcopy(check),
                    "configuration_fingerprint": protocol["configuration_fingerprint"]})
                return check
        verifier = ObservedVerifier()
        score = scorer or (lambda example, calls: score_example(example, calls, checker_path=args.checker))
        for example in examples:
            current_id, phase = example["id"], "model_operations"
            # Strip IDs, categories and labels before BOTH generator and verifier.
            public = {key: deepcopy(example[key]) for key in ("messages", "tools")}
            initial = deepcopy(initials[current_id]["proposal"])
            produced = []
            for name, method in METHODS:
                current_method = name
                started = time.perf_counter()
                row = run_example(public, budget, verifier, method=method, max_repairs=args.max_repairs,
                                  exhaustion_policy="abstain", initial_proposal=deepcopy(initial), executor=None)
                row.update(id=current_id, category=example["category"], method=name, loop_method=method,
                           synthetic=False, replicate=args.replicate, final_rule="keep_first",
                           configuration_fingerprint=protocol["configuration_fingerprint"],
                           execution_wall_time_s=time.perf_counter() - started,
                           standalone_initial_generation_latency_s=initial.get("latency_s"))
                produced.append(row)
            phase = "offline_scoring"
            # Complete this case's model operations before exposing any gold-derived scores.
            for row in produced:
                current_method = row["method"]
                for step in row["trace"]:
                    step["score"] = score(example, step["proposal"]["calls"])
                final = 0 if row["method"] == "perplexity_decider_joint" and not row["accepted"] else len(row["trace"]) - 1
                row.update(final_attempt=final, calls=deepcopy(row["trace"][final]["proposal"]["calls"]),
                           score=deepcopy(row["trace"][final]["score"]))
                row["accounting"] = accounting(row)
                append_jsonl(args.output / "records.jsonl", row)
                records.append(row)
        summary = {"status": "completed", "synthetic": False, "records": len(records),
                   "expected_records": protocol["expected_records"], "new_generator_requests_attempted": budget.attempted,
                   "generator_http_retries": 0, "native_http_retries": 0, "tool_executions": 0,
                   "configuration_fingerprint": protocol["configuration_fingerprint"],
                   "methods": {name: {"n": sum(row["method"] == name for row in records),
                       "correct": sum(bool(row["score"]["correct"]) for row in records if row["method"] == name)}
                               for name, _ in METHODS}}
        write_new_json(args.output / "summary.json", summary)
        return summary
    except Exception as error:
        event = {**safe_error(error), "id": current_id, "method": current_method, "phase": phase,
                 "new_generator_requests_attempted": budget.attempted if budget else 0,
                 "configuration_fingerprint": protocol["configuration_fingerprint"],
                 "completed_records": len(records), "failed_attempt_token_usage": None}
        append_jsonl(args.output / "errors.jsonl", event)
        raise RuntimeError("Experiment stopped; sanitized failure was preserved in errors.jsonl.") from None
    finally:
        for client in owned:
            client.close()


def main() -> int:
    args = parser().parse_args()
    try:
        protocol, examples, initials = preflight(args)
        if not args.execute:
            args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
            write_new_json(args.output / "preflight.json", protocol)
            print(json.dumps({"status": "ready_for_explicit_execution", "model_api_calls": 0,
                              "output": str(args.output), "selected_cases": len(examples),
                              "configuration_fingerprint": protocol["configuration_fingerprint"]}))
            return 0
        summary = execute(args, protocol, examples, initials)
        print(json.dumps(summary))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", **safe_error(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
