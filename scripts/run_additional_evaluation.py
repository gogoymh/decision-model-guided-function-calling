"""Matched Direct/DMFC evaluation on prepared BFCL or adapted API-Bank first-call data.

The default is an offline preflight. Only --run enables generator/verifier requests.
No benchmark function executes, labels stay out of model inputs, and existing
paper experiments are never imported or overwritten. Price flags (or the
ADDITIONAL_*_PRICE environment variables) are explicit per-model rates in USD
per million tokens; unknown rates stay unavailable rather than inheriting GPT prices.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jev_fc import __version__  # noqa: E402
from jev_fc.benchmarks import BFCL_REVISION, DEFAULT_CATEGORIES, load_examples  # noqa: E402
from jev_fc.cli import _fingerprint, _source_hash, stable_split  # noqa: E402
from jev_fc.jev import JevVerifier, validate_calls  # noqa: E402
from jev_fc.llm import FunctionCallingLLM  # noqa: E402
from jev_fc.pipeline import run_example  # noqa: E402
from jev_fc.reporting import build_summary, paired_delta, percentile, read_jsonl, token_cost  # noqa: E402
from jev_fc.scoring import load_official_checker, score_example as score_bfcl  # noqa: E402
import prepare_apibank  # noqa: E402

METHODS = ("direct", "jev_joint")
FORMAT = "additional-matched-v1"
PRICE_NAMES = ("llm_input", "llm_cached_input", "llm_output", "llm_cache_write_5m",
               "llm_cache_write_1h", "jev_input")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_input(example):
    """Strip IDs, categories, target calls, source filenames, and all other audit/label fields."""
    return deepcopy({name: example[name] for name in ("messages", "tools")})


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def _price(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Prices must be nonnegative finite numbers.")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--benchmark", choices=("bfcl", "apibank"), required=True)
    result.add_argument("--model", required=True, help="Exact configured model/deployment identifier")
    result.add_argument("--api-style", choices=("chat_completions", "azure_chat_completions", "responses",
                                               "anthropic_messages"), required=True)
    result.add_argument("--gateway", choices=("apim", "direct"), default="apim")
    result.add_argument("--jev-gateway", choices=("apim", "direct"), default="direct")
    result.add_argument("--jev-model", default="jev-1.13.0")
    result.add_argument("--base-url", help="Optional explicit generator base URL")
    result.add_argument("--jev-base-url", help="Optional explicit verifier base URL")
    result.add_argument("--data", type=Path, required=True, help="Prepared JSONL file or dataset directory")
    result.add_argument("--checker", type=Path, default=ROOT / "data/bfcl/upstream/ast_checker.py")
    result.add_argument("--split", choices=("all", "dev", "test"), default="test")
    result.add_argument("--split-seed", type=int, default=1729)
    result.add_argument("--replicates", type=_positive, default=3)
    result.add_argument("--workers", type=_positive, default=8)
    result.add_argument("--limit", type=_positive, help="Prespecified limited smoke run; separate output fingerprint")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--run", action="store_true", help="Explicitly enable live requests after offline preflight")
    result.add_argument("--threshold", type=float, default=.5)
    result.add_argument("--reasoning-effort", default="none", help="OpenAI effort; Anthropic records None")
    result.add_argument("--temperature", type=float)
    result.add_argument("--max-completion-tokens", type=_positive, default=4096)
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--jev-timeout", type=float, default=60)
    result.add_argument("--max-retries", type=int, default=3)
    for name in PRICE_NAMES:
        result.add_argument("--" + name.replace("_", "-") + "-price", type=_price,
                            help="USD per million tokens; otherwise ADDITIONAL_" + name.upper() + "_PRICE")
    return result


def explicit_prices(args):
    prices = {}
    for name in PRICE_NAMES:
        value = getattr(args, name + "_price")
        raw = os.getenv("ADDITIONAL_" + name.upper() + "_PRICE", "").strip()
        if value is None and raw:
            try:
                value = _price(raw)
            except (ValueError, argparse.ArgumentTypeError):
                raise ValueError("Invalid additional-evaluation price: " + name) from None
        prices[name] = value
    return prices


def load_data(args):
    path = args.data / "examples.jsonl" if args.data.is_dir() else args.data
    path = path.resolve()
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Additional evaluation requires a prepared dataset manifest.")
    manifest = json.loads(manifest_path.read_text())
    digests = manifest.get("files_sha256", {}) if args.benchmark == "apibank" else manifest.get("normalized_sha256", {})
    if digests.get(path.name) != sha256(path):
        raise ValueError("Prepared data checksum mismatch or missing manifest entry.")
    examples = load_examples(path)
    if not examples:
        raise ValueError("Prepared dataset is empty.")
    for example in examples:
        if (not isinstance(example.get("id"), str) or not example["id"]
                or not isinstance(example.get("messages"), list) or not example["messages"]
                or not isinstance(example.get("tools"), list)):
            raise ValueError("Invalid prepared example identity, messages, or tools.")
        if any(message.get("role") not in {"user", "assistant", "system"}
               or not isinstance(message.get("content"), str) for message in example["messages"]):
            raise ValueError("Prepared messages must be text without tool observations.")
        validate_calls(public_input(example), [])  # Validate every supplied schema before any model request.
    if args.benchmark == "apibank":
        if (manifest.get("revision") != prepare_apibank.REVISION
                or manifest.get("format_version") != "apibank-firstcall-v1"
                or manifest.get("preparer_sha256") != sha256(Path(prepare_apibank.__file__))):
            raise ValueError("API-Bank preparation revision/code does not match the current audited preparer.")
        if any(example.get("category") != "apibank_firstcall" or not example.get("scorable")
               or example.get("dataset_revision") != prepare_apibank.REVISION
               or example.get("split") not in {"dev", "test"} for example in examples):
            raise ValueError("Only prespecified scorable API-Bank first-call examples are supported.")
        if any(example["tools"] != examples[0]["tools"] for example in examples):
            raise ValueError("API-Bank examples must use one fixed independent tool catalog.")
        selected = [example for example in examples if args.split == "all" or example["split"] == args.split]
        score = prepare_apibank.score_example
        scorer_hash = sha256(Path(prepare_apibank.__file__))
        checker_hash = None
        split_rule = manifest["split"]
    else:
        if any(example.get("category") not in DEFAULT_CATEGORIES
               or example.get("dataset_revision") != BFCL_REVISION for example in examples):
            raise ValueError("BFCL examples must use the pinned Python/call-relevance categories and revision.")
        load_official_checker(args.checker)
        selected = stable_split(examples, args.split, args.split_seed)
        def score(example, calls):
            return score_bfcl(example, calls, checker_path=args.checker)
        scorer_hash = sha256(ROOT / "src/jev_fc/scoring.py")
        checker_hash = sha256(args.checker)
        split_rule = "sha256(seed:id), per-category 10% development; original all split unchanged"
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("Selected split is empty.")
    provenance = {"data_path": str(path), "dataset_sha256": sha256(path),
                  "prepared_manifest_sha256": sha256(manifest_path),
                  "dataset_revisions": sorted({example["dataset_revision"] for example in selected}),
                  "dataset_preparer_sha256": manifest.get("preparer_sha256"),
                  "scorer_sha256": scorer_hash, "checker_sha256": checker_hash,
                  "split_rule": split_rule}
    return selected, score, provenance


def make_clients(args, *, preflight=False):
    # Client construction validates routes but sends no requests. Preflight does not require keys.
    key = "additional-evaluation-preflight-only" if preflight else None
    generator = FunctionCallingLLM(api_key=key, model=args.model, api_style=args.api_style,
                                   gateway=args.gateway, base_url=args.base_url,
                                   reasoning_effort=args.reasoning_effort, temperature=args.temperature,
                                   max_completion_tokens=args.max_completion_tokens, timeout=args.timeout,
                                   max_retries=args.max_retries)
    try:
        # Prevent unrelated sampling environment variables from changing the explicitly recorded protocol.
        generator.temperature = args.temperature
        verifier = JevVerifier(api_key=key, model=args.jev_model, gateway=args.jev_gateway,
                               base_url=args.jev_base_url, function_threshold=args.threshold,
                               arguments_threshold=args.threshold, timeout=args.jev_timeout,
                               max_retries=args.max_retries)
    except BaseException:
        generator.close()
        raise
    return generator, verifier


def settings_for(args, examples, provenance, generator, verifier):
    return {"format_version": FORMAT, "version": __version__, "benchmark": args.benchmark,
            "source_sha256": _source_hash(), "runner_sha256": sha256(Path(__file__)),
            **provenance, "ids": [example["id"] for example in examples], "split": args.split,
            "split_seed": args.split_seed, "limit": args.limit, "methods": list(METHODS),
            "shared_initial_methods": list(METHODS), "replicates": args.replicates,
            "max_repairs": 3, "threshold": args.threshold, "final_rule": "keep_first",
            "exhaustion_policy": "execute_last", "external_tool_execution": False,
            "llm_model": generator.model, "llm_base_url": generator.base_url,
            "llm_provider": generator.provider, "llm_api_style": generator.api_style,
            "llm_gateway": generator.gateway, "llm_endpoint": generator.endpoint,
            "llm_auth_header": generator.auth_header, "llm_timeout_s": generator.timeout,
            "llm_max_retries": generator.max_retries, "reasoning_effort": generator.reasoning_effort,
            "temperature": generator.temperature, "max_completion_tokens": generator.max_completion_tokens,
            "jev_model": verifier.model, "jev_gateway": verifier.gateway, "jev_base_url": verifier.base_url,
            "jev_endpoint": verifier.base_url + "/v1/systemone", "jev_timeout_s": verifier.timeout,
            "jev_max_retries": verifier.max_retries, "workers": args.workers,
            "prices_usd_per_million": explicit_prices(args),
            "expected_records": len(examples) * args.replicates * len(METHODS), "synthetic": False,
            "metric": "apibank_firstcall_exact" if args.benchmark == "apibank" else "bfcl_python_ast_or_presence",
            "timing": {"api": "sum of generation and verification durations, including shared initial once per arm",
                       "wall": "initial generation wall time plus method loop; excludes offline scoring and queue waits",
                       "retry_wait": "reported separately; excluded from API duration"},
            "bootstrap": {"unit": "example ID, averaged across repetitions", "seed": 1729, "samples": 2000}}


def validate_resume(directory, settings, resume):
    manifest_path = directory / "manifest.json"
    config_id = _fingerprint(settings)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous_settings = {key: value for key, value in previous.items() if key not in {"config_id", "created_at"}}
        if previous.get("format_version") != FORMAT or previous.get("config_id") != config_id or previous_settings != settings:
            raise ValueError("Output contains a different code/data/model/interface/price fingerprint; choose a new output.")
        if not resume:
            raise ValueError("Output already exists; use --resume with an identical fingerprint or choose a new output.")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("Nonempty output has no additional-evaluation manifest; choose a new output.")
    elif resume:
        raise ValueError("Cannot resume an output without a manifest.")
    rows = read_jsonl(directory / "records.jsonl")
    expected = {(example, replicate, method) for example in settings["ids"]
                for replicate in range(settings["replicates"]) for method in METHODS}
    completed = set()
    for row in rows:
        key = (row["id"], row["replicate"], row["method"])
        if key not in expected or key in completed or row.get("config_id") != config_id:
            raise ValueError("Duplicate, foreign, or changed-fingerprint result record in resume files.")
        completed.add(key)
    initials = {}
    for initial in read_jsonl(directory / "initial.jsonl"):
        key = (initial["id"], initial["replicate"])
        if key + ("direct",) not in expected or key in initials or initial.get("config_id") != config_id:
            raise ValueError("Duplicate, foreign, or changed-fingerprint initial cache.")
        if not isinstance(initial.get("proposal"), dict) or not isinstance(initial.get("initial_wall_time_s"), (float, int)):
            raise ValueError("Invalid cached initial proposal/timing.")
        initials[key] = initial
    for row in rows:
        initial = initials.get((row["id"], row["replicate"]))
        if initial is None or initial["proposal"] != row["trace"][0]["proposal"]:
            raise ValueError("Resumed method record does not match its cached initial proposal.")
    return config_id, completed, initials


def _safe_error(exc, example, replicate, method):
    # Never emit raw exception text, response bodies, headers, or credentials, even from unknown exceptions.
    event = {"id": example["id"], "replicate": replicate, "method": method,
             "error_type": type(exc).__name__, "at": datetime.now(timezone.utc).isoformat()}
    for name in ("status_code", "attempts"):
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            event[name] = value
    code = getattr(exc, "provider_code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code):
        event["provider_code"] = code
    return event


def evaluation_summary(directory):
    summary = build_summary(directory)
    rows = read_jsonl(directory / "records.jsonl")
    manifest = summary["manifest"]
    for section in ("methods", "by_category", "by_replicate"):
        for key, metrics in summary[section].items():
            group = [row for row in rows if (row["method"] == key if section == "methods" else
                     f"{row['category']}/{row['method']}" == key if section == "by_category" else
                     f"{row['replicate']}/{row['method']}" == key)]
            metrics["incomplete_final_n"] = sum(bool(row["trace"][row["final_attempt"]]["proposal"].get("incomplete"))
                                                 for row in group)
            def function_names(calls):
                return [call.get("name") for call in calls] if isinstance(calls, list) else None

            metrics["function_change_rate"] = (statistics.fmean(
                function_names(row["trace"][0]["proposal"]["calls"]) != function_names(row["calls"])
                for row in group) if group else None)
            metrics["mean_wall_time_s"] = statistics.fmean(row["wall_time_s"] for row in group) if group else None
            metrics["p95_wall_time_s"] = percentile([row["wall_time_s"] for row in group], .95)
            metrics["mean_retry_wait_s"] = statistics.fmean(sum(
                step["proposal"].get("retry_wait_s", 0) + step["verification"].get("retry_wait_s", 0)
                for step in row["trace"]) for row in group) if group else None
            known = [row["accounting"]["estimated_usd"] for row in group if row["accounting"]["complete"]]
            metrics["known_usage_usd_per_1k"] = statistics.fmean(known) * 1000 if known else None
            metrics["unknown_cost_n"] = len(group) - len(known)
    repeats = {}
    for method in METHODS:
        per_run = [summary["by_replicate"][f"{replicate}/{method}"] for replicate in range(manifest["replicates"])]
        complete = all(sum(row["replicate"] == replicate and row["method"] == method for row in rows)
                       == len(manifest["ids"]) for replicate in range(manifest["replicates"]))
        values = [item["accuracy"] for item in per_run]
        repeats[method] = {"runs": [{"replicate": replicate, **item} for replicate, item in enumerate(per_run)],
                           "accuracy_mean": statistics.fmean(values) if complete and None not in values else None,
                           "accuracy_sample_sd": statistics.stdev(values) if complete and len(values) > 1 and None not in values else None,
                           "complete_repetitions": complete}
    summary["repetition_report"] = repeats
    summary["paired_dmfc_minus_direct"] = paired_delta(rows, "jev_joint", seed=1729, samples=2000)
    _write_json(directory / "summary.json", summary)
    # Rich additional-evaluation metrics live in JSON; the base summary.csv stays a compatible raw metric table.
    return summary


def execute(args, examples, score, settings, generator, verifier):
    directory = args.output.resolve()
    config_id, completed, initials = validate_resume(directory, settings, args.resume)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "manifest.json").exists():
        _write_json(directory / "manifest.json", {**settings, "config_id": config_id,
                                                  "created_at": datetime.now(timezone.utc).isoformat()})
    lock, stop = threading.Lock(), threading.Event()

    def append(name, value):
        with lock:
            with (directory / name).open("a") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def task(example, replicate):
        method = "initial"
        key = (example["id"], replicate)
        if all(key + (arm,) in completed for arm in METHODS) or stop.is_set():
            return
        try:
            initial = initials.get(key)
            if initial is None:
                started = time.perf_counter()
                proposal = generator.generate(public_input(example), [])
                initial = {"id": example["id"], "replicate": replicate, "config_id": config_id,
                           "proposal": proposal, "initial_wall_time_s": time.perf_counter() - started}
                append("initial.jsonl", initial)
            # Finish all model operations for the case before scoring either arm: no offline score feeds a later request.
            produced = []
            for method in METHODS:
                if key + (method,) in completed or stop.is_set():
                    continue
                record = run_example(public_input(example), generator, verifier, method=method, max_repairs=3,
                                     exhaustion_policy="execute_last", initial_proposal=initial["proposal"], executor=None)
                record["wall_time_s"] += initial["initial_wall_time_s"]
                record.update(id=example["id"], category=example["category"], replicate=replicate,
                              synthetic=False, config_id=config_id, final_rule="keep_first")
                produced.append(record)
            for record in produced:
                method = record["method"]
                for step in record["trace"]:
                    step["score"] = score(example, step["proposal"]["calls"])
                final_attempt = 0 if method == "jev_joint" and record["accepted"] is False else len(record["trace"]) - 1
                record["final_attempt"] = final_attempt
                record["calls"] = deepcopy(record["trace"][final_attempt]["proposal"]["calls"])
                record["score"] = deepcopy(record["trace"][final_attempt]["score"])
                record["accounting"] = token_cost(record, settings["prices_usd_per_million"])
                append("records.jsonl", record)
                with lock:
                    completed.add(key + (method,))
                    print(f"Completed {len(completed)}/{settings['expected_records']} method records.", flush=True)
        except Exception as exc:
            append("errors.jsonl", _safe_error(exc, example, replicate, method))
            stop.set()
            raise RuntimeError("Additional evaluation stopped after a recorded sanitized error; resume uses cached initials.") from None

    work = iter((example, replicate) for replicate in range(args.replicates) for example in examples)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = set()
            for _ in range(args.workers):
                item = next(work, None)
                if item is not None:
                    futures.add(pool.submit(task, *item))
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    item = next(work, None)
                    if item is not None and not stop.is_set():
                        futures.add(pool.submit(task, *item))
    finally:
        summary = evaluation_summary(directory)
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1 or args.max_retries < 0:
        raise ValueError("Use a finite threshold in [0,1] and nonnegative retries.")
    examples, score, provenance = load_data(args)
    generator, verifier = make_clients(args, preflight=True)
    try:
        settings = settings_for(args, examples, provenance, generator, verifier)
        config_id, completed, _ = validate_resume(args.output.resolve(), settings, args.resume)
    finally:
        generator.close()
        verifier.close()
    preflight = {"network_requests": 0, "run_enabled": args.run, "config_id": config_id,
                 "benchmark": args.benchmark, "model": args.model, "selected_examples": len(examples),
                 "main_scored_examples": sum(example["category"] != "live_relevance" for example in examples),
                 "replicates": args.replicates, "expected_records": settings["expected_records"],
                 "completed_records": len(completed), "llm_endpoint": settings["llm_endpoint"],
                 "jev_endpoint": settings["jev_endpoint"], "prices_usd_per_million": settings["prices_usd_per_million"]}
    print(json.dumps(preflight, indent=2))
    if not args.run:
        return preflight
    generator, verifier = make_clients(args)
    try:
        # Recheck exact live configuration before writing a manifest or sending a request.
        if settings_for(args, examples, provenance, generator, verifier) != settings:
            raise ValueError("Live client configuration differs from preflight.")
        summary = execute(args, examples, score, settings, generator, verifier)
    finally:
        generator.close()
        verifier.close()
    print(json.dumps({"status": summary["status"], "output": str(args.output.resolve()),
                      "completed_records": summary["completed_records"],
                      "repetition_report": summary["repetition_report"],
                      "paired_dmfc_minus_direct": summary["paired_dmfc_minus_direct"]}, indent=2))
    return summary


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print("Evaluation stopped: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from None
