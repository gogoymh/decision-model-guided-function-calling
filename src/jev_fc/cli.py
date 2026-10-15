"""Dataset preparation, paired API experiments, resume, and paper reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .benchmarks import BFCL_REVISION, DEFAULT_CATEGORIES, load_examples, prepare_bfcl
from .jev import JevVerifier
from .llm import FunctionCallingLLM
from .pipeline import METHODS, run_example
from .reporting import build_summary, read_jsonl, token_cost, update_paper
from .scoring import load_official_checker, score_example


def stable_split(examples, split="test", seed=1729):
    """Fixed per-category 10% development set, minimum one case, before any API call."""
    by_category = defaultdict(list)
    for example in examples:
        by_category[example["category"]].append(example)
    result = []
    for category in sorted(by_category):
        ordered = sorted(by_category[category], key=lambda e: hashlib.sha256(f"{seed}:{e['id']}".encode()).hexdigest())
        count = max(1, int(len(ordered) * .1))
        result.extend(ordered if split == "all" else ordered[:count] if split == "dev" else ordered[count:])
    # Interleave categories for representative limited smoke runs.
    random.Random(seed).shuffle(result)
    return result


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _source_hash():
    source = b"".join(p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
    return hashlib.sha256(source).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _prices(generator):
    known = generator.provider == "openai" and generator.model == "gpt-5.4-mini-2026-03-17"
    defaults = {"llm_input": .75 if known else None, "llm_cached_input": .075 if known else None,
                "llm_output": 4.5 if known else None, "jev_input": .042,
                "llm_cache_write_5m": None, "llm_cache_write_1h": None}
    result = {}
    for name, default in defaults.items():
        variable = name.upper() + "_PRICE"
        raw = os.getenv(variable, "").strip()
        try:
            value = float(raw) if raw else default
        except ValueError:
            raise ValueError(f"{variable} must be a nonnegative finite number or blank.") from None
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{variable} must be a nonnegative finite number or blank.")
        result[name] = value
    return result


def check_config():
    """Validate routing and options without sending a request or showing secrets."""
    generator = FunctionCallingLLM(api_key="configuration-check-only")
    verifier = None
    try:
        verifier = JevVerifier(api_key="configuration-check-only")
        if generator.gateway == "apim":
            llm_key = jev_key = bool(os.getenv("APIM_SUBSCRIPTION_KEY", "").strip())
        else:
            provider_key = "ANTHROPIC_API_KEY" if generator.provider == "anthropic" else "OPENAI_API_KEY"
            llm_key = bool(os.getenv("LLM_API_KEY") or os.getenv(provider_key))
            jev_key = bool(os.getenv("TYPESAFE_API_KEY"))
        print(json.dumps({
            "network_requests": 0, "gateway": generator.gateway,
            "llm_provider": generator.provider, "llm_api_style": generator.api_style,
            "llm_endpoint": generator.endpoint, "llm_model": generator.model,
            "jev_endpoint": verifier.base_url + "/v1/systemone", "jev_model": verifier.model,
            "llm_key_configured": llm_key, "jev_key_configured": jev_key,
            "prices_usd_per_million": _prices(generator),
        }, indent=2))
        return 0 if llm_key and jev_key else 1
    finally:
        generator.close()
        if verifier is not None:
            verifier.close()


def run(args):
    directory = Path(args.output)
    data_path = Path(args.data)
    if data_path.is_dir():
        data_path = data_path / "examples.jsonl"
    examples = load_examples(data_path)
    load_official_checker(args.checker)
    if args.categories:
        categories = set(args.categories.split(","))
        examples = [e for e in examples if e["category"] in categories]
        if not examples:
            raise ValueError("No examples match the requested categories.")
    selected = stable_split(examples, args.split, args.split_seed)
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("Selected split is empty")
    methods = args.methods.split(",")
    if len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
        raise ValueError(f"Methods must be unique entries from {METHODS}")
    if not 0 <= args.max_repairs <= 3 or args.replicates < 1 or args.workers < 1:
        raise ValueError("Use 0 <= max-repairs <= 3, replicates >= 1, workers >= 1")
    generator = FunctionCallingLLM()
    verifier = JevVerifier(function_threshold=args.threshold, arguments_threshold=args.threshold) if any(m.startswith("jev_") for m in methods) else None
    prices = _prices(generator)
    settings = {
        "version": __version__, "source_sha256": _source_hash(),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_revisions": sorted({e.get("dataset_revision", "unknown") for e in selected}),
        "ids": [e["id"] for e in selected], "split": args.split, "split_seed": args.split_seed,
        "limit": args.limit, "methods": methods, "replicates": args.replicates,
        "max_repairs": args.max_repairs, "threshold": args.threshold,
        "exhaustion_policy": args.exhaustion_policy,
        "llm_model": generator.model, "llm_base_url": generator.base_url,
        "llm_provider": generator.provider, "llm_api_style": generator.api_style,
        "llm_gateway": generator.gateway, "llm_endpoint": generator.endpoint,
        "llm_auth_header": generator.auth_header,
        "llm_timeout_s": generator.timeout, "llm_max_retries": generator.max_retries,
        "reasoning_effort": generator.reasoning_effort, "temperature": generator.temperature,
        "max_completion_tokens": generator.max_completion_tokens,
        "jev_model": verifier.model if verifier is not None else None,
        "jev_base_url": verifier.base_url if verifier is not None else None,
        "jev_gateway": verifier.gateway if verifier is not None else None,
        "jev_endpoint": verifier.base_url + "/v1/systemone" if verifier is not None else None,
        "jev_timeout_s": verifier.timeout if verifier is not None else None,
        "jev_max_retries": verifier.max_retries if verifier is not None else None,
        "workers": args.workers,
        "prices_usd_per_million": prices,
        "expected_records": len(selected) * args.replicates * len(methods),
        "synthetic": False,
    }
    config_id = _fingerprint(settings)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("config_id") != config_id:
            raise ValueError("Output directory has a different code/data/config fingerprint. Choose a new --output directory.")
        if not args.resume:
            raise ValueError("Output directory already has a run. Pass --resume or select a new directory.")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "records.jsonl").exists() or (directory / "initial.jsonl").exists():
            raise ValueError("Found orphan run files without a manifest; choose a new output directory.")
        _write_json(manifest_path, {**settings, "config_id": config_id, "created_at": datetime.now(timezone.utc).isoformat()})
    rows = read_jsonl(directory / "records.jsonl")
    completed = {(r["id"], r["replicate"], r["method"]) for r in rows}
    initials = {(r["id"], r["replicate"]): r["proposal"] for r in read_jsonl(directory / "initial.jsonl")}
    write_lock = threading.Lock()
    stop = threading.Event()
    count = len(completed)

    def append(name, record):
        with write_lock:
            with (directory / name).open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def task(example, replicate):
        nonlocal count
        current_method = "initial"
        if all((example["id"], replicate, method) in completed for method in methods) or stop.is_set():
            return
        try:
            key = (example["id"], replicate)
            initial = initials.get(key)
            if initial is None:
                initial = generator.generate({k: example[k] for k in ("id", "category", "messages", "tools")})
                append("initial.jsonl", {"id": example["id"], "replicate": replicate, "proposal": initial})
            for method in methods:
                current_method = method
                if (example["id"], replicate, method) in completed or stop.is_set():
                    continue
                record = run_example(example, generator, verifier, method, args.max_repairs,
                                     args.exhaustion_policy, initial_proposal=initial)
                # Scoring is strictly after all proposal/verification operations.
                for step in record["trace"]:
                    step["score"] = score_example(example, step["proposal"]["calls"], checker_path=args.checker)
                record.update({"score": record["trace"][-1]["score"], "replicate": replicate,
                               "synthetic": False, "config_id": config_id})
                record["accounting"] = token_cost(record, prices)
                append("records.jsonl", record)
                with write_lock:
                    count += 1
                    print(f"[{count}/{settings['expected_records']}] {example['id']} {method}", flush=True)
        except Exception as exc:
            # No raw response, HTTP headers or API keys enter error logs.
            event = {"id": example["id"], "replicate": replicate,
                     "method": current_method, "error_type": type(exc).__name__,
                     "at": datetime.now(timezone.utc).isoformat()}
            for field in ("status_code", "request_id", "attempts"):
                value = getattr(exc, field, None)
                if value is not None:
                    event[field] = value
            append("errors.jsonl", event)
            stop.set()
            raise

    work = iter((example, replicate) for replicate in range(args.replicates) for example in selected)
    print(f"Prepared {len(selected)} {args.split} examples; {settings['expected_records']} method records; {len(completed)} already complete.")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = set()
            for _ in range(args.workers):
                item = next(work, None)
                if item:
                    futures.add(pool.submit(task, *item))
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    item = next(work, None)
                    if item and not stop.is_set():
                        futures.add(pool.submit(task, *item))
    finally:
        generator.close()
        if verifier and hasattr(verifier, "close"):
            verifier.close()
    summary = build_summary(directory)
    print(f"Saved {summary['completed_records']} records and summary.json to {directory}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download the pinned official BFCL data and AST checker")
    prepare.add_argument("--output", default="data/bfcl")
    prepare.add_argument("--revision", default=BFCL_REVISION)
    prepare.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    experiment = sub.add_parser("run", help="Run a paired live experiment; requires keys")
    experiment.add_argument("--data", default="data/bfcl/examples.jsonl")
    experiment.add_argument("--checker", default="data/bfcl/upstream/ast_checker.py")
    experiment.add_argument("--output", default="results/main")
    experiment.add_argument("--methods", default=",".join(METHODS))
    experiment.add_argument("--split", choices=("dev", "test", "all"), default="test")
    experiment.add_argument("--split-seed", type=int, default=1729)
    experiment.add_argument("--categories")
    experiment.add_argument("--max-repairs", type=int, default=3)
    experiment.add_argument("--threshold", type=float, default=.8)
    experiment.add_argument("--exhaustion-policy", choices=("execute_last", "abstain"), default="execute_last")
    experiment.add_argument("--replicates", type=int, default=3)
    experiment.add_argument("--workers", type=int, default=4)
    experiment.add_argument("--limit", type=int)
    experiment.add_argument("--resume", action="store_true")
    report = sub.add_parser("report", help="Aggregate empirical traces, optionally populate manuscript table")
    report.add_argument("run_dir")
    report.add_argument("--paper")
    template = sub.add_parser("paper-template", help="Restore the unrun main-results table")
    template.add_argument("--paper", default="paper/main.tex")
    sub.add_parser("demo", help="Run an offline, synthetic bounded-repair demo (not benchmark results)")
    sub.add_parser("check-config", help="Validate API settings without making any network requests")
    args = parser.parse_args()
    try:
        if args.command == "check-config":
            return check_config()
        elif args.command == "prepare":
            manifest = prepare_bfcl(args.output, args.revision, args.categories.split(","))
            print(json.dumps(manifest, indent=2))
        elif args.command == "run":
            run(args)
        elif args.command == "report":
            summary = build_summary(args.run_dir)
            if args.paper:
                update_paper(args.paper, summary)
            print(json.dumps({"status": summary["status"], "methods": summary["methods"]}, indent=2))
        elif args.command == "paper-template":
            update_paper(args.paper)
        elif args.command == "demo":
            from .demo import demo
            print(json.dumps(demo(), indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
