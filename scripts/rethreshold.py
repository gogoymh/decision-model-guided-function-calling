"""Run JBFC again with another Jev threshold, from the first proposals of a finished run.

Every example keeps the first proposal it had in the source run (its Direct row), so
Direct is unchanged and each new JBFC row pairs with the source's JBFC row: the only
difference is the threshold of both Jev checks, which decides acceptance and appears in
the feedback the generator sees, as in any run with that threshold. Code
(source_sha256), data, generator settings, interface, prices, repair limit and
exhaustion policy must equal the source's; Jev's route may change on request
(--jev-gateway or JEV_GATEWAY) and is recorded. The output is a normal run with Direct
(copied) and JBFC (new), so summaries and paired analyses work as for any run.

This is an exploratory run: the thresholds of the paper's tables stay at the source's.

Usage: python scripts/rethreshold.py SOURCE --threshold 0.7 --output OUT [--workers 8] [--resume]
                                     [--jev-gateway direct|apim]
"""

import argparse
import json
import math
import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from extend_run import _relative, _sha256, current_settings  # noqa: E402
from raise_limit import JEV_FIELDS, LLM_FIELDS, initial_proposals, load_run  # noqa: E402

from jev_fc import cli  # noqa: E402
from jev_fc.benchmarks import load_examples  # noqa: E402
from jev_fc.jev import JevAPIError  # noqa: E402
from jev_fc.llm import LLMAPIError  # noqa: E402
from jev_fc.pipeline import run_example  # noqa: E402
from jev_fc.reporting import build_summary, read_jsonl, token_cost  # noqa: E402
from jev_fc.scoring import score_example  # noqa: E402

METHOD = "jev_joint"
NOT_COPIED = {"config_id", "created_at", "methods", "shared_initial_methods", "expected_records", "workers",
              "extends", "reused", "added", "jev_routes", "rethreshold"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a finished run with Direct rows (its first proposals)")
    parser.add_argument("--threshold", type=float, required=True, help="Jev threshold for both checks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default=str(ROOT / "data/bfcl/examples.jsonl"))
    parser.add_argument("--checker", default=str(ROOT / "data/bfcl/upstream/ast_checker.py"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jev-gateway", choices=("direct", "apim"), default=os.getenv("JEV_GATEWAY") or None)
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    threshold = args.threshold
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("--threshold must be in (0, 1].")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    source_dir, directory = Path(args.source), Path(args.output)
    source, rows = load_run(source_dir)
    if "direct" not in source["methods"]:
        raise ValueError("The source run needs Direct rows: they hold the first proposals.")
    if threshold == source["threshold"]:
        raise ValueError("The threshold equals the source run's; there is nothing to run again.")
    if source.get("source_sha256") != cli._source_hash():
        raise ValueError("The source run used different code; run again with the code that produced it.")
    if _sha256(args.data) != source["dataset_sha256"]:
        raise ValueError("The prepared data differ from the data of the source run.")
    shared = initial_proposals(source_dir, rows)
    examples = {e["id"]: e for e in load_examples(args.data)}
    limit, prices = source["max_repairs"], source["prices_usd_per_million"]

    generator = cli.FunctionCallingLLM()
    route_option = {"gateway": args.jev_gateway} if args.jev_gateway else {}
    verifier = cli.JevVerifier(function_threshold=threshold, arguments_threshold=threshold, **route_option)
    try:
        current = current_settings(generator, verifier)
        changed = [key for key in LLM_FIELDS if current[key] != source.get(key)]
        if source.get("jev_model") is not None and current["jev_model"] != source["jev_model"]:
            changed.append("jev_model")
        if changed:
            raise ValueError("Settings differ from the source run: " + ", ".join(changed))
        settings = {key: value for key, value in source.items() if key not in NOT_COPIED}
        settings.update({key: current[key] for key in JEV_FIELDS})
        settings.update(
            threshold=threshold, methods=["direct", METHOD], shared_initial_methods=["direct", METHOD],
            workers=args.workers, expected_records=len(source["ids"]) * source["replicates"] * 2, added=[METHOD],
            rethreshold={"method": METHOD, "from": source["threshold"], "to": threshold},
            extends={"run_dir": _relative(source_dir), "config_id": source["config_id"], "max_repairs": limit,
                     "threshold": source["threshold"], "records_sha256": _sha256(source_dir / "records.jsonl")})
        config_id = cli._fingerprint(settings)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()).get("config_id") != config_id:
                raise ValueError("Output directory holds a different run; choose a new --output directory.")
            if not args.resume:
                raise ValueError("Output directory already has a run. Pass --resume or select a new directory.")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            if any(directory.glob("*.jsonl")):
                raise ValueError("Found orphan run files without a manifest; choose a new output directory.")
            cli._write_json(manifest_path, {**settings, "config_id": config_id,
                                            "created_at": datetime.now(timezone.utc).isoformat()})
        done = {(r["id"], r["replicate"], r["method"]) for r in read_jsonl(directory / "records.jsonl")}
        lock, stop = threading.Lock(), threading.Event()

        def append(name, record):
            with lock:
                with (directory / name).open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        for row in rows:  # Direct is the shared first proposal: copied, never requested again
            if row["method"] == "direct" and (row["id"], row["replicate"], "direct") not in done:
                record = deepcopy(row)
                record.update(config_id=config_id, extended=False)
                record["accounting"] = token_cost(record, prices)
                append("records.jsonl", record)
        todo = [{"id": example_id, "replicate": replicate, "first": first}
                for (example_id, replicate), first in shared.items() if (example_id, replicate, METHOD) not in done]
        total = len(shared)
        count = total - len(todo)
        print(f"{count} of {total} JBFC rows are done; running {len(todo)} with threshold {threshold} "
              f"(source {source['threshold']}), r={limit}.", flush=True)

        def task(item):
            nonlocal count
            if stop.is_set():
                return
            try:
                example = examples[item["id"]]
                record = run_example(example, generator, verifier, METHOD, limit, source["exhaustion_policy"],
                                     initial_proposal=deepcopy(item["first"]))
                # Scoring is strictly after all proposal and verification work.
                for step in record["trace"]:
                    step["score"] = score_example(example, step["proposal"]["calls"], checker_path=args.checker)
                record.update(score=record["trace"][-1]["score"], replicate=item["replicate"], synthetic=False,
                              config_id=config_id, extended=False, added=True)
                record["accounting"] = token_cost(record, prices)
                append("records.jsonl", record)
                with lock:
                    count += 1
                    print(f"[{count}/{total}] {item['id']} {METHOD} r={record['repairs']}", flush=True)
            except Exception as exc:
                event = {"id": item["id"], "replicate": item["replicate"], "method": METHOD,
                         "error_type": type(exc).__name__, "at": datetime.now(timezone.utc).isoformat()}
                for field in ("status_code", "request_id", "attempts", "provider_code"):
                    value = getattr(exc, field, None)
                    if value is not None:
                        event[field] = value
                if isinstance(exc, (LLMAPIError, JevAPIError, ValueError)):
                    event["message"] = str(exc)  # fixed, sanitized text
                append("errors.jsonl", event)
                stop.set()
                raise

        work = iter(todo)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(task, item) for item in (next(work, None) for _ in range(args.workers)) if item}
            while futures:
                finished, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                    item = next(work, None)
                    if item and not stop.is_set():
                        futures.add(pool.submit(task, item))
    finally:
        generator.close()
        verifier.close()
    summary = build_summary(directory)
    print(f"Saved {summary['completed_records']} records and summary.json to {directory}")
    return summary


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
