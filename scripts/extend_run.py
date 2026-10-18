"""Extend the JBFC trajectories of a finished run to a higher repair limit (Figure 2).

Neither model sees the repair limit and no function runs before the end, so the
trajectory for a limit r is the first r+1 attempts of a trajectory for any larger
limit. Rows accepted within the source limit therefore stay as they are. Rows that
reached it continue from their saved history: the saved attempts are replayed
through the same pipeline and checked against the record, so every new request is
exactly the one an uninterrupted run with the larger limit would have sent.

The source run must come from the same code (source_sha256) and the same models,
interfaces and prices. Only Jev's route may change, and only on request
(--jev-gateway or JEV_GATEWAY): the same Jev model then answers the same questions
directly or through the gateway, and the manifest lists the route of each attempt
range in jev_routes. The result is a normal run directory with methods=[jev_joint].

Usage: python scripts/extend_run.py SOURCE_RUN --max-repairs 7 --output OUT [--workers 8] [--resume]
                                   [--jev-gateway direct|apim]
"""

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from jev_fc import cli  # noqa: E402
from jev_fc.benchmarks import load_examples  # noqa: E402
from jev_fc.jev import JevAPIError  # noqa: E402
from jev_fc.llm import LLMAPIError  # noqa: E402
from jev_fc.pipeline import run_example  # noqa: E402
from jev_fc.reporting import build_summary, read_jsonl, token_cost  # noqa: E402
from jev_fc.scoring import score_example  # noqa: E402

METHOD = "jev_joint"
# Everything that shapes a request, a verdict or a cost must equal the source run.
SAME_AS_SOURCE = ("llm_model", "llm_base_url", "llm_provider", "llm_api_style", "llm_gateway", "llm_endpoint",
                  "llm_auth_header", "llm_timeout_s", "llm_max_retries", "reasoning_effort", "temperature",
                  "max_completion_tokens", "jev_model", "jev_base_url", "jev_gateway", "jev_endpoint",
                  "jev_timeout_s", "jev_max_retries", "prices_usd_per_million")
# How requests reach Jev; may differ from the source run only when requested and recorded.
JEV_ROUTE = ("jev_gateway", "jev_base_url", "jev_endpoint")


def current_settings(generator, verifier):
    return {
        "llm_model": generator.model, "llm_base_url": generator.base_url, "llm_provider": generator.provider,
        "llm_api_style": generator.api_style, "llm_gateway": generator.gateway, "llm_endpoint": generator.endpoint,
        "llm_auth_header": generator.auth_header, "llm_timeout_s": generator.timeout,
        "llm_max_retries": generator.max_retries, "reasoning_effort": generator.reasoning_effort,
        "temperature": generator.temperature, "max_completion_tokens": generator.max_completion_tokens,
        "jev_model": verifier.model, "jev_base_url": verifier.base_url, "jev_gateway": verifier.gateway,
        "jev_endpoint": verifier.base_url + "/v1/systemone", "jev_timeout_s": verifier.timeout,
        "jev_max_retries": verifier.max_retries, "prices_usd_per_million": cli._prices(generator),
    }


def compare_settings(source, current, jev_gateway=None):
    """Settings that differ from the source run, and the Jev route fields to record.

    Jev's route may differ only when jev_gateway was requested. Then, and whenever the
    source already lists routes, jev_routes gives the route used from each attempt on.
    """
    route_change = [key for key in JEV_ROUTE if current[key] != source.get(key)]
    changed = [key for key in SAME_AS_SOURCE if current[key] != source.get(key)
               and not (jev_gateway and key in JEV_ROUTE)]
    record = {}
    if (route_change and jev_gateway) or "jev_routes" in source:
        routes = deepcopy(source.get("jev_routes") or [{"first_attempt": 0, **{key: source.get(key) for key in JEV_ROUTE}}])
        if route_change:
            routes.append({"first_attempt": source["max_repairs"] + 1, **{key: current[key] for key in JEV_ROUTE}})
        record = {"jev_routes": routes, **{key: current[key] for key in JEV_ROUTE}}
    return changed, record


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_source(source_dir, max_repairs):
    """Return the manifest and JBFC rows of a complete source run that may be extended."""
    source_dir = Path(source_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    rows = read_jsonl(source_dir / "records.jsonl")
    methods, cap = manifest.get("methods") or [], manifest.get("max_repairs")
    if manifest.get("synthetic") is not False or METHOD not in methods:
        raise ValueError("The source must be a real run that includes JBFC (jev_joint).")
    if isinstance(cap, bool) or not isinstance(cap, int) or isinstance(max_repairs, bool) or max_repairs <= cap:
        raise ValueError("The new repair limit must be larger than the source run's limit.")
    if manifest.get("exhaustion_policy") != "execute_last":
        raise ValueError("Only execute_last runs can be extended.")
    if manifest.get("source_sha256") != cli._source_hash():
        raise ValueError("The source run used different code; extend it with the code that produced it.")
    expected = {(i, r, m) for i in manifest["ids"] for r in range(manifest["replicates"]) for m in methods}
    seen = {(row["id"], row["replicate"], row["method"]) for row in rows}
    if (len(seen) != len(rows) or seen != expected
            or any(row.get("config_id") != manifest.get("config_id") for row in rows)):
        raise ValueError("The source run must be complete, without foreign or duplicate records.")
    joint = [row for row in rows if row["method"] == METHOD]
    for row in joint:
        trace = row["trace"]
        if (not trace or len(trace) > cap + 1 or row.get("max_repairs") != cap
                or [step.get("attempt") for step in trace] != list(range(len(trace)))
                or any(step["gate_pass"] for step in trace[:-1])
                or row.get("accepted") is not trace[-1]["gate_pass"]
                or (not trace[-1]["gate_pass"] and len(trace) != cap + 1)):
            raise ValueError(f"Source trajectory {row['id']} violates its stopping rule.")
    return manifest, joint


class ReplayGenerator:
    """Return the saved proposals for recorded attempts, then ask the real generator."""

    def __init__(self, generator, trace):
        self.generator, self.trace, self.attempt = generator, trace, 0

    def generate(self, example, history=None):
        self.attempt += 1  # attempt 0 is the shared initial proposal, not a generate() call
        if self.attempt <= len(self.trace):
            # Up to the first new attempt, the history must be exactly the saved records.
            saved = [{"calls": step["proposal"]["calls"], "feedback": step["feedback"]}
                     for step in self.trace[:self.attempt]]
            if [{"calls": item["calls"], "feedback": item["feedback"]} for item in (history or [])] != saved:
                raise ValueError("The rebuilt repair history differs from the saved trajectory.")
        if self.attempt < len(self.trace):
            return deepcopy(self.trace[self.attempt]["proposal"])
        return self.generator.generate(example, history)


class ReplayVerifier:
    """Return the saved verdicts for recorded attempts, then ask the real verifier."""

    def __init__(self, verifier, trace):
        self.verifier, self.trace, self.attempt = verifier, trace, 0

    def verify(self, example, calls):
        attempt, self.attempt = self.attempt, self.attempt + 1
        if attempt < len(self.trace):
            if calls != self.trace[attempt]["proposal"]["calls"]:
                raise ValueError("A replayed proposal differs from the saved trajectory.")
            return deepcopy(self.trace[attempt]["verification"])
        return self.verifier.verify(example, calls)


def continue_row(example, row, generator, verifier, max_repairs):
    """Continue one exhausted JBFC row; the replayed prefix must match the record exactly."""
    saved = row["trace"]
    record = run_example(example, ReplayGenerator(generator, saved), ReplayVerifier(verifier, saved),
                         METHOD, max_repairs, "execute_last", initial_proposal=saved[0]["proposal"])
    for old, new in zip(saved, record["trace"]):
        if any(old[key] != new[key] for key in ("proposal", "verification", "schema", "feedback", "gate_pass")):
            raise ValueError(f"Replay of {row['id']} differs from the saved trajectory.")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--max-repairs", type=int, default=7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default=str(ROOT / "data/bfcl/examples.jsonl"))
    parser.add_argument("--checker", default=str(ROOT / "data/bfcl/upstream/ast_checker.py"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jev-gateway", choices=("direct", "apim"), default=os.getenv("JEV_GATEWAY") or None,
                        help="send Jev requests directly to TypeSafe (TYPESAFE_API_KEY) or through the gateway; "
                             "recorded in the manifest when it differs from the source run")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    source_dir, directory = Path(args.source), Path(args.output)
    source, joint = check_source(source_dir, args.max_repairs)
    data_path = Path(args.data)
    if _sha256(data_path) != source["dataset_sha256"]:
        raise ValueError("The prepared data differ from the data of the source run.")
    examples = {e["id"]: e for e in load_examples(data_path)}

    generator = cli.FunctionCallingLLM()
    route_option = {"gateway": args.jev_gateway} if args.jev_gateway else {}
    verifier = cli.JevVerifier(function_threshold=source["threshold"], arguments_threshold=source["threshold"],
                               **route_option)
    try:
        current = current_settings(generator, verifier)
        changed, route = compare_settings(source, current, args.jev_gateway)
        if changed:
            raise ValueError("Settings differ from the source run: " + ", ".join(changed))
        if route.get("jev_routes") and len(route["jev_routes"]) > len(source.get("jev_routes") or [None]):
            print(f"Jev route: {current['jev_gateway']} ({current['jev_base_url']}) from attempt "
                  f"{source['max_repairs'] + 1}; earlier attempts used {source.get('jev_gateway')}.", flush=True)
        skip = {"config_id", "created_at", "methods", "max_repairs", "expected_records", "workers"}
        settings = {key: value for key, value in source.items() if key not in skip}
        settings.update(route)
        settings.update(methods=[METHOD], max_repairs=args.max_repairs, workers=args.workers,
                        expected_records=len(joint),
                        extends={"run_dir": _relative(source_dir), "config_id": source["config_id"],
                                 "max_repairs": source["max_repairs"],
                                 "records_sha256": _sha256(source_dir / "records.jsonl")})
        config_id = cli._fingerprint(settings)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()).get("config_id") != config_id:
                raise ValueError("Output directory holds a different extension; choose a new --output directory.")
            if not args.resume:
                raise ValueError("Output directory already has a run. Pass --resume or select a new directory.")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            if any(directory.glob("*.jsonl")):
                raise ValueError("Found orphan run files without a manifest; choose a new output directory.")
            cli._write_json(manifest_path, {**settings, "config_id": config_id,
                                            "created_at": datetime.now(timezone.utc).isoformat()})
        done = {(r["id"], r["replicate"]) for r in read_jsonl(directory / "records.jsonl")}
        prices = source["prices_usd_per_million"]
        lock, stop = threading.Lock(), threading.Event()

        def append(name, record):
            with lock:
                with (directory / name).open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        todo = []
        for row in joint:
            if (row["id"], row["replicate"]) in done:
                continue
            if row["accepted"]:
                # Accepted within the source limit: a larger limit stops at the same attempt.
                record = deepcopy(row)
                record.update(max_repairs=args.max_repairs, config_id=config_id, extended=False)
                record["accounting"] = token_cost(record, prices)
                append("records.jsonl", record)
            else:
                todo.append(row)
        count = len(joint) - len(todo)
        print(f"{count} of {len(joint)} JBFC rows stopped within r={source['max_repairs']}; "
              f"continuing {len(todo)} rows up to r={args.max_repairs}.", flush=True)

        def task(row):
            nonlocal count
            if stop.is_set():
                return
            try:
                example = examples[row["id"]]
                record = continue_row(example, row, generator, verifier, args.max_repairs)
                # Scoring is strictly after all proposal and verification work.
                for step in record["trace"]:
                    step["score"] = score_example(example, step["proposal"]["calls"], checker_path=args.checker)
                record.update(score=record["trace"][-1]["score"], replicate=row["replicate"], synthetic=False,
                              config_id=config_id, extended=True)
                record["accounting"] = token_cost(record, prices)
                append("records.jsonl", record)
                with lock:
                    count += 1
                    print(f"[{count}/{len(joint)}] {row['id']} r={record['repairs']} "
                          f"{'accepted' if record['accepted'] else 'exhausted'}", flush=True)
            except Exception as exc:
                event = {"id": row["id"], "replicate": row["replicate"], "method": METHOD,
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
