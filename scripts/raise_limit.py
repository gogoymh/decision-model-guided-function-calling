"""Raise the repair limit of a finished run, reusing every attempt it already recorded.

Neither model sees the repair limit and no function runs before the end, so the
trajectory under a limit r is the first r+1 attempts of a trajectory under any
larger limit. For each method of the source run:

- direct makes no repairs, so its rows stay as they are;
- schema_only, jev_function, jev_arguments, jev_joint and jev_joint_latest stop at
  the first passing proposal: rows that passed within the old limit stay, and rows
  that reached it continue from their saved history up to the new limit;
- self_refine always uses every repair, so every row continues.

Continued rows replay their saved attempts through the same pipeline and must match
the record, so every new request is the one an uninterrupted run with the new limit
would have sent. --reuse METHOD=RUN takes a method's rows from a run of the same
trajectories with a limit at least as large, cut at the new limit (for JBFC, the
Figure 2 extension, so Table 2 agrees with Figure 2). --add METHOD runs a method the
source lacks, starting from the source's shared first proposals (JBFC next to a
Direct-only run). The source must come from the same code (source_sha256), data,
models, interfaces and prices; Jev's route may change only on request
(--jev-gateway or JEV_GATEWAY) and is then recorded. The output is a normal run.

Usage: python scripts/raise_limit.py SOURCE --max-repairs 3 --output OUT [--reuse jev_joint=RUN]
                                    [--add jev_joint] [--workers 8] [--resume] [--jev-gateway direct|apim]
"""

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from extend_run import _relative, _sha256, compare_settings, current_settings  # noqa: E402

from jev_fc import cli  # noqa: E402
from jev_fc.benchmarks import load_examples  # noqa: E402
from jev_fc.jev import JevAPIError  # noqa: E402
from jev_fc.llm import LLMAPIError  # noqa: E402
from jev_fc.pipeline import METHODS, run_example  # noqa: E402
from jev_fc.reporting import build_summary, read_jsonl, token_cost  # noqa: E402
from jev_fc.scoring import score_example  # noqa: E402

STOP_AT_PASS = ("schema_only", "jev_function", "jev_arguments", "jev_joint", "jev_joint_latest")
STEP_FIELDS = ("proposal", "verification", "schema", "feedback", "gate_pass")
JEV_FIELDS = ("jev_model", "jev_base_url", "jev_gateway", "jev_endpoint", "jev_timeout_s", "jev_max_retries")
LLM_FIELDS = ("llm_model", "llm_base_url", "llm_provider", "llm_api_style", "llm_gateway", "llm_endpoint",
              "llm_auth_header", "llm_timeout_s", "llm_max_retries", "reasoning_effort", "temperature",
              "max_completion_tokens", "prices_usd_per_million")


def check_stopping_rule(row, cap):
    """Each method's recorded trajectory must follow the rule that ended it."""
    trace, method = row.get("trace") or [], row["method"]
    budget = 0 if method == "direct" else cap
    gates = [step.get("gate_pass") for step in trace]
    ok = (bool(trace) and len(trace) <= budget + 1 and row.get("max_repairs") == cap
          and [step.get("attempt") for step in trace] == list(range(len(trace)))
          and row.get("repairs") == len(trace) - 1 and not any(gates[:-1]))
    if method == "self_refine":
        ok = ok and len(trace) == budget + 1 and gates[-1] is True
    elif method in STOP_AT_PASS:
        ok = ok and (gates[-1] is True or len(trace) == budget + 1) and row.get("accepted") is gates[-1]
    else:
        ok = ok and len(trace) == 1
    if not ok:
        raise ValueError(f"{method} trajectory {row['id']} violates its stopping rule.")


def load_run(run_dir):
    """Manifest and rows of a complete, real execute_last run."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = read_jsonl(run_dir / "records.jsonl")
    methods, cap = manifest.get("methods") or [], manifest.get("max_repairs")
    if manifest.get("synthetic") is not False or manifest.get("exhaustion_policy") != "execute_last":
        raise ValueError(f"{run_dir} must be a real execute_last run.")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0 or not set(methods) <= set(METHODS):
        raise ValueError(f"{run_dir} has an invalid manifest.")
    expected = {(i, r, m) for i in manifest["ids"] for r in range(manifest["replicates"]) for m in methods}
    seen = [(row["id"], row["replicate"], row["method"]) for row in rows]
    if (len(set(seen)) != len(seen) or set(seen) != expected
            or any(row.get("config_id") != manifest.get("config_id") for row in rows)):
        raise ValueError(f"{run_dir} must be complete, without foreign or duplicate records.")
    for row in rows:
        check_stopping_rule(row, cap)
    return manifest, rows


def initial_proposals(run_dir, rows):
    """The shared first proposal of every (id, replicate); all methods must start from it."""
    shared = {}
    for row in rows:
        key = (row["id"], row["replicate"])
        first = row["trace"][0]["proposal"]
        if shared.setdefault(key, first) != first or row.get("initial_proposal_shared") is not True:
            raise ValueError(f"Methods of {row['id']} do not share one first proposal.")
    saved = {(r["id"], r["replicate"]): r["proposal"] for r in read_jsonl(Path(run_dir) / "initial.jsonl")}
    if saved and saved != shared:
        raise ValueError("initial.jsonl differs from the records' first proposals.")
    return shared


def cut_row(row, limit, prices):
    """The record a run with a smaller limit would have written (stop-at-pass methods, execute_last)."""
    row = deepcopy(row)
    row["trace"] = row["trace"][:limit + 1]
    terminal = row["trace"][-1]
    accepted = terminal["gate_pass"]
    row.update(calls=terminal["proposal"]["calls"], score=terminal["score"], accepted=accepted,
               exhausted=not accepted, repairs=len(row["trace"]) - 1, max_repairs=limit,
               dispatched=True, abstained=False, wall_time_s=None,
               logical_latency_s=sum(t["proposal"].get("latency_s", 0) + t["verification"].get("latency_s", 0)
                                     for t in row["trace"]))
    row["accounting"] = token_cost(row, prices)
    return row


class Replay:
    """Saved proposals and verdicts for recorded attempts, then the real generator and verifier."""

    def __init__(self, generator, verifier, trace, method):
        self.generator, self.verifier, self.trace, self.method = generator, verifier, trace, method
        self.generated = self.checked = 0

    def generate(self, example, history=None):
        self.generated += 1  # attempt 0 is the shared first proposal, not a generate() call
        attempt = self.generated
        if attempt <= len(self.trace):
            # Up to the first new attempt, the prompt history must be exactly the saved records.
            saved = [{"calls": step["proposal"]["calls"], "feedback": step["feedback"]}
                     for step in self.trace[:attempt]]
            if self.method == "jev_joint_latest":
                saved = saved[-1:]
            if [{"calls": item["calls"], "feedback": item["feedback"]} for item in (history or [])] != saved:
                raise ValueError("The rebuilt repair history differs from the saved trajectory.")
        if attempt < len(self.trace):
            return deepcopy(self.trace[attempt]["proposal"])
        return self.generator.generate(example, history)

    def verify(self, example, calls):
        attempt, self.checked = self.checked, self.checked + 1
        if attempt < len(self.trace):
            if calls != self.trace[attempt]["proposal"]["calls"]:
                raise ValueError("A replayed proposal differs from the saved trajectory.")
            return deepcopy(self.trace[attempt]["verification"])
        return self.verifier.verify(example, calls)


def continue_row(example, row, generator, verifier, limit):
    """Continue one row to the new limit; the replayed prefix must match the record."""
    method, saved = row["method"], row["trace"]
    replay = Replay(generator, verifier, saved, method)
    record = run_example(example, replay, replay if method.startswith("jev_") else None, method, limit,
                         "execute_last", initial_proposal=saved[0]["proposal"])
    for index, (old, new) in enumerate(zip(saved, record["trace"])):
        # Self-refine marks its last proposal as final; under a larger limit that attempt is not last.
        fields = STEP_FIELDS[:-1] if method == "self_refine" and index == len(saved) - 1 else STEP_FIELDS
        if any(old[key] != new[key] for key in fields):
            raise ValueError(f"Replay of {method} {row['id']} differs from the saved trajectory.")
    return record


def reused_rows(path, method, limit, source, shared):
    """Rows of `method` from another run of the same trajectories, cut at the new limit."""
    manifest, rows = load_run(path)
    if method not in manifest["methods"] or method not in STOP_AT_PASS or manifest["max_repairs"] < limit:
        raise ValueError(f"--reuse {method} needs a run of that method with a limit of at least {limit}.")
    for key in ("source_sha256", "dataset_sha256", "ids", "replicates", "threshold", "llm_model",
                "llm_api_style", "reasoning_effort", "max_completion_tokens", "jev_model", "prices_usd_per_million"):
        if manifest.get(key) != source.get(key):
            raise ValueError(f"--reuse {method}: the run differs from the source in {key}.")
    out = {}
    for row in rows:
        if row["method"] != method:
            continue
        if row["trace"][0]["proposal"] != shared[(row["id"], row["replicate"])]:
            raise ValueError(f"--reuse {method}: {row['id']} starts from another first proposal.")
        out[(row["id"], row["replicate"])] = cut_row(row, limit, source["prices_usd_per_million"])
    return manifest, out


def parse_reuse(items):
    reuse = {}
    for item in items or []:
        method, _, path = item.partition("=")
        if method not in METHODS or not path or method in reuse:
            raise ValueError("--reuse takes METHOD=RUN, once per method.")
        reuse[method] = Path(path)
    return reuse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--max-repairs", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse", action="append", help="METHOD=RUN: take METHOD's rows from RUN, cut")
    parser.add_argument("--add", action="append", default=[], help="a method the source lacks, run from its first proposals")
    parser.add_argument("--data", default=str(ROOT / "data/bfcl/examples.jsonl"))
    parser.add_argument("--checker", default=str(ROOT / "data/bfcl/upstream/ast_checker.py"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jev-gateway", choices=("direct", "apim"), default=os.getenv("JEV_GATEWAY") or None)
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    source_dir, directory, limit = Path(args.source), Path(args.output), args.max_repairs
    source, rows = load_run(source_dir)
    cap = source["max_repairs"]
    if isinstance(limit, bool) or limit <= cap:
        raise ValueError("The new repair limit must be larger than the source run's limit.")
    if source.get("source_sha256") != cli._source_hash():
        raise ValueError("The source run used different code; raise its limit with the code that produced it.")
    if _sha256(args.data) != source["dataset_sha256"]:
        raise ValueError("The prepared data differ from the data of the source run.")
    added = list(dict.fromkeys(args.add))
    if any(m not in METHODS or m in source["methods"] or m == "direct" for m in added):
        raise ValueError("--add takes repair methods the source run does not have.")
    reuse = parse_reuse(args.reuse)
    if any(m not in source["methods"] for m in reuse):
        raise ValueError("--reuse takes methods of the source run.")
    shared = initial_proposals(source_dir, rows)
    examples = {e["id"]: e for e in load_examples(args.data)}
    methods = [m for m in METHODS if m in source["methods"] or m in added]
    needs_jev = any(m.startswith("jev_") for m in methods)

    generator = cli.FunctionCallingLLM()
    route_option = {"gateway": args.jev_gateway} if args.jev_gateway else {}
    verifier = cli.JevVerifier(function_threshold=source["threshold"], arguments_threshold=source["threshold"],
                               **route_option) if needs_jev else None
    try:
        settings = {key: value for key, value in source.items()
                    if key not in {"config_id", "created_at", "methods", "shared_initial_methods", "max_repairs",
                                   "expected_records", "workers", "extends", "reused", "added"}}
        if verifier is not None and source.get("jev_model") is not None:
            changed, route = compare_settings(source, current_settings(generator, verifier), args.jev_gateway)
            if changed:
                raise ValueError("Settings differ from the source run: " + ", ".join(changed))
            settings.update(route)
        else:
            current = current_settings(generator, verifier) if verifier is not None else None
            llm_now = {key: value for key, value in (current or {}).items() if key in LLM_FIELDS}
            if current is None:
                llm_now = {"llm_model": generator.model, "reasoning_effort": generator.reasoning_effort,
                           "prices_usd_per_million": cli._prices(generator)}
            changed = [key for key, value in llm_now.items() if source.get(key) != value]
            if changed:
                raise ValueError("Settings differ from the source run: " + ", ".join(changed))
            if current is not None:
                settings.update({key: current[key] for key in JEV_FIELDS})
        settings.update(
            methods=methods, shared_initial_methods=methods, max_repairs=limit, workers=args.workers,
            expected_records=len(source["ids"]) * source["replicates"] * len(methods), added=added,
            extends={"run_dir": _relative(source_dir), "config_id": source["config_id"], "max_repairs": cap,
                     "records_sha256": _sha256(source_dir / "records.jsonl")})
        reused = {}
        for method, path in reuse.items():
            manifest, cut = reused_rows(path, method, limit, source, shared)
            reused[method] = cut
            settings.setdefault("reused", {})[method] = {
                "run_dir": _relative(path), "config_id": manifest["config_id"], "max_repairs": manifest["max_repairs"],
                "records_sha256": _sha256(Path(path) / "records.jsonl"), "jev_routes": manifest.get("jev_routes")}
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
        prices = source["prices_usd_per_million"]
        lock, stop = threading.Lock(), threading.Event()

        def append(name, record):
            with lock:
                with (directory / name).open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        def keep(record, **marks):
            record = deepcopy(record)
            record.update(max_repairs=limit, config_id=config_id, **marks)
            record["accounting"] = token_cost(record, prices)
            append("records.jsonl", record)

        todo = []
        for row in rows:
            key = (row["id"], row["replicate"], row["method"])
            if key in done:
                continue
            if row["method"] in reused:
                keep(reused[row["method"]][key[:2]], reused=True)
            elif row["method"] == "direct" or (row["method"] in STOP_AT_PASS and row["accepted"]):
                keep(row, extended=False)  # a larger limit stops at the same attempt
            else:
                todo.append(("continue", row))
        for method in added:
            for (example_id, replicate), first in shared.items():
                if (example_id, replicate, method) not in done:
                    todo.append(("add", {"id": example_id, "replicate": replicate, "method": method, "first": first}))
        total = len(source["ids"]) * source["replicates"] * len(methods)
        count = total - len(todo)
        print(f"{count} of {total} rows need no new request; running {len(todo)} rows up to r={limit}.", flush=True)

        def task(item):
            nonlocal count
            kind, row = item
            if stop.is_set():
                return
            try:
                example = examples[row["id"]]
                if kind == "continue":
                    record = continue_row(example, row, generator, verifier, limit)
                else:
                    record = run_example(example, generator, verifier if row["method"].startswith("jev_") else None,
                                         row["method"], limit, "execute_last", initial_proposal=deepcopy(row["first"]))
                # Scoring is strictly after all proposal and verification work.
                for step in record["trace"]:
                    step["score"] = score_example(example, step["proposal"]["calls"], checker_path=args.checker)
                record.update(score=record["trace"][-1]["score"], replicate=row["replicate"], synthetic=False,
                              config_id=config_id, extended=kind == "continue", added=kind == "add")
                record["accounting"] = token_cost(record, prices)
                append("records.jsonl", record)
                with lock:
                    count += 1
                    print(f"[{count}/{total}] {row['id']} {row['method']} r={record['repairs']}", flush=True)
            except Exception as exc:
                event = {"id": row["id"], "replicate": row["replicate"], "method": row["method"],
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
        if verifier is not None:
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
