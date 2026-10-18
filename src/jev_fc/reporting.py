"""Offline metrics. Synthetic demos cannot be promoted to empirical paper results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean

from .pipeline import METHODS

LABELS = {"direct": "Direct", "schema_only": "Schema repair", "self_refine": "Self-refine",
          "jev_function": "JBFC-F", "jev_arguments": "JBFC-A", "jev_joint": "JBFC",
          "jev_joint_latest": "JBFC-L"}
# Table 2 lists the baselines, then the JBFC variants, with JBFC itself in the last row.
TABLE_METHOD_ORDER = ("direct", "schema_only", "self_refine", "jev_function", "jev_arguments",
                      "jev_joint_latest", "jev_joint")

# The paper's protocol for Tables 1 and 2: every prepared example, one run, a fixed
# repair limit, and the declared models and interface. Nothing is tuned on these data.
PAPER_REPAIRS = 2
PAPER_SETTINGS = {
    "split": "all", "limit": None, "synthetic": False, "replicates": 1, "max_repairs": PAPER_REPAIRS,
    "threshold": .8, "exhaustion_policy": "execute_last",
    "llm_model": "gpt-5-nano", "jev_model": "jev-1.13.0", "reasoning_effort": "none", "temperature": None,
    "max_completion_tokens": 4096, "llm_provider": "openai", "llm_api_style": "azure_chat_completions",
    "llm_gateway": "apim", "jev_gateway": "apim",
}
# USD per million tokens for gpt-5-nano and Jev, as stated in the paper.
PAPER_PRICES = {"llm_input": .05, "llm_cached_input": .005, "llm_output": .4, "jev_input": .042}
TABLE1_METHODS = ("direct", "jev_joint")


def paper_protocol_matches(manifest, methods=METHODS):
    """Whether a run follows the paper's protocol and includes the given methods.

    Table 2 needs all seven methods; Table 1 needs Direct and JBFC. Example IDs
    and the data digest are checked separately against the prepared dataset.
    """
    prices = manifest.get("prices_usd_per_million")
    run_methods = manifest.get("methods")
    return (isinstance(prices, dict) and isinstance(run_methods, (list, tuple))
            and len(run_methods) == len(set(run_methods)) and set(methods) <= set(run_methods) <= set(METHODS)
            and not any(isinstance(manifest.get(key), bool) for key in ("replicates", "max_repairs"))
            and all(key in manifest and manifest[key] == value for key, value in PAPER_SETTINGS.items())
            and manifest["synthetic"] is False
            and all(prices.get(key) == value for key, value in PAPER_PRICES.items()))


def _check_all_examples(manifest):
    """The run must cover every prepared example of the pinned dataset."""
    from .benchmarks import load_examples
    canonical_path = Path(__file__).resolve().parents[2] / "data/bfcl/examples.jsonl"
    canonical = load_examples(canonical_path)
    if len(canonical) != 3051 or sorted(manifest.get("ids") or []) != sorted(e["id"] for e in canonical):
        raise ValueError("The manuscript requires all 3,051 canonical example IDs; a subset cannot populate it.")
    if manifest.get("dataset_sha256") != hashlib.sha256(canonical_path.read_bytes()).hexdigest():
        raise ValueError("Manuscript data digest differs from the canonical prepared BFCL dataset.")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def token_cost(record, prices):
    result = dict(llm_input=0, llm_cached_input=0, llm_cache_write_5m=0,
                  llm_cache_write_1h=0, llm_cache_write_unknown=0,
                  llm_output=0, jev_input=0, estimated_usd=0.)
    complete = tokens_complete = True
    for step in record["trace"]:
        usage = step["proposal"].get("usage") or {}
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            complete = tokens_complete = False
        result["llm_input"] += input_tokens or 0
        result["llm_output"] += output_tokens or 0
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        result["llm_cached_input"] += details.get("cached_tokens") or 0
        # Anthropic cache writes are additional input, billed by cache lifetime.
        # The adapter includes them in normalized prompt_tokens exactly once.
        created = usage.get("cache_creation_input_tokens") or 0
        lifetime = usage.get("cache_creation") or {}
        five_minute = lifetime.get("ephemeral_5m_input_tokens") or 0
        one_hour = lifetime.get("ephemeral_1h_input_tokens") or 0
        result["llm_cache_write_5m"] += five_minute
        result["llm_cache_write_1h"] += one_hour
        result["llm_cache_write_unknown"] += max(0, created - five_minute - one_hour)
        if five_minute + one_hour > created:
            complete = False
        verification = step["verification"]
        jev_input = (verification.get("usage") or {}).get("input_tokens")
        if verification.get("api_calls", 0) > 0 and jev_input is None:
            complete = tokens_complete = False
        result["jev_input"] += jev_input or 0
    uncached = (result["llm_input"] - result["llm_cached_input"] - result["llm_cache_write_5m"]
                - result["llm_cache_write_1h"] - result["llm_cache_write_unknown"])
    if uncached < 0 or result["llm_cache_write_unknown"]:
        complete = False
    for name, count in (("llm_input", max(0, uncached)), ("llm_cached_input", result["llm_cached_input"]),
                        ("llm_cache_write_5m", result["llm_cache_write_5m"]),
                        ("llm_cache_write_1h", result["llm_cache_write_1h"]),
                        ("llm_output", result["llm_output"]), ("jev_input", result["jev_input"])):
        rate = prices.get(name)
        if count and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                      or not math.isfinite(rate) or rate < 0):
            complete = False
        elif count:
            result["estimated_usd"] += count * rate / 1_000_000
    result["complete"] = complete
    result["tokens_complete"] = tokens_complete
    if not complete:
        result["estimated_usd"] = None
    return result


def summarize_group(rows):
    scored = [r for r in rows if r.get("score", {}).get("correct") is not None]
    # live_relevance has call-presence labels, not exact function/argument labels.
    exact = [r for r in scored if r["category"] != "live_relevance"]
    accepted = [r for r in exact if r.get("accepted") is True]
    rejected = [r for r in exact if r.get("accepted") is False]
    relevant = [r for r in scored if r["category"] == "live_relevance"]
    ast_rows = [r for r in scored if r["category"] in {"simple_python", "multiple", "live_simple", "live_multiple"}]
    no_call = [r for r in scored if r["category"] in {"irrelevance", "live_irrelevance"}]
    gated = [r for r in exact if r.get("accepted") is not None]
    wrong = [r for r in gated if not r["score"]["correct"]]
    right = [r for r in gated if r["score"]["correct"]]
    verifications = [t for r in rows for t in r["trace"]
                     if any(t.get("verification", {}).get(k) is not None
                            for k in ("function_probability", "arguments_probability"))]
    brier_function = []
    brier_arguments = []
    for step in verifications:
        score = step.get("score", {})
        check = step["verification"]
        if score.get("function_correct") is not None and check.get("function_probability") is not None:
            brier_function.append((check["function_probability"] - int(score["function_correct"]))**2)
        if score.get("arguments_correct") is not None and check.get("arguments_probability") is not None:
            brier_arguments.append((check["arguments_probability"] - int(score["arguments_correct"]))**2)
    # Token use is reported only when every record's usage is known.
    accounting = [r.get("accounting") or {} for r in rows]
    token_rows = [a for a in accounting if a.get("tokens_complete", a.get("complete", False))]
    tokens_known = bool(rows) and len(token_rows) == len(rows)

    def token_mean(name):
        return avg([a.get(name) for a in token_rows]) if tokens_known else None

    def names(calls):
        if not isinstance(calls, list):
            return None
        return sorted(str(c.get("name")) if isinstance(c, dict) else "<invalid>" for c in calls)

    # Initial-to-final transitions on the same records (function changes, fixes and new errors).
    function_pairs = [(r["trace"][0].get("score", {}).get("function_correct"), r["score"].get("function_correct"))
                      for r in exact]
    function_pairs = [(a, b) for a, b in function_pairs if a is not None and b is not None]
    argument_pairs = [(r["trace"][0].get("score", {}).get("arguments_correct"), r["score"].get("arguments_correct"))
                      for r in exact if r["trace"][0].get("score", {}).get("function_correct") is True
                      and r["score"].get("function_correct") is True]
    argument_pairs = [(a, b) for a, b in argument_pairs if a is not None and b is not None]
    return {
        "n": len(rows), "exact_n": len(exact),
        "accuracy": avg([int(r["score"]["correct"]) for r in exact]),
        "ast_n": len(ast_rows), "ast_accuracy": avg([r["score"]["correct"] for r in ast_rows]),
        "no_call_n": len(no_call), "no_call_accuracy": avg([r["score"]["correct"] for r in no_call]),
        "function_accuracy": avg([r["score"].get("function_correct") for r in exact]),
        "function_scored_n": sum(r["score"].get("function_correct") is not None for r in exact),
        "arguments_accuracy_conditional": avg([r["score"].get("arguments_correct") for r in exact]),
        "arguments_scored_n": sum(r["score"].get("arguments_correct") is not None for r in exact),
        "relevance_n": len(relevant), "relevance_call_rate": avg([r["score"]["correct"] for r in relevant]),
        "coverage": len(accepted) / len(exact) if exact and (accepted or rejected) else None,
        "gate_diagnostic_n": len(gated),
        "accepted_precision": avg([r["score"]["correct"] for r in accepted]),
        "selective_risk": avg([not r["score"]["correct"] for r in accepted]),
        "false_accept_rate": avg([r["accepted"] for r in wrong]),
        "false_reject_rate": avg([not r["accepted"] for r in right]),
        "accepted_n": len(accepted), "gate_correct_n": len(right), "gate_incorrect_n": len(wrong),
        "mean_repairs": avg([r["repairs"] for r in rows]),
        "exhaustion_rate": avg([r["exhausted"] for r in rows]),
        "dispatch_rate": avg([r["dispatched"] for r in rows]),
        "dispatch_correct_per_example": avg([r["dispatched"] and r["score"]["correct"] for r in exact]),
        "mean_latency_s": avg([r["logical_latency_s"] for r in rows]),
        "p95_latency_s": percentile([r["logical_latency_s"] for r in rows], .95),
        "mean_cost_usd": avg([r.get("accounting", {}).get("estimated_usd") for r in rows])
        if rows and all(r.get("accounting", {}).get("complete", False) for r in rows) else None,
        "cost_observed_n": sum(r.get("accounting", {}).get("complete", False) for r in rows),
        "mean_llm_proposals": avg([len(r["trace"]) for r in rows]),
        "mean_jev_requests": avg([sum(t["verification"].get("api_calls", 0) for t in r["trace"]) for r in rows]),
        # Generator responses cut off, filtered or refused; each was an unreadable proposal.
        "incomplete_proposal_n": sum(bool(t["proposal"].get("incomplete")) for r in rows for t in r["trace"]),
        "incomplete_final_n": sum(bool(r["trace"][-1]["proposal"].get("incomplete")) for r in rows),
        "brier_function": avg(brier_function), "brier_arguments": avg(brier_arguments),
        "brier_function_n": len(brier_function), "brier_arguments_n": len(brier_arguments),
        "verified_candidates_n": len(verifications),
        "initial_wrong_final_right": avg([not r["trace"][0]["score"]["correct"] and r["score"]["correct"] for r in exact]),
        "initial_right_final_wrong": avg([r["trace"][0]["score"]["correct"] and not r["score"]["correct"] for r in exact]),
        "function_change_rate": avg([names(r["trace"][0]["proposal"].get("calls"))
                                     != names(r["trace"][-1]["proposal"].get("calls")) for r in exact]),
        "function_transition_n": len(function_pairs),
        "function_fix_rate": avg([not a and b for a, b in function_pairs]),
        "function_regression_rate": avg([a and not b for a, b in function_pairs]),
        "argument_transition_n": len(argument_pairs),
        "argument_fix_rate": avg([not a and b for a, b in argument_pairs]),
        "argument_regression_rate": avg([a and not b for a, b in argument_pairs]),
        "token_observed_n": len(token_rows),
        "mean_llm_input_tokens": token_mean("llm_input"),
        "mean_llm_cached_input_tokens": token_mean("llm_cached_input"),
        "mean_llm_output_tokens": token_mean("llm_output"),
        "mean_jev_input_tokens": token_mean("jev_input"),
    }


def paired_delta(rows, method, seed=1729, samples=2000, *, baseline="direct"):
    """Estimate method minus baseline on matching example/replicate pairs."""
    # Cluster by example ID so repeated runs are not treated as independent examples.
    groups = defaultdict(dict)
    for row in rows:
        if (row["method"] in {baseline, method} and row["category"] != "live_relevance"
            and row.get("score", {}).get("correct") is not None):
            key = (row["id"], row["replicate"])
            groups[key][row["method"]] = int(row["score"]["correct"])
    deltas = defaultdict(list)
    for (example_id, _), pair in groups.items():
        if baseline in pair and method in pair:
            deltas[example_id].append(pair[method] - pair[baseline])
    values = [mean(v) for v in deltas.values()]
    paired_records = sum(len(v) for v in deltas.values())
    if not values:
        return {"n_paired_ids": 0, "n_paired_records": 0, "delta": None, "ci95": None}
    rng = random.Random(seed)
    boot = [mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return {"n_paired_ids": len(values), "n_paired_records": paired_records, "delta": mean(values),
            "ci95": [percentile(boot, .025), percentile(boot, .975)]}


def _history_ablation(rows, manifest):
    """Compare observed cumulative/latest runs, never synthesize one from the other."""
    method, baseline = "jev_joint", "jev_joint_latest"
    if not {method, baseline}.issubset(manifest["methods"]):
        return None
    pairs = defaultdict(dict)
    for row in rows:
        if row["method"] in {method, baseline}:
            pairs[(row["id"], row["replicate"])][row["method"]] = row
    for pair in pairs.values():
        if not {method, baseline}.issubset(pair):
            continue  # Partial reports use only observed pairs.
        main, latest = pair[method], pair[baseline]
        if (main.get("initial_proposal_shared") is not True
            or latest.get("initial_proposal_shared") is not True
            or not main.get("trace") or not latest.get("trace")
            or main["trace"][0].get("proposal") != latest["trace"][0].get("proposal")
            or main["trace"][0].get("score") != latest["trace"][0].get("score")):
            raise ValueError("History ablation requires the same shared initial proposal and score.")
        cap, policy = manifest.get("max_repairs"), manifest.get("exhaustion_policy")
        if (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0
            or policy not in {"execute_last", "abstain"}
            or any(row.get("max_repairs") != cap or row.get("exhaustion_policy") != policy
                   for row in (main, latest))):
            raise ValueError("History ablation requires the same repair limit and dispatch policy.")
    return {
        "method": method, "baseline": baseline, "contrast": "JBFC minus JBFC-L",
        "outcome": "final_call_accuracy", "pairing": "example_id_and_replicate",
        "excluded_category": "live_relevance",
        **paired_delta(rows, method, baseline=baseline),
    }


def build_summary(run_dir: str | Path) -> dict:
    directory = Path(run_dir)
    rows = read_jsonl(directory / "records.jsonl")
    manifest = json.loads((directory / "manifest.json").read_text())
    methods = manifest.get("methods")
    if (not isinstance(methods, (list, tuple)) or not methods
        or any(not isinstance(method, str) for method in methods)
        or len(methods) != len(set(methods)) or not set(methods).issubset(METHODS)):
        raise ValueError("Manifest methods must be unique supported methods.")
    if any(r.get("synthetic") for r in rows) or manifest.get("synthetic"):
        raise ValueError("Synthetic demonstrations are excluded from benchmark and paper reports.")
    expected = {(example_id, replicate, method) for example_id in manifest["ids"]
                for replicate in range(manifest["replicates"]) for method in manifest["methods"]}
    seen = set()
    if len(expected) != manifest["expected_records"]:
        raise ValueError("Manifest expected-record count is inconsistent with its ID/replicate/method grid.")
    for row in rows:
        key = (row["id"], row["replicate"], row["method"])
        if key in seen or key not in expected or row.get("config_id") != manifest.get("config_id"):
            raise ValueError("Duplicate, foreign or mismatched-config result record detected.")
        seen.add(key)
    methods = {method: summarize_group([r for r in rows if r["method"] == method]) for method in manifest["methods"]}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["method"])].append(row)
    summary = {
        "run_dir": str(directory.resolve()),
        "status": "complete" if len(rows) == manifest["expected_records"] else "partial",
        "expected_records": manifest["expected_records"], "completed_records": len(rows),
        "api_error_events": len(read_jsonl(directory / "errors.jsonl")),
        "manifest": manifest, "methods": methods,
        "by_category": {f"{c}/{m}": summarize_group(v) for (c, m), v in groups.items()},
        "by_replicate": {f"{replicate}/{m}": summarize_group([r for r in rows if r["replicate"] == replicate and r["method"] == m])
                         for replicate in range(manifest["replicates"]) for m in manifest["methods"]},
        "initial_candidate_diagnostics": {
            m: {key: value for key, value in summarize_group([
                {**r, "trace": r["trace"][:1], "score": r["trace"][0]["score"],
                 "accepted": r["trace"][0]["gate_pass"]}
                for r in rows if r["method"] == m]).items()
                if key in {"n", "exact_n", "accuracy", "gate_diagnostic_n", "coverage", "accepted_precision",
                           "selective_risk", "false_accept_rate", "false_reject_rate", "brier_function",
                           "brier_arguments", "brier_function_n", "brier_arguments_n", "verified_candidates_n"}}
            for m in manifest["methods"] if m.startswith("jev_")
        },
        "paired_vs_direct": {m: paired_delta(rows, m) for m in methods if m != "direct"},
        "history_ablation": _history_ablation(rows, manifest),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if methods:
        with (directory / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method"] + list(next(iter(methods.values())).keys()))
            writer.writeheader()
            for method, metrics in methods.items():
                writer.writerow({"method": method, **metrics})
    return summary


def derive_policy_run(source_dir, output_dir, max_repairs=None, exhaustion_policy="execute_last"):
    """Truncate actual joint trajectories; changing only a stopping cap needs no new API call.

    This is valid because no benchmark tool executes before termination and the
    cap is never sent to either model. It must not be used to replay new prompts,
    thresholds, models, or repair feedback. In particular, a cumulative JBFC
    trajectory cannot stand in for the latest-feedback-only JBFC-L method.
    """
    if (max_repairs is not None and (isinstance(max_repairs, bool)
                                    or not isinstance(max_repairs, int) or max_repairs < 0)
        or exhaustion_policy not in {"execute_last", "abstain"}):
        raise ValueError("Invalid bounded policy")
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("Replay output must differ from its source directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Replay output must be a directory; existing files are not overwritten")
    manifest = json.loads((source_dir / "manifest.json").read_text())
    rows = read_jsonl(source_dir / "records.jsonl")
    ids, methods, replicates = manifest.get("ids"), manifest.get("methods"), manifest.get("replicates")
    source_cap = manifest.get("max_repairs")
    if (manifest.get("synthetic") is not False
        or not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
        or len(ids) != len(set(ids))
        or not isinstance(methods, list) or any(not isinstance(m, str) for m in methods)
        or len(methods) != len(set(methods))
        or "jev_joint" not in methods or not set(methods).issubset(METHODS)
        or isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1
        or isinstance(source_cap, bool) or not isinstance(source_cap, int) or source_cap < 0
        or not isinstance(manifest.get("config_id"), str) or not manifest["config_id"]):
        raise ValueError("A complete empirical joint run is required for cap replay")
    expected = {(example_id, replicate, method) for example_id in ids
                for replicate in range(replicates) for method in methods}
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
            or not isinstance(row.get("method"), str) or isinstance(row.get("replicate"), bool)
            or not isinstance(row.get("replicate"), int)):
            raise ValueError("Malformed source result record")
        key = (row.get("id"), row.get("replicate"), row.get("method"))
        if (key in seen or key not in expected or row.get("synthetic") is not False
            or row.get("config_id") != manifest["config_id"]):
            raise ValueError("Duplicate, foreign or mismatched-config source record detected")
        seen.add(key)
    if seen != expected or manifest.get("expected_records") != len(expected):
        raise ValueError("A complete empirical source run is required for cap replay")
    source = [r for r in rows if r["method"] == "jev_joint"]
    for row in source:
        trace = row.get("trace")
        if (not isinstance(trace, list) or not trace or len(trace) > source_cap + 1
            or isinstance(row.get("max_repairs"), bool) or row.get("max_repairs") != source_cap
            or isinstance(row.get("repairs"), bool) or row.get("repairs") != len(trace) - 1
            or any(not isinstance(step, dict) or isinstance(step.get("attempt"), bool)
                   or step.get("attempt") != index or not isinstance(step.get("gate_pass"), bool)
                   for index, step in enumerate(trace))
            or any(step["gate_pass"] for step in trace[:-1])
            or (not trace[-1]["gate_pass"] and len(trace) != source_cap + 1)
            or row.get("accepted") != trace[-1]["gate_pass"]):
            raise ValueError("Source joint trajectory is incomplete or violates its recorded stopping rule")
    max_repairs = source_cap if max_repairs is None else max_repairs
    if max_repairs > source_cap:
        raise ValueError("Cannot replay a longer trajectory than was recorded")
    source_hash = hashlib.sha256((source_dir / "records.jsonl").read_bytes()).hexdigest()
    derived = []
    for row in source:
        row = deepcopy(row)
        row["trace"] = row["trace"][:max_repairs + 1]
        terminal = row["trace"][-1]
        accepted = terminal["gate_pass"]
        row.update(calls=terminal["proposal"]["calls"], score=terminal["score"],
                   accepted=accepted, exhausted=not accepted,
                   repairs=len(row["trace"]) - 1, max_repairs=max_repairs,
                   exhaustion_policy=exhaustion_policy,
                   dispatched=accepted or exhaustion_policy == "execute_last",
                   abstained=not accepted and exhaustion_policy == "abstain",
                   derived_from=str(source_dir), wall_time_s=None,
                   logical_latency_s=sum(t["proposal"].get("latency_s", 0) + t["verification"].get("latency_s", 0) for t in row["trace"]))
        row["accounting"] = token_cost(row, manifest["prices_usd_per_million"])
        derived.append(row)
    manifest.update(methods=["jev_joint"], max_repairs=max_repairs, exhaustion_policy=exhaustion_policy,
                    expected_records=len(derived), derived_from=str(source_dir), source_records_sha256=source_hash)
    manifest["config_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for row in derived:
        row["config_id"] = manifest["config_id"]
    record_text = "".join(json.dumps(r) + "\n" for r in derived)
    if output_dir.exists() and any(output_dir.iterdir()):
        # A resumed suite can repeat a replay only when the complete expected
        # manifest and records still match. Recompute from source first; never
        # trust an existing summary or silently replace different trajectories.
        try:
            saved_manifest = json.loads((output_dir / "manifest.json").read_text())
            saved_records = (output_dir / "records.jsonl").read_text()
        except (OSError, json.JSONDecodeError):
            raise ValueError("Existing replay results are incomplete and are not overwritten") from None
        if saved_manifest != manifest or saved_records != record_text:
            raise ValueError("Existing replay results differ from the source or policy and are not overwritten")
        return build_summary(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "records.jsonl").write_text(record_text)
    return build_summary(output_dir)


def _bold_best(column):
    """Bold the highest shown value in a column; values tied for highest are all bold."""
    shown = [float(cell) for cell in column if cell != "--"]
    if len(shown) < 2:
        return column, False
    best = max(shown)
    return [rf"\textbf{{{cell}}}" if cell != "--" and float(cell) == best else cell for cell in column], True


def latex_table(summary=None):
    """Table 2: the seven methods on all prepared examples (one run, fixed repair limit)."""
    def fmt(value, scale=1, precision=2):
        return "--" if value is None else f"{value * scale:.{precision}f}"
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.7pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule", r"Method & Call & Func. & Args. & Sec. & USD/1k \\", r"\midrule"]
    rows = []
    for method in TABLE_METHOD_ORDER:
        values = (summary or {}).get("methods", {}).get(method, {})
        rows.append([fmt(values.get("accuracy"), 100, 1), fmt(values.get("function_accuracy"), 100, 1),
                     fmt(values.get("arguments_accuracy_conditional"), 100, 1), fmt(values.get("mean_latency_s")),
                     fmt(values.get("mean_cost_usd"), 1000, 3)])
    missing = any("--" in row for row in rows)
    bolded = False
    for column in range(3):  # accuracy columns: higher is better
        cells, bold = _bold_best([row[column] for row in rows])
        bolded = bolded or bold
        for row, cell in zip(rows, cells):
            row[column] = cell
    for method, row in zip(TABLE_METHOD_ORDER, rows):
        label = LABELS[method]
        if label.startswith("JBFC"):
            label = r"\textbf{" + label + "}"
        lines.append(label + " & " + " & ".join(row) + r" \\")
    model = (summary or {}).get("manifest", {}).get("llm_model")
    generator = rf" Generator: \texttt{{{_tex_escape(model)}}}." if model else ""
    best = " Bold marks the highest accuracy in each column." if bolded else ""
    unrun = " -- means unavailable or unrun." if missing else ""
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  rf"\caption{{Methods on all BFCL examples, from one run with repair limit $r={PAPER_REPAIRS}$."
                  + generator + best
                  + r" Call and function accuracy (\%) exclude live relevance. Argument accuracy (\%) uses calls with the correct function and argument labels."
                  + unrun + "}",
                  r"\label{tab:main}", r"\end{table}"])
    return "\n".join(lines)


# Table 1 shows Direct and JBFC by category, from the same protocol as Table 2.
TABLE1_REPAIRS = PAPER_REPAIRS
TABLE1_CATEGORIES = (("simple_python", "Simple Python", 400), ("multiple", "Multiple", 200),
                     ("irrelevance", "Irrelevance", 240), ("live_simple", "Live simple", 258),
                     ("live_multiple", "Live multiple", 1053), ("live_irrelevance", "Live irrelevance", 884),
                     ("live_relevance", r"Live relevance$^\dagger$", 16))


def _tex_escape(text):
    special = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(char, char) for char in str(text))


def _replace_block(source, marker, content):
    begin, finish = f"% BEGIN AUTO {marker}", f"% END AUTO {marker}"
    if source.count(begin) != 1 or source.count(finish) != 1:
        raise ValueError(f"Manuscript must have exactly one pair of automatic {marker} markers.")
    prefix, rest = source.split(begin)
    _, suffix = rest.split(finish)
    return prefix + begin + "\n" + content + "\n" + finish + suffix


def _bold_higher(cells):
    """Bold the strictly higher of two shown values; ties and missing values stay plain."""
    if "--" in cells or float(cells[0]) == float(cells[1]):
        return cells, False
    best = 0 if float(cells[0]) > float(cells[1]) else 1
    return [rf"\textbf{{{cell}}}" if index == best else cell for index, cell in enumerate(cells)], True


def latex_category_table(summary=None):
    """Table 1: Direct and JBFC accuracy on all prepared examples (one run, fixed repair limit)."""
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrr}", r"\toprule", r"Category & Examples & Direct & \textbf{JBFC}\\", r"\midrule"]
    missing = bolded = False
    for category, label, count in TABLE1_CATEGORIES:
        cells = []
        for method in ("direct", "jev_joint"):
            values = (summary or {}).get("by_category", {}).get(f"{category}/{method}", {})
            accuracy = values.get("relevance_call_rate" if category == "live_relevance" else "accuracy")
            cells.append("--" if accuracy is None else f"{accuracy * 100:.1f}")
        missing = missing or "--" in cells
        cells, bold = _bold_higher(cells)
        bolded = bolded or bold
        lines.append(f"{label} & {count} & " + " & ".join(cells) + r"\\")
    total = [(summary or {}).get("methods", {}).get(m, {}).get("accuracy") for m in ("direct", "jev_joint")]
    missing = missing or None in total
    total, bold = _bold_higher(["--" if x is None else f"{100*x:.1f}" for x in total])
    bolded = bolded or bold
    model = (summary or {}).get("manifest", {}).get("llm_model")
    generator = rf" Generator: \texttt{{{_tex_escape(model)}}}." if model else ""
    best = " Bold marks the higher accuracy when the two differ." if bolded else ""
    # The note on empty cells appears only while some cell is still empty.
    unrun = " -- means the experiment has not been run." if missing else ""
    lines += [r"\midrule", "Total & 3051 & " + " & ".join(total) + r"\\",
              r"\bottomrule", r"\end{tabular}",
              rf"\caption{{BFCL example counts and accuracy (\%) on all examples, from one run with repair limit $r={TABLE1_REPAIRS}$."
              + generator + best
              + r" $^\dagger$Live relevance checks only whether a call is present; total accuracy excludes these 16 cases."
              + unrun + "}",
              r"\label{tab:categories}", r"\end{table}"]
    return "\n".join(lines)


def table1_protocol_matches(manifest):
    """Whether a run can fill Table 1: Direct and JBFC under the paper's protocol."""
    return paper_protocol_matches(manifest, TABLE1_METHODS)


def update_table1(paper_path: str | Path, summary: dict | None = None):
    """Fill only Table 1 from a complete all-example run; the other tables stay as they are."""
    if summary is not None:
        if summary.get("status") != "complete":
            raise ValueError("Only a complete run can populate Table 1.")
        if not table1_protocol_matches(summary["manifest"]):
            raise ValueError("Table 1 needs Direct and JBFC on all examples (--split all), one run, r=2, "
                             "threshold 0.8, execute_last and the paper's model settings and prices.")
        _check_all_examples(summary["manifest"])
    path = Path(paper_path)
    path.write_text(_replace_block(path.read_text(), "CATEGORIES", latex_category_table(summary)))


def update_paper(paper_path: str | Path, summary: dict | None = None):
    """Fill Tables 1 and 2 from one complete seven-method run on all examples.

    Both tables then come from the same records, so their Direct and JBFC values
    agree. Without a summary, both tables are reset to their unrun form.
    """
    if summary is not None:
        if summary.get("status") != "complete":
            raise ValueError("Only a complete run can populate the manuscript; keep partial or smoke outputs separate.")
        manifest = summary["manifest"]
        if not paper_protocol_matches(manifest):
            raise ValueError("Run differs from the manuscript's declared protocol: all seven methods on all "
                             "examples, one run, r=2, threshold 0.8, execute_last, and the paper's models and prices.")
        _check_all_examples(manifest)
    path = Path(paper_path)
    source = path.read_text()
    source = _replace_block(source, "RESULTS", latex_table(summary))
    if "% BEGIN AUTO CATEGORIES" in source or "% END AUTO CATEGORIES" in source:
        source = _replace_block(source, "CATEGORIES", latex_category_table(summary))
    path.write_text(source)
