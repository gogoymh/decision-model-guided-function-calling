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
          "jev_function": "JBFC-F", "jev_arguments": "JBFC-A", "jev_joint": "JBFC"}

PAPER_SETTINGS = {
    "replicates": 3, "max_repairs": 3, "threshold": .8,
    "exhaustion_policy": "execute_last", "split_seed": 1729,
    "llm_model": "gpt-5.4-mini-2026-03-17", "jev_model": "jev-1.13.0",
    "reasoning_effort": "none", "temperature": None, "max_completion_tokens": 4096,
    "llm_provider": "openai", "llm_api_style": "responses", "llm_gateway": "apim",
    "jev_gateway": "apim",
}


def paper_protocol_matches(manifest):
    """Whether a run uses the paper's declared model/interface protocol.

    Dataset identity and completeness are additionally checked by update_paper.
    """
    prices = manifest.get("prices_usd_per_million", {})
    return (manifest.get("split") == "test" and manifest.get("limit") is None
            and set(manifest.get("methods", [])) == set(METHODS)
            and all(manifest.get(k) == v for k, v in PAPER_SETTINGS.items())
            and all(prices.get(k) == v for k, v in {
                "llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5, "jev_input": .042,
            }.items()))


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
    complete = True
    for step in record["trace"]:
        usage = step["proposal"].get("usage") or {}
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            complete = False
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
            complete = False
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
    verifications = [t for r in rows for t in r["trace"] if t.get("verification", {}).get("function_probability") is not None]
    brier_function = []
    brier_arguments = []
    for step in verifications:
        score = step.get("score", {})
        check = step["verification"]
        if score.get("function_correct") is not None:
            brier_function.append((check["function_probability"] - int(score["function_correct"]))**2)
        if score.get("arguments_correct") is not None:
            brier_arguments.append((check["arguments_probability"] - int(score["arguments_correct"]))**2)
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
        "brier_function": avg(brier_function), "brier_arguments": avg(brier_arguments),
        "brier_function_n": len(brier_function), "brier_arguments_n": len(brier_arguments),
        "verified_candidates_n": len(verifications),
        "initial_wrong_final_right": avg([not r["trace"][0]["score"]["correct"] and r["score"]["correct"] for r in exact]),
        "initial_right_final_wrong": avg([r["trace"][0]["score"]["correct"] and not r["score"]["correct"] for r in exact]),
    }


def paired_delta(rows, method, seed=1729, samples=2000):
    # Cluster by example ID so repeated runs are not treated as independent examples.
    groups = defaultdict(dict)
    for row in rows:
        if row["method"] in {"direct", method} and row["category"] != "live_relevance":
            key = (row["id"], row["replicate"])
            groups[key][row["method"]] = int(row["score"]["correct"])
    deltas = defaultdict(list)
    for (example_id, _), pair in groups.items():
        if "direct" in pair and method in pair:
            deltas[example_id].append(pair[method] - pair["direct"])
    values = [mean(v) for v in deltas.values()]
    if not values:
        return {"n_paired_ids": 0, "delta": None, "ci95": None}
    rng = random.Random(seed)
    boot = [mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return {"n_paired_ids": len(values), "delta": mean(values),
            "ci95": [percentile(boot, .025), percentile(boot, .975)]}


def build_summary(run_dir: str | Path) -> dict:
    directory = Path(run_dir)
    rows = read_jsonl(directory / "records.jsonl")
    manifest = json.loads((directory / "manifest.json").read_text())
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
                 "accepted": r["trace"][0]["gate_pass"]} for r in rows if r["method"] == m]).items()
                if key in {"n", "exact_n", "accuracy", "gate_diagnostic_n", "coverage", "accepted_precision",
                           "selective_risk", "false_accept_rate", "false_reject_rate", "brier_function",
                           "brier_arguments", "brier_function_n", "brier_arguments_n", "verified_candidates_n"}}
            for m in manifest["methods"] if m.startswith("jev_")
        },
        "paired_vs_direct": {m: paired_delta(rows, m) for m in methods if m != "direct"},
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if methods:
        with (directory / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method"] + list(next(iter(methods.values())).keys()))
            writer.writeheader()
            for method, metrics in methods.items():
                writer.writerow({"method": method, **metrics})
    return summary


def derive_policy_run(source_dir, output_dir, max_repairs=3, exhaustion_policy="execute_last"):
    """Truncate actual joint trajectories; changing only a stopping cap needs no new API call.

    This is valid because no benchmark tool executes before termination and the
    cap is never sent to either model. It must not be used to replay new prompts,
    thresholds, models, or repair feedback.
    """
    if not 0 <= max_repairs <= 3 or exhaustion_policy not in {"execute_last", "abstain"}:
        raise ValueError("Invalid bounded policy")
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    source = [r for r in read_jsonl(source_dir / "records.jsonl") if r["method"] == "jev_joint"]
    if manifest.get("synthetic") or len(source) != len(manifest["ids"]) * manifest["replicates"]:
        raise ValueError("A complete empirical joint run is required for cap replay")
    if max_repairs > manifest["max_repairs"]:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in derived))
    return build_summary(output_dir)


def latex_table(summary=None):
    def fmt(value, scale=1, precision=2):
        return "--" if value is None else f"{value * scale:.{precision}f}"
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.2pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule", r"Method & Acc. (\%) & Repairs & Sec. & USD/1k \\", r"\midrule"]
    for method in METHODS:
        values = (summary or {}).get("methods", {}).get(method, {})
        label = LABELS[method]
        if label in {"JBFC", "JBFC-F", "JBFC-A"}:
            label = r"\textbf{" + label + "}"
        lines.append(label + " & " + " & ".join([
            fmt(values.get("accuracy"), 100, 1), fmt(values.get("mean_repairs")),
            fmt(values.get("mean_latency_s")), fmt(values.get("mean_cost_usd"), 1000, 3)]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\caption{Primary comparison on the held-out BFCL subset. Accuracy excludes the call-presence-only live relevance category. -- denotes an unavailable or unrun result; no empirical values are imputed.}",
                  r"\label{tab:main}", r"\end{table}"])
    return "\n".join(lines)


def latex_category_table(summary=None):
    categories = [("simple_python", "Simple Python", 400, 360), ("multiple", "Multiple", 200, 180),
                  ("irrelevance", "Irrelevance", 240, 216), ("live_simple", "Live simple", 258, 233),
                  ("live_multiple", "Live multiple", 1053, 948), ("live_irrelevance", "Live irrelevance", 884, 796),
                  ("live_relevance", r"Live relevance$^\dagger$", 16, 15)]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.1pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule", r"Category & All & Test & Direct & \textbf{JBFC}\\", r"\midrule"]
    for category, label, full, test in categories:
        cells = []
        for method in ("direct", "jev_joint"):
            values = (summary or {}).get("by_category", {}).get(f"{category}/{method}", {})
            accuracy = values.get("relevance_call_rate" if category == "live_relevance" else "accuracy")
            cells.append("--" if accuracy is None else f"{accuracy * 100:.1f}")
        lines.append(f"{label} & {full} & {test} & " + " & ".join(cells) + r"\\")
    total = [(summary or {}).get("methods", {}).get(m, {}).get("accuracy") for m in ("direct", "jev_joint")]
    lines += [r"\midrule", "Total & 3051 & 2748 & " + " & ".join("--" if x is None else f"{100*x:.1f}" for x in total) + r"\\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Prepared BFCL data and test accuracy (\%). Counts are data statistics. $^\dagger$Live relevance measures call presence only; total accuracy excludes these 15 cases. -- denotes an unrun experiment.}",
              r"\label{tab:categories}", r"\end{table}"]
    return "\n".join(lines)


def latex_budget_table(budgets=None):
    rows = [("Final correctness (\\%)", "accuracy", 100), ("Acceptance coverage (\\%)", "coverage", 100),
            ("Accepted precision (\\%)", "accepted_precision", 100), ("False acceptance (\\%)", "false_accept_rate", 100),
            ("False rejection (\\%)", "false_reject_rate", 100), ("Correction (\\%)", "initial_wrong_final_right", 100),
            ("Regression (\\%)", "initial_right_final_wrong", 100), ("Cap exhaustion (\\%)", "exhaustion_rate", 100),
            ("Mean proposals", "mean_llm_proposals", 1), ("p95 latency (s)", "p95_latency_s", 1)]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule", r"\textbf{JBFC} diagnostic & $r{=}0$ & $1$ & $2$ & $3$\\", r"\midrule"]
    for label, metric, scale in rows:
        values = [(budgets or {}).get(budget, {}).get("methods", {}).get("jev_joint", {}).get(metric) for budget in range(4)]
        lines.append(label + " & " + " & ".join("--" if x is None else f"{x*scale:.1f}" for x in values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Budget and verifier diagnostics; -- denotes an unrun experiment. False acceptance conditions on incorrect candidates. Correction and regression are fractions of all scored requests.}",
              r"\label{tab:diagnostics}", r"\end{table}"]
    return "\n".join(lines)


def update_paper(paper_path: str | Path, summary: dict | None = None):
    if summary is not None:
        if summary["status"] != "complete" or summary["manifest"]["split"] != "test":
            raise ValueError("Only complete test runs may populate the manuscript; keep dev/smoke outputs separate.")
        if summary["manifest"].get("limit"):
            raise ValueError("A limited smoke run cannot populate the manuscript's main table.")
        from .benchmarks import load_examples, stratified_split
        canonical_path = Path(__file__).resolve().parents[2] / "data/bfcl/examples.jsonl"
        canonical = load_examples(canonical_path)
        _, test = stratified_split(canonical, seed=1729)
        manifest = summary["manifest"]
        if set(manifest["ids"]) != {e["id"] for e in test} or len(test) != 2748:
            raise ValueError("The manuscript requires all 2,748 canonical test IDs; a custom subset cannot populate it.")
        if manifest["dataset_sha256"] != hashlib.sha256(canonical_path.read_bytes()).hexdigest():
            raise ValueError("Manuscript data digest differs from the canonical prepared BFCL dataset.")
        if not paper_protocol_matches(manifest):
            raise ValueError("Run differs from the manuscript's declared protocol. Update its protocol before using custom-model/config results.")
    path = Path(paper_path)
    source = path.read_text()
    start, end = "% BEGIN AUTO RESULTS", "% END AUTO RESULTS"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("Manuscript must have exactly one pair of automatic-results markers.")
    before, remainder = source.split(start)
    _, after = remainder.split(end)
    source = before + start + "\n" + latex_table(summary) + "\n" + end + after
    for marker, content in [("CATEGORIES", latex_category_table(summary))]:
        begin, finish = f"% BEGIN AUTO {marker}", f"% END AUTO {marker}"
        if begin in source and finish in source:
            prefix, rest = source.split(begin)
            _, suffix = rest.split(finish)
            source = prefix + begin + "\n" + content + "\n" + finish + suffix
    budgets = {3: summary} if summary else {}
    if summary:
        main_dir = Path(summary["run_dir"])
        for budget in range(3):
            candidate = main_dir.parent / f"budget_{budget}" / "summary.json"
            if candidate.exists():
                item = json.loads(candidate.read_text())
                if (item["status"] == "complete"
                    and item["manifest"].get("source_records_sha256") == hashlib.sha256((main_dir / "records.jsonl").read_bytes()).hexdigest()
                    and item["manifest"]["max_repairs"] == budget):
                    budgets[budget] = item
    begin, finish = "% BEGIN AUTO DIAGNOSTICS", "% END AUTO DIAGNOSTICS"
    if begin in source and finish in source:
        prefix, rest = source.split(begin)
        _, suffix = rest.split(finish)
        source = prefix + begin + "\n" + latex_budget_table(budgets) + "\n" + finish + suffix
    path.write_text(source)
