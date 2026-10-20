"""The paper's tables and Figure 2 at its final JBFC settings (make final).

Final settings (user decision, 2026-10-18): Jev threshold 0.5 for both checks and, when no
proposal passes within the repair limit r=3, the first proposal is returned (rule "keep_first")
instead of the last one. The rule changes only which recorded proposal is final, never what is
generated or checked, so the final runs are derived from recorded runs without new requests:

* Table 1: three runs per generator. Direct and JBFC (threshold 0.5) come from the threshold
  reruns (make threshold THRESHOLD=0.5), which share each Table 1 run's first proposals.
* Table 2: gpt-5-nano's run 1 with all seven methods. Direct, schema repair and self-refine come
  from the r=3 run; JBFC from the threshold rerun; JBFC-F, JBFC-A and JBFC-L at 0.5 from make t05.
* Figure 2: JBFC at 0.5 continued to r=10 (make t05), cut at each limit.
* Table 3: JBFC at 0.5 returning the last or the first proposal when none passes, on the same
  trajectories (so with the same requests and cost).

The rule applies to every method that stops at its first passing proposal; Direct and
self-refine always return their last proposal. Derived runs go to results/final/ (the runs
returning the last proposal, for Table 3, to results/final/last/).

Usage: python scripts/final_tables.py [--paper paper/main.tex] [--json results/final/final.json]

The ACM short-paper format keeps a four-row main table, the seven-method ablation table,
and a separate accuracy/mean/p95 repair plot. Its author-edited captions and layout are
preserved; the original inline manuscript and the earlier ACM draft keep their full tables.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_tables  # noqa: E402
import table1_repeat  # noqa: E402
import table3  # noqa: E402
from raise_limit import STOP_AT_PASS, load_run  # noqa: E402

from jev_fc import cli  # noqa: E402
from jev_fc.pipeline import METHODS  # noqa: E402
from jev_fc.reporting import (  # noqa: E402
    PAPER_PRICES,
    PAPER_SETTINGS,
    _check_all_examples,
    _replace_block,
    _tex_escape,
    avg,
    build_summary,
    paired_delta,
    read_jsonl,
)

THRESHOLD = 0.5
RULE = "keep_first"
LIMIT = 3
RUNS = 3
MODELS = ("gpt-5-nano", "gpt-5.4-mini")
OUT = "results/final"
JEV_METHODS = ("jev_function", "jev_arguments", "jev_joint", "jev_joint_latest")
SAME_EVERYWHERE = ("source_sha256", "dataset_sha256", "ids", "split", "split_seed", "limit", "replicates",
                   "max_repairs", "exhaustion_policy", "synthetic", "llm_provider", "llm_api_style", "llm_gateway",
                   "max_completion_tokens", "temperature", "jev_model")
SAME_PER_MODEL = ("llm_model", "reasoning_effort", "prices_usd_per_million")
# Provenance of the run a derived run is taken from; the derived run records its parts instead.
NOT_COPIED = ("config_id", "created_at", "extends", "added", "rethreshold", "reused", "workers")
FIGURE_RUN = f"results/table3_gpt-5-nano_r10_t{THRESHOLD:g}"
ACM_SHORT = "% MANUSCRIPT FORMAT: ACM SHORT"


def rerun(model, run):
    """JBFC at the final threshold on run `run` of `model` (make threshold THRESHOLD=0.5)."""
    return f"results/jbfc_t{THRESHOLD:g}_{model}_run{run}"


def table2_parts():
    """gpt-5-nano's run 1: where each of the seven methods comes from."""
    base, model = "results/table2_gpt-5-nano_r3", "gpt-5-nano"
    parts = {method: base for method in ("direct", "schema_only", "self_refine")}
    parts["jev_joint"] = rerun(model, 1)
    for method in ("jev_function", "jev_arguments", "jev_joint_latest"):
        parts[method] = f"results/{method}_t{THRESHOLD:g}_{model}_run1"
    return parts


def keep_first(row):
    """The record with the first proposal as final when nothing passed (stop-at-pass methods).

    The trajectory, acceptance, repairs, time and cost stay as recorded: the loop runs the same way
    and only the returned proposal differs. final_attempt names the returned attempt."""
    row = deepcopy(row)
    if row["method"] in STOP_AT_PASS and not row["accepted"]:
        first = row["trace"][0]
        row.update(calls=deepcopy(first["proposal"]["calls"]), score=deepcopy(first["score"]), final_attempt=0)
    else:
        row["final_attempt"] = len(row["trace"]) - 1
    row["final_rule"] = RULE
    return row


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path, root=None):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(root or ROOT).resolve()))
    except ValueError:
        return str(path)


def _names(calls):
    if not isinstance(calls, list):
        return None
    return sorted(str(call.get("name")) if isinstance(call, dict) else "<invalid>" for call in calls)


def _final_step(row):
    return row["trace"][row.get("final_attempt", len(row["trace"]) - 1)]


def final_proposal_metrics(rows):
    """The two summary metrics that read the final proposal from the trace. build_summary takes the last
    attempt, which is not the final proposal when keep_first returned the first one."""
    exact = [row for row in rows
             if (row.get("score") or {}).get("correct") is not None and row["category"] != "live_relevance"]
    return {"incomplete_final_n": sum(bool(_final_step(row)["proposal"].get("incomplete")) for row in rows),
            "function_change_rate": avg([_names(row["trace"][0]["proposal"].get("calls"))
                                         != _names(_final_step(row)["proposal"].get("calls")) for row in exact])}


def _with_final_metrics(output, summary, rows):
    """Correct those two metrics in every group of a keep_first summary and rewrite summary.json and .csv."""
    groups = {}
    for row in rows:
        for section, key in (("methods", row["method"]), ("by_category", f"{row['category']}/{row['method']}"),
                             ("by_replicate", f"{row['replicate']}/{row['method']}")):
            groups.setdefault((section, key), []).append(row)
    for (section, key), members in groups.items():
        summary[section][key].update(final_proposal_metrics(members))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    methods = summary["methods"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method"] + list(next(iter(methods.values())).keys()))
        writer.writeheader()
        for method, metrics in methods.items():
            writer.writerow({"method": method, **metrics})
    return summary


def _write_run(output, manifest, rows, rule):
    """Write a derived run (manifest with a new config_id, records) and return its summary."""
    manifest = {key: value for key, value in manifest.items() if key != "config_id"}
    manifest["config_id"] = cli._fingerprint(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = [{**(keep_first(row) if rule else deepcopy(row)), "config_id": manifest["config_id"]} for row in rows]
    with (output / "records.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for stale in ("errors.jsonl", "summary.json", "summary.csv"):
        if (output / stale).exists():
            (output / stale).unlink()
    summary = build_summary(output)
    return _with_final_metrics(output, summary, rows) if rule else summary


def derive(output, parts, root=None, rule=True):
    """A run made of recorded rows: parts maps each method to the run it comes from. The settings are
    those of the JBFC run; with rule=True the final rule is applied. Returns its summary."""
    root = Path(root or ROOT)
    output = root / output
    loaded, rows, manifests = {}, [], {}
    for method, run_dir in parts.items():
        if run_dir not in loaded:
            loaded[run_dir] = load_run(root / run_dir)
        manifest, run_rows = loaded[run_dir]
        if method not in manifest["methods"]:
            raise ValueError(f"{run_dir} has no {method} rows.")
        if method in JEV_METHODS and manifest["threshold"] != THRESHOLD:
            raise ValueError(f"{method} must come from a run with threshold {THRESHOLD} "
                             f"({run_dir} has {manifest['threshold']}).")
        manifests[method] = manifest
        rows += [row for row in run_rows if row["method"] == method]
    main = manifests.get("jev_joint") or next(iter(manifests.values()))
    for method, manifest in manifests.items():
        for key in SAME_EVERYWHERE + SAME_PER_MODEL:
            if manifest.get(key) != main.get(key):
                raise ValueError(f"{parts[method]} differs from {parts.get('jev_joint')} in {key}.")
    first = {}
    for row in rows:  # every method of an example must start from the same first proposal
        key = (row["id"], row["replicate"])
        if first.setdefault(key, row["trace"][0]["proposal"]) != row["trace"][0]["proposal"]:
            raise ValueError(f"The parts do not share the first proposal of {row['id']}.")
    methods = [method for method in METHODS if method in parts]
    settings = {key: value for key, value in main.items() if key not in NOT_COPIED}
    settings.update(methods=methods, shared_initial_methods=methods, threshold=THRESHOLD,
                    expected_records=len(main["ids"]) * main["replicates"] * len(methods),
                    derived={"rule": RULE if rule else "execute_last", "parts": [
                        {"run_dir": _relative(root / run_dir, root),
                         "methods": [m for m in methods if parts[m] == run_dir],
                         "config_id": loaded[run_dir][0]["config_id"],
                         "records_sha256": _sha256(root / run_dir / "records.jsonl")}
                        for run_dir in dict.fromkeys(parts[m] for m in methods)]})
    if rule:
        settings["final_rule"] = RULE
    rows.sort(key=lambda row: (row["id"], row["replicate"], methods.index(row["method"])))
    return _write_run(output, settings, rows, rule)


def check_final(summary, model, methods, rule=True):
    """A derived run follows the paper's final protocol (with rule=False: the same, returning the last
    proposal)."""
    manifest = summary["manifest"]
    if summary["status"] != "complete" or manifest.get("methods") != methods:
        raise ValueError(f"{summary['run_dir']} is not a complete run of {methods}.")
    expected = {**PAPER_SETTINGS, "max_repairs": LIMIT, "threshold": THRESHOLD, "llm_model": model,
                "final_rule": RULE if rule else None}
    for key, value in expected.items():
        if key != "jev_gateway" and manifest.get(key) != value:
            raise ValueError(f"{summary['run_dir']} differs from the final protocol in {key}.")
    prices = manifest.get("prices_usd_per_million") or {}
    if model == PAPER_SETTINGS["llm_model"] and any(prices.get(k) != v for k, v in PAPER_PRICES.items()):
        raise ValueError("gpt-5-nano runs must use the paper's prices.")
    if any(prices.get(k) is None for k in ("llm_input", "llm_cached_input", "llm_output")):
        raise ValueError(f"{summary['run_dir']} has no prices.")
    paper_tables.check_route(manifest)
    _check_all_examples(manifest)


def build(root=None):
    """Derive every final run and its twin that returns the last proposal; return their summaries."""
    root = Path(root or ROOT)
    out = {"table1": {}, "last": {}}
    for model in MODELS:
        summaries, last = [], []
        for run in range(1, RUNS + 1):
            if model == "gpt-5-nano" and run == 1:
                parts = table2_parts()
            else:
                parts = {"direct": rerun(model, run), "jev_joint": rerun(model, run)}
            summary = derive(f"{OUT}/{model}_run{run}", parts, root)
            check_final(summary, model, [m for m in METHODS if m in parts])
            summaries.append(summary)
            twin = derive(f"{OUT}/last/{model}_run{run}", {"direct": rerun(model, run),
                                                            "jev_joint": rerun(model, run)}, root, rule=False)
            check_final(twin, model, ["direct", "jev_joint"], rule=False)
            last.append(twin)
        out["table1"][model], out["last"][model] = summaries, last
    reference = out["table1"]["gpt-5-nano"][0]["manifest"]
    for runs in (*out["table1"].values(), *out["last"].values()):  # one code, data and protocol
        for summary in runs:
            for key in ("source_sha256", "dataset_sha256", "ids", "max_repairs", "threshold", "jev_model"):
                if summary["manifest"].get(key) != reference.get(key):
                    raise ValueError(f"Final runs differ in {key}.")
    return out


def check_figure_run(manifest):
    """Figure 2's run: JBFC with gpt-5-nano at the final threshold, continued to the largest limit."""
    expected = {**PAPER_SETTINGS, "max_repairs": max(table3.LIMITS), "threshold": THRESHOLD, "methods": ["jev_joint"]}
    for key, value in expected.items():
        if key != "jev_gateway" and manifest.get(key) != value:
            raise ValueError(f"{FIGURE_RUN} differs from the final protocol in {key}.")
    if any((manifest.get("prices_usd_per_million") or {}).get(k) != v for k, v in PAPER_PRICES.items()):
        raise ValueError(f"{FIGURE_RUN} must use the paper's prices.")
    paper_tables.check_route(manifest)
    _check_all_examples(manifest)


def figure_summaries(root=None):
    """Figure 2: the r=10 run cut at each limit, with the final rule."""
    root = Path(root or ROOT)
    run_dir = root / FIGURE_RUN
    source = root / rerun("gpt-5-nano", 1)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    extends = manifest.get("extends") or {}
    if (root / extends.get("run_dir", "")).resolve() != source.resolve():
        raise ValueError(f"{FIGURE_RUN} must continue {rerun('gpt-5-nano', 1)}.")
    if _sha256(source / "records.jsonl") != extends.get("records_sha256"):
        raise ValueError("The threshold rerun changed after Figure 2's run continued it.")
    check_figure_run(manifest)
    cuts = table3.derive_limits(run_dir)
    table3.check_matches_source(run_dir, source)
    summaries = {}
    for limit, summary in cuts.items():
        cut_dir = Path(summary["run_dir"])
        cut = json.loads((cut_dir / "manifest.json").read_text())
        cut.update(final_rule=RULE, derived={"rule": RULE, "parts": [
            {"run_dir": _relative(cut_dir, root), "methods": ["jev_joint"], "config_id": cut["config_id"],
             "records_sha256": _sha256(cut_dir / "records.jsonl")}]})
        summaries[limit] = _write_run(root / OUT / "figure2" / f"r{limit}", cut,
                                      read_jsonl(cut_dir / "records.jsonl"), rule=True)
    return summaries


def _rows(summary):
    return read_jsonl(Path(summary["run_dir"]) / "records.jsonl")


def known_cost(summary, method):
    """Mean cost (USD per request) over the requests whose usage was reported, and how many were not.
    A generation blocked by the provider's filter reports no usage."""
    rows = [row for row in _rows(summary) if row["method"] == method]
    known = [row["accounting"]["estimated_usd"] for row in rows if (row.get("accounting") or {}).get("complete")]
    return {"mean": statistics.fmean(known) if known else None, "unknown": len(rows) - len(known)}


def table1_costs(final):
    """Table 1's USD/1k: per column, the mean over runs of each run's known-usage mean cost."""
    columns = []
    for model in MODELS:
        for method in ("direct", "jev_joint"):
            per_run = [known_cost(summary, method) for summary in final[model]]
            means = [item["mean"] for item in per_run]
            columns.append({"column": f"{model}/{method}", "runs": means,
                            "mean": None if None in means else statistics.fmean(means),
                            "unknown": sum(item["unknown"] for item in per_run)})
    return columns


def history_labels(summary):
    """JBFC's history as labeled examples: Jev seconds per check, generator seconds per proposal and Jev's
    USD per 1,000 checks, over every JBFC attempt of a run."""
    steps = [step for row in _rows(summary) if row["method"] == "jev_joint" for step in row["trace"]]
    checks = [step["verification"] for step in steps if step["verification"].get("api_calls", 0) > 0]
    tokens = [(check.get("usage") or {}).get("input_tokens") for check in checks]
    price = summary["manifest"]["prices_usd_per_million"]["jev_input"]
    return {"proposals": len(steps), "checks": len(checks),
            "jev_s_per_check": statistics.fmean(check["latency_s"] for check in checks),
            "generator_s_per_proposal": statistics.fmean(step["proposal"].get("latency_s", 0) for step in steps),
            "jev_usd_per_1k_checks": None if None in tokens else statistics.fmean(tokens) * price / 1000}


def _no_call(row):
    return isinstance(row["calls"], list) and not row["calls"]


def limit_changes(figure, low=LIMIT):
    """Examples whose final call changes correctness between r=low and the largest limit of Figure 2."""
    high = max(figure)
    before = {row["id"]: row for row in _rows(figure[low])}
    after = {row["id"]: row for row in _rows(figure[high])}
    scored = [key for key, row in before.items() if row["category"] != "live_relevance"]
    right = [key for key in scored if not before[key]["score"]["correct"] and after[key]["score"]["correct"]]
    wrong = [key for key in scored if before[key]["score"]["correct"] and not after[key]["score"]["correct"]]
    low_methods, high_methods = figure[low]["methods"]["jev_joint"], figure[high]["methods"]["jev_joint"]
    return {"from": low, "to": high, "turned_right": len(right), "turned_wrong": len(wrong),
            "turned_right_no_call": sum(after[key]["category"] in ("irrelevance", "live_irrelevance")
                                        and _no_call(after[key]) for key in right),
            "turned_right_passed": sum(after[key]["accepted"] is True for key in right),
            "time_ratio": high_methods["mean_latency_s"] / low_methods["mean_latency_s"],
            "cost_ratio": None if None in (high_methods["mean_cost_usd"], low_methods["mean_cost_usd"])
            else high_methods["mean_cost_usd"] / low_methods["mean_cost_usd"]}


def _tagged(rows_by_run, rename):
    return [{**row, "method": rename[row["method"]], "replicate": index}
            for index, rows in enumerate(rows_by_run) for row in rows if row["method"] in rename]


def _with_initial(rows):
    return [{**row, "initial_correct": row["trace"][0]["score"]["correct"]} for row in rows]


def rule_report(final, last):
    """Table 3's numbers: Direct, JBFC returning the last proposal, JBFC returning the first."""
    report = {}
    for model in MODELS:
        first_rows = [[r for r in _rows(s) if r["method"] in ("direct", "jev_joint")] for s in final[model]]
        last_rows = [[r for r in _rows(s) if r["method"] in ("direct", "jev_joint")] for s in last[model]]
        combined = []
        for first, last_ in zip(first_rows, last_rows):
            if ({(r["id"], r["method"]): r["trace"] for r in first}
                    != {(r["id"], r["method"]): r["trace"] for r in last_}):
                raise ValueError("The rules must be compared on the same trajectories.")
            combined.append([*({**r, "method": "first"} for r in first if r["method"] == "jev_joint"),
                             *({**r, "method": "last"} for r in last_ if r["method"] == "jev_joint"),
                             *(r for r in first if r["method"] == "direct")])
        report[model] = {
            "direct": table1_repeat.spread([s["methods"]["direct"]["accuracy"] for s in final[model]]),
            "last": table1_repeat.spread([s["methods"]["jev_joint"]["accuracy"] for s in last[model]]),
            "first": table1_repeat.spread([s["methods"]["jev_joint"]["accuracy"] for s in final[model]]),
            "first - last": paired_delta(_tagged(combined, {"first": "method", "last": "baseline"}), "method",
                                         baseline="baseline"),
            "first - direct": paired_delta(_tagged(combined, {"first": "method", "direct": "baseline"}), "method",
                                           baseline="baseline"),
            "last - direct": paired_delta(_tagged(combined, {"last": "method", "direct": "baseline"}), "method",
                                          baseline="baseline"),
            "transitions_first": [table1_repeat.transitions(_with_initial(rows)) for rows in first_rows],
            "transitions_last": [table1_repeat.transitions(_with_initial(rows)) for rows in last_rows],
            "exhausted": [s["methods"]["jev_joint"]["exhaustion_rate"] for s in final[model]],
        }
    return report


def _signed(value, digits=2):
    """Points with a sign, as LaTeX (minus as $-$); zero without one. Two decimals, so a bound near zero keeps
    its sign."""
    text = f"{100 * value:+.{digits}f}"
    return text[1:] if float(text) == 0 else text.replace("-", "$-$")


def _plain(value, digits=2):
    return _signed(value, digits).lstrip("+")


def latex_rules(report):
    """Table 3: the proposal JBFC returns when none passes."""
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Generator & Direct & Last & First & First $-$ last\\", r"\midrule"]
    for model, item in report.items():
        cells, _ = paper_tables._bold_top([paper_tables._cell(item[key]["mean"]) for key in ("direct", "last", "first")])
        diff = item["first - last"]
        low, high = diff["ci95"]
        lines.append(rf"\texttt{{{_tex_escape(model)}}} & " + " & ".join(cells)
                     + rf" & {_signed(diff['delta'])} [{_plain(low)}, {_plain(high)}]\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Call accuracy (\%), mean of three runs, when \textbf{JBFC} returns the last or the first "
              r"proposal if none passes. First $-$ last: paired difference [95\% interval].}",
              r"\label{tab:rules}", r"\end{table}"]
    return "\n".join(lines)


def compact_main_rows(final, costs, method_name="DMFC"):
    """Four main-result rows: per-run mean accuracy and sample s.d., mean API seconds, known-usage cost."""
    cost_by_column = {column["column"]: column["mean"] for column in costs}
    lines = []
    for model in MODELS:
        spreads = [table1_repeat.spread([summary["methods"][method]["accuracy"] for summary in final[model]])
                   for method in ("direct", "jev_joint")]
        cells, _ = paper_tables._bold_top([paper_tables._cell(item["mean"]) for item in spreads])
        for index, method in enumerate(("direct", "jev_joint")):
            spread = spreads[index]
            accuracy = cells[index]
            if spread["sd"] is not None:
                accuracy += rf"$\pm${100 * spread['sd']:.1f}"
            latency = table1_repeat.spread([summary["methods"][method].get("mean_latency_s")
                                           for summary in final[model]])["mean"]
            label = rf"\texttt{{{_tex_escape(model)}}}" if index == 0 else ""
            method_label = "Direct" if method == "direct" else rf"\textbf{{{method_name}}}"
            lines.append(f"{label} & {method_label} & {accuracy} & "
                         + paper_tables._cell(latency, 1, 2) + " & "
                         + paper_tables._cell(cost_by_column[f"{model}/{method}"], 1000, 3) + r"\\")
    return "\n".join(lines)


def _table_rows(source):
    match = re.search(r"\\midrule\s*\n(.*?)\s*\\bottomrule", source, re.DOTALL)
    if not match:
        raise ValueError("A measured table needs midrule and bottomrule row delimiters.")
    return match.group(1)


def _replace_table_rows(source, name, rows):
    """Replace only a marked table's measured rows, retaining its edited caption, header, and layout."""
    block = re.search(rf"(% BEGIN AUTO {name}\s*\n)(.*?)(\n% END AUTO {name})", source, re.DOTALL)
    if not block:
        raise ValueError(f"Missing AUTO {name} block.")
    table, count = re.subn(r"(\\midrule\s*\n).*?(\n\\bottomrule)",
                          lambda match: match.group(1) + rows + match.group(2), block.group(2),
                          count=1, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"AUTO {name} needs midrule and bottomrule row delimiters.")
    return source[:block.start(2)] + table + source[block.end(2):]


def measured_tradeoff(source, figure):
    """Update the short-paper plot's three measured series while retaining its caption and layout."""
    series = (("tradeoffaccuracy", "accuracy", 100, 4),
              ("tradeoffmean", "mean_latency_s", 1, 6),
              ("tradeofftail", "p95_latency_s", 1, 6))
    for color, key, scale, digits in series:
        values = [(limit, figure[limit]["methods"]["jev_joint"].get(key)) for limit in table3.LIMITS]
        if any(value is None for _, value in values):
            raise ValueError(f"The measured repair plot requires {key} at every repair limit.")
        coordinates = " ".join(f"({limit},{value * scale:.{digits}f})" for limit, value in values)
        pattern = rf"(\\addplot\[color={color},[^\]]*\]\s*coordinates\s*\{{)[^}}]*(\}};)"
        source, count = re.subn(pattern, lambda match: match.group(1) + coordinates + match.group(2),
                                source, count=1)
        if count != 1:
            raise ValueError(f"The repair plot needs one {color} coordinate series.")
    # Keep the accessible description's endpoint values in sync without changing its prose.
    first = figure[min(table3.LIMITS)]["methods"]["jev_joint"]
    last = figure[max(table3.LIMITS)]["methods"]["jev_joint"]
    descriptions = (
        (r"(Call accuracy rises from )\d+(?:\.\d+)?( to )\d+(?:\.\d+)?( percent)",
         first["accuracy"] * 100, last["accuracy"] * 100, 1),
        (r"(Mean API time rises from )\d+(?:\.\d+)?( to )\d+(?:\.\d+)?( seconds)",
         first["mean_latency_s"], last["mean_latency_s"], 2),
        (r"(the 95th percentile rises from )\d+(?:\.\d+)?( to )\d+(?:\.\d+)?( seconds)",
         first["p95_latency_s"], last["p95_latency_s"], 2),
    )
    for pattern, low, high, digits in descriptions:
        source = re.sub(pattern, lambda match: (match.group(1) + f"{low:.{digits}f}" + match.group(2)
                                                + f"{high:.{digits}f}" + match.group(3)), source)
    return source


def update_paper(paper, final, figure, rules, costs):
    path = Path(paper)
    source = path.read_text()
    method_name = "JBFC"
    section_input = r"\input{sec/experiment_setting_and_results}"
    if section_input in source and "% BEGIN AUTO CATEGORIES" not in source:
        # The ACM draft keeps measured tables in its experiment section; leave the supplied front matter alone.
        method_name = "DMFC" if "Decision-Model-Guided Function Calling" in source else "JBFC"
        path = path.parent / "sec" / "experiment_setting_and_results.tex"
        source = path.read_text()
    if ACM_SHORT in source:
        # Only regenerate data: the short-paper captions, headings and plot layout are author-edited.
        source = _replace_table_rows(source, "CATEGORIES", compact_main_rows(final, costs, method_name))
        result_rows = _table_rows(paper_tables.latex_table2(final["gpt-5-nano"][0]))
        if method_name == "DMFC":
            result_rows = result_rows.replace(r"\textbf{JBFC", r"\textbf{DMFC")
        source = _replace_table_rows(source, "RESULTS", result_rows)
        repairs = re.search(r"% BEGIN AUTO REPAIRS\s*\n(.*?)\n% END AUTO REPAIRS", source, re.DOTALL)
        if not repairs or repairs.group(1).strip() != r"\input{sec/repair_tradeoff}":
            raise ValueError("ACM SHORT's REPAIRS block must input sec/repair_tradeoff.")
        plot = path.parent / "repair_tradeoff.tex"
        plot_source = measured_tradeoff(plot.read_text(), figure)
        plot.write_text(plot_source)
        path.write_text(source)
        return
    source = _replace_block(source, "CATEGORIES", table1_repeat.latex_table1(
        final["gpt-5-nano"], final["gpt-5.4-mini"], costs=[column["mean"] for column in costs]))
    source = _replace_block(source, "RESULTS", paper_tables.latex_table2(final["gpt-5-nano"][0]))
    source = _replace_block(source, "REPAIRS", table3.latex_repairs_figure(figure))
    source = _replace_block(source, "RULES", latex_rules(rules))
    if method_name == "DMFC":
        source = source.replace(r"\textbf{JBFC", r"\textbf{DMFC")
    path.write_text(source)


def table1_report(final):
    """Table 1's numbers with the final rule, as scripts/table1_repeat.py reports them."""
    dirs = {model: [s["run_dir"] for s in final[model]] for model in MODELS}
    rows = {model: [table1_repeat.load_rows(run_dir) for run_dir in dirs[model]] for model in MODELS}
    for model in MODELS:
        table1_repeat.check_new_samples(rows[model], dirs[model])
    small, large = MODELS
    return table1_repeat.build_report(dirs[small], dirs[large], final[small], final[large], rows[small], rows[large])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", help="update this manuscript's measured tables and repair plot")
    parser.add_argument("--json", help=f"report file (default {OUT}/final.json)")
    args = parser.parse_args(argv)
    built = build(ROOT)
    figure = figure_summaries(ROOT)
    nano_run1 = built["table1"]["gpt-5-nano"][0]
    if abs(figure[LIMIT]["methods"]["jev_joint"]["accuracy"] - nano_run1["methods"]["jev_joint"]["accuracy"]) > 1e-12:
        raise ValueError("Figure 2 at r=3 must equal Table 2's JBFC.")
    rules = rule_report(built["table1"], built["last"])
    costs = table1_costs(built["table1"])
    if args.paper:
        update_paper(args.paper, built["table1"], figure, rules, costs)
    report = {
        "settings": {"threshold": THRESHOLD, "final_rule": RULE, "max_repairs": LIMIT, "runs": RUNS},
        "table1": table1_report(built["table1"]),
        "table1_costs": costs,
        "table2": {"run_dir": nano_run1["run_dir"], "parts": nano_run1["manifest"]["derived"]["parts"],
                   "methods": {m: {k: v.get(k) for k in ("accuracy", "function_accuracy",
                                                          "arguments_accuracy_conditional", "mean_latency_s",
                                                          "p95_latency_s",
                                                          "mean_cost_usd", "mean_repairs", "mean_llm_proposals",
                                                          "coverage", "accepted_precision", "exhaustion_rate",
                                                          "incomplete_proposal_n")}
                               for m, v in nano_run1["methods"].items()},
                   "paired_vs_direct": nano_run1["paired_vs_direct"],
                   "history_ablation": nano_run1["history_ablation"]},
        "figure2": {r: {k: s["methods"]["jev_joint"].get(k) for k in ("accuracy", "function_accuracy", "mean_repairs",
                                                                       "exhaustion_rate", "mean_latency_s", "p95_latency_s",
                                                                       "mean_cost_usd", "coverage",
                                                                       "accepted_precision")}
                    for r, s in figure.items()},
        "rules": rules,
        "history_labels": history_labels(nano_run1),
        "limit_changes": limit_changes(figure),
    }
    target = Path(args.json) if args.json else ROOT / OUT / "final.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.paper:
        print(f"Measured tables and repair plot in {args.paper} now use threshold {THRESHOLD:g} and the first proposal "
              "when none passes; rebuild the PDF.", file=sys.stderr)
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
