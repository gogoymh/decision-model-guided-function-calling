"""JBFC with another Jev threshold against Direct and JBFC at the run's own threshold.

For each generator, every Table 1 run with a rerun from scripts/rethreshold.py
(results/jbfc_t<THRESHOLD>_<model>_run<k>) is paired with it: the rerun starts from the
same first proposals, so Direct, JBFC at the run's threshold and JBFC at the new one are
compared example by example. The report gives each run's values and their means
(accuracy, cost, time, repairs, acceptance and its precision, fixes and breaks against
the first proposal) and paired differences with 95% intervals: each example is averaged
over the runs first, then examples are resampled (seed 1729, 2,000 times, as in the
paper). Exploratory: it never changes the paper.

Usage: python scripts/threshold_compare.py [--threshold 0.7] [--json OUT]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_tables  # noqa: E402
import table1_repeat  # noqa: E402

from jev_fc.reporting import TABLE1_CATEGORIES, paired_delta  # noqa: E402

RUN_KEYS = ("accuracy", "function_accuracy", "mean_cost_usd", "mean_latency_s", "mean_repairs", "exhaustion_rate",
            "coverage", "accepted_precision")


def rerun_dir(threshold, model, run):
    return f"results/jbfc_t{threshold:g}_{model}_run{run}"


def check_pair(base, base_dir, rerun, rerun_dir_, threshold):
    """The rerun must start from this run's first proposals and change only the threshold."""
    manifest, source = rerun["manifest"], base["manifest"]
    if (manifest.get("rethreshold") or {}).get("to") != threshold or manifest.get("threshold") != threshold:
        raise ValueError(f"{rerun_dir_} is not a rerun with threshold {threshold}.")
    if Path(ROOT / (manifest.get("extends") or {}).get("run_dir", "")).resolve() != Path(ROOT / base_dir).resolve():
        raise ValueError(f"{rerun_dir_} was not made from {base_dir}.")
    for key in ("source_sha256", "dataset_sha256", "ids", "replicates", "max_repairs", "llm_model", "reasoning_effort",
                "max_completion_tokens", "prices_usd_per_million", "jev_model", "exhaustion_policy"):
        if manifest.get(key) != source.get(key):
            raise ValueError(f"{rerun_dir_} differs from {base_dir} in {key}.")


def compare_model(model, base_dirs, threshold, rerun_dirs=None):
    """Report for one generator; runs without a finished rerun are left out."""
    rerun_dirs = rerun_dirs or [rerun_dir(threshold, model, run) for run in range(1, len(base_dirs) + 1)]
    pairs = []
    for run, (base_dir, new_dir) in enumerate(zip(base_dirs, rerun_dirs), start=1):
        if not table1_repeat._complete(ROOT / new_dir):
            continue
        base, rerun = paper_tables.load_summary(ROOT / base_dir), paper_tables.load_summary(ROOT / new_dir)
        check_pair(base, base_dir, rerun, new_dir, threshold)
        base_rows, new_rows = table1_repeat.load_rows(ROOT / base_dir), table1_repeat.load_rows(ROOT / new_dir)
        if table1_repeat._first(base_rows, "first_full") != table1_repeat._first(new_rows, "first_full"):
            raise ValueError(f"{new_dir} does not share the first proposals of {base_dir}.")
        pairs.append({"run": run, "base_dir": base_dir, "rerun_dir": new_dir, "base": base, "rerun": rerun,
                      "rows": [*base_rows, *({**row, "method": "rerun"} for row in new_rows if row["method"] == "jev_joint")]})
    if not pairs:
        return None
    columns = {"direct": ("base", "direct"), "jbfc": ("base", "jev_joint"), f"jbfc_t{threshold:g}": ("rerun", "jev_joint")}
    report = {"runs": [pair["run"] for pair in pairs], "threshold": threshold,
              "base_threshold": pairs[0]["base"]["manifest"]["threshold"], "columns": {}, "per_run": [], "paired": {}}
    for name, (which, method) in columns.items():
        report["columns"][name] = {
            key: table1_repeat.spread([(pair[which]["methods"].get(method) or {}).get(key) for pair in pairs])
            for key in RUN_KEYS}
        report["columns"][name]["categories"] = {
            category: table1_repeat.spread([paper_tables._value(pair[which], method,
                                                                "relevance_call_rate" if category == "live_relevance"
                                                                else "accuracy", category) for pair in pairs])
            for category, _, _ in TABLE1_CATEGORIES}
    for pair in pairs:
        rerun_rows = [{**row, "method": "jev_joint"} for row in pair["rows"] if row["method"] == "rerun"]
        report["per_run"].append({
            "run": pair["run"], "base_dir": pair["base_dir"], "rerun_dir": pair["rerun_dir"],
            "jev_gateway": pair["rerun"]["manifest"].get("jev_gateway"),
            "jbfc": table1_repeat.transitions(pair["rows"]),
            f"jbfc_t{threshold:g}": table1_repeat.transitions(rerun_rows)})
    rows_by_run = [pair["rows"] for pair in pairs]
    for label, method, baseline in ((f"jbfc_t{threshold:g} - direct", "rerun", "direct"),
                                    (f"jbfc_t{threshold:g} - jbfc", "rerun", "jev_joint"),
                                    ("jbfc - direct", "jev_joint", "direct")):
        report["paired"][label] = paired_delta(table1_repeat._tagged(rows_by_run, {method: "method", baseline: "baseline"}),
                                               "method", baseline="baseline")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--json", help="also write the report to this file")
    args = parser.parse_args(argv)
    report = {}
    for model, base_dirs in (("gpt-5-nano", table1_repeat.SMALL_RUNS), ("gpt-5.4-mini", table1_repeat.LARGE_RUNS)):
        result = compare_model(model, base_dirs, args.threshold)
        if result is not None:
            report[model] = result
    if not report:
        raise ValueError(f"No finished rerun with threshold {args.threshold:g} (results/jbfc_t{args.threshold:g}_*).")
    text = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
