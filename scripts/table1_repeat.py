"""Table 1 as the mean of three runs per generator (make repeat).

Run 1 of each generator is its r=3 run from make r3: the gpt-5-nano seven-method run
(also the source of Table 2 and Figure 2) and the gpt-5.4-mini Direct/JBFC run. Runs 2
and 3 repeat Direct and JBFC with new first proposals (scripts/run_repeat.sh: a Direct
run, then JBFC added from its first proposals with scripts/raise_limit.py, the way run 1
of gpt-5.4-mini was made). Every run covers all 3,051 examples with the same code, data,
interface, settings and prices; only the generator's samples differ.

Table 1 shows each cell's mean over the runs, the standard deviation of the total across
runs (the s.d. row), and the mean cost per 1,000 requests. The JSON report adds each run's values and
paired differences with 95% intervals: each example's difference is averaged over the
runs first, then examples are resampled (seed 1729, 2,000 times, as in the paper).

Usage: python scripts/table1_repeat.py [--small RUN RUN RUN] [--large RUN RUN RUN]
                                       [--paper paper/main.tex] [--json OUT] [--if-complete]
--if-complete leaves the paper and OUT unchanged while a run is missing or unfinished.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_tables  # noqa: E402

from jev_fc.reporting import TABLE1_CATEGORIES, _replace_block, _tex_escape, paired_delta  # noqa: E402

SMALL_RUNS = ("results/table2_gpt-5-nano_r3", "results/table1_gpt-5-nano_r3_run2",
              "results/table1_gpt-5-nano_r3_run3")
LARGE_RUNS = ("results/table1_gpt-5.4-mini_r3", "results/table1_gpt-5.4-mini_r3_run2",
              "results/table1_gpt-5.4-mini_r3_run3")
TABLE_METHODS = ("direct", "jev_joint")
REPEAT_SAME = ("llm_model", "reasoning_effort", "prices_usd_per_million")
RUN_KEYS = ("accuracy", "function_accuracy", "arguments_accuracy_conditional", "mean_cost_usd", "mean_latency_s",
            "mean_repairs", "exhaustion_rate", "coverage", "accepted_precision", "incomplete_proposal_n")
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def check_repeat(first, run, label):
    """A later run of run 1's generator: Direct and JBFC from new first proposals, all else as run 1."""
    manifest, reference = run["manifest"], first["manifest"]
    if manifest.get("methods") != list(TABLE_METHODS):
        raise ValueError(f"{label}: a repeat has Direct and JBFC only (make repeat).")
    for key in (*paper_tables.SAME_AS_MAIN, *REPEAT_SAME):
        if manifest.get(key) != reference.get(key):
            raise ValueError(f"{label} differs from run 1 of {reference.get('llm_model')} in {key}.")
    if manifest.get("added") != ["jev_joint"] or not (manifest.get("extends") or {}).get("run_dir"):
        raise ValueError(f"{label}: JBFC must be added to a new Direct run (scripts/run_repeat.sh).")
    paper_tables.check_route(manifest)
    paper_tables._check_all_examples(manifest)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_rows(run_dir):
    """Direct and JBFC rows of a run: example, category, initial and final correctness, first proposal."""
    rows = []
    with (Path(run_dir) / "records.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["method"] not in TABLE_METHODS:
                continue
            first = record["trace"][0]
            proposal = first.get("proposal") or {}
            rows.append({"id": record["id"], "replicate": record["replicate"], "method": record["method"],
                         "category": record["category"], "score": {"correct": record["score"].get("correct")},
                         "initial_correct": (first.get("score") or {}).get("correct"),
                         "first_calls": _digest(proposal.get("calls")),
                         "first_full": _digest([proposal.get("calls"), proposal.get("content")])})
    return rows


def _first(rows, key):
    return {row["id"]: row[key] for row in rows if row["method"] == "direct"}


def check_new_samples(rows_by_run, labels):
    """Every later run must have generated its own first proposals, not copied an earlier run's."""
    firsts = [_first(rows, "first_full") for rows in rows_by_run]
    for later in range(1, len(firsts)):
        for earlier in range(later):
            if firsts[later] == firsts[earlier]:
                raise ValueError(f"{labels[later]} has the same first proposals as {labels[earlier]}; "
                                 "a repeat needs new generations.")


def same_first_calls(rows, reference):
    """Share of examples whose first proposal makes the same calls (or no call) as run 1's."""
    mine, base = _first(rows, "first_calls"), _first(reference, "first_calls")
    shared = [example for example in mine if example in base]
    return sum(mine[example] == base[example] for example in shared) / len(shared) if shared else None


def transitions(rows):
    """JBFC's fixes and breaks against the shared first proposal, live relevance excluded."""
    scored = [row for row in rows if row["method"] == "jev_joint" and row["category"] != "live_relevance"]
    return {"fixes": sum(1 for row in scored if row["initial_correct"] is False and row["score"]["correct"]),
            "breaks": sum(1 for row in scored if row["initial_correct"] and row["score"]["correct"] is False)}


def _tagged(rows_by_run, rename):
    """Rows of every run, renamed for pairing, with the run number as the replicate."""
    return [{**row, "method": rename[row["method"]], "replicate": index}
            for index, rows in enumerate(rows_by_run) for row in rows if row["method"] in rename]


def paired(rows_by_run, method, baseline):
    """method minus baseline, averaging each example over the runs before resampling examples."""
    return paired_delta(_tagged(rows_by_run, {method: "method", baseline: "baseline"}), "method", baseline="baseline")


def across(small_rows, small_method, large_rows, large_method):
    """The larger generator's method minus the smaller generator's method, paired by example."""
    rows = _tagged(small_rows, {small_method: "baseline"}) + _tagged(large_rows, {large_method: "method"})
    return paired_delta(rows, "method", baseline="baseline")


def spread(values):
    """Mean, standard deviation across runs (n-1), minimum and maximum; None when a run lacks the value."""
    if not values or any(value is None for value in values):
        return {"runs": values, "mean": None, "sd": None, "min": None, "max": None}
    return {"runs": values, "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None, "min": min(values), "max": max(values)}


def columns(small, large):
    return [(small, "direct"), (small, "jev_joint"), (large, "direct"), (large, "jev_joint")]


def column_spreads(small, large, key, category=None):
    return [spread([paper_tables._value(summary, method, key, category) for summary in runs])
            for runs, method in columns(small, large)]


def latex_table1(small, large):
    """Table 1: mean accuracy over the runs by category, the total's run-to-run spread and the mean cost."""
    small_model, large_model = small[0]["manifest"]["llm_model"], large[0]["manifest"]["llm_model"]
    limit, count = small[0]["manifest"]["max_repairs"], len(small)
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             rf" & & \multicolumn{{2}}{{c}}{{\texttt{{{_tex_escape(small_model)}}}}}"
             rf" & \multicolumn{{2}}{{c}}{{\texttt{{{_tex_escape(large_model)}}}}}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Category & Examples & Direct & \textbf{JBFC} & Direct & \textbf{JBFC}\\", r"\midrule"]
    bolded = False
    for category, label, examples in TABLE1_CATEGORIES:
        key = "relevance_call_rate" if category == "live_relevance" else "accuracy"
        cells, bold = paper_tables._bold_top([paper_tables._cell(item["mean"])
                                              for item in column_spreads(small, large, key, category)])
        bolded = bolded or bold
        lines.append(f"{label} & {examples} & " + " & ".join(cells) + r"\\")
    totals = column_spreads(small, large, "accuracy")
    total, bold = paper_tables._bold_top([paper_tables._cell(item["mean"]) for item in totals])
    bolded = bolded or bold
    deviation = [paper_tables._cell(item["sd"]) for item in totals]
    cost = [paper_tables._cell(item["mean"], 1000, 3) for item in column_spreads(small, large, "mean_cost_usd")]
    effort = large[0]["manifest"].get("reasoning_effort")
    effort_note = "" if effort in (None, "none") else \
        rf" \texttt{{{_tex_escape(large_model)}}} uses reasoning effort \texttt{{{_tex_escape(effort)}}}."
    lines += [r"\midrule", "Total & 3051 & " + " & ".join(total) + r"\\",
              "s.d. & & " + " & ".join(deviation) + r"\\",
              "USD/1k & & " + " & ".join(cost) + r"\\",
              r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Accuracy (\%) on all BFCL examples, mean of {NUMBER_WORDS.get(count, count)} runs, "
              rf"$r={limit}$ (Direct makes no repairs)." + effort_note
              + (" Bold: highest per row." if bolded else "")
              + " s.d.: of the total across runs. USD/1k: mean cost per 1,000 requests."
              + r" $^\dagger$Call presence only; not in the total.}",
              r"\label{tab:categories}", r"\end{table}"]
    return "\n".join(lines)


def _complete(run_dir):
    try:
        paper_tables.load_summary(run_dir)
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError):
        return False
    return True


def build_report(small_dirs, large_dirs, small, large, small_rows, large_rows):
    models = {"small": small[0]["manifest"]["llm_model"], "large": large[0]["manifest"]["llm_model"]}
    report = {"max_repairs": small[0]["manifest"]["max_repairs"], "runs": len(small_dirs),
              "run_dirs": {models["small"]: list(small_dirs), models["large"]: list(large_dirs)}, "columns": {}}
    names = [f"{models['small']}/direct", f"{models['small']}/jbfc", f"{models['large']}/direct",
             f"{models['large']}/jbfc"]
    for index, name in enumerate(names):
        report["columns"][name] = {
            "accuracy": column_spreads(small, large, "accuracy")[index],
            "categories": {category: column_spreads(small, large, "relevance_call_rate" if category == "live_relevance"
                                                    else "accuracy", category)[index]
                           for category, _, _ in TABLE1_CATEGORIES},
            **{key: column_spreads(small, large, key)[index] for key in RUN_KEYS if key != "accuracy"}}
    report["per_run"] = {}
    for model, dirs, summaries, rows_by_run in ((models["small"], small_dirs, small, small_rows),
                                               (models["large"], large_dirs, large, large_rows)):
        report["per_run"][model] = [{
            "run_dir": str(run_dir), "jev_gateway": summary["manifest"].get("jev_gateway"),
            "jev_routes": summary["manifest"].get("jev_routes"),
            **{method: {key: (summary["methods"].get(method) or {}).get(key) for key in RUN_KEYS}
               for method in TABLE_METHODS},
            "jbfc_minus_direct": paired([rows], "jev_joint", "direct"), **transitions(rows),
            "same_first_calls_as_run_1": same_first_calls(rows, rows_by_run[0])}
            for run_dir, summary, rows in zip(dirs, summaries, rows_by_run)]
    report["paired"] = {
        f"{models['small']} jbfc - direct": paired(small_rows, "jev_joint", "direct"),
        f"{models['large']} jbfc - direct": paired(large_rows, "jev_joint", "direct"),
        f"{models['large']} direct - {models['small']} jbfc": across(small_rows, "jev_joint", large_rows, "direct"),
        f"{models['large']} jbfc - {models['small']} jbfc": across(small_rows, "jev_joint", large_rows, "jev_joint"),
        f"{models['large']} direct - {models['small']} direct": across(small_rows, "direct", large_rows, "direct"),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", nargs="+", default=list(SMALL_RUNS), help="gpt-5-nano runs; run 1 first")
    parser.add_argument("--large", nargs="+", default=list(LARGE_RUNS), help="gpt-5.4-mini runs; run 1 first")
    parser.add_argument("--paper", help="write Table 1 (the CATEGORIES block) into this manuscript")
    parser.add_argument("--json", help="also write the report to this file")
    parser.add_argument("--if-complete", action="store_true", help="do nothing while a run is missing or unfinished")
    args = parser.parse_args(argv)
    dirs = [*args.small, *args.large]
    if len(args.small) != len(args.large) or len(args.small) < 2:
        raise ValueError("Give each generator the same number of runs, at least two, run 1 first.")
    if len({Path(run_dir).resolve() for run_dir in dirs}) != len(dirs):
        raise ValueError("Each run must be a different directory.")
    if args.if_complete:
        waiting = [run_dir for run_dir in dirs if not _complete(run_dir)]
        if waiting:
            print("Table 1 stays a single run until these runs are complete: " + ", ".join(waiting), file=sys.stderr)
            return None
    small = [paper_tables.load_summary(run_dir) for run_dir in args.small]
    large = [paper_tables.load_summary(run_dir) for run_dir in args.large]
    paper_tables.check_main(small[0])
    paper_tables.check_compare(small[0], large[0])
    for dirs_, runs in ((args.small, small), (args.large, large)):
        for run_dir, run in zip(dirs_[1:], runs[1:]):
            check_repeat(runs[0], run, run_dir)
    small_rows = [load_rows(run_dir) for run_dir in args.small]
    large_rows = [load_rows(run_dir) for run_dir in args.large]
    check_new_samples(small_rows, args.small)
    check_new_samples(large_rows, args.large)
    report = build_report(args.small, args.large, small, large, small_rows, large_rows)
    if args.paper:
        path = Path(args.paper)
        path.write_text(_replace_block(path.read_text(), "CATEGORIES", latex_table1(small, large)))
        print(f"Table 1 in {args.paper} is now the mean of {len(small)} runs per generator; rebuild the PDF.",
              file=sys.stderr)
    text = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
        print(f"Numbers: {args.json}", file=sys.stderr)
    print(text)
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
