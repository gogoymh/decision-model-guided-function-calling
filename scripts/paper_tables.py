"""Tables 1 and 2 at the paper's repair limit (r=3) from the raised runs (make r3).

Table 2 compares the seven methods with gpt-5-nano. Table 1 shows Direct and JBFC by
category for gpt-5-nano and for the larger, costlier gpt-5.4-mini, with each column's
cost per 1,000 requests. Both come from scripts/raise_limit.py: the gpt-5-nano run
raises the Table 2 run's limit (its JBFC rows are the Figure 2 extension cut at r=3),
and the gpt-5.4-mini run adds JBFC to that model's Direct run. Each run covers all
3,051 examples once, Direct and JBFC share their first proposals, and both runs use
the same code, data, interface and settings; only the generator and its prices differ.

Usage: python scripts/paper_tables.py MAIN_RUN [--compare RUN] [--model NAME] [--paper paper/main.tex]
Without a finished --compare run, its two columns are drawn empty for --model.
These are the threshold-0.8 runs. The paper's tables use threshold 0.5 and the first proposal when none
passes (user decision 2026-10-18) and come from scripts/final_tables.py (make final); --paper here would
overwrite them with the earlier settings.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_fc import reporting  # noqa: E402
from jev_fc.pipeline import METHODS  # noqa: E402
from jev_fc.reporting import (  # noqa: E402
    PAPER_PRICES,
    PAPER_SETTINGS,
    TABLE1_CATEGORIES,
    _check_all_examples,
    _replace_block,
    _tex_escape,
)

LIMIT = 3  # the paper's repair limit, chosen from Figure 2 (user decision, 2026-10-18)
DEFAULT_MODEL = "gpt-5.4-mini"
JEV_ROUTES = ("apim", "direct")
SAME_AS_MAIN = ("source_sha256", "dataset_sha256", "ids", "split", "split_seed", "limit", "replicates",
                "max_repairs", "threshold", "exhaustion_policy", "llm_provider", "llm_api_style", "llm_gateway",
                "max_completion_tokens", "temperature", "jev_model")
LLM_PRICES = ("llm_input", "llm_cached_input", "llm_output")


def load_summary(run_dir):
    """A run's summary.json, checked to be complete and to cover the records on disk."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    with (run_dir / "records.jsonl").open() as handle:
        records = sum(1 for line in handle if line.strip())
    if summary.get("status") != "complete" or records != summary.get("expected_records"):
        raise ValueError(f"{run_dir} is not a complete run.")
    if summary.get("completed_records") != records:
        raise ValueError(f"{run_dir}/summary.json is older than its records; run jev-fc report on it first.")
    return summary


def check_route(manifest):
    """Jev answers through the gateway or directly; any change between attempts is recorded."""
    routes = manifest.get("jev_routes") or [{"jev_gateway": manifest.get("jev_gateway")}]
    if any(route.get("jev_gateway") not in JEV_ROUTES for route in routes) or \
            routes[-1].get("jev_gateway") != manifest.get("jev_gateway"):
        raise ValueError("Jev routes must be the gateway or direct, as recorded in the manifest.")


def check_main(summary):
    manifest = summary["manifest"]
    if sorted(manifest.get("methods") or []) != sorted(METHODS):
        raise ValueError("The main run must have all seven methods (make r3).")
    for key, value in {**PAPER_SETTINGS, "max_repairs": LIMIT}.items():
        if key != "jev_gateway" and manifest.get(key) != value:
            raise ValueError(f"The main run differs from the paper's protocol in {key}.")
    if any(manifest.get("prices_usd_per_million", {}).get(key) != value for key, value in PAPER_PRICES.items()):
        raise ValueError("The main run must use the paper's prices.")
    check_route(manifest)
    _check_all_examples(manifest)


def check_compare(main, compare):
    """The comparison run: Direct and JBFC, same code, data and settings as the main run, another generator."""
    manifest, main_manifest = compare["manifest"], main["manifest"]
    if manifest.get("methods") != ["direct", "jev_joint"]:
        raise ValueError("The comparison run must have Direct and JBFC (make r3).")
    for key in SAME_AS_MAIN:
        if manifest.get(key) != main_manifest.get(key):
            raise ValueError(f"The comparison run differs from the main run in {key}.")
    if not manifest.get("llm_model") or manifest.get("llm_model") == main_manifest.get("llm_model"):
        raise ValueError("The comparison run needs a different generator.")
    prices = manifest.get("prices_usd_per_million") or {}
    if any(prices.get(name) is None for name in LLM_PRICES) or prices.get("jev_input") != PAPER_PRICES["jev_input"]:
        raise ValueError("Set the comparison model's prices (LLM_INPUT_PRICE, LLM_CACHED_INPUT_PRICE, "
                         "LLM_OUTPUT_PRICE) so its cost can be compared.")
    check_route(manifest)
    _check_all_examples(manifest)


def _cell(value, scale=100, precision=1):
    return "--" if value is None else f"{value * scale:.{precision}f}"


def _bold_top(cells):
    """Bold every shown value tied for the highest; nothing when fewer than two values differ."""
    shown = {float(cell) for cell in cells if cell != "--"}
    if len(shown) < 2:
        return cells, False
    best = max(shown)
    return [rf"\textbf{{{cell}}}" if cell != "--" and float(cell) == best else cell for cell in cells], True


def columns(main, compare=None):
    """(summary, method) for Direct and JBFC with the paper's generator, then with the larger one."""
    return [(main, "direct"), (main, "jev_joint"), (compare, "direct"), (compare, "jev_joint")]


def _value(summary, method, key, category=None):
    group = (summary or {}).get("by_category" if category else "methods") or {}
    return group.get(f"{category}/{method}" if category else method, {}).get(key)


def latex_table1(main, compare=None, model=DEFAULT_MODEL):
    """Table 1: Direct and JBFC by category for two generators, with cost per 1,000 requests."""
    small = main["manifest"]["llm_model"]
    large = compare["manifest"]["llm_model"] if compare else model
    limit = main["manifest"]["max_repairs"]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             rf" & & \multicolumn{{2}}{{c}}{{\texttt{{{_tex_escape(small)}}}}}"
             rf" & \multicolumn{{2}}{{c}}{{\texttt{{{_tex_escape(large)}}}}}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Category & Examples & Direct & \textbf{JBFC} & Direct & \textbf{JBFC}\\", r"\midrule"]
    missing = bolded = False
    for category, label, count in TABLE1_CATEGORIES:
        key = "relevance_call_rate" if category == "live_relevance" else "accuracy"
        cells = [_cell(_value(summary, method, key, category)) for summary, method in columns(main, compare)]
        missing = missing or "--" in cells
        cells, bold = _bold_top(cells)
        bolded = bolded or bold
        lines.append(f"{label} & {count} & " + " & ".join(cells) + r"\\")
    total = [_cell(_value(summary, method, "accuracy")) for summary, method in columns(main, compare)]
    cost = [_cell(_value(summary, method, "mean_cost_usd"), 1000, 3) for summary, method in columns(main, compare)]
    missing = missing or "--" in total or "--" in cost
    total, bold = _bold_top(total)
    bolded = bolded or bold
    effort = (compare or {}).get("manifest", {}).get("reasoning_effort")
    effort_note = "" if effort in (None, PAPER_SETTINGS["reasoning_effort"]) else \
        rf" \texttt{{{_tex_escape(large)}}} uses reasoning effort \texttt{{{_tex_escape(effort or 'default')}}}."
    lines += [r"\midrule", "Total & 3051 & " + " & ".join(total) + r"\\",
              "USD/1k & & " + " & ".join(cost) + r"\\",
              r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Accuracy (\%) on all BFCL examples, one run per generator, $r={limit}$ "
              r"(Direct makes no repairs)." + effort_note
              + (" Bold: highest accuracy per row." if bolded else "")
              + " USD/1k: cost per 1,000 requests."
              + r" $^\dagger$Call presence only; excluded from the total."
              + (" -- means the experiment has not been run." if missing else "") + "}",
              r"\label{tab:categories}", r"\end{table}"]
    return "\n".join(lines)


def latex_table2(main):
    """Table 2: the seven methods at the main run's repair limit (reporting's layout and bolding)."""
    saved = reporting.PAPER_REPAIRS
    reporting.PAPER_REPAIRS = main["manifest"]["max_repairs"]
    try:
        table = reporting.latex_table(main)
    finally:
        reporting.PAPER_REPAIRS = saved
    # A shorter caption than reporting's, with the same content.
    model = _tex_escape(main["manifest"]["llm_model"])
    long_form = (rf", from one run with repair limit $r={main['manifest']['max_repairs']}$. Generator: \texttt{{{model}}}.",
                 "Bold marks the highest accuracy in each column.",
                 r"exclude live relevance. Argument accuracy (\%) uses calls with the correct function and argument labels.")
    short_form = (rf" (one run, $r={main['manifest']['max_repairs']}$, \texttt{{{model}}}).",
                  "Bold: highest accuracy per column.",
                  r"exclude live relevance; argument accuracy (\%) uses calls with the correct function.")
    for old, new in zip(long_form, short_form):
        table = table.replace(old, new)
    return table


def update_paper_tables(paper_path, main, compare=None, model=DEFAULT_MODEL):
    path = Path(paper_path)
    source = _replace_block(path.read_text(), "CATEGORIES", latex_table1(main, compare, model))
    path.write_text(_replace_block(source, "RESULTS", latex_table2(main)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("main_run", help="the raised gpt-5-nano run with all seven methods")
    parser.add_argument("--compare", help="the larger generator's Direct and JBFC run")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="column header while the comparison is unrun")
    parser.add_argument("--paper")
    args = parser.parse_args(argv)
    main_summary = load_summary(args.main_run)
    check_main(main_summary)
    compare = None
    if args.compare and (Path(args.compare) / "summary.json").exists():
        compare = load_summary(args.compare)
        check_compare(main_summary, compare)
    if args.paper:
        update_paper_tables(args.paper, main_summary, compare, args.model)
    report = {"main": args.main_run, "compare": args.compare if compare else None,
              "model": compare["manifest"]["llm_model"] if compare else args.model,
              "max_repairs": main_summary["manifest"]["max_repairs"]}
    for name, (summary, method) in zip(("direct", "jbfc", "compare_direct", "compare_jbfc"),
                                       columns(main_summary, compare)):
        values = ((summary or {}).get("methods") or {}).get(method, {})
        report[name] = {key: values.get(key) for key in ("accuracy", "function_accuracy", "mean_cost_usd",
                                                         "mean_latency_s", "mean_repairs", "exhaustion_rate",
                                                         "incomplete_proposal_n")}
    report["methods"] = {method: {key: values.get(key) for key in ("accuracy", "function_accuracy",
                                                                   "arguments_accuracy_conditional",
                                                                   "mean_latency_s", "mean_cost_usd")}
                         for method, values in main_summary["methods"].items()}
    report["paired_vs_direct"] = main_summary.get("paired_vs_direct")
    report["history_ablation"] = main_summary.get("history_ablation")
    if compare:
        report["compare_paired_vs_direct"] = compare.get("paired_vs_direct")
        report["compare_reasoning_effort"] = compare["manifest"].get("reasoning_effort")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
