"""Table 1 with a third column: Direct with a larger generator.

Table 1 shows Direct and JBFC with the paper's generator (gpt-5-nano) by category,
from the seven-method run that also fills Table 2. The third column is Direct with
a larger, more expensive generator (make compare), run once on the same 3,051
examples with the same code, interface and settings; only the model and its prices
differ. A last row gives the cost per 1,000 requests, so the table shows what each
column pays for its accuracy. The comparison run never asks Jev.

Usage: python scripts/table1_compare.py MAIN_RUN [--compare RUN] [--model NAME] [--paper paper/main.tex]
Without a finished --compare run, the third column is drawn empty for --model.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_fc.reporting import (  # noqa: E402
    PAPER_REPAIRS,
    PAPER_SETTINGS,
    TABLE1_CATEGORIES,
    _check_all_examples,
    _replace_block,
    _tex_escape,
    read_jsonl,
    table1_protocol_matches,
)

DEFAULT_MODEL = "gpt-5.4-mini"
# The comparison run may differ from the paper's protocol only in its generator (and so its
# reasoning-effort fallback); it has no verifier, so the Jev settings are empty.
FREE = {"llm_model", "reasoning_effort", "jev_model", "jev_gateway"}
SAME_AS_MAIN = ("source_sha256", "dataset_sha256", "ids", "split", "split_seed", "limit", "llm_provider",
                "llm_api_style", "llm_gateway", "max_completion_tokens", "temperature")
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


def check_main(summary):
    manifest = summary["manifest"]
    if not table1_protocol_matches(manifest):
        raise ValueError("The main run must be the paper's run (Direct and JBFC, all examples, r=2, gpt-5-nano).")
    _check_all_examples(manifest)


def check_compare(main, compare):
    """The comparison run: Direct only, same code, data and settings as the main run, another generator."""
    manifest, main_manifest = compare["manifest"], main["manifest"]
    if manifest.get("methods") != ["direct"]:
        raise ValueError("The comparison run must run Direct only (make compare).")
    for key, value in PAPER_SETTINGS.items():
        if key not in FREE and manifest.get(key) != value:
            raise ValueError(f"The comparison run differs from the paper's protocol in {key}.")
    for key in SAME_AS_MAIN:
        if manifest.get(key) != main_manifest.get(key):
            raise ValueError(f"The comparison run differs from the main run in {key}.")
    if manifest.get("jev_model") is not None:
        raise ValueError("The comparison run must not use Jev.")
    if not manifest.get("llm_model") or manifest.get("llm_model") == main_manifest.get("llm_model"):
        raise ValueError("The comparison run needs a different generator.")
    prices = manifest.get("prices_usd_per_million") or {}
    if any(prices.get(name) is None for name in LLM_PRICES):
        raise ValueError("Set the comparison model's prices (LLM_INPUT_PRICE, LLM_CACHED_INPUT_PRICE, "
                         "LLM_OUTPUT_PRICE) so its cost can be compared.")
    _check_all_examples(manifest)


def _cell(value, scale=100, precision=1):
    return "--" if value is None else f"{value * scale:.{precision}f}"


def _bold_top(cells):
    """Bold every shown value tied for the highest; nothing when fewer than two differ."""
    shown = {float(cell) for cell in cells if cell != "--"}
    if len(shown) < 2:
        return cells, False
    best = max(shown)
    return [rf"\textbf{{{cell}}}" if cell != "--" and float(cell) == best else cell for cell in cells], True


def columns(main, compare=None):
    """(summary, method) for the three columns: main Direct, main JBFC, comparison Direct."""
    return [(main, "direct"), (main, "jev_joint"), (compare, "direct")]


def latex_table1(main, compare=None, model=DEFAULT_MODEL):
    """Table 1: Direct and JBFC with the paper's generator, and Direct with a larger one."""
    small = main["manifest"]["llm_model"]
    large = compare["manifest"]["llm_model"] if compare else model
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             rf" & & \multicolumn{{2}}{{c}}{{\texttt{{{_tex_escape(small)}}}}} & \texttt{{{_tex_escape(large)}}}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}",
             r"Category & Examples & Direct & \textbf{JBFC} & Direct\\", r"\midrule"]
    missing = bolded = False
    for category, label, count in TABLE1_CATEGORIES:
        key = "relevance_call_rate" if category == "live_relevance" else "accuracy"
        cells = [_cell(((summary or {}).get("by_category") or {}).get(f"{category}/{method}", {}).get(key))
                 for summary, method in columns(main, compare)]
        missing = missing or "--" in cells
        cells, bold = _bold_top(cells)
        bolded = bolded or bold
        lines.append(f"{label} & {count} & " + " & ".join(cells) + r"\\")
    total = [_cell(((summary or {}).get("methods") or {}).get(method, {}).get("accuracy"))
             for summary, method in columns(main, compare)]
    cost = [_cell(((summary or {}).get("methods") or {}).get(method, {}).get("mean_cost_usd"), 1000, 3)
            for summary, method in columns(main, compare)]
    missing = missing or "--" in total or "--" in cost
    total, bold = _bold_top(total)
    bolded = bolded or bold
    effort = (compare or {}).get("manifest", {}).get("reasoning_effort")
    effort_note = "" if effort in (None, PAPER_SETTINGS["reasoning_effort"]) else \
        rf" \texttt{{{_tex_escape(large)}}} uses reasoning effort \texttt{{{_tex_escape(effort or 'default')}}}."
    lines += [r"\midrule", "Total & 3051 & " + " & ".join(total) + r"\\",
              "USD/1k & & " + " & ".join(cost) + r"\\",
              r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Accuracy (\%) on all BFCL examples, one run per generator with repair limit "
              rf"$r={PAPER_REPAIRS}$ (Direct makes no repairs). \textbf{{JBFC}} adds \textit{{Jev}} to "
              rf"\texttt{{{_tex_escape(small)}}}." + effort_note
              + (" Bold marks the highest accuracy in each row;" if bolded else "")
              + " USD/1k is the cost per 1,000 requests."
              + r" $^\dagger$Call presence only; the total excludes these 16 cases."
              + (" -- means the experiment has not been run." if missing else "") + "}",
              r"\label{tab:categories}", r"\end{table}"]
    return "\n".join(lines)


def update_paper_table1(paper_path, main, compare=None, model=DEFAULT_MODEL):
    path = Path(paper_path)
    path.write_text(_replace_block(path.read_text(), "CATEGORIES", latex_table1(main, compare, model)))


def returned_models(run_dir):
    """Model snapshots the API reported for the comparison run's proposals."""
    return sorted({row["proposal"].get("model") for row in read_jsonl(Path(run_dir) / "initial.jsonl")} - {None})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("main_run", help="the paper's run (Table 2 run)")
    parser.add_argument("--compare", help="Direct run with a larger generator (make compare)")
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
        update_paper_table1(args.paper, main_summary, compare, args.model)
    report = {"main": args.main_run, "compare": args.compare if compare else None,
              "model": compare["manifest"]["llm_model"] if compare else args.model}
    for name, (summary, method) in zip(("direct", "jbfc", "compare_direct"), columns(main_summary, compare)):
        values = ((summary or {}).get("methods") or {}).get(method, {})
        report[name] = {key: values.get(key) for key in ("accuracy", "function_accuracy", "mean_cost_usd",
                                                         "mean_latency_s", "incomplete_proposal_n")}
    if compare:
        report["compare_returned_models"] = returned_models(args.compare)
        report["compare_reasoning_effort"] = compare["manifest"].get("reasoning_effort")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
