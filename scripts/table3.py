"""Table 3: JBFC with repair limits 1, 2, 3, 5 and 7 on all examples, from one extended run.

The run comes from scripts/extend_run.py, which continues the r=2 JBFC trajectories
of the Table 2 run up to r=7. Each row cuts those trajectories at its limit; the r=2
cut must reproduce the source run exactly, so Table 3 and Table 2 agree at r=2.

Usage: python scripts/table3.py EXTENDED_RUN [--paper paper/main.tex]
       python scripts/table3.py --template --paper paper/main.tex   (unrun table)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_fc.reporting import (  # noqa: E402
    PAPER_REPAIRS,
    _bold_best,
    _check_all_examples,
    _replace_block,
    _tex_escape,
    build_summary,
    derive_policy_run,
    paper_protocol_matches,
    read_jsonl,
)

LIMITS = (1, 2, 3, 5, 7)
METHOD = "jev_joint"


def check_extension(run_dir):
    """Check that a complete extended run continues an unchanged source run; return both paths."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    extends = manifest.get("extends") or {}
    if manifest.get("methods") != [METHOD] or not extends:
        raise ValueError("Table 3 needs a JBFC run made by scripts/extend_run.py.")
    source = Path(extends["run_dir"])
    source = source if source.is_absolute() else ROOT / source
    source_manifest = json.loads((source / "manifest.json").read_text())
    if source_manifest.get("config_id") != extends.get("config_id") or source_manifest.get("max_repairs") != extends.get("max_repairs"):
        raise ValueError("The source run of this extension was replaced.")
    if hashlib.sha256((source / "records.jsonl").read_bytes()).hexdigest() != extends.get("records_sha256"):
        raise ValueError("The source run's records changed after the extension.")
    return run_dir, source


def derive_limits(run_dir, limits=LIMITS):
    """Summaries for each limit: the run itself at its own limit, cuts of it below."""
    run_dir = Path(run_dir)
    summary = build_summary(run_dir)
    if summary["status"] != "complete":
        raise ValueError("The extended run is incomplete.")
    cap = summary["manifest"]["max_repairs"]
    if max(limits) != cap:
        raise ValueError(f"The extended run must stop at r={max(limits)}, not r={cap}.")
    return {r: summary if r == cap else derive_policy_run(run_dir, run_dir / "limits" / f"r{r}", max_repairs=r)
            for r in limits}


def check_matches_source(run_dir, source_dir):
    """The cut at the source limit must reproduce every source JBFC row."""
    run_dir, source_dir = Path(run_dir), Path(source_dir)
    cap = json.loads((source_dir / "manifest.json").read_text())["max_repairs"]
    cut = run_dir / "limits" / f"r{cap}"
    if not (cut / "records.jsonl").exists():
        derive_policy_run(run_dir, cut, max_repairs=cap)
    keys = ("calls", "score", "accepted", "exhausted", "repairs")
    derived = {(r["id"], r["replicate"]): r for r in read_jsonl(cut / "records.jsonl")}
    source = {(r["id"], r["replicate"]): r for r in read_jsonl(source_dir / "records.jsonl") if r["method"] == METHOD}
    if derived.keys() != source.keys():
        raise ValueError("The extended run does not cover the same examples as its source.")
    for key, row in source.items():
        other = derived[key]
        if (any(row[k] != other[k] for k in keys)
                or [s["proposal"] for s in row["trace"]] != [s["proposal"] for s in other["trace"]]):
            raise ValueError(f"The r={cap} cut differs from the source run at {key[0]}.")


def check_paper_protocol(summary):
    """The extended run must follow the paper's protocol apart from its larger limit."""
    manifest = summary["manifest"]
    if not paper_protocol_matches({**manifest, "max_repairs": PAPER_REPAIRS}, (METHOD,)):
        raise ValueError("Table 3 needs the paper's models, interface, prices and settings.")
    if manifest.get("extends", {}).get("max_repairs") != PAPER_REPAIRS:
        raise ValueError(f"Table 3 must extend the paper's r={PAPER_REPAIRS} run.")
    _check_all_examples(manifest)


def latex_table3(summaries=None):
    def fmt(value, scale=1, precision=2):
        return "--" if value is None else f"{value * scale:.{precision}f}"
    rows = []
    for r in LIMITS:
        values = ((summaries or {}).get(r) or {}).get("methods", {}).get(METHOD, {})
        rows.append([fmt(values.get("accuracy"), 100, 1), fmt(values.get("function_accuracy"), 100, 1),
                     fmt(values.get("mean_repairs")), fmt(values.get("exhaustion_rate"), 100, 1),
                     fmt(values.get("mean_latency_s")), fmt(values.get("mean_cost_usd"), 1000, 3)])
    missing = any("--" in row for row in rows)
    bolded = False
    for column in range(2):  # call and function accuracy: higher is better
        cells, bold = _bold_best([row[column] for row in rows])
        bolded = bolded or bold
        for row, cell in zip(rows, cells):
            row[column] = cell
    lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.7pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"$r$ & Call & Func. & Repairs & Exhaust. & Sec. & USD/1k \\", r"\midrule"]
    lines += [f"${r}$ & " + " & ".join(row) + r" \\" for r, row in zip(LIMITS, rows)]
    model = ((summaries or {}).get(max(LIMITS)) or {}).get("manifest", {}).get("llm_model")
    generator = rf" Generator: \texttt{{{_tex_escape(model)}}}." if model else ""
    best = " Bold marks the highest accuracy in each column." if bolded else ""
    unrun = " -- means unrun." if missing else ""
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{JBFC} on all BFCL examples by repair limit $r$, cut from the run of "
              r"Table~\ref{tab:main} extended to $r=7$, so $r=2$ matches it." + generator + best
              + r" Accuracy (\%) excludes live relevance; Exhaust.: examples reaching the limit without passing (\%)."
              + unrun + "}",
              r"\label{tab:repairs}", r"\end{table}"]
    return "\n".join(lines)


def update_paper_table3(paper_path, summaries=None):
    path = Path(paper_path)
    path.write_text(_replace_block(path.read_text(), "REPAIRS", latex_table3(summaries)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--paper")
    parser.add_argument("--template", action="store_true", help="write the unrun Table 3 into --paper")
    args = parser.parse_args(argv)
    if args.template:
        if not args.paper:
            raise ValueError("--template needs --paper")
        update_paper_table3(args.paper)
        return None
    if not args.run_dir:
        raise ValueError("Give the extended run directory")
    run_dir, source = check_extension(args.run_dir)
    summaries = derive_limits(run_dir)
    check_matches_source(run_dir, source)
    if args.paper:
        check_paper_protocol(summaries[max(LIMITS)])
        update_paper_table3(args.paper, summaries)
    rows = {r: {key: summaries[r]["methods"][METHOD].get(key) for key in
                ("accuracy", "function_accuracy", "mean_repairs", "exhaustion_rate", "mean_latency_s",
                 "mean_cost_usd", "incomplete_proposal_n")} for r in LIMITS}
    print(json.dumps({"source": str(source), "rows": rows}, indent=2))
    return summaries


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
