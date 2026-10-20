"""Figure 2 (formerly Table 3): DMFC with repair limits 1, 2, 3, 5, 7 and 10 on all examples, from one extended run.

The run comes from scripts/extend_run.py, which continues the r=2 DMFC trajectories
of the Table 2 run to a larger limit. An extension can itself be continued
(r=2 -> 7 -> 10), so rows already reported keep their values. Each row of the table
cuts the final trajectories at its limit. The cut at the limit of every run in the
chain must reproduce that run exactly, so the figure agrees with Table 2 at r=2.

Usage: python scripts/table3.py EXTENDED_RUN [--paper paper/main.tex]
       python scripts/table3.py --template --paper paper/main.tex   (unrun figure)
The paper's Figure 2 uses threshold 0.5 and the first proposal when none passes; scripts/final_tables.py
(make final) cuts that run with derive_limits and draws it with latex_repairs_figure. --paper here would
draw the threshold-0.8 run instead.
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
    _check_all_examples,
    _replace_block,
    build_summary,
    derive_policy_run,
    paper_protocol_matches,
    read_jsonl,
)

LIMITS = (1, 2, 3, 5, 7, 10)
TABLE_LIMIT = 3  # the repair limit of Tables 1 and 2 (scripts/paper_tables.py)
METHOD = "jev_joint"
JEV_ROUTE = ("jev_gateway", "jev_base_url", "jev_endpoint")


def _source_of(run_dir):
    """The run that run_dir extends, checked to be unchanged since; None for an original run."""
    extends = json.loads((Path(run_dir) / "manifest.json").read_text()).get("extends")
    if not extends:
        return None
    source = Path(extends["run_dir"])
    source = source if source.is_absolute() else ROOT / source
    source_manifest = json.loads((source / "manifest.json").read_text())
    if (source_manifest.get("config_id") != extends.get("config_id")
            or source_manifest.get("max_repairs") != extends.get("max_repairs")):
        raise ValueError("The source run of this extension was replaced.")
    if hashlib.sha256((source / "records.jsonl").read_bytes()).hexdigest() != extends.get("records_sha256"):
        raise ValueError("The source run's records changed after the extension.")
    return source


def check_extension(run_dir):
    """Check an extended run and every run it continues; return it and its sources, nearest first."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("methods") != [METHOD] or not manifest.get("extends"):
        raise ValueError("Figure 2 needs a DMFC run made by scripts/extend_run.py.")
    chain, current = [], run_dir
    while (source := _source_of(current)) is not None:
        if source.resolve() in {path.resolve() for path in chain} or len(chain) >= len(LIMITS):
            raise ValueError("The chain of extended runs loops.")
        chain.append(source)
        current = source
    return run_dir, chain


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
    """The cut at the source limit must reproduce every source DMFC row."""
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


def check_paper_protocol(summary, root_manifest):
    """The extended run follows the paper's protocol apart from its larger limit, and its chain
    of extensions starts from the paper's r=2 run (root_manifest)."""
    manifest = summary["manifest"]
    routes = manifest.get("jev_routes")
    if routes is not None:
        # A recorded change of Jev's route (same model and questions) for later attempts is allowed;
        # the first route must be the original run's and the last one this run's.
        starts = [route.get("first_attempt") for route in routes] if isinstance(routes, list) else []
        if (not starts or starts[0] != 0 or starts != sorted(set(starts))
                or any(routes[0].get(key) != root_manifest.get(key) for key in JEV_ROUTE)
                or any(routes[-1].get(key) != manifest.get(key) for key in JEV_ROUTE)):
            raise ValueError("The extended run's Jev routes do not match its runs.")
        manifest = {**manifest, **{key: root_manifest.get(key) for key in JEV_ROUTE}}
    if not paper_protocol_matches({**manifest, "max_repairs": PAPER_REPAIRS}, (METHOD,)):
        raise ValueError("Figure 2 needs the paper's models, interface, prices and settings.")
    if (root_manifest.get("extends") or root_manifest.get("max_repairs") != PAPER_REPAIRS
            or not paper_protocol_matches(root_manifest, (METHOD,))):
        raise ValueError(f"Figure 2 must extend the paper's r={PAPER_REPAIRS} run.")
    _check_all_examples(manifest)


# Figure colors: a categorical pair validated for color-blind separation, plus a neutral ink.
COLORS = {"call": "2A78D6", "function": "EB6834", "cost": "52514E"}


def _points(summaries, key, scale):
    points = []
    for r in LIMITS:
        value = ((summaries or {}).get(r) or {}).get("methods", {}).get(METHOD, {}).get(key)
        if value is not None:
            points.append((r, value * scale))
    return points


def _plot(points, color, mark):
    coordinates = " ".join(f"({r},{value:.4f})" for r, value in points)
    return rf"\addplot[color=repair{color}, mark={mark}, mark size=1.5pt, thick] coordinates {{{coordinates}}};"


def _labels(points, text, above):
    anchor, shift = ("south", "1.5pt") if above else ("north", "-1.5pt")
    return [rf"\node[font=\tiny, anchor={anchor}, yshift={shift}, inner sep=1pt] at (axis cs:{r},{value:.4f}) {{{text(value)}}};"
            for r, value in points]


def latex_repairs_figure(summaries=None):
    """Figure 2: DMFC call and function accuracy by repair limit, with each limit's cost under its tick."""
    call = _points(summaries, "accuracy", 100)
    function = _points(summaries, "function_accuracy", 100)
    cost = dict(_points(summaries, "mean_cost_usd", 1000))
    lines = [r"\begin{figure}[t]", r"\centering"]
    lines += [rf"\definecolor{{repair{name}}}{{HTML}}{{{value}}}" for name, value in COLORS.items()]
    ticks = ",".join(str(r) for r in LIMITS)
    # Cost per 1,000 requests sits under each limit's tick, in gray, instead of on a second axis.
    labels = ",".join("{" + str(r) + (rf"\\\color{{repaircost}}{cost[r]:.2f}" if r in cost else "") + "}"
                      for r in LIMITS)
    xlabel = r"Repair limit $r$" + (r" (gray: USD per 1,000 requests)" if cost else "")
    lines += [
        r"\begin{tikzpicture}",
        rf"\begin{{axis}}[scale only axis, width=0.80\columnwidth, height=2.0cm, xmin=0.3, xmax={max(LIMITS) + 0.7},",
        rf"  xtick={{{ticks}}}, xticklabels={{{labels}}}, xticklabel style={{align=center}},",
        r"  ymin=70, ymax=91, ytick={70,75,80,85,90}, ylabel={Accuracy (\%)}, xlabel={" + xlabel + "},",
        r"  axis x line*=bottom, axis y line*=left, ymajorgrids, grid style={black!12},",
        r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}, tick align=outside,",
        r"  major tick length=2pt, ylabel style={yshift=-3pt},",
        r"  legend style={font=\scriptsize, draw=none, fill=none, at={(0.99,0.03)}, anchor=south east,",
        r"  legend columns=2, /tikz/every even column/.append style={column sep=6pt}}]",
    ]
    if call or function:
        lines += [_plot(call, "call", "*"), r"\addlegendentry{Call}",
                  _plot(function, "function", "square*"), r"\addlegendentry{Function}"]
        lines += _labels(call, lambda v: f"{v:.1f}", above=True) + _labels(function, lambda v: f"{v:.1f}", above=True)
    else:
        lines.append(r"\node[font=\scriptsize] at (rel axis cs:0.5,0.5) {not yet run};")
    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    final = ((summaries or {}).get(max(LIMITS)) or {}).get("manifest", {})
    notes = ""  # the generator is Table 2's, named in its caption
    routes = final.get("jev_routes") or []
    for index, route in enumerate(routes[1:], start=1):
        way = "directly" if route.get("jev_gateway") == "direct" else "through the gateway"
        last = routes[index + 1]["first_attempt"] - 1 if index + 1 < len(routes) else max(LIMITS)
        notes += rf" Repairs {route['first_attempt']}--{last} call \textit{{Jev}} {way}."
    if len(call) < len(LIMITS):
        notes += " Limits without points are unrun."
    lines += [r"\caption{\textbf{DMFC} accuracy (\%) by repair limit $r$: one run continued to "
              rf"$r={max(LIMITS)}$ and cut at each limit ($r={TABLE_LIMIT}$ is Table~\ref{{tab:main}})"
              + "." + notes + "}",  # the x-axis label explains the gray costs
              r"\label{fig:repairs}", r"\end{figure}"]
    return "\n".join(lines)


def update_paper_repairs(paper_path, summaries=None):
    path = Path(paper_path)
    path.write_text(_replace_block(path.read_text(), "REPAIRS", latex_repairs_figure(summaries)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--paper")
    parser.add_argument("--template", action="store_true", help="write the unrun repair-limit figure into --paper")
    args = parser.parse_args(argv)
    if args.template:
        if not args.paper:
            raise ValueError("--template needs --paper")
        update_paper_repairs(args.paper)
        return None
    if not args.run_dir:
        raise ValueError("Give the extended run directory")
    run_dir, chain = check_extension(args.run_dir)
    summaries = derive_limits(run_dir)
    for source in chain:
        check_matches_source(run_dir, source)
    if args.paper:
        check_paper_protocol(summaries[max(LIMITS)], json.loads((chain[-1] / "manifest.json").read_text()))
        update_paper_repairs(args.paper, summaries)
    rows = {r: {key: summaries[r]["methods"][METHOD].get(key) for key in
                ("accuracy", "function_accuracy", "mean_repairs", "exhaustion_rate", "mean_latency_s",
                 "mean_cost_usd", "incomplete_proposal_n")} for r in LIMITS}
    print(json.dumps({"source": str(chain[0]), "chain": [str(path) for path in chain], "rows": rows}, indent=2))
    return summaries


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
