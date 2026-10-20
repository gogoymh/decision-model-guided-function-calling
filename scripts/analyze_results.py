"""Print the numbers quoted in the paper's Results section from a finished run.

Usage: uv run python scripts/analyze_results.py results/table2_gpt-5-nano_r3
       (or results/table1_gpt-5.4-mini_r3 for the gpt-5.4-mini columns)

It reads summary.json and the Direct and DMFC (jev_joint) records: accuracy and the
paired difference, fixes and regressions against the shared initial proposal, repairs,
time and cost, then the no-call analysis for Irrelevance and Live irrelevance and the
reject/fix/break counts for categories with reference calls.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REFERENCE = {"simple_python", "multiple", "live_simple", "live_multiple"}
DISPLAY_NAMES = {"direct": "Direct", "schema_only": "Schema repair", "self_refine": "Self-refine",
                 "jev_function": "DMFC-F", "jev_arguments": "DMFC-A", "jev_joint": "DMFC",
                 "jev_joint_latest": "DMFC-L"}


def no_call(calls):
    return isinstance(calls, list) and not calls


def main(run: Path) -> None:
    summary = json.loads((run / "summary.json").read_text())
    rows = defaultdict(dict)
    with (run / "records.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["method"] in ("direct", "jev_joint"):
                rows[record["method"]][record["id"]] = record
    direct, joint = rows["direct"], rows["jev_joint"]
    if set(direct) != set(joint) or not joint:
        raise SystemExit("The run needs Direct and DMFC records for the same examples.")
    scored = [i for i, r in joint.items() if r["category"] != "live_relevance"]
    accuracy = {m: sum(rows[m][i]["score"]["correct"] for i in scored) / len(scored) for m in ("direct", "jev_joint")}
    fixes = sum(1 for i in scored if not joint[i]["trace"][0]["score"]["correct"] and joint[i]["score"]["correct"])
    breaks = sum(1 for i in scored if joint[i]["trace"][0]["score"]["correct"] and not joint[i]["score"]["correct"])
    methods, paired = summary["methods"], summary["paired_vs_direct"]["jev_joint"]
    print(f"accuracy {100 * accuracy['direct']:.1f} -> {100 * accuracy['jev_joint']:.1f}; paired difference "
          f"{100 * paired['delta']:.1f} [{100 * paired['ci95'][0]:.1f}, {100 * paired['ci95'][1]:.1f}] "
          f"over {paired['n_paired_ids']} examples")
    print(f"fixes {fixes}, regressions {breaks}; mean repairs {methods['jev_joint']['mean_repairs']:.2f}; "
          f"exhausted {100 * methods['jev_joint']['exhaustion_rate']:.1f}%; API seconds "
          f"{methods['direct']['mean_latency_s']:.2f} -> {methods['jev_joint']['mean_latency_s']:.2f}; USD per 1k "
          f"{1000 * methods['direct']['mean_cost_usd']:.3f} -> {1000 * methods['jev_joint']['mean_cost_usd']:.3f}")
    final = methods["jev_joint"]
    if final.get("coverage") is not None:
        print(f"final proposals passing all checks: {100 * final['coverage']:.1f}% of examples, "
              f"{100 * final['accepted_precision']:.1f}% correct")
    for category in ("irrelevance", "live_irrelevance"):
        ids = [i for i, r in joint.items() if r["category"] == category]
        first = {i: joint[i]["trace"][0] for i in ids}
        calls = [i for i in ids if not no_call(first[i]["proposal"]["calls"])]
        nones = [i for i in ids if no_call(first[i]["proposal"]["calls"])]
        rejected = [i for i in calls if not first[i]["gate_pass"]]
        withheld = [i for i in rejected if no_call(joint[i]["calls"])]
        lost = [i for i in nones if not first[i]["gate_pass"] and not no_call(joint[i]["calls"])]

        def median(group, first=first):
            values = [first[i]["verification"]["function_probability"] for i in group]
            return statistics.median(values) if values else None

        correct = sum(joint[i]["score"]["correct"] for i in ids)
        print(f"{category}: {len(ids)} examples; Direct calls {sum(not no_call(direct[i]['calls']) for i in ids)}; "
              f"median Jev function probability {median(calls)} for initial calls, {median(nones)} for "
              f"{len(nones)} initial no-calls; rejected calls {len(rejected)}, withheld after feedback "
              f"{len(withheld)}, accepted calls {len(calls) - len(rejected)}, kept calling "
              f"{len(rejected) - len(withheld)}; no-calls turned into calls after a rejection {len(lost)}; "
              f"DMFC errors {len(ids) - correct}")
    reference = [i for i, r in joint.items() if r["category"] in REFERENCE]
    wrong = [i for i in reference if not joint[i]["trace"][0]["score"]["correct"]]
    print(f"reference-call categories: {len(wrong)} wrong initial proposals, "
          f"{sum(not joint[i]['trace'][0]['gate_pass'] for i in wrong)} rejected, "
          f"{sum(joint[i]['score']['correct'] for i in wrong)} fixed, "
          f"{sum(joint[i]['trace'][0]['score']['correct'] and not joint[i]['score']['correct'] for i in reference)} broken")
    for method, value in summary["paired_vs_direct"].items():
        print(f"{DISPLAY_NAMES.get(method, method)} minus Direct: {100 * value['delta']:.1f} [{100 * value['ci95'][0]:.1f}, "
              f"{100 * value['ci95'][1]:.1f}]")
    history = summary.get("history_ablation")
    if history and history.get("delta") is not None:
        print(f"DMFC minus DMFC-L: {100 * history['delta']:.2f} [{100 * history['ci95'][0]:.2f}, "
              f"{100 * history['ci95'][1]:.2f}]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
