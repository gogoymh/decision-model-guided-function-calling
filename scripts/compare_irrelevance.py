"""Compare a Direct re-run on the Irrelevance category with an earlier run (Table 1 by default).

Usage: python scripts/compare_irrelevance.py NEW_RUN [REFERENCE_RUN] [--show N]

Irrelevance examples offer functions that cannot serve the request, so the correct
answer is no call. The report shows the no-call rate per run, how often both runs
made the same decision per example, which functions were called, and a few calls.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(run_dir, category="irrelevance", method="direct"):
    path = Path(run_dir) / "records.jsonl"
    if not path.exists():
        return {}
    runs = defaultdict(dict)
    for line in path.open():
        row = json.loads(line)
        if row["category"] == category and row["method"] == method:
            runs[row["replicate"]][row["id"]] = row
    return runs


def decision(row):
    calls = row["calls"]
    if not isinstance(calls, list):
        return "unreadable"
    return "call" if calls else "no call"


def settings(run_dir):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    return (f"model={manifest.get('llm_model')} effort={manifest.get('reasoning_effort')!r} "
            f"max_output={manifest.get('max_completion_tokens')} runs={manifest.get('replicates')}")


def rate(rows):
    right = sum(bool(r["score"]["correct"]) for r in rows.values())
    return f"{right}/{len(rows)} = {100 * right / len(rows):.1f}%" if rows else "no records"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_run")
    parser.add_argument("reference_run", nargs="?", default=str(ROOT / "results/table1_gpt-5-nano_r2"))
    parser.add_argument("--show", type=int, default=5, help="number of example calls to print")
    args = parser.parse_args()

    new = load(args.new_run)
    if not new:
        raise SystemExit(f"No Direct/Irrelevance records in {args.new_run}")
    print(f"new run: {args.new_run} ({settings(args.new_run)})")
    for replicate, rows in sorted(new.items()):
        print(f"  run {replicate}: correct (no call) {rate(rows)}; "
              + ", ".join(f"{k} {v}" for k, v in sorted(Counter(map(decision, rows.values())).items())))

    reference = load(args.reference_run)
    first = new[min(new)]
    if reference:
        ref = reference[min(reference)]
        print(f"reference: {args.reference_run} ({settings(args.reference_run)})")
        print(f"  correct (no call) {rate(ref)}")
        common = sorted(set(ref) & set(first))
        same = sum(decision(ref[i]) == decision(first[i]) for i in common)
        moves = Counter((decision(ref[i]), decision(first[i])) for i in common if decision(ref[i]) != decision(first[i]))
        print(f"  same decision on {same}/{len(common)} examples"
              + "".join(f"; {a} -> {b}: {n}" for (a, b), n in sorted(moves.items())))
    if len(new) > 1:
        calls_per_example = Counter(sum(decision(rows[i]) == "call" for rows in new.values() if i in rows) for i in first)
        print("examples by number of runs with a call:", dict(sorted(calls_per_example.items())))

    called = [row for row in first.values() if decision(row) == "call"]
    print("most called functions:", Counter(call["name"] for row in called for call in row["calls"]).most_common(5))
    examples = {}
    data = ROOT / "data/bfcl/examples.jsonl"
    if data.exists():
        for line in data.open():
            row = json.loads(line)
            if row["category"] == "irrelevance":
                examples[row["id"]] = row
    for row in sorted(called, key=lambda r: r["id"])[:max(0, args.show)]:
        example = examples.get(row["id"], {})
        request = next((m["content"] for m in reversed(example.get("messages", [])) if m.get("role") == "user"), "")
        tools = [tool["function"]["name"] for tool in example.get("tools", [])]
        print(f"- {row['id']}: {str(request)[:110]!r}")
        print(f"  offered {tools}; called {json.dumps(row['calls'], ensure_ascii=False)[:160]}")


if __name__ == "__main__":
    main()
