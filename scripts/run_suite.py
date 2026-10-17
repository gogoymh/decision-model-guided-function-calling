"""Choose a repair budget on development data, then run the paired test suite.

Development selection always happens first. Optional test-prefix reports only
describe smaller stopping caps; they never choose the budget. No keys are
embedded here. --resume requires matching code, data, models and configuration.
Alternative configurations save reports without replacing manuscript tables.
The development search uses JBFC only. Its chosen limit is shared by all test
repair methods. All seven methods share their initial LLM proposal. JBFC-L runs
its own repairs with only the latest feedback; it is never replayed from JBFC.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from jev_fc.pipeline import METHODS
from jev_fc.reporting import derive_policy_run, paper_protocol_matches
from jev_fc.tuning import BUDGET_SELECTION_METHOD, DEFAULT_REPAIR_GRID, select_repair_budget


def parse_repair_grid(value):
    try:
        parts = value.split(",")
        grid = tuple(int(part.strip()) for part in parts)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Repair grid must be comma-separated nonnegative integers.") from None
    if not grid or any(number < 0 for number in grid) or len(set(grid)) != len(grid):
        raise argparse.ArgumentTypeError("Repair grid must contain distinct nonnegative integers.")
    return tuple(sorted(grid))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--repair-grid", type=parse_repair_grid, default=DEFAULT_REPAIR_GRID,
                        help="Development candidates; default 0,1,2,3,4,5")
    parser.add_argument("--results-dir", default="results/tuned", help="Root for development, selection and test outputs")
    parser.add_argument("--dev-only", action="store_true", help="Run development data and save the selection; do not run test data")
    parser.add_argument("--main-only", action="store_true", help="Skip optional lower-cap test diagnostics; development selection remains mandatory")
    parser.add_argument("--include-abstain", action="store_true", help="Also report abstention at the selected test repair budget")
    parser.add_argument("--skip-paper", action="store_true", help="Save experiment reports without updating the manuscript")
    args = parser.parse_args(argv)
    if args.replicates < 1 or args.workers < 1:
        parser.error("replicates and workers must be positive integers")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    root = Path(__file__).resolve().parents[1]
    results = Path(args.results_dir)
    if not results.is_absolute():
        results = root / results
    development = results / "dev_budget"
    selection_path = results / "repair_budget_selection.json"
    test = results / "test_selected"
    common = [sys.executable, "-m", "jev_fc.cli", "run", "--resume",
              "--replicates", str(args.replicates), "--workers", str(args.workers),
              "--threshold", str(args.threshold)]
    subprocess.run(common + ["--split", "dev", "--methods", BUDGET_SELECTION_METHOD,
                             "--output", str(development), "--max-repairs", str(max(args.repair_grid))],
                   cwd=root, check=True)
    selection = select_repair_budget(development, grid=args.repair_grid)
    if selection.get("selection_method") != BUDGET_SELECTION_METHOD:
        raise ValueError("The repair budget must be selected from the JBFC development method.")
    selected_budget = selection["selected_max_repairs"]
    if isinstance(selected_budget, bool) or not isinstance(selected_budget, int) or selected_budget not in args.repair_grid:
        raise ValueError("The development selection did not return a budget in the requested grid.")
    results.mkdir(parents=True, exist_ok=True)
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("The saved repair-budget selection differs; choose a new --results-dir.")
    else:
        selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Development selection chose r={selected_budget} using JBFC; saved {selection_path}.")
    if args.dev_only:
        print("Development run and selection complete. Test data were not used.")
        return 0

    print(f"Test evaluation compares {len(METHODS)} methods with the shared JBFC-selected repair limit.")
    subprocess.run(common + ["--split", "test", "--methods", ",".join(METHODS),
                             "--output", str(test), "--max-repairs", str(selected_budget),
                             "--repair-budget-selection", str(selection_path)], cwd=root, check=True)
    if not args.main_only:
        # These stopping-cap diagnostics use only recorded cumulative JBFC runs.
        for repairs in args.repair_grid:
            if repairs < selected_budget:
                derive_policy_run(test, results / "test_prefix_diagnostics" / f"budget_{repairs}",
                                  max_repairs=repairs)
    if args.include_abstain:
        derive_policy_run(test, results / "test_abstain_selected", max_repairs=selected_budget,
                          exhaustion_policy="abstain")
    manifest = json.loads((test / "manifest.json").read_text())
    update_manuscript = not args.skip_paper and paper_protocol_matches(manifest)
    report_command = [sys.executable, "-m", "jev_fc.cli", "report", str(test)]
    if update_manuscript:
        # The report command independently checks complete canonical data,
        # provenance and the development artifact before changing the paper.
        report_command.extend(["--paper", "paper/main.tex"])
    subprocess.run(report_command, cwd=root, check=True)
    if update_manuscript:
        print("Experiment suite complete. Review summaries and update the manuscript's empirical discussion before submission.")
    else:
        reason = "--skip-paper was requested" if args.skip_paper else "the run differs from the manuscript's declared protocol"
        print(f"Experiment suite complete; reports saved. Paper update skipped because {reason}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
