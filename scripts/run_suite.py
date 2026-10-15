"""Run the documented primary comparison and repair-budget ablations.

No keys are embedded here. Each completed example is saved and --resume reuses
it only when code, data, split, models and configuration hashes still match.
Alternative configurations still finish the suite and save their reports; only
the manuscript's declared protocol is eligible for automatic table insertion.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from jev_fc.reporting import derive_policy_run, paper_protocol_matches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--include-abstain", action="store_true")
    parser.add_argument("--skip-paper", action="store_true", help="Save experiment reports without updating the manuscript")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    common = [sys.executable, "-m", "jev_fc.cli", "run", "--resume", "--split", "test",
              "--replicates", str(args.replicates), "--workers", str(args.workers),
              "--threshold", str(args.threshold)]
    subprocess.run(common + ["--output", "results/main", "--max-repairs", "3"], cwd=root, check=True)
    if not args.main_only:
        for repairs in range(3):
            derive_policy_run(root / "results/main", root / f"results/budget_{repairs}", max_repairs=repairs)
    if args.include_abstain:
        derive_policy_run(root / "results/main", root / "results/abstain", exhaustion_policy="abstain")
    manifest = json.loads((root / "results/main/manifest.json").read_text())
    update_manuscript = not args.skip_paper and paper_protocol_matches(manifest)
    report_command = [sys.executable, "-m", "jev_fc.cli", "report", "results/main"]
    if update_manuscript:
        # update_paper still enforces complete canonical data, split and digest.
        # Matching model settings alone never bypass those stronger guards.
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
