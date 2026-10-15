"""Run the documented primary comparison and repair-budget ablations.

No keys are embedded here. Each completed example is saved and --resume reuses
it only when code, data, split, models and configuration hashes still match.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from jev_fc.reporting import derive_policy_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--include-abstain", action="store_true")
    args = parser.parse_args()
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
    subprocess.run([sys.executable, "-m", "jev_fc.cli", "report", "results/main", "--paper", "paper/main.tex"],
                   cwd=root, check=True)
    print("Experiment suite complete. Review summaries and update the manuscript's empirical discussion before submission.")


if __name__ == "__main__":
    main()
