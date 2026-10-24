#!/usr/bin/env bash
# The paper's final DMFC settings (user decision, 2026-10-18): Jev threshold 0.5 for both checks and, when no
# proposal passes within the limit, the first proposal. Table 1 and the rule comparison come from the threshold
# reruns (make threshold THRESHOLD=0.5); this adds what Table 2 and Figure 2 need, on gpt-5-nano's run 1:
#  1-3. DMFC-F, DMFC-A and DMFC-L again with threshold 0.5 from the run's first proposals (scripts/rethreshold.py)
#  4.   DMFC at 0.5 continued from r=3 to r=10 (scripts/extend_run.py)
# Jev is called directly with TYPESAFE_API_KEY, as in the threshold reruns; the generator stays on the gateway.
# Running the script again resumes from the completed records. The paper is filled afterwards (no API calls).
# Usage: make t05 [WORKERS=8]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORKERS="${WORKERS:-8}"
MODEL=gpt-5-nano
SOURCE="results/table2_gpt-5-nano_r3"
JBFC="results/jbfc_t0.5_gpt-5-nano_run1"
LOG="results/t05_run.log"
mkdir -p results

stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date '+%H:%M:%S')" "$line"; done | tee -a "$LOG"; }
note() { echo "== $*" | stamp; }

export LLM_API_STYLE=azure_chat_completions LLM_MODEL="$MODEL" JEV_GATEWAY=direct
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
export LLM_INPUT_PRICE=0.05 LLM_CACHED_INPUT_PRICE=0.005 LLM_OUTPUT_PRICE=0.40
[ -f "$JBFC/summary.json" ] || { note "missing $JBFC: run make threshold THRESHOLD=0.5 first"; exit 1; }

note "preflight $MODEL: one generator call and one Jev call (Jev directly)"
uv run python scripts/preflight.py results/t05_preflight.env none 2>&1 | stamp
# shellcheck disable=SC1091
source results/t05_preflight.env

for method in jev_function jev_arguments jev_joint_latest; do
  note "$method with threshold 0.5 from the first proposals of $SOURCE -> results/${method}_t0.5_${MODEL}_run1"
  uv run python scripts/rethreshold.py "$SOURCE" --method "$method" --threshold 0.5 \
    --output "results/${method}_t0.5_${MODEL}_run1" --workers "$WORKERS" --resume 2>&1 | stamp
done
note "DMGFC at 0.5 continued from r=3 to r=10 -> results/table3_${MODEL}_r10_t0.5"
uv run python scripts/extend_run.py "$JBFC" --max-repairs 10 --output "results/table3_${MODEL}_r10_t0.5" \
  --workers "$WORKERS" --resume 2>&1 | stamp
note "done: the runs for Table 2 and Figure 2 at threshold 0.5 are complete (the paper is filled next)"
