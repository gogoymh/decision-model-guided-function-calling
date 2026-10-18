#!/usr/bin/env bash
# Re-run Direct on the 240 Irrelevance examples only, as a check of Table 1's Irrelevance/Direct
# cell, then compare the decisions with the Table 1 run. The result never fills the paper.
# Usage: make irrelevance-check [MODEL=gpt-5-nano] [WORKERS=8] [REPLICATES=1] [EFFORT=none]
#   EFFORT empty: the same reasoning-effort check as Table 1 (none, minimal, low, model default).
#   EFFORT=low (or minimal, medium, high, default) tests another reasoning effort.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-gpt-5-nano}"
WORKERS="${WORKERS:-8}"
REPLICATES="${REPLICATES:-1}"
EFFORT="${EFFORT:-}"
REF="${REF:-results/table1_${MODEL}_r2}"
OUT="${OUT:-results/irrelevance_direct_${MODEL}_${EFFORT:-same}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

export LLM_API_STYLE=azure_chat_completions
export LLM_MODEL="$MODEL"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
case "$MODEL" in  # USD per million tokens; for other deployments set LLM_*_PRICE or cost stays unknown.
  gpt-5-nano*) export LLM_INPUT_PRICE=0.05 LLM_CACHED_INPUT_PRICE=0.005 LLM_OUTPUT_PRICE=0.40 ;;
  gpt-5.4-mini*) export LLM_INPUT_PRICE=0.75 LLM_CACHED_INPUT_PRICE=0.075 LLM_OUTPUT_PRICE=4.50 ;;
esac

stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date '+%H:%M:%S')" "$line"; done | tee -a "$OUT/run.log"; }
note() { echo "== $*" | stamp; }

note "1/3 preflight (deployment=$MODEL${EFFORT:+, effort=$EFFORT})"
uv run python scripts/preflight.py "$OUT/preflight.env" ${EFFORT:+"$EFFORT"} 2>&1 | stamp
source "$OUT/preflight.env"
note "settings: deployment=$LLM_MODEL effort='${LLM_REASONING_EFFORT}' runs=$REPLICATES workers=$WORKERS"

note "2/3 run: Direct on the 240 Irrelevance examples"
uv run jev-fc run --split all --categories irrelevance --methods direct --max-repairs 0 \
  --replicates "$REPLICATES" --workers "$WORKERS" --output "$OUT" --resume 2>&1 | stamp

note "3/3 compare with $REF"
uv run python scripts/compare_irrelevance.py "$OUT" "$REF" 2>&1 | stamp
note "done: $OUT"
