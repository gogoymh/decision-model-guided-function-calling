#!/usr/bin/env bash
# Direct with a larger generator on all 3,051 examples, once, with the paper's code, interface and
# settings (Azure Chat Completions, reasoning effort none if the model accepts it, 4,096 output tokens).
# Only the generator is called; Jev is not used. This is the source of Table 1's gpt-5.4-mini columns:
# make r3 then adds JBFC from the saved first proposals.
# Usage: make compare [MODEL=gpt-5.4-mini] [WORKERS=8]
#   Another Azure deployment: MODEL=<deployment>, with LLM_INPUT_PRICE, LLM_CACHED_INPUT_PRICE and
#   LLM_OUTPUT_PRICE (USD per million tokens) set unless listed below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-gpt-5.4-mini}"
WORKERS="${WORKERS:-8}"
OUT="${OUT:-results/direct_${MODEL}}"
mkdir -p "$OUT"

export LLM_API_STYLE=azure_chat_completions
export LLM_MODEL="$MODEL"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
case "$MODEL" in  # USD per million tokens (OpenAI list prices); other deployments need LLM_*_PRICE.
  gpt-5-nano*) export LLM_INPUT_PRICE=0.05 LLM_CACHED_INPUT_PRICE=0.005 LLM_OUTPUT_PRICE=0.40 ;;
  gpt-5.4-mini*) export LLM_INPUT_PRICE=0.75 LLM_CACHED_INPUT_PRICE=0.075 LLM_OUTPUT_PRICE=4.50 ;;
esac

stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date '+%H:%M:%S')" "$line"; done | tee -a "$OUT/run.log"; }
note() { echo "== $*" | stamp; }

note "1/2 preflight: one generator call (deployment=$MODEL; this run does not call Jev)"
uv run python scripts/preflight.py --generator-only "$OUT/preflight.env" 2>&1 | stamp
source "$OUT/preflight.env"
note "settings: deployment=$LLM_MODEL effort='${LLM_REASONING_EFFORT}' api-version=$AZURE_OPENAI_API_VERSION workers=$WORKERS"

note "2/2 run: Direct on all 3,051 examples, one run (r=2 recorded; Direct makes no repairs)"
uv run jev-fc run --split all --methods direct --max-repairs 2 --replicates 1 \
  --workers "$WORKERS" --output "$OUT" --resume 2>&1 | stamp

note "done: $OUT (next: make r3 adds JBFC at r=3 and fills Tables 1 and 2)"
