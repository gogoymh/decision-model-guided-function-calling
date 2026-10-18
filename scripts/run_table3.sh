#!/usr/bin/env bash
# Table 3: JBFC with repair limits 1, 2, 3, 5 and 7. Continues the r=2 JBFC trajectories of the
# Table 2 run up to r=7 (only rows that reached r=2 make new requests), then cuts them at each limit.
# Needs the finished Table 2 run and the same code and settings that produced it.
# Usage: make table3 [MODEL=gpt-5-nano] [WORKERS=8] [SOURCE=results/table2_gpt-5-nano_r2]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-gpt-5-nano}"
WORKERS="${WORKERS:-4}"
SOURCE="${SOURCE:-results/table2_${MODEL}_r2}"
OUT="${OUT:-results/table3_${MODEL}_r7}"
mkdir -p "$OUT"

export LLM_API_STYLE=azure_chat_completions
export LLM_MODEL="$MODEL"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
case "$MODEL" in  # USD per million tokens; must equal the prices of the source run.
  gpt-5-nano*) export LLM_INPUT_PRICE=0.05 LLM_CACHED_INPUT_PRICE=0.005 LLM_OUTPUT_PRICE=0.40 ;;
  gpt-5.4-mini*) export LLM_INPUT_PRICE=0.75 LLM_CACHED_INPUT_PRICE=0.075 LLM_OUTPUT_PRICE=4.50 ;;
esac

stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date '+%H:%M:%S')" "$line"; done | tee -a "$OUT/run.log"; }
note() { echo "== $*" | stamp; }

note "1/3 preflight: one generator call and one Jev call (deployment=$MODEL)"
uv run python scripts/preflight.py "$OUT/preflight.env" 2>&1 | stamp
source "$OUT/preflight.env"
note "settings: deployment=$LLM_MODEL effort='${LLM_REASONING_EFFORT}' workers=$WORKERS source=$SOURCE"

note "2/3 extend JBFC from r=2 to r=7"
uv run python scripts/extend_run.py "$SOURCE" --max-repairs 7 --workers "$WORKERS" --output "$OUT" --resume 2>&1 | stamp

note "3/3 cut at r=1,2,3,5,7 and fill Table 3 in paper/main.tex"
uv run python scripts/table3.py "$OUT" --paper paper/main.tex > "$OUT/table3.json"
note "done: $OUT/table3.json; Table 3 is filled in paper/main.tex (rebuild the PDF next)"
