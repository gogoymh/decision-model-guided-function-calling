#!/usr/bin/env bash
# Table 3: JBFC with repair limits 1, 2, 3, 5, 7 and 10. Continues JBFC trajectories that reached their
# limit up to r=10, then cuts them at each limit. Only rows that reached the source limit make new requests.
# The source is the r=7 extension when it exists, so the rows up to r=7 stay exactly as before; otherwise
# the Table 2 run (r=2). Needs the same code and settings that produced the Table 2 run.
# Usage: make table3 [MODEL=gpt-5-nano] [WORKERS=8] [SOURCE=results/table3_gpt-5-nano_r7]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-gpt-5-nano}"
WORKERS="${WORKERS:-4}"
MAX_R=10
if [ -z "${SOURCE:-}" ]; then
  if [ -f "results/table3_${MODEL}_r7/summary.json" ]; then SOURCE="results/table3_${MODEL}_r7"
  else SOURCE="results/table2_${MODEL}_r2"; fi
fi
OUT="${OUT:-results/table3_${MODEL}_r${MAX_R}}"
mkdir -p "$OUT"

# Jev goes directly to TypeSafe with TYPESAFE_API_KEY (the gateway keeps serving the generator);
# JEV_GATEWAY=apim routes Jev through the gateway as in the Table 2 run. Recorded in the manifest.
export JEV_GATEWAY="${JEV_GATEWAY:-direct}"
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
note "settings: deployment=$LLM_MODEL effort='${LLM_REASONING_EFFORT}' jev=$JEV_GATEWAY workers=$WORKERS source=$SOURCE"

note "2/3 extend JBFC from $SOURCE to r=$MAX_R"
uv run python scripts/extend_run.py "$SOURCE" --max-repairs "$MAX_R" --workers "$WORKERS" --output "$OUT" --resume 2>&1 | stamp

note "3/3 cut at r=1,2,3,5,7,10 and fill Table 3 in paper/main.tex"
uv run python scripts/table3.py "$OUT" --paper paper/main.tex > "$OUT/table3.json"
note "done: $OUT/table3.json; Table 3 is filled in paper/main.tex (rebuild the PDF next)"
