#!/usr/bin/env bash
# Table 2: all seven methods once on all 3,051 prepared examples with repair limit r=2
# (3,051 x 7 = 21,357 records). The report fills Tables 1 and 2 from this same run.
# The generator is an Azure OpenAI deployment behind the APIM host and key in .env.
# Usage: make table2 [MODEL=gpt-5-nano] [WORKERS=8]   (or MODEL=... bash scripts/run_table2.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-gpt-5-nano}"
WORKERS="${WORKERS:-4}"
OUT="${OUT:-results/table2_${MODEL}_r2}"
METHODS="direct,schema_only,self_refine,jev_function,jev_arguments,jev_joint,jev_joint_latest"
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

note "1/3 preflight: one generator call and one Jev call (deployment=$MODEL)"
uv run python scripts/preflight.py "$OUT/preflight.env" 2>&1 | stamp
source "$OUT/preflight.env"
note "settings: deployment=$LLM_MODEL effort='${LLM_REASONING_EFFORT}' api-version=$AZURE_OPENAI_API_VERSION workers=$WORKERS"

note "2/3 run: seven methods on all 3,051 examples, one run, r=2"
uv run jev-fc run --split all --methods "$METHODS" --max-repairs 2 --replicates 1 \
  --workers "$WORKERS" --output "$OUT" --resume 2>&1 | stamp

note "3/3 fill Tables 1 and 2 in paper/main.tex from this run"
uv run jev-fc report "$OUT" --paper paper/main.tex > "$OUT/report.json"
note "done: $OUT/summary.json; Tables 1 and 2 are filled in paper/main.tex (rebuild the PDF next)"
