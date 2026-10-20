#!/usr/bin/env bash
# Direct and DMFC once on all 3,051 prepared examples with repair limit r=2: a cheaper standalone run and the
# reference of make irrelevance-check. The paper's tables come from make table2, table3, compare and r3.
# The generator is an Azure OpenAI deployment behind the APIM host and key in .env.
# Usage: make table1 [MODEL=gpt-5-nano] [WORKERS=4]   (or MODEL=... bash scripts/run_table1.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-gpt-5-nano}"
WORKERS="${WORKERS:-4}"
OUT="${OUT:-results/table1_${MODEL}_r2}"
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

note "2/3 run: Direct and DMFC on all 3,051 examples, one run, r=2"
uv run jev-fc run --split all --methods direct,jev_joint --max-repairs 2 --replicates 1 \
  --workers "$WORKERS" --output "$OUT" --resume 2>&1 | stamp

note "3/3 summarize the run (paper/main.tex is not changed; make r3 fills its tables)"
uv run jev-fc report "$OUT" > "$OUT/report.json"
note "done: $OUT/summary.json"
