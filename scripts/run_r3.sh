#!/usr/bin/env bash
# Tables 1 and 2 at repair limit r=3 (chosen from Figure 2; user decision 2026-10-18), reusing
# every attempt already recorded:
#  1. gpt-5-nano: raise the Table 2 run (seven methods, r=2) to r=3. JBFC rows are the Figure 2
#     extension cut at r=3 (no new request); Direct stays; every self-refine row and the rows of the
#     other methods that reached r=2 get one more attempt on their saved history.
#  2. gpt-5.4-mini: add JBFC (r=3) to its Direct run, starting from the saved first proposals.
#  3. Fill Tables 1 and 2 in paper/main.tex.
# Jev answers through the gateway when it can, otherwise directly with TYPESAFE_API_KEY (same model);
# each run's manifest records the route. Usage: make r3 [WORKERS=8]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORKERS="${WORKERS:-8}"
LIMIT=3
NANO_SOURCE="${NANO_SOURCE:-results/table2_gpt-5-nano_r2}"
NANO_JBFC="${NANO_JBFC:-results/table3_gpt-5-nano_r10}"
NANO_OUT="${NANO_OUT:-results/table2_gpt-5-nano_r${LIMIT}}"
MINI_SOURCE="${MINI_SOURCE:-results/direct_gpt-5.4-mini}"
MINI_OUT="${MINI_OUT:-results/table1_gpt-5.4-mini_r${LIMIT}}"
LOG="results/r${LIMIT}_run.log"
mkdir -p results

stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date '+%H:%M:%S')" "$line"; done | tee -a "$LOG"; }
note() { echo "== $*" | stamp; }

use_model() {  # Azure deployment and its USD prices per million tokens (as in the source runs)
  export LLM_API_STYLE=azure_chat_completions LLM_MODEL="$1"
  export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
  case "$1" in
    gpt-5-nano*) export LLM_INPUT_PRICE=0.05 LLM_CACHED_INPUT_PRICE=0.005 LLM_OUTPUT_PRICE=0.40 ;;
    gpt-5.4-mini*) export LLM_INPUT_PRICE=0.75 LLM_CACHED_INPUT_PRICE=0.075 LLM_OUTPUT_PRICE=4.50 ;;
  esac
}

preflight() {  # reasoning effort none, as in the source runs; Jev through the gateway, else directly
  local route
  for route in "${JEV_GATEWAY:-apim}" direct; do
    if JEV_GATEWAY="$route" uv run python scripts/preflight.py "$1" none 2>&1 | stamp; then
      export JEV_GATEWAY="$route"
      # shellcheck disable=SC1090
      source "$1"
      note "Jev route: $JEV_GATEWAY"
      return 0
    fi
    note "Jev did not answer through '$route'; trying the next route"
  done
  return 1
}

note "1/5 preflight gpt-5-nano"
use_model gpt-5-nano
preflight "results/r${LIMIT}_nano_preflight.env"
note "2/5 gpt-5-nano: raise $NANO_SOURCE from r=2 to r=$LIMIT (JBFC from $NANO_JBFC)"
uv run python scripts/raise_limit.py "$NANO_SOURCE" --max-repairs "$LIMIT" --reuse "jev_joint=$NANO_JBFC" \
  --output "$NANO_OUT" --workers "$WORKERS" --resume 2>&1 | stamp

note "3/5 preflight gpt-5.4-mini"
use_model gpt-5.4-mini
preflight "results/r${LIMIT}_mini_preflight.env"
note "4/5 gpt-5.4-mini: add JBFC (r=$LIMIT) to $MINI_SOURCE from its saved first proposals"
uv run python scripts/raise_limit.py "$MINI_SOURCE" --max-repairs "$LIMIT" --add jev_joint \
  --output "$MINI_OUT" --workers "$WORKERS" --resume 2>&1 | stamp

note "5/5 fill Tables 1 and 2 in paper/main.tex"
uv run python scripts/paper_tables.py "$NANO_OUT" --compare "$MINI_OUT" --paper paper/main.tex \
  > "results/tables_r${LIMIT}.json"
note "done: Tables 1 and 2 are filled in paper/main.tex (rebuild the PDF next); numbers in results/tables_r${LIMIT}.json"
