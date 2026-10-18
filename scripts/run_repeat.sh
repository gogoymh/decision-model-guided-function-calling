#!/usr/bin/env bash
# Table 1 over three runs (user request, 2026-10-18): Direct and JBFC (r=3) for gpt-5-nano and gpt-5.4-mini.
# Run 1 of each generator is its r=3 run from make r3. This script adds runs 2 and 3. Each is a Direct run on
# all 3,051 examples with new first proposals (generator only), then JBFC (r=3) added from those proposals with
# scripts/raise_limit.py --add jev_joint, the way run 1 of gpt-5.4-mini was made. Code, data, interface,
# settings and prices are those of run 1; only the generator's samples differ. Jev answers through the gateway
# when it can, otherwise directly with TYPESAFE_API_KEY (same model); each manifest records the route.
# The two generators run at the same time (PARALLEL=0: one after the other). Running the script again resumes
# from the completed records. Once every run is complete, scripts/table1_repeat.py reports the mean of the three
# runs at threshold 0.8 (results/table1_repeat.json); the paper's Table 1 comes from make final.
# Usage: make repeat [WORKERS=8] [PARALLEL=0] [MODELS="gpt-5-nano gpt-5.4-mini"] [RUNS="2 3"]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORKERS="${WORKERS:-8}"
MODELS="${MODELS:-gpt-5-nano gpt-5.4-mini}"
RUNS="${RUNS:-2 3}"
PARALLEL="${PARALLEL:-1}"
LIMIT=3
LOG="results/repeat.log"
mkdir -p results

stamp() {  # timestamp and tag every line, and append it to a log
  local tag="$1" log="$2" line
  while IFS= read -r line; do printf '%s [%s] %s\n' "$(date '+%H:%M:%S')" "$tag" "$line"; done | tee -a "$log"
}
note() { local tag="$1" log="$2"; shift 2; echo "== $*" | stamp "$tag" "$log"; }

use_model() {  # Azure deployment and its USD prices per million tokens (as in run 1)
  export LLM_API_STYLE=azure_chat_completions LLM_MODEL="$1"
  export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
  case "$1" in
    gpt-5-nano*) export LLM_INPUT_PRICE=0.05 LLM_CACHED_INPUT_PRICE=0.005 LLM_OUTPUT_PRICE=0.40 ;;
    gpt-5.4-mini*) export LLM_INPUT_PRICE=0.75 LLM_CACHED_INPUT_PRICE=0.075 LLM_OUTPUT_PRICE=4.50 ;;
    *) echo "unknown generator $1: Table 1 has gpt-5-nano and gpt-5.4-mini" >&2; return 1 ;;
  esac
}

run_model() {  # runs $RUNS of one generator
  local model="$1" log="results/repeat_${1}.log" env_file="results/repeat_${1}_preflight.env" route run direct out
  use_model "$model"
  note "$model" "$log" "preflight $model: one generator call and one Jev call (reasoning effort none, as in run 1)"
  for route in "${JEV_GATEWAY:-apim}" direct ""; do
    if [ -z "$route" ]; then note "$model" "$log" "Jev did not answer on any route"; return 1; fi
    if JEV_GATEWAY="$route" uv run python scripts/preflight.py "$env_file" none 2>&1 | stamp "$model" "$log"; then
      export JEV_GATEWAY="$route"
      # shellcheck disable=SC1090
      source "$env_file"
      note "$model" "$log" "Jev route: $JEV_GATEWAY"
      break
    fi
    note "$model" "$log" "Jev did not answer through '$route'; trying the next route"
  done
  for run in $RUNS; do
    direct="results/direct_${model}_run${run}"
    out="results/table1_${model}_r${LIMIT}_run${run}"
    note "$model" "$log" "run $run: Direct on all 3,051 examples with new first proposals -> $direct"
    uv run jev-fc run --split all --methods direct --max-repairs 2 --replicates 1 \
      --workers "$WORKERS" --output "$direct" --resume 2>&1 | stamp "$model" "$log"
    note "$model" "$log" "run $run: add JBFC (r=$LIMIT) from those first proposals -> $out"
    uv run python scripts/raise_limit.py "$direct" --max-repairs "$LIMIT" --add jev_joint \
      --output "$out" --workers "$WORKERS" --resume 2>&1 | stamp "$model" "$log"
  done
  note "$model" "$log" "done: runs $RUNS of $model"
}

note all "$LOG" "Table 1 runs $RUNS for: $MODELS (workers=$WORKERS per generator, parallel=$PARALLEL)"
# Each generator runs in a background subshell so that any failing step stops it (errexit stays active there).
failed=0
pids=()
for model in $MODELS; do
  ( run_model "$model" ) &
  pids+=("$!")
  if [ "$PARALLEL" != 1 ]; then wait "$!" || { failed=1; break; }; fi
done
if [ "$PARALLEL" = 1 ]; then
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
fi
if [ "$failed" != 0 ]; then
  note all "$LOG" "stopped: see the lines above (and results/repeat_<model>.log); make repeat resumes from here"
  exit 1
fi

note all "$LOG" "mean of the three runs once all are complete (scripts/table1_repeat.py; the paper is unchanged)"
uv run python scripts/table1_repeat.py --if-complete --json results/table1_repeat.json \
  2>&1 >/dev/null | stamp all "$LOG"
note all "$LOG" "done"
