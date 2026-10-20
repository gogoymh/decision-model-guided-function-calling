#!/usr/bin/env bash
# DMFC again with another Jev threshold (default 0.7, both checks; user request, 2026-10-18), on each of Table 1's
# three runs per generator. Every rerun keeps its run's first proposals (Direct is copied, not requested again)
# and only DMFC runs (scripts/rethreshold.py), so Direct, DMFC at 0.8 and DMFC at THRESHOLD are paired per
# example. Code, data, generator settings, prices and r=3 are those of the run; Jev answers through the gateway
# when it can, otherwise directly with TYPESAFE_API_KEY (same model), as recorded in each manifest. The two
# generators run at the same time (PARALLEL=0: one after the other); running the script again resumes from the
# completed records. Exploratory: the paper is not changed.
# Results: results/jbfc_t<THRESHOLD>_<model>_run<k>; comparison: results/threshold_<THRESHOLD>.json; the table
# of every threshold run so far: docs/threshold.md (scripts/threshold_table.py).
# Usage: make threshold [THRESHOLD=0.7] [WORKERS=8] [PARALLEL=0] [MODELS="gpt-5-nano gpt-5.4-mini"] [RUNS="1 2 3"]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
THRESHOLD="${THRESHOLD:-0.7}"
WORKERS="${WORKERS:-8}"
MODELS="${MODELS:-gpt-5-nano gpt-5.4-mini}"
RUNS="${RUNS:-1 2 3}"
PARALLEL="${PARALLEL:-1}"
TAG="t$(uv run python -c "import sys; print(f'{float(sys.argv[1]):g}')" "$THRESHOLD")"  # 0.70 -> t0.7
LOG="results/threshold_${TAG}.log"
mkdir -p results

stamp() {  # timestamp and tag every line, and append it to a log
  local tag="$1" log="$2" line
  while IFS= read -r line; do printf '%s [%s] %s\n' "$(date '+%H:%M:%S')" "$tag" "$line"; done | tee -a "$log"
}
note() { local tag="$1" log="$2"; shift 2; echo "== $*" | stamp "$tag" "$log"; }

use_model() {  # Azure deployment and its USD prices per million tokens (as in Table 1's runs)
  export LLM_API_STYLE=azure_chat_completions LLM_MODEL="$1"
  export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
  case "$1" in
    gpt-5-nano*) export LLM_INPUT_PRICE=0.05 LLM_CACHED_INPUT_PRICE=0.005 LLM_OUTPUT_PRICE=0.40 ;;
    gpt-5.4-mini*) export LLM_INPUT_PRICE=0.75 LLM_CACHED_INPUT_PRICE=0.075 LLM_OUTPUT_PRICE=4.50 ;;
    *) echo "unknown generator $1: Table 1 has gpt-5-nano and gpt-5.4-mini" >&2; return 1 ;;
  esac
}

source_run() {  # Table 1's run k of a generator (run 1 of gpt-5-nano is the seven-method run)
  case "$1:$2" in
    gpt-5-nano:1) echo "results/table2_gpt-5-nano_r3" ;;
    gpt-5.4-mini:1) echo "results/table1_gpt-5.4-mini_r3" ;;
    *) echo "results/table1_${1}_r3_run${2}" ;;
  esac
}

run_model() {  # reruns DMFC on $RUNS of one generator
  local model="$1" log="results/threshold_${TAG}_${1}.log" env_file="results/threshold_${TAG}_${1}_preflight.env"
  local route run source out
  use_model "$model"
  note "$model" "$log" "preflight $model: one generator call and one Jev call (reasoning effort none, as in Table 1)"
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
    source="$(source_run "$model" "$run")"
    out="results/jbfc_${TAG}_${model}_run${run}"
    note "$model" "$log" "run $run: DMFC with threshold $THRESHOLD from the first proposals of $source -> $out"
    uv run python scripts/rethreshold.py "$source" --threshold "$THRESHOLD" --output "$out" \
      --workers "$WORKERS" --resume 2>&1 | stamp "$model" "$log"
  done
  note "$model" "$log" "done: runs $RUNS of $model"
}

note all "$LOG" "DMFC with threshold $THRESHOLD, runs $RUNS for: $MODELS (workers=$WORKERS per generator, parallel=$PARALLEL)"
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
  note all "$LOG" "stopped: see the lines above (and results/threshold_${TAG}_<model>.log); make threshold resumes from here"
  exit 1
fi

note all "$LOG" "comparison with Direct and DMFC at the runs' threshold (scripts/threshold_compare.py)"
uv run python scripts/threshold_compare.py --threshold "$THRESHOLD" --json "results/threshold_${TAG#t}.json" \
  2>&1 >/dev/null | stamp all "$LOG"
note all "$LOG" "table of every threshold run so far: docs/threshold.md (scripts/threshold_table.py)"
uv run python scripts/threshold_table.py 2>&1 | stamp all "$LOG"
note all "$LOG" "done: results/threshold_${TAG#t}.json and docs/threshold.md (the paper is unchanged)"
