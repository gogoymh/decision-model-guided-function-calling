.PHONY: setup prepare check-config test demo smoke table1 table2 table3 compare r3 repeat threshold threshold-table t05 final irrelevance-check report paper verify

WORKERS ?= 4
MODEL ?= gpt-5-nano
CHECK_RUNS ?= 1
COMPARE_MODEL ?= gpt-5.4-mini
EFFORT ?=

setup:
	uv sync --extra dev
	@test -f .env || cp .env.example .env

prepare:
	uv run jev-fc prepare

check-config:
	uv run jev-fc check-config

test:
	uv run pytest -q
	uv run ruff check src tests scripts

demo:
	uv run jev-fc demo

smoke:
	uv run jev-fc run --split all --limit 8 --replicates 1 --max-repairs 2 --output results/smoke --resume

# Paper protocol: all 3,051 examples, repair limit r=3, no development split. The runs start at threshold 0.8
# and r=2: make table2, table3 and compare, then make r3 raises them to r=3 and make repeat adds runs 2 and 3.
# The final settings (threshold 0.5, the first proposal when none passes; user decision 2026-10-18) rerun the
# Jev methods from the same first proposals (make threshold THRESHOLD=0.5, make t05); make final derives the
# paper's runs and fills its tables. make final updates the measured experiment section and plot.
# Direct and DMFC only, r=2 (6,102 records): a cheaper standalone run, not used by the tables.
table1:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" bash scripts/run_table1.sh

# Source run of Tables 1 and 2: all seven methods at r=2 (21,357 records).
table2:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" bash scripts/run_table2.sh

# Figure 2 (formerly Table 3): DMFC with r=1,2,3,5,7,10, continuing the DMFC rows of the r=7 extension
# (or the Table 2 run) to r=10.
table3:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" bash scripts/run_table3.sh

# Direct with a larger generator (default gpt-5.4-mini; MODEL=... for another deployment) on all
# examples, without Jev: the source of the gpt-5.4-mini columns that make r3 completes.
compare:
	MODEL="$(if $(filter command line,$(origin MODEL)),$(MODEL),$(COMPARE_MODEL))" WORKERS="$(WORKERS)" bash scripts/run_compare.sh

# The paper's repair limit r=3: raise the Table 2 run to r=3 (DMFC from the Figure 2 extension) and add
# DMFC (r=3) to the gpt-5.4-mini Direct run (threshold 0.8). Reuses every recorded attempt.
r3:
	WORKERS="$(WORKERS)" bash scripts/run_r3.sh

# Table 1 over three runs (user request, 2026-10-18): runs 2 and 3 of Direct and DMFC (r=3, threshold 0.8)
# for both generators, at the same time (PARALLEL=0: one after the other). Running it again resumes from the
# completed records.
repeat:
	WORKERS="$(WORKERS)" PARALLEL="$(PARALLEL)" MODELS="$(MODELS)" RUNS="$(RUNS)" bash scripts/run_repeat.sh

# DMFC again with another Jev threshold (THRESHOLD, default 0.7) on Table 1's three runs per generator, from
# the same first proposals (user request, 2026-10-18). Writes results/threshold_<THRESHOLD>.json and
# docs/threshold.md. The paper uses THRESHOLD=0.5 through make final.
threshold:
	THRESHOLD="$(THRESHOLD)" WORKERS="$(WORKERS)" PARALLEL="$(PARALLEL)" MODELS="$(MODELS)" RUNS="$(RUNS)" bash scripts/run_threshold.sh

# Table of every threshold run on disk (docs/threshold.md); make threshold refreshes it at the end.
threshold-table:
	uv run python scripts/threshold_table.py

# The paper's final DMFC settings (threshold 0.5, first proposal when nothing passes): the extra gpt-5-nano runs
# Table 2 and Figure 2 need (DMFC-F, DMFC-A and DMFC-L at 0.5; DMFC at 0.5 continued to r=10).
t05:
	WORKERS="$(WORKERS)" bash scripts/run_t05.sh

# Direct only on the 240 Irrelevance examples; compares with the Table 1 run. EFFORT=low etc. is optional.
irrelevance-check:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" REPLICATES="$(CHECK_RUNS)" EFFORT="$(EFFORT)" bash scripts/run_irrelevance_check.sh

# The paper at its final settings: derive the final runs from the recorded ones (threshold 0.5 reruns, the first
# candidate when none passes; no API calls) into results/final, then update short-format Tables 1-2 and Figure 2. Numbers:
# results/final/final.json. Rebuild the PDF next (make paper).
final:
	uv run python scripts/final_tables.py --paper paper/main.tex > /dev/null

report: final

paper:
	python3 scripts/build_paper.py

verify:
	uv run python scripts/verify_paper.py paper/main.pdf --format acm-review
