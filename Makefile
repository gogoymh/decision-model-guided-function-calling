.PHONY: setup prepare check-config test demo smoke table1 table2 table3 compare irrelevance-check report paper verify

WORKERS ?= 4
MODEL ?= gpt-5-nano
RUN ?= results/table2_$(MODEL)_r2
CHECK_RUNS ?= 1
COMPARE_MODEL ?= gpt-5.4-mini
COMPARE ?= results/direct_$(COMPARE_MODEL)
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

# Paper protocol: all 3,051 examples, one run, repair limit r=2 (no development split or tuning).
# Table 1: Direct and JBFC (6,102 records).
table1:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" bash scripts/run_table1.sh

# Table 2: all seven methods (21,357 records); its report fills Tables 1 and 2 from the same run.
table2:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" bash scripts/run_table2.sh

# Figure 2 (formerly Table 3): JBFC with r=1,2,3,5,7,10, continuing the JBFC rows of the r=7 extension
# (or the Table 2 run) to r=10.
table3:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" bash scripts/run_table3.sh

# Table 1, third column: Direct with a larger generator (default gpt-5.4-mini; MODEL=... for another
# deployment) on all examples, without Jev. Redraws Table 1 next to the Table 2 run's gpt-5-nano columns.
compare:
	MODEL="$(if $(filter command line,$(origin MODEL)),$(MODEL),$(COMPARE_MODEL))" WORKERS="$(WORKERS)" bash scripts/run_compare.sh

# Direct only on the 240 Irrelevance examples; compares with the Table 1 run. EFFORT=low etc. is optional.
irrelevance-check:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" REPLICATES="$(CHECK_RUNS)" EFFORT="$(EFFORT)" bash scripts/run_irrelevance_check.sh

# Refill the paper's tables from a finished run (default: the Table 2 run), then Table 1's third
# column from the comparison run (empty until make compare has run).
report:
	uv run jev-fc report "$(RUN)" --paper paper/main.tex
	uv run python scripts/table1_compare.py "$(RUN)" --compare "$(COMPARE)" --model "$(COMPARE_MODEL)" --paper paper/main.tex > /dev/null

paper:
	python3 scripts/build_paper.py

verify:
	uv run python scripts/verify_paper.py paper/main.pdf
