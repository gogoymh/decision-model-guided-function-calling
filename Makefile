.PHONY: setup prepare check-config test demo smoke table1 table2 table3 compare r3 irrelevance-check report paper verify

WORKERS ?= 4
MODEL ?= gpt-5-nano
RUN ?= results/table2_$(MODEL)_r3
CHECK_RUNS ?= 1
COMPARE_MODEL ?= gpt-5.4-mini
COMPARE ?= results/table1_$(COMPARE_MODEL)_r3
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

# Direct with a larger generator (default gpt-5.4-mini; MODEL=... for another deployment) on all
# examples, without Jev: the source of the gpt-5.4-mini columns that make r3 completes.
compare:
	MODEL="$(if $(filter command line,$(origin MODEL)),$(MODEL),$(COMPARE_MODEL))" WORKERS="$(WORKERS)" bash scripts/run_compare.sh

# The paper's repair limit r=3: raise the Table 2 run to r=3 (JBFC from the Figure 2 extension), add
# JBFC (r=3) to the gpt-5.4-mini Direct run, then fill Tables 1 and 2. Reuses every recorded attempt.
r3:
	WORKERS="$(WORKERS)" bash scripts/run_r3.sh

# Direct only on the 240 Irrelevance examples; compares with the Table 1 run. EFFORT=low etc. is optional.
irrelevance-check:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" REPLICATES="$(CHECK_RUNS)" EFFORT="$(EFFORT)" bash scripts/run_irrelevance_check.sh

# Refill Tables 1 and 2 from the r=3 runs (make r3); the gpt-5.4-mini columns stay empty until they exist.
report:
	uv run python scripts/paper_tables.py "$(RUN)" --compare "$(COMPARE)" --model "$(COMPARE_MODEL)" --paper paper/main.tex > /dev/null

paper:
	python3 scripts/build_paper.py

verify:
	uv run python scripts/verify_paper.py paper/main.pdf
