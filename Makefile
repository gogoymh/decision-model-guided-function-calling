.PHONY: setup prepare check-config test demo smoke table1 table2 table3 irrelevance-check report paper verify

WORKERS ?= 4
MODEL ?= gpt-5-nano
RUN ?= results/table2_$(MODEL)_r2
CHECK_RUNS ?= 1
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

# Table 3: JBFC with r=1,2,3,5,7, continuing the Table 2 run's JBFC rows from r=2 to r=7.
table3:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" bash scripts/run_table3.sh

# Direct only on the 240 Irrelevance examples; compares with the Table 1 run. EFFORT=low etc. is optional.
irrelevance-check:
	MODEL="$(MODEL)" WORKERS="$(WORKERS)" REPLICATES="$(CHECK_RUNS)" EFFORT="$(EFFORT)" bash scripts/run_irrelevance_check.sh

# Refill the paper's tables from a finished run (default: the Table 2 run).
report:
	uv run jev-fc report "$(RUN)" --paper paper/main.tex

paper:
	python3 scripts/build_paper.py

verify:
	uv run python scripts/verify_paper.py paper/main.pdf
