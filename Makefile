.PHONY: setup prepare check-config test demo smoke tune experiment report paper verify

REPAIR_GRID ?= 0,1,2,3,4,5
RESULTS_DIR ?= results/tuned
REPLICATES ?= 3
WORKERS ?= 4

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
	uv run jev-fc run --split dev --limit 8 --replicates 1 --max-repairs 1 --output results/smoke --resume

tune:
	uv run python scripts/run_suite.py --dev-only --repair-grid "$(REPAIR_GRID)" --results-dir "$(RESULTS_DIR)" --replicates $(REPLICATES) --workers $(WORKERS)

experiment:
	uv run python scripts/run_suite.py --repair-grid "$(REPAIR_GRID)" --results-dir "$(RESULTS_DIR)" --replicates $(REPLICATES) --workers $(WORKERS)

report:
	uv run jev-fc report "$(RESULTS_DIR)/test_selected" --paper paper/main.tex

paper:
	python3 scripts/build_paper.py

verify:
	uv run python scripts/verify_paper.py paper/main.pdf
