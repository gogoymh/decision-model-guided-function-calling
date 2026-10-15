.PHONY: setup prepare test demo smoke experiment report paper verify

setup:
	uv sync --extra dev
	@test -f .env || cp .env.example .env

prepare:
	uv run jev-fc prepare

test:
	uv run pytest -q
	uv run ruff check src tests scripts

demo:
	uv run jev-fc demo

smoke:
	uv run jev-fc run --split dev --limit 8 --replicates 1 --output results/smoke --resume

experiment:
	uv run python scripts/run_suite.py

report:
	uv run jev-fc report results/main --paper paper/main.tex

paper:
	python3 scripts/build_paper.py

verify:
	uv run python scripts/verify_paper.py paper/main.pdf
