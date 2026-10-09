PY := .venv/bin/python

.PHONY: setup test lint fmt report notebook clean

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

# Needs network on first run (FRED public CSV, cached to data/cache/).
report:
	$(PY) -m meridian.report

notebook:
	$(PY) notebooks/build_notebook.py

clean:
	rm -rf .pytest_cache .ruff_cache data/cache/*.csv
