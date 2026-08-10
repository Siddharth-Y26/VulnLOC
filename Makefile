# VulnLoc developer tasks.
# Windows users without GNU make: every target below has a PowerShell twin,
#   .\make.ps1 <target>
.DEFAULT_GOAL := help
.PHONY: help setup setup-cuda gpu-check test test-fast lint format typecheck imports \
        train-debug train-gpu mlflow-ui docker-build docker-train clean freeze

VENV    ?= .venv
ifeq ($(OS),Windows_NT)
PY      ?= $(VENV)/Scripts/python.exe
else
PY      ?= $(VENV)/bin/python
endif
SRC     := src/vulnloc
PATHS   := src tests scripts

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Editable install with dev extras + pre-commit hooks
	$(PY) -m pip install --upgrade pip setuptools wheel
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pre_commit install

# The default PyPI wheel is CPU-only on Windows and Linux alike. CUDA_CHANNEL must
# match the host driver: `nvidia-smi` reports the maximum CUDA version it supports.
CUDA_CHANNEL ?= cu130

setup-cuda:  ## Replace CPU torch with a CUDA build (see CUDA_CHANNEL)
	$(PY) -m pip install --index-url https://download.pytorch.org/whl/$(CUDA_CHANNEL) \
	  --extra-index-url https://pypi.org/simple --force-reinstall torch

gpu-check:  ## Report the accelerator that `device.kind=auto` would select
	$(PY) -c "from vulnloc.training.device import log_accelerator, resolve_device; \
	from vulnloc.utils.logging import setup_logging; \
	setup_logging('INFO', json_to_file=False); log_accelerator(resolve_device('auto'))"

test:  ## Run the full test suite with coverage
	$(PY) -m pytest

test-fast:  ## Run tests, skipping anything marked slow
	$(PY) -m pytest -m "not slow" --no-cov -q

lint: format-check imports typecheck  ## ruff + black --check + import-linter + mypy
	$(PY) -m ruff check $(PATHS)

format:  ## Auto-format and auto-fix
	$(PY) -m black $(PATHS)
	$(PY) -m ruff check --fix $(PATHS)

format-check:  ## Verify formatting without writing
	$(PY) -m black --check $(PATHS)

typecheck:  ## Static type check
	$(PY) -m mypy $(SRC)

imports:  ## Enforce the L0->L6 layer rule (Architecture.md §3)
	$(PY) -m importlinter.cli lint

train-debug:  ## Phase 0 smoke run: no-op loop, logs a dummy metric to MLflow
	$(PY) scripts/train.py train=debug

train-gpu:  ## Same smoke run, but fail loudly if no GPU is usable
	$(PY) scripts/train.py train=debug device.kind=cuda

mlflow-ui:  ## Browse tracked runs at http://127.0.0.1:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db \
	  --default-artifact-root ./artifacts/mlartifacts

freeze:  ## Pin the current environment into requirements.lock.txt
	$(PY) -m pip freeze --exclude-editable > requirements.lock.txt

docker-build:  ## Build the CPU image
	docker compose build

docker-train:  ## Run the Phase 0 smoke run inside the image
	docker compose run --rm vulnloc python scripts/train.py train=debug

clean:  ## Remove caches and build leftovers (keeps artifacts/ and .venv/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
