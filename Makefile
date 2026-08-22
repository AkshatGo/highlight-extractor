# ============================================================
# Highlight Extraction Service — Common Dev Commands
# ============================================================
# Usage: make help (shows all commands)
# ============================================================

.PHONY: help install test lint typecheck run run-api docker-build docker-up docker-down clean

# Default target
.DEFAULT_GOAL := help

# Colors
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m # No Color

# Python executable
PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest

# --- Help ---
help: ## Show this help message
	@echo "$(GREEN)Highlight Extraction Service$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make install      Install all dependencies (production + dev)"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make test         Run unit tests (fast, no ML models)"
	@echo "  make test-slow    Run slow e2e tests (real models, GPU)"
	@echo "  make test-all     Run all tests"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo "  make format       Auto-format code"
	@echo ""
	@echo "$(YELLOW)Run:$(NC)"
	@echo "  make run          Start dev server (uvicorn, hot reload)"
	@echo "  make run-api      Start production server (gunicorn)"
	@echo "  make pipeline     Run CLI pipeline on a file"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-build Build production Docker image"
	@echo "  make docker-up    Start production containers"
	@echo "  make docker-dev   Start dev containers (hot reload)"
	@echo "  make docker-down  Stop all containers"
	@echo ""
	@echo "$(YELLOW)Cleanup:$(NC)"
	@echo "  make clean        Remove build artifacts and caches"

# --- Setup ---
install: ## Install all dependencies (production + dev)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install ruff mypy gunicorn
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# --- Testing ---
test: ## Run unit tests (fast, no ML models)
	$(PYTEST) tests/ -v -m "not slow" --tb=short

test-slow: ## Run slow e2e tests (real models, GPU required)
	$(PYTEST) tests/ -v -m "slow" --tb=short

test-all: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v -m "not slow" --cov=highlight_extractor --cov-report=term-missing --cov-report=html:htmlcov

# --- Quality ---
lint: ## Run linter (ruff)
	.venv/bin/ruff check src/ tests/

format: ## Auto-format code
	.venv/bin/ruff format src/ tests/

typecheck: ## Run type checker (mypy)
	.venv/bin/mypy src/ --ignore-missing-imports

# --- Run ---
run: ## Start dev server (uvicorn, hot reload)
	PYTHONPATH=src $(PYTHON) -m uvicorn highlight_extractor.api.app:app \
		--host 0.0.0.0 --port $${PORT:-8000} --reload --log-level info

run-api: ## Start production server (gunicorn)
	PYTHONPATH=src $(PYTHON) -m gunicorn highlight_extractor.api.app:app \
		--bind 0.0.0.0:$${PORT:-8000} \
		--workers $${WORKERS:-2} \
		--worker-class uvicorn.workers.UvicornWorker \
		--timeout 120 --graceful-timeout 30 \
		--access-logfile - --error-logfile - --log-level info

pipeline: ## Run CLI pipeline on a file (usage: make pipeline FILE=path/to/audio.mp3)
	@if [ -z "$(FILE)" ]; then echo "$(RED)Error: FILE not set. Usage: make pipeline FILE=path/to/audio.mp3$(NC)"; exit 1; fi
	PYTHONPATH=src $(PYTHON) scripts/run_pipeline.py $(FILE) --top_n $(or $(TOP_N),15)

# --- Docker ---
docker-build: ## Build production Docker image
	docker build -t highlight-extractor:latest --target production .

docker-up: ## Start production containers
	docker compose up -d api

docker-dev: ## Start dev containers (hot reload)
	docker compose up -d api-dev

docker-down: ## Stop all containers
	docker compose down

docker-logs: ## Tail container logs
	docker compose logs -f api

# --- Cleanup ---
clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache htmlcov .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"
