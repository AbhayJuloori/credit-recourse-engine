.PHONY: install train serve test lint clean

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "✓ Environment ready. Activate with: source .venv/bin/activate"

# ── Training ─────────────────────────────────────────────────────────────────
train:
	@echo "Starting training pipeline (90–180 min)…"
	.venv/bin/python -m backend.scripts.train

resume:
	@echo "Resuming from saved model (MAPIE + DiCE only)…"
	.venv/bin/python -m backend.scripts.resume_training

# ── API server ────────────────────────────────────────────────────────────────
serve:
	.venv/bin/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	.venv/bin/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	.venv/bin/pytest backend/tests/ -v --tb=short

test-fast:
	.venv/bin/pytest backend/tests/test_effort_ranker.py backend/tests/test_constraints.py -v

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	.venv/bin/pip install ruff --quiet
	.venv/bin/ruff check backend/ --fix

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf backend/artifacts/*.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-all: clean
	rm -rf .venv

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Credit Recourse Engine"
	@echo ""
	@echo "  make install      Set up virtualenv + install dependencies"
	@echo "  make train        Run full training pipeline (needs dataset)"
	@echo "  make serve        Start development server (localhost:8000)"
	@echo "  make serve-prod   Start production server"
	@echo "  make test         Run all tests"
	@echo "  make test-fast    Run constraint + ranker tests only (no ML needed)"
	@echo "  make lint         Run ruff linter"
	@echo "  make clean        Delete model artifacts"
