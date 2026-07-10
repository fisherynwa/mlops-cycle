# One-command demo + local dev shortcuts.
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db

.PHONY: setup data train serve monitor test lint demo up down clean

setup:            ## install deps into a local venv via uv
	uv sync

data:                ## generate reference + no-drift + data-drift datasets
	uv run python data/data_generating.py

train:            ## fit the GAM, verify shape recovery, register @champion
	uv run python -m src.train

serve:            ## run the FastAPI model server locally (http://localhost:8000/docs)
	uv run uvicorn src.serve:app --reload

monitor:          ## build an Evidently drift report for the concept-drift batch
	uv run python -m src.monitor --current data/prod_concept_smoker.csv

test:             ## run the test suite (shape-recovery + JSD drift gates)
	uv run pytest

lint:             ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .

demo: up          ## bring up the full stack, then print where to look
	@echo "Waiting for API..." && for i in $$(seq 1 40); do curl -sf http://localhost:8000/health >/dev/null && break || sleep 5; done
	@docker compose --profile monitor run --rm monitor || true
	@echo ""
	@echo "  MLflow UI ........ http://localhost:5000"
	@echo "  Model API docs ... http://localhost:8000/docs"
	@echo "  Drift explorer ... http://localhost:8080/drift_explorer.html"
	@echo "  Evidently report . http://localhost:8080/reports/report.html"

up:               ## build + start postgres, mlflow, trainer, api, dashboard
	docker compose up -d --build

down:             ## stop everything and remove volumes
	docker compose down -v

clean:            ## remove local artifacts
	rm -rf mlruns mlflow.db reports/*.html data/predictions.jsonl
