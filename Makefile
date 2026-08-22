# MIMIC-IV angina lakehouse — common tasks.
#
#   make demo     one command: synthetic data -> lake -> warehouse -> dbt -> model
#   make up       start the stack (MinIO, Spark, Postgres, Airflow)
#   make test     unit + integration tests, no Docker required

SHELL := /bin/bash
COMPOSE := docker compose
PYTHON ?= python3
VENV := .venv
VENV_PY := $(VENV)/bin/python
PATIENTS ?= 5000
RUN_ID ?= $(shell date +%F)

# Spark 3.5 refuses to start on JDK 21+ without this; harmless on JDK 17.
export JDK_JAVA_OPTIONS := -Djava.security.manager=allow

.DEFAULT_GOAL := help
.PHONY: help venv data seed up down restart logs demo pipeline dbt-run dbt-test \
        dbt-docs train test lint format clean status psql shell-airflow

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the local dev environment
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --quiet --upgrade pip
	$(VENV_PY) -m pip install --quiet -r requirements/dev.txt
	@echo "dev environment ready: source $(VENV)/bin/activate"

data: ## Generate a synthetic MIMIC-IV-shaped extract into data/raw
	$(PYTHON) scripts/generate_synthetic_mimic.py --patients $(PATIENTS) --out-dir data/raw

seed: ## Upload data/raw into the MinIO raw zone
	S3_ENDPOINT=http://localhost:9000 $(PYTHON) scripts/seed_lake.py --source-dir data/raw

up: ## Build images and start the whole stack
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  Airflow      http://localhost:8080   (admin / admin)"
	@echo "  MinIO        http://localhost:9001   (minioadmin / minioadmin)"
	@echo "  Spark master http://localhost:8081"
	@echo "  Warehouse    postgresql://mimic:mimic@localhost:5433/mimic"

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

clean: ## Stop the stack and delete all data volumes
	$(COMPOSE) down -v
	rm -rf data/raw data/lake ml/artifacts dbt/target dbt/dbt_packages .pytest_cache

restart: down up ## Restart everything

logs: ## Tail logs from the scheduler
	$(COMPOSE) logs -f airflow-scheduler

status: ## Show container health
	$(COMPOSE) ps

demo: up data seed pipeline ## Full end-to-end demo from an empty machine
	@echo "demo complete — open http://localhost:8080 to see the DAG runs"

pipeline: ## Trigger the ELT DAG and wait for it to finish
	$(COMPOSE) exec -T airflow-scheduler airflow dags unpause mimic_lakehouse_elt
	$(COMPOSE) exec -T airflow-scheduler airflow dags trigger mimic_lakehouse_elt --run-id manual__$(RUN_ID)
	@echo "triggered mimic_lakehouse_elt — follow along at http://localhost:8080"

dbt-run: ## Run dbt models inside the Airflow container
	$(COMPOSE) exec -T airflow-scheduler bash -c \
	  "cd /opt/dbt && dbt deps && dbt run --profiles-dir /opt/dbt --target prod"

dbt-test: ## Run dbt tests
	$(COMPOSE) exec -T airflow-scheduler bash -c \
	  "cd /opt/dbt && dbt test --profiles-dir /opt/dbt --target prod"

dbt-docs: ## Generate and serve the dbt docs site on :8088
	$(COMPOSE) exec -T airflow-scheduler bash -c \
	  "cd /opt/dbt && dbt docs generate --profiles-dir /opt/dbt --target prod"
	$(COMPOSE) exec -d airflow-scheduler bash -c \
	  "cd /opt/dbt && dbt docs serve --profiles-dir /opt/dbt --port 8088 --no-browser"
	@echo "dbt docs: http://localhost:8088"

train: ## Train models on the published mart
	$(COMPOSE) exec -T airflow-scheduler python /opt/ml/train_angina_model.py \
	  --source warehouse --run-id $(RUN_ID)

test: ## Run the test suite locally (no Docker)
	cd spark && ../$(VENV_PY) -m pytest tests/ -v

test-fast: ## Unit tests only, skipping the slow end-to-end test
	cd spark && ../$(VENV_PY) -m pytest tests/ -v --ignore=tests/test_pipeline_end_to_end.py

lint: ## Lint Python and SQL
	$(VENV_PY) -m ruff check spark scripts ml airflow
	$(VENV_PY) -m sqlfluff lint dbt/models --dialect postgres || true

format: ## Auto-fix formatting
	$(VENV_PY) -m ruff check --fix spark scripts ml airflow
	$(VENV_PY) -m ruff format spark scripts ml airflow

psql: ## Open a warehouse shell
	$(COMPOSE) exec warehouse psql -U mimic -d mimic

shell-airflow: ## Shell into the Airflow scheduler
	$(COMPOSE) exec airflow-scheduler bash
