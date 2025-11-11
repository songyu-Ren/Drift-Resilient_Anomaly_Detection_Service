VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
BLACK=$(VENV)/bin/black
RUFF=$(VENV)/bin/ruff
PYTEST=$(VENV)/bin/pytest
UVICORN=$(VENV)/bin/uvicorn

.PHONY: help setup fmt lint test train run docker-build docker-run

help:
	@echo "Targets: setup, fmt, lint, test, train, run, docker-build, docker-run"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

fmt:
	$(BLACK) .

lint:
	$(RUFF) check .

test:
	$(PYTEST) -q

train:
	$(PY) -c 'from src.drift_detect_service.train import train_and_save; train_and_save(); print("Trained")'

run:
	$(UVICORN) src.drift_detect_service.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t drift-detect:latest -f docker/Dockerfile .

docker-run:
	docker compose up --build -d