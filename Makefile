VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
BLACK=$(VENV)/bin/black
RUFF=$(VENV)/bin/ruff
PYTEST=$(VENV)/bin/pytest
UVICORN=$(VENV)/bin/uvicorn

.PHONY: help install format lint test train run docker-build docker-up docker-down

help:
	@echo "Targets: install, format, lint, test, train, run, docker-build, docker-up, docker-down"

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

format:
	$(BLACK) .

lint:
	$(RUFF) check --strict .

test:
	$(PYTEST) -q

train:
	$(PY) -c 'from src.drift_detect_service.train import train_and_save; train_and_save(); print("Trained")'

run:
	$(UVICORN) src.drift_detect_service.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t drift-detect:latest -f docker/Dockerfile .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down