# Drift Detect Service

Production-minded FastAPI microservice for real-time anomaly detection using IsolationForest.

## Features
- Python 3.11, PEP 8, Black (line length 100), Ruff (strict), PyTest
- FastAPI app: `/predict`, `/health`, `/metrics` (Prometheus exposition)
- Training script: saves `artifacts/model.joblib` and `artifacts/metadata.json`
- Baseline model: IsolationForest on deterministic synthetic 3-feature data
- Observability: request counters, latency histogram, anomaly counter, model_loaded gauge
- Data quality: Pydantic request schema + numeric/finite checks; extensible stubs for GE/Evidently
- Reproducibility: Dockerfile, docker-compose, GitHub Actions CI (lint+tests), Makefile, requirements, pyproject

## Quickstart
1. Install deps and run tests:
   - `make install`
   - `make test`

2. Train the baseline model:
   - `make train`

3. Run the API locally:
   - `make run`
   - Visit `http://localhost:8000/health` and `http://localhost:8000/docs`

4. Docker:
   - `make docker-build`
   - `make docker-up` (service at `http://localhost:8000`)
   - `make docker-down`

## API
- `POST /predict`:
  - Request: `{ "instances": [[f1, f2, f3], ...] }`
  - Response: `{ "predictions": [1|-1, ...], "scores": [float, ...] }`
- `GET /health`: service status and model load indicator
- `GET /metrics`: Prometheus exposition text

## Config
- `config/settings.yaml` controls training parameters and artifact locations.
- Override via `CONFIG_PATH=/custom/path/settings.yaml`.

## Testing & Linting
- `make lint` for Ruff strict checks
- `make format` to format with Black
- `make test` to run PyTest (includes a smoke test)

## Notes
- IsolationForest returns `-1` for anomalies and `1` for normal.
- The synthetic dataset includes obvious outliers to validate behavior.