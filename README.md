# Drift Detect Service

Production-minded FastAPI microservice for real-time anomaly detection using IsolationForest.

## Features
- Python 3.11, PEP 8, Black (line length 100), Ruff (pragmatic), PyTest
- FastAPI app: `/predict`, `/health`, `/metrics` (Prometheus exposition)
- Training script: saves `artifacts/model.joblib` and `artifacts/metadata.json`
- Baseline model: IsolationForest on deterministic synthetic 3-feature Gaussian data
- Observability: `dds_requests_total{endpoint,method,status}`, `dds_request_latency_seconds{endpoint}` (histogram), `dds_anomalies_total`, `dds_model_loaded`
- Data quality: Pydantic v2 model enforcing exact keys `f1,f2,f3` and finite numeric values; stubs for GE/Evidently integration
- Reproducibility: Dockerfile (trains at build), docker-compose, GitHub Actions CI (lint+tests), Makefile, requirements, pyproject

## Quickstart
1. Setup and tests:
   - `make setup`
   - `make test`

2. Train the baseline model:
   - `make train`

3. Run the API locally:
   - `make run`
   - Visit `http://localhost:8000/health` and `http://localhost:8000/docs`

4. Docker:
   - `make docker-build`
   - `make docker-run` (service at `http://localhost:8000`)

## API
- `POST /predict`:
  - Request: `{ "features": { "f1": <float>, "f2": <float>, "f3": <float> } }`
  - Response: `{ "anomaly_score": <float>, "is_anomaly": <bool>, "n_features": 3 }`
- `GET /health`: includes `model_loaded`, `env`, and training `metadata` if present
- `GET /metrics`: Prometheus exposition text with `dds_*` metrics

Example:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"f1":0.1,"f2":-0.2,"f3":0.05}}'
curl -s localhost:8000/metrics | head -n 20
```

## Config
- `config/settings.yaml` controls training, monitoring (latency buckets), and artifact locations.
- Override via `DDS_SETTINGS=/custom/path/settings.yaml`.

## Testing & Linting
- `make lint` for Ruff checks (ignores E203/E266/E501/PLR2004)
- `make fmt` to format with Black
- `make test` to run PyTest (includes a smoke test)

## Notes
- IsolationForest returns `-1` for anomalies and `1` for normal; `anomaly_score = -score_samples[0]` so higher means more anomalous.
- Docker image trains during build to ensure artifacts are present.
- Extend data quality with Great Expectations/Evidently by filling `run_ge_checks`/`run_evidently_checks` in `data_quality.py`.