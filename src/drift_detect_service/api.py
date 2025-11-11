from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .data_quality import PredictRequest, run_data_quality_checks
from .model import AnomalyModel
from .monitoring import (
    anomalies_total,
    metrics_exposition_text,
    model_loaded,
    request_latency_seconds,
    requests_total,
)
from .settings import get_settings

app = FastAPI(title="Drift Detect Service", version="0.1.0")

_model = AnomalyModel()


@app.on_event("startup")
def _load_model_on_startup() -> None:
    settings = get_settings()
    loaded = _model.load(Path(settings.model_path))
    model_loaded.set(1 if loaded else 0)


def _ensure_model_loaded() -> bool:
    if not _model.is_loaded():
        settings = get_settings()
        loaded = _model.load(Path(settings.model_path))
        model_loaded.set(1 if loaded else 0)
    return _model.is_loaded()


@app.get("/health")
def health() -> dict[str, Any]:
    requests_total.labels(endpoint="/health").inc()
    return {"status": "ok", "model_loaded": int(_ensure_model_loaded())}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    requests_total.labels(endpoint="/predict").inc()
    # Measure latency manually since we want explicit control
    # (alternatively use a decorator)
    import time

    start = time.perf_counter()
    try:
        X = np.asarray(req.instances, dtype=float)
        run_data_quality_checks(X)
        if not _ensure_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        out = _model.predict(X)
        # IsolationForest: -1 means anomaly
        anomalies = sum(1 for p in out["predictions"] if p == -1)
        anomalies_total.labels(endpoint="/predict").inc(anomalies)
        return out
    finally:
        duration = time.perf_counter() - start
        request_latency_seconds.labels(endpoint="/predict").observe(duration)


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    requests_total.labels(endpoint="/metrics").inc()
    return PlainTextResponse(
        content=metrics_exposition_text(), media_type="text/plain; version=0.0.4"
    )
