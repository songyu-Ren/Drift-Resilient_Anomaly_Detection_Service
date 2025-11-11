from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .data_quality import PredictRequest, run_data_quality_checks
from .model import AnomalyModel
from .monitoring import (
    anomalies_total,
    latency_timer,
    metrics_exposition_text,
    model_loaded,
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
    settings = get_settings()
    loaded = int(_ensure_model_loaded())
    # Load metadata if available
    metadata: dict[str, Any] | None = None
    try:
        meta_path = Path(settings.metadata_path)
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        metadata = None
    resp = {"status": "ok", "model_loaded": bool(loaded), "env": settings.service.env, "metadata": metadata}
    requests_total.labels(endpoint="/health", method="GET", status="200").inc()
    return resp


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    with latency_timer(endpoint="/predict"):
        X = req.as_array()
        run_data_quality_checks(X)
        if not _ensure_model_loaded():
            # Count request with 503
            requests_total.labels(endpoint="/predict", method="POST", status="503").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        out = _model.predict(X)
        # IsolationForest: -1 means anomaly
        pred = out["predictions"][0]
        score = out["scores"][0]
        is_anomaly = pred == -1
        if is_anomaly:
            anomalies_total.labels(endpoint="/predict").inc(1)
        resp = {
            "anomaly_score": -float(score),
            "is_anomaly": bool(is_anomaly),
            "n_features": int(X.shape[1]),
        }
        # Count request with 200
        requests_total.labels(endpoint="/predict", method="POST", status="200").inc()
        return resp


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    requests_total.labels(endpoint="/metrics", method="GET", status="200").inc()
    return PlainTextResponse(
        content=metrics_exposition_text(), media_type="text/plain; version=0.0.4"
    )
