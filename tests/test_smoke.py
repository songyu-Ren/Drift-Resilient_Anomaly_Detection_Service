from __future__ import annotations

from pathlib import Path

from drift_detect_service.api import app
from drift_detect_service.settings import get_settings
from drift_detect_service.train import train_and_save
from fastapi.testclient import TestClient


def ensure_model_artifacts() -> Path:
    settings = get_settings()
    model_path = Path(settings.model_path)
    if not model_path.exists():
        return train_and_save()
    return model_path


def test_smoke_end_to_end():
    ensure_model_artifacts()

    client = TestClient(app)

    # Health
    h = client.get("/health")
    assert h.status_code == 200
    assert "model_loaded" in h.json()
    assert h.json()["model_loaded"] == 1

    # Predict
    payload = {"instances": [[0.1, -0.2, 0.3], [9.0, 10.0, 11.0]]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    # Expect at least the obvious outlier to be an anomaly (-1)
    assert -1 in body["predictions"]

    # Metrics
    m = client.get("/metrics")
    assert m.status_code == 200
    text = m.text
    # Basic presence checks for required metrics
    assert "requests_total" in text
    assert "request_latency_seconds" in text
    assert "anomalies_total" in text
    assert "model_loaded" in text
