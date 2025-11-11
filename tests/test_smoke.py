from __future__ import annotations

from pathlib import Path

import pytest
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


@pytest.fixture(scope="session", autouse=True)
def _train_once_if_missing():
    ensure_model_artifacts()


def test_smoke_end_to_end():
    client = TestClient(app)

    # Health
    h = client.get("/health")
    assert h.status_code == 200
    assert "model_loaded" in h.json()
    assert h.json()["model_loaded"] is True
    assert "env" in h.json()

    # Predict
    payload = {"features": {"f1": 0.1, "f2": -0.2, "f3": 0.05}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"anomaly_score", "is_anomaly", "n_features"}
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["is_anomaly"], bool)
    assert body["n_features"] == 3

    # Metrics
    m = client.get("/metrics")
    assert m.status_code == 200
    text = m.text
    # Basic presence checks for required metrics
    assert "dds_requests_total" in text
    assert "dds_request_latency_seconds" in text
    assert "dds_anomalies_total" in text
    assert "dds_model_loaded" in text
    # end
