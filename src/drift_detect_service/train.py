from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .settings import get_settings


def _make_synthetic_data(
    n_samples: int, n_anomalies: int, n_features: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_samples - n_anomalies, n_features))
    # Create outliers far from the normal distribution
    anomalies = rng.uniform(low=8.0, high=12.0, size=(n_anomalies, n_features))
    X = np.vstack([normal, anomalies])
    rng.shuffle(X)
    return X


def train_and_save() -> Path:
    settings = get_settings()

    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    n_samples = settings.training.n_samples
    n_anomalies = settings.training.n_anomalies
    n_features = settings.training.n_features
    seed = settings.training.random_seed

    X = _make_synthetic_data(n_samples, n_anomalies, n_features, seed)

    iso_cfg = settings.training.isolation_forest
    model = IsolationForest(
        n_estimators=iso_cfg.n_estimators,
        contamination=iso_cfg.contamination,
        random_state=iso_cfg.random_state,
    )
    model.fit(X)

    model_path = Path(settings.model_path)
    joblib.dump(model, model_path)

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": seed,
        "n_samples": n_samples,
        "n_anomalies": n_anomalies,
        "n_features": n_features,
        "model": {
            "type": "IsolationForest",
            "n_estimators": iso_cfg.n_estimators,
            "contamination": iso_cfg.contamination,
            "random_state": iso_cfg.random_state,
        },
    }
    metadata_path = Path(settings.metadata_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path
