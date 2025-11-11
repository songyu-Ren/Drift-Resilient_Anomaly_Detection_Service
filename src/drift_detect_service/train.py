from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .settings import get_settings


def _make_synthetic_data(
    n_samples: int, n_anomalies: int, n_features: int, seed: int, cluster_std: float, anomaly_scale: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=cluster_std, size=(n_samples - n_anomalies, n_features))
    # Create outliers using a broader Gaussian to simulate anomalies
    anomalies = rng.normal(loc=0.0, scale=anomaly_scale, size=(n_anomalies, n_features))
    X = np.vstack([normal, anomalies])
    rng.shuffle(X)
    return X


def train_and_save() -> Path:
    settings = get_settings()

    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    n_samples = settings.training.n_samples
    n_features = settings.training.n_features
    seed = settings.training.random_seed
    iso_cfg = settings.training.isolation_forest

    # Derive anomaly count based on contamination fraction
    n_anomalies = max(1, int(iso_cfg.contamination * n_samples))
    synth = settings.training.synthetic

    X = _make_synthetic_data(
        n_samples,
        n_anomalies,
        n_features,
        seed,
        synth.cluster_std,
        synth.anomaly_scale,
    )

    model = IsolationForest(
        n_estimators=iso_cfg.n_estimators,
        contamination=iso_cfg.contamination,
        max_samples=iso_cfg.max_samples,
        random_state=iso_cfg.random_state,
    )
    model.fit(X)

    model_path = Path(settings.model_path)
    joblib.dump(model, model_path)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "framework": "scikit-learn",
        "model": "IsolationForest",
        "n_features": n_features,
        "params": {
            "n_estimators": iso_cfg.n_estimators,
            "contamination": iso_cfg.contamination,
            "max_samples": iso_cfg.max_samples,
            "random_state": iso_cfg.random_state,
            "n_samples": n_samples,
            "seed": seed,
            "synthetic": {
                "cluster_std": synth.cluster_std,
                "anomaly_scale": synth.anomaly_scale,
                "anomaly_fraction": n_anomalies / n_samples,
            },
        },
    }
    metadata_path = Path(settings.metadata_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path
