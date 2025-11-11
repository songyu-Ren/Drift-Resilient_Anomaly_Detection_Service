from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class IsolationForestConfig(BaseModel):
    n_estimators: int = Field(default=200)
    contamination: float = Field(default=0.02)
    max_samples: int | str = Field(default="auto")
    random_state: int = Field(default=42)


class SyntheticConfig(BaseModel):
    cluster_std: float = Field(default=1.0)
    anomaly_scale: float = Field(default=10.0)


class TrainingConfig(BaseModel):
    random_seed: int = Field(default=42)
    n_samples: int = Field(default=5000)
    n_features: int = Field(default=3)
    isolation_forest: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)


class ServiceConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    env: str = Field(default="dev")


class MonitoringConfig(BaseModel):
    enabled: bool = Field(default=True)
    latency_buckets: list[float] = Field(
        default_factory=lambda: [
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
        ]
    )


class Settings(BaseModel):
    artifacts_dir: str = Field(default="artifacts")
    model_path: str = Field(default="artifacts/model.joblib")
    metadata_path: str = Field(default="artifacts/metadata.json")
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def _default_config_path() -> Path:
    # Assume project root two levels above src/drift_detect_service/
    root = Path(__file__).resolve().parents[2]
    return root / "config" / "settings.yaml"


def load_settings(path: Path | None = None) -> Settings:
    env_path = os.getenv("DDS_SETTINGS")
    config_path = Path(env_path) if env_path else (path or _default_config_path())
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
