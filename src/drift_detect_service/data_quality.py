from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator


class Features(BaseModel):
    f1: float
    f2: float
    f3: float

    @field_validator("f1", "f2", "f3")
    @classmethod
    def validate_numeric_and_finite(cls, v: float) -> float:
        if not isinstance(v, int | float):
            raise ValueError("feature must be numeric")
        if not np.isfinite(v):
            raise ValueError("feature must be finite")
        return float(v)


class PredictRequest(BaseModel):
    features: Features

    def as_array(self) -> np.ndarray:
        return np.asarray([[self.features.f1, self.features.f2, self.features.f3]], dtype=float)


def ensure_numeric_and_finite(X: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("Input must be of shape (n_samples, 3)")
    if not np.isfinite(X).all():
        raise ValueError("All values must be finite")


# Stubs for extensibility with Great Expectations / Evidently
def run_ge_checks(_X: np.ndarray) -> None:
    # Placeholder: integrate Great Expectations suites here
    return None


def run_evidently_checks(_X: np.ndarray) -> None:
    # Placeholder: integrate Evidently reports here
    return None


def run_data_quality_checks(X: np.ndarray) -> None:
    ensure_numeric_and_finite(X)
    run_ge_checks(X)
    run_evidently_checks(X)
