from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    instances: list[list[float]]

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("instances must not be empty")
        for row in v:
            if len(row) != 3:
                raise ValueError("each instance must have exactly 3 features")
            for val in row:
                if not isinstance(val, int | float):
                    raise ValueError("features must be numeric")
                if not np.isfinite(val):
                    raise ValueError("features must be finite")
        return v


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
