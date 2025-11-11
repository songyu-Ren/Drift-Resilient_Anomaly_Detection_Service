from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyModel:
    def __init__(self) -> None:
        self._model: IsolationForest | None = None

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        self._model = joblib.load(path)
        return True

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, X: np.ndarray) -> dict[str, list[float]]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        preds = self._model.predict(X)
        scores = self._model.score_samples(X)
        return {"predictions": preds.tolist(), "scores": scores.tolist()}
