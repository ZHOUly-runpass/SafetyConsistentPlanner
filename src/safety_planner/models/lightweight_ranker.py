from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class GBDTRanker:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._model: Any | None = None
        self._linear_weights: NDArray[np.float64] | None = None

    def fit(self, features: NDArray[Any], labels: NDArray[Any]) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must have shape [N, D].")
        if y.shape != (x.shape[0],):
            raise ValueError("labels must have shape [N].")

        backend = self._config.get("backend", "auto")
        if backend in ("auto", "sklearn"):
            try:
                from sklearn.ensemble import GradientBoostingRegressor

                model = GradientBoostingRegressor(
                    random_state=int(self._config.get("random_state", 0)),
                    n_estimators=int(self._config.get("n_estimators", 100)),
                    max_depth=int(self._config.get("max_depth", 3)),
                )
                model.fit(x, y)
                self._model = model
                return
            except ImportError:
                if backend == "sklearn":
                    raise

        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        self._linear_weights = np.linalg.pinv(x_aug) @ y

    def predict(self, features: NDArray[Any]) -> NDArray[np.float64]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must have shape [N, D].")
        if self._model is not None:
            return np.asarray(self._model.predict(x), dtype=np.float64)
        if self._linear_weights is None:
            raise RuntimeError("Ranker must be fitted before predict().")
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        return x_aug @ self._linear_weights
