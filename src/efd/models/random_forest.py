"""Detector tabular basado en Random Forest (grupos G1 y G2)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from efd.data import ExperimentData
from efd.models.base import BaseDetector
from efd.models.registry import register_detector


@register_detector
class RandomForestDetector(BaseDetector):
    """Random Forest de scikit-learn sobre la vista tabular.

    El mismo detector sirve a G1 y G2; la diferencia entre grupos está en las columnas de
    ``data.features``, no en el modelo. El desbalance se compensa con ``class_weight``, que por defecto
    es ``balanced``. La partición de validación no se usa para entrenar.
    """

    name = "random_forest"

    def __init__(self, seed: int, **hyperparams: Any) -> None:
        """
        Args:
            seed: semilla de la corrida, usada como ``random_state``.
            **hyperparams: argumentos de ``sklearn.ensemble.RandomForestClassifier`` tomados de la
                configuración.
        """
        self.seed = seed
        self.hyperparams = {"class_weight": "balanced", **hyperparams}
        self.model: RandomForestClassifier | None = None

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        X, y = data.tabular(train_idx)
        self.model = RandomForestClassifier(random_state=self.seed, **self.hyperparams)
        self.model.fit(X, y)

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El detector no está entrenado.")
        X, _ = data.tabular(idx)
        positive = list(self.model.classes_).index(1)
        return self.model.predict_proba(X)[:, positive]
