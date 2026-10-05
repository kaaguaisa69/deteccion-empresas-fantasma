"""Detector tabular basado en Random Forest (grupos G1 y G2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from efd.data import ExperimentData
from efd.models.base import BaseDetector
from efd.models.registry import register_detector


@register_detector
class RandomForestDetector(BaseDetector):
    """Random Forest de scikit-learn sobre la vista tabular.

    El mismo detector sirve a G1 y G2; la diferencia entre grupos está en las columnas de
    ``data.features``, no en el modelo.
    """

    name = "random_forest"

    def __init__(self, seed: int, **hyperparams: Any) -> None:
        """
        Args:
            seed: semilla del experimento, usada como ``random_state``.
            **hyperparams: argumentos de ``sklearn.ensemble.RandomForestClassifier`` tomados de la
                configuración.
        """
        self.seed = seed
        self.hyperparams = hyperparams

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Ajusta el bosque con las filas de entrenamiento de la vista tabular."""
        raise NotImplementedError

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        """Devuelve la columna de probabilidad de la clase positiva de ``predict_proba``."""
        raise NotImplementedError
