"""Detector tabular basado en XGBoost (grupos G1 y G2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from efd.data import ExperimentData
from efd.models.base import BaseDetector
from efd.models.registry import register_detector


@register_detector
class XGBoostDetector(BaseDetector):
    """Clasificador de gradient boosting de XGBoost sobre la vista tabular.

    La partición de validación se usa como conjunto de evaluación para la parada temprana.
    """

    name = "xgboost"

    def __init__(self, seed: int, **hyperparams: Any) -> None:
        """
        Args:
            seed: semilla del experimento, usada como ``random_state``.
            **hyperparams: argumentos de ``xgboost.XGBClassifier`` tomados de la configuración.
        """
        self.seed = seed
        self.hyperparams = hyperparams

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Ajusta el modelo con entrenamiento y usa validación para la parada temprana."""
        raise NotImplementedError

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        """Devuelve la columna de probabilidad de la clase positiva de ``predict_proba``."""
        raise NotImplementedError
