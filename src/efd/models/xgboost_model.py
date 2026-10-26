"""Detector tabular basado en XGBoost (grupos G1 y G2)."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from efd.data import ExperimentData
from efd.models.base import BaseDetector
from efd.models.registry import register_detector


@register_detector
class XGBoostDetector(BaseDetector):
    """Clasificador de gradient boosting de XGBoost sobre la vista tabular.

    El desbalance se compensa con ``scale_pos_weight``; si la configuración no lo fija, se calcula como
    negativos ÷ positivos de entrenamiento. La partición de validación es el conjunto de evaluación de
    la parada temprana (``early_stopping_rounds``), medida con ``aucpr``.
    """

    name = "xgboost"

    def __init__(self, seed: int, **hyperparams: Any) -> None:
        """
        Args:
            seed: semilla de la corrida, usada como ``random_state``.
            **hyperparams: argumentos de ``xgboost.XGBClassifier`` tomados de la configuración.
        """
        self.seed = seed
        self.hyperparams = {"eval_metric": "aucpr", **hyperparams}
        self.model: XGBClassifier | None = None

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        X_train, y_train = data.tabular(train_idx)
        X_val, y_val = data.tabular(val_idx)
        params = dict(self.hyperparams)
        if params.get("scale_pos_weight") is None:
            positives = int(y_train.sum())
            params["scale_pos_weight"] = (y_train.size - positives) / max(positives, 1)
        self.model = XGBClassifier(random_state=self.seed, **params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El detector no está entrenado.")
        X, _ = data.tabular(idx)
        return self.model.predict_proba(X)[:, 1]
