"""Explicaciones SHAP para los detectores tabulares (G1 y G2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from efd.data import ExperimentData
from efd.explainability.base import BaseExplainer, Explanation
from efd.explainability.registry import register_explainer
from efd.models.base import BaseDetector


@register_explainer
class ShapExplainer(BaseExplainer):
    """Valores SHAP de cada variable tabular para la probabilidad de la clase positiva.

    Permite comparar el peso de las variables transaccionales frente al de las estructurales en G2.
    """

    name = "shap"

    def __init__(self, seed: int, **params: Any) -> None:
        """
        Args:
            seed: semilla del experimento, para el muestreo del conjunto de referencia.
            **params: opciones del explicador de ``shap`` tomadas de la configuración.
        """
        self.seed = seed
        self.params = params

    def _explain(self, detector: BaseDetector, data: ExperimentData, idx: np.ndarray) -> Explanation:
        """Calcula los valores SHAP de las filas ``idx`` de la vista tabular."""
        raise NotImplementedError
