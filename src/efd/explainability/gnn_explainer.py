"""Explicaciones con GNNExplainer para los detectores de grafo (G3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from efd.data import ExperimentData
from efd.explainability.base import BaseExplainer, Explanation
from efd.explainability.registry import register_explainer
from efd.models.base import BaseDetector


@register_explainer
class GNNExplainerAdapter(BaseExplainer):
    """Adapta ``torch_geometric.explain.GNNExplainer`` a la interfaz ``BaseExplainer``.

    Devuelve la máscara de variables de nodo como atribuciones de variables y la máscara de aristas
    como atribuciones de aristas, para identificar qué relaciones de facturación sustentan la
    predicción.
    """

    name = "gnn_explainer"

    def __init__(self, seed: int, **params: Any) -> None:
        """
        Args:
            seed: semilla del experimento, para la inicialización de las máscaras.
            **params: opciones de ``GNNExplainer`` (épocas, tasa de aprendizaje) tomadas de la
                configuración.
        """
        self.seed = seed
        self.params = params

    def _explain(self, detector: BaseDetector, data: ExperimentData, idx: np.ndarray) -> Explanation:
        """Optimiza las máscaras de variables y aristas para cada nodo de ``idx``."""
        raise NotImplementedError
