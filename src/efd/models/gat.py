"""Detector de grafo basado en Graph Attention Networks (grupo G3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from efd.data import ExperimentData
from efd.models.base import BaseDetector
from efd.models.registry import register_detector


@register_detector
class GATDetector(BaseDetector):
    """Clasificación de nodos con GAT de PyTorch Geometric en modo transductivo.

    El grafo completo se propaga en cada época; la pérdida solo se calcula sobre los nodos de
    entrenamiento y la validación se usa para la parada temprana.
    """

    name = "gat"

    def __init__(self, seed: int, **hyperparams: Any) -> None:
        """
        Args:
            seed: semilla del experimento, usada para inicializar pesos y el orden de entrenamiento.
            **hyperparams: arquitectura y entrenamiento (capas, dimensión oculta, cabezas de atención,
                tasa de aprendizaje, épocas, paciencia) tomados de la configuración.
        """
        self.seed = seed
        self.hyperparams = hyperparams

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Entrena la red sobre ``data.require_graph()`` con máscara de nodos de entrenamiento."""
        raise NotImplementedError

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        """Aplica softmax (o sigmoide) a la salida de la red y devuelve la de los nodos ``idx``."""
        raise NotImplementedError
