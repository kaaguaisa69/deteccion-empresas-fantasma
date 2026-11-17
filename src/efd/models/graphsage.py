"""Detector de grafo basado en GraphSAGE (grupo G3)."""

from __future__ import annotations

from typing import Any

from efd.models.neural import HeteroGNNDetector
from efd.models.registry import register_detector


@register_detector
class GraphSAGEDetector(HeteroGNNDetector):
    """Clasificación de nodos con GraphSAGE de PyTorch Geometric en modo transductivo.

    El grafo completo se propaga en cada época; la pérdida solo se calcula sobre los nodos de
    entrenamiento y la validación se usa para la parada temprana. No lee ``edge_attr``.
    """

    name = "graphsage"
    conv = "sage"

    def __init__(self, seed: int, **hyperparams: Any) -> None:
        """
        Args:
            seed: semilla de pesos y dropout.
            **hyperparams: ``hidden_channels``, ``num_layers``, ``lr``, ``dropout``, ``weight_decay``,
                ``epochs``, ``patience`` y ``num_threads``.
        """
        if "heads" in hyperparams:
            raise TypeError("graphsage no acepta el parámetro 'heads'.")
        super().__init__(seed, **hyperparams)
