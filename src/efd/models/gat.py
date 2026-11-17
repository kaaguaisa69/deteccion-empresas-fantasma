"""Detector de grafo basado en Graph Attention Networks (grupo G3)."""

from __future__ import annotations

from typing import Any

from efd.models.neural import HeteroGNNDetector
from efd.models.registry import register_detector


@register_detector
class GATDetector(HeteroGNNDetector):
    """Clasificación de nodos con GAT de PyTorch Geometric en modo transductivo.

    El grafo completo se propaga en cada época; la pérdida solo se calcula sobre los nodos de
    entrenamiento y la validación se usa para la parada temprana. Usa ``edge_attr`` (4 columnas) en las
    aristas de factura.
    """

    name = "gat"
    conv = "gat"

    def __init__(self, seed: int, **hyperparams: Any) -> None:
        """
        Args:
            seed: semilla de pesos y dropout.
            **hyperparams: ``hidden_channels`` (por cabeza), ``heads``, ``num_layers``, ``lr``, ``dropout``,
                ``weight_decay``, ``epochs``, ``patience`` y ``num_threads``.
        """
        super().__init__(seed, **hyperparams)
