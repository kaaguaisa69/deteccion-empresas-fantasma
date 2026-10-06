"""Variables estructurales de grafo por empresa (grupo G2)."""

from __future__ import annotations

import pandas as pd

from efd.features.base import BaseFeatureBuilder
from efd.features.registry import register_feature_builder
from efd.network import InvoiceNetwork


@register_feature_builder
class StructuralFeatureBuilder(BaseFeatureBuilder):
    """Expone como variables tabulares las métricas estructurales de cada empresa.

    Construye el grafo con ``efd.graph.InvoiceGraphBuilder`` y calcula grado, PageRank, clustering y
    comunidad con ``efd.graph.StructuralMetricsCalculator``.
    """

    name = "structural"

    def __init__(self, seed: int) -> None:
        """
        Args:
            seed: semilla del experimento, que se propaga al cálculo de comunidades.
        """
        self.seed = seed

    def build(self, network: InvoiceNetwork) -> pd.DataFrame:
        raise NotImplementedError
