"""Variables estructurales de red por contribuyente (grupo G2)."""

from __future__ import annotations

import pandas as pd

from efd.features.base import BaseFeatureBuilder, FeatureContext
from efd.features.registry import register_feature_builder
from efd.graph.builder import InvoiceGraphBuilder
from efd.graph.structural import StructuralMetricsCalculator
from efd.network import InvoiceNetwork


@register_feature_builder
class StructuralFeatureBuilder(BaseFeatureBuilder):
    """Expone como variables tabulares las métricas de red de ``efd.graph``.

    Solo consume las métricas; su cálculo está en ``StructuralMetricsCalculator``.
    """

    name = "structural"

    def build(self, network: InvoiceNetwork, context: FeatureContext) -> pd.DataFrame:
        graph = InvoiceGraphBuilder().to_networkx(network)
        metrics = StructuralMetricsCalculator(seed=context.seed).compute(graph, network.representatives)
        return metrics.reindex(pd.Index(network.taxpayers["taxpayer_id"], name="taxpayer_id"))
