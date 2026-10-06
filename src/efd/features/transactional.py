"""Variables transaccionales por empresa (grupo G1)."""

from __future__ import annotations

import pandas as pd

from efd.features.base import BaseFeatureBuilder
from efd.features.registry import register_feature_builder
from efd.network import InvoiceNetwork


@register_feature_builder
class TransactionalFeatureBuilder(BaseFeatureBuilder):
    """Agrega las facturas emitidas y recibidas de cada empresa (montos, volúmenes, frecuencias).

    Solo usa atributos de las facturas propias de la empresa, sin información de la estructura de la
    red más allá de sus contrapartes directas.
    """

    name = "transactional"

    def build(self, network: InvoiceNetwork) -> pd.DataFrame:
        raise NotImplementedError
