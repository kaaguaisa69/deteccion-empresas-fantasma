"""Interfaz común de los constructores de variables tabulares."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from efd.generation.network import InvoiceNetwork


class BaseFeatureBuilder(ABC):
    """Produce un bloque de variables por empresa a partir de la red de facturación.

    Cada grupo experimental combina bloques: G1 usa el transaccional y G2 el transaccional más el
    estructural. Los bloques se concatenan por columnas sobre el mismo índice de empresas.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def build(self, network: InvoiceNetwork) -> pd.DataFrame:
        """Devuelve una fila por empresa, indexada por ``company_id`` en el orden de
        ``network.companies``, y una columna por variable."""
