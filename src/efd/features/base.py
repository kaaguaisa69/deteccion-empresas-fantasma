"""Interfaz común de los constructores de variables tabulares."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from efd.network import InvoiceNetwork


@dataclass(frozen=True)
class FeatureContext:
    """Información del escenario y de la corrida que necesitan los constructores de variables.

    Attributes:
        period_start: primer día del período simulado.
        period_end: último día del período simulado.
        seed: semilla de la corrida, para los cálculos aleatorios (por ejemplo, comunidades).
    """

    period_start: pd.Timestamp
    period_end: pd.Timestamp
    seed: int

    @property
    def period_days(self) -> int:
        """Duración del período en días, ambos extremos inclusive."""
        return (self.period_end - self.period_start).days + 1


class BaseFeatureBuilder(ABC):
    """Produce un bloque de variables por contribuyente a partir de la red de facturación.

    Cada grupo experimental combina bloques: G1 usa el transaccional y G2 el transaccional más el
    estructural. Los bloques se concatenan por columnas sobre el mismo índice de contribuyentes. Ningún
    bloque puede usar las columnas de etiqueta de ``network.taxpayers``.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def build(self, network: InvoiceNetwork, context: FeatureContext) -> pd.DataFrame:
        """Devuelve una fila por contribuyente, indexada por ``taxpayer_id`` en el orden de
        ``network.taxpayers``, y una columna numérica por variable."""
