"""Generador de la red sintética de facturación electrónica."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from efd.generation.network import InvoiceNetwork


class InvoiceNetworkGenerator:
    """Simula empresas legítimas y fantasmas y las facturas que intercambian.

    Todos los parámetros de la simulación provienen de la sección ``dataset`` de la configuración, lo
    que permite repetir el experimento variando uno de ellos (por ejemplo, la proporción de empresas
    fantasmas) para el análisis de sensibilidad.
    """

    def __init__(self, params: Mapping[str, Any], seed: int) -> None:
        """
        Args:
            params: parámetros de la sección ``dataset`` de la configuración.
            seed: semilla del experimento; con la misma semilla y parámetros la red es idéntica.
        """
        self.params = dict(params)
        self.seed = seed

    def generate(self) -> InvoiceNetwork:
        """Genera la red completa de empresas y facturas con sus etiquetas."""
        raise NotImplementedError
