"""Métricas estructurales de los nodos del grafo de facturación."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import networkx as nx


class StructuralMetricsCalculator:
    """Calcula grado, PageRank, coeficiente de clustering y comunidad de cada empresa."""

    def __init__(self, seed: int) -> None:
        """
        Args:
            seed: semilla del experimento, para los algoritmos de detección de comunidades que la
                requieren.
        """
        self.seed = seed

    def compute(self, graph: nx.DiGraph) -> pd.DataFrame:
        """Devuelve una fila por nodo (índice = ``taxpayer_id``) y una columna por métrica."""
        raise NotImplementedError
