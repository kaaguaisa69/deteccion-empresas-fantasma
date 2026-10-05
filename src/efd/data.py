"""Contenedor de datos compartido por todos los grupos experimentales."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from torch_geometric.data import Data


@dataclass(frozen=True)
class ExperimentData:
    """Vista tabular y vista de grafo de la misma población de empresas.

    La posición ``i`` identifica al mismo nodo (empresa) en ambas vistas: es la fila ``i`` de
    ``features`` y ``labels`` y el nodo ``i`` de ``graph``. Las particiones se expresan como índices
    posicionales de nodo, por lo que G1, G2 y G3 se entrenan y evalúan exactamente sobre las mismas
    empresas.

    Attributes:
        features: variables tabulares, una fila por empresa; el índice contiene el identificador de
            empresa y debe ser único.
        labels: etiqueta binaria (1 = empresa fantasma, 0 = legítima) con el mismo índice que
            ``features``.
        graph: grafo de facturación en formato PyTorch Geometric con un nodo por empresa, o ``None``
            cuando el experimento solo usa la vista tabular.
    """

    features: pd.DataFrame
    labels: pd.Series
    graph: Data | None = None

    def __post_init__(self) -> None:
        index = self.features.index
        if index.has_duplicates:
            raise ValueError("Los identificadores de empresa en 'features' deben ser únicos.")
        if index.hasnans:
            raise ValueError("Los identificadores de empresa en 'features' no pueden ser nulos.")
        if not self.labels.index.equals(index):
            raise ValueError("'labels' debe tener el mismo índice, en el mismo orden, que 'features'.")
        if not self.labels.isin([0, 1]).all():
            raise ValueError("'labels' solo puede contener los valores 0 y 1.")
        if self.graph is not None:
            self._check_graph_alignment(self.graph)

    def _check_graph_alignment(self, graph: Data) -> None:
        """Verifica que la vista de grafo describa la misma población que la vista tabular."""
        if graph.num_nodes != self.n_nodes:
            raise ValueError(
                f"El grafo tiene {graph.num_nodes} nodos y la vista tabular {self.n_nodes} filas."
            )
        graph_labels = getattr(graph, "y", None)
        if graph_labels is not None and not np.array_equal(
            np.asarray(graph_labels).reshape(-1), self.y
        ):
            raise ValueError("Las etiquetas del grafo no coinciden con las de la vista tabular.")

    @property
    def n_nodes(self) -> int:
        """Número de empresas (nodos)."""
        return len(self.features)

    @property
    def node_ids(self) -> pd.Index:
        """Identificadores de empresa en orden posicional."""
        return self.features.index

    @property
    def y(self) -> np.ndarray:
        """Etiquetas como arreglo de enteros en orden posicional."""
        return self.labels.to_numpy(dtype=np.int64)

    def check_idx(self, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        """Convierte ``idx`` en un arreglo 1-D de posiciones de nodo válidas.

        Raises:
            ValueError: si ``idx`` no es unidimensional, no es entero o contiene posiciones fuera de
                rango o repetidas.
        """
        positions = np.asarray(idx)
        if positions.ndim != 1:
            raise ValueError("Los índices de nodo deben ser un arreglo unidimensional.")
        if positions.size == 0:
            return positions.astype(np.int64)
        if not np.issubdtype(positions.dtype, np.integer):
            raise ValueError("Los índices de nodo deben ser enteros.")
        if positions.min() < 0 or positions.max() >= self.n_nodes:
            raise ValueError(f"Hay índices de nodo fuera del rango [0, {self.n_nodes}).")
        if np.unique(positions).size != positions.size:
            raise ValueError("Los índices de nodo no pueden repetirse.")
        return positions.astype(np.int64)

    def tabular(self, idx: Sequence[int] | np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """Devuelve ``(X, y)`` de la vista tabular para las posiciones ``idx``."""
        positions = self.check_idx(idx)
        return self.features.iloc[positions], self.y[positions]

    def require_graph(self) -> Data:
        """Devuelve la vista de grafo.

        Raises:
            ValueError: si el contenedor se construyó sin vista de grafo.
        """
        if self.graph is None:
            raise ValueError("Este ExperimentData no contiene la vista de grafo.")
        return self.graph

    def select_features(self, columns: Sequence[str]) -> ExperimentData:
        """Devuelve un contenedor con un subconjunto de columnas tabulares.

        Permite derivar la vista de G1 (solo variables transaccionales) a partir de la de G2 sin
        alterar las etiquetas, el grafo ni el orden de los nodos.

        Raises:
            KeyError: si alguna columna no existe.
        """
        missing = [column for column in columns if column not in self.features.columns]
        if missing:
            raise KeyError(f"Columnas inexistentes: {missing}")
        return replace(self, features=self.features.loc[:, list(columns)])
