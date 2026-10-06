"""Construcción del grafo de facturación a partir de la red de empresas y facturas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from efd.network import InvoiceNetwork

if TYPE_CHECKING:
    import networkx as nx
    from torch_geometric.data import Data


class InvoiceGraphBuilder:
    """Convierte la red en un grafo dirigido emisor → receptor con un nodo por contribuyente.

    Las aristas agregan las facturas entre cada par de empresas. El orden de los nodos sigue el de
    ``network.taxpayers``, el mismo que usa la vista tabular de ``ExperimentData``.
    """

    def to_networkx(self, network: InvoiceNetwork) -> nx.DiGraph:
        """Construye el grafo de NetworkX usado para calcular las métricas estructurales de G2."""
        raise NotImplementedError

    def to_pyg(self, network: InvoiceNetwork, node_features: pd.DataFrame) -> Data:
        """Construye la vista de grafo de PyTorch Geometric usada por G3.

        Args:
            network: red de contribuyentes, facturas y representantes legales.
            node_features: variables de nodo, indexadas por ``taxpayer_id`` en el orden de
                ``network.taxpayers``.

        Returns:
            Objeto ``Data`` con ``x``, ``edge_index``, ``edge_attr`` e ``y``.
        """
        raise NotImplementedError
