"""Métricas estructurales de los nodos del grafo de facturación (grupo G2)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

STRUCTURAL_COLUMNS = (
    "pagerank",
    "clustering",
    "triangles",
    "core_number",
    "community_size",
    "community_density",
    "reciprocity",
    "scc_size",
    "shared_representative_count",
)


def to_undirected_by_count(graph: nx.DiGraph) -> nx.Graph:
    """Grafo no dirigido cuyas aristas suman el número de facturas en ambos sentidos."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for issuer, receiver, count in graph.edges(data="count", default=1):
        if undirected.has_edge(issuer, receiver):
            undirected[issuer][receiver]["count"] += count
        else:
            undirected.add_edge(issuer, receiver, count=count)
    return undirected


def node_reciprocity(graph: nx.DiGraph) -> pd.Series:
    """Proporción de las aristas de cada nodo (entrantes y salientes) que tienen la arista inversa.

    Vale 0 para los nodos sin aristas.
    """
    edges = pd.DataFrame(list(graph.edges), columns=["source", "target"])
    nodes = pd.Index(list(graph.nodes))
    if edges.empty:
        return pd.Series(0.0, index=nodes)
    reverse = set(zip(edges["target"], edges["source"]))
    edges["reciprocated"] = [pair in reverse for pair in zip(edges["source"], edges["target"])]
    incident = pd.concat(
        [edges[["source", "reciprocated"]].set_axis(["node", "reciprocated"], axis=1),
         edges[["target", "reciprocated"]].set_axis(["node", "reciprocated"], axis=1)]
    )
    return incident.groupby("node")["reciprocated"].mean().reindex(nodes, fill_value=0.0).astype(float)


def shared_representative_counts(representatives: pd.DataFrame, nodes: pd.Index) -> pd.Series:
    """Número de otros contribuyentes con los que cada contribuyente comparte representante legal."""
    pairs = representatives.merge(representatives, on="representative_id", suffixes=("", "_other"))
    pairs = pairs[pairs["taxpayer_id"] != pairs["taxpayer_id_other"]]
    counts = pairs.groupby("taxpayer_id")["taxpayer_id_other"].nunique()
    return counts.reindex(nodes, fill_value=0).astype(np.int64)


class StructuralMetricsCalculator:
    """Calcula las métricas de red de cada contribuyente.

    Sobre el grafo dirigido: PageRank ponderado por monto, reciprocidad y tamaño del componente
    fuertemente conexo. Sobre el grafo no dirigido: clustering, triángulos, k-core y tamaño y densidad
    de la comunidad Louvain. Con la tabla de representantes: empresas con representante compartido.
    """

    def __init__(self, seed: int) -> None:
        """
        Args:
            seed: semilla de la corrida, usada por el algoritmo de Louvain.
        """
        self.seed = seed

    def compute(self, graph: nx.DiGraph, representatives: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una fila por nodo, en el orden de ``graph.nodes``, y una columna por métrica.

        Args:
            graph: grafo dirigido emisor → receptor con atributos ``amount`` y ``count``.
            representatives: aristas representante → contribuyente.
        """
        nodes = pd.Index(list(graph.nodes), name="taxpayer_id")
        undirected = to_undirected_by_count(graph)

        community_size, community_density = {}, {}
        for community in nx.community.louvain_communities(undirected, weight="count", seed=self.seed):
            size = len(community)
            edges = undirected.subgraph(community).number_of_edges()
            density = 2 * edges / (size * (size - 1)) if size > 1 else 0.0
            for node in community:
                community_size[node] = size
                community_density[node] = density

        scc_size = {}
        for component in nx.strongly_connected_components(graph):
            for node in component:
                scc_size[node] = len(component)

        metrics = pd.DataFrame(
            {
                "pagerank": pd.Series(nx.pagerank(graph, weight="amount")),
                "clustering": pd.Series(nx.clustering(undirected)),
                "triangles": pd.Series(nx.triangles(undirected)),
                "core_number": pd.Series(nx.core_number(undirected)),
                "community_size": pd.Series(community_size),
                "community_density": pd.Series(community_density),
                "reciprocity": node_reciprocity(graph),
                "scc_size": pd.Series(scc_size),
                "shared_representative_count": shared_representative_counts(representatives, nodes),
            }
        ).reindex(nodes)
        return metrics[list(STRUCTURAL_COLUMNS)].astype(np.float64)
