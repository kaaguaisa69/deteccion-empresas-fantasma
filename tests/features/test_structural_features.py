"""Métricas de red (G2) sobre grafos con resultado conocido y prueba de fuga de etiquetas."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from efd.features import build_feature_set
from efd.graph.builder import InvoiceGraphBuilder
from efd.graph.structural import STRUCTURAL_COLUMNS, StructuralMetricsCalculator
from efd.network import InvoiceNetwork

from tiny_network import tiny_context, tiny_network

NO_REPRESENTATIVES = pd.DataFrame({"representative_id": [], "taxpayer_id": []})


@pytest.fixture(scope="module")
def tiny_metrics() -> pd.DataFrame:
    network = tiny_network()
    graph = InvoiceGraphBuilder().to_networkx(network)
    return StructuralMetricsCalculator(seed=0).compute(graph, network.representatives)


def test_networkx_graph_aggregates_pairs() -> None:
    graph = InvoiceGraphBuilder().to_networkx(tiny_network())
    assert list(graph.nodes) == ["A", "B", "C", "D"]
    assert graph["A"]["B"] == {"amount": 400.0, "count": 2}
    assert graph.number_of_edges() == 4


def test_tiny_network_metrics(tiny_metrics: pd.DataFrame) -> None:
    assert list(tiny_metrics.columns) == list(STRUCTURAL_COLUMNS)
    assert list(tiny_metrics.index) == ["A", "B", "C", "D"]
    assert tiny_metrics["pagerank"].sum() == pytest.approx(1.0)
    assert tiny_metrics["pagerank"].idxmax() == "A"
    assert tiny_metrics["clustering"].tolist() == [0, 0, 0, 0]
    assert tiny_metrics["triangles"].tolist() == [0, 0, 0, 0]
    assert tiny_metrics["core_number"].tolist() == [1, 1, 1, 0]
    assert tiny_metrics["reciprocity"].tolist() == [1, 1, 1, 0]
    assert tiny_metrics["scc_size"].tolist() == [3, 3, 3, 1]
    assert tiny_metrics["shared_representative_count"].tolist() == [1, 0, 1, 0]
    assert tiny_metrics.loc["D", ["community_size", "community_density"]].tolist() == [1, 0]


def test_directed_triangle_metrics() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(["X", "Y", "Z", "W"])
    graph.add_edges_from([("X", "Y"), ("Y", "Z"), ("Z", "X")], amount=10.0, count=1)
    metrics = StructuralMetricsCalculator(seed=0).compute(graph, NO_REPRESENTATIVES)
    assert metrics.loc[["X", "Y", "Z"], "clustering"].tolist() == [1, 1, 1]
    assert metrics.loc[["X", "Y", "Z"], "triangles"].tolist() == [1, 1, 1]
    assert metrics.loc[["X", "Y", "Z"], "core_number"].tolist() == [2, 2, 2]
    assert metrics.loc[["X", "Y", "Z"], "scc_size"].tolist() == [3, 3, 3]
    assert metrics["reciprocity"].tolist() == [0, 0, 0, 0]
    assert metrics.loc["X", "community_size"] == 3
    assert metrics.loc["X", "community_density"] == pytest.approx(1.0)


def test_community_detection_is_seeded(written_dir) -> None:
    network = InvoiceNetwork.load(written_dir)
    first = build_feature_set("G2", network, tiny_context(seed=5))
    second = build_feature_set("G2", network, tiny_context(seed=5))
    pd.testing.assert_frame_equal(first, second)


def test_g2_extends_g1(written_dir) -> None:
    network = InvoiceNetwork.load(written_dir)
    g1 = build_feature_set("G1", network, tiny_context())
    g2 = build_feature_set("G2", network, tiny_context())
    assert list(g2.columns) == list(g1.columns) + list(STRUCTURAL_COLUMNS)
    pd.testing.assert_frame_equal(g2[g1.columns], g1)
    assert not g2.isna().any().any()


@pytest.mark.parametrize("feature_set", ["G1", "G2"])
def test_features_do_not_depend_on_labels(written_dir, feature_set: str) -> None:
    """Permutar ``is_shell`` y ``pattern`` no puede cambiar ninguna variable."""
    network = InvoiceNetwork.load(written_dir)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(network.taxpayers))
    shuffled = network.taxpayers.assign(
        is_shell=network.taxpayers["is_shell"].to_numpy()[order],
        pattern=network.taxpayers["pattern"].to_numpy()[order],
    )
    assert not shuffled["is_shell"].equals(network.taxpayers["is_shell"])
    permuted = InvoiceNetwork(shuffled, network.invoices, network.representatives)
    pd.testing.assert_frame_equal(
        build_feature_set(feature_set, network, tiny_context()),
        build_feature_set(feature_set, permuted, tiny_context()),
    )
