"""Vista de grafo heterogénea para G3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efd.data import TAXPAYER_NODE_TYPE, ExperimentData
from efd.features import create_feature_builder
from efd.graph.builder import (
    INVOICE_EDGE,
    INVOICE_REVERSE_EDGE,
    REPRESENTATION_EDGE,
    REPRESENTATION_REVERSE_EDGE,
    REPRESENTATIVE_NODE_TYPE,
    InvoiceGraphBuilder,
)

from tiny_network import PERIOD_END, PERIOD_START, tiny_context, tiny_network

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return create_feature_builder("transactional").build(tiny_network(), tiny_context())


@pytest.fixture(scope="module")
def hetero(features: pd.DataFrame):
    return InvoiceGraphBuilder().to_pyg(tiny_network(), features, PERIOD_START, PERIOD_END)


def test_taxpayer_nodes(hetero, features: pd.DataFrame) -> None:
    nodes = hetero[TAXPAYER_NODE_TYPE]
    np.testing.assert_allclose(nodes.x.numpy(), features.to_numpy(np.float32))
    assert nodes.y.tolist() == [0, 1, 0, 0]


def test_invoice_edges_aggregate_pairs(hetero) -> None:
    edges = hetero[INVOICE_EDGE]
    pairs = [tuple(pair) for pair in edges.edge_index.t().tolist()]
    assert pairs == [(0, 1), (0, 2), (1, 0), (2, 0)]  # A→B, A→C, B→A, C→A
    a_to_b = edges.edge_attr[0].tolist()
    span = (PERIOD_END - PERIOD_START).days
    assert a_to_b == pytest.approx(
        [2, np.log1p(400), (pd.Timestamp("2025-03-10") - PERIOD_START).days / span,
         (pd.Timestamp("2025-03-20") - PERIOD_START).days / span],
        rel=1e-6,
    )
    assert edges.edge_attr[3, 3].item() == pytest.approx(1.0)  # C→A el último día del período


def test_reverse_edges(hetero) -> None:
    assert torch.equal(hetero[INVOICE_REVERSE_EDGE].edge_index, hetero[INVOICE_EDGE].edge_index.flip(0))
    assert torch.equal(hetero[INVOICE_REVERSE_EDGE].edge_attr, hetero[INVOICE_EDGE].edge_attr)
    assert torch.equal(
        hetero[REPRESENTATION_REVERSE_EDGE].edge_index, hetero[REPRESENTATION_EDGE].edge_index.flip(0)
    )


def test_representative_nodes(hetero) -> None:
    assert hetero[REPRESENTATIVE_NODE_TYPE].x.tolist() == [[1.0], [1.0]]
    # RL1 → A, RL1 → C, RL2 → D
    assert hetero[REPRESENTATION_EDGE].edge_index.tolist() == [[0, 0, 1], [0, 2, 3]]


def test_experiment_data_accepts_hetero_graph(hetero, features: pd.DataFrame) -> None:
    labels = pd.Series([0, 1, 0, 0], index=features.index, name="is_shell")
    data = ExperimentData(features=features, labels=labels, graph=hetero)
    assert data.require_graph() is hetero
    with pytest.raises(ValueError, match="etiquetas"):
        ExperimentData(features=features, labels=labels.replace({1: 0, 0: 1}), graph=hetero)


def test_node_features_must_follow_taxpayer_order(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        InvoiceGraphBuilder().to_pyg(tiny_network(), features.iloc[::-1], PERIOD_START, PERIOD_END)
