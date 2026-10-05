from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efd.data import ExperimentData


def test_views_share_positional_order(small_data: ExperimentData) -> None:
    X, y = small_data.tabular([0, 7, 19])
    assert list(X.index) == ["E000", "E007", "E019"]
    np.testing.assert_array_equal(y, [1, 0, 0])
    assert small_data.n_nodes == 20


def test_rejects_misaligned_labels(small_frame: tuple[pd.DataFrame, pd.Series]) -> None:
    features, labels = small_frame
    with pytest.raises(ValueError, match="mismo índice"):
        ExperimentData(features=features, labels=labels.iloc[::-1])


def test_rejects_duplicate_ids(small_frame: tuple[pd.DataFrame, pd.Series]) -> None:
    features, labels = small_frame
    duplicated = features.index.to_list()
    duplicated[1] = duplicated[0]
    features = features.set_axis(duplicated)
    with pytest.raises(ValueError, match="únicos"):
        ExperimentData(features=features, labels=labels.set_axis(duplicated))


def test_rejects_non_binary_labels(small_frame: tuple[pd.DataFrame, pd.Series]) -> None:
    features, labels = small_frame
    with pytest.raises(ValueError, match="0 y 1"):
        ExperimentData(features=features, labels=labels.replace(1, 2))


@pytest.mark.parametrize("idx", [[0, 20], [-1], [1, 1], [[0, 1]], [0.5]])
def test_check_idx_rejects_invalid_positions(small_data: ExperimentData, idx: list) -> None:
    with pytest.raises(ValueError):
        small_data.check_idx(idx)


def test_select_features_keeps_labels_and_order(small_data: ExperimentData) -> None:
    subset = small_data.select_features(["amount_total"])
    assert list(subset.features.columns) == ["amount_total"]
    assert subset.node_ids.equals(small_data.node_ids)
    np.testing.assert_array_equal(subset.y, small_data.y)
    with pytest.raises(KeyError):
        small_data.select_features(["inexistente"])


def test_require_graph_without_graph(small_data: ExperimentData) -> None:
    with pytest.raises(ValueError):
        small_data.require_graph()


def test_graph_view_must_match_tabular_view(small_frame: tuple[pd.DataFrame, pd.Series]) -> None:
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    features, labels = small_frame
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    y = torch.tensor(labels.to_numpy())

    graph = pyg_data.Data(edge_index=edge_index, y=y, num_nodes=20)
    data = ExperimentData(features=features, labels=labels, graph=graph)
    assert data.require_graph() is graph

    with pytest.raises(ValueError, match="nodos"):
        ExperimentData(
            features=features,
            labels=labels,
            graph=pyg_data.Data(edge_index=edge_index, num_nodes=19),
        )
    with pytest.raises(ValueError, match="etiquetas"):
        ExperimentData(
            features=features,
            labels=labels,
            graph=pyg_data.Data(edge_index=edge_index, y=1 - y, num_nodes=20),
        )
