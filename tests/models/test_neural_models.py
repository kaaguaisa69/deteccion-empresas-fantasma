from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.parameter import UninitializedParameter
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

from efd.data import ExperimentData
from efd.evaluation.metrics import auc_pr
from efd.evaluation.splits import stratified_folds
from efd.experiments import ExperimentConfig, ExperimentRunner
from efd.experiments.assembly import assemble_experiment_data, load_scenario
from efd.graph.builder import REPRESENTATION_EDGE, REPRESENTATION_REVERSE_EDGE
from efd.models import create_detector
from efd.models.neural import (
    EDGE_TYPES,
    INVOICE_EDGE_TYPES,
    HeteroGNN,
    fit_standardizer,
    signed_log1p,
)

PARAMS = {"hidden_channels": 16, "epochs": 30, "patience": 5}


@pytest.fixture
def separable_data() -> ExperimentData:
    """400 empresas, 40 fantasmas; la variable ``signal`` separa las clases y ``noise`` no aporta."""
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40, dtype=np.int64), np.zeros(360, dtype=np.int64)]
    ids = pd.Index([f"T{i:03d}" for i in range(y.size)], name="taxpayer_id")
    features = pd.DataFrame(
        {"signal": y * 2.0 + rng.normal(0, 0.3, y.size), "noise": rng.normal(0, 1, y.size)}, index=ids
    )
    return ExperimentData(features=features, labels=pd.Series(y, index=ids, name="is_shell"))


def _split(data: ExperimentData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold = stratified_folds(data.y, n_splits=4, val_size=0.25, seed=0)[0]
    return fold.train_idx, fold.val_idx, fold.test_idx


def test_mlp_learns(separable_data: ExperimentData) -> None:
    train, val, test = _split(separable_data)
    detector = create_detector("mlp", seed=0, **PARAMS).fit(separable_data, train, val)
    assert not detector.requires_graph
    assert auc_pr(separable_data.y[test], detector.predict_proba(separable_data, test)) > 0.9
    assert 1 <= detector.best_epoch_ <= len(detector.history_) <= PARAMS["epochs"]
    assert set(detector.history_[0]) == {"epoch", "train_loss", "val_auc_pr"}


def test_mlp_is_deterministic_and_depends_on_seed(separable_data: ExperimentData) -> None:
    train, val, test = _split(separable_data)
    first = create_detector("mlp", seed=1, **PARAMS).fit(separable_data, train, val)
    again = create_detector("mlp", seed=1, **PARAMS).fit(separable_data, train, val)
    other = create_detector("mlp", seed=2, **PARAMS).fit(separable_data, train, val)
    scores = first.predict_proba(separable_data, test)
    assert np.array_equal(again.predict_proba(separable_data, test), scores)
    assert not np.array_equal(other.predict_proba(separable_data, test), scores)


def test_standardizer_uses_only_train_rows(separable_data: ExperimentData) -> None:
    x = signed_log1p(separable_data.features.to_numpy() * 1e6)
    train = np.arange(0, 300)
    mean, _ = fit_standardizer(x, train)
    np.testing.assert_allclose(
        mean, signed_log1p(separable_data.features.to_numpy() * 1e6)[train].mean(axis=0)
    )


def test_standardizer_replaces_zero_std() -> None:
    x = np.column_stack([np.arange(10.0), np.full(10, 7.0)])
    _, std = fit_standardizer(x, np.arange(10))
    assert std[1] == 1.0 and std[0] > 0


def test_signed_log1p_keeps_sign() -> None:
    np.testing.assert_allclose(signed_log1p(np.array([1 - np.e, 0.0, np.e - 1])), [-1.0, 0.0, 1.0])


_INV = ("taxpayer", "invoices", "taxpayer")
_REV = ("taxpayer", "rev_invoices", "taxpayer")
_REP = ("taxpayer", "represented_by", "representative")
_RREP = ("representative", "rev_represented_by", "taxpayer")


def _synthetic_hetero() -> HeteroData:
    g = torch.Generator().manual_seed(0)
    graph = HeteroData()
    graph["taxpayer"].x = torch.randn(50, 6, generator=g)
    graph["representative"].x = torch.randn(10, 1, generator=g)
    src = torch.randint(0, 50, (80,), generator=g)
    dst = torch.randint(0, 50, (80,), generator=g)
    graph[_INV].edge_index = torch.stack([src, dst])
    graph[_INV].edge_attr = torch.randn(80, 4, generator=g)
    graph[_REV].edge_index = torch.stack([dst, src])
    graph[_REV].edge_attr = torch.randn(80, 4, generator=g)
    rep = torch.randint(0, 10, (50,), generator=g)
    tax = torch.arange(50)
    graph[_REP].edge_index = torch.stack([tax, rep])
    graph[_RREP].edge_index = torch.stack([rep, tax])
    return graph


@pytest.mark.parametrize("conv", ["sage", "gat"])
def test_hetero_conv_runs_under_deterministic_algorithms(conv: str) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        graph = _synthetic_hetero()
        edges = (_INV, _REV, _REP, _RREP)
        if conv == "sage":
            layers = {edge: SAGEConv((-1, -1), 8) for edge in edges}
        else:
            layers = {
                edge: GATConv(
                    (-1, -1), 4, heads=2, add_self_loops=False, edge_dim=4 if edge in (_INV, _REV) else None
                )
                for edge in edges
            }
        hetero = HeteroConv(layers, aggr="sum")
        x_dict = {"taxpayer": graph["taxpayer"].x, "representative": graph["representative"].x}
        kwargs = {}
        if conv == "gat":
            kwargs["edge_attr_dict"] = {edge: graph[edge].edge_attr for edge in (_INV, _REV)}
        out = hetero(x_dict, graph.edge_index_dict, **kwargs)
        out["taxpayer"].sum().backward()
        assert out["taxpayer"].shape[0] == 50
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in hetero.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)


# --- GraphSAGE y GAT sobre el grafo heterogéneo ---------------------------------------------------

GNN_PARAMS = {
    "graphsage": {"hidden_channels": 8, "num_layers": 2, "epochs": 6, "patience": 6},
    "gat": {"hidden_channels": 4, "heads": 2, "num_layers": 2, "epochs": 6, "patience": 6},
}
GNNS = list(GNN_PARAMS)


@pytest.fixture(scope="module")
def graph_data(written_dir: Path) -> ExperimentData:
    scenario = load_scenario(written_dir)
    return assemble_experiment_data(scenario, "G1", {"transactional": {"round_multiple": 100}}, 0, True).data


@pytest.fixture(scope="module")
def fold(graph_data: ExperimentData):
    return stratified_folds(graph_data.y, 5, 0.2, 0)[0]


def _with_flipped_labels(data: ExperimentData, flipped: np.ndarray) -> ExperimentData:
    """Copia de ``data`` con las etiquetas de ``flipped`` invertidas, en ``labels`` y en el grafo."""
    y = data.y.copy()
    y[flipped] = 1 - y[flipped]
    graph = copy.deepcopy(data.require_graph())
    graph["taxpayer"].y = torch.tensor(y)
    return ExperimentData(features=data.features, labels=pd.Series(y, index=data.features.index), graph=graph)


@pytest.mark.parametrize("name", GNNS)
def test_gnn_is_deterministic(name: str, graph_data: ExperimentData, fold) -> None:
    first = create_detector(name, seed=3, **GNN_PARAMS[name]).fit(graph_data, fold.train_idx, fold.val_idx)
    again = create_detector(name, seed=3, **GNN_PARAMS[name]).fit(graph_data, fold.train_idx, fold.val_idx)
    assert first.requires_graph
    scores = first.predict_proba(graph_data, fold.test_idx)
    assert np.array_equal(again.predict_proba(graph_data, fold.test_idx), scores)
    assert set(first.history_[0]) == {"epoch", "train_loss", "val_auc_pr"}
    assert 1 <= first.best_epoch_ <= len(first.history_)


@pytest.mark.parametrize("name", GNNS)
def test_test_labels_do_not_influence_gnn(name: str, graph_data: ExperimentData, fold) -> None:
    flipped = _with_flipped_labels(graph_data, fold.test_idx)
    original = create_detector(name, seed=3, **GNN_PARAMS[name]).fit(graph_data, fold.train_idx, fold.val_idx)
    changed = create_detector(name, seed=3, **GNN_PARAMS[name]).fit(flipped, fold.train_idx, fold.val_idx)
    assert np.array_equal(
        original.predict_proba(graph_data, fold.test_idx), changed.predict_proba(flipped, fold.test_idx)
    )


@pytest.mark.parametrize("name", GNNS)
def test_validation_labels_do_not_enter_the_loss(name: str, graph_data: ExperimentData, fold) -> None:
    flipped = _with_flipped_labels(graph_data, fold.val_idx)
    params = GNN_PARAMS[name]  # patience == epochs: sin parada temprana
    first = create_detector(name, seed=3, **params).fit(graph_data, fold.train_idx, fold.val_idx)
    second = create_detector(name, seed=3, **params).fit(flipped, fold.train_idx, fold.val_idx)
    loss_a = [row["train_loss"] for row in first.history_]
    loss_b = [row["train_loss"] for row in second.history_]
    assert len(loss_a) == len(loss_b) == params["epochs"]
    assert loss_a == loss_b
    early = {**params, "epochs": 12, "patience": 1}
    first = create_detector(name, seed=3, **early).fit(graph_data, fold.train_idx, fold.val_idx)
    second = create_detector(name, seed=3, **early).fit(flipped, fold.train_idx, fold.val_idx)
    common = min(len(first.history_), len(second.history_))
    assert common >= 2
    assert [r["train_loss"] for r in first.history_[:common]] == [
        r["train_loss"] for r in second.history_[:common]
    ]


@pytest.mark.parametrize("name", GNNS)
def test_fit_does_not_modify_the_graph(name: str, graph_data: ExperimentData, fold) -> None:
    graph = graph_data.require_graph()
    before = copy.deepcopy(graph)
    create_detector(name, seed=3, **GNN_PARAMS[name]).fit(graph_data, fold.train_idx, fold.val_idx)
    assert torch.equal(graph["taxpayer"].x, before["taxpayer"].x)
    assert torch.equal(graph["representative"].x, before["representative"].x)
    for edge in EDGE_TYPES:
        assert torch.equal(graph[edge].edge_index, before[edge].edge_index)
    for edge in INVOICE_EDGE_TYPES:
        assert torch.equal(graph[edge].edge_attr, before[edge].edge_attr)


def _zero_edge_attr(data: ExperimentData) -> ExperimentData:
    graph = copy.deepcopy(data.require_graph())
    for edge in INVOICE_EDGE_TYPES:
        graph[edge].edge_attr = torch.zeros_like(graph[edge].edge_attr)
    return ExperimentData(features=data.features, labels=data.labels, graph=graph)


def test_edge_attr_is_read_only_by_gat(graph_data: ExperimentData, fold) -> None:
    zeroed = _zero_edge_attr(graph_data)
    sage = create_detector("graphsage", seed=3, **GNN_PARAMS["graphsage"]).fit(
        graph_data, fold.train_idx, fold.val_idx
    )
    assert np.array_equal(
        sage.predict_proba(graph_data, fold.test_idx), sage.predict_proba(zeroed, fold.test_idx)
    )
    assert not any(hasattr(conv, "edge_dim") for conv in sage.network.layers[0].convs.values())

    gat = create_detector("gat", seed=3, **GNN_PARAMS["gat"]).fit(graph_data, fold.train_idx, fold.val_idx)
    assert not np.array_equal(
        gat.predict_proba(graph_data, fold.test_idx), gat.predict_proba(zeroed, fold.test_idx)
    )
    for layer in gat.network.layers:
        for edge in INVOICE_EDGE_TYPES:
            assert layer.convs[edge].edge_dim == 4
        for edge in (REPRESENTATION_EDGE, REPRESENTATION_REVERSE_EDGE):
            assert layer.convs[edge].edge_dim is None


@pytest.mark.parametrize("conv", ["sage", "gat"])
def test_lazy_parameters_exist_after_initialization(conv: str, graph_data: ExperimentData) -> None:
    graph = graph_data.require_graph()
    x_dict = {"taxpayer": graph["taxpayer"].x, "representative": graph["representative"].x}
    edge_index = {edge: graph[edge].edge_index for edge in EDGE_TYPES}
    edge_attr = {edge: graph[edge].edge_attr for edge in INVOICE_EDGE_TYPES}
    network = HeteroGNN(conv, {nt: x.shape[1] for nt, x in x_dict.items()}, 4, 2, 0.5, heads=2)
    network.initialize_lazy(x_dict, edge_index, edge_attr)
    parameters = list(network.parameters())
    assert not any(isinstance(p, UninitializedParameter) for p in parameters)
    assert sum(p.numel() for p in parameters) > 0
    assert network(x_dict, edge_index, edge_attr).shape == (graph["taxpayer"].num_nodes,)


def test_runner_with_graphsage(written_dir: Path, tmp_path: Path) -> None:
    raw = {
        "experiment": {"name": "sage_test", "scenario": str(written_dir), "feature_set": "G1", "seeds": [1]},
        "model": {
            "name": "graphsage",
            "params": {"epochs": 3, "patience": 3, "num_threads": 2},
            "grid": [{"hidden_channels": 4}, {"hidden_channels": 8}],
        },
        "features": {"transactional": {"round_multiple": 100}},
        "splits": {"n_splits": 3, "val_size": 0.25},
        "threshold": {"strategy": "max_f1"},
        "evaluation": {
            "metrics": ["auc_pr", "auc_roc", "f1", "precision", "recall", "fpr"],
            "precision_at_k": [10],
            "subpopulations": {"sin_p1": ["isolated_emitter"]},
        },
    }
    threads = torch.get_num_threads()
    try:
        run = ExperimentRunner(ExperimentConfig.from_dict(raw), tmp_path).run()
    finally:
        torch.set_num_threads(threads)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["torch_num_threads"] == 2
    assert all(len(fold["tuning"]["val_auc_pr"]) == 2 for fold in metrics["folds"])
    assert "selected_counts" in metrics["summary"]["tuning"]
    assert "auc_pr" in metrics["summary"]["subpopulations"]["sin_p1"]["metrics"]
