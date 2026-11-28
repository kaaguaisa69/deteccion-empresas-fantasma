"""Pruebas del GAT homogéneo sobre un Elliptic sintético (sin descargar datos)."""

import numpy as np
import pytest
import torch

from efd.elliptic import (
    EllipticData,
    HomogeneousGAT,
    fit_gat,
    predict_test,
    temporal_split,
)
from efd.evaluation.metrics import auc_pr


def synthetic(cross_step_edge: bool = False) -> EllipticData:
    rng = np.random.default_rng(0)
    per_step, steps = 20, 49
    n = per_step * steps
    time_step = np.repeat(np.arange(1, steps + 1), per_step)
    y = rng.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2])
    x = rng.normal(size=(n, 6))
    x[:, 0] += (y == 1) * 3.0
    edges = []
    for s in range(steps):
        nodes = np.arange(s * per_step, (s + 1) * per_step)
        src, dst = rng.choice(nodes, 30), rng.choice(nodes, 30)
        edges += [src, dst]
    edge_index = np.stack([np.concatenate(edges[0::2]), np.concatenate(edges[1::2])])
    if cross_step_edge:
        edge_index = np.concatenate([edge_index, [[0], [n - 1]]], axis=1)
    return EllipticData(torch.tensor(x, dtype=torch.float32), torch.tensor(edge_index), y, time_step)


def test_split_is_temporal_and_strictly_inductive() -> None:
    data = synthetic()
    split = temporal_split(data)
    assert split.train.x.shape[0] == 20 * 34 and split.test.x.shape[0] == 20 * 15
    train_y, test_y = split.train.y, split.test.y
    assert (train_y[split.train.idx] != 2).all() and (train_y[split.train.val_idx] != 2).all()
    assert (test_y[split.test.idx] != 2).all()
    train_steps = data.time_step[data.time_step <= 34]
    assert (train_steps[split.train.idx] <= 29).all()
    assert ((train_steps[split.train.val_idx] >= 30) & (train_steps[split.train.val_idx] <= 34)).all()
    assert split.train.edge_index.max() < split.train.x.shape[0]
    assert split.test.edge_index.max() < split.test.x.shape[0]


def test_cross_step_edge_stops_the_protocol() -> None:
    with pytest.raises(ValueError, match="pasos de tiempo distintos"):
        temporal_split(synthetic(cross_step_edge=True))


def test_gat_learns_and_is_deterministic_on_cpu() -> None:
    split = temporal_split(synthetic())
    params = {"hidden_channels": 4, "heads": 2, "epochs": 40, "patience": 10}
    first = fit_gat(split, seed=1, **params)
    again = fit_gat(split, seed=1, **params)
    scores = predict_test(first, split)
    assert np.array_equal(scores, predict_test(again, split))
    assert auc_pr(split.test.y[split.test.idx], scores) > 0.8
    assert 1 <= first.best_epoch <= len(first.history) <= 40


def test_test_graph_does_not_influence_the_fit() -> None:
    data = synthetic()
    changed = EllipticData(data.x.clone(), data.edge_index, data.y.copy(), data.time_step)
    changed.x[data.time_step > 34] += 10.0
    params = {"hidden_channels": 4, "heads": 2, "epochs": 15, "patience": 15}
    first = fit_gat(temporal_split(data), seed=1, **params)
    other = fit_gat(temporal_split(changed), seed=1, **params)
    assert first.history == other.history


def test_model_shape() -> None:
    out = HomogeneousGAT(6, 4, 2, 2, 0.1)(torch.randn(10, 6), torch.tensor([[0, 1], [1, 2]]))
    assert out.shape == (10,)
