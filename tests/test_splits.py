from __future__ import annotations

import json

import numpy as np
import pytest

from efd.evaluation.splits import stratified_folds

Y = np.array([1] * 20 + [0] * 80)


def test_partitions_are_disjoint_and_cover_all_nodes() -> None:
    folds = stratified_folds(Y, n_splits=5, val_size=0.25, seed=1)
    assert len(folds) == 5
    for fold in folds:
        parts = [fold.train_idx, fold.val_idx, fold.test_idx]
        combined = np.concatenate(parts)
        assert combined.size == Y.size
        assert np.unique(combined).size == Y.size


def test_each_node_is_tested_exactly_once() -> None:
    folds = stratified_folds(Y, n_splits=5, val_size=0.25, seed=1)
    tested = np.sort(np.concatenate([fold.test_idx for fold in folds]))
    np.testing.assert_array_equal(tested, np.arange(Y.size))


def test_partitions_are_stratified() -> None:
    for fold in stratified_folds(Y, n_splits=5, val_size=0.25, seed=1):
        assert Y[fold.test_idx].sum() == 4
        assert Y[fold.val_idx].sum() == 4
        assert Y[fold.train_idx].sum() == 12


def test_same_seed_same_partitions_different_seed_different_partitions() -> None:
    first = stratified_folds(Y, n_splits=5, val_size=0.25, seed=7)
    second = stratified_folds(Y, n_splits=5, val_size=0.25, seed=7)
    other = stratified_folds(Y, n_splits=5, val_size=0.25, seed=8)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.test_idx, b.test_idx)
        np.testing.assert_array_equal(a.val_idx, b.val_idx)
    assert any(not np.array_equal(a.test_idx, c.test_idx) for a, c in zip(first, other))


def test_fold_is_json_serializable() -> None:
    fold = stratified_folds(Y, n_splits=5, val_size=0.25, seed=1)[0]
    restored = json.loads(json.dumps(fold.to_dict()))
    assert restored["index"] == 0
    assert restored["test_idx"] == fold.test_idx.tolist()


@pytest.mark.parametrize(
    ("y", "n_splits", "val_size"),
    [
        (Y, 1, 0.2),
        (Y, 5, 0.0),
        (Y, 5, 1.0),
        (np.array([1, 1, 0, 0, 0, 0]), 3, 0.2),
        (np.array([1, 2, 0, 0]), 2, 0.2),
    ],
)
def test_invalid_arguments(y: np.ndarray, n_splits: int, val_size: float) -> None:
    with pytest.raises(ValueError):
        stratified_folds(y, n_splits=n_splits, val_size=val_size, seed=0)
