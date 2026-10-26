from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efd.data import ExperimentData
from efd.evaluation.metrics import auc_pr
from efd.evaluation.splits import stratified_folds
from efd.models import BaseDetector, available_detectors, create_detector


class _PrevalenceDetector(BaseDetector):
    """Detector trivial que predice la prevalencia de entrenamiento para todas las empresas."""

    name = "prevalence"

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        _, y = data.tabular(train_idx)
        self.rate = float(y.mean())

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        return np.full(idx.size, self.rate)


class _BrokenDetector(_PrevalenceDetector):
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        return self.output


def test_fit_and_predict_contract(small_data: ExperimentData) -> None:
    detector = _PrevalenceDetector().fit(small_data, np.arange(0, 15), np.arange(15, 20))
    scores = detector.predict_proba(small_data, [3, 17])
    assert scores.shape == (2,)
    np.testing.assert_allclose(scores, 5 / 15)


def test_fit_rejects_overlapping_or_empty_partitions(small_data: ExperimentData) -> None:
    with pytest.raises(ValueError, match="solapan"):
        _PrevalenceDetector().fit(small_data, [0, 1, 2], [2, 3])
    with pytest.raises(ValueError, match="vacías"):
        _PrevalenceDetector().fit(small_data, [0, 1, 2], [])


@pytest.mark.parametrize(
    "output", [np.array([0.1, 0.2, 0.3]), np.array([0.1, 1.5]), np.array([0.1, np.nan])]
)
def test_predict_proba_rejects_invalid_output(small_data: ExperimentData, output: np.ndarray) -> None:
    detector = _BrokenDetector(output)
    with pytest.raises(ValueError):
        detector.predict_proba(small_data, [0, 1])


def test_concrete_detectors_are_registered() -> None:
    assert {"random_forest", "xgboost", "graphsage", "gat"} <= set(available_detectors())


@pytest.mark.parametrize("name", ["graphsage", "gat"])
def test_graph_detectors_are_pending(name: str, small_data: ExperimentData) -> None:
    detector = create_detector(name, seed=0)
    assert isinstance(detector, BaseDetector)
    assert detector.requires_graph
    with pytest.raises(NotImplementedError):
        detector.fit(small_data, np.arange(0, 15), np.arange(15, 20))


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


@pytest.mark.parametrize(
    ("name", "params"),
    [("random_forest", {"n_estimators": 50}), ("xgboost", {"n_estimators": 50})],
)
def test_tabular_detectors_learn_and_are_deterministic(
    name: str, params: dict, separable_data: ExperimentData
) -> None:
    train, val, test = _split(separable_data)
    detector = create_detector(name, seed=3, **params).fit(separable_data, train, val)
    assert not detector.requires_graph
    scores = detector.predict_proba(separable_data, test)
    assert auc_pr(separable_data.y[test], scores) > 0.9
    again = create_detector(name, seed=3, **params).fit(separable_data, train, val)
    np.testing.assert_array_equal(again.predict_proba(separable_data, test), scores)


def test_random_forest_balances_classes_by_default() -> None:
    assert create_detector("random_forest", seed=0).hyperparams["class_weight"] == "balanced"
    custom = create_detector("random_forest", seed=0, class_weight=None)
    assert custom.hyperparams["class_weight"] is None


def test_xgboost_scale_pos_weight(separable_data: ExperimentData) -> None:
    train, val, _ = _split(separable_data)
    y_train = separable_data.y[train]
    computed = create_detector("xgboost", seed=0, n_estimators=5).fit(separable_data, train, val)
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    assert computed.model.get_params()["scale_pos_weight"] == pytest.approx(expected)
    fixed = create_detector("xgboost", seed=0, n_estimators=5, scale_pos_weight=2.0)
    fixed.fit(separable_data, train, val)
    assert fixed.model.get_params()["scale_pos_weight"] == 2.0


def test_untrained_detector_cannot_predict(separable_data: ExperimentData) -> None:
    with pytest.raises(RuntimeError):
        create_detector("random_forest", seed=0).predict_proba(separable_data, [0, 1])
