from __future__ import annotations

import numpy as np
import pytest

from efd.data import ExperimentData
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


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "graphsage", "gat"])
def test_registered_detectors_are_interchangeable(name: str, small_data: ExperimentData) -> None:
    detector = create_detector(name, seed=0)
    assert isinstance(detector, BaseDetector)
    assert detector.name == name
    with pytest.raises(NotImplementedError):
        detector.fit(small_data, np.arange(0, 15), np.arange(15, 20))
