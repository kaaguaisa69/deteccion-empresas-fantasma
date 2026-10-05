from __future__ import annotations

import numpy as np
import pytest

from efd.evaluation.metrics import (
    METRICS,
    auc_pr,
    auc_roc,
    compute_metrics,
    f1,
    false_positive_rate,
    precision,
    precision_at_k,
    recall,
)

Y_TRUE = np.array([1, 1, 0, 0, 0, 1, 0, 0])
Y_SCORE = np.array([0.9, 0.4, 0.8, 0.1, 0.3, 0.7, 0.2, 0.05])


def test_auc_pr_hand_computed() -> None:
    # Orden por puntuación: 1, 0, 1, 1, 0, 0, 0, 0; precisión en cada positivo: 1, 2/3 y 3/4.
    assert auc_pr(Y_TRUE, Y_SCORE) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)


def test_auc_pr_perfect_ranking_is_one() -> None:
    assert auc_pr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auc_roc_hand_computed() -> None:
    # Pares (positivo, negativo) correctamente ordenados: 13 de 15.
    assert auc_roc(Y_TRUE, Y_SCORE) == pytest.approx(13 / 15)


def test_label_metrics_hand_computed() -> None:
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 0])  # TP=2, FP=1, FN=1, TN=4
    assert precision(Y_TRUE, y_pred) == pytest.approx(2 / 3)
    assert recall(Y_TRUE, y_pred) == pytest.approx(2 / 3)
    assert f1(Y_TRUE, y_pred) == pytest.approx(2 / 3)
    assert false_positive_rate(Y_TRUE, y_pred) == pytest.approx(1 / 5)


def test_label_metrics_degenerate_cases_return_zero() -> None:
    assert precision([0, 1], [0, 0]) == 0.0
    assert recall([0, 0], [1, 0]) == 0.0
    assert false_positive_rate([1, 1], [1, 0]) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(Y_TRUE, Y_SCORE, 1) == 1.0
    assert precision_at_k(Y_TRUE, Y_SCORE, 3) == pytest.approx(2 / 3)
    assert precision_at_k(Y_TRUE, Y_SCORE, 8) == pytest.approx(3 / 8)


def test_precision_at_k_ties_are_deterministic() -> None:
    assert precision_at_k([1, 0, 0], [0.5, 0.5, 0.5], 1) == 1.0
    assert precision_at_k([0, 1, 0], [0.5, 0.5, 0.5], 1) == 0.0


@pytest.mark.parametrize("k", [0, 9, 2.5, True])
def test_precision_at_k_rejects_invalid_k(k: object) -> None:
    with pytest.raises(ValueError):
        precision_at_k(Y_TRUE, Y_SCORE, k)  # type: ignore[arg-type]


def test_score_metrics_require_both_classes() -> None:
    with pytest.raises(ValueError):
        auc_pr([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        auc_roc([1, 1], [0.1, 0.2])


@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [([1, 0], [0.5]), ([1, 2], [0.5, 0.5]), ([1, 0], [np.nan, 0.5]), ([], [])],
)
def test_invalid_inputs_are_rejected(y_true: list, y_score: list) -> None:
    with pytest.raises(ValueError):
        auc_pr(y_true, y_score)


def test_label_metrics_reject_probabilities() -> None:
    with pytest.raises(ValueError):
        f1([1, 0], [0.7, 0.2])


def test_registered_metric_names() -> None:
    assert set(METRICS.names()) == {"auc_pr", "auc_roc", "f1", "precision", "recall", "fpr"}


def test_compute_metrics_applies_threshold_and_k() -> None:
    results = compute_metrics(Y_TRUE, Y_SCORE, threshold=0.7, k_values=[2])
    assert results["auc_pr"] == pytest.approx(auc_pr(Y_TRUE, Y_SCORE))
    assert results["precision"] == pytest.approx(2 / 3)  # 0.9, 0.8 y 0.7 superan el umbral
    assert results["recall"] == pytest.approx(2 / 3)
    assert results["fpr"] == pytest.approx(1 / 5)
    assert results["precision_at_2"] == pytest.approx(1 / 2)


def test_compute_metrics_subset_and_unknown_name() -> None:
    assert set(compute_metrics(Y_TRUE, Y_SCORE, 0.5, metric_names=["auc_pr"])) == {"auc_pr"}
    with pytest.raises(KeyError):
        compute_metrics(Y_TRUE, Y_SCORE, 0.5, metric_names=["accuracy"])
