from __future__ import annotations

import numpy as np
import pytest

from efd.evaluation.metrics import f1
from efd.evaluation.threshold import (
    THRESHOLD_POLICIES,
    FixedThreshold,
    MaxF1Threshold,
    binarize,
    create_threshold_policy,
)


def test_binarize_uses_greater_or_equal() -> None:
    np.testing.assert_array_equal(binarize([0.2, 0.5, 0.7], 0.5), [0, 1, 1])


def test_max_f1_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 200)
    y_score = np.clip(0.3 * y_true + rng.uniform(0, 0.7, 200), 0, 1)

    threshold = MaxF1Threshold().select(y_true, y_score)
    best = max(f1(y_true, binarize(y_score, t)) for t in np.unique(y_score))
    assert f1(y_true, binarize(y_score, threshold)) == pytest.approx(best)


def test_max_f1_separable_scores() -> None:
    threshold = MaxF1Threshold().select(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.6, 0.9]))
    assert threshold == pytest.approx(0.6)


def test_max_f1_prefers_highest_threshold_on_ties() -> None:
    # Con umbral 0.8 (TP=1, FP=0, FN=1) y con 0.4 (TP=2, FP=2, FN=0) F1 es 2/3 en ambos casos.
    y_true = np.array([1, 0, 0, 1])
    y_score = np.array([0.8, 0.6, 0.5, 0.4])
    assert MaxF1Threshold().select(y_true, y_score) == pytest.approx(0.8)


def test_max_f1_requires_positives() -> None:
    with pytest.raises(ValueError):
        MaxF1Threshold().select(np.array([0, 0]), np.array([0.2, 0.4]))


def test_fixed_threshold() -> None:
    assert FixedThreshold(0.3).select(np.array([0, 1]), np.array([0.1, 0.9])) == 0.3
    with pytest.raises(ValueError):
        FixedThreshold(1.5)


def test_policies_created_by_name() -> None:
    assert set(THRESHOLD_POLICIES.names()) == {"max_f1", "fixed"}
    assert isinstance(create_threshold_policy(), MaxF1Threshold)
    policy = create_threshold_policy("fixed", value=0.4)
    assert isinstance(policy, FixedThreshold) and policy.value == 0.4
