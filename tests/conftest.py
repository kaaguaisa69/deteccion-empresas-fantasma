"""Datos mínimos compartidos por las pruebas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efd.data import ExperimentData


@pytest.fixture
def small_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Veinte empresas, cinco de ellas fantasmas, con dos variables numéricas."""
    rng = np.random.default_rng(0)
    ids = pd.Index([f"E{i:03d}" for i in range(20)], name="taxpayer_id")
    features = pd.DataFrame(
        {"amount_total": rng.uniform(0, 100, 20), "degree": rng.integers(0, 10, 20)}, index=ids
    )
    labels = pd.Series([1] * 5 + [0] * 15, index=ids, name="is_shell")
    return features, labels


@pytest.fixture
def small_data(small_frame: tuple[pd.DataFrame, pd.Series]) -> ExperimentData:
    features, labels = small_frame
    return ExperimentData(features=features, labels=labels)
