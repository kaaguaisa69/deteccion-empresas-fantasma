"""Datos compartidos por las pruebas: un contenedor mínimo y un escenario de 2 000 contribuyentes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from efd.data import ExperimentData
from efd.generation import GeneratedScenario, GeneratorConfig, InvoiceNetworkGenerator
from generation_helpers import small_raw


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


@pytest.fixture(scope="session")
def small_config() -> GeneratorConfig:
    return GeneratorConfig.from_dict(small_raw())


@pytest.fixture(scope="session")
def small_scenario(small_config: GeneratorConfig) -> GeneratedScenario:
    return InvoiceNetworkGenerator(small_config).generate()


@pytest.fixture(scope="session")
def written_dir(small_config: GeneratorConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Carpeta del escenario pequeño escrito en disco."""
    return InvoiceNetworkGenerator(small_config).run(tmp_path_factory.mktemp("scenario"))
