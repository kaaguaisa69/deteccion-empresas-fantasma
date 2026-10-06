"""Escenario pequeño compartido por las pruebas del generador."""

from __future__ import annotations

from pathlib import Path

import pytest

from efd.generation import GeneratedScenario, GeneratorConfig, InvoiceNetworkGenerator
from generation_helpers import small_raw


@pytest.fixture(scope="session")
def small_config() -> GeneratorConfig:
    return GeneratorConfig.from_dict(small_raw())


@pytest.fixture(scope="session")
def small_scenario(small_config: GeneratorConfig) -> GeneratedScenario:
    return InvoiceNetworkGenerator(small_config).generate()


@pytest.fixture(scope="session")
def written_dir(small_config: GeneratorConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return InvoiceNetworkGenerator(small_config).run(tmp_path_factory.mktemp("first"))
