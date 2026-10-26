"""Lectura de un escenario generado y armado de ``ExperimentData``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from efd.data import ExperimentData
from efd.features import FeatureContext, build_feature_set
from efd.generation.config import GeneratorConfig
from efd.graph.builder import InvoiceGraphBuilder
from efd.network import METADATA_FILE, InvoiceNetwork


@dataclass(frozen=True)
class Scenario:
    """Escenario generado: sus tablas y la configuración con que se produjo.

    Attributes:
        path: carpeta del escenario.
        network: tablas de contribuyentes, facturas y representantes.
        config: configuración del generador leída de ``metadata.json``.
        config_hash: huella de esa configuración.
    """

    path: Path
    network: InvoiceNetwork
    config: GeneratorConfig
    config_hash: str

    def feature_context(self, seed: int) -> FeatureContext:
        """Período del escenario y semilla de la corrida para los constructores de variables."""
        return FeatureContext(
            period_start=pd.Timestamp(self.config.period.start),
            period_end=pd.Timestamp(self.config.period.end),
            seed=seed,
        )


@dataclass(frozen=True)
class AssembledData:
    """Datos de una corrida.

    Attributes:
        data: vistas tabular y, si el detector la requiere, de grafo.
        patterns: patrón de fraude de cada nodo (nulo para legítimos), en el orden de ``data``. Solo lo
            usa el evaluador; queda fuera de ``ExperimentData`` para que ningún detector pueda leerlo.
    """

    data: ExperimentData
    patterns: np.ndarray


def load_scenario(path: Path) -> Scenario:
    """Lee las tablas y los metadatos de un escenario.

    Raises:
        ValueError: si la huella guardada no coincide con la de la configuración guardada.
    """
    path = Path(path)
    with open(path / METADATA_FILE, encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = GeneratorConfig.from_dict(metadata["config"])
    if config.config_hash() != metadata["config_hash"]:
        raise ValueError(f"La huella de {path / METADATA_FILE} no coincide con su configuración.")
    return Scenario(path=path, network=InvoiceNetwork.load(path), config=config, config_hash=metadata["config_hash"])


def assemble_experiment_data(
    scenario: Scenario,
    feature_set: str,
    feature_params: Mapping[str, Mapping[str, Any]],
    seed: int,
    with_graph: bool,
) -> AssembledData:
    """Construye las variables del grupo y, si se pide, la vista de grafo heterogénea.

    Args:
        scenario: escenario leído con ``load_scenario``.
        feature_set: grupo de variables (``G1`` o ``G2``).
        feature_params: parámetros por constructor de variables.
        seed: semilla de la corrida.
        with_graph: si se construye la vista de grafo (detectores con ``requires_graph``).
    """
    network = scenario.network
    context = scenario.feature_context(seed)
    features = build_feature_set(feature_set, network, context, feature_params)
    taxpayers = network.taxpayers.set_index("taxpayer_id")
    labels = taxpayers["is_shell"].astype(np.int64).rename("is_shell")
    graph = (
        InvoiceGraphBuilder().to_pyg(network, features, context.period_start, context.period_end)
        if with_graph
        else None
    )
    return AssembledData(
        data=ExperimentData(features=features, labels=labels, graph=graph),
        patterns=taxpayers["pattern"].to_numpy(dtype=object),
    )
