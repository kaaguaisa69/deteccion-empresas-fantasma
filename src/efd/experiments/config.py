"""Configuración de un experimento, cargada y validada desde YAML."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from efd.evaluation.metrics import METRICS
from efd.evaluation.threshold import THRESHOLD_POLICIES
from efd.features import FEATURE_BUILDERS, FEATURE_SETS
from efd.models import available_detectors

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ExperimentConfig:
    """Un experimento: un escenario, un grupo de variables, un modelo y una lista de semillas.

    Cada semilla repite el k-fold completo con sus propias particiones, modelo y cálculos aleatorios
    de las variables.

    Attributes:
        name: nombre del experimento; forma parte de la carpeta de la corrida.
        scenario: carpeta del escenario generado (``data/synthetic/<escenario>``).
        feature_set: grupo de variables (``G1`` o ``G2``).
        seeds: semillas de las repeticiones.
        model_name: detector registrado.
        model_params: hiperparámetros del detector.
        feature_params: parámetros por constructor de variables.
        n_splits: número de pliegues.
        val_size: fracción de los nodos fuera de prueba reservada para validación.
        threshold: política de umbral (``strategy`` y sus parámetros).
        metrics: métricas registradas que se calculan por pliegue.
        precision_at_k: valores de ``k`` para precision@k sobre todos los pliegues juntos.
        raw: diccionario original, que se guarda tal cual con la corrida.
    """

    name: str
    scenario: Path
    feature_set: str
    seeds: tuple[int, ...]
    model_name: str
    model_params: dict[str, Any]
    feature_params: dict[str, dict[str, Any]]
    n_splits: int
    val_size: float
    threshold: dict[str, Any]
    metrics: tuple[str, ...]
    precision_at_k: tuple[int, ...]
    raw: dict[str, Any] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not _SAFE_NAME.match(self.name):
            raise ValueError("El nombre del experimento solo admite letras, dígitos, '_', '-' y '.'.")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"Grupo de variables desconocido: '{self.feature_set}'.")
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("'seeds' debe ser una lista no vacía de enteros no negativos.")
        if self.model_name not in available_detectors():
            raise ValueError(f"Modelo desconocido: '{self.model_name}'.")
        unknown_builders = set(self.feature_params) - set(FEATURE_BUILDERS.names())
        if unknown_builders:
            raise ValueError(f"Constructores de variables desconocidos: {sorted(unknown_builders)}.")
        if self.threshold.get("strategy") not in THRESHOLD_POLICIES:
            raise ValueError(f"Política de umbral desconocida: '{self.threshold.get('strategy')}'.")
        unknown_metrics = set(self.metrics) - set(METRICS.names())
        if unknown_metrics:
            raise ValueError(f"Métricas desconocidas: {sorted(unknown_metrics)}.")
        if any(k < 1 for k in self.precision_at_k):
            raise ValueError("Los valores de precision_at_k deben ser positivos.")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        """Construye la configuración a partir del diccionario leído del YAML."""
        experiment = raw["experiment"]
        evaluation = raw["evaluation"]
        return cls(
            name=experiment["name"],
            scenario=Path(experiment["scenario"]),
            feature_set=experiment["feature_set"],
            seeds=tuple(experiment["seeds"]),
            model_name=raw["model"]["name"],
            model_params=dict(raw["model"].get("params") or {}),
            feature_params={name: dict(params or {}) for name, params in (raw.get("features") or {}).items()},
            n_splits=raw["splits"]["n_splits"],
            val_size=raw["splits"]["val_size"],
            threshold=dict(raw["threshold"]),
            metrics=tuple(evaluation["metrics"]),
            precision_at_k=tuple(evaluation.get("precision_at_k") or ()),
            raw=dict(raw),
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        """Lee y valida un archivo YAML de experimento."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))
