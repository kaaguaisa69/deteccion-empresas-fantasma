"""Registro de corridas en ``results/runs/<AAAAMMDD-HHMMSS>_<experimento>/``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from efd.reproducibility import library_versions


class RunRecorder:
    """Guarda todo lo necesario para auditar y repetir una corrida.

    Cada corrida tiene su propia carpeta, que nunca se sobrescribe, con la configuración usada, el
    escenario, las particiones, las métricas, las predicciones y las versiones de las librerías.
    """

    def __init__(self, root: Path, experiment_name: str) -> None:
        """
        Args:
            root: carpeta raíz de las corridas (``results/runs``).
            experiment_name: nombre del experimento declarado en la configuración.
        """
        self.root = Path(root)
        self.experiment_name = experiment_name
        self.directory: Path | None = None

    def create_run_dir(self, now: datetime | None = None) -> Path:
        """Crea la carpeta ``<AAAAMMDD-HHMMSS>_<experimento>`` de la corrida.

        Raises:
            FileExistsError: si ya existe una corrida con el mismo nombre y segundo.
        """
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        self.directory = self.root / f"{stamp}_{self.experiment_name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        return self.directory

    def _path(self, file: str) -> Path:
        if self.directory is None:
            raise RuntimeError("Primero hay que crear la carpeta de la corrida.")
        return self.directory / file

    def _write_json(self, file: str, content: Any) -> None:
        with open(self._path(file), "w", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def save_config(self, config: Mapping[str, Any]) -> None:
        """Guarda la configuración del experimento en YAML."""
        with open(self._path("config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config), handle, allow_unicode=True, sort_keys=False)

    def save_scenario(self, scenario: Mapping[str, Any]) -> None:
        """Guarda la ruta del escenario y su huella de configuración."""
        self._write_json("scenario.json", dict(scenario))

    def save_folds(self, folds: Mapping[str, Any]) -> None:
        """Guarda los índices de nodo de cada pliegue, por semilla."""
        self._write_json("folds.json", dict(folds))

    def save_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Guarda las métricas por semilla y pliegue, precision@k por semilla y el resumen."""
        self._write_json("metrics.json", dict(metrics))

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        """Guarda las predicciones de prueba de todos los pliegues y semillas."""
        predictions.to_parquet(self._path("predictions.parquet"), index=False)

    def save_environment(self) -> None:
        """Guarda la versión de Python y de cada librería del stack."""
        self._write_json("environment.json", library_versions())
