"""Registro de corridas en ``results/runs/<AAAAMMDD-HHMMSS>_<experimento>/``."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from efd.reproducibility import git_state, library_versions

PROGRESS_DIR = "progress"


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
        self.info: dict[str, Any] = {}

    def create_run_dir(self, now: datetime | None = None) -> Path:
        """Crea la carpeta ``<AAAAMMDD-HHMMSS>_<experimento>`` de la corrida.

        Raises:
            FileExistsError: si ya existe una corrida con el mismo nombre y segundo.
        """
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        self.directory = self.root / f"{stamp}_{self.experiment_name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        return self.directory

    def start(self, config: Mapping[str, Any], scenario: Mapping[str, Any]) -> tuple[Path, bool]:
        """Crea la carpeta de la corrida o retoma una interrumpida del mismo experimento.

        Se retoma la corrida más reciente de ``experiment_name`` que tiene carpeta ``progress/``, no tiene
        ``metrics.json`` y cuya configuración y commit coinciden con los actuales. Una corrida interrumpida
        con otra configuración se ignora; con otro commit es un error, para no mezclar códigos distintos.
        Los cambios sin commit se registran en ``run_info.json`` (``dirty``) pero no impiden retomar.

        Returns:
            ``(carpeta, retomada)``.

        Raises:
            RuntimeError: si la corrida interrumpida se hizo con otro commit.
        """
        state = git_state()
        for candidate in sorted(self.root.glob(f"*_{self.experiment_name}"), reverse=True):
            if not (candidate / PROGRESS_DIR).is_dir() or (candidate / "metrics.json").exists():
                continue
            saved = yaml.safe_load((candidate / "config.yaml").read_text(encoding="utf-8"))
            if saved != yaml.safe_load(yaml.safe_dump(dict(config))):
                continue
            info = json.loads((candidate / "run_info.json").read_text(encoding="utf-8"))
            if info["commit"] != state["commit"]:
                raise RuntimeError(
                    f"{candidate} se interrumpió con el commit {info['commit']} y ahora hay "
                    f"{state['commit']}. No se mezclan códigos distintos: borra esa carpeta para empezar "
                    "de nuevo."
                )
            self.directory = candidate
            self.info = info
            return candidate, True
        directory = self.create_run_dir()
        (directory / PROGRESS_DIR).mkdir()
        self.info = {**state, "started_at": datetime.now().isoformat(timespec="seconds")}
        self.save_config(config)
        self.save_scenario(scenario)
        self._write_json("run_info.json", self.info)
        return directory, False

    def save_fold_checkpoint(
        self, seed: int, fold: int, result: Mapping[str, Any], predictions: pd.DataFrame, seconds: float
    ) -> None:
        """Guarda al instante una partición terminada (resultado, predicciones y tiempo de cálculo)."""
        stem = f"{PROGRESS_DIR}/seed{seed}_fold{fold}"
        predictions.to_parquet(self._path(f"{stem}.parquet"), index=False)
        temporary = self._path(f"{stem}.json.tmp")
        temporary.write_text(
            json.dumps({"result": dict(result), "seconds": seconds}, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self._path(f"{stem}.json"))  # el .json aparece solo con el .parquet listo

    def load_fold_checkpoint(self, seed: int, fold: int) -> tuple[dict[str, Any], pd.DataFrame, float] | None:
        """Devuelve ``(resultado, predicciones, segundos)`` de una partición ya guardada, o ``None``."""
        stem = f"{PROGRESS_DIR}/seed{seed}_fold{fold}"
        if not self._path(f"{stem}.json").exists():
            return None
        saved = json.loads(self._path(f"{stem}.json").read_text(encoding="utf-8"))
        return saved["result"], pd.read_parquet(self._path(f"{stem}.parquet")), saved["seconds"]

    def save_seed_meta(self, seed: int, meta: Mapping[str, Any]) -> None:
        """Guarda los tiempos de variables de una semilla (se conservan al retomar)."""
        self._write_json(f"{PROGRESS_DIR}/seed{seed}_meta.json", dict(meta))

    def load_seed_meta(self, seed: int) -> dict[str, Any] | None:
        """Devuelve los datos guardados de una semilla, o ``None`` si aún no hay."""
        path = self._path(f"{PROGRESS_DIR}/seed{seed}_meta.json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

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
