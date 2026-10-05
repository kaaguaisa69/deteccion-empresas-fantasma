"""Registro de ejecuciones en ``results/runs/<AAAAMMDD-HHMMSS>_<experimento>/``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class RunRecorder:
    """Guarda todo lo necesario para auditar y repetir una ejecución.

    Cada ejecución tiene su propia carpeta con la configuración usada, la semilla, las particiones,
    las métricas en JSON y las versiones de las librerías.
    """

    def __init__(self, root: Path, experiment_name: str) -> None:
        """
        Args:
            root: carpeta raíz de las ejecuciones (``results/runs``).
            experiment_name: nombre del experimento declarado en la configuración.
        """
        self.root = root
        self.experiment_name = experiment_name

    def create_run_dir(self) -> Path:
        """Crea y devuelve la carpeta ``<AAAAMMDD-HHMMSS>_<experimento>`` de la ejecución."""
        raise NotImplementedError

    def save_config(self, config: Mapping[str, Any]) -> None:
        """Guarda la configuración efectiva, incluida la semilla, en YAML."""
        raise NotImplementedError

    def save_folds(self, folds: list[Mapping[str, Any]]) -> None:
        """Guarda los índices de nodo de cada pliegue en JSON."""
        raise NotImplementedError

    def save_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Guarda las métricas por grupo, modelo y pliegue en JSON."""
        raise NotImplementedError

    def save_environment(self) -> None:
        """Guarda la versión de Python y de cada librería del stack en JSON."""
        raise NotImplementedError
