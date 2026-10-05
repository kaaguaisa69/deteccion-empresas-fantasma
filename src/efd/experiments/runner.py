"""Orquestador de experimentos.

Lee la configuración, genera los datos, construye ``ExperimentData``, recorre los pliegues y, para
cada grupo y modelo, entrena, elige el umbral en validación, evalúa en prueba y registra la ejecución.
Solo depende de las abstracciones (``BaseDetector``, ``BaseExplainer``, ``BaseFeatureBuilder``,
``ThresholdPolicy``) y de los registros por nombre.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ExperimentRunner:
    """Ejecuta un experimento completo descrito por un archivo YAML de ``configs/``."""

    def __init__(self, config: Mapping[str, Any], runs_root: Path = Path("results/runs")) -> None:
        """
        Args:
            config: configuración del experimento ya cargada.
            runs_root: carpeta donde se crea la carpeta de cada ejecución.
        """
        self.config = config
        self.runs_root = runs_root

    @classmethod
    def from_yaml(cls, path: Path, runs_root: Path = Path("results/runs")) -> ExperimentRunner:
        """Crea el orquestador a partir de un archivo de configuración YAML."""
        raise NotImplementedError

    def run(self) -> Path:
        """Ejecuta el experimento y devuelve la carpeta de la ejecución."""
        raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de línea de comandos: ``python -m efd.experiments.runner --config <yaml>``."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
