"""Ejecuta un experimento: ``python -m efd.experiments --config configs/experiments/<experimento>.yaml``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from efd.experiments.runner import DEFAULT_RUNS_ROOT, ExperimentRunner


def _format(stat: dict) -> str:
    std = "n/a" if stat["std"] is None else f"{stat['std']:.4f}"
    # Solo caracteres ASCII: la consola de Windows puede no mostrar "±".
    return f"{stat['mean']:.4f} +/- {std}"


def main(argv: list[str] | None = None) -> Path:
    """Ejecuta el experimento, imprime el resumen y devuelve la carpeta de la corrida."""
    parser = argparse.ArgumentParser(description="Ejecuta un experimento de detección de empresas fantasmas.")
    parser.add_argument("--config", type=Path, required=True, help="archivo YAML del experimento")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="carpeta raíz de las corridas")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    directory = ExperimentRunner.from_yaml(args.config, args.runs_root).run()
    elapsed = time.perf_counter() - started

    summary = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["summary"]
    print(f"Corrida guardada en {directory} ({elapsed:.1f} s)")
    for name, stat in summary["metrics"].items():
        print(f"  {name:<22} {_format(stat)}")
    for name, stat in summary["precision_at_k"].items():
        print(f"  {name:<22} {_format(stat)}")
    for pattern, stat in summary["recall_by_pattern"].items():
        print(f"  recall {pattern:<15} {_format(stat)}")
    return directory


if __name__ == "__main__":
    main()
