"""Escritura de las tablas y metadatos de un escenario."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from efd.generation.config import GeneratorConfig
from efd.network import METADATA_FILE, TABLE_FILES, InvoiceNetwork

LIBRARIES = ("numpy", "pandas", "pyarrow", "networkx", "pyyaml")


def library_versions() -> dict[str, str]:
    """Versión de Python y de las librerías que intervienen en la generación."""
    versions = {"python": platform.python_version()}
    for library in LIBRARIES:
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "no instalada"
    return versions


def build_metadata(config: GeneratorConfig, network: InvoiceNetwork) -> dict[str, Any]:
    """Metadatos del escenario: configuración, semilla, huella, conteos y versiones.

    No incluye marcas de tiempo, para que la misma semilla y configuración produzcan archivos
    idénticos.
    """
    taxpayers = network.taxpayers
    shells = taxpayers[taxpayers["is_shell"] == 1]
    return {
        "scenario": config.scenario_name,
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "counts": {
            "taxpayers": len(taxpayers),
            "shells": len(shells),
            "shells_by_pattern": {
                name: int((shells["pattern"] == name).sum()) for name in config.patterns
            },
            "invoices": len(network.invoices),
            "representatives": int(network.representatives["representative_id"].nunique()),
        },
        "library_versions": library_versions(),
        "config": config.to_dict(),
    }


def write_scenario(network: InvoiceNetwork, metadata: dict[str, Any], directory: Path) -> Path:
    """Escribe las tablas en Parquet y los metadatos en JSON dentro de ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for table, file in TABLE_FILES.items():
        getattr(network, table).to_parquet(directory / file, index=False)
    with open(directory / METADATA_FILE, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return directory
