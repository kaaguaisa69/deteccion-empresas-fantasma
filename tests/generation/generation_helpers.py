"""Configuración pequeña (2 000 contribuyentes) para las pruebas del generador."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "generator.yaml"
SMALL_N_TAXPAYERS = 2000

with open(CONFIG_PATH, encoding="utf-8") as _handle:
    _BASE_RAW = yaml.safe_load(_handle)


def small_raw() -> dict[str, Any]:
    """Copia del YAML base con 2 000 contribuyentes, modificable por cada prueba."""
    raw = copy.deepcopy(_BASE_RAW)
    raw["population"]["n_taxpayers"] = SMALL_N_TAXPAYERS
    return raw
