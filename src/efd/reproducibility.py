"""Control de semillas aleatorias y registro de versiones de librerías.

La semilla se declara una sola vez en la configuración del experimento y se propaga desde aquí a todas
las fuentes de aleatoriedad del proceso. Las versiones de las librerías se guardan con cada escenario y
cada corrida, porque un mismo código puede producir resultados distintos con versiones distintas.
"""

from __future__ import annotations

import platform
import random
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

import numpy as np

STACK_LIBRARIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "torch",
    "torch_geometric",
    "networkx",
    "shap",
    "pyyaml",
    "pyarrow",
)


def library_versions() -> dict[str, str]:
    """Versión de Python y de cada librería del stack."""
    versions = {"python": platform.python_version()}
    for library in STACK_LIBRARIES:
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "no instalada"
    return versions


def set_global_seed(seed: int) -> None:
    """Fija la semilla de ``random``, NumPy y, si está instalado, PyTorch (CPU y CUDA).

    Args:
        seed: entero no negativo tomado de la configuración del experimento.

    Raises:
        ValueError: si la semilla es negativa o no es entera.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"La semilla debe ser un entero no negativo; se recibió {seed!r}.")
    random.seed(seed)
    np.random.seed(seed)
    if find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
