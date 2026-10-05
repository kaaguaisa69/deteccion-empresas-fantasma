"""Control de semillas aleatorias.

La semilla se declara una sola vez en la configuración del experimento y se propaga desde aquí a todas
las fuentes de aleatoriedad del proceso.
"""

from __future__ import annotations

import random
from importlib.util import find_spec

import numpy as np


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
