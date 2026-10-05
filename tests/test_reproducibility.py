from __future__ import annotations

import random

import numpy as np
import pytest

from efd.reproducibility import set_global_seed


def _draw() -> tuple[float, float]:
    return random.random(), float(np.random.rand())


def test_same_seed_same_draws() -> None:
    set_global_seed(123)
    first = _draw()
    set_global_seed(123)
    assert _draw() == first


def test_seed_controls_torch_when_installed() -> None:
    torch = pytest.importorskip("torch")
    set_global_seed(5)
    first = torch.rand(3)
    set_global_seed(5)
    assert torch.equal(torch.rand(3), first)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError):
        set_global_seed(seed)  # type: ignore[arg-type]
