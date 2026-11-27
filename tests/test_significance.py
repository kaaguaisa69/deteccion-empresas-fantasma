import numpy as np
import pytest

from efd.evaluation.significance import (
    error_reduction,
    friedman,
    holm_adjust,
    paired_wilcoxon,
)


def test_holm_ejemplo():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_acotado_y_monotono():
    p = [0.5, 0.001, 0.2, 0.9, 0.04]
    adj = holm_adjust(p)
    assert max(adj) <= 1.0
    ordenado = [adj[i] for i in np.argsort(p)]
    assert all(x <= y for x, y in zip(ordenado, ordenado[1:], strict=False))


def test_wilcoxon_exacto():
    b = np.arange(15) * 0.1
    a = b + 1 + np.arange(15) * 0.001
    res = paired_wilcoxon(a, b)
    assert res["p_value"] == pytest.approx(2 / 2**15)
    assert res["mean_difference"] == pytest.approx(np.mean(a - b))


def test_wilcoxon_diferencias_cero():
    b = np.arange(15) * 0.1
    assert paired_wilcoxon(b, b)["p_value"] == 1.0


def test_friedman_ranking_identico():
    fila = np.arange(7, 0, -1, dtype=float)
    res = friedman(np.tile(fila, (15, 1)))
    assert res["mean_ranks"] == pytest.approx([1, 2, 3, 4, 5, 6, 7])
    assert res["p_value"] < 0.001


def test_error_reduction():
    assert error_reduction(0.9, 0.95) == pytest.approx(0.5)
