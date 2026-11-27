"""Funciones estadísticas puras para el análisis de significancia."""

from collections.abc import Sequence

import numpy as np
from scipy import stats


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Ajusta valores p con el método de Holm-Bonferroni.

    Ordena los p de menor a mayor y aplica
    ``ajustado_k = máx(ajustado_{k-1}, mín(1, (m - k + 1) * p_k))``.
    Devuelve los valores ajustados en el orden original.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for k, idx in enumerate(order):
        value = min(1.0, (m - k) * float(p_values[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def friedman(scores: np.ndarray) -> dict:
    """Prueba de Friedman sobre puntajes de forma ``(n_mediciones, n_modelos)``.

    Mayor puntaje es mejor: el rango 1 corresponde al mayor puntaje y los
    empates reciben rango medio. Devuelve ``statistic``, ``p_value`` y
    ``mean_ranks`` (una entrada por modelo).
    """
    scores = np.asarray(scores, dtype=float)
    result = stats.friedmanchisquare(*[scores[:, j] for j in range(scores.shape[1])])
    ranks = np.apply_along_axis(stats.rankdata, 1, -scores)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_ranks": [float(r) for r in ranks.mean(axis=0)],
    }


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Prueba de Wilcoxon de rangos con signo, bilateral, sobre pares ``a``, ``b``.

    Si todas las diferencias son 0, el valor p es 1.0. Devuelve ``statistic``,
    ``p_value`` y ``mean_difference`` (media de ``a - b``).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_difference": 0.0}
    result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_difference": float(diff.mean()),
    }


def error_reduction(mean_baseline: float, mean_candidate: float) -> float:
    """Reducción relativa del error (H1) respecto de la línea base.

    El error es ``1 - media de AUC-PR``; la reducción es
    ``(error_base - error_candidato) / error_base``.
    """
    error_baseline = 1.0 - mean_baseline
    error_candidate = 1.0 - mean_candidate
    return (error_baseline - error_candidate) / error_baseline
