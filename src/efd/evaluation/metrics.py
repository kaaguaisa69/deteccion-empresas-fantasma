"""Única fuente de métricas de evaluación.

Cada métrica se registra con un nombre y declara si recibe probabilidades o predicciones binarias.
``compute_metrics`` calcula las métricas pedidas por la configuración sin conocerlas de antemano, por lo
que agregar una métrica nueva solo requiere registrarla aquí.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from efd.evaluation.threshold import binarize
from efd.registry import Registry

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Metric:
    """Métrica registrada.

    Attributes:
        name: nombre usado en la configuración y en el JSON de resultados.
        fn: función ``(y_true, y) -> float``.
        uses_scores: ``True`` si ``y`` son probabilidades; ``False`` si son predicciones binarias.
    """

    name: str
    fn: MetricFn
    uses_scores: bool


METRICS: Registry[Metric] = Registry("métrica")


def register_metric(name: str, *, uses_scores: bool) -> Callable[[MetricFn], MetricFn]:
    """Decorador que registra una función de métrica bajo ``name``."""

    def decorator(fn: MetricFn) -> MetricFn:
        METRICS.add(name, Metric(name=name, fn=fn, uses_scores=uses_scores))
        return fn

    return decorator


def _check_labels(y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true debe ser un arreglo 1-D no vacío.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true solo puede contener 0 y 1.")
    return y_true.astype(np.int64)


def _check_pair(y_true: np.ndarray, y: np.ndarray, *, binary: bool) -> tuple[np.ndarray, np.ndarray]:
    y_true = _check_labels(y_true)
    y = np.asarray(y, dtype=np.float64)
    if y.shape != y_true.shape:
        raise ValueError(f"Formas incompatibles: y_true {y_true.shape}, y {y.shape}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Las puntuaciones contienen valores no finitos.")
    if binary and not np.isin(y, (0, 1)).all():
        raise ValueError("Las predicciones binarias solo pueden contener 0 y 1.")
    return y_true, y


def _require_both_classes(y_true: np.ndarray, metric: str) -> None:
    if np.unique(y_true).size < 2:
        raise ValueError(f"{metric} requiere ejemplos positivos y negativos en y_true.")


@register_metric("auc_pr", uses_scores=True)
def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Área bajo la curva precisión-exhaustividad, estimada como average precision.

    Es la métrica principal: con clases desbalanceadas no se infla por los verdaderos negativos.
    """
    y_true, y_score = _check_pair(y_true, y_score, binary=False)
    _require_both_classes(y_true, "AUC-PR")
    return float(average_precision_score(y_true, y_score))


@register_metric("auc_roc", uses_scores=True)
def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Área bajo la curva ROC."""
    y_true, y_score = _check_pair(y_true, y_score, binary=False)
    _require_both_classes(y_true, "AUC-ROC")
    return float(roc_auc_score(y_true, y_score))


@register_metric("precision", uses_scores=False)
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fracción de empresas señaladas que son fantasmas; 0 si no se señala ninguna."""
    y_true, y_pred = _check_pair(y_true, y_pred, binary=True)
    return float(precision_score(y_true, y_pred, zero_division=0))


@register_metric("recall", uses_scores=False)
def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fracción de empresas fantasmas detectadas; 0 si no hay positivos."""
    y_true, y_pred = _check_pair(y_true, y_pred, binary=True)
    return float(recall_score(y_true, y_pred, zero_division=0))


@register_metric("f1", uses_scores=False)
def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Media armónica de precisión y exhaustividad; 0 si ambas son 0."""
    y_true, y_pred = _check_pair(y_true, y_pred, binary=True)
    return float(f1_score(y_true, y_pred, zero_division=0))


@register_metric("fpr", uses_scores=False)
def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fracción de empresas legítimas señaladas como fantasmas; 0 si no hay negativos."""
    y_true, y_pred = _check_pair(y_true, y_pred, binary=True)
    negatives = y_true == 0
    if not negatives.any():
        return 0.0
    return float(np.mean(y_pred[negatives] == 1))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fracción de fantasmas entre las ``k`` empresas con mayor probabilidad.

    Representa una capacidad de fiscalización limitada a ``k`` empresas. Los empates se resuelven por
    orden de aparición para que el resultado sea determinista.

    Raises:
        ValueError: si ``k`` no está entre 1 y el número de empresas evaluadas.
    """
    y_true, y_score = _check_pair(y_true, y_score, binary=False)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= y_true.size:
        raise ValueError(f"k debe ser un entero entre 1 y {y_true.size}; se recibió {k!r}.")
    top_k = np.argsort(-y_score, kind="stable")[:k]
    return float(y_true[top_k].mean())


def precision_at_k_key(k: int) -> str:
    """Nombre con el que se reporta precision@k en los resultados."""
    return f"precision_at_{k}"


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    metric_names: Sequence[str] | None = None,
    k_values: Iterable[int] = (),
) -> dict[str, float]:
    """Calcula las métricas solicitadas sobre una partición.

    Args:
        y_true: etiquetas reales (0/1).
        y_score: probabilidades de la clase positiva.
        threshold: umbral elegido en validación; se aplica con ``binarize``.
        metric_names: métricas registradas a calcular; ``None`` calcula todas.
        k_values: valores de ``k`` para precision@k.

    Returns:
        Diccionario ``nombre -> valor``, con precision@k reportada como ``precision_at_<k>``.
    """
    y_true, y_score = _check_pair(y_true, y_score, binary=False)
    y_pred = binarize(y_score, threshold)
    names = METRICS.names() if metric_names is None else list(metric_names)
    results: dict[str, float] = {}
    for name in names:
        metric = METRICS.get(name)
        results[name] = metric.fn(y_true, y_score if metric.uses_scores else y_pred)
    for k in k_values:
        results[precision_at_k_key(k)] = precision_at_k(y_true, y_score, k)
    return results
