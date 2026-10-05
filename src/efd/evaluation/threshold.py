"""Política de umbral para convertir probabilidades en decisiones binarias.

El umbral se elige siempre sobre la partición de validación y luego se aplica, sin cambios, a la de
prueba. La política por defecto maximiza F1; otras políticas se registran por nombre y se seleccionan
desde la configuración.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from sklearn.metrics import precision_recall_curve

from efd.registry import ClassRegistry

DEFAULT_STRATEGY = "max_f1"


def binarize(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Asigna la clase positiva cuando la probabilidad es mayor o igual que el umbral.

    Es la única regla de decisión del proyecto; tanto la selección del umbral como las métricas la usan.
    """
    return (np.asarray(y_score, dtype=np.float64) >= threshold).astype(np.int64)


class ThresholdPolicy(ABC):
    """Estrategia que elige un umbral a partir de etiquetas y probabilidades de validación."""

    name: ClassVar[str] = ""

    @abstractmethod
    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Devuelve el umbral en [0, 1] que se aplicará a la partición de prueba."""


THRESHOLD_POLICIES: ClassRegistry[ThresholdPolicy] = ClassRegistry(
    "política de umbral", ThresholdPolicy
)


@THRESHOLD_POLICIES.register
class MaxF1Threshold(ThresholdPolicy):
    """Umbral que maximiza F1 en validación.

    Los candidatos son las probabilidades observadas, de modo que el umbral reproduce exactamente un
    punto de la curva precisión-exhaustividad. Ante empates se elige el umbral más alto, que produce
    menos falsos positivos con el mismo F1.
    """

    name = "max_f1"

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=np.float64)
        if y_true.shape != y_score.shape or y_true.ndim != 1:
            raise ValueError("y_true e y_score deben ser arreglos 1-D de la misma longitud.")
        if not np.any(y_true == 1):
            raise ValueError("La validación no contiene positivos; F1 no está definido.")
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        # El último punto de la curva (precisión 1, exhaustividad 0) no tiene umbral asociado.
        precision, recall = precision[:-1], recall[:-1]
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        best = np.flatnonzero(np.isclose(f1, f1.max()))[-1]
        return float(thresholds[best])


@THRESHOLD_POLICIES.register
class FixedThreshold(ThresholdPolicy):
    """Umbral constante declarado en la configuración; útil para análisis de sensibilidad."""

    name = "fixed"

    def __init__(self, value: float) -> None:
        """
        Args:
            value: umbral en [0, 1].
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"El umbral fijo debe estar en [0, 1]; se recibió {value}.")
        self.value = float(value)

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self.value


def create_threshold_policy(strategy: str = DEFAULT_STRATEGY, **params: Any) -> ThresholdPolicy:
    """Instancia la política registrada como ``strategy`` con los parámetros dados."""
    return THRESHOLD_POLICIES.create(strategy, **params)
