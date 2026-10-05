"""Interfaz común de los detectores de empresas fantasmas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from efd.data import ExperimentData


class BaseDetector(ABC):
    """Contrato que cumplen todos los detectores, tabulares (G1, G2) o de grafo (G3).

    Los métodos públicos validan entradas y salidas y delegan el trabajo en ``_fit`` y
    ``_predict_proba``. Así cualquier detector puede sustituir a otro en el orquestador: todos reciben
    las mismas particiones y devuelven puntuaciones con la misma forma y rango.

    Los detectores tabulares leen ``data.tabular(idx)``; los de grafo, ``data.require_graph()``.
    """

    name: ClassVar[str] = ""

    def fit(
        self,
        data: ExperimentData,
        train_idx: Sequence[int] | np.ndarray,
        val_idx: Sequence[int] | np.ndarray,
    ) -> BaseDetector:
        """Entrena el detector.

        Args:
            data: contenedor con ambas vistas.
            train_idx: posiciones de nodo usadas para ajustar parámetros.
            val_idx: posiciones de nodo reservadas para parada temprana o selección interna; nunca
                se usan para ajustar parámetros.

        Returns:
            El propio detector, para encadenar llamadas.

        Raises:
            ValueError: si alguna partición está vacía o si entrenamiento y validación se solapan.
        """
        train = data.check_idx(train_idx)
        val = data.check_idx(val_idx)
        if train.size == 0 or val.size == 0:
            raise ValueError("Las particiones de entrenamiento y validación no pueden estar vacías.")
        if np.intersect1d(train, val).size > 0:
            raise ValueError("Las particiones de entrenamiento y validación se solapan.")
        self._fit(data, train, val)
        return self

    def predict_proba(self, data: ExperimentData, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        """Devuelve la probabilidad de que cada empresa de ``idx`` sea fantasma.

        Returns:
            Arreglo 1-D de flotantes en [0, 1] con un valor por posición de ``idx``, en el mismo orden.

        Raises:
            ValueError: si la implementación devuelve un arreglo con forma o valores inválidos.
        """
        positions = data.check_idx(idx)
        scores = np.asarray(self._predict_proba(data, positions), dtype=np.float64)
        if scores.shape != positions.shape:
            raise ValueError(
                f"{type(self).__name__} devolvió forma {scores.shape}; se esperaba {positions.shape}."
            )
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"{type(self).__name__} devolvió probabilidades fuera de [0, 1].")
        return scores

    @abstractmethod
    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Ajusta el modelo con índices ya validados."""

    @abstractmethod
    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        """Calcula la probabilidad de la clase positiva para índices ya validados."""
