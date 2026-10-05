"""Interfaz común de los explicadores de predicciones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from efd.data import ExperimentData
from efd.models.base import BaseDetector


@dataclass(frozen=True)
class Explanation:
    """Atribuciones de un explicador para un conjunto de empresas.

    Attributes:
        feature_attributions: una fila por empresa explicada (índice = identificador de empresa) y
            una columna por variable con su contribución a la predicción.
        edge_attributions: importancia de las aristas del grafo, con columnas ``source``, ``target``
            e ``importance``; ``None`` para explicadores que no consideran la estructura.
    """

    feature_attributions: pd.DataFrame
    edge_attributions: pd.DataFrame | None = None


class BaseExplainer(ABC):
    """Contrato de los explicadores: SHAP para G1/G2 y GNNExplainer para G3."""

    name: ClassVar[str] = ""

    def explain(
        self, detector: BaseDetector, data: ExperimentData, idx: Sequence[int] | np.ndarray
    ) -> Explanation:
        """Explica las predicciones de ``detector`` para las empresas en las posiciones ``idx``.

        Args:
            detector: detector ya entrenado.
            data: contenedor con ambas vistas.
            idx: posiciones de nodo a explicar.

        Raises:
            ValueError: si las atribuciones no corresponden, fila a fila, a las empresas pedidas.
        """
        positions = data.check_idx(idx)
        explanation = self._explain(detector, data, positions)
        if not explanation.feature_attributions.index.equals(data.node_ids[positions]):
            raise ValueError(
                f"{type(self).__name__} devolvió atribuciones para empresas distintas de las pedidas."
            )
        return explanation

    @abstractmethod
    def _explain(self, detector: BaseDetector, data: ExperimentData, idx: np.ndarray) -> Explanation:
        """Calcula las atribuciones para índices ya validados."""
