"""Particiones estratificadas sobre nodos, compartidas por todos los grupos experimentales.

Cada pliegue de un k-fold estratificado actúa una vez como partición de prueba. El resto de nodos se
divide, también de forma estratificada, en entrenamiento y validación. Las particiones son posiciones
de nodo en ``ExperimentData`` y dependen solo de las etiquetas y la semilla, nunca de las variables ni
del modelo; por eso G1, G2 y G3 se evalúan exactamente sobre las mismas empresas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class Fold:
    """Partición de un pliegue en entrenamiento, validación y prueba (posiciones de nodo ordenadas).

    Attributes:
        index: número de pliegue, desde 0.
        train_idx: nodos para ajustar el modelo.
        val_idx: nodos para parada temprana y selección del umbral.
        test_idx: nodos para reportar métricas.
    """

    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def to_dict(self) -> dict[str, int | list[int]]:
        """Representación serializable a JSON, para guardar las particiones de cada ejecución."""
        return {
            "index": self.index,
            "train_idx": self.train_idx.tolist(),
            "val_idx": self.val_idx.tolist(),
            "test_idx": self.test_idx.tolist(),
        }


def stratified_folds(y: np.ndarray, n_splits: int, val_size: float, seed: int) -> list[Fold]:
    """Genera ``n_splits`` pliegues estratificados sobre los nodos.

    Args:
        y: etiquetas binarias en orden posicional (``ExperimentData.y``).
        n_splits: número de pliegues; cada nodo aparece en prueba exactamente una vez.
        val_size: fracción de los nodos no usados en prueba que se reserva para validación.
        seed: semilla del experimento; con la misma semilla las particiones son idénticas.

    Raises:
        ValueError: si los parámetros son inválidos o alguna clase tiene menos de ``n_splits``
            ejemplos, lo que impediría estratificar.
    """
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError("y debe ser un arreglo 1-D con valores 0 y 1.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError(f"n_splits debe ser un entero mayor o igual que 2; se recibió {n_splits!r}.")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size debe estar en (0, 1); se recibió {val_size}.")
    class_counts = np.bincount(y.astype(np.int64), minlength=2)
    if class_counts.min() < n_splits:
        raise ValueError(
            f"Cada clase necesita al menos {n_splits} ejemplos; conteos por clase: {class_counts.tolist()}."
        )

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    positions = np.arange(y.size)
    folds = []
    for index, (pool, test_idx) in enumerate(outer.split(positions, y)):
        train_idx, val_idx = train_test_split(
            pool, test_size=val_size, stratify=y[pool], random_state=seed
        )
        folds.append(
            Fold(
                index=index,
                train_idx=np.sort(train_idx),
                val_idx=np.sort(val_idx),
                test_idx=np.sort(test_idx),
            )
        )
    return folds
