"""Control sin grafo: perceptrón multicapa sobre las variables de nodo (G1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from efd.data import ExperimentData
from efd.models.base import BaseDetector
from efd.models.neural import fit_standardizer, signed_log1p, train_full_batch
from efd.models.registry import register_detector


def _build_mlp(in_features: int, hidden_channels: int, num_layers: int, dropout: float) -> nn.Sequential:
    """``num_layers`` capas lineales, de ``in_features`` a ``hidden_channels`` y de ahí a 1, con ReLU y dropout."""
    layers: list[nn.Module] = []
    width = in_features
    for _ in range(num_layers - 1):
        layers += [nn.Linear(width, hidden_channels), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden_channels
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


@register_detector
class MLPDetector(BaseDetector):
    """Perceptrón multicapa sobre la vista tabular, sin usar el grafo.

    Las variables pasan por ``signed_log1p`` y se estandarizan con los nodos de entrenamiento. Se entrena
    con el protocolo común de ``train_full_batch`` (pérdida ponderada, parada temprana por AUC-PR de
    validación). Tras ``fit`` quedan ``history_`` y ``best_epoch_``.
    """

    name = "mlp"
    requires_graph = False

    def __init__(
        self,
        seed: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        lr: float = 0.01,
        dropout: float = 0.5,
        weight_decay: float = 0.0005,
        epochs: int = 200,
        patience: int = 20,
        num_threads: int | None = None,
        **extra: Any,
    ) -> None:
        """
        Args:
            seed: semilla de pesos y dropout.
            hidden_channels: ancho de las capas ocultas.
            num_layers: número de capas lineales (>= 1).
            lr: tasa de aprendizaje de Adam.
            dropout: probabilidad de dropout entre capas.
            weight_decay: regularización L2 de Adam.
            epochs: máximo de épocas.
            patience: épocas sin mejora antes de detener.
            num_threads: hilos de torch; si falta, no se fijan.
        """
        if extra:
            raise TypeError(f"Parámetros desconocidos para mlp: {sorted(extra)}")
        self.seed = seed
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.num_threads = num_threads
        self.network: nn.Sequential | None = None
        self.history_: list[dict[str, float]] = []
        self.best_epoch_: int = 0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _transform(self, data: ExperimentData) -> torch.Tensor:
        values = signed_log1p(data.features.to_numpy(dtype=np.float64))
        return torch.as_tensor((values - self._mean) / self._std, dtype=torch.float32)

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)
        self._mean, self._std = fit_standardizer(
            signed_log1p(data.features.to_numpy(dtype=np.float64)), train_idx
        )
        x = self._transform(data)
        network = _build_mlp(x.shape[1], self.hidden_channels, self.num_layers, self.dropout)
        self.history_, self.best_epoch_ = train_full_batch(
            network,
            lambda: network(x).squeeze(-1),
            data.y,
            train_idx,
            val_idx,
            self.lr,
            self.weight_decay,
            self.epochs,
            self.patience,
        )
        self.network = network

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("El detector no está entrenado.")
        x = self._transform(data)
        with torch.no_grad():
            logits = self.network(x[torch.as_tensor(idx, dtype=torch.long)]).squeeze(-1)
        return torch.sigmoid(logits).numpy()
