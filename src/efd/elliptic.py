"""GAT homogéneo sobre Elliptic con partición temporal e inductivo estricto.

Elliptic (203 769 transacciones, 165 variables, 49 pasos de tiempo) es un grafo homogéneo sin atributos de
arista. Las aristas solo unen nodos del mismo paso, así que el subgrafo de los pasos 1–34 y el de los pasos
35–49 no comparten ninguna arista. Protocolo (inductivo estricto):

- Entrenamiento: subgrafo de los pasos 1–29 y validación (parada temprana) de los pasos 30–34, en un mismo
  subgrafo de los pasos 1–34; la pérdida usa solo las etiquetas de los pasos 1–29. Al no haber aristas entre
  pasos, la validación no recibe mensajes del entrenamiento.
- Prueba: subgrafo inducido de los pasos 35–49, sin ver nada de ellos durante el ajuste. La estandarización de
  las variables se ajusta solo con los nodos etiquetados de entrenamiento.
- Etiquetas: 0 = legítima, 1 = ilícita, 2 = desconocida (entra al grafo, no a la pérdida ni a las métricas).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import subgraph

from efd.models.neural import (
    configure_determinism,
    fit_standardizer,
    resolve_device,
    train_full_batch,
)

TRAIN_LAST_STEP = 34
VALIDATION_FIRST_STEP = 30
UNKNOWN = 2


@dataclass
class EllipticData:
    """Elliptic completo: variables, aristas, etiquetas (0, 1, 2) y paso de tiempo de cada nodo."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: np.ndarray
    time_step: np.ndarray


@dataclass
class GraphPart:
    """Subgrafo con sus nodos etiquetados: ``idx`` y, en entrenamiento, ``val_idx``."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: np.ndarray
    idx: np.ndarray
    val_idx: np.ndarray | None = None


@dataclass
class EllipticSplit:
    """Partición temporal: grafo de entrenamiento y validación (pasos 1–34) y grafo de prueba (35–49)."""

    train: GraphPart
    test: GraphPart


def load_elliptic(root: str | Path) -> EllipticData:
    """Lee Elliptic con ``EllipticBitcoinDataset`` (lo descarga en ``root`` si falta) y el paso de tiempo.

    El paso de tiempo está en la columna 1 de ``raw/elliptic_txs_features.csv``, en el mismo orden de nodos
    que usa PyG.
    """
    import pandas as pd
    from torch_geometric.datasets import EllipticBitcoinDataset

    root = Path(root)
    data = EllipticBitcoinDataset(root=str(root))[0]
    steps = pd.read_csv(root / "raw" / "elliptic_txs_features.csv", header=None, usecols=[1])[1].to_numpy()
    if steps.shape[0] != data.num_nodes:
        raise ValueError("El número de pasos de tiempo no coincide con el número de nodos de Elliptic.")
    return EllipticData(data.x, data.edge_index, data.y.numpy(), steps.astype(np.int64))


def _induced(data: EllipticData, nodes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    subset = torch.as_tensor(nodes)
    edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.x.shape[0])
    return data.x[subset], edge_index, data.y[nodes]


def temporal_split(data: EllipticData) -> EllipticSplit:
    """Arma el grafo de entrenamiento y validación (pasos 1–34) y el de prueba (35–49).

    Raises:
        ValueError: si alguna arista une nodos de pasos distintos (el protocolo lo supone imposible).
    """
    step = data.time_step
    source, target = data.edge_index
    if (step[source.numpy()] != step[target.numpy()]).any():
        raise ValueError("Hay aristas entre pasos de tiempo distintos; el protocolo inductivo no aplica.")
    train_nodes = np.flatnonzero(step <= TRAIN_LAST_STEP)
    test_nodes = np.flatnonzero(step > TRAIN_LAST_STEP)
    x, edge_index, y = _induced(data, train_nodes)
    train_step = step[train_nodes]
    labeled = y != UNKNOWN
    train = GraphPart(
        x,
        edge_index,
        y,
        np.flatnonzero(labeled & (train_step < VALIDATION_FIRST_STEP)),
        np.flatnonzero(labeled & (train_step >= VALIDATION_FIRST_STEP)),
    )
    x_t, edge_index_t, y_t = _induced(data, test_nodes)
    test = GraphPart(x_t, edge_index_t, y_t, np.flatnonzero(y_t != UNKNOWN))
    return EllipticSplit(train, test)


class HomogeneousGAT(nn.Module):
    """GAT homogéneo: proyección lineal, ``num_layers`` capas ``GATConv`` (``concat=True``, sin lazos propios)
    con término raíz lineal, ELU y dropout, y una capa lineal que produce el logit de cada nodo.

    La dimensión oculta es ``heads × hidden_channels``, como en el GAT heterogéneo del proyecto.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, heads: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.dropout = dropout
        dim = hidden_channels * heads
        self.projection = nn.Linear(in_channels, dim)
        self.convs = nn.ModuleList(
            [
                GATConv(dim, hidden_channels, heads=heads, concat=True, add_self_loops=False)
                for _ in range(num_layers)
            ]
        )
        self.roots = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])
        self.output = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Logit de cada nodo, forma ``(n_nodos,)``."""
        h = self.projection(x)
        for conv, root in zip(self.convs, self.roots, strict=True):
            h = functional.dropout(functional.elu(conv(h, edge_index) + root(h)), self.dropout, self.training)
        return self.output(h).squeeze(-1)


@dataclass
class EllipticFit:
    """Resultado del ajuste: red, estandarizador, historial y época elegida."""

    network: HomogeneousGAT
    mean: np.ndarray
    std: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    device: torch.device


def fit_gat(
    split: EllipticSplit,
    seed: int,
    hidden_channels: int = 8,
    heads: int = 4,
    num_layers: int = 2,
    lr: float = 0.005,
    dropout: float = 0.3,
    weight_decay: float = 0.0005,
    epochs: int = 200,
    patience: int = 20,
    device: str = "cpu",
) -> EllipticFit:
    """Entrena el GAT con el grafo de entrenamiento (pérdida en los pasos 1–29, parada en los 30–34)."""
    dev = resolve_device(device)
    configure_determinism(seed, dev)
    train = split.train
    assert train.val_idx is not None
    mean, std = fit_standardizer(train.x.numpy(), train.idx)
    x = torch.as_tensor((train.x.numpy() - mean) / std, dtype=torch.float32)
    network = HomogeneousGAT(x.shape[1], hidden_channels, heads, num_layers, dropout).to(dev)
    x, edge_index = x.to(dev), train.edge_index.to(dev)
    history, best_epoch = train_full_batch(
        network,
        lambda: network(x, edge_index),
        train.y,
        train.idx,
        train.val_idx,
        lr,
        weight_decay,
        epochs,
        patience,
        dev,
    )
    return EllipticFit(network, mean, std, history, best_epoch, dev)


def predict_test(fit: EllipticFit, split: EllipticSplit) -> np.ndarray:
    """Probabilidad de ilícita de los nodos etiquetados de prueba, con el grafo de prueba inducido."""
    test = split.test
    x = torch.as_tensor((test.x.numpy() - fit.mean) / fit.std, dtype=torch.float32).to(fit.device)
    fit.network.eval()
    with torch.no_grad():
        logits = fit.network(x, test.edge_index.to(fit.device))
    rows = torch.as_tensor(test.idx, dtype=torch.long, device=fit.device)
    return torch.sigmoid(logits)[rows].cpu().numpy()
