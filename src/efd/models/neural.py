"""Utilidades comunes de los modelos neuronales: preprocesamiento y entrenamiento de grafo completo."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

from efd.data import TAXPAYER_NODE_TYPE, ExperimentData
from efd.evaluation.metrics import auc_pr
from efd.graph.builder import (
    INVOICE_EDGE,
    INVOICE_REVERSE_EDGE,
    REPRESENTATION_EDGE,
    REPRESENTATION_REVERSE_EDGE,
    REPRESENTATIVE_NODE_TYPE,
)
from efd.models.base import BaseDetector

EDGE_ATTR_DIM = 4
EDGE_TYPES = (INVOICE_EDGE, INVOICE_REVERSE_EDGE, REPRESENTATION_EDGE, REPRESENTATION_REVERSE_EDGE)
INVOICE_EDGE_TYPES = (INVOICE_EDGE, INVOICE_REVERSE_EDGE)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Transformación fija sin ajuste: ``signo(x) · log(1 + |x|)``.

    Comprime montos de hasta 10⁸ conservando el signo.
    """
    values = np.asarray(x, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def fit_standardizer(x: np.ndarray, train_idx: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula media y desviación por columna con las filas de entrenamiento de ``x``.

    Args:
        x: matriz ya transformada, una fila por nodo.
        train_idx: posiciones de las filas de entrenamiento.

    Returns:
        ``(mean, std)``; las desviaciones iguales a 0 se reemplazan por 1.
    """
    rows = np.asarray(x, dtype=np.float64)[np.asarray(train_idx)]
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def train_full_batch(
    module: nn.Module,
    forward: Callable[[], torch.Tensor],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
) -> tuple[list[dict[str, float]], int]:
    """Entrena ``module`` con todo el grafo en cada época y parada temprana por AUC-PR de validación.

    La pérdida es ``BCEWithLogitsLoss`` con ``pos_weight`` = legítimos ÷ fantasmas de entrenamiento, y se
    calcula solo con ``logits[train_idx]``. Al final de cada época se mide el AUC-PR de validación en modo
    ``eval``; solo una mejora estricta guarda los pesos. Se detiene tras ``patience`` épocas sin mejora o al
    llegar a ``epochs`` y restaura los mejores pesos.

    Args:
        module: red con los parámetros a optimizar (ya inicializados).
        forward: función sin argumentos que devuelve el logit de todos los nodos, forma ``(n_nodos,)``.
        y: etiquetas de todos los nodos, tomadas de ``data.y``.
        train_idx: nodos de entrenamiento.
        val_idx: nodos de validación (solo parada temprana).
        lr: tasa de aprendizaje de Adam.
        weight_decay: regularización L2 de Adam.
        epochs: máximo de épocas.
        patience: épocas sin mejora antes de detenerse.

    Returns:
        ``(history, best_epoch)``: lista de ``{"epoch", "train_loss", "val_auc_pr"}`` (épocas desde 1) y la
        época elegida.
    """
    labels = torch.tensor(np.asarray(y), dtype=torch.float32)
    train = torch.as_tensor(np.asarray(train_idx), dtype=torch.long)
    val = torch.as_tensor(np.asarray(val_idx), dtype=torch.long)
    y_train = labels[train]
    positives = float(y_train.sum())
    pos_weight = torch.tensor((y_train.numel() - positives) / max(positives, 1.0), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    y_val = np.asarray(y)[np.asarray(val_idx)]

    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(module.state_dict())
    for epoch in range(1, epochs + 1):
        module.train()
        optimizer.zero_grad()
        loss = loss_fn(forward()[train], y_train)
        loss.backward()
        optimizer.step()

        module.eval()
        with torch.no_grad():
            scores = torch.sigmoid(forward()[val]).numpy()
        val_score = auc_pr(y_val, scores)
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_auc_pr": val_score})
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_state = copy.deepcopy(module.state_dict())
        elif epoch - best_epoch >= patience:
            break
    module.load_state_dict(best_state)
    module.eval()
    return history, best_epoch


class HeteroGNN(nn.Module):
    """Red heterogénea común de GraphSAGE y GAT sobre los nodos ``taxpayer`` y ``representative``.

    Una proyección lineal por tipo de nodo lleva las variables a la dimensión oculta; después van
    ``num_layers`` capas ``HeteroConv(aggr="sum")`` sobre los 4 tipos de arista, cada una con activación
    (ReLU en GraphSAGE, ELU en GAT) y dropout; una capa lineal produce el logit de cada ``taxpayer``.

    GAT usa ``GATConv`` con ``concat=True`` y sin lazos propios, con ``edge_dim`` solo en las aristas de
    factura, y suma un término raíz lineal por tipo de nodo; su dimensión oculta es
    ``heads × hidden_channels``. Los pesos de ``SAGEConv`` y ``GATConv`` con entradas ``(-1, -1)`` se crean
    en la primera pasada: ``initialize_lazy`` la hace antes de crear el optimizador.
    """

    def __init__(
        self,
        conv: Literal["sage", "gat"],
        in_channels: Mapping[str, int],
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        heads: int = 1,
    ) -> None:
        """
        Args:
            conv: ``"sage"`` o ``"gat"``.
            in_channels: número de variables por tipo de nodo.
            hidden_channels: dimensión oculta (por cabeza en GAT).
            num_layers: número de capas de mensajes (>= 1).
            dropout: probabilidad de dropout tras cada activación.
            heads: cabezas de atención (solo GAT).
        """
        super().__init__()
        if conv not in ("sage", "gat"):
            raise ValueError(f"conv debe ser 'sage' o 'gat', no {conv!r}.")
        self.conv = conv
        self.dropout = dropout
        self.hidden_dim = hidden_channels * heads if conv == "gat" else hidden_channels
        self.projection = nn.ModuleDict({nt: nn.Linear(n, self.hidden_dim) for nt, n in in_channels.items()})
        self.layers = nn.ModuleList(
            [
                HeteroConv(
                    {edge: self._edge_conv(edge, hidden_channels, heads) for edge in EDGE_TYPES}, aggr="sum"
                )
                for _ in range(num_layers)
            ]
        )
        self.roots = nn.ModuleList(
            [
                nn.ModuleDict(
                    {nt: nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for nt in in_channels}
                )
                for _ in range(num_layers if conv == "gat" else 0)
            ]
        )
        self.output = nn.Linear(self.hidden_dim, 1)

    def _edge_conv(self, edge: tuple[str, str, str], hidden_channels: int, heads: int) -> nn.Module:
        if self.conv == "sage":
            return SAGEConv((-1, -1), hidden_channels, aggr="mean")
        return GATConv(
            (-1, -1),
            hidden_channels,
            heads=heads,
            concat=True,
            add_self_loops=False,
            edge_dim=EDGE_ATTR_DIM if edge in INVOICE_EDGE_TYPES else None,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Logit de cada nodo ``taxpayer``, forma ``(n_taxpayer,)``."""
        activation = functional.relu if self.conv == "sage" else functional.elu
        h = {nt: self.projection[nt](x) for nt, x in x_dict.items()}
        kwargs = {"edge_attr_dict": edge_attr_dict} if self.conv == "gat" and edge_attr_dict else {}
        for index, layer in enumerate(self.layers):
            messages = layer(h, edge_index_dict, **kwargs)
            if self.conv == "gat":
                messages = {nt: out + self.roots[index][nt](h[nt]) for nt, out in messages.items()}
            h = {
                nt: functional.dropout(activation(out), self.dropout, self.training)
                for nt, out in messages.items()
            }
        return self.output(h[TAXPAYER_NODE_TYPE]).squeeze(-1)

    def initialize_lazy(self, *inputs: Any) -> None:
        """Crea los pesos perezosos con una pasada hacia adelante sin gradientes."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(*inputs)
        self.train(was_training)


class HeteroGNNDetector(BaseDetector):
    """Base de ``GraphSAGEDetector`` y ``GATDetector``: entrenamiento transductivo con ``HeteroGNN``.

    Los atributos escalados se arman en tensores nuevos (``data.graph`` no se modifica). Las variables de los
    nodos ``taxpayer`` pasan por ``signed_log1p`` y se estandarizan con los nodos de entrenamiento (D2); en
    GAT, ``edge_attr`` recibe ``log(1+x)`` en la columna 0 y se estandariza con todas las aristas (D3). Las
    etiquetas se leen solo desde ``data.y``. Tras ``fit`` quedan ``history_`` y ``best_epoch_``.
    """

    requires_graph = True
    conv: Literal["sage", "gat"]

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
        heads: int = 1,
    ) -> None:
        """
        Args:
            seed: semilla de pesos y dropout.
            hidden_channels: dimensión oculta (por cabeza en GAT).
            num_layers: capas de mensajes (>= 1).
            lr: tasa de aprendizaje de Adam.
            dropout: probabilidad de dropout.
            weight_decay: regularización L2 de Adam.
            epochs: máximo de épocas.
            patience: épocas sin mejora antes de detener.
            num_threads: hilos de torch; si falta, no se fijan.
            heads: cabezas de atención (solo GAT).
        """
        self.seed = seed
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.num_threads = num_threads
        self.heads = heads
        self.network: HeteroGNN | None = None
        self.history_: list[dict[str, float]] = []
        self.best_epoch_: int = 0
        self._node_stats: tuple[np.ndarray, np.ndarray] | None = None
        self._edge_stats: tuple[np.ndarray, np.ndarray] | None = None

    def _inputs(
        self, data: ExperimentData
    ) -> tuple[dict[str, torch.Tensor], dict[Any, torch.Tensor], dict[Any, torch.Tensor] | None]:
        """Tensores de entrada escalados, construidos sin tocar ``data.graph``."""
        assert self._node_stats is not None
        graph = data.require_graph()
        mean, std = self._node_stats
        taxpayer = signed_log1p(graph[TAXPAYER_NODE_TYPE].x.numpy())
        x_dict = {
            TAXPAYER_NODE_TYPE: torch.as_tensor((taxpayer - mean) / std, dtype=torch.float32),
            REPRESENTATIVE_NODE_TYPE: graph[REPRESENTATIVE_NODE_TYPE].x.clone(),
        }
        edge_index_dict = {edge: graph[edge].edge_index for edge in EDGE_TYPES}
        edge_attr_dict = None
        if self.conv == "gat":
            assert self._edge_stats is not None
            mean_e, std_e = self._edge_stats
            edge_attr_dict = {
                edge: torch.as_tensor(
                    (self._log_count(graph[edge].edge_attr) - mean_e) / std_e, dtype=torch.float32
                )
                for edge in INVOICE_EDGE_TYPES
            }
        return x_dict, edge_index_dict, edge_attr_dict

    @staticmethod
    def _log_count(edge_attr: torch.Tensor) -> np.ndarray:
        """``edge_attr`` como float64 con ``log(1+x)`` solo en la columna 0 (número de facturas)."""
        values = edge_attr.numpy().astype(np.float64)
        values[:, 0] = np.log1p(values[:, 0])
        return values

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)
        graph = data.require_graph()
        self._node_stats = fit_standardizer(signed_log1p(graph[TAXPAYER_NODE_TYPE].x.numpy()), train_idx)
        if self.conv == "gat":
            attr = self._log_count(graph[INVOICE_EDGE].edge_attr)
            self._edge_stats = fit_standardizer(attr, np.arange(attr.shape[0]))
        inputs = self._inputs(data)
        network = HeteroGNN(
            self.conv,
            {nt: x.shape[1] for nt, x in inputs[0].items()},
            self.hidden_channels,
            self.num_layers,
            self.dropout,
            self.heads,
        )
        network.initialize_lazy(*inputs)
        self.history_, self.best_epoch_ = train_full_batch(
            network,
            lambda: network(*inputs),
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
        self.network.eval()
        with torch.no_grad():
            logits = self.network(*self._inputs(data))
        return torch.sigmoid(logits)[torch.as_tensor(idx, dtype=torch.long)].numpy()
