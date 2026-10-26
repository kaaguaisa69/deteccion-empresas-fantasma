"""Construcción del grafo de facturación a partir de la red de contribuyentes, facturas y representantes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from efd.data import TAXPAYER_NODE_TYPE
from efd.network import InvoiceNetwork, invoice_amounts

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

REPRESENTATIVE_NODE_TYPE = "representative"
INVOICE_EDGE = (TAXPAYER_NODE_TYPE, "invoices", TAXPAYER_NODE_TYPE)
INVOICE_REVERSE_EDGE = (TAXPAYER_NODE_TYPE, "invoiced_by", TAXPAYER_NODE_TYPE)
REPRESENTATION_EDGE = (REPRESENTATIVE_NODE_TYPE, "represents", TAXPAYER_NODE_TYPE)
REPRESENTATION_REVERSE_EDGE = (TAXPAYER_NODE_TYPE, "represented_by", REPRESENTATIVE_NODE_TYPE)


def aggregate_invoice_pairs(network: InvoiceNetwork) -> pd.DataFrame:
    """Agrega las facturas por par emisor → receptor.

    Returns:
        Una fila por par con ``issuer_id``, ``receiver_id``, ``count`` (número de facturas),
        ``amount`` (monto total sin IVA), ``first_date`` y ``last_date``.
    """
    invoices = network.invoices.assign(amount=invoice_amounts(network.invoices))
    return (
        invoices.groupby(["issuer_id", "receiver_id"], sort=True)
        .agg(
            count=("invoice_id", "size"),
            amount=("amount", "sum"),
            first_date=("issue_date", "min"),
            last_date=("issue_date", "max"),
        )
        .reset_index()
    )


class InvoiceGraphBuilder:
    """Convierte la red en un grafo con un nodo por contribuyente.

    El orden de los nodos sigue el de ``network.taxpayers``, el mismo que usa la vista tabular de
    ``ExperimentData``.
    """

    def to_networkx(self, network: InvoiceNetwork) -> nx.DiGraph:
        """Grafo dirigido emisor → receptor para las métricas estructurales de G2.

        Cada arista agrega las facturas del par y tiene los atributos ``amount`` (monto total sin IVA)
        y ``count`` (número de facturas). Los contribuyentes sin facturas se incluyen como nodos
        aislados.
        """
        pairs = aggregate_invoice_pairs(network)
        graph = nx.DiGraph()
        graph.add_nodes_from(network.taxpayers["taxpayer_id"])
        graph.add_edges_from(
            (issuer, receiver, {"amount": float(amount), "count": int(count)})
            for issuer, receiver, amount, count in pairs[
                ["issuer_id", "receiver_id", "amount", "count"]
            ].itertuples(index=False)
        )
        return graph

    def to_pyg(
        self,
        network: InvoiceNetwork,
        node_features: pd.DataFrame,
        period_start: pd.Timestamp,
        period_end: pd.Timestamp,
    ) -> HeteroData:
        """Vista de grafo heterogénea de PyTorch Geometric usada por G3.

        Args:
            network: red de contribuyentes, facturas y representantes legales.
            node_features: variables de los nodos ``taxpayer``, indexadas por ``taxpayer_id`` en el
                orden de ``network.taxpayers``; se guardan sin escalar.
            period_start: inicio del período, para normalizar las fechas de las aristas.
            period_end: fin del período.

        Returns:
            ``HeteroData`` con nodos ``taxpayer`` (``x``, ``y``) y ``representative`` (``x``
            constante), aristas de factura agregadas por par con atributos (número de facturas,
            ``log(1 + monto)``, primera y última fecha normalizadas), aristas de representación y las
            inversas de ambos tipos.

        Raises:
            ValueError: si ``node_features`` no sigue el orden de ``network.taxpayers``.
        """
        import torch
        from torch_geometric.data import HeteroData

        taxpayer_ids = pd.Index(network.taxpayers["taxpayer_id"])
        if not node_features.index.equals(taxpayer_ids):
            raise ValueError("Las variables de nodo deben seguir el orden de network.taxpayers.")

        data = HeteroData()
        data[TAXPAYER_NODE_TYPE].x = torch.tensor(node_features.to_numpy(np.float32))
        data[TAXPAYER_NODE_TYPE].y = torch.tensor(network.taxpayers["is_shell"].to_numpy(np.int64))

        pairs = aggregate_invoice_pairs(network)
        span = max((period_end - period_start).days, 1)
        edge_index = torch.tensor(
            np.vstack(
                [taxpayer_ids.get_indexer(pairs["issuer_id"]), taxpayer_ids.get_indexer(pairs["receiver_id"])]
            ),
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            np.column_stack(
                [
                    pairs["count"].to_numpy(np.float64),
                    np.log1p(pairs["amount"].to_numpy(np.float64)),
                    (pairs["first_date"] - period_start).dt.days.to_numpy(np.float64) / span,
                    (pairs["last_date"] - period_start).dt.days.to_numpy(np.float64) / span,
                ]
            ).astype(np.float32)
        )
        data[INVOICE_EDGE].edge_index = edge_index
        data[INVOICE_EDGE].edge_attr = edge_attr
        data[INVOICE_REVERSE_EDGE].edge_index = edge_index.flip(0)
        data[INVOICE_REVERSE_EDGE].edge_attr = edge_attr.clone()

        links = network.representatives
        representative_ids = pd.Index(links["representative_id"].unique())
        data[REPRESENTATIVE_NODE_TYPE].x = torch.ones((len(representative_ids), 1), dtype=torch.float32)
        representation = torch.tensor(
            np.vstack(
                [
                    representative_ids.get_indexer(links["representative_id"]),
                    taxpayer_ids.get_indexer(links["taxpayer_id"]),
                ]
            ),
            dtype=torch.long,
        )
        data[REPRESENTATION_EDGE].edge_index = representation
        data[REPRESENTATION_REVERSE_EDGE].edge_index = representation.flip(0)
        return data
