"""Variables transaccionales por contribuyente (grupo G1).

Solo usan los registros propios de cada contribuyente: su ficha y sus facturas emitidas y recibidas,
lo que equivale a una consulta SQL agrupada por contribuyente. Las definiciones están en
``docs/features_design.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from efd.features.base import BaseFeatureBuilder, FeatureContext
from efd.features.registry import register_feature_builder
from efd.network import InvoiceNetwork, invoice_amounts

SIZE_ORDER = ("micro", "pequena", "mediana", "grande")
COMPANY_TYPE = "sociedad"
SIDES = {"sales": ("issuer_id", "receiver_id"), "purchases": ("receiver_id", "issuer_id")}


def is_round_amount(amounts: pd.Series, multiple: int) -> np.ndarray:
    """Indica si cada monto es múltiplo exacto de ``multiple`` (comparado en centavos)."""
    cents = np.rint(amounts.to_numpy(np.float64) * 100).astype(np.int64)
    return cents % (multiple * 100) == 0


def _one_hot(values: pd.Series, prefix: str) -> pd.DataFrame:
    dummies = pd.get_dummies(values, prefix=prefix, prefix_sep="_", dtype=np.float64)
    dummies.columns = [column.replace(" ", "_") for column in dummies.columns]
    return dummies[sorted(dummies.columns)]


@register_feature_builder
class TransactionalFeatureBuilder(BaseFeatureBuilder):
    """Ficha, ventas, compras, relaciones, montos relativos y comportamiento temporal."""

    name = "transactional"

    def __init__(self, round_multiple: int = 100) -> None:
        """
        Args:
            round_multiple: múltiplo que define un monto redondo.
        """
        if round_multiple < 1:
            raise ValueError("'round_multiple' debe ser un entero positivo.")
        self.round_multiple = round_multiple

    def build(self, network: InvoiceNetwork, context: FeatureContext) -> pd.DataFrame:
        ids = pd.Index(network.taxpayers["taxpayer_id"], name="taxpayer_id")
        taxpayers = network.taxpayers.set_index("taxpayer_id")
        invoices = network.invoices.assign(amount=invoice_amounts(network.invoices))

        sides = pd.concat(
            [self._side(invoices, own, other, ids, prefix) for prefix, (own, other) in SIDES.items()],
            axis=1,
        )
        return pd.concat(
            [
                self._profile(taxpayers, context),
                sides,
                self._ratios(invoices, sides, ids),
                self._relative_means(taxpayers, sides),
                self._temporal(invoices, ids, context),
            ],
            axis=1,
        ).astype(np.float64)

    @staticmethod
    def _profile(taxpayers: pd.DataFrame, context: FeatureContext) -> pd.DataFrame:
        """Tipo, tamaño ordinal, sector y provincia en columnas 0/1 y antigüedad en días."""
        size_rank = taxpayers["size"].map({size: rank for rank, size in enumerate(SIZE_ORDER)})
        if size_rank.isna().any():
            raise ValueError(f"Tamaños desconocidos; se esperaban {SIZE_ORDER}.")
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "is_company": (taxpayers["taxpayer_type"] == COMPANY_TYPE).astype(np.float64),
                        "size_rank": size_rank.astype(np.float64),
                        "age_days": (context.period_end - taxpayers["start_date"]).dt.days.astype(np.float64),
                    }
                ),
                _one_hot(taxpayers["sector"], "sector"),
                _one_hot(taxpayers["province"], "province"),
            ],
            axis=1,
        )

    def _side(
        self, invoices: pd.DataFrame, own: str, other: str, ids: pd.Index, prefix: str
    ) -> pd.DataFrame:
        """Estadísticas de las facturas en que el contribuyente aparece en la columna ``own``."""
        frame = pd.DataFrame(
            {
                "own": invoices[own].to_numpy(),
                "other": invoices[other].to_numpy(),
                "amount": invoices["amount"].to_numpy(),
                "round": is_round_amount(invoices["amount"], self.round_multiple),
                "zero_rated": (invoices["subtotal_0"] > 0).to_numpy(),
            }
        )
        grouped = frame.groupby("own")
        amount = grouped["amount"]
        by_pair = frame.groupby(["own", "other"])["amount"].sum()
        pair_share = by_pair / by_pair.groupby(level="own").transform("sum")
        stats = pd.DataFrame(
            {
                "count": amount.size(),
                "total": amount.sum(),
                "mean": amount.mean(),
                "max": amount.max(),
                "std": amount.std(ddof=0),
                "round_share": grouped["round"].mean(),
                "zero_rated_share": grouped["zero_rated"].mean(),
                "counterparties": grouped["other"].nunique(),
                "herfindahl": (pair_share**2).groupby(level="own").sum(),
            }
        )
        stats = stats.reindex(ids).fillna(0.0)
        stats.columns = [f"{prefix}_{column}" for column in stats.columns]
        return stats

    @staticmethod
    def _ratios(invoices: pd.DataFrame, sides: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
        """Compras frente a ventas e IVA pagado frente a IVA cobrado, como ``log((1 + a)/(1 + b))``."""
        vat_collected = invoices.groupby("issuer_id")["vat"].sum().reindex(ids, fill_value=0.0)
        vat_paid = invoices.groupby("receiver_id")["vat"].sum().reindex(ids, fill_value=0.0)
        return pd.DataFrame(
            {
                "purchases_to_sales": np.log1p(sides["purchases_total"]) - np.log1p(sides["sales_total"]),
                "vat_paid_to_collected": np.log1p(vat_paid) - np.log1p(vat_collected),
            },
            index=ids,
        )

    @staticmethod
    def _relative_means(taxpayers: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
        """Monto medio dividido por la mediana de los montos medios de su sector y tamaño."""
        relative = {}
        for prefix in SIDES:
            mean = sides[f"{prefix}_mean"]
            active = sides[f"{prefix}_count"] > 0
            peers = mean[active].groupby(
                [taxpayers.loc[active, "sector"], taxpayers.loc[active, "size"]]
            ).transform("median")
            relative[f"{prefix}_mean_relative"] = (mean[active] / peers).reindex(sides.index, fill_value=0.0)
        return pd.DataFrame(relative)

    @staticmethod
    def _temporal(invoices: pd.DataFrame, ids: pd.Index, context: FeatureContext) -> pd.DataFrame:
        """Meses activos, concentración de ventas en el mes pico y días medios entre facturas."""
        month = invoices["issue_date"].dt.year * 12 + invoices["issue_date"].dt.month
        participation = pd.concat(
            [
                pd.DataFrame({"taxpayer": invoices[column].to_numpy(), "date": invoices["issue_date"].to_numpy(),
                              "month": month.to_numpy()})
                for column in ("issuer_id", "receiver_id")
            ],
            ignore_index=True,
        )
        grouped = participation.groupby("taxpayer")
        count = grouped.size().reindex(ids, fill_value=0)
        span_days = (grouped["date"].max() - grouped["date"].min()).dt.days.reindex(ids, fill_value=0)
        # La media de los intervalos entre facturas consecutivas es (última − primera) / (n − 1).
        mean_gap = np.where(count >= 2, span_days / np.maximum(count - 1, 1), float(context.period_days))

        monthly_sales = invoices.assign(month=month).groupby(["issuer_id", "month"])["amount"].sum()
        by_issuer = monthly_sales.groupby(level="issuer_id")
        peak_share = (by_issuer.max() / by_issuer.sum()).reindex(ids, fill_value=0.0)
        return pd.DataFrame(
            {
                "active_months": grouped["month"].nunique().reindex(ids, fill_value=0),
                "peak_month_sales_share": peak_share,
                "mean_days_between_invoices": mean_gap,
            },
            index=ids,
        )
