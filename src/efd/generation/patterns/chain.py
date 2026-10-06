"""P3 — Cadena de intermediarios."""

from __future__ import annotations

import numpy as np

from efd.generation.config import check_int_range
from efd.generation.patterns.base import FraudPattern, GenerationState, draw_int
from efd.generation.patterns.registry import register_pattern


@register_pattern
class ChainPattern(FraudPattern):
    """Cadena F1 → F2 → … → Fh → beneficiario con ``h`` saltos y ``h`` fantasmas.

    En cada operación la factura atraviesa la cadena en orden cronológico y cada intermediario añade
    un pequeño margen; el último salto llega a un beneficiario legítimo.
    """

    name = "chain"

    def __init__(
        self,
        hops: list[int],
        operations: list[int],
        amount_log_mean: float,
        amount_log_sigma: float,
        margin: float,
        max_gap_days: int,
    ) -> None:
        """
        Args:
            hops: rango de saltos por cadena; cada cadena tiene tantas fantasmas como saltos.
            operations: rango de operaciones completas de la cadena en el período.
            amount_log_mean: media del logaritmo del monto inicial de cada operación.
            amount_log_sigma: desviación del logaritmo del monto inicial.
            margin: margen relativo que añade cada salto.
            max_gap_days: días máximos entre saltos consecutivos.
        """
        self.hops = tuple(hops)
        self.operations = tuple(operations)
        check_int_range(self.hops, "hops", minimum=2)
        check_int_range(self.operations, "operations", minimum=1)
        if max_gap_days < 1:
            raise ValueError("'max_gap_days' debe ser positivo.")
        self.amount_log_mean = amount_log_mean
        self.amount_log_sigma = amount_log_sigma
        self.margin = margin
        self.max_gap_days = max_gap_days

    @property
    def group_size(self) -> tuple[int, int]:
        return self.hops

    def inject(self, state: GenerationState, rng: np.random.Generator) -> None:
        market = state.market
        for members in self.groups(state, rng):
            hops = members.size
            lower, upper = state.window(members)
            n_operations = draw_int(rng, self.operations)
            gaps = rng.integers(1, self.max_gap_days + 1, size=(n_operations, hops - 1))
            offsets = np.concatenate([np.zeros((n_operations, 1), dtype=np.int64), gaps.cumsum(axis=1)], axis=1)
            starts = market.sample_dates(
                rng,
                np.full(n_operations, lower),
                np.full(n_operations, upper - np.timedelta64((hops - 1) * self.max_gap_days, "D")),
            )
            beneficiaries = market.choose_counterparts(
                rng, n_operations, market.sector[members[-1]], lower, "buyer", replace=True
            )
            amounts = rng.lognormal(self.amount_log_mean, self.amount_log_sigma, n_operations)
            for hop in range(hops):
                receiver = members[hop + 1] if hop + 1 < hops else beneficiaries
                state.add_invoices(
                    "fraud",
                    members[hop],
                    receiver,
                    starts + offsets[:, hop].astype("timedelta64[D]"),
                    np.round(amounts * (1 + self.margin) ** hop, 2),
                )
