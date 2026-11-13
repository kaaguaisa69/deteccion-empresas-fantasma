"""P2 — Anillo circular."""

from __future__ import annotations

import numpy as np

from efd.generation.config import check_int_range
from efd.generation.patterns.base import (
    FraudPattern,
    GenerationState,
    add_proportional_purchases,
    draw_int,
)
from efd.generation.patterns.registry import register_pattern


@register_pattern
class RingPattern(FraudPattern):
    """Grupo de fantasmas que se facturan en ciclo A1 → A2 → … → Ak → A1.

    El monto de cada ronda circula con pequeñas variaciones. Cada miembro vende además a algunos
    beneficiarios legítimos, por lo que el anillo no queda aislado del resto de la red.
    """

    name = "ring"

    def __init__(
        self,
        group_size: list[int],
        rounds: list[int],
        amount_surcharge: float,
        amount_jitter: float,
        beneficiaries_per_member: list[int],
        invoices_per_beneficiary: list[int],
    ) -> None:
        """
        Args:
            group_size: rango de fantasmas por anillo.
            rounds: rango de vueltas completas del anillo en el período.
            amount_surcharge: recargo relativo sobre el monto legítimo de cada ronda y de cada venta.
            amount_jitter: variación relativa máxima del monto entre saltos de una ronda.
            beneficiaries_per_member: rango de beneficiarios legítimos por miembro.
            invoices_per_beneficiary: rango de facturas a cada beneficiario.
        """
        self._group_size = tuple(group_size)
        self.rounds = tuple(rounds)
        self.beneficiaries_per_member = tuple(beneficiaries_per_member)
        self.invoices_per_beneficiary = tuple(invoices_per_beneficiary)
        check_int_range(self._group_size, "group_size", minimum=2)
        check_int_range(self.rounds, "rounds", minimum=1)
        check_int_range(self.beneficiaries_per_member, "beneficiaries_per_member")
        check_int_range(self.invoices_per_beneficiary, "invoices_per_beneficiary", minimum=1)
        if amount_surcharge < 0:
            raise ValueError("'amount_surcharge' no puede ser negativo.")
        self.amount_surcharge = amount_surcharge
        self.amount_jitter = amount_jitter

    @property
    def group_size(self) -> tuple[int, int]:
        return self._group_size

    def inject(self, state: GenerationState, rng: np.random.Generator) -> None:
        market = state.market
        for members in self.groups(state, rng):
            k = members.size
            lower, upper = state.window(members)
            n_rounds = draw_int(rng, self.rounds)
            # Cada ronda ocupa k días consecutivos, uno por salto.
            round_dates = market.sample_dates(
                rng,
                np.full(n_rounds, lower),
                np.full(n_rounds, upper - np.timedelta64(k - 1, "D")),
            )
            round_amounts = market.sample_amounts(
                rng, np.full(n_rounds, members[1]), surcharge=self.amount_surcharge
            )
            for hop in range(k):
                jitter = rng.uniform(1 - self.amount_jitter, 1 + self.amount_jitter, n_rounds)
                state.add_invoices(
                    "fraud",
                    members[hop],
                    members[(hop + 1) % k],
                    round_dates + np.timedelta64(hop, "D"),
                    np.round(round_amounts * jitter, 2),
                )
            for member in members:
                buyers = market.choose_counterparts(
                    rng,
                    draw_int(rng, self.beneficiaries_per_member),
                    market.sector[member],
                    upper,
                    "buyer",
                    replace=False,
                )
                receivers = np.repeat(buyers, draw_int(rng, self.invoices_per_beneficiary, size=buyers.size))
                amounts = market.sample_amounts(rng, receivers, surcharge=self.amount_surcharge)
                state.add_invoices(
                    "fraud", member, receivers, state.counterpart_dates(rng, member, receivers), amounts
                )
        add_proportional_purchases(
            state, rng, state.shells_of(self.name), state.config.shells.purchase_ratio_quantiles
        )
