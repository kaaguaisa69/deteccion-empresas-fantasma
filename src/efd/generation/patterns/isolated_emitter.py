"""P1 — Emisor aislado."""

from __future__ import annotations

import numpy as np
import pandas as pd

from efd.generation.config import check_int_range
from efd.generation.market import round_to_multiples
from efd.generation.patterns.base import FraudPattern, GenerationState, draw_int
from efd.generation.patterns.registry import register_pattern


@register_pattern
class IsolatedEmitterPattern(FraudPattern):
    """Fantasma de vida corta que vende a muchos beneficiarios sin compras proporcionales.

    Deja una señal tabular fuerte: antigüedad corta, montos redondos y ventas sin compras (salvo las
    de camuflaje).
    """

    name = "isolated_emitter"

    def __init__(
        self,
        life_months: list[int],
        beneficiaries: list[int],
        invoices_per_beneficiary: list[int],
        amount_log_mean: float,
        amount_log_sigma: float,
        round_multiples: list[int],
    ) -> None:
        """
        Args:
            life_months: rango de meses de actividad.
            beneficiaries: rango de beneficiarios por fantasma.
            invoices_per_beneficiary: rango de facturas a cada beneficiario.
            amount_log_mean: media del logaritmo del monto antes de redondear.
            amount_log_sigma: desviación del logaritmo del monto.
            round_multiples: múltiplos a los que se redondean los montos.
        """
        self.life_months = tuple(life_months)
        self.beneficiaries = tuple(beneficiaries)
        self.invoices_per_beneficiary = tuple(invoices_per_beneficiary)
        check_int_range(self.life_months, "life_months", minimum=1)
        check_int_range(self.beneficiaries, "beneficiaries", minimum=1)
        check_int_range(self.invoices_per_beneficiary, "invoices_per_beneficiary", minimum=1)
        self.amount_log_mean = amount_log_mean
        self.amount_log_sigma = amount_log_sigma
        self.round_multiples = tuple(round_multiples)

    @property
    def group_size(self) -> tuple[int, int]:
        return (1, 1)

    def inject(self, state: GenerationState, rng: np.random.Generator) -> None:
        market = state.market
        period_start = pd.Timestamp(market.period_start)
        period_end = pd.Timestamp(market.period_end)
        for shell in state.shells_of(self.name):
            months = draw_int(rng, self.life_months)
            # El último inicio posible deja la vida completa dentro del período.
            latest_start = period_end + pd.Timedelta(days=1) - pd.DateOffset(months=months)
            span = max((latest_start - period_start).days, 0)
            start = period_start + pd.Timedelta(days=int(rng.integers(0, span + 1)))
            end = start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
            state.set_lifetime(shell, np.datetime64(start, "D"), np.datetime64(min(end, period_end), "D"))

            _, upper = state.window(shell)
            buyers = market.choose_counterparts(
                rng, draw_int(rng, self.beneficiaries), market.sector[shell], upper, "buyer", replace=False
            )
            receivers = np.repeat(buyers, draw_int(rng, self.invoices_per_beneficiary, size=buyers.size))
            amounts = round_to_multiples(
                rng,
                rng.lognormal(self.amount_log_mean, self.amount_log_sigma, receivers.size),
                self.round_multiples,
            )
            state.add_invoices(
                "fraud", shell, receivers, state.counterpart_dates(rng, shell, receivers), amounts
            )
