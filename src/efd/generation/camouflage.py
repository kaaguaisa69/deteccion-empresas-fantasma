"""Aplicación del nivel de camuflaje."""

from __future__ import annotations

import numpy as np
import pandas as pd

from efd.generation.patterns.base import GenerationState


def apply_camouflage(state: GenerationState, rng: np.random.Generator) -> None:
    """Agrega a cada fantasma comercio que imita el legítimo.

    Si una fantasma participa en ``n`` facturas de su patrón, recibe ``round(c / (1 - c) · n)``
    facturas de camuflaje, de modo que una proporción ``c`` de sus transacciones luce normal. Cada
    factura de camuflaje es, con igual probabilidad, una venta a un comprador o una compra a un
    proveedor elegidos como en el mercado legítimo, con montos legítimos y fechas dentro de la
    actividad de la fantasma.
    """
    share = state.config.camouflage.share
    if share == 0 or not state.invoices["fraud"]:
        return
    market = state.market
    fraud = pd.concat(state.invoices["fraud"], ignore_index=True)
    involvement = np.bincount(fraud["issuer"], minlength=market.n) + np.bincount(
        fraud["receiver"], minlength=market.n
    )
    for shell in np.flatnonzero(state.taxpayers["is_shell"].to_numpy() == 1):
        n_camouflage = round(share / (1 - share) * involvement[shell])
        n_sales = int(rng.binomial(n_camouflage, 0.5))
        _, upper = state.window(shell)
        sector = market.sector[shell]

        buyers = market.choose_counterparts(rng, n_sales, sector, upper, "buyer", replace=True)
        state.add_invoices(
            "camouflage",
            shell,
            buyers,
            state.counterpart_dates(rng, shell, buyers),
            market.sample_amounts(rng, buyers),
        )
        suppliers = market.choose_counterparts(
            rng, n_camouflage - n_sales, sector, upper, "supplier", replace=True
        )
        state.add_invoices(
            "camouflage",
            suppliers,
            shell,
            state.counterpart_dates(rng, shell, suppliers),
            market.sample_amounts(rng, np.full(suppliers.size, shell)),
        )
