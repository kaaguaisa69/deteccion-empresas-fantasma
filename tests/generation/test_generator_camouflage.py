from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efd.generation import GeneratorConfig
from efd.generation.camouflage import apply_camouflage
from efd.generation.market import MarketModel
from efd.generation.patterns import GenerationState
from efd.generation.population import build_population

from generation_helpers import small_raw


def _state(level: str) -> GenerationState:
    raw = small_raw()
    raw["camouflage"]["level"] = level
    config = GeneratorConfig.from_dict(raw)
    rng = np.random.default_rng(0)
    taxpayers, representatives = build_population(config, rng)
    market = MarketModel(config, taxpayers)
    return GenerationState(
        config=config,
        market=market,
        taxpayers=taxpayers,
        representatives=representatives,
        active_until=np.full(len(taxpayers), market.period_end),
    )


@pytest.mark.parametrize("level", ["bajo", "medio", "alto"])
def test_camouflage_share_per_shell(level: str) -> None:
    state = _state(level)
    shells = np.flatnonzero(state.taxpayers["is_shell"].to_numpy() == 1)
    legit = np.flatnonzero(state.taxpayers["is_shell"].to_numpy() == 0)
    fraud_per_shell = np.arange(1, shells.size + 1) * 3
    for shell, n in zip(shells, fraud_per_shell):
        state.add_invoices(
            "fraud", shell, legit[:n], np.full(n, np.datetime64("2025-06-01")), np.full(n, 100.0)
        )

    apply_camouflage(state, np.random.default_rng(1))

    camouflage = pd.concat(state.invoices["camouflage"], ignore_index=True)
    share = state.config.camouflage.share
    for shell, n in zip(shells, fraud_per_shell):
        involved = int(((camouflage["issuer"] == shell) | (camouflage["receiver"] == shell)).sum())
        assert involved == round(share / (1 - share) * n)
    # Las contrapartes del camuflaje son siempre legítimas.
    assert state.taxpayers["is_shell"].to_numpy()[
        np.where(np.isin(camouflage["issuer"], shells), camouflage["receiver"], camouflage["issuer"])
    ].sum() == 0
