"""Perfil legítimo (tamaño y antigüedad) de las fantasmas P2–P4."""

from __future__ import annotations

import numpy as np
import pandas as pd
from generation_helpers import small_raw

from efd.generation import GeneratorConfig, InvoiceNetworkGenerator
from efd.generation.population import SOCIEDAD, build_population

PROFILED = ("ring", "chain", "shared_representative")


def _profiled(raw: dict) -> tuple[GeneratorConfig, pd.DataFrame]:
    """Reproduce ``generate`` hasta antes de ``MarketModel``."""
    config = GeneratorConfig.from_dict(raw)
    generator = InvoiceNetworkGenerator(config)
    rng = np.random.default_rng(config.seed)
    taxpayers, _ = build_population(config, rng)
    generator._assign_patterns(taxpayers, rng)
    generator._apply_legit_profiles(taxpayers, rng)
    return config, taxpayers


def _starts(taxpayers: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    return taxpayers["start_date"].to_numpy()[mask].astype("datetime64[D]")


def test_sizes_follow_their_distributions() -> None:
    config, taxpayers = _profiled(small_raw())
    isolated = taxpayers[taxpayers["pattern"] == "isolated_emitter"]
    assert len(isolated) > 0
    assert set(isolated["size"]) <= set(config.shells.sizes)
    for name in PROFILED:
        group = taxpayers[taxpayers["pattern"] == name]
        assert len(group) > 0
        for taxpayer_type, rows in group.groupby("taxpayer_type"):
            assert set(rows["size"]) <= set(config.population.sizes_by_type[taxpayer_type])
    shells = taxpayers[taxpayers["pattern"].isin(PROFILED)]
    assert not ((shells["taxpayer_type"] != SOCIEDAD) & (shells["size"] == "grande")).any()


def test_shared_representative_shells_are_companies() -> None:
    _, taxpayers = _profiled(small_raw())
    group = taxpayers[taxpayers["pattern"] == "shared_representative"]
    assert len(group) > 0
    assert (group["taxpayer_type"] == SOCIEDAD).all()


def test_all_reactivated_start_before_period() -> None:
    raw = small_raw()
    raw["shells"]["reactivated_fraction"] = 1.0
    config, taxpayers = _profiled(raw)
    mask = taxpayers["pattern"].isin(PROFILED).to_numpy()
    assert mask.any()
    assert (_starts(taxpayers, mask) < np.datetime64(config.period.start, "D")).all()


def test_none_reactivated_keep_recent_start() -> None:
    raw = small_raw()
    raw["shells"]["reactivated_fraction"] = 0.0
    config, taxpayers = _profiled(raw)
    mask = taxpayers["pattern"].isin(PROFILED).to_numpy()
    assert mask.any()
    assert (_starts(taxpayers, mask) >= np.datetime64(config.shells.earliest_start, "D")).all()


def test_legitimate_start_dates_are_untouched() -> None:
    raw = small_raw()
    config, profiled = _profiled(raw)
    base, _ = build_population(config, np.random.default_rng(config.seed))
    legit = (profiled["is_shell"] == 0).to_numpy()
    assert np.array_equal(_starts(profiled, legit), _starts(base, legit))
