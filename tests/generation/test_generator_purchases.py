"""Montos y compras de las fantasmas P2 y P3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from generation_helpers import small_raw

from efd.generation import GeneratorConfig, InvoiceNetworkGenerator
from efd.generation.market import MarketModel
from efd.generation.population import build_population

PURCHASING = ("ring", "chain")


def _sales_purchases(scenario) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    network = scenario.network
    taxpayers = network.taxpayers.set_index("taxpayer_id")
    invoices = network.invoices
    subtotal = invoices["subtotal_0"] + invoices["subtotal_15"]
    sales = subtotal.groupby(invoices["issuer_id"]).sum().reindex(taxpayers.index, fill_value=0.0)
    purchases = subtotal.groupby(invoices["receiver_id"]).sum().reindex(taxpayers.index, fill_value=0.0)
    return taxpayers, sales, purchases


def test_ring_and_chain_shells_receive_legitimate_invoices() -> None:
    scenario = InvoiceNetworkGenerator(GeneratorConfig.from_dict(small_raw())).generate()
    taxpayers = scenario.network.taxpayers
    invoices = scenario.network.invoices
    legit_ids = set(taxpayers.loc[taxpayers["is_shell"] == 0, "taxpayer_id"])
    from_legit = invoices[invoices["issuer_id"].isin(legit_ids)]
    received = set(from_legit["receiver_id"])
    shells = taxpayers[taxpayers["pattern"].isin(PURCHASING)]
    assert len(shells) > 0
    assert shells["taxpayer_id"].isin(received).all()


def test_purchase_ratio_is_within_legitimate_range() -> None:
    scenario = InvoiceNetworkGenerator(GeneratorConfig.from_dict(small_raw())).generate()
    taxpayers, sales, purchases = _sales_purchases(scenario)
    log_gap = np.log1p(purchases) - np.log1p(sales)
    both = (sales > 0) & (purchases > 0)
    legit = both & (taxpayers["is_shell"] == 0)
    shells = taxpayers["pattern"].isin(PURCHASING) & both
    assert shells.any()
    low, high = np.quantile(log_gap[legit], [0.10, 0.90])
    assert low <= log_gap[shells].median() <= high


def test_zero_surcharge_matches_default() -> None:
    config = GeneratorConfig.from_dict(small_raw())
    taxpayers, _ = build_population(config, np.random.default_rng(1))
    market = MarketModel(config, taxpayers)
    buyers = np.arange(200)
    plain = market.sample_amounts(np.random.default_rng(7), buyers)
    zero = market.sample_amounts(np.random.default_rng(7), buyers, surcharge=0.0)
    assert np.array_equal(plain, zero)
