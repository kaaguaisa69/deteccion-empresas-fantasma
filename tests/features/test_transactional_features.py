"""Variables G1 comparadas con cálculos a mano sobre la red mínima."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efd.features import build_feature_set, create_feature_builder
from efd.network import LABEL_COLUMNS

from tiny_network import PERIOD_END, tiny_context, tiny_network


@pytest.fixture(scope="module")
def g1() -> pd.DataFrame:
    return create_feature_builder("transactional").build(tiny_network(), tiny_context())


def test_index_and_no_missing_values(g1: pd.DataFrame) -> None:
    assert list(g1.index) == ["A", "B", "C", "D"]
    assert not g1.isna().any().any()
    assert not set(g1.columns) & set(LABEL_COLUMNS)


def test_profile(g1: pd.DataFrame) -> None:
    assert g1["is_company"].tolist() == [1, 0, 1, 1]
    assert g1["size_rank"].tolist() == [3, 0, 3, 0]
    assert g1.loc["A", "age_days"] == (PERIOD_END - pd.Timestamp("2020-01-01")).days == 2191
    assert g1.loc["B", "age_days"] == 305
    assert g1.loc["A", ["sector_comercio", "sector_manufactura"]].tolist() == [1, 0]
    assert g1.loc["D", ["province_El_Oro", "province_Guayas", "province_Pichincha"]].tolist() == [1, 0, 0]


def test_sales_of_a(g1: pd.DataFrame) -> None:
    amounts = np.array([100.0, 300.0, 40.5])
    a = g1.loc["A"]
    assert a["sales_count"] == 3
    assert a["sales_total"] == pytest.approx(440.5)
    assert a["sales_mean"] == pytest.approx(amounts.mean())
    assert a["sales_max"] == 300
    assert a["sales_std"] == pytest.approx(amounts.std())
    assert a["sales_round_share"] == pytest.approx(2 / 3)
    assert a["sales_zero_rated_share"] == pytest.approx(1 / 3)
    assert a["sales_counterparties"] == 2
    assert a["sales_herfindahl"] == pytest.approx((400 / 440.5) ** 2 + (40.5 / 440.5) ** 2)


def test_purchases_of_a(g1: pd.DataFrame) -> None:
    a = g1.loc["A"]
    assert a["purchases_count"] == 2
    assert a["purchases_total"] == pytest.approx(1060)
    assert a["purchases_mean"] == pytest.approx(530)
    assert a["purchases_max"] == 1000
    assert a["purchases_std"] == pytest.approx(470)
    assert a["purchases_round_share"] == pytest.approx(0.5)
    assert a["purchases_zero_rated_share"] == 0
    assert a["purchases_counterparties"] == 2
    assert a["purchases_herfindahl"] == pytest.approx((1000 / 1060) ** 2 + (60 / 1060) ** 2)


def test_ratios(g1: pd.DataFrame) -> None:
    a = g1.loc["A"]
    assert a["purchases_to_sales"] == pytest.approx(np.log(1061 / 441.5))
    assert a["vat_paid_to_collected"] == pytest.approx(np.log(160 / 59.58))
    assert g1.loc["D", "purchases_to_sales"] == 0


def test_relative_means(g1: pd.DataFrame) -> None:
    # A y C son comercio-grande: medianas de 146.83 y 60 en ventas, 530 y 40.5 en compras.
    sales_median = (440.5 / 3 + 60) / 2
    purchases_median = (530 + 40.5) / 2
    assert g1.loc["A", "sales_mean_relative"] == pytest.approx((440.5 / 3) / sales_median)
    assert g1.loc["C", "sales_mean_relative"] == pytest.approx(60 / sales_median)
    assert g1.loc["A", "purchases_mean_relative"] == pytest.approx(530 / purchases_median)
    assert g1.loc["B", "sales_mean_relative"] == 1.0  # único manufactura-micro con ventas
    assert g1.loc["D", "sales_mean_relative"] == 0


def test_temporal(g1: pd.DataFrame) -> None:
    assert g1["active_months"].tolist() == [4, 2, 2, 0]
    assert g1.loc["A", "peak_month_sales_share"] == pytest.approx(400 / 440.5)
    assert g1.loc["B", "peak_month_sales_share"] == 1.0
    span_a = (pd.Timestamp("2025-12-31") - pd.Timestamp("2025-03-10")).days
    assert g1.loc["A", "mean_days_between_invoices"] == pytest.approx(span_a / 4)
    assert g1.loc["D", "mean_days_between_invoices"] == 365  # sin facturas: duración del período


def test_round_multiple_is_configurable() -> None:
    g1 = create_feature_builder("transactional", round_multiple=1000).build(tiny_network(), tiny_context())
    assert g1.loc["A", "sales_round_share"] == 0
    assert g1.loc["B", "sales_round_share"] == 1


def test_feature_set_g1_matches_builder(g1: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(build_feature_set("G1", tiny_network(), tiny_context()), g1)
    with pytest.raises(KeyError):
        build_feature_set("G9", tiny_network(), tiny_context())
