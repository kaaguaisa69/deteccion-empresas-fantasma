from __future__ import annotations

import pandas as pd
import pytest

from efd.network import (
    INVOICE_COLUMNS,
    LABEL_COLUMNS,
    REPRESENTATIVE_COLUMNS,
    TAXPAYER_COLUMNS,
    InvoiceNetwork,
)


@pytest.fixture
def tables() -> dict[str, pd.DataFrame]:
    taxpayers = pd.DataFrame(
        {
            "taxpayer_id": ["TP1", "TP2", "TP3"],
            "taxpayer_type": ["sociedad", "persona_natural", "sociedad"],
            "sector": ["G", "C", "F"],
            "province": ["Pichincha", "Guayas", "El Oro"],
            "start_date": pd.to_datetime(["2010-03-01", "2018-07-15", "2025-02-01"]),
            "size": ["grande", "micro", "micro"],
            "is_shell": [0, 0, 1],
            "pattern": [None, None, "isolated_emitter"],
        }
    )
    invoices = pd.DataFrame(
        {
            "invoice_id": ["F1", "F2"],
            "issuer_id": ["TP3", "TP1"],
            "receiver_id": ["TP1", "TP2"],
            "issue_date": pd.to_datetime(["2025-03-10", "2025-04-02"]),
            "subtotal_0": [0.0, 10.0],
            "subtotal_15": [1000.0, 20.0],
            "vat": [150.0, 3.0],
            "total": [1150.0, 33.0],
        }
    )
    representatives = pd.DataFrame(
        {"representative_id": ["RL1", "RL2"], "taxpayer_id": ["TP1", "TP3"]}
    )
    return {"taxpayers": taxpayers, "invoices": invoices, "representatives": representatives}


def test_valid_network(tables: dict[str, pd.DataFrame]) -> None:
    network = InvoiceNetwork(**tables)
    assert list(network.taxpayers.columns) == list(TAXPAYER_COLUMNS)
    assert list(network.invoices.columns) == list(INVOICE_COLUMNS)
    assert list(network.representatives.columns) == list(REPRESENTATIVE_COLUMNS)


def test_labels_are_taxpayer_columns() -> None:
    assert set(LABEL_COLUMNS) <= set(TAXPAYER_COLUMNS)


@pytest.mark.parametrize("table", ["taxpayers", "invoices", "representatives"])
def test_rejects_missing_or_extra_columns(tables: dict[str, pd.DataFrame], table: str) -> None:
    missing = dict(tables, **{table: tables[table].iloc[:, 1:]})
    with pytest.raises(ValueError, match="faltan"):
        InvoiceNetwork(**missing)
    extra = dict(tables, **{table: tables[table].assign(extra=1)})
    with pytest.raises(ValueError, match="sobran"):
        InvoiceNetwork(**extra)


def test_rejects_duplicate_ids(tables: dict[str, pd.DataFrame]) -> None:
    taxpayers = tables["taxpayers"].assign(taxpayer_id=["TP1", "TP1", "TP3"])
    with pytest.raises(ValueError, match="taxpayer_id"):
        InvoiceNetwork(**dict(tables, taxpayers=taxpayers))
    invoices = tables["invoices"].assign(invoice_id=["F1", "F1"])
    with pytest.raises(ValueError, match="invoice_id"):
        InvoiceNetwork(**dict(tables, invoices=invoices))


@pytest.mark.parametrize(
    ("table", "column"),
    [("invoices", "issuer_id"), ("invoices", "receiver_id"), ("representatives", "taxpayer_id")],
)
def test_rejects_unknown_taxpayers(
    tables: dict[str, pd.DataFrame], table: str, column: str
) -> None:
    broken = tables[table].copy()
    broken.loc[0, column] = "TP999"
    with pytest.raises(ValueError, match="inexistentes"):
        InvoiceNetwork(**dict(tables, **{table: broken}))
