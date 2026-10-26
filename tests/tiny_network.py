"""Red mínima de cuatro contribuyentes cuyas variables pueden calcularse a mano.

Facturas (monto sin IVA):
    F1  A → B  2025-03-10  100    (redondo)
    F2  A → B  2025-03-20  300    (redondo; 50 con tarifa 0 %)
    F3  A → C  2025-05-05   40.5
    F4  B → A  2025-06-01  1000   (redondo)
    F5  C → A  2025-12-31   60
D no tiene facturas. A y C comparten el representante RL1.
"""

from __future__ import annotations

import pandas as pd

from efd.features import FeatureContext
from efd.network import InvoiceNetwork

PERIOD_START = pd.Timestamp("2025-01-01")
PERIOD_END = pd.Timestamp("2025-12-31")


def tiny_context(seed: int = 0) -> FeatureContext:
    return FeatureContext(period_start=PERIOD_START, period_end=PERIOD_END, seed=seed)


def tiny_network() -> InvoiceNetwork:
    taxpayers = pd.DataFrame(
        {
            "taxpayer_id": ["A", "B", "C", "D"],
            "taxpayer_type": ["sociedad", "persona_natural", "sociedad", "sociedad"],
            "sector": ["comercio", "manufactura", "comercio", "comercio"],
            "province": ["Guayas", "Pichincha", "Guayas", "El Oro"],
            "start_date": pd.to_datetime(["2020-01-01", "2025-03-01", "2024-06-01", "2010-01-01"]),
            "size": ["grande", "micro", "grande", "micro"],
            "is_shell": [0, 1, 0, 0],
            "pattern": [None, "isolated_emitter", None, None],
        }
    )
    invoices = pd.DataFrame(
        {
            "invoice_id": ["F1", "F2", "F3", "F4", "F5"],
            "issuer_id": ["A", "A", "A", "B", "C"],
            "receiver_id": ["B", "B", "C", "A", "A"],
            "issue_date": pd.to_datetime(
                ["2025-03-10", "2025-03-20", "2025-05-05", "2025-06-01", "2025-12-31"]
            ),
            "subtotal_0": [0.0, 50.0, 0.0, 0.0, 0.0],
            "subtotal_15": [100.0, 250.0, 40.5, 1000.0, 60.0],
            "vat": [15.0, 37.5, 6.08, 150.0, 9.0],
            "total": [115.0, 337.5, 46.58, 1150.0, 69.0],
        }
    )
    representatives = pd.DataFrame(
        {"representative_id": ["RL1", "RL1", "RL2"], "taxpayer_id": ["A", "C", "D"]}
    )
    return InvoiceNetwork(taxpayers=taxpayers, invoices=invoices, representatives=representatives)
