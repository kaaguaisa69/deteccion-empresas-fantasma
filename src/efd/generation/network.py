"""Estructura de la red sintética de facturación electrónica."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class InvoiceNetwork:
    """Empresas y facturas generadas, punto de partida de todas las vistas del experimento.

    Attributes:
        companies: una fila por empresa con, al menos, ``company_id`` (único) e ``is_shell``
            (1 = empresa fantasma, 0 = legítima).
        invoices: una fila por factura con, al menos, ``issuer_id`` y ``receiver_id`` (referencias a
            ``company_id``), ``amount`` e ``issued_at``.
    """

    companies: pd.DataFrame
    invoices: pd.DataFrame

    def save(self, directory: Path) -> None:
        """Guarda empresas y facturas en ``directory`` para reutilizarlas sin regenerarlas."""
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> InvoiceNetwork:
        """Carga una red guardada previamente con ``save``."""
        raise NotImplementedError
