"""Red heterogénea de facturación electrónica: contrato entre generación, grafo y variables.

Las columnas y los nombres de archivo de cada tabla son los del esquema descrito en
``docs/generator_design.md`` y se definen solo aquí.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TAXPAYER_COLUMNS = (
    "taxpayer_id",
    "taxpayer_type",
    "sector",
    "province",
    "start_date",
    "size",
    "is_shell",
    "pattern",
)
INVOICE_COLUMNS = (
    "invoice_id",
    "issuer_id",
    "receiver_id",
    "issue_date",
    "subtotal_0",
    "subtotal_15",
    "vat",
    "total",
)
REPRESENTATIVE_COLUMNS = ("representative_id", "taxpayer_id")

# Etiquetas: ningún constructor de variables puede usarlas como entrada.
LABEL_COLUMNS = ("is_shell", "pattern")

TABLE_FILES = {
    "taxpayers": "taxpayers.parquet",
    "invoices": "invoices.parquet",
    "representatives": "representatives.parquet",
}
METADATA_FILE = "metadata.json"


def _check_columns(frame: pd.DataFrame, expected: tuple[str, ...], table: str) -> None:
    if set(frame.columns) != set(expected):
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        raise ValueError(f"Columnas inválidas en '{table}': faltan {missing}, sobran {extra}.")


@dataclass(frozen=True)
class InvoiceNetwork:
    """Contribuyentes, facturas y representantes legales de la red heterogénea.

    Attributes:
        taxpayers: una fila por contribuyente con ``TAXPAYER_COLUMNS``; ``is_shell`` (1 = fantasma) y
            ``pattern`` (patrón de fraude, nulo para legítimos) son etiquetas.
        invoices: una fila por comprobante (arista emisor → receptor) con ``INVOICE_COLUMNS``.
        representatives: aristas representante legal → contribuyente con ``REPRESENTATIVE_COLUMNS``.
    """

    taxpayers: pd.DataFrame
    invoices: pd.DataFrame
    representatives: pd.DataFrame

    def __post_init__(self) -> None:
        _check_columns(self.taxpayers, TAXPAYER_COLUMNS, "taxpayers")
        _check_columns(self.invoices, INVOICE_COLUMNS, "invoices")
        _check_columns(self.representatives, REPRESENTATIVE_COLUMNS, "representatives")
        if self.taxpayers["taxpayer_id"].duplicated().any():
            raise ValueError("'taxpayer_id' debe ser único.")
        if self.invoices["invoice_id"].duplicated().any():
            raise ValueError("'invoice_id' debe ser único.")
        known = self.taxpayers["taxpayer_id"]
        for table, column in (
            (self.invoices, "issuer_id"),
            (self.invoices, "receiver_id"),
            (self.representatives, "taxpayer_id"),
        ):
            if not table[column].isin(known).all():
                raise ValueError(f"'{column}' referencia contribuyentes inexistentes.")

    @classmethod
    def load(cls, directory: Path) -> InvoiceNetwork:
        """Carga las tablas de un escenario escrito en ``directory``."""
        directory = Path(directory)
        return cls(
            **{table: pd.read_parquet(directory / file) for table, file in TABLE_FILES.items()}
        )
