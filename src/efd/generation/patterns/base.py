"""Interfaz de los patrones de fraude y estado de la generación que reciben."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from efd.generation.config import GeneratorConfig
from efd.generation.market import MarketModel, invoice_block
from efd.generation.population import SOCIEDAD, is_splittable, split_into_groups

INVOICE_KINDS = ("legit", "fraud", "camouflage")


@dataclass
class GenerationState:
    """Estado mutable que recorre población → mercado → patrones → camuflaje.

    Los contribuyentes se identifican por su posición en ``taxpayers``; los identificadores
    definitivos de facturas y representantes se asignan al ensamblar la red.

    Attributes:
        config: configuración del escenario.
        market: modelo del comercio legítimo, con sus muestreadores.
        taxpayers: contribuyentes con índice posicional.
        representatives: representantes con identificador entero.
        active_until: último día de actividad de cada contribuyente (fin del período salvo P1).
        invoices: bloques de facturas por tipo (``legit``, ``fraud`` o ``camouflage``).
    """

    config: GeneratorConfig
    market: MarketModel
    taxpayers: pd.DataFrame
    representatives: pd.DataFrame
    active_until: np.ndarray
    invoices: dict[str, list[pd.DataFrame]] = field(
        default_factory=lambda: {kind: [] for kind in INVOICE_KINDS}
    )

    @property
    def start_dates(self) -> np.ndarray:
        return self.taxpayers["start_date"].to_numpy().astype("datetime64[D]")

    def shells_of(self, pattern: str) -> np.ndarray:
        """Posiciones de las fantasmas asignadas a ``pattern``."""
        return np.flatnonzero(self.taxpayers["pattern"].to_numpy() == pattern)

    def window(self, positions: np.ndarray) -> tuple[np.datetime64, np.datetime64]:
        """Intervalo en que todos los contribuyentes de ``positions`` están activos a la vez."""
        positions = np.atleast_1d(positions)
        lower = max(self.start_dates[positions].max(), self.market.period_start)
        upper = self.active_until[positions].min()
        return lower, upper

    def set_lifetime(self, position: int, start: np.datetime64, end: np.datetime64) -> None:
        """Fija el inicio de actividades y el último día de actividad de un contribuyente."""
        self.taxpayers.loc[position, "start_date"] = pd.Timestamp(start)
        self.active_until[position] = end

    def share_representative(self, positions: np.ndarray) -> None:
        """Convierte a ``positions`` en sociedades con un único representante legal compartido."""
        ids = self.taxpayers.loc[positions, "taxpayer_id"]
        self.taxpayers.loc[positions, "taxpayer_type"] = SOCIEDAD
        kept = self.representatives[~self.representatives["taxpayer_id"].isin(ids)]
        new_id = int(self.representatives["representative_id"].max()) + 1 if len(self.representatives) else 0
        added = pd.DataFrame({"representative_id": new_id, "taxpayer_id": ids.to_numpy()})
        self.representatives = pd.concat([kept, added], ignore_index=True)

    def add_invoices(
        self,
        kind: str,
        issuer: np.ndarray | int,
        receiver: np.ndarray | int,
        issue_date: np.ndarray,
        subtotal: np.ndarray,
    ) -> None:
        """Agrega facturas; un emisor o receptor escalar se repite para todas."""
        size = len(issue_date)
        self.invoices[kind].append(
            invoice_block(
                np.broadcast_to(issuer, size), np.broadcast_to(receiver, size), issue_date, subtotal
            )
        )

    def counterpart_dates(
        self, rng: np.random.Generator, position: int, counterparts: np.ndarray
    ) -> np.ndarray:
        """Fechas de facturas entre ``position`` y cada contraparte, dentro de la actividad de ambos."""
        lower, upper = self.window(position)
        return self.market.sample_dates(
            rng,
            np.maximum(self.start_dates[counterparts], lower),
            np.full(len(counterparts), upper),
        )


class FraudPattern(ABC):
    """Patrón de comportamiento de un tipo de empresa fantasma.

    Cada patrón recibe sus fantasmas ya etiquetadas en ``state.taxpayers['pattern']``, las agrupa
    según ``group_size`` e inyecta sus facturas en el estado.
    """

    name: ClassVar[str] = ""

    @property
    @abstractmethod
    def group_size(self) -> tuple[int, int]:
        """Tamaño mínimo y máximo de los grupos de fantasmas que actúan juntas."""

    def check_count(self, n_shells: int) -> None:
        """Verifica que ``n_shells`` fantasmas puedan repartirse en grupos válidos.

        Raises:
            ValueError: si no es posible; por ejemplo, un anillo con menos de tres fantasmas.
        """
        if not is_splittable(n_shells, *self.group_size):
            low, high = self.group_size
            raise ValueError(
                f"El patrón '{self.name}' no puede formar grupos de {low} a {high} con {n_shells} fantasmas."
            )

    def groups(self, state: GenerationState, rng: np.random.Generator) -> list[np.ndarray]:
        """Agrupa al azar las fantasmas del patrón."""
        return split_into_groups(state.shells_of(self.name), self.group_size, rng)

    @abstractmethod
    def inject(self, state: GenerationState, rng: np.random.Generator) -> None:
        """Agrega al estado las facturas (y, si corresponde, los atributos) del patrón."""


def draw_int(rng: np.random.Generator, bounds: tuple[int, int], size: int | None = None) -> np.ndarray | int:
    """Entero uniforme en ``[bounds[0], bounds[1]]``, inclusive."""
    value = rng.integers(bounds[0], bounds[1] + 1, size=size)
    return value if size is not None else int(value)
