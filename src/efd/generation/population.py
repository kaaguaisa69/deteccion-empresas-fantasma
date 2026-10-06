"""Contribuyentes y representantes legales."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from efd.generation.config import GeneratorConfig
from efd.network import TAXPAYER_COLUMNS

SOCIEDAD = "sociedad"
DAYS_PER_YEAR = 365.25


def draw_categories(rng: np.random.Generator, distribution: Mapping[str, float], n: int) -> np.ndarray:
    """Muestrea ``n`` categorías según una distribución ``categoría -> probabilidad``."""
    labels = np.array(list(distribution), dtype=object)
    probabilities = np.array(list(distribution.values()), dtype=np.float64)
    return labels[rng.choice(labels.size, size=n, p=probabilities / probabilities.sum())]


def _splittable_table(n: int, low: int, high: int) -> np.ndarray:
    """``table[k]`` indica si ``k`` elementos pueden repartirse en grupos de ``low`` a ``high``."""
    table = np.zeros(n + 1, dtype=bool)
    table[0] = True
    for k in range(low, n + 1):
        table[k] = table[max(0, k - high) : k - low + 1].any()
    return table


def is_splittable(n: int, low: int, high: int) -> bool:
    """Indica si ``n`` elementos pueden repartirse en grupos con tamaños entre ``low`` y ``high``."""
    return bool(_splittable_table(n, low, high)[n])


def split_into_groups(
    items: np.ndarray, size_range: tuple[int, int], rng: np.random.Generator
) -> list[np.ndarray]:
    """Reparte ``items`` al azar en grupos cuyos tamaños están dentro de ``size_range``.

    Raises:
        ValueError: si el número de elementos no admite tal reparto.
    """
    low, high = size_range
    table = _splittable_table(len(items), low, high)
    if not table[len(items)]:
        raise ValueError(f"{len(items)} elementos no pueden formar grupos de {low} a {high}.")
    shuffled = rng.permutation(items)
    groups, start = [], 0
    while start < len(shuffled):
        remaining = len(shuffled) - start
        feasible = [
            size for size in range(low, min(high, remaining) + 1) if table[remaining - size]
        ]
        size = int(rng.choice(feasible))
        groups.append(shuffled[start : start + size])
        start += size
    return groups


def _start_dates(
    rng: np.random.Generator, config: GeneratorConfig, is_shell: np.ndarray
) -> np.ndarray:
    """Fechas de inicio de actividades.

    Una fracción de legítimos inicia dentro del período para que la antigüedad no delate a las
    fantasmas; el resto tiene una antigüedad exponencial truncada. Las fantasmas inician entre
    ``shells.earliest_start`` y ``min_active_days`` antes del fin del período.
    """
    population, shells = config.population, config.shells
    period_start = np.datetime64(config.period.start, "D")
    period_end = np.datetime64(config.period.end, "D")
    n = is_shell.size
    period_days = int((period_end - period_start) // np.timedelta64(1, "D")) + 1

    age_days = np.minimum(
        rng.exponential(population.mean_age_years * DAYS_PER_YEAR, n),
        population.max_age_years * DAYS_PER_YEAR,
    ).astype(np.int64)
    established = period_start - np.timedelta64(1, "D") - age_days.astype("timedelta64[D]")
    recent = period_start + rng.integers(0, period_days, n).astype("timedelta64[D]")
    starts = np.where(rng.random(n) < population.in_period_fraction, recent, established)

    shell_earliest = np.datetime64(shells.earliest_start, "D")
    shell_latest = period_end - np.timedelta64(shells.min_active_days, "D")
    shell_span = int((shell_latest - shell_earliest) // np.timedelta64(1, "D")) + 1
    shell_starts = shell_earliest + rng.integers(0, shell_span, n).astype("timedelta64[D]")
    return np.where(is_shell, shell_starts, starts).astype("datetime64[D]")


def build_population(
    config: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea los contribuyentes y sus representantes legales.

    Las fantasmas ocupan posiciones aleatorias, de modo que el identificador no revela la etiqueta.
    Cada sociedad tiene un representante; una fracción de sociedades legítimas forma grupos
    empresariales que comparten representante.

    Returns:
        ``(taxpayers, representatives)``. ``taxpayers`` tiene índice posicional y las columnas de
        ``TAXPAYER_COLUMNS`` con ``pattern`` vacío; ``representatives`` usa identificadores enteros de
        representante que se reasignan al final de la generación.
    """
    population, shells = config.population, config.shells
    n = population.n_taxpayers
    is_shell = np.zeros(n, dtype=bool)
    is_shell[rng.choice(n, size=config.n_shells, replace=False)] = True

    types = draw_categories(rng, population.taxpayer_types, n)
    sizes = np.empty(n, dtype=object)
    for taxpayer_type, distribution in population.sizes_by_type.items():
        mask = ~is_shell & (types == taxpayer_type)
        sizes[mask] = draw_categories(rng, distribution, int(mask.sum()))
    sizes[is_shell] = draw_categories(rng, shells.sizes, int(is_shell.sum()))
    sectors = np.where(
        is_shell,
        draw_categories(rng, shells.sectors, n),
        draw_categories(rng, population.sectors, n),
    )
    provinces = np.where(
        is_shell,
        draw_categories(rng, shells.provinces, n),
        draw_categories(rng, population.provinces, n),
    )
    width = max(6, len(str(n)))
    taxpayers = pd.DataFrame(
        {
            "taxpayer_id": [f"TP{i + 1:0{width}d}" for i in range(n)],
            "taxpayer_type": types,
            "sector": sectors,
            "province": provinces,
            "start_date": _start_dates(rng, config, is_shell),
            "size": sizes,
            "is_shell": is_shell.astype(np.int8),
            "pattern": pd.Series([None] * n, dtype=object),
        }
    )[list(TAXPAYER_COLUMNS)]
    return taxpayers, _build_representatives(config, rng, taxpayers)


def _build_representatives(
    config: GeneratorConfig, rng: np.random.Generator, taxpayers: pd.DataFrame
) -> pd.DataFrame:
    """Asigna un representante por sociedad; los grupos legítimos comparten uno."""
    population = config.population
    companies = np.flatnonzero(taxpayers["taxpayer_type"].to_numpy() == SOCIEDAD)
    legit = companies[taxpayers["is_shell"].to_numpy()[companies] == 0]
    low, high = population.legit_group_size
    n_grouped = round(population.legit_group_fraction * legit.size)
    while n_grouped > 0 and not is_splittable(n_grouped, low, high):
        n_grouped -= 1
    grouped = rng.choice(legit, size=n_grouped, replace=False)
    groups = split_into_groups(grouped, (low, high), rng) if n_grouped else []

    representative = np.full(len(taxpayers), -1, dtype=np.int64)
    for group_id, members in enumerate(groups):
        representative[members] = group_id
    alone = companies[representative[companies] < 0]
    representative[alone] = np.arange(len(groups), len(groups) + alone.size)
    return pd.DataFrame(
        {
            "representative_id": representative[companies],
            "taxpayer_id": taxpayers["taxpayer_id"].to_numpy()[companies],
        }
    )
