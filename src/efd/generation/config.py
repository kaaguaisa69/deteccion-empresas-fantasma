"""Configuración del generador, cargada y validada desde YAML."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_TOLERANCE = 1e-6


def check_distribution(values: Mapping[str, float], name: str) -> None:
    """Verifica que ``values`` sea una distribución de probabilidad no vacía.

    Raises:
        ValueError: si está vacía, tiene valores negativos o no suma 1.
    """
    if not values:
        raise ValueError(f"'{name}' no puede estar vacía.")
    if any(value < 0 for value in values.values()):
        raise ValueError(f"'{name}' no puede tener probabilidades negativas.")
    if abs(sum(values.values()) - 1.0) > _TOLERANCE:
        raise ValueError(f"'{name}' debe sumar 1; suma {sum(values.values())}.")


def check_int_range(value: tuple[int, int], name: str, minimum: int = 0) -> None:
    """Verifica que ``value`` sea un rango entero ``(mínimo, máximo)`` con ``minimum <= mínimo <= máximo``."""
    if (
        len(value) != 2
        or not all(isinstance(bound, int) and not isinstance(bound, bool) for bound in value)
        or not minimum <= value[0] <= value[1]
    ):
        raise ValueError(f"'{name}' debe ser un rango entero [a, b] con {minimum} <= a <= b.")


def check_fraction(value: float, name: str) -> None:
    """Verifica que ``value`` esté en [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"'{name}' debe estar en [0, 1]; se recibió {value}.")


def _check_keys(values: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(values) != expected:
        raise ValueError(f"'{name}' debe definir exactamente {sorted(expected)}.")


@dataclass(frozen=True)
class PeriodConfig:
    """Período simulado; ambas fechas inclusive."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("El inicio del período debe ser anterior a su fin.")


@dataclass(frozen=True)
class PopulationConfig:
    """Tamaño y atributos de la población de contribuyentes legítimos."""

    n_taxpayers: int
    prevalence: float
    taxpayer_types: dict[str, float]
    sizes_by_type: dict[str, dict[str, float]]
    sectors: dict[str, float]
    provinces: dict[str, float]
    mean_age_years: float
    max_age_years: float
    in_period_fraction: float
    legit_group_fraction: float
    legit_group_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.n_taxpayers < 1:
            raise ValueError("'n_taxpayers' debe ser positivo.")
        if not 0.0 < self.prevalence < 1.0:
            raise ValueError("'prevalence' debe estar en (0, 1).")
        check_distribution(self.taxpayer_types, "taxpayer_types")
        _check_keys(self.sizes_by_type, set(self.taxpayer_types), "sizes_by_type")
        sizes = set(next(iter(self.sizes_by_type.values())))
        for taxpayer_type, distribution in self.sizes_by_type.items():
            check_distribution(distribution, f"sizes_by_type.{taxpayer_type}")
            _check_keys(distribution, sizes, f"sizes_by_type.{taxpayer_type}")
        check_distribution(self.sectors, "sectors")
        check_distribution(self.provinces, "provinces")
        if not 0 < self.mean_age_years <= self.max_age_years:
            raise ValueError("Se requiere 0 < mean_age_years <= max_age_years.")
        check_fraction(self.in_period_fraction, "in_period_fraction")
        check_fraction(self.legit_group_fraction, "legit_group_fraction")
        check_int_range(self.legit_group_size, "legit_group_size", minimum=2)

    @property
    def sizes(self) -> tuple[str, ...]:
        """Categorías de tamaño, en el orden declarado."""
        return tuple(next(iter(self.sizes_by_type.values())))


@dataclass(frozen=True)
class ShellProfileConfig:
    """Atributos de las empresas fantasmas antes de aplicar su patrón."""

    sectors: dict[str, float]
    provinces: dict[str, float]
    sizes: dict[str, float]
    earliest_start: date
    min_active_days: int

    def __post_init__(self) -> None:
        check_distribution(self.sectors, "shells.sectors")
        check_distribution(self.provinces, "shells.provinces")
        check_distribution(self.sizes, "shells.sizes")
        if self.min_active_days < 1:
            raise ValueError("'min_active_days' debe ser positivo.")


@dataclass(frozen=True)
class MarketConfig:
    """Parámetros de la facturación legítima."""

    invoices_per_taxpayer: float
    extra_suppliers_mean: dict[str, float]
    attractiveness: dict[str, int]
    uniform_choice_probability: float
    purchase_weight: dict[str, float]
    relation_intensity_sigma: float
    sector_matrix: dict[str, dict[str, float]]
    amount_log_mean: dict[str, float]
    amount_log_sigma: float
    round_amount_fraction: float
    round_multiples: tuple[int, ...]
    vat_rate: float
    zero_rated_fraction: float
    monthly_seasonality: tuple[float, ...]
    reciprocity_fraction: float
    group_trade_probability: float
    cycles_per_1000: float
    cycle_length: tuple[int, int]

    def __post_init__(self) -> None:
        if self.invoices_per_taxpayer <= 0:
            raise ValueError("'invoices_per_taxpayer' debe ser positivo.")
        if any(value < 1 for value in self.attractiveness.values()):
            raise ValueError("'attractiveness' debe tener enteros mayores o iguales que 1.")
        for name in ("uniform_choice_probability", "round_amount_fraction", "vat_rate",
                     "zero_rated_fraction", "reciprocity_fraction", "group_trade_probability"):
            check_fraction(getattr(self, name), name)
        if not self.round_multiples or any(multiple <= 0 for multiple in self.round_multiples):
            raise ValueError("'round_multiples' debe contener enteros positivos.")
        if len(self.monthly_seasonality) != 12 or any(w <= 0 for w in self.monthly_seasonality):
            raise ValueError("'monthly_seasonality' debe tener 12 pesos positivos.")
        for name, row in self.sector_matrix.items():
            if any(weight < 0 for weight in row.values()) or sum(row.values()) <= 0:
                raise ValueError(f"La fila '{name}' de 'sector_matrix' debe tener pesos no negativos.")
        check_int_range(self.cycle_length, "cycle_length", minimum=3)


@dataclass(frozen=True)
class PatternSpec:
    """Proporción de fantasmas asignadas a un patrón y parámetros de ese patrón."""

    proportion: float
    params: dict[str, Any]


@dataclass(frozen=True)
class CamouflageConfig:
    """Nivel de camuflaje del escenario y proporción asociada a cada nivel."""

    level: str
    levels: dict[str, float]

    def __post_init__(self) -> None:
        if self.level not in self.levels:
            raise ValueError(f"Nivel de camuflaje desconocido: '{self.level}'.")
        for name, share in self.levels.items():
            if not 0.0 <= share < 1.0:
                raise ValueError(f"El camuflaje '{name}' debe estar en [0, 1).")

    @property
    def share(self) -> float:
        """Proporción de transacciones de cada fantasma que imitan comercio normal."""
        return self.levels[self.level]


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuración completa de un escenario del generador."""

    seed: int
    output_root: Path
    period: PeriodConfig
    population: PopulationConfig
    shells: ShellProfileConfig
    market: MarketConfig
    patterns: dict[str, PatternSpec]
    camouflage: CamouflageConfig

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("'seed' debe ser un entero no negativo.")
        sectors = set(self.population.sectors)
        provinces = set(self.population.provinces)
        sizes = set(self.population.sizes)
        if not set(self.shells.sectors) <= sectors:
            raise ValueError("'shells.sectors' contiene sectores que no existen en la población.")
        if not set(self.shells.provinces) <= provinces:
            raise ValueError("'shells.provinces' contiene provincias que no existen en la población.")
        if not set(self.shells.sizes) <= sizes:
            raise ValueError("'shells.sizes' contiene tamaños que no existen en la población.")
        _check_keys(self.market.sector_matrix, sectors, "sector_matrix")
        for name, row in self.market.sector_matrix.items():
            if not set(row) <= sectors:
                raise ValueError(f"La fila '{name}' de 'sector_matrix' tiene sectores desconocidos.")
        for name in ("extra_suppliers_mean", "attractiveness", "purchase_weight", "amount_log_mean"):
            _check_keys(getattr(self.market, name), sizes, name)
        if not self.shells.earliest_start < self.period.end:
            raise ValueError("'shells.earliest_start' debe ser anterior al fin del período.")
        if not self.patterns:
            raise ValueError("Se requiere al menos un patrón de fraude.")
        check_distribution(
            {name: spec.proportion for name, spec in self.patterns.items()}, "patterns.proportion"
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GeneratorConfig:
        """Construye la configuración a partir del diccionario leído del YAML."""
        population = dict(raw["population"])
        population["legit_group_size"] = tuple(population["legit_group_size"])
        market = dict(raw["market"])
        for name in ("round_multiples", "monthly_seasonality", "cycle_length"):
            market[name] = tuple(market[name])
        shells = dict(raw["shells"])
        shells["earliest_start"] = _as_date(shells["earliest_start"])
        patterns = {
            name: PatternSpec(
                proportion=float(spec["proportion"]),
                params={key: value for key, value in spec.items() if key != "proportion"},
            )
            for name, spec in raw["patterns"].items()
        }
        return cls(
            seed=raw["seed"],
            output_root=Path(raw["output_root"]),
            period=PeriodConfig(
                start=_as_date(raw["period"]["start"]), end=_as_date(raw["period"]["end"])
            ),
            population=PopulationConfig(**population),
            shells=ShellProfileConfig(**shells),
            market=MarketConfig(**market),
            patterns=patterns,
            camouflage=CamouflageConfig(**raw["camouflage"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Representación serializable a JSON con la misma estructura que el YAML."""
        raw = asdict(self)
        raw["patterns"] = {
            name: {"proportion": spec.proportion, **spec.params} for name, spec in self.patterns.items()
        }
        return json.loads(json.dumps(raw, default=str))

    def config_hash(self) -> str:
        """Huella SHA-256 de los parámetros que determinan los datos generados.

        Es independiente del formato del YAML y excluye ``output_root``: la carpeta de salida no
        altera los datos y su representación cambia según el sistema operativo.
        """
        parameters = {key: value for key, value in self.to_dict().items() if key != "output_root"}
        canonical = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def scenario_name(self) -> str:
        """Nombre de la carpeta del escenario: ``prev<prevalencia %>_<camuflaje>_s<semilla>``."""
        return f"prev{self.population.prevalence * 100:g}_{self.camouflage.level}_s{self.seed}"

    @property
    def n_shells(self) -> int:
        """Número exacto de empresas fantasmas."""
        return round(self.population.n_taxpayers * self.population.prevalence)

    def shell_counts(self) -> dict[str, int]:
        """Fantasmas por patrón, repartidas por resto mayor para que la suma sea exacta.

        Los empates en el resto se resuelven por el orden en que los patrones aparecen en la
        configuración.
        """
        quotas = {name: self.n_shells * spec.proportion for name, spec in self.patterns.items()}
        counts = {name: math.floor(quota) for name, quota in quotas.items()}
        remaining = self.n_shells - sum(counts.values())
        by_remainder = sorted(quotas, key=lambda name: quotas[name] - counts[name], reverse=True)
        for name in by_remainder[:remaining]:
            counts[name] += 1
        return counts


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def load_generator_config(path: Path) -> GeneratorConfig:
    """Lee y valida un archivo YAML de configuración del generador."""
    with open(path, encoding="utf-8") as handle:
        return GeneratorConfig.from_dict(yaml.safe_load(handle))
