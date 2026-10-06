"""Orquestación: población → mercado → patrones → camuflaje → validación → escritura."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from efd.generation.camouflage import apply_camouflage
from efd.generation.config import GeneratorConfig
from efd.generation.market import MarketModel
from efd.generation.patterns import FraudPattern, GenerationState, create_pattern
from efd.generation.population import build_population
from efd.generation.validation import validate_network
from efd.generation.writer import build_metadata, write_scenario
from efd.network import INVOICE_COLUMNS, REPRESENTATIVE_COLUMNS, TAXPAYER_COLUMNS, InvoiceNetwork


@dataclass(frozen=True)
class GeneratedScenario:
    """Resultado de una generación.

    Attributes:
        network: tablas del escenario.
        active_until: último día de actividad de cada contribuyente (índice = ``taxpayer_id``); no se
            escribe en disco, pero permite validar la vida de los emisores aislados.
        metadata: metadatos que se escriben junto a las tablas.
    """

    network: InvoiceNetwork
    active_until: pd.Series
    metadata: dict[str, Any]


class InvoiceNetworkGenerator:
    """Genera un escenario completo a partir de una ``GeneratorConfig``.

    Toda la aleatoriedad proviene de un único ``numpy.random.Generator`` creado con la semilla de la
    configuración y pasado explícitamente a cada etapa.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        """
        Raises:
            KeyError: si la configuración nombra un patrón no registrado.
            ValueError: si las fantasmas de algún patrón no pueden formar grupos válidos.
        """
        self.config = config
        self.patterns: dict[str, FraudPattern] = {
            name: create_pattern(name, **spec.params) for name, spec in config.patterns.items()
        }
        for name, count in config.shell_counts().items():
            self.patterns[name].check_count(count)

    def generate(self) -> GeneratedScenario:
        """Genera y valida el escenario sin escribirlo.

        Raises:
            ValidationError: si la red generada no supera alguna validación.
        """
        config = self.config
        rng = np.random.default_rng(config.seed)
        taxpayers, representatives = build_population(config, rng)
        self._assign_patterns(taxpayers, rng)

        market = MarketModel(config, taxpayers)
        state = GenerationState(
            config=config,
            market=market,
            taxpayers=taxpayers,
            representatives=representatives,
            active_until=np.full(len(taxpayers), market.period_end),
        )
        state.invoices["legit"].append(
            market.build_legitimate_invoices(rng, representatives, taxpayers["taxpayer_id"].to_numpy())
        )
        for pattern in self.patterns.values():
            pattern.inject(state, rng)
        apply_camouflage(state, rng)

        network = self._assemble(state, rng)
        active_until = pd.Series(
            pd.to_datetime(state.active_until), index=state.taxpayers["taxpayer_id"].to_numpy()
        )
        validate_network(network, config, active_until)
        return GeneratedScenario(network, active_until, build_metadata(config, network))

    def run(self, output_root: Path | None = None) -> Path:
        """Genera, valida y escribe el escenario en ``<output_root>/<escenario>/``."""
        scenario = self.generate()
        root = self.config.output_root if output_root is None else Path(output_root)
        return write_scenario(scenario.network, scenario.metadata, root / self.config.scenario_name)

    def _assign_patterns(self, taxpayers: pd.DataFrame, rng: np.random.Generator) -> None:
        """Reparte al azar las fantasmas entre los patrones según los conteos exactos."""
        shells = rng.permutation(np.flatnonzero(taxpayers["is_shell"].to_numpy() == 1))
        start = 0
        for name, count in self.config.shell_counts().items():
            taxpayers.loc[shells[start : start + count], "pattern"] = name
            start += count

    @staticmethod
    def _assemble(state: GenerationState, rng: np.random.Generator) -> InvoiceNetwork:
        """Construye las tablas finales con identificadores que no revelan el orden de generación.

        Las facturas se ordenan por fecha con desempate aleatorio antes de numerarse, y los
        representantes se renumeran al azar.
        """
        ids = state.taxpayers["taxpayer_id"].to_numpy()
        blocks = pd.concat(
            [block for kind in state.invoices.values() for block in kind], ignore_index=True
        )
        shuffled = rng.permutation(len(blocks))
        order = shuffled[np.argsort(blocks["issue_date"].to_numpy()[shuffled], kind="stable")]
        blocks = blocks.iloc[order].reset_index(drop=True)
        amounts = state.market.split_vat(rng, blocks["subtotal"].to_numpy())
        width = max(8, len(str(len(blocks))))
        invoices = pd.DataFrame(
            {
                "invoice_id": [f"FAC{i + 1:0{width}d}" for i in range(len(blocks))],
                "issuer_id": ids[blocks["issuer"].to_numpy()],
                "receiver_id": ids[blocks["receiver"].to_numpy()],
                "issue_date": pd.to_datetime(blocks["issue_date"]),
                **{column: amounts[column].to_numpy() for column in amounts},
            }
        )[list(INVOICE_COLUMNS)]

        representatives = state.representatives
        unique = np.sort(representatives["representative_id"].unique())
        relabel = dict(zip(unique, rng.permutation(unique.size)))
        width = max(6, len(str(unique.size)))
        representatives = pd.DataFrame(
            {
                "representative_id": [
                    f"RL{relabel[value] + 1:0{width}d}" for value in representatives["representative_id"]
                ],
                "taxpayer_id": representatives["taxpayer_id"].to_numpy(),
            }
        ).sort_values(["representative_id", "taxpayer_id"], ignore_index=True)[list(REPRESENTATIVE_COLUMNS)]

        taxpayers = state.taxpayers.copy()
        taxpayers["start_date"] = pd.to_datetime(taxpayers["start_date"])
        taxpayers = taxpayers.sort_values("taxpayer_id", ignore_index=True)[list(TAXPAYER_COLUMNS)]
        return InvoiceNetwork(taxpayers=taxpayers, invoices=invoices, representatives=representatives)
