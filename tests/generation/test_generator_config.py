from __future__ import annotations

import pytest

from efd.generation import GeneratorConfig, InvoiceNetworkGenerator, load_generator_config
from efd.generation.patterns import PATTERNS

from generation_helpers import CONFIG_PATH, small_raw


def test_base_scenario_matches_design() -> None:
    config = load_generator_config(CONFIG_PATH)
    assert config.seed == 42
    assert config.population.n_taxpayers == 40000
    assert config.n_shells == 400
    assert config.camouflage.level == "medio"
    assert config.camouflage.share == pytest.approx(0.30)
    assert config.shell_counts() == {
        "isolated_emitter": 160,
        "ring": 80,
        "chain": 80,
        "shared_representative": 80,
    }
    assert config.scenario_name == "prev1_medio_s42"
    assert set(config.patterns) <= set(PATTERNS.names())


@pytest.mark.parametrize(
    ("prevalence", "level", "seed", "expected"),
    [(0.005, "bajo", 7, "prev0.5_bajo_s7"), (0.02, "alto", 42, "prev2_alto_s42")],
)
def test_scenario_name_format(prevalence: float, level: str, seed: int, expected: str) -> None:
    raw = small_raw()
    raw["population"]["prevalence"] = prevalence
    raw["camouflage"]["level"] = level
    raw["seed"] = seed
    assert GeneratorConfig.from_dict(raw).scenario_name == expected


def test_shell_counts_use_largest_remainder() -> None:
    raw = small_raw()
    raw["population"].update(n_taxpayers=1000, prevalence=0.013)
    counts = GeneratorConfig.from_dict(raw).shell_counts()
    # Cuotas 5.2, 2.6, 2.6, 2.6: los dos restos mayores (en orden de aparición) reciben una más.
    assert counts == {"isolated_emitter": 5, "ring": 3, "chain": 3, "shared_representative": 2}
    assert sum(counts.values()) == 13


def test_config_hash_is_stable_and_sensitive() -> None:
    first = GeneratorConfig.from_dict(small_raw())
    assert first.config_hash() == GeneratorConfig.from_dict(small_raw()).config_hash()
    raw = small_raw()
    raw["seed"] = 43
    assert GeneratorConfig.from_dict(raw).config_hash() != first.config_hash()


@pytest.mark.parametrize(
    "output_root", ["data/synthetic", "data\\synthetic", "C:\\Users\\otra\\salida", "/home/otra/salida"]
)
def test_config_hash_ignores_output_root(output_root: str) -> None:
    reference = GeneratorConfig.from_dict(small_raw()).config_hash()
    raw = small_raw()
    raw["output_root"] = output_root
    assert GeneratorConfig.from_dict(raw).config_hash() == reference


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["patterns"]["ring"].update(proportion=0.5),
        lambda raw: raw["camouflage"].update(level="extremo"),
        lambda raw: raw["shells"]["provinces"].update({"Atlántida": 0.0}),
        lambda raw: raw["market"]["sector_matrix"].pop("comercio"),
        lambda raw: raw["population"].update(prevalence=0.0),
        lambda raw: raw["population"]["sectors"].update(comercio=0.5),
        lambda raw: raw["market"].update(monthly_seasonality=[1.0] * 11),
    ],
)
def test_invalid_configurations_are_rejected(mutate) -> None:
    raw = small_raw()
    mutate(raw)
    with pytest.raises(ValueError):
        GeneratorConfig.from_dict(raw)


def test_generator_rejects_patterns_that_cannot_form_groups() -> None:
    raw = small_raw()
    raw["population"]["n_taxpayers"] = 1000  # 10 fantasmas: el anillo recibiría solo 2
    with pytest.raises(ValueError, match="ring"):
        InvoiceNetworkGenerator(GeneratorConfig.from_dict(raw))


def test_generator_rejects_unknown_patterns() -> None:
    raw = small_raw()
    raw["patterns"]["inventado"] = raw["patterns"].pop("chain")
    with pytest.raises(KeyError):
        InvoiceNetworkGenerator(GeneratorConfig.from_dict(raw))
