"""Verifica que las configuraciones de ``configs/experiments/`` sean válidas y coherentes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from efd.evaluation.threshold import create_threshold_policy
from efd.experiments import ExperimentConfig

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"
CONFIG_FILES = sorted(EXPERIMENTS_DIR.glob("*.yaml"))


def test_configs_exist() -> None:
    assert CONFIG_FILES


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda path: path.name)
def test_experiment_configs_are_valid(path: Path) -> None:
    config = ExperimentConfig.load(path)
    assert config.name == path.stem
    threshold = dict(config.threshold)
    create_threshold_policy(threshold.pop("strategy"), **threshold)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("experiment", "feature_set", "G9"),
        ("experiment", "seeds", []),
        ("model", "name", "inexistente"),
        ("evaluation", "metrics", ["accuracy"]),
        ("threshold", "strategy", "otro"),
        ("experiment", "name", "con espacios"),
        ("evaluation", "subpopulations", {"sin_p1": []}),
        ("evaluation", "subpopulations", {"mal nombre": ["isolated_emitter"]}),
    ],
)
def test_invalid_experiment_configs_are_rejected(section: str, key: str, value: object) -> None:
    raw = yaml.safe_load(CONFIG_FILES[0].read_text(encoding="utf-8"))
    raw[section][key] = value
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [{"max_features": 0.1 + i / 100} for i in range(13)],
        [{"max_features": "sqrt"}, {}],
        [{"max_features": "sqrt"}, {"max_features": "sqrt"}],
        ["sqrt"],
    ],
    ids=["vacia", "trece", "elemento_vacio", "repetidos", "no_diccionario"],
)
def test_invalid_model_grid_is_rejected(grid: object) -> None:
    raw = yaml.safe_load(CONFIG_FILES[0].read_text(encoding="utf-8"))
    raw["model"]["grid"] = grid
    with pytest.raises(ValueError, match="model.grid"):
        ExperimentConfig.from_dict(raw)


def test_model_grid_accepts_up_to_twelve_combinations() -> None:
    raw = yaml.safe_load(CONFIG_FILES[0].read_text(encoding="utf-8"))
    raw["model"]["grid"] = [{"min_samples_leaf": i + 1} for i in range(12)]
    assert len(ExperimentConfig.from_dict(raw).model_grid) == 12
