"""Verifica que las configuraciones de ``configs/`` solo referencien componentes registrados."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from efd.evaluation.metrics import METRICS
from efd.evaluation.threshold import THRESHOLD_POLICIES, create_threshold_policy
from efd.explainability import EXPLAINERS
from efd.features import FEATURE_BUILDERS
from efd.models import available_detectors

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _is_experiment_config(path: Path) -> bool:
    """Las configuraciones de experimento tienen la sección ``experiment``; el generador tiene la suya."""
    return "experiment" in yaml.safe_load(path.read_text(encoding="utf-8"))


CONFIG_FILES = sorted(path for path in CONFIGS_DIR.glob("*.yaml") if _is_experiment_config(path))


@pytest.fixture(params=CONFIG_FILES, ids=lambda path: path.name)
def config(request: pytest.FixtureRequest) -> dict:
    return yaml.safe_load(request.param.read_text(encoding="utf-8"))


def test_configs_exist() -> None:
    assert CONFIG_FILES


def test_experiment_section(config: dict) -> None:
    assert config["experiment"]["name"]
    assert isinstance(config["experiment"]["seed"], int)


def test_referenced_components_are_registered(config: dict) -> None:
    assert set(config["evaluation"]["metrics"]) <= set(METRICS.names())
    threshold = dict(config["threshold"])
    assert threshold["strategy"] in THRESHOLD_POLICIES
    create_threshold_policy(threshold.pop("strategy"), **threshold)
    for group in config["groups"].values():
        assert set(group["features"]) <= set(FEATURE_BUILDERS.names())
        assert group["explainer"] in EXPLAINERS
        for model in group["models"]:
            assert model["name"] in available_detectors()
