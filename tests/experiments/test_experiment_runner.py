"""Corrida completa del orquestador sobre el escenario de 2 000 contribuyentes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from efd.evaluation.metrics import precision_at_k
from efd.experiments import ExperimentConfig, ExperimentRunner
from efd.experiments.__main__ import main
from efd.experiments.runner import mean_std

RUN_FILES = (
    "config.yaml",
    "scenario.json",
    "folds.json",
    "metrics.json",
    "predictions.parquet",
    "environment.json",
)


def _raw(scenario: Path) -> dict[str, Any]:
    return {
        "experiment": {
            "name": "rf_g1_test",
            "scenario": str(scenario),
            "feature_set": "G1",
            "seeds": [1, 2],
        },
        "model": {"name": "random_forest", "params": {"n_estimators": 30}},
        "features": {"transactional": {"round_multiple": 100}},
        "splits": {"n_splits": 3, "val_size": 0.25},
        "threshold": {"strategy": "max_f1"},
        "evaluation": {
            "metrics": ["auc_pr", "auc_roc", "f1", "precision", "recall", "fpr"],
            # 1 000 supera el tamaño de cualquier pliegue de prueba (unos 667 nodos).
            "precision_at_k": [10, 1000],
        },
    }


@pytest.fixture(scope="module")
def run_dir(written_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = ExperimentConfig.from_dict(_raw(written_dir))
    return ExperimentRunner(config, tmp_path_factory.mktemp("runs")).run()


def _metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_run_directory_contents(run_dir: Path) -> None:
    assert run_dir.name.endswith("_rf_g1_test")
    assert all((run_dir / file).exists() for file in RUN_FILES)
    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["experiment"]["seeds"] == [1, 2]
    scenario = json.loads((run_dir / "scenario.json").read_text(encoding="utf-8"))
    assert scenario["name"] == "prev1_medio_s42" and len(scenario["config_hash"]) == 64
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert {"python", "scikit-learn", "torch_geometric"} <= set(environment)


def test_every_taxpayer_is_tested_once_per_seed(run_dir: Path) -> None:
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    for _, by_seed in predictions.groupby("seed"):
        assert len(by_seed) == 2000
        assert by_seed["taxpayer_id"].is_unique
        assert by_seed["label"].sum() == 20


def test_precision_at_k_uses_all_folds_together(run_dir: Path) -> None:
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    pooled = _metrics(run_dir)["precision_at_k"]
    assert [entry["seed"] for entry in pooled] == [1, 2]
    for entry in pooled:
        by_seed = predictions[predictions["seed"] == entry["seed"]]
        for k in (10, 1000):
            expected = precision_at_k(by_seed["label"].to_numpy(), by_seed["score"].to_numpy(), k)
            assert entry[f"precision_at_{k}"] == pytest.approx(expected)
    fold_keys = {key for fold in _metrics(run_dir)["folds"] for key in fold["metrics"]}
    assert not any(key.startswith("precision_at_") for key in fold_keys)


def test_metrics_per_fold_and_summary(run_dir: Path) -> None:
    metrics = _metrics(run_dir)
    assert len(metrics["folds"]) == 6  # 2 semillas × 3 pliegues
    for fold in metrics["folds"]:
        assert set(fold["metrics"]) == {"auc_pr", "auc_roc", "f1", "precision", "recall", "fpr"}
        assert set(fold["recall_by_pattern"]) <= {"isolated_emitter", "ring", "chain", "shared_representative"}
    summary = metrics["summary"]
    auc_pr = [fold["metrics"]["auc_pr"] for fold in metrics["folds"]]
    assert summary["metrics"]["auc_pr"] == mean_std(auc_pr)
    assert summary["precision_at_k"]["precision_at_10"]["n"] == 2
    assert set(summary["recall_by_pattern"]) == {"isolated_emitter", "ring", "chain", "shared_representative"}


def test_same_configuration_gives_same_results(run_dir: Path, written_dir: Path, tmp_path: Path) -> None:
    again = ExperimentRunner(ExperimentConfig.from_dict(_raw(written_dir)), tmp_path).run()
    first, second = _metrics(run_dir), _metrics(again)
    assert first["folds"] == second["folds"]
    assert first["summary"] == second["summary"]


def test_mean_std() -> None:
    assert mean_std([1.0, 3.0]) == {"mean": 2.0, "std": pytest.approx(2**0.5), "n": 2}
    assert mean_std([0.5]) == {"mean": 0.5, "std": None, "n": 1}


def test_command_line(written_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    raw = _raw(written_dir)
    raw["experiment"]["seeds"] = [1]
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    directory = main(["--config", str(config_path), "--runs-root", str(tmp_path / "runs")])
    output = capsys.readouterr().out
    assert directory.parent == tmp_path / "runs"
    assert "auc_pr" in output and "precision_at_1000" in output and "recall isolated_emitter" in output
