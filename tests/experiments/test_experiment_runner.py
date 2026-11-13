"""Corrida completa del orquestador sobre el escenario de 2 000 contribuyentes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from efd.data import ExperimentData
from efd.evaluation.metrics import precision_at_k
from efd.experiments import ExperimentConfig, ExperimentRunner
from efd.experiments.__main__ import main
from efd.experiments.runner import mean_std
from efd.models import BaseDetector, register_detector

RUN_FILES = (
    "config.yaml",
    "scenario.json",
    "folds.json",
    "metrics.json",
    "predictions.parquet",
    "environment.json",
)


@register_detector
class _TuningProbe(BaseDetector):
    """Puntúa como la etiqueta en validación y al revés en prueba (``quality="val"``), o viceversa."""

    name = "tuning_probe_test"

    def __init__(self, seed: int, quality: str = "val", tag: int = 0) -> None:
        self.quality = quality
        self.val_idx = np.empty(0, dtype=int)

    def _fit(self, data: ExperimentData, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        self.val_idx = np.asarray(val_idx)

    def _predict_proba(self, data: ExperimentData, idx: np.ndarray) -> np.ndarray:
        y = data.y[idx].astype(float)
        on_validation = np.array_equal(idx, self.val_idx)
        return y if on_validation == (self.quality == "val") else 1.0 - y


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
            "subpopulations": {"sin_p1": ["isolated_emitter"]},
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
    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["experiment"]["seeds"] == [
        1,
        2,
    ]
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
        assert set(fold["recall_by_pattern"]) <= {
            "isolated_emitter",
            "ring",
            "chain",
            "shared_representative",
        }
    summary = metrics["summary"]
    auc_pr = [fold["metrics"]["auc_pr"] for fold in metrics["folds"]]
    assert summary["metrics"]["auc_pr"] == mean_std(auc_pr)
    assert summary["precision_at_k"]["precision_at_10"]["n"] == 2
    assert set(summary["recall_by_pattern"]) == {"isolated_emitter", "ring", "chain", "shared_representative"}


def test_subpopulation_metrics(run_dir: Path) -> None:
    metrics = _metrics(run_dir)
    for fold in metrics["folds"]:
        sin_p1 = fold["subpopulations"]["sin_p1"]
        assert set(sin_p1["metrics"]) == set(fold["metrics"])
        assert "isolated_emitter" not in sin_p1["recall_by_pattern"]
    summary = metrics["summary"]["subpopulations"]["sin_p1"]
    values = [fold["subpopulations"]["sin_p1"]["metrics"]["auc_pr"] for fold in metrics["folds"]]
    assert summary["metrics"]["auc_pr"] == mean_std(values)
    assert "isolated_emitter" not in summary["recall_by_pattern"]


def test_feature_seconds_per_seed(run_dir: Path) -> None:
    metrics = _metrics(run_dir)
    assert [entry["seed"] for entry in metrics["feature_seconds"]] == [1, 2]
    assert all(entry["transactional"] >= 0 for entry in metrics["feature_seconds"])
    assert metrics["summary"]["feature_seconds"]["transactional"]["n"] == 2


def test_run_without_subpopulations_has_no_such_key(written_dir: Path, tmp_path: Path) -> None:
    raw = _raw(written_dir)
    raw["experiment"]["seeds"] = [1]
    del raw["evaluation"]["subpopulations"]
    metrics = _metrics(ExperimentRunner(ExperimentConfig.from_dict(raw), tmp_path).run())
    assert all("subpopulations" not in fold for fold in metrics["folds"])
    assert "subpopulations" not in metrics["summary"]


def test_same_configuration_gives_same_results(run_dir: Path, written_dir: Path, tmp_path: Path) -> None:
    again = ExperimentRunner(ExperimentConfig.from_dict(_raw(written_dir)), tmp_path).run()
    first, second = _metrics(run_dir), _metrics(again)
    assert first["folds"] == second["folds"]
    # Los tiempos de cálculo varían entre corridas; lo demás debe coincidir.
    assert {k: v for k, v in first["summary"].items() if k != "feature_seconds"} == {
        k: v for k, v in second["summary"].items() if k != "feature_seconds"
    }


def test_mean_std() -> None:
    assert mean_std([1.0, 3.0]) == {"mean": 2.0, "std": pytest.approx(2**0.5), "n": 2}
    assert mean_std([0.5]) == {"mean": 0.5, "std": None, "n": 1}


GRID = [{"max_features": "sqrt"}, {"max_features": 0.5}]


@pytest.fixture(scope="module")
def tuned_metrics(written_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    raw = _raw(written_dir)
    raw["model"]["grid"] = GRID
    return _metrics(ExperimentRunner(ExperimentConfig.from_dict(raw), tmp_path_factory.mktemp("runs")).run())


def test_tuning_selects_best_validation_auc_pr(tuned_metrics: dict[str, Any]) -> None:
    for fold in tuned_metrics["folds"]:
        tuning = fold["tuning"]
        values = tuning["val_auc_pr"]
        assert len(values) == 2
        assert tuning["selected"] == values.index(max(values))  # index() da el menor en un empate
        assert tuning["params"] == GRID[tuning["selected"]]
    counts = tuned_metrics["summary"]["tuning"]["selected_counts"]
    assert sum(counts.values()) == len(tuned_metrics["folds"])
    assert list(counts) == sorted(counts)


def test_run_without_grid_has_no_tuning(run_dir: Path) -> None:
    metrics = _metrics(run_dir)
    assert all("tuning" not in fold for fold in metrics["folds"])
    assert "tuning" not in metrics["summary"]


def _probe_metrics(written_dir: Path, tmp_path: Path, grid: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _raw(written_dir)
    raw["experiment"]["seeds"] = [1]
    raw["model"] = {"name": "tuning_probe_test", "grid": grid}
    return _metrics(ExperimentRunner(ExperimentConfig.from_dict(raw), tmp_path).run())


def test_tuning_ties_choose_lowest_index(written_dir: Path, tmp_path: Path) -> None:
    metrics = _probe_metrics(
        written_dir, tmp_path, [{"quality": "val", "tag": 0}, {"quality": "val", "tag": 1}]
    )  # ambas puntúan igual
    assert all(fold["tuning"]["selected"] == 0 for fold in metrics["folds"])
    assert all(len(set(fold["tuning"]["val_auc_pr"])) == 1 for fold in metrics["folds"])


def test_tuning_does_not_use_the_test_partition(written_dir: Path, tmp_path: Path) -> None:
    metrics = _probe_metrics(written_dir, tmp_path, [{"quality": "test"}, {"quality": "val"}])
    for fold in metrics["folds"]:
        assert fold["tuning"]["selected"] == 1
        assert fold["tuning"]["val_auc_pr"][1] > fold["tuning"]["val_auc_pr"][0]
        # La combinación 0 habría ganado en prueba; la elegida (1) puntúa al revés allí.
        assert fold["metrics"]["auc_pr"] < 0.5


def test_command_line(written_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    raw = _raw(written_dir)
    raw["experiment"]["seeds"] = [1]
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    directory = main(["--config", str(config_path), "--runs-root", str(tmp_path / "runs")])
    output = capsys.readouterr().out
    assert directory.parent == tmp_path / "runs"
    assert "auc_pr" in output and "precision_at_1000" in output and "recall isolated_emitter" in output
