"""Orquestador de experimentos.

Para cada semilla: arma ``ExperimentData``, genera las particiones y, en cada pliegue, entrena el
detector, elige el umbral en validación y evalúa en prueba. precision@k se calcula al final de cada
semilla sobre las predicciones de prueba de todos los pliegues. Solo depende de abstracciones
(``BaseDetector``, ``ThresholdPolicy``) y de los registros por nombre.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from efd.evaluation.metrics import (
    compute_metrics,
    compute_metrics_excluding,
    exclusion_mask,
    precision_at_k_values,
    recall_by_group,
)
from efd.evaluation.splits import stratified_folds
from efd.evaluation.threshold import binarize, create_threshold_policy
from efd.experiments.assembly import AssembledData, Scenario, assemble_experiment_data, load_scenario
from efd.experiments.config import ExperimentConfig
from efd.experiments.run_record import RunRecorder
from efd.models import create_detector
from efd.models.registry import DETECTORS
from efd.reproducibility import set_global_seed

DEFAULT_RUNS_ROOT = Path("results/runs")


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    """Media, desviación estándar muestral (``None`` con un solo valor) y número de valores."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else None,
        "n": int(array.size),
    }


def _summarize_block(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Media ± desviación de las métricas y del recall por patrón de una lista de resultados."""
    patterns = sorted({pattern for result in results for pattern in result["recall_by_pattern"]})
    return {
        "metrics": {name: mean_std([r["metrics"][name] for r in results]) for name in results[0]["metrics"]},
        "recall_by_pattern": {
            pattern: mean_std(
                [r["recall_by_pattern"][pattern] for r in results if pattern in r["recall_by_pattern"]]
            )
            for pattern in patterns
        },
    }


def summarize(
    fold_results: list[dict[str, Any]],
    pooled_results: list[dict[str, Any]],
    feature_seconds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resume las métricas por pliegue y precision@k por semilla con media ± desviación.

    El recall de un patrón se promedia sobre los pliegues cuyo conjunto de prueba tiene fantasmas de
    ese patrón. Las subpoblaciones (si las hay) y los tiempos de las variables se resumen igual.
    """
    k_keys = [key for key in pooled_results[0] if key != "seed"]
    summary = {
        **_summarize_block(fold_results),
        "precision_at_k": {key: mean_std([r[key] for r in pooled_results]) for key in k_keys},
    }
    if "subpopulations" in fold_results[0]:
        summary["subpopulations"] = {
            name: _summarize_block([r["subpopulations"][name] for r in fold_results])
            for name in fold_results[0]["subpopulations"]
        }
    if feature_seconds:
        builders = [key for key in feature_seconds[0] if key != "seed"]
        summary["feature_seconds"] = {
            builder: mean_std([entry[builder] for entry in feature_seconds]) for builder in builders
        }
    return summary


class ExperimentRunner:
    """Ejecuta un experimento descrito por una ``ExperimentConfig``."""

    def __init__(self, config: ExperimentConfig, runs_root: Path = DEFAULT_RUNS_ROOT) -> None:
        """
        Args:
            config: configuración del experimento.
            runs_root: carpeta donde se crea la carpeta de cada corrida.
        """
        self.config = config
        self.runs_root = Path(runs_root)

    @classmethod
    def from_yaml(cls, path: Path, runs_root: Path = DEFAULT_RUNS_ROOT) -> ExperimentRunner:
        """Crea el orquestador a partir de un archivo YAML de ``configs/experiments/``."""
        return cls(ExperimentConfig.load(path), runs_root)

    def run(self) -> Path:
        """Ejecuta todas las semillas y pliegues, guarda la corrida y devuelve su carpeta."""
        config = self.config
        started = time.perf_counter()
        scenario = load_scenario(config.scenario)
        requires_graph = DETECTORS.get(config.model_name).requires_graph

        fold_results, pooled_results, predictions, folds_by_seed, feature_seconds = [], [], [], {}, []
        for seed in config.seeds:
            set_global_seed(seed)
            assembled = assemble_experiment_data(
                scenario, config.feature_set, config.feature_params, seed, requires_graph
            )
            feature_seconds.append({"seed": seed, **assembled.feature_seconds})
            seed_predictions = self._run_seed(assembled, seed, fold_results, folds_by_seed)
            pooled_results.append(
                {
                    "seed": seed,
                    **precision_at_k_values(
                        seed_predictions["label"].to_numpy(),
                        seed_predictions["score"].to_numpy(),
                        config.precision_at_k,
                    ),
                }
            )
            predictions.append(seed_predictions)

        recorder = RunRecorder(self.runs_root, config.name)
        directory = recorder.create_run_dir()
        recorder.save_config(config.raw)
        recorder.save_scenario(self._scenario_info(scenario))
        recorder.save_folds(folds_by_seed)
        recorder.save_predictions(pd.concat(predictions, ignore_index=True))
        recorder.save_metrics(
            {
                "experiment": config.name,
                "feature_set": config.feature_set,
                "model": config.model_name,
                "seeds": list(config.seeds),
                "duration_seconds": round(time.perf_counter() - started, 1),
                "folds": fold_results,
                "precision_at_k": pooled_results,
                "feature_seconds": feature_seconds,
                "summary": summarize(fold_results, pooled_results, feature_seconds),
            }
        )
        recorder.save_environment()
        return directory

    def _run_seed(
        self,
        assembled: AssembledData,
        seed: int,
        fold_results: list[dict[str, Any]],
        folds_by_seed: dict[str, Any],
    ) -> pd.DataFrame:
        """Recorre los pliegues de una semilla y devuelve sus predicciones de prueba."""
        config = self.config
        data, patterns = assembled.data, assembled.patterns
        y = data.y
        threshold_params = dict(config.threshold)
        policy = create_threshold_policy(threshold_params.pop("strategy"), **threshold_params)
        folds = stratified_folds(y, config.n_splits, config.val_size, seed)
        folds_by_seed[str(seed)] = {
            "node_ids": data.node_ids.tolist(),
            "folds": [fold.to_dict() for fold in folds],
        }

        seed_predictions = []
        for fold in folds:
            detector = create_detector(config.model_name, seed=seed, **config.model_params)
            detector.fit(data, fold.train_idx, fold.val_idx)
            threshold = policy.select(y[fold.val_idx], detector.predict_proba(data, fold.val_idx))
            scores = detector.predict_proba(data, fold.test_idx)
            y_test = y[fold.test_idx]
            test_patterns = patterns[fold.test_idx]
            predicted = binarize(scores, threshold)
            fold_result: dict[str, Any] = {
                "seed": seed,
                "fold": fold.index,
                "threshold": threshold,
                "metrics": compute_metrics(y_test, scores, threshold, config.metrics),
                "recall_by_pattern": recall_by_group(y_test, predicted, test_patterns),
            }
            if config.subpopulations:
                fold_result["subpopulations"] = {}
                for name, excluded in config.subpopulations.items():
                    kept = exclusion_mask(test_patterns, excluded)
                    fold_result["subpopulations"][name] = {
                        "metrics": compute_metrics_excluding(
                            y_test, scores, threshold, test_patterns, excluded, config.metrics
                        ),
                        "recall_by_pattern": recall_by_group(
                            y_test[kept], predicted[kept], test_patterns[kept]
                        ),
                    }
            fold_results.append(fold_result)
            seed_predictions.append(
                pd.DataFrame(
                    {
                        "taxpayer_id": data.node_ids[fold.test_idx],
                        "seed": seed,
                        "fold": fold.index,
                        "score": scores,
                        "threshold": threshold,
                        "label": y_test,
                    }
                )
            )
        return pd.concat(seed_predictions, ignore_index=True)

    @staticmethod
    def _scenario_info(scenario: Scenario) -> dict[str, Any]:
        return {
            "path": str(scenario.path).replace("\\", "/"),
            "name": scenario.path.name,
            "config_hash": scenario.config_hash,
            "taxpayers": len(scenario.network.taxpayers),
            "invoices": len(scenario.network.invoices),
        }
