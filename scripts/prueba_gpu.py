"""Prueba puntual de GAT en CPU/GPU: semilla 42, partición 0, una combinación de la rejilla.

Uso: python results/prueba_gpu.py --device cpu|cuda --combo <índice> [--compare-run]
Guarda results/prueba_gpu/<device>_c<combo>.json y .npy (puntuaciones de prueba). No usa el runner.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from efd.evaluation.metrics import compute_metrics_excluding
from efd.evaluation.splits import stratified_folds
from efd.evaluation.threshold import create_threshold_policy
from efd.experiments.assembly import assemble_experiment_data, load_scenario
from efd.experiments.config import ExperimentConfig
from efd.models import create_detector
from efd.reproducibility import set_global_seed

SEED = 42
OUT = Path("results/prueba_gpu")


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combo", type=int, required=True)
    parser.add_argument("--compare-run", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(Path("configs/experiments/gat_g3_v2_ajuste.yaml"))
    combo = config.model_grid[args.combo]
    set_global_seed(SEED)
    assembled = assemble_experiment_data(
        load_scenario(config.scenario), config.feature_set, config.feature_params, SEED, True
    )
    data = assembled.data
    fold = stratified_folds(data.y, config.n_splits, config.val_size, SEED)[0]

    result: dict = {
        "device": args.device,
        "combo_index": args.combo,
        "combo": combo,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
    if args.device == "cuda":
        result["gpu"] = torch.cuda.get_device_name(0)
        torch.zeros(1, device="cuda")  # crea el contexto CUDA antes de medir
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    params = {**config.model_params, **combo, "device": args.device}
    detector = create_detector(config.model_name, seed=SEED, **params)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            detector.fit(data, fold.train_idx, fold.val_idx)
            sync(args.device)
            result["fit_seconds"] = round(time.perf_counter() - started, 2)
            val_scores = detector.predict_proba(data, fold.val_idx)
            scores = detector.predict_proba(data, fold.test_idx)
        except torch.OutOfMemoryError as error:
            result["out_of_memory"] = str(error).splitlines()[0]
            sync(args.device)
            result["fit_seconds"] = round(time.perf_counter() - started, 2)
            if args.device == "cuda":
                result["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                result["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            (OUT / f"{args.device}_c{args.combo}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            return
    result["nondeterministic_warnings"] = sorted(
        {str(w.message)[:200] for w in caught if "determin" in str(w.message).lower()}
    )
    result["epochs_run"] = len(detector.history_)
    result["best_epoch"] = detector.best_epoch_
    if args.device == "cuda":
        result["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        result["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)

    policy_params = dict(config.threshold)
    policy = create_threshold_policy(policy_params.pop("strategy"), **policy_params)
    threshold = policy.select(data.y[fold.val_idx], val_scores)
    y_test = data.y[fold.test_idx]
    excluded = config.subpopulations["sin_p1"]
    metrics = compute_metrics_excluding(
        y_test, scores, threshold, assembled.patterns[fold.test_idx], excluded, config.metrics
    )
    result["auc_pr_sin_p1"] = metrics["auc_pr"]
    result["val_auc_pr_best"] = max(h["val_auc_pr"] for h in detector.history_)

    if args.compare_run:
        run = sorted(glob.glob("results/runs/*_gat_g3_v2_ajuste"))[-1]
        pred = pd.read_parquet(Path(run) / "predictions.parquet")
        pred = pred[(pred["seed"] == SEED) & (pred["fold"] == 0)].set_index("taxpayer_id")
        reference = pred.loc[data.node_ids[fold.test_idx], "score"].to_numpy()
        result["vs_run_max_abs_diff"] = float(np.abs(reference - scores).max())
        result["vs_run_identical"] = bool(np.array_equal(reference, scores))

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{args.device}_c{args.combo}.npy", scores)
    (OUT / f"{args.device}_c{args.combo}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
