"""MediciÃ³n de tiempo con una particiÃ³n (paso P4 del plan de G3 y MLP).

Arma los datos como el runner, toma el pliegue 0 de la primera semilla y entrena las combinaciones de la
rejilla. Solo usa entrenamiento y validaciÃ³n; no calcula nada sobre la prueba.

Uso: python scripts/medir_tiempo_g3.py configs/experiments/<yaml> [salida.json]
"""

from __future__ import annotations

import ctypes
import gc
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

from efd.evaluation.metrics import auc_pr
from efd.evaluation.splits import stratified_folds
from efd.experiments.assembly import assemble_experiment_data, load_scenario
from efd.experiments.config import ExperimentConfig
from efd.models import create_detector
from efd.models.registry import DETECTORS
from efd.reproducibility import set_global_seed


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_gb() -> tuple[float, float]:
    """RAM actual y mÃ¡xima del proceso en GB (Windows: working set)."""
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    ok = kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError("GetProcessMemoryInfo fallÃ³")
    return counters.WorkingSetSize / 2**30, counters.PeakWorkingSetSize / 2**30


def main() -> None:
    config = ExperimentConfig.load(Path(sys.argv[1]))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    seed = config.seeds[0]
    set_global_seed(seed)

    start = time.perf_counter()
    scenario = load_scenario(config.scenario)
    requires_graph = DETECTORS.get(config.model_name).requires_graph
    data = assemble_experiment_data(
        scenario, config.feature_set, config.feature_params, seed, requires_graph
    ).data
    assembly_seconds = time.perf_counter() - start

    fold = stratified_folds(data.y, config.n_splits, config.val_size, seed)[0]
    rows = []
    for index, combo in enumerate(config.model_grid):
        start = time.perf_counter()
        detector = create_detector(config.model_name, seed=seed, **{**config.model_params, **combo})
        detector.fit(data, fold.train_idx, fold.val_idx)
        val = auc_pr(data.y[fold.val_idx], detector.predict_proba(data, fold.val_idx))
        seconds = time.perf_counter() - start
        epochs_run, best_epoch = len(detector.history_), detector.best_epoch_
        del detector  # liberaciÃ³n de memoria entre combinaciones, como en el runner
        gc.collect()
        ram_now, ram_peak = memory_gb()
        rows.append(
            {
                "combo": index,
                "params": combo,
                "seconds": seconds,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "val_auc_pr": val,
                "ram_gb": ram_now,
                "ram_peak_gb": ram_peak,
            }
        )
        print(
            f"{index:2d} {combo} {seconds:7.1f} s  Ã©pocas {epochs_run:3d}  mejor {best_epoch:3d}  "
            f"val AUC-PR {val:.6f}  RAM {ram_now:.2f} GB (mÃ¡x {ram_peak:.2f})",
            flush=True,
        )

    total_fold = sum(row["seconds"] for row in rows)
    n_measurements = len(config.seeds) * config.n_splits
    estimate = total_fold * n_measurements + len(config.seeds) * assembly_seconds
    print(f"Armado de datos: {assembly_seconds:.1f} s; pliegue (12 combinaciones): {total_fold:.1f} s")
    print(f"EstimaciÃ³n corrida completa ({n_measurements} mediciones): {estimate / 3600:.2f} h")
    if out:
        out.write_text(
            json.dumps(
                {
                    "config": sys.argv[1],
                    "assembly_seconds": assembly_seconds,
                    "fold_seconds": total_fold,
                    "estimate_seconds": estimate,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
