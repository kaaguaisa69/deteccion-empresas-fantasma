"""Mide GAT sobre Elliptic (partición temporal 1–34 / 35–49, inductivo estricto): una combinación, una semilla.

Uso: python scripts/medir_gat_elliptic.py --device cpu|cuda [--root data/elliptic] [--seed 42]
Registra tiempo de ajuste y por época, memoria máxima de GPU y AUC-PR de la clase ilícita en la prueba.
Guarda results/elliptic_gat_<device>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from efd.elliptic import fit_gat, load_elliptic, predict_test, temporal_split
from efd.evaluation.metrics import auc_pr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=Path("data/elliptic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    data = load_elliptic(args.root)
    split = temporal_split(data)
    result: dict = {
        "device": args.device,
        "torch": torch.__version__,
        "params": {"heads": args.heads, "hidden_channels": args.hidden, "dropout": args.dropout},
        "seed": args.seed,
        "train_nodes": int(split.train.x.shape[0]),
        "train_edges": int(split.train.edge_index.shape[1]),
        "train_labeled": int(split.train.idx.size),
        "val_labeled": int(split.train.val_idx.size),
        "test_nodes": int(split.test.x.shape[0]),
        "test_edges": int(split.test.edge_index.shape[1]),
        "test_labeled": int(split.test.idx.size),
        "test_illicit": int(split.test.y[split.test.idx].sum()),
    }
    if args.device == "cuda":
        result["gpu"] = torch.cuda.get_device_name(0)
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        fit = fit_gat(split, args.seed, args.hidden, args.heads, dropout=args.dropout, device=args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
    except torch.OutOfMemoryError as error:
        result["out_of_memory"] = str(error).splitlines()[0]
        fit = None
    result["fit_seconds"] = round(time.perf_counter() - started, 2)
    if args.device == "cuda":
        result["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        result["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    if fit is not None:
        scores = predict_test(fit, split)
        y_test = split.test.y[split.test.idx]
        result["epochs_run"] = len(fit.history)
        result["best_epoch"] = fit.best_epoch
        result["seconds_per_epoch"] = round(result["fit_seconds"] / len(fit.history), 3)
        result["test_auc_pr_illicit"] = auc_pr(y_test, scores)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/elliptic_gat_{args.device}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
