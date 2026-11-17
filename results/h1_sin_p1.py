"""Línea base de G1 sin P1 con las predicciones guardadas de la corrida v2 (paso P2, sin reentrenar)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from efd.evaluation.metrics import compute_metrics, compute_metrics_excluding, exclusion_mask, recall_by_group
from efd.evaluation.threshold import binarize

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/runs/20261113-101209_rf_g1_prev1_medio"
SCENARIO = ROOT / "data/synthetic/prev1_medio_s42"
OUT = ROOT / "results/h1_sin_p1.md"
EXCLUDED = ["isolated_emitter"]
METRICS = ["auc_pr", "f1"]
RELATIONAL = ["ring", "chain", "shared_representative"]


def stat(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main() -> None:
    run_hash = json.loads((RUN / "scenario.json").read_text(encoding="utf-8"))["config_hash"]
    data_hash = json.loads((SCENARIO / "metadata.json").read_text(encoding="utf-8"))["config_hash"]
    if run_hash != data_hash:
        raise SystemExit(f"config_hash distinto: corrida {run_hash[:8]}, escenario {data_hash[:8]}")
    summary = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))["summary"]

    predictions = pd.read_parquet(RUN / "predictions.parquet")
    patterns = pd.read_parquet(SCENARIO / "taxpayers.parquet", columns=["taxpayer_id", "pattern"])
    frame = predictions.merge(patterns, on="taxpayer_id", how="left", validate="many_to_one")

    full: dict[str, list[float]] = {m: [] for m in METRICS}
    sub: dict[str, list[float]] = {m: [] for m in METRICS}
    recalls: dict[str, list[float]] = {p: [] for p in RELATIONAL}
    n_folds = 0
    for (_, _), fold in frame.groupby(["seed", "fold"], sort=True):
        y, s, t = fold["label"].to_numpy(), fold["score"].to_numpy(), float(fold["threshold"].iloc[0])
        groups = fold["pattern"].to_numpy(dtype=object)
        for name, value in compute_metrics(y, s, t, METRICS).items():
            full[name].append(value)
        for name, value in compute_metrics_excluding(y, s, t, groups, EXCLUDED, METRICS).items():
            sub[name].append(value)
        keep = exclusion_mask(groups, EXCLUDED)
        by_pattern = recall_by_group(y[keep], binarize(s[keep], t), groups[keep])
        for pattern in RELATIONAL:
            recalls[pattern].append(by_pattern[pattern])
        n_folds += 1

    lines = ["# Línea base G1 sin P1 (corrida v2, sin reentrenar)", ""]
    lines += [
        f"- Corrida: `{RUN.name}`; escenario `{SCENARIO.relative_to(ROOT).as_posix()}` (`config_hash {run_hash[:8]}`).",
        f"- Mediciones: {n_folds} (5 pliegues × 1 semilla). La regla de H1 usa 15 (5 × 3 semillas): valor preliminar.",
        "- Población sin P1: legítimos + P2 (ring), P3 (chain) y P4 (shared_representative); umbral de cada pliegue.",
        "",
        "## Comprobación: métricas globales recalculadas frente a metrics.json",
        "",
        "| Métrica | recalculada | metrics.json |",
        "|---|---|---|",
    ]
    for name in METRICS:
        mean, _ = stat(full[name])
        stored = summary["metrics"][name]["mean"]
        if not np.isclose(mean, stored, rtol=0, atol=1e-12):
            raise SystemExit(f"{name}: recalculado {mean} distinto de metrics.json {stored}")
        lines.append(f"| {name} | {mean:.4f} | {stored:.4f} |")

    lines += ["", "## Resultado", "", "| Métrica | Global (con P1) | Sin P1 |", "|---|---|---|"]
    for name in METRICS:
        (fm, fs), (sm, ss) = stat(full[name]), stat(sub[name])
        lines.append(f"| {name} | {fm:.4f} ± {fs:.4f} | {sm:.4f} ± {ss:.4f} |")
    for pattern in RELATIONAL:
        m, s = stat(recalls[pattern])
        lines.append(f"| recall {pattern} | — | {m:.4f} ± {s:.4f} |")

    baseline = stat(sub["auc_pr"])[0]
    lines += ["", "## Criterio de H1 que aplica", ""]
    if baseline > 0.87:
        target = baseline + 0.15 * (1 - baseline)
        lines.append(
            f"Línea base sin P1 = {baseline:.4f} > 0,87: el criterio es cerrar ≥ 15 % de la brecha hasta 1, "
            f"es decir, AUC-PR de G3 ≥ {target:.4f} (y Wilcoxon significativo)."
        )
    else:
        target = baseline * 1.15
        lines.append(
            f"Línea base sin P1 = {baseline:.4f} ≤ 0,87: el criterio es la mejora relativa ≥ 15 %, "
            f"es decir, AUC-PR de G3 ≥ {target:.4f} (y Wilcoxon significativo)."
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Escrito {OUT}")


if __name__ == "__main__":
    main()
