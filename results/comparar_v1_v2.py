"""Comparación de la prueba de dificultad v1 y v2 (paso P6b del plan de recalibración)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from efd.features.base import FeatureContext
from efd.features.transactional import TransactionalFeatureBuilder
from efd.network import InvoiceNetwork

ROOT = Path(__file__).resolve().parents[1]
V1 = {
    "run": ROOT / "results/runs/20261112-114831_rf_g1_prev1_medio",
    "data": ROOT / "data/synthetic/prev1_medio_s42_v1",
}
V2 = {
    "run": ROOT / "results/runs/20261113-101209_rf_g1_prev1_medio",
    "data": ROOT / "data/synthetic/prev1_medio_s42",
}
OUT = ROOT / "results/comparacion_v1_v2.md"
PATTERNS = {"isolated_emitter": "P1", "ring": "P2", "chain": "P3", "shared_representative": "P4"}
RELATIONAL = ("ring", "chain", "shared_representative")
FEATURES = ("age_days", "sales_mean_relative", "sales_counterparties", "purchases_to_sales")


def load(run: Path) -> tuple[dict, dict]:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["summary"]
    scenario = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    return metrics, scenario


def pm(stat: dict) -> str:
    return f"{stat['mean']:.4f}" if stat["std"] is None else f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def verdict(summary: dict) -> str:
    auc = summary["metrics"]["auc_pr"]["mean"]
    recalls = [summary["recall_by_pattern"][p]["mean"] for p in RELATIONAL]
    if auc >= 0.95 or all(r >= 0.90 for r in recalls):
        return "demasiado fácil"
    if auc <= 0.30:
        return "demasiado difícil"
    return "adecuado"


def feature_means(directory: Path) -> pd.DataFrame:
    network = InvoiceNetwork.load(directory)
    context = FeatureContext(
        period_start=pd.Timestamp("2025-01-01"), period_end=pd.Timestamp("2025-12-31"), seed=42
    )
    features = TransactionalFeatureBuilder(round_multiple=100).build(network, context)
    taxpayers = network.taxpayers.set_index("taxpayer_id")
    group = np.where(taxpayers["is_shell"] == 1, taxpayers["pattern"], "legit")
    frame = features.loc[taxpayers.index, list(FEATURES)].assign(group=group)
    return frame.groupby("group").mean()


def volume(directory: Path) -> pd.DataFrame:
    network = InvoiceNetwork.load(directory)
    taxpayers = network.taxpayers
    pattern = taxpayers.set_index("taxpayer_id")["pattern"]
    invoices = network.invoices
    rows = []
    for name in PATTERNS:
        ids = set(pattern[pattern == name].index)
        involved = invoices["issuer_id"].isin(ids) | invoices["receiver_id"].isin(ids)
        rows.append((PATTERNS[name], name, len(ids), int(involved.sum())))
    return pd.DataFrame(rows, columns=["patrón", "clave", "fantasmas", "facturas"])


def main() -> None:
    (m1, s1), (m2, s2) = load(V1["run"]), load(V2["run"])
    lines = ["# Prueba de dificultad: v1 frente a v2", ""]
    lines += [f"- v1: `{V1['run'].name}` sobre `{V1['data'].relative_to(ROOT).as_posix()}`"]
    lines += [f"- v2: `{V2['run'].name}` sobre `{V2['data'].relative_to(ROOT).as_posix()}`", ""]

    lines += ["## 1. Métricas (RF, G1, 5 pliegues, semilla 42)", "", "| Métrica | v1 | v2 |", "|---|---|---|"]
    lines.append(f"| config_hash | `{s1['config_hash'][:8]}` | `{s2['config_hash'][:8]}` |")
    lines.append(f"| facturas | {s1['invoices']:,} | {s2['invoices']:,} |".replace(",", " "))
    for name in ("auc_pr", "auc_roc", "f1"):
        lines.append(f"| {name} | {pm(m1['metrics'][name])} | {pm(m2['metrics'][name])} |")
    for name in ("precision_at_100", "precision_at_200", "precision_at_400"):
        lines.append(f"| {name} | {pm(m1['precision_at_k'][name])} | {pm(m2['precision_at_k'][name])} |")
    for key, label in PATTERNS.items():
        lines.append(
            f"| recall {label} ({key}) | {pm(m1['recall_by_pattern'][key])} | {pm(m2['recall_by_pattern'][key])} |"
        )

    lines += ["", "## 2. Veredicto (regla del 2026-10-05, lectura literal)", ""]
    lines += [
        "Demasiado fácil: AUC-PR ≥ 0,95, o recall ≥ 0,90 en P2, P3 y P4 a la vez. "
        "Demasiado difícil: AUC-PR ≤ 0,30. Adecuado: en otro caso.",
        "",
        f"- v1: **{verdict(m1)}** (AUC-PR {m1['metrics']['auc_pr']['mean']:.4f})",
        f"- v2: **{verdict(m2)}** (AUC-PR {m2['metrics']['auc_pr']['mean']:.4f})",
        "",
        "Hallazgos por patrón (recall medio ≥ 0,90):",
        "",
    ]
    for key in RELATIONAL:
        r1, r2 = m1["recall_by_pattern"][key]["mean"], m2["recall_by_pattern"][key]["mean"]
        lines.append(
            f"- {PATTERNS[key]} ({key}): v1 {r1:.3f} ({'sí' if r1 >= 0.90 else 'no'}); "
            f"v2 {r2:.3f} ({'sí' if r2 >= 0.90 else 'no'})"
        )

    f1, f2 = feature_means(V1["data"]), feature_means(V2["data"])
    lines += ["", "## 3. Rasgos que delataban a las fantasmas en la v1 (medias)", ""]
    lines += ["| Rasgo | Grupo | v1 | v2 |", "|---|---|---|---|"]
    groups = {
        "legit": "legítimas",
        "ring": "P2 (ring)",
        "chain": "P3 (chain)",
        "shared_representative": "P4 (shared_representative)",
    }
    for feature in FEATURES:
        for key, label in groups.items():
            lines.append(f"| {feature} | {label} | {f1.loc[key, feature]:.2f} | {f2.loc[key, feature]:.2f} |")

    lines += [
        "",
        "## 4. Volumen de la v2 por patrón",
        "",
        "| Patrón | Fantasmas | Facturas en que participan |",
        "|---|---|---|",
    ]
    for row in volume(V2["data"]).itertuples(index=False):
        lines.append(f"| {row[0]} ({row[1]}) | {row[2]} | {row[3]:,} |".replace(",", " "))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Escrito {OUT}")


if __name__ == "__main__":
    main()
