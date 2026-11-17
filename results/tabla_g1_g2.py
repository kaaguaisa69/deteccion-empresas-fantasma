"""Tabla de G1 y G2 (RF y XGBoost, 3 semillas) sobre la v2: métricas globales y sin P1 (paso P4)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/runs"
OUT = ROOT / "results/tabla_g1_g2.md"
REFERENCE = RUNS / "20261113-101209_rf_g1_prev1_medio"
EXPERIMENTS = [
    ("RF · G1", "rf_g1_v2_3s"),
    ("XGB · G1", "xgb_g1_v2_3s"),
    ("RF · G2", "rf_g2_v2_3s"),
    ("XGB · G2", "xgb_g2_v2_3s"),
]


def latest(name: str) -> Path:
    runs = sorted(RUNS.glob(f"*_{name}"))
    if not runs:
        raise SystemExit(f"No hay corrida de {name}")
    return runs[-1]


def load(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def pm(stat: dict) -> str:
    std = "n/a" if stat["std"] is None else f"{stat['std']:.3f}"
    return f"{stat['mean']:.3f} ± {std}"


def main() -> None:
    lines = ["# G1 y G2 sobre la v2 (3 semillas × 5 pliegues = 15 mediciones)", ""]
    header = "| Modelo | AUC-PR global | AUC-PR sin P1 | F1 sin P1 | Recall P1 | Recall P2 | Recall P3 | Recall P4 | n |"
    lines += [header, "|---|---|---|---|---|---|---|---|---|"]
    details = []
    for label, name in EXPERIMENTS:
        run = latest(name)
        metrics = load(run)
        summary = metrics["summary"]
        sub = summary["subpopulations"]["sin_p1"]
        lines.append(
            f"| {label} | {pm(summary['metrics']['auc_pr'])} | {pm(sub['metrics']['auc_pr'])} | "
            f"{pm(sub['metrics']['f1'])} | {pm(summary['recall_by_pattern']['isolated_emitter'])} | "
            f"{pm(sub['recall_by_pattern']['ring'])} | {pm(sub['recall_by_pattern']['chain'])} | "
            f"{pm(sub['recall_by_pattern']['shared_representative'])} | {summary['metrics']['auc_pr']['n']} |"
        )
        seconds = summary["feature_seconds"]
        details.append(
            f"| {label} | `{run.name}` | {metrics['duration_seconds']:.0f} | "
            f"{seconds['transactional']['mean']:.1f} | "
            f"{seconds['structural']['mean']:.1f} ± {seconds['structural']['std']:.1f} |"
            if "structural" in seconds
            else f"| {label} | `{run.name}` | {metrics['duration_seconds']:.0f} | "
            f"{seconds['transactional']['mean']:.1f} | — |"
        )

    lines += [
        "",
        "Recall P1 es global; P2–P4 y F1 son sobre la población sin P1. Media ± desviación muestral.",
        "",
    ]
    lines += ["## Corridas y tiempos (segundos)", ""]
    lines += [
        "| Modelo | Corrida | Duración total | Variables G1 por semilla | Métricas de red (G2) por semilla |",
        "|---|---|---|---|---|",
        *details,
    ]

    reference = {f["fold"]: f["metrics"]["auc_pr"] for f in load(REFERENCE)["folds"]}
    rf_g1 = {
        f["fold"]: f["metrics"]["auc_pr"] for f in load(latest("rf_g1_v2_3s"))["folds"] if f["seed"] == 42
    }
    same = reference == rf_g1
    lines += [
        "",
        "## Comprobación",
        "",
        f"RF · G1 con semilla 42 reproduce los pliegues de `{REFERENCE.name}`: {'sí' if same else 'NO'}.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Escrito {OUT}; reproducción semilla 42: {same}")


if __name__ == "__main__":
    main()
