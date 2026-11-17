"""Tabla de G1 y G2 con ajuste de hiperparámetros frente a sin ajuste, y línea base de H1 (paso P4)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/runs"
OUT = ROOT / "results/tabla_ajuste.md"
MODELS = [("RF · G1", "rf_g1"), ("XGB · G1", "xgb_g1"), ("RF · G2", "rf_g2"), ("XGB · G2", "xgb_g2")]


def latest(name: str) -> dict:
    runs = sorted(RUNS.glob(f"*_{name}"))
    if not runs:
        raise SystemExit(f"No hay corrida de {name}")
    metrics = json.loads((runs[-1] / "metrics.json").read_text(encoding="utf-8"))
    metrics["_run"] = runs[-1].name
    return metrics


def pm(stat: dict) -> str:
    std = "n/a" if stat["std"] is None else f"{stat['std']:.3f}"
    return f"{stat['mean']:.3f} ± {std}"


def row(label: str, metrics: dict, combination: str) -> str:
    summary = metrics["summary"]
    sub = summary["subpopulations"]["sin_p1"]
    recall = sub["recall_by_pattern"]
    return (
        f"| {label} | {pm(summary['metrics']['auc_pr'])} | {pm(sub['metrics']['auc_pr'])} | "
        f"{pm(sub['metrics']['f1'])} | {pm(summary['recall_by_pattern']['isolated_emitter'])} | "
        f"{pm(recall['ring'])} | {pm(recall['chain'])} | {pm(recall['shared_representative'])} | {combination} |"
    )


def most_frequent(metrics: dict) -> str:
    counts = Counter(json.dumps(f["tuning"]["params"], sort_keys=True) for f in metrics["folds"])
    params, n = counts.most_common(1)[0]
    total = len(metrics["folds"])
    text = ", ".join(f"{k}={v}" for k, v in json.loads(params).items())
    return f"{text} ({n}/{total})"


def main() -> None:
    header = (
        "| Modelo | AUC-PR global | AUC-PR sin P1 | F1 sin P1 | Recall P1 | Recall P2 | Recall P3 | Recall P4 | "
        "Combinación más elegida |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|"
    tuned = {key: latest(f"{key}_v2_ajuste") for _, key in MODELS}
    plain = {key: latest(f"{key}_v2_3s") for _, key in MODELS}

    lines = ["# G1 y G2 con y sin ajuste de hiperparámetros (v2, 3 semillas × 5 pliegues = 15)", ""]
    lines += ["## Con ajuste (≤ 12 combinaciones, elección por AUC-PR en validación)", "", header, rule]
    lines += [row(label, tuned[key], most_frequent(tuned[key])) for label, key in MODELS]
    lines += ["", "## Sin ajuste (13/11)", "", header, rule]
    lines += [row(label, plain[key], "fija") for label, key in MODELS]
    lines += ["", "Recall P1 es global; F1 y recall P2–P4 son sin P1. Media ± desviación muestral.", ""]

    lines += ["## Distribución de la combinación elegida (índice en la rejilla: pliegues)", ""]
    for label, key in MODELS:
        counts = tuned[key]["summary"]["tuning"]["selected_counts"]
        lines.append(f"- {label}: " + ", ".join(f"{i}: {n}" for i, n in counts.items()))

    best_label, best_key = max(
        [(label, key) for label, key in MODELS if key.endswith("g1")],
        key=lambda item: tuned[item[1]]["summary"]["subpopulations"]["sin_p1"]["metrics"]["auc_pr"]["mean"],
    )
    base = tuned[best_key]["summary"]["subpopulations"]["sin_p1"]["metrics"]["auc_pr"]["mean"]
    error = 1 - base
    target = 1 - 0.85 * error
    best_g2_label, best_g2_key = max(
        [(label, key) for label, key in MODELS if key.endswith("g2")],
        key=lambda item: tuned[item[1]]["summary"]["subpopulations"]["sin_p1"]["metrics"]["auc_pr"]["mean"],
    )
    g2 = tuned[best_g2_key]["summary"]["subpopulations"]["sin_p1"]["metrics"]["auc_pr"]["mean"]
    lines += [
        "",
        "## Línea base definitiva de H1",
        "",
        f"- Mejor G1 (AUC-PR sin P1 medio): {best_label}, {base:.4f}; error = {error:.4f}.",
        f"- G3 debe alcanzar un AUC-PR sin P1 ≥ {target:.4f} (error ≤ {0.85 * error:.4f}) y Wilcoxon significativo.",
        f"- Mejor G2: {best_g2_label}, {g2:.4f}; reducción descriptiva del error frente al mejor G1: "
        f"{(error - (1 - g2)) / error:.1%} (sin prueba estadística).",
        "",
        "## Corridas",
        "",
        *[
            f"- {label}: `{tuned[key]['_run']}` ({tuned[key]['duration_seconds']:.0f} s)"
            for label, key in MODELS
        ],
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Escrito {OUT}")


if __name__ == "__main__":
    main()
