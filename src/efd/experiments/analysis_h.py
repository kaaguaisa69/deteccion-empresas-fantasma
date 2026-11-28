"""H1 corregida (AUC-PR y F1 sin P1) y H2 (FPR y precisión sin P1) sobre las siete corridas ``*_v2_ajuste``.

Orden: ``python -m efd.experiments.analysis_h [--runs-root results/runs] [--output results/analisis]``.
Escribe ``analisis_h1.{md,json}`` y ``analisis_h2.{md,json}``. Criterios registrados en la bitácora del 27-nov.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from efd.evaluation.significance import holm_adjust, paired_wilcoxon
from efd.experiments.analysis import (
    ALPHA,
    DEFAULT_OUTPUT,
    DEFAULT_RUNS_ROOT,
    LIMITATION,
    MODELS,
    _latest_run,
    analyze,
    load_scores,
    render,
)

H2_BASELINES = ("XGBoost G2", "MLP G1")
# (clave de la métrica, nombre, True si un valor menor favorece a GAT)
H2_METRICS = (("fpr", "FPR sin P1", True), ("precision", "Precisión sin P1", False))


def analyze_h1(runs_root: Path) -> dict[str, Any]:
    """H1: el criterio de cuatro condiciones para AUC-PR sin P1 y para F1 sin P1; se cumple si se cumplen ambos.

    Los mejores modelos del F1 son los elegidos por AUC-PR media (``best_by_own_mean`` informa los del F1).
    """
    auc_pr, labels, runs = load_scores(runs_root, "auc_pr")
    f1, _, _ = load_scores(runs_root, "f1")
    result_auc = analyze(auc_pr, labels)
    result_f1 = analyze(f1, labels, best=result_auc["best"])
    return {
        "auc_pr": result_auc,
        "f1": result_f1,
        "holds": bool(result_auc["h1"]["holds"] and result_f1["h1"]["holds"]),
        "runs": runs,
    }


def _wilcoxon_row(a: np.ndarray, b: np.ndarray, lower_is_better: bool) -> dict[str, Any]:
    test = paired_wilcoxon(a, b)
    diff = test["mean_difference"]
    return {
        "mean_gat": float(np.mean(a)),
        "mean_baseline": float(np.mean(b)),
        "difference": diff,
        "statistic": test["statistic"],
        "p_value": test["p_value"],
        "favors_gat": bool(diff < 0 if lower_is_better else diff > 0),
    }


def analyze_h2(runs_root: Path) -> dict[str, Any]:
    """H2: GAT contra XGBoost G1 con FPR y precisión sin P1 (Wilcoxon bilateral, Holm m = 2).

    Se acepta si alguna de las dos métricas es significativamente mejor para GAT. GAT contra XGBoost G2 y
    MLP se informan como descriptivas (sin corrección). También se informan FPR absoluta y precision@k.
    """
    auc_pr, labels, runs = load_scores(runs_root, "auc_pr")
    best = analyze(auc_pr, labels)["best"]
    gat, baseline = best["G3"], best["G1"]
    index = {label: i for i, label in enumerate(labels)}

    columns: dict[str, np.ndarray] = {}
    for key, _, _ in H2_METRICS:
        columns[key] = load_scores(runs_root, key)[0]

    primary = []
    for key, name, lower in H2_METRICS:
        row = _wilcoxon_row(columns[key][:, index[gat]], columns[key][:, index[baseline]], lower)
        primary.append({"metric": key, "name": name, **row})
    for row, adjusted in zip(primary, holm_adjust([r["p_value"] for r in primary]), strict=True):
        row["p_adjusted"] = adjusted
        row["significant_for_gat"] = bool(adjusted < ALPHA and row["favors_gat"])

    descriptive = []
    for other in (best["G2"], *[m for m in H2_BASELINES if m != best["G2"]]):
        for key, name, lower in H2_METRICS:
            row = _wilcoxon_row(columns[key][:, index[gat]], columns[key][:, index[other]], lower)
            descriptive.append({"baseline": other, "metric": key, "name": name, **row})

    absolute = {
        label: {
            key: {
                "mean": float(columns[key][:, i].mean()),
                "std": float(columns[key][:, i].std(ddof=1)),
            }
            for key, _, _ in H2_METRICS
        }
        for label, i in index.items()
    }
    precision_at_k = {}
    for label in labels:
        summary = json.loads((_latest_run(runs_root, MODELS[label]) / "metrics.json").read_text("utf-8"))
        precision_at_k[label] = {k: v["mean"] for k, v in summary["summary"]["precision_at_k"].items()}
    return {
        "gat": gat,
        "baseline": baseline,
        "primary": primary,
        "descriptive": descriptive,
        "absolute": absolute,
        "precision_at_k": precision_at_k,
        "holds": bool(any(r["significant_for_gat"] for r in primary)),
        "runs": runs,
    }


def render_h1(result: dict[str, Any]) -> str:
    """Informe de H1: los dos análisis (AUC-PR y F1) y la decisión conjunta."""
    auc, f1 = result["auc_pr"], result["f1"]
    differs = f1["best_by_own_mean"] != f1["best"]
    lines = [
        "# H1 corregida: AUC-PR y F1 sin P1",
        "",
        "H1 se cumple solo si se cumple el criterio (cuatro condiciones) para AUC-PR sin P1 y para F1 sin P1.",
        "",
        f"- AUC-PR sin P1: {'se cumple' if auc['h1']['holds'] else 'no se cumple'}.",
        f"- F1 sin P1: {'se cumple' if f1['h1']['holds'] else 'no se cumple'}.",
        "",
        "**Decisión conjunta: H1 se cumple.**"
        if result["holds"]
        else "**Decisión conjunta: H1 no se cumple.**",
        "",
        "El F1 se agregó después de ver los resultados de G3, para alinear el criterio con el Plan.",
        "",
        "---",
        "",
        render(auc, "AUC-PR sin P1"),
        "---",
        "",
        render(f1, "F1 sin P1"),
        "Mejores modelos del F1 fijados desde AUC-PR: "
        + ", ".join(f"{g} = {m}" for g, m in f1["best"].items())
        + ". Por la media del F1 saldrían: "
        + ", ".join(f"{g} = {m}" for g, m in f1["best_by_own_mean"].items())
        + (" (difieren)." if differs else " (coinciden)."),
        "",
    ]
    return "\n".join(lines)


def render_h2(result: dict[str, Any]) -> str:
    """Informe de H2: dos pruebas con Holm (m = 2), descriptivas, FPR absoluta y precision@k."""
    lines = [
        "# H2: falsos positivos",
        "",
        f"{result['gat']} contra {result['baseline']}. Se acepta si la FPR sin P1 es significativamente menor "
        "o la precisión sin P1 es significativamente mayor (Wilcoxon bilateral, Holm con m = 2, diferencia a "
        "favor de GAT). Sin umbral de reducción.",
        "",
        "## Prueba principal",
        "",
        "| Métrica | Media GAT | Media base | Diferencia | W | p | p ajustado (Holm) | Decisión |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["primary"]:
        decision = (
            "significativa a favor de GAT"
            if r["significant_for_gat"]
            else (
                "sin diferencia significativa"
                if r["p_adjusted"] >= ALPHA
                else "significativa en contra de GAT"
            )
        )
        lines.append(
            f"| {r['name']} | {r['mean_gat']:.5f} | {r['mean_baseline']:.5f} | {r['difference']:+.5f} | "
            f"{r['statistic']:.1f} | {r['p_value']:.4g} | {r['p_adjusted']:.4g} | {decision} |"
        )
    lines += [
        "",
        "**Decisión: H2 se cumple.**" if result["holds"] else "**Decisión: H2 no se cumple.**",
        "",
        "## Comparaciones descriptivas (sin corrección ni decisión)",
        "",
        "| Contra | Métrica | Media GAT | Media base | Diferencia | p sin corregir |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["descriptive"]:
        lines.append(
            f"| {r['baseline']} | {r['name']} | {r['mean_gat']:.5f} | {r['mean_baseline']:.5f} | "
            f"{r['difference']:+.5f} | {r['p_value']:.4g} |"
        )
    lines += ["", "## FPR y precisión absolutas (media ± desviación, 15 mediciones)", ""]
    lines += [
        "| Modelo | FPR sin P1 | Precisión sin P1 | Falsos positivos por 10 000 legítimos |",
        "| --- | --- | --- | --- |",
    ]
    for label, m in result["absolute"].items():
        f, p = m["fpr"], m["precision"]
        lines.append(
            f"| {label} | {f['mean']:.5f} ± {f['std']:.5f} | {p['mean']:.4f} ± {p['std']:.4f} | "
            f"{f['mean'] * 10000:.1f} |"
        )
    ks = sorted({k for v in result["precision_at_k"].values() for k in v})
    lines += ["", "## precision@k (media de las 3 semillas)", ""]
    lines += ["| Modelo | " + " | ".join(ks) + " |", "| --- | " + " | ".join("---" for _ in ks) + " |"]
    for label, v in result["precision_at_k"].items():
        lines.append(f"| {label} | " + " | ".join(f"{v[k]:.4f}" for k in ks) + " |")
    lines += ["", "## Limitación", "", LIMITATION, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Calcula H1 corregida y H2 y escribe los informes Markdown y JSON."""
    parser = argparse.ArgumentParser(description="H1 corregida (AUC-PR y F1) y H2 sobre las siete corridas.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    h1, h2 = analyze_h1(args.runs_root), analyze_h2(args.runs_root)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {"analisis_h1": (h1, render_h1(h1)), "analisis_h2": (h2, render_h2(h2))}
    for name, (data, report) in reports.items():
        (args.output / f"{name}.md").write_text(report, encoding="utf-8")
        (args.output / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    sys.stdout.reconfigure(encoding="utf-8")
    print(reports["analisis_h1"][1])
    print(reports["analisis_h2"][1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
