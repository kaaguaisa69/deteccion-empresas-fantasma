"""Análisis estadístico de las siete corridas ``*_v2_ajuste`` (AUC-PR sin P1, Friedman, Wilcoxon, Holm, H1).

Orden: ``python -m efd.experiments.analysis [--runs-root results/runs] [--output results/analisis]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from efd.evaluation.significance import error_reduction, friedman, holm_adjust, paired_wilcoxon

MODELS: dict[str, str] = {
    "RF G1": "rf_g1_v2_ajuste",
    "XGBoost G1": "xgb_g1_v2_ajuste",
    "MLP G1": "mlp_g1_v2_ajuste",
    "RF G2": "rf_g2_v2_ajuste",
    "XGBoost G2": "xgb_g2_v2_ajuste",
    "GraphSAGE G3": "sage_g3_v2_ajuste",
    "GAT G3": "gat_g3_v2_ajuste",
}
G1 = ("RF G1", "XGBoost G1")
G2 = ("RF G2", "XGBoost G2")
G3 = ("GraphSAGE G3", "GAT G3")
EXPECTED_BEST_G1 = "XGBoost G1"
ALPHA = 0.05
MIN_REDUCTION = 0.15
LIMITATION = (
    "Las 15 mediciones no son independientes: las particiones comparten datos de entrenamiento, por lo que "
    "los valores p son orientativos y tienden a ser optimistas (Demšar, 2006)."
)
DEFAULT_RUNS_ROOT = Path("results/runs")
DEFAULT_OUTPUT = Path("results/analisis")


def _latest_run(runs_root: Path, experiment: str) -> Path:
    candidates = sorted(
        d for d in runs_root.glob(f"*_{experiment}") if d.is_dir() and (d / "metrics.json").exists()
    )
    candidates = [d for d in candidates if d.name.split("_", 1)[1] == experiment]
    if not candidates:
        raise FileNotFoundError(f"No hay corrida con metrics.json para {experiment} en {runs_root}.")
    return candidates[-1]


def _commit_of(directory: Path) -> str | None:
    info_path = directory / "run_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for key in ("commit", "git_commit", "commit_hash"):
        if key in info:
            return info[key]
    return None


def _partitions(directory: Path) -> dict[tuple[int, int], frozenset]:
    frame = pd.read_parquet(directory / "predictions.parquet", columns=["taxpayer_id", "seed", "fold"])
    return {
        (int(seed), int(fold)): frozenset(group["taxpayer_id"].tolist())
        for (seed, fold), group in frame.groupby(["seed", "fold"])
    }


def load_scores(runs_root: Path) -> tuple[np.ndarray, list[str], dict[str, dict[str, Any]]]:
    """Lee las siete corridas y devuelve ``(scores, labels, runs)``.

    ``scores`` tiene forma ``(n_mediciones, n_modelos)`` con el AUC-PR sin P1 de cada pliegue, ordenado
    por (``seed``, ``fold``). ``labels`` sigue el orden de ``MODELS`` y ``runs`` guarda la carpeta y el
    commit de cada corrida.

    Raises:
        FileNotFoundError: si falta la corrida de algún modelo.
        ValueError: si los modelos no usan las mismas particiones.
    """
    runs_root = Path(runs_root)
    labels = list(MODELS)
    columns: list[list[float]] = []
    runs: dict[str, dict[str, Any]] = {}
    reference_keys: list[tuple[int, int]] | None = None
    reference_partitions: dict[tuple[int, int], frozenset] | None = None
    for label in labels:
        directory = _latest_run(runs_root, MODELS[label])
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        folds = sorted(metrics["folds"], key=lambda f: (f["seed"], f["fold"]))
        keys = [(int(f["seed"]), int(f["fold"])) for f in folds]
        partitions = _partitions(directory)
        if reference_keys is None:
            reference_keys, reference_partitions = keys, partitions
        elif keys != reference_keys or partitions != reference_partitions:
            raise ValueError(
                f"Las particiones de {label} difieren de las de {labels[0]}; no hay parejas válidas."
            )
        columns.append([float(f["subpopulations"]["sin_p1"]["metrics"]["auc_pr"]) for f in folds])
        runs[label] = {"folder": directory.name, "commit": _commit_of(directory)}
    return np.column_stack(columns), labels, runs


def _best(group: tuple[str, ...], means: dict[str, float]) -> str:
    return max(group, key=lambda label: means[label])


def analyze(scores: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Calcula Friedman, las tres Wilcoxon con Holm y la decisión de H1.

    Raises:
        ValueError: si el mejor modelo G1 no es XGBoost G1.
    """
    scores = np.asarray(scores, dtype=float)
    column = {label: scores[:, i] for i, label in enumerate(labels)}
    means = {label: float(column[label].mean()) for label in labels}
    fried = friedman(scores)
    models = {
        label: {
            "mean": means[label],
            "std": float(column[label].std(ddof=1)),
            "mean_rank": fried["mean_ranks"][i],
        }
        for i, label in enumerate(labels)
    }

    best_g1, best_g2, best_g3 = _best(G1, means), _best(G2, means), _best(G3, means)
    if best_g1 != EXPECTED_BEST_G1:
        raise ValueError(f"El mejor G1 es {best_g1} y debía ser {EXPECTED_BEST_G1}; el análisis se detiene.")

    friedman_significant = fried["p_value"] < ALPHA
    pairs = [
        ("(a) H1: mejor G3 contra mejor G1", best_g3, best_g1),
        ("(b) secundaria: mejor G3 contra mejor G2", best_g3, best_g2),
        ("(c) control: mejor G3 contra MLP G1", best_g3, "MLP G1"),
    ]
    tests = [paired_wilcoxon(column[a], column[b]) for _, a, b in pairs]
    adjusted = holm_adjust([t["p_value"] for t in tests])
    comparisons = []
    for (name, a, b), test, p_adj in zip(pairs, tests, adjusted, strict=True):
        if not friedman_significant:
            decision = "descriptiva, Friedman no significativa"
        elif p_adj < ALPHA:
            decision = "diferencia significativa"
        else:
            decision = "sin diferencia significativa"
        comparisons.append(
            {
                "name": name,
                "a": a,
                "b": b,
                "mean_a": means[a],
                "mean_b": means[b],
                "difference": means[a] - means[b],
                "statistic": test["statistic"],
                "p_value": test["p_value"],
                "p_adjusted": p_adj,
                "decision": decision,
            }
        )

    primary = comparisons[0]
    reduction = error_reduction(means[best_g1], means[best_g3])
    conditions = {
        "friedman_significativa": bool(friedman_significant),
        "reduccion_minima": bool(reduction >= MIN_REDUCTION),
        "holm_significativo": bool(primary["p_adjusted"] < ALPHA),
        "diferencia_a_favor_de_g3": bool(primary["difference"] > 0),
    }
    return {
        "n_measurements": int(scores.shape[0]),
        "models": models,
        "friedman": {
            "statistic": fried["statistic"],
            "p_value": fried["p_value"],
            "significant": bool(friedman_significant),
        },
        "best": {"G1": best_g1, "G2": best_g2, "G3": best_g3},
        "comparisons": comparisons,
        "h1": {
            "error_g1": 1.0 - means[best_g1],
            "error_g3": 1.0 - means[best_g3],
            "reduction": reduction,
            "conditions": conditions,
            "holds": all(conditions.values()),
        },
    }


_CONDITION_TEXT = {
    "friedman_significativa": "la prueba de Friedman no es significativa (p >= 0.05)",
    "reduccion_minima": "la reducción del error es menor que 0.15",
    "holm_significativo": "el p ajustado de la comparación (a) no es menor que 0.05",
    "diferencia_a_favor_de_g3": "la diferencia media no favorece a G3",
}


def render(result: dict[str, Any]) -> str:
    """Genera el informe en Markdown con tablas, la decisión de H1, la explicación y la limitación."""
    fried, h1 = result["friedman"], result["h1"]
    lines = [
        "# Análisis estadístico",
        "",
        f"Métrica: AUC-PR sin P1, {result['n_measurements']} mediciones por modelo (semilla × pliegue).",
        "",
        "## Medias por modelo",
        "",
        "| Modelo | AUC-PR sin P1 (media ± desviación) | Rango medio |",
        "| --- | --- | --- |",
    ]
    for label, m in result["models"].items():
        lines.append(f"| {label} | {m['mean']:.4f} ± {m['std']:.4f} | {m['mean_rank']:.2f} |")
    lines += [
        "",
        "## Friedman",
        "",
        f"Estadístico = {fried['statistic']:.4f}, p = {fried['p_value']:.4g} "
        f"({'significativa' if fried['significant'] else 'no significativa'} con α = {ALPHA}).",
        "",
        "## Comparaciones pareadas (Wilcoxon bilateral, corrección de Holm, m = 3)",
        "",
        "| Comparación | A | B | Media A | Media B | Diferencia | W | p | p ajustado (Holm) | Decisión |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in result["comparisons"]:
        lines.append(
            f"| {c['name']} | {c['a']} | {c['b']} | {c['mean_a']:.4f} | {c['mean_b']:.4f} | "
            f"{c['difference']:+.4f} | {c['statistic']:.1f} | {c['p_value']:.4g} | "
            f"{c['p_adjusted']:.4g} | {c['decision']} |"
        )
    if not fried["significant"]:
        lines += [
            "",
            "Como la prueba de Friedman no es significativa, las comparaciones de Wilcoxon se reportan solo "
            "como descriptivas (protocolo 2.6.5).",
        ]
    failed = [_CONDITION_TEXT[k] for k, ok in h1["conditions"].items() if not ok]
    lines += [
        "",
        "## H1",
        "",
        f"Error del mejor G1 ({result['best']['G1']}) = {h1['error_g1']:.4f}; "
        f"error del mejor G3 ({result['best']['G3']}) = {h1['error_g3']:.4f}; "
        f"reducción del error = {h1['reduction']:.4f} (mínimo exigido {MIN_REDUCTION}).",
        "",
        "H1 se cumple solo si se cumplen las cuatro condiciones: Friedman significativa, reducción >= 0.15, "
        "p ajustado de (a) < 0.05 y diferencia media a favor de G3.",
        "",
        "**Decisión: H1 se cumple.**"
        if h1["holds"]
        else "**Decisión: H1 no se cumple.** Condiciones que fallan: " + "; ".join(failed) + ".",
        "",
        "## Explicación en lenguaje sencillo",
        "",
        f"Se compararon siete modelos con las mismas {result['n_measurements']} particiones de los datos. "
        f"La prueba de Friedman {'sí' if fried['significant'] else 'no'} encontró diferencias entre los modelos "
        f"(p = {fried['p_value']:.4g}). El mejor modelo del grupo G3 ({result['best']['G3']}) obtuvo en "
        f"promedio {result['comparisons'][0]['mean_a']:.4f} de AUC-PR sin P1, frente a "
        f"{result['comparisons'][0]['mean_b']:.4f} del mejor del grupo G1 ({result['best']['G1']}); eso "
        f"equivale a una reducción del error de {h1['reduction']:.1%}.",
        "",
        "## Limitación",
        "",
        LIMITATION,
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Lee las corridas, calcula el análisis y escribe el Markdown y el JSON."""
    parser = argparse.ArgumentParser(description="Análisis estadístico de las siete corridas.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    scores, labels, runs = load_scores(args.runs_root)
    result = analyze(scores, labels)
    result["runs"] = runs
    report = render(result)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analisis_estadistico.md").write_text(report, encoding="utf-8")
    (args.output / "analisis_estadistico.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
