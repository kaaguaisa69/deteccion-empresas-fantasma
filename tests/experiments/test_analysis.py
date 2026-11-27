"""Pruebas del análisis estadístico con corridas falsas en ``tmp_path``."""

import json

import numpy as np
import pandas as pd
import pytest

from efd.experiments.analysis import LIMITATION, MODELS, analyze, load_scores, main, render

LABELS = list(MODELS)
SEEDS = (11, 22, 33)
FOLDS = range(5)
KEYS = [(s, f) for s in SEEDS for f in FOLDS]

STRONG = {
    "RF G1": 0.40,
    "XGBoost G1": 0.50,
    "MLP G1": 0.30,
    "RF G2": 0.45,
    "XGBoost G2": 0.55,
    "GraphSAGE G3": 0.70,
    "GAT G3": 0.80,
}
SMALL_GAIN = {
    "RF G1": 0.80,
    "XGBoost G1": 0.90,
    "MLP G1": 0.70,
    "RF G2": 0.85,
    "XGBoost G2": 0.88,
    "GraphSAGE G3": 0.89,
    "GAT G3": 0.905,
}


def make_matrix(bases: dict[str, float], noise: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([bases[m] + rng.uniform(-noise, noise, len(KEYS)) for m in LABELS])


def write_run(root, label, matrix, *, shuffle=False, ids_suffix="", stamp="20261127-100000"):
    folder = root / f"{stamp}_{MODELS[label]}"
    folder.mkdir(parents=True)
    column = matrix[:, LABELS.index(label)]
    folds = [
        {
            "seed": s,
            "fold": f,
            "subpopulations": {"sin_p1": {"metrics": {"auc_pr": float(column[i])}}},
        }
        for i, (s, f) in enumerate(KEYS)
    ]
    if shuffle:
        folds = folds[::-1]
    (folder / "metrics.json").write_text(json.dumps({"folds": folds}), encoding="utf-8")
    (folder / "run_info.json").write_text(json.dumps({"commit": "abc1234"}), encoding="utf-8")
    rows = [
        {"taxpayer_id": f"t{s}-{f}-{k}{ids_suffix}", "seed": s, "fold": f, "score": 0.5, "label": 0}
        for s, f in KEYS
        for k in range(3)
    ]
    pd.DataFrame(rows).to_parquet(folder / "predictions.parquet", index=False)


def write_all(root, matrix, **kwargs):
    for label in LABELS:
        write_run(root, label, matrix, **kwargs)


def test_load_scores_orders_by_seed_and_fold(tmp_path):
    matrix = np.tile(np.array([s * 0.01 + f * 0.001 for s, f in KEYS])[:, None], (1, 7))
    write_all(tmp_path, matrix, shuffle=True)
    scores, labels, runs = load_scores(tmp_path)
    assert scores.shape == (15, 7)
    assert labels == LABELS
    np.testing.assert_allclose(scores, matrix)
    assert runs["GAT G3"]["folder"].endswith("_gat_g3_v2_ajuste")
    assert runs["GAT G3"]["commit"] == "abc1234"


def test_load_scores_stops_if_partitions_differ(tmp_path):
    matrix = make_matrix(STRONG, 0.01)
    for label in LABELS[:-1]:
        write_run(tmp_path, label, matrix)
    write_run(tmp_path, LABELS[-1], matrix, ids_suffix="x")
    with pytest.raises(ValueError, match="particiones"):
        load_scores(tmp_path)


def test_analyze_stops_if_best_g1_is_not_xgboost():
    bases = {**STRONG, "RF G1": 0.60}
    with pytest.raises(ValueError, match="XGBoost G1"):
        analyze(make_matrix(bases, 0.01), LABELS)


def test_h1_holds_and_report_has_required_parts():
    result = analyze(make_matrix(STRONG, 0.01), LABELS)
    assert result["h1"]["holds"]
    assert result["h1"]["reduction"] == pytest.approx((0.5 - 0.2) / 0.5, abs=0.05)
    report = render(result)
    assert "(a) H1" in report and "(b) secundaria" in report and "(c) control" in report
    assert "p ajustado (Holm)" in report
    assert "reducción del error" in report
    assert LIMITATION in report
    assert "H1 se cumple" in report


def test_h1_fails_if_reduction_below_015_even_with_small_p():
    result = analyze(make_matrix(SMALL_GAIN, 0.001), LABELS)
    assert result["friedman"]["significant"]
    assert result["comparisons"][0]["p_adjusted"] < 0.05
    assert result["h1"]["reduction"] < 0.15
    assert not result["h1"]["holds"]
    report = render(result)
    assert "H1 no se cumple" in report
    assert "menor que 0.15" in report


def test_friedman_not_significant_makes_wilcoxon_descriptive():
    bases = {m: 0.5 for m in LABELS}
    bases["XGBoost G1"] = 0.5005
    result = analyze(make_matrix(bases, 0.05, seed=3), LABELS)
    assert not result["friedman"]["significant"]
    assert not result["h1"]["holds"]
    assert all(c["decision"] == "descriptiva, Friedman no significativa" for c in result["comparisons"])
    report = render(result)
    assert "H1 no se cumple" in report
    assert "Friedman no es significativa" in report
    assert "descriptiva, Friedman no significativa" in report


def test_main_writes_markdown_and_json(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "out"
    runs.mkdir()
    write_all(runs, make_matrix(STRONG, 0.01))
    assert main(["--runs-root", str(runs), "--output", str(out)]) == 0
    assert (out / "analisis_estadistico.md").exists()
    data = json.loads((out / "analisis_estadistico.json").read_text(encoding="utf-8"))
    assert len(data["runs"]) == 7
    assert data["runs"]["XGBoost G1"]["commit"] == "abc1234"
