"""Pruebas de H1 con F1 y de H2 con corridas falsas en ``tmp_path``."""

import json

import numpy as np
import pandas as pd

from efd.experiments.analysis import MODELS
from efd.experiments.analysis_h import analyze_h1, analyze_h2, main, render_h1, render_h2

LABELS = list(MODELS)
KEYS = [(s, f) for s in (11, 22, 33) for f in range(5)]
AUC = {"RF G1": 0.4, "XGBoost G1": 0.5, "MLP G1": 0.3, "RF G2": 0.45, "XGBoost G2": 0.55,
       "GraphSAGE G3": 0.7, "GAT G3": 0.8}  # fmt: skip


def write_runs(root, *, f1_gain=True, fpr_gat=0.001, fpr_base=0.004, precision_gat=0.9, precision_base=0.5):
    rng = np.random.default_rng(0)
    for label in LABELS:
        gat = label == "GAT G3"
        folds = []
        for s, f in KEYS:
            noise = rng.uniform(-0.01, 0.01)
            f1 = AUC[label] + noise
            if not f1_gain and label in ("GAT G3", "GraphSAGE G3"):
                f1 = (0.40 if gat else 0.45) + noise
            folds.append(
                {
                    "seed": s,
                    "fold": f,
                    "subpopulations": {
                        "sin_p1": {
                            "metrics": {
                                "auc_pr": AUC[label] + noise,
                                "f1": f1,
                                "fpr": (fpr_gat if gat else fpr_base) + rng.uniform(0, 0.0002),
                                "precision": (precision_gat if gat else precision_base)
                                + rng.uniform(0, 0.01),
                            }
                        }
                    },
                }
            )
        folder = root / f"20261127-100000_{MODELS[label]}"
        folder.mkdir(parents=True)
        summary = {"precision_at_k": {"precision_at_100": {"mean": 0.9 if gat else 0.5}}}
        (folder / "metrics.json").write_text(
            json.dumps({"folds": folds, "summary": summary}), encoding="utf-8"
        )
        (folder / "run_info.json").write_text(json.dumps({"commit": "abc1234"}), encoding="utf-8")
        rows = [
            {"taxpayer_id": f"t{s}-{f}-{k}", "seed": s, "fold": f, "score": 0.5, "label": 0}
            for s, f in KEYS
            for k in range(3)
        ]
        pd.DataFrame(rows).to_parquet(folder / "predictions.parquet", index=False)


def test_h1_requires_both_auc_pr_and_f1(tmp_path):
    (tmp_path / "ok").mkdir()
    write_runs(tmp_path / "ok")
    both = analyze_h1(tmp_path / "ok")
    assert both["auc_pr"]["h1"]["holds"] and both["f1"]["h1"]["holds"] and both["holds"]
    assert "Decisión conjunta: H1 se cumple" in render_h1(both)

    (tmp_path / "nof1").mkdir()
    write_runs(tmp_path / "nof1", f1_gain=False)
    partial = analyze_h1(tmp_path / "nof1")
    assert partial["auc_pr"]["h1"]["holds"]
    assert not partial["f1"]["h1"]["holds"]
    assert not partial["holds"]
    assert "Decisión conjunta: H1 no se cumple" in render_h1(partial)


def test_h1_f1_uses_best_models_chosen_by_auc_pr(tmp_path):
    write_runs(tmp_path, f1_gain=False)
    result = analyze_h1(tmp_path)
    assert result["f1"]["best"] == result["auc_pr"]["best"]
    assert result["f1"]["best_by_own_mean"] != result["f1"]["best"]


def test_h2_holds_if_fpr_or_precision_significantly_better(tmp_path):
    write_runs(tmp_path)
    result = analyze_h2(tmp_path)
    assert result["holds"]
    assert result["gat"] == "GAT G3" and result["baseline"] == "XGBoost G1"
    assert {r["metric"] for r in result["primary"]} == {"fpr", "precision"}
    assert all(r["p_adjusted"] >= r["p_value"] for r in result["primary"])
    assert len(result["descriptive"]) == 4
    report = render_h2(result)
    assert "H2 se cumple" in report and "precision@k" in report and "p ajustado (Holm)" in report


def test_h2_fails_without_difference(tmp_path):
    write_runs(tmp_path, fpr_gat=0.004, precision_gat=0.5)
    result = analyze_h2(tmp_path)
    assert not result["holds"]
    assert "H2 no se cumple" in render_h2(result)


def test_h2_difference_against_gat_does_not_count(tmp_path):
    write_runs(tmp_path, fpr_gat=0.01, precision_gat=0.2)
    result = analyze_h2(tmp_path)
    assert not result["holds"]
    assert all(not r["significant_for_gat"] for r in result["primary"])


def test_main_writes_both_reports(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "out"
    runs.mkdir()
    write_runs(runs)
    assert main(["--runs-root", str(runs), "--output", str(out)]) == 0
    for name in ("analisis_h1.md", "analisis_h1.json", "analisis_h2.md", "analisis_h2.json"):
        assert (out / name).exists()
    assert json.loads((out / "analisis_h2.json").read_text(encoding="utf-8"))["holds"] is True
