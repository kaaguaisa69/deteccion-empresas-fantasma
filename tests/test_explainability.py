from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from efd.data import ExperimentData
from efd.explainability import EXPLAINERS, BaseExplainer, Explanation, create_explainer
from efd.models import BaseDetector


class _FeatureCopyExplainer(BaseExplainer):
    """Explicador trivial que usa los valores de las variables como atribuciones."""

    name = "feature_copy"

    def __init__(self, shift: int = 0) -> None:
        self.shift = shift

    def _explain(self, detector: BaseDetector, data: ExperimentData, idx: np.ndarray) -> Explanation:
        X, _ = data.tabular((idx + self.shift) % data.n_nodes)
        return Explanation(feature_attributions=X)


def test_explain_returns_one_row_per_requested_company(small_data: ExperimentData) -> None:
    explanation = _FeatureCopyExplainer().explain(None, small_data, [2, 4])  # type: ignore[arg-type]
    assert list(explanation.feature_attributions.index) == ["E002", "E004"]
    assert explanation.edge_attributions is None


def test_explain_rejects_misaligned_attributions(small_data: ExperimentData) -> None:
    with pytest.raises(ValueError, match="distintas"):
        _FeatureCopyExplainer(shift=1).explain(None, small_data, [2, 4])  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["shap", "gnn_explainer"])
def test_concrete_explainers_are_registered(name: str, small_data: ExperimentData) -> None:
    assert name in EXPLAINERS
    explainer = create_explainer(name, seed=0)
    with pytest.raises(NotImplementedError):
        explainer.explain(None, small_data, [0])  # type: ignore[arg-type]


def test_explanation_accepts_edge_attributions() -> None:
    edges = pd.DataFrame({"source": ["E000"], "target": ["E001"], "importance": [0.9]})
    explanation = Explanation(feature_attributions=pd.DataFrame(), edge_attributions=edges)
    assert explanation.edge_attributions is edges
