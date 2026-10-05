"""Explicabilidad de las predicciones.

Importar este paquete registra los explicadores concretos para que puedan instanciarse por nombre.
"""

from efd.explainability.base import BaseExplainer, Explanation
from efd.explainability.registry import EXPLAINERS, create_explainer, register_explainer
from efd.explainability import gnn_explainer, shap_explainer  # noqa: F401  (registro)

__all__ = ["EXPLAINERS", "BaseExplainer", "Explanation", "create_explainer", "register_explainer"]
