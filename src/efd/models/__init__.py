"""Detectores de empresas fantasmas.

Importar este paquete registra los detectores concretos para que puedan instanciarse por nombre.
"""

from efd.models.base import BaseDetector
from efd.models.registry import available_detectors, create_detector, register_detector
from efd.models import gat, graphsage, random_forest, xgboost_model  # noqa: F401  (registro)

__all__ = ["BaseDetector", "available_detectors", "create_detector", "register_detector"]
