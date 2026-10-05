"""Constructores de variables tabulares (G1) y estructurales (G2).

Importar este paquete registra los constructores concretos para que puedan seleccionarse por nombre.
"""

from efd.features.base import BaseFeatureBuilder
from efd.features.registry import FEATURE_BUILDERS, create_feature_builder, register_feature_builder
from efd.features import structural, transactional  # noqa: F401  (registro)

__all__ = [
    "FEATURE_BUILDERS",
    "BaseFeatureBuilder",
    "create_feature_builder",
    "register_feature_builder",
]
