"""Constructores de variables tabulares (G1) y estructurales (G2).

Importar este paquete registra los constructores concretos para que puedan seleccionarse por nombre.
"""

from efd.features.base import BaseFeatureBuilder, FeatureContext
from efd.features.registry import (
    FEATURE_BUILDERS,
    FEATURE_SETS,
    build_feature_set,
    create_feature_builder,
    register_feature_builder,
)
from efd.features import structural, transactional  # noqa: F401  (registro)

__all__ = [
    "FEATURE_BUILDERS",
    "FEATURE_SETS",
    "BaseFeatureBuilder",
    "FeatureContext",
    "build_feature_set",
    "create_feature_builder",
    "register_feature_builder",
]
