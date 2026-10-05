"""Registro de constructores de variables por nombre para seleccionarlos desde la configuración."""

from __future__ import annotations

from typing import Any

from efd.features.base import BaseFeatureBuilder
from efd.registry import ClassRegistry

FEATURE_BUILDERS: ClassRegistry[BaseFeatureBuilder] = ClassRegistry(
    "constructor de variables", BaseFeatureBuilder
)


def register_feature_builder(cls: type[BaseFeatureBuilder]) -> type[BaseFeatureBuilder]:
    """Decorador que registra un constructor de variables bajo su atributo ``name``."""
    return FEATURE_BUILDERS.register(cls)


def create_feature_builder(name: str, **params: Any) -> BaseFeatureBuilder:
    """Instancia el constructor registrado como ``name`` con los parámetros dados."""
    return FEATURE_BUILDERS.create(name, **params)
