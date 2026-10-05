"""Registro de explicadores por nombre para instanciarlos desde la configuración."""

from __future__ import annotations

from typing import Any

from efd.explainability.base import BaseExplainer
from efd.registry import ClassRegistry

EXPLAINERS: ClassRegistry[BaseExplainer] = ClassRegistry("explicador", BaseExplainer)


def register_explainer(cls: type[BaseExplainer]) -> type[BaseExplainer]:
    """Decorador que registra un explicador bajo su atributo ``name``."""
    return EXPLAINERS.register(cls)


def create_explainer(name: str, **params: Any) -> BaseExplainer:
    """Instancia el explicador registrado como ``name`` con los parámetros dados."""
    return EXPLAINERS.create(name, **params)
