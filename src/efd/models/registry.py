"""Registro de detectores por nombre para instanciarlos desde la configuración."""

from __future__ import annotations

from typing import Any

from efd.models.base import BaseDetector
from efd.registry import ClassRegistry

DETECTORS: ClassRegistry[BaseDetector] = ClassRegistry("detector", BaseDetector)


def register_detector(cls: type[BaseDetector]) -> type[BaseDetector]:
    """Decorador que registra un detector bajo su atributo ``name``."""
    return DETECTORS.register(cls)


def create_detector(name: str, **params: Any) -> BaseDetector:
    """Instancia el detector registrado como ``name`` con los hiperparámetros dados."""
    return DETECTORS.create(name, **params)


def available_detectors() -> list[str]:
    """Nombres de los detectores registrados."""
    return DETECTORS.names()
