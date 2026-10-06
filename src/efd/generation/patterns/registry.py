"""Registro de patrones de fraude por nombre; agregar uno nuevo no modifica el resto del generador."""

from __future__ import annotations

from typing import Any

from efd.generation.patterns.base import FraudPattern
from efd.registry import ClassRegistry

PATTERNS: ClassRegistry[FraudPattern] = ClassRegistry("patrón de fraude", FraudPattern)


def register_pattern(cls: type[FraudPattern]) -> type[FraudPattern]:
    """Decorador que registra un patrón bajo su atributo ``name``."""
    return PATTERNS.register(cls)


def create_pattern(name: str, **params: Any) -> FraudPattern:
    """Instancia el patrón registrado como ``name`` con los parámetros de la configuración."""
    return PATTERNS.create(name, **params)
