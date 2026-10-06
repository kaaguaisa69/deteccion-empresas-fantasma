"""Patrones de empresas fantasmas.

Importar este paquete registra los patrones concretos para que puedan seleccionarse por nombre.
"""

from efd.generation.patterns.base import FraudPattern, GenerationState
from efd.generation.patterns.registry import PATTERNS, create_pattern, register_pattern
from efd.generation.patterns import chain, isolated_emitter, ring, shared_representative  # noqa: F401  (registro)

__all__ = ["PATTERNS", "FraudPattern", "GenerationState", "create_pattern", "register_pattern"]
