"""Registro de constructores de variables y composición de los grupos experimentales."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from efd.features.base import BaseFeatureBuilder, FeatureContext
from efd.network import LABEL_COLUMNS, InvoiceNetwork
from efd.registry import ClassRegistry

FEATURE_BUILDERS: ClassRegistry[BaseFeatureBuilder] = ClassRegistry(
    "constructor de variables", BaseFeatureBuilder
)

# Bloques de variables de cada grupo experimental.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "G1": ("transactional",),
    "G2": ("transactional", "structural"),
}


def register_feature_builder(cls: type[BaseFeatureBuilder]) -> type[BaseFeatureBuilder]:
    """Decorador que registra un constructor de variables bajo su atributo ``name``."""
    return FEATURE_BUILDERS.register(cls)


def create_feature_builder(name: str, **params: Any) -> BaseFeatureBuilder:
    """Instancia el constructor registrado como ``name`` con los parámetros dados."""
    return FEATURE_BUILDERS.create(name, **params)


def build_feature_set(
    feature_set: str,
    network: InvoiceNetwork,
    context: FeatureContext,
    params: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Construye y concatena los bloques de variables de un grupo experimental.

    Args:
        feature_set: nombre del grupo (``G1`` o ``G2``).
        network: red de facturación.
        context: período y semilla de la corrida.
        params: parámetros por constructor, ``{nombre: {parámetro: valor}}``.

    Raises:
        KeyError: si el grupo no existe.
        ValueError: si algún bloque devuelve columnas de etiqueta, columnas repetidas u otro orden de
            contribuyentes.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Grupo de variables desconocido: '{feature_set}'. Disponibles: {sorted(FEATURE_SETS)}")
    params = params or {}
    ids = pd.Index(network.taxpayers["taxpayer_id"], name="taxpayer_id")
    blocks = []
    for name in FEATURE_SETS[feature_set]:
        block = create_feature_builder(name, **params.get(name, {})).build(network, context)
        if not block.index.equals(ids):
            raise ValueError(f"El bloque '{name}' no sigue el orden de network.taxpayers.")
        blocks.append(block)
    features = pd.concat(blocks, axis=1)
    if features.columns.has_duplicates:
        raise ValueError("Hay variables repetidas entre bloques.")
    leaked = set(features.columns) & set(LABEL_COLUMNS)
    if leaked:
        raise ValueError(f"Las variables no pueden incluir etiquetas: {sorted(leaked)}.")
    return features
