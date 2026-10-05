from __future__ import annotations

from typing import ClassVar

import pytest

from efd.registry import ClassRegistry, Registry


class _Base:
    name: ClassVar[str] = ""


def test_registry_add_and_get() -> None:
    registry: Registry[int] = Registry("componente")
    registry.add("uno", 1)
    assert registry.get("uno") == 1
    assert "uno" in registry
    assert registry.names() == ["uno"]


def test_registry_rejects_duplicates_and_empty_names() -> None:
    registry: Registry[int] = Registry("componente")
    registry.add("uno", 1)
    with pytest.raises(ValueError):
        registry.add("uno", 2)
    with pytest.raises(ValueError):
        registry.add("", 3)


def test_registry_unknown_name_lists_available() -> None:
    registry: Registry[int] = Registry("componente")
    registry.add("uno", 1)
    with pytest.raises(KeyError, match="uno"):
        registry.get("dos")


def test_class_registry_registers_and_creates() -> None:
    registry = ClassRegistry("componente", _Base)

    @registry.register
    class Concrete(_Base):
        name = "concreto"

        def __init__(self, value: int) -> None:
            self.value = value

    instance = registry.create("concreto", value=7)
    assert isinstance(instance, Concrete)
    assert instance.value == 7


def test_class_registry_rejects_foreign_classes_and_missing_name() -> None:
    registry = ClassRegistry("componente", _Base)

    class Foreign:
        name = "ajeno"

    class Unnamed(_Base):
        pass

    with pytest.raises(TypeError):
        registry.register(Foreign)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        registry.register(Unnamed)
