"""Registro genérico de componentes por nombre.

Modelos, métricas, políticas de umbral, explicadores y constructores de variables se registran con un
nombre único. La configuración YAML los referencia por ese nombre, de modo que agregar un componente
nuevo no exige modificar el orquestador ni el evaluador.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")
B = TypeVar("B")


class Registry(Generic[T]):
    """Diccionario de componentes indexado por nombre, sin sobrescrituras silenciosas."""

    def __init__(self, kind: str) -> None:
        """
        Args:
            kind: descripción del tipo de componente; se usa en los mensajes de error.
        """
        self._kind = kind
        self._items: dict[str, T] = {}

    def add(self, name: str, item: T) -> None:
        """Registra ``item`` bajo ``name``.

        Raises:
            ValueError: si el nombre está vacío o ya existe; un nombre duplicado haría ambigua la
                configuración.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"El nombre de un {self._kind} debe ser una cadena no vacía.")
        if name in self._items:
            raise ValueError(f"Ya existe un {self._kind} registrado como '{name}'.")
        self._items[name] = item

    def get(self, name: str) -> T:
        """Devuelve el componente registrado como ``name``.

        Raises:
            KeyError: si el nombre no existe; el mensaje lista los nombres disponibles.
        """
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"{self._kind.capitalize()} desconocido: '{name}'. Disponibles: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        """Nombres registrados en orden alfabético."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


class ClassRegistry(Registry[type[B]]):
    """Registro de clases que comparten una clase base y declaran su nombre en el atributo ``name``."""

    def __init__(self, kind: str, base: type[B]) -> None:
        """
        Args:
            kind: descripción del tipo de componente.
            base: clase base que deben extender todas las clases registradas.
        """
        super().__init__(kind)
        self._base = base

    def register(self, cls: type[B]) -> type[B]:
        """Registra ``cls`` bajo ``cls.name``; se usa como decorador de clase.

        Raises:
            TypeError: si ``cls`` no extiende la clase base del registro.
            ValueError: si ``cls.name`` está vacío o duplicado.
        """
        if not isinstance(cls, type) or not issubclass(cls, self._base):
            raise TypeError(f"{cls!r} no es subclase de {self._base.__name__}.")
        self.add(getattr(cls, "name", ""), cls)
        return cls

    def create(self, name: str, **params: Any) -> B:
        """Instancia la clase registrada como ``name`` con los parámetros dados."""
        return self.get(name)(**params)
