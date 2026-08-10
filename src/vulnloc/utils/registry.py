"""Generic name -> class registry (``docs/Architecture.md`` §1.6, §9).

The project bans ``if/elif`` dispatch on config strings. Every extension point is a
registry: a new encoder, GNN, graph type, dataset, explainer, or metric is *one class
plus one decorator plus one YAML line*, with no edits to existing files.

This module lives in ``utils`` (below every layer) so that L0..L6 can each own their
registries without importing sideways or upward.

Example:
    >>> from vulnloc.utils.registry import Registry
    >>> encoders: Registry[object] = Registry("semantic_encoder")
    >>> @encoders.register("codebert")
    ... class CodeBERT:
    ...     pass
    >>> encoders.get("codebert") is CodeBERT
    True
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Generic, TypeVar

from vulnloc.utils.exceptions import RegistryError

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """A namespaced mapping from config keys to implementations.

    Registration is strict: re-registering a key raises rather than silently shadowing
    an implementation, because a silently-overridden encoder would invalidate every
    experiment run under that config name (``Rules.md`` §5.1).

    Args:
        name: Namespace label used in error messages, e.g. ``"fusion"``.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type[T]] = {}

    @property
    def name(self) -> str:
        """The registry's namespace label."""
        return self._name

    def register(self, key: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
        """Return a decorator that registers a class under ``key``.

        Args:
            key: Config-facing name, e.g. ``"cross_attention"``.
            override: Allow replacing an existing entry. Intended for tests only.

        Returns:
            A class decorator that returns the class unchanged.

        Raises:
            RegistryError: If ``key`` is empty, or already registered and
                ``override`` is False.
        """
        if not key:
            raise RegistryError(f"Cannot register an empty key in registry {self._name!r}.")

        def decorator(cls: type[T]) -> type[T]:
            if key in self._entries and not override:
                existing = self._entries[key]
                raise RegistryError(
                    f"Key {key!r} is already registered in registry {self._name!r}.",
                    registry=self._name,
                    key=key,
                    existing=f"{existing.__module__}.{existing.__qualname__}",
                    incoming=f"{cls.__module__}.{cls.__qualname__}",
                )
            self._entries[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Look up a registered class.

        Args:
            key: The name it was registered under.

        Returns:
            The registered class.

        Raises:
            RegistryError: If ``key`` is not registered. The message lists the
                available keys, since this is almost always a config typo.
        """
        try:
            return self._entries[key]
        except KeyError:
            raise RegistryError(
                f"Unknown {self._name} {key!r}. Available: {sorted(self._entries) or '<none>'}.",
                registry=self._name,
                key=key,
                available=sorted(self._entries),
            ) from None

    def keys(self) -> list[str]:
        """Return the registered keys, sorted."""
        return sorted(self._entries)

    def items(self) -> Mapping[str, type[T]]:
        """Return a read-only view of the registry contents."""
        return dict(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={self.keys()})"
