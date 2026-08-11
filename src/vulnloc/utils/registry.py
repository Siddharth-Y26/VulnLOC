"""Generic name -> implementation registry (``docs/Architecture.md`` §1.6, §9).

The project bans ``if/elif`` dispatch on config strings. Every extension point is a
registry: a new encoder, GNN, graph type, dataset, explainer, metric or split strategy
is *one implementation plus one decorator plus one YAML line*, with no edits to existing
files. Implementations may be classes or plain functions — split strategies are
functions, encoders are classes, and the registry does not care.

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


def _describe(impl: object) -> str:
    """Return a dotted path for an implementation, for error messages.

    Args:
        impl: A class or function.

    Returns:
        ``module.qualname``, falling back to ``repr`` for objects lacking them.
    """
    module = getattr(impl, "__module__", "?")
    qualname = getattr(impl, "__qualname__", repr(impl))
    return f"{module}.{qualname}"


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
        self._entries: dict[str, T] = {}

    @property
    def name(self) -> str:
        """The registry's namespace label."""
        return self._name

    def register(self, key: str, *, override: bool = False) -> Callable[[T], T]:
        """Return a decorator that registers an implementation under ``key``.

        Args:
            key: Config-facing name, e.g. ``"cross_attention"``.
            override: Allow replacing an existing entry. Intended for tests only.

        Returns:
            A decorator that returns its argument unchanged, so the decorated class or
            function remains directly usable.

        Raises:
            RegistryError: If ``key`` is empty, or already registered and
                ``override`` is False.
        """
        if not key:
            raise RegistryError(f"Cannot register an empty key in registry {self._name!r}.")

        def decorator(impl: T) -> T:
            if key in self._entries and not override:
                raise RegistryError(
                    f"Key {key!r} is already registered in registry {self._name!r}.",
                    registry=self._name,
                    key=key,
                    existing=_describe(self._entries[key]),
                    incoming=_describe(impl),
                )
            self._entries[key] = impl
            return impl

        return decorator

    def get(self, key: str) -> T:
        """Look up a registered implementation.

        Args:
            key: The name it was registered under.

        Returns:
            The registered class or function.

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

    def items(self) -> Mapping[str, T]:
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
