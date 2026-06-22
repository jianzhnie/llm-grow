"""In-memory expander registry — decorator-based, symmetric with safetensor/auto.py.

Usage::

    from llm_grow.expanders.registry import register_expander, get_expander

    @register_expander("zero_block_insert")
    class ZeroBlockInsertExpander(AbstractExpander[ZeroBlockInsertConfig]):
        ...

    expander = get_expander("zero_block_insert")()
    expanded = expander(model, config)
"""

from __future__ import annotations

from typing import TypeVar

from llm_grow.expanders.base import AbstractExpander

_E = TypeVar("_E", bound=AbstractExpander)

_EXPANDER_REGISTRY: dict[str, type[AbstractExpander]] = {}


def register_expander(name: str):
    """Decorator that registers an in-memory expander class under *name*."""

    def decorator(cls: type[_E]) -> type[_E]:
        _EXPANDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_expander(name: str) -> type[AbstractExpander]:
    """Look up a registered in-memory expander by name.

    Raises:
        KeyError: if no expander is registered under *name*.
    """
    if name not in _EXPANDER_REGISTRY:
        available = sorted(_EXPANDER_REGISTRY)
        raise KeyError(
            f"Unknown expander {name!r}. "
            f"Available: {available or '(none — did you import the expander modules?)'}"
        )
    return _EXPANDER_REGISTRY[name]


def list_expanders() -> list[str]:
    """Return sorted list of registered expander names."""
    return sorted(_EXPANDER_REGISTRY)
