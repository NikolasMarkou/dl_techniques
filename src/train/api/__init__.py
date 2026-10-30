"""Training API: strict configs, a key=value command line, one function per task.

The package surface is lazy (PEP 562): ``import train.api`` loads nothing,
and a name is imported from its module on first access. That keeps
``python -m train help`` free of TensorFlow, Keras and ``yaml``.

A name enters ``__all__`` when its module exists (D-007).
"""

import importlib
from typing import Dict, List

_LAZY: Dict[str, str] = {
    "DLTrainError": ".errors",
    "ConfigError": ".errors",
    "ConfigValidationError": ".errors",
    "UnknownModelError": ".errors",
    "UnresolvedAutoError": ".errors",
    "main": ".cli",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    """Import ``name`` from its module on first access and cache it."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
