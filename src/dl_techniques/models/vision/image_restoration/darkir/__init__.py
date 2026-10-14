"""DarkIR low-light image restoration — public API re-exports.

Provides the ``keras.Model`` subclass :class:`DarkIR` and its factory
:func:`create_darkir_model`, which returns a :class:`DarkIR` (it builds no
functional model). ``include_top=True`` is the paper's network, tail included;
``include_top=False`` returns the last decoder stage's features.
"""
from .model import DarkIR, create_darkir_model

__all__ = [
    "DarkIR",
    "create_darkir_model",
]
