"""DarkIR low-light image restoration — public API re-exports.

Provides both a ``keras.Model`` subclass (:class:`DarkIR`) with ``include_top``
support and a functional factory (:func:`create_darkir_model`) for backward
compatibility.
"""
from .model import DarkIR, create_darkir_model

__all__ = [
    "DarkIR",
    "create_darkir_model",
]
