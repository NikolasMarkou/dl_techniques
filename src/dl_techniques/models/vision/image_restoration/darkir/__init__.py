"""DarkIR low-light image restoration — public API re-exports.

A functional builder, not a `keras.Model` subclass: `create_darkir_model`
returns `keras.Model(inputs, outputs)`. Kept functional deliberately — every
existing DarkIR checkpoint was saved from that graph.
"""
from .model import create_darkir_model

__all__ = [
    "create_darkir_model",
]
