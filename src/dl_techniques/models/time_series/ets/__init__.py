"""
Pure additive exponential smoothing (ETS) with trainable smoothing parameters.

Exports the model class and its factory. See ``model.py`` for the state-space
equations and ``README.md`` for background: this is the only recursive
forecaster in the tree.
"""

from .model import ETS_VARIANTS, ETSModel, create_ets

__all__ = ["ETSModel", "create_ets", "ETS_VARIANTS"]
