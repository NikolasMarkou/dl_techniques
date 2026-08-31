"""
Pure additive exponential smoothing (ETS) with trainable smoothing parameters.

Public API mirrors the rest of ``models/time_series/``: the model class and its
factory. See ``model.py`` for the state-space equations and ``README.md`` for
why this package exists (it is the only recursive forecaster in the tree, and
therefore the only one on which the multistep-loss shrinkage result can be
reproduced rather than cited).
"""

from .model import ETS_VARIANTS, ETSModel, create_ets

__all__ = ["ETSModel", "create_ets", "ETS_VARIANTS"]
