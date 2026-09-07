"""Shared ensemble scaling-vector vocabulary for ``layers/tabular/``.

A batched-ensemble layer gives each of its ``k`` members a small per-member
perturbation instead of a full private copy of the weights. That perturbation is
a scaling vector, and how it is initialized decides whether the members start as
different functions at all: ``'random-signs'`` (the TabM paper's choice) and
``'normal'`` both break symmetry between members, while ``'ones'`` leaves every
member's effective weight matrix identical at initialization.

Two symbols express that one idea, and both are needed in more than one sibling
module, so they live here once rather than being restated per file. Measured in
this package: :data:`EnsembleInitDistribution` is the annotation for the
``init_distribution`` constructor argument of FOUR layers -- ``ScaleEnsemble``,
``LinearEfficientEnsemble``, ``TabMMLPBlock`` and ``TabMBackbone`` -- and
:func:`_ensemble_scaling_initializer` is called by TWO of them,
``ScaleEnsemble`` and ``LinearEfficientEnsemble``, each once in ``__init__``.

Nothing here is a registered Keras object: the type alias is a
:class:`typing.Literal`, and the function only ever returns stock
``keras.initializers`` instances plus
:class:`dl_techniques.initializers.RandomSigns`, which carries its own
registration. So this module declares no ``@register_dl_technique``, and it is
underscore-prefixed because it is private to this package -- import the leaf
layer modules, not this one.

Follows the ``layers/norms/_masking.py`` precedent (underscore-prefixed,
non-registered, ``__all__``-declared, imported by siblings via its full dotted
path).

See ``decisions.md`` D-007 of plan ``plan-2026-09-07T095804-b821967f``.
"""

import keras
from typing import Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import RandomSigns

# ---------------------------------------------------------------------

__all__ = ["EnsembleInitDistribution", "_ensemble_scaling_initializer"]

# ---------------------------------------------------------------------

# DECISION plan-2026-09-07T095804-b821967f/D-007
# These two symbols live in exactly ONE place. `EnsembleInitDistribution`
# annotates `init_distribution` in 4 sibling layer modules and
# `_ensemble_scaling_initializer` is called from 2 of them.
# Do NOT copy either back into a layer module "so the file reads standalone":
# a second `Literal[...]` drifts silently when a fourth distribution name is
# added, and a second resolver makes `'random-signs'` mean two different
# initializers depending on which layer you asked. Import from here instead.
# Do NOT promote this to a public `common.py` either -- that name is reserved
# for semi-public infrastructure, which neither symbol is. See decisions.md D-007.

EnsembleInitDistribution = Literal['ones', 'normal', 'random-signs']

# ---------------------------------------------------------------------

def _ensemble_scaling_initializer(
        init_distribution: EnsembleInitDistribution
) -> keras.initializers.Initializer:
    """
    Resolve an ensemble scaling-vector initializer by name.

    :param init_distribution: One of ``'ones'``, ``'normal'``, ``'random-signs'``.
    :type init_distribution: str

    :return: The matching initializer instance.
    :rtype: keras.initializers.Initializer

    :raises ValueError: If ``init_distribution`` is not one of the three names.
    """
    if init_distribution == 'ones':
        return keras.initializers.Ones()
    if init_distribution == 'normal':
        return keras.initializers.RandomNormal(mean=1.0, stddev=0.1)
    if init_distribution == 'random-signs':
        return RandomSigns()
    raise ValueError(
        f"init_distribution must be one of 'ones', 'normal', 'random-signs'; "
        f"got {init_distribution!r}"
    )

# ---------------------------------------------------------------------
