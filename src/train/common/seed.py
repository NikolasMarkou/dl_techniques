"""Single-call multi-source seed helper shared across training scripts.

Promoted from ``rms_variants_train/seed_utils.py`` (plan_2026-06-02_30721a0f,
F3) so every trainer can seed Python / NumPy / TF / Keras RNGs through one
canonical helper instead of re-inlining the 4-call cluster.

LESSONS L73: ``keras.utils.set_random_seed`` seeds Python ``random`` + NumPy +
TF + Keras backend simultaneously. This wrapper exists only to (a) provide a
local import boundary so trainers do not couple directly to keras for seeding,
(b) emit a single logger line so multi-seed runs are auditable from the log.

It also holds :func:`seed_generators`, the one walk over a model's layers that returns every
``keras.random.SeedGenerator`` (used by the denoising homogeneity probe, the TopoLM live
diagnostics and the PowerMLP baseline pass, which save and restore the generators' state).
"""
from __future__ import annotations

import os
import random as _py_random
from typing import Any, List

import keras
import numpy as np

from dl_techniques.utils.logger import logger


#: The house ``--seed`` default (D-247 of plan-2026-10-14T062616-81938e81). Every
#: trainer's ``--seed`` defaults to it: by name where the module already imports
#: ``train.common``, as a literal of equal value where it does not.
#: ``tests/test_train/test_every_trainer_seeds_through_the_shared_helper.py`` reads it by
#: AST and holds every trainer's default equal to it.
DEFAULT_SEED: int = 42


def set_seeds(seed: int) -> None:
    """Seed every RNG source used by training.

    Order matters: PYTHONHASHSEED first (best-effort, no-op once Python is up),
    then the rest. ``keras.utils.set_random_seed`` covers Python ``random``,
    NumPy, TF, and the Keras backend -- but explicit calls remain useful when
    a later TF op is reseeded by a third-party library.

    Args:
        seed: Integer seed applied to every RNG source.

    Returns:
        None.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    _py_random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    logger.info(f"[seed] all RNGs seeded with seed={seed}")


def seed_generators(model: Any) -> List[Any]:
    """Every ``keras.random.SeedGenerator`` reachable from ``model``, once each.

    Interface contract: ``model`` is a Keras model or layer; returns the generators in layer
    order, de-duplicated by identity (empty for a model with no stochastic layer). Never raises
    for a layer that has no generators.

    Walks ``model._flatten_layers()`` and reads each layer's ``_seed_generators`` (Keras 3.8
    private attributes, the only registry there is). If a Keras release renames them this
    returns an empty list, so a caller that uses it as a tripwire (a draw must leave every
    generator where it found it) would pass vacuously: such a caller counts what it found
    against what the model is known to hold.

    Args:
        model: A Keras model or layer.

    Returns:
        The generators, in layer order, without duplicates.
    """
    found: List[Any] = []
    seen = set()
    for layer in model._flatten_layers():
        for generator in getattr(layer, "_seed_generators", ()):
            if id(generator) not in seen:
                seen.add(id(generator))
                found.append(generator)
    return found


__all__ = ["DEFAULT_SEED", "seed_generators", "set_seeds"]
