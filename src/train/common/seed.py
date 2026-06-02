"""Single-call multi-source seed helper shared across training scripts.

Promoted from ``rms_variants_train/seed_utils.py`` (plan_2026-06-02_30721a0f,
F3) so every trainer can seed Python / NumPy / TF / Keras RNGs through one
canonical helper instead of re-inlining the 4-call cluster.

LESSONS L73: ``keras.utils.set_random_seed`` seeds Python ``random`` + NumPy +
TF + Keras backend simultaneously. This wrapper exists only to (a) provide a
local import boundary so trainers do not couple directly to keras for seeding,
(b) emit a single logger line so multi-seed runs are auditable from the log.
"""
from __future__ import annotations

import os
import random as _py_random

import keras
import numpy as np

from dl_techniques.utils.logger import logger


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


__all__ = ["set_seeds"]
