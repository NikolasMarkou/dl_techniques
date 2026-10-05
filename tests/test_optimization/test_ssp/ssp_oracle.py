"""Shared instruments for the SSP guards.

No ``test_`` prefix, so pytest does not collect it (see
``src/dl_techniques/AGENTS.md`` § Testing, "shared instruments").

Each function here is an ORACLE or a MUTATOR: something the guards check against,
or something that injects the specific defect a guard exists to catch. Both kinds
are needed, because a guard that has only ever passed has not been shown to work.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

import keras

# ---------------------------------------------------------------------
# oracles
# ---------------------------------------------------------------------


def binary_entropy_oracle(p) -> np.ndarray:
    """``H(p)`` in exact Python floats, with ``0 log 0`` handled by a branch.

    Independent of the ``keras.ops`` implementation on purpose: it must not share a
    line of code with the thing it judges.
    """
    values = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(values)
    for index, value in np.ndenumerate(values):
        if value <= 0.0 or value >= 1.0:
            out[index] = 0.0
        else:
            out[index] = -value * math.log(value) - (1.0 - value) * math.log1p(-value)
    return out


def bernoulli_kl_oracle(p, p0: float = 0.5) -> np.ndarray:
    """``KL(Bern(p) || Bern(p0))`` from the defining sum, not from an entropy identity.

    ``sum_x p(x) log(p(x)/q(x))`` over the two-point support, with the ``0 log 0``
    term dropped exactly as the definition drops it. Deliberately does NOT use
    ``H`` -- deriving the quantity under test from the identity it is supposed to
    satisfy would make every identity check vacuous.
    """
    values = np.asarray(p, dtype=np.float64)
    out = np.empty_like(values)
    for index, value in np.ndenumerate(values):
        total = 0.0
        for x, px in ((1.0, value), (0.0, 1.0 - value)):
            qx = p0 if x == 1.0 else 1.0 - p0
            if px > 0.0:
                total += px * math.log(px / qx)
        out[index] = total
    return out


def group_relative_advantage_oracle(rewards, eps: float = 1e-6) -> np.ndarray:
    """``(r - mu) / (sigma + eps)`` per group, in plain numpy.

    Independent of the ``keras.ops`` implementation, and it uses the same
    ``eps``-in-the-denominator convention -- so a guard comparing the two is testing
    the port, not restating the formula.
    """
    values = np.asarray(rewards, dtype=np.float64)
    mean = values.mean(axis=-1, keepdims=True)
    centred = values - mean
    stddev = np.sqrt((centred**2).mean(axis=-1, keepdims=True))
    return centred / (stddev + eps)


def clipped_surrogate_oracle(
        log_probs: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        clip_eps: float = 0.2,
) -> np.ndarray:
    """The PPO clipped surrogate, token by token, with the advantage inside the min.

    Written out longhand so it cannot inherit the implementation's structure. The
    advantage is broadcast onto the token grid first, the same promotion the
    implementation performs, because indexing a 1-D advantage with a 2-D index is a
    shape error rather than a numeric one.
    """
    current = np.asarray(log_probs, dtype=np.float64)
    reference = np.asarray(old_log_probs, dtype=np.float64)
    weights = np.asarray(advantages, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    weights = np.broadcast_to(weights, current.shape)
    out = np.empty_like(current)
    for index in np.ndindex(current.shape):
        ratio = math.exp(current[index] - reference[index])
        low = 1.0 - clip_eps
        high = 1.0 + clip_eps
        clipped = min(max(ratio, low), high)
        out[index] = -min(ratio * weights[index], clipped * weights[index])
    return out


# ---------------------------------------------------------------------
# mutators -- inject the exact defect a guard watches, so the guard is
# proven able to go red
# ---------------------------------------------------------------------


def degenerate_boundary_mutant(n_minus_c: int, k: int) -> bool:
    """The WRONG degenerate test for ``C(n - c, k) == 0``, i.e. ``<=`` not ``<``.

    Returned so a guard can assert that the boundary it pins is the one that matters:
    at ``n - c == k`` the falling factorial lands on ``1``, not on ``0``.
    """
    return n_minus_c - k <= 0


def log1p_confusion_mutant(p) -> np.ndarray:
    """``log1p(1 - p)`` where ``log(1 - p)`` was required: ``log(2 - p)``.

    This is the one-character slip that made the max-entropy weight peak at the
    wrong ``p`` while raising nothing. Exposed so the guard can assert the two differ
    materially rather than being told they do.
    """
    values = np.asarray(p, dtype=np.float64)
    out = np.empty_like(values)
    for index, value in np.ndenumerate(values):
        complement = 1.0 - value
        out[index] = math.log1p(complement) if complement > 0.0 else 0.0
    return out
