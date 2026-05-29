"""Polar weight reparameterization, after PolarQuant (Han et al., 2025).

This module repurposes PolarQuant's recursive Cartesian->polar transform -- the
paper uses it to *quantize* KV-cache vectors -- as a *trainable weight
reparameterization*. A weight vector ``w`` of dimension ``d`` (a power of two) is
represented by a single radius ``r = ||w||`` and ``d - 1`` angles organized into
``log2(d)`` hierarchical levels. The forward map (angles -> Cartesian) is smooth
and differentiable, so the angles can be optimized directly by gradient descent.

This generalizes Weight Normalization (Salimans & Kingma, 2016), which splits a
weight into ``(g = ||w||, v / ||v||)``: here the *direction* is itself given a
full hierarchical angular coordinate system rather than a free unit vector, and
the magnitude (radius) is an explicit, separately-regularizable parameter that
equals the exact per-unit weight norm.

The module exposes:

- :func:`polar_encode` / :func:`polar_decode` -- the differentiable, backend
  agnostic (``keras.ops``) recursive transform pair (paper Definition 1 /
  Algorithm 1), operating on 2-D tensors ``(N, d)``.
- :class:`PolarWeightNorm` -- a ``Dense``-style layer whose trainable parameters
  are a per-unit ``radius`` and hierarchical ``angles``.
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Recursive Cartesian <-> Polar transform (paper Definition 1 / Algorithm 1)
# ---------------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    """Smallest power of two >= ``n`` (for ``n >= 1``)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _level_sizes(d: int) -> List[int]:
    """Angle counts per level for dimension ``d``: ``[d/2, d/4, ..., 1]``.

    The list has ``log2(d)`` entries (empty when ``d == 1``) summing to ``d - 1``.
    """
    sizes: List[int] = []
    m = d
    while m > 1:
        m //= 2
        sizes.append(m)
    return sizes


def polar_encode(
    x: keras.KerasTensor,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Cartesian -> polar (paper ``Polar`` procedure, Algorithm 1).

    Args:
        x: 2-D tensor ``(N, d)`` with ``d`` a power of two.

    Returns:
        ``(radius, angles)`` where ``radius`` is ``(N, 1)`` and ``angles`` is
        ``(N, d - 1)`` (the concatenation of levels ``1..log2(d)``, sizes
        ``[d/2, d/4, ..., 1]``). For ``d == 1`` ``angles`` is ``(N, 0)``.

    The transform is bijective: :func:`polar_decode` inverts it exactly (up to
    floating-point error).
    """
    d = x.shape[-1]
    if d is None:
        raise ValueError("polar_encode requires a statically known last dim.")
    if not _is_power_of_two(d):
        raise ValueError(f"Last dim must be a power of two, got d={d}.")

    r = x
    angle_levels: List[keras.KerasTensor] = []
    m = d
    while m > 1:
        # Pair up adjacent coordinates: (..., m) -> (..., m/2, 2).
        pair = ops.reshape(r, (-1, m // 2, 2))
        a = pair[:, :, 0]  # r_{2j-1}
        b = pair[:, :, 1]  # r_{2j}
        angle_levels.append(ops.arctan2(b, a))  # psi in [0, 2pi) (lvl 1) / [0, pi/2]
        r = ops.sqrt(ops.square(a) + ops.square(b))  # new radius vector
        m //= 2

    radius = r  # (N, 1)
    if angle_levels:
        angles = ops.concatenate(angle_levels, axis=-1)  # (N, d-1)
    else:
        angles = radius[:, :0]  # (N, 0) for d == 1
    return radius, angles


def polar_decode(
    radius: keras.KerasTensor,
    angles: keras.KerasTensor,
) -> keras.KerasTensor:
    """Polar -> Cartesian (paper ``DeQuant`` procedure, Algorithm 1).

    Args:
        radius: ``(N, 1)`` top-level radius.
        angles: ``(N, d - 1)`` angle levels as produced by :func:`polar_encode`.

    Returns:
        ``(N, d)`` Cartesian reconstruction.
    """
    a_dim = angles.shape[-1]
    if a_dim is None:
        raise ValueError("polar_decode requires a statically known angle dim.")
    d = a_dim + 1

    # Split the flat angle vector back into levels [d/2, d/4, ..., 1].
    splits: List[keras.KerasTensor] = []
    start = 0
    for s in _level_sizes(d):
        splits.append(angles[:, start:start + s])
        start += s

    r = radius  # (N, 1) == r^(L)
    # Walk levels top-down (smallest level first) interleaving cos/sin children.
    for psi in reversed(splits):
        a = r * ops.cos(psi)
        b = r * ops.sin(psi)
        stacked = ops.stack([a, b], axis=-1)  # (N, m, 2)
        m = psi.shape[-1]
        r = ops.reshape(stacked, (-1, 2 * m))  # (N, 2m)
    return r
