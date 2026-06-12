# DECISION plan_2026-06-11_f662207d/D-003
# Placed as a pure-function module (NOT a keras.layers.Layer) because make_grid /
# interpolate_grid are STATELESS pure functions with no weights, no build(), and no
# serialization state -- wrapping them in a Layer would be a single-use empty shell.
# Both functions must be callable BARE so the step-9 Jacobian-TV helper can invoke
# interpolate_grid inside a nested tf.GradientTape (a built Layer would add nothing).
# TF-native tf.gather + manual lerp is chosen over keras.ops here for the differentiable
# bilinear (order=1) path: backend-agnosticism is explicitly WAIVED for this op (scope
# decision Q4) and a hand-written 4-corner lerp gives an auditable gradient path to
# coords. order=0 (nearest) is THERA's DEFAULT sampling mode; the coordinate Jacobian
# in THERA flows through the DIRECT `coords` term of rel_coords (= coords -
# interpolate_grid(...)), NOT through the (a.e. zero-gradient) nearest sampler -- see
# decisions.md D-003. We still implement a fully differentiable order=1 path for
# completeness and future use.
"""TF-native coordinate-grid helpers ported from THERA (`model/utils.py`).

This module provides two stateless pure functions used by the THERA neural heat
field for arbitrary-scale super-resolution:

- :func:`make_grid` builds a fixed pixel-center normalized coordinate grid.
- :func:`interpolate_grid` samples a feature grid at continuous query coordinates
  (nearest or bilinear), with edge clamping (``mode='nearest'``).

Coordinate convention (verified against THERA reference ``model/utils.py``)
---------------------------------------------------------------------------
``make_grid`` uses PIXEL-CENTER normalized coordinates. For an axis of length
``n`` the sample positions are ``linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n)`` -- the
centers of ``n`` equal cells tiling ``[-0.5, 0.5]``. The output has shape
``(h, w, 2)`` with channel order ``[h_coord, w_coord]`` (index 0 = vertical/h,
index 1 = horizontal/w), matching ``np.meshgrid(..., indexing='ij')``.

NOTE: ``dl_techniques.layers.spatial_layer.SpatialLayer`` is NOT equivalent -- it
samples linspace ENDPOINTS ``[-0.5, 0.5]`` and then z-score normalizes the
channels. THERA needs the un-normalized pixel-center grid, so make_grid is
implemented faithfully here rather than reusing SpatialLayer.

Sampling (``interpolate_grid``)
-------------------------------
Given a feature grid of shape ``(B, H', W', C)`` and query coordinates of shape
``(B, Hq, Wq, 2)`` in ``[-0.5, 0.5]`` (channel order ``[h, w]``), the
coordinate -> continuous-pixel-index mapping per axis is::

    pix = coord * size + (size - 1) / 2

where ``size`` is ``H'`` for axis 0 and ``W'`` for axis 1. Border handling is
``mode='nearest'`` (CLAMP to ``[0, size-1]``). ``order=0`` rounds to the nearest
integer index (nearest-neighbour); ``order=1`` performs a 4-corner bilinear lerp.
Output shape is ``(B, Hq, Wq, C)``.

Differentiability
-----------------
THERA's forward pass uses ``order=0`` (nearest) to sample the phi-params and the
source coordinates. The aliasing TV Jacobian is the Jacobian of the heat FIELD
w.r.t. its local spatial input ``rel_coords``, where
``rel_coords = coords - interpolate_grid(coords, source_coords)``. The nearest
sampling term has zero gradient almost everywhere, so the coordinate gradient
flows through the DIRECT ``coords`` term, not through the sampler. Therefore
``interpolate_grid`` need not be differentiable for THERA's exact-Jacobian
deliverable. We nonetheless implement a fully differentiable ``order=1`` path: its
lerp weights are a smooth function of ``coords``, so gradients propagate to the
query coordinates (required for any future differentiable-sampling use).
"""

from typing import Tuple, Union

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------


def make_grid(patch_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """Build a pixel-center normalized coordinate grid.

    For an axis of length ``n``, sample positions are the centers of ``n`` equal
    cells tiling ``[-0.5, 0.5]``: ``linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n)``.

    Args:
        patch_size: Grid size. An ``int`` ``n`` yields an ``(n, n)`` grid; a
            ``(h, w)`` tuple yields an ``(h, w)`` grid.

    Returns:
        A ``float32`` numpy array of shape ``(h, w, 2)``. The last axis holds
        ``[h_coord, w_coord]`` (index 0 = vertical/h, index 1 = horizontal/w),
        with ``indexing='ij'`` semantics.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    h, w = int(patch_size[0]), int(patch_size[1])

    offset_h = 1.0 / (2.0 * h)
    offset_w = 1.0 / (2.0 * w)
    space_h = np.linspace(-0.5 + offset_h, 0.5 - offset_h, h)
    space_w = np.linspace(-0.5 + offset_w, 0.5 - offset_w, w)
    grid = np.stack(np.meshgrid(space_h, space_w, indexing="ij"), axis=-1)
    return grid.astype(np.float32)


# ---------------------------------------------------------------------


def _gather_hw(grid: tf.Tensor, idx_h: tf.Tensor, idx_w: tf.Tensor) -> tf.Tensor:
    """Gather ``grid[b, idx_h, idx_w, :]`` for batched integer index maps.

    Args:
        grid: ``(B, H', W', C)`` feature grid.
        idx_h: ``(B, Hq, Wq)`` int32 clamped row indices.
        idx_w: ``(B, Hq, Wq)`` int32 clamped col indices.

    Returns:
        ``(B, Hq, Wq, C)`` gathered values.
    """
    b = tf.shape(idx_h)[0]
    hq = tf.shape(idx_h)[1]
    wq = tf.shape(idx_h)[2]

    batch_idx = tf.reshape(tf.range(b), (b, 1, 1))
    batch_idx = tf.broadcast_to(batch_idx, (b, hq, wq))

    gather_idx = tf.stack([batch_idx, idx_h, idx_w], axis=-1)  # (B, Hq, Wq, 3)
    return tf.gather_nd(grid, gather_idx)


def interpolate_grid(
    coords: Union[tf.Tensor, np.ndarray],
    grid: Union[tf.Tensor, np.ndarray],
    order: int = 0,
) -> tf.Tensor:
    """Sample a feature grid at continuous query coordinates (edge-clamped).

    Args:
        coords: ``(B, Hq, Wq, 2)`` query coordinates in ``[-0.5, 0.5]``, channel
            order ``[h, w]``.
        grid: ``(B, H', W', C)`` feature grid to sample.
        order: Interpolation order. ``0`` = nearest-neighbour (default, THERA's
            forward mode); ``1`` = bilinear (4-corner lerp, differentiable w.r.t.
            ``coords``).

    Returns:
        ``(B, Hq, Wq, C)`` ``float32`` sampled values.

    Raises:
        ValueError: If ``order`` is not ``0`` or ``1``.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}")

    coords = tf.convert_to_tensor(coords, dtype=tf.float32)
    grid = tf.convert_to_tensor(grid, dtype=tf.float32)

    grid_shape = tf.shape(grid)
    size_h = grid_shape[-3]
    size_w = grid_shape[-2]
    size_h_f = tf.cast(size_h, tf.float32)
    size_w_f = tf.cast(size_w, tf.float32)

    # coord -> continuous pixel index per axis: pix = coord * size + (size - 1) / 2
    coord_h = coords[..., 0]  # (B, Hq, Wq)
    coord_w = coords[..., 1]
    pix_h = coord_h * size_h_f + (size_h_f - 1.0) / 2.0
    pix_w = coord_w * size_w_f + (size_w_f - 1.0) / 2.0

    max_h = size_h - 1
    max_w = size_w - 1

    if order == 0:
        # Nearest: round, then clamp to [0, size-1].
        idx_h = tf.cast(tf.round(pix_h), tf.int32)
        idx_w = tf.cast(tf.round(pix_w), tf.int32)
        idx_h = tf.clip_by_value(idx_h, 0, max_h)
        idx_w = tf.clip_by_value(idx_w, 0, max_w)
        return _gather_hw(grid, idx_h, idx_w)

    # order == 1: bilinear. Floor/ceil corners, clamp each, lerp by frac part.
    # Floor in float, derive fractional weights, THEN clamp integer corners. The
    # weights are computed from the UNCLAMPED frac so they stay a smooth function
    # of coords (gradient to coords); clamping only affects which pixel is read,
    # reproducing mode='nearest' edge replication.
    h0_f = tf.floor(pix_h)
    w0_f = tf.floor(pix_w)
    frac_h = pix_h - h0_f  # (B, Hq, Wq), differentiable in coords
    frac_w = pix_w - w0_f

    h0 = tf.cast(h0_f, tf.int32)
    w0 = tf.cast(w0_f, tf.int32)
    h1 = h0 + 1
    w1 = w0 + 1

    h0c = tf.clip_by_value(h0, 0, max_h)
    h1c = tf.clip_by_value(h1, 0, max_h)
    w0c = tf.clip_by_value(w0, 0, max_w)
    w1c = tf.clip_by_value(w1, 0, max_w)

    v00 = _gather_hw(grid, h0c, w0c)  # (B, Hq, Wq, C)
    v01 = _gather_hw(grid, h0c, w1c)
    v10 = _gather_hw(grid, h1c, w0c)
    v11 = _gather_hw(grid, h1c, w1c)

    # Expand weights to (B, Hq, Wq, 1) to broadcast over channels.
    fh = frac_h[..., tf.newaxis]
    fw = frac_w[..., tf.newaxis]

    top = v00 * (1.0 - fw) + v01 * fw
    bot = v10 * (1.0 - fw) + v11 * fw
    return top * (1.0 - fh) + bot * fh

# ---------------------------------------------------------------------
