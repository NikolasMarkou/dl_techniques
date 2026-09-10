"""The one sampling convention this port has, expressed as two pure functions.

DocScanner is a RAFT-lineage network, and every spatial operation in its
rectification stage is stated in ABSOLUTE PIXEL coordinates: the coordinate
field it refines, the feature resample that feeds the motion encoder, and the
backward map it finally emits are all pixel-valued. Upstream, those coordinates
reach ``torch.nn.functional.grid_sample`` through ``bilinear_sampler``
(``model.py:9-22``), which normalizes them with ``2 * x / (W - 1) - 1`` and
passes ``align_corners=True``.

This module is that convention, and nothing else. It holds no layer, no weight
and no registered class -- only :func:`coords_grid`,
:func:`sample_at_pixel_coords` and :func:`convex_upsample`, the three pure
functions that every other part of the port routes its resampling through. It
is deliberately separate from ``components.py`` (which holds the width table
and the Keras layers) so that the sampling convention is findable as one thing
rather than being distributed through a layer zoo: it is invariant #1 of this
port, and a violation of it is sub-pixel, silent, and invisible to every shape
test.

:func:`convex_upsample` belongs here for the same reason: it is a resampling
operator (a learned 3x3-neighbourhood blend followed by a sub-pixel
scatter), it holds no weights -- the mask it consumes is produced by the
update block's mask head -- and its own failure mode is a permuted or
transposed backward map with no shape symptom at all.

Why there is no new sampler here
--------------------------------
``layers/spatial_layer.py::interpolate_grid`` already exists, is differentiable
w.r.t. its coordinates, and is edge-clamped. It is a HALF-PIXEL sampler:
internally it computes ``pix = coord * S + (S - 1) / 2`` for a coordinate in
``[-0.5, 0.5]``, which is ``align_corners=False`` semantics. That does NOT mean
it cannot reproduce ``align_corners=True`` -- it means the difference lives
entirely in the coordinate transform, so an ADAPTER suffices and a second
sampler in this repo would be a duplicate definition of the same operation.

Inverting the sampler's own formula for an absolute pixel index ``p`` on an
axis of size ``S`` gives::

    coord = (p - (S - 1) / 2) / S

which is what :func:`sample_at_pixel_coords` applies, per axis, using that
axis's own size. Composing it with torch's ``align_corners=True`` normalization
``p = (n + 1) / 2 * (S - 1)`` recovers the upstream mapping end to end, so
``sample_at_pixel_coords(fmap, pix)`` is equivalent to upstream's
``bilinear_sampler(img, coords)``.

Stride-independence
-------------------
Neither :func:`coords_grid` nor :func:`sample_at_pixel_coords` mentions
:data:`~.components.SPATIAL_DIVISOR`. That is a property, not an omission: each
one reads the extent it needs from its own argument, so the SAME code samples
the stride-8 feature map and the full-resolution image. The rectifier calls
:func:`coords_grid` three times -- once at full resolution and twice at
``H // SPATIAL_DIVISOR`` (upstream ``model.py:47-49``) -- and the divisor lives
at those call sites, never here.

:func:`convex_upsample` is the exception, and necessarily so: the divisor is
not an extent it can read off an argument, it is the RATIO between its input
and its output. It reads :data:`~.components.SPATIAL_DIVISOR` directly, and no
``8`` appears in this module as a literal.

References:
    - Upstream release: https://github.com/fh2019ustc/DocScanner --
      ``model.py:9-22`` (``bilinear_sampler``), ``model.py:25-28``
      (``coords_grid``), ``model.py:45-51`` (``initialize_flow``).
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2.
"""

from typing import List, Union

import keras

from dl_techniques.layers.spatial_layer import interpolate_grid

from .components import SPATIAL_DIVISOR

# ---------------------------------------------------------------------


def coords_grid(
    batch: Union[int, "keras.KerasTensor"],
    height: int,
    width: int,
    dtype: str = "float32",
) -> "keras.KerasTensor":
    """Build the identity coordinate field, in ``(x, y)`` channel order.

    The returned field is the identity of :func:`sample_at_pixel_coords`:
    sampling any feature map at ``coords_grid(B, H, W)`` returns that feature
    map unchanged. It is also the ``coords0`` that DocScanner's refinement loop
    subtracts to turn an absolute coordinate field back into a flow
    (``model.py:87``).

    :param batch: Batch size to tile the field over. May be a symbolic scalar,
        e.g. ``keras.ops.shape(x)[0]``.
    :type batch: int or keras.KerasTensor
    :param height: Field height ``H``, in pixels. Must be a Python int.
    :type height: int
    :param width: Field width ``W``, in pixels. Must be a Python int.
    :type width: int
    :param dtype: Floating dtype of the returned field.
    :type dtype: str
    :return: ``(batch, H, W, 2)``. Channel ``0`` is ``x`` (the COLUMN index,
        ranging over ``[0, W - 1]``); channel ``1`` is ``y`` (the ROW index,
        ranging over ``[0, H - 1]``).
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-009: channel 0 is x (column),
    # channel 1 is y (row) -- and this is built from an EXPLICIT `indexing="ij"`
    # meshgrid whose two outputs are then stacked in the deliberate order
    # `[cols, rows]`. Do NOT port `model.py:26-27`'s `coords[::-1]` literally on
    # top of a `keras.ops.meshgrid` default: upstream reverses because pre-1.10
    # torch `meshgrid` had NO `indexing` argument and behaved as `'ij'`, while
    # `keras.ops.meshgrid` defaults to `'xy'` and has already reversed. A 1:1
    # translation double-reverses, emitting (y, x), and NOTHING downstream has a
    # shape symptom -- the field is (B, H, W, 2) either way, and at the square
    # 288x288 training resolution it is not even out of range. See decisions.md
    # D-009 and findings.md G-1. Guarded by `TestCoordsGridChannelOrder` (which
    # is deliberately NON-SQUARE; a square fixture cannot see this at all).
    rows, cols = keras.ops.meshgrid(
        keras.ops.arange(height, dtype=dtype),
        keras.ops.arange(width, dtype=dtype),
        indexing="ij",
    )
    grid = keras.ops.stack([cols, rows], axis=-1)
    grid = keras.ops.expand_dims(grid, axis=0)
    return keras.ops.repeat(grid, batch, axis=0)

# ---------------------------------------------------------------------


def sample_at_pixel_coords(
    fmap: "keras.KerasTensor",
    pix_xy: "keras.KerasTensor",
) -> "keras.KerasTensor":
    """Bilinearly sample ``fmap`` at absolute PIXEL coordinates, edge-clamped.

    Reproduces upstream ``bilinear_sampler`` (``model.py:9-22``), i.e.
    ``F.grid_sample(..., align_corners=True)`` over pixel coordinates, exactly:
    an integer coordinate ``p`` reads pixel ``p`` with no sub-pixel offset, and
    coordinates outside ``[0, S - 1]`` clamp to the edge rather than zero-pad.

    :param fmap: ``(B, H, W, C)`` feature map to read, channels-last.
    :type fmap: keras.KerasTensor
    :param pix_xy: ``(B, Hq, Wq, 2)`` query coordinates in PIXEL units, channel
        ``0`` = ``x`` (column, over ``[0, W - 1]``), channel ``1`` = ``y``
        (row, over ``[0, H - 1]``) -- the order :func:`coords_grid` emits.
    :type pix_xy: keras.KerasTensor
    :return: ``(B, Hq, Wq, C)`` sampled values, in ``fmap``'s own dtype.
        Differentiable w.r.t. ``pix_xy``.
    :rtype: keras.KerasTensor
    """
    fmap = keras.ops.convert_to_tensor(fmap)
    pix_xy = keras.ops.convert_to_tensor(pix_xy)

    fmap_shape = keras.ops.shape(fmap)
    size_h = keras.ops.cast(fmap_shape[1], "float32")
    size_w = keras.ops.cast(fmap_shape[2], "float32")

    pix_x = keras.ops.cast(pix_xy[..., 0], "float32")
    pix_y = keras.ops.cast(pix_xy[..., 1], "float32")

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-009: TWO independent conversions
    # happen on these four lines, and each one is silent when wrong.
    #
    # (1) THE SWAP. `pix_xy` is (x, y); `interpolate_grid` documents its `coords`
    #     as channel order [h, w], i.e. 'ij'. So x -> the W slot and y -> the H
    #     slot. Do NOT pass `pix_xy` through in its own order: at the square
    #     288x288 training resolution the result is merely a TRANSPOSED warp --
    #     same shape, same dtype, finite, trains. Every non-square guard in
    #     `test_components.py` exists for this line.
    #
    # (2) THE ADAPTER. `interpolate_grid` is half-pixel (`pix = coord * S +
    #     (S - 1) / 2`), so recovering an absolute pixel index needs
    #     `coord = (p - (S - 1) / 2) / S` -- and each axis divides by its OWN
    #     size. Do NOT use the more natural-looking half-pixel form `p / S - 0.5`
    #     (which is `align_corners=False`): it is a HALF-PIXEL systematic bias,
    #     invisible in every shape, dtype, finiteness and serialization test, and
    #     visible only against a hand-computed reference. Do NOT collapse the two
    #     sizes into one either -- that is inert only while H == W.
    #
    # See decisions.md D-009, findings.md F-14 (measured), and the RED proofs
    # recorded there for both mutations.
    coord_h = (pix_y - (size_h - 1.0) / 2.0) / size_h
    coord_w = (pix_x - (size_w - 1.0) / 2.0) / size_w
    coords_hw = keras.ops.stack([coord_h, coord_w], axis=-1)

    return interpolate_grid(coords=coords_hw, grid=fmap, order=1)

# ---------------------------------------------------------------------

# The size of the convex-upsample neighbourhood: a 3x3 window, so 9 candidate
# source pixels per destination sub-pixel. This is the ONLY place the 9 is
# written down. It is structural (upstream ``model.py:58``'s ``F.unfold(...,
# [3, 3], padding=1)``), not a tunable width, which is why it is a module
# constant here rather than a `_VARIANT_SPEC` row -- but the update block's mask
# head must emit `SPATIAL_DIVISOR ** 2 * CONVEX_NEIGHBOURS` (= 576) channels and
# derives that count from this name rather than restating it.
CONVEX_NEIGHBOURS: int = 9

# ---------------------------------------------------------------------


def convex_upsample(
    flow: "keras.KerasTensor",
    mask: "keras.KerasTensor",
) -> "keras.KerasTensor":
    """Learned convex 8x upsample of a coordinate/flow field (RAFT-style).

    Ports upstream ``RAFT.upsample_flow`` (``model.py:54-65``) to channels-last.
    Every destination sub-pixel is a CONVEX combination -- the weights are a
    softmax, so they are non-negative and sum to one -- of the 9 pixels in the
    3x3 neighbourhood of its source pixel, with the field values pre-scaled by
    :data:`~.components.SPATIAL_DIVISOR` because the coordinate units grow with
    the resolution.

    ``mask`` is consumed as RAW LOGITS. Upstream applies its ``0.25`` scaling in
    the update block (``update.py:104``, ``mask = .25 * self.mask(net)``), i.e.
    strictly before this function is reached, so that factor belongs to the mask
    head and is deliberately NOT applied here.

    :param flow: ``(B, H, W, C)`` field to upsample, channels-last. ``C`` is 2
        in this port, but nothing here depends on that.
    :type flow: keras.KerasTensor
    :param mask: ``(B, H, W, CONVEX_NEIGHBOURS * SPATIAL_DIVISOR ** 2)`` raw
        logits, i.e. ``(B, H, W, 576)``. The trailing axis decomposes
        major-to-minor as ``(neighbour, sub_row, sub_col)`` -- see the anchor in
        the body, which is the whole content of this function.
    :type mask: keras.KerasTensor
    :return: ``(B, H * SPATIAL_DIVISOR, W * SPATIAL_DIVISOR, C)``. Destination
        pixel ``(8 * h + i, 8 * w + j)`` is the mask-weighted sum of
        ``SPATIAL_DIVISOR * flow`` over the 3x3 neighbourhood of ``(h, w)``,
        zero-padded at the field border.
    :rtype: keras.KerasTensor
    """
    divisor = SPATIAL_DIVISOR

    scaled = keras.ops.multiply(
        flow, keras.ops.cast(divisor, flow.dtype)
    )

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-010: THREE orderings meet on the
    # next dozen lines and none of them has a shape symptom when reversed.
    #
    # (1) THE PATCH AXIS. `keras.ops.image.extract_patches` is channels-LAST and
    #     lays its `9 * C` trailing axis out as `(kh, kw, c)` major-to-minor --
    #     MEASURED, not assumed (probe: a (1,4,7,2) position-encoded input gives
    #     `[103,-103, 104,-104, 105,-105, 203,-203, ...]` at (h=2,w=3), so `c` is
    #     MINOR). `torch.nn.functional.unfold` lays the SAME data out as
    #     `(c, kh, kw)` -- channel MAJOR. So `up_flow.view(N, 2, 9, ...)` from
    #     `model.py:59` must NOT be transcribed as a `(..., C, 9)` reshape here;
    #     the correct channels-last split is `(..., 9, C)`. Both reshapes accept
    #     the same 18-element vector.
    #
    # (2) THE 576 SPLIT. `mask.view(N, 1, 9, 8, 8, H, W)` at `model.py:56` splits
    #     a channels-FIRST 576 axis as `(9, 8, 8)` major-to-minor, i.e. channel
    #     `c = k * 64 + i * 8 + j`. That index arithmetic is a property of the
    #     NUMBER, not of the memory layout, so the channels-last trailing 576
    #     axis splits identically: `(9, 8, 8)` = `(neighbour, sub_row, sub_col)`.
    #     Do NOT "translate" it to `(8, 8, 9)` by reversing it as if it were a
    #     channels-first/last conversion -- it is not one, and `8 * 8 * 9 == 576`
    #     either way.
    #
    # (3) THE INTERLEAVE. `model.py:63`'s `permute(0, 1, 4, 2, 5, 3)` maps
    #     `(N, C, i, j, H, W) -> (N, C, H, i, W, j)`, so the FIRST 8 (`i`) is the
    #     sub-ROW and the SECOND (`j`) is the sub-COLUMN: destination
    #     `(8h + i, 8w + j)`. Swapping them transposes every 8x8 block while
    #     leaving the output shape, dtype and finiteness untouched, and at the
    #     square 288x288 training resolution it does not even go out of range.
    #
    # (1) and (3) are pinned by the delta-impulse tests in
    # `TestConvexUpsampleNeighbourOrdering` / `TestConvexUpsampleSubPixelInterleave`
    # (deliberately NON-SQUARE, H=4 W=7 -- a square fixture cannot see (3)), and
    # (2) additionally by the uniform-mask control against an independently
    # computed numpy reference. Both mutations were run and observed RED; see
    # decisions.md D-010.
    patches = keras.ops.image.extract_patches(
        scaled, size=3, strides=1, dilation_rate=1, padding="same"
    )

    shape = keras.ops.shape(flow)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]

    patches = keras.ops.reshape(
        patches, (batch, height, width, CONVEX_NEIGHBOURS, 1, 1, channels)
    )
    weights = keras.ops.reshape(
        mask, (batch, height, width, CONVEX_NEIGHBOURS, divisor, divisor, 1)
    )

    # `torch.softmax(mask, dim=2)` -- over the 9 NEIGHBOURS, never over the 64
    # sub-pixels. Each destination sub-pixel gets its own independent convex
    # combination; the 64 of them are not in competition with one another.
    weights = keras.ops.softmax(weights, axis=3)

    # (B, H, W, 9, 8, 8, C) -> (B, H, W, 8, 8, C), the `dim=2` sum of
    # `model.py:62`.
    blended = keras.ops.sum(weights * patches, axis=3)

    # (B, H, W, i, j, C) -> (B, H, i, W, j, C), then fold each (H, i) and
    # (W, j) pair into one axis. This is `model.py:63-65`.
    blended = keras.ops.transpose(blended, (0, 1, 3, 2, 4, 5))
    return keras.ops.reshape(
        blended, (batch, height * divisor, width * divisor, channels)
    )

# ---------------------------------------------------------------------

__all__: List[str] = [
    "CONVEX_NEIGHBOURS",
    "convex_upsample",
    "coords_grid",
    "sample_at_pixel_coords",
]
