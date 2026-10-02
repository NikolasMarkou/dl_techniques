"""Batched, graph-compatible SuperPoint decode: heatmap, NMS, top-k, descriptors.

SuperPoint emits dense outputs (cell logits ``(B, H/8, W/8, 65)`` and a dense
descriptor map ``(B, H, W, D)``). Matchers such as LightGlue consume a sparse,
padded set ``(keypoints, descriptors, mask)``. This module is that bridge. Every
function is pure, built from ``keras.ops`` only, has static output shapes (so it
runs inside ``tf.function`` and under ``jit_compile``) and computes in float32
regardless of the global dtype policy: callers may feed float16 / bfloat16
tensors, results are always float32.

The numpy single-image oracles are ``grid_logits_to_heatmap`` and ``simple_nms`` in
``src/train/superpoint/homographic_adaptation.py``. Semantics kept identical:

- the heatmap is the softmax over the 65 channels with the dustbin (last
  channel) dropped, scattered to full resolution with channel ``k`` mapped to
  within-cell ``(row=k // 8, col=k % 8)``;
- a pixel survives NMS iff it equals the max over the
  ``(2*radius+1) x (2*radius+1)`` window (SAME padding, ties all survive) and is
  ``> threshold``. The oracle does a single max-pool equality test; there are no
  repeated suppression passes, so none are done here.

Conventions
-----------
- Keypoints are ``(x, y)`` in pixel units, pixel centers at integer
  coordinates: keypoint ``(3, 5)`` is exactly the pixel at column 3, row 5.
- Descriptor sampling is bilinear with align-corners semantics: coordinates are
  clamped into ``[0, W-1] x [0, H-1]`` and the four neighbours are the pixels at
  ``floor`` and ``floor + 1`` (the latter clamped). At an integer coordinate the
  result is exactly that pixel's descriptor.
- Padded slots: keypoints, scores and descriptors are exactly zero and
  ``mask`` is ``False``.
- Tie-breaking in top-k: equal scores are ordered by ascending row-major pixel
  index (``y * W + x``), i.e. the pixel nearer the top-left wins. This is
  ``keras.ops.top_k``'s documented lower-index-first behaviour.

Dtype note: the SuperPoint model itself is untouched; its own
``np.finfo``-derived behaviour under ``mixed_bfloat16`` remains as is. This
decode only casts its outputs to float32 first.
"""

import keras
from keras import ops
from typing import Dict

from dl_techniques.utils.dtype_policy import stability_floor

# ---------------------------------------------------------------------

_F32 = "float32"


def _dims(x):
    """Return ``(static_or_dynamic)`` dims, preferring static ints."""
    static = tuple(x.shape)
    dynamic = ops.shape(x)
    return tuple(
        static[i] if static[i] is not None else dynamic[i] for i in range(len(static))
    )


# ---------------------------------------------------------------------


def superpoint_heatmap(logits):
    """Decode cell logits to a dense keypoint-probability heatmap.

    :param logits: ``(B, Hc, Wc, cell*cell + 1)`` raw detector logits (any float
        dtype). The last channel is the dustbin.
    :return: ``(B, Hc*cell, Wc*cell)`` float32 heatmap in ``[0, 1]``.
    :raises ValueError: if the channel count is not ``cell*cell + 1``.
    """
    logits = ops.cast(logits, _F32)
    channels = logits.shape[-1]
    cell = int(round((channels - 1) ** 0.5))
    if channels is None or cell * cell + 1 != channels or cell < 1:
        raise ValueError(
            f"superpoint_heatmap expects last dim cell*cell+1 (65 for cell 8), got {channels}."
        )
    b, hc, wc, _ = _dims(logits)
    probs = ops.softmax(logits, axis=-1)[..., :-1]
    grid = ops.reshape(probs, (b, hc, wc, cell, cell))  # (B, Hc, Wc, row, col)
    grid = ops.transpose(grid, (0, 1, 3, 2, 4))  # (B, Hc, row, Wc, col)
    return ops.reshape(grid, (b, hc * cell, wc * cell))


def heatmap_nms(heatmap, radius: int):
    """Local-maximum mask of a heatmap.

    A pixel is True iff it equals the max of its ``(2*radius+1)^2`` window
    (SAME padding; ties all survive). Threshold and border are applied
    separately by :func:`select_keypoints`.

    :param heatmap: ``(B, H, W)`` score map.
    :param radius: window radius in pixels, ``>= 0`` (0 keeps every pixel).
    :return: ``(B, H, W)`` bool mask.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    heatmap = ops.cast(heatmap, _F32)
    pooled = ops.max_pool(
        ops.expand_dims(heatmap, -1), 2 * radius + 1, strides=1, padding="same"
    )[..., 0]
    return ops.equal(heatmap, pooled)


def border_mask(height, width, border: int):
    """Boolean ``(H, W)`` mask, True for pixels at least ``border`` from every edge."""
    ys = ops.arange(height)[:, None]
    xs = ops.arange(width)[None, :]
    return (
        (ys >= border) & (ys < height - border) & (xs >= border) & (xs < width - border)
    )


def select_keypoints(
    heatmap,
    max_keypoints: int,
    threshold: float = 0.0,
    nms_radius: int = 4,
    border: int = 0,
):
    """NMS, threshold, border removal and padded top-k selection.

    :param heatmap: ``(B, H, W)`` score map.
    :param max_keypoints: output slots ``N``. If ``N > H*W`` the surplus slots are
        padding.
    :param threshold: a pixel must have score ``> threshold``.
    :param nms_radius: see :func:`heatmap_nms`.
    :param border: pixels within ``border`` of any edge are discarded.
    :return: ``(keypoints, scores, mask)`` with shapes ``(B, N, 2)`` float32
        ``(x, y)``, ``(B, N)`` float32 and ``(B, N)`` bool. Real points come first
        in descending score order (ties: ascending ``y*W + x``); padded slots are
        zero with ``mask`` False.
    """
    if max_keypoints < 1:
        raise ValueError(f"max_keypoints must be >= 1, got {max_keypoints}.")
    heatmap = ops.cast(heatmap, _F32)
    b, h, w = _dims(heatmap)
    keep = heatmap_nms(heatmap, nms_radius) & (heatmap > threshold)
    if border > 0:
        keep = keep & border_mask(h, w, border)[None]

    # Dropped pixels rank below every real score (scores are >= 0 for a heatmap,
    # but -1 is also below any threshold >= 0; the mask is the source of truth).
    ranked = ops.where(keep, heatmap, ops.full_like(heatmap, -1.0))
    flat = ops.reshape(ranked, (b, h * w))
    k = max_keypoints
    n_pix = h * w if isinstance(h, int) and isinstance(w, int) else None
    k_eff = k if n_pix is None else min(k, n_pix)
    values, index = ops.top_k(flat, k_eff, sorted=True)
    if k_eff < k:
        values = ops.pad(values, [[0, 0], [0, k - k_eff]], constant_values=-1.0)
        index = ops.pad(index, [[0, 0], [0, k - k_eff]])

    # Heatmap scores are >= 0 and dropped / padded slots hold -1, so the sign is the mask.
    mask = values >= 0.0
    mask_f = ops.cast(mask, _F32)
    ys = ops.cast(index // w, _F32)
    xs = ops.cast(index % w, _F32)
    keypoints = ops.stack([xs, ys], axis=-1) * mask_f[..., None]
    scores = ops.where(mask, values, ops.zeros_like(values))
    return keypoints, scores, mask


def sample_descriptors(descriptor_map, keypoints):
    """Bilinearly sample a dense descriptor map at pixel ``(x, y)`` and L2-normalise.

    :param descriptor_map: ``(B, H, W, D)`` dense descriptors.
    :param keypoints: ``(B, N, 2)`` ``(x, y)`` pixel coordinates, centers at
        integers, clamped to ``[0, W-1] x [0, H-1]`` (align-corners).
    :return: ``(B, N, D)`` float32 unit-norm descriptors. A zero sampled vector
        stays zero (norm clamped by a float32 stability floor).
    """
    desc = ops.cast(descriptor_map, _F32)
    kp = ops.cast(keypoints, _F32)
    b, h, w, d = _dims(desc)
    flat = ops.reshape(desc, (b, h * w, d))

    x = ops.clip(kp[..., 0], 0.0, ops.cast(w - 1, _F32))
    y = ops.clip(kp[..., 1], 0.0, ops.cast(h - 1, _F32))
    x0f, y0f = ops.floor(x), ops.floor(y)
    wx = (x - x0f)[..., None]
    wy = (y - y0f)[..., None]
    x0 = ops.cast(x0f, "int32")
    y0 = ops.cast(y0f, "int32")
    x1 = ops.minimum(x0 + 1, w - 1)
    y1 = ops.minimum(y0 + 1, h - 1)

    def gather(yy, xx):
        return ops.take_along_axis(flat, (yy * w + xx)[..., None], axis=1)

    out = (
        gather(y0, x0) * (1.0 - wy) * (1.0 - wx)
        + gather(y0, x1) * (1.0 - wy) * wx
        + gather(y1, x0) * wy * (1.0 - wx)
        + gather(y1, x1) * wy * wx
    )
    norm = ops.sqrt(ops.sum(ops.square(out), axis=-1, keepdims=True))
    return out / ops.maximum(norm, stability_floor(_F32, 1e-12))


def decode_superpoint(
    outputs: Dict[str, "keras.KerasTensor"],
    max_keypoints: int = 512,
    threshold: float = 0.005,
    nms_radius: int = 4,
    border: int = 4,
) -> Dict[str, "keras.KerasTensor"]:
    """Decode a SuperPoint output dict into padded sparse keypoints.

    :param outputs: dict with ``"keypoints"`` ``(B, H/8, W/8, 65)`` logits and
        ``"descriptors"`` ``(B, H, W, D)`` unit-norm dense descriptors.
    :param max_keypoints: padded slot count ``N``.
    :param threshold: minimum heatmap probability.
    :param nms_radius: NMS radius in pixels.
    :param border: edge margin in pixels.
    :return: dict ``keypoints (B,N,2)`` xy float32, ``scores (B,N)``,
        ``descriptors (B,N,D)`` float32 unit norm on real slots and zero on
        padded slots, ``mask (B,N)`` bool.
    """
    heat = superpoint_heatmap(outputs["keypoints"])
    keypoints, scores, mask = select_keypoints(
        heat, max_keypoints, threshold, nms_radius, border
    )
    descriptors = sample_descriptors(outputs["descriptors"], keypoints)
    descriptors = descriptors * ops.cast(mask, _F32)[..., None]
    return {
        "keypoints": keypoints,
        "scores": scores,
        "descriptors": descriptors,
        "mask": mask,
    }
