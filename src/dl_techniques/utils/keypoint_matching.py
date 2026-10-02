"""Keypoint-level ground-truth match labels from a known homography.

Stage-1 LightGlue training uses image pairs related by a known homography.
This module turns two padded keypoint sets and ``H0to1`` into per-keypoint
labels. Everything is pure ``keras.ops``, has static shapes, runs inside
``tf.function`` and computes in float32 regardless of the dtype policy.

Label format (``matches0`` is ``(B, M)``, ``matches1`` is ``(B, N)``, int32)
-------------------------------------------------------------------------
- ``j >= 0``: matched to keypoint ``j`` of the other image;
- ``-1``: dustbin (a real keypoint with no counterpart);
- ``-2``: ignored (ambiguous) or padded.

Rules
-----
Keypoints are ``(x, y)`` pixels. Keypoints of image 0 are warped with ``H0to1``
and keypoints of image 1 with ``H0to1^-1``. For a pair ``(i, j)`` the distance
is ``max(|H p0_i - p1_j|, |p0_i - H^-1 p1_j|)``, the larger of the two
reprojection errors, one measured in each image (glue-factory's choice).

- Positive: ``i`` and ``j`` are each other's nearest neighbour (mutual NN) and
  their distance is ``<= pos_threshold``.
- Dustbin: no pair of the keypoint has a distance ``<= neg_threshold``, or the
  keypoint warps outside the other image (``0 <= x < w``, ``0 <= y < h``).
  Warping outside is dustbin, never ignored.
- Ignored: a real, in-image keypoint that is not positive but has a candidate
  within ``neg_threshold`` (it lost a mutual-NN contest, or sits in the band
  ``pos_threshold < dist <= neg_threshold``).
- Padded slots are always ``-2``.

Decision on the threshold wording: the plan text says "within 3 px" for both
positives and dustbin and also names a band ``(1 px, 3 px]``, which cannot all
hold at once. The glue-factory reading is implemented: ``pos_threshold=3`` and
``neg_threshold=3`` by default, so the ignore set is "has a candidate within
3 px but did not win the mutual-NN contest"; raising ``neg_threshold`` above
``pos_threshold`` widens it into a genuine band. ``neg_threshold`` must be
``>= pos_threshold``.

Ties: ``argmin`` picks the lowest index, so with exact duplicates only the
first copy can be positive; later copies are ignored (-2).

Numerics: ``H^-1`` is the adjugate over the determinant. If ``|det|`` is
below ``1e-6 * ||H||_F^3`` (near singular, including a zero matrix) the
homography is declared unusable: nothing is positive and every real keypoint
is dustbin. Points whose homogeneous ``w`` is ``<= 1e-6`` (at or behind the
horizon) or non-finite count as warping outside the image. No NaN is ever
produced.
"""

import keras
from keras import ops
from typing import Dict

# ---------------------------------------------------------------------

_F32 = "float32"
_W_EPS = 1e-6
_DET_EPS = 1e-6
_INF = 1e30


def invert_3x3(h):
    """Invert batched 3x3 matrices by the adjugate, flagging near-singular ones.

    :param h: ``(B, 3, 3)`` matrices.
    :return: ``(h_inv, ok)``: ``h_inv`` is ``(B, 3, 3)`` float32 (identity where
        not ``ok``, so downstream maths stays finite) and ``ok`` is a ``(B,)``
        bool, False when ``|det| < 1e-6 * ||H||_F^3`` or non-finite.
    """
    h = ops.cast(h, _F32)
    a, b, c = h[:, 0, 0], h[:, 0, 1], h[:, 0, 2]
    d, e, f = h[:, 1, 0], h[:, 1, 1], h[:, 1, 2]
    g, i, j = h[:, 2, 0], h[:, 2, 1], h[:, 2, 2]
    c00, c01, c02 = e * j - f * i, f * g - d * j, d * i - e * g
    det = a * c00 + b * c01 + c * c02
    norm = ops.sqrt(ops.sum(ops.square(h), axis=(1, 2)))
    ok = ops.abs(det) >= _DET_EPS * norm * norm * norm
    ok = ops.logical_and(ok, ops.isfinite(det))
    safe = ops.where(ok, det, ops.ones_like(det))
    adj = ops.stack(
        [
            ops.stack([c00, c * i - b * j, b * f - c * e], axis=-1),
            ops.stack([c01, a * j - c * g, c * d - a * f], axis=-1),
            ops.stack([c02, b * g - a * i, a * e - b * d], axis=-1),
        ],
        axis=1,
    )
    inv = adj / safe[:, None, None]
    eye = ops.broadcast_to(ops.eye(3, dtype=_F32)[None], ops.shape(inv))
    return ops.where(ok[:, None, None], inv, eye), ok


def _project(points, h, size):
    """Project ``(B, K, 2)`` points with ``h`` and test the result against ``size``.

    :return: ``(projected (B, K, 2), inside (B, K) bool)``; ``projected`` is
        zero where the projection is invalid.
    """
    ones = ops.ones_like(points[..., :1])
    homo = ops.concatenate([points, ones], axis=-1)
    out = ops.einsum("bij,bkj->bki", h, homo)
    w = out[..., 2:3]
    valid = ops.logical_and(w > _W_EPS, ops.isfinite(w))
    xy = out[..., :2] / ops.where(valid, w, ops.ones_like(w))
    valid = ops.logical_and(valid[..., 0], ops.all(ops.isfinite(xy), axis=-1))
    xy = ops.where(valid[..., None], xy, ops.zeros_like(xy))
    wh = ops.cast(size, _F32)[:, None, :]
    inside = ops.all(ops.logical_and(xy >= 0.0, xy < wh), axis=-1)
    return xy, ops.logical_and(inside, valid)


def _mutual_nn(dist, pos_threshold):
    """Mutual nearest neighbours of a ``(B, M, N)`` distance matrix.

    :return: ``(nn0, nn1, min0, min1, pos0, pos1)``: argmin along each axis,
        the minima, and the bool positive masks (mutual and within threshold).
    """
    m, n = dist.shape[1], dist.shape[2]
    nn0 = ops.argmin(dist, axis=2)
    nn1 = ops.argmin(dist, axis=1)
    min0 = ops.min(dist, axis=2)
    min1 = ops.min(dist, axis=1)
    back0 = ops.take_along_axis(nn1, nn0, axis=1)
    back1 = ops.take_along_axis(nn0, nn1, axis=1)
    mutual0 = back0 == ops.arange(m, dtype=nn0.dtype)[None]
    mutual1 = back1 == ops.arange(n, dtype=nn1.dtype)[None]
    pos0 = ops.logical_and(mutual0, min0 <= pos_threshold)
    pos1 = ops.logical_and(mutual1, min1 <= pos_threshold)
    return nn0, nn1, min0, min1, pos0, pos1


def homography_matches(
    keypoints0,
    keypoints1,
    mask0,
    mask1,
    H0to1,
    image_size0,
    image_size1,
    pos_threshold: float = 3.0,
    neg_threshold: float = 3.0,
) -> Dict[str, "keras.KerasTensor"]:
    """Ground-truth match labels for two keypoint sets related by a homography.

    :param keypoints0: ``(B, M, 2)`` xy pixels of image 0.
    :param keypoints1: ``(B, N, 2)`` xy pixels of image 1.
    :param mask0: ``(B, M)`` truthy for real keypoints.
    :param mask1: ``(B, N)`` truthy for real keypoints.
    :param H0to1: ``(B, 3, 3)`` homography mapping image-0 pixels to image 1.
    :param image_size0: ``(B, 2)`` ``(w, h)`` of image 0.
    :param image_size1: ``(B, 2)`` ``(w, h)`` of image 1.
    :param pos_threshold: maximal distance (pixels) of a positive pair.
    :param neg_threshold: keypoints without a candidate within this distance are
        dustbin; must be ``>= pos_threshold``.
    :return: dict with ``matches0`` ``(B, M)`` and ``matches1`` ``(B, N)``, int32,
        format and rules in the module docstring. ``matches0[i] == j`` for a
        match iff ``matches1[j] == i``.
    :raises ValueError: if ``neg_threshold < pos_threshold``.
    """
    if neg_threshold < pos_threshold:
        raise ValueError(
            f"neg_threshold ({neg_threshold}) must be >= pos_threshold ({pos_threshold})."
        )
    kp0 = ops.cast(keypoints0, _F32)
    kp1 = ops.cast(keypoints1, _F32)
    real0 = ops.cast(mask0, "bool")
    real1 = ops.cast(mask1, "bool")
    h = ops.cast(H0to1, _F32)
    h_inv, ok = invert_3x3(h)

    kp0_in1, inside0 = _project(kp0, h, image_size1)
    kp1_in0, inside1 = _project(kp1, h_inv, image_size0)
    usable = ok[:, None]
    vis0 = ops.logical_and(ops.logical_and(inside0, usable), real0)
    vis1 = ops.logical_and(ops.logical_and(inside1, usable), real1)

    def pairwise(a, b):
        diff = a[:, :, None, :] - b[:, None, :, :]
        return ops.sqrt(ops.sum(ops.square(diff), axis=-1))

    dist = ops.maximum(pairwise(kp0_in1, kp1), pairwise(kp0, kp1_in0))
    pair_ok = ops.logical_and(vis0[:, :, None], vis1[:, None, :])
    dist = ops.where(pair_ok, dist, ops.full_like(dist, _INF))

    nn0, nn1, min0, min1, pos0, pos1 = _mutual_nn(dist, pos_threshold)
    dust0 = ops.logical_and(real0, min0 > neg_threshold)
    dust1 = ops.logical_and(real1, min1 > neg_threshold)

    def labels(pos, nn, dust):
        out = ops.where(dust, -1, -2)
        return ops.cast(ops.where(pos, nn, out), "int32")

    return {
        "matches0": labels(pos0, nn0, dust0),
        "matches1": labels(pos1, nn1, dust1),
    }


def label_statistics(matches, mask) -> Dict[str, "keras.KerasTensor"]:
    """Fractions of positive, dustbin and ignored labels among real keypoints.

    :param matches: ``(B, K)`` int32 labels from :func:`homography_matches`.
    :param mask: ``(B, K)`` truthy for real keypoints.
    :return: scalar float32 ``positive``, ``dustbin``, ``ignored`` fractions of the
        real keypoints of the whole batch (0 when there are none).
    """
    real = ops.cast(mask, "bool")
    total = ops.maximum(ops.sum(ops.cast(real, _F32)), 1.0)

    def frac(cond):
        return ops.sum(ops.cast(ops.logical_and(cond, real), _F32)) / total

    return {
        "positive": frac(matches >= 0),
        "dustbin": frac(matches == -1),
        "ignored": frac(matches == -2),
    }
