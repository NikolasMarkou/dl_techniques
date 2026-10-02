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
and keypoints of image 1 with ``H0to1^-1``. For a pair ``(i, j)`` the forward
error is ``|H p0_i - p1_j|`` (measured in image 1) and the backward error is
``|p0_i - H^-1 p1_j|`` (measured in image 0). This follows glue-factory's
``gt_matches_from_homography``:

- Positive: ``i`` and ``j`` are each other's nearest neighbour (mutual NN)
  under the pair distance ``max(forward, backward)`` and that distance is
  ``< pos_threshold`` (strict, as glue-factory's ``dist < pos_th**2``; a pair at
  exactly 3.0 px is not positive).
- Dustbin of image 0: every candidate ``j`` has a FORWARD error
  ``> neg_threshold`` (one-sided: the backward error plays no part).
  Dustbin of image 1: every candidate ``i`` has a BACKWARD error
  ``> neg_threshold`` (one-sided: the forward error plays no part).
- Extension to glue-factory (documented, not in its function): a keypoint that
  warps outside the other image (``0 <= x < w``, ``0 <= y < h``) is dustbin,
  never ignored, and a pair with a warped-outside member is never positive.
- Ignored: a real, in-image keypoint that is neither positive nor dustbin (it
  lost a mutual-NN contest, or the max-distance partner is farther than
  ``pos_threshold`` while a one-sided candidate is within ``neg_threshold``).
- Padded slots are always ``-2``.

Decision history: D-011 first took the dustbin from the two-sided max distance;
a run of glue-factory's function on real keypoints showed that mislabels 3.3%
of keypoints as dustbin that the reference leaves ignored. D-016 replaced it
with the one-sided rule above. Both thresholds default to 3 px and
``neg_threshold`` must be ``>= pos_threshold``; raising it above
``pos_threshold`` widens the ignore set.

Border: the outside-image extension above equals glue-factory only when the
detector keeps keypoints at least ``pos_threshold`` pixels from the image
border (``--border >= 3`` in the trainer): a keypoint within ``pos_threshold``
of the edge can have a partner that warps just outside the other image, which
this module labels dustbin and glue-factory (no bounds rule) labels positive.

Ties: ``argmin`` picks the lowest index, so with exact duplicates only the
first copy can be positive; later copies are ignored (-2).

Numerics: ``H^-1`` is the adjugate over the determinant. If ``|det Hn|`` is
below ``1e-6 * ||Hn||_F^3``, with ``Hn = diag(1/w1, 1/h1, 1) @ H @ diag(w0, h0, 1)``
the size-normalised matrix (near singular, including a zero matrix; D-020: a
pixel-unit test flags ordinary homographies at 240 px and above), the
homography is declared unusable: nothing is positive and every real keypoint
is dustbin. Points whose homogeneous ``w`` is ``<= 1e-6`` (at or behind the
horizon) or non-finite count as warping outside the image. No NaN is ever
produced.
"""

import keras
from typing import Dict, Tuple

# ---------------------------------------------------------------------

_F32 = "float32"
_W_EPS = 1e-6
_DET_EPS = 1e-6
_INF = 1e30


def _size_scale(size: "keras.KerasTensor") -> "keras.KerasTensor":
    """``(B, 3, 3)`` matrices ``diag(w, h, 1)`` from ``(B, 2)`` sizes."""
    size = keras.ops.cast(size, _F32)
    diag = keras.ops.concatenate([size, keras.ops.ones_like(size[:, :1])], axis=-1)
    return diag[:, :, None] * keras.ops.eye(3, dtype=_F32)[None]


def invert_3x3(
    h: "keras.KerasTensor",
    size0: "keras.KerasTensor" = None,
    size1: "keras.KerasTensor" = None,
) -> Tuple["keras.KerasTensor", "keras.KerasTensor"]:
    """Invert batched 3x3 homographies by the adjugate, flagging near-singular ones.

    With both sizes given, the conditioning test runs on the size-normalised
    matrix ``Hn = diag(1/w1, 1/h1, 1) @ H @ diag(w0, h0, 1)``. In pixel units
    ``||H||_F`` is dominated by the translation entries, so a well-conditioned
    homography (det near 1) at 240 px fails a pixel-unit test (D-020). The
    inverse is taken from ``Hn``: ``H^-1 = diag(w0, h0, 1) @ Hn^-1 @ diag(1/w1, 1/h1, 1)``,
    so the test and the inverse see one matrix.

    :param h: ``(B, 3, 3)`` matrices (image-0 pixels to image-1 pixels).
    :param size0: ``(B, 2)`` ``(w, h)`` of image 0, or None (no normalisation,
        only meaningful for matrices already in unit coordinates).
    :param size1: ``(B, 2)`` ``(w, h)`` of image 1, or None.
    :return: ``(h_inv, ok)``: ``h_inv`` is ``(B, 3, 3)`` float32 (identity where
        not ``ok``, so downstream maths stays finite) and ``ok`` is a ``(B,)``
        bool, False when ``|det Hn| < 1e-6 * ||Hn||_F^3`` or non-finite.
    """
    h = keras.ops.cast(h, _F32)
    normalise = size0 is not None and size1 is not None
    if normalise:
        s0 = _size_scale(size0)
        s1_inv = _size_scale(1.0 / keras.ops.cast(size1, _F32))
        hn = keras.ops.matmul(s1_inv, keras.ops.matmul(h, s0))
    else:
        hn = h
    a, b, c = hn[:, 0, 0], hn[:, 0, 1], hn[:, 0, 2]
    d, e, f = hn[:, 1, 0], hn[:, 1, 1], hn[:, 1, 2]
    g, i, j = hn[:, 2, 0], hn[:, 2, 1], hn[:, 2, 2]
    c00, c01, c02 = e * j - f * i, f * g - d * j, d * i - e * g
    det = a * c00 + b * c01 + c * c02
    norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(hn), axis=(1, 2)))
    # DECISION plan-2026-10-02T084508-dd2c07ac/D-020
    # The conditioning test is on the SIZE-NORMALISED matrix. Do NOT test the
    # pixel-unit matrix: its Frobenius norm is dominated by the translation
    # entries, so 8% of ordinary 240 px pairs (50% at 480 px) were declared
    # singular and labelled all-dustbin. Guard:
    # tests/test_utils/test_keypoint_matching.py::TestSingularityScale
    ok = keras.ops.abs(det) >= _DET_EPS * norm * norm * norm
    ok = keras.ops.logical_and(ok, keras.ops.isfinite(det))
    safe = keras.ops.where(ok, det, keras.ops.ones_like(det))
    adj = keras.ops.stack(
        [
            keras.ops.stack([c00, c * i - b * j, b * f - c * e], axis=-1),
            keras.ops.stack([c01, a * j - c * g, c * d - a * f], axis=-1),
            keras.ops.stack([c02, b * g - a * i, a * e - b * d], axis=-1),
        ],
        axis=1,
    )
    inv = adj / safe[:, None, None]
    if normalise:
        inv = keras.ops.matmul(s0, keras.ops.matmul(inv, s1_inv))
    eye = keras.ops.broadcast_to(keras.ops.eye(3, dtype=_F32)[None], keras.ops.shape(inv))
    return keras.ops.where(ok[:, None, None], inv, eye), ok


def _project(
    points: "keras.KerasTensor", h: "keras.KerasTensor", size: "keras.KerasTensor"
) -> Tuple["keras.KerasTensor", "keras.KerasTensor"]:
    """Project ``(B, K, 2)`` points with ``h`` and test the result against ``size``.

    :return: ``(projected (B, K, 2), inside (B, K) bool)``; ``projected`` is
        zero where the projection is invalid.
    """
    ones = keras.ops.ones_like(points[..., :1])
    homo = keras.ops.concatenate([points, ones], axis=-1)
    out = keras.ops.einsum("bij,bkj->bki", h, homo)
    w = out[..., 2:3]
    valid = keras.ops.logical_and(w > _W_EPS, keras.ops.isfinite(w))
    xy = out[..., :2] / keras.ops.where(valid, w, keras.ops.ones_like(w))
    valid = keras.ops.logical_and(valid[..., 0], keras.ops.all(keras.ops.isfinite(xy), axis=-1))
    xy = keras.ops.where(valid[..., None], xy, keras.ops.zeros_like(xy))
    wh = keras.ops.cast(size, _F32)[:, None, :]
    inside = keras.ops.all(keras.ops.logical_and(xy >= 0.0, xy < wh), axis=-1)
    return xy, keras.ops.logical_and(inside, valid)


def _mutual_nn(
    dist: "keras.KerasTensor", pos_threshold: float
) -> Tuple["keras.KerasTensor", "keras.KerasTensor", "keras.KerasTensor", "keras.KerasTensor", "keras.KerasTensor", "keras.KerasTensor"]:
    """Mutual nearest neighbours of a ``(B, M, N)`` distance matrix.

    :return: ``(nn0, nn1, min0, min1, pos0, pos1)``: argmin along each axis,
        the minima, and the bool positive masks (mutual and within threshold).
    """
    m, n = dist.shape[1], dist.shape[2]
    nn0 = keras.ops.argmin(dist, axis=2)
    nn1 = keras.ops.argmin(dist, axis=1)
    min0 = keras.ops.min(dist, axis=2)
    min1 = keras.ops.min(dist, axis=1)
    back0 = keras.ops.take_along_axis(nn1, nn0, axis=1)
    back1 = keras.ops.take_along_axis(nn0, nn1, axis=1)
    mutual0 = back0 == keras.ops.arange(m, dtype=nn0.dtype)[None]
    mutual1 = back1 == keras.ops.arange(n, dtype=nn1.dtype)[None]
    pos0 = keras.ops.logical_and(mutual0, min0 < pos_threshold)
    pos1 = keras.ops.logical_and(mutual1, min1 < pos_threshold)
    return nn0, nn1, min0, min1, pos0, pos1


def homography_matches(
    keypoints0: "keras.KerasTensor",
    keypoints1: "keras.KerasTensor",
    mask0: "keras.KerasTensor",
    mask1: "keras.KerasTensor",
    H0to1: "keras.KerasTensor",
    image_size0: "keras.KerasTensor",
    image_size1: "keras.KerasTensor",
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
    kp0 = keras.ops.cast(keypoints0, _F32)
    kp1 = keras.ops.cast(keypoints1, _F32)
    real0 = keras.ops.cast(mask0, "bool")
    real1 = keras.ops.cast(mask1, "bool")
    h = keras.ops.cast(H0to1, _F32)
    h_inv, ok = invert_3x3(h, image_size0, image_size1)

    kp0_in1, inside0 = _project(kp0, h, image_size1)
    kp1_in0, inside1 = _project(kp1, h_inv, image_size0)
    usable = ok[:, None]
    vis0 = keras.ops.logical_and(keras.ops.logical_and(inside0, usable), real0)
    vis1 = keras.ops.logical_and(keras.ops.logical_and(inside1, usable), real1)

    def pairwise(a: "keras.KerasTensor", b: "keras.KerasTensor") -> "keras.KerasTensor":
        diff = a[:, :, None, :] - b[:, None, :, :]
        return keras.ops.sqrt(keras.ops.sum(keras.ops.square(diff), axis=-1))

    fwd = pairwise(kp0_in1, kp1)
    bwd = pairwise(kp0, kp1_in0)
    dist = keras.ops.maximum(fwd, bwd)
    pair_ok = keras.ops.logical_and(vis0[:, :, None], vis1[:, None, :])
    dist = keras.ops.where(pair_ok, dist, keras.ops.full_like(dist, _INF))

    nn0, nn1, min0, min1, pos0, pos1 = _mutual_nn(dist, pos_threshold)

    # DECISION plan-2026-10-02T084508-dd2c07ac/D-016
    # Dustbin uses ONE-SIDED distances, exactly as glue-factory
    # gt_matches_from_homography: image 0 from the forward error only,
    # image 1 from the backward error only. Do NOT reuse the two-sided max
    # (min0 / min1) here: it labels as dustbin keypoints that glue-factory
    # leaves ignored (3.3% of real keypoints on the reviewer's 64-pair probe).
    # Guard: tests/test_utils/test_keypoint_matching.py (one-sided cases and
    # the frozen glue-factory fixture). Supersedes the max rule of D-011.
    big = keras.ops.full_like(fwd, _INF)
    fwd_min = keras.ops.min(keras.ops.where(real1[:, None, :], fwd, big), axis=2)
    bwd_min = keras.ops.min(keras.ops.where(real0[:, :, None], bwd, big), axis=1)
    dust0 = keras.ops.logical_and(real0, keras.ops.logical_or(keras.ops.logical_not(vis0), fwd_min > neg_threshold))
    dust1 = keras.ops.logical_and(real1, keras.ops.logical_or(keras.ops.logical_not(vis1), bwd_min > neg_threshold))

    def labels(pos: "keras.KerasTensor", nn: "keras.KerasTensor", dust: "keras.KerasTensor") -> "keras.KerasTensor":
        out = keras.ops.where(dust, -1, -2)
        return keras.ops.cast(keras.ops.where(pos, nn, out), "int32")

    return {
        "matches0": labels(pos0, nn0, dust0),
        "matches1": labels(pos1, nn1, dust1),
    }


def label_statistics(matches: "keras.KerasTensor", mask: "keras.KerasTensor") -> Dict[str, "keras.KerasTensor"]:
    """Fractions of positive, dustbin and ignored labels among real keypoints.

    :param matches: ``(B, K)`` int32 labels from :func:`homography_matches`.
    :param mask: ``(B, K)`` truthy for real keypoints.
    :return: scalar float32 ``positive``, ``dustbin``, ``ignored`` fractions of the
        real keypoints of the whole batch (0 when there are none).
    """
    real = keras.ops.cast(mask, "bool")
    total = keras.ops.maximum(keras.ops.sum(keras.ops.cast(real, _F32)), 1.0)

    def frac(cond: "keras.KerasTensor") -> "keras.KerasTensor":
        return keras.ops.sum(keras.ops.cast(keras.ops.logical_and(cond, real), _F32)) / total

    return {
        "positive": frac(matches >= 0),
        "dustbin": frac(matches == -1),
        "ignored": frac(matches == -2),
    }
