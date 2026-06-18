"""Random homography sampling and image/point warping for the data pipeline.

This module provides the geometric primitives used to build SuperPoint-style
self-supervised training pairs (random homographic adaptation): a random 3x3
homography sampler, a batched image warp, and a point warp for mapping
ground-truth keypoints into the warped frame.

**Scope**: these are DATA-PIPELINE utilities, NOT model forward-path code. Raw
TensorFlow ops (``tf.raw_ops.ImageProjectiveTransformV3``) and NumPy are
therefore acceptable here. Do NOT import these into a Keras ``call()``.

Coordinate convention
---------------------
All homographies operate on PIXEL coordinates with the origin at the TOP-LEFT
corner of the image: ``x`` increases to the right (columns), ``y`` increases
downward (rows). A point ``p = (x, y)`` is mapped FORWARD (input -> warped
output) by::

    [x', y', w']^T = H @ [x, y, 1]^T
    p_out = (x' / w', y' / w')

``warp_points`` implements exactly this forward map. ``warp_image`` produces an
output image such that ``output[p_out] == input[p]`` for every pixel ``p`` (i.e.
content located at ``p`` in the input lands at ``H @ p`` in the output). The two
functions are guaranteed consistent by construction: ``warp_image`` internally
inverts ``H`` because ``tf.raw_ops.ImageProjectiveTransformV3`` expects the
OUTPUT->INPUT (inverse) transform.
"""

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# SuperPoint-like default magnitude ranges
# ---------------------------------------------------------------------

# Rotation in radians (+- 25 degrees).
_DEFAULT_ROTATION: float = np.deg2rad(25.0)
# Per-axis scale multiplier sampled uniformly in [lo, hi].
_DEFAULT_SCALE: Tuple[float, float] = (0.8, 1.2)
# Perspective coefficient magnitude (applied to the bottom row of H, in
# normalized-by-image-size units). Small => mild perspective.
_DEFAULT_PERSPECTIVE: float = 0.0008
# Translation as a fraction of (W, H).
_DEFAULT_TRANSLATION: float = 0.1
# Shear in radians (disabled by default).
_DEFAULT_SHEAR: float = 0.0

ArrayLike = Union[np.ndarray, "tf.Tensor"]

# ---------------------------------------------------------------------


def sample_homography(
    image_size: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    *,
    seed: Optional[int] = None,
    rotation: float = _DEFAULT_ROTATION,
    scale: Tuple[float, float] = _DEFAULT_SCALE,
    perspective: float = _DEFAULT_PERSPECTIVE,
    translation: float = _DEFAULT_TRANSLATION,
    shear: float = _DEFAULT_SHEAR,
) -> np.ndarray:
    """Sample a random 3x3 homography in pixel coordinates.

    The returned matrix maps INPUT pixel coordinates to WARPED-OUTPUT pixel
    coordinates (forward direction, see module docstring). It is composed, about
    the image center, from random rotation, anisotropic scale, shear,
    perspective, and translation within the supplied magnitude ranges. Defaults
    are SuperPoint-like (mild perspective, +-25 deg rotation, 0.8-1.2 scale).

    The composition is::

        H = T_center @ T_translate @ P @ S_scale @ Sh_shear @ R_rotate @ T_-center

    where ``T_center`` / ``T_-center`` move the origin to/from the image center
    so rotation/scale/shear pivot about the center rather than the top-left.

    Args:
        image_size: ``(H, W)`` image size in pixels. Used to center the
            transform and to scale translation/perspective into pixel units.
        rng: Optional ``numpy.random.Generator`` for deterministic sampling. If
            ``None`` and ``seed`` is given, ``np.random.default_rng(seed)`` is
            used; otherwise a fresh default generator is created.
        seed: Convenience seed used only when ``rng`` is ``None``.
        rotation: Maximum absolute rotation in RADIANS; the angle is sampled
            uniformly in ``[-rotation, rotation]``.
        scale: ``(lo, hi)`` range for the per-axis uniform scale factor.
        perspective: Maximum absolute perspective coefficient (normalized by
            image size). Sampled uniformly in ``[-perspective, perspective]``
            per bottom-row entry. Set to ``0.0`` to disable.
        translation: Maximum absolute translation as a fraction of ``(W, H)``;
            sampled uniformly per axis in ``[-translation, translation]``.
        shear: Maximum absolute shear angle in RADIANS; sampled uniformly in
            ``[-shear, shear]``. ``0.0`` disables shear.

    Returns:
        A ``(3, 3)`` ``float32`` NumPy array with element ``[2, 2]`` normalized
        to ``1.0``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    height, width = float(image_size[0]), float(image_size[1])

    # --- rotation about origin ---
    theta = rng.uniform(-rotation, rotation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    r_mat = np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # --- anisotropic scale ---
    sx = rng.uniform(scale[0], scale[1])
    sy = rng.uniform(scale[0], scale[1])
    s_mat = np.array(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # --- shear ---
    sh = rng.uniform(-shear, shear)
    sh_mat = np.array(
        [[1.0, np.tan(sh), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # --- translation (fraction of image size -> pixels) ---
    tx = rng.uniform(-translation, translation) * width
    ty = rng.uniform(-translation, translation) * height
    t_mat = np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # --- perspective (bottom row), normalized by image size ---
    g = rng.uniform(-perspective, perspective) / width
    h = rng.uniform(-perspective, perspective) / height
    p_mat = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [g, h, 1.0]], dtype=np.float64
    )

    # --- center pivot so rotation/scale/shear act about the image center ---
    cx, cy = width / 2.0, height / 2.0
    to_center = np.array(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    from_center = np.array(
        [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    h_mat = (
        from_center
        @ t_mat
        @ p_mat
        @ s_mat
        @ sh_mat
        @ r_mat
        @ to_center
    )

    # Normalize so element [2, 2] == 1.0 (well-defined since the pivot terms
    # keep it non-zero for the default magnitude ranges).
    if abs(h_mat[2, 2]) > 1e-12:
        h_mat = h_mat / h_mat[2, 2]

    return h_mat.astype(np.float32)


# ---------------------------------------------------------------------


def _homography_to_flat_inverse(h_mat: np.ndarray) -> np.ndarray:
    """Convert a forward 3x3 homography to the 8-param OUTPUT->INPUT transform.

    ``tf.raw_ops.ImageProjectiveTransformV3`` consumes the inverse (output ->
    input) transform as a flat vector ``[a0, a1, a2, a3, a4, a5, a6, a7]`` with
    the implicit ``a8 == 1``. Given a forward (input -> output) homography
    ``H``, the required transform is ``inv(H)`` normalized so that its ``[2, 2]``
    entry equals 1, with the last element dropped.

    Args:
        h_mat: ``(3, 3)`` forward homography (input -> output pixel coords).

    Returns:
        ``(8,)`` ``float32`` vector for ``ImageProjectiveTransformV3``.
    """
    inv = np.linalg.inv(h_mat.astype(np.float64))
    if abs(inv[2, 2]) > 1e-12:
        inv = inv / inv[2, 2]
    return inv.reshape(-1)[:8].astype(np.float32)


def warp_image(
    image: ArrayLike,
    homography: ArrayLike,
    output_size: Optional[Tuple[int, int]] = None,
    interpolation: str = "bilinear",
) -> tf.Tensor:
    """Warp an image (or batch) by a forward homography.

    Content located at input pixel ``p`` is moved to ``H @ p`` in the output,
    consistent with :func:`warp_points`. Internally the forward homography is
    inverted because the underlying op expects an OUTPUT->INPUT transform.

    Args:
        image: ``(H, W, C)`` single image or ``(B, H, W, C)`` batch. Any
            array-like convertible to a float tensor.
        homography: ``(3, 3)`` forward homography, or ``(B, 3, 3)`` batch of
            them matching the image batch.
        output_size: Optional ``(H_out, W_out)``. Defaults to the input spatial
            size.
        interpolation: ``"bilinear"`` or ``"nearest"``.

    Returns:
        Warped image tensor with the same rank as the input (``(H, W, C)`` or
        ``(B, H, W, C)``), ``float32``.
    """
    img = tf.convert_to_tensor(image, dtype=tf.float32)

    squeeze_batch = False
    if len(img.shape) == 3:
        img = img[tf.newaxis, ...]
        squeeze_batch = True
    elif len(img.shape) != 4:
        raise ValueError(
            f"warp_image expects rank-3 (H,W,C) or rank-4 (B,H,W,C) input, "
            f"got shape {tuple(img.shape)}."
        )

    batch = int(img.shape[0])

    h_np = np.asarray(homography, dtype=np.float64)
    if h_np.shape == (3, 3):
        flat = np.stack([_homography_to_flat_inverse(h_np)] * batch, axis=0)
    elif h_np.shape == (batch, 3, 3):
        flat = np.stack(
            [_homography_to_flat_inverse(h_np[i]) for i in range(batch)], axis=0
        )
    else:
        raise ValueError(
            f"homography must be (3,3) or (B,3,3) with B={batch}, "
            f"got shape {h_np.shape}."
        )

    if output_size is None:
        out_h, out_w = int(img.shape[1]), int(img.shape[2])
    else:
        out_h, out_w = int(output_size[0]), int(output_size[1])

    interp = interpolation.upper()
    if interp not in ("BILINEAR", "NEAREST"):
        raise ValueError(
            f"interpolation must be 'bilinear' or 'nearest', got "
            f"'{interpolation}'."
        )

    transforms = tf.convert_to_tensor(flat, dtype=tf.float32)
    warped = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transforms,
        output_shape=tf.constant([out_h, out_w], dtype=tf.int32),
        fill_value=0.0,
        interpolation=interp,
        fill_mode="CONSTANT",
    )

    if squeeze_batch:
        warped = warped[0]
    return warped


# ---------------------------------------------------------------------


def warp_points(points: ArrayLike, homography: ArrayLike) -> np.ndarray:
    """Map ``(x, y)`` points FORWARD (input -> warped output) by ``H``.

    Applies the homogeneous transform ``[x', y', w'] = H @ [x, y, 1]`` then
    divides by ``w'``. This is the exact inverse-direction partner of
    :func:`warp_image`: a feature at input pixel ``p`` warps to ``warp_points(p,
    H)`` in the output produced by ``warp_image(img, H)``.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` pixel coordinates (also accepts a
            single ``(2,)`` point).
        homography: ``(3, 3)`` forward homography.

    Returns:
        ``(N, 2)`` ``float32`` NumPy array of warped ``(x, y)`` coordinates.
    """
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts[np.newaxis, :]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points must be (N,2) or (2,), got shape {pts.shape}."
        )

    h_mat = np.asarray(homography, dtype=np.float64)
    if h_mat.shape != (3, 3):
        raise ValueError(f"homography must be (3,3), got {h_mat.shape}.")

    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=1)  # (N, 3)
    warped = homo @ h_mat.T  # (N, 3)

    w = warped[:, 2:3]
    # Guard against division by ~0 for points on the line at infinity.
    w = np.where(np.abs(w) < 1e-12, np.sign(w) * 1e-12 + 1e-12, w)
    out = warped[:, :2] / w

    if single:
        out = out[0]
    return out.astype(np.float32)
