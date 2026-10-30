"""Test-time corruptions for the mean shift experiments: patch occlusion and Gaussian noise.

Pure numpy on purpose: the draws are exactly reproducible, the tests are fast, and the
module imports without TensorFlow. Every arm of an experiment must score the same
corrupted images, so the random generator is keyed by ``(seed, condition)`` and nothing
else; the functions take no arm and never read the pixel values to decide a mask.

Interface contract (callers: the experiment workers, once per condition on the whole
test array):

* Inputs are ``(N, H, W, C)`` float arrays in the model input domain, which is the
  ``[0, 1]`` pixel domain in these experiments. Outputs are new arrays of the same
  shape and dtype; the input is never modified.
* The draw depends on ``N`` and on the row order, so corrupt a split in ONE call.
  Calling it on batches would give each batch the same leading random numbers.
* Failure mode: ``ValueError`` for a bad shape, fraction or sigma, ``KeyError`` for an
  unknown condition name. Nothing is silently coerced.
"""

import hashlib
from typing import Dict, Tuple

import numpy as np

#: Side of one occluded block, in pixels: the ViT patch size of both experiments.
PATCH_SIZE = 4

#: Condition name -> ``(kind, parameter)``. The name is also the generator key.
CONDITIONS: Dict[str, Tuple[str, float]] = {
    "clean": ("clean", 0.0),
    "occ25": ("occlude", 0.25),
    "occ50": ("occlude", 0.5),
    "noise0.05": ("noise", 0.05),
    "noise0.1": ("noise", 0.1),
    "noise0.2": ("noise", 0.2),
}


def _generator(seed: int, condition: str) -> np.random.Generator:
    """The generator for ``(seed, condition)``; stable across processes and Python versions."""
    digest = hashlib.sha256(condition.encode("utf-8")).digest()
    key = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([int(seed), key]))


def _patch_count(fraction: float, n_patches: int) -> int:
    """How many of ``n_patches`` a ``fraction`` occludes (16 of 64 at 0.25, 32 at 0.5)."""
    return int(round(fraction * n_patches))


# DECISION plan-2026-10-30T025213-f75ab4e9/D-005
# Noisy images are clipped to [0, 1] (RobustnessProbe does not): do NOT drop the clip, inputs must stay in the training range.
def _clip_unit(values: np.ndarray) -> np.ndarray:
    """Clip to the ``[0, 1]`` pixel domain."""
    return np.clip(values, 0.0, 1.0)


def _check_batch(x: np.ndarray) -> None:
    if x.ndim != 4:
        raise ValueError(f"expected an (N, H, W, C) batch, got shape {x.shape}")


def occlude(x: np.ndarray, fraction: float, seed: int, condition: str) -> np.ndarray:
    """Zero exactly ``round(fraction * n_patches)`` distinct patch blocks of every image.

    The image is cut into a grid of ``PATCH_SIZE`` x ``PATCH_SIZE`` blocks (8 x 8 = 64
    for a 32 x 32 image). Each image draws its own blocks, uniformly without
    replacement, so a 25 % occlusion hides 16 blocks and a 50 % one hides 32.

    Args:
        x: ``(N, H, W, C)`` images in the model input domain, ``H`` and ``W`` multiples
            of ``PATCH_SIZE``.
        fraction: Share of the blocks to hide, in ``[0, 1]``.
        seed: Experiment seed.
        condition: Condition name; together with ``seed`` it keys the draw.

    Returns:
        A new array like ``x`` with the chosen blocks set to zero.
    """
    x = np.asarray(x)
    _check_batch(x)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1], got {fraction}")
    count, height, width = len(x), x.shape[1], x.shape[2]
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(
            f"image size {height}x{width} is not a multiple of the {PATCH_SIZE}-pixel patch")
    grid_h, grid_w = height // PATCH_SIZE, width // PATCH_SIZE
    n_patches = grid_h * grid_w
    hidden = _patch_count(fraction, n_patches)

    scores = _generator(seed, condition).random((count, n_patches))
    chosen = np.argsort(scores, axis=1)[:, :hidden]
    mask = np.zeros((count, n_patches), dtype=bool)
    np.put_along_axis(mask, chosen, True, axis=1)
    pixels = mask.reshape(count, grid_h, grid_w)
    pixels = pixels.repeat(PATCH_SIZE, axis=1).repeat(PATCH_SIZE, axis=2)
    return np.where(pixels[..., None], x.dtype.type(0), x)


def add_noise(x: np.ndarray, sigma: float, seed: int, condition: str) -> np.ndarray:
    """Add ``N(0, sigma)`` in the ``[0, 1]`` pixel domain, then clip to ``[0, 1]``.

    The noise is drawn in float64 and rounded once, so a float32 and a float64 copy of
    the same images receive the same noise.

    Args:
        x: ``(N, H, W, C)`` images in ``[0, 1]``.
        sigma: Noise standard deviation in pixel units, ``>= 0``.
        seed: Experiment seed.
        condition: Condition name; together with ``seed`` it keys the draw.

    Returns:
        A new array like ``x``.

    Raises:
        ValueError: ``sigma`` is negative or not finite, or ``x`` leaves ``[0, 1]``
            (standardised images would be clipped into a different picture).
    """
    x = np.asarray(x)
    _check_batch(x)
    if not (np.isfinite(sigma) and sigma >= 0.0):
        raise ValueError(f"sigma must be finite and >= 0, got {sigma}")
    if x.min() < 0.0 or x.max() > 1.0:
        raise ValueError("add_noise expects images in the [0, 1] pixel domain")
    noise = _generator(seed, condition).normal(0.0, sigma, size=x.shape)
    return _clip_unit(x.astype(np.float64) + noise).astype(x.dtype)


def apply_condition(x: np.ndarray, condition: str, seed: int) -> np.ndarray:
    """Apply the named entry of ``CONDITIONS``; ``clean`` returns a copy of ``x``."""
    kind, parameter = CONDITIONS[condition]
    if kind == "occlude":
        return occlude(x, parameter, seed, condition)
    if kind == "noise":
        return add_noise(x, parameter, seed, condition)
    return np.array(x, copy=True)
