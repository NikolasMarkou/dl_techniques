"""Normalization for Timestral: causal per-patch mean/std scaling with asinh squash.

Each patch is normalized with the mean and std of every observed value up to the
end of that patch, so the context never sees future statistics. Forecasts come
back in the normalized (asinh) space of the last context patch and are inverted
with sinh.
"""

import functools

import keras
import numpy as np
from typing import Tuple, Optional


def compute_causal_statistics(
    data,
    observation_mask,
    minimum_scale: float,
    stabilize_with_global: bool,
    scale_factor_exponent: float = 10.0,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Causal running mean and std along last axis (Welford form, in float64).

    Returns (means, scales), both shaped like data; position t uses observed
    values at positions 0..t only. Unobserved positions get zero weight.
    Variance uses Bessel's correction; std clamped to minimum_scale after sqrt.
    """
    weights = keras.ops.cast(observation_mask, "float64")
    x = keras.ops.cast(data, "float64")
    w = weights

    # Weighted cumulative statistics
    weighted = w * x
    cum_w = keras.ops.cumsum(w, axis=-1)
    cum_x = keras.ops.cumsum(weighted, axis=-1)

    denominator = keras.ops.maximum(cum_w, 1.0)
    means = cum_x / denominator

    # Welford variance: M2 = sum(w * (x - mean)^2)
    # Use the recurrence: shifted_mean[t] = mean[t-1], shifted_mean[0] = 0
    shifted_means = keras.ops.concatenate(
        [keras.ops.zeros_like(means[..., :1]), means[..., :-1]], axis=-1
    )
    increment = (x - shifted_means) * (x - means) * w
    m2 = keras.ops.cumsum(increment, axis=-1)

    variance = m2 / keras.ops.maximum(denominator - 1.0, 1.0)
    scales = keras.ops.sqrt(variance)
    scales = keras.ops.maximum(scales, minimum_scale)

    if stabilize_with_global:
        lo = 10.0 ** (-scale_factor_exponent)
        hi = 10.0 ** scale_factor_exponent
        global_den = keras.ops.sum(weights * observation_mask, axis=-1, keepdims=True)
        global_den = keras.ops.maximum(global_den, 1.0)
        global_means = keras.ops.sum(weighted * observation_mask, axis=-1, keepdims=True) / global_den
        global_var = keras.ops.sum(
            ((x - global_means) * weights * observation_mask) ** 2,
            axis=-1, keepdims=True
        ) / global_den
        global_scale = keras.ops.sqrt(global_var + minimum_scale)
        global_scale = keras.ops.broadcast_to(global_scale, keras.ops.shape(scales))
        floor = keras.ops.maximum(minimum_scale, global_scale * lo)
        scales = keras.ops.clip(scales, floor, global_scale * hi)

    return keras.ops.cast(means, data.dtype), keras.ops.cast(scales, data.dtype)


def causal_patch_loc_scale(
    data,
    observation_mask,
    patch_size: int,
    minimum_scale: float,
    stabilize_with_global: bool,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Per-patch (loc, scale) broadcast back to every time step of the patch.

    Each patch uses causal statistics at its last time step (everything observed
    up to the end of that patch).
    """
    seq_len = keras.ops.shape(data)[-1]
    if seq_len % patch_size != 0:
        raise ValueError(f"Length {seq_len} not a multiple of patch_size {patch_size}")

    means, scales = compute_causal_statistics(
        data, observation_mask, minimum_scale, stabilize_with_global
    )

    # Unfold to patches, take last position of each patch
    # means: (..., seq_len) -> (..., n_patches, patch_size) -> (..., n_patches)
    n_patches = seq_len // patch_size
    means_reshaped = keras.ops.reshape(means, (-1, n_patches, patch_size))
    scales_reshaped = keras.ops.reshape(scales, (-1, n_patches, patch_size))

    patch_means = means_reshaped[..., -1]  # (..., n_patches)
    patch_scales = scales_reshaped[..., -1]  # (..., n_patches)

    # Repeat each patch stat patch_size times
    patch_means = keras.ops.repeat(patch_means, patch_size, axis=-1)
    patch_scales = keras.ops.repeat(patch_scales, patch_size, axis=-1)

    return patch_means, patch_scales


def _fp32_io(*dtypes):
    """(out_dtype, work_dtype): compute in at least fp32, return promoted input dtype."""
    names = [d if isinstance(d, str) else getattr(d, "name", str(d)) for d in dtypes]
    out_dtype = str(np.result_type(*names).name)
    return out_dtype, str(np.result_type(out_dtype, "float32").name)


def apply_affine_asinh_masked(
    data,
    loc,
    scale,
    observation_mask,
    use_asinh: bool,
):
    """asinh((data - loc) / scale) on observed positions; unobserved become 0."""
    out_dtype, work = _fp32_io(data.dtype, loc.dtype, scale.dtype)
    scaled = (keras.ops.cast(data, work) - keras.ops.cast(loc, work)) / keras.ops.maximum(
        keras.ops.cast(scale, work), np.finfo(work).eps
    )
    scaled = keras.ops.where(observation_mask, scaled, keras.ops.zeros_like(scaled))
    if use_asinh:
        scaled = keras.ops.where(observation_mask, keras.ops.arcsinh(scaled), scaled)
    return keras.ops.cast(scaled, out_dtype)


def invert_affine_asinh(x, loc, scale, use_asinh: bool):
    """Inverse: sinh(x) * scale + loc.

    loc/scale follow x's last-but-one axis: same rank as x (elementwise) or one
    lower (broadcast as a per-step trailing axis).
    """
    out_dtype, work = _fp32_io(x.dtype, loc.dtype, scale.dtype)
    x = keras.ops.cast(x, work)
    if use_asinh:
        x = keras.ops.sinh(x)
    if loc.ndim == x.ndim - 1:
        loc = keras.ops.expand_dims(loc, axis=-1)
        scale = keras.ops.expand_dims(scale, axis=-1)
    return keras.ops.cast(x * keras.ops.cast(scale, work) + keras.ops.cast(loc, work), out_dtype)


def clamp_nonfinite(vals):
    """Map +/-inf (sinh overflow) and NaN to finite values along last axis.

    +/-inf clamp to the max/min finite value of their row; NaN becomes 0.
    """
    finite = keras.ops.isfinite(vals)
    max_finite = keras.ops.max(keras.ops.where(finite, vals, -np.inf), axis=-1, keepdims=True)
    min_finite = keras.ops.min(keras.ops.where(finite, vals, np.inf), axis=-1, keepdims=True)
    vals = keras.ops.where(keras.ops.equal(vals, np.inf), max_finite, vals)
    vals = keras.ops.where(keras.ops.equal(vals, -np.inf), min_finite, vals)
    vals = keras.ops.where(keras.ops.isnan(vals), keras.ops.zeros_like(vals), vals)
    return vals


def denormalize_quantiles(preds, loc, scale, use_asinh: bool):
    """Invert normalization and clamp infinities; monotone quantiles stay in order."""
    return clamp_nonfinite(invert_affine_asinh(preds, loc, scale, use_asinh))


def welford_combine(
    w_a, mean_a, m2_a, w_b, mean_b, m2_b
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
    """Merge two Welford states (weight, mean, m2) element-wise."""
    w = w_a + w_b
    w_safe = keras.ops.maximum(w, np.finfo(np.float32).tiny)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + delta * delta * w_a * w_b / w_safe
    is_zero = keras.ops.equal(w, 0)
    mean = keras.ops.where(is_zero, keras.ops.zeros_like(mean), mean)
    m2 = keras.ops.where(is_zero, keras.ops.zeros_like(m2), m2)
    return w, mean, m2


def cumulative_welford(w, mean, m2):
    """Inclusive prefix scan of Welford states along dim 1 (Hillis-Steele)."""
    n = keras.ops.shape(w)[1]
    d = 1
    while d < n:
        # Pad left by d
        pad = ((0, 0), (d, 0)) + ((0, 0),) * (w.ndim - 2)
        w_l = keras.ops.pad(w[:, :n-d], pad)
        mean_l = keras.ops.pad(mean[:, :n-d], pad)
        m2_l = keras.ops.pad(m2[:, :n-d], pad)
        w, mean, m2 = welford_combine(w_l, mean_l, m2_l, w, mean, m2)
        d *= 2
    return w, mean, m2


def advancing_block_frames(
    context,
    observation_mask,
    sinh_median,
    minimum_scale: float,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Per-patch (loc, scale) of a decoded block in the advancing refine frame.

    Horizon patch j is normalized with statistics of observed context plus the
    block's medians of patches 0..j-1, each median denormalized in its own frame.
    Statistics are float64 Welford states, accumulated exactly as in training.

    Args:
        context: (R, T) block context, T multiple of patch_size.
        observation_mask: (R, T).
        sinh_median: (R, H, patch_size) block median in last context patch frame
            with asinh undone: (median - loc) / scale.
        minimum_scale: Scale floor.
    """
    r = int(keras.ops.shape(sinh_median)[0])
    n_pred = int(keras.ops.shape(sinh_median)[1])
    ps = int(keras.ops.shape(sinh_median)[2])
    sinh_median = keras.ops.cast(sinh_median, "float64")
    # Per-patch moments of context, scanned with zero-weight horizon patches
    values = keras.ops.concatenate([
        keras.ops.reshape(context, (r, -1, ps)),
        keras.ops.zeros((r, n_pred, ps), dtype=context.dtype),
    ], axis=1)
    weights = keras.ops.concatenate([
        keras.ops.reshape(observation_mask, (r, -1, ps)),
        keras.ops.zeros((r, n_pred, ps), dtype=observation_mask.dtype),
    ], axis=1)

    # Convert to float64 for accumulation
    values = keras.ops.cast(values, "float64")
    weights = keras.ops.cast(weights, "float64")

    cnt = keras.ops.sum(weights, axis=-1)
    mean = keras.ops.sum(values * weights, axis=-1) / keras.ops.maximum(cnt, 1.0)
    m2 = keras.ops.sum(weights * (values - keras.ops.expand_dims(mean, -1)) ** 2, axis=-1)

    is_zero = keras.ops.equal(cnt, 0)
    mean = keras.ops.where(is_zero, keras.ops.zeros_like(mean), mean)
    m2 = keras.ops.where(is_zero, keras.ops.zeros_like(m2), m2)

    # Cumulative Welford scan
    cnt, mean, m2 = cumulative_welford(cnt, mean, m2)

    # We need the state BEFORE each horizon patch (i.e., at index -n_pred-1)
    # cum_welford gives inclusive scan; we want the state after context patches
    cnt = cnt[:, -n_pred - 1]
    mean = mean[:, -n_pred - 1]
    m2 = m2[:, -n_pred - 1]

    offset = mean  # medians clamped relative to context mean
    clip = keras.ops.cast(np.float32(np.finfo(np.float32).max) ** 0.5 / 1e3, "float64")

    locs, scales = [], []
    for h in range(n_pred):
        var = m2 / keras.ops.maximum(cnt - 1.0, 1.0)
        var = keras.ops.maximum(var, 0.0)
        scale = keras.ops.sqrt(var)
        scale = keras.ops.maximum(scale, minimum_scale)

        locs.append(keras.ops.cast(mean, "float32"))
        scales.append(keras.ops.cast(scale, "float32"))

        # Fold in this patch's median (denormalized in its own frame)
        folded = sinh_median[:, h] * keras.ops.expand_dims(scale, -1) + keras.ops.expand_dims(mean, -1)
        folded = folded - keras.ops.expand_dims(offset, -1)
        folded = keras.ops.clip(folded, -clip, clip)
        folded = folded + keras.ops.expand_dims(offset, -1)

        b_mean = keras.ops.mean(folded, axis=-1)
        b_m2 = keras.ops.sum((folded - keras.ops.expand_dims(b_mean, -1)) ** 2, axis=-1)
        cnt, mean, m2 = welford_combine(
            cnt, mean, m2,
            keras.ops.full_like(cnt, ps), b_mean, b_m2
        )

    return keras.ops.stack(locs, axis=1), keras.ops.stack(scales, axis=1)


def instance_norm_stats(values, observation_mask, minimum_scale: float):
    """Non-causal per-row mean/std over observed positions, for known-future covariates.

    Rows with all observed values in {0, 1} (indicator covariates) keep loc=0, scale=1.
    Returns (loc, scale, is_binary), each (R, 1).
    """
    orig_dtype = values.dtype
    stat_dtype = str(np.result_type(getattr(orig_dtype, "name", str(orig_dtype)), "float32").name)
    x = keras.ops.cast(values, stat_dtype)
    mask_f = keras.ops.cast(observation_mask, stat_dtype)

    n = keras.ops.sum(mask_f, axis=-1, keepdims=True)
    n = keras.ops.maximum(n, 1.0)
    loc = keras.ops.sum(x * mask_f, axis=-1, keepdims=True) / n
    var = keras.ops.sum(((x - loc) * mask_f) ** 2, axis=-1, keepdims=True) / n
    scale = keras.ops.sqrt(var)
    scale = keras.ops.maximum(scale, minimum_scale)

    is_zero_one = keras.ops.all(
        keras.ops.logical_or(
            keras.ops.logical_or(keras.ops.equal(values, 0.0), keras.ops.equal(values, 1.0)),
            keras.ops.logical_not(observation_mask)
        ),
        axis=-1, keepdims=True
    )
    is_binary = keras.ops.logical_and(is_zero_one, keras.ops.any(observation_mask, axis=-1, keepdims=True))

    loc = keras.ops.where(is_binary, keras.ops.zeros_like(loc), loc)
    scale = keras.ops.where(is_binary, keras.ops.ones_like(scale), scale)

    return keras.ops.cast(loc, orig_dtype), keras.ops.cast(scale, orig_dtype), is_binary