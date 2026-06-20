"""Miyasawa / Tweedie compliance tooling for per-pixel multiplicative Gaussian noise.

This module is the library-level home for the multiplicative-noise empirical-Bayes
machinery used by the ConvUNeXt bias-free denoiser ecosystem. It bundles three things:

1. ``apply_multiplicative_gaussian`` — the pure-TensorFlow noise primitive
   ``y = x * (1 + N(0, 1) * sigma)`` (equivalently ``y = x * n, n ~ N(1, sigma^2)``).
   It is written to be ``tf.data.Dataset.map`` graph-safe (uses ``tf.random.normal``,
   not ``keras.ops.random.*`` which is finicky inside ``tf.data``). The trainer and the
   visualization callback reuse this single primitive.

2. The Monte-Carlo empirical-Bayes checkers (``mc_posterior_mean`` + ``relation_A`` /
   ``relation_B`` + ``rel_rmse``). For a *signed* (``[-1, +1]``-style) prior there is **no**
   clean linear-domain ``D(y) - y = sigma^2 * grad log p`` identity. Instead two relations
   hold (validated numerically, see ``research/2026_miyasawa_multiplicative_noise.md`` /
   ``findings/multiplicative-tweedie-derivation.md``):

       (A) exact, any sigma:
           E[x|y] = y + sigma^2 * d/dy[ E[x^2|y] * p(y) ] / p(y)

       (B) small-sigma approximation:
           D(y) - y ~= 2*sigma^2*y + sigma^2*y^2 * d/dy log p(y)

   These are the **hard** compliance gates of the accompanying pytest.

3. ``sure_divergence_consistency`` — a generalized-SURE / Hutchinson divergence
   diagnostic for a trained denoiser, estimable from noisy data without clean
   references. Its trustworthiness is gated on reproducing the *additive* closed-form
   SURE risk on a known linear toy (the Pre-Mortem STOP-IF). The accompanying test runs
   that additive self-check; if it cannot be made consistent the function is demoted to
   *diagnostic-only* (still returns finite numbers, but is not a hard compliance gate)
   and the MC relation (A)/(B) becomes the sole hard assertion. See decisions.md D-002
   if such a demotion was recorded.

Layering: this module is part of ``dl_techniques`` and MUST NOT import from ``src/train``
(asserted by construction — there is no such import here). It depends only on numpy,
tensorflow, and ``dl_techniques.utils.logger``.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# 1. Noise primitive (tf.data-graph-safe; reused by trainer + viz)
# ---------------------------------------------------------------------


def apply_multiplicative_gaussian(
        x: tf.Tensor,
        sigma: Union[float, tf.Tensor, tf.Variable],
) -> tf.Tensor:
    """Apply per-pixel multiplicative Gaussian noise ``y = x * (1 + N(0,1) * sigma)``.

    This is the multiplicative analog of additive AWGN: each element of ``x`` is scaled
    by an independent draw of ``n ~ N(1, sigma^2)``, so ``y | x ~ N(x, sigma^2 * x^2)``
    (signal-dependent variance). The function is written to be safe inside a
    ``tf.data.Dataset.map`` graph: it uses ``tf.random.normal`` (the same op the additive
    training path uses) rather than ``keras.ops.random.*``, and derives the noise shape
    dynamically from ``tf.shape(x)`` so it works for any rank / unknown batch dim.

    Args:
        x: Input tensor of any shape (e.g. a clean image patch ``[H, W, C]`` or a batch
            ``[B, H, W, C]``).
        sigma: Noise standard deviation. May be a python float, a scalar tf.Tensor, or a
            scalar ``tf.Variable`` (e.g. a curriculum-widened variable read inside
            ``tf.data.map``). It is cast to ``x.dtype``.

    Returns:
        The noisy tensor ``y`` with the same shape and dtype as ``x``. No clipping is
        applied here (callers clip to their domain, e.g. ``[-1, +1]``, separately).
    """
    sigma = tf.cast(sigma, x.dtype)
    noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
    return x * (1.0 + noise * sigma)


# ---------------------------------------------------------------------
# 2. Monte-Carlo posterior moments + relation (A)/(B) evaluators
#    (offline checker; numpy is fine here — NOT in the training graph)
# ---------------------------------------------------------------------

# Default count threshold mirrors the reference (/tmp/mult_tweedie_check.py) at 8M
# samples (cnt > 2000). It is scaled with n_samples so a smaller MC run keeps a
# comparable relative population floor.
_REFERENCE_N_SAMPLES = 8_000_000
_REFERENCE_COUNT_THRESHOLD = 2000


def mc_posterior_mean(
        prior_samples: np.ndarray,
        sigma: float,
        n_bins: int = 600,
        n_samples: Optional[int] = None,
        count_threshold: Optional[int] = None,
        trim_percentile: float = 0.5,
        seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Estimate posterior moments ``E[x|y]``, ``E[x^2|y]`` and the marginal ``p(y)`` by binning.

    Draws ``n_samples`` clean values ``x`` from ``prior_samples`` (with replacement),
    forms ``y = x * (1 + N(0,1) * sigma)``, then bins over ``y`` to build non-parametric
    posterior moments. This mirrors ``/tmp/mult_tweedie_check.py`` exactly: percentile
    trim of the y-range, ``np.digitize`` binning, and a population ``count_threshold``
    mask. All randomness is seeded for reproducibility.

    Args:
        prior_samples: 1-D array of samples from the prior ``p(x)``.
        sigma: Multiplicative noise std (``n ~ N(1, sigma^2)``).
        n_bins: Number of y-bins. Defaults to 600 (reference value).
        n_samples: Monte-Carlo sample count. Defaults to ``len(prior_samples)`` if the
            prior array is already large, else falls back to the reference 8M. Pass a
            smaller value (e.g. 1-2M) to keep a pytest fast.
        count_threshold: Minimum bin population to be considered "well populated".
            Defaults to the reference ``2000`` scaled by ``n_samples / 8M`` (min 1).
        trim_percentile: Lower/upper percentile to trim the y-range (default 0.5, i.e.
            keep [0.5, 99.5]).
        seed: RNG seed.

    Returns:
        Dict with numpy arrays (all length ``n_bins`` unless noted):
            ``ctr``   — bin centers (the y-grid).
            ``Ex``    — ``E[x|y]``.
            ``Ex2``   — ``E[x^2|y]``.
            ``p``     — marginal density ``p(y)``.
            ``mask``  — boolean populated-bin mask (``cnt > count_threshold``).
            ``cnt``   — raw per-bin counts.
            ``dy``    — scalar bin width (numpy float).
            ``sigma`` — scalar sigma (numpy float).
    """
    prior_samples = np.asarray(prior_samples, dtype=np.float64).ravel()
    if prior_samples.size == 0:
        raise ValueError("prior_samples must be non-empty")

    if n_samples is None:
        n_samples = max(prior_samples.size, _REFERENCE_N_SAMPLES)
    if count_threshold is None:
        scaled = int(round(_REFERENCE_COUNT_THRESHOLD * n_samples / _REFERENCE_N_SAMPLES))
        count_threshold = max(1, scaled)

    rng = np.random.default_rng(seed)

    # Draw clean x from the provided prior samples (with replacement), then corrupt.
    x = rng.choice(prior_samples, size=n_samples, replace=True)
    n = rng.normal(1.0, sigma, n_samples)
    y = x * n

    lo, hi = np.percentile(y, [trim_percentile, 100.0 - trim_percentile])
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)

    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    sx = np.bincount(idx, weights=x, minlength=n_bins)
    sx2 = np.bincount(idx, weights=x * x, minlength=n_bins)

    ctr = 0.5 * (edges[:-1] + edges[1:])
    dy = ctr[1] - ctr[0]

    Ex = np.divide(sx, cnt, out=np.zeros_like(sx), where=cnt > 0)
    Ex2 = np.divide(sx2, cnt, out=np.zeros_like(sx2), where=cnt > 0)
    p = cnt / (n_samples * dy)
    mask = cnt > count_threshold

    logger.info(
        "mc_posterior_mean: sigma=%.4f n_samples=%d n_bins=%d populated=%d (thr=%d)",
        sigma, n_samples, n_bins, int(mask.sum()), count_threshold,
    )

    return {
        "ctr": ctr,
        "Ex": Ex,
        "Ex2": Ex2,
        "p": p,
        "mask": mask,
        "cnt": cnt,
        "dy": float(dy),
        "sigma": float(sigma),
    }


def relation_A(mc: Dict[str, np.ndarray]) -> np.ndarray:
    """RHS of exact relation (A): ``y + sigma^2 * d/dy[E[x^2|y]*p(y)] / p(y)``.

    Args:
        mc: The dict returned by :func:`mc_posterior_mean`.

    Returns:
        Array (length ``n_bins``) of the relation-(A) RHS evaluated on the y-grid; this
        is the predicted ``E[x|y]`` and should be compared against ``mc["Ex"]``.
    """
    ctr, Ex2, p = mc["ctr"], mc["Ex2"], mc["p"]
    dy = mc["dy"]
    sigma = mc["sigma"]
    g = Ex2 * p
    dg = np.gradient(g, dy)
    return ctr + sigma ** 2 * np.divide(dg, p, out=np.zeros_like(dg), where=p > 0)


def relation_B(mc: Dict[str, np.ndarray]) -> np.ndarray:
    """RHS of small-sigma relation (B) as a predicted ``E[x|y]``.

    The derivation gives ``D(y) - y ~= 2*sigma^2*y + sigma^2*y^2 * dlog p/dy``; adding
    ``y`` yields the predicted posterior mean ``D(y)`` returned here, comparable against
    ``mc["Ex"]`` on the same footing as :func:`relation_A`.

    Args:
        mc: The dict returned by :func:`mc_posterior_mean`.

    Returns:
        Array (length ``n_bins``) of the relation-(B) predicted ``E[x|y]``.
    """
    ctr, p = mc["ctr"], mc["p"]
    dy = mc["dy"]
    sigma = mc["sigma"]
    dlogp = np.gradient(np.log(np.where(p > 0, p, 1e-12)), dy)
    return ctr + 2.0 * sigma ** 2 * ctr + sigma ** 2 * ctr ** 2 * dlogp


def rel_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray, edge_trim: int = 5) -> float:
    """Relative RMSE of ``a`` vs ``b`` over masked bins, trimming ``edge_trim`` bins each side.

    Mirrors the reference comparison: ``sqrt(mean((a-b)^2)) / (sqrt(mean(b^2)) + 1e-12)``
    over the populated bins, with the first/last ``edge_trim`` bins excluded (where
    ``np.gradient`` is noisy).

    Args:
        a: Predicted array (e.g. relation-(A) RHS).
        b: Reference array (e.g. ``E[x|y]``).
        mask: Populated-bin boolean mask.
        edge_trim: Number of bins to drop from each end. Defaults to 5.

    Returns:
        The relative RMSE as a python float.
    """
    m = np.asarray(mask).copy()
    if edge_trim > 0:
        m[:edge_trim] = False
        m[-edge_trim:] = False
    if not np.any(m):
        raise ValueError("rel_rmse: no bins left after masking/trimming")
    num = np.sqrt(np.mean((a[m] - b[m]) ** 2))
    den = np.sqrt(np.mean(b[m] ** 2)) + 1e-12
    return float(num / den)


# ---------------------------------------------------------------------
# 3. Generalized-SURE / Hutchinson divergence consistency
# ---------------------------------------------------------------------


def hutchinson_divergence(
        denoiser: Callable[[tf.Tensor], tf.Tensor],
        y: tf.Tensor,
        n_hutchinson: int = 8,
        eps: float = 1e-3,
        rademacher: bool = True,
        seed: int = 0,
) -> float:
    """Estimate the divergence ``div_y D(y)`` via random-projection finite differences.

    Uses the Hutchinson estimator with a finite-difference Jacobian-vector product:

        div(D) ~= E_v[ v . (D(y + eps*v) - D(y)) / eps ]

    where ``v`` are i.i.d. Rademacher (``+/-1``) or standard-normal probes. The estimate
    is the *total* divergence summed over all elements of the batch (the trace of the
    Jacobian of the flattened map), averaged over ``n_hutchinson`` probes.

    Args:
        denoiser: Callable mapping a noisy tensor to a denoised tensor of the same shape.
        y: Noisy input tensor (any shape).
        n_hutchinson: Number of random probes to average. Defaults to 8.
        eps: Finite-difference step. Defaults to 1e-3.
        rademacher: If True use Rademacher probes, else standard normal. Defaults to True.
        seed: RNG seed.

    Returns:
        Scalar divergence estimate (python float).
    """
    y = tf.convert_to_tensor(y)
    d0 = denoiser(y)
    acc = 0.0
    for i in range(n_hutchinson):
        if rademacher:
            v = tf.cast(
                tf.where(
                    tf.random.stateless_uniform(tf.shape(y), seed=[seed, i]) < 0.5,
                    -1.0, 1.0,
                ),
                y.dtype,
            )
        else:
            v = tf.random.stateless_normal(tf.shape(y), seed=[seed, i], dtype=y.dtype)
        d1 = denoiser(y + eps * v)
        jvp = (d1 - d0) / eps
        acc += float(tf.reduce_sum(v * jvp).numpy())
    return acc / float(n_hutchinson)


def additive_sure_risk(
        denoiser: Callable[[tf.Tensor], tf.Tensor],
        y: tf.Tensor,
        sigma: float,
        n_hutchinson: int = 8,
        eps: float = 1e-3,
        seed: int = 0,
) -> Dict[str, float]:
    """Stein's Unbiased Risk Estimate (SURE) for the **additive** Gaussian model.

    For ``y = x + N(0, sigma^2 I)``, SURE estimates ``E||D(y) - x||^2`` from noisy data
    alone as ``||D(y) - y||^2 + 2*sigma^2*div(D) - sigma^2*N`` where ``N`` is the number
    of elements. This is the closed-form anchor used by the Pre-Mortem self-check: a
    linear toy denoiser ``D(y) = a*y`` has analytic divergence ``a*N``, so its true MSE
    ``E||a*y - x||^2`` can be compared against this estimate within MC tolerance to
    validate the divergence estimator's scale.

    Args:
        denoiser: Callable noisy -> denoised.
        y: Noisy batch tensor.
        sigma: Additive noise std.
        n_hutchinson: Probes for the divergence estimate.
        eps: Finite-difference step.
        seed: RNG seed.

    Returns:
        Dict with ``divergence``, ``residual_sq`` (``||D(y)-y||^2``), ``n_elements``,
        and ``sure_risk`` (the SURE risk estimate).
    """
    y = tf.convert_to_tensor(y)
    n_elements = float(tf.size(y).numpy())
    d = denoiser(y)
    residual_sq = float(tf.reduce_sum(tf.square(d - y)).numpy())
    div = hutchinson_divergence(
        denoiser, y, n_hutchinson=n_hutchinson, eps=eps, rademacher=True, seed=seed
    )
    sure_risk = residual_sq + 2.0 * sigma ** 2 * div - sigma ** 2 * n_elements
    return {
        "divergence": div,
        "residual_sq": residual_sq,
        "n_elements": n_elements,
        "sure_risk": sure_risk,
    }


def sure_divergence_consistency(
        denoiser: Callable[[tf.Tensor], tf.Tensor],
        noisy_batch: tf.Tensor,
        sigma: float,
        n_hutchinson: int = 8,
        eps: float = 1e-3,
        seed: int = 0,
) -> Dict[str, float]:
    """Generalized-SURE divergence-consistency diagnostic for the multiplicative model.

    For per-pixel multiplicative Gaussian noise (``y | x ~ N(x, sigma^2 x^2)``) the data
    covariance is signal-dependent (``Sigma(y) = diag(sigma^2 y^2)`` to leading order).
    The generalized-SURE risk estimate (Eldar; Raphan-Simoncelli) replaces the constant
    ``sigma^2`` weighting of the divergence with the per-element variance, giving a
    residual-consistency scalar estimable from noisy data + the denoiser alone (no clean
    references):

        gsure_residual = ||D(y) - y||^2
                         + 2 * sum_i sigma^2 * y_i^2 * (dD_i/dy_i)
                         - sigma^2 * sum_i y_i^2

    We estimate the variance-weighted divergence ``sum_i sigma^2 y_i^2 (dD_i/dy_i)`` with
    a Hutchinson probe whose components are pre-scaled by ``sigma * |y|`` so that
    ``E[v_i v_j] = sigma^2 y_i^2 delta_ij``.

    NOTE on trust: the unweighted Hutchinson divergence estimator underlying this is
    validated against the additive closed form (:func:`additive_sure_risk`) in the
    accompanying pytest (the Pre-Mortem STOP-IF). If that self-check could not be made
    consistent, this function is treated as **diagnostic-only** (it still returns finite
    numbers but is not a hard compliance gate) and the MC relation (A)/(B) test is the
    sole hard gate — see decisions.md D-002.

    Args:
        denoiser: Callable mapping the noisy batch to a denoised batch of the same shape.
        noisy_batch: Noisy input tensor (e.g. ``[B, H, W, C]``).
        sigma: Multiplicative noise std.
        n_hutchinson: Number of random probes. Defaults to 8.
        eps: Finite-difference step for the JVP. Defaults to 1e-3.
        seed: RNG seed.

    Returns:
        Dict with python floats:
            ``divergence``         — plain (unweighted) Hutchinson divergence of D.
            ``weighted_divergence``— ``sum_i sigma^2 y_i^2 (dD_i/dy_i)`` estimate.
            ``residual_sq``        — ``||D(y) - y||^2``.
            ``n_elements``         — element count.
            ``gsure_residual``     — the generalized-SURE residual-consistency scalar.
    """
    y = tf.convert_to_tensor(noisy_batch)
    n_elements = float(tf.size(y).numpy())
    d0 = denoiser(y)
    residual_sq = float(tf.reduce_sum(tf.square(d0 - y)).numpy())

    # Plain divergence (reuses the validated estimator).
    div = hutchinson_divergence(
        denoiser, y, n_hutchinson=n_hutchinson, eps=eps, rademacher=True, seed=seed
    )

    # Variance-weighted divergence: probe v_i = sigma * |y_i| * r_i, r_i Rademacher,
    # so v . (D(y+eps v) - D(y))/eps  estimates  sum_i sigma^2 y_i^2 (dD_i/dy_i).
    scale = tf.cast(sigma, y.dtype) * tf.abs(y)
    acc = 0.0
    for i in range(n_hutchinson):
        r = tf.cast(
            tf.where(
                tf.random.stateless_uniform(tf.shape(y), seed=[seed + 101, i]) < 0.5,
                -1.0, 1.0,
            ),
            y.dtype,
        )
        v = scale * r
        d1 = denoiser(y + eps * v)
        jvp = (d1 - d0) / eps
        acc += float(tf.reduce_sum(v * jvp).numpy())
    weighted_div = acc / float(n_hutchinson)

    y2_sum = float(tf.reduce_sum(tf.square(y)).numpy())
    gsure_residual = (
        residual_sq + 2.0 * weighted_div - sigma ** 2 * y2_sum
    )

    logger.info(
        "sure_divergence_consistency: div=%.4g weighted_div=%.4g residual_sq=%.4g "
        "gsure_residual=%.4g (N=%d)",
        div, weighted_div, residual_sq, gsure_residual, int(n_elements),
    )

    return {
        "divergence": div,
        "weighted_divergence": weighted_div,
        "residual_sq": residual_sq,
        "n_elements": n_elements,
        "gsure_residual": gsure_residual,
    }


# ---------------------------------------------------------------------
# Synthetic prior helper (shared by the reference + the pytest)
# ---------------------------------------------------------------------


def signed_mixture_prior(n: int, seed: int = 0) -> np.ndarray:
    """Draw ``n`` samples from the signed 3-component Gaussian mixture prior.

    This is the exact prior from ``/tmp/mult_tweedie_check.py`` (normalized-image-like,
    signed, supported roughly in ``[-1, +1]``): a mixture of ``N(-0.5, 0.15)``,
    ``N(0.2, 0.1)`` and ``N(0.6, 0.2)`` with equal weights.

    Args:
        n: Number of samples.
        seed: RNG seed.

    Returns:
        1-D numpy array of ``n`` prior samples.
    """
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 3, n)
    return np.where(
        comp == 0, rng.normal(-0.5, 0.15, n),
        np.where(comp == 1, rng.normal(0.2, 0.1, n), rng.normal(0.6, 0.2, n)),
    )


# ---------------------------------------------------------------------
# 4. Optional checkpoint diagnostic (CLI entry; NOT exercised by pytest)
# ---------------------------------------------------------------------


def run_checkpoint_diagnostic(
        checkpoint_path: Optional[str] = None,
        sigma: float = 0.15,
        batch: int = 4,
        spatial: int = 64,
        channels: int = 3,
        n_hutchinson: int = 8,
        eps: float = 1e-3,
        seed: int = 0,
) -> Dict[str, Any]:
    """Load a real ``.keras`` denoiser (or build a tiny one) and print the SURE diagnostic.

    This is the heavy "check a real checkpoint" path the plan keeps OUT of the fast
    pytest. It loads a trained denoiser, synthesizes a noisy batch with
    :func:`apply_multiplicative_gaussian` (no dataset dependency — random-uniform clean
    input in the ``[-1, +1]`` domain), and reports the generalized-SURE
    divergence-consistency diagnostic via :func:`sure_divergence_consistency`. The
    diagnostic needs no clean references.

    Args:
        checkpoint_path: Path to a saved ``.keras`` denoiser. If ``None`` or the file is
            missing, a tiny randomly-initialized ``create_convunext_denoiser`` is built so
            the diagnostic still runs end-to-end (smoke fallback).
        sigma: Multiplicative noise std for the synthetic noisy batch.
        batch: Batch size of the synthetic input.
        spatial: Spatial size; the synthetic input is ``[batch, spatial, spatial, channels]``.
            Ignored when a checkpoint with a fixed input shape is loaded.
        channels: Channel count for the fallback tiny model / synthetic input.
        n_hutchinson: Hutchinson probes for the divergence estimate.
        eps: Finite-difference step for the JVP.
        seed: RNG seed.

    Returns:
        The diagnostic dict from :func:`sure_divergence_consistency`.
    """
    import os
    import keras

    # Importing these modules registers the custom Keras objects (Gabor stem
    # initializer + ConvUNeXt denoiser layers) needed to deserialize a checkpoint.
    # Imported lazily so library import of this module stays light and src/train-free.
    import dl_techniques.initializers.gabor_filters_initializer  # noqa: F401
    from dl_techniques.models.bias_free_denoisers.bfconvunext import (
        create_convunext_denoiser,
    )

    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info("loading denoiser checkpoint: %s", checkpoint_path)
        model = keras.models.load_model(checkpoint_path, compile=False)
        in_shape = model.input_shape  # (None, H, W, C)
        h = in_shape[1] if in_shape[1] is not None else spatial
        w = in_shape[2] if in_shape[2] is not None else spatial
        c = in_shape[3] if in_shape[3] is not None else channels
    else:
        if checkpoint_path:
            logger.warning(
                "checkpoint not found at %s; building a tiny fallback model",
                checkpoint_path,
            )
        else:
            logger.info("no checkpoint given; building a tiny fallback model")
        h = w = max(16, spatial)
        c = channels
        model = create_convunext_denoiser(
            input_shape=(h, w, c),
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            drop_path_rate=0.0,
        )

    tf.random.set_seed(seed)
    clean = tf.random.uniform([batch, h, w, c], minval=-1.0, maxval=1.0)
    noisy = apply_multiplicative_gaussian(clean, sigma)

    diag = sure_divergence_consistency(
        model, noisy, sigma=sigma, n_hutchinson=n_hutchinson, eps=eps, seed=seed
    )
    return diag


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generalized-SURE divergence-consistency diagnostic for a multiplicative-"
            "noise denoiser checkpoint (no clean references required)."
        )
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a saved .keras denoiser. If omitted/missing, a tiny model is built.",
    )
    parser.add_argument("--sigma", type=float, default=0.15, help="Multiplicative noise std.")
    parser.add_argument("--batch", type=int, default=4, help="Synthetic batch size.")
    parser.add_argument("--spatial", type=int, default=64, help="Synthetic spatial size.")
    parser.add_argument("--channels", type=int, default=3, help="Synthetic channel count.")
    parser.add_argument("--n-hutchinson", type=int, default=8, help="Hutchinson probes.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Finite-difference step.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    diag = run_checkpoint_diagnostic(
        checkpoint_path=args.checkpoint,
        sigma=args.sigma,
        batch=args.batch,
        spatial=args.spatial,
        channels=args.channels,
        n_hutchinson=args.n_hutchinson,
        eps=args.eps,
        seed=args.seed,
    )

    # CLI stdout summary (a __main__ entry may print; library code uses the logger).
    print("=" * 60)
    print("SURE divergence-consistency diagnostic (multiplicative noise)")
    print("=" * 60)
    for key, value in diag.items():
        print(f"  {key:>22s} : {value:.6g}")
    print("=" * 60)

    finite = all(np.isfinite(v) for v in diag.values())
    return 0 if finite else 1


if __name__ == "__main__":
    import sys

    sys.exit(_main())
