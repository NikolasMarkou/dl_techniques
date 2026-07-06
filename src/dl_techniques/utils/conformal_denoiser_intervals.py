"""Post-hoc per-pixel split-conformal intervals for a FROZEN bias-free denoiser.

This module turns a trained, **frozen** additive-Gaussian denoiser ``D`` into a
per-pixel interval predictor without any retraining and without touching the
model factories. Given a held-out calibration set of clean/noisy pairs it
produces a per-sigma scalar radius ``q`` such that the symmetric interval
``[mu - q, mu + q]`` (with ``mu = D(y)``) attains the target marginal coverage
on an independent test split.

Guarantees and caveats (read before trusting an interval):

* **Post-hoc / frozen.** The model is called black-box (a forward pass only);
  its weights are never modified and no factory is edited. Everything here is a
  stateless function over host (numpy) arrays plus keras forward passes.
* **Distribution-free finite-sample coverage.** Split conformal gives a
  finite-sample guarantee of coverage ``>= 1 - alpha`` using the finite-sample
  quantile ``ceil((n+1)(1-alpha))/n`` of the nonconformity scores (see
  :func:`conformal_quantile`). No parametric noise assumption is required.
* **Coverage is MARGINAL, not per-pixel-conditional.** The guarantee holds on
  average over the exchangeable population, not conditionally at every pixel.
  The interval width is a single scalar ``q`` per sigma bin (homoscedastic), not
  a heteroscedastic per-pixel band.
* **Exchangeability -> per-sigma (Mondrian) bins.** The training noise
  curriculum sweeps sigma, so residuals are NOT exchangeable across sigmas.
  Calibration is therefore done independently per sigma bin
  (:func:`calibrate_per_sigma`); mixing sigmas breaks the guarantee.
* **Domain convention ``[-0.5, +0.5]``.** All model outputs are clipped to
  ``[-0.5, +0.5]`` to match the denoiser domain (``src/train/bfunet/common.py``
  ``denoise_k_passes`` / ``_mean_psnr``, ``max_val=1.0``). No new domain is
  invented here.

Deliberate design choice — coverage/width are computed with **plain numpy
reductions** inline, NOT via a Keras metric. There are two colliding
``CoverageMetric`` classes in the repo
(``dl_techniques/metrics/probabilistic_forecast_metrics.py`` and
``dl_techniques/models/cliffordnet/confidence_denoiser.py``); importing either
would couple this post-hoc utility to an unrelated stateful metric and pick a
side of a name collision. The math (a boolean mean and a width mean) is trivial,
so we keep it stateless and dependency-free.

Layering: part of ``dl_techniques.utils``; depends only on numpy, tensorflow,
keras, and ``dl_techniques.utils.logger``. It MUST NOT import from ``src/train``.
"""

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Domain / defaults (single source of truth for this module)
# ---------------------------------------------------------------------

DOMAIN_MIN: float = -0.5
DOMAIN_MAX: float = 0.5
DEFAULT_BATCH_SIZE: int = 8


# ---------------------------------------------------------------------
# 0. Output-contract unwrap (deep-supervision: index-0 point estimate)
# ---------------------------------------------------------------------


def _unwrap_point_estimate(model_output: Any) -> Any:
    """Return the point-estimate (index-0) element of a denoiser output.

    Bias-free denoisers with deep supervision return a ``list``/``tuple`` whose
    first element is the primary point estimate; a plain denoiser returns a
    single tensor. This mirrors the deep-supervision unwrap in
    ``src/train/bfunet/common.py`` (``denoise_k_passes``, lines 479-480).

    Unlike ``multiplicative_miyasawa._denoiser_point_estimate`` (which takes the
    ``(denoiser, y)`` callable + input and performs the forward pass), this
    helper operates on an ALREADY-COMPUTED output, so it composes with a batched
    forward loop without re-invoking the model.

    Args:
        model_output: The raw output of ``model(y)`` — a tensor, or a list/tuple
            whose index-0 element is the point estimate.

    Returns:
        The point-estimate tensor (index-0 of a list/tuple output, else the
        output itself).
    """
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    return model_output


# ---------------------------------------------------------------------
# 1. Core correctness primitive: finite-sample conformal quantile
# ---------------------------------------------------------------------


def conformal_quantile(scores_1d: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of 1-D nonconformity scores.

    Returns the ``ceil((n+1)*(1-alpha))``-th smallest score (1-indexed order
    statistic), i.e. the empirical quantile at level ``ceil((n+1)(1-alpha))/n``.
    This is the standard split-conformal correction that yields a finite-sample
    marginal coverage guarantee of at least ``1 - alpha`` when the scores are
    exchangeable.

    Small-``n`` behaviour: when ``(n+1)*(1-alpha) > n`` (equivalently the order
    index ``k = ceil((n+1)(1-alpha))`` exceeds ``n``), no finite score is large
    enough to guarantee coverage, so the quantile is ``+inf`` (an unbounded
    interval covering everything). This happens for small ``n`` relative to
    ``alpha`` (e.g. ``n < 1/alpha - 1``); callers should treat ``+inf`` as
    "calibration set too small for this alpha".

    Args:
        scores_1d: 1-D (or flattenable) array of nonconformity scores. Ravelled
            internally.
        alpha: Miscoverage level in ``(0, 1)``; target coverage is ``1 - alpha``.

    Returns:
        The finite-sample conformal quantile as a float (possibly ``+inf`` for
        very small ``n``).

    Raises:
        ValueError: If ``scores_1d`` is empty or ``alpha`` is not in ``(0, 1)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    scores = np.asarray(scores_1d, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("conformal_quantile received an empty score array")

    # 1-indexed order-statistic rank; exact split-conformal correction.
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        # (n+1)(1-alpha) > n: interval must be unbounded to guarantee coverage.
        return float(np.inf)

    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


# ---------------------------------------------------------------------
# 2. Batched forward pass -> clipped point estimate (host numpy array)
# ---------------------------------------------------------------------


def _predict_mu(
        model: Any,
        noisy: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Run ``model`` over ``noisy`` in batches, unwrap index-0, clip to domain.

    Args:
        model: A frozen denoiser callable. Called as ``model(batch,
            training=False)``; a list/tuple output is unwrapped to index-0.
        noisy: Array of noisy inputs, shape ``(N, ...)``.
        batch_size: Forward-pass batch size.

    Returns:
        Numpy array ``mu`` of denoised point estimates clipped to
        ``[DOMAIN_MIN, DOMAIN_MAX]``, same shape as ``noisy``.
    """
    noisy_arr = np.asarray(noisy)
    n = noisy_arr.shape[0]
    chunks: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        batch = noisy_arr[start:start + batch_size]
        out = model(tf.convert_to_tensor(batch), training=False)
        out = _unwrap_point_estimate(out)
        out = np.asarray(out)
        chunks.append(out)
    mu = np.concatenate(chunks, axis=0)
    return np.clip(mu, DOMAIN_MIN, DOMAIN_MAX)


# ---------------------------------------------------------------------
# 3. Per-sigma (Mondrian) calibration
# ---------------------------------------------------------------------


def calibrate_per_sigma(
        model: Any,
        clean: np.ndarray,
        noisy: np.ndarray,
        sigmas: Union[Sequence[float], np.ndarray],
        alpha: float = 0.1,
        batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[float, float]:
    """Calibrate one conformal radius ``q`` per sigma bin (Mondrian split conformal).

    For each distinct sigma label, the corresponding calibration subset is run
    through the frozen model, the per-pixel nonconformity ``s = |mu - clean|``
    (with ``mu`` clipped to ``[-0.5, +0.5]``) is flattened over ALL pixels of
    that bin, and :func:`conformal_quantile` yields the bin's radius ``q``.

    Per-sigma (rather than pooled) calibration is REQUIRED: the noise curriculum
    sweeps sigma, so residuals are not exchangeable across sigmas and a pooled
    quantile would miscover. Within a fixed sigma bin, additive-Gaussian
    residuals across held-out patches are exchangeable, so the finite-sample
    guarantee holds per bin.

    Args:
        model: Frozen denoiser callable (black-box forward pass only).
        clean: Clean targets, shape ``(N, ...)``.
        noisy: Noisy inputs, shape ``(N, ...)`` aligned with ``clean``.
        sigmas: Per-sample sigma labels, length ``N`` (one label per sample). The
            distinct values define the Mondrian bins.
        alpha: Miscoverage level; target coverage per bin is ``1 - alpha``.
        batch_size: Forward-pass batch size.

    Returns:
        Dict mapping each distinct sigma (as ``float``) to its conformal radius
        ``q`` (a ``float``; possibly ``+inf`` if a bin is too small for ``alpha``).

    Raises:
        ValueError: If inputs are empty, misaligned, or ``sigmas`` length does
            not match ``clean``/``noisy``.
    """
    clean_arr = np.asarray(clean)
    noisy_arr = np.asarray(noisy)
    sigma_arr = np.asarray(sigmas, dtype=np.float64).ravel()

    n = clean_arr.shape[0]
    if n == 0:
        raise ValueError("calibrate_per_sigma received an empty calibration set")
    if noisy_arr.shape[0] != n:
        raise ValueError(
            f"clean/noisy sample-count mismatch: {n} vs {noisy_arr.shape[0]}"
        )
    if sigma_arr.shape[0] != n:
        raise ValueError(
            f"sigmas length {sigma_arr.shape[0]} != sample count {n}"
        )

    q_by_sigma: Dict[float, float] = {}
    for sigma in np.unique(sigma_arr):
        mask = sigma_arr == sigma
        clean_bin = clean_arr[mask]
        noisy_bin = noisy_arr[mask]

        mu = _predict_mu(model, noisy_bin, batch_size=batch_size)
        scores = np.abs(mu - clean_bin).ravel()
        q = conformal_quantile(scores, alpha)

        sigma_key = float(sigma)
        q_by_sigma[sigma_key] = q
        logger.info(
            "calibrate_per_sigma: sigma=%.4f  n_samples=%d  n_pixels=%d  "
            "alpha=%.3f  q=%.6g",
            sigma_key, int(mask.sum()), int(scores.size), alpha, q,
        )

    return q_by_sigma


# ---------------------------------------------------------------------
# 4. Interval prediction
# ---------------------------------------------------------------------


def predict_intervals(
        model: Any,
        y: np.ndarray,
        q: float,
        batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict symmetric per-pixel intervals ``[mu - q, mu + q]`` for noisy ``y``.

    ``mu`` is the frozen model's index-0 point estimate clipped to
    ``[-0.5, +0.5]``. ``q`` is a scalar radius (single sigma bin); applying the
    per-sigma radius to the matching sigma subset is the caller's job.

    Args:
        model: Frozen denoiser callable.
        y: Noisy inputs, shape ``(N, ...)``.
        q: Scalar conformal radius (from :func:`calibrate_per_sigma`).
        batch_size: Forward-pass batch size.

    Returns:
        Tuple ``(mu, lower, upper)`` of numpy arrays, each the shape of the
        model output for ``y``. ``lower = mu - q``, ``upper = mu + q``.
    """
    mu = _predict_mu(model, y, batch_size=batch_size)
    lower = mu - q
    upper = mu + q
    return mu, lower, upper


# ---------------------------------------------------------------------
# 5. Empirical coverage / mean width (plain numpy reductions)
# ---------------------------------------------------------------------


def evaluate_coverage(
        model: Any,
        clean: np.ndarray,
        noisy: np.ndarray,
        q: float,
        batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, float]:
    """Empirical per-pixel coverage and mean interval width for radius ``q``.

    Coverage is the fraction of pixels whose clean value falls inside
    ``[mu - q, mu + q]``; mean width is the average of ``upper - lower`` (equal
    to ``2*q`` for a scalar ``q``). Both are plain numpy reductions — no Keras
    metric is used (see module docstring on the ``CoverageMetric`` name
    collision).

    Args:
        model: Frozen denoiser callable.
        clean: Clean targets, shape ``(N, ...)``.
        noisy: Noisy inputs, shape ``(N, ...)`` aligned with ``clean``.
        q: Scalar conformal radius applied to every pixel.
        batch_size: Forward-pass batch size.

    Returns:
        Dict ``{"coverage": float, "mean_width": float}``.
    """
    clean_arr = np.asarray(clean)
    _mu, lower, upper = predict_intervals(
        model, noisy, q, batch_size=batch_size
    )
    covered = (clean_arr >= lower) & (clean_arr <= upper)
    coverage = float(np.mean(covered))
    mean_width = float(np.mean(upper - lower))
    logger.info(
        "evaluate_coverage: n_pixels=%d  q=%.6g  coverage=%.4f  mean_width=%.6g",
        int(covered.size), q, coverage, mean_width,
    )
    return {"coverage": coverage, "mean_width": mean_width}
