"""Equivariance metrics for topographic capsule models.

Two metrics, both from Keller & Welling (2022), and both **plain NumPy
functions rather than `keras.metrics.Metric`**. That is not a shortcut, it is
forced by what each quantity is:

- :func:`equivariance_error` is a sum over *every pair* of timesteps
  ``(l, l + delta)``. A streaming `update_state` sees one batch of consecutive
  windows, so it cannot accumulate the full pairwise sum without state that
  grows with the sequence length.
- :func:`capcorr_correlation` is a Pearson correlation between a *measured
  per-example integer* and the ground-truth factor shift, over the entire
  dataset. A running mean is computable in principle, but the metric is only
  meaningful once every example has contributed its argmax, and the paper
  reports it as a dataset-level number.

This matches the convention already documented in `dl_techniques.metrics.AGENTS.md`
(`sequence_metrics.py`, `embedding_quality.py` and `perplexity_metric.py` all
export plain functions for the same reason).

Why both metrics, given that they can disagree
-----------------------------------------------
This is the interesting part, and the paper is explicit about it (Section 6.3):
the two measure **different things** and can rank models differently.

`equivariance_error` is a continuous L1 distance between a rolled activation and
the next one. It is LOW whenever the representation is *smooth in time* — which
includes a representation that is entirely **invariant**. The BubbleVAE
baseline, whose capsules encode an invariance rather than an equivariance, scores
a low `equivariance_error` (3369 on MNIST vs 13274 for a plain VAE) while its
`capcorr` is no better than a random baseline's.

`capcorr` measures the actual **correspondence** between the input transformation
and the transformation of the representation: the `argmax` of a periodic
cross-correlation must equal the ground-truth factor shift. An invariant
representation rolls by zero no matter what happens to the input, so it scores
`capcorr ~= 0`.

**Read `capcorr` as the equivariance metric.** `equivariance_error` is a
smoothness diagnostic and is reported alongside it, never in place of it.

Why not a rolling window
------------------------
:func:`equivariance_error` sums over ``delta = 1 .. sequence_length - 1``, i.e.
every offset within the sequence, not a fixed lag. That is what makes it a
measure of whether the representation is *globally* consistent under the roll
rather than locally smooth. A one-lag-only variant would be a different metric.

References:
    - Keller & Welling, 2022. Topographic VAEs learn Equivariant Capsules.
      NeurIPS 2021. (https://arxiv.org/abs/2109.01394)
"""

from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


def _as_numpy(value) -> np.ndarray:
    """Convert a tensor or array to float64 NumPy, rejecting a wrong rank.

    :param value: A tensor or array of at least rank 3 ``(batch, sequence, ...)``.
    :type value: Any
    :return: A float64 array.
    :rtype: np.ndarray
    :raises ValueError: If the value has fewer than 3 axes. Both metrics index a
        time axis and a capsule axis, and a silently-wrong axis would produce a
        number rather than an error.
    """
    array = np.asarray(
        value.numpy() if hasattr(value, "numpy") else value, dtype=np.float64
    )
    if array.ndim < 3:
        raise ValueError(
            f"expected at least rank 3 (batch, sequence, capsules, dims), got "
            f"shape {array.shape}"
        )
    return array


def roll_capsules(values: np.ndarray, shift: int) -> np.ndarray:
    """Cyclically permute each capsule by ``shift`` steps, in NumPy.

    The NumPy twin of :class:`dl_techniques.layers.capsules.CapsuleRoll`, with
    the same convention: ``out[..., c, j] = values[..., c, (j - shift) % D]``.

    :param values: Array shaped ``(..., num_capsules, capsule_dim)``.
    :type values: np.ndarray
    :param shift: Cyclic steps.
    :type shift: int
    :return: An array of the same shape.
    :rtype: np.ndarray
    """
    capsule_dim = values.shape[-1]
    offset = int(shift) % capsule_dim
    if offset == 0:
        return values.copy()
    return np.roll(values, offset, axis=-1)


def _periodic_cross_correlation(t0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Argmax over capsule shifts of the periodic cross-correlation of two capsules.

    Computes, for every candidate shift ``d``, ``sum_j t0[..., j] * t[..., j + d]``
    and returns the ``d`` maximizing it. A positive ``d`` therefore means ``t`` is
    ``t0`` advanced by ``d`` — the same sign convention as ``CapsuleRoll``.

    :param t0: **Earlier** reference capsule activations ``(batch, capsule_dim)``.
    :type t0: np.ndarray
    :param t: **Later** comparison activations ``(batch, capsule_dim)``.
    :type t: np.ndarray
    :return: Integer shifts ``(batch,)`` in ``[0, capsule_dim)``, positive for a
        forward roll.
    :rtype: np.ndarray
    """
    capsule_dim = t0.shape[-1]
    # Candidate shift ``d`` scores ``sum_j t0[j] * t[j + d]``, so a positive ``d``
    # means `t` is `t0` advanced by ``d` — the same sign convention as
    # `CapsuleRoll`, where `roll(v, d)[j] == v[j - d]` and the cross-correlation
    # is taken between the *later* and the *earlier* activation.
    #
    # Note that `np.roll(t, d)` gives `t[j - d]`, i.e. the OPPOSITE sign; using it
    # directly here would report every roll as `(-d) mod D`, which is invisible
    # in a symmetric metric and silently flips the sign of the correlation for
    # any asymmetric one. The explicit index is what pins the direction.
    indices = (
        np.arange(capsule_dim)[None, :] + np.arange(capsule_dim)[:, None]
    ) % capsule_dim  # (D candidates, D source slots)
    candidates = t[:, indices.reshape(-1)].reshape(
        t.shape[0], capsule_dim, capsule_dim
    )
    scores = np.sum(candidates * t0[:, None, :], axis=-1)  # (batch, D)
    return np.argmax(scores, axis=-1)


def observed_roll(
    t_earlier: np.ndarray,
    t_later: np.ndarray,
) -> np.ndarray:
    """Per-capsule estimate of the roll from one timestep to a LATER one.

    :param t_earlier: Activations at the earlier timestep ``(batch, C, D)``.
    :type t_earlier: np.ndarray
    :param t_later: Activations at the later timestep ``(batch, C, D)``.
    :type t_later: np.ndarray
    :return: Integer rolls ``(batch, C)`` in ``[0, D)``, positive for a forward
        roll.
    :rtype: np.ndarray

    The paper writes ``ObservedRoll(t_Omega, t_0) = argmax[t_Omega ? t_0]`` and
    defines ``?`` only as "the discrete periodic cross-correlation", which leaves
    the argument order — and therefore the SIGN of every reported shift — up to
    the reader. The direction implemented here is the one that makes ``CapCorr``
    equal ``+1`` for a perfectly equivariant representation, which is the property
    the metric is read against. MEASURED: with the arguments the other way round,
    a latent that rolls by ``+start`` reports ``D - start`` and CapCorr comes out
    as a correlation against the WRONG shift.
    """
    earlier = t_earlier.reshape(t_earlier.shape[0], -1, t_earlier.shape[-1])
    later = t_later.reshape(t_later.shape[0], -1, t_later.shape[-1])
    return np.stack(
        [
            _periodic_cross_correlation(earlier[:, c], later[:, c])
            for c in range(later.shape[1])
        ],
        axis=-1,
    )


def equivariance_error(
    latent_sequence,
    normalize: bool = True,
) -> float:
    """Mean absolute error between a rolled latent activation and the next one.

    Eq. 13 of Keller & Welling (2022). For a sequence of length ``S`` with
    L2-normalised activations ``t_hat = t / ||t||_2``, the error is

    .. code-block:: text

        E_eq = sum_{l=1..S-1} sum_{delta=1..S-l} || Roll_delta(t_hat_l) - t_hat_{l+delta} ||_1
        divided by the number of terms (S(S-1)/2) and by the latent width

    The normalization matters and is on by default: without it the metric would
    partly measure activation magnitude rather than direction.

    **This is a smoothness measure, not an equivariance measure.** It is low for
    an *invariant* representation just as it is low for an equivariant one, which
    is why :func:`capcorr_correlation` exists and why the paper reports both. See
    the module docstring.

    :param latent_sequence: Activations shaped ``(batch, sequence, num_capsules,
        capsule_dim)``, or ``(batch, sequence, latent_dim)`` for a single
        capsule. Sequence length must be at least 2.
    :type latent_sequence: Any
    :param normalize: Whether to L2-normalise each timestep before comparing.
        Defaults to ``True``.
    :type normalize: bool
    :return: The mean L1 error, a non-negative float.
    :rtype: float
    :raises ValueError: If the sequence length is less than 2, in which case the
        sum is empty and a mean over zero terms would be ``NaN`` -- an undefined
        metric that reads like a measurement.
    """
    activations = _as_numpy(latent_sequence)
    if activations.shape[1] < 2:
        raise ValueError(
            f"equivariance_error needs a sequence length of at least 2, got "
            f"{activations.shape[1]}"
        )

    if normalize:
        norms = np.sqrt(np.sum(activations**2, axis=-1, keepdims=True))
        activations = activations / np.maximum(norms, 1e-12)

    sequence_length = activations.shape[1]
    total = 0.0
    terms = 0
    for lag in range(1, sequence_length):
        for index in range(0, sequence_length - lag):
            rolled = roll_capsules(activations[:, index], lag)
            total += float(np.abs(rolled - activations[:, index + lag]).sum())
            terms += 1

    if terms == 0:
        return float("nan")
    return total / terms


def capcorr_correlation(
    latent_sequence,
    factors,
    canonical_factor: Optional[float] = None,
) -> float:
    """Pearson correlation between observed capsule roll and ground-truth shift.

    Eq. 15/16 of Keller & Welling (2022): the ``CapCorr`` metric. For every
    example, the activation at the canonical timestep ``Omega`` (where the ground
    truth factor sits at its reference position) is cross-correlated against the
    activation at timestep 0; the per-capsule ``argmax`` is reduced to a single
    integer by its **mode** across capsules, and that integer is correlated with
    ``|y_Omega - y_0|``.

    A perfectly equivariant representation gives ``1.0``. An invariant one gives
    approximately ``0``.

    The mode reduction assumes the capsules roll *simultaneously*, which the
    paper reports empirically holds. This function returns the per-capsule
    correlations alongside the pooled one so that assumption is observable
    rather than assumed: a large spread across capsules is the signal that the
    pooled number is hiding disagreement.

    :param latent_sequence: Activations ``(batch, sequence, num_capsules,
        capsule_dim)``.
    :type latent_sequence: Any
    :param factors: Ground-truth transformation parameter per timestep,
        ``(batch, sequence)``.
    :type factors: Any
    :param canonical_factor: The value of ``factors`` that counts as timestep
        ``Omega``. When ``None``, the per-example minimum of ``factors`` is used.
        For a cyclic sequence that is the natural canonical position; for a
        sequence that does not wrap this is a guess, so pass it explicitly when
        one is known.
    :type canonical_factor: Optional[float]
    :return: The pooled Pearson correlation, a float in ``[-1, 1]``. ``nan`` if
        fewer than two examples have a defined shift.
    :rtype: float
    :raises ValueError: If ``factors`` and ``latent_sequence`` disagree on the
        batch size or sequence length.
    """
    activations = _as_numpy(latent_sequence)
    factor_values = np.asarray(
        factors.numpy() if hasattr(factors, "numpy") else factors,
        dtype=np.float64,
    )
    if factor_values.shape != activations.shape[:2]:
        raise ValueError(
            f"factors shape {factor_values.shape} must match the (batch, "
            f"sequence) prefix of latent_sequence {activations.shape[:2]}"
        )

    if canonical_factor is None:
        canonical_index = np.argmin(factor_values, axis=-1)
    else:
        matches = np.isclose(factor_values, canonical_factor)
        if not np.all(matches.any(axis=-1)):
            missing = int((~matches.any(axis=-1)).sum())
            raise ValueError(
                f"{missing} example(s) never reach canonical_factor="
                f"{canonical_factor}, so Omega is undefined for them"
            )
        canonical_index = np.argmax(matches, axis=-1)

    rows = np.arange(activations.shape[0])
    earlier = activations[:, 0]  # (batch, C, D), timestep 0
    later = activations[rows, canonical_index]  # (batch, C, D), timestep Omega

    # Earlier FIRST: `observed_roll` reports a POSITIVE roll for a forward move.
    per_capsule = observed_roll(earlier, later)  # (batch, C)
    # Mode across capsules. Ties resolve to the smaller shift, matching
    # numpy's argmax-first convention.
    modes = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=per_capsule.shape[-1]).argmax(),
        axis=-1,
        arr=per_capsule,
    )

    shift = np.abs(
        factor_values[rows, canonical_index] - factor_values[:, 0]
    )
    pooled = _pearson(modes.astype(np.float64), shift)
    logger.info(
        f"CapCorr pooled={pooled:.4f} over {activations.shape[0]} example(s), "
        f"{per_capsule.shape[1]} capsule(s)"
    )
    return pooled


def capcorr_per_capsule(
    latent_sequence,
    factors,
    canonical_factor: Optional[float] = None,
) -> np.ndarray:
    """Per-capsule CapCorr, the diagnostic behind the pooled number.

    :param latent_sequence: Activations ``(batch, sequence, capsules, dims)``.
    :type latent_sequence: Any
    :param factors: Ground-truth parameters ``(batch, sequence)``.
    :type factors: Any
    :param canonical_factor: See :func:`capcorr_correlation`.
    :type canonical_factor: Optional[float]
    :return: One correlation per capsule, ``(num_capsules,)``.
    :rtype: np.ndarray
    """
    activations = _as_numpy(latent_sequence)
    factor_values = np.asarray(
        factors.numpy() if hasattr(factors, "numpy") else factors,
        dtype=np.float64,
    )
    if canonical_factor is None:
        canonical_index = np.argmin(factor_values, axis=-1)
    else:
        canonical_index = np.argmax(
            np.isclose(factor_values, canonical_factor), axis=-1
        )

    rows = np.arange(activations.shape[0])
    per_capsule = observed_roll(
        activations[:, 0], activations[rows, canonical_index]
    )
    shift = np.abs(factor_values[rows, canonical_index] - factor_values[:, 0])
    return np.array(
        [
            _pearson(per_capsule[:, c].astype(np.float64), shift)
            for c in range(per_capsule.shape[1])
        ],
        dtype=np.float64,
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two 1-D arrays, ``nan`` when it is undefined.

    Returns ``nan`` rather than 0 for a constant input: a correlation against a
    zero-variance series is undefined, and reporting 0 would make "this metric
    could not be computed" indistinguishable from "this model is uncorrelated".

    :param a: First series.
    :type a: np.ndarray
    :param b: Second series.
    :type b: np.ndarray
    :return: The correlation, or ``nan``.
    :rtype: float
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"series shapes differ: {a.shape} vs {b.shape}")
    if a.size < 2:
        return float("nan")

    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(a_centered * b_centered) / denominator)

# ---------------------------------------------------------------------