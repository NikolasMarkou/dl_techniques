"""Pure mean-shift operations shared by the mean-shift layers.

Mean shift moves a query ``x`` to the kernel-weighted mean of a set of keys,
``m(x) = sum_j K(x, x_j) v_j / sum_j K(x, x_j)``. With the Gaussian kernel
``K(x, y) = exp(-||x - y||^2 / (2 h^2))`` this is a softmax over the keys,
``m(x) = softmax_j(x.x_j / h^2 - ||x_j||^2 / (2 h^2)) @ V``: the query-norm term
is constant along the softmax axis and is dropped, the key-norm term is kept.
The shift ``m(x) - x`` equals ``h^2 * grad log p_hat(x)`` for the Gaussian kernel
density estimate ``p_hat`` (the Miyasawa/Tweedie form). The von Mises-Fisher
kernel on the unit sphere is the same expression with the key-norm term constant.
This module holds only stateless functions: pairwise squared distance, kernel
logits, one step, a kernel density, a bounded iteration and a median bandwidth
heuristic. Every logit, softmax and density is computed in at least float32
(``utils.dtype_policy.accumulation_dtype``), a masked key has exactly zero weight
whatever the scale of the logits (no fixed sentinel is written into them), and a
row with no kept key never produces NaN. Squared distances and Gaussian logits
are formed on points centred by a key mean, so a constant offset of the data does
not cost float32 digits. The module is not exported from ``layers.mixtures`` and
registers nothing.

References:
    - Fukunaga and Hostetler, 1975. The estimation of the gradient of a density
      function, with applications in pattern recognition.
    - Comaniciu and Meer, 2002. Mean shift: a robust approach toward feature
      space analysis.
    - Efron, 2011. Tweedie's formula and selection bias.
    - Banerjee, Dhillon, Ghosh and Sra, 2005. Clustering on the unit hypersphere
      using von Mises-Fisher distributions.
"""

import math
from typing import Any, Optional, Tuple, Union

import keras

from dl_techniques.utils.dtype_policy import (
    accumulation_dtype,
    mask_sentinel,
    stability_floor,
)
from dl_techniques.utils.tensors import l2_normalize

__all__ = [
    "pairwise_sq_dist",
    "kernel_logits",
    "mean_shift_step",
    "kde_log_density",
    "mean_shift_iterate",
    "median_heuristic_bandwidth",
]

_KERNELS = ("gaussian", "vmf")

_NORMALIZE_EPS = 1e-12

_MEDIAN_FLOOR = 1e-6


def _check_kernel(kernel: str) -> None:
    """Raise if ``kernel`` is not one of the supported kernel names.

    :param kernel: kernel name to validate.
    :type kernel: str
    :raises ValueError: if ``kernel`` is not ``"gaussian"`` or ``"vmf"``.
    """
    if kernel not in _KERNELS:
        raise ValueError(
            f"kernel must be one of {list(_KERNELS)}, got {kernel!r}"
        )


def _dtype_names(x: Any) -> Tuple[str, str]:
    """Return ``(dtype name, accumulation dtype name)`` of a tensor.

    :param x: tensor whose dtype is read.
    :type x: Any
    :return: the tensor's own dtype name and the dtype the math runs in.
    :rtype: Tuple[str, str]
    """
    name = getattr(x.dtype, "name", None) or str(x.dtype)
    return name, accumulation_dtype(name)


def _as_dtype(x: Any, dtype: str) -> Any:
    """Convert a tensor or Python number to ``dtype`` without a float32 detour.

    ``keras.ops.cast(0.8, "float64")`` first materialises 0.8 as float32; this
    converts straight to the target dtype.

    :param x: tensor, array or Python number.
    :type x: Any
    :param dtype: target dtype name.
    :type dtype: str
    :return: tensor of dtype ``dtype``.
    :rtype: Any
    """
    return keras.ops.convert_to_tensor(x, dtype=dtype)


def _normalize_rows(x: Any) -> Any:
    """L2-normalise the last axis; an all-zero row stays all zeros (no NaN).

    :param x: tensor ``(..., D)`` already in its accumulation dtype.
    :type x: Any
    :return: tensor of the same shape and dtype with unit-norm rows.
    :rtype: Any
    """
    return l2_normalize(
        x,
        axis=-1,
        eps=stability_floor(_dtype_names(x)[0], _NORMALIZE_EPS),
    )


def _keep_mask(mask: Any, rank: int) -> Any:
    """Turn a keep-mask into a boolean tensor of rank ``rank``.

    Axes are inserted just before the last (key) axis, so a ``(B, M)`` mask
    becomes ``(B, 1, M)`` against 3-D logits and ``(B, 1, 1, M)`` against 4-D
    logits. A mask whose rank already equals ``rank`` is used as given.

    :param mask: ``None`` or a tensor ``(..., M)`` with 1/True meaning keep.
    :type mask: Any
    :param rank: rank of the logits the mask is applied to.
    :type rank: int
    :return: boolean tensor broadcastable to the logits, or ``None``.
    :rtype: Any
    :raises ValueError: if the mask rank exceeds ``rank``.
    """
    if mask is None:
        return None
    keep = keras.ops.cast(keras.ops.convert_to_tensor(mask), "bool")
    missing = rank - len(keep.shape)
    if missing < 0:
        raise ValueError(
            f"mask has rank {len(keep.shape)}, more than the logits rank {rank}"
        )
    for _ in range(missing):
        keep = keras.ops.expand_dims(keep, axis=-2)
    return keep


def _keep_all_rescue(keep: Any) -> Any:
    """Make a row with no kept key keep every key (used where the count matters).

    :param keep: boolean keep-mask ``(..., 1 or N, M)``.
    :type keep: Any
    :return: boolean mask of the same shape in which no row is empty.
    :rtype: Any
    """
    any_kept = keras.ops.any(keep, axis=-1, keepdims=True)
    return keras.ops.logical_or(keep, keras.ops.logical_not(any_kept))


def _to_acc(x: Any, acc: str) -> Any:
    """Convert an array, Variable or tensor to a tensor of dtype ``acc``.

    :param x: tensor-like operand.
    :type x: Any
    :param acc: accumulation dtype name.
    :type acc: str
    :return: tensor of dtype ``acc``.
    :rtype: Any
    """
    return keras.ops.cast(keras.ops.convert_to_tensor(x), acc)


def _key_centre(keys: Any, key_mask: Optional[Any] = None) -> Any:
    """Mean of the keys over their row axis, as a constant reference point.

    Distances and Gaussian logits are exactly invariant to subtracting one point
    from queries and keys, and the expanded form loses digits in float32 when the
    points sit far from the origin; the centre removes that offset. With a
    keep-mask the mean runs over the kept keys only, so a value stored at a masked
    position (a large finite junk value, as padding often is) cannot set the centre.
    The result carries no gradient.

    :param keys: tensor ``(..., M, D)`` in its accumulation dtype.
    :type keys: Any
    :param key_mask: ``None`` or a keep-mask ``(..., M)`` whose leading axes
        broadcast with those of ``keys``; 1/True keeps.
    :type key_mask: Optional[Any]
    :return: tensor ``(..., 1, D)``; the zero vector when no key is kept.
    :rtype: Any
    """
    if key_mask is None:
        centre = keras.ops.mean(keys, axis=-2, keepdims=True)
    else:
        weight = keras.ops.expand_dims(
            keras.ops.cast(keras.ops.convert_to_tensor(key_mask), keys.dtype), axis=-2
        )
        count = keras.ops.sum(weight, axis=-1, keepdims=True)
        centre = keras.ops.matmul(weight, keys) / keras.ops.maximum(count, 1.0)
    return keras.ops.stop_gradient(centre)


def _masked_exp(logits: Any, keep: Any) -> Tuple[Any, Any]:
    """Exponentiate max-shifted logits; a masked entry is exactly 0.

    The row maximum is taken over the kept logits only (a masked entry is set to
    negative infinity for the maximum), so the shift does not depend on the scale
    of the logits and a kept entry is at most ``exp(0) = 1``. A row with no kept
    entry has maximum negative infinity and an all-zero result; callers handle it.

    :param logits: tensor ``(..., N, M)`` in the accumulation dtype.
    :type logits: Any
    :param keep: boolean tensor broadcastable to ``logits``.
    :type keep: Any
    :return: ``(exp(logits - row_max) with masked entries 0, row_max)``, the
        maximum shaped ``(..., N, 1)`` and carrying no gradient.
    :rtype: Tuple[Any, Any]
    """
    row_max = keras.ops.stop_gradient(
        keras.ops.max(
            keras.ops.where(keep, logits, keras.ops.full_like(logits, -math.inf)),
            axis=-1,
            keepdims=True,
        )
    )
    shifted = keras.ops.where(keep, logits - row_max, keras.ops.zeros_like(logits))
    zeros = keras.ops.zeros_like(logits)
    return keras.ops.where(keep, keras.ops.exp(shifted), zeros), row_max


def pairwise_sq_dist(a: Any, b: Any, key_mask: Optional[Any] = None) -> Any:
    """Squared Euclidean distance between every row of ``a`` and every row of ``b``.

    Uses the expanded form ``||a||^2 - 2 a.b + ||b||^2`` clamped at 0, so the
    result is never negative and the diagonal of ``pairwise_sq_dist(a, a)`` is
    zero to float tolerance. Both operands are first shifted by the mean of ``b``
    over its row axis: the distance is exactly invariant to the shift, and the
    expanded form no longer cancels two large squared norms when the points sit
    far from the origin. The math runs in at least float32 and the result is
    returned in that accumulation dtype; callers cast back if they need to.

    ``utils.tensors.pairwise_squared_distance`` is not reused: it takes centers
    of rank 2 only, one shared set for all batch rows, and subtracts directly,
    which would materialise an ``(..., N, M, D)`` tensor for per-batch keys.

    :param a: tensor-like ``(..., N, D)``.
    :type a: Any
    :param b: tensor-like ``(..., M, D)`` whose leading axes broadcast with ``a``.
    :type b: Any
    :param key_mask: ``None`` or a keep-mask ``(..., M)`` for ``b``; the shift is
        then the mean of the kept rows of ``b``.
    :type key_mask: Optional[Any]
    :return: squared distances ``(..., N, M)`` in the accumulation dtype.
    :rtype: Any
    """
    a = keras.ops.convert_to_tensor(a)
    _, acc = _dtype_names(a)
    a = keras.ops.cast(a, acc)
    b = _to_acc(b, acc)
    # DECISION plan-2026-10-29T104646-27f045c5/D-027: shift both operands by a key mean before the expanded form.
    # Do not drop the shift as redundant: the distance is invariant but float32 cancels without it.
    centre = _key_centre(b, key_mask)
    a = a - centre
    b = b - centre
    # DECISION plan-2026-10-29T104646-27f045c5/D-016: batched matmul form, own function.
    # Do not wrap the rank-2-centers direct-subtraction helper here. See decisions.md.
    a_sq = keras.ops.sum(keras.ops.square(a), axis=-1, keepdims=True)
    b_sq = keras.ops.expand_dims(
        keras.ops.sum(keras.ops.square(b), axis=-1), axis=-2
    )
    cross = keras.ops.matmul(a, keras.ops.swapaxes(b, -1, -2))
    return keras.ops.maximum(a_sq - 2.0 * cross + b_sq, 0.0)


def kernel_logits(
    q: Any,
    k: Any,
    bandwidth: Any,
    kernel: str = "gaussian",
    drop_query_norm: bool = True,
    key_mask: Optional[Any] = None,
) -> Any:
    """Log of the unnormalised kernel weight of every (query, key) pair.

    Gaussian: ``(q - c).(k - c) / h^2 - ||k - c||^2 / (2 h^2)``, minus
    ``||q - c||^2 / (2 h^2)`` when ``drop_query_norm`` is False, where ``c`` is the
    mean of the keys (of the kept keys when ``key_mask`` is given), held constant
    for differentiation. Centring leaves every difference of logits along the key
    axis equal to that of ``q.k / h^2 - ||k||^2 / (2 h^2)``, so the logits are
    defined up to a per-query constant: a softmax over the keys is the same with
    or without the query-norm term and with or without the centre, and only the
    centred form keeps its float32 digits when the points sit far from the
    origin. vMF: ``q.k / h^2`` on L2-normalised ``q`` and ``k`` (zero rows stay
    zero, no NaN); ``drop_query_norm`` and ``key_mask`` have no effect. The
    bandwidth is not clipped here.

    :param q: queries, tensor-like ``(..., N, D)``.
    :type q: Any
    :param k: keys, tensor-like ``(..., M, D)``.
    :type k: Any
    :param bandwidth: ``h``; a scalar, ``(B, 1, 1)``, ``(B, H, 1, 1)`` or per-query
        ``(..., N, 1)``, aligned from the right with the logits ``(..., N, M)``.
    :type bandwidth: Any
    :param kernel: ``"gaussian"`` or ``"vmf"``.
    :type kernel: str
    :param drop_query_norm: omit the per-query constant of the Gaussian kernel.
    :type drop_query_norm: bool
    :param key_mask: ``None`` or a keep-mask ``(..., M)`` selecting the keys that
        set the Gaussian centre; ``None`` uses every key.
    :type key_mask: Optional[Any]
    :return: logits ``(..., N, M)`` in the accumulation dtype.
    :rtype: Any
    :raises ValueError: if ``kernel`` is not a supported name.
    """
    _check_kernel(kernel)
    q = keras.ops.convert_to_tensor(q)
    _, acc = _dtype_names(q)
    q = keras.ops.cast(q, acc)
    k = _to_acc(k, acc)
    h = _as_dtype(bandwidth, acc)
    inv_h2 = 1.0 / (h * h)

    if kernel == "vmf":
        q = _normalize_rows(q)
        k = _normalize_rows(k)
        return keras.ops.matmul(q, keras.ops.swapaxes(k, -1, -2)) * inv_h2

    # DECISION plan-2026-10-29T104646-27f045c5/D-027: centre queries and keys by a key mean.
    # Do not form q.k - ||k||^2 / 2 on raw points: float32 loses the distances far from the origin.
    centre = _key_centre(k, key_mask)
    q = q - centre
    k = k - centre
    cross = keras.ops.matmul(q, keras.ops.swapaxes(k, -1, -2))
    k_sq = keras.ops.expand_dims(
        keras.ops.sum(keras.ops.square(k), axis=-1), axis=-2
    )
    logits = cross * inv_h2 - 0.5 * k_sq * inv_h2
    if not drop_query_norm:
        q_sq = keras.ops.sum(keras.ops.square(q), axis=-1, keepdims=True)
        logits = logits - 0.5 * q_sq * inv_h2
    return logits


def _logit_rank(q: Any, k: Any) -> int:
    """Rank of the ``(..., N, M)`` logits of two ``(..., N or M, D)`` operands."""
    return max(len(q.shape), len(k.shape))


def mean_shift_step(
    q: Any,
    k: Any,
    v: Any,
    bandwidth: Any,
    mask: Optional[Any] = None,
    kernel: str = "gaussian",
) -> Tuple[Any, Any]:
    """One mean-shift step: the kernel-weighted mean of ``v`` for every query.

    Weights are ``softmax(kernel_logits(q, k, h))`` over the key axis. A masked
    key is excluded without any fixed sentinel: the row maximum is taken over the
    kept logits only and a masked weight is set to exactly 0.0, so the result does
    not depend on the scale of the logits (a tiny bandwidth gives logits of
    ``-1e12`` and below). A row with no kept key gets uniform weights over all
    keys, so its output is the plain mean of ``v``. With a mask the Gaussian centre
    is the mean of the kept keys. For the vMF kernel only the logits normalise
    ``q`` and ``k``; ``v`` is used as given.

    :param q: queries, tensor-like ``(..., N, D)``.
    :type q: Any
    :param k: keys, tensor-like ``(..., M, D)``.
    :type k: Any
    :param v: values, tensor-like ``(..., M, Dv)``, or ``None`` to use ``k``.
    :type v: Optional[Any]
    :param bandwidth: ``h``, as in :func:`kernel_logits`.
    :type bandwidth: Any
    :param mask: ``None`` or a keep-mask (1/True keeps) ``(..., M)``. Missing axes
        are inserted before the key axis: ``(B, M)`` serves 3-D logits ``(B, N, M)``
        and 4-D logits ``(B, H, N, M)``; a mask of the logits' own rank is used as
        given (for example a pairwise ``(B, N, M)`` mask).
    :type mask: Optional[Any]
    :param kernel: ``"gaussian"`` or ``"vmf"``.
    :type kernel: str
    :return: ``(m, weights)``: ``m`` is ``(..., N, Dv)`` in the dtype of ``q``;
        ``weights`` is ``(..., N, M)`` in the accumulation dtype.
    :rtype: Tuple[Any, Any]
    :raises ValueError: if ``kernel`` is unsupported or the mask rank is too high.
    """
    _check_kernel(kernel)
    q = keras.ops.convert_to_tensor(q)
    out_dtype, acc = _dtype_names(q)
    q = keras.ops.cast(q, acc)
    k = _to_acc(k, acc)
    v = k if v is None else _to_acc(v, acc)

    keep = _keep_mask(mask, _logit_rank(q, k))
    key_mask = None if keep is None else keras.ops.any(keep, axis=-2)
    logits = kernel_logits(q, k, bandwidth, kernel, key_mask=key_mask)
    if keep is None:
        weights = keras.ops.softmax(logits, axis=-1)
    else:
        # DECISION plan-2026-10-29T104646-27f045c5/D-027: masked weights come from a max over kept logits only.
        # Do not write a fixed sentinel into the logits: at a tiny bandwidth the kept ones are below it.
        expd, _ = _masked_exp(logits, keep)
        total = keras.ops.sum(expd, axis=-1, keepdims=True)
        any_kept = keras.ops.any(keep, axis=-1, keepdims=True)
        weights = expd / keras.ops.where(any_kept, total, keras.ops.ones_like(total))
        # DECISION plan-2026-10-29T104646-27f045c5/D-004: a row with no kept key gets uniform weights.
        # Do not leave it to the sentinel softmax or a kernel-weighted keep-all rescue. See decisions.md.
        uniform = keras.ops.ones_like(weights) / keras.ops.cast(
            keras.ops.shape(weights)[-1], acc
        )
        weights = keras.ops.where(any_kept, weights, uniform)

    # The weights sum to 1, so averaging the centred values and adding the centre back
    # is the same mean, without float32 rounding of values far from the origin.
    v_centre = _key_centre(v, key_mask)
    shifted = keras.ops.matmul(weights, v - v_centre) + v_centre
    return keras.ops.cast(shifted, out_dtype), weights


def kde_log_density(
    q: Any,
    k: Any,
    bandwidth: Any,
    mask: Optional[Any] = None,
) -> Any:
    """Gaussian kernel density estimate of the keys, as a log density at ``q``.

    ``log p(x) = logsumexp_j(-||x - x_j||^2 / (2 h^2)) - log N_valid
    - (d / 2) log(2 pi h^2)`` where ``N_valid`` counts the kept keys (at least 1).
    The query-norm term is kept: this is a density, not a softmax. Masked keys are
    excluded from the logsumexp without a fixed sentinel (the maximum is taken
    over the kept terms only), so the value is the density of the kept subset at
    any bandwidth. A row with no kept key treats every key as kept (``N_valid`` is
    then the key count), so it is finite.

    :param q: query points, tensor-like ``(..., N, D)``.
    :type q: Any
    :param k: keys, tensor-like ``(..., M, D)``.
    :type k: Any
    :param bandwidth: ``h``, as in :func:`kernel_logits`.
    :type bandwidth: Any
    :param mask: ``None`` or a keep-mask ``(..., M)``, as in :func:`mean_shift_step`.
    :type mask: Optional[Any]
    :return: log density ``(..., N)`` in the accumulation dtype.
    :rtype: Any
    """
    q = keras.ops.convert_to_tensor(q)
    _, acc = _dtype_names(q)
    k = _to_acc(k, acc)
    h = _as_dtype(bandwidth, acc)
    dim = q.shape[-1]

    keep = _keep_mask(mask, _logit_rank(q, k))
    key_mask = None if keep is None else keras.ops.any(keep, axis=-2)
    log_kernel = -pairwise_sq_dist(q, k, key_mask) / (2.0 * h * h)
    if keep is None:
        n_valid = keras.ops.cast(keras.ops.shape(log_kernel)[-1], acc)
        log_sum = keras.ops.logsumexp(log_kernel, axis=-1, keepdims=True)
    else:
        keep = _keep_all_rescue(keep)
        n_valid = keras.ops.sum(
            keras.ops.cast(keep, acc), axis=-1, keepdims=True
        )
        expd, row_max = _masked_exp(log_kernel, keep)
        log_sum = row_max + keras.ops.log(
            keras.ops.sum(expd, axis=-1, keepdims=True)
        )
    n_valid = keras.ops.maximum(n_valid, 1.0)

    log_norm = 0.5 * dim * (math.log(2.0 * math.pi) + 2.0 * keras.ops.log(h))
    return keras.ops.squeeze(
        log_sum - keras.ops.log(n_valid) - log_norm, axis=-1
    )


def mean_shift_iterate(
    x: Any,
    bandwidth: Any,
    num_steps: int,
    step_size: float = 1.0,
    blurring: bool = True,
    mask: Optional[Any] = None,
    kernel: str = "gaussian",
    return_weights: bool = False,
) -> Union[Any, Tuple[Any, Any]]:
    """Run ``num_steps`` damped mean-shift steps on a point set.

    Each step is ``z <- (1 - eta) z + eta m(z)``. Blurring uses the current ``z``
    as keys and values; non-blurring uses the fixed input ``x``. With the vMF
    kernel the inputs and every iterate are projected back onto the unit sphere.
    The mask applies to the keys at every step; masked query rows are computed
    like any other and stay finite, callers ignore them.

    :param x: tensor-like points ``(B, N, D)``.
    :type x: Any
    :param bandwidth: ``h``, as in :func:`kernel_logits`.
    :type bandwidth: Any
    :param num_steps: number of steps, a Python int >= 0; 0 returns ``x``.
    :type num_steps: int
    :param step_size: damping ``eta`` in the open interval (0, 2).
    :type step_size: float
    :param blurring: move the keys together with the points.
    :type blurring: bool
    :param mask: ``None`` or a keep-mask ``(B, N)``.
    :type mask: Optional[Any]
    :param kernel: ``"gaussian"`` or ``"vmf"``.
    :type kernel: str
    :param return_weights: also return the weights ``(B, N, N)`` of the last step.
    :type return_weights: bool
    :return: ``z`` ``(B, N, D)`` in the dtype of ``x``, or ``(z, weights)``.
    :rtype: Union[Any, Tuple[Any, Any]]
    :raises ValueError: for a bad ``kernel``, ``num_steps`` or ``step_size``, or
        for ``return_weights`` with ``num_steps == 0``.
    """
    _check_kernel(kernel)
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError(f"num_steps must be an int >= 0, got {num_steps!r}")
    if (
        isinstance(step_size, bool)
        or not isinstance(step_size, (int, float))
        or not 0.0 < step_size < 2.0
    ):
        raise ValueError(f"step_size must be a number in (0, 2), got {step_size!r}")
    if return_weights and num_steps == 0:
        raise ValueError("return_weights=True needs num_steps >= 1, got 0")

    x = keras.ops.convert_to_tensor(x)
    if num_steps == 0:
        return x

    _, acc = _dtype_names(x)
    if kernel == "vmf":
        x = keras.ops.cast(_normalize_rows(keras.ops.cast(x, acc)), x.dtype)
    z = x
    weights = None
    for _ in range(num_steps):
        keys = z if blurring else x
        moved, weights = mean_shift_step(z, keys, keys, bandwidth, mask, kernel)
        z = (1.0 - step_size) * z + step_size * moved
        if kernel == "vmf":
            z = keras.ops.cast(_normalize_rows(keras.ops.cast(z, acc)), x.dtype)
    if return_weights:
        return z, weights
    return z


def median_heuristic_bandwidth(x: Any, mask: Optional[Any] = None) -> Any:
    """Median pairwise distance among the valid points of every sample.

    Distances to oneself and any pair with a masked member are excluded. The
    median is taken by sorting a ``(B, N * N)`` vector whose excluded entries are
    pushed to a large value and gathering the middle element(s) with a flat-row
    ``take``, so it needs no data-dependent Python control flow and works with an
    unknown ``N``. A sample with fewer than two valid points gets 1.0. The result
    is floored at a small positive value so it is never 0, and carries no gradient.
    The distances are formed on points centred by the mean of the kept points, so
    a constant offset of ``x`` does not change the median beyond float32 rounding.

    :param x: tensor-like points ``(B, N, D)``.
    :type x: Any
    :param mask: ``None`` or a keep-mask ``(B, N)``.
    :type mask: Optional[Any]
    :return: bandwidth ``(B, 1, 1)`` in the dtype of ``x``, wrapped in
        ``stop_gradient``.
    :rtype: Any
    :raises ValueError: if ``x`` is not rank 3.
    """
    x = keras.ops.convert_to_tensor(x)
    if len(x.shape) != 3:
        raise ValueError(f"x must have shape (B, N, D), got rank {len(x.shape)}")
    out_dtype, acc = _dtype_names(x)
    x = keras.ops.cast(x, acc)

    keep = _keep_mask(mask, 2)
    dist = keras.ops.sqrt(pairwise_sq_dist(x, x, keep))
    size = keras.ops.shape(dist)[1]
    ids = keras.ops.arange(size)
    off_diagonal = keras.ops.expand_dims(
        keras.ops.not_equal(
            keras.ops.expand_dims(ids, axis=1), keras.ops.expand_dims(ids, axis=0)
        ),
        axis=0,
    )
    valid = keras.ops.logical_and(
        off_diagonal, keras.ops.ones_like(dist, dtype="bool")
    )
    if keep is not None:
        pair = keras.ops.logical_and(
            keras.ops.expand_dims(keep, axis=2), keras.ops.expand_dims(keep, axis=1)
        )
        valid = keras.ops.logical_and(valid, pair)

    length = size * size
    flat_valid = keras.ops.reshape(valid, (keras.ops.shape(dist)[0], length))
    flat_dist = keras.ops.reshape(dist, (keras.ops.shape(dist)[0], length))
    far = keras.ops.full_like(flat_dist, -mask_sentinel(acc))
    ordered = keras.ops.sort(keras.ops.where(flat_valid, flat_dist, far), axis=-1)

    count = keras.ops.sum(keras.ops.cast(flat_valid, "int32"), axis=-1)
    row = keras.ops.cumsum(keras.ops.ones_like(count), axis=0) - 1
    base = row * keras.ops.cast(length, "int32")
    flat_ordered = keras.ops.reshape(ordered, (-1,))
    lower = keras.ops.take(
        flat_ordered, base + keras.ops.maximum(count - 1, 0) // 2, axis=0
    )
    upper = keras.ops.take(flat_ordered, base + count // 2, axis=0)
    median = 0.5 * (lower + upper)

    median = keras.ops.where(count > 0, median, keras.ops.ones_like(median))
    median = keras.ops.maximum(median, stability_floor(out_dtype, _MEDIAN_FLOOR))
    median = keras.ops.reshape(median, (-1, 1, 1))
    return keras.ops.stop_gradient(keras.ops.cast(median, out_dtype))
