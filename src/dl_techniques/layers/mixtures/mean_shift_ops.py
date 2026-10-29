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
(``utils.dtype_policy.accumulation_dtype``), a masked key has exactly zero weight,
and a row with no kept key never produces NaN. The module is not exported from
``layers.mixtures`` and registers nothing.

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


def pairwise_sq_dist(a: Any, b: Any) -> Any:
    """Squared Euclidean distance between every row of ``a`` and every row of ``b``.

    Uses the expanded form ``||a||^2 - 2 a.b + ||b||^2`` clamped at 0, so the
    result is never negative and the diagonal of ``pairwise_sq_dist(a, a)`` is
    zero to float tolerance. The math runs in at least float32 and the result is
    returned in that accumulation dtype; callers cast back if they need to.

    ``utils.tensors.pairwise_squared_distance`` is not reused: it takes centers
    of rank 2 only, one shared set for all batch rows, and subtracts directly,
    which would materialise an ``(..., N, M, D)`` tensor for per-batch keys.

    :param a: tensor ``(..., N, D)``.
    :type a: Any
    :param b: tensor ``(..., M, D)`` whose leading axes broadcast with ``a``.
    :type b: Any
    :return: squared distances ``(..., N, M)`` in the accumulation dtype.
    :rtype: Any
    """
    a = keras.ops.convert_to_tensor(a)
    _, acc = _dtype_names(a)
    a = keras.ops.cast(a, acc)
    b = keras.ops.cast(b, acc)
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
) -> Any:
    """Log of the unnormalised kernel weight of every (query, key) pair.

    Gaussian: ``q.k / h^2 - ||k||^2 / (2 h^2)``, minus ``||q||^2 / (2 h^2)`` when
    ``drop_query_norm`` is False. The query-norm term is constant along the key
    axis, so a softmax over the keys is the same with or without it. vMF:
    ``q.k / h^2`` on L2-normalised ``q`` and ``k`` (zero rows stay zero, no NaN);
    ``drop_query_norm`` has no effect. The bandwidth is not clipped here.

    :param q: queries ``(..., N, D)``.
    :type q: Any
    :param k: keys ``(..., M, D)``.
    :type k: Any
    :param bandwidth: ``h``; a scalar, ``(B, 1, 1)``, ``(B, H, 1, 1)`` or per-query
        ``(..., N, 1)``, aligned from the right with the logits ``(..., N, M)``.
    :type bandwidth: Any
    :param kernel: ``"gaussian"`` or ``"vmf"``.
    :type kernel: str
    :param drop_query_norm: omit the per-query constant of the Gaussian kernel.
    :type drop_query_norm: bool
    :return: logits ``(..., N, M)`` in the accumulation dtype.
    :rtype: Any
    :raises ValueError: if ``kernel`` is not a supported name.
    """
    _check_kernel(kernel)
    q = keras.ops.convert_to_tensor(q)
    _, acc = _dtype_names(q)
    q = keras.ops.cast(q, acc)
    k = keras.ops.cast(k, acc)
    h = _as_dtype(bandwidth, acc)
    inv_h2 = 1.0 / (h * h)

    if kernel == "vmf":
        q = _normalize_rows(q)
        k = _normalize_rows(k)
        return keras.ops.matmul(q, keras.ops.swapaxes(k, -1, -2)) * inv_h2

    cross = keras.ops.matmul(q, keras.ops.swapaxes(k, -1, -2))
    k_sq = keras.ops.expand_dims(
        keras.ops.sum(keras.ops.square(k), axis=-1), axis=-2
    )
    logits = cross * inv_h2 - 0.5 * k_sq * inv_h2
    if not drop_query_norm:
        q_sq = keras.ops.sum(keras.ops.square(q), axis=-1, keepdims=True)
        logits = logits - 0.5 * q_sq * inv_h2
    return logits


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
    key is removed by ``where`` selection on the logits and again on the weights,
    so its weight is exactly 0.0. A row with no kept key gets uniform weights over
    all keys, so its output is the plain mean of ``v``. For the vMF kernel only the
    logits normalise ``q`` and ``k``; ``v`` is used as given.

    :param q: queries ``(..., N, D)``.
    :type q: Any
    :param k: keys ``(..., M, D)``.
    :type k: Any
    :param v: values ``(..., M, Dv)``, or ``None`` to use ``k``.
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
    k = keras.ops.cast(k, acc)
    v = k if v is None else keras.ops.cast(v, acc)

    logits = kernel_logits(q, k, bandwidth, kernel)
    keep = _keep_mask(mask, len(logits.shape))
    if keep is None:
        weights = keras.ops.softmax(logits, axis=-1)
    else:
        sentinel = keras.ops.full_like(logits, mask_sentinel(acc))
        weights = keras.ops.softmax(keras.ops.where(keep, logits, sentinel), axis=-1)
        weights = keras.ops.where(keep, weights, keras.ops.zeros_like(weights))
        any_kept = keras.ops.any(keep, axis=-1, keepdims=True)
        # DECISION plan-2026-10-29T104646-27f045c5/D-004: a row with no kept key gets uniform weights.
        # Do not leave it to the sentinel softmax or a kernel-weighted keep-all rescue. See decisions.md.
        uniform = keras.ops.ones_like(weights) / keras.ops.cast(
            keras.ops.shape(weights)[-1], acc
        )
        weights = keras.ops.where(any_kept, weights, uniform)

    shifted = keras.ops.matmul(weights, v)
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
    excluded from the logsumexp by ``where`` selection. A row with no kept key
    treats every key as kept (``N_valid`` is then the key count), so it is finite.

    :param q: query points ``(..., N, D)``.
    :type q: Any
    :param k: keys ``(..., M, D)``.
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
    h = _as_dtype(bandwidth, acc)
    dim = q.shape[-1]

    log_kernel = -pairwise_sq_dist(q, k) / (2.0 * h * h)
    keep = _keep_mask(mask, len(log_kernel.shape))
    if keep is None:
        n_valid = keras.ops.cast(keras.ops.shape(log_kernel)[-1], acc)
    else:
        keep = _keep_all_rescue(keep)
        sentinel = keras.ops.full_like(log_kernel, mask_sentinel(acc))
        log_kernel = keras.ops.where(keep, log_kernel, sentinel)
        n_valid = keras.ops.sum(
            keras.ops.cast(keep, acc), axis=-1, keepdims=True
        )
    n_valid = keras.ops.maximum(n_valid, 1.0)

    log_sum = keras.ops.logsumexp(log_kernel, axis=-1, keepdims=True)
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

    :param x: points ``(B, N, D)``.
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

    :param x: points ``(B, N, D)``.
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

    dist = keras.ops.sqrt(pairwise_sq_dist(x, x))
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
    keep = _keep_mask(mask, 2)
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
