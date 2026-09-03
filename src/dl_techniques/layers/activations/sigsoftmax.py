"""Sigsoftmax: softmax with a sigmoid factor added to the numerator.

``sigsoftmax`` and ``log_sigsoftmax`` normalise ``exp(z) * sigmoid(z)`` along
one axis. Softmax normalises ``exp(z)``, whose logarithm is linear in ``z``.
Sigsoftmax normalises ``exp(z) * sigmoid(z)``, whose logarithm carries a
softplus term and is not linear, which lifts the output out of the
rank-limited subspace softmax occupies.

The computation runs entirely in log space:

    log_sigsoftmax(z) = w - logsumexp(w),  where w = z + log_sigmoid(z)

``sigsoftmax`` is ``exp(log_sigsoftmax)``. The reduction widens float16 and
bfloat16 inputs to float32 and casts back, so the output carries the input
dtype and sums to 1 along the reduction axis.

References:
    - Kanai et al., 2018. Sigsoftmax: Reanalysis of the Softmax Bottleneck.
      (https://arxiv.org/abs/1805.10829)
"""

import keras
from typing import Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _log_sigsoftmax_widened(
        inputs: Any,
        axis: int
) -> Tuple[Any, str]:
    """Compute log-sigsoftmax in a dtype at least as wide as the input's.

    Returns the log-probabilities still in the widened dtype together with the
    dtype the caller should cast back to.

    :param inputs: Logits, any shape and rank.
    :type inputs: Any
    :param axis: Axis to normalise along. Not range-checked here.
    :type axis: int
    :return: Log-probabilities in the reduction dtype, and the input dtype name.
    :rtype: Tuple[Any, str]
    """
    input_dtype = getattr(inputs.dtype, "name", None) or str(inputs.dtype)
    reduction_dtype = (
        "float32" if input_dtype in ("float16", "bfloat16") else input_dtype
    )
    z = keras.ops.cast(inputs, reduction_dtype)

    # DECISION plan-2026-09-03T085145-3384c4dc/D-001: log space, not sparsemax's max shift.
    # sigmoid(z) does not shift with the max, so an all-negative row is 0/0 = NaN. See decisions.md.
    w = z + keras.ops.log_sigmoid(z)
    log_probabilities = w - keras.ops.logsumexp(w, axis=axis, keepdims=True)
    return log_probabilities, input_dtype


@register_dl_technique("dl_techniques.layers.activations.sigsoftmax")
def log_sigsoftmax(
        x: Any,
        axis: int = -1
) -> Any:
    """Return the log-probabilities of sigsoftmax along ``axis``.

    Use this rather than ``log(sigsoftmax(x))`` in a cross-entropy loss: the
    logarithm of an underflowed probability is ``-inf``, while this stays
    finite wherever the input is finite.

    :param x: Logits, any shape and rank.
    :type x: Any
    :param axis: Axis to normalise along. Defaults to ``-1``.
    :type axis: int
    :return: Log-probabilities, same shape and dtype as ``x``.
    :rtype: Any
    """
    log_probabilities, input_dtype = _log_sigsoftmax_widened(x, axis)
    return keras.ops.cast(log_probabilities, input_dtype)


@register_dl_technique("dl_techniques.layers.activations.sigsoftmax")
def sigsoftmax(
        x: Any,
        axis: int = -1
) -> Any:
    """Return the sigsoftmax probabilities along ``axis``.

    Outputs are non-negative and sum to 1 along ``axis``.

    :param x: Logits, any shape and rank.
    :type x: Any
    :param axis: Axis to normalise along. Defaults to ``-1``.
    :type axis: int
    :return: Probabilities, same shape and dtype as ``x``.
    :rtype: Any
    """
    log_probabilities, input_dtype = _log_sigsoftmax_widened(x, axis)
    return keras.ops.cast(keras.ops.exp(log_probabilities), input_dtype)

# ---------------------------------------------------------------------
