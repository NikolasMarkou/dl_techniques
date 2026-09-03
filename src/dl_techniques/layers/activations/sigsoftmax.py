"""
Sigsoftmax: softmax with a sigmoid factor added to the numerator.

The problem it solves is the softmax bottleneck. A classifier or language
model computes its logits as ``z = W h(x)``, an affine map of a
``d``-dimensional context vector. Softmax normalises ``exp(z)``, and
``log(exp(z))`` is just ``z``, so the log-probabilities it can produce all
lie in a vector space of dimension at most ``d + 1``. When the true
log-probability matrix has higher rank, no amount of training closes the gap;
the output layer is the ceiling. Widening ``d`` is the usual remedy and costs
parameters in the largest matrix in the model.

``sigsoftmax`` and ``log_sigsoftmax`` normalise ``exp(z) * sigmoid(z)``
instead. Its logarithm is ``2z - softplus(z)``, which is not linear in ``z``,
so the reachable log-probabilities leave that subspace. The paper also shows
the range of softmax is a subset of sigsoftmax's when the all-ones vector
lies in the input space, so nothing representable is lost, and no parameters
are added.

The computation runs entirely in log space:

    log_sigsoftmax(z) = w - logsumexp(w),  where w = z + log_sigmoid(z)

``sigsoftmax`` is ``exp(log_sigsoftmax)`` and the ``SigSoftmax`` layer wraps
it. Use ``log_sigsoftmax`` directly in a cross-entropy loss. The reduction
widens float16 and bfloat16 to float32 and casts back, which keeps a float16
row of large logits off ``nan``; see :func:`log_sigsoftmax` for the two range
limits.

References:
    - Kanai et al., 2018. Sigsoftmax: Reanalysis of the Softmax Bottleneck.
      (https://arxiv.org/abs/1805.10829)
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import axis_is_in_range
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

    # DECISION plan-2026-09-03T085145-3384c4dc/D-008: the widen above is not a precision tweak.
    # Deleting it returns nan on a float16 row past |z| ~ 32752, since w = 2z. See decisions.md.

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
    finite over the range described below.

    Two separate limits bound the range, and they are not the same limit.

    In float32 and float64 the reduction runs in the input dtype, so the
    intermediate ``2z`` overflows once ``|z|`` passes half the dtype's
    maximum, 1.7e38 in float32. One lane past it saturates to ``-inf``, the
    correct limit while another lane is in range; a row whose entries all pass
    it returns ``nan``.

    In float16 and bfloat16 the reduction is widened to float32, so ``2z``
    never overflows and the result is never ``nan``. What saturates is the
    cast back: an output entry is about ``2 * (z_i - max z)``, so the limit
    falls on the row's SPREAD, not on ``|z|``. A float16 row spanning more
    than about 32752 loses its far lane to ``-inf`` while a row of large but
    close logits, such as ``[-40000, -35000, -33000]``, comes back finite.

    Takes a tensor and returns a tensor. It does not validate ``axis`` or
    coerce its input: an out-of-range ``axis`` surfaces as the backend's own
    reduction error, and a Python list has no ``dtype``. Use
    :class:`SigSoftmax` for the validated form.

    :param x: Logits, any shape and rank.
    :type x: Any
    :param axis: Axis to normalise along. Defaults to ``-1``. Not range-checked.
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

    Outputs are non-negative and sum to 1 along ``axis``. Values below the
    dtype's smallest subnormal round to zero, and whether that happens is a
    property of the device: ``exp(-100)`` in float32 is 3.8e-44 on some
    backends and exactly 0.0 on others. Use :func:`log_sigsoftmax` where the
    small end matters.

    Takes a tensor and returns a tensor. It does not validate ``axis`` or
    coerce its input; see :func:`log_sigsoftmax`. Use :class:`SigSoftmax` for
    the validated form.

    :param x: Logits, any shape and rank.
    :type x: Any
    :param axis: Axis to normalise along. Defaults to ``-1``. Not range-checked.
    :type axis: int
    :return: Probabilities, same shape and dtype as ``x``.
    :rtype: Any
    """
    log_probabilities, input_dtype = _log_sigsoftmax_widened(x, axis)
    return keras.ops.cast(keras.ops.exp(log_probabilities), input_dtype)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.sigsoftmax")
class SigSoftmax(keras.layers.Layer):
    """Normalise ``exp(z) * sigmoid(z)`` along one axis.

    A stateless wrapper over :func:`sigsoftmax`. Outputs are non-negative and
    sum to 1 along ``axis``. Output shape and dtype equal the input's, and the
    layer owns no weights. The sigmoid factor makes the logarithm of the
    output non-linear in ``z``, which is what separates it from softmax.

    Architecture:

    .. code-block:: text

        z  logits  [..., K]
                      ▼
        ┌───────────────────────────┐
        │ cast to reduction dtype   │  dtype island
        └─────────────┬─────────────┘
                      ▼  [..., K]
        ┌───────────────────────────┐
        │ w = z + log_sigmoid(z)    │
        └──────┬─────────────┬──────┘
               │             ▼  [..., K]
               │  ┌─────────────────────┐
               │  │ logsumexp on axis   │
               │  └──────────┬──────────┘
               ▼             ▼  [..., 1]
        ┌───────────────────────────┐
        │ subtract                  │
        └─────────────┬─────────────┘
                      ▼  [..., K]  log probabilities
        ┌───────────────────────────┐
        │ exp                       │
        └─────────────┬─────────────┘
                      ▼  [..., K]
        ┌───────────────────────────┐
        │ cast back to input dtype  │  dtype island
        └─────────────┬─────────────┘
                      ▼
        p  [..., K]   sums to 1 along axis

    ``K`` is the size of ``axis``. The two dtype-island stages are a pair:
    float16 and bfloat16 widen to float32 for the reduction, float32 and
    float64 pass through unchanged, and the cast back restores the input
    dtype.

    :param axis: Axis to normalise along. Defaults to -1. The valid range,
        ``[-ndim, ndim - 1]``, depends on the rank of the tensor the layer is
        called on, so it can only be checked at call time. ``__init__``
        checks the type; ``call`` and ``compute_output_shape`` check the
        range, with identical predicates.
    :type axis: int
    :param kwargs: Additional keyword arguments passed to the Layer base
        class.

    :raises ValueError: If ``axis`` is not an ``int``, or is a ``bool``.
        Raised from ``__init__``.

    ``supports_masking`` is left at ``False``, matching :class:`Sparsemax`.
    Propagating a mask unchanged would be right only when ``axis`` is not the
    masked axis; when it is, masked positions still enter the reduction and
    the mask would have to zero them first. Applying a mask is the caller's
    job, before the layer.

    Input shape:
        Arbitrary, rank 1 or higher. ``axis`` must address one of its
        dimensions.

    Output shape:
        Same as the input shape.
    """

    def __init__(
            self,
            axis: int = -1,
            **kwargs: Any
    ) -> None:
        """Validate the type of ``axis`` and store it.

        :param axis: Axis to normalise along.
        :type axis: int
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``axis`` is not an integer, or is a bool. The
            range of ``axis`` depends on the input rank and is therefore
            validated in :meth:`call`, not here.
        """
        super().__init__(**kwargs)
        # `bool` is a subclass of `int`, so `isinstance(True, int)` is True and
        # an unguarded `SigSoftmax(axis=True)` would behave as `axis=1`.
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError(f"axis must be an integer, got {type(axis).__name__}")
        self.axis = axis

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Return the sigsoftmax probabilities of ``inputs`` along ``axis``.

        :param inputs: Logits, any shape and rank.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Non-negative tensor of the same shape and dtype as
            ``inputs``, summing to 1 along ``axis``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``axis`` is out of range for the rank of
            ``inputs``, i.e. outside ``[-ndim, ndim - 1]``.
        """
        ndim = len(inputs.shape)
        if not axis_is_in_range(self.axis, ndim):
            raise ValueError(
                f"axis={self.axis} is out of range for an input of rank "
                f"{ndim} (shape {tuple(inputs.shape)}); axis must be in "
                f"[{-ndim}, {ndim - 1}]"
            )
        return sigsoftmax(inputs, self.axis)

    def compute_output_shape(
            self,
            input_shape: tuple
    ) -> tuple:
        """Return the input shape unchanged, after the same axis check.

        Reads only ``self.axis``, so it answers correctly on an unbuilt layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple, identical to the input.
        :rtype: tuple
        :raises ValueError: If ``axis`` is out of range for ``input_shape``'s
            rank. The range is the one :meth:`call` enforces, because both
            read the same ``common.axis_is_in_range`` predicate.
        """
        ndim = len(input_shape)
        if not axis_is_in_range(self.axis, ndim):
            raise ValueError(
                f"axis={self.axis} is out of range for an input of rank "
                f"{ndim} (shape {tuple(input_shape)}); axis must be in "
                f"[{-ndim}, {ndim - 1}]"
            )
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

# ---------------------------------------------------------------------
