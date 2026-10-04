"""HarMax: harmonic-max normalization of distances onto the probability simplex.

Baek, Liu, Tyagi, Tegmark. "Harmonic Loss Trains Interpretable AI Models",
TMLR 2025.

Softmax gives every logit a non-zero share via ``exp(z)``. HarMax instead
takes non-negative **distances** ``d`` and normalizes inverse powers::

    HarMax(d)_i = d_i^-n / sum_j d_j^-n == softmax(-n * log d)_i

so harmonic loss is sparse categorical crossentropy on logits ``-n * log d``.
The ``logits`` form (see :class:`HarmonicDense` with
``output_mode="logits"``) is the numerically stable training path; this layer
is the drop-in ``Softmax`` analog for the ``distances`` form.

The computation runs as ``softmax(-n * log(max(d, eps)))`` along ``axis``.
Outputs are non-negative, sum to 1 along ``axis``, and preserve the input
shape and dtype. The layer owns no weights.

References:
    - Baek et al., 2025. "Harmonic Loss Trains Interpretable AI Models".
"""

import keras
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import axis_is_in_range
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.harmax")
class HarMax(keras.layers.Layer):
    """Normalize non-negative distances into probabilities via inverse powers.

    Computes ``softmax(-n * log(max(d, epsilon)))`` along ``axis``. Smaller
    distances receive larger probabilities; ``n`` controls the sharpness
    (larger ``n`` is closer to argmin). Output shape and dtype equal the
    input's, and the layer owns no weights.

    **Architecture Overview:**

    .. code-block:: text

        d  distances  [..., K]   `axis` selects the K dimension
                      ▼
        ┌───────────────────────────┐
        │ clip to [epsilon, +inf)   │
        └─────────────┬─────────────┘
                      ▼  [..., K]
        ┌───────────────────────────┐
        │ logits = -n * log(d)      │
        └─────────────┬─────────────┘
                      ▼  [..., K]
        ┌───────────────────────────┐
        │ softmax(logits, axis)     │
        └─────────────┬─────────────┘
                      ▼
        p  [..., K]   sums to 1 along axis

    ``K`` is the size of ``axis``. Negative inputs are not rejected; they
    clip to ``epsilon`` like exact zeros do, because the contract is
    distances (``>= 0``) and the clip is the documented out-of-contract
    behavior rather than a branch.

    :param n: Harmonic exponent. Must be positive. Defaults to 1.0.
    :type n: float
    :param epsilon: Floor for distances before the logarithm. Must be
        positive. Defaults to 1e-8.
    :type epsilon: float
    :param axis: Axis to normalize along. Defaults to -1. The valid range,
        ``[-ndim, ndim - 1]``, depends on the rank of the tensor the layer is
        called on, so it can only be checked at call time. ``__init__``
        checks the type; ``call`` and ``compute_output_shape`` check the
        range, with identical predicates.
    :type axis: int
    :param kwargs: Additional keyword arguments passed to the Layer base
        class.

    :raises ValueError: If ``n`` or ``epsilon`` is not positive, or if
        ``axis`` is not an ``int`` or is a ``bool``. Raised from
        ``__init__``.
    """

    def __init__(
            self,
            n: float = 1.0,
            epsilon: float = 1e-8,
            axis: int = -1,
            **kwargs: Any
    ) -> None:
        """Validate the scalars and store the configuration.

        No weight is created here; the layer is stateless.

        :param n: Harmonic exponent. Must be positive. Defaults to 1.0.
        :type n: float
        :param epsilon: Floor for distances before the logarithm. Must be
            positive. Defaults to 1e-8.
        :type epsilon: float
        :param axis: Axis to normalize along. Defaults to -1.
        :type axis: int
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``n`` or ``epsilon`` is not positive, or if
            ``axis`` is not an integer or is a bool. The RANGE of ``axis``
            depends on the input rank and is therefore validated in
            :meth:`call`, not here.
        """
        super().__init__(**kwargs)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError(f"axis must be an integer, got {type(axis).__name__}")
        n = float(n)
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.n = n
        self.epsilon = float(epsilon)
        self.axis = axis

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Normalize distances into probabilities along ``axis``.

        :param inputs: Distances, any shape and rank. Values below
            ``epsilon`` (including negatives) clip to ``epsilon``.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Non-negative tensor of the same shape as ``inputs``,
            summing to 1 along ``axis``.
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
        clipped = keras.ops.maximum(inputs, self.epsilon)
        logits = -self.n * keras.ops.log(clipped)
        return keras.ops.softmax(logits, axis=self.axis)

    def compute_output_shape(
            self,
            input_shape: tuple
    ) -> tuple:
        """Return the input shape unchanged, after the same axis check.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        :return: Output shape tuple, identical to input.
        :rtype: tuple
        :raises ValueError: If ``axis`` is out of range for ``input_shape``'s
            rank. The range is the same one :meth:`call` enforces, because
            both call ``common.axis_is_in_range``.
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
            "n": self.n,
            "epsilon": self.epsilon,
            "axis": self.axis,
        })
        return config

# ---------------------------------------------------------------------
