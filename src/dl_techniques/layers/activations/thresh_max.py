"""
A softmax variant that suppresses low-confidence classes.

A plain softmax gives every class some probability, including ones the model
clearly rejects. ThreshMax runs a softmax, then multiplies each probability
by a smooth gate that opens above the uniform value ``1/N`` and closes below
it, then renormalizes so the row sums to 1 again.

The gate is ``g = 0.5 * (tanh(slope * (p - 1/N)) + 1)``, applied
element-wise. It is a smooth stand-in for a hard "is this class above
average?" test, which is what makes the layer differentiable.

**"Sparse" here means suppressed, not zero.** ``tanh`` never reaches -1, so
``g`` is never 0 and no output is ever exactly 0. Measured on logits
``[3, 1, 2, 0.5, -1]`` at the default ``slope=10``: the smallest softmax
probability, 1.142147e-02, becomes 3.288895e-04 -- a 34.7x suppression, not
a zero. If you need exact zeros use ``Sparsemax``.

Class order is preserved, because ``g`` is increasing in ``p`` and the
renormalization is a single positive divisor. Relative *magnitudes* are not
preserved: on the same input the ratio between the largest and second
largest probability goes from 2.71828 to 4.22701. The gate stretches the
distribution as well as suppressing its tail.

Uniform input is a fixed point. Every ``p`` equals ``1/N``, so every gate is
exactly 0.5, and renormalization divides the constant back out. Measured on
7 equal logits: the output is 7 copies of 0.14285715. ThreshMax cannot
sparsify a maximum-entropy distribution.

There is no special case for that in the code and none is needed, but the
bound depends on ``slope``. At ``slope=10``, the default, the gated sum before
renormalization bottoms out at 0.500000, hit exactly at uniform input
(measured minimum over 120,000 random logit draws, ``N`` from 2 to 200, logit
scale 0.01 to 5). A larger slope goes lower. At ``slope=50``, the top of the
default ``ValueRangeConstraint(1.0, 50.0)`` and the direction the negative
``L2_custom(-1e-4)`` pushes a trainable slope, a grid search over
``k``-elevated logit families for ``N`` in 2 to 59 bottoms out at 0.34528
(``N=6``, one logit raised by 0.74). That is still more than eleven orders of
magnitude above the default ``epsilon=1e-12`` (the ratio is 3.45e11), so the
denominator never approaches it at any slope the constraint allows.

References:
    - Sparse softmax variants, of which ``Sparsemax`` is the L2-projection
      relative in this same package.
    - Differentiable relaxations: replacing a hard step by ``tanh`` so a
      threshold can be trained through.

"""

import keras
from typing import Optional, Any, Tuple, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.regularizers.l2_custom import L2_custom
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint


# ---------------------------------------------------------------------
# Keras layer implementation
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ThreshMax(keras.layers.Layer):
    """Softmax, then a smooth gate against the uniform probability ``1/N``.

    Computes ``softmax(x)``, multiplies it by
    ``g = 0.5 * (tanh(slope * (p - 1/N)) + 1)``, and renormalizes. Classes
    scoring above ``1/N`` keep most of their mass; classes below it lose most
    of theirs. Output shape equals input shape and each row sums to 1.

    Class order survives, because ``g`` increases with ``p``. Exact zeros do
    not appear: ``tanh`` never reaches -1. This suppresses a tail, it does
    not prune it. Use ``Sparsemax`` if you need true zeros.

    The layer always creates one scalar weight, ``slope``, whether or not
    ``trainable_slope`` is set. With ``trainable_slope=False`` it is a
    non-trainable constant.

    **Architecture Overview:**

    .. code-block:: text

                      x  logits  [B, ..., N]
                                  │
                                  ▼
            ┌───────────────────────────────────┐
            │ p = softmax(x, axis)              │
            └─────────────────┬─────────────────┘
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │ p (unchanged)     │   │ d = p - 1/N       │
        │                   │   │ g = (tanh(slope   │
        │                   │   │       * d) + 1)/2 │
        └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │ g in (0, 1)
                  └───────────┬───────────┘
                              │  p_gated = p * g
                              ▼
            ┌───────────────────────────────────┐
            │ p_gated / (sum(p_gated) + eps)    │
            └─────────────────┬─────────────────┘
                              ▼
                      y  [B, ..., N]

    ``N`` is the size of the axis given by ``axis``, read from the input at
    call time. Both branches read the same ``p``; the left one is the tensor
    itself, not a sub-layer.

    **What ``slope`` does, measured at the defaults:** larger ``slope`` makes
    the gate a harder step around ``1/N``, so more mass moves to the classes
    above it. ``slope`` is clipped to ``[1.0, 50.0]`` and regularized with
    ``L2_custom(-1e-4)``. The coefficient is negative on purpose: it rewards
    a larger slope, so a trainable slope drifts toward a harder threshold
    over training. Do not flip that sign.

    :param axis: Axis to normalize and gate over. Defaults to -1.
    :type axis: int
    :param slope: Starting steepness of the gate. Must be positive. Defaults
        to 10.0. See the note below on how it interacts with
        ``slope_initializer``.
    :type slope: float
    :param epsilon: Added to the renormalization denominator. Must be
        positive. Defaults to 1e-12.
    :type epsilon: float
    :param trainable_slope: Whether the ``slope`` weight is trained.
        Defaults to False.
    :type trainable_slope: bool
    :param slope_initializer: Initializer for the ``slope`` weight. Defaults
        to ``"ones"``.
    :type slope_initializer: Union[str, keras.initializers.Initializer]
    :param slope_regularizer: Regularizer for the ``slope`` weight. ``None``
        resolves to ``L2_custom(-1e-4)``.
    :type slope_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param slope_constraint: Constraint on the ``slope`` weight. ``None``
        resolves to ``ValueRangeConstraint(1.0, 50.0)``.
    :type slope_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param kwargs: Additional keyword arguments passed to the Layer base
        class.

    :raises ValueError: If ``epsilon`` or ``slope`` is not positive. Raised
        from ``__init__``.

    :ivar slope_weight: The scalar slope. ``None`` until ``build`` runs.
    :vartype slope_weight: Optional[keras.Variable]

    Note:
        ``slope`` and ``slope_initializer`` fight, and the initializer wins.
        ``build`` only honours ``slope`` when the initializer is the default
        ``Ones``; any other initializer sets the weight and ``slope`` is
        ignored for the initial value while still being written to
        ``get_config``. Measured: ``ThreshMax(slope=10.0)`` builds a weight of
        10.0, ``ThreshMax(slope=10.0, slope_initializer="ones")`` also builds
        10.0, but
        ``ThreshMax(slope=10.0, slope_initializer=Constant(2.0))`` builds 2.0.
        Set one or the other, not both.
    """

    def __init__(
            self,
            axis: int = -1,
            slope: float = 10.0,
            epsilon: float = 1e-12,
            trainable_slope: bool = False,
            slope_initializer: Union[str, keras.initializers.Initializer] = "ones",
            # None resolves to L2_custom(-1e-4) in the body.
            slope_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            # None resolves to ValueRangeConstraint(1.0, 50.0) in the body.
            slope_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the scalars and store the configuration.

        No weight is created here; ``build`` does that. ``slope_regularizer``
        and ``slope_constraint`` default to ``None`` in the signature and are
        resolved in the body, because a mutable default in a signature is
        evaluated once at import and would be shared by every layer in the
        process.

        :param axis: Axis to normalize and gate over. Defaults to -1.
        :type axis: int
        :param slope: Starting steepness of the gate. Must be positive.
            Defaults to 10.0. Stored as ``slope_initial_value``; only used
            by ``build`` when ``slope_initializer`` is the default ``Ones``.
        :type slope: float
        :param epsilon: Added to the renormalization denominator. Must be
            positive. Defaults to 1e-12.
        :type epsilon: float
        :param trainable_slope: Whether the ``slope`` weight is trained.
            Defaults to False.
        :type trainable_slope: bool
        :param slope_initializer: Initializer for the ``slope`` weight.
            Defaults to ``"ones"``. Overrides ``slope`` when it is anything
            other than ``Ones``.
        :type slope_initializer: Union[str, keras.initializers.Initializer]
        :param slope_regularizer: Regularizer for the ``slope`` weight.
            ``None`` resolves to ``L2_custom(-1e-4)``.
        :type slope_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param slope_constraint: Constraint on the ``slope`` weight. ``None``
            resolves to ``ValueRangeConstraint(1.0, 50.0)``.
        :type slope_constraint: Optional[Union[str, keras.constraints.Constraint]]
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``epsilon`` or ``slope`` is not positive.
        """
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if slope <= 0:
            raise ValueError(f"slope must be positive, got {slope}")

        # Resolved here, not in the signature: see this method's docstring.
        if slope_regularizer is None:
            slope_regularizer = L2_custom(-1e-4)
        if slope_constraint is None:
            slope_constraint = ValueRangeConstraint(1.0, 50.0)

        self.axis = axis
        self.slope_initial_value = float(slope)
        self.epsilon = float(epsilon)
        self.trainable_slope = trainable_slope
        self.slope_initializer = keras.initializers.get(slope_initializer)
        self.slope_regularizer = keras.regularizers.get(slope_regularizer)
        self.slope_constraint = keras.constraints.get(slope_constraint)
        self.slope_weight = None

        logger.info(
            f"Initialized ThreshMax(axis={axis}, slope={slope}, "
            f"trainable_slope={trainable_slope})"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the scalar ``slope`` weight.

        The weight is created whether or not ``trainable_slope`` is set; the
        flag only decides whether the optimizer may move it. The constraint
        and regularizer apply either way.

        The initializer is swapped for ``Constant(slope)`` only when the
        caller left ``slope_initializer`` at its ``Ones`` default and asked
        for a ``slope`` other than 1.0. Pass a real initializer and ``slope``
        is ignored for the weight's value.

        :param input_shape: Shape of the input tensor. Unused except to
            forward to the base class; the weight is a scalar.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        init = self.slope_initializer
        is_default_ones = (
                isinstance(init, keras.initializers.Ones) or
                (isinstance(init, str) and init == 'ones') or
                (hasattr(init, 'get_config') and init.get_config().get('class_name') == 'Ones')
        )

        if is_default_ones and self.slope_initial_value != 1.0:
            init = keras.initializers.Constant(self.slope_initial_value)

        self.slope_weight = self.add_weight(
            name='slope',
            shape=(),
            initializer=init,
            regularizer=self.slope_regularizer,
            constraint=self.slope_constraint,
            trainable=self.trainable_slope
        )

        super().build(input_shape)

    @staticmethod
    def _differentiable_step(
            x: keras.KerasTensor,
            slope: Union[float, keras.KerasTensor] = 1.0,
            shift: Union[float, keras.KerasTensor] = 0.0
    ) -> keras.KerasTensor:
        """Smooth stand-in for a Heaviside step.

        Computes ``(tanh(slope * (x - shift)) + 1) / 2``. The result is in
        the open interval ``(0, 1)`` -- it approaches the endpoints but never
        reaches them, which is why ThreshMax produces no exact zeros.

        :param x: Input tensor.
        :type x: keras.KerasTensor
        :param slope: Steepness. Larger is closer to a hard step.
        :type slope: Union[float, keras.KerasTensor]
        :param shift: Where the output is 0.5.
        :type shift: Union[float, keras.KerasTensor]
        :return: Tensor of the same shape as ``x``, values in ``(0, 1)``.
        :rtype: keras.KerasTensor
        """
        # Python scalars are converted to x's dtype so the arithmetic below
        # broadcasts instead of promoting.
        if isinstance(slope, (int, float)):
            slope = keras.ops.convert_to_tensor(slope, dtype=x.dtype)
        if isinstance(shift, (int, float)):
            shift = keras.ops.convert_to_tensor(shift, dtype=x.dtype)

        scaled_shifted_x = slope * (x - shift)
        return (keras.ops.tanh(scaled_shifted_x) + 1.0) / 2.0

    def _compute_threshmax(
            self,
            x: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Run the whole ThreshMax computation.

        Softmax, gate on the deviation from ``1/N``, multiply, renormalize.

        :param x: Logits.
        :type x: keras.KerasTensor
        :return: Probabilities summing to 1 over ``self.axis``. Suppressed,
            not zeroed, below ``1/N``.
        :rtype: keras.KerasTensor
        """
        y_soft = keras.activations.softmax(x, axis=self.axis)

        # N is read from the tensor, so it can vary between calls.
        num_classes = keras.ops.shape(x)[self.axis]
        uniform_prob = 1.0 / keras.ops.cast(num_classes, x.dtype)
        confidence_diff = y_soft - uniform_prob

        gate = self._differentiable_step(confidence_diff, slope=self.slope_weight, shift=0.0)

        # Multiplicative, not a hard mask: the gate is increasing in p, so
        # class ORDER survives. Ratios between classes do not.
        y_stepped = y_soft * gate

        # No degenerate-case branch needed. The gated sum's floor depends on
        # slope: measured 0.500000 at slope=10 (uniform input) and 0.34528 at
        # slope=50, the top of the default constraint. Both are far above the
        # default epsilon=1e-12. See the module docstring for the sweeps.
        total_sum = keras.ops.sum(y_stepped, axis=self.axis, keepdims=True)
        return y_stepped / (total_sum + self.epsilon)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ThreshMax to the logits.

        :param inputs: Logits.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Probabilities of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        return self._compute_threshmax(inputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        ``slope`` is written from ``slope_initial_value``, the constructor
        argument, not from the current value of the ``slope`` weight. A
        trained slope is restored from the weights file, not from here.

        :return: The base Layer config plus ``axis``, ``slope``,
            ``epsilon``, ``trainable_slope``, and the serialized slope
            initializer, regularizer and constraint.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'slope': self.slope_initial_value,
            'epsilon': self.epsilon,
            'trainable_slope': self.trainable_slope,
            'slope_initializer': keras.initializers.serialize(self.slope_initializer),
            'slope_regularizer': keras.regularizers.serialize(self.slope_regularizer),
            'slope_constraint': keras.constraints.serialize(self.slope_constraint),
        })
        return config

    def __repr__(self) -> str:
        """Return a short representation showing the config and layer name.

        :return: A string such as
            ``ThreshMax(axis=-1, slope=10.0, mode='fixed', name='thresh_max')``.
        :rtype: str
        """
        mode = "learnable" if self.trainable_slope else "fixed"
        return (f"ThreshMax(axis={self.axis}, slope={self.slope_initial_value}, "
                f"mode='{mode}', name='{self.name}')")


# ---------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------


def thresh_max(
        x: keras.KerasTensor,
        axis: int = -1,
        slope: Union[float, keras.KerasTensor] = 10.0,
        epsilon: float = 1e-12
) -> keras.KerasTensor:
    """Apply ThreshMax without holding on to a layer.

    Builds a fresh, non-trainable :class:`ThreshMax` and calls it. Convenient
    for a one-off; not a fast path. Every call constructs a layer and emits
    the constructor's INFO log line, so do not call this inside a loop that
    runs per step -- build one :class:`ThreshMax` and reuse it.

    ``slope`` may be an eager scalar tensor as well as a float. It may not be
    a symbolic ``KerasTensor``: ``ThreshMax.__init__`` compares it against 0
    and converts it with ``float()``, and a symbolic tensor raises
    ``TypeError: A symbolic KerasTensor cannot be used as a boolean``.

    :param x: Logits.
    :type x: keras.KerasTensor
    :param axis: Axis to normalize and gate over. Defaults to -1.
    :type axis: int
    :param slope: Gate steepness. Must be positive. Defaults to 10.0.
    :type slope: Union[float, keras.KerasTensor]
    :param epsilon: Added to the renormalization denominator. Must be
        positive. Defaults to 1e-12.
    :type epsilon: float
    :return: Probabilities of the same shape as ``x``.
    :rtype: keras.KerasTensor
    :raises ValueError: If ``epsilon`` is not positive, or if ``slope`` is a
        Python number that is not positive. A non-numeric ``slope`` skips
        the check here and is validated by ``ThreshMax.__init__``.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if isinstance(slope, (int, float)) and slope <= 0:
        raise ValueError(f"slope must be positive, got {slope}")

    layer = ThreshMax(axis=axis, slope=slope, epsilon=epsilon, trainable_slope=False)
    return layer(x)

# ---------------------------------------------------------------------
