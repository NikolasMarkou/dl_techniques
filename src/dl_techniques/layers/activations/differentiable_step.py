"""
A learnable, differentiable step function.

The Heaviside step is discontinuous and its derivative is zero everywhere it
is defined, so gradient descent cannot train through it. This layer replaces
it with a shifted, scaled ``tanh``, which is smooth everywhere. The two things
that define a step -- where it happens and how sharp it is -- become trainable
weights.

The function is::

    f(x) = 0.5 * (tanh(slope * (x - shift)) + 1)

Each part does one job:

1. **tanh** is the differentiable switch. It maps any real input into
   ``[-1, 1]`` with an S-shaped transition centred at zero.
2. **shift** moves the transition left or right. Learning it means learning
   the threshold, so the network picks where the gate sits.
3. **slope** controls sharpness. Large ``slope`` makes the tanh approach a
   true hard step, so the decision is nearly binary. Small ``slope`` gives a
   soft ramp.
4. **0.5 * (... + 1)** rescales from ``[-1, 1]`` to ``[0, 1]``, so the output
   reads as a probability or a soft mask.

This is the same shape as the gates in a GRU or LSTM, with the gate
parameters exposed as weights instead of computed from the input.

References:
    - Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical
      Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
      (Provides a prominent example of using sigmoid-like gating functions).
"""

import keras
from typing import Optional, Union, Any, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.l2_custom import L2_custom
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.activations.differentiable_step")
class DifferentiableStep(keras.layers.Layer):
    """
    A learnable, differentiable step function, per-tensor or per-feature.

    Computes ``0.5 * (tanh(slope * (x - shift)) + 1)``, so the output is
    always in ``[0, 1]`` and has the same shape as the input. Both ``slope``
    and ``shift`` are trainable weights.

    There are two modes, picked by ``axis``:

    1. **Scalar** (``axis=None``): one ``slope`` and one ``shift`` for the
       whole tensor.
    2. **Per-axis** (``axis`` is an int, the default is ``-1``): a vector of
       ``slope`` and ``shift`` values, one per index along that axis, shaped
       to broadcast. This gives per-feature or per-channel thresholding.

    **Architecture Overview:**

    .. code-block:: text

               x  [B, ..., F]
                      │
                      ▼
        ┌───────────────────────────┐
        │ x - shift                 │
        │ (learned threshold)       │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ * slope                   │
        │ (learned steepness)       │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ tanh(...)  -> [-1, 1]     │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ (tanh + 1) / 2            │
        └─────────────┬─────────────┘
                      │
                      ▼
                y  in [0, 1]

    ``slope`` and ``shift`` broadcast against ``x``, so the two subtract and
    multiply steps do not change shape.

    **Weight shapes by axis, for a rank-4 input (B, H, W, C):**

    .. code-block:: text

        axis    slope / shift shape    meaning
        None    ()                     one value for the whole tensor
        -1      (1, 1, 1, C)           one value per feature
        1       (1, H, 1, 1)           one value per row

    **Fixed settings on the slope weight:**

    ``slope`` is always clipped to ``[1.0, 10.0]`` and always regularized with
    ``L2_custom(-1e-3)``. Both are hard-coded in ``build`` and are not
    constructor arguments, so they are not in ``get_config`` either. The
    negative L2 coefficient is intentional: it rewards a *larger* slope, which
    pushes the layer toward a harder step over training. ``shift``, by
    contrast, takes whatever regularizer and constraint you pass.

    :param axis: Axis to learn per-index parameters over. ``None`` gives a
        single scalar ``slope`` and ``shift`` for the whole input. An integer
        gives one of each per index along that axis. Defaults to ``-1``.
    :type axis: Optional[int]
    :param slope_initializer: Initializer for the ``slope`` weight. A string
        name or an Initializer instance. Defaults to ``'ones'``, which starts
        at the lower end of the ``[1.0, 10.0]`` clip range.
    :type slope_initializer: Union[str, keras.initializers.Initializer]
    :param shift_initializer: Initializer for the ``shift`` weight. A string
        name or an Initializer instance. Defaults to ``'zeros'``.
    :type shift_initializer: Union[str, keras.initializers.Initializer]
    :param shift_regularizer: Regularizer for the ``shift`` weight. If
        ``None``, ``keras.regularizers.L2(1e-3)`` is used, which pulls the
        threshold toward 0.
    :type shift_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param shift_constraint: Constraint on the ``shift`` weight. If ``None``,
        ``ValueRangeConstraint(-1, +1)`` is used, which clips the threshold
        into that range after every update.
    :type shift_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    :raises TypeError: If ``axis`` is neither an ``int`` nor ``None``. Raised
        from ``__init__``.
    :raises ValueError: From ``build``, if ``axis`` is out of bounds for the
        input rank, or if the input dimension along ``axis`` is ``None``.

    :ivar slope: Trainable steepness weight. ``None`` until ``build`` runs.
    :vartype slope: Optional[keras.Variable]
    :ivar shift: Trainable threshold weight. ``None`` until ``build`` runs.
    :vartype shift: Optional[keras.Variable]
    """

    def __init__(
            self,
            axis: Optional[int] = -1,
            slope_initializer: Union[str, keras.initializers.Initializer] = 'ones',
            shift_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            # None resolves to L2(1e-3) in the body, pulling shift toward 0.
            shift_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            # None resolves to ValueRangeConstraint(-1, +1) in the body.
            shift_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            **kwargs: Any
    ) -> None:
        """
        Validate ``axis`` and store the initializers, regularizer and
        constraint. No weight is created here; ``build`` does that once the
        input shape is known.

        ``shift_regularizer`` and ``shift_constraint`` default to ``None`` in
        the signature and are resolved to real objects in the body. A mutable
        default in the signature is evaluated once at import, so every layer
        in the process would share one regularizer and one constraint object.

        :param axis: Axis to learn per-index parameters over. ``None`` gives
            a single scalar ``slope`` and ``shift``. Defaults to ``-1``.
        :type axis: Optional[int]
        :param slope_initializer: Initializer for the ``slope`` weight.
            Defaults to ``'ones'``.
        :type slope_initializer: Union[str, keras.initializers.Initializer]
        :param shift_initializer: Initializer for the ``shift`` weight.
            Defaults to ``'zeros'``.
        :type shift_initializer: Union[str, keras.initializers.Initializer]
        :param shift_regularizer: Regularizer for the ``shift`` weight.
            ``None`` resolves to ``keras.regularizers.L2(1e-3)``.
        :type shift_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param shift_constraint: Constraint on the ``shift`` weight. ``None``
            resolves to ``ValueRangeConstraint(-1, +1)``.
        :type shift_constraint: Optional[Union[str, keras.constraints.Constraint]]
        :param kwargs: Additional arguments for the Layer base class.
        :raises TypeError: If ``axis`` is neither an ``int`` nor ``None``.
        """
        super().__init__(**kwargs)

        if axis is not None and not isinstance(axis, int):
            raise TypeError(f"Expected `axis` to be an int or None, but got: {axis}")

        # Built here, not in the signature: see this method's docstring.
        if shift_regularizer is None:
            shift_regularizer = keras.regularizers.L2(1e-3)
        if shift_constraint is None:
            shift_constraint = ValueRangeConstraint(min_value=-1, max_value=+1)

        # get_config() serializes each of these, so all five must be stored.
        self.axis = axis
        self.slope_initializer = keras.initializers.get(slope_initializer)
        self.shift_initializer = keras.initializers.get(shift_initializer)
        self.shift_regularizer = keras.regularizers.get(shift_regularizer)
        self.shift_constraint = keras.constraints.get(shift_constraint)

        # Created in build(), once the input shape is known.
        self.slope = None
        self.shift = None

    def build(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> None:
        """
        Create the ``slope`` and ``shift`` weights.

        With ``axis=None`` both are scalars. With an integer ``axis`` both get
        a shape that is 1 everywhere except at ``axis``, so they broadcast
        against the input.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``axis`` is out of bounds for the input rank,
            or if the input dimension along ``axis`` is ``None``.
        """
        if self.built:
            return

        if self.axis is None:
            param_shape = ()
        else:
            rank = len(input_shape)

            # A negative axis is resolved against the rank here, so the bounds
            # check below and the shape build both see the same index.
            axis = self.axis if self.axis >= 0 else rank + self.axis
            if axis < 0 or axis >= rank:
                raise ValueError(
                    f"Invalid axis: {self.axis}. It is out of bounds for an "
                    f"input of rank {rank}."
                )

            # Broadcast shape: 1 on every axis except the chosen one.
            param_shape = [1] * rank
            if input_shape[axis] is None:
                raise ValueError(
                    f"The dimension for axis {axis} must be defined, but it is None. "
                    f"Input shape received: {input_shape}"
                )
            param_shape[axis] = input_shape[axis]
            param_shape = tuple(param_shape)

        self.slope = self.add_weight(
            name='slope',
            shape=param_shape,
            initializer=self.slope_initializer,
            # Clipped to [1, 10]: 1 is an ordinary tanh, 10 is close to a hard
            # step. Hard-coded, so it is not a constructor argument.
            constraint=ValueRangeConstraint(min_value=+1.0, max_value=+10.0),
            # NEGATIVE L2 coefficient, on purpose: it rewards a larger slope,
            # pushing the layer toward a harder step. Do not flip the sign.
            regularizer=L2_custom(-1e-3),
            trainable=True,
        )

        self.shift = self.add_weight(
            name='shift',
            shape=param_shape,
            initializer=self.shift_initializer,
            constraint=self.shift_constraint,
            regularizer=self.shift_regularizer,
            trainable=True,
        )

        # Parent build() last, so self.built flips only once the weights exist.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply ``0.5 * (tanh(slope * (x - shift)) + 1)`` element-wise.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``, with every value in
            ``[0, 1]``.
        :rtype: keras.KerasTensor
        """
        scaled_shifted_x = self.slope * (inputs - self.shift)
        return (keras.ops.tanh(scaled_shifted_x) + 1.0) / 2.0

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config needed to rebuild the layer.

        The ``slope`` constraint and regularizer are absent because they are
        hard-coded in ``build``, not constructor arguments.

        :return: The base Layer config plus ``axis``, both initializers, and
            the ``shift`` regularizer and constraint.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'slope_initializer': keras.initializers.serialize(self.slope_initializer),
            'shift_initializer': keras.initializers.serialize(self.shift_initializer),
            'shift_regularizer': keras.regularizers.serialize(self.shift_regularizer),
            'shift_constraint': keras.constraints.serialize(self.shift_constraint),
        })
        return config

# ---------------------------------------------------------------------
