"""
Turns unconstrained predictions into a non-decreasing sequence.

Given raw scores ``[r_0, r_1, ..., r_n]`` along one axis, this module returns
values that satisfy ``y[i] <= y[i+1]``. That is what quantile regression,
dose-response curves, survival curves and ranking heads need: the network is
free to predict anything, and the layer enforces the ordering.

Five of the six methods work the same way. The first score is kept as an
anchor. The remaining scores are pushed through a function that cannot go
negative -- softplus, ``exp``, or a square -- and the running sum of those
non-negative deltas is added to the anchor. The sixth method, ``"sigmoid"``,
does not accumulate anything: it maps each score independently into a bounded
window around a fixed target position.

The ordering is non-strict, and it really can be flat. Two adjacent outputs
come out exactly equal whenever a delta is zero (``squared`` on a zero input)
or whenever a delta is too small to register against the accumulated total in
float32. Measured over 2000 random rows of 7 values drawn from N(0, 3): zero
negative adjacent differences for every method, but a minimum adjacent
difference of exactly 0.0 for ``exponential``, ``cumulative_exp``,
``squared`` and ``normalized_softmax``.

References:
    - Koenker, R. (2005). Quantile Regression. Cambridge University Press.
    - Cannon, A. J. (2011). Quantile regression neural networks.
      Journal of Computational and Graphical Statistics.
"""

import keras
import warnings
from typing import Optional, Literal, Tuple, Any

# ---------------------------------------------------------------------

MonotonicityMethod = Literal[
    "cumulative_softplus",
    "exponential",
    "sigmoid",
    "normalized_softmax",
    "squared",
    "cumulative_exp"
]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MonotonicityLayer(keras.layers.Layer):
    """Enforces a non-decreasing ordering along one axis.

    Reads raw scores and returns values with ``y[..., i] <= y[..., i+1]``.
    Output shape equals input shape. The layer owns no weights; every
    argument is validated in ``__init__`` or ``build``, so a bad
    configuration fails before the first forward pass.

    Five methods keep ``r_0`` as an anchor, turn ``r_1..r_n`` into
    non-negative deltas, and add the running sum of those deltas to the
    anchor. For those five the first output is the raw first input,
    unchanged. The ``"sigmoid"`` method is different: it ignores the anchor,
    places each position at a fixed target inside ``value_range``, and lets
    the sigmoid move it by a bounded amount around that target.

    **Architecture Overview:**

    .. code-block:: text

                   r  raw scores  [..., n]  (n >= 2)
                                │
                   ┌────────────┴────────────┐
                   ▼ 5 delta methods         ▼ sigmoid
        ┌─────────────────────┐   ┌─────────────────────┐
        │ split: r0 | r1..rn  │   │ p = sigmoid(r)      │
        └──────────┬──────────┘   │ t = lo + span*i/m   │
                   ▼              │ y = t + span        │
        ┌─────────────────────┐   │     *(p-0.5)/m      │
        │ d = f(r1..rn) >= 0  │   │ clip to [lo, hi]    │
        └──────────┬──────────┘   └──────────┬──────────┘
                   ▼                         │
        ┌─────────────────────┐              │
        │ d += min_spacing    │              │
        │ d = min(d, max_sp)  │              │
        └──────────┬──────────┘              │
                   ▼                         │
        ┌─────────────────────┐              │
        │ c = cumsum(d)       │              │
        └──────────┬──────────┘              │
                   ▼                         │
        ┌─────────────────────┐              │
        │ concat[r0, r0 + c]  │              │
        └──────────┬──────────┘              │
                   └────────────┬────────────┘
                                ▼
                   y  [..., n]   y[i] <= y[i+1]

    ``n`` is the size of the axis given by ``axis``, ``m = n - 1``,
    ``lo, hi = value_range`` and ``span = hi - lo``. ``i`` is the position
    along the axis. The spacing box is skipped when both spacing arguments
    are ``None``, which is the default.

    **Methods and which arguments reach them:**

    .. code-block:: text

        method               min_sp  max_sp  clip_inputs  value_range
        -------------------  ------  ------  -----------  -----------
        cumulative_softplus  yes     yes     honoured     ignored
        exponential          yes     yes     honoured     ignored
        cumulative_exp       yes     yes     forced on    ignored
        squared              yes     yes     ignored      ignored
        normalized_softmax   yes     no      ignored      required
        sigmoid              no      no      ignored      required

    The delta functions are ``softplus(r)`` for ``cumulative_softplus``,
    ``exp(r)`` for ``exponential`` and ``cumulative_exp``, ``r * r`` for
    ``squared``, and ``softmax(r) * span`` for ``normalized_softmax``.

    Every "ignored" and "no" cell above was measured, not read off the code:
    the same input was run with and without the argument and the outputs
    compared. Do not assume an argument reaches a method because it is
    accepted by the constructor.

    ``value_range`` means two different things. Under ``"sigmoid"`` it bounds
    the output, because the last step clips to it. Under
    ``"normalized_softmax"`` it only sets the total spread ``hi - lo``; the
    sequence still starts at the raw first input, so outputs fall outside
    ``value_range`` routinely. Measured on 5 rows of 6 scores from N(0, 3)
    with ``value_range=(0.0, 10.0)``: spread 10.000 on every row, but the
    observed values ran from -0.865 to 14.901.

    :param method: Which strategy to use. One of ``"cumulative_softplus"``
        (default), ``"exponential"``, ``"cumulative_exp"``, ``"sigmoid"``,
        ``"squared"``, ``"normalized_softmax"``.
    :type method: MonotonicityMethod
    :param axis: Axis to enforce the ordering along. Defaults to -1. It is
        normalized against the input rank in ``build``.
    :type axis: int
    :param min_spacing: Constant added to every delta, so consecutive outputs
        are at least this far apart. Must be non-negative. ``None`` (default)
        adds nothing. Ignored by ``"sigmoid"``.
    :type min_spacing: Optional[float]
    :param max_spacing: Ceiling applied to every delta. Must be positive.
        ``None`` (default) applies no ceiling. Ignored by ``"sigmoid"`` and
        ``"normalized_softmax"``.
    :type max_spacing: Optional[float]
    :param value_range: ``(min_val, max_val)`` with ``min_val < max_val``.
        Required by ``"sigmoid"`` and ``"normalized_softmax"``; never read by
        the other four.
    :type value_range: Optional[Tuple[float, float]]
    :param clip_inputs: Whether to clip raw scores to ``input_clip_range``
        before transforming them. ``None`` (default) means True for
        ``"exponential"`` and ``"cumulative_exp"`` and False otherwise.
        ``"cumulative_exp"`` clips regardless of this flag; ``"squared"``,
        ``"sigmoid"`` and ``"normalized_softmax"`` never clip.
    :type clip_inputs: Optional[bool]
    :param input_clip_range: ``(min, max)`` used when clipping is active.
        Defaults to ``(-20.0, 20.0)``.
    :type input_clip_range: Tuple[float, float]
    :param epsilon: Floor for the ``n - 1`` divisor in ``"sigmoid"``.
        Defaults to 1e-7. ``build`` already rejects ``n < 2``, so with a
        static axis size this floor is never the binding term.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``method`` is unknown, if a required
        ``value_range`` is missing or not increasing, if a spacing argument
        has the wrong sign, or if ``min_spacing > max_spacing``. Raised from
        ``__init__``. ``build`` raises for an out-of-bounds ``axis`` or an
        axis shorter than 2.

    :ivar axis_normalized: ``axis`` converted to a non-negative index against
        the input rank. Set in ``build``, so it does not exist before the
        first call.
    :vartype axis_normalized: int

    Note:
        ``exponential`` with ``clip_inputs=False`` overflows, and the
        constructor warns about it. Measured on three scores of 100.0:
        unclipped output ``[100.0, inf, inf]``; the clipped default gives
        ``[1.0000000e+02, 4.8516531e+08, 9.7033056e+08]``. Prefer
        ``"cumulative_exp"``, which cannot be talked out of clipping.
    """

    def __init__(
            self,
            method: MonotonicityMethod = "cumulative_softplus",
            axis: int = -1,
            min_spacing: Optional[float] = None,
            max_spacing: Optional[float] = None,
            value_range: Optional[Tuple[float, float]] = None,
            clip_inputs: Optional[bool] = None,
            input_clip_range: Tuple[float, float] = (-20.0, 20.0),
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and store it.

        No weight is created and ``axis`` is not resolved here; ``build``
        does that once the input rank is known.

        :param method: Which strategy to use.
        :type method: MonotonicityMethod
        :param axis: Axis to enforce the ordering along.
        :type axis: int
        :param min_spacing: Constant added to every delta, or ``None``.
        :type min_spacing: Optional[float]
        :param max_spacing: Ceiling applied to every delta, or ``None``.
        :type max_spacing: Optional[float]
        :param value_range: ``(min_val, max_val)``, required by ``"sigmoid"``
            and ``"normalized_softmax"``.
        :type value_range: Optional[Tuple[float, float]]
        :param clip_inputs: Whether to clip raw scores. ``None`` picks the
            per-method default.
        :type clip_inputs: Optional[bool]
        :param input_clip_range: ``(min, max)`` used when clipping is active.
        :type input_clip_range: Tuple[float, float]
        :param epsilon: Floor for the ``n - 1`` divisor in ``"sigmoid"``.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``method`` is unknown, if a required
            ``value_range`` is missing or not increasing, if a spacing
            argument has the wrong sign, or if ``min_spacing`` exceeds
            ``max_spacing``.
        """
        super().__init__(**kwargs)

        # Validate method
        valid_methods = [
            "cumulative_softplus", "exponential", "sigmoid",
            "normalized_softmax", "squared", "cumulative_exp"
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown monotonicity method: {method}. "
                f"Must be one of {valid_methods}"
            )

        # These two methods read value_range; the other four never do.
        if method in ["sigmoid", "normalized_softmax"]:
            if value_range is None:
                raise ValueError(
                    f"Method '{method}' requires 'value_range' to be specified as (min, max)"
                )
            if len(value_range) != 2 or value_range[0] >= value_range[1]:
                raise ValueError(
                    f"value_range must be (min, max) with min < max, got {value_range}"
                )

        # Validate spacing constraints
        if min_spacing is not None and min_spacing < 0:
            raise ValueError(f"min_spacing must be non-negative, got {min_spacing}")
        if max_spacing is not None and max_spacing <= 0:
            raise ValueError(f"max_spacing must be positive, got {max_spacing}")
        if min_spacing is not None and max_spacing is not None:
            if min_spacing > max_spacing:
                raise ValueError(
                    f"min_spacing ({min_spacing}) cannot exceed max_spacing ({max_spacing})"
                )

        # Store configuration
        self.method = method
        self.axis = axis
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.value_range = value_range
        self.epsilon = epsilon
        self.input_clip_range = input_clip_range

        # The two exponential methods overflow easily, so clip by default.
        if clip_inputs is None:
            self.clip_inputs = method in ["exponential", "cumulative_exp"]
        else:
            self.clip_inputs = clip_inputs

        # `cumulative_exp` clips regardless, so only `exponential` can be
        # talked into an unclipped exp().
        if method == "exponential" and not self.clip_inputs:
            warnings.warn(
                "Using exponential method without input clipping can cause numerical "
                "overflow. Consider setting clip_inputs=True or using 'cumulative_exp' method.",
                UserWarning
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Resolve ``axis`` against the input rank and check the axis size.

        Creates no weights. Sets ``self.axis_normalized``, which every method
        below relies on.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``axis`` is out of bounds for the input rank,
            or if the axis is statically known and shorter than 2.
        """
        if self.built:
            return

        # Normalize axis to positive index
        ndim = len(input_shape)
        if self.axis < 0:
            self.axis_normalized = ndim + self.axis
        else:
            self.axis_normalized = self.axis

        # Validate axis
        if self.axis_normalized < 0 or self.axis_normalized >= ndim:
            raise ValueError(
                f"axis {self.axis} is out of bounds for input with {ndim} dimensions"
            )

        # An ordering needs at least two values. A dynamic (None) axis size
        # cannot be checked here and is left to the ops.
        axis_size = input_shape[self.axis_normalized]
        if axis_size is not None and axis_size < 2:
            raise ValueError(
                f"Monotonicity requires at least 2 values along axis {self.axis}, "
                f"but got {axis_size}"
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Dispatch to the configured method.

        :param inputs: Raw scores. ``inputs.shape[axis]`` must be at least 2.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Non-decreasing tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``self.method`` is unknown. ``__init__``
            already rejects that, so this branch is unreachable through the
            public API.
        """
        # Apply the selected monotonicity method
        if self.method == "cumulative_softplus":
            return self._cumulative_softplus(inputs)
        elif self.method == "exponential":
            return self._exponential(inputs)
        elif self.method == "cumulative_exp":
            return self._cumulative_exp(inputs)
        elif self.method == "sigmoid":
            return self._sigmoid(inputs)
        elif self.method == "squared":
            return self._squared(inputs)
        elif self.method == "normalized_softmax":
            return self._normalized_softmax(inputs)
        else:
            # Unreachable: __init__ rejects any other method.
            raise ValueError(f"Unknown method: {self.method}")

    def _split_first_and_rest(
            self,
            inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Split off the anchor value from the rest along the axis.

        Both parts keep the input rank. ``first`` has size 1 along the axis
        so it broadcasts against the accumulated deltas.

        :param inputs: Raw scores.
        :type inputs: keras.KerasTensor
        :return: ``(first, rest)``, of sizes 1 and ``n - 1`` along the axis.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Slice the anchor: everything on the other axes, index 0 on this one.
        indices_first = [slice(None)] * len(inputs.shape)
        indices_first[self.axis_normalized] = slice(0, 1)
        first = inputs[tuple(indices_first)]

        # Remaining values along axis
        indices_rest = [slice(None)] * len(inputs.shape)
        indices_rest[self.axis_normalized] = slice(1, None)
        rest = inputs[tuple(indices_rest)]

        return first, rest

    def _apply_spacing_constraints(
            self,
            deltas: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Add ``min_spacing`` and cap at ``max_spacing``.

        A no-op when both are ``None``. Only the four accumulating methods
        call this; ``_sigmoid`` and ``_normalized_softmax`` do not.

        :param deltas: Non-negative increments between consecutive values.
        :type deltas: keras.KerasTensor
        :return: Constrained deltas, same shape.
        :rtype: keras.KerasTensor
        """
        if self.min_spacing is not None:
            deltas = deltas + self.min_spacing

        if self.max_spacing is not None:
            deltas = keras.ops.minimum(deltas, self.max_spacing)

        return deltas

    def _cumulative_softplus(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Accumulate ``softplus`` deltas: ``Q_i = Q_0 + sum(softplus(r_j))``.

        The default. ``softplus`` is strictly positive and its gradient never
        saturates to zero, so deltas stay strictly positive and the ordering
        is strict up to float rounding.

        :param inputs: Raw scores.
        :type inputs: keras.KerasTensor
        :return: Non-decreasing predictions, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Clip inputs if requested
        if self.clip_inputs:
            rest = keras.ops.clip(rest, *self.input_clip_range)

        # Apply softplus to ensure positive deltas
        deltas = keras.ops.softplus(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum of deltas
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        # Add to first value
        subsequent_values = first + accumulated_deltas

        # Concatenate first and subsequent values
        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _exponential(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Accumulate ``exp`` deltas: ``Q_i = Q_{i-1} + exp(r_i)``.

        Gives large spacing for large scores. Honours ``clip_inputs``, which
        defaults to True here; with clipping off this overflows to ``inf``.

        :param inputs: Raw scores.
        :type inputs: keras.KerasTensor
        :return: Non-decreasing predictions, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Clip to prevent overflow
        if self.clip_inputs:
            rest = keras.ops.clip(rest, *self.input_clip_range)

        # Exponential transformation for positive deltas
        deltas = keras.ops.exp(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        # Add to first value
        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _cumulative_exp(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """``exponential`` with clipping that cannot be switched off.

        Identical to :meth:`_exponential` except that ``input_clip_range`` is
        applied whatever ``clip_inputs`` says. Use this instead of
        ``"exponential"`` unless you need the unclipped behaviour.

        :param inputs: Raw scores.
        :type inputs: keras.KerasTensor
        :return: Non-decreasing predictions, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Always clip for this method
        rest = keras.ops.clip(rest, *self.input_clip_range)

        # Exponential transformation
        deltas = keras.ops.exp(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _squared(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Accumulate squared deltas: ``Q_i = Q_{i-1} + r_i^2``.

        Cheap and free of exponential blow-up, but the delta is exactly zero
        at ``r_i = 0``, so adjacent outputs can be equal. Measured: a row of
        four zeros comes back as ``[0., 0., 0., 0.]``. The gradient also
        vanishes there. Does not clip, whatever ``clip_inputs`` says.

        :param inputs: Raw scores.
        :type inputs: keras.KerasTensor
        :return: Non-decreasing predictions, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Square to ensure positive deltas
        deltas = keras.ops.square(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _sigmoid(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Place each position at a fixed target, then nudge it by a sigmoid.

        The only method that does not accumulate. Position ``i`` of ``n``
        gets target ``min_val + span * i / (n - 1)``, and the sigmoid moves
        it by at most ``span * 0.5 / (n - 1)`` either way. Adjacent targets
        are ``span / (n - 1)`` apart, so two nudges can never close the gap.
        The result is then clipped into ``value_range``, which preserves the
        ordering. ``min_spacing``, ``max_spacing`` and ``clip_inputs`` are
        all ignored here.

        Measured on ``[-2, 0, 1, 3, -1]`` with ``value_range=(0.0, 10.0)``:
        output ``[0.0, 2.5, 5.5776463, 8.631435, 9.422354]``. Note the last
        two: the scores fall from 3 to -1 while the outputs still rise,
        because the target dominates the nudge. Do not read this method as
        ``sigmoid(r) * span + min_val`` -- that formula is not what runs and
        is not monotonic.

        :param inputs: Raw scores.
        :type inputs: keras.KerasTensor
        :return: Non-decreasing predictions inside ``value_range``, same
            shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        min_val, max_val = self.value_range

        # Apply sigmoid to map to [0, 1]
        normalized = keras.ops.sigmoid(inputs)

        # Each position gets its own target, so the targets alone are already
        # ordered before the sigmoid moves anything.
        axis_size = keras.ops.shape(inputs)[self.axis_normalized]

        # Generate indices: [0, 1, 2, ..., n-1]
        indices = keras.ops.cast(
            keras.ops.arange(axis_size, dtype="float32"),
            inputs.dtype
        )

        # Normalize indices to [0, 1]: [0, 1/(n-1), 2/(n-1), ..., 1]
        normalized_indices = indices / keras.ops.maximum(
            keras.ops.cast(axis_size - 1, inputs.dtype),
            self.epsilon
        )

        # Reshape to broadcast along the monotonicity axis: 1 everywhere
        # except the axis being ordered.
        broadcast_shape = [1] * len(inputs.shape)
        broadcast_shape[self.axis_normalized] = axis_size
        normalized_indices = keras.ops.reshape(normalized_indices, broadcast_shape)

        # The ordered backbone: evenly spaced positions across value_range.
        target_values = min_val + (max_val - min_val) * normalized_indices

        # The sigmoid lets each position deviate from its target. Centering
        # on 0.5 means sigmoid < 0.5 pulls down and > 0.5 pulls up. The
        # deviation is bounded by (max-min) * 0.5 * flexibility while
        # adjacent targets sit (max-min)/(n-1) apart, so flexibility =
        # 1/(n-1) makes the worst-case adjacent difference exactly zero.
        # Raising flexibility would break the ordering this layer promises.
        flexibility = 1.0 / keras.ops.maximum(
            keras.ops.cast(axis_size - 1, inputs.dtype), self.epsilon
        )
        output = target_values + (max_val - min_val) * (normalized - 0.5) * flexibility

        # Clip to ensure we stay in bounds
        output = keras.ops.clip(output, min_val, max_val)

        return output

    def _normalized_softmax(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Split a fixed total spread across the deltas with a softmax.

        The deltas are ``softmax(r_1..r_n) * (max_val - min_val)``, so they
        sum to exactly the width of ``value_range``. The sequence still
        starts at the raw first input, so ``value_range`` fixes the spread,
        not the output bounds -- measured spread 10.000 per row with
        ``value_range=(0.0, 10.0)`` while the values themselves ran -0.865 to
        14.901. ``min_spacing`` is added afterwards and pushes the spread
        past that width; ``max_spacing`` and ``clip_inputs`` are ignored.

        :param inputs: Raw scores.
        :type inputs: keras.KerasTensor
        :return: Non-decreasing predictions with a controlled total spread,
            same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        min_val, max_val = self.value_range
        total_range = max_val - min_val

        # Softmax to get normalized weights that sum to 1
        weights = keras.ops.softmax(rest, axis=self.axis_normalized)

        # Scale weights by total range to get deltas
        deltas = weights * total_range

        # Apply min_spacing if specified (may exceed total_range)
        if self.min_spacing is not None:
            deltas = deltas + self.min_spacing

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the input shape unchanged.

        The split and the concatenate cancel: 1 plus ``n - 1`` is ``n``.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output tensor shape, identical to ``input_shape``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration for serialization.

        ``clip_inputs`` is written out as the resolved boolean, not the
        ``None`` that may have been passed in, so a reloaded layer keeps the
        clipping behaviour it had.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "method": self.method,
            "axis": self.axis,
            "min_spacing": self.min_spacing,
            "max_spacing": self.max_spacing,
            "value_range": self.value_range,
            "clip_inputs": self.clip_inputs,
            "input_clip_range": self.input_clip_range,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
