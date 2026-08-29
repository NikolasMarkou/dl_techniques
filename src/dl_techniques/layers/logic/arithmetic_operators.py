"""
A layer that learns which arithmetic operation to apply.

The layer runs all seven primitive operations on its inputs — add, multiply,
subtract, divide, power, max and min — and returns their weighted sum. One
learnable weight per operation decides the mix, so the network trains the
choice instead of having it fixed at design time. The idea is the one
Neural Architecture Search uses: relax a discrete choice into a weighted
average and let gradients pick.

A forward pass has three stages:

    1. Run every operation named in ``operation_types``. Divide and power
       go through guarded helpers so neither can produce an infinity.

    2. Take a softmax over the learnable weights to get one probability per
       operation, then sum the results with those probabilities.

    3. Multiply by a learnable scale, if ``use_scaling`` is on.

The math:

    With weights w, one per operation f_i, and temperature T, the
    probability of operation i is

        p_i = exp(w_i / T) / sum_j(exp(w_j / T))

    T is learnable and controls how sharp the choice is. Small T pushes the
    distribution toward one-hot, so a single operation dominates. Large T
    flattens it and averages the operations. Training usually starts flat
    and sharpens as the weights separate.

    The output is

        Y = s * sum_i(p_i * f_i(X))

    where s is the learnable scale. Gradients reach w, T and s, so all
    three are learned.

References:
    - Liu, H., Simonyan, K., & Yang, Y. (2018). "DARTS: Differentiable
      Architecture Search". The continuous relaxation of a discrete
      operation choice.

    - Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the
      Knowledge in a Neural Network". The softmax temperature used here to
      control the sharpness of the selection.
"""

import math
import keras
from typing import List, Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LearnableArithmeticOperator(keras.layers.Layer):
    """
    A learnable choice among seven arithmetic operations.

    Give the layer two tensors of the same shape, or one tensor to use for
    both operands. It runs each operation named in ``operation_types``,
    sums the results weighted by a softmax over learnable weights, and
    scales the sum by a learnable factor. All seven operations are selected
    by default.

    Divide and power are guarded. ``_safe_divide`` never divides by
    something smaller than ``epsilon`` in magnitude, and ``_safe_power``
    clips both the base and the exponent into ``power_clip_range`` and
    ``exponent_clip_range``. A zero denominator or a negative base
    therefore produces a finite number, not an infinity or a NaN.

    **Architecture Overview:**

    .. code-block:: text

        x1            x2      one tensor sets x2 = x1
         │             │
         └──────┬──────┘
                ▼
        ┌───────────────────────────────┐
        │ every op in operation_types   │
        │ runs in parallel -> N results │
        │ divide -> _safe_divide        │
        │ power  -> _safe_power         │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ weighted sum, p from          │
        │ _operation_probs              │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ * scaling_factor              │
        │ (use_scaling only; magnitude  │
        │ floored at 1e-7, sign kept)   │
        └───────────────┬───────────────┘
                        ▼
        output, same shape as x1

    **The seven operations, read from call():**

    .. code-block:: text

        key       f(x1, x2)
        --------  ---------------------------
        add       x1 + x2
        multiply  x1 * x2
        subtract  x1 - x2
        divide    _safe_divide(x1, x2)
        power     _safe_power(x1, x2)
        max       maximum(x1, x2)
        min       minimum(x1, x2)

        Every key is selected when operation_types
        is None. There is no unary operation here:
        a single input tensor is used for both
        operands, so subtract returns 0 everywhere
        and multiply squares the input.

    :param operation_types: Operations to select among, named by the keys
        in the table above. ``None`` selects all seven.
    :type operation_types: Optional[List[str]]
    :param use_temperature: Divide the selection weights by a learnable
        temperature before the softmax.
    :type use_temperature: bool
    :param temperature_init: Starting temperature. Must be positive.
    :type temperature_init: float
    :param use_scaling: Multiply the output by a learnable scale.
    :type use_scaling: bool
    :param scaling_init: Starting scale. Must be positive, though training
        may drive the learned value negative.
    :type scaling_init: float
    :param operation_initializer: Initializer for the selection weights.
        ``"zeros"`` starts every operation equally likely.
    :type operation_initializer: Union[str, keras.initializers.Initializer]
    :param temperature_initializer: Initializer for the temperature weight.
        Read only when ``softplus_temperature`` is False; the softplus path
        computes its own raw value from ``temperature_init``.
    :type temperature_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param scaling_initializer: Initializer for the scale weight.
    :type scaling_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param epsilon: Floor for the divide guard. Must be positive.
    :type epsilon: float
    :param power_clip_range: ``(min_base, max_base)`` applied to ``|x1|``
        before the power. Both must be positive and increasing.
    :type power_clip_range: Tuple[float, float]
    :param exponent_clip_range: ``(min_exp, max_exp)`` applied to the
        exponent, hard or smooth per ``exponent_clip_mode``.
    :type exponent_clip_range: Tuple[float, float]
    :param softplus_temperature: Store the temperature as a raw value and
        read it back through softplus, which keeps it positive without a
        constraint. The stored weight is then not the temperature itself.
    :type softplus_temperature: bool
    :param safe_divide_mode: ``"hard_clamp"`` floors ``|x2|`` at
        ``epsilon``. ``"smooth"`` uses a bounded-gradient approximation
        that returns 0 at ``x2 = 0``. See :meth:`_safe_divide`.
    :type safe_divide_mode: str
    :param gumbel_softmax: Add Gumbel noise to the weights during training
        so the selection is sampled rather than averaged.
    :type gumbel_softmax: bool
    :param gumbel_hard: With ``gumbel_softmax``, make the forward value a
        one-hot vector while the gradient stays soft. Ignored when
        ``gumbel_softmax`` is False.
    :type gumbel_hard: bool
    :param entropy_coefficient: Weight of an added loss that penalizes a
        flat selection distribution. 0 adds no loss.
    :type entropy_coefficient: float
    :param selection_mode: ``"global"`` learns one operation choice for the
        whole tensor. ``"per_channel"`` learns one per channel and needs a
        known last axis at build time.
    :type selection_mode: str
    :param exponent_clip_mode: ``"hard"`` clips the exponent, which zeroes
        the gradient outside the range. ``"smooth"`` squashes it with tanh
        and keeps a gradient everywhere.
    :type exponent_clip_mode: str
    :param kwargs: Passed to ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar operation_types: The operation keys this layer selects among.
    :vartype operation_types: List[str]
    :ivar use_temperature: Whether a temperature weight exists.
    :vartype use_temperature: bool
    :ivar temperature_init: The requested starting temperature.
    :vartype temperature_init: float
    :ivar use_scaling: Whether a scale weight exists.
    :vartype use_scaling: bool
    :ivar scaling_init: The requested starting scale.
    :vartype scaling_init: float
    :ivar num_operations: ``len(operation_types)``; the N of every shape
        note in this file.
    :vartype num_operations: int
    :ivar operation_initializer: The resolved selection-weight initializer.
    :vartype operation_initializer: keras.initializers.Initializer
    :ivar temperature_initializer: The resolved temperature initializer.
    :vartype temperature_initializer: keras.initializers.Initializer
    :ivar scaling_initializer: The resolved scale initializer.
    :vartype scaling_initializer: keras.initializers.Initializer
    :ivar epsilon: The stored divide floor.
    :vartype epsilon: float
    :ivar power_clip_range: The stored base clip range.
    :vartype power_clip_range: Tuple[float, float]
    :ivar exponent_clip_range: The stored exponent clip range.
    :vartype exponent_clip_range: Tuple[float, float]
    :ivar softplus_temperature: Whether the temperature weight is raw.
    :vartype softplus_temperature: bool
    :ivar safe_divide_mode: ``"hard_clamp"`` or ``"smooth"``.
    :vartype safe_divide_mode: str
    :ivar gumbel_softmax: Whether training samples the selection.
    :vartype gumbel_softmax: bool
    :ivar gumbel_hard: Whether the sampled selection is straight-through.
    :vartype gumbel_hard: bool
    :ivar entropy_coefficient: Weight of the entropy loss.
    :vartype entropy_coefficient: float
    :ivar selection_mode: ``"global"`` or ``"per_channel"``.
    :vartype selection_mode: str
    :ivar exponent_clip_mode: ``"hard"`` or ``"smooth"``.
    :vartype exponent_clip_mode: str
    :ivar operation_weights: Selection weights, ``(N,)`` in global mode and
        ``(C, N)`` per channel. ``None`` until ``build``.
    :vartype operation_weights: Optional[keras.Variable]
    :ivar temperature: Scalar temperature weight, or ``None`` when
        ``use_temperature`` is False or the layer is not built.
    :vartype temperature: Optional[keras.Variable]
    :ivar scaling_factor: Scalar scale weight, or ``None`` when
        ``use_scaling`` is False or the layer is not built.
    :vartype scaling_factor: Optional[keras.Variable]

    :raises ValueError: From the constructor if ``exponent_clip_mode`` or
        ``selection_mode`` or ``safe_divide_mode`` is not one of its two
        keys, ``operation_types`` is empty or names an unknown operation,
        ``temperature_init`` or ``scaling_init`` or ``epsilon`` is not
        positive, either clip range is not increasing, or
        ``entropy_coefficient`` is negative.
    :raises ValueError: From ``build`` if two input shapes differ, or
        ``selection_mode="per_channel"`` gets an unknown last axis.
    :raises ValueError: From ``call`` if a list of more than two tensors is
        given.
    :raises RuntimeError: From ``to_symbolic`` before the layer is built.

    Input shape:
        One tensor of any shape, or a list of one or two tensors of the same
        shape. In ``per_channel`` mode the last axis must be known.

    Output shape:
        The same shape as the first input.

    Example:
        .. code-block:: python

            import keras
            from dl_techniques.layers.logic import (
                LearnableArithmeticOperator,
            )

            a = keras.random.normal((2, 8))
            b = keras.random.normal((2, 8))

            op = LearnableArithmeticOperator(
                operation_types=['add', 'multiply', 'divide']
            )
            y = op([a, b])
            y.shape  # (2, 8)

            # After training, ask which operation won.
            op.to_symbolic(top_k=2)
    """

    def __init__(
            self,
            operation_types: Optional[List[str]] = None,
            use_temperature: bool = True,
            temperature_init: float = 1.0,
            use_scaling: bool = True,
            scaling_init: float = 1.0,
            operation_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            temperature_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            scaling_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            epsilon: float = 1e-7,
            power_clip_range: Tuple[float, float] = (1e-7, 10.0),
            exponent_clip_range: Tuple[float, float] = (-2.0, 2.0),
            softplus_temperature: bool = True,
            safe_divide_mode: str = "hard_clamp",
            gumbel_softmax: bool = False,
            gumbel_hard: bool = False,
            entropy_coefficient: float = 0.0,
            selection_mode: str = "global",
            exponent_clip_mode: str = "hard",
            **kwargs: Any
    ) -> None:
        """
        Validate the arguments and store the configuration.

        No weight is created here. The selection weights need the channel
        count in ``per_channel`` mode, so every weight is created in
        :meth:`build`. The class docstring documents each parameter.
        """
        super().__init__(**kwargs)

        # exponent_clip_mode='smooth' replaces the hard clip with a
        # tanh-based squash that still has a gradient at the boundary.
        # 'hard' is the default and keeps the plain clip.
        if exponent_clip_mode not in ("hard", "smooth"):
            raise ValueError(
                f"exponent_clip_mode must be 'hard' or 'smooth', got "
                f"{exponent_clip_mode!r}."
            )

        # selection_mode='per_channel' creates (channels, num_operations)
        # weights, so each channel picks its own operation. 'global' is the
        # default and keeps a single (num_operations,) weight vector.
        if selection_mode not in ("global", "per_channel"):
            raise ValueError(
                f"selection_mode must be 'global' or 'per_channel', got "
                f"{selection_mode!r}."
            )

        # Validate and set operation types
        if operation_types is None:
            operation_types = ['add', 'multiply', 'subtract', 'divide', 'power', 'max', 'min']

        valid_operations = {'add', 'multiply', 'subtract', 'divide', 'power', 'max', 'min'}
        invalid_ops = set(operation_types) - valid_operations
        if invalid_ops:
            raise ValueError(
                f"Invalid operation types: {invalid_ops}. "
                f"Valid operations are: {valid_operations}"
            )

        # Validate parameters
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive.")

        if scaling_init <= 0:
            raise ValueError("scaling_init must be positive.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        if power_clip_range[0] <= 0 or power_clip_range[1] <= power_clip_range[0]:
            raise ValueError("power_clip_range must be (min, max) with 0 < min < max.")

        if exponent_clip_range[1] <= exponent_clip_range[0]:
            raise ValueError("exponent_clip_range must be (min, max) with min < max.")

        if not operation_types:
            raise ValueError("operation_types must be a non-empty list.")

        if safe_divide_mode not in ("hard_clamp", "smooth"):
            raise ValueError(
                f"safe_divide_mode must be 'hard_clamp' or 'smooth', got "
                f"{safe_divide_mode!r}."
            )

        if entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative.")

        # Store ALL configuration parameters
        self.operation_types = operation_types
        self.use_temperature = use_temperature
        self.temperature_init = temperature_init
        self.use_scaling = use_scaling
        self.scaling_init = scaling_init
        self.num_operations = len(operation_types)
        self.operation_initializer = keras.initializers.get(operation_initializer)

        # Set default initializers if not provided
        if temperature_initializer is None or temperature_initializer == "constant":
            self.temperature_initializer = keras.initializers.Constant(temperature_init)
        else:
            self.temperature_initializer = keras.initializers.get(temperature_initializer)

        if scaling_initializer is None or scaling_initializer == "constant":
            self.scaling_initializer = keras.initializers.Constant(scaling_init)
        else:
            self.scaling_initializer = keras.initializers.get(scaling_initializer)

        self.epsilon = epsilon
        self.power_clip_range = power_clip_range
        self.exponent_clip_range = exponent_clip_range
        self.softplus_temperature = softplus_temperature
        self.safe_divide_mode = safe_divide_mode
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_hard = gumbel_hard
        self.entropy_coefficient = entropy_coefficient
        self.selection_mode = selection_mode
        self.exponent_clip_mode = exponent_clip_mode

        # Initialize weight attributes - these will be created in build()
        self.operation_weights = None
        self.temperature = None
        self.scaling_factor = None

        logger.debug(
            f"LearnableArithmeticOperator initialized with operations: {operation_types}, "
            f"use_temperature: {use_temperature}, temperature_init: {temperature_init}, "
            f"use_scaling: {use_scaling}, scaling_init: {scaling_init}"
        )

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> None:
        """
        Create the selection weights, the temperature and the scale.

        The temperature exists only under ``use_temperature`` and the scale
        only under ``use_scaling``. In ``per_channel`` mode the selection
        weights are shaped ``(channels, num_operations)``, so the last axis
        of the input must be known here.

        :param input_shape: Shape of the input tensor, or a list of one or
            two such shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :raises ValueError: If two shapes differ, or ``per_channel`` mode
            gets an unknown last axis.
        """
        # Same list-of-shapes detection as compute_output_shape and
        # logic_operators.py: one shape can itself arrive as a list, so
        # look at the first element to tell the two cases apart.
        is_list_of_shapes = (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], (int, type(None)))
        )

        # Validate input shapes for binary operations
        if is_list_of_shapes and len(input_shape) == 2:
            if list(input_shape[0]) != list(input_shape[1]):
                raise ValueError(
                    f"Input tensors must have the same shape for binary operations. "
                    f"Got shapes: {input_shape[0]} and {input_shape[1]}"
                )

        # The learnable selection weights. per_channel mode stores
        # (channels, num_operations) so each channel picks its own
        # operation; global mode stores (num_operations,).
        if self.selection_mode == "per_channel":
            if is_list_of_shapes:
                shape_for_channels = tuple(input_shape[0])
            else:
                shape_for_channels = tuple(input_shape)
            if shape_for_channels[-1] is None:
                raise ValueError(
                    "selection_mode='per_channel' requires a concrete "
                    f"last-axis dimension; got {shape_for_channels}."
                )
            self._channels = int(shape_for_channels[-1])
            weight_shape = (self._channels, self.num_operations)
        else:
            self._channels = None
            weight_shape = (self.num_operations,)

        self.operation_weights = self.add_weight(
            name="operation_weights",
            shape=weight_shape,
            initializer=self.operation_initializer,
            trainable=True,
        )

        # The temperature weight, when one was asked for. Under
        # softplus_temperature the stored weight is the pre-softplus raw
        # value, initialized so that softplus(raw) == temperature_init.
        if self.use_temperature:
            if self.softplus_temperature:
                # The inverse of softplus is log(exp(y) - 1), which is
                # close to y once y is well above 0.
                raw_init = float(math.log(math.expm1(self.temperature_init)))
                temp_initializer = keras.initializers.Constant(raw_init)
            else:
                temp_initializer = self.temperature_initializer
            self.temperature = self.add_weight(
                name="temperature",
                shape=(),
                initializer=temp_initializer,
                trainable=True,
            )

        # The output scale, when one was asked for.
        if self.use_scaling:
            self.scaling_factor = self.add_weight(
                name="scaling_factor",
                shape=(),
                initializer=self.scaling_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def _safe_divide(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Divide x1 by x2 without letting the denominator reach zero.

        ``safe_divide_mode`` picks between two guards:

        .. code-block:: text

            'hard_clamp'            'smooth'
            s = sign(x2), 0 -> +1        x1 * x2
            d = s * max(|x2|, eps)    -------------
            result = x1 / d            x2^2 + eps^2

            hard_clamp is exactly x1/x2 whenever
            |x2| >= eps, and its gradient steps at
            x2 = 0.
            smooth is within O((eps/x2)^2) of x1/x2
            far from zero, returns 0 at x2 = 0, and
            its gradient in x2 never exceeds
            |x1| / (2 * eps).

        In ``'smooth'`` mode ``f(x1, 0)`` is 0, not the limit of x1/x2.
        That is the price of the bounded gradient. Use
        ``safe_divide_mode='hard_clamp'`` when you need exact divide
        semantics near zero.

        :param x1: Numerator tensor.
        :type x1: keras.KerasTensor
        :param x2: Denominator tensor, same shape as ``x1``.
        :type x2: keras.KerasTensor
        :return: The guarded quotient.
        :rtype: keras.KerasTensor
        """
        if self.safe_divide_mode == "smooth":
            # DECISION plan_2026-05-13_a2b0f17b/D-001 — bounded-gradient
            # smooth division. Far from zero this is x1/x2 to within
            # O((eps/x2)^2). At x2=0 it is 0 and the gradient in x2 is at
            # most |x1| / (2 * eps). Do not replace it with a plain divide.
            # Owning plan dir gone; this comment is the record.
            denom = keras.ops.add(keras.ops.square(x2), keras.ops.cast(self.epsilon ** 2, x2.dtype))
            return keras.ops.divide(keras.ops.multiply(x1, x2), denom)

        # hard_clamp: floor the magnitude, keep the sign, then divide.
        sign_x2 = keras.ops.sign(x2)
        sign_x2 = keras.ops.where(keras.ops.equal(sign_x2, 0.0), keras.ops.ones_like(sign_x2), sign_x2)
        safe_x2 = sign_x2 * keras.ops.maximum(keras.ops.abs(x2), self.epsilon)
        return keras.ops.divide(x1, safe_x2)

    def _safe_power(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Raise x1 to the power x2, with both sides clipped.

        A negative base with a fractional exponent has no real value, so
        this returns the real part of the complex power:
        ``cos(pi*y) * |x1|^y``. That agrees with ``x1 ** x2`` whenever the
        base is non-negative or the exponent is a whole number.

        **Block internals:**

        .. code-block:: text

            |x1| ─► clip to power_clip_range ─► base
            x2   ─► clip, or tanh squash into ─► y
                    exponent_clip_range
                    ('smooth' keeps a gradient
                     outside the range)
                             │
            sign ─► cos(pi*y) where x1 < 0,
                    +1 where x1 >= 0
                             │
                             ▼
                   sign * base ** y

            The base clip floors |x1| away from 0 and
            caps it, so the result stays finite for
            every exponent in range.

        :param x1: Base tensor.
        :type x1: keras.KerasTensor
        :param x2: Exponent tensor, same shape as ``x1``.
        :type x2: keras.KerasTensor
        :return: The guarded power, keeping the sign of the base.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-05-13_a2b0f17b/D-001 — real restriction of the
        # complex power: Re((-|x|)^y) = cos(pi*y) * |x|^y. On a negative
        # base that gives +1 for even y, -1 for odd y, 0 at half-integer y.
        # Do not drop the sign and return |x|^y, as an earlier version did.
        # Owning plan dir gone; this comment is the record.
        x1_abs_safe = keras.ops.clip(
            keras.ops.abs(x1), self.power_clip_range[0], self.power_clip_range[1]
        )
        if self.exponent_clip_mode == "smooth":
            # tanh squash into the range, with a gradient everywhere.
            lo, hi = self.exponent_clip_range
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            x2_safe = keras.ops.add(mid, keras.ops.multiply(half, keras.ops.tanh(keras.ops.divide(keras.ops.subtract(x2, mid), half))))
        else:
            x2_safe = keras.ops.clip(x2, self.exponent_clip_range[0], self.exponent_clip_range[1])
        magnitude = keras.ops.power(x1_abs_safe, x2_safe)
        # The sign: +1 where the base is non-negative, cos(pi*y) where it
        # is negative.
        is_negative = keras.ops.cast(keras.ops.less(x1, 0.0), x1.dtype)
        sign_component = (
            keras.ops.cos(keras.ops.multiply(math.pi, x2_safe)) * is_negative
            + (keras.ops.cast(1.0, x1.dtype) - is_negative)
        )
        return keras.ops.multiply(sign_component, magnitude)

    def _soft_max(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Return the elementwise maximum of the two inputs.

        The name says soft, but this is the plain maximum. The softness of
        the layer comes from the weighted mix over operations, not from
        this operation.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor, same shape as ``x1``.
        :type x2: keras.KerasTensor
        :return: The elementwise maximum.
        :rtype: keras.KerasTensor
        """
        return keras.ops.maximum(x1, x2)

    def _soft_min(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Return the elementwise minimum of the two inputs.

        As with :meth:`_soft_max`, this is the plain minimum.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor, same shape as ``x1``.
        :type x2: keras.KerasTensor
        :return: The elementwise minimum.
        :rtype: keras.KerasTensor
        """
        return keras.ops.minimum(x1, x2)

    def _resolve_temperature(self) -> keras.KerasTensor:
        """
        Return the temperature actually used by the softmax.

        Under ``softplus_temperature`` the stored weight is a raw value and
        the temperature is ``softplus(raw)``. Either way the result is
        floored at 1e-7, so the division can never blow up.

        :return: A positive scalar temperature.
        :rtype: keras.KerasTensor
        """
        if self.softplus_temperature:
            # softplus(raw) is already > 0; the floor is a final guard.
            return keras.ops.maximum(keras.ops.softplus(self.temperature), 1e-7)
        return keras.ops.maximum(self.temperature, 1e-7)

    def _operation_probs(
        self,
        training: Optional[bool] = None,
        deterministic: bool = False,
    ) -> keras.KerasTensor:
        """
        Return one selection probability per operation.

        The result is ``(N,)`` in global mode and ``(C, N)`` per channel,
        and sums to 1 on the last axis.

        **Selection head:**

        .. code-block:: text

            operation_weights w    (N,) or (C, N)
                     │
                     ▼
            gumbel_softmax, training is True and
            deterministic is False?
                     │
              no ────┴──── yes
               │            │
               ▼            ▼
            logits       g = -log(-log(U))
            = w / T      logits = (w + g) / T
               │            │
               ▼            ▼
            softmax      softmax
               │            │
               │            ▼
               │         gumbel_hard: one-hot(argmax)
               │         added through stop_gradient
               │            │
               └─────┬──────┘
                     ▼
                   probs

            T is _resolve_temperature(). The division
            is skipped when use_temperature is False.

        # DECISION plan_2026-05-13_3a2f1d23/D-001
        # Canonical Jang (2017) Gumbel-softmax form: softmax((w + g) / T).
        # Do not compute softmax((w / T) + g).
        # Owning plan dir gone; this comment is the record.

        # DECISION plan_2026-05-13_e33114da/D-003
        # Gumbel noise is added only when training is True. training=False,
        # training=None and deterministic=True all skip it, so predict() is
        # reproducible. Do not key this on gumbel_softmax alone.
        # Owning plan dir gone; this comment is the record.

        :param training: Keras training flag. Noise is added only when this
            is exactly ``True``.
        :type training: Optional[bool]
        :param deterministic: Skip the noise whatever ``training`` says.
            ``to_symbolic()`` passes True so its output repeats.
        :type deterministic: bool
        :return: Probability of each operation.
        :rtype: keras.KerasTensor
        """
        weights = self.operation_weights
        skip_gumbel = deterministic or (training is not True)

        if self.gumbel_softmax and not skip_gumbel:
            # Gumbel(0,1) is -log(-log(U(0,1))), written out because
            # keras.ops has no sampler for it.
            uniform = keras.random.uniform(
                shape=keras.ops.shape(weights), minval=1e-9, maxval=1.0
            )
            gumbel = keras.ops.negative(keras.ops.log(keras.ops.negative(keras.ops.log(uniform))))
            # Canonical form: (w + g) / T then softmax (NOT softmax(w/T) + g).
            noisy = keras.ops.add(weights, gumbel)
            if self.use_temperature:
                temp = self._resolve_temperature()
                logits = keras.ops.divide(noisy, temp)
            else:
                logits = noisy
            soft = keras.ops.softmax(logits, axis=-1)
            if self.gumbel_hard:
                # Straight-through: the forward value is the one-hot, the
                # gradient is the soft sample's.
                idx = keras.ops.argmax(soft, axis=-1)
                hard = keras.ops.one_hot(idx, num_classes=self.num_operations)
                hard = keras.ops.cast(hard, soft.dtype)
                return keras.ops.add(soft, keras.ops.stop_gradient(keras.ops.subtract(hard, soft)))
            return soft

        # No noise, or deterministic=True: plain temperature-scaled
        # softmax.
        if self.use_temperature:
            temp = self._resolve_temperature()
            logits = keras.ops.divide(weights, temp)
        else:
            logits = weights
        return keras.ops.softmax(logits, axis=-1)

    def _maybe_add_entropy_loss(
        self, probs: keras.KerasTensor
    ) -> None:
        """
        Add a loss that pushes the selection toward one operation.

        The added term is ``entropy_coefficient * H(probs)``. Since high
        entropy costs more, training is pushed toward a peaked
        distribution. Nothing is added when the coefficient is 0.

        :param probs: The selection probabilities from
            :meth:`_operation_probs`.
        :type probs: keras.KerasTensor
        :return: Nothing. The loss is registered with ``add_loss``.
        :rtype: None
        """
        if self.entropy_coefficient > 0:
            log_p = keras.ops.log(keras.ops.add(probs, 1e-12))
            ent = keras.ops.negative(keras.ops.sum(keras.ops.multiply(probs, log_p)))
            # High entropy costs more, so training sharpens the selection.
            self.add_loss(keras.ops.multiply(self.entropy_coefficient, ent))

    def to_symbolic(self, top_k: int = 1, deterministic: bool = True) -> str:
        """
        Report the operations the layer currently favours.

        In ``per_channel`` mode the probabilities are averaged over the
        channels first, so the answer is one ranking for the whole layer.
        Read ``operation_weights`` directly if you need per-channel detail.

        :param top_k: How many operations to report, highest probability
            first.
        :type top_k: int
        :param deterministic: Skip the Gumbel noise so repeated calls agree.
            Pass False only if you want a sample instead.
        :type deterministic: bool
        :return: A string like ``"multiply(0.812), add(0.101)"``.
        :rtype: str
        :raises RuntimeError: If the layer has not been built.
        """
        if self.operation_weights is None:
            raise RuntimeError("Layer has not been built yet.")
        probs_arr = keras.ops.convert_to_numpy(
            self._operation_probs(deterministic=deterministic)
        )
        if self.selection_mode == "per_channel":
            # Averaging over channels gives one ranking, which is what a
            # top-k summary needs. The per-channel detail is lost here;
            # read operation_weights for it.
            probs = probs_arr.mean(axis=0).tolist()
        else:
            probs = probs_arr.tolist()
        ranked = sorted(
            zip(self.operation_types, probs), key=lambda kv: -kv[1]
        )[:top_k]
        return ", ".join(f"{name}({p:.3f})" for name, p in ranked)

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run every selected operation and return their weighted sum.

        **How the two selection modes combine the operations:**

        .. code-block:: text

            'global'              'per_channel'
            probs (N,)            probs (C, N)
            stack on axis 0       stack on axis -1
             -> (N, *x.shape)      -> (*x.shape, N)
            reshape probs to      reshape probs to
             (N, 1, ..., 1)        (1, ..., 1, C, N)
            sum on axis 0         sum on axis -1
                  │                      │
                  └──────────┬───────────┘
                             ▼
                  * scaling_factor, if any
                             ▼
                 output, shaped like x1

            selection_mode picks the column.

        :param inputs: One tensor, or a list of one or two tensors of the
            same shape.
        :type inputs: Union[keras.KerasTensor, List[keras.KerasTensor]]
        :param training: Keras training flag. Only the Gumbel path reads
            it; the operations themselves behave the same either way.
        :type training: Optional[bool]
        :return: The combined result, shaped like the first input.
        :rtype: keras.KerasTensor
        :raises ValueError: If more than two tensors are given.
        """
        # Input parsing. A single tensor is used for both operands.
        if isinstance(inputs, list):
            if len(inputs) == 2:
                x1, x2 = inputs
            elif len(inputs) == 1:
                x1 = inputs[0]
                # One tensor: use it for both operands.
                x2 = inputs[0]
            else:
                raise ValueError(f"Expected 1 or 2 inputs, got {len(inputs)}")
        else:
            x1 = inputs
            x2 = inputs

        # One probability per operation, plus the optional entropy penalty.
        operation_probs = self._operation_probs(training=training)
        self._maybe_add_entropy_loss(operation_probs)

        # Run every selected operation.
        operations = []
        for op_type in self.operation_types:
            if op_type == 'add':
                result = keras.ops.add(x1, x2)
            elif op_type == 'multiply':
                result = keras.ops.multiply(x1, x2)
            elif op_type == 'subtract':
                result = keras.ops.subtract(x1, x2)
            elif op_type == 'divide':
                result = self._safe_divide(x1, x2)
            elif op_type == 'power':
                result = self._safe_power(x1, x2)
            elif op_type == 'max':
                result = self._soft_max(x1, x2)
            elif op_type == 'min':
                result = self._soft_min(x1, x2)
            else:
                logger.warning(f"Unknown operation type: {op_type}, using identity")
                result = x1
            operations.append(result)

        # Weighted combination, stacked and summed in one shot.
        if self.selection_mode == "per_channel":
            # operation_probs is (C, N), so stack the results on a new last
            # axis to match, broadcast the probs to (1,...,1,C,N) and sum
            # the last axis.
            # Shape after this: (..., C, N)
            stacked = keras.ops.stack(operations, axis=-1)
            rank = len(stacked.shape)
            probs_bshape = (1,) * (rank - 2) + (self._channels, self.num_operations)
            weights = keras.ops.reshape(operation_probs, probs_bshape)
            output = keras.ops.sum(keras.ops.multiply(weights, stacked), axis=-1)
        else:
            # Global: one weight vector for the whole tensor.
            stacked = keras.ops.stack(operations, axis=0)
            n = self.num_operations
            weight_shape = (n,) + (1,) * (len(stacked.shape) - 1)
            weights = keras.ops.reshape(operation_probs, weight_shape)
            output = keras.ops.sum(keras.ops.multiply(weights, stacked), axis=0)

        # Apply the learnable scale.
        if self.use_scaling:
            # Floor the magnitude at 1e-7 but keep the sign. Taking abs()
            # here instead would make a negative scale unreachable.
            abs_s = keras.ops.maximum(keras.ops.abs(self.scaling_factor), 1e-7)
            sign_s = keras.ops.sign(self.scaling_factor)
            sign_s = keras.ops.where(keras.ops.equal(sign_s, 0.0), keras.ops.ones_like(sign_s), sign_s)
            scale = keras.ops.multiply(sign_s, abs_s)
            output = keras.ops.multiply(output, scale)

        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the first input shape.

        :param input_shape: Shape of the input, or a list of one or two
            such shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: The shape of the first input.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If two shapes are given and they differ.
        """
        is_list_of_shapes = (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], (int, type(None)))
        )
        if is_list_of_shapes:
            # Two inputs must agree in shape, same rule as build().
            if len(input_shape) == 2 and list(input_shape[0]) != list(input_shape[1]):
                raise ValueError(
                    f"Input tensors must have the same shape for binary operations. "
                    f"Got shapes: {input_shape[0]} and {input_shape[1]}"
                )
            return tuple(input_shape[0])
        return tuple(input_shape) if isinstance(input_shape, list) else input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return every constructor argument, for serialization.

        The three initializers are serialized, so a round trip restores the
        same objects. Nothing created in ``build`` appears here.

        :return: The layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "operation_types": self.operation_types,
            "use_temperature": self.use_temperature,
            "temperature_init": self.temperature_init,
            "use_scaling": self.use_scaling,
            "scaling_init": self.scaling_init,
            "operation_initializer": keras.initializers.serialize(self.operation_initializer),
            "temperature_initializer": keras.initializers.serialize(self.temperature_initializer),
            "scaling_initializer": keras.initializers.serialize(self.scaling_initializer),
            "epsilon": self.epsilon,
            "power_clip_range": self.power_clip_range,
            "exponent_clip_range": self.exponent_clip_range,
            "softplus_temperature": self.softplus_temperature,
            "safe_divide_mode": self.safe_divide_mode,
            "gumbel_softmax": self.gumbel_softmax,
            "gumbel_hard": self.gumbel_hard,
            "entropy_coefficient": self.entropy_coefficient,
            "selection_mode": self.selection_mode,
            "exponent_clip_mode": self.exponent_clip_mode,
        })
        return config

# ---------------------------------------------------------------------