"""
A layer that learns which logic gate to apply.

The layer holds a set of soft Boolean gates and one learnable weight per
gate. It runs every gate on the inputs and returns their weighted sum, so
the choice of gate is trained by gradient descent along with everything
else. Use it where you would otherwise add or concatenate two feature
tensors and would rather have the combination learned.

A forward pass has three stages:

    1. Normalize. Inputs are mapped to [0, 1] by a sigmoid, so they read as
       fuzzy truth values: 0 is false and 1 is true. Turn the sigmoid off
       when the caller already supplies values in that range.

    2. Run every gate. Each gate named in ``operation_types`` is applied to
       the normalized inputs. Every gate is smooth, and each one agrees
       with its Boolean counterpart at the corners 0 and 1.

    3. Combine. A softmax over the learnable weights gives one probability
       per gate, and the output is the weighted sum of the gate results.

The math:

    The gates come from fuzzy logic. For p and q in [0, 1] the six default
    gates are

        NOT(p)     = 1 - p
        AND(p, q)  = p * q              (the product t-norm)
        OR(p, q)   = p + q - p*q        (the probabilistic sum)
        XOR(p, q)  = p + q - 2*p*q
        NAND(p, q) = 1 - p*q
        NOR(p, q)  = 1 - (p + q - p*q)

    Twelve more gates can be selected: the Lukasiewicz, Godel, Hamacher and
    Yager t-norm families plus four implications. The class docstring lists
    all eighteen with their formulas.

    Selection is a softmax over the learnable weight vector w with
    temperature T. The probability of gate i is

        alpha_i = exp(w_i / T) / sum_j(exp(w_j / T))

    and the output is

        Y = sum_i(alpha_i * f_i(X))

    Gradients reach both w and T, so the logical structure is learned from
    data rather than fixed by hand.

Every weight clones the initializer it is given, so one ``Initializer``
INSTANCE passed to two parameters, or handed down by a parent to every
child, still leaves each weight with an independent draw. A seeded
instance keeps its seed and so keeps drawing the same values.

References:
    - Liu, H., Simonyan, K., & Yang, Y. (2018). "DARTS: Differentiable
      Architecture Search". The continuous relaxation of a discrete choice
      that this layer applies to logic gates.

    - Zadeh, L. A. (1965). "Fuzzy sets". Information and Control. The
      source of the soft gate forms.

    - Garcez, A. S., Broda, K., & Gabbay, D. M. (2002). "Neural-Symbolic
      Learning Systems: Foundations and Applications". The wider
      neuro-symbolic setting.
"""

import math
import keras
from typing import List, Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers.clone import clone_initializer


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LearnableLogicOperator(keras.layers.Layer):
    """
    A learnable choice among 18 soft logic gates.

    Give the layer two tensors of the same shape, or one tensor when every
    gate you asked for is unary. It maps the inputs into [0, 1], runs each
    gate named in ``operation_types``, and returns the weighted sum of the
    results. The weights are trained, so the layer learns which gate the
    task wants.

    The map into [0, 1] is a sigmoid by default. Pass
    ``apply_sigmoid=False`` when the caller already produces values in that
    range, which is what you want when stacking these layers. If the
    upstream layer can leave the range, add
    ``force_clip_when_no_sigmoid=True``.

    Passing one tensor for a binary gate would set ``x2 = x1``, and then
    XOR(p, p) is always 0 and AND(p, p) is just p. The layer raises instead.
    ``allow_unary_degenerate=True`` brings the old rebinding back.

    **Architecture Overview:**

    .. code-block:: text

        x1            x2      one tensor sets x2 = x1
         │             │
         └──────┬──────┘
                ▼
        ┌───────────────────────────────┐
        │ unary guard                   │
        │ raises on a binary gate with  │
        │ one tensor, unless            │
        │ allow_unary_degenerate        │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ sigmoid      apply_sigmoid    │
        │ clip 0..1    force_clip only  │
        │ pass through neither flag     │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ every gate in operation_types │
        │ runs in parallel -> N results │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ weighted sum, alpha from      │
        │ _operation_probs              │
        └───────────────┬───────────────┘
                        ▼
        output, same shape as x1

    **The 18 gates, read from VALID_OPS and call():**

    .. code-block:: text

        key                  f(p, q)
        -------------------  --------------------------
        and                  p * q
        or                   p + q - p*q
        xor                  p + q - 2*p*q
        not                  1 - p
        nand                 1 - p*q
        nor                  1 - (p + q - p*q)
        lukasiewicz_and      max(0, p + q - 1)
        lukasiewicz_or       min(1, p + q)
        godel_and            min(p, q)
        godel_or             max(p, q)
        implies              max(1 - p, q)
        lukasiewicz_implies  min(1, 1 - p + q)
        reichenbach_implies  1 - p + p*q
        goguen_implies       min(1, q / max(p, 1e-9))
        hamacher_and         p*q / (p + q - p*q)
        hamacher_or          (p+q-2*p*q) / (1 - p*q)
        yager_and            1 - min(1, S(1-p, 1-q))
        yager_or             min(1, S(p, q))

        S(a, b) = (a^w + b^w)^(1/w), w = yager_p.
        'not' is the only unary gate; it reads x1 and
        ignores x2. The default operation_types is the
        first six rows. The two Hamacher gates return
        their limit at the corner where the ratio is
        0/0: 0 for AND at (0,0), 1 for OR at (1,1).

    :param operation_types: Gates to select among, named by the keys in the
        table above. ``None`` gives the six default gates, not all 18.
    :type operation_types: Optional[List[str]]
    :param use_temperature: Divide the selection weights by a learnable
        temperature before the softmax.
    :type use_temperature: bool
    :param temperature_init: Starting temperature. Must be positive.
    :type temperature_init: float
    :param operation_initializer: Initializer for the selection weights.
        ``"zeros"`` starts every gate equally likely.
    :type operation_initializer: Union[str, keras.initializers.Initializer]
    :param temperature_initializer: Initializer for the temperature weight.
        Read only when ``softplus_temperature`` is False; the softplus path
        computes its own raw value from ``temperature_init``.
    :type temperature_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param apply_sigmoid: Map the inputs through a sigmoid first. Set False
        when they already lie in [0, 1].
    :type apply_sigmoid: bool
    :param force_clip_when_no_sigmoid: Clip the inputs to [0, 1] when
        ``apply_sigmoid`` is False. Inert when it is True.
    :type force_clip_when_no_sigmoid: bool
    :param softplus_temperature: Store the temperature as a raw value and
        read it back through softplus, which keeps it positive without a
        constraint. The stored weight is then not the temperature itself.
    :type softplus_temperature: bool
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
    :param allow_unary_degenerate: Allow one input tensor to be used for
        both operands of a binary gate. Off by default because it makes
        binary gates meaningless.
    :type allow_unary_degenerate: bool
    :param selection_mode: ``"global"`` learns one gate choice for the whole
        tensor. ``"per_channel"`` learns one per channel and needs a known
        last axis at build time.
    :type selection_mode: str
    :param yager_p: The w exponent of the Yager t-norm pair. Must be > 0.
        Larger w moves the Yager gates toward min and max.
    :type yager_p: float
    :param kwargs: Passed to ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar operation_types: The gate keys this layer selects among.
    :vartype operation_types: List[str]
    :ivar use_temperature: Whether a temperature weight exists.
    :vartype use_temperature: bool
    :ivar temperature_init: The requested starting temperature.
    :vartype temperature_init: float
    :ivar apply_sigmoid: Whether ``call`` applies the sigmoid.
    :vartype apply_sigmoid: bool
    :ivar force_clip_when_no_sigmoid: Whether ``call`` clips instead.
    :vartype force_clip_when_no_sigmoid: bool
    :ivar softplus_temperature: Whether the temperature weight is raw.
    :vartype softplus_temperature: bool
    :ivar gumbel_softmax: Whether training samples the selection.
    :vartype gumbel_softmax: bool
    :ivar gumbel_hard: Whether the sampled selection is straight-through.
    :vartype gumbel_hard: bool
    :ivar entropy_coefficient: Weight of the entropy loss.
    :vartype entropy_coefficient: float
    :ivar allow_unary_degenerate: Whether the unary guard is disabled.
    :vartype allow_unary_degenerate: bool
    :ivar selection_mode: ``"global"`` or ``"per_channel"``.
    :vartype selection_mode: str
    :ivar yager_p: The stored Yager exponent, as a float.
    :vartype yager_p: float
    :ivar num_operations: ``len(operation_types)``; the N of every shape
        note in this file.
    :vartype num_operations: int
    :ivar operation_initializer: The resolved selection-weight initializer.
    :vartype operation_initializer: keras.initializers.Initializer
    :ivar temperature_initializer: The resolved temperature initializer.
    :vartype temperature_initializer: keras.initializers.Initializer
    :ivar operation_weights: Selection weights, ``(N,)`` in global mode and
        ``(C, N)`` per channel. ``None`` until ``build``.
    :vartype operation_weights: Optional[keras.Variable]
    :ivar temperature: Scalar temperature weight, or ``None`` when
        ``use_temperature`` is False or the layer is not built.
    :vartype temperature: Optional[keras.Variable]
    :ivar VALID_OPS: Class constant. Every accepted gate key.
    :vartype VALID_OPS: frozenset
    :ivar UNARY_OPS: Class constant. The gate keys that read one operand.
    :vartype UNARY_OPS: frozenset
    :ivar BINARY_OPS: Class constant. ``VALID_OPS - UNARY_OPS``; the unary
        guard fires on these.
    :vartype BINARY_OPS: frozenset

    :raises ValueError: From the constructor if ``selection_mode`` is not
        one of the two keys, ``yager_p`` is not positive,
        ``operation_types`` is empty or names an unknown gate,
        ``temperature_init`` is not positive, or ``entropy_coefficient`` is
        negative.
    :raises ValueError: From ``build`` if two input shapes differ, more than
        two are given, or ``selection_mode="per_channel"`` gets an unknown
        last axis.
    :raises ValueError: From ``call`` if a list of more than two tensors is
        given, or a single tensor is given for a binary gate without
        ``allow_unary_degenerate``.
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
                LearnableLogicOperator,
            )

            p = keras.random.normal((2, 8))
            q = keras.random.normal((2, 8))

            op = LearnableLogicOperator(
                operation_types=['and', 'or', 'xor']
            )
            y = op([p, q])
            y.shape  # (2, 8)

            # Stacked: the second layer reads values already in [0, 1].
            op2 = LearnableLogicOperator(
                operation_types=['and', 'or'], apply_sigmoid=False
            )
            y2 = op2([y, y])
    """

    # Every accepted gate key. A key is binary unless it is in UNARY_OPS.
    # The class docstring holds the formula for each one.
    VALID_OPS = frozenset({
        'and', 'or', 'xor', 'not', 'nand', 'nor',
        'lukasiewicz_and', 'lukasiewicz_or',
        'godel_and', 'godel_or',
        'implies',
        # The Hamacher and Yager t-norm families.
        'hamacher_and', 'hamacher_or',
        'yager_and', 'yager_or',
        # Three implications beyond the Kleene-Dienes 'implies' above.
        'lukasiewicz_implies', 'reichenbach_implies', 'goguen_implies',
    })
    UNARY_OPS = frozenset({'not'})
    BINARY_OPS = VALID_OPS - UNARY_OPS

    def __init__(
            self,
            operation_types: Optional[List[str]] = None,
            use_temperature: bool = True,
            temperature_init: float = 1.0,
            operation_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            temperature_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            apply_sigmoid: bool = True,
            force_clip_when_no_sigmoid: bool = False,
            softplus_temperature: bool = True,
            gumbel_softmax: bool = False,
            gumbel_hard: bool = False,
            entropy_coefficient: float = 0.0,
            allow_unary_degenerate: bool = False,
            selection_mode: str = "global",
            yager_p: float = 2.0,
            **kwargs: Any
    ) -> None:
        """
        Validate the arguments and store the configuration.

        No weight is created here. The selection weights need the channel
        count in ``per_channel`` mode, so every weight is created in
        :meth:`build`. The class docstring documents each parameter.
        """
        super().__init__(**kwargs)

        if selection_mode not in ("global", "per_channel"):
            raise ValueError(
                f"selection_mode must be 'global' or 'per_channel', got "
                f"{selection_mode!r}."
            )
        # yager_p is the w exponent of the Yager pair and must be positive.
        if yager_p <= 0:
            raise ValueError(f"yager_p must be > 0, got {yager_p}.")

        # Validate and set operation types
        if operation_types is None:
            operation_types = ['and', 'or', 'xor', 'not', 'nand', 'nor']

        if not operation_types:
            raise ValueError("operation_types must be a non-empty list.")

        invalid_ops = set(operation_types) - self.VALID_OPS
        if invalid_ops:
            raise ValueError(
                f"Invalid operation types: {invalid_ops}. "
                f"Valid operations are: {sorted(self.VALID_OPS)}"
            )

        # Validate temperature initialization
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive.")
        if entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative.")

        # Store ALL configuration parameters
        self.operation_types = operation_types
        self.use_temperature = use_temperature
        self.temperature_init = temperature_init
        # DECISION plan_2026-05-13_e52a5ac8/D-001 — apply_sigmoid=False is the
        # intended path for stacking. Default True keeps the legacy reading of
        # inputs as raw logits, mapped to [0,1] before the fuzzy gates run.
        # Do not flip the default: it would change every saved caller's math.
        # Owning plan dir gone; this comment is the record.
        self.apply_sigmoid = apply_sigmoid
        # With apply_sigmoid=False the layer trusts the caller to stay in
        # [0, 1]. An arithmetic expert upstream does not. Setting
        # force_clip_when_no_sigmoid=True clips to [0, 1] in that case.
        self.force_clip_when_no_sigmoid = force_clip_when_no_sigmoid
        self.softplus_temperature = softplus_temperature
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_hard = gumbel_hard
        self.entropy_coefficient = entropy_coefficient
        self.allow_unary_degenerate = allow_unary_degenerate
        self.selection_mode = selection_mode
        self.yager_p = float(yager_p)
        self.num_operations = len(operation_types)
        # Set in build() for per_channel mode; stays None in global mode.
        self._channels = None
        self.operation_initializer = keras.initializers.get(operation_initializer)

        # Set default initializer if not provided or if 'constant' is specified
        if temperature_initializer is None or temperature_initializer == "constant":
            self.temperature_initializer = keras.initializers.Constant(temperature_init)
        else:
            self.temperature_initializer = keras.initializers.get(temperature_initializer)

        # Initialize weight attributes - these will be created in build()
        self.operation_weights = None
        self.temperature = None

        logger.debug(
            f"LearnableLogicOperator initialized with operations: {operation_types}, "
            f"use_temperature: {use_temperature}, temperature_init: {temperature_init}"
        )

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> None:
        """
        Create the selection weights and, if enabled, the temperature.

        In ``per_channel`` mode the selection weights are shaped
        ``(channels, num_operations)``, so the last axis of the input must
        be known here.

        :param input_shape: Shape of the input tensor, or a list of one or
            two such shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :raises ValueError: If two shapes differ, more than two shapes are
            given, or ``per_channel`` mode gets an unknown last axis.
        """
        # One shape can itself arrive as a list, for example after
        # deserialization. A list OF shapes has non-dimension elements, so
        # look at the first element to tell the two apart.
        is_list_of_shapes = (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], (int, type(None)))
        )

        # Validate input shapes for binary operations
        if is_list_of_shapes:
            if len(input_shape) == 2:
                if input_shape[0] != input_shape[1]:
                    raise ValueError(
                        f"Input tensors must have the same shape for binary operations. "
                        f"Got shapes: {input_shape[0]} and {input_shape[1]}"
                    )
            elif len(input_shape) > 2:
                raise ValueError(
                    f"Expected 1 or 2 inputs, got {len(input_shape)}"
                )

        # per_channel mode shapes the weight tensor as
        # (channels, num_operations), one gate choice per channel.
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
            weight_shape = (self.num_operations,)

        # The learnable selection weights.
        # DECISION plan-2026-08-29T112804-aff039c4/D-001 -- clone at the
        # add_weight site, not at the hand-off. One resolved Initializer
        # INSTANCE redraws identical values at every weight whose shape
        # matches, so a parent handing one object to every child, or a
        # caller aliasing one object across two roles, gets bit-identical
        # weights (MEASURED max|delta| 0.0 on 8 pairs in this package). A
        # raw string is safe -- Keras resolves it once per consumer. A
        # seeded instance defeats the clone by design, which is why the
        # guards use an unseeded one. Do not put self.operation_initializer
        # back into this call. See decisions.md D-001.
        self.operation_weights = self.add_weight(
            name="operation_weights",
            shape=weight_shape,
            initializer=clone_initializer(self.operation_initializer),
            trainable=True,
        )

        # The temperature weight, when one was asked for. Under
        # softplus_temperature the stored value is the pre-softplus raw
        # value, initialized so that softplus(raw) == temperature_init.
        if self.use_temperature:
            if self.softplus_temperature:
                raw_init = float(math.log(math.expm1(self.temperature_init)))
                temp_initializer = keras.initializers.Constant(raw_init)
            else:
                temp_initializer = self.temperature_initializer
            self.temperature = self.add_weight(
                name="temperature",
                shape=(),
                initializer=clone_initializer(temp_initializer),
                trainable=True,
            )

        super().build(input_shape)

    def _soft_logic_and(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Soft differentiable AND: ``p * q``.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor.
        :type x2: keras.KerasTensor
        :return: Result of soft AND operation.
        :rtype: keras.KerasTensor
        """
        return keras.ops.multiply(x1, x2)

    def _soft_logic_or(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Soft differentiable OR: ``p + q - p*q``.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor.
        :type x2: keras.KerasTensor
        :return: Result of soft OR operation.
        :rtype: keras.KerasTensor
        """
        return keras.ops.add(
            keras.ops.add(x1, x2),
            keras.ops.negative(
                keras.ops.multiply(x1, x2)))

    def _soft_logic_xor(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Soft differentiable XOR: ``p + q - 2*p*q``.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor.
        :type x2: keras.KerasTensor
        :return: Result of soft XOR operation.
        :rtype: keras.KerasTensor
        """
        return keras.ops.subtract(
            keras.ops.add(x1, x2),
            keras.ops.multiply(2.0, keras.ops.multiply(x1, x2)))

    def _soft_logic_not(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Soft differentiable NOT: ``1 - p``.

        :param x: Input tensor.
        :type x: keras.KerasTensor
        :return: Result of soft NOT operation.
        :rtype: keras.KerasTensor
        """
        return keras.ops.subtract(1.0, x)

    def _soft_logic_nand(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Soft differentiable NAND: ``1 - p*q``.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor.
        :type x2: keras.KerasTensor
        :return: Result of soft NAND operation.
        :rtype: keras.KerasTensor
        """
        return keras.ops.subtract(1.0, keras.ops.multiply(x1, x2))

    def _soft_logic_nor(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Soft differentiable NOR: ``1 - (p + q - p*q)``.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor.
        :type x2: keras.KerasTensor
        :return: Result of soft NOR operation.
        :rtype: keras.KerasTensor
        """
        or_result = (
            keras.ops.add(
                keras.ops.add(x1, x2),
                keras.ops.negative(
                    keras.ops.multiply(x1, x2)
                )
            )
        )
        return keras.ops.subtract(1.0, or_result)

    # --- Łukasiewicz t-norm / t-conorm -----------------------------------
    def _luk_and(self, x1, x2):
        """
        Łukasiewicz AND: ``max(0, p + q - 1)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        return keras.ops.maximum(0.0, keras.ops.subtract(keras.ops.add(x1, x2), 1.0))

    def _luk_or(self, x1, x2):
        """
        Łukasiewicz OR: ``min(1, p + q)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        return keras.ops.minimum(1.0, keras.ops.add(x1, x2))

    # --- Gödel t-norm / t-conorm -----------------------------------------
    def _godel_and(self, x1, x2):
        """
        Gödel AND: ``min(p, q)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        return keras.ops.minimum(x1, x2)

    def _godel_or(self, x1, x2):
        """
        Gödel OR: ``max(p, q)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        return keras.ops.maximum(x1, x2)

    # --- Implication family ----------------------------------------------
    def _implies(self, x1, x2):
        """
        Kleene-Dienes implication: ``max(1 - p, q)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        return keras.ops.maximum(keras.ops.subtract(1.0, x1), x2)

    def _lukasiewicz_implies(self, x1, x2):
        """
        Łukasiewicz implication: ``min(1, 1 - p + q)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        return keras.ops.minimum(1.0, keras.ops.add(keras.ops.subtract(1.0, x1), x2))

    def _reichenbach_implies(self, x1, x2):
        """
        Reichenbach implication: ``1 - p + p*q``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        return keras.ops.add(keras.ops.subtract(1.0, x1), keras.ops.multiply(x1, x2))

    def _goguen_implies(self, x1, x2):
        """
        Goguen implication: ``min(1, q / max(p, 1e-9))``.

        Returns 1 wherever p <= q.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        # The 1e-9 floor keeps the ratio finite at p=0, where the formal
        # value is 1: false implies anything.
        p_safe = keras.ops.maximum(x1, 1e-9)
        return keras.ops.minimum(1.0, keras.ops.divide(x2, p_safe))

    # --- Hamacher / Yager t-norms ----------------------------------------

    # DECISION plan_2026-05-13_e33114da/D-002 — both Hamacher gates hit 0/0
    # at one corner: AND at (0,0), OR at (1,1). keras.ops.where returns the
    # limit there, 0 for AND and 1 for OR. Do not go back to per-gate eps
    # clamps: that made OR(1,1) return 0 instead of 1.
    # Owning plan dir gone; this comment is the record.
    _HAMACHER_SINGULAR_EPS = 1e-7

    def _hamacher_and(self, x1, x2):
        """
        Hamacher product t-norm: ``p*q / (p + q - p*q)``.

        The denominator is 0 only at p = q = 0, where this returns 0.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        pq = keras.ops.multiply(x1, x2)
        denom = keras.ops.subtract(keras.ops.add(x1, x2), pq)
        denom_safe = keras.ops.maximum(denom, 1e-9)
        singular = keras.ops.less(denom, self._HAMACHER_SINGULAR_EPS)
        ratio = keras.ops.divide(pq, denom_safe)
        return keras.ops.where(singular, keras.ops.zeros_like(ratio), ratio)

    def _hamacher_or(self, x1, x2):
        """
        Hamacher sum t-conorm: ``(p + q - 2*p*q) / (1 - p*q)``.

        The denominator is 0 only at p = q = 1, where this returns 1.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        pq = keras.ops.multiply(x1, x2)
        num = keras.ops.subtract(keras.ops.add(x1, x2), keras.ops.multiply(2.0, pq))
        denom = keras.ops.subtract(1.0, pq)
        denom_safe = keras.ops.maximum(denom, 1e-9)
        singular = keras.ops.less(denom, self._HAMACHER_SINGULAR_EPS)
        ratio = keras.ops.divide(num, denom_safe)
        return keras.ops.where(singular, keras.ops.ones_like(ratio), ratio)

    def _yager_and(self, x1, x2):
        """
        Yager t-norm: ``1 - min(1, ((1-p)^w + (1-q)^w)^(1/w))``.

        w is ``yager_p``. Larger w moves this gate toward ``min(p, q)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        w = self.yager_p
        a = keras.ops.power(keras.ops.maximum(keras.ops.subtract(1.0, x1), 0.0), w)
        b = keras.ops.power(keras.ops.maximum(keras.ops.subtract(1.0, x2), 0.0), w)
        s = keras.ops.power(keras.ops.add(a, b), 1.0 / w)
        return keras.ops.subtract(1.0, keras.ops.minimum(s, 1.0))

    def _yager_or(self, x1, x2):
        """
        Yager t-conorm: ``min(1, (p^w + q^w)^(1/w))``.

        w is ``yager_p``. Larger w moves this gate toward ``max(p, q)``.

        :param x1: First operand, expected in [0, 1].
        :type x1: keras.KerasTensor
        :param x2: Second operand, expected in [0, 1].
        :type x2: keras.KerasTensor
        :return: The gate output, elementwise.
        :rtype: keras.KerasTensor
        """
        w = self.yager_p
        a = keras.ops.power(keras.ops.maximum(x1, 0.0), w)
        b = keras.ops.power(keras.ops.maximum(x2, 0.0), w)
        s = keras.ops.power(keras.ops.add(a, b), 1.0 / w)
        return keras.ops.minimum(s, 1.0)

    # --- Selection helpers -----------------------------------------------
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
            return keras.ops.maximum(keras.ops.softplus(self.temperature), 1e-7)
        return keras.ops.maximum(self.temperature, 1e-7)

    def _operation_probs(
        self,
        training: Optional[bool] = None,
        deterministic: bool = False,
    ) -> keras.KerasTensor:
        """
        Return one selection probability per gate.

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
        # Do not compute softmax((w / T) + g): that over-weights the noise
        # at low temperature and breaks the Concrete distribution.
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
        :return: Probability of each gate.
        :rtype: keras.KerasTensor
        """
        weights = self.operation_weights
        skip_gumbel = deterministic or (training is not True)

        if self.gumbel_softmax and not skip_gumbel:
            uniform = keras.random.uniform(
                shape=keras.ops.shape(weights), minval=1e-9, maxval=1.0
            )
            gumbel = keras.ops.negative(keras.ops.log(keras.ops.negative(keras.ops.log(uniform))))
            noisy = keras.ops.add(weights, gumbel)
            if self.use_temperature:
                temp = self._resolve_temperature()
                logits = keras.ops.divide(noisy, temp)
            else:
                logits = noisy
            soft = keras.ops.softmax(logits, axis=-1)
            if self.gumbel_hard:
                idx = keras.ops.argmax(soft, axis=-1)
                hard = keras.ops.cast(
                    keras.ops.one_hot(idx, num_classes=self.num_operations), soft.dtype
                )
                return keras.ops.add(soft, keras.ops.stop_gradient(keras.ops.subtract(hard, soft)))
            return soft

        if self.use_temperature:
            temp = self._resolve_temperature()
            logits = keras.ops.divide(weights, temp)
        else:
            logits = weights
        return keras.ops.softmax(logits, axis=-1)

    def _maybe_add_entropy_loss(self, probs: keras.KerasTensor) -> None:
        """
        Add a loss that pushes the selection toward one gate.

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
            self.add_loss(keras.ops.multiply(self.entropy_coefficient, ent))

    def to_symbolic(self, top_k: int = 1, deterministic: bool = True) -> str:
        """
        Report the gates the layer currently favours.

        In ``per_channel`` mode the probabilities are averaged over the
        channels first, so the answer is one ranking for the whole layer.
        Read ``operation_weights`` directly if you need per-channel detail.

        :param top_k: How many gates to report, highest probability first.
        :type top_k: int
        :param deterministic: Skip the Gumbel noise so repeated calls agree.
            Pass False only if you want a sample instead.
        :type deterministic: bool
        :return: A string like ``"xor(0.812), and(0.101)"``.
        :rtype: str
        :raises RuntimeError: If the layer has not been built.
        """
        if self.operation_weights is None:
            raise RuntimeError("Layer has not been built yet.")
        probs_arr = keras.ops.convert_to_numpy(
            self._operation_probs(deterministic=deterministic)
        )
        if self.selection_mode == "per_channel":
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
        Run every selected gate and return their weighted sum.

        **How the two selection modes combine the gates:**

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
                 output, shaped like x1

            selection_mode picks the column.

        :param inputs: One tensor, or a list of one or two tensors of the
            same shape.
        :type inputs: Union[keras.KerasTensor, List[keras.KerasTensor]]
        :param training: Keras training flag. Only the Gumbel path reads
            it; the gates themselves behave the same either way.
        :type training: Optional[bool]
        :return: The combined gate output, shaped like the first input.
        :rtype: keras.KerasTensor
        :raises ValueError: If more than two tensors are given, or one
            tensor is given for a binary gate without
            ``allow_unary_degenerate``.
        """
        # Input parsing. Three cases: two tensors, one tensor inside a list,
        # and a bare tensor.
        unary_input = False
        if isinstance(inputs, list):
            if len(inputs) == 2:
                x1, x2 = inputs
            elif len(inputs) == 1:
                x1 = inputs[0]
                x2 = inputs[0]
                unary_input = True
            else:
                raise ValueError(f"Expected 1 or 2 inputs, got {len(inputs)}")
        else:
            x1 = inputs
            x2 = inputs
            unary_input = True

        # DECISION plan_2026-05-13_a2b0f17b/D-001 — one tensor plus a binary
        # gate raises. Do not silently rebind x2 = x1 instead: that makes
        # XOR(p,p) always 0 and AND(p,p) just p, with no error anywhere.
        # allow_unary_degenerate=True opts back into the rebinding.
        # Owning plan dir gone; this comment is the record.
        if (
            unary_input
            and not self.allow_unary_degenerate
            and any(op in self.BINARY_OPS for op in self.operation_types)
        ):
            raise ValueError(
                "LearnableLogicOperator received a single tensor input but "
                f"operation_types contains binary ops {sorted(set(self.operation_types) & self.BINARY_OPS)}. "
                "Pass two tensors as a list `[x1, x2]`, or set "
                "allow_unary_degenerate=True to opt into legacy x2=x1 "
                "rebinding (mathematically incorrect for binary ops)."
            )

        # Map into [0, 1]. Skipped when the caller already provides values
        # in that range, which is the case for stacked logic layers.
        if self.apply_sigmoid:
            x1 = keras.ops.sigmoid(x1)
            x2 = keras.ops.sigmoid(x2)
        elif self.force_clip_when_no_sigmoid:
            # Clip when the layer above can produce unbounded outputs, for
            # example an arithmetic expert feeding a logic expert.
            x1 = keras.ops.clip(x1, 0.0, 1.0)
            x2 = keras.ops.clip(x2, 0.0, 1.0)

        # One probability per gate, plus the optional entropy penalty.
        operation_probs = self._operation_probs(training=training)
        self._maybe_add_entropy_loss(operation_probs)

        # Run every selected gate.
        operations = []
        for op_type in self.operation_types:
            if op_type == 'and':
                result = self._soft_logic_and(x1, x2)
            elif op_type == 'or':
                result = self._soft_logic_or(x1, x2)
            elif op_type == 'xor':
                result = self._soft_logic_xor(x1, x2)
            elif op_type == 'not':
                result = self._soft_logic_not(x1)
            elif op_type == 'nand':
                result = self._soft_logic_nand(x1, x2)
            elif op_type == 'nor':
                result = self._soft_logic_nor(x1, x2)
            elif op_type == 'lukasiewicz_and':
                result = self._luk_and(x1, x2)
            elif op_type == 'lukasiewicz_or':
                result = self._luk_or(x1, x2)
            elif op_type == 'godel_and':
                result = self._godel_and(x1, x2)
            elif op_type == 'godel_or':
                result = self._godel_or(x1, x2)
            elif op_type == 'implies':
                result = self._implies(x1, x2)
            elif op_type == 'lukasiewicz_implies':
                result = self._lukasiewicz_implies(x1, x2)
            elif op_type == 'reichenbach_implies':
                result = self._reichenbach_implies(x1, x2)
            elif op_type == 'goguen_implies':
                result = self._goguen_implies(x1, x2)
            elif op_type == 'hamacher_and':
                result = self._hamacher_and(x1, x2)
            elif op_type == 'hamacher_or':
                result = self._hamacher_or(x1, x2)
            elif op_type == 'yager_and':
                result = self._yager_and(x1, x2)
            elif op_type == 'yager_or':
                result = self._yager_or(x1, x2)
            else:
                logger.warning(f"Unknown operation type: {op_type}, using identity")
                result = x1
            operations.append(result)

        # Weighted combination, stacked and summed in one shot.
        if self.selection_mode == "per_channel":
            # Shape after this: (..., C, N)
            stacked = keras.ops.stack(operations, axis=-1)
            rank = len(stacked.shape)
            probs_bshape = (1,) * (rank - 2) + (self._channels, self.num_operations)
            weights = keras.ops.reshape(operation_probs, probs_bshape)
            output = keras.ops.sum(
                keras.ops.multiply(weights, stacked),
                axis=-1
            )
        else:
            stacked = keras.ops.stack(operations, axis=0)
            weight_shape = (self.num_operations,) + (1,) * (len(stacked.shape) - 1)
            weights = keras.ops.reshape(operation_probs, weight_shape)
            output = keras.ops.sum(keras.ops.multiply(weights, stacked), axis=0)
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

        The two initializers are serialized, so a round trip restores the
        same objects. Nothing created in ``build`` appears here.

        :return: The layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "operation_types": self.operation_types,
            "use_temperature": self.use_temperature,
            "temperature_init": self.temperature_init,
            "operation_initializer": keras.initializers.serialize(self.operation_initializer),
            "temperature_initializer": keras.initializers.serialize(self.temperature_initializer),
            "apply_sigmoid": self.apply_sigmoid,
            "force_clip_when_no_sigmoid": self.force_clip_when_no_sigmoid,
            "softplus_temperature": self.softplus_temperature,
            "gumbel_softmax": self.gumbel_softmax,
            "gumbel_hard": self.gumbel_hard,
            "entropy_coefficient": self.entropy_coefficient,
            "allow_unary_degenerate": self.allow_unary_degenerate,
            "selection_mode": self.selection_mode,
            "yager_p": self.yager_p,
        })
        return config

# ---------------------------------------------------------------------
