"""
Shape-preserving layers that run a mix of logic and arithmetic experts.

Two layers live here. ``CircuitDepthLayer`` is one stage: it runs several
``LearnableLogicOperator`` and ``LearnableArithmeticOperator`` children on
the same input and returns their weighted sum, so the layer learns which
operators the task wants. ``LearnableNeuralCircuit`` stacks
``circuit_depth`` of those stages with an optional LayerNorm after each.

Both keep the input shape. Give them any tensor of rank 2 or more and you
get the same shape back, so they drop into the middle of a network where
you would otherwise put a residual block.

Two routing modes:

    ``circuit_routing='output_only'`` is the default. Every expert reads
    the full input X and only the fusion is gated:

        Y = sum_i(beta_i * f_i(X)) [+ X]

    where beta is a softmax over the combination weights.

    ``circuit_routing='classic'`` also scales each expert's input by its
    own routing weight:

        Y = sum_i(beta_i * f_i(alpha_i * X)) [+ X]

    where alpha is a softmax over the routing weights. Each expert then
    sees X / N on average, so the signal shrinks as you add experts. Keep
    this mode only to load models trained before the default changed.

Extras, all off by default:

    - ``gate_entropy_coefficient`` > 0 adds an auxiliary loss that pushes
      expert use toward uniform. The old name ``load_balance_coefficient``
      still works and raises a DeprecationWarning.
    - ``diversity_coefficient`` > 0 adds a cosine-similarity penalty
      between experts, so they settle on different operators.
    - ``channel_mix='dense'`` appends a ``Dense(C)`` mixing layer after
      the fusion.
    - ``selection_mode='per_channel'`` learns one fusion weight vector per
      channel instead of one for the whole tensor.
    - ``use_layer_norm``, on the circuit only, adds a LayerNorm after
      every depth.

``apply_sigmoid_per_depth`` on ``LearnableNeuralCircuit`` says which
depths let their logic children sigmoid their input. ``'first_only'`` is
the default and only depth 0 does, because a sigmoid of a sigmoid keeps
squeezing the range and a 3-deep stack converges to a constant. ``'all'``
is the old behavior; ``'none'`` trusts the caller to stay in [0, 1].

Every weight clones the initializer it is given, so one ``Initializer``
INSTANCE passed to two parameters, or handed down by a parent to every
child, still leaves each weight with an independent draw. A seeded
instance keeps its seed and so keeps drawing the same values.
"""

import copy
import keras
import warnings
from typing import List, Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers.clone import clone_initializer
from .logic_operators import LearnableLogicOperator
from .arithmetic_operators import LearnableArithmeticOperator
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

def _resolve_gate_entropy_coefficient(
    gate_entropy_coefficient: Optional[float],
    load_balance_coefficient: Optional[float],
    cls_name: str,
) -> float:
    """
    Resolve the gate-entropy coefficient, honoring the deprecated alias.

    ``load_balance_coefficient`` is the old name for the same number. If
    only the old name is given and it is non-zero, this warns and uses it.
    If both are given, the new name wins. If neither is set, the result is
    0.0, which turns the loss off.

    :param gate_entropy_coefficient: The canonical argument, or ``None``.
    :type gate_entropy_coefficient: Optional[float]
    :param load_balance_coefficient: The deprecated alias, or ``None``.
    :type load_balance_coefficient: Optional[float]
    :param cls_name: Class name to name in the warning message.
    :type cls_name: str
    :return: The resolved coefficient, 0.0 when neither name is set.
    :rtype: float
    """
    if (
        load_balance_coefficient is not None
        and load_balance_coefficient != 0.0
        and gate_entropy_coefficient is None
    ):
        warnings.warn(
            f"{cls_name}: 'load_balance_coefficient' is deprecated; rename "
            f"to 'gate_entropy_coefficient' (plan_2026-05-13_3a2f1d23 H6). "
            f"The old name continues to work for now.",
            DeprecationWarning,
            stacklevel=3,
        )
        return float(load_balance_coefficient)
    if gate_entropy_coefficient is not None:
        return float(gate_entropy_coefficient)
    return float(load_balance_coefficient or 0.0)


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.logic.neural_circuit")
class CircuitDepthLayer(keras.layers.Layer):
    """
    One circuit stage: parallel logic and arithmetic experts, fused.

    The layer builds ``num_logic_ops`` ``LearnableLogicOperator`` children
    and ``num_arithmetic_ops`` ``LearnableArithmeticOperator`` children in
    ``__init__``. Every child runs on the input, and the output is their
    weighted sum with weights ``beta = softmax(combination_weights)``. The
    weights are trained, so the stage learns which experts matter.

    The output has the same shape as the input, so this drops in wherever
    a residual block would go. With ``use_residual=True``, the default,
    the input is added back at the end.

    Every child -- the operators, and the ``Dense`` channel mixer built in
    ``build()`` -- receives this layer's own ``dtype_policy``. Pinning a
    stage with ``dtype='float32'`` inside a ``mixed_float16`` model
    therefore runs its experts in float32 too.

    **Architecture Overview:**

    .. code-block:: text

        X = inputs [B, ..., C]
          │
          ├────────────────────────────────────────────┐
          ▼                                            │
        ┌────────────────────────────────────────────┐ │
        │ beta = softmax(combination_weights, -1)    │ │
        │ the gate-entropy and diversity aux losses  │ │
        │ are added here, when their coefs are > 0   │ │
        └─────────────────────┬──────────────────────┘ │
                              ▼                        │
        ┌────────────────────────────────────────────┐ │
        │ circuit_routing                            │ │
        │   'output_only'  every expert reads X      │ │
        │   'classic'      expert i reads            │ │
        │                  alpha_i * X, alpha =      │ │
        │                  softmax(routing_weights)  │ │
        └─────────────────────┬──────────────────────┘ │
                              ▼                        │
        ┌────────────────────────────────────────────┐ │
        │ N experts run in parallel on that input    │ │
        │ N = num_logic_ops + num_arithmetic_ops     │ │
        │ the logic children first, then arithmetic  │ │
        └─────────────────────┬──────────────────────┘ │
                              ▼                        │
        ┌────────────────────────────────────────────┐ │
        │ stack the N outputs, weight them by beta   │ │
        │   'global'       one beta for all of C     │ │
        │   'per_channel'  one beta per channel      │ │
        └─────────────────────┬──────────────────────┘ │
                              ▼                        │
        ┌────────────────────────────────────────────┐ │
        │ Dense(C)   only when channel_mix ==        │ │
        │            'dense'                         │ │
        └─────────────────────┬──────────────────────┘ │
                              ▼                        │
                             add ◄─────────────────────┘
                              │   only when use_residual
                              ▼
                   Y = output [B, ..., C]

    :param num_logic_ops: How many logic experts to run. Must be > 0.
    :type num_logic_ops: int
    :param num_arithmetic_ops: How many arithmetic experts to run. Must be
        > 0.
    :type num_arithmetic_ops: int
    :param use_residual: Add the input back to the fused output.
    :type use_residual: bool
    :param logic_op_types: Gate keys passed to every logic child. ``None``
        leaves that child on its own default set.
    :type logic_op_types: Optional[List[str]]
    :param arithmetic_op_types: Operation keys passed to every arithmetic
        child. ``None`` leaves that child on its own default set.
    :type arithmetic_op_types: Optional[List[str]]
    :param routing_initializer: Initializer for the routing weights. Those
        weights are created either way, but only ``'classic'`` routing
        reads them.
    :type routing_initializer: Union[str, keras.initializers.Initializer]
    :param combination_initializer: Initializer for the combination
        weights. ``"zeros"`` starts every expert equally weighted.
    :type combination_initializer: Union[str, keras.initializers.Initializer]
    :param circuit_routing: ``'output_only'`` (default) or ``'classic'``.
        See the module docstring for the two formulas.
    :type circuit_routing: str
    :param apply_sigmoid: Passed to every logic child as its
        ``apply_sigmoid``. ``True`` matches the old behavior.
    :type apply_sigmoid: bool
    :param gate_entropy_coefficient: Weight of the auxiliary loss that
        pushes expert use toward uniform. 0 or ``None`` adds no loss.
    :type gate_entropy_coefficient: Optional[float]
    :param load_balance_coefficient: Deprecated name for
        ``gate_entropy_coefficient``. Passing it warns. If both are
        passed, the new name wins.
    :type load_balance_coefficient: Optional[float]
    :param channel_mix: ``'dense'`` appends a ``Dense(C)`` layer after the
        fusion. ``None`` keeps the stage pointwise.
    :type channel_mix: Optional[str]
    :param force_logic_input_clip: Passed to every logic child as its
        ``force_clip_when_no_sigmoid``. Use it when the input can leave
        [0, 1] and ``apply_sigmoid`` is False.
    :type force_logic_input_clip: bool
    :param selection_mode: ``"global"`` learns one beta for the tensor,
        ``"per_channel"`` one per channel. Passed on to the children too,
        and ``"per_channel"`` needs a known last axis at build time.
    :type selection_mode: str
    :param diversity_coefficient: Weight of the pairwise cosine-similarity
        penalty between experts. 0 adds no loss.
    :type diversity_coefficient: float
    :param inner_logic_kwargs: Extra arguments forwarded to every logic
        child. Keys this layer sets itself are dropped with a warning.
    :type inner_logic_kwargs: Optional[Dict[str, Any]]
    :param inner_arithmetic_kwargs: Extra arguments forwarded to every
        arithmetic child, filtered the same way.
    :type inner_arithmetic_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Passed to ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar num_logic_ops: Number of logic experts.
    :vartype num_logic_ops: int
    :ivar num_arithmetic_ops: Number of arithmetic experts.
    :vartype num_arithmetic_ops: int
    :ivar use_residual: Whether the input is added back.
    :vartype use_residual: bool
    :ivar logic_op_types: Gate keys given to the logic children.
    :vartype logic_op_types: Optional[List[str]]
    :ivar arithmetic_op_types: Operation keys given to the arithmetic
        children.
    :vartype arithmetic_op_types: Optional[List[str]]
    :ivar routing_initializer: The resolved routing-weight initializer.
    :vartype routing_initializer: keras.initializers.Initializer
    :ivar combination_initializer: The resolved combination-weight
        initializer.
    :vartype combination_initializer: keras.initializers.Initializer
    :ivar circuit_routing: ``'output_only'`` or ``'classic'``.
    :vartype circuit_routing: str
    :ivar apply_sigmoid: What the logic children were given.
    :vartype apply_sigmoid: bool
    :ivar gate_entropy_coefficient: The resolved coefficient, as a float.
    :vartype gate_entropy_coefficient: float
    :ivar load_balance_coefficient: Read-only alias holding the same
        value, for callers still on the old name. Nothing in this class
        reads it: ``_maybe_load_balance_loss`` reads the canonical
        ``gate_entropy_coefficient``.
    :vartype load_balance_coefficient: float
    :ivar channel_mix: ``'dense'`` or ``None``.
    :vartype channel_mix: Optional[str]
    :ivar force_logic_input_clip: What the logic children were given.
    :vartype force_logic_input_clip: bool
    :ivar selection_mode: ``"global"`` or ``"per_channel"``.
    :vartype selection_mode: str
    :ivar diversity_coefficient: The stored penalty weight, as a float.
    :vartype diversity_coefficient: float
    :ivar inner_logic_kwargs: The filtered extra logic arguments.
    :vartype inner_logic_kwargs: Dict[str, Any]
    :ivar inner_arithmetic_kwargs: The filtered extra arithmetic
        arguments.
    :vartype inner_arithmetic_kwargs: Dict[str, Any]
    :ivar logic_operators: The logic children, built in ``__init__``.
    :vartype logic_operators: List[LearnableLogicOperator]
    :ivar arithmetic_operators: The arithmetic children, built in
        ``__init__``.
    :vartype arithmetic_operators: List[LearnableArithmeticOperator]
    :ivar routing_weights: Shape ``(N,)``. ``None`` until ``build``.
    :vartype routing_weights: Optional[keras.Variable]
    :ivar combination_weights: Shape ``(N,)`` in global mode and
        ``(C, N)`` per channel. ``None`` until ``build``.
    :vartype combination_weights: Optional[keras.Variable]

    :raises ValueError: From the constructor if ``diversity_coefficient``
        is negative, ``selection_mode`` or ``circuit_routing`` or
        ``channel_mix`` is not one of its allowed values, either operator
        count is not positive, or the resolved gate-entropy coefficient is
        negative.
    :raises ValueError: From ``build`` if the input rank is below 2, or
        ``selection_mode="per_channel"`` gets an unknown last axis.
    :raises ValueError: From ``call`` if the input rank is below 2, or the
        last axis differs from the one ``build`` sized channel-dependent
        state for.
    :raises RuntimeError: From ``to_symbolic`` before the layer is built.

    Input shape:
        A tensor of rank 2 or more, ``(batch, ..., channels)``. In
        ``per_channel`` mode the last axis must be known.

    Output shape:
        The same shape as the input.

    Example:
        .. code-block:: python

            import keras
            from dl_techniques.layers.logic import CircuitDepthLayer

            x = keras.random.normal((4, 16))

            layer = CircuitDepthLayer(
                num_logic_ops=2, num_arithmetic_ops=2
            )
            y = layer(x)
            y.shape  # (4, 16)

            # A wider stage that also mixes channels and reports
            # which experts it settled on.
            layer2 = CircuitDepthLayer(
                num_logic_ops=3,
                num_arithmetic_ops=1,
                channel_mix='dense',
                diversity_coefficient=0.01,
            )
            layer2(x)
            print(layer2.to_symbolic(top_k=2))
    """

    def __init__(
            self,
            num_logic_ops: int = 2,
            num_arithmetic_ops: int = 2,
            use_residual: bool = True,
            logic_op_types: Optional[List[str]] = None,
            arithmetic_op_types: Optional[List[str]] = None,
            routing_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            combination_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            circuit_routing: str = "output_only",
            apply_sigmoid: bool = True,
            gate_entropy_coefficient: Optional[float] = None,
            load_balance_coefficient: Optional[float] = None,
            channel_mix: Optional[str] = None,
            force_logic_input_clip: bool = False,
            selection_mode: str = "global",
            diversity_coefficient: float = 0.0,
            inner_logic_kwargs: Optional[Dict[str, Any]] = None,
            inner_arithmetic_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """
        Validate the arguments and build the expert children.

        The children are constructed here, not in :meth:`build`, so that a
        caller can inspect ``logic_operators`` and ``arithmetic_operators``
        before the layer ever runs. This layer's own weights need the
        channel count in ``per_channel`` mode and are created in
        :meth:`build`. The class docstring documents every parameter.
        """
        super().__init__(**kwargs)

        # Accept the canonical name and the deprecated alias.
        resolved_coef = _resolve_gate_entropy_coefficient(
            gate_entropy_coefficient,
            load_balance_coefficient,
            self.__class__.__name__,
        )

        if diversity_coefficient < 0:
            raise ValueError("diversity_coefficient must be non-negative.")

        if selection_mode not in ("global", "per_channel"):
            raise ValueError(
                f"selection_mode must be 'global' or 'per_channel', got "
                f"{selection_mode!r}."
            )

        # Validate parameters
        if num_logic_ops <= 0:
            raise ValueError("num_logic_ops must be positive.")
        if num_arithmetic_ops <= 0:
            raise ValueError("num_arithmetic_ops must be positive.")
        if circuit_routing not in ("output_only", "classic"):
            raise ValueError(
                f"circuit_routing must be 'output_only' or 'classic', got "
                f"{circuit_routing!r}."
            )
        if resolved_coef < 0:
            raise ValueError("gate_entropy_coefficient must be non-negative.")
        if channel_mix not in (None, "dense"):
            raise ValueError(
                f"channel_mix must be None or 'dense', got {channel_mix!r}."
            )

        # Store ALL configuration parameters
        self.num_logic_ops = num_logic_ops
        self.num_arithmetic_ops = num_arithmetic_ops
        self.use_residual = use_residual
        self.logic_op_types = logic_op_types
        self.arithmetic_op_types = arithmetic_op_types
        self.routing_initializer = keras.initializers.get(routing_initializer)
        self.combination_initializer = keras.initializers.get(combination_initializer)
        self.circuit_routing = circuit_routing
        self.apply_sigmoid = apply_sigmoid
        # gate_entropy_coefficient is the canonical name and the only
        # one any code path reads. load_balance_coefficient mirrors it
        # for callers on the old name.
        self.gate_entropy_coefficient = resolved_coef
        self.load_balance_coefficient = resolved_coef
        self.channel_mix = channel_mix
        # Passed to the logic children as force_clip_when_no_sigmoid.
        self.force_logic_input_clip = force_logic_input_clip
        # selection_mode goes to the children and also shapes this layer's
        # own combination weights.
        self.selection_mode = selection_mode
        # Set in build() for per_channel mode; stays None in global mode.
        self._channels = None
        # The last axis build() sized channel-dependent state for --
        # per_channel combination weights, or the channel-mix Dense.
        # None when nothing built here depends on it.
        self._build_last_dim: Optional[int] = None
        self.diversity_coefficient = float(diversity_coefficient)
        # Deep, so an Initializer instance in here is this layer's own copy
        # and a caller mutating the dict afterwards cannot reach the children.
        self.inner_logic_kwargs = copy.deepcopy(inner_logic_kwargs) if inner_logic_kwargs else {}
        self.inner_arithmetic_kwargs = copy.deepcopy(inner_arithmetic_kwargs) if inner_arithmetic_kwargs else {}

        # DECISION plan_2026-05-13_a2b0f17b/D-002 — the children are built
        # here in __init__, and the logic ones get
        # allow_unary_degenerate=True because this layer hands each child a
        # single tensor. Do not move the construction into build().
        # Owning plan dir gone; this comment is the record.

        # DECISION plan_2026-05-13_e33114da/D-006 — keys this layer sets
        # itself are dropped from inner_*_kwargs, with a warning. Do not
        # forward them: the child would get two values for one argument.
        # Owning plan dir gone; this comment is the record.
        logic_owned = {
            "operation_types", "apply_sigmoid", "allow_unary_degenerate",
            "force_clip_when_no_sigmoid", "selection_mode", "name",
        }
        arith_owned = {
            "operation_types", "selection_mode", "name",
        }
        logic_extra = {k: v for k, v in self.inner_logic_kwargs.items() if k not in logic_owned}
        arith_extra = {k: v for k, v in self.inner_arithmetic_kwargs.items() if k not in arith_owned}
        collided_logic = set(self.inner_logic_kwargs) & logic_owned
        collided_arith = set(self.inner_arithmetic_kwargs) & arith_owned
        if collided_logic:
            warnings.warn(
                f"CircuitDepthLayer: inner_logic_kwargs keys {sorted(collided_logic)} "
                f"are wrapper-controlled and will be ignored.",
                UserWarning,
                stacklevel=3,
            )
        if collided_arith:
            warnings.warn(
                f"CircuitDepthLayer: inner_arithmetic_kwargs keys {sorted(collided_arith)} "
                f"are wrapper-controlled and will be ignored.",
                UserWarning,
                stacklevel=3,
            )

        # DECISION plan-2026-08-29T112804-aff039c4/D-007 -- pass
        # dtype=self.dtype_policy, never dtype=self.dtype, to every child
        # constructed here. Under mixed_float16 self.dtype reads
        # 'float32' (the VARIABLE dtype), so the 41-file house spelling
        # would silently build pure-float32 children. See D-007.
        self.logic_operators = [
            LearnableLogicOperator(
                operation_types=self.logic_op_types,
                apply_sigmoid=self.apply_sigmoid,
                allow_unary_degenerate=True,
                force_clip_when_no_sigmoid=self.force_logic_input_clip,
                selection_mode=self.selection_mode,
                name=f"logic_op_{i}",
                dtype=self.dtype_policy,
                **logic_extra,
            )
            for i in range(self.num_logic_ops)
        ]
        self.arithmetic_operators = [
            LearnableArithmeticOperator(
                operation_types=self.arithmetic_op_types,
                selection_mode=self.selection_mode,
                name=f"arithmetic_op_{i}",
                dtype=self.dtype_policy,
                **arith_extra,
            )
            for i in range(self.num_arithmetic_ops)
        ]

        # The channel-mix sublayer needs the channel count, so it waits
        # for build().
        self._channel_mix_layer: Optional[keras.layers.Dense] = None

        # Weights, created in build().
        self.routing_weights = None
        self.combination_weights = None

        logger.debug(
            f"CircuitDepthLayer: routing={circuit_routing}, "
            f"{num_logic_ops}+{num_arithmetic_ops} ops, residual={use_residual}, "
            f"gate_entropy={resolved_coef}, channel_mix={channel_mix}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create this layer's weights and build every child.

        The children are built here as well. Keras 3 requires the parent to
        create all child state during its own build, or a saved model
        reloads with fresh child weights.

        :param input_shape: Shape of the input tensor, rank 2 or more.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the rank is below 2, or
            ``selection_mode="per_channel"`` gets an unknown last axis.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"CircuitDepthLayer expects rank >= 2 input, "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        total_operators = self.num_logic_ops + self.num_arithmetic_ops

        # per_channel shapes the combination weights (channels, N). The
        # routing weights stay 1-D: 'classic' routing predates per-channel
        # and that API is not being widened.
        if self.selection_mode == "per_channel":
            if input_shape[-1] is None:
                raise ValueError(
                    "selection_mode='per_channel' requires a concrete "
                    f"last-axis dimension; got {input_shape}."
                )
            self._channels = int(input_shape[-1])
            self._build_last_dim = self._channels
            combination_shape = (self._channels, total_operators)
        else:
            combination_shape = (total_operators,)

        # The routing weights are created in both modes so a checkpoint
        # from either one loads into either one. Only 'classic' reads them.
        # DECISION plan-2026-08-29T112804-aff039c4/D-001 -- clone at the
        # add_weight site: one resolved Initializer INSTANCE redraws
        # identical values at every weight whose shape matches. A string
        # is safe; a seeded instance defeats the clone.
        # See decisions.md D-001.
        self.routing_weights = self.add_weight(
            name="routing_weights",
            shape=(total_operators,),
            initializer=clone_initializer(self.routing_initializer),
            trainable=True,
        )
        self.combination_weights = self.add_weight(
            name="combination_weights",
            shape=combination_shape,
            initializer=clone_initializer(self.combination_initializer),
            trainable=True,
        )

        for op in self.logic_operators:
            op.build(input_shape)
        for op in self.arithmetic_operators:
            op.build(input_shape)

        if self.channel_mix == "dense":
            channel_dim = int(input_shape[-1])
            self._build_last_dim = channel_dim
            self._channel_mix_layer = keras.layers.Dense(
                channel_dim,
                use_bias=True,
                name="channel_mix",
                dtype=self.dtype_policy,
            )
            self._channel_mix_layer.build(input_shape)

        super().build(input_shape)

    def _maybe_load_balance_loss(
        self, combination_probs: keras.KerasTensor
    ) -> None:
        """
        Add the Shazeer (2017) importance loss, if it is switched on.

        The value is ``coef * N * mean(sum_i(beta_i^2))``, where the sum
        runs over the expert axis and the mean over any leading axes. It
        is the squared L2 of the gate vector, not an entropy, despite the
        ``gate_entropy_coefficient`` name, which is the attribute this
        method reads. It is smallest when beta is uniform, so it pushes
        the layer to use every expert. Nothing is added when the
        coefficient is 0.

        # DECISION plan_2026-05-13_e33114da/D-005 — take the L2 per channel
        # first and average after. Do not average the probs across channels
        # before the L2: per-channel-peaky rows that average to uniform
        # then escape the penalty.
        # Owning plan dir gone; this comment is the record.

        :param combination_probs: Softmax gate probabilities, ``(N,)`` in
            global mode and ``(C, N)`` per channel.
        :type combination_probs: keras.KerasTensor
        :return: Nothing. The loss is registered with ``add_loss``.
        :rtype: None
        """
        if self.gate_entropy_coefficient <= 0:
            return
        n = float(self.num_logic_ops + self.num_arithmetic_ops)
        # axis=-1 is the expert axis in both shapes. For (N,) this is just
        # sum(beta^2); for (C, N) it is a per-channel L2, averaged below.
        per_row_l2 = keras.ops.sum(keras.ops.square(combination_probs), axis=-1)
        aux = keras.ops.mean(per_row_l2) if len(combination_probs.shape) > 1 else per_row_l2
        self.add_loss(self.gate_entropy_coefficient * n * aux)

    def _maybe_diversity_loss(self) -> None:
        """
        Add the expert-diversity loss, if it is switched on.

        The value is the mean pairwise cosine similarity between the
        experts' own operation-probability vectors. Penalizing it pushes
        the experts onto different operators instead of all picking the
        same one. Nothing is added when the coefficient is 0.

        Logic and arithmetic experts are scored separately, because their
        probability vectors live in op spaces of different size and cannot
        be compared. Each group is scored with one Gram matrix rather than
        a pair loop.

        :return: Nothing. The loss is registered with ``add_loss``.
        :rtype: None
        """
        if self.diversity_coefficient <= 0:
            return

        def _group_sim(ops_group: List[Any]) -> Tuple[Optional[keras.KerasTensor], int]:
            """
            Sum the pairwise cosine similarities within one group.

            :param ops_group: Experts sharing one op space.
            :type ops_group: List[Any]
            :return: The summed upper-triangle similarity and the number
                of pairs, or ``(None, 0)`` for fewer than 2 experts.
            :rtype: Tuple[Optional[keras.KerasTensor], int]
            """
            if len(ops_group) < 2:
                return None, 0
            vecs = []
            for op in ops_group:
                p = op._operation_probs(deterministic=True)
                if len(p.shape) > 1:
                    p = keras.ops.mean(p, axis=0)
                vecs.append(p)
            # stacked is (K experts, M operations).
            stacked = keras.ops.stack(vecs, axis=0)
            norms = keras.ops.add(
                keras.ops.norm(stacked, axis=-1, keepdims=True), 1e-12)
            stacked = keras.ops.divide(stacked, norms)
            gram = keras.ops.matmul(stacked, keras.ops.transpose(stacked))
            # The gram matrix is symmetric with a unit diagonal, so the
            # upper triangle is (sum(gram) - trace) / 2.
            diag = keras.ops.sum(keras.ops.multiply(gram, keras.ops.eye(len(ops_group), dtype=gram.dtype)))
            upper_sum = keras.ops.divide(keras.ops.subtract(keras.ops.sum(gram), diag), 2.0)
            k = len(ops_group)
            pair_count = k * (k - 1) // 2
            return upper_sum, pair_count

        logic_sum, logic_pairs = _group_sim(self.logic_operators)
        arith_sum, arith_pairs = _group_sim(self.arithmetic_operators)

        total_pairs = logic_pairs + arith_pairs
        if total_pairs == 0:
            return

        total_sim: Optional[keras.KerasTensor] = None
        if logic_sum is not None:
            total_sim = logic_sum
        if arith_sum is not None:
            total_sim = arith_sum if total_sim is None else keras.ops.add(total_sim, arith_sum)

        mean_sim = keras.ops.divide(total_sim, float(total_pairs))
        self.add_loss(keras.ops.multiply(self.diversity_coefficient, mean_sim))

    def _assert_call_shape_contract(
            self,
            inputs: keras.KerasTensor
    ) -> None:
        """
        Re-assert in ``call`` the static shape contract ``build`` checked.

        A contract checked only in ``build`` is checked once, against
        whatever shape arrived first: the layer stays built and every
        later call is unchecked. ``InputSpec`` cannot close this, because
        ``assert_input_compatibility`` tests ``shape[axis] not in
        {value, None}`` and so accepts an unknown dimension. A dimension
        that is ``None`` here is genuinely unknown at trace time and is
        skipped rather than guessed.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :raises ValueError: If the rank is below 2, or the last axis
            differs from the one ``build`` sized channel-dependent state
            for.
        """
        shape = tuple(inputs.shape)
        if len(shape) < 2:
            raise ValueError(
                f"{type(self).__name__} expects rank >= 2 input, "
                f"got shape with {len(shape)} dimensions: {shape}"
            )
        if self._build_last_dim is None:
            return
        if shape[-1] is not None and int(shape[-1]) != self._build_last_dim:
            raise ValueError(
                f"{type(self).__name__} was built for a last axis of "
                f"{self._build_last_dim} and cannot run on shape "
                f"{shape}. Build a separate layer per input width."
            )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run every expert on the input and return their weighted sum.

        **How the two selection modes fuse the N expert outputs:**

        .. code-block:: text

            'global'                'per_channel'
            beta (N,)               beta (C, N)
            stack on axis 0         stack on axis -1
             -> (N, *x.shape)        -> (*x.shape, N)
            reshape beta to         reshape beta to
             (N, 1, ..., 1)          (1, ..., 1, C, N)
            sum on axis 0           sum on axis -1
                    │                        │
                    └───────────┬────────────┘
                                ▼
                     fused, shaped like inputs

            selection_mode picks the column. The channel
            mix and the residual come after the join.

        :param inputs: A tensor of rank 2 or more.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every expert
            and to the channel-mix ``Dense``. ``Dense`` ignores it; it is
            forwarded anyway, because Keras 3 propagates ``training``
            through one mutable call-context slot that a sibling layer
            can leave set.
        :type training: Optional[bool]
        :return: The fused output, shaped like ``inputs``.
        :rtype: keras.KerasTensor
        """
        # The static shape contract build() checked, re-checked here,
        # before any loss is registered.
        self._assert_call_shape_contract(inputs)

        combination_probs = keras.ops.softmax(self.combination_weights, axis=-1)
        # Both aux losses are no-ops when their coefficients are 0.
        self._maybe_load_balance_loss(combination_probs)
        self._maybe_diversity_loss()

        all_outputs: List[keras.KerasTensor] = []

        if self.circuit_routing == "classic":
            # Legacy path: each expert's input is scaled by its own
            # softmax(routing_weights) entry before it runs.
            routing_probs = keras.ops.softmax(self.routing_weights)

            for i, logic_op in enumerate(self.logic_operators):
                weight = routing_probs[i]
                weighted_input = keras.ops.multiply(inputs, weight)
                all_outputs.append(logic_op(weighted_input, training=training))
            for j, arithmetic_op in enumerate(self.arithmetic_operators):
                weight = routing_probs[self.num_logic_ops + j]
                weighted_input = keras.ops.multiply(inputs, weight)
                all_outputs.append(arithmetic_op(weighted_input, training=training))
        else:
            # output_only: every expert reads the input unchanged, and only
            # the fusion below is gated.
            for logic_op in self.logic_operators:
                all_outputs.append(logic_op(inputs, training=training))
            for arithmetic_op in self.arithmetic_operators:
                all_outputs.append(arithmetic_op(inputs, training=training))

        # Weighted fusion, done as one stack-and-sum rather than a loop.
        n = self.num_logic_ops + self.num_arithmetic_ops
        if self.selection_mode == "per_channel":
            # stacked is (..., C, N).
            stacked = keras.ops.stack(all_outputs, axis=-1)
            rank = len(stacked.shape)
            weight_shape = (1,) * (rank - 2) + (self._channels, n)
            weights = keras.ops.reshape(combination_probs, weight_shape)
            combined_output = keras.ops.sum(keras.ops.multiply(weights, stacked), axis=-1)
        else:
            stacked = keras.ops.stack(all_outputs, axis=0)
            weight_shape = (n,) + (1,) * (len(stacked.shape) - 1)
            weights = keras.ops.reshape(combination_probs, weight_shape)
            combined_output = keras.ops.sum(keras.ops.multiply(weights, stacked), axis=0)

        if self._channel_mix_layer is not None:
            combined_output = self._channel_mix_layer(
                combined_output, training=training
            )

        if self.use_residual:
            combined_output = keras.ops.add(combined_output, inputs)

        return combined_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def to_symbolic(self, top_k: int = 1) -> str:
        """
        Report which operator each expert settled on.

        One line per expert naming its ``top_k`` operators with their
        probabilities, then one line ranking the experts by their
        combination weight. In ``per_channel`` mode the weights are
        averaged over channels first. This works on a stage used on its
        own, outside a ``LearnableNeuralCircuit``.

        :param top_k: How many entries to keep on each line.
        :type top_k: int
        :return: A multi-line summary.
        :rtype: str
        :raises RuntimeError: If the layer is not built yet.
        """
        if not self.built:
            raise RuntimeError(
                "CircuitDepthLayer.to_symbolic() requires the layer to be "
                "built. Call the layer on a sample input first."
            )
        lines: List[str] = []
        for i, op in enumerate(self.logic_operators):
            lines.append(f"logic_op_{i}: {op.to_symbolic(top_k=top_k)}")
        for j, op in enumerate(self.arithmetic_operators):
            lines.append(f"arithmetic_op_{j}: {op.to_symbolic(top_k=top_k)}")
        cw = keras.ops.convert_to_numpy(keras.ops.softmax(self.combination_weights, axis=-1))
        if cw.ndim > 1:
            cw = cw.mean(axis=0)
        names = (
            [f"logic_op_{i}" for i in range(self.num_logic_ops)]
            + [f"arithmetic_op_{j}" for j in range(self.num_arithmetic_ops)]
        )
        ranked = sorted(zip(names, cw.tolist()), key=lambda kv: -kv[1])[:top_k]
        lines.append("combination: " + ", ".join(f"{n}({p:.3f})" for n, p in ranked))
        return "\n".join(lines)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this layer.

        Every constructor argument appears as a key, including the
        deprecated ``load_balance_coefficient``. That key is always
        emitted as ``None``, never as the resolved value: the coefficient
        travels in the canonical ``gate_entropy_coefficient`` key, and
        ``None`` is exactly what the alias parameter contributed once
        ``__init__`` resolved the two. Emitting the value instead would
        make every load hand ``__init__`` the deprecated name, and a
        round trip would then either warn on every load or silently
        depend on which of the two names wins. A config written by an
        older version carrying a non-``None``
        ``load_balance_coefficient`` still loads, because ``__init__``
        accepts both names.

        :return: A serializable config dict.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_logic_ops": self.num_logic_ops,
            "num_arithmetic_ops": self.num_arithmetic_ops,
            "use_residual": self.use_residual,
            "logic_op_types": self.logic_op_types,
            "arithmetic_op_types": self.arithmetic_op_types,
            "routing_initializer": keras.initializers.serialize(self.routing_initializer),
            "combination_initializer": keras.initializers.serialize(self.combination_initializer),
            "circuit_routing": self.circuit_routing,
            "apply_sigmoid": self.apply_sigmoid,
            "gate_entropy_coefficient": self.gate_entropy_coefficient,
            # DECISION plan-2026-08-29T112804-aff039c4/D-003 -- the
            # deprecated alias key is always None. Do not emit the
            # resolved value: every load would then hand __init__ the
            # deprecated name. See decisions.md D-003.
            "load_balance_coefficient": None,
            "channel_mix": self.channel_mix,
            "force_logic_input_clip": self.force_logic_input_clip,
            "selection_mode": self.selection_mode,
            "diversity_coefficient": self.diversity_coefficient,
            "inner_logic_kwargs": self.inner_logic_kwargs or None,
            "inner_arithmetic_kwargs": self.inner_arithmetic_kwargs or None,
        })
        return config


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.logic.neural_circuit")
class LearnableNeuralCircuit(keras.layers.Layer):
    """
    A stack of ``circuit_depth`` ``CircuitDepthLayer`` stages.

    Each depth is its own ``CircuitDepthLayer`` with its own experts and
    its own weights, and the output of one is the input of the next. With
    ``use_layer_norm=True`` a ``LayerNormalization`` follows every depth.
    Every stage and every norm receives this circuit's own
    ``dtype_policy``, so a ``dtype=`` override reaches the whole tree.
    The shape never changes, so the whole stack drops in wherever a single
    stage would.

    Two settings vary by depth. The first is whether the logic children
    sigmoid their input, which ``apply_sigmoid_per_depth`` controls. Its
    default ``'first_only'`` sigmoids depth 0 and no other, because
    stacking sigmoids narrows the range until the stack outputs a
    constant. The second follows from it: when that default is set on a
    stack of 2 or more that also has arithmetic experts or a residual,
    ``force_logic_input_clip`` is switched on for depths 1 and up, and the
    constructor logs a warning saying so. Everything else is identical
    across depths.

    **Architecture Overview:**

    .. code-block:: text

        X = inputs [B, ..., C]
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │ depth 0  CircuitDepthLayer             │
        │   num_logic_ops_per_depth logic and    │
        │   num_arithmetic_ops_per_depth arith   │
        │   experts, fused; residual inside      │
        └───────────────────┬────────────────────┘
                            ▼
        ┌────────────────────────────────────────┐
        │ LayerNormalization                     │
        │   only when use_layer_norm             │
        └───────────────────┬────────────────────┘
                            ▼
              repeated circuit_depth times.
              Every depth is its own layer with
              its own weights; nothing is shared.
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ depth circuit_depth - 1                │
        │   CircuitDepthLayer [+ LayerNorm]      │
        └───────────────────┬────────────────────┘
                            ▼
             Y = output [B, ..., C]

        Two things differ between depths: apply_sigmoid,
        per _sigmoid_for_depth, and force_logic_input_clip,
        which is True on depths >= 1 of a risky stack.

    :param circuit_depth: How many stages to stack. Must be > 0.
    :type circuit_depth: int
    :param num_logic_ops_per_depth: Logic experts inside each stage. Must
        be > 0.
    :type num_logic_ops_per_depth: int
    :param num_arithmetic_ops_per_depth: Arithmetic experts inside each
        stage. Must be > 0.
    :type num_arithmetic_ops_per_depth: int
    :param use_residual: Passed to every stage. Each stage adds its own
        input back; there is no residual across the whole stack.
    :type use_residual: bool
    :param use_layer_norm: Insert a ``LayerNormalization`` after every
        depth, including the last.
    :type use_layer_norm: bool
    :param logic_op_types: Gate keys passed down to every logic child.
    :type logic_op_types: Optional[List[str]]
    :param arithmetic_op_types: Operation keys passed down to every
        arithmetic child.
    :type arithmetic_op_types: Optional[List[str]]
    :param routing_initializer: Initializer for each stage's routing
        weights. Only ``'classic'`` routing reads them.
    :type routing_initializer: Union[str, keras.initializers.Initializer]
    :param combination_initializer: Initializer for each stage's
        combination weights.
    :type combination_initializer: Union[str, keras.initializers.Initializer]
    :param circuit_routing: ``'output_only'`` (default) or ``'classic'``,
        passed to every stage.
    :type circuit_routing: str
    :param apply_sigmoid_per_depth: ``'first_only'`` (default), ``'all'``
        or ``'none'``. See :meth:`_sigmoid_for_depth`.
    :type apply_sigmoid_per_depth: str
    :param gate_entropy_coefficient: Weight of each stage's auxiliary
        uniform-use loss. 0 or ``None`` adds no loss.
    :type gate_entropy_coefficient: Optional[float]
    :param load_balance_coefficient: Deprecated name for
        ``gate_entropy_coefficient``. Passing it warns.
    :type load_balance_coefficient: Optional[float]
    :param channel_mix: ``'dense'`` gives every stage a ``Dense(C)`` after
        its fusion. ``None`` keeps the stack pointwise.
    :type channel_mix: Optional[str]
    :param selection_mode: ``"global"`` or ``"per_channel"``, passed to
        every stage.
    :type selection_mode: str
    :param diversity_coefficient: Weight of each stage's expert-diversity
        penalty. 0 adds no loss.
    :type diversity_coefficient: float
    :param inner_logic_kwargs: Extra arguments forwarded down to every
        logic child. Keys the stage owns are dropped with a warning.
    :type inner_logic_kwargs: Optional[Dict[str, Any]]
    :param inner_arithmetic_kwargs: Extra arguments forwarded down to
        every arithmetic child, filtered the same way.
    :type inner_arithmetic_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Passed to ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar circuit_depth: Number of stages.
    :vartype circuit_depth: int
    :ivar num_logic_ops_per_depth: Logic experts per stage.
    :vartype num_logic_ops_per_depth: int
    :ivar num_arithmetic_ops_per_depth: Arithmetic experts per stage.
    :vartype num_arithmetic_ops_per_depth: int
    :ivar use_residual: What the stages were given.
    :vartype use_residual: bool
    :ivar use_layer_norm: Whether ``layer_norms`` is populated.
    :vartype use_layer_norm: bool
    :ivar logic_op_types: Gate keys given to the logic children.
    :vartype logic_op_types: Optional[List[str]]
    :ivar arithmetic_op_types: Operation keys given to the arithmetic
        children.
    :vartype arithmetic_op_types: Optional[List[str]]
    :ivar routing_initializer: The resolved routing-weight initializer.
    :vartype routing_initializer: keras.initializers.Initializer
    :ivar combination_initializer: The resolved combination-weight
        initializer.
    :vartype combination_initializer: keras.initializers.Initializer
    :ivar circuit_routing: ``'output_only'`` or ``'classic'``.
    :vartype circuit_routing: str
    :ivar apply_sigmoid_per_depth: The stored sigmoid rule.
    :vartype apply_sigmoid_per_depth: str
    :ivar gate_entropy_coefficient: The resolved coefficient, as a float.
    :vartype gate_entropy_coefficient: float
    :ivar load_balance_coefficient: Alias holding the same value, kept
        for callers still on the old name.
    :vartype load_balance_coefficient: float
    :ivar channel_mix: ``'dense'`` or ``None``.
    :vartype channel_mix: Optional[str]
    :ivar selection_mode: ``"global"`` or ``"per_channel"``.
    :vartype selection_mode: str
    :ivar diversity_coefficient: The stored penalty weight, as a float.
    :vartype diversity_coefficient: float
    :ivar inner_logic_kwargs: The extra logic arguments, as a dict.
    :vartype inner_logic_kwargs: Dict[str, Any]
    :ivar inner_arithmetic_kwargs: The extra arithmetic arguments, as a
        dict.
    :vartype inner_arithmetic_kwargs: Dict[str, Any]
    :ivar circuit_layers: The stages, in order, built in ``__init__``.
    :vartype circuit_layers: List[CircuitDepthLayer]
    :ivar layer_norms: One norm per depth, or an empty list when
        ``use_layer_norm`` is False.
    :vartype layer_norms: List[keras.layers.LayerNormalization]

    :raises ValueError: From the constructor if ``selection_mode`` is not
        one of the two keys, ``circuit_depth`` or either per-depth expert
        count is not positive, ``apply_sigmoid_per_depth`` is not one of
        the three keys, or ``diversity_coefficient`` is negative.
    :raises ValueError: Also from the constructor, raised by the stages it
        builds there, if ``circuit_routing`` or ``channel_mix`` is not one
        of its allowed values or the resolved gate-entropy coefficient is
        negative. This class does not check those three itself.
    :raises ValueError: From ``build`` if the input rank is below 2.
    :raises ValueError: From ``call`` if the input rank is below 2, or the
        last axis differs from the one ``build`` sized channel-dependent
        stage state for.
    :raises RuntimeError: From ``to_symbolic`` before the layer is built.

    Input shape:
        A tensor of rank 2 or more, ``(batch, ..., channels)``. In
        ``per_channel`` mode the last axis must be known.

    Output shape:
        The same shape as the input.

    Example:
        .. code-block:: python

            import keras
            from dl_techniques.layers.logic import (
                LearnableNeuralCircuit,
            )

            x = keras.random.normal((4, 16))

            circuit = LearnableNeuralCircuit(circuit_depth=3)
            y = circuit(x)
            y.shape  # (4, 16)

            # Deeper, normalized between depths, and reporting the
            # operators each depth settled on.
            circuit2 = LearnableNeuralCircuit(
                circuit_depth=4,
                use_layer_norm=True,
                apply_sigmoid_per_depth='first_only',
            )
            circuit2(x)
            print(circuit2.to_symbolic())
    """

    def __init__(
            self,
            circuit_depth: int = 3,
            num_logic_ops_per_depth: int = 2,
            num_arithmetic_ops_per_depth: int = 2,
            use_residual: bool = True,
            use_layer_norm: bool = False,
            logic_op_types: Optional[List[str]] = None,
            arithmetic_op_types: Optional[List[str]] = None,
            routing_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            combination_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            circuit_routing: str = "output_only",
            apply_sigmoid_per_depth: str = "first_only",
            gate_entropy_coefficient: Optional[float] = None,
            load_balance_coefficient: Optional[float] = None,
            channel_mix: Optional[str] = None,
            selection_mode: str = "global",
            diversity_coefficient: float = 0.0,
            inner_logic_kwargs: Optional[Dict[str, Any]] = None,
            inner_arithmetic_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """
        Validate the arguments and build every stage.

        The stages are constructed here, one ``CircuitDepthLayer`` per
        depth, so they can be inspected through ``circuit_layers`` before
        the stack runs. This layer owns no weights of its own; the stages
        create theirs in :meth:`build`. The class docstring documents
        every parameter.
        """
        super().__init__(**kwargs)

        # Accept the canonical name and the deprecated alias.
        resolved_coef = _resolve_gate_entropy_coefficient(
            gate_entropy_coefficient,
            load_balance_coefficient,
            self.__class__.__name__,
        )

        if selection_mode not in ("global", "per_channel"):
            raise ValueError(
                f"selection_mode must be 'global' or 'per_channel', got "
                f"{selection_mode!r}."
            )

        # Validate
        if circuit_depth <= 0:
            raise ValueError("circuit_depth must be positive.")
        if num_logic_ops_per_depth <= 0:
            raise ValueError("num_logic_ops_per_depth must be positive.")
        if num_arithmetic_ops_per_depth <= 0:
            raise ValueError("num_arithmetic_ops_per_depth must be positive.")
        if apply_sigmoid_per_depth not in ("first_only", "all", "none"):
            raise ValueError(
                f"apply_sigmoid_per_depth must be 'first_only'|'all'|'none', "
                f"got {apply_sigmoid_per_depth!r}."
            )
        if diversity_coefficient < 0:
            raise ValueError("diversity_coefficient must be non-negative.")

        self.circuit_depth = circuit_depth
        self.num_logic_ops_per_depth = num_logic_ops_per_depth
        self.num_arithmetic_ops_per_depth = num_arithmetic_ops_per_depth
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.logic_op_types = logic_op_types
        self.arithmetic_op_types = arithmetic_op_types
        self.routing_initializer = keras.initializers.get(routing_initializer)
        self.combination_initializer = keras.initializers.get(combination_initializer)
        self.circuit_routing = circuit_routing
        self.apply_sigmoid_per_depth = apply_sigmoid_per_depth
        self.gate_entropy_coefficient = resolved_coef
        # Alias holding the same value, kept for callers on the old name.
        self.load_balance_coefficient = resolved_coef
        self.channel_mix = channel_mix
        self.selection_mode = selection_mode
        # The last axis build() sized channel-dependent stage state for --
        # per_channel combination weights, or a channel-mix Dense. None
        # when no stage builds anything that depends on it.
        self._build_last_dim: Optional[int] = None
        self.diversity_coefficient = float(diversity_coefficient)
        # These go to the stages, which pass them on to their children.
        # Keys a stage sets itself (operation_types, apply_sigmoid,
        # selection_mode, force_clip_when_no_sigmoid, name) are dropped
        # there with a warning and cannot be overridden from here. The copy
        # is deep, so a caller mutating the dict afterwards cannot reach
        # the stages.
        self.inner_logic_kwargs = copy.deepcopy(inner_logic_kwargs) if inner_logic_kwargs else {}
        self.inner_arithmetic_kwargs = copy.deepcopy(inner_arithmetic_kwargs) if inner_arithmetic_kwargs else {}

        # A 'first_only' stack only sigmoids depth 0, so depths >= 1 read
        # whatever the depth below produced. An arithmetic expert or a
        # residual add can leave [0, 1], and a logic child needs [0, 1].
        # When that combination is set up, clipping is turned on for those
        # depths and the caller is told.
        risky_stack = (
            self.apply_sigmoid_per_depth == "first_only"
            and self.circuit_depth >= 2
            and (
                self.num_arithmetic_ops_per_depth > 0
                or self.use_residual
            )
        )
        if risky_stack:
            logger.warning(
                "LearnableNeuralCircuit: apply_sigmoid_per_depth='first_only' "
                "with depth>=2 and (arithmetic experts OR use_residual=True) — "
                "auto-enabling force_logic_input_clip on depths >= 1 to "
                "guarantee logic-op inputs in [0, 1] "
                "(plan_2026-05-13_e33114da/D-004). Set apply_sigmoid_per_depth="
                "'all', use_residual=False with num_arithmetic_ops_per_depth=0 "
                "to silence."
            )

        self.circuit_layers: List[CircuitDepthLayer] = []
        for depth in range(self.circuit_depth):
            apply_sigmoid = self._sigmoid_for_depth(depth)
            self.circuit_layers.append(
                CircuitDepthLayer(
                    num_logic_ops=self.num_logic_ops_per_depth,
                    num_arithmetic_ops=self.num_arithmetic_ops_per_depth,
                    use_residual=self.use_residual,
                    logic_op_types=self.logic_op_types,
                    arithmetic_op_types=self.arithmetic_op_types,
                    routing_initializer=self.routing_initializer,
                    combination_initializer=self.combination_initializer,
                    circuit_routing=self.circuit_routing,
                    apply_sigmoid=apply_sigmoid,
                    gate_entropy_coefficient=self.gate_entropy_coefficient,
                    channel_mix=self.channel_mix,
                    force_logic_input_clip=risky_stack and depth >= 1,
                    selection_mode=self.selection_mode,
                    diversity_coefficient=self.diversity_coefficient,
                    inner_logic_kwargs=self.inner_logic_kwargs or None,
                    inner_arithmetic_kwargs=self.inner_arithmetic_kwargs or None,
                    name=f"circuit_depth_{depth}",
                    dtype=self.dtype_policy,
                )
            )
        self.layer_norms: List[keras.layers.LayerNormalization] = []
        if self.use_layer_norm:
            self.layer_norms = [
                keras.layers.LayerNormalization(
                    name=f"layer_norm_{depth}", dtype=self.dtype_policy
                )
                for depth in range(self.circuit_depth)
            ]

        logger.debug(
            f"LearnableNeuralCircuit: depth={circuit_depth}, "
            f"sigmoid_mode={apply_sigmoid_per_depth}, routing={circuit_routing}, "
            f"layer_norm={use_layer_norm}"
        )

    def _sigmoid_for_depth(self, depth: int) -> bool:
        """
        Say whether the stage at ``depth`` sigmoids its logic inputs.

        **One row per mode:**

        .. code-block:: text

            mode          depth 0  depth 1  depth 2  ...
            ------------  -------  -------  -----------
            'first_only'  sigmoid  no       no
            'all'         sigmoid  sigmoid  sigmoid
            'none'        no       no       no

            'first_only' is the default. A sigmoid of a
            sigmoid keeps narrowing the range, so an
            'all' stack of 3 can settle on a constant.
            'none' needs the caller to supply [0, 1].

        :param depth: Index of the stage, starting at 0.
        :type depth: int
        :return: What to pass that stage as ``apply_sigmoid``.
        :rtype: bool
        """
        if self.apply_sigmoid_per_depth == "all":
            return True
        if self.apply_sigmoid_per_depth == "none":
            return False
        # 'first_only'
        return depth == 0

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build every stage and, if enabled, every norm.

        This layer has no weights of its own. It builds its children
        explicitly because Keras 3 requires the parent to create all child
        state during its own build; otherwise a saved model reloads with
        fresh child weights.

        :param input_shape: Shape of the input tensor, rank 2 or more.
            Every stage sees the same shape, since the shape is preserved.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the rank is below 2.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"LearnableNeuralCircuit expects rank >= 2 input, "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )
        if self.selection_mode == "per_channel" or self.channel_mix == "dense":
            self._build_last_dim = int(input_shape[-1])
        for circuit_layer in self.circuit_layers:
            circuit_layer.build(input_shape)
        for layer_norm in self.layer_norms:
            layer_norm.build(input_shape)
        super().build(input_shape)

    def _assert_call_shape_contract(
            self,
            inputs: keras.KerasTensor
    ) -> None:
        """
        Re-assert in ``call`` the static shape contract ``build`` checked.

        A contract checked only in ``build`` is checked once, against
        whatever shape arrived first: the layer stays built and every
        later call is unchecked. ``InputSpec`` cannot close this, because
        ``assert_input_compatibility`` tests ``shape[axis] not in
        {value, None}`` and so accepts an unknown dimension. A dimension
        that is ``None`` here is genuinely unknown at trace time and is
        skipped rather than guessed.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :raises ValueError: If the rank is below 2, or the last axis
            differs from the one ``build`` sized channel-dependent state
            for.
        """
        shape = tuple(inputs.shape)
        if len(shape) < 2:
            raise ValueError(
                f"{type(self).__name__} expects rank >= 2 input, "
                f"got shape with {len(shape)} dimensions: {shape}"
            )
        if self._build_last_dim is None:
            return
        if shape[-1] is not None and int(shape[-1]) != self._build_last_dim:
            raise ValueError(
                f"{type(self).__name__} was built for a last axis of "
                f"{self._build_last_dim} and cannot run on shape "
                f"{shape}. Build a separate layer per input width."
            )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the input through every stage in order.

        :param inputs: A tensor of rank 2 or more.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every stage and
            every norm.
        :type training: Optional[bool]
        :return: The output of the last stage, shaped like ``inputs``.
        :rtype: keras.KerasTensor
        """
        # The static shape contract build() checked, re-checked here.
        self._assert_call_shape_contract(inputs)

        x = inputs
        for depth in range(self.circuit_depth):
            x = self.circuit_layers[depth](x, training=training)
            if self.use_layer_norm:
                x = self.layer_norms[depth](x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def to_symbolic(self, top_k: int = 1) -> str:
        """
        Report what every depth settled on, as indented text.

        Each depth gets a ``depth N:`` header followed by that stage's own
        summary, indented two spaces. The per-stage lines come from
        :meth:`CircuitDepthLayer.to_symbolic`.

        :param top_k: How many entries to keep on each line.
        :type top_k: int
        :return: A multi-line summary covering every depth.
        :rtype: str
        :raises RuntimeError: If the layer is not built yet.
        """
        if not self.built:
            raise RuntimeError(
                "LearnableNeuralCircuit.to_symbolic() requires the layer to "
                "be built. Call the layer on a sample input first."
            )
        lines: List[str] = []
        for depth, cl in enumerate(self.circuit_layers):
            lines.append(f"depth {depth}:")
            for line in cl.to_symbolic(top_k=top_k).split("\n"):
                lines.append(f"  {line}")
        return "\n".join(lines)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this layer.

        Every constructor argument appears as a key, including the
        deprecated ``load_balance_coefficient``. That key is always
        emitted as ``None``, never as the resolved value: the coefficient
        travels in the canonical ``gate_entropy_coefficient`` key, and
        ``None`` is exactly what the alias parameter contributed once
        ``__init__`` resolved the two. Emitting the value instead would
        make every load hand ``__init__`` the deprecated name, and a
        round trip would then either warn on every load or silently
        depend on which of the two names wins. A config written by an
        older version carrying a non-``None``
        ``load_balance_coefficient`` still loads, because ``__init__``
        accepts both names.

        :return: A serializable config dict.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "circuit_depth": self.circuit_depth,
            "num_logic_ops_per_depth": self.num_logic_ops_per_depth,
            "num_arithmetic_ops_per_depth": self.num_arithmetic_ops_per_depth,
            "use_residual": self.use_residual,
            "use_layer_norm": self.use_layer_norm,
            "logic_op_types": self.logic_op_types,
            "arithmetic_op_types": self.arithmetic_op_types,
            "routing_initializer": keras.initializers.serialize(self.routing_initializer),
            "combination_initializer": keras.initializers.serialize(self.combination_initializer),
            "circuit_routing": self.circuit_routing,
            "apply_sigmoid_per_depth": self.apply_sigmoid_per_depth,
            "gate_entropy_coefficient": self.gate_entropy_coefficient,
            # DECISION plan-2026-08-29T112804-aff039c4/D-003 -- the
            # deprecated alias key is always None. Do not emit the
            # resolved value: every load would then hand __init__ the
            # deprecated name. See decisions.md D-003.
            "load_balance_coefficient": None,
            "channel_mix": self.channel_mix,
            "selection_mode": self.selection_mode,
            "diversity_coefficient": self.diversity_coefficient,
            "inner_logic_kwargs": self.inner_logic_kwargs or None,
            "inner_arithmetic_kwargs": self.inner_arithmetic_kwargs or None,
        })
        return config

# ---------------------------------------------------------------------
