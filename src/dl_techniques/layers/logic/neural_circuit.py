"""
A parallelized, learnable computational block for a neural circuit.

This layer represents a single, complex computational stage designed to be
stacked in a deep "neural circuit." It departs from monolithic layers like
convolution by creating a parallel ensemble of diverse, learnable operators
(both logical and arithmetic) and then learning how to route information
between them.

Architecture (post plan_2026-05-13_a2b0f17b):
    The depth layer's design is inspired by Mixture of Experts (MoE) but
    has TWO selectable routing modes:

    - ``circuit_routing='output_only'`` (NEW DEFAULT):
          ``Y = sum_i beta_i * f_i(X) [+ X]``
      Each expert sees the full input X. Only the *output* fusion is gated
      by the softmax-normalized combination weights ``beta``. This avoids
      the input attenuation problem of the classic mode (each expert seeing
      ``X / N`` on average) and makes the layer behave like a true soft-MoE.

    - ``circuit_routing='classic'``:
          ``Y = sum_i beta_i * f_i(alpha_i * X) [+ X]``
      The original behavior preserved for backwards compatibility with
      models trained against the prior (attenuated) routing.

Optional features (all opt-in, default off):

  - ``load_balance_coefficient`` > 0 enables Shazeer-style auxiliary loss
    that pushes expert utilization toward uniform.
  - ``channel_mix='dense'`` appends a learnable per-channel ``Dense`` mixing
    layer after the fusion.
  - ``apply_sigmoid_per_depth`` (on ``LearnableNeuralCircuit``) controls
    where ``LearnableLogicOperator`` instances apply their input sigmoid:
    ``'first_only'`` (default, recommended for stacking — avoids the
    sigmoid-of-sigmoid range collapse), ``'all'`` (legacy), or ``'none'``.
"""

import warnings

import keras
from keras import ops
from typing import List, Optional, Union, Any, Dict, Tuple


# DECISION plan_2026-05-13_3a2f1d23/D-002
# H6: ``load_balance_coefficient`` was a misnomer — the loss it controls is
# actually the Shazeer (2017) gate-entropy regularizer, not load-balance.
# The new canonical name is ``gate_entropy_coefficient``. The old name is
# kept as a deprecated alias with a one-time DeprecationWarning to avoid
# silently breaking external callers.
def _resolve_gate_entropy_coefficient(
    gate_entropy_coefficient: Optional[float],
    load_balance_coefficient: Optional[float],
    cls_name: str,
) -> float:
    """Resolve the gate-entropy coefficient honoring back-compat aliasing.

    If only the deprecated ``load_balance_coefficient`` is passed, emit a
    DeprecationWarning and use its value. If both are passed, the new name
    wins. Returns the resolved float (default 0.0).
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
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .logic_operators import LearnableLogicOperator
from .arithmetic_operators import LearnableArithmeticOperator

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CircuitDepthLayer(keras.layers.Layer):
    """
    Single depth layer of a neural circuit with parallel expert operators.

    Implements a MoE-inspired computational stage containing parallel logic and
    arithmetic operator "experts" with learnable output fusion. With the
    default ``circuit_routing='output_only'`` mode, each expert sees the full
    input and only the output is gated by ``beta = softmax(w_combination)``:
    ``Y = sum_i(beta_i * f_i(X)) [+ X]``.

    See module docstring for the legacy ``'classic'`` mode and the load-balance
    / channel-mix opt-ins.

    :param num_logic_ops: Number of logic operators to run in parallel.
    :param num_arithmetic_ops: Number of arithmetic operators to run in parallel.
    :param use_residual: Whether to add an input-skip residual to the fusion.
    :param logic_op_types: Optional list of logic operation types to expose.
    :param arithmetic_op_types: Optional list of arithmetic operation types.
    :param routing_initializer: Initializer for routing weights (only used in
        ``circuit_routing='classic'`` mode; preserved on the layer for
        serialization compatibility regardless).
    :param combination_initializer: Initializer for combination weights.
    :param circuit_routing: ``'output_only'`` (default — fixed math) or
        ``'classic'`` (legacy attenuated input gating).
    :param apply_sigmoid: Forwarded to inner ``LearnableLogicOperator`` instances.
        Default ``True`` matches legacy behavior.
    :param load_balance_coefficient: If > 0, add a Shazeer-style aux loss that
        encourages uniform expert utilization. Default 0 (off).
    :param channel_mix: ``'dense'`` to append a per-channel ``Dense(C, C)``
        mixing layer; ``None`` (default) for shape-pure pointwise behavior.
    :param kwargs: Additional keyword arguments for the Layer base class.
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
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # H6: resolve canonical name + deprecated alias.
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
        # H6: canonical name is gate_entropy_coefficient. The attribute
        # load_balance_coefficient remains as a read-only alias for back-compat.
        self.gate_entropy_coefficient = resolved_coef
        self.load_balance_coefficient = resolved_coef  # deprecated alias
        self.channel_mix = channel_mix
        # C4: forwards force_clip to inner LearnableLogicOperator instances.
        self.force_logic_input_clip = force_logic_input_clip
        # C3: selection_mode forwarded to inner experts AND used for own
        # combination weights shape.
        self.selection_mode = selection_mode
        self._channels = None  # set in build() for per_channel mode
        # M5: diversity regularizer coefficient.
        self.diversity_coefficient = float(diversity_coefficient)

        # DECISION plan_2026-05-13_a2b0f17b/D-002 — children created in
        # __init__, NOT manually pre-built in build(). Auto-build via
        # __call__ is the Keras 3 idiomatic pattern; manual pre-build was
        # cargo-culted and risks double-build.
        # CircuitDepthLayer intentionally feeds inner logic ops a single
        # tensor X (each expert sees the full input; only output fusion is
        # gated). Opt these inner instances in to the legacy x2=x1 rebinding
        # explicitly — M8 (plan_2026-05-13_3a2f1d23) flipped the public
        # default to False but inside the circuit the contract has always
        # been unary-input.
        self.logic_operators = [
            LearnableLogicOperator(
                operation_types=self.logic_op_types,
                apply_sigmoid=self.apply_sigmoid,
                allow_unary_degenerate=True,
                force_clip_when_no_sigmoid=self.force_logic_input_clip,
                selection_mode=self.selection_mode,
                name=f"logic_op_{i}",
            )
            for i in range(self.num_logic_ops)
        ]
        self.arithmetic_operators = [
            LearnableArithmeticOperator(
                operation_types=self.arithmetic_op_types,
                selection_mode=self.selection_mode,
                name=f"arithmetic_op_{i}",
            )
            for i in range(self.num_arithmetic_ops)
        ]

        # Channel-mix sublayer is built lazily once we know the channel size.
        self._channel_mix_layer: Optional[keras.layers.Dense] = None

        # Weights — created in build()
        self.routing_weights = None
        self.combination_weights = None

        logger.debug(
            f"CircuitDepthLayer: routing={circuit_routing}, "
            f"{num_logic_ops}+{num_arithmetic_ops} ops, residual={use_residual}, "
            f"gate_entropy={resolved_coef}, channel_mix={channel_mix}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build own weights AND explicitly build all children (Keras 3
        invariant for serialization — parent.build must create child state)."""
        if len(input_shape) < 2:
            raise ValueError(
                f"CircuitDepthLayer expects rank >= 2 input, "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        total_operators = self.num_logic_ops + self.num_arithmetic_ops

        # C3: per-channel selection shapes combination weights as
        # (channels, total_operators). Routing weights stay 1-D since
        # 'classic' mode predates per-channel and we don't expand that API.
        if self.selection_mode == "per_channel":
            if input_shape[-1] is None:
                raise ValueError(
                    "selection_mode='per_channel' requires a concrete "
                    f"last-axis dimension; got {input_shape}."
                )
            self._channels = int(input_shape[-1])
            combination_shape = (self._channels, total_operators)
        else:
            combination_shape = (total_operators,)

        # Routing weights still created in 'output_only' for serialization
        # compatibility (zero-cost; only consulted in 'classic' mode).
        self.routing_weights = self.add_weight(
            name="routing_weights",
            shape=(total_operators,),
            initializer=self.routing_initializer,
            trainable=True,
        )
        self.combination_weights = self.add_weight(
            name="combination_weights",
            shape=combination_shape,
            initializer=self.combination_initializer,
            trainable=True,
        )

        # Children must be built by the parent's build() per Keras 3 contract.
        # H9 (plan_2026-05-13_3a2f1d23): keep these explicit child.build()
        # calls — removing them was attempted in plan_a2b0f17b and reversed
        # because Keras 3 does NOT auto-build sub-layers that aren't called
        # via __call__ in the parent's call(); .keras save/load round-trip
        # then fails. See LESSONS L42 and plan_a2b0f17b D-003.
        for op in self.logic_operators:
            op.build(input_shape)
        for op in self.arithmetic_operators:
            op.build(input_shape)

        if self.channel_mix == "dense":
            channel_dim = int(input_shape[-1])
            self._channel_mix_layer = keras.layers.Dense(
                channel_dim,
                use_bias=True,
                name="channel_mix",
            )
            self._channel_mix_layer.build(input_shape)

        super().build(input_shape)

    def _maybe_load_balance_loss(
        self, combination_probs: keras.KerasTensor
    ) -> None:
        """
        Shazeer (2017) load-balancing aux loss.

        For a single-example setting (no batch routing decisions), we use
        ``coef * N * sum(beta^2)`` which penalizes peaky combination
        distributions and rewards uniform utilization. This is a degenerate
        but standard variant when there is no per-token routing.
        """
        if self.load_balance_coefficient > 0:
            n = float(self.num_logic_ops + self.num_arithmetic_ops)
            l2 = ops.sum(ops.square(combination_probs))
            self.add_loss(self.load_balance_coefficient * n * l2)

    def _maybe_diversity_loss(self) -> None:
        """M5: pairwise cosine-similarity penalty over inner ops' operation
        probability vectors. Encourages experts to specialize on distinct
        operators rather than collapsing onto the same one."""
        if self.diversity_coefficient <= 0:
            return
        inner = list(self.logic_operators) + list(self.arithmetic_operators)
        # Each inner op's operation_probs has shape (N,) (global) or (C, N)
        # (per_channel). Reduce to a single vector per expert by mean over
        # the first axis when rank > 1.
        vecs = []
        for op in inner:
            p = op._operation_probs(deterministic=True)
            if len(p.shape) > 1:
                p = ops.mean(p, axis=0)
            # L2-normalize.
            p = ops.divide(p, ops.add(ops.norm(p), 1e-12))
            vecs.append(p)
        # Note: experts may have different num_operations (logic vs arith),
        # so we cannot stack directly. Compute pairwise cos-sim only between
        # same-arity pairs.
        sim_sum = 0.0
        pair_count = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if vecs[i].shape == vecs[j].shape:
                    sim_sum = ops.add(sim_sum, ops.sum(ops.multiply(vecs[i], vecs[j])))
                    pair_count += 1
        if pair_count == 0:
            return
        # Mean similarity in [-1, 1]; aux loss is coef * mean(sim).
        mean_sim = ops.divide(sim_sum, float(pair_count))
        self.add_loss(ops.multiply(self.diversity_coefficient, mean_sim))

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the depth layer."""
        combination_probs = ops.softmax(self.combination_weights, axis=-1)
        # _maybe_load_balance_loss expects a 1-D combination_probs; under
        # per_channel it's (C, N). Reduce by mean over channels for the aux.
        if self.selection_mode == "per_channel":
            self._maybe_load_balance_loss(ops.mean(combination_probs, axis=0))
        else:
            self._maybe_load_balance_loss(combination_probs)
        # M5: diversity regularizer.
        self._maybe_diversity_loss()

        all_outputs: List[keras.KerasTensor] = []

        if self.circuit_routing == "classic":
            # Legacy behavior — input attenuation by softmax(routing_weights).
            routing_probs = ops.softmax(self.routing_weights)
            input_rank = len(ops.shape(inputs))

            for i, logic_op in enumerate(self.logic_operators):
                weight = routing_probs[i]
                weighted_input = ops.multiply(inputs, weight)
                all_outputs.append(logic_op(weighted_input, training=training))
            for j, arithmetic_op in enumerate(self.arithmetic_operators):
                weight = routing_probs[self.num_logic_ops + j]
                weighted_input = ops.multiply(inputs, weight)
                all_outputs.append(arithmetic_op(weighted_input, training=training))
        else:
            # output_only — every expert sees full X; only fusion is gated.
            for logic_op in self.logic_operators:
                all_outputs.append(logic_op(inputs, training=training))
            for arithmetic_op in self.arithmetic_operators:
                all_outputs.append(arithmetic_op(inputs, training=training))

        # Vectorized weighted fusion.
        n = self.num_logic_ops + self.num_arithmetic_ops
        if self.selection_mode == "per_channel":
            stacked = ops.stack(all_outputs, axis=-1)  # (..., C, N)
            rank = len(stacked.shape)
            weight_shape = (1,) * (rank - 2) + (self._channels, n)
            weights = ops.reshape(combination_probs, weight_shape)
            combined_output = ops.sum(ops.multiply(weights, stacked), axis=-1)
        else:
            stacked = ops.stack(all_outputs, axis=0)
            weight_shape = (n,) + (1,) * (len(stacked.shape) - 1)
            weights = ops.reshape(combination_probs, weight_shape)
            combined_output = ops.sum(ops.multiply(weights, stacked), axis=0)

        if self._channel_mix_layer is not None:
            combined_output = self._channel_mix_layer(combined_output)

        if self.use_residual:
            combined_output = ops.add(combined_output, inputs)

        return combined_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
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
            # H6: emit canonical name. Old key omitted to keep config clean;
            # round-trip works because __init__ accepts both names.
            "gate_entropy_coefficient": self.gate_entropy_coefficient,
            "channel_mix": self.channel_mix,
            "force_logic_input_clip": self.force_logic_input_clip,
            "selection_mode": self.selection_mode,
            "diversity_coefficient": self.diversity_coefficient,
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LearnableNeuralCircuit(keras.layers.Layer):
    """
    Deep learnable neural circuit with stacked parallel operator layers.

    Implements a multi-depth neural circuit where each depth level is a
    ``CircuitDepthLayer``. ``apply_sigmoid_per_depth`` controls where the
    inner ``LearnableLogicOperator`` instances sigmoid-normalize their input:

    - ``'first_only'`` (NEW DEFAULT): only the first depth applies sigmoid.
      Subsequent depths assume signals are already in ``[0, 1]``-ish from
      the prior fuzzy-logic outputs. This is the recommended mode for
      stacking — it prevents the sigmoid-of-sigmoid range collapse where
      a 3-layer stack converges to a constant.
    - ``'all'``: every depth applies sigmoid (legacy behavior, causes
      stack-collapse for unbounded inputs).
    - ``'none'``: never apply sigmoid (caller guarantees ``[0, 1]`` inputs).
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
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # H6: resolve canonical name + deprecated alias.
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
        self.load_balance_coefficient = resolved_coef  # deprecated alias
        self.channel_mix = channel_mix
        self.selection_mode = selection_mode

        # C4 (plan_2026-05-13_3a2f1d23): when first_only mode is on and
        # depths >= 1 inner logic ops have apply_sigmoid=False, any source of
        # out-of-[0,1] values feeding into them breaks fuzzy semantics. Two
        # sources: (a) arithmetic experts at depth 0 (unbounded outputs);
        # (b) use_residual=True propagates the raw input X to depth 1.
        # DECISION plan_2026-05-13_e33114da/D-004 — risky_stack now triggers
        # on EITHER source. B3 widened the prior guard which only caught (a).
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

        # DECISION plan_2026-05-13_a2b0f17b/D-002 — sublayers created in
        # __init__, build lazily via __call__.
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
                    name=f"circuit_depth_{depth}",
                )
            )
        self.layer_norms: List[keras.layers.LayerNormalization] = []
        if self.use_layer_norm:
            self.layer_norms = [
                keras.layers.LayerNormalization(name=f"layer_norm_{depth}")
                for depth in range(self.circuit_depth)
            ]

        logger.debug(
            f"LearnableNeuralCircuit: depth={circuit_depth}, "
            f"sigmoid_mode={apply_sigmoid_per_depth}, routing={circuit_routing}, "
            f"layer_norm={use_layer_norm}"
        )

    def _sigmoid_for_depth(self, depth: int) -> bool:
        if self.apply_sigmoid_per_depth == "all":
            return True
        if self.apply_sigmoid_per_depth == "none":
            return False
        # 'first_only'
        return depth == 0

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) < 2:
            raise ValueError(
                f"LearnableNeuralCircuit expects rank >= 2 input, "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )
        # Build sublayers explicitly per Keras 3 serialization contract.
        for circuit_layer in self.circuit_layers:
            circuit_layer.build(input_shape)
        for layer_norm in self.layer_norms:
            layer_norm.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        x = inputs
        for depth in range(self.circuit_depth):
            x = self.circuit_layers[depth](x, training=training)
            if self.use_layer_norm:
                x = self.layer_norms[depth](x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def to_symbolic(self, top_k: int = 1) -> str:
        """Walk all depths and return a multi-line symbolic summary.

        M1 (plan_2026-05-13_3a2f1d23): for each depth print the dominant
        operator per inner expert plus a combination-weight ranking. Useful
        for post-training interpretation.
        """
        if not self.built:
            raise RuntimeError(
                "LearnableNeuralCircuit.to_symbolic() requires the layer to "
                "be built. Call the layer on a sample input first."
            )
        lines: List[str] = []
        for depth, cl in enumerate(self.circuit_layers):
            lines.append(f"depth {depth}:")
            for i, op in enumerate(cl.logic_operators):
                lines.append(f"  logic_op_{i}: {op.to_symbolic(top_k=top_k)}")
            for j, op in enumerate(cl.arithmetic_operators):
                lines.append(f"  arithmetic_op_{j}: {op.to_symbolic(top_k=top_k)}")
            # Combination weights ranking. Per-channel mode collapses to mean.
            cw = ops.convert_to_numpy(
                ops.softmax(cl.combination_weights, axis=-1)
            )
            if cw.ndim > 1:
                cw = cw.mean(axis=0)
            total = cl.num_logic_ops + cl.num_arithmetic_ops
            names = (
                [f"logic_op_{i}" for i in range(cl.num_logic_ops)]
                + [f"arithmetic_op_{j}" for j in range(cl.num_arithmetic_ops)]
            )
            ranked = sorted(zip(names, cw.tolist()), key=lambda kv: -kv[1])[:top_k]
            lines.append(
                "  combination: "
                + ", ".join(f"{n}({p:.3f})" for n, p in ranked)
            )
        return "\n".join(lines)

    def get_config(self) -> Dict[str, Any]:
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
            "channel_mix": self.channel_mix,
            "selection_mode": self.selection_mode,
        })
        return config

# ---------------------------------------------------------------------
