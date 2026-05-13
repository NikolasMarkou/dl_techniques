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

import keras
from keras import ops
from typing import List, Optional, Union, Any, Dict, Tuple

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
            routing_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            combination_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            circuit_routing: str = "output_only",
            apply_sigmoid: bool = True,
            load_balance_coefficient: float = 0.0,
            channel_mix: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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
        if load_balance_coefficient < 0:
            raise ValueError("load_balance_coefficient must be non-negative.")
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
        self.load_balance_coefficient = load_balance_coefficient
        self.channel_mix = channel_mix

        # DECISION plan_2026-05-13_a2b0f17b/D-002 — children created in
        # __init__, NOT manually pre-built in build(). Auto-build via
        # __call__ is the Keras 3 idiomatic pattern; manual pre-build was
        # cargo-culted and risks double-build.
        self.logic_operators = [
            LearnableLogicOperator(
                operation_types=self.logic_op_types,
                apply_sigmoid=self.apply_sigmoid,
                name=f"logic_op_{i}",
            )
            for i in range(self.num_logic_ops)
        ]
        self.arithmetic_operators = [
            LearnableArithmeticOperator(
                operation_types=self.arithmetic_op_types,
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
            f"load_balance={load_balance_coefficient}, channel_mix={channel_mix}"
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
            shape=(total_operators,),
            initializer=self.combination_initializer,
            trainable=True,
        )

        # Children must be built by the parent's build() per Keras 3 contract.
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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the depth layer."""
        combination_probs = ops.softmax(self.combination_weights)
        self._maybe_load_balance_loss(combination_probs)

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
        stacked = ops.stack(all_outputs, axis=0)
        n = self.num_logic_ops + self.num_arithmetic_ops
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
            "load_balance_coefficient": self.load_balance_coefficient,
            "channel_mix": self.channel_mix,
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
            routing_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            combination_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            circuit_routing: str = "output_only",
            apply_sigmoid_per_depth: str = "first_only",
            load_balance_coefficient: float = 0.0,
            channel_mix: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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
        self.load_balance_coefficient = load_balance_coefficient
        self.channel_mix = channel_mix

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
                    load_balance_coefficient=self.load_balance_coefficient,
                    channel_mix=self.channel_mix,
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
            "load_balance_coefficient": self.load_balance_coefficient,
            "channel_mix": self.channel_mix,
        })
        return config

# ---------------------------------------------------------------------
