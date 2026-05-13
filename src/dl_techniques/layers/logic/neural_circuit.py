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


# DECISION plan_2026-05-13_3a2f1d23/D-002 (rationale corrected in
# plan_2026-05-13_e33114da/D-007)
# The H6 rename swapped ``load_balance_coefficient`` for the canonical name
# ``gate_entropy_coefficient``. The implementation computes
# ``coef * N * mean(sum(beta^2, axis=-1))`` which is the **Shazeer (2017)
# importance regularizer** — algebraically equivalent (up to constant) to
# the CV^2 of importance values when N is fixed — and is **L2 of the gate
# probability vector**, NOT entropy. Both forms are convex measures of
# peakiness and have the same optimum (uniform β), but the *name* is a
# misnomer in the strict sense. We keep the canonical name for back-compat
# (anchored in saved models) and document the math accurately here.
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
            inner_logic_kwargs: Optional[Dict[str, Any]] = None,
            inner_arithmetic_kwargs: Optional[Dict[str, Any]] = None,
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

        # G1 (plan_2026-05-13_e33114da): inner_*_kwargs forwards arbitrary
        # configuration into the inner LearnableLogicOperator and
        # LearnableArithmeticOperator instances. Wrapper-controlled keys
        # (listed in _WRAPPER_OWNED_*) cannot be overridden — the wrapper
        # always wins. Collisions are warned, not errored.
        self.inner_logic_kwargs = dict(inner_logic_kwargs) if inner_logic_kwargs else {}
        self.inner_arithmetic_kwargs = dict(inner_arithmetic_kwargs) if inner_arithmetic_kwargs else {}

        # DECISION plan_2026-05-13_a2b0f17b/D-002 — children created in
        # __init__. Inner logic ops opt into legacy unary x2=x1 rebinding
        # because CircuitDepthLayer feeds them a single tensor.
        # DECISION plan_2026-05-13_e33114da/D-006 — wrapper-owned keys are
        # popped from user dicts with a warning if collision detected.
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

        self.logic_operators = [
            LearnableLogicOperator(
                operation_types=self.logic_op_types,
                apply_sigmoid=self.apply_sigmoid,
                allow_unary_degenerate=True,
                force_clip_when_no_sigmoid=self.force_logic_input_clip,
                selection_mode=self.selection_mode,
                name=f"logic_op_{i}",
                **logic_extra,
            )
            for i in range(self.num_logic_ops)
        ]
        self.arithmetic_operators = [
            LearnableArithmeticOperator(
                operation_types=self.arithmetic_op_types,
                selection_mode=self.selection_mode,
                name=f"arithmetic_op_{i}",
                **arith_extra,
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
        Shazeer (2017)-style importance regularizer aux loss.

        Implementation: ``coef * N * mean_over_extra_axes(sum_op(beta^2))``.
        Equivalent (up to constant) to the CV² importance loss when N is
        fixed — penalizes peaky combination distributions. Note: this is
        L2 of the gate vector, NOT entropy, despite the
        ``gate_entropy_coefficient`` field name (kept for back-compat).

        # DECISION plan_2026-05-13_e33114da/D-005 — per_channel mode
        # previously averaged probs across channels BEFORE the L2, which let
        # per-channel-peaky distributions that average to uniform escape the
        # regularizer (B4). Fixed by per-channel L2 then mean.
        """
        if self.load_balance_coefficient <= 0:
            return
        n = float(self.num_logic_ops + self.num_arithmetic_ops)
        # Per-row L2 (axis=-1 = op axis). For 1-D global (N,), this is just
        # sum(beta^2); for per-channel (C, N), it's per-channel L2 then mean.
        per_row_l2 = ops.sum(ops.square(combination_probs), axis=-1)
        aux = ops.mean(per_row_l2) if len(combination_probs.shape) > 1 else per_row_l2
        self.add_loss(self.load_balance_coefficient * n * aux)

    def _maybe_diversity_loss(self) -> None:
        """M5: pairwise cosine-similarity penalty over inner ops' operation
        probability vectors. Encourages experts to specialize on distinct
        operators rather than collapsing onto the same one.

        D4 (plan_2026-05-13_e33114da): vectorized via per-arity Gram matrix.
        Cross-arity pairs (logic vs arithmetic) have different op-space
        dimensionality so are kept separate (matches prior behavior).
        """
        if self.diversity_coefficient <= 0:
            return

        def _group_sim(ops_group: List[Any]) -> Tuple[Optional[keras.KerasTensor], int]:
            """Compute mean pairwise cosine similarity for a same-arity group.
            Returns (sim_tensor, pair_count) or (None, 0) if <2 experts."""
            if len(ops_group) < 2:
                return None, 0
            vecs = []
            for op in ops_group:
                p = op._operation_probs(deterministic=True)
                if len(p.shape) > 1:
                    p = ops.mean(p, axis=0)
                vecs.append(p)
            stacked = ops.stack(vecs, axis=0)  # (K, M)
            norms = ops.add(ops.norm(stacked, axis=-1, keepdims=True), 1e-12)
            stacked = ops.divide(stacked, norms)
            gram = ops.matmul(stacked, ops.transpose(stacked))
            # Upper triangle sum excluding diagonal = (sum(gram) - trace) / 2.
            diag = ops.sum(ops.multiply(gram, ops.eye(len(ops_group), dtype=gram.dtype)))
            upper_sum = ops.divide(ops.subtract(ops.sum(gram), diag), 2.0)
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
            total_sim = arith_sum if total_sim is None else ops.add(total_sim, arith_sum)

        mean_sim = ops.divide(total_sim, float(total_pairs))
        self.add_loss(ops.multiply(self.diversity_coefficient, mean_sim))

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the depth layer."""
        combination_probs = ops.softmax(self.combination_weights, axis=-1)
        # _maybe_load_balance_loss now handles both (N,) and (C, N) shapes
        # natively — per-channel L2 then mean (B4 fix, D-005).
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

    def to_symbolic(self, top_k: int = 1) -> str:
        """Return a symbolic summary of this depth layer.

        G2 (plan_2026-05-13_e33114da): standalone depth-level to_symbolic
        mirroring the per-depth body of LearnableNeuralCircuit.to_symbolic.
        Useful when CircuitDepthLayer is used outside a LearnableNeuralCircuit.
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
        cw = ops.convert_to_numpy(ops.softmax(self.combination_weights, axis=-1))
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
            "inner_logic_kwargs": self.inner_logic_kwargs or None,
            "inner_arithmetic_kwargs": self.inner_arithmetic_kwargs or None,
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
            diversity_coefficient: float = 0.0,
            inner_logic_kwargs: Optional[Dict[str, Any]] = None,
            inner_arithmetic_kwargs: Optional[Dict[str, Any]] = None,
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
        self.load_balance_coefficient = resolved_coef  # deprecated alias
        self.channel_mix = channel_mix
        self.selection_mode = selection_mode
        # B5 fix: diversity_coefficient now reachable through the wrapper.
        self.diversity_coefficient = float(diversity_coefficient)
        # G1 fix: inner_*_kwargs forwarded verbatim to inner ops via the
        # child CircuitDepthLayer. Wrapper-controlled keys (operation_types,
        # apply_sigmoid, selection_mode, force_clip_when_no_sigmoid, name)
        # cannot be overridden — those are set by the wrapper itself.
        self.inner_logic_kwargs = dict(inner_logic_kwargs) if inner_logic_kwargs else {}
        self.inner_arithmetic_kwargs = dict(inner_arithmetic_kwargs) if inner_arithmetic_kwargs else {}

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
                    diversity_coefficient=self.diversity_coefficient,
                    inner_logic_kwargs=self.inner_logic_kwargs or None,
                    inner_arithmetic_kwargs=self.inner_arithmetic_kwargs or None,
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
        operator per inner expert plus a combination-weight ranking.
        Delegates to CircuitDepthLayer.to_symbolic per depth (G2,
        plan_2026-05-13_e33114da).
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
            "diversity_coefficient": self.diversity_coefficient,
            "inner_logic_kwargs": self.inner_logic_kwargs or None,
            "inner_arithmetic_kwargs": self.inner_arithmetic_kwargs or None,
        })
        return config

# ---------------------------------------------------------------------
