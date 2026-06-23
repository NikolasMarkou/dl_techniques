"""
A differentiable operator that learns logical functions.

This layer embeds principles of fuzzy logic into a neural network,
enabling it to learn logical combinations of input features. It provides a
differentiable framework for selecting and applying logical operations
(e.g., AND, OR, XOR), moving beyond traditional feature summation or
concatenation and toward a form of neuro-symbolic reasoning.

Architecture:
    The layer's architecture is designed to create a continuous and
    differentiable proxy for discrete Boolean logic. The process involves
    three main stages:

    1.  **Input Normalization:** Input tensors, which can have any real
        values, are first passed through a sigmoid function. This maps all
        values to the range `[0, 1]`, allowing them to be interpreted as
        probabilistic or "fuzzy" truth values, where 0 represents `False`
        and 1 represents `True`.

    2.  **Soft Logic Operations:** A predefined set of "soft" logical
        operations are applied in parallel to the normalized inputs. Each
        operation is a differentiable function that emulates the behavior
        of its discrete Boolean counterpart at the boundaries (0 and 1)
        while providing smooth gradients for intermediate values.

    3.  **Differentiable Selection:** The final output is a convex
        combination of the results from all soft logic operations. This is
        achieved using a learnable weight vector, passed through a softmax
        function, which assigns a probability to each operation. The
        network learns to increase the weights for operations that are
        most effective for the task.

Foundational Mathematics:
    The core of this layer lies in its formulation of differentiable logic
    gates, which draw heavily from probability theory and fuzzy logic. For
    inputs `p` and `q` in the range `[0, 1]`:

    -   **Soft NOT:** The standard complement is used:
        `NOT(p) = 1 - p`

    -   **Soft AND:** This is modeled by the product of probabilities,
        corresponding to the 'product t-norm' in fuzzy logic:
        `AND(p, q) = p * q`

    -   **Soft OR:** Modeled using the probabilistic sum (derived from the
        inclusion-exclusion principle):
        `OR(p, q) = P(p U q) = P(p) + P(q) - P(p intersect q) = p + q - p*q`

    -   **Soft XOR:** Derived from its definition `(p OR q) AND (NOT(p AND q))`,
        a common differentiable form is:
        `XOR(p, q) = p + q - 2*p*q`

    The weighted combination of these operations is controlled by a softmax
    distribution over a learnable weight vector `w`, often with a
    temperature `T`. The probability `alpha_i` for selecting the i-th
    logical operation `f_i` is:

        alpha_i = exp(w_i / T) / sum_j(exp(w_j / T))

    The final output `Y` is the weighted sum of all operation results:

        Y = sum_i(alpha_i * f_i(X))

    This design allows gradients to flow back to the weights `w` and the
    temperature `T`, enabling the model to learn the optimal logical
    structure from data.

References:
    - The concept of continuous relaxations for discrete choices is central
      to Differentiable Architecture Search (DARTS).
      Liu, H., Simonyan, K., & Yang, Y. (2018). "DARTS: Differentiable
      Architecture Search".

    - The mathematical forms of the soft logic gates are standard in
      fuzzy logic literature.
      Zadeh, L. A. (1965). "Fuzzy sets". Information and Control.

    - This approach is part of a broader field of neuro-symbolic AI, which
      aims to integrate neural learning with symbolic reasoning.
      Garcez, A. S., Broda, K., & Gabbay, D. M. (2002). "Neural-Symbolic
      Learning Systems: Foundations and Applications".
"""

import math

import keras
from keras import ops
from typing import List, Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LearnableLogicOperator(keras.layers.Layer):
    """
    Differentiable learnable logic operator layer using fuzzy logic.

    Embeds principles of fuzzy logic into a neural network layer, implementing
    soft differentiable versions of Boolean gates. Inputs are first normalized
    to ``[0, 1]`` via sigmoid, then soft operations are applied:
    ``AND(p,q) = p*q``, ``OR(p,q) = p+q-p*q``, ``XOR(p,q) = p+q-2*p*q``,
    ``NOT(p) = 1-p``, ``NAND(p,q) = 1-p*q``, ``NOR(p,q) = 1-(p+q-p*q)``.
    The output is a weighted combination
    ``Y = sum_i(alpha_i * f_i(X))`` where ``alpha_i = softmax(w_i / T)``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │       LearnableLogicOperator             │
        │                                          │
        │  Input(s): x1, x2                        │
        │         │                                │
        │         ▼                                │
        │  sigmoid(x1), sigmoid(x2)                │
        │         │                                │
        │         ▼                                │
        │  ┌────┬────┬────┬────┬─────┬────┐        │
        │  │AND │ OR │XOR │NOT │NAND │NOR │        │
        │  └─┬──┴─┬──┴─┬──┴─┬──┴──┬──┴─┬──┘        │
        │    │    │    │    │     │    │           │
        │    ▼    ▼    ▼    ▼     ▼    ▼           │
        │  Weighted sum: alpha_i * f_i(p, q)       │
        │         │                                │
        │         ▼                                │
        │  Output (same shape as input)            │
        └──────────────────────────────────────────┘

    :param operation_types: List of operation types. Available:
        ``['and', 'or', 'xor', 'not', 'nand', 'nor']``. If None, all included.
    :type operation_types: Optional[List[str]]
    :param use_temperature: Whether to use temperature scaling for soft selection.
    :type use_temperature: bool
    :param temperature_init: Initial temperature value. Must be positive.
    :type temperature_init: float
    :param operation_initializer: Initializer for operation weights.
    :type operation_initializer: Union[str, keras.initializers.Initializer]
    :param temperature_initializer: Initializer for temperature parameter.
    :type temperature_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
    """

    # Set of all valid op tokens. Binary unless listed in UNARY_OPS.
    VALID_OPS = frozenset({
        'and', 'or', 'xor', 'not', 'nand', 'nor',
        'lukasiewicz_and', 'lukasiewicz_or',
        'godel_and', 'godel_or',
        'implies',
        # M4 (plan_2026-05-13_3a2f1d23):
        'hamacher_and', 'hamacher_or',
        'yager_and', 'yager_or',
        # G4 (plan_2026-05-13_e33114da): additional implications.
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
        super().__init__(**kwargs)

        if selection_mode not in ("global", "per_channel"):
            raise ValueError(
                f"selection_mode must be 'global' or 'per_channel', got "
                f"{selection_mode!r}."
            )
        # M4: Yager p > 0 controls the t-norm sharpness.
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
        # intended path for stacking. Default True preserves legacy behavior
        # (inputs interpreted as raw logits, mapped to [0,1] before fuzzy ops).
        self.apply_sigmoid = apply_sigmoid
        # C4 (plan_2026-05-13_3a2f1d23): when apply_sigmoid=False the layer
        # assumes inputs already lie in [0, 1]. Stacking arithmetic experts
        # upstream violates that. force_clip_when_no_sigmoid=True applies
        # ops.clip(x, 0, 1) defensively.
        self.force_clip_when_no_sigmoid = force_clip_when_no_sigmoid
        self.softplus_temperature = softplus_temperature
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_hard = gumbel_hard
        self.entropy_coefficient = entropy_coefficient
        self.allow_unary_degenerate = allow_unary_degenerate
        self.selection_mode = selection_mode
        self.yager_p = float(yager_p)
        self.num_operations = len(operation_types)
        self._channels = None  # Set in build() for per_channel mode.
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
        Build the layer weights.

        :param input_shape: Shape of the input tensor(s).
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        # A single shape can be a list (e.g., from serialization), but a list
        # of shapes will be a list of lists/tuples/TensorShapes.
        # We differentiate by checking if the list's elements are dimensions (int/None).
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

        # C3 (plan_2026-05-13_3a2f1d23): per-channel mode shapes the weight
        # tensor as (channels, num_operations).
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

        # Create learnable operation selection weights
        self.operation_weights = self.add_weight(
            name="operation_weights",
            shape=weight_shape,
            initializer=self.operation_initializer,
            trainable=True,
        )

        # Create temperature parameter if enabled
        if self.use_temperature:
            if self.softplus_temperature:
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
        return ops.multiply(x1, x2)

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
        return ops.add(ops.add(x1, x2), ops.negative(ops.multiply(x1, x2)))

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
        return ops.subtract(ops.add(x1, x2), ops.multiply(2.0, ops.multiply(x1, x2)))

    def _soft_logic_not(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Soft differentiable NOT: ``1 - p``.

        :param x: Input tensor.
        :type x: keras.KerasTensor
        :return: Result of soft NOT operation.
        :rtype: keras.KerasTensor
        """
        return ops.subtract(1.0, x)

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
        return ops.subtract(1.0, ops.multiply(x1, x2))

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
        or_result = ops.add(ops.add(x1, x2), ops.negative(ops.multiply(x1, x2)))
        return ops.subtract(1.0, or_result)

    # --- Łukasiewicz t-norm / t-conorm -----------------------------------
    def _luk_and(self, x1, x2):
        """Łukasiewicz AND: max(0, p + q - 1)."""
        return ops.maximum(0.0, ops.subtract(ops.add(x1, x2), 1.0))

    def _luk_or(self, x1, x2):
        """Łukasiewicz OR: min(1, p + q)."""
        return ops.minimum(1.0, ops.add(x1, x2))

    # --- Gödel t-norm / t-conorm -----------------------------------------
    def _godel_and(self, x1, x2):
        """Gödel AND: min(p, q)."""
        return ops.minimum(x1, x2)

    def _godel_or(self, x1, x2):
        """Gödel OR: max(p, q)."""
        return ops.maximum(x1, x2)

    # --- Implication family ----------------------------------------------
    def _implies(self, x1, x2):
        """Kleene-Dienes implication: max(1 - p, q)."""
        return ops.maximum(ops.subtract(1.0, x1), x2)

    def _lukasiewicz_implies(self, x1, x2):
        """Łukasiewicz implication: min(1, 1 - p + q)."""
        return ops.minimum(1.0, ops.add(ops.subtract(1.0, x1), x2))

    def _reichenbach_implies(self, x1, x2):
        """Reichenbach (probabilistic) implication: 1 - p + p*q."""
        return ops.add(ops.subtract(1.0, x1), ops.multiply(x1, x2))

    def _goguen_implies(self, x1, x2):
        """Goguen implication: min(1, q / max(p, eps)). Identity when p<=q."""
        # eps prevents 0/0 at p=0 (where formal value should be 1 because
        # ⊥→anything is vacuously true). Clamp p, take ratio, clip to <=1.
        p_safe = ops.maximum(x1, 1e-9)
        return ops.minimum(1.0, ops.divide(x2, p_safe))

    # --- Hamacher / Yager t-norms (M4) -----------------------------------
    # DECISION plan_2026-05-13_e33114da/D-002 — Both Hamacher t-norms have a
    # 0/0 singularity at one corner: AND at (0,0), OR at (1,1). The limit by
    # continuity is 0 for AND and 1 for OR. Prior implementation used
    # asymmetric eps strategies (additive for AND, max-clamp for OR) which
    # gave wrong limits — most visibly, OR(1,1) returned 0 instead of 1.
    # Unified with ops.where: when denom is near-singular, return the
    # mathematical limit; otherwise return the standard ratio.
    _HAMACHER_SINGULAR_EPS = 1e-7

    def _hamacher_and(self, x1, x2):
        """Hamacher product t-norm: p*q / (p + q - p*q). Limit at (0,0) = 0."""
        pq = ops.multiply(x1, x2)
        denom = ops.subtract(ops.add(x1, x2), pq)
        denom_safe = ops.maximum(denom, 1e-9)
        singular = ops.less(denom, self._HAMACHER_SINGULAR_EPS)
        ratio = ops.divide(pq, denom_safe)
        return ops.where(singular, ops.zeros_like(ratio), ratio)

    def _hamacher_or(self, x1, x2):
        """Hamacher sum t-conorm: (p + q - 2 p q) / (1 - p q). Limit at (1,1) = 1."""
        pq = ops.multiply(x1, x2)
        num = ops.subtract(ops.add(x1, x2), ops.multiply(2.0, pq))
        denom = ops.subtract(1.0, pq)
        denom_safe = ops.maximum(denom, 1e-9)
        singular = ops.less(denom, self._HAMACHER_SINGULAR_EPS)
        ratio = ops.divide(num, denom_safe)
        return ops.where(singular, ops.ones_like(ratio), ratio)

    def _yager_and(self, x1, x2):
        """Yager t-norm: 1 - min(1, ((1-p)^w + (1-q)^w)^(1/w))."""
        w = self.yager_p
        a = ops.power(ops.maximum(ops.subtract(1.0, x1), 0.0), w)
        b = ops.power(ops.maximum(ops.subtract(1.0, x2), 0.0), w)
        s = ops.power(ops.add(a, b), 1.0 / w)
        return ops.subtract(1.0, ops.minimum(s, 1.0))

    def _yager_or(self, x1, x2):
        """Yager t-conorm: min(1, (p^w + q^w)^(1/w))."""
        w = self.yager_p
        a = ops.power(ops.maximum(x1, 0.0), w)
        b = ops.power(ops.maximum(x2, 0.0), w)
        s = ops.power(ops.add(a, b), 1.0 / w)
        return ops.minimum(s, 1.0)

    # --- DARTS-style helpers ---------------------------------------------
    def _resolve_temperature(self) -> keras.KerasTensor:
        if self.softplus_temperature:
            return ops.maximum(ops.softplus(self.temperature), 1e-7)
        return ops.maximum(self.temperature, 1e-7)

    def _operation_probs(
        self,
        training: Optional[bool] = None,
        deterministic: bool = False,
    ) -> keras.KerasTensor:
        """
        Compute the operation-selection probability vector.

        # DECISION plan_2026-05-13_3a2f1d23/D-001
        # Canonical Jang (2017) Gumbel-softmax form: softmax((w + g) / T).
        # Previously the implementation computed softmax((w/T) + g), which
        # over-weights the noise term at low temperatures and breaks the
        # Concrete distribution semantics (issue C1 in the residual review).

        # DECISION plan_2026-05-13_e33114da/D-003
        # Gumbel noise is injected ONLY when ``training is True`` (or
        # explicitly via ``deterministic=False`` from a training-path caller).
        # ``model.predict(...)``/``training=False``/``training=None`` skip
        # noise — fixes B2 (non-deterministic inference with gumbel_softmax).

        Args:
            training: Keras training flag. Gumbel noise is injected only when
                ``training is True``. ``None`` and ``False`` are treated as
                inference and skip noise.
            deterministic: Force-skip Gumbel noise regardless of training.
                Used by ``to_symbolic()`` so the printed selection is
                reproducible.
        """
        weights = self.operation_weights
        skip_gumbel = deterministic or (training is not True)

        if self.gumbel_softmax and not skip_gumbel:
            uniform = keras.random.uniform(
                shape=ops.shape(weights), minval=1e-9, maxval=1.0
            )
            gumbel = ops.negative(ops.log(ops.negative(ops.log(uniform))))
            noisy = ops.add(weights, gumbel)
            if self.use_temperature:
                temp = self._resolve_temperature()
                logits = ops.divide(noisy, temp)
            else:
                logits = noisy
            soft = ops.softmax(logits, axis=-1)
            if self.gumbel_hard:
                idx = ops.argmax(soft, axis=-1)
                hard = ops.cast(
                    ops.one_hot(idx, num_classes=self.num_operations), soft.dtype
                )
                return ops.add(soft, ops.stop_gradient(ops.subtract(hard, soft)))
            return soft

        if self.use_temperature:
            temp = self._resolve_temperature()
            logits = ops.divide(weights, temp)
        else:
            logits = weights
        return ops.softmax(logits, axis=-1)

    def _maybe_add_entropy_loss(self, probs: keras.KerasTensor) -> None:
        if self.entropy_coefficient > 0:
            log_p = ops.log(ops.add(probs, 1e-12))
            ent = ops.negative(ops.sum(ops.multiply(probs, log_p)))
            self.add_loss(ops.multiply(self.entropy_coefficient, ent))

    def to_symbolic(self, top_k: int = 1, deterministic: bool = True) -> str:
        """Return a string of the dominant op(s) by selection probability.

        :param deterministic: If True (default), skip Gumbel noise so the
            output is reproducible regardless of ``self.gumbel_softmax``.
            Fixes issue C5 (plan_2026-05-13_3a2f1d23).
        """
        if self.operation_weights is None:
            raise RuntimeError("Layer has not been built yet.")
        probs_arr = ops.convert_to_numpy(
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
        Forward pass through the logic operator.

        :param inputs: Input tensor(s). Single tensor or list of two tensors.
        :type inputs: Union[keras.KerasTensor, List[keras.KerasTensor]]
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Output tensor after applying learnable logic operations.
        :rtype: keras.KerasTensor
        """
        # Input parsing — distinguish unary, single-tensor-supplied-as-list,
        # and binary inputs.
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

        # DECISION plan_2026-05-13_a2b0f17b/D-001 — strict guard against the
        # unary-input footgun (LESSONS L38). When allow_unary_degenerate is
        # False, raise rather than silently rebinding x2 = x1, which makes
        # binary ops like XOR collapse to nonsense (XOR(p,p) should be 0).
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

        # Normalize inputs to [0, 1] range using sigmoid (skip when caller
        # already provides values in [0, 1] — e.g. stacked logic layers).
        if self.apply_sigmoid:
            x1 = ops.sigmoid(x1)
            x2 = ops.sigmoid(x2)
        elif self.force_clip_when_no_sigmoid:
            # C4: defensive clipping when upstream may produce unbounded
            # outputs (e.g. an arithmetic expert above a logic expert).
            x1 = ops.clip(x1, 0.0, 1.0)
            x2 = ops.clip(x2, 0.0, 1.0)

        # Compute operation selection probabilities
        operation_probs = self._operation_probs(training=training)
        self._maybe_add_entropy_loss(operation_probs)

        # Compute all operations
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

        # Vectorized weighted combination.
        if self.selection_mode == "per_channel":
            stacked = ops.stack(operations, axis=-1)  # (..., C, N)
            rank = len(stacked.shape)
            probs_bshape = (1,) * (rank - 2) + (self._channels, self.num_operations)
            weights = ops.reshape(operation_probs, probs_bshape)
            output = ops.sum(ops.multiply(weights, stacked), axis=-1)
        else:
            stacked = ops.stack(operations, axis=0)
            weight_shape = (self.num_operations,) + (1,) * (len(stacked.shape) - 1)
            weights = ops.reshape(operation_probs, weight_shape)
            output = ops.sum(ops.multiply(weights, stacked), axis=0)
        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        :param input_shape: Shape of the input(s).
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        is_list_of_shapes = (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], (int, type(None)))
        )
        if is_list_of_shapes:
            # D9: validate shape consistency for binary inputs.
            if len(input_shape) == 2 and list(input_shape[0]) != list(input_shape[1]):
                raise ValueError(
                    f"Input tensors must have the same shape for binary operations. "
                    f"Got shapes: {input_shape[0]} and {input_shape[1]}"
                )
            return tuple(input_shape[0])
        return tuple(input_shape) if isinstance(input_shape, list) else input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
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
