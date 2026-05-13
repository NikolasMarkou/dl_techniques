"""
 A differentiable, learnable arithmetic operator.

This layer provides a mechanism for a neural network to learn the optimal
arithmetic combination of its inputs, moving beyond fixed operations like
addition or concatenation. It is inspired by techniques in Neural
Architecture Search (NAS), where the choice of operation is made a
learnable part of the network itself.

Architecture:
    The core principle is to create a "soft," differentiable selection over
    a predefined set of primitive arithmetic operations (e.g., add,
    multiply, max). Instead of making a discrete, non-differentiable
    choice of one operation, this layer computes the result of *all*
    candidate operations and then combines them through a weighted sum.

    The weights for this combination are determined by a learnable parameter
    vector, where each element corresponds to an operation. This vector is
    passed through a softmax function to produce a probability
    distribution, representing the "importance" of each operation.

Foundational Mathematics:
    The selection of operations is governed by the softmax function, often
    with a temperature parameter `T`. Given a vector of learnable weights
    `w` (one `w_i` for each operation `f_i`), the probability `p_i` for
    selecting the i-th operation is:

        p_i = exp(w_i / T) / sum_j(exp(w_j / T))

    The temperature `T` is a learnable parameter that controls the sharpness
    of the probability distribution. As `T -> 0`, the distribution
    approaches a one-hot vector (a "hard" selection), concentrating all
    probability on a single operation. As `T -> infinity`, it approaches a
    uniform distribution, treating all operations equally. This allows the
    model to explore different operations during early training phases and
    converge to a more decisive choice later.

    The final output `Y` is a convex combination of the results of each
    operation `f_i(X)` applied to the input tensor(s) `X`, scaled by a
    learnable factor `s`:

        Y = s * sum_i(p_i * f_i(X))

    This formulation makes the entire process end-to-end differentiable.
    Gradients can flow back through the weighted sum and the softmax
    function to update the operation weights `w`, the temperature `T`, and
    the scaling factor `s`, allowing the network to learn the most
    suitable arithmetic transformation for the task at hand.

References:
    - The concept of a continuous relaxation over a discrete set of
      operations is a cornerstone of differentiable NAS, famously
      popularized by the DARTS framework.
      Liu, H., Simonyan, K., & Yang, Y. (2018). "DARTS: Differentiable
      Architecture Search".

    - The use of temperature to control the sharpness of a softmax
      distribution is a widely used technique, notably in knowledge
      distillation to create "soft targets."
      Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the
      Knowledge in a Neural Network".
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
class LearnableArithmeticOperator(keras.layers.Layer):
    """
    Differentiable learnable arithmetic operator layer.

    Implements a soft selection over a set of primitive arithmetic operations
    (add, multiply, subtract, divide, power, max, min) using learnable weights
    passed through a temperature-scaled softmax:
    ``p_i = exp(w_i / T) / sum_j(exp(w_j / T))``. The output is a convex
    combination ``Y = s * sum_i(p_i * f_i(X))`` where ``s`` is a learnable
    scaling factor. This formulation makes the operation selection end-to-end
    differentiable, inspired by DARTS-style continuous relaxation.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────┐
        │        LearnableArithmeticOperator               │
        │                                                  │
        │  Input(s): x1, x2                                │
        │         │                                        │
        │         ▼                                        │
        │  ┌─────┬─────────┬──────┬───────┬─────┬───────┐  │
        │  │ add │multiply │ sub  │divide │power│max/min│  │
        │  └──┬──┴────┬────┴───┬──┴───┬───┴──┬──┴──┬────┘  │
        │     │       │        │      │      │     │       │
        │     ▼       ▼        ▼      ▼      ▼     ▼       │
        │  Weighted sum: p_i * f_i(x1, x2)                 │
        │         │                                        │
        │         ├──► * scaling_factor                    │
        │         ▼                                        │
        │  Output (same shape as input)                    │
        └──────────────────────────────────────────────────┘

    :param operation_types: List of operation types. Available:
        ``['add', 'multiply', 'subtract', 'divide', 'power', 'max', 'min']``.
        If None, all operations are included.
    :type operation_types: Optional[List[str]]
    :param use_temperature: Whether to use temperature scaling for soft selection.
    :type use_temperature: bool
    :param temperature_init: Initial temperature value. Must be positive.
    :type temperature_init: float
    :param use_scaling: Whether to use a learnable output scaling factor.
    :type use_scaling: bool
    :param scaling_init: Initial scaling factor value. Must be positive.
    :type scaling_init: float
    :param operation_initializer: Initializer for operation weights.
    :type operation_initializer: Union[str, keras.initializers.Initializer]
    :param temperature_initializer: Initializer for temperature parameter.
    :type temperature_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param scaling_initializer: Initializer for scaling factor.
    :type scaling_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param epsilon: Small constant for numerical stability in division.
    :type epsilon: float
    :param power_clip_range: ``(min_base, max_base)`` for clipping in power operations.
    :type power_clip_range: Tuple[float, float]
    :param exponent_clip_range: ``(min_exp, max_exp)`` for clipping exponents.
    :type exponent_clip_range: Tuple[float, float]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
    """

    def __init__(
            self,
            operation_types: Optional[List[str]] = None,
            use_temperature: bool = True,
            temperature_init: float = 1.0,
            use_scaling: bool = True,
            scaling_init: float = 1.0,
            operation_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            temperature_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            scaling_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            epsilon: float = 1e-7,
            power_clip_range: Tuple[float, float] = (1e-7, 10.0),
            exponent_clip_range: Tuple[float, float] = (-2.0, 2.0),
            softplus_temperature: bool = False,
            safe_divide_mode: str = "hard_clamp",
            gumbel_softmax: bool = False,
            gumbel_hard: bool = False,
            entropy_coefficient: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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
        Build the layer weights.

        :param input_shape: Shape of the input tensor(s).
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        # Validate input shapes for binary operations
        if isinstance(input_shape, list):
            # To distinguish a list of shapes from a single shape deserialized as a list (e.g., [None, 32]),
            # we check if the first element is iterable (like a tuple, list, or TensorShape).
            if len(input_shape) == 2 and input_shape[0] is not None and hasattr(input_shape[0], '__iter__'):
                if list(input_shape[0]) != list(input_shape[1]):
                    raise ValueError(
                        f"Input tensors must have the same shape for binary operations. "
                        f"Got shapes: {input_shape[0]} and {input_shape[1]}"
                    )

        # Create learnable operation selection weights
        self.operation_weights = self.add_weight(
            name="operation_weights",
            shape=(self.num_operations,),
            initializer=self.operation_initializer,
            trainable=True,
        )

        # Create temperature parameter if enabled. If softplus_temperature is
        # True, the stored weight is the *raw* pre-softplus value; init the
        # raw value so that softplus(raw) == temperature_init.
        if self.use_temperature:
            if self.softplus_temperature:
                # softplus_inv(y) = log(exp(y) - 1); for y >> 0, ~= y.
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

        # Create scaling factor if enabled
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
        Safe division.

        Two modes (selected by ``safe_divide_mode`` constructor arg):

        - ``'hard_clamp'`` (default, legacy): clamp ``|x2|`` to be at least
          ``epsilon``. Forward-bounded but produces a step in the gradient at
          ``x2 = 0`` and grows like ``1/epsilon`` for very small denominators.
        - ``'smooth'``: use the smooth approximation
          ``x1 * x2 / (x2**2 + epsilon**2)``. Equivalent to ``x1/x2`` whenever
          ``|x2| >> epsilon``; bounded gradient ``|d/dx2| <= |x1| / (2 * epsilon)``
          everywhere, including at ``x2 == 0``.

        **C2 clarification (plan_2026-05-13_3a2f1d23):** In ``'smooth'`` mode,
        ``f(x1, 0) = 0`` by design (not the mathematical limit of x1/x2). This
        is a deliberate trade-off — see LESSONS L44 and the D-001 anchor of
        plan_2026-05-13_a2b0f17b on line 318 — that exchanges exact-divide
        semantics near zero for a globally bounded gradient that makes the
        operator trainable end-to-end. If you need exact divide semantics,
        use ``safe_divide_mode='hard_clamp'``.

        :param x1: Numerator tensor.
        :param x2: Denominator tensor.
        :return: Result of the safe division.
        """
        if self.safe_divide_mode == "smooth":
            # DECISION plan_2026-05-13_a2b0f17b/D-001 — bounded-gradient
            # smooth division. Far from zero this is x1/x2 to within
            # O((eps/x2)^2). At x2=0 the value is 0 and the gradient wrt x2 is
            # bounded by |x1| / (2 * eps).
            denom = ops.add(ops.square(x2), ops.cast(self.epsilon ** 2, x2.dtype))
            return ops.divide(ops.multiply(x1, x2), denom)

        # Legacy hard_clamp behavior — bit-exact with prior versions.
        sign_x2 = ops.sign(x2)
        sign_x2 = ops.where(ops.equal(sign_x2, 0.0), ops.ones_like(sign_x2), sign_x2)
        safe_x2 = sign_x2 * ops.maximum(ops.abs(x2), self.epsilon)
        return ops.divide(x1, safe_x2)

    def _safe_power(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Safe power operation with clipping for numerical stability.

        Sign of the base is preserved via the standard ``sign(x1) * |x1|^x2``
        decomposition. For non-integer exponents, ``power(negative, frac)`` is
        complex-valued; this implementation returns the **real** restriction
        ``sign(x1) * |x1|^x2`` which is well-defined for any real ``x2`` and
        coincides with ``power(x1, x2)`` whenever ``x1 >= 0`` or ``x2`` is an
        integer.

        :param x1: Base tensor.
        :param x2: Exponent tensor.
        :return: Result of safe power operation, sign-preserving.
        """
        # DECISION plan_2026-05-13_a2b0f17b/D-001 — real restriction of complex
        # power: Re((-|x|)^y) = cos(pi*y) * |x|^y. Equals x^y for non-negative
        # x; reproduces +1 for even-integer y, -1 for odd-integer y, 0 for
        # half-integer y on negative bases. Old impl dropped sign entirely.
        x1_abs_safe = ops.clip(
            ops.abs(x1), self.power_clip_range[0], self.power_clip_range[1]
        )
        x2_safe = ops.clip(x2, self.exponent_clip_range[0], self.exponent_clip_range[1])
        magnitude = ops.power(x1_abs_safe, x2_safe)
        # sign component: +1 for non-negative bases, cos(pi*y) for negative.
        is_negative = ops.cast(ops.less(x1, 0.0), x1.dtype)
        sign_component = (
            ops.cos(ops.multiply(math.pi, x2_safe)) * is_negative
            + (ops.cast(1.0, x1.dtype) - is_negative)
        )
        return ops.multiply(sign_component, magnitude)

    def _soft_max(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Element-wise maximum operation.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor.
        :type x2: keras.KerasTensor
        :return: Element-wise maximum of the inputs.
        :rtype: keras.KerasTensor
        """
        return ops.maximum(x1, x2)

    def _soft_min(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        Element-wise minimum operation.

        :param x1: First input tensor.
        :type x1: keras.KerasTensor
        :param x2: Second input tensor.
        :type x2: keras.KerasTensor
        :return: Element-wise minimum of the inputs.
        :rtype: keras.KerasTensor
        """
        return ops.minimum(x1, x2)

    def _resolve_temperature(self) -> keras.KerasTensor:
        """Return the effective positive temperature."""
        if self.softplus_temperature:
            # softplus(raw) is always > 0; clamp epsilon as a final guard.
            return ops.maximum(ops.softplus(self.temperature), 1e-7)
        return ops.maximum(self.temperature, 1e-7)

    def _operation_probs(self, deterministic: bool = False) -> keras.KerasTensor:
        """
        Compute the operation-selection probability vector.

        Honors ``use_temperature`` and ``gumbel_softmax`` modes.

        # DECISION plan_2026-05-13_3a2f1d23/D-001
        # Canonical Jang (2017) Gumbel-softmax form: softmax((w + g) / T).
        # Previously the implementation computed softmax((w/T) + g), which
        # over-weights the noise term at low temperatures and breaks the
        # Concrete distribution semantics (issue C1 in the residual review).

        Args:
            deterministic: If True, skip Gumbel noise injection regardless of
                ``self.gumbel_softmax``. Used by ``to_symbolic()`` so that the
                printed operator selection is reproducible during training.
        """
        weights = self.operation_weights

        if self.gumbel_softmax and not deterministic:
            # Gumbel(0,1) = -log(-log(U(0,1))). Manual implementation since
            # keras.ops doesn't expose it directly.
            uniform = keras.random.uniform(
                shape=ops.shape(weights), minval=1e-9, maxval=1.0
            )
            gumbel = ops.negative(ops.log(ops.negative(ops.log(uniform))))
            # Canonical form: (w + g) / T then softmax (NOT softmax(w/T) + g).
            noisy = ops.add(weights, gumbel)
            if self.use_temperature:
                temp = self._resolve_temperature()
                logits = ops.divide(noisy, temp)
            else:
                logits = noisy
            soft = ops.softmax(logits)
            if self.gumbel_hard:
                # Straight-through estimator: forward-pass uses the one-hot,
                # backward-pass uses the soft sample.
                idx = ops.argmax(soft)
                hard = ops.one_hot(idx, num_classes=self.num_operations)
                hard = ops.cast(hard, soft.dtype)
                return ops.add(soft, ops.stop_gradient(ops.subtract(hard, soft)))
            return soft

        # No gumbel (or deterministic=True): plain temperature-scaled softmax.
        if self.use_temperature:
            temp = self._resolve_temperature()
            logits = ops.divide(weights, temp)
        else:
            logits = weights
        return ops.softmax(logits)

    def _maybe_add_entropy_loss(
        self, probs: keras.KerasTensor
    ) -> None:
        if self.entropy_coefficient > 0:
            log_p = ops.log(ops.add(probs, 1e-12))
            ent = ops.negative(ops.sum(ops.multiply(probs, log_p)))
            # Penalize HIGH entropy (push toward sharp selection).
            self.add_loss(ops.multiply(self.entropy_coefficient, ent))

    def to_symbolic(self, top_k: int = 1, deterministic: bool = True) -> str:
        """
        Return a human-readable string of the dominant op(s) post-training.

        :param top_k: Return the top-k operations by softmax probability.
        :param deterministic: If True (default), skip Gumbel noise so the
            output is reproducible regardless of ``self.gumbel_softmax``.
            Set False only if you explicitly want sample variability.
            Fixes issue C5 (plan_2026-05-13_3a2f1d23).
        :return: Comma-separated operation names ranked by probability,
            optionally with their probabilities in parentheses.
        """
        if self.operation_weights is None:
            raise RuntimeError("Layer has not been built yet.")
        probs = ops.convert_to_numpy(
            self._operation_probs(deterministic=deterministic)
        ).tolist()
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
        Forward pass through the arithmetic operator.

        :param inputs: Input tensor(s). Single tensor or list of two tensors.
        :type inputs: Union[keras.KerasTensor, List[keras.KerasTensor]]
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Output tensor after applying learnable arithmetic operations.
        :rtype: keras.KerasTensor
        """
        # Handle input parsing
        if isinstance(inputs, list):
            if len(inputs) == 2:
                x1, x2 = inputs
            elif len(inputs) == 1:
                x1 = inputs[0]
                x2 = inputs[0]  # Use same input for unary operations
            else:
                raise ValueError(f"Expected 1 or 2 inputs, got {len(inputs)}")
        else:
            x1 = inputs
            x2 = inputs

        # Compute operation selection probabilities
        operation_probs = self._operation_probs()
        self._maybe_add_entropy_loss(operation_probs)

        # Compute all operations
        operations = []
        for op_type in self.operation_types:
            if op_type == 'add':
                result = ops.add(x1, x2)
            elif op_type == 'multiply':
                result = ops.multiply(x1, x2)
            elif op_type == 'subtract':
                result = ops.subtract(x1, x2)
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

        # Weighted combination of operations — vectorized stack-and-sum.
        # `stacked` has shape (N, ...x.shape...). `weights` is reshaped to
        # broadcast against the leading op axis.
        stacked = ops.stack(operations, axis=0)
        n = self.num_operations
        weight_shape = (n,) + (1,) * (len(stacked.shape) - 1)
        weights = ops.reshape(operation_probs, weight_shape)
        output = ops.sum(ops.multiply(weights, stacked), axis=0)

        # Apply scaling factor if enabled
        if self.use_scaling:
            # Clamp scaling factor to prevent numerical issues
            scale = ops.maximum(ops.abs(self.scaling_factor), 1e-7)
            output = ops.multiply(output, scale)

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
        # Distinguish [(s1,), (s2,)] (list of two shapes) from [None, 32] (one
        # shape deserialized as a list). Single shapes have int/None elements.
        is_list_of_shapes = (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], (int, type(None)))
        )
        if is_list_of_shapes:
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
        })
        return config

# ---------------------------------------------------------------------