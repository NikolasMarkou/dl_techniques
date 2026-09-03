"""``BitLinear``, a bit-quantized dense layer for quantization-aware training.

The layer replaces the plain matmul ``y = matmul(x, W)`` with a quantized
version: both the input and the kernel are scaled and quantized on every
forward pass, and the result is rescaled by the inverse of both scales.
Scaling never reduces over the batch axis, so the layer computes the same
function at any batch size. Gradients cross the non-differentiable round
step through a straight-through estimator, so the underlying full-precision
weights still train while the forward pass runs on quantized values.

    y = matmul(round(x / gamma_x), round(W / gamma_w)) * gamma_x * gamma_w

By default weights use 1.58-bit (ternary) quantization and drop the bias,
matching the BitNet paper; activations use 8-bit quantization. See
``weight_bits`` and the class docstring for other bit widths.

References:
    - Wang et al., 2024. The Era of 1-bit LLMs: All Large Language Models
      are in 1.58 Bits. (https://arxiv.org/abs/2402.17764)
    - Bengio et al., 2013. Estimating or Propagating Gradients Through
      Stochastic Neurons for Conditional Computation.
      (https://arxiv.org/abs/1308.3432)
"""

import keras
from typing import Optional, Dict, Any, Callable, Union, Tuple, Sequence

from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.bitlinear_layer")
class BitLinear(keras.layers.Layer):
    """Bit-aware linear layer for quantization-aware training.

    Implements a quantization-aware dense transformation that scales and
    quantizes both weights and activations to low-bit representations during
    the forward pass, while maintaining full-precision gradients through a
    Straight-Through Estimator (STE). The quantization pipeline is:
    ``T_scaled = T * (Q_max / gamma)``, ``T_quant = clip(round(T_scaled), Q_min, Q_max)``,
    ``y = matmul(x_quant, W_quant) / (alpha_x * alpha_W)``. For 1.58-bit
    weights the quantized range is the ternary set ``{-1, 0, 1}``.

    Activation ``gamma`` is reduced over the **feature** axis (one factor per
    token), and weight ``gamma`` over the **input** axis when
    ``weight_per_channel`` is set (one factor per output unit) or over the
    whole kernel otherwise. Neither statistic crosses the batch axis, so the
    layer computes the same function at every batch size.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────┐    ┌──────────────────────────────┐
        │  Input (..., D_in)           │    │  Kernel (D_in, units)        │
        └──────────────┬───────────────┘    └──────────────┬───────────────┘
                       ▼                                   │
        ┌──────────────────────────────┐                   │
        │  [Optional] LayerNorm        │                   │
        └──────────────┬───────────────┘                   ▼
                       ▼                    ┌──────────────────────────────┐
        ┌──────────────────────────────┐    │  Scale + Quantize Weights    │
        │  Scale + Quantize Activations│    │  alpha_w = Q_max / gamma(W)  │
        │  alpha_x = Q_max / gamma(x)  │    │    -> (1, units) or scalar   │
        │    -> (..., 1) per token     │    │  W_q = clip(round(W*alpha))  │
        │  x_q = clip(round(x*alpha))  │    └──────────────┬───────────────┘
        └──────────────┬───────────────┘                   │
                       └─────────────┬─────────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │  matmul(x_q, W_q)            │
                      │  / (alpha_x * alpha_w)       │
                      │  + bias (optional)           │
                      │  activation (optional)       │
                      └──────────────┬───────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │  Output (..., units)         │
                      └──────────────────────────────┘

    :param units: Positive integer, dimensionality of the output space.
    :type units: int
    :param weight_bits: Number of bits for weight quantization, or an explicit
        ``(min, max)`` range. Default is 1.58 bits (ternary weights). ``1.58``
        and ``2`` both map to the ternary range ``{-1, 0, 1}``; ``1`` is a
        distinct **binary** path producing ``{-1, +1}`` via ``sign``.
    :type weight_bits: Union[float, int, Tuple[float, float]]
    :param activation_bits: Number of bits for activation quantization or
        explicit range. Default is 8 bits.
    :type activation_bits: Union[float, int, Tuple[float, float]]
    :param weight_scale_method: Method to compute weight scaling factor.
        One of ``"abs_max"``, ``"abs_mean"``, ``"abs_median"``. Defaults to
        ``"abs_mean"``, the value advocated by the BitNet paper.
    :type weight_scale_method: str
    :param activation_scale_method: Method to compute activation scaling
        factor. One of ``"abs_max"``, ``"abs_mean"``, ``"abs_median"``.
    :type activation_scale_method: str
    :param weight_per_channel: If ``True``, compute one weight scale per
        output unit (shape ``(1, units)``); if ``False``, one scale for the
        whole kernel, which is the BitNet formulation and the default.
    :type weight_per_channel: bool
    :param quantization_method: Quantization strategy. One of
        ``"round_clip"``, ``"stochastic"``. Stochastic rounding is a
        training-time regulariser and falls back to deterministic rounding
        whenever ``training`` is not true, so inference stays deterministic.
    :type quantization_method: str
    :param activation: Activation applied to the de-quantized output, after
        the bias. Accepts anything ``keras.activations.get`` accepts. Defaults
        to ``None``, which ``keras.activations.get`` resolves to the identity,
        matching ``keras.layers.Dense``.
    :type activation: Optional[Union[str, Callable]]
    :param use_bias: Whether the layer uses a bias vector. Defaults to
        ``False``, following BitNet, which drops the bias in its quantized
        projections -- unlike ``keras.layers.Dense``, whose default is
        ``True``. See the migration note below.
    :type use_bias: bool
    :param use_input_norm: Whether to apply layer normalization to inputs
        before quantization.
    :type use_input_norm: bool
    :param ste_lambda: Scaling factor for the straight-through estimator
        gradient. Affects the backward pass only; the forward value is the
        quantized tensor for every value of this parameter.
    :type ste_lambda: float
    :param ste_clip_gradient: If ``True`` (default), the straight-through
        gradient is masked outside the representable range, so saturated
        values do not receive gradient for a forward output that is locally
        constant. This is the canonical STE of Bengio et al.
    :type ste_clip_gradient: bool
    :param epsilon: Small constant for numerical stability, used as a floor
        on ``gamma``.
    :type epsilon: float
    :param norm_epsilon: Epsilon of the optional input ``LayerNormalization``.
        Named separately from ``epsilon`` because the two guard unrelated
        denominators.
    :type norm_epsilon: float
    :param seed: Seed for the stochastic-rounding random draws.
    :type seed: Optional[int]
    :param kernel_initializer: Initializer for the kernel weights matrix.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias vector.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer for bias vector.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kernel_constraint: Constraint on the latent full-precision kernel.
        BitNet-style QAT commonly clips these shadow weights to ``[-1, 1]``.
    :type kernel_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to ``keras.layers.Layer``.

    Input shape:
        N-D tensor ``(batch_size, ..., input_dim)``.

    Output shape:
        N-D tensor ``(batch_size, ..., units)``.

    .. note::

        **Migration, 2026-09-02.** ``use_bias`` changed default from ``True``
        to ``False`` so the layer matches the BitNet formulation it cites.
        Existing archives are unaffected: ``use_bias`` has always been written
        into ``get_config``, so a saved model rebuilds with whatever it was
        constructed with. Code that constructed ``BitLinear(units=...)``
        without naming ``use_bias`` and relied on the bias must now pass
        ``use_bias=True`` explicitly; a layer built the new way has one
        trainable variable instead of two, so a weight file written before
        this change will refuse to load into it rather than load silently.
    """

    #: Accepted values for the two ``*_scale_method`` parameters.
    VALID_SCALE_METHODS = ("abs_max", "abs_mean", "abs_median")

    #: Accepted values for ``quantization_method``.
    VALID_QUANT_METHODS = ("round_clip", "stochastic")

    def __init__(
        self,
        units: int,
        weight_bits: Union[float, int, Tuple[float, float]] = 1.58,
        activation_bits: Union[float, int, Tuple[float, float]] = 8,
        weight_scale_method: str = "abs_mean",
        activation_scale_method: str = "abs_max",
        weight_per_channel: bool = False,
        quantization_method: str = "round_clip",
        activation: Optional[Union[str, Callable]] = None,
        use_bias: bool = False,
        use_input_norm: bool = False,
        ste_lambda: float = 1.0,
        ste_clip_gradient: bool = True,
        epsilon: float = 1e-5,
        norm_epsilon: float = 1e-6,
        seed: Optional[int] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_constraint: Optional[keras.constraints.Constraint] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the BitLinear layer."""
        super().__init__(**kwargs)

        # Validate units parameter
        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")

        # Validate scale methods
        if weight_scale_method not in self.VALID_SCALE_METHODS:
            raise ValueError(
                f"weight_scale_method must be one of {list(self.VALID_SCALE_METHODS)}, "
                f"got {weight_scale_method}"
            )
        if activation_scale_method not in self.VALID_SCALE_METHODS:
            raise ValueError(
                f"activation_scale_method must be one of {list(self.VALID_SCALE_METHODS)}, "
                f"got {activation_scale_method}"
            )

        # Validate quantization method
        if quantization_method not in self.VALID_QUANT_METHODS:
            raise ValueError(
                f"quantization_method must be one of {list(self.VALID_QUANT_METHODS)}, "
                f"got {quantization_method}"
            )

        # Validate numerical parameters
        if ste_lambda <= 0:
            raise ValueError(f"ste_lambda must be positive, got {ste_lambda}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if norm_epsilon <= 0:
            raise ValueError(f"norm_epsilon must be positive, got {norm_epsilon}")

        # Store configuration
        self.units = units
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_scale_method = weight_scale_method
        self.activation_scale_method = activation_scale_method
        self.weight_per_channel = weight_per_channel
        self.quantization_method = quantization_method
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_input_norm = use_input_norm
        self.ste_lambda = ste_lambda
        self.ste_clip_gradient = ste_clip_gradient
        self.epsilon = epsilon
        self.norm_epsilon = norm_epsilon
        self.seed = seed

        # Store initializers, regularizers and constraints
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)

        # ``*_is_binary`` selects the ``sign`` path, needed to reach the
        # two-valued set {-1, +1}: rounding into [-1, 1] can land on zero.
        self.weight_range, self._weight_is_binary = self._parse_bits(
            weight_bits, "weight_bits"
        )
        self.activation_range, self._activation_is_binary = self._parse_bits(
            activation_bits, "activation_bits"
        )

        # A traced/JIT-compiled forward pass needs a serializable RNG.
        self.seed_generator = keras.random.SeedGenerator(seed)

        # The transformation is elementwise on the feature axis and every
        # statistic is per token, so a mask passes through unchanged.
        self.supports_masking = True

        # Optional input normalization sub-layer (created in __init__, built in build)
        if self.use_input_norm:
            self.input_norm = keras.layers.LayerNormalization(
                axis=-1,
                epsilon=self.norm_epsilon,
                center=True,
                scale=True,
                name="input_norm",
            )
        else:
            self.input_norm = None

        # Layer weights (created in build)
        self.kernel = None
        self.bias = None

    @staticmethod
    def _parse_bits(
        bits: Union[float, int, Sequence[float]],
        name: str
    ) -> Tuple[Tuple[float, float], bool]:
        """Convert a bit specification to a quantization range.

        A JSON round trip turns a ``tuple`` config value into a ``list``, so
        both are accepted here; rejecting the list would make every
        explicit-range model unloadable.

        :param bits: Number of bits, or an explicit ``(min, max)`` range.
        :type bits: Union[float, int, Sequence[float]]
        :param name: Parameter name, used in error messages.
        :type name: str
        :return: ``((min_value, max_value), is_binary)``.
        :rtype: Tuple[Tuple[float, float], bool]
        :raises ValueError: If the specification is not a positive number or a
            two-element increasing range.
        """
        if isinstance(bits, (tuple, list)):
            if len(bits) != 2:
                raise ValueError(
                    f"{name} given as an explicit range must have exactly 2 "
                    f"elements, got {bits}"
                )
            low, high = float(bits[0]), float(bits[1])
            if not low < high:
                raise ValueError(
                    f"{name} range must satisfy min < max, got {bits}"
                )
            return (low, high), False

        # ``bool`` is a subclass of ``int``; ``weight_bits=True`` is a mistake.
        if isinstance(bits, bool) or not isinstance(bits, (int, float)):
            raise ValueError(f"Invalid bit specification for {name}: {bits!r}")

        if bits <= 0:
            raise ValueError(f"{name} must be positive, got {bits}")

        if bits == 1:
            # True binary {-1, +1}, taken by the sign path.
            return (-1.0, 1.0), True
        if bits < 3:
            # 1.58 and 2 bits alike -> ternary {-1, 0, 1}.
            return (-1.0, 1.0), False

        # n-bit quantization: [-(2^(n-1)-1), 2^(n-1)-1]
        n = int(bits)
        max_val = float(2 ** (n - 1) - 1)
        return (-max_val, max_val), False

    def _compute_gamma(
        self,
        tensor: keras.KerasTensor,
        method: str,
        reduce_axis: Optional[int]
    ) -> keras.KerasTensor:
        """Compute the magnitude statistic ``gamma`` used to set the scale.

        The scale is ``Q_max / gamma``; this helper returns ``gamma`` itself so
        that callers can divide by it rather than materialize a reciprocal that
        overflows float16 for small ``gamma``.

        ``gamma`` is returned under ``stop_gradient``: it is a differentiable
        function of the tensor, and letting the gradient flow through it adds a
        spurious ``-T/gamma**2 * dgamma/dT`` term. For ``abs_max`` that entire
        term is routed to whichever single element attained the maximum.

        :param tensor: Tensor to compute the statistic for.
        :type tensor: keras.KerasTensor
        :param method: One of ``"abs_max"``, ``"abs_mean"``, ``"abs_median"``.
        :type method: str
        :param reduce_axis: Axis to reduce over, keeping dimensions, or ``None``
            to reduce the whole tensor to a scalar.
        :type reduce_axis: Optional[int]
        :return: Magnitude statistic, floored at ``epsilon``.
        :rtype: keras.KerasTensor
        """
        abs_tensor = keras.ops.abs(tensor)

        if reduce_axis is None:
            if method == "abs_max":
                gamma = keras.ops.max(abs_tensor)
            elif method == "abs_mean":
                gamma = keras.ops.mean(abs_tensor)
            elif method == "abs_median":
                gamma = keras.ops.quantile(
                    keras.ops.reshape(abs_tensor, (-1,)), 0.5
                )
            else:
                raise ValueError(f"Unknown scaling method: {method}")
        else:
            if method == "abs_max":
                gamma = keras.ops.max(abs_tensor, axis=reduce_axis, keepdims=True)
            elif method == "abs_mean":
                gamma = keras.ops.mean(abs_tensor, axis=reduce_axis, keepdims=True)
            elif method == "abs_median":
                gamma = keras.ops.quantile(
                    abs_tensor, 0.5, axis=reduce_axis, keepdims=True
                )
            else:
                raise ValueError(f"Unknown scaling method: {method}")

        # Prevent division by zero.
        gamma = keras.ops.maximum(gamma, self.epsilon)

        return keras.ops.stop_gradient(gamma)

    def _quantize_tensor(
        self,
        tensor: keras.KerasTensor,
        target_range: Tuple[float, float],
        is_binary: bool = False,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply quantization with a straight-through estimator.

        :param tensor: Scaled tensor to quantize.
        :type tensor: keras.KerasTensor
        :param target_range: Target quantization range.
        :type target_range: Tuple[float, float]
        :param is_binary: If ``True``, map to ``{-1, +1}`` with ``sign``
            (ties resolved to ``+1``) instead of rounding.
        :type is_binary: bool
        :param training: Whether the call is in training mode. Stochastic
            rounding only applies when this is true.
        :type training: Optional[bool]
        :return: Quantized tensor whose forward value is exactly the quantized
            grid and whose gradient is the straight-through pass-through.
        :rtype: keras.KerasTensor
        """
        min_val, max_val = target_range

        if is_binary:
            # sign() is 0 at 0, which would produce a ternary set; break the
            # tie towards +1 so the output really is two-valued.
            quantized = keras.ops.sign(tensor) + keras.ops.cast(
                keras.ops.equal(tensor, 0.0), tensor.dtype
            )
        else:
            if self.quantization_method == "stochastic" and training:
                floor_val = keras.ops.floor(tensor)
                prob_ceil = tensor - floor_val
                # Draw in float32: a float16 uniform has ~1e-3 resolution,
                # which would coarsely discretize the rounding probability.
                random_uniform = keras.random.uniform(
                    keras.ops.shape(tensor),
                    minval=0.0,
                    maxval=1.0,
                    dtype="float32",
                    seed=self.seed_generator,
                )
                random_uniform = keras.ops.cast(random_uniform, tensor.dtype)
                rounded = keras.ops.where(
                    random_uniform < prob_ceil,
                    floor_val + 1.0,
                    floor_val
                )
            else:
                rounded = keras.ops.round(tensor)

            quantized = keras.ops.clip(rounded, min_val, max_val)

        # Straight-through estimator. Keeping ``stop_gradient`` around the whole
        # residual makes the forward value exactly ``quantized`` for every
        # ``ste_lambda`` and the gradient exactly ``ste_lambda``.
        passthrough = self.ste_lambda * tensor
        if self.ste_clip_gradient:
            in_range = keras.ops.cast(
                keras.ops.logical_and(
                    keras.ops.greater_equal(tensor, min_val),
                    keras.ops.less_equal(tensor, max_val),
                ),
                tensor.dtype,
            )
            passthrough = passthrough * in_range

        return keras.ops.stop_gradient(quantized - passthrough) + passthrough

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights based on input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Extract input dimension
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the input shape must be defined. "
                f"Received input_shape={input_shape}"
            )

        # Create kernel weights
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        # Create bias weights if needed
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )

        # Build the optional input normalization sub-layer
        if self.use_input_norm and self.input_norm is not None:
            self.input_norm.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Perform quantized linear transformation.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean flag indicating training mode.
        :type training: Optional[bool]
        :return: Transformed output tensor.
        :rtype: keras.KerasTensor
        """
        # Apply input normalization if configured
        if self.use_input_norm and self.input_norm is not None:
            x = self.input_norm(inputs, training=training)
        else:
            x = inputs

        # Activation statistic per token: reduce the feature axis, never the
        # batch axis, so the quantization of one sample is independent of the
        # others and batch_size=1 is the same function as batch_size=N.
        activation_max = max(abs(v) for v in self.activation_range)
        gamma_x = self._compute_gamma(x, self.activation_scale_method, -1)
        gamma_x = keras.ops.cast(gamma_x, x.dtype)
        x_quantized = self._quantize_tensor(
            x / gamma_x * activation_max,
            self.activation_range,
            is_binary=self._activation_is_binary,
            training=training,
        )

        # Weight statistic per output channel (axis 0) or per tensor.
        weight_max = max(abs(v) for v in self.weight_range)
        weight_axis = 0 if self.weight_per_channel else None
        kernel = keras.ops.cast(self.kernel, x.dtype)
        gamma_w = self._compute_gamma(kernel, self.weight_scale_method, weight_axis)
        w_quantized = self._quantize_tensor(
            kernel / gamma_w * weight_max,
            self.weight_range,
            is_binary=self._weight_is_binary,
            training=training,
        )

        # gamma_x has shape (..., 1) and broadcasts over the output rows;
        # gamma_w has shape (1, units) or is a scalar and broadcasts over the
        # output columns. Neither axis is contracted by the matmul, so the
        # de-quantization folds into the output and the matmul operands stay
        # on the integer grid.
        output = keras.ops.matmul(x_quantized, w_quantized)
        output = output * (gamma_x / activation_max) * (gamma_w / weight_max)

        # Add bias if present
        if self.use_bias and self.bias is not None:
            output = output + keras.ops.cast(self.bias, output.dtype)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape tuple of output.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "weight_scale_method": self.weight_scale_method,
            "activation_scale_method": self.activation_scale_method,
            "weight_per_channel": self.weight_per_channel,
            "quantization_method": self.quantization_method,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "use_input_norm": self.use_input_norm,
            "ste_lambda": self.ste_lambda,
            "ste_clip_gradient": self.ste_clip_gradient,
            "epsilon": self.epsilon,
            "norm_epsilon": self.norm_epsilon,
            "seed": self.seed,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitLinear":
        """Create a layer instance from configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: New BitLinear instance.
        :rtype: BitLinear
        """
        config = dict(config)

        for key in ("kernel_initializer", "bias_initializer"):
            if isinstance(config.get(key), dict):
                config[key] = keras.initializers.deserialize(config[key])
        for key in ("kernel_regularizer", "bias_regularizer"):
            if isinstance(config.get(key), dict):
                config[key] = keras.regularizers.deserialize(config[key])
        if isinstance(config.get("kernel_constraint"), dict):
            config["kernel_constraint"] = keras.constraints.deserialize(
                config["kernel_constraint"]
            )

        return cls(**config)
