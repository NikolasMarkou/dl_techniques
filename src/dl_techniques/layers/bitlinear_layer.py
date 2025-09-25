"""Bit-quantized linear layer for efficient language models.

This layer provides a Keras implementation of the linear layer proposed in the
BitNet architecture, designed to enable 1-bit Large Language Models (LLMs). It
replaces standard floating-point matrix multiplication with low-bit integer
operations, drastically reducing memory footprint and computational cost.

References:
    - Wang et al. "The Era of 1-bit LLMs: All Large Language Models are in
      1.58 Bits". https://arxiv.org/abs/2402.17764
    - Bengio et al. "Estimating or Propagating Gradients Through Stochastic
      Neurons for Conditional Computation". https://arxiv.org/abs/1308.3432
"""

import keras
from typing import Optional, Dict, Any, Union, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BitLinear(keras.layers.Layer):
    """Bit-aware linear layer for quantization-aware training.

    This layer implements a quantization-aware dense transformation that scales
    and quantizes both weights and activations to low-bit representations during
    the forward pass, while maintaining full precision gradients through a
    straight-through estimator.

    The layer automatically infers input dimensions from the first call, similar
    to keras.layers.Dense, making it suitable for functional model construction.

    Args:
        units: Positive integer, dimensionality of the output space.
        weight_bits: Number of bits for weight quantization or explicit range tuple.
            Default is 1.58 bits (ternary weights: -1, 0, 1).
        activation_bits: Number of bits for activation quantization or explicit range.
            Default is 8 bits.
        weight_scale_method: Method to compute weight scaling factor.
            Options: "abs_max", "abs_mean", "abs_median". Default: "abs_median".
        activation_scale_method: Method to compute activation scaling factor.
            Options: "abs_max", "abs_mean", "abs_median". Default: "abs_max".
        quantization_method: Quantization strategy to use.
            Options: "round_clip", "stochastic". Default: "round_clip".
        use_bias: Boolean, whether the layer uses a bias vector. Default: True.
        use_input_norm: Boolean, whether to apply layer normalization to inputs
            before quantization. Default: False.
        ste_lambda: Float, scaling factor for straight-through estimator gradient.
            Controls gradient flow through quantization. Default: 1.0.
        epsilon: Float, small constant for numerical stability. Default: 1e-5.
        kernel_initializer: Initializer for the kernel weights matrix.
            Default: "glorot_uniform".
        bias_initializer: Initializer for the bias vector. Default: "zeros".
        kernel_regularizer: Regularizer function applied to kernel weights matrix.
            Default: None.
        bias_regularizer: Regularizer function applied to bias vector.
            Default: None.
        **kwargs: Additional keyword arguments passed to keras.layers.Layer.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        The most common situation would be a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.

    Example:
        ```python
        # Basic usage - input dimension inferred automatically
        model = keras.Sequential([
            keras.layers.Input(shape=(784,)),
            BitLinear(units=256, weight_bits=1.58),
            keras.layers.ReLU(),
            BitLinear(units=128, weight_bits=2, activation_bits=4),
            BitLinear(units=10)
        ])

        # Functional API with automatic dimension inference
        inputs = keras.layers.Input(shape=(768,))
        x = BitLinear(
            units=512,
            weight_bits=1.58,
            weight_scale_method="abs_mean",
            use_input_norm=True,
            kernel_regularizer=keras.regularizers.L2(1e-5)
        )(inputs)
        x = keras.layers.ReLU()(x)
        outputs = BitLinear(units=256)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Can also be used without specifying input shape initially
        layer = BitLinear(units=128, weight_bits=2)
        # Input dimension will be inferred on first call
        output = layer(keras.random.normal((32, 64)))  # Infers input_dim=64
        ```
    """

    def __init__(
        self,
        units: int,
        weight_bits: Union[float, int, Tuple[float, float]] = 1.58,
        activation_bits: Union[float, int, Tuple[float, float]] = 8,
        weight_scale_method: str = "abs_median",
        activation_scale_method: str = "abs_max",
        quantization_method: str = "round_clip",
        use_bias: bool = True,
        use_input_norm: bool = False,
        ste_lambda: float = 1.0,
        epsilon: float = 1e-5,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the BitLinear layer.

        Raises:
            ValueError: If units is not a positive integer.
            ValueError: If scale methods or quantization method are invalid.
            ValueError: If ste_lambda or epsilon are not positive.
        """
        super().__init__(**kwargs)

        # Validate units parameter
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")

        # Validate scale methods
        valid_scale_methods = ["abs_max", "abs_mean", "abs_median"]
        if weight_scale_method not in valid_scale_methods:
            raise ValueError(
                f"weight_scale_method must be one of {valid_scale_methods}, "
                f"got {weight_scale_method}"
            )
        if activation_scale_method not in valid_scale_methods:
            raise ValueError(
                f"activation_scale_method must be one of {valid_scale_methods}, "
                f"got {activation_scale_method}"
            )

        # Validate quantization method
        valid_quant_methods = ["round_clip", "stochastic"]
        if quantization_method not in valid_quant_methods:
            raise ValueError(
                f"quantization_method must be one of {valid_quant_methods}, "
                f"got {quantization_method}"
            )

        # Validate numerical parameters
        if ste_lambda <= 0:
            raise ValueError(f"ste_lambda must be positive, got {ste_lambda}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.units = units
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_scale_method = weight_scale_method
        self.activation_scale_method = activation_scale_method
        self.quantization_method = quantization_method
        self.use_bias = use_bias
        self.use_input_norm = use_input_norm
        self.ste_lambda = ste_lambda
        self.epsilon = epsilon

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Convert bit specifications to ranges
        self.weight_range = self._bits_to_range(weight_bits)
        self.activation_range = self._bits_to_range(activation_bits)

        # Layer weights (created in build)
        self.kernel = None
        self.bias = None
        self.input_norm = None

    def _bits_to_range(
        self,
        bits: Union[float, int, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Convert bit specification to quantization range.

        Args:
            bits: Number of bits or explicit range tuple.

        Returns:
            Tuple of (min_value, max_value) for quantization.
        """
        if isinstance(bits, tuple):
            return bits

        if isinstance(bits, (int, float)):
            if bits <= 1:
                # Binary quantization
                return (-1.0, 1.0)
            elif bits < 2:
                # 1.58 bits = ternary {-1, 0, 1}
                return (-1.0, 1.0)
            elif bits == 2:
                # 2-bit = {-1, 0, 1}
                return (-1.0, 1.0)
            else:
                # n-bit quantization: [-(2^(n-1)-1), 2^(n-1)-1]
                n = int(bits)
                max_val = float(2 ** (n - 1) - 1)
                return (-max_val, max_val)

        raise ValueError(f"Invalid bit specification: {bits}")

    def _compute_scale(
        self,
        tensor: keras.KerasTensor,
        method: str,
        per_channel: bool,
        target_range: Tuple[float, float]
    ) -> keras.KerasTensor:
        """Compute scaling factor for quantization.

        Args:
            tensor: Tensor to compute scale for.
            method: Scaling method ("abs_max", "abs_mean", or "abs_median").
            per_channel: If True, compute per-channel scaling (last dimension).
            target_range: Target quantization range.

        Returns:
            Scaling factor tensor.
        """
        min_val, max_val = target_range
        abs_tensor = keras.ops.abs(tensor)

        # Determine reduction axes
        if per_channel:
            # Reduce all dimensions except the last one
            ndims = len(keras.ops.shape(tensor))
            axes = list(range(ndims - 1))
            keepdims = True
        else:
            # Reduce all dimensions
            axes = None
            keepdims = False

        # Compute representative value based on method
        if method == "abs_max":
            if axes:
                gamma = keras.ops.max(abs_tensor, axis=axes, keepdims=keepdims)
            else:
                gamma = keras.ops.max(abs_tensor)
        elif method == "abs_mean":
            if axes:
                gamma = keras.ops.mean(abs_tensor, axis=axes, keepdims=keepdims)
            else:
                gamma = keras.ops.mean(abs_tensor)
        elif method == "abs_median":
            # Approximate median using 50th percentile
            if per_channel:
                # Compute per-channel median
                shape = keras.ops.shape(tensor)
                # Flatten all dimensions except last
                flat_shape = (-1, shape[-1])
                flat_tensor = keras.ops.reshape(abs_tensor, flat_shape)
                gamma = keras.ops.quantile(flat_tensor, 0.5, axis=0, keepdims=True)
                # Ensure proper broadcasting shape
                new_shape = [1] * (len(shape) - 1) + [shape[-1]]
                gamma = keras.ops.reshape(gamma, new_shape)
            else:
                flat_tensor = keras.ops.reshape(abs_tensor, (-1,))
                gamma = keras.ops.quantile(flat_tensor, 0.5)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Prevent division by zero
        gamma = keras.ops.maximum(gamma, self.epsilon)

        # Compute scale to map to target range
        target_max = max(abs(min_val), abs(max_val))
        scale = target_max / gamma

        return scale

    def _quantize_tensor(
        self,
        tensor: keras.KerasTensor,
        target_range: Tuple[float, float]
    ) -> keras.KerasTensor:
        """Apply quantization with straight-through estimator.

        Args:
            tensor: Scaled tensor to quantize.
            target_range: Target quantization range.

        Returns:
            Quantized tensor with straight-through gradient.
        """
        min_val, max_val = target_range

        if self.quantization_method == "round_clip":
            # Round to nearest integer and clip to range
            rounded = keras.ops.round(tensor)
            clipped = keras.ops.clip(rounded, min_val, max_val)

            # Straight-through estimator: use quantized forward, original gradient
            quantized = keras.ops.stop_gradient(clipped - tensor) + tensor * self.ste_lambda

        elif self.quantization_method == "stochastic":
            # Stochastic rounding for better gradient estimation
            floor_val = keras.ops.floor(tensor)
            ceil_val = keras.ops.ceil(tensor)

            # Probability of rounding up
            prob_ceil = tensor - floor_val

            # Generate random values for stochastic rounding
            random_uniform = keras.random.uniform(
                keras.ops.shape(tensor),
                minval=0.0,
                maxval=1.0,
                dtype=tensor.dtype
            )

            # Stochastic rounding
            rounded = keras.ops.where(
                random_uniform < prob_ceil,
                ceil_val,
                floor_val
            )
            clipped = keras.ops.clip(rounded, min_val, max_val)

            # Straight-through estimator
            quantized = keras.ops.stop_gradient(clipped - tensor) + tensor * self.ste_lambda

        else:
            raise ValueError(f"Unknown quantization method: {self.quantization_method}")

        return quantized

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights based on input shape.

        Args:
            input_shape: Shape tuple of the input.

        Raises:
            ValueError: If the last dimension of input_shape is None.
        """
        super().build(input_shape)

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
            trainable=True,
            dtype=self.dtype
        )

        # Create bias weights if needed
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype=self.dtype
            )

        # Create input normalization layer if needed
        if self.use_input_norm:
            self.input_norm = keras.layers.LayerNormalization(
                axis=-1,
                epsilon=1e-6,
                center=True,
                scale=True,
                name=f"{self.name}_input_norm" if self.name else "input_norm"
            )
            self.input_norm.build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Perform quantized linear transformation.

        Args:
            inputs: Input tensor.
            training: Boolean flag indicating training mode.

        Returns:
            Transformed output tensor.
        """
        # Apply input normalization if configured
        if self.use_input_norm and self.input_norm is not None:
            x = self.input_norm(inputs, training=training)
        else:
            x = inputs

        # Scale and quantize activations
        activation_scale = self._compute_scale(
            x,
            method=self.activation_scale_method,
            per_channel=True,  # Per-channel scaling for activations
            target_range=self.activation_range
        )
        x_scaled = x * activation_scale
        x_quantized = self._quantize_tensor(x_scaled, self.activation_range)

        # Scale and quantize weights
        weight_scale = self._compute_scale(
            self.kernel,
            method=self.weight_scale_method,
            per_channel=False,  # Global scaling for weights
            target_range=self.weight_range
        )
        w_scaled = self.kernel * weight_scale
        w_quantized = self._quantize_tensor(w_scaled, self.weight_range)

        # Perform quantized matrix multiplication
        output = keras.ops.matmul(x_quantized, w_quantized)

        # Add bias if present
        if self.use_bias and self.bias is not None:
            output = output + self.bias

        # Rescale output to original magnitude
        output = output / (activation_scale * weight_scale)

        return output

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of input.

        Returns:
            Shape tuple of output.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "weight_scale_method": self.weight_scale_method,
            "activation_scale_method": self.activation_scale_method,
            "quantization_method": self.quantization_method,
            "use_bias": self.use_bias,
            "use_input_norm": self.use_input_norm,
            "ste_lambda": self.ste_lambda,
            "epsilon": self.epsilon,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
                                  if self.kernel_regularizer else None,
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer)
                               if self.bias_regularizer else None,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitLinear":
        """Create layer instance from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            New BitLinear instance.
        """
        # Deserialize initializers
        if "kernel_initializer" in config:
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "bias_initializer" in config:
            config["bias_initializer"] = keras.initializers.deserialize(
                config["bias_initializer"]
            )

        # Deserialize regularizers
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if "bias_regularizer" in config and config["bias_regularizer"]:
            config["bias_regularizer"] = keras.regularizers.deserialize(
                config["bias_regularizer"]
            )

        return cls(**config)

# ---------------------------------------------------------------------
