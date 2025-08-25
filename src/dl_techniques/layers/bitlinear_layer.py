"""
BitLinear Layer Implementation

A Keras implementation of bit-aware linear layer for quantization-aware training,
based on "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits".

Key features:
- Low-bit quantization (e.g., 1.58-bit with values -1, 0, 1)
- Quantization-aware training with straight-through estimator
- Customizable quantization strategies and bit-widths
- Different measurement methods for scaling
- Optional layer normalization
"""

import copy
import keras
from keras import ops
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.scaling import *

# ---------------------------------------------------------------------


@dataclass
class BitLinearConfig:
    """Configuration for BitLinear layer parameters.

    Args:
        in_features: Size of each input sample. Must be positive.
        out_features: Size of each output sample. Must be positive.
        weight_range: Range of weight values (tuple (min, max) or number of bits).
        weight_measure: Method for computing representative value for weights.
            Options: "AbsMax", "AbsMean", "AbsMedian".
        activation_range: Range of activation values (tuple (min, max) or number of bits).
        activation_measure: Method for computing representative value for activations.
            Options: "AbsMax", "AbsMean", "AbsMedian".
        strategy: Quantization strategy. Options: "round_clamp", "sample".
        use_bias: Whether to include a bias term.
        eps: Small value to avoid division by zero. Must be positive.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Optional regularization for kernel weights.
        bias_initializer: Initializer for bias weights.
        bias_regularizer: Optional regularization for bias weights.

    Example:
        ```python
        config = BitLinearConfig(
            in_features=768,
            out_features=768,
            weight_range=1.58,
            weight_measure="AbsMedian"
        )
        ```
    """
    in_features: int
    out_features: int
    weight_range: Union[Tuple[int, int], float, int] = 1.58
    weight_measure: str = "AbsMedian"
    activation_range: Union[Tuple[int, int], float, int] = 8
    activation_measure: str = "AbsMax"
    strategy: str = "round_clamp"
    use_bias: bool = True
    eps: float = 1e-5
    kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform"
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
    bias_initializer: Union[str, keras.initializers.Initializer] = "zeros"
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.in_features, int) or self.in_features <= 0:
            raise ValueError(f"in_features must be positive integer, got {self.in_features}")

        if not isinstance(self.out_features, int) or self.out_features <= 0:
            raise ValueError(f"out_features must be positive integer, got {self.out_features}")

        valid_measures = ["AbsMax", "AbsMean", "AbsMedian"]
        if self.weight_measure not in valid_measures:
            raise ValueError(f"weight_measure must be one of {valid_measures}")

        if self.activation_measure not in valid_measures:
            raise ValueError(f"activation_measure must be one of {valid_measures}")

        valid_strategies = ["round_clamp", "sample"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BitLinear(keras.layers.Layer):
    """Bit-aware linear layer for quantization-aware training.

    Implements a quantization-aware dense layer that scales and quantizes both
    weights and activations to low-bit representations during training, using
    a straight-through estimator for gradient propagation.

    The computation flow is:
    input -> normalize (optional) -> scale_and_quantize ->
    matrix_multiply(quantized_weights) -> rescale -> output

    Args:
        bit_config: Configuration object containing all layer parameters.
        lambda_: Scaling factor for straight-through estimator. Must be positive.
        use_norm: Whether to use layer normalization before quantization.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        Tensor with shape `(..., in_features)`.

    Output shape:
        Tensor with shape `(..., out_features)`.

    Example:
        ```python
        config = BitLinearConfig(in_features=768, out_features=256)
        layer = BitLinear(config, lambda_=1.0, use_norm=True)

        inputs = keras.random.normal([32, 768])
        outputs = layer(inputs)  # Shape: (32, 256)

        # Advanced configuration
        config = BitLinearConfig(
            in_features=1024,
            out_features=4096,
            weight_range=2,
            weight_measure="AbsMean",
            kernel_regularizer=keras.regularizers.L2(1e-5)
        )
        layer = BitLinear(bit_config=config, lambda_=0.8)
        ```

    Raises:
        ValueError: If lambda_ is not positive.
        ValueError: If input shape is incompatible with configured dimensions.
    """

    def __init__(
        self,
        bit_config: BitLinearConfig,
        lambda_: float = 1.0,
        use_norm: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate lambda parameter
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")

        # Store ALL configuration parameters
        self.bit_config = bit_config
        self.lambda_ = lambda_
        self.use_norm = use_norm

        # Process ranges from configuration
        if isinstance(self.bit_config.weight_range, (int, float)):
            self.weight_range = range_from_bits(self.bit_config.weight_range)
        else:
            self.weight_range = self.bit_config.weight_range

        if isinstance(self.bit_config.activation_range, (int, float)):
            self.activation_range = range_from_bits(self.bit_config.activation_range)
        else:
            self.activation_range = self.bit_config.activation_range

        # Create optional sub-layer in __init__
        if self.use_norm:
            self.norm_layer = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name=f"{self.name}_norm" if self.name else "bit_linear_norm"
            )
        else:
            self.norm_layer = None

        # Initialize weight attributes - created in build()
        self.kernel = None
        self.bias = None
        self.weight_measure_fn = None
        self.activation_measure_fn = None
        self.strategy_fn = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create layer weights and build sub-layers."""
        # Validate input shape compatibility
        if input_shape[-1] != self.bit_config.in_features:
            raise ValueError(
                f"Input shape last dimension ({input_shape[-1]}) must match "
                f"in_features ({self.bit_config.in_features})"
            )

        # Create layer's own weights
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.bit_config.in_features, self.bit_config.out_features),
            initializer=keras.initializers.get(self.bit_config.kernel_initializer),
            regularizer=self.bit_config.kernel_regularizer,
            trainable=True,
            dtype=self.dtype
        )

        if self.bit_config.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.bit_config.out_features,),
                initializer=keras.initializers.get(self.bit_config.bias_initializer),
                regularizer=self.bit_config.bias_regularizer,
                trainable=True,
                dtype=self.dtype
            )

        # Initialize function mappings
        measure_functions = {
            "AbsMax": abs_max,
            "AbsMean": abs_mean,
            "AbsMedian": abs_median
        }

        strategy_functions = {
            "round_clamp": round_clamp,
            "sample": sample
        }

        self.weight_measure_fn = measure_functions[self.bit_config.weight_measure]
        self.activation_measure_fn = measure_functions[self.bit_config.activation_measure]
        self.strategy_fn = strategy_functions[self.bit_config.strategy]

        # Build sub-layer if it exists
        if self.norm_layer is not None:
            self.norm_layer.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing quantization-aware linear transformation.

        Args:
            inputs: Input tensor of shape (..., in_features).
            training: Training mode flag.

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Apply optional normalization
        if self.norm_layer is not None:
            x_norm = self.norm_layer(inputs, training=training)
        else:
            x_norm = inputs

        # Scale and quantize activations
        if self.activation_measure_fn is None:
            x_scale = ops.cast(1.0, dtype=inputs.dtype)
            x_quant = x_norm
        else:
            x_scale = scale(
                x_norm,
                self.activation_range,
                self.activation_measure_fn,
                True,
                self.bit_config.eps
            )
            x_quant = self.strategy_fn(x_norm * x_scale, self.activation_range, self.lambda_)

        # Scale and quantize weights
        if self.weight_measure_fn is None:
            w_scale = ops.cast(1.0, dtype=self.kernel.dtype)
            w_quant = self.kernel
        else:
            w_scale = scale(
                self.kernel,
                self.weight_range,
                self.weight_measure_fn,
                False,
                self.bit_config.eps
            )
            w_quant = self.strategy_fn(self.kernel * w_scale, self.weight_range, self.lambda_)

        # Perform linear transformation
        outputs = ops.matmul(x_quant, w_quant)

        # Add bias if configured
        if self.bias is not None:
            outputs = outputs + self.bias

        # Rescale outputs
        return outputs / (w_scale * x_scale)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output tensor shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.bit_config.out_features
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'bit_config': asdict(self.bit_config),
            'lambda_': self.lambda_,
            'use_norm': self.use_norm,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BitLinear':
        """Create BitLinear layer from configuration.

        Args:
            config: Layer configuration dictionary.

        Returns:
            BitLinear layer instance.
        """
        # Extract bit_config and recreate dataclass
        config_copy = copy.deepcopy(config)
        bit_config_dict = config_copy.pop('bit_config', {})
        bit_config = BitLinearConfig(**bit_config_dict)

        return cls(bit_config=bit_config, **config_copy)

# ---------------------------------------------------------------------

