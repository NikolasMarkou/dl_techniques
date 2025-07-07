"""
BitLinear Layer Implementation
============================

A Keras implementation of bit-aware linear layer for quantization-aware training,
based on concepts from "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
https://arxiv.org/pdf/2402.17764.pdf

Key Features:
------------
- Supports low-bit quantization (e.g., 1.58-bit with values -1, 0, 1)
- Quantization-aware training with straight-through estimator
- Customizable quantization strategies and bit-widths
- Different measurement methods for scaling (AbsMax, AbsMean, AbsMedian)
- Optional layer normalization
- Configurable kernel initializers and regularizers

Architecture:
------------
The BitLinear layer consists of:
1. Input normalization (optional)
2. Input scaling and quantization
3. Weight scaling and quantization
4. Linear transformation (matrix multiplication)
5. Output rescaling

The computation flow is:
input -> normalize -> scale_and_quantize ->
        matrix_multiply(quantized_weights) -> rescale -> output

Configuration:
-------------
Supports extensive customization through BitLinearConfig:
- Input/output dimensions
- Quantization bit-widths for weights and activations
- Measurement methods for scaling
- Quantization strategies
- Bias terms
- Regularization strategies
- Normalization

Usage Examples:
-------------
```python
# Basic configuration with 1.58-bit weights
config = BitLinearConfig(
    in_features=768,
    out_features=768,
    weight_range=1.58,
    weight_measure="AbsMedian",
    activation_range=8,
    activation_measure="AbsMax"
)
layer = BitLinear(config)

# Advanced configuration with regularization
config = BitLinearConfig(
    in_features=1024,
    out_features=4096,
    weight_range=2,
    weight_measure="AbsMean",
    kernel_initializer="glorot_uniform",
    kernel_regularizer=keras.regularizers.L2(1e-5)
)
layer = BitLinear(
    bit_config=config,
    lambda_=0.8,
    use_norm=True
)
```
"""

import copy
import keras
import tensorflow as tf
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Union, List, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.scaling import *

# ---------------------------------------------------------------------

@dataclass
class BitLinearConfig:
    """Configuration for BitLinear layer parameters.

    This dataclass holds all configurable parameters for the BitLinear layer,
    including dimensions, quantization settings, and initializers.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        weight_range: Range of weight values (either tuple (min, max) or number of bits)
        weight_measure: Method for computing representative value for weights
                       ("AbsMax", "AbsMean", or "AbsMedian")
        activation_range: Range of activation values (either tuple (min, max) or number of bits)
        activation_measure: Method for computing representative value for activations
                          ("AbsMax", "AbsMean", or "AbsMedian")
        strategy: Quantization strategy to use ("round_clamp" or "sample")
        use_bias: Whether to include a bias term
        eps: Small value to avoid division by zero
        kernel_initializer: Initializer for the kernel weights
        kernel_regularizer: Optional regularization for kernel weights
        bias_initializer: Initializer for the bias
        bias_regularizer: Optional regularization for bias

    Examples:
        >>> config = BitLinearConfig(
        ...     in_features=768,
        ...     out_features=768,
        ...     weight_range=1.58,
        ...     weight_measure="AbsMedian"
        ... )
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
        """Validate configuration after initialization."""
        if not isinstance(self.in_features, int) or self.in_features <= 0:
            raise ValueError(f"in_features must be a positive integer, got {self.in_features}")

        if not isinstance(self.out_features, int) or self.out_features <= 0:
            raise ValueError(f"out_features must be a positive integer, got {self.out_features}")

        valid_measures = ["AbsMax", "AbsMean", "AbsMedian"]
        if self.weight_measure not in valid_measures:
            raise ValueError(f"weight_measure must be one of {valid_measures}, got {self.weight_measure}")

        if self.activation_measure not in valid_measures:
            raise ValueError(f"activation_measure must be one of {valid_measures}, got {self.activation_measure}")

        valid_strategies = ["round_clamp", "sample"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {self.strategy}")

        if not isinstance(self.eps, float) or self.eps <= 0:
            raise ValueError(f"eps must be a positive float, got {self.eps}")

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable(package="BitLinear")
class BitLinear(keras.layers.Layer):
    """Bit-aware linear layer for quantization-aware training.

    This layer implements a quantization-aware linear (dense) layer that scales and
    quantizes both weights and activations to low-bit representations during training.
    It uses a straight-through estimator for gradient propagation through the
    quantization operations.

    Args:
        bit_config: Configuration for BitLinear parameters
        lambda_: Scaling factor for the straight-through estimator
        use_norm: Whether to use layer normalization before quantization
        name: Name of the layer

    Attributes:
        bit_config: Configuration for the layer
        lambda_: Scaling factor for the straight-through estimator
        use_norm: Whether layer normalization is used
        weight_range: Computed range for weight quantization
        activation_range: Computed range for activation quantization

    Examples:
        >>> config = BitLinearConfig(in_features=4, out_features=2)
        >>> layer = BitLinear(config)
        >>> x = tf.random.normal((3, 4))
        >>> y = layer(x)
        >>> y.shape
        TensorShape([3, 2])
    """

    def __init__(
            self,
            bit_config: BitLinearConfig,
            lambda_: float = 1.0,
            use_norm: bool = False,
            **kwargs
    ):
        """Initialize the BitLinear layer.

        Args:
            bit_config: Configuration for BitLinear parameters
            lambda_: Scaling factor for the straight-through estimator
            use_norm: Whether to use layer normalization
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)

        # Store configurations
        self.bit_config = bit_config
        self.lambda_ = lambda_
        self.use_norm = use_norm

        # Validate lambda
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")

        # Process ranges
        if isinstance(self.bit_config.weight_range, (int, float)):
            self.weight_range = range_from_bits(self.bit_config.weight_range)
        else:
            self.weight_range = self.bit_config.weight_range

        if isinstance(self.bit_config.activation_range, (int, float)):
            self.activation_range = range_from_bits(self.bit_config.activation_range)
        else:
            self.activation_range = self.bit_config.activation_range

        # These will be initialized in build()
        self.kernel = None
        self.bias = None
        self.weight_measure_fn = None
        self.activation_measure_fn = None
        self.strategy_fn = None
        self.norm_layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer by creating weights and initializing sublayers.

        Args:
            input_shape: Shape of the input tensor

        Raises:
            ValueError: If input shape is incompatible with configured dimensions
        """
        # Validate input shape
        if input_shape[-1] != self.bit_config.in_features:
            raise ValueError(
                f"Last dimension of input shape ({input_shape[-1]}) must match "
                f"in_features ({self.bit_config.in_features})"
            )

        # Create the kernel weight variable
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.bit_config.in_features, self.bit_config.out_features),
            initializer=keras.initializers.get(self.bit_config.kernel_initializer),
            regularizer=self.bit_config.kernel_regularizer,
            trainable=True
        )

        # Create the bias variable if needed
        if self.bit_config.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.bit_config.out_features,),
                initializer=keras.initializers.get(self.bit_config.bias_initializer),
                regularizer=self.bit_config.bias_regularizer,
                trainable=True
            )
        else:
            self.bias = None

        # Initialize measure functions based on configuration
        measure_functions = {
            "AbsMax": abs_max,
            "AbsMean": abs_mean,
            "AbsMedian": abs_median
        }

        self.weight_measure_fn = measure_functions[self.bit_config.weight_measure]
        self.activation_measure_fn = measure_functions[self.bit_config.activation_measure]

        # Initialize strategy function
        strategy_functions = {
            "round_clamp": round_clamp,
            "sample": sample
        }

        self.strategy_fn = strategy_functions[self.bit_config.strategy]

        # Initialize normalization if needed
        if self.use_norm:
            self.norm_layer = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name=f"{self.name}_norm" if self.name else None
            )

        self.built = True

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the BitLinear layer.

        This method implements the full computation flow of the BitLinear layer:
        1. Input normalization (optional)
        2. Input scaling and quantization
        3. Weight scaling and quantization
        4. Linear transformation (matrix multiplication)
        5. Output rescaling

        Args:
            inputs: Input tensor of shape (..., in_features)
            training: Whether in training mode

        Returns:
            tf.Tensor: Output tensor of shape (..., out_features)
        """
        # Apply normalization if provided
        if self.use_norm:
            x_norm = self.norm_layer(inputs, training=training)
        else:
            x_norm = inputs

        # Scale and quantize activations if needed
        if self.activation_measure_fn is None:
            x_scale, x_quant = tf.constant(1.0, dtype=inputs.dtype), x_norm
        else:
            x_scale = scale(x_norm, self.activation_range, self.activation_measure_fn, True, self.bit_config.eps)
            x_quant = self.strategy_fn(x_norm * x_scale, self.activation_range, self.lambda_)

        # Scale and quantize weights
        if self.weight_measure_fn is None:
            w_scale, w_quant = tf.constant(1.0, dtype=self.kernel.dtype), self.kernel
        else:
            w_scale = scale(self.kernel, self.weight_range, self.weight_measure_fn, False, self.bit_config.eps)
            w_quant = self.strategy_fn(self.kernel * w_scale, self.weight_range, self.lambda_)

        # Perform the linear transformation
        outputs = tf.matmul(x_quant, w_quant)

        # Add bias if needed
        if self.bias is not None:
            outputs = outputs + self.bias

        # Rescale the outputs
        return outputs / (w_scale * x_scale)

    def compute_output_shape(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]) -> (
            Union)[tf.TensorShape, List[tf.TensorShape]]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor or list of shapes

        Returns:
            Union[tf.TensorShape, List[tf.TensorShape]]: Shape of the output tensor or list of shapes
        """
        if isinstance(input_shape, list):
            return [self.compute_output_shape(shape) for shape in input_shape]

        batch_dims = input_shape[:-1]
        return tf.TensorShape(batch_dims.as_list() + [self.bit_config.out_features])

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer for serialization.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "bit_config": asdict(self.bit_config),
            "lambda_": self.lambda_,
            "use_norm": self.use_norm,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BitLinear':
        """Creates a BitLinear layer from its config.

        Args:
            config: Layer configuration dictionary

        Returns:
            BitLinear: Instantiated BitLinear layer
        """
        # Make a copy of the config to avoid modifying the original
        config_copy = copy.deepcopy(config)

        # Extract the bit_config dictionary
        bit_config_dict = config_copy.pop("bit_config", {})

        # Check for core layer parameters and restore them if needed
        for key in ["kernel_initializer", "kernel_regularizer", "bias_initializer", "bias_regularizer"]:
            if key in config:
                bit_config_dict[key] = config.pop(key)

        # Recreate the BitLinearConfig object
        bit_config = BitLinearConfig(**bit_config_dict)

        # Create the BitLinear with the recreated BitLinearConfig
        return cls(bit_config=bit_config, **config_copy)

# ---------------------------------------------------------------------
