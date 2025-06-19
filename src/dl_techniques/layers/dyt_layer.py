"""
Dynamic Tanh (DyT) Layer Implementation for Keras 3.x

Based on the paper:
"Transformers without Normalization" by Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu
CVPR 2025
Paper URL: https://arxiv.org/abs/2503.10622

Key Insights from the Paper:
1. Problem addressed: Layer Normalization is a standard component in modern Transformer architectures,
   but comes with computational costs and potential limitations in expressivity.

2. Proposed solution: DynamicTanh (DyT) - a simple element-wise operation defined as DyT(x) = tanh(α·x),
   where α is a learnable scalar. This is followed by an affine transformation (weight * output + bias).

3. Benefits:
   - Simpler computation than LayerNormalization (no mean/variance calculations)
   - Improved training stability without batch statistics
   - Competitive performance with standard LayerNorm transformers
   - Potential computational efficiency advantages

4. How it works:
   - The learnable α parameter controls the "steepness" of the tanh activation
   - The tanh function naturally bounds the output, providing a form of regularization
   - The affine transformation (weight & bias) provides scaling and shifting capabilities

5. Implementation details:
   - This layer can be a drop-in replacement for LayerNormalization in transformer architectures
   - α is typically initialized to values around 0.5
   - Weight is initialized to ones and bias to zeros

This implementation supports different axis configurations and provides full support
for regularization, constraints, and different initialization strategies.

Example usage in a transformer block:

```python
def transformer_block_with_dyt(
    inputs: keras.KerasTensor,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    dropout_rate: float = 0.1,
    kernel_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
) -> keras.KerasTensor:
    # Self-attention part
    dyt1 = DynamicTanh(alpha_init_value=0.7)(inputs)
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(dyt1, dyt1)
    attention_output = keras.layers.Dropout(dropout_rate)(attention_output)
    attention_output = keras.layers.Add()([inputs, attention_output])

    # Feed-forward part
    dyt2 = DynamicTanh(alpha_init_value=0.15)(attention_output)
    ffn_output = keras.layers.Dense(
        intermediate_size,
        activation="gelu",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(dyt2)
    ffn_output = keras.layers.Dense(
        hidden_size,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(ffn_output)
    ffn_output = keras.layers.Dropout(dropout_rate)(ffn_output)

    return keras.layers.Add()([attention_output, ffn_output])
```
"""

import keras
from keras import ops, constraints, initializers, regularizers
from typing import Optional, Union, Dict, Any, List
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class DynamicTanh(keras.layers.Layer):
    """
    Dynamic Tanh (DyT) layer as described in "Transformers without Normalization" (Zhu et al., 2025).

    This layer replaces normalization layers in Transformers with a learnable scaled
    hyperbolic tangent and an affine transformation: DyT(x) = weight * tanh(alpha * x) + bias.

    The layer applies the following transformation:
    1. Scale input by learnable parameter alpha: alpha * x
    2. Apply hyperbolic tangent activation: tanh(alpha * x)
    3. Apply affine transformation: weight * tanh(alpha * x) + bias

    Args:
        axis: Integer or list of integers, the axis that should be normalized
            (typically the features axis). Default is -1 (last axis).
        alpha_init_value: Float, the initial value for the learnable scalar alpha.
            The paper suggests different initialization values depending on the layer's position:

            - For attention normalization: ~0.6-0.8
            - For FFN normalization: ~0.1-0.2
            - For final decoder normalization: ~0.1-0.2

            Default is 0.5.
        kernel_initializer: Initializer for the weight parameters. Default is 'ones'.
        bias_initializer: Initializer for the bias parameters. Default is 'zeros'.
        kernel_regularizer: Regularizer function applied to the weight parameters.
        bias_regularizer: Regularizer function applied to the bias parameters.
        kernel_constraint: Constraint function applied to the weight parameters.
        bias_constraint: Constraint function applied to the bias parameters.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    def __init__(
        self,
        axis: Union[int, List[int]] = -1,
        alpha_init_value: float = 0.5,
        kernel_initializer: Union[str, initializers.Initializer] = 'ones',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate alpha_init_value first
        if not isinstance(alpha_init_value, (int, float)):
            raise ValueError(f"alpha_init_value must be a number, got {type(alpha_init_value)}")

        # Store and validate configuration parameters
        self.axis = list(axis) if isinstance(axis, (list, tuple)) else [axis]
        self.alpha_init_value = float(alpha_init_value)

        # Initialize serializable objects
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Initialize weights to None (will be created in build)
        self.alpha = None
        self.weight = None
        self.bias = None

        # Store build input shape for serialization
        self._build_input_shape = None

        # Enable masking support
        self.supports_masking = True

        logger.debug(f"Initialized DynamicTanh layer with axis={self.axis}, alpha_init_value={self.alpha_init_value}")

    def build(self, input_shape: tuple) -> None:
        """
        Build the layer weights based on input shape.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If axis is out of bounds for the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        ndims = len(input_shape)

        # Validate and normalize axes
        normalized_axis = []
        for ax in self.axis:
            if ax >= ndims or ax < -ndims:
                raise ValueError(
                    f"Axis {ax} is out of bounds for tensor of dimension {ndims}"
                )
            # Convert negative axes to positive
            normalized_ax = ndims + ax if ax < 0 else ax
            normalized_axis.append(normalized_ax)

        self.axis = normalized_axis

        # Calculate parameter shape
        param_shape = [input_shape[ax] for ax in self.axis]
        # Ensure param_shape is always a tuple for add_weight
        if len(param_shape) == 1:
            param_shape = (param_shape[0],)
        else:
            param_shape = tuple(param_shape)

        logger.debug(f"Building DynamicTanh layer with param_shape={param_shape}")

        # Create learnable alpha parameter (scalar)
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),  # Scalar
            initializer=lambda shape, dtype: ops.cast(self.alpha_init_value, dtype),
            trainable=True,
            dtype=self.dtype
        )

        # Create weight parameter
        self.weight = self.add_weight(
            name="weight",
            shape=param_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )

        # Create bias parameter
        self.bias = self.add_weight(
            name="bias",
            shape=param_shape,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype
        )

        super().build(input_shape)
        logger.debug("DynamicTanh layer built successfully")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs
    ) -> keras.KerasTensor:
        """
        Forward computation of the DynamicTanh layer.

        Applies the transformation: weight * tanh(alpha * inputs) + bias

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer.
            **kwargs: Additional keyword arguments.

        Returns:
            Output tensor after applying dynamic tanh transformation.
        """
        # Apply dynamic tanh transformation: tanh(alpha * x)
        scaled_inputs = self.alpha * inputs
        tanh_outputs = ops.tanh(scaled_inputs)

        # Apply affine transformation with proper broadcasting
        # Get dynamic shape for runtime dimensions
        dynamic_shape = ops.shape(inputs)
        ndims = len(inputs.shape)

        # Create broadcast shape for weight and bias
        # Start with all dimensions as 1, then set the normalized axes to their actual sizes
        broadcast_shape = []
        for i in range(ndims):
            if i in self.axis:
                broadcast_shape.append(dynamic_shape[i])
            else:
                broadcast_shape.append(1)

        # Reshape weight and bias for proper broadcasting
        weight_broadcasted = ops.reshape(self.weight, broadcast_shape)
        bias_broadcasted = ops.reshape(self.bias, broadcast_shape)

        # Apply affine transformation
        outputs = tanh_outputs * weight_broadcasted + bias_broadcasted

        return outputs

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'alpha_init_value': self.alpha_init_value,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from build configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])