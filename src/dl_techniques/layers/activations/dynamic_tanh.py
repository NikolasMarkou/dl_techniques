"""
Dynamic Tanh (DyT) Layer Implementation for Keras 3.x

Based on "Transformers without Normalization" by Zhu et al., CVPR 2025.
Paper: https://arxiv.org/abs/2503.10622

Key innovation: Replace LayerNormalization with learnable scaled hyperbolic tangent
plus affine transformation for improved computational efficiency in Transformers.

The transformation is: DyT(x) = weight * tanh(alpha * x) + bias
where alpha is a learnable scalar parameter.

Benefits:
- Simpler computation than LayerNormalization
- No batch statistics required
- Improved training stability
- Competitive performance with LayerNorm
"""

import keras
from typing import Optional, Union, Dict, Any, List, Tuple
from keras import ops, constraints, initializers, regularizers

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DynamicTanh(keras.layers.Layer):
    """Dynamic Tanh (DyT) layer as described in "Transformers without Normalization".

    Applies learnable scaled hyperbolic tangent followed by affine transformation:
    output = weight * tanh(alpha * input) + bias

    This serves as a drop-in replacement for LayerNormalization in Transformers,
    providing normalization-like benefits without batch statistics computation.

    Args:
        axis: Integer or list of integers specifying normalization axes.
            Typically -1 (features axis). Defaults to -1.
        alpha_init_value: Float, initial value for learnable alpha parameter.
            Paper suggests:
            - Attention normalization: 0.6-0.8
            - FFN normalization: 0.1-0.2
            - Final decoder normalization: 0.1-0.2
            Defaults to 0.5.
        kernel_initializer: Initializer for weight parameters. Defaults to 'ones'.
        bias_initializer: Initializer for bias parameters. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for weight parameters.
        bias_regularizer: Optional regularizer for bias parameters.
        kernel_constraint: Optional constraint for weight parameters.
        bias_constraint: Optional constraint for bias parameters.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        Arbitrary tensor shape.

    Output shape:
        Same as input shape.

    Example:
        ```python
        # Basic usage as LayerNorm replacement
        layer = DynamicTanh(alpha_init_value=0.7)

        # For attention normalization (higher alpha)
        attn_norm = DynamicTanh(
            alpha_init_value=0.8,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # For FFN normalization (lower alpha)
        ffn_norm = DynamicTanh(alpha_init_value=0.15)

        # Multi-axis normalization
        layer = DynamicTanh(axis=[1, 2])

        # In transformer block
        def transformer_block(x):
            # Attention normalization
            x_norm = DynamicTanh(alpha_init_value=0.7)(x)
            attn_out = MultiHeadAttention(...)(x_norm, x_norm)
            x = Add()([x, attn_out])

            # FFN normalization
            x_norm = DynamicTanh(alpha_init_value=0.15)(x)
            ffn_out = Dense(...)(x_norm)
            return Add()([x, ffn_out])
        ```

    Raises:
        ValueError: If alpha_init_value is not a number.
        ValueError: If axis is out of bounds for input tensor.
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

        # Validate alpha initialization value
        if not isinstance(alpha_init_value, (int, float)):
            raise ValueError(f"alpha_init_value must be a number, got {type(alpha_init_value)}")

        # Store ALL configuration parameters
        self.axis = list(axis) if isinstance(axis, (list, tuple)) else [axis]
        self.alpha_init_value = float(alpha_init_value)

        # Store serializable initializers/regularizers/constraints
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Initialize weight attributes - created in build()
        self.alpha = None
        self.weight = None
        self.bias = None

        # Enable masking support
        self.supports_masking = True

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's learnable parameters.

        Creates alpha (scalar), weight, and bias parameters based on input shape
        and configured normalization axes.
        """
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

        # Calculate parameter shape for weight and bias
        param_shape = tuple(input_shape[ax] for ax in self.axis)

        # Create layer's own weights
        # Alpha: learnable scalar parameter
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),  # Scalar shape
            initializer=lambda shape, dtype: ops.cast(self.alpha_init_value, dtype),
            trainable=True,
            dtype=self.dtype
        )

        # Weight: affine transformation scaling
        self.weight = self.add_weight(
            name="weight",
            shape=param_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )

        # Bias: affine transformation offset
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

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward computation: weight * tanh(alpha * inputs) + bias.

        Args:
            inputs: Input tensor.
            training: Training mode flag (unused but kept for interface consistency).

        Returns:
            Transformed tensor with same shape as input.
        """
        # Step 1: Scale inputs by learnable alpha
        scaled_inputs = self.alpha * inputs

        # Step 2: Apply hyperbolic tangent
        tanh_outputs = ops.tanh(scaled_inputs)

        # Step 3: Apply affine transformation with proper broadcasting
        input_shape = ops.shape(inputs)
        ndims = len(inputs.shape)

        # Create broadcast shape for weight and bias
        broadcast_shape = []
        for i in range(ndims):
            if i in self.axis:
                broadcast_shape.append(input_shape[i])
            else:
                broadcast_shape.append(1)

        # Reshape parameters for broadcasting
        weight_broadcasted = ops.reshape(self.weight, broadcast_shape)
        bias_broadcasted = ops.reshape(self.bias, broadcast_shape)

        # Final affine transformation
        outputs = tanh_outputs * weight_broadcasted + bias_broadcasted

        return outputs

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as input shape)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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

# ---------------------------------------------------------------------
