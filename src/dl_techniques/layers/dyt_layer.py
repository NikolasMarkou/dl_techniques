"""
Dynamic Tanh (DyT) Layer Implementation for TensorFlow Keras

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

This implementation supports both channels-first and channels-last formats and provides full support
for regularization, constraints, and different initialization strategies.

A transformer block using Dynamic Tanh instead of LayerNormalization.

def transformer_block_with_dyt(
        inputs: tf.Tensor,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout_rate: float = 0.1,
        kernel_initializer: Optional[Union[str, initializers.Initializer]] = None,
        kernel_regularizer: Optional[regularizers.Regularizer] = None
) -> tf.Tensor:
    # Self-attention part
    dyt1 = DynamicTanh()(inputs)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(dyt1, dyt1)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.Add()([inputs, attention_output])

    # Feed-forward part
    dyt2 = DynamicTanh()(attention_output)
    ffn_output = layers.Dense(
        intermediate_size,
        activation="gelu",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(dyt2)
    ffn_output = layers.Dense(
        hidden_size,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(ffn_output)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)

    return layers.Add()([attention_output, ffn_output])

"""

import keras
import tensorflow as tf
from keras import constraints
from keras import initializers
from keras import regularizers
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class DynamicTanh(keras.layers.Layer):
    """
    Dynamic Tanh (DyT) layer as described in "Transformers without Normalization" (Zhu et al., 2025).

    This layer replaces normalization layers in Transformers with a learnable scaled
    hyperbolic tangent and an affine transformation: DyT(x) = weight * tanh(alpha * x) + bias.

    Args:
        axis: Integer or list of integers, the axis that should be normalized
            (typically the features axis for CNN and the last axis for RNN).
        channels_last: Boolean, whether the channel dimension is the last dimension.
        alpha_init_value: Float, the initial value for the learnable scalar alpha.
            The paper suggests different initialization values depending on the layer's position:
            - For attention normalization: ~0.6-0.8
            - For FFN normalization: ~0.1-0.2
            - For final decoder normalization: ~0.1-0.2
        kernel_initializer: Initializer for the weight parameters.
        kernel_regularizer: Regularizer function applied to the weight parameters.
        kernel_constraint: Constraint function applied to the weight parameters.
        bias_initializer: Initializer for the bias parameters.
        bias_regularizer: Regularizer function applied to the bias parameters.
        bias_constraint: Constraint function applied to the bias parameters.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    def __init__(
            self,
            axis: int = -1,
            channels_last: bool = True,
            alpha_init_value: float = 0.5,
            kernel_initializer: Union[str, initializers.Initializer] = 'ones',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            kernel_constraint: Optional[constraints.Constraint] = None,
            **kwargs
    ):
        super(DynamicTanh, self).__init__(**kwargs)
        self.axis = axis if isinstance(axis, (list, tuple)) else [axis]
        self.channels_last = channels_last
        self.alpha_init_value = alpha_init_value
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.supports_masking = True

        # For pre-computing broadcast shapes
        self.broadcast_shape = None
        self.alpha = None
        self.weight = None
        self.bias = None

    def build(self, input_shape):
        ndims = len(input_shape)

        # Validate axes
        for ax in self.axis:
            if ax >= ndims or ax < -ndims:
                raise ValueError(
                    f"Axis {ax} is out of bounds for tensor of dimension {ndims}"
                )

        # Convert negative axes to positive
        axis = [ndims + ax if ax < 0 else ax for ax in self.axis]

        # Calculate normalized shape
        param_shape = [input_shape[ax] for ax in axis]
        if len(param_shape) == 1:
            param_shape = param_shape[0]

        # Create trainable parameters
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.constant_initializer(self.alpha_init_value),
            trainable=True,
            dtype=self.dtype
        )

        self.weight = self.add_weight(
            name="weight",
            shape=param_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )

        self.bias = self.add_weight(
            name="bias",
            shape=param_shape,
            trainable=True,
            dtype=self.dtype
        )

        # Pre-compute broadcast shape for channels-first format
        if not self.channels_last:
            self.broadcast_shape = [1] * ndims
            for ax in axis:
                self.broadcast_shape[ax] = input_shape[ax]

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # Apply dynamic tanh transformation
        outputs = tf.tanh(self.alpha * inputs)

        # Apply affine transformation based on the channel dimension
        if self.channels_last:
            # For channels-last format, directly apply weight and bias
            outputs = outputs * self.weight + self.bias
        else:
            # For channels-first format, use pre-computed broadcast shape
            weight_broadcasted = tf.reshape(self.weight, self.broadcast_shape)
            bias_broadcasted = tf.reshape(self.bias, self.broadcast_shape)
            outputs = outputs * weight_broadcasted + bias_broadcasted

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = {
            'axis': self.axis,
            'channels_last': self.channels_last,
            'alpha_init_value': self.alpha_init_value,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(DynamicTanh, self).get_config()
        return {**base_config, **config}

# ---------------------------------------------------------------------


