"""
Squeeze-and-Excitation Layer
===========================

Modern Keras 3 implementation of the Squeeze-and-Excitation (SE) block as described in:
"Squeeze-and-Excitation Networks" (Hu et al., 2019)
https://arxiv.org/abs/1709.01507

This implementation follows modern Keras 3 patterns while acknowledging that SE blocks
fundamentally require input channel information to create properly sized sub-layers.

Architecture:
------------
The SE block consists of:
1. Global average pooling (squeeze)
2. Two-layer bottleneck (excitation):
   - Dimensionality reduction (1x1 conv)
   - Configurable activation (default: ReLU)
   - Dimensionality restoration (1x1 conv)
   - Sigmoid activation
3. Channel-wise scaling

The computation flow is:
input -> global_pool -> conv_reduce -> activation -> conv_restore -> sigmoid -> scale -> output

Design Note:
-----------
This layer creates sub-layers in build() because the SE mechanism fundamentally
requires knowledge of input channels to determine bottleneck dimensions. This is
a necessary exception to the general Keras 3 pattern, handled with extra care
for serialization robustness.
"""

import keras
from typing import Dict, Optional, Tuple, Union, Callable, Any
from keras import layers, activations, ops, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SqueezeExcitation(layers.Layer):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.

    This layer implements the Squeeze-and-Excitation mechanism that adaptively
    recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels. The SE block enhances the representational
    power of a network by enabling it to perform feature recalibration, through
    which it can learn to use global information to selectively emphasise
    informative features and suppress less useful ones.

    The SE block operates by:
    1. **Squeeze**: Global information embedding via global average pooling
    2. **Excitation**: Adaptive recalibration via a bottleneck transformation
    3. **Scale**: Feature recalibration through channel-wise multiplication

    Mathematical formulation:
        Given input X ∈ R^(H×W×C), the SE block computes:

        z = GlobalAvgPool(X)  # Shape: (1, 1, C)
        s = σ(W₂ · δ(W₁ · z))  # Shape: (1, 1, C)
        output = X ⊙ s  # Element-wise multiplication

    Where δ is the reduction activation, σ is sigmoid, W₁ reduces channels by
    reduction_ratio, and W₂ restores original channel count.

    Args:
        reduction_ratio: Float between 0 and 1, determining the bottleneck width.
            Controls the capacity and computational cost of the SE block.
            Smaller values create tighter bottlenecks. Must be positive.
            Defaults to 0.25.
        activation: Activation function for the reduction layer. Can be string
            identifier ('relu', 'swish', 'gelu') or callable. The final
            activation is always sigmoid. Defaults to 'relu'.
        use_bias: Boolean, whether convolution layers use bias vectors.
            Defaults to False as recommended in the original paper.
        kernel_initializer: Initializer for convolution kernel weights.
            Accepts string names ('glorot_normal', 'he_normal') or Initializer
            instances. Defaults to 'glorot_normal'.
        kernel_regularizer: Optional regularizer applied to kernel weights.
            Accepts string names ('l1', 'l2') or Regularizer instances.
            Defaults to None.
        bias_initializer: Initializer for bias vectors (if use_bias=True).
            Defaults to 'zeros'.
        bias_regularizer: Optional regularizer applied to bias vectors.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: ``(batch_size, height, width, channels)``

    Output shape:
        4D tensor with shape: ``(batch_size, height, width, channels)``
        Same shape as input.

    Attributes:
        global_pool: GlobalAveragePooling2D layer for squeeze operation.
        conv_reduce: Conv2D layer for channel dimensionality reduction.
        conv_restore: Conv2D layer for channel dimensionality restoration.
        reduction_activation: Activation function applied after reduction.
        input_channels: Number of input channels (set during build).
        bottleneck_channels: Number of bottleneck channels (set during build).

    Example:
        ```python
        # Basic usage - channels inferred from input
        inputs = keras.Input(shape=(32, 32, 64))
        se_layer = SqueezeExcitation(reduction_ratio=0.25)
        outputs = se_layer(inputs)
        print(outputs.shape)  # (None, 32, 32, 64)

        # Advanced configuration
        se_layer = SqueezeExcitation(
            reduction_ratio=0.125,  # Tighter bottleneck
            activation='swish',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a ResNet-style model
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(64, 7, strides=2, activation='relu')(inputs)
        x = SqueezeExcitation(reduction_ratio=0.25)(x)
        x = keras.layers.Conv2D(128, 3, activation='relu')(x)
        x = SqueezeExcitation(reduction_ratio=0.25)(x)
        outputs = keras.layers.GlobalAveragePooling2D()(x)
        model = keras.Model(inputs, outputs)
        ```

    References:
        - Squeeze-and-Excitation Networks, Hu et al., 2018
        - https://arxiv.org/abs/1709.01507

    Raises:
        ValueError: If reduction_ratio is not in (0, 1] or input shape is invalid.

    Note:
        This layer creates sub-layers in build() rather than __init__() because
        the SE mechanism requires knowledge of input channels. Extra care is taken
        to ensure robust serialization by explicitly building all sub-layers.
        The original paper recommends reduction_ratio=0.25 for most cases.
    """

    def __init__(
        self,
        reduction_ratio: float = 0.25,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'relu',
        use_bias: bool = False,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_normal',
        kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not 0 < reduction_ratio <= 1.0:
            raise ValueError(
                f"reduction_ratio must be in range (0, 1], got {reduction_ratio}"
            )

        # Store ALL configuration parameters
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Get activation function
        self.reduction_activation = activations.get(activation)

        # Shape-dependent attributes (set during build)
        self.input_channels: Optional[int] = None
        self.bottleneck_channels: Optional[int] = None

        # Sub-layers (created during build due to shape dependency)
        self.global_pool: Optional[layers.GlobalAveragePooling2D] = None
        self.conv_reduce: Optional[layers.Conv2D] = None
        self.conv_restore: Optional[layers.Conv2D] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create all sub-layers.

        This method creates sub-layers based on input shape since the SE block
        fundamentally needs to know the number of input channels. Extra care
        is taken to ensure robust serialization by explicitly building all
        sub-layers.

        Args:
            input_shape: Shape of the input tensor.
                Expected format: (batch_size, height, width, channels)

        Raises:
            ValueError: If input_shape is invalid or channels are undefined.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        self.input_channels = input_shape[-1]
        if self.input_channels is None:
            raise ValueError("Last dimension (channels) of input must be defined")

        # Calculate bottleneck channels
        self.bottleneck_channels = max(1, int(round(
            self.input_channels * self.reduction_ratio
        )))

        logger.info(
            f"Building SqueezeExcitation: input_channels={self.input_channels}, "
            f"bottleneck_channels={self.bottleneck_channels}"
        )

        # CREATE all sub-layers (necessary exception to general Keras 3 pattern)
        self.global_pool = layers.GlobalAveragePooling2D(
            keepdims=True,
            name='global_pool'
        )

        self.conv_reduce = layers.Conv2D(
            filters=self.bottleneck_channels,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            name='conv_reduce'
        )

        self.conv_restore = layers.Conv2D(
            filters=self.input_channels,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            name='conv_restore'
        )

        # BUILD all sub-layers explicitly for robust serialization
        # This is CRITICAL for proper weight loading after model deserialization
        pooled_shape = (input_shape[0], 1, 1, self.input_channels)

        self.global_pool.build(input_shape)
        self.conv_reduce.build(pooled_shape)

        reduced_shape = (pooled_shape[0], 1, 1, self.bottleneck_channels)
        self.conv_restore.build(reduced_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the SE block.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Passed to sub-layers that
                support training behavior.

        Returns:
            Output tensor after applying SE operations with same shape as input.

        Raises:
            RuntimeError: If layer hasn't been built (sub-layers are None).
        """
        # Ensure layer is built
        if (self.global_pool is None or
            self.conv_reduce is None or
            self.conv_restore is None):
            raise RuntimeError(
                "Layer must be built before calling. "
                "This usually happens automatically on first call."
            )

        # Squeeze: Global average pooling to capture channel statistics
        squeezed = self.global_pool(inputs)

        # Excitation: Two-step channel recalibration
        # Step 1: Dimensionality reduction with activation
        excited = self.conv_reduce(squeezed, training=training)
        excited = self.reduction_activation(excited)

        # Step 2: Dimensionality restoration with sigmoid gating
        excited = self.conv_restore(excited, training=training)
        attention_weights = activations.sigmoid(excited)

        # Scale: Apply learned attention weights to original features
        return ops.multiply(inputs, attention_weights)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (same as input shape for SE blocks).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        This method returns all parameters needed to reconstruct the layer.
        Note that shape-dependent attributes (input_channels, bottleneck_channels)
        are not included as they are derived during build().

        Returns:
            Dictionary containing all configuration parameters needed to
            reconstruct the layer.
        """
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'activation': activations.serialize(self.reduction_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config