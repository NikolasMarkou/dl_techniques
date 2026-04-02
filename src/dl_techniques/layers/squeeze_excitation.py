"""
Implement a Squeeze-and-Excitation block for channel-wise feature recalibration.

This layer introduces a mechanism for adaptive, channel-wise feature
recalibration. It enhances the representational power of a convolutional
network by explicitly modeling the interdependencies between channels. The core
idea is to use global information from the entire feature map to selectively
emphasize informative feature channels and suppress less useful ones, allowing
the network to learn what to "pay attention to."

Architectural and Mathematical Underpinnings:

The Squeeze-and-Excitation (SE) block is a lightweight computational unit that
can be integrated into any standard convolutional block. It operates in three
distinct stages:

1.  **Squeeze (Global Information Embedding)**: The first step aggregates global
    spatial information into a channel descriptor. This is achieved by applying
    Global Average Pooling (GAP) to the input feature map `X ∈ ℝ^(H×W×C)`,
    producing a vector `z ∈ ℝ^(1×1×C)`. Each element `z_c` is the mean
    activation for the c-th channel across all spatial locations:

        z_c = (1 / (H * W)) * Σ_{i=1 to H} Σ_{j=1 to W} X_c(i, j)

    This operation effectively creates a compact summary of the global receptive
    field for each channel.

2.  **Excitation (Adaptive Recalibration)**: This stage learns a non-linear,
    non-mutually-exclusive relationship between channels. The channel
    descriptor `z` is passed through a simple gating mechanism, typically a
    two-layer bottleneck MLP implemented with 1x1 convolutions. This network
    first reduces the channel dimension by a `reduction_ratio` `r` and then
    restores it, followed by a sigmoid activation to produce a set of channel
    weights `s ∈ ℝ^(1×1×C)`:

        s = σ(W₂ * δ(W₁ * z))

    Here, `W₁ ∈ ℝ^((C/r)×C)` and `W₂ ∈ ℝ^(C×(C/r))` are the weights of the two
    convolutional layers, `δ` is a ReLU activation, and `σ` is the sigmoid
    function. The sigmoid ensures that the learned weights are normalized
    between 0 and 1, representing the relative importance of each channel.

3.  **Scale (Feature Recalibration)**: The final step applies the learned channel
    weights to the original input feature map. The output of the SE block is
    obtained by rescaling the input `X` with the activations `s`:

        X_scaled_c = s_c * X_c

    This is an element-wise multiplication where the scalar `s_c` (the c-th
    element of `s`) is broadcast across the entire spatial extent of the
    corresponding input channel `X_c`. This operation adaptively modulates the
    activations of each feature channel based on the global context captured
    by the squeeze-and-excitation mechanism.

References:
    - Hu, J., et al. (2018). Squeeze-and-Excitation Networks. *CVPR*.
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
    interdependencies between channels. Given input ``X``, the SE block
    computes: ``z = GAP(X)``, ``s = sigmoid(W2 * act(W1 * z))``,
    ``output = X * s``, where ``W1`` reduces channels by ``reduction_ratio``
    and ``W2`` restores the original channel count. The layer supports 2D, 3D,
    and 4D inputs by internally expanding to 4D for the convolutional
    infrastructure.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────┐
        │  Input [B, ..., C]                 │
        └──────────────┬─────────────────────┘
                       ▼
        ┌────────────────────────────────────┐
        │  Squeeze: GlobalAvgPool → [B,1,1,C]│
        └──────────────┬─────────────────────┘
                       ▼
        ┌────────────────────────────────────┐
        │  Excitation:                       │
        │    Conv1x1(C→C*r) → Activation     │
        │    Conv1x1(C*r→C) → Sigmoid        │
        │    → attention weights [B,1,1,C]   │
        └──────────────┬─────────────────────┘
                       ▼
        ┌────────────────────────────────────┐
        │  Scale: Input * attention_weights  │
        │  → Output [B, ..., C]              │
        └────────────────────────────────────┘

    :param reduction_ratio: Float in ``(0, 1]`` determining the bottleneck width.
        Defaults to 0.25.
    :type reduction_ratio: float
    :param activation: Activation function for the reduction layer. String identifier
        or callable. Final activation is always sigmoid. Defaults to ``'relu'``.
    :type activation: str or callable
    :param use_bias: Whether convolution layers use bias vectors. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Initializer for convolution kernel weights.
        Defaults to ``'glorot_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_initializer: Initializer for bias vectors. Defaults to ``'zeros'``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param bias_regularizer: Optional regularizer for bias vectors.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
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

        :param input_shape: Shape of the input tensor in format
            ``(batch, [spatial_dims...], channels)``.
        :type input_shape: tuple
        """
        if len(input_shape) not in (2, 3, 4):
            raise ValueError(
                f"Expected 2D (B, C), 3D (B, S, C), or 4D (B, H, W, C) input shape, "
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
        # Determine internal shape seen by sublayers after expansion
        if len(input_shape) == 2:
            # (B, C) -> (B, 1, 1, C)
            internal_shape = (input_shape[0], 1, 1, self.input_channels)
        elif len(input_shape) == 3:
            # (B, S, C) -> (B, S, 1, C)
            internal_shape = (input_shape[0], input_shape[1], 1, self.input_channels)
        else:
            # (B, H, W, C)
            internal_shape = input_shape

        self.global_pool.build(internal_shape)

        # The output of GAP is always (B, 1, 1, C)
        pooled_shape = (input_shape[0], 1, 1, self.input_channels)
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

        :param inputs: Input tensor of shape ``(B, C)``, ``(B, S, C)``, or ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode.
        :type training: bool or None
        :return: Output tensor with same shape as input after SE recalibration.
        :rtype: keras.KerasTensor
        """
        # Ensure layer is built
        if (self.global_pool is None or
            self.conv_reduce is None or
            self.conv_restore is None):
            raise RuntimeError(
                "Layer must be built before calling. "
                "This usually happens automatically on first call."
            )

        # Handle dimension expansion for 2D/3D inputs
        x = inputs
        input_rank = len(inputs.shape)

        if input_rank == 2:
            # (B, C) -> (B, 1, 1, C)
            x = ops.expand_dims(x, axis=1)
            x = ops.expand_dims(x, axis=1)
        elif input_rank == 3:
            # (B, S, C) -> (B, S, 1, C)
            x = ops.expand_dims(x, axis=2)

        # Squeeze: Global average pooling to capture channel statistics
        # Input to global_pool is effectively 4D
        squeezed = self.global_pool(x)

        # Excitation: Two-step channel recalibration
        # Step 1: Dimensionality reduction with activation
        excited = self.conv_reduce(squeezed, training=training)
        excited = self.reduction_activation(excited)

        # Step 2: Dimensionality restoration with sigmoid gating
        excited = self.conv_restore(excited, training=training)
        attention_weights = activations.sigmoid(excited)

        # Scale: Apply learned attention weights to original features
        output = ops.multiply(x, attention_weights)

        # Restore original dimensions
        if input_rank == 2:
            output = ops.squeeze(output, axis=[1, 2])
        elif input_rank == 3:
            output = ops.squeeze(output, axis=2)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Dictionary containing all configuration parameters.
        :rtype: dict
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

# ---------------------------------------------------------------------
