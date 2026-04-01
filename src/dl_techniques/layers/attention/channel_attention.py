"""
Channel-wise attention weights for convolutional feature maps.

This module implements the channel attention mechanism from the Convolutional
Block Attention Module (CBAM). Its purpose is to learn the importance of
each feature channel in a convolutional network, allowing the model to
dynamically re-weight channels to focus on the most informative features.
It answers the question of "what" is important in the input features.

Architecture:
    The core principle is to aggregate spatial information from each
    channel into a compact channel descriptor and then use these descriptors
    to learn the non-linear, cross-channel relationships. This is achieved
    through a dual-path architecture:

    1.  **Spatial Information Aggregation:** The module processes the input
        feature map through two parallel global pooling operations to create
        two distinct channel descriptors:
        -   **Global Average Pooling:** Captures the overall statistical
            distribution and global context of each feature channel.
        -   **Global Max Pooling:** Captures the most salient, high-activation
            part of each feature channel, representing its most distinctive
            local feature.

    2.  **Shared Multi-Layer Perceptron (MLP):** Both channel descriptors are
        then fed through the *same* lightweight MLP. This MLP, which
        consists of a bottleneck structure (a reduction layer followed by an
        expansion layer), learns to model the complex interdependencies
        between channels. Sharing the MLP for both descriptors reduces
        parameters and encourages the learning of a more general relationship
        model.

    3.  **Merging and Activation:** The output feature vectors from the
        shared MLP are merged via element-wise summation. This combined
        vector is then passed through a sigmoid activation function to
        produce the final channel attention weights, scaled between 0 and 1.

Foundational Mathematics:
    The channel attention map `M_c` for an input feature map `F` is computed
    as follows:

        M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )

    where `σ` is the sigmoid function. The MLP consists of two weight
    matrices, `W_0` (for dimensionality reduction) and `W_1` (for
    expansion), shared across both paths:

        M_c(F) = σ( W_1(ReLU(W_0(F_avg))) + W_1(ReLU(W_0(F_max))) )

    Here, `F_avg` and `F_max` are the channel descriptors produced by
    average and max pooling, respectively. This formulation allows the
    model to learn which channels to emphasize or suppress based on a
    combination of their global context and most salient features.

References:
    - The foundational paper for this module:
      Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). "CBAM:
      Convolutional Block Attention Module". European Conference on
      Computer Vision (ECCV).
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    """
    Channel attention module from CBAM that learns per-channel importance weights.

    Implements the channel attention mechanism from the Convolutional Block
    Attention Module (CBAM). The module aggregates spatial information via
    dual-path global pooling (average and max), processes both descriptors
    through a shared bottleneck MLP, and produces sigmoid-activated channel
    weights via ``M_c(F) = sigma(W_1(ReLU(W_0(F_avg))) + W_1(ReLU(W_0(F_max))))``.

    **Architecture Overview:**

    .. code-block:: text

        Input [B, H, W, C]
              │
              ├──────────────────────┐
              ▼                      ▼
        ┌───────────────┐   ┌───────────────┐
        │ Global AvgPool│   │ Global MaxPool│
        │  (H,W) → (1,1)│   │  (H,W) → (1,1)│
        └───────┬───────┘   └───────┬───────┘
                ▼                    ▼
          [B, C] (flat)        [B, C] (flat)
                │                    │
                ├────── Shared ──────┤
                ▼       MLP         ▼
        ┌───────────────┐   ┌───────────────┐
        │ Dense(C//r)   │   │ Dense(C//r)   │
        │ + ReLU        │   │ + ReLU        │
        ├───────────────┤   ├───────────────┤
        │ Dense(C)      │   │ Dense(C)      │
        └───────┬───────┘   └───────┬───────┘
                │                    │
                └────── Add ─────────┘
                         ▼
                    ┌─────────┐
                    │ Sigmoid │
                    └────┬────┘
                         ▼
                  [B, 1, 1, C] weights

    :param channels: Number of input channels. Must be positive and
        divisible by ``ratio``.
    :type channels: int
    :param ratio: Reduction ratio for the shared MLP bottleneck dimension
        (``channels // ratio``). Must be positive and divide evenly into
        ``channels``. Defaults to 8.
    :type ratio: int
    :param kernel_initializer: Initializer for the dense layer kernels.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the dense layer
        kernels. Defaults to ``None``.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param use_bias: Whether to include bias in dense layers.
        Defaults to ``False``.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``channels`` is not positive.
    :raises ValueError: If ``ratio`` is not positive or does not divide
        evenly into ``channels``.
    """

    def __init__(
            self,
            channels: int,
            ratio: int = 8,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if channels % ratio != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by ratio ({ratio})"
            )

        # Store configuration
        self.channels = channels
        self.ratio = ratio
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # CREATE sub-layers in __init__ following modern Keras 3 pattern
        # These layers are unbuilt at this point
        self.dense1 = keras.layers.Dense(
            units=channels // ratio,
            activation='relu',
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='channel_attention_dense_1'
        )

        self.dense2 = keras.layers.Dense(
            units=channels,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='channel_attention_dense_2'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization, ensuring
        weight variables exist before weight restoration during model loading.

        :param input_shape: Shape tuple of the input tensor. Expected to be
            ``(batch_size, height, width, channels)``.
        :type input_shape: tuple
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        if input_shape[-1] != self.channels:
            raise ValueError(
                f"Expected input channels ({input_shape[-1]}) to match "
                f"layer channels ({self.channels})"
            )

        # Shape for MLP input after global pooling: (batch_size, channels)
        mlp_input_shape = (input_shape[0], self.channels)

        # Build sub-layers in computational order for robust serialization
        self.dense1.build(mlp_input_shape)

        # Compute intermediate shape after first dense layer
        dense1_output_shape = self.dense1.compute_output_shape(mlp_input_shape)

        # Build second dense layer
        self.dense2.build(dense1_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply channel attention to the input tensor.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: bool or None
        :return: Channel attention weights of shape
            ``(batch_size, 1, 1, channels)``.
        :rtype: keras.KerasTensor
        """
        # Apply global pooling operations
        # Shape: (batch_size, height, width, channels) -> (batch_size, 1, 1, channels)
        avg_pool = keras.ops.mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = keras.ops.max(inputs, axis=[1, 2], keepdims=True)

        # Reshape for MLP processing
        # Shape: (batch_size, 1, 1, channels) -> (batch_size, channels)
        avg_pool_flat = keras.ops.reshape(avg_pool, (-1, self.channels))
        max_pool_flat = keras.ops.reshape(max_pool, (-1, self.channels))

        # Pass through shared MLP
        avg_out = self.dense1(avg_pool_flat, training=training)
        avg_out = self.dense2(avg_out, training=training)

        max_out = self.dense1(max_pool_flat, training=training)
        max_out = self.dense2(max_out, training=training)

        # Combine outputs and apply sigmoid activation
        # Shape: (batch_size, channels)
        attention_weights = keras.ops.sigmoid(avg_out + max_out)

        # Reshape back to spatial format for broadcasting
        # Shape: (batch_size, channels) -> (batch_size, 1, 1, channels)
        attention_weights = keras.ops.reshape(
            attention_weights, (-1, 1, 1, self.channels)
        )

        return attention_weights

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple ``(batch_size, 1, 1, channels)``.
        :rtype: tuple
        """
        return (input_shape[0], 1, 1, self.channels)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config
