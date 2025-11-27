"""
Approximate self-attention by aggregating hierarchical neighborhood context.

This layer implements a convolutional mechanism to model long-range spatial
dependencies, serving as an efficient alternative to the self-attention
modules found in Vision Transformers. The core challenge this layer addresses
is the trade-off between the local receptive fields of standard convolutions
and the quadratic computational complexity of true self-attention. It achieves
this by reformulating the concept of global context from an all-to-all pixel
comparison to a comparison between each pixel and statistical summaries of its
surrounding neighborhoods at multiple scales.

The architectural design is a form of dynamic, learned spatial pyramid
pooling. The process unfolds as follows:
1.  **Multi-Scale Context Extraction:** For a hierarchy of `k-1` scales
    (e.g., corresponding to 2x2, 4x4, 8x8 receptive fields), the input
    feature map is downsampled using two complementary pooling operations:
    -   **Average Pooling:** Captures the mean feature response, representing
        the general context or texture of a neighborhood.
    -   **Max Pooling:** Captures the most salient feature activations,
        highlighting dominant edges, corners, or object parts.
2.  **Context Re-Projection:** The downsampled context maps from each scale
    and pooling type are upsampled back to the original input resolution,
    ensuring spatial alignment.
3.  **Hierarchical Feature Fusion:** The original feature map is concatenated
    with all the upsampled context maps along the channel axis. This creates
    an enriched representation where each pixel's feature vector is augmented
    with explicit information about the average and maximum feature values
    in its local, regional, and global surroundings.
4.  **Learned Aggregation:** A final 1x1 convolution processes this wide,
    concatenated tensor. This projection acts as a learned, channel-wise
    attention mechanism, allowing the model to weigh the importance of the
    original features against the contextual summaries from each scale to
    produce the final output.

Mathematically, this layer provides a computationally tractable approximation
of the self-attention operation. While self-attention has a complexity of
`O(N^2)`, where `N` is the number of spatial locations, this hierarchical
aggregation has a complexity linear in `N` (`O(N*k)` where `k` is the number
of scales). It replaces the expensive all-pairs similarity calculation
(`QK^T`) with a highly efficient feature fusion that achieves a similar goal:
making each pixel's representation aware of the broader context in which it
exists.

References:
    - Yan et al., 2023. ACC-UNet: An adaptive context and contrast-aware UNet
      for seismic facies identification.
    - Zhao et al., 2017. Pyramid Scene Parsing Network (PSPNet).
      (For the concept of spatial pyramid pooling)
    - Vaswani et al., 2017. Attention Is All You Need.
      (For the foundational self-attention concept)

"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any, List, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HANCLayer(keras.layers.Layer):
    """
    Hierarchical Aggregation of Neighborhood Context (HANC) Layer.

    This layer approximates global self-attention by aggregating statistical
    summaries (mean and max) from local neighborhoods at multiple scales.
    It combines these multi-scale context features with the original input
    to create a rich, context-aware representation with linear complexity O(k).

    **Intent**: To provide a computationally efficient alternative to
    self-attention for high-resolution feature maps, bridging the gap between
    local convolution and global transformer contexts.

    **Architecture**:
    ```
    Input(shape=[H, W, C])
           ↓
    Path 1: Identity
    Path 2..k:
       ├─ AvgPool(2^s) → Resize(H, W)
       └─ MaxPool(2^s) → Resize(H, W)
           ↓
    Concatenate(axis=-1)  [Channels = C * (2k - 1)]
           ↓
    Conv1x1(out_channels)
           ↓
    BatchNorm → LeakyReLU
           ↓
    Output(shape=[H, W, out_channels])
    ```

    **Mathematical Operations**:
    Let $X$ be the input. For scales $s \in \{1, \dots, k-1\}$:
    1.  $C_{avg}^{(s)} = \text{Up}(\text{AvgPool}_{2^s}(X))$
    2.  $C_{max}^{(s)} = \text{Up}(\text{MaxPool}_{2^s}(X))$
    3.  $X_{concat} = [X, C_{avg}^{(1)}, C_{max}^{(1)}, \dots, C_{avg}^{(k-1)}, C_{max}^{(k-1)}]$
    4.  $Y = \sigma(BN(W * X_{concat}))$

    Args:
        in_channels: Integer, number of input channels. Must be positive.
        out_channels: Integer, number of output channels after projection.
            Must be positive.
        k: Integer, number of hierarchical levels (1-5).
            k=1: Identity only (no pooling).
            k=2: Adds 2x2 pooling context.
            k=3: Adds 2x2 and 4x4 pooling context.
        kernel_initializer: Initializer for the 1x1 convolution kernel.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional Regularizer for the convolution kernel.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape `(batch_size, height, width, in_channels)`.

    Output shape:
        4D tensor with shape `(batch_size, height, width, out_channels)`.

    Attributes:
        conv: 1x1 Convolution for feature fusion and dimension reduction.
        batch_norm: BatchNormalization applied after convolution.
        avg_pooling_layers: List of AveragePooling2D layers.
        max_pooling_layers: List of MaxPooling2D layers.

    Example:
        ```python
        # Standard usage (Context from 2x2 and 4x4 neighborhoods)
        layer = HANCLayer(in_channels=64, out_channels=64, k=3)
        y = layer(x)

        # High-level context (up to 16x16 neighborhoods)
        layer = HANCLayer(
            in_channels=128,
            out_channels=256,
            k=5,
            kernel_regularizer='l2'
        )
        ```

    Raises:
        ValueError: If in_channels, out_channels are not positive.
        ValueError: If k is not between 1 and 5.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            k: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if k < 1 or k > 5:
            raise ValueError(f"k must be between 1 and 5, got {k}")

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Compute derived parameters
        # Original (1) + (Avg + Max) * (k-1) scales
        self.total_concat_channels = in_channels * (1 + 2 * (k - 1))

        # ---------------------------------------------------------------------
        # CREATE sub-layers (Golden Rule: Create in __init__)
        # ---------------------------------------------------------------------

        # 1. Pooling layers for hierarchical scales
        # We create exactly the layers needed for the specified k
        self.avg_pooling_layers: List[keras.layers.Layer] = []
        self.max_pooling_layers: List[keras.layers.Layer] = []

        for scale in range(1, self.k):
            pool_size = 2 ** scale

            self.avg_pooling_layers.append(
                keras.layers.AveragePooling2D(
                    pool_size=pool_size,
                    strides=pool_size,
                    padding='same',
                    name=f'avg_pool_{pool_size}x{pool_size}'
                )
            )

            self.max_pooling_layers.append(
                keras.layers.MaxPooling2D(
                    pool_size=pool_size,
                    strides=pool_size,
                    padding='same',
                    name=f'max_pool_{pool_size}x{pool_size}'
                )
            )

        # 2. Concatenation
        self.concatenate = keras.layers.Concatenate(axis=-1, name='concat_features')

        # 3. Fusion and Projection
        self.conv = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='fusion_conv'
        )

        self.batch_norm = keras.layers.BatchNormalization(name='fusion_bn')
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly builds sub-layers in computational order.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, "
                f"got {input_shape[-1]}"
            )

        # ---------------------------------------------------------------------
        # BUILD sub-layers (Golden Rule: Build in build)
        # ---------------------------------------------------------------------

        # 1. Build pooling layers
        # Pooling layers generally don't have weights, but we build them for completeness
        for avg_pool, max_pool in zip(self.avg_pooling_layers, self.max_pooling_layers):
            avg_pool.build(input_shape)
            max_pool.build(input_shape)

        # 2. Compute shape after concatenation to build the convolution
        # Shape: (Batch, H, W, total_concat_channels)
        concat_shape = tuple(input_shape[:-1]) + (self.total_concat_channels,)

        # 3. Build Convolution
        self.conv.build(concat_shape)
        conv_output_shape = self.conv.compute_output_shape(concat_shape)

        # 4. Build Batch Norm
        self.batch_norm.build(conv_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation.

        Args:
            inputs: Input tensor of shape (batch, height, width, in_channels).
            training: Boolean indicating training mode for batch normalization.

        Returns:
            Output tensor of shape (batch, height, width, out_channels).
        """
        if self.k == 1:
            # Even if k=1, we might need to project channels if in != out
            # However, logic dictates we still proceed through conv/bn logic below
            # treating inputs as the 'concatenated' tensor.
            concatenated = inputs
        else:
            # 1. Gather Multi-scale Context
            features_list = [inputs]

            # Use dynamic shape for resizing
            input_shape = ops.shape(inputs)
            height, width = input_shape[1], input_shape[2]

            for avg_pool, max_pool in zip(self.avg_pooling_layers, self.max_pooling_layers):
                # Average pooling path
                avg_feat = avg_pool(inputs)
                avg_resized = ops.image.resize(
                    avg_feat,
                    size=(height, width),
                    interpolation='nearest'
                )
                features_list.append(avg_resized)

                # Max pooling path
                max_feat = max_pool(inputs)
                max_resized = ops.image.resize(
                    max_feat,
                    size=(height, width),
                    interpolation='nearest'
                )
                features_list.append(max_resized)

            # 2. Concatenate
            concatenated = self.concatenate(features_list)

        # 3. Fusion & Projection
        x = self.conv(concatenated)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        return tuple(list(input_shape[:-1]) + [self.out_channels])

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'k': self.k,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------