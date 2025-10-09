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
from typing import Optional, Union, Tuple, Any, List

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HANCLayer(keras.layers.Layer):
    """
    Hierarchical Aggregation of Neighborhood Context (HANC) Layer.

    This layer implements hierarchical context aggregation by computing average
    and max pooling at multiple scales (2×2, 4×4, 8×8, 16×16, 32×32) and concatenating
    them along the channel dimension. This provides an approximate version of
    self-attention by comparing pixels with neighborhood statistics at multiple scales.

    The layer concatenates:
    - Original features
    - Average pooled features at k-1 different scales (upsampled back)
    - Max pooled features at k-1 different scales (upsampled back)

    Total output channels after concatenation = input_channels × (2×k - 1)
    Final output channels = out_channels (after 1×1 convolution)

    Args:
        in_channels: Integer, number of input channels. Must be positive.
        out_channels: Integer, number of output channels after final 1×1 convolution.
            Must be positive.
        k: Integer, number of hierarchical levels. Must be between 1 and 5.
            k=1 means no pooling (original only), k=2 adds 2×2 pooling,
            k=3 adds 2×2 and 4×4, etc. Higher k values provide more contextual
            information but increase computational cost.
        kernel_initializer: String or Initializer, initializer for the 1×1 convolution kernel.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional Regularizer, regularizer for the 1×1 convolution kernel.
            Defaults to None.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, in_channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, out_channels).

    Attributes:
        conv: 1×1 convolution layer for dimensional compression.
        batch_norm: Batch normalization layer for training stability.
        activation: LeakyReLU activation function.
        avg_pooling_layers: List of average pooling layers for each scale.
        max_pooling_layers: List of max pooling layers for each scale.
        concatenate: Concatenation layer for combining multi-scale features.

    Example:
        ```python
        # Basic usage with default parameters
        hanc = HANCLayer(in_channels=64, out_channels=64, k=3)

        # Custom configuration with regularization
        hanc = HANCLayer(
            in_channels=128,
            out_channels=128,
            k=4,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(32, 32, 64))
        outputs = HANCLayer(in_channels=64, out_channels=64, k=3)(inputs)
        ```

    Raises:
        ValueError: If in_channels is not positive.
        ValueError: If out_channels is not positive.
        ValueError: If k is not between 1 and 5.

    Note:
        Higher k values provide more contextual information but increase
        computational cost and memory usage. k=3 (up to 8×8 patches) is
        recommended for most applications as it provides comprehensive
        context modeling with reasonable computational overhead.

        The layer automatically handles spatial dimension alignment through
        `keras.ops.image.resize`, making it robust to dynamic input shapes.
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

        # Store ALL configuration parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Calculate total channels after concatenation
        self.total_channels = in_channels * (2 * k - 1)

        # CREATE all sub-layers in __init__ (following Modern Keras 3 pattern)

        # Main processing layers
        self.conv = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='hanc_conv'
        )

        self.batch_norm = keras.layers.BatchNormalization(name='hanc_bn')
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='hanc_activation')

        # Concatenation layer
        self.concatenate = keras.layers.Concatenate(axis=-1, name='hanc_concat')

        # Pre-create pooling layers for all possible scales (1 to k-1)
        max_k = 5  # Maximum supported k value
        self.avg_pooling_layers: List[keras.layers.Layer] = []
        self.max_pooling_layers: List[keras.layers.Layer] = []

        for scale in range(1, max_k):  # scales 1, 2, 3, 4 (for k up to 5)
            pool_size = 2 ** scale  # 2, 4, 8, 16

            # Average pooling
            avg_pool = keras.layers.AveragePooling2D(
                pool_size=pool_size,
                strides=pool_size,
                padding='same',
                name=f'hanc_avg_pool_{scale}'
            )

            # Max pooling
            max_pool = keras.layers.MaxPooling2D(
                pool_size=pool_size,
                strides=pool_size,
                padding='same',
                name=f'hanc_max_pool_{scale}'
            )

            self.avg_pooling_layers.append(avg_pool)
            self.max_pooling_layers.append(max_pool)


    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration during
        model loading.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, "
                f"got {input_shape[-1]}"
            )

        # Build main processing layers
        # For concatenation, we need to compute the expected concatenated shape
        concat_channels = self.in_channels * (2 * self.k - 1)
        concat_shape = tuple(input_shape[:-1]) + (concat_channels,)

        self.conv.build(concat_shape)
        conv_output_shape = self.conv.compute_output_shape(concat_shape)
        self.batch_norm.build(conv_output_shape)

        # Build pooling and upsampling layers that will be used (based on k)
        for scale in range(min(self.k - 1, len(self.avg_pooling_layers))):
            self.avg_pooling_layers[scale].build(input_shape)
            self.max_pooling_layers[scale].build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation through hierarchical context aggregation.

        Implements the six-stage processing pipeline:
        1. Original feature preservation
        2. Multi-scale average pooling and upsampling
        3. Multi-scale max pooling and upsampling
        4. Hierarchical concatenation
        5. Dimensional compression via 1×1 convolution
        6. Feature normalization and activation

        Args:
            inputs: Input tensor of shape (batch_size, height, width, in_channels).
            training: Boolean indicating training mode for batch normalization.

        Returns:
            Output tensor of shape (batch_size, height, width, out_channels).
        """
        # Stage 1: Start with original features
        features_list = [inputs]

        if self.k == 1:
            # No hierarchical pooling, just use original features
            concatenated = inputs
        else:
            # Get target shape for resizing, compatible with graph mode
            target_shape = ops.shape(inputs)
            target_height, target_width = target_shape[1], target_shape[2]

            # Stage 2 & 3: Add average and max pooled features at different scales
            for scale in range(self.k - 1):  # scales 0 to k-2, representing 2^1 to 2^(k-1)
                # Average pooling path
                avg_pooled = self.avg_pooling_layers[scale](inputs)
                avg_upsampled = ops.image.resize(
                    avg_pooled,
                    size=(target_height, target_width),
                    interpolation='nearest'
                )
                features_list.append(avg_upsampled)

                # Max pooling path
                max_pooled = self.max_pooling_layers[scale](inputs)
                max_upsampled = ops.image.resize(
                    max_pooled,
                    size=(target_height, target_width),
                    interpolation='nearest'
                )
                features_list.append(max_upsampled)

            # Stage 4: Hierarchical concatenation along channel dimension
            concatenated = self.concatenate(features_list)

        # Stage 5: Dimensional compression via 1×1 convolution
        x = self.conv(concatenated)

        # Stage 6: Feature normalization and activation
        x = self.batch_norm(x, training=training)
        x = self.activation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple with same spatial dimensions and out_channels.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        return tuple(list(input_shape[:-1]) + [self.out_channels])

    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.

        Returns ALL constructor parameters for proper serialization/deserialization.
        This is critical for model saving and loading.

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