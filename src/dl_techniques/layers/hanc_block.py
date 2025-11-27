"""
Model long-range dependencies using hierarchical context aggregation.

This block is the core component of the ACC-UNet architecture, designed to
bridge the gap between the efficiency of Convolutional Neural Networks (CNNs)
and the global context modeling capabilities of Transformers. It addresses the
inherent limitation of standard convolutions—their restricted receptive
field—by introducing a novel mechanism to efficiently aggregate multi-scale
contextual information.

Architecturally, the HANC block synthesizes several successful design
principles from modern deep learning into a coherent structure. It adopts the
inverted bottleneck from MobileNetV2, expanding the channel dimension with a
1x1 convolution to create a richer feature space for processing. Spatial
feature extraction is then performed efficiently using a depthwise separable
convolution. The block's central innovation, however, is the Hierarchical
Aggregation of Neighborhood Context (HANC) layer, which is subsequently
followed by a projection 1x1 convolution, a Squeeze-and-Excitation (SE)
module for channel recalibration, and a residual connection for stable
gradient flow.

The foundational mathematical concept is a convolutional approximation of the
self-attention mechanism found in Transformers. Standard self-attention
computes a pixel's relationship to every other pixel, incurring a quadratic
computational cost `O(N^2)` with respect to the number of pixels `N`. The
HANC mechanism circumvents this by reformulating the problem: instead of
all-to-all comparisons, each pixel's feature representation is enriched by
comparing it to statistical summaries of its surrounding neighborhoods at
multiple scales. The process is as follows:

1.  **Hierarchical Pooling:** For `k` different scales (e.g., corresponding
    to 2x2, 4x4, 8x8 receptive fields), the feature map is downsampled using
    both average and max pooling to create a set of low-resolution context
    maps. These maps summarize the feature statistics at different levels of
    granularity.

2.  **Context Concatenation:** These multi-scale context maps are concatenated
    with the original, full-resolution feature map along the channel axis. This
    creates an augmented representation where each pixel is now explicitly
    aware of not just its own features, but also the average and maximum
    feature values in its local, medium, and large-scale neighborhoods.

3.  **Learned Aggregation:** A final 1x1 convolution processes this augmented
    tensor. This step acts as a learned, weighted aggregator, allowing the
    model to determine the optimal way to combine local information with the
    multi-scale contextual signals.

This approach provides a powerful proxy for global context with computational
complexity that is linear with respect to the number of scales `k`, making it
far more efficient than true self-attention for high-resolution inputs.

References:
    - Yan et al., 2023. ACC-UNet: An adaptive context and contrast-aware UNet
      for seismic facies identification. (Inspired this architecture)
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear
      Bottlenecks. (Inverted bottleneck concept)
    - Hu et al., 2018. Squeeze-and-Excitation Networks. (Channel attention)

"""

import keras
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .hanc_layer import HANCLayer
from .squeeze_excitation import SqueezeExcitation

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HANCBlock(keras.layers.Layer):
    """
    Hierarchical Aggregation of Neighborhood Context (HANC) Block.

    This block implements the main building block from ACC-UNet, providing
    long-range dependencies through hierarchical pooling operations while
    maintaining efficiency through depthwise separable convolutions.

    **Intent**: To serve as a drop-in replacement for standard convolutional
    blocks or Transformer blocks in U-Net architectures, providing global
    context awareness with linear computational complexity O(k).

    **Architecture**:
    ```
    Input(shape=[..., input_channels])
           ↓
    Expand: Conv1x1(expanded_channels) → BN → LeakyReLU
           ↓
    Spatial: DepthwiseConv3x3 → BN → LeakyReLU
           ↓
    Context: HANCLayer(k levels)
           ↓
    Residual: + Input (if input_channels == filters) → BN
           ↓
    Project: Conv1x1(filters) → BN → LeakyReLU
           ↓
    Calibrate: SqueezeExcitation
           ↓
    Output(shape=[..., filters])
    ```

    **Mathematical Operation**:
    The block approximates self-attention by aggregating multi-scale statistics:
    1.  **Expansion**: $X_{exp} = \sigma(BN(W_{exp} * X))$
    2.  **Spatial**: $X_{dw} = \sigma(BN(W_{dw} * X_{exp}))$
    3.  **HANC**: $X_{ctx} = \text{Agg}(\{P_s(X_{dw})\}_{s=1}^k)$
        where $P_s$ are pooling operations at scale $s$.
    4.  **Projection**: $Y = \text{SE}(\sigma(BN(W_{proj} * X_{ctx})))$

    Args:
        filters: Integer, number of output filters. Must be positive.
        input_channels: Integer, number of input channels. Must be positive.
        k: Integer, hierarchical levels for HANC operation (1-5 supported).
            Determines the granularity of context aggregation.
        inv_factor: Integer, inverted bottleneck expansion factor.
            Determines the channel width of the internal processing.
        kernel_initializer: Initializer for convolution kernels.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer for convolution kernels.
        bias_regularizer: Optional Regularizer for bias vectors.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape `(batch_size, height, width, input_channels)`.

    Output shape:
        4D tensor with shape `(batch_size, height, width, filters)`.

    Attributes:
        expand_conv: 1x1 convolution for channel expansion.
        depthwise_conv: 3x3 depthwise convolution for spatial processing.
        hanc_layer: Hierarchical context aggregation layer.
        output_conv: 1x1 convolution for output projection.
        squeeze_excitation: Squeeze-Excitation attention mechanism.

    Example:
        ```python
        # Basic usage in a model
        inputs = keras.Input(shape=(256, 256, 32))
        x = HANCBlock(
            filters=64,
            input_channels=32,
            k=3,
            inv_factor=3
        )(inputs)

        # High-capacity configuration
        x = HANCBlock(
            filters=128,
            input_channels=64,
            k=5,
            inv_factor=4,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )(x)
        ```

    Raises:
        ValueError: If filters, input_channels, or inv_factor are not positive.
        ValueError: If k is not between 1 and 5.
    """

    def __init__(
        self,
        filters: int,
        input_channels: int,
        k: int = 3,
        inv_factor: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if k < 1 or k > 5:
            raise ValueError(f"k must be between 1 and 5, got {k}")
        if inv_factor <= 0:
            raise ValueError(f"inv_factor must be positive, got {inv_factor}")

        # Store ALL configuration parameters
        self.filters = filters
        self.input_channels = input_channels
        self.k = k
        self.inv_factor = inv_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Compute derived parameters
        self.expanded_channels = self.input_channels * self.inv_factor
        self.use_residual = (self.input_channels == self.filters)

        # ---------------------------------------------------------------------
        # CREATE sub-layers (Golden Rule: Create in __init__)
        # ---------------------------------------------------------------------

        # 1. Expansion layers
        self.expand_conv = keras.layers.Conv2D(
            filters=self.expanded_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='expand_conv'
        )
        self.expand_bn = keras.layers.BatchNormalization(name='expand_bn')

        # 2. Depthwise layers
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            use_bias=False,
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name='depthwise_conv'
        )
        self.depthwise_bn = keras.layers.BatchNormalization(name='depthwise_bn')

        # 3. HANC layer
        self.hanc_layer = HANCLayer(
            in_channels=self.expanded_channels,
            out_channels=self.input_channels,
            k=self.k,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='hanc'
        )

        # 4. Residual connection batch norm
        if self.use_residual:
            self.residual_bn = keras.layers.BatchNormalization(name='residual_bn')
        else:
            self.residual_bn = None

        # 5. Output layers
        self.output_conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='output_conv'
        )
        self.output_bn = keras.layers.BatchNormalization(name='output_bn')

        # 6. Squeeze-Excitation
        self.squeeze_excitation = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='se'
        )

        # 7. Activation
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly builds sub-layers in computational order to ensure
        weights are created and shapes are validated before training/loading.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] != self.input_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.input_channels}, "
                f"got {input_shape[-1]}"
            )

        # ---------------------------------------------------------------------
        # BUILD sub-layers (Golden Rule: Build in build)
        # ---------------------------------------------------------------------

        # 1. Expansion
        self.expand_conv.build(input_shape)
        expand_output_shape = self.expand_conv.compute_output_shape(input_shape)
        self.expand_bn.build(expand_output_shape)

        # 2. Depthwise
        self.depthwise_conv.build(expand_output_shape)
        depthwise_output_shape = self.depthwise_conv.compute_output_shape(expand_output_shape)
        self.depthwise_bn.build(depthwise_output_shape)

        # 3. HANC
        self.hanc_layer.build(depthwise_output_shape)
        hanc_output_shape = self.hanc_layer.compute_output_shape(depthwise_output_shape)

        # 4. Residual
        if self.residual_bn is not None:
            self.residual_bn.build(hanc_output_shape)

        # 5. Output Projection
        # Note: Shape is same as HANC output (concatenation logic is handled inside HANCLayer
        # or it preserves spatial dims, input_channels was restored by HANCLayer out_channels)
        output_input_shape = hanc_output_shape
        self.output_conv.build(output_input_shape)
        output_conv_shape = self.output_conv.compute_output_shape(output_input_shape)
        self.output_bn.build(output_conv_shape)

        # 6. SE Block
        self.squeeze_excitation.build(output_conv_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation.

        Args:
            inputs: Input tensor of shape (batch, height, width, input_channels).
            training: Boolean indicating training mode for batch normalization.

        Returns:
            Output tensor of shape (batch, height, width, filters).
        """
        # 1. Expansion phase
        x = self.expand_conv(inputs)
        x = self.expand_bn(x, training=training)
        x = self.activation(x)

        # 2. Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.activation(x)

        # 3. Hierarchical context aggregation
        x = self.hanc_layer(x, training=training)

        # 4. Residual connection (if applicable)
        if self.use_residual and self.residual_bn is not None:
            x = x + inputs
            x = self.residual_bn(x, training=training)

        # 5. Output projection
        x = self.output_conv(x)
        x = self.output_bn(x, training=training)
        x = self.activation(x)

        # 6. Squeeze-Excitation
        x = self.squeeze_excitation(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the layer."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        return tuple(list(input_shape[:-1]) + [self.filters])

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'input_channels': self.input_channels,
            'k': self.k,
            'inv_factor': self.inv_factor,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config