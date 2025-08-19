"""
Hierarchical Aggregation of Neighborhood Context (HANC) Block - Core Building Block of ACC-UNet.

This module implements the revolutionary HANC block, which represents the fundamental
architectural innovation of ACC-UNet. By combining transformer-inspired design principles
with convolutional efficiency, HANC blocks provide long-range contextual dependencies
while maintaining the computational advantages and inductive biases of convolutional
neural networks. This design enables ACC-UNet to achieve transformer-level performance
with significantly fewer parameters and computational overhead.

Core Innovation Philosophy:
    The HANC block addresses a fundamental limitation of standard convolutional blocks:
    their restricted receptive field and inability to model long-range dependencies
    efficiently. While transformers excel at capturing global context through self-attention,
    they lack the spatial inductive biases and efficiency of convolutions. HANC blocks
    bridge this gap by implementing a convolutional approximation of self-attention
    through hierarchical neighborhood context aggregation.

    Key insight: Instead of computing expensive attention matrices, HANC blocks approximate
    self-attention by comparing each pixel with statistical summaries (mean and max) of
    its neighborhoods at multiple scales, providing transformer-like global modeling
    with convolutional efficiency.

Architectural Design Principles:
    The HANC block integrates five complementary design elements inspired by modern
    deep learning architectures:

    1. **Inverted Bottleneck Expansion**: Inspired by MobileNetV2 and transformers' MLP layers
       - Expands channels by inv_factor (typically 3-4x) for increased expressivity
       - Provides wider intermediate representation for richer feature learning

    2. **Efficient Convolution**: Uses depthwise separable convolutions for parameter efficiency
       - 3×3 depthwise convolution reduces computational complexity
       - Maintains spatial feature extraction with minimal parameter overhead

    3. **Hierarchical Context Aggregation**: Novel HANC layer for long-range dependencies
       - Multi-scale neighborhood analysis through hierarchical pooling
       - Convolutional approximation of transformer self-attention mechanism

    4. **Residual Learning**: Skip connections for stable gradient flow and feature preservation
       - Enables training of very deep networks without degradation
       - Preserves low-level spatial information throughout processing

    5. **Adaptive Channel Recalibration**: Squeeze-Excitation for feature refinement
       - Learns channel-wise importance weights adaptively
       - Enhances feature discriminability and suppresses irrelevant channels

Detailed Processing Pipeline:
    The HANC block implements a sophisticated seven-stage processing pipeline:

    ```
    Input Features (H×W×C_in)
           ↓
    [1] Channel Expansion: 1×1 Conv (C_in → C_in×inv_factor)
           ↓
    [2] Spatial Processing: 3×3 Depthwise Conv (preserve channels)
           ↓
    [3] Hierarchical Context: HANC Layer (C_in×inv_factor → C_in)
           ↓
    [4] Residual Integration: Addition + Batch Norm (if channels match)
           ↓
    [5] Output Projection: 1×1 Conv (C_in → filters)
           ↓
    [6] Channel Recalibration: Squeeze-Excitation attention
           ↓
    Output Features (H×W×filters)
    ```

Mathematical Formulation:
    For input X ∈ ℝ^(H×W×C_in), the HANC block transformation is:

    ```python
    # Stage 1: Channel expansion with inverted bottleneck
    X_exp = ReLU(BN(Conv1x1(X, C_in → C_in×inv_factor)))

    # Stage 2: Efficient spatial feature extraction
    X_spatial = ReLU(BN(DepthwiseConv3x3(X_exp)))

    # Stage 3: Hierarchical context aggregation (core innovation)
    X_context = HANC(X_spatial, k_levels)  # Multi-scale pooling and aggregation

    # Stage 4: Residual connection (conditional on channel compatibility)
    if C_in == filters:
        X_residual = BN(X_context + X)  # Identity shortcut
    else:
        X_residual = X_context  # No residual when dimensions differ

    # Stage 5: Output projection and feature transformation
    X_proj = ReLU(BN(Conv1x1(X_residual, C_in → filters)))

    # Stage 6: Adaptive channel recalibration
    X_out = SE(X_proj)  # Squeeze-Excitation attention
    ```

Hierarchical Context Aggregation (HANC) Mechanism:
    The core innovation lies in the HANC layer's ability to capture multi-scale context:

    - **Scale Hierarchy**: Analyzes neighborhoods at k different scales: [2¹, 2², ..., 2^(k-1)]
    - **Statistical Aggregation**: Computes mean and max pooling at each scale
    - **Context Compilation**: Concatenates multi-scale statistics with original features
    - **Dimensional Reduction**: Uses 1×1 convolution to compress enriched representation

    This provides O(k) complexity for multi-scale context vs. O(n²) for full self-attention,
    where k≪n for typical feature map sizes.

Adaptive Configuration Strategy:
    HANC blocks use context-aware parameter selection based on network depth and scale:

    - **Shallow Levels (k=3)**: Maximum context aggregation for fine-grained features
      * Processes scales: [2×2, 4×4, 8×8] patches
      * Balances local details with broader spatial context

    - **Intermediate Levels (k=2)**: Moderate context for mid-level representations
      * Processes scales: [2×2, 4×4] patches
      * Focuses on structural patterns and object parts

    - **Deep Levels (k=1)**: Minimal context for high-level semantic features
      * Original features only (no additional pooling)
      * Preserves semantic abstraction without excessive context mixing
"""

import keras
from typing import Optional, Union, Tuple, Any

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

    This block implements the main building block from ACC-UNet, combining:
    1. Inverted bottleneck expansion (1x1 conv to expand channels)
    2. Depthwise 3x3 convolution
    3. Hierarchical context aggregation (HANC layer)
    4. Residual connection (when input/output channels match)
    5. Final 1x1 convolution with Squeeze-Excitation

    The block provides long-range dependencies through hierarchical pooling
    operations while maintaining efficiency through depthwise convolutions.

    Args:
        filters: Integer, number of output filters. Must be positive.
        input_channels: Integer, number of input channels. Must be positive.
        k: Integer, hierarchical levels for HANC operation (1-5 supported).
            Must be between 1 and 5.
        inv_factor: Integer, inverted bottleneck expansion factor. Must be positive.
        kernel_initializer: String or Initializer, initializer for convolution kernels.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer, regularizer for convolution kernels.
            Defaults to None.
        bias_regularizer: Optional Regularizer, regularizer for bias vectors.
            Defaults to None.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, input_channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, filters).

    Attributes:
        expand_conv: 1x1 convolution for channel expansion.
        expand_bn: Batch normalization after expansion.
        depthwise_conv: 3x3 depthwise convolution for spatial processing.
        depthwise_bn: Batch normalization after depthwise convolution.
        hanc_layer: Hierarchical context aggregation layer.
        residual_bn: Batch normalization applied after residual addition.
        output_conv: 1x1 convolution for output projection.
        output_bn: Batch normalization after output projection.
        squeeze_excitation: Squeeze-Excitation attention mechanism.
        activation: LeakyReLU activation function.

    Example:
        ```python
        # Basic usage
        block = HANCBlock(filters=64, input_channels=32, k=3, inv_factor=3)

        # Custom configuration
        block = HANCBlock(
            filters=128,
            input_channels=64,
            k=4,
            inv_factor=4,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(224, 224, 32))
        x = HANCBlock(filters=64, input_channels=32)(inputs)
        ```

    Raises:
        ValueError: If filters is not positive.
        ValueError: If input_channels is not positive.
        ValueError: If k is not between 1 and 5.
        ValueError: If inv_factor is not positive.

    Note:
        The block includes automatic residual connections when input and output
        channels match. The hierarchical context aggregation provides transformer-like
        global modeling with convolutional efficiency.
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

        # CREATE all sub-layers in __init__ (following Modern Keras 3 pattern)

        # 1. Expansion convolution (1x1)
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

        # 2. Depthwise convolution (3x3)
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            use_bias=False,
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name='depthwise_conv'
        )
        self.depthwise_bn = keras.layers.BatchNormalization(name='depthwise_bn')

        # 3. HANC layer for hierarchical context aggregation
        self.hanc_layer = HANCLayer(
            in_channels=self.expanded_channels,
            out_channels=self.input_channels,
            k=self.k,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='hanc'
        )

        # 4. Residual connection batch norm (applied after adding residual)
        if self.use_residual:
            self.residual_bn = keras.layers.BatchNormalization(name='residual_bn')
        else:
            self.residual_bn = None

        # 5. Output convolution (1x1)
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

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration during
        model loading.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] != self.input_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.input_channels}, "
                f"got {input_shape[-1]}"
            )

        # Build sub-layers in computational order

        # 1. Expansion layers
        self.expand_conv.build(input_shape)
        expand_output_shape = self.expand_conv.compute_output_shape(input_shape)
        self.expand_bn.build(expand_output_shape)

        # 2. Depthwise layers
        self.depthwise_conv.build(expand_output_shape)
        depthwise_output_shape = self.depthwise_conv.compute_output_shape(expand_output_shape)
        self.depthwise_bn.build(depthwise_output_shape)

        # 3. HANC layer
        self.hanc_layer.build(depthwise_output_shape)
        hanc_output_shape = self.hanc_layer.compute_output_shape(depthwise_output_shape)

        # 4. Residual batch norm (if applicable)
        if self.residual_bn is not None:
            self.residual_bn.build(hanc_output_shape)

        # 5. Output layers
        output_input_shape = hanc_output_shape  # Same shape after residual
        self.output_conv.build(output_input_shape)
        output_conv_shape = self.output_conv.compute_output_shape(output_input_shape)
        self.output_bn.build(output_conv_shape)

        # 6. Squeeze-Excitation
        self.squeeze_excitation.build(output_conv_shape)

        # 7. Activation doesn't need explicit building

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation through the HANC block.

        Implements the seven-stage processing pipeline:
        1. Channel expansion via 1x1 convolution
        2. Spatial processing via depthwise convolution
        3. Hierarchical context aggregation via HANC layer
        4. Optional residual connection with batch normalization
        5. Output projection via 1x1 convolution
        6. Channel recalibration via Squeeze-Excitation

        Args:
            inputs: Input tensor of shape (batch_size, height, width, input_channels).
            training: Boolean indicating training mode for batch normalization
                and dropout layers.

        Returns:
            Output tensor of shape (batch_size, height, width, filters).
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
        """
        Compute output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple with same spatial dimensions and filters channels.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        return tuple(list(input_shape[:-1]) + [self.filters])

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