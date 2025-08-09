"""
Residual Path (ResPath) Layer for Enhanced U-Net Skip Connections.

This layer implements a sophisticated skip connection enhancement mechanism from ACC-UNet
that addresses one of the fundamental limitations of standard U-Net architectures: the
semantic gap between encoder and decoder features. ResPath processes encoder features
through a series of residual blocks before they participate in skip connections,
significantly improving the quality and semantic consistency of feature propagation.

Problem Statement:
    In standard U-Net architectures, encoder features are directly concatenated with
    decoder features at corresponding levels. However, this creates a semantic mismatch:
    - **Encoder features**: Rich in spatial detail but semantically "raw"
    - **Decoder features**: Semantically refined through multiple processing stages
    - **Semantic gap**: Direct concatenation can lead to feature conflicts and suboptimal fusion

Core Innovation:
    ResPath bridges this semantic gap by implementing a learnable transformation pathway
    that progressively refines encoder features to match the semantic level of decoder
    features. This is achieved through:

    1. **Progressive Refinement**: Multiple residual blocks gradually enhance feature semantics
    2. **Depth-Adaptive Processing**: Deeper encoder levels receive more processing blocks
    3. **Feature Preservation**: Residual connections maintain spatial detail integrity
    4. **Channel Recalibration**: Squeeze-excitation modules adaptively weight feature channels
    5. **Gradient Optimization**: Residual structure ensures stable training dynamics

Architectural Design:
    The ResPath layer applies a variable number of residual processing blocks based on
    the hierarchical depth difference between encoder and decoder:

    - **Level 0 (shallowest)**: 4 residual blocks - maximum semantic refinement needed
    - **Level 1**: 3 residual blocks - substantial refinement for mid-level features
    - **Level 2**: 2 residual blocks - moderate refinement for deeper features
    - **Level 3 (deepest)**: 1 residual block - minimal refinement for bottleneck-adjacent features

    This depth-adaptive design reflects that shallower encoder features require more
    semantic enhancement to match the abstraction level of their decoder counterparts.

Residual Block Architecture:
    Each residual block implements a sophisticated processing pipeline:
    ```
    Input Features
         ↓
    3×3 Convolution (spatial feature extraction)
         ↓
    Batch Normalization (training stability)
         ↓
    Squeeze-Excitation (channel recalibration)
         ↓
    LeakyReLU Activation (non-linearity)
         ↓
    Residual Addition (+) ← Input Features
         ↓
    Enhanced Features
    ```

Mathematical Formulation:
    For input features F_in and num_blocks residual transformations:
    ```
    F_0 = F_in
    For i in [1, 2, ..., num_blocks]:
        R_i = Conv3x3(F_{i-1})
        R_i = BatchNorm(R_i)
        R_i = SE(R_i)  # Squeeze-Excitation
        R_i = LeakyReLU(R_i)
        F_i = R_i + F_{i-1}  # Residual connection

    F_out = SE(F_num_blocks)  # Final channel recalibration
    ```

Technical Implementation Details:
    - **Convolution Type**: 3×3 spatial convolutions for local feature extraction
    - **Channel Preservation**: Number of channels remains constant throughout processing
    - **Normalization Strategy**: Batch normalization for each transformation step
    - **Activation Function**: LeakyReLU (negative_slope=0.01) for stable gradients
    - **Attention Mechanism**: Squeeze-excitation with 25% reduction ratio for efficiency
    - **Residual Connections**: Direct addition to preserve original information

Performance Characteristics:
    - **Parameter Efficiency**: Lightweight design with minimal parameter overhead
    - **Computational Cost**: O(HW × C × K) where K is kernel size (3×3)
    - **Memory Usage**: Moderate increase due to intermediate feature storage
    - **Training Stability**: Excellent due to residual connections and batch normalization
    - **Inference Speed**: Fast processing with parallelizable operations

Integration with ACC-UNet:
    ResPath layers are strategically positioned in the skip connection pathway:

    1. **Encoder Output**: Features extracted from encoder levels
    2. **ResPath Processing**: Semantic refinement through residual blocks
    3. **MLFC Compilation**: Cross-level feature fusion and enrichment
    4. **Decoder Input**: Enhanced features concatenated with decoder features

    This creates a sophisticated skip connection pipeline that combines:
    - **Local refinement** (ResPath) with **global fusion** (MLFC)
    - **Semantic enhancement** with **spatial detail preservation**
    - **Individual processing** with **cross-level compilation**

Comparison to Standard Skip Connections:
    - **Standard U-Net**: Direct feature concatenation, potential semantic mismatch
    - **ResPath Enhanced**: Progressive semantic refinement, improved feature compatibility
    - **Performance Impact**: Significant improvement in segmentation accuracy
    - **Parameter Cost**: Minimal increase (~2M parameters for entire ACC-UNet)

Comparison to Alternative Approaches:
    - **vs. Attention Gates**: More computationally efficient, less memory intensive
    - **vs. Feature Pyramid Networks**: Better semantic alignment, residual stability
    - **vs. Dense Connections**: More targeted refinement, cleaner architectural design
    - **vs. Transformer Skip Connections**: Maintains convolutional efficiency and locality

Use Cases and Applications:
    - **Medical Image Segmentation**: Excellent for multi-scale pathology detection
    - **Satellite Imagery Analysis**: Effective for objects with varying semantic complexity
    - **Industrial Quality Control**: Handles both fine defects and structural patterns
    - **Biological Microscopy**: Processes cellular details with tissue-level context
    - **Autonomous Driving**: Manages road elements at different abstraction levels

Training Considerations:
    - **Gradient Flow**: Residual connections ensure stable backpropagation
    - **Learning Rate**: Can use standard learning rates due to stable training dynamics
    - **Regularization**: Built-in batch normalization provides regularization effects
    - **Initialization**: Works well with standard weight initialization schemes
    - **Fine-tuning**: Individual block depths can be adjusted for specific domains

Implementation Benefits:
    - **Modular Design**: Easy to integrate into existing U-Net architectures
    - **Configurable Depth**: Adaptable to different network depths and complexities
    - **Robust Training**: Residual structure prevents vanishing gradient problems
    - **Semantic Consistency**: Improves feature alignment between encoder and decoder
    - **Performance Gains**: Measurable improvements in segmentation metrics
"""

import keras
from typing import Optional, Union, Tuple, Any

from .squeeze_excitation import SqueezeExcitation


class ResPath(keras.layers.Layer):
    """
    Residual Path layer for improved skip connections.

    This layer implements a series of residual blocks along the skip connection
    path to reduce the semantic gap between encoder and decoder features.
    Each residual block consists of:
    1. 3x3 convolution
    2. Batch normalization
    3. Squeeze-Excitation
    4. Residual connection

    The number of residual blocks is typically set based on the level difference
    between encoder and decoder (deeper levels use more blocks).

    Args:
        channels: Number of channels (kept constant throughout).
        num_blocks: Number of residual blocks to apply.
        kernel_initializer: Initializer for convolution kernels.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for convolution kernels.
        bias_regularizer: Regularizer for bias vectors.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, channels).

    Example:
        ```python
        # Basic usage - 4 blocks for deepest skip connection
        res_path = ResPath(channels=32, num_blocks=4)

        # Fewer blocks for shallower connections
        res_path = ResPath(channels=64, num_blocks=2)

        # Custom initialization
        res_path = ResPath(
            channels=128,
            num_blocks=3,
            kernel_initializer='he_normal'
        )
        ```

    Note:
        The channel count remains constant throughout the ResPath.
        This layer is typically used in U-Net skip connections to
        process encoder features before concatenating with decoder features.
    """

    def __init__(
            self,
            channels: int,
            num_blocks: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Validate parameters
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        # Will be initialized in build()
        self.conv_blocks = []
        self.bn_blocks = []
        self.se_blocks = []
        self.final_bn = None
        self.activation = None
        self.final_se = None

        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the residual blocks."""
        self._build_input_shape = input_shape

        # Create residual blocks
        self.conv_blocks = []
        self.bn_blocks = []
        self.se_blocks = []

        for i in range(self.num_blocks):
            # 3x3 Convolution
            conv = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                padding='same',
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'conv_block_{i}'
            )
            self.conv_blocks.append(conv)

            # Batch Normalization
            bn = keras.layers.BatchNormalization(name=f'bn_block_{i}')
            self.bn_blocks.append(bn)

            # Squeeze-Excitation
            se = SqueezeExcitation(
                reduction_ratio=0.25,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'se_block_{i}'
            )
            self.se_blocks.append(se)

        # Final batch normalization
        self.final_bn = keras.layers.BatchNormalization(name='final_bn')

        # Activation
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

        # Final squeeze-excitation
        self.final_se = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='final_se'
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation."""
        x = inputs

        # Apply residual blocks
        for i in range(self.num_blocks):
            # Residual block: conv -> bn -> se -> activation + residual
            residual = x

            x = self.conv_blocks[i](x)
            x = self.bn_blocks[i](x, training=training)
            x = self.se_blocks[i](x)
            x = self.activation(x)

            # Add residual connection
            x = x + residual

        # Final processing
        x = self.final_se(x)
        x = self.activation(x)
        x = self.final_bn(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape  # Shape remains unchanged

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_blocks': self.num_blocks,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])