"""
Multi Level Feature Compilation (MLFC) Layer for Cross-Scale Feature Fusion.

This layer implements one of the key innovations of ACC-UNet: multi-level feature compilation
that enables cross-scale information exchange between different encoder levels. Unlike standard
U-Net skip connections that only pass features from corresponding encoder-decoder levels,
MLFC allows each encoder level to be enriched with semantic information from all other levels.

Core Functionality:
    The MLFC layer addresses the semantic gap problem in U-Net architectures by implementing
    a sophisticated feature fusion mechanism that:

    1. **Multi-Scale Aggregation**: Collects features from all 4 encoder levels simultaneously
    2. **Adaptive Resizing**: Intelligently resizes features to match target spatial dimensions
       using appropriate pooling (downsampling) or interpolation (upsampling) strategies
    3. **Cross-Level Fusion**: Concatenates multi-scale features to create rich representations
    4. **Iterative Refinement**: Applies multiple compilation iterations to strengthen feature mixing
    5. **Residual Enhancement**: Uses residual connections to preserve original feature information
    6. **Channel Recalibration**: Applies squeeze-excitation for adaptive channel weighting

Architectural Innovation:
    Traditional U-Net skip connections suffer from semantic gaps between encoder and decoder
    features due to the absence of sufficient abstraction. MLFC solves this by:

    - **Semantic Bridging**: Low-level features gain high-level semantic context
    - **Detail Preservation**: High-level features retain fine-grained spatial details
    - **Multi-Scale Context**: Each level receives information from all other scales
    - **Progressive Refinement**: Multiple iterations allow gradual feature enhancement

Technical Implementation:
    For each compilation iteration and each encoder level, the layer:

    1. **Collects** features from all 4 encoder levels: [h1×w1×c1, h2×w2×c2, h3×w3×c3, h4×w4×c4]
    2. **Resizes** all features to current level's spatial dimensions using:
       - Average pooling for downsampling (higher → lower resolution)
       - Bilinear upsampling for upsampling (lower → higher resolution)
       - Exact resizing to ensure precise dimension matching
    3. **Concatenates** resized features: total_channels = c1 + c2 + c3 + c4
    4. **Compiles** through 1×1 convolution: total_channels → current_level_channels
    5. **Merges** with original features via concatenation and residual connection
    6. **Refines** through batch normalization and activation functions

Multi-Iteration Process:
    The layer performs num_iterations compilation cycles, where each iteration:
    - Uses outputs from the previous iteration as inputs
    - Strengthens cross-level feature mixing progressively
    - Allows gradual semantic information propagation
    - Typical range: 1-4 iterations (3 recommended for optimal performance)

Mathematical Formulation:
    For level i at iteration t:
    ```
    # Feature collection and resizing
    F_resized = [Resize(F_j^t, size_i) for j in [1,2,3,4]]

    # Multi-level compilation
    F_concat = Concat(F_resized)  # shape: (H_i, W_i, C_total)
    F_compiled = Conv1x1(F_concat)  # shape: (H_i, W_i, C_i)

    # Residual enhancement
    F_merged = Conv1x1(Concat([F_compiled, F_i^t]))  # shape: (H_i, W_i, C_i)
    F_i^{t+1} = F_merged + F_i^t  # Residual connection
    ```

Performance Characteristics:
    - **Memory Overhead**: Moderate increase due to feature concatenation and resizing
    - **Computational Cost**: O(HW × C_total × num_iterations) for compilation convolutions
    - **Parameter Count**: Relatively lightweight (mainly 1×1 convolutions)
    - **Training Stability**: Residual connections ensure stable gradient flow
    - **Inference Speed**: Parallelizable operations with minimal sequential dependencies

Use Cases:
    - **Medical Image Segmentation**: Excellent for multi-scale pathology detection
    - **Remote Sensing**: Effective for objects at different scales in satellite imagery
    - **Industrial Inspection**: Handles both fine defects and large structural patterns
    - **Biological Imaging**: Processes cellular details alongside tissue-level structures

Integration with ACC-UNet:
    MLFC layers are strategically placed in the skip connection pathway:
    - Applied after ResPath processing of encoder features
    - Typically 3 MLFC layers in series for optimal feature compilation
    - Feeds enriched features to decoder levels for improved reconstruction
    - Works synergistically with HANC blocks for comprehensive context modeling

Comparison to Alternatives:
    - **vs. Standard Skip Connections**: Provides cross-level information vs. single-level
    - **vs. Feature Pyramid Networks**: More sophisticated fusion vs. simple concatenation
    - **vs. Attention Mechanisms**: Computationally efficient vs. quadratic complexity
    - **vs. Transformer Approaches**: Convolutional efficiency vs. global modeling

Implementation Notes:
    - Automatically handles different input spatial dimensions through intelligent resizing
    - Uses bilinear interpolation for smooth upsampling transitions
    - Employs average pooling for information-preserving downsampling
    - Applies batch normalization for training stability
    - Includes squeeze-excitation for adaptive channel importance weighting
    - Supports gradient checkpointing for memory-efficient training
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, List, Any

from dl_techniques.layers.squeeze_excitation import SqueezeExcitation


class MLFCLayer(keras.layers.Layer):
    """
    Multi Level Feature Compilation (MLFC) Layer.

    This layer implements multi-level feature fusion by:
    1. Resizing feature maps from different encoder levels to common dimensions
    2. Concatenating them along the channel dimension
    3. Processing through pointwise convolutions
    4. Applying residual connections and squeeze-excitation

    The layer processes features from 4 different encoder levels simultaneously,
    enriching each level with information from other levels through cross-level
    feature compilation.

    Args:
        channels_list: List of channel counts for each of the 4 levels [c1, c2, c3, c4].
        num_iterations: Number of compilation iterations to apply.
        kernel_initializer: Initializer for convolution kernels.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for convolution kernels.
        bias_regularizer: Regularizer for bias vectors.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        List of 4 tensors with shapes:
        - Level 1: (batch_size, h1, w1, c1) - highest resolution
        - Level 2: (batch_size, h2, w2, c2) - h2=h1/2, w2=w1/2
        - Level 3: (batch_size, h3, w3, c3) - h3=h1/4, w3=w1/4
        - Level 4: (batch_size, h4, w4, c4) - h4=h1/8, w4=w1/8

    Output shape:
        List of 4 tensors with same shapes as input but enriched features.

    Example:
        ```python
        # For standard U-Net with base filters=32
        mlfc = MLFCLayer(
            channels_list=[32, 64, 128, 256],
            num_iterations=1
        )

        # Multiple iterations for stronger feature mixing
        mlfc = MLFCLayer(
            channels_list=[32, 64, 128, 256],
            num_iterations=3
        )

        # Usage in model
        x1, x2, x3, x4 = encoder_features  # From different levels
        x1, x2, x3, x4 = mlfc([x1, x2, x3, x4])
        ```

    Note:
        This layer expects exactly 4 input feature maps from different
        encoder levels with 2x downsampling between adjacent levels.
        The spatial dimensions are handled automatically through resizing.
    """

    def __init__(
            self,
            channels_list: List[int],
            num_iterations: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if len(channels_list) != 4:
            raise ValueError(f"channels_list must have exactly 4 elements, got {len(channels_list)}")

        self.channels_list = channels_list
        self.num_iterations = num_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Validate parameters
        if any(c <= 0 for c in channels_list):
            raise ValueError(f"All channel counts must be positive, got {channels_list}")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {num_iterations}")

        self.total_channels = sum(channels_list)

        # Will be initialized in build()
        self.compilation_convs = []  # [num_iterations][4 levels]
        self.merge_convs = []  # [num_iterations][4 levels]
        self.batch_norms = []  # [num_iterations][4 levels]
        self.merge_batch_norms = []  # [num_iterations][4 levels]
        self.squeeze_excitations = []  # [4 levels]
        self.activation = None

        self._build_input_shape = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the compilation layers."""
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError(
                f"input_shape must be a list of 4 shapes, got {type(input_shape)} with length {len(input_shape) if isinstance(input_shape, list) else 'N/A'}")

        self._build_input_shape = input_shape

        # Initialize layer lists
        self.compilation_convs = []
        self.merge_convs = []
        self.batch_norms = []
        self.merge_batch_norms = []

        # Create layers for each iteration
        for iter_idx in range(self.num_iterations):
            iter_comp_convs = []
            iter_merge_convs = []
            iter_bns = []
            iter_merge_bns = []

            for level_idx in range(4):
                channels = self.channels_list[level_idx]

                # Compilation convolution (total_channels -> level_channels)
                comp_conv = keras.layers.Conv2D(
                    filters=channels,
                    kernel_size=1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'comp_conv_iter{iter_idx}_level{level_idx}'
                )
                iter_comp_convs.append(comp_conv)

                # Compilation batch norm
                comp_bn = keras.layers.BatchNormalization(
                    name=f'comp_bn_iter{iter_idx}_level{level_idx}'
                )
                iter_bns.append(comp_bn)

                # Merge convolution (2*channels -> channels)
                merge_conv = keras.layers.Conv2D(
                    filters=channels,
                    kernel_size=1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'merge_conv_iter{iter_idx}_level{level_idx}'
                )
                iter_merge_convs.append(merge_conv)

                # Merge batch norm
                merge_bn = keras.layers.BatchNormalization(
                    name=f'merge_bn_iter{iter_idx}_level{level_idx}'
                )
                iter_merge_bns.append(merge_bn)

            self.compilation_convs.append(iter_comp_convs)
            self.merge_convs.append(iter_merge_convs)
            self.batch_norms.append(iter_bns)
            self.merge_batch_norms.append(iter_merge_bns)

        # Squeeze-excitation for each level (applied once at the end)
        self.squeeze_excitations = []
        for level_idx in range(4):
            se = SqueezeExcitation(
                reduction_ratio=0.25,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'se_level{level_idx}'
            )
            self.squeeze_excitations.append(se)

        # Activation
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """Forward pass computation."""
        if len(inputs) != 4:
            raise ValueError(f"Expected 4 input tensors, got {len(inputs)}")

        x1, x2, x3, x4 = inputs

        # Apply multiple compilation iterations
        for iter_idx in range(self.num_iterations):
            # Get current feature maps
            current_features = [x1, x2, x3, x4]
            new_features = []

            # Process each level
            for level_idx in range(4):
                target_shape = ops.shape(current_features[level_idx])
                batch_size = target_shape[0]
                target_height = target_shape[1]
                target_width = target_shape[2]

                # Resize all features to current level's spatial dimensions
                resized_features = []
                for feat_idx, feat in enumerate(current_features):
                    if feat_idx == level_idx:
                        # Same level, no resizing needed
                        resized_features.append(feat)
                    else:
                        # Need to resize to target dimensions
                        feat_height = ops.shape(feat)[1]
                        feat_width = ops.shape(feat)[2]

                        if feat_height > target_height:
                            # Downsample using average pooling
                            scale_h = feat_height // target_height
                            scale_w = feat_width // target_width
                            feat_resized = keras.layers.AveragePooling2D(
                                pool_size=(scale_h, scale_w),
                                strides=(scale_h, scale_w),
                                padding='same'
                            )(feat)
                        else:
                            # Upsample using bilinear interpolation
                            scale_h = target_height // feat_height
                            scale_w = target_width // feat_width
                            feat_resized = keras.layers.UpSampling2D(
                                size=(scale_h, scale_w),
                                interpolation='bilinear'
                            )(feat)

                        # Ensure exact dimensions match
                        feat_resized = keras.layers.Resizing(
                            height=target_height,
                            width=target_width,
                            interpolation='bilinear'
                        )(feat_resized)

                        resized_features.append(feat_resized)

                # Concatenate all resized features
                concatenated = keras.layers.Concatenate(axis=-1)(resized_features)

                # Apply compilation convolution
                compiled_feat = self.compilation_convs[iter_idx][level_idx](concatenated)
                compiled_feat = self.batch_norms[iter_idx][level_idx](compiled_feat, training=training)
                compiled_feat = self.activation(compiled_feat)

                # Merge with original features using residual connection
                original_feat = current_features[level_idx]
                merged_input = keras.layers.Concatenate(axis=-1)([compiled_feat, original_feat])

                # Apply merge convolution with residual
                merged_feat = self.merge_convs[iter_idx][level_idx](merged_input)
                merged_feat = self.merge_batch_norms[iter_idx][level_idx](merged_feat, training=training)
                merged_feat = merged_feat + original_feat  # Residual connection
                merged_feat = self.activation(merged_feat)

                new_features.append(merged_feat)

            # Update features for next iteration
            x1, x2, x3, x4 = new_features

        # Apply final squeeze-excitation to each level
        final_features = []
        for level_idx, feat in enumerate([x1, x2, x3, x4]):
            feat = self.squeeze_excitations[level_idx](feat)
            final_features.append(feat)

        return final_features

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> List[Tuple[Optional[int], ...]]:
        """Compute output shapes."""
        return input_shape  # Shapes remain unchanged

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'channels_list': self.channels_list,
            'num_iterations': self.num_iterations,
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