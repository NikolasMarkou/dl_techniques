"""
ACC-UNet: A Completely Convolutional UNet model for the 2020s.

This module implements the complete ACC-UNet architecture from the paper
"ACC-UNet: A Completely Convolutional UNet model for the 2020s" by Ibtehaz & Kihara (MICCAI 2023).
ACC-UNet combines the benefits of convolutional networks with transformer-inspired design decisions,
achieving state-of-the-art performance on medical image segmentation while using significantly
fewer parameters than transformer-based alternatives.

Key Innovations:
    1. **HANC Blocks**: Hierarchical Aggregation of Neighborhood Context blocks replace standard
       convolution blocks, providing transformer-like long-range dependencies through multi-scale
       pooling operations (2x2, 4x4, 8x8, 16x16 patches).

    2. **MLFC Layers**: Multi-Level Feature Compilation layers in skip connections enable
       cross-level feature fusion by aggregating information from all encoder levels,
       enriching features with multi-scale semantic information.

    3. **Enhanced Skip Connections**: ResPath layers with residual blocks reduce the semantic
       gap between encoder and decoder features, improving information flow.

    4. **Efficient Design**: Purely convolutional architecture with inverted bottlenecks,
       depthwise convolutions, and squeeze-excitation for parameter efficiency.

Architecture Overview:
    - **Encoder**: 5 levels with [32, 64, 128, 256, 512] filters (configurable base_filters)
    - **Decoder**: 4 levels with transposed convolutions for upsampling
    - **Skip Connections**: ResPath + MLFC processing for enhanced feature compilation
    - **Context Aggregation**: HANC layers with hierarchical k values [3, 3, 3, 2, 1]
    - **Parameter Count**: ~16.8M parameters (comparable to standard U-Net)

Performance Characteristics:
    - Outperforms Swin-UNet by 2.64±2.54% dice score with 59.26% fewer parameters
    - Outperforms UCTransNet by 0.45±1.61% dice score with 24.24% fewer parameters
    - Maintains computational efficiency comparable to standard U-Net
    - Supports both binary and multi-class segmentation tasks
    - Handles variable input sizes (dynamic shape support)

Model Components:
    - **HANCBlock**: Core building block with hierarchical context aggregation
    - **ResPath**: Enhanced skip connection processing with residual blocks
    - **MLFCLayer**: Multi-level feature compilation for cross-scale fusion
    - **AccUNet**: Complete model implementation with factory functions

References:
    Ibtehaz, N., & Kihara, D. (2023). ACC-UNet: A Completely Convolutional UNet
    model for the 2020s. In International Conference on Medical Image Computing
    and Computer-Assisted Intervention (pp. 1-11). Springer.

    Original paper: https://arxiv.org/abs/2308.13680
    GitHub: https://github.com/kiharalab/ACC-UNet
"""

import keras
from typing import Optional, Union, Tuple, Any, List, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.res_path import ResPath
from dl_techniques.layers.hanc_block import HANCBlock
from dl_techniques.layers.multi_level_feature_compilation import MLFCLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AccUNet(keras.Model):
    """
    ACC-UNet: A Completely Convolutional UNet model for the 2020s.

    This model implements the ACC-UNet architecture with:
    - HANC blocks for hierarchical context aggregation
    - ResPath for improved skip connections
    - MLFC layers for multi-level feature compilation
    - Squeeze-excitation throughout the network

    The architecture follows a U-Net structure but replaces standard convolution
    blocks with HANC blocks that provide transformer-like long-range dependencies
    through convolutional operations.

    Args:
        input_channels: Integer, number of input channels (e.g., 3 for RGB, 1 for grayscale).
            Must be positive.
        num_classes: Integer, number of output classes for segmentation. Must be positive.
        base_filters: Integer, base number of filters. The model uses
            [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]
            for the 5 encoder levels. Must be positive. Defaults to 32.
        mlfc_iterations: Integer, number of MLFC iterations for feature compilation.
            Must be positive. Defaults to 3.
        kernel_initializer: String or Initializer instance for convolution kernels.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer instance for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer instance for convolution kernels.
            Defaults to None.
        bias_regularizer: Optional Regularizer instance for bias vectors.
            Defaults to None.
        **kwargs: Additional arguments for the Model base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, input_channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, num_classes) for
        multi-class segmentation, or (batch_size, height, width, 1) for
        binary segmentation.

    Attributes:
        encoder_blocks: List of lists containing HANC blocks for each encoder level.
        pooling_layers: List of max pooling layers for encoder downsampling.
        decoder_upsamples: List of transposed convolution layers for decoder upsampling.
        decoder_blocks: List of lists containing HANC blocks for each decoder level.
        res_paths: List of ResPath layers for enhanced skip connection processing.
        mlfc_layers: List of MLFC layers for multi-level feature compilation.
        output_conv: Final convolution layer for class prediction.
        output_activation: Final activation layer (sigmoid or softmax).

    Example:
        ```python
        # Binary segmentation
        model = AccUNet(input_channels=3, num_classes=1)

        # Multi-class segmentation
        model = AccUNet(input_channels=1, num_classes=5)

        # Custom configuration
        model = AccUNet(
            input_channels=3,
            num_classes=2,
            base_filters=64,
            mlfc_iterations=4,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Compile and use
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        ```

    Raises:
        ValueError: If input_channels is not positive.
        ValueError: If num_classes is not positive.
        ValueError: If base_filters is not positive.
        ValueError: If mlfc_iterations is not positive.

    Note:
        The model automatically determines the final activation (sigmoid for
        num_classes=1, softmax for num_classes>1). For training, use logits
        by setting the appropriate loss function.

        Following modern Keras 3 patterns, all sub-layers are created in __init__
        without helper methods, ensuring proper serialization and build handling.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        base_filters: int = 32,
        mlfc_iterations: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if base_filters <= 0:
            raise ValueError(f"base_filters must be positive, got {base_filters}")
        if mlfc_iterations <= 0:
            raise ValueError(f"mlfc_iterations must be positive, got {mlfc_iterations}")

        # Store ALL configuration parameters for serialization
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.mlfc_iterations = mlfc_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate filter sizes for each level
        self.filter_sizes = [
            base_filters,        # Level 0: 32
            base_filters * 2,    # Level 1: 64
            base_filters * 4,    # Level 2: 128
            base_filters * 8,    # Level 3: 256
            base_filters * 16    # Level 4 (bottleneck): 512
        ]

        # CREATE all sub-layers in __init__ following Modern Keras 3 pattern
        # No helper methods - all layers created here directly

        # === ENCODER BLOCKS ===
        self.encoder_blocks: List[List[HANCBlock]] = []

        for level in range(5):  # 5 encoder levels
            if level == 0:
                # First level: input_channels -> base_filters
                input_ch = input_channels
                output_ch = self.filter_sizes[0]
                k = 3
            else:
                # Other levels: prev_filters -> curr_filters
                input_ch = self.filter_sizes[level - 1]
                output_ch = self.filter_sizes[level]
                # Determine k based on level (as per paper)
                if level <= 2:
                    k = 3
                elif level == 3:
                    k = 2
                else:  # level 4 (bottleneck)
                    k = 1

            # Create 2 HANC blocks per level
            block1 = HANCBlock(
                filters=output_ch,
                input_channels=input_ch,  # FIX: First block always takes input_ch for the level
                k=k,
                inv_factor=3,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'encoder_l{level}_block1'
            )

            block2 = HANCBlock(
                filters=output_ch,
                input_channels=output_ch,  # Second block always has same in/out
                k=k,
                inv_factor=3,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'encoder_l{level}_block2'
            )

            self.encoder_blocks.append([block1, block2])

        # === POOLING LAYERS ===
        self.pooling_layers: List[keras.layers.Layer] = []
        for level in range(4):  # Only 4 pooling layers (not needed for bottleneck)
            pool = keras.layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                name=f'pool_{level}'
            )
            self.pooling_layers.append(pool)

        # === DECODER BLOCKS ===
        self.decoder_upsamples: List[keras.layers.Layer] = []
        self.decoder_blocks: List[List[HANCBlock]] = []

        for level in range(4):  # 4 decoder levels
            # Upsample layer
            # curr_filters = self.filter_sizes[4 - level]  # 512, 256, 128, 64
            next_filters = self.filter_sizes[3 - level]  # 256, 128, 64, 32

            upsample = keras.layers.Conv2DTranspose(
                filters=next_filters,
                kernel_size=2,
                strides=2,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'upsample_{level}'
            )
            self.decoder_upsamples.append(upsample)

            # Determine k and inv_factor based on level
            if level <= 1:
                k = 2
                inv_factor = 3
            elif level == 2:
                k = 3
                inv_factor = 3
            else:  # level 3
                k = 3
                inv_factor = 4  # Special case as per paper

            # Decoder blocks (2 per level)
            # Input: next_filters (from upsample) + next_filters (from skip) = 2*next_filters
            block1 = HANCBlock(
                filters=next_filters,
                input_channels=2 * next_filters,  # Concatenated channels
                k=k,
                inv_factor=inv_factor,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'decoder_l{level}_block1'
            )

            block2 = HANCBlock(
                filters=next_filters,
                input_channels=next_filters,  # Output of block1
                k=k,
                inv_factor=inv_factor,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'decoder_l{level}_block2'
            )

            self.decoder_blocks.append([block1, block2])

        # === SKIP CONNECTION PROCESSING ===
        # ResPath layers for each encoder level (except bottleneck)
        self.res_paths: List[ResPath] = []
        res_path_blocks = [4, 3, 2, 1]  # Number of blocks for each level

        for level in range(4):
            res_path = ResPath(
                channels=self.filter_sizes[level],
                num_blocks=res_path_blocks[level],
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'res_path_{level}'
            )
            self.res_paths.append(res_path)

        # MLFC layers for multi-level feature compilation
        self.mlfc_layers: List[MLFCLayer] = []
        channels_list = self.filter_sizes[:4]  # First 4 levels only

        for i in range(self.mlfc_iterations):
            mlfc = MLFCLayer(
                channels_list=channels_list,
                num_iterations=1,  # Each MLFC layer does 1 iteration
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'mlfc_{i}'
            )
            self.mlfc_layers.append(mlfc)

        # === OUTPUT LAYER ===
        # Output convolution and activation
        self.output_conv = keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='output_conv'
        )

        if self.num_classes == 1:
            # Binary segmentation
            self.output_activation = keras.layers.Activation("sigmoid", name='output_activation')
        else:
            # Multi-class segmentation
            self.output_activation = keras.layers.Softmax(name='output_activation')

        # === CONCATENATION LAYERS ===
        # Create concatenation layers for decoder skip connections
        self.concat_layers: List[keras.layers.Layer] = []
        for level in range(4):
            concat = keras.layers.Concatenate(axis=-1, name=f'concat_{level}')
            self.concat_layers.append(concat)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, input_channels).
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, height, width, num_classes).
        """
        # === ENCODER FORWARD PASS ===
        encoder_features: List[keras.KerasTensor] = []
        x = inputs

        for level in range(5):
            # Apply encoder blocks
            for block in self.encoder_blocks[level]:
                x = block(x, training=training)

            # Store features for skip connections (except bottleneck)
            if level < 4:
                encoder_features.append(x)
                x = self.pooling_layers[level](x)
            else:
                # Bottleneck features (level 4)
                bottleneck_features = x

        # === SKIP CONNECTION PROCESSING ===
        # Apply ResPath to encoder features
        processed_features: List[keras.KerasTensor] = []
        for level, features in enumerate(encoder_features):
            processed = self.res_paths[level](features, training=training)
            processed_features.append(processed)

        # Apply MLFC layers iteratively
        for mlfc_layer in self.mlfc_layers:
            processed_features = mlfc_layer(processed_features, training=training)

        # === DECODER FORWARD PASS ===
        x = bottleneck_features

        for level in range(4):
            # Upsample
            x = self.decoder_upsamples[level](x)

            # Concatenate with processed skip connection features
            skip_features = processed_features[3 - level]  # Reverse order
            x = self.concat_layers[level]([x, skip_features])

            # Apply decoder blocks
            for block in self.decoder_blocks[level]:
                x = block(x, training=training)

        # === OUTPUT LAYER ===
        x = self.output_conv(x)
        x = self.output_activation(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns:
            Dictionary containing all model configuration parameters.
        """
        config = super().get_config()
        config.update({
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'base_filters': self.base_filters,
            'mlfc_iterations': self.mlfc_iterations,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AccUNet':
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_acc_unet(
    input_channels: int,
    num_classes: int,
    base_filters: int = 32,
    mlfc_iterations: int = 3,
    input_shape: Optional[Tuple[int, int]] = None,
    **kwargs: Any
) -> keras.Model:
    """
    Create ACC-UNet model with functional API.

    Args:
        input_channels: Integer, number of input channels.
        num_classes: Integer, number of output classes.
        base_filters: Integer, base number of filters. Defaults to 32.
        mlfc_iterations: Integer, number of MLFC iterations. Defaults to 3.
        input_shape: Optional tuple, input spatial dimensions (height, width).
            If None, uses dynamic shape.
        **kwargs: Additional arguments for AccUNet.

    Returns:
        Keras Model instance.

    Example:
        ```python
        # Binary segmentation with fixed input size
        model = create_acc_unet(
            input_channels=3,
            num_classes=1,
            input_shape=(256, 256)
        )

        # Multi-class with dynamic input size
        model = create_acc_unet(
            input_channels=1,
            num_classes=5,
            base_filters=64
        )
        ```
    """
    if input_shape is not None:
        input_spec = keras.Input(shape=input_shape + (input_channels,))
    else:
        input_spec = keras.Input(shape=(None, None, input_channels))

    # Create the model instance
    acc_unet = AccUNet(
        input_channels=input_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        mlfc_iterations=mlfc_iterations,
        **kwargs
    )

    # Build the model by calling it
    outputs = acc_unet(input_spec)

    # Create functional model
    model = keras.Model(inputs=input_spec, outputs=outputs, name='ACC_UNet')

    return model


def create_acc_unet_binary(
    input_channels: int,
    input_shape: Optional[Tuple[int, int]] = None,
    base_filters: int = 32,
    mlfc_iterations: int = 3,
    **kwargs: Any
) -> keras.Model:
    """
    Create ACC-UNet model for binary segmentation.

    Args:
        input_channels: Integer, number of input channels.
        input_shape: Optional tuple, input spatial dimensions (height, width).
            If None, uses dynamic shape.
        base_filters: Integer, base number of filters. Defaults to 32.
        mlfc_iterations: Integer, number of MLFC iterations. Defaults to 3.
        **kwargs: Additional arguments for AccUNet.

    Returns:
        Keras Model instance configured for binary segmentation with sigmoid activation.

    Example:
        ```python
        # Grayscale medical image segmentation
        model = create_acc_unet_binary(
            input_channels=1,
            input_shape=(512, 512),
            base_filters=32
        )

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy', 'dice_coefficient']
        )
        ```
    """
    return create_acc_unet(
        input_channels=input_channels,
        num_classes=1,  # Binary segmentation
        base_filters=base_filters,
        mlfc_iterations=mlfc_iterations,
        input_shape=input_shape,
        **kwargs
    )


def create_acc_unet_multiclass(
    input_channels: int,
    num_classes: int,
    input_shape: Optional[Tuple[int, int]] = None,
    base_filters: int = 32,
    mlfc_iterations: int = 3,
    **kwargs: Any
) -> keras.Model:
    """
    Create ACC-UNet model for multi-class segmentation.

    Args:
        input_channels: Integer, number of input channels.
        num_classes: Integer, number of output classes (must be > 1).
        input_shape: Optional tuple, input spatial dimensions (height, width).
            If None, uses dynamic shape.
        base_filters: Integer, base number of filters. Defaults to 32.
        mlfc_iterations: Integer, number of MLFC iterations. Defaults to 3.
        **kwargs: Additional arguments for AccUNet.

    Returns:
        Keras Model instance configured for multi-class segmentation with softmax activation.

    Raises:
        ValueError: If num_classes is not greater than 1.

    Example:
        ```python
        # RGB image with 5 semantic classes
        model = create_acc_unet_multiclass(
            input_channels=3,
            num_classes=5,
            input_shape=(256, 256),
            base_filters=64
        )

        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy', 'mean_iou']
        )
        ```
    """
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1 for multi-class segmentation, got {num_classes}")

    return create_acc_unet(
        input_channels=input_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        mlfc_iterations=mlfc_iterations,
        input_shape=input_shape,
        **kwargs
    )

# ---------------------------------------------------------------------
