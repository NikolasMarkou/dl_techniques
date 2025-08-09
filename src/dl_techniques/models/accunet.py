"""
ACC-UNet: A Completely Convolutional UNet model for the 2020s.

This module implements the complete ACC-UNet architecture, which combines
the benefits of convolutional networks with transformer-inspired design decisions.
Key innovations include HANC blocks for hierarchical context aggregation and
MLFC layers for multi-level feature compilation in skip connections.

Reference:
    "ACC-UNet: A Completely Convolutional UNet model for the 2020s"
    Ibtehaz, N. and Kihara, D.
    MICCAI 2023
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any

from ..layers.hanc_block import HANCBlock
from ..layers.res_path import ResPath
from ..layers.mlfc_layer import MLFCLayer


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
        input_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale).
        num_classes: Number of output classes for segmentation.
        base_filters: Base number of filters (default: 32). The model uses
            [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]
            for the 5 encoder levels.
        mlfc_iterations: Number of MLFC iterations for feature compilation (default: 3).
        kernel_initializer: Initializer for convolution kernels.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for convolution kernels.
        bias_regularizer: Regularizer for bias vectors.
        **kwargs: Additional arguments for the Model base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, input_channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, num_classes) for
        multi-class segmentation, or (batch_size, height, width, 1) for
        binary segmentation.

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
            mlfc_iterations=4
        )

        # Compile and use
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        ```

    Note:
        The model automatically determines the final activation (sigmoid for
        num_classes=1, softmax for num_classes>1). For training, use logits
        by setting the appropriate loss function.
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

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.mlfc_iterations = mlfc_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Validate parameters
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if base_filters <= 0:
            raise ValueError(f"base_filters must be positive, got {base_filters}")
        if mlfc_iterations <= 0:
            raise ValueError(f"mlfc_iterations must be positive, got {mlfc_iterations}")

        # Calculate filter sizes for each level
        self.filter_sizes = [
            base_filters,  # Level 1: 32
            base_filters * 2,  # Level 2: 64
            base_filters * 4,  # Level 3: 128
            base_filters * 8,  # Level 4: 256
            base_filters * 16  # Level 5 (bottleneck): 512
        ]

        # Build the network components
        self._build_encoder()
        self._build_decoder()
        self._build_skip_connections()
        self._build_output_layer()

    def _build_encoder(self) -> None:
        """Build encoder blocks."""
        # Encoder blocks - each level has 2 HANC blocks
        self.encoder_blocks = []
        self.pooling_layers = []

        for level in range(5):  # 5 levels total
            level_blocks = []

            if level == 0:
                # First level: input_channels -> base_filters
                block1 = HANCBlock(
                    filters=self.filter_sizes[0],
                    k=3, inv_factor=3,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'encoder_l{level}_block1'
                )
                block2 = HANCBlock(
                    filters=self.filter_sizes[0],
                    k=3, inv_factor=3,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'encoder_l{level}_block2'
                )
                level_blocks = [block1, block2]
            else:
                # Other levels: prev_filters -> curr_filters
                prev_filters = self.filter_sizes[level - 1]
                curr_filters = self.filter_sizes[level]

                # Determine k based on level (as per paper)
                if level <= 2:
                    k = 3
                elif level == 3:
                    k = 2
                else:  # level 4 (bottleneck)
                    k = 1

                block1 = HANCBlock(
                    filters=curr_filters,
                    k=k, inv_factor=3,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'encoder_l{level}_block1'
                )
                block2 = HANCBlock(
                    filters=curr_filters,
                    k=k, inv_factor=3,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'encoder_l{level}_block2'
                )
                level_blocks = [block1, block2]

            self.encoder_blocks.append(level_blocks)

            # Pooling layers (not needed for last level)
            if level < 4:
                pool = keras.layers.MaxPooling2D(
                    pool_size=2,
                    strides=2,
                    name=f'pool_{level}'
                )
                self.pooling_layers.append(pool)

    def _build_decoder(self) -> None:
        """Build decoder blocks."""
        self.decoder_upsamples = []
        self.decoder_blocks = []

        for level in range(4):  # 4 decoder levels
            # Upsample layer
            curr_filters = self.filter_sizes[4 - level]  # 512, 256, 128, 64
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
                inv_factor = 34  # Special case as per paper

            # Decoder blocks (2 per level)
            # Input channels: next_filters (from upsample) + next_filters (from skip) = 2*next_filters
            block1 = HANCBlock(
                filters=next_filters,
                k=k, inv_factor=inv_factor,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'decoder_l{level}_block1'
            )
            block2 = HANCBlock(
                filters=next_filters,
                k=k, inv_factor=inv_factor,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'decoder_l{level}_block2'
            )

            self.decoder_blocks.append([block1, block2])

    def _build_skip_connections(self) -> None:
        """Build skip connection processing."""
        # ResPath layers for each encoder level (except bottleneck)
        self.res_paths = []
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
        self.mlfc_layers = []
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

    def _build_output_layer(self) -> None:
        """Build output layer."""
        # Output convolution
        if self.num_classes == 1:
            # Binary segmentation
            self.output_conv = keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='output_conv'
            )
            self.output_activation = keras.layers.Sigmoid(name='output_activation')
        else:
            # Multi-class segmentation
            self.output_conv = keras.layers.Conv2D(
                filters=self.num_classes,
                kernel_size=1,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='output_conv'
            )
            self.output_activation = keras.layers.Softmax(name='output_activation')

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation."""
        # Encoder forward pass
        encoder_features = []
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

        # Process skip connection features
        # Apply ResPath to encoder features
        processed_features = []
        for level, features in enumerate(encoder_features):
            processed = self.res_paths[level](features, training=training)
            processed_features.append(processed)

        # Apply MLFC layers iteratively
        for mlfc_layer in self.mlfc_layers:
            processed_features = mlfc_layer(processed_features, training=training)

        # Decoder forward pass
        x = bottleneck_features

        for level in range(4):
            # Upsample
            x = self.decoder_upsamples[level](x)

            # Concatenate with processed skip connection features
            skip_features = processed_features[3 - level]  # Reverse order
            x = keras.layers.Concatenate(axis=-1)([x, skip_features])

            # Apply decoder blocks
            for block in self.decoder_blocks[level]:
                x = block(x, training=training)

        # Output layer
        x = self.output_conv(x)
        x = self.output_activation(x)

        return x

    def get_config(self) -> dict:
        """Get model configuration."""
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
    def from_config(cls, config: dict) -> 'AccUNet':
        """Create model from configuration."""
        return cls(**config)


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
        input_channels: Number of input channels.
        num_classes: Number of output classes.
        base_filters: Base number of filters.
        mlfc_iterations: Number of MLFC iterations.
        input_shape: Input spatial dimensions (height, width). If None,
            uses dynamic shape.
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
        input_channels: int = 3,
        input_shape: Tuple[int, int] = (256, 256),
        base_filters: int = 32,
        **kwargs: Any
) -> keras.Model:
    """
    Create ACC-UNet for binary segmentation.

    Args:
        input_channels: Number of input channels (default: 3 for RGB).
        input_shape: Input spatial dimensions (default: (256, 256)).
        base_filters: Base number of filters (default: 32).
        **kwargs: Additional arguments for create_acc_unet.

    Returns:
        Keras Model for binary segmentation.
    """
    return create_acc_unet(
        input_channels=input_channels,
        num_classes=1,
        base_filters=base_filters,
        input_shape=input_shape,
        **kwargs
    )


def create_acc_unet_multiclass(
        input_channels: int,
        num_classes: int,
        input_shape: Tuple[int, int] = (256, 256),
        base_filters: int = 32,
        **kwargs: Any
) -> keras.Model:
    """
    Create ACC-UNet for multi-class segmentation.

    Args:
        input_channels: Number of input channels.
        num_classes: Number of output classes (> 1).
        input_shape: Input spatial dimensions (default: (256, 256)).
        base_filters: Base number of filters (default: 32).
        **kwargs: Additional arguments for create_acc_unet.

    Returns:
        Keras Model for multi-class segmentation.
    """
    if num_classes <= 1:
        raise ValueError(f"For multi-class segmentation, num_classes must be > 1, got {num_classes}")

    return create_acc_unet(
        input_channels=input_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        input_shape=input_shape,
        **kwargs
    )