"""
SqueezeNet architecture for efficient classification.

This model encapsulates the SqueezeNet architecture, a convolutional neural
network designed to achieve AlexNet-level accuracy on ImageNet with
approximately 50 times fewer parameters. The design centers on a novel
building block, the "Fire module," which systematically reduces the
parameter count while preserving representational power.

Architectural Overview:
    SqueezeNet's macro-architecture consists of a standalone convolutional
    layer (the "stem"), followed by a series of eight Fire modules, and
    concludes with a final convolutional layer and global pooling (the
    "head"). Max-pooling layers are strategically placed after the stem and
    certain Fire modules to downsample feature maps.

    The core innovation is the Fire module, which comprises two stages:
    1.  A "squeeze" stage: This consists of a convolutional layer with only
        1x1 filters. Its purpose is to reduce the channel dimensionality of
    its input tensor, acting as a bottleneck.
    2.  An "expand" stage: This stage takes the output of the squeeze layer
        and feeds it into two parallel convolutional layers—one with 1x1
        filters and another with 3x3 filters. The outputs of these two
        layers are then concatenated along the channel dimension.

    This model also supports the integration of residual "bypass"
    connections, similar to those in ResNet. These connections are added
    around certain Fire modules to improve gradient flow and enable the
    training of deeper SqueezeNet variants.

Foundational Principles and Intuition:
    The architecture is guided by three primary strategies for creating a
    parameter-efficient CNN:

    1.  Replace 3x3 filters with 1x1 filters: A significant portion of the
        computation and parameters in a CNN is in its 3x3 convolutions.
        A 1x1 filter has 9 times fewer parameters than a 3x3 filter,
        providing a direct and effective way to reduce model size. The
        Fire module's expand layer heavily utilizes 1x1 convolutions.

    2.  Decrease input channels to 3x3 filters: The number of parameters
        in a convolutional layer is calculated as `(input_channels) *
        (output_channels) * (kernel_height * kernel_width)`. The "squeeze"
        layer's primary function is to reduce the `input_channels` term
        for the subsequent 3x3 convolution in the "expand" layer. This
        bottleneck design is the most critical factor in SqueezeNet's
        parameter efficiency.

    3.  Downsample late in the network: The authors postulate that larger
        activation maps (achieved by delaying pooling) can lead to higher
        classification accuracy by preserving more spatial information.
        SqueezeNet performs max-pooling later in the network compared to
        its predecessors, allowing representations to develop over a larger
        spatial extent before being downsampled.

References:
    -   Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer
        parameters and <0.5MB model size" (2016).
        https://arxiv.org/abs/1602.07360
    -   He et al., "Deep Residual Learning for Image Recognition" (2015)
        (for the bypass connection concept).
        https://arxiv.org/abs/1512.03385
"""

import keras
from keras import layers, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FireModule(keras.layers.Layer):
    """
    Fire module - the fundamental building block of SqueezeNet.

    A Fire module consists of a squeeze convolution layer (1x1 filters) feeding into
    an expand layer that has a mix of 1x1 and 3x3 convolution filters. This design
    significantly reduces the number of parameters while maintaining representational power.

    **Architecture**:
    ```
    Input → Squeeze(1x1) → ReLU → Expand(1x1 + 3x3) → ReLU → Concatenate → Output
    ```

    Args:
        s1x1: Number of filters in squeeze layer (all 1x1).
        e1x1: Number of 1x1 filters in expand layer.
        e3x3: Number of 3x3 filters in expand layer.
        kernel_regularizer: Regularizer for convolution kernels.
        kernel_initializer: Initializer for convolution kernels.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, e1x1 + e3x3)`.
    """

    def __init__(
            self,
            s1x1: int,
            e1x1: int,
            e3x3: int,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if s1x1 <= 0 or e1x1 <= 0 or e3x3 <= 0:
            raise ValueError("All filter counts must be positive integers")

        # Store configuration
        self.s1x1 = s1x1
        self.e1x1 = e1x1
        self.e3x3 = e3x3
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        # Create squeeze layer (1x1 convolution)
        self.squeeze = layers.Conv2D(
            filters=s1x1,
            kernel_size=1,
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name='squeeze'
        )

        # Create expand layers
        self.expand_1x1 = layers.Conv2D(
            filters=e1x1,
            kernel_size=1,
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name='expand_1x1'
        )

        self.expand_3x3 = layers.Conv2D(
            filters=e3x3,
            kernel_size=3,
            padding='same',  # Maintain spatial dimensions
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name='expand_3x3'
        )

        # Concatenation layer
        self.concat = layers.Concatenate(axis=-1)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the Fire module by building all sub-layers."""
        # Build squeeze layer
        self.squeeze.build(input_shape)

        # Compute squeeze output shape
        squeeze_output_shape = self.squeeze.compute_output_shape(input_shape)

        # Build expand layers with squeeze output shape
        self.expand_1x1.build(squeeze_output_shape)
        self.expand_3x3.build(squeeze_output_shape)

        # Build concatenation layer
        expand_1x1_shape = self.expand_1x1.compute_output_shape(squeeze_output_shape)
        expand_3x3_shape = self.expand_3x3.compute_output_shape(squeeze_output_shape)
        self.concat.build([expand_1x1_shape, expand_3x3_shape])

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the Fire module."""
        # Squeeze
        squeezed = self.squeeze(inputs, training=training)

        # Expand
        expanded_1x1 = self.expand_1x1(squeezed, training=training)
        expanded_3x3 = self.expand_3x3(squeezed, training=training)

        # Concatenate
        output = self.concat([expanded_1x1, expanded_3x3])

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape of Fire module."""
        output_shape = list(input_shape)
        output_shape[-1] = self.e1x1 + self.e3x3
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            's1x1': self.s1x1,
            'e1x1': self.e1x1,
            'e3x3': self.e3x3,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


@keras.saving.register_keras_serializable()
class SqueezeNetV1(keras.Model):
    """
    SqueezeNet V1 model implementation.

    A highly efficient CNN architecture that achieves AlexNet-level accuracy
    with 50x fewer parameters through strategic use of Fire modules and
    architectural design choices.

    Args:
        num_classes: Integer, number of output classes for classification.
        variant_config: Dictionary defining the Fire module configurations.
        use_bypass: Boolean or string, whether to use bypass connections.
            Can be False, 'simple', or 'complex'.
        dropout_rate: Float, dropout rate after final Fire module.
        kernel_regularizer: Regularizer for all convolution kernels.
        kernel_initializer: Initializer for all convolution kernels.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape (height, width, channels).
        **kwargs: Additional arguments for Model base class.

    Raises:
        ValueError: If invalid configuration is provided.

    Example:
        >>> # Create standard SqueezeNet for ImageNet
        >>> model = SqueezeNetV1.from_variant("1.0", num_classes=1000)
        >>>
        >>> # Create SqueezeNet with simple bypass for CIFAR-10
        >>> model = SqueezeNetV1.from_variant("1.0_bypass", num_classes=10,
        >>>                                    input_shape=(32, 32, 3))
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "1.0": {
            "fire_configs": [
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},  # fire2
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},  # fire3
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},  # fire4
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},  # fire5
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},  # fire6
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},  # fire7
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},  # fire8
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},  # fire9
            ],
            "use_bypass": False,
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8]  # After conv1, fire4, fire8
        },
        "1.1": {
            "fire_configs": [
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},  # fire2
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},  # fire3
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},  # fire4
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},  # fire5
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},  # fire6
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},  # fire7
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},  # fire8
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},  # fire9
            ],
            "use_bypass": False,
            "conv1_filters": 64,
            "conv1_kernel": 3,
            "conv1_stride": 2,
            "pool_indices": [1, 3, 5]  # Different pooling strategy
        },
        "1.0_bypass": {
            "fire_configs": [
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},  # fire2
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},  # fire3
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},  # fire4
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},  # fire5
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},  # fire6
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},  # fire7
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},  # fire8
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},  # fire9
            ],
            "use_bypass": "simple",
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8]
        }
    }

    # Architecture constants
    STEM_INITIALIZER = "glorot_uniform"
    HEAD_INITIALIZER = "glorot_uniform"

    def __init__(
            self,
            num_classes: int = 1000,
            variant_config: Optional[Dict[str, Any]] = None,
            use_bypass: Union[bool, str] = False,
            dropout_rate: float = 0.5,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            include_top: bool = True,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            **kwargs: Any
    ) -> None:
        # Use default configuration if none provided
        if variant_config is None:
            variant_config = self.MODEL_VARIANTS["1.0"]

        # Validate inputs
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in range [0, 1)")

        # Store configuration
        self.num_classes = num_classes
        self.variant_config = variant_config
        self.use_bypass = use_bypass if use_bypass else variant_config.get("use_bypass", False)
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self._input_shape = input_shape

        # Extract variant configuration
        self.fire_configs = variant_config["fire_configs"]
        self.conv1_filters = variant_config["conv1_filters"]
        self.conv1_kernel = variant_config["conv1_kernel"]
        self.conv1_stride = variant_config["conv1_stride"]
        self.pool_indices = variant_config["pool_indices"]

        # Initialize layer lists
        self.stem_layers = []
        self.fire_modules = []
        self.pool_layers = []
        self.bypass_layers = []
        self.head_layers = []

        # Build the model
        inputs = keras.Input(shape=input_shape)
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete SqueezeNet model architecture."""
        x = inputs

        # Build stem (initial convolution)
        x = self._build_stem(x)

        # Build Fire modules with pooling and bypass
        x = self._build_fire_modules(x)

        # Build classification head if requested
        if self.include_top:
            x = self._build_head(x)

        return x

    def _build_stem(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the stem (initial convolution) layer."""
        conv1 = layers.Conv2D(
            filters=self.conv1_filters,
            kernel_size=self.conv1_kernel,
            strides=self.conv1_stride,
            activation='relu',
            padding='same' if self.conv1_stride == 1 else 'valid',
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.STEM_INITIALIZER,
            name='conv1'
        )
        x = conv1(x)
        self.stem_layers.append(conv1)

        # Add first pooling if specified
        if 1 in self.pool_indices:
            maxpool1 = layers.MaxPooling2D(
                pool_size=3,
                strides=2,
                padding='valid',
                name='maxpool1'
            )
            x = maxpool1(x)
            self.pool_layers.append(maxpool1)

        return x

    def _build_fire_modules(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build all Fire modules with optional pooling and bypass."""
        bypass_indices = []
        if self.use_bypass == "simple":
            bypass_indices = [2, 4, 6, 8]  # Fire3, 5, 7, 9 (0-indexed: 2, 4, 6, 7)
        elif self.use_bypass == "complex":
            bypass_indices = list(range(len(self.fire_configs)))

        for idx, fire_config in enumerate(self.fire_configs):
            fire_name = f'fire{idx + 2}'  # Fire modules start from fire2

            # Store identity for bypass if needed
            identity = x

            # Create and apply Fire module
            fire_module = FireModule(
                s1x1=fire_config['s1x1'],
                e1x1=fire_config['e1x1'],
                e3x3=fire_config['e3x3'],
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                name=fire_name
            )
            x = fire_module(x)
            self.fire_modules.append(fire_module)

            # Add bypass connection if needed
            if idx in bypass_indices:
                if self.use_bypass == "simple" and identity.shape[-1] == x.shape[-1]:
                    # Simple bypass (just addition)
                    add_layer = layers.Add(name=f'add_{fire_name}')
                    x = add_layer([x, identity])
                    self.bypass_layers.append(add_layer)
                elif self.use_bypass == "complex":
                    # Complex bypass with 1x1 conv to match dimensions
                    if identity.shape[-1] != x.shape[-1]:
                        bypass_conv = layers.Conv2D(
                            filters=x.shape[-1],
                            kernel_size=1,
                            activation=None,
                            kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.kernel_initializer,
                            name=f'bypass_conv_{fire_name}'
                        )
                        identity = bypass_conv(identity)
                        self.bypass_layers.append(bypass_conv)

                    add_layer = layers.Add(name=f'add_{fire_name}')
                    x = add_layer([x, identity])
                    self.bypass_layers.append(add_layer)

            # Add pooling after specific Fire modules
            fire_number = idx + 2  # Convert to 1-based fire module number
            if fire_number in self.pool_indices:
                pool_layer = layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    name=f'maxpool{fire_number}'
                )
                x = pool_layer(x)
                self.pool_layers.append(pool_layer)

            # Add dropout after last Fire module
            if idx == len(self.fire_configs) - 1:
                dropout = layers.Dropout(
                    rate=self.dropout_rate,
                    name='dropout'
                )
                x = dropout(x)
                self.head_layers.append(dropout)

        return x

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the classification head."""
        # Final convolution for classification
        conv10 = layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            activation='relu',
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.HEAD_INITIALIZER,
            name='conv10'
        )
        x = conv10(x)
        self.head_layers.append(conv10)

        # Global average pooling
        avgpool = layers.GlobalAveragePooling2D(name='avgpool')
        x = avgpool(x)
        self.head_layers.append(avgpool)

        # Softmax activation
        softmax = layers.Activation('softmax', name='predictions')
        x = softmax(x)
        self.head_layers.append(softmax)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            **kwargs: Any
    ) -> "SqueezeNetV1":
        """
        Create a SqueezeNet model from a predefined variant.

        Args:
            variant: String, one of "1.0", "1.1", "1.0_bypass"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape (height, width, channels)
            **kwargs: Additional arguments passed to the constructor

        Returns:
            SqueezeNetV1 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Standard SqueezeNet for ImageNet
            >>> model = SqueezeNetV1.from_variant("1.0", num_classes=1000)
            >>>
            >>> # SqueezeNet 1.1 for CIFAR-10
            >>> model = SqueezeNetV1.from_variant("1.1", num_classes=10,
            >>>                                   input_shape=(32, 32, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        variant_config = cls.MODEL_VARIANTS[variant]

        return cls(
            num_classes=num_classes,
            variant_config=variant_config,
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'variant_config': self.variant_config,
            'use_bypass': self.use_bypass,
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'include_top': self.include_top,
            'input_shape': self._input_shape
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SqueezeNetV1":
        """Create model from configuration."""
        # Deserialize regularizer if present
        if config.get('kernel_regularizer'):
            config['kernel_regularizer'] = regularizers.deserialize(
                config['kernel_regularizer']
            )
        if config.get('kernel_initializer'):
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )

        # Remove base config keys that are handled by Model
        base_config = {}
        for key in ['name', 'layers', 'input_layers', 'output_layers']:
            if key in config:
                base_config[key] = config.pop(key)

        return cls(**config)

    def summary_with_details(self) -> None:
        """Print detailed model summary with configuration information."""
        self.summary()

        logger.info("SqueezeNet V1 Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Number of Fire modules: {len(self.fire_configs)}")
        logger.info(f"  - Use bypass: {self.use_bypass}")
        logger.info(f"  - Conv1 filters: {self.conv1_filters}")
        logger.info(f"  - Conv1 kernel: {self.conv1_kernel}")
        logger.info(f"  - Conv1 stride: {self.conv1_stride}")
        logger.info(f"  - Pooling after modules: {self.pool_indices}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")

        # Calculate total parameters reduction
        total_params = sum(
            fire['s1x1'] + fire['e1x1'] * 1 + fire['e3x3'] * 9
            for fire in self.fire_configs
        )
        logger.info(f"  - Estimated parameter reduction: ~50x vs AlexNet")

# ---------------------------------------------------------------------

def create_squeezenet_v1(
        variant: str = "1.0",
        num_classes: int = 1000,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        weights: Optional[str] = None,
        **kwargs: Any
) -> SqueezeNetV1:
    """
    Convenience function to create SqueezeNet V1 models.

    Args:
        variant: String, model variant ("1.0", "1.1", "1.0_bypass")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape (height, width, channels)
        weights: String, pretrained weights to load (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        SqueezeNetV1 model instance

    Example:
        >>> # Create SqueezeNet 1.0 for ImageNet
        >>> model = create_squeezenet_v1("1.0", num_classes=1000)
        >>>
        >>> # Create SqueezeNet 1.1 for CIFAR-10
        >>> model = create_squeezenet_v1("1.1", num_classes=10,
        >>>                              input_shape=(32, 32, 3))
        >>>
        >>> # Create SqueezeNet with bypass for custom dataset
        >>> model = create_squeezenet_v1("1.0_bypass", num_classes=100,
        >>>                              input_shape=(64, 64, 3))
    """
    if weights is not None:
        logger.info("Warning: Pretrained weights are not yet implemented")

    model = SqueezeNetV1.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------