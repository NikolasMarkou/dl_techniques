"""
SqueezeNodule-Net V2 Model Implementation
==================================================

An improved SqueezeNet architecture for efficient lung nodule classification
achieving better accuracy with comparable or fewer parameters.

Based on: "An improved SqueezeNet model for the diagnosis of lung cancer in CT scans"
(Tsivgoulis et al., 2022)
https://doi.org/10.1016/j.mlwa.2022.100399

Key Features:
------------
- Simplified Fire modules with only 3x3 expand convolutions
- Two variants with different squeeze ratios
- Optimized for medical image classification
- Better runtime and classification performance than SqueezeNet
- Support for both 2D and 3D image inputs

Architecture Improvements:
-------------------------
- Removes 1x1 expand convolutions from Fire modules
- Uses higher squeeze ratios for better information flow
- V1: SR=0.25 with s1x1=16
- V2: SR=0.50 with s1x1=32 (except last 2 modules with SR=0.25)

Model Variants:
--------------
- SqueezeNodule-Net V1: Lighter, 15.8% fewer parameters than SqueezeNet
- SqueezeNodule-Net V2: Better accuracy, 23% more parameters but faster convergence

Performance Improvements (2D):
-----------------------------
- V1: 93.2% accuracy, 94.6% specificity, 89.2% sensitivity
- V2: 94.3% accuracy, 95.3% specificity, 91.3% sensitivity

Usage Examples:
-------------
```python
# For lung nodule classification
model = SqueezeNoduleNetV2(num_classes=2, input_shape=(50, 50, 1))

# For 3D CT scans
model = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2,
                                        input_shape=(32, 32, 32, 1))

# For standard ImageNet
model = create_squeezenodule_net_v2("v2", num_classes=1000)
```
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
class SimplifiedFireModule(keras.layers.Layer):
    """
    Simplified Fire module - the core building block of SqueezeNodule-Net.

    A simplified version of the Fire module that removes the 1x1 expand convolution,
    keeping only the 3x3 expand convolution. This reduces parameters while maintaining
    spatial and channel expansion capabilities.

    **Architecture**:
    ```
    Input → Squeeze(1x1) → ReLU → Expand(3x3 only) → ReLU → Output
    ```

    Args:
        s1x1: Number of filters in squeeze layer (all 1x1).
        e3x3: Number of 3x3 filters in expand layer.
        kernel_regularizer: Regularizer for convolution kernels.
        kernel_initializer: Initializer for convolution kernels.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, e3x3)`.

    Note:
        The squeeze ratio (SR) is defined as s1x1 / e3x3.
        Unlike standard Fire modules, there are no 1x1 expand filters.
    """

    def __init__(
            self,
            s1x1: int,
            e3x3: int,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if s1x1 <= 0 or e3x3 <= 0:
            raise ValueError("All filter counts must be positive integers")
        if s1x1 >= e3x3:
            raise ValueError("Squeeze filters should be less than expand filters for compression")

        # Store configuration
        self.s1x1 = s1x1
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

        # Create simplified expand layer (3x3 convolution only)
        self.expand_3x3 = layers.Conv2D(
            filters=e3x3,
            kernel_size=3,
            padding='same',  # Maintain spatial dimensions
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name='expand_3x3'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the Simplified Fire module by building all sub-layers."""
        # Build squeeze layer
        self.squeeze.build(input_shape)

        # Compute squeeze output shape
        squeeze_output_shape = self.squeeze.compute_output_shape(input_shape)

        # Build expand layer with squeeze output shape
        self.expand_3x3.build(squeeze_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the Simplified Fire module."""
        # Squeeze
        squeezed = self.squeeze(inputs, training=training)

        # Expand (3x3 only)
        output = self.expand_3x3(squeezed, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape of Simplified Fire module."""
        output_shape = list(input_shape)
        output_shape[-1] = self.e3x3
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            's1x1': self.s1x1,
            'e3x3': self.e3x3,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SqueezeNoduleNetV2(keras.Model):
    """
    SqueezeNodule-Net V2 model implementation.

    An improved SqueezeNet architecture that achieves better accuracy with
    simplified Fire modules. V2 uses a heavier squeeze layer for better
    compression and feature extraction.

    Args:
        num_classes: Integer, number of output classes for classification.
        variant_config: Dictionary defining the Fire module configurations.
        dropout_rate: Float, dropout rate after final Fire module.
        kernel_regularizer: Regularizer for all convolution kernels.
        kernel_initializer: Initializer for all convolution kernels.
        include_top: Boolean, whether to include the classification head.
        use_3d: Boolean, whether to use 3D convolutions for volumetric data.
        input_shape: Tuple, input shape (height, width, channels) or
                     (depth, height, width, channels) for 3D.
        **kwargs: Additional arguments for Model base class.

    Raises:
        ValueError: If invalid configuration is provided.

    Example:
        >>> # Create SqueezeNodule-Net V2 for lung nodule classification
        >>> model = SqueezeNoduleNetV2.from_variant("v2", num_classes=2,
        >>>                                          input_shape=(50, 50, 1))
        >>>
        >>> # Create 3D version for CT scans
        >>> model = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2,
        >>>                                          input_shape=(32, 32, 32, 1))
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "v1": {
            "fire_configs": [
                {'s1x1': 16, 'e3x3': 64},  # fire2
                {'s1x1': 16, 'e3x3': 64},  # fire3
                {'s1x1': 32, 'e3x3': 128},  # fire4
                {'s1x1': 32, 'e3x3': 128},  # fire5
                {'s1x1': 48, 'e3x3': 192},  # fire6
                {'s1x1': 48, 'e3x3': 192},  # fire7
                {'s1x1': 64, 'e3x3': 256},  # fire8
                {'s1x1': 64, 'e3x3': 256},  # fire9
            ],
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8]  # After conv1, fire4, fire8
        },
        "v2": {
            "fire_configs": [
                {'s1x1': 32, 'e3x3': 64},  # fire2 (SR=0.50)
                {'s1x1': 32, 'e3x3': 64},  # fire3 (SR=0.50)
                {'s1x1': 64, 'e3x3': 128},  # fire4 (SR=0.50)
                {'s1x1': 64, 'e3x3': 128},  # fire5 (SR=0.50)
                {'s1x1': 96, 'e3x3': 192},  # fire6 (SR=0.50)
                {'s1x1': 96, 'e3x3': 192},  # fire7 (SR=0.50)
                {'s1x1': 64, 'e3x3': 256},  # fire8 (SR=0.25)
                {'s1x1': 64, 'e3x3': 256},  # fire9 (SR=0.25)
            ],
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8]
        },
        "v1_3d": {
            "fire_configs": [
                {'s1x1': 16, 'e3x3': 64},
                {'s1x1': 16, 'e3x3': 64},
                {'s1x1': 32, 'e3x3': 128},
                {'s1x1': 32, 'e3x3': 128},
                {'s1x1': 48, 'e3x3': 192},
                {'s1x1': 48, 'e3x3': 192},
                {'s1x1': 64, 'e3x3': 256},
                {'s1x1': 64, 'e3x3': 256},
            ],
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8],
            "use_3d": True
        },
        "v2_3d": {
            "fire_configs": [
                {'s1x1': 32, 'e3x3': 64},
                {'s1x1': 32, 'e3x3': 64},
                {'s1x1': 64, 'e3x3': 128},
                {'s1x1': 64, 'e3x3': 128},
                {'s1x1': 96, 'e3x3': 192},
                {'s1x1': 96, 'e3x3': 192},
                {'s1x1': 64, 'e3x3': 256},
                {'s1x1': 64, 'e3x3': 256},
            ],
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8],
            "use_3d": True
        }
    }

    # Architecture constants
    STEM_INITIALIZER = "glorot_uniform"
    HEAD_INITIALIZER = "glorot_uniform"

    def __init__(
            self,
            num_classes: int = 1000,
            variant_config: Optional[Dict[str, Any]] = None,
            dropout_rate: float = 0.5,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            include_top: bool = True,
            use_3d: bool = False,
            input_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (224, 224, 3),
            **kwargs: Any
    ) -> None:
        # Use default V2 configuration if none provided
        if variant_config is None:
            variant_config = self.MODEL_VARIANTS["v2"]

        # Validate inputs
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in range [0, 1)")

        # Store configuration
        self.num_classes = num_classes
        self.variant_config = variant_config
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self.use_3d = use_3d or variant_config.get("use_3d", False)
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
        self.head_layers = []

        # Build the model
        inputs = keras.Input(shape=input_shape)
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete SqueezeNodule-Net model architecture."""
        x = inputs

        # Build stem (initial convolution)
        x = self._build_stem(x)

        # Build Fire modules with pooling
        x = self._build_fire_modules(x)

        # Build classification head if requested
        if self.include_top:
            x = self._build_head(x)

        return x

    def _build_stem(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the stem (initial convolution) layer."""
        if self.use_3d:
            Conv = layers.Conv3D
            MaxPool = layers.MaxPooling3D
        else:
            Conv = layers.Conv2D
            MaxPool = layers.MaxPooling2D

        conv1 = Conv(
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
            maxpool1 = MaxPool(
                pool_size=3,
                strides=2,
                padding='valid',
                name='maxpool1'
            )
            x = maxpool1(x)
            self.pool_layers.append(maxpool1)

        return x

    def _build_fire_modules(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build all Simplified Fire modules with pooling."""
        if self.use_3d:
            MaxPool = layers.MaxPooling3D
        else:
            MaxPool = layers.MaxPooling2D

        for idx, fire_config in enumerate(self.fire_configs):
            fire_name = f'simpfire{idx + 2}'  # Fire modules start from simpfire2

            # Create and apply Simplified Fire module
            if self.use_3d:
                # For 3D, we need to create a 3D version of SimplifiedFireModule
                fire_module = self._create_3d_fire_module(
                    s1x1=fire_config['s1x1'],
                    e3x3=fire_config['e3x3'],
                    name=fire_name
                )
            else:
                fire_module = SimplifiedFireModule(
                    s1x1=fire_config['s1x1'],
                    e3x3=fire_config['e3x3'],
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_initializer=self.kernel_initializer,
                    name=fire_name
                )
            x = fire_module(x)
            self.fire_modules.append(fire_module)

            # Add pooling after specific Fire modules
            fire_number = idx + 2  # Convert to 1-based fire module number
            if fire_number in self.pool_indices:
                pool_layer = MaxPool(
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    name=f'pool{len(self.pool_layers) + 1}'
                )
                x = pool_layer(x)
                self.pool_layers.append(pool_layer)

        # Add dropout after last Fire module
        dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='drop9'
        )
        x = dropout(x)
        self.head_layers.append(dropout)

        return x

    def _create_3d_fire_module(
            self,
            s1x1: int,
            e3x3: int,
            name: str
    ) -> keras.Sequential:
        """Create a 3D version of the Simplified Fire module."""
        return keras.Sequential([
            layers.Conv3D(
                filters=s1x1,
                kernel_size=1,
                activation='relu',
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                name=f'{name}_squeeze'
            ),
            layers.Conv3D(
                filters=e3x3,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                name=f'{name}_expand'
            )
        ], name=name)

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the classification head."""
        if self.use_3d:
            Conv = layers.Conv3D
            GlobalPool = layers.GlobalAveragePooling3D
        else:
            Conv = layers.Conv2D
            GlobalPool = layers.GlobalAveragePooling2D

        # Final convolution for classification
        conv10 = Conv(
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
        globalpool = GlobalPool(name='globalpool')
        x = globalpool(x)
        self.head_layers.append(globalpool)

        # Softmax activation for binary classification
        if self.num_classes == 2:
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        final_activation = layers.Activation(activation, name='predictions')
        x = final_activation(x)
        self.head_layers.append(final_activation)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (224, 224, 3),
            **kwargs: Any
    ) -> "SqueezeNoduleNetV2":
        """
        Create a SqueezeNodule-Net model from a predefined variant.

        Args:
            variant: String, one of "v1", "v2", "v1_3d", "v2_3d"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape
            **kwargs: Additional arguments passed to the constructor

        Returns:
            SqueezeNoduleNetV2 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # SqueezeNodule-Net V2 for lung nodule classification
            >>> model = SqueezeNoduleNetV2.from_variant("v2", num_classes=2,
            >>>                                          input_shape=(50, 50, 1))
            >>>
            >>> # 3D version for CT scans
            >>> model = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2,
            >>>                                          input_shape=(32, 32, 32, 1))
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
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'include_top': self.include_top,
            'use_3d': self.use_3d,
            'input_shape': self._input_shape
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SqueezeNoduleNetV2":
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

        logger.info("\nSqueezeNodule-Net V2 Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - 3D mode: {self.use_3d}")
        logger.info(f"  - Number of Fire modules: {len(self.fire_configs)}")

        # Calculate and display squeeze ratios
        logger.info("  - Squeeze Ratios:")
        for i, config in enumerate(self.fire_configs):
            sr = config['s1x1'] / config['e3x3']
            logger.info(f"    - Fire{i + 2}: SR={sr:.2f} (s1x1={config['s1x1']}, e3x3={config['e3x3']})")

        logger.info(f"  - Conv1 filters: {self.conv1_filters}")
        logger.info(f"  - Conv1 kernel: {self.conv1_kernel}")
        logger.info(f"  - Conv1 stride: {self.conv1_stride}")
        logger.info(f"  - Pooling after modules: {self.pool_indices}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")

        # Calculate total parameters
        total_params = self.count_params()
        logger.info(f"  - Total parameters: {total_params:,}")

        # Compare with original SqueezeNet
        squeezenet_params = 1_248_424  # Original SqueezeNet parameter count
        reduction = (squeezenet_params - total_params) / squeezenet_params * 100
        if reduction > 0:
            logger.info(f"  - Parameter reduction vs SqueezeNet: {reduction:.1f}%")
        else:
            logger.info(f"  - Parameter increase vs SqueezeNet: {-reduction:.1f}%")

# ---------------------------------------------------------------------

def create_squeezenodule_net_v2(
        variant: str = "v2",
        num_classes: int = 1000,
        input_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (224, 224, 3),
        weights: Optional[str] = None,
        **kwargs: Any
) -> SqueezeNoduleNetV2:
    """
    Convenience function to create SqueezeNodule-Net V2 models.

    Args:
        variant: String, model variant ("v1", "v2", "v1_3d", "v2_3d")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape
        weights: String, pretrained weights to load (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        SqueezeNoduleNetV2 model instance

    Example:
        >>> # Create SqueezeNodule-Net V2 for lung nodules
        >>> model = create_squeezenodule_net_v2("v2", num_classes=2,
        >>>                                     input_shape=(50, 50, 1))
        >>>
        >>> # Create V1 (lighter version)
        >>> model = create_squeezenodule_net_v2("v1", num_classes=2,
        >>>                                     input_shape=(50, 50, 1))
        >>>
        >>> # Create 3D version for CT volumes
        >>> model = create_squeezenodule_net_v2("v2_3d", num_classes=2,
        >>>                                     input_shape=(32, 32, 32, 1))
    """
    if weights is not None:
        logger.info("Warning: Pretrained weights are not yet implemented")

    model = SqueezeNoduleNetV2.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------
