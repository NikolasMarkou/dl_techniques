"""
SqueezeNodule-Net architecture for medical imaging.

This model presents an evolution of the SqueezeNet architecture, specifically
optimized for tasks such as lung nodule classification from CT scans. It
achieves improved accuracy and computational performance by introducing
targeted modifications to SqueezeNet's core building block, the Fire module,
and by adjusting the network's information bottleneck.

Architectural Overview:
    The macro-architecture of SqueezeNodule-Net is largely inherited from
    the original SqueezeNet: it begins with a convolutional stem, followed
    by a series of eight modified "Fire" modules interspersed with
    max-pooling layers for downsampling, and concludes with a classification
    head.

    The primary innovation lies in the micro-architecture of its fundamental
    building block, the `SimplifiedFireModule`. This module alters the
    original Fire module design in two significant ways:
    1.  It completely removes the 1x1 convolutional path within the "expand"
        layer, retaining only the 3x3 convolutional path.
    2.  It employs a different strategy for the "squeeze ratio," which
        governs the degree of channel compression.

Foundational Principles and Intuition:
    The design of SqueezeNodule-Net is motivated by hypotheses about feature
    learning in the context of medical imaging, leading to two key changes
    from the original SqueezeNet principles:

    -   Enforced Spatial Feature Extraction: The original Fire module's
        "expand" layer contained parallel 1x1 and 3x3 convolutions. The 1x1
        path learns channel-wise combinations without spatial context,
        while the 3x3 path captures local spatial patterns. By eliminating
        the 1x1 expand path, SqueezeNodule-Net forces the module to learn
        features that are exclusively derived from local spatial context.
        The underlying assumption is that for tasks like nodule
        classification, where texture and local shape are paramount, such
        spatially-aware feature learning is more effective and parameter-
        efficient than a mixed approach.

    -   Widened Information Bottleneck: The "squeeze ratio" (SR), defined
        as the ratio of squeeze filters (`s1x1`) to expand filters (`e3x3`),
        controls the severity of the information bottleneck in each module.
        The original SqueezeNet used a very low SR (e.g., 0.125), creating
        an aggressive bottleneck to maximize parameter reduction. In
        contrast, SqueezeNodule-Net variants use a significantly higher
        SR (e.g., 0.25 or 0.50). This creates a wider bottleneck, allowing
        more channels of information to flow through the module. The
        intuition is that retaining a richer feature representation at
        each stage is critical for distinguishing subtle diagnostic
        patterns in medical images, leading to faster convergence and
        higher final accuracy, even if it slightly increases the parameter
        count compared to the most aggressive SqueezeNet variants.

References:
    -   Tsivgoulis et al., "An improved SqueezeNet model for the diagnosis
        of lung cancer in CT scans" (2022).
        https://doi.org/10.1016/j.mlwa.2022.100399
    -   Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer
        parameters and <0.5MB model size" (2016).
        https://arxiv.org/abs/1602.07360
"""

import keras
from keras import layers, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .spatial_guard import validate_spatial_extent

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

        if s1x1 <= 0 or e3x3 <= 0:
            raise ValueError("All filter counts must be positive integers")
        if s1x1 >= e3x3:
            raise ValueError("Squeeze filters should be less than expand filters for compression")

        self.s1x1 = s1x1
        self.e3x3 = e3x3
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

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
        self.squeeze.build(input_shape)

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
        ValueError: If invalid configuration is provided, or if the input's
            spatial extent is below the variant's minimum (see
            `spatial_guard.minimum_spatial_extent`): every downsampling stage
            uses `padding='valid'`, and a stage that collapses an axis to length
            zero yields an all-NaN output of the correct shape. All four
            variants share one stem and pooling schedule, so the computed
            minimum is 35 on every spatial axis, 2D and 3D alike.

    Example:
        >>> # Create SqueezeNodule-Net V2 for lung nodule classification
        >>> model = SqueezeNoduleNetV2.from_variant("v2", num_classes=2,
        >>>                                          input_shape=(50, 50, 1))
        >>>
        >>> # Create 3D version for CT scans. Every variant of this model shares
        >>> # the 7x7/stride-2 stem and pools after fire4 and fire8, so the
        >>> # minimum extent on EVERY spatial axis is 35 -- a 32-voxel cube
        >>> # collapses the last pooling stage to length 0.
        >>> model = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2,
        >>>                                          input_shape=(48, 48, 48, 1))
    """

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

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in range [0, 1)")

        # DECISION plan-2026-08-17T183311-79c63e38/D-020
        # Validate here, in __init__, NOT in build(): input_shape is a required
        # constructor argument already resolved to concrete ints, and this class
        # calls super().__init__(inputs=..., outputs=...) -- by the time a
        # functional Model's build() would run, the all-NaN graph is already
        # assembled. Applies to every spatial axis, so it covers the 3D variants.
        validate_spatial_extent(input_shape[:-1], variant_config, type(self).__name__)

        self.num_classes = num_classes
        self.variant_config = variant_config
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self.use_3d = use_3d or variant_config.get("use_3d", False)
        self._input_shape = input_shape

        self.fire_configs = variant_config["fire_configs"]
        self.conv1_filters = variant_config["conv1_filters"]
        self.conv1_kernel = variant_config["conv1_kernel"]
        self.conv1_stride = variant_config["conv1_stride"]
        self.pool_indices = variant_config["pool_indices"]

        self.stem_layers = []
        self.fire_modules = []
        self.pool_layers = []
        self.head_layers = []

        inputs = keras.Input(shape=input_shape)
        outputs = self._build_model(inputs)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete SqueezeNodule-Net model architecture."""
        x = inputs

        x = self._build_stem(x)

        # Build Fire modules with pooling
        x = self._build_fire_modules(x)

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

        globalpool = GlobalPool(name='globalpool')
        x = globalpool(x)
        self.head_layers.append(globalpool)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-063: softmax at EVERY
        # num_classes, including 2. Do NOT restore the num_classes == 2 ->
        # 'sigmoid' special case: the head is Conv2D(num_classes) -> GAP, so with
        # sigmoid the two outputs are independent and do not sum to 1, while the
        # package's own num_classes=2 examples are compiled with
        # categorical_crossentropy. SqueezeNetV1 softmaxes on the same argument.
        # A single-logit sigmoid head would be the other consistent option; it is
        # rejected because it changes the output SHAPE. See decisions.md D-063.
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
            **kwargs: Additional arguments passed to the constructor. A
                non-None ``weights`` here raises NotImplementedError; no
                pretrained checkpoints are distributed with this package.

        Returns:
            SqueezeNoduleNetV2 model instance

        Raises:
            NotImplementedError: If a non-None ``weights`` is passed.
            ValueError: If variant is not recognized, or if `input_shape`'s
                spatial extent is below the computed minimum of 35 (shared by
                all four variants, which share one stem and pooling schedule).

        Example:
            >>> # SqueezeNodule-Net V2 for lung nodule classification
            >>> model = SqueezeNoduleNetV2.from_variant("v2", num_classes=2,
            >>>                                          input_shape=(50, 50, 1))
            >>>
            >>> # 3D version for CT scans (48 voxels per axis: the minimum is 35)
            >>> model = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2,
            >>>                                          input_shape=(48, 48, 48, 1))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        if kwargs.pop("weights", None) is not None:
            # Guard lives HERE, not in create_squeezenet_v2: ``from_variant`` is the
            # chokepoint both public entry points reach, and ``**kwargs``
            # swallowed ``weights`` silently, returning a random model.
            raise NotImplementedError(
                f"No pretrained SqueezeNodule-Net weights are distributed with dl_techniques. "
                f"Train from scratch, or load a local checkpoint with "
                f"keras.models.load_model()."
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
        if config.get('kernel_regularizer'):
            config['kernel_regularizer'] = regularizers.deserialize(
                config['kernel_regularizer']
            )
        if config.get('kernel_initializer'):
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-129
        # Drop only the Functional-graph keys that ``__init__`` rebuilds. Do NOT
        # add 'name' back to this list: dropping it renamed a nested backbone
        # from 'backbone' to 'squeeze_net_v1' on reload, and
        # ``utils/weight_transfer.py`` keys its layer map by ``layer.name``, so
        # the whole backbone landed in missing_in_source and kept its random
        # init while the call still returned normally.
        for key in ('layers', 'input_layers', 'output_layers'):
            config.pop(key, None)

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
        weights: Unsupported; any non-None value raises NotImplementedError.
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        SqueezeNoduleNetV2 model instance

    Raises:
        NotImplementedError: If `weights` is not None.
        ValueError: If `input_shape`'s spatial extent is below 35 on any axis.

    Example:
        >>> # Create SqueezeNodule-Net V2 for lung nodules
        >>> model = create_squeezenodule_net_v2("v2", num_classes=2,
        >>>                                     input_shape=(50, 50, 1))
        >>>
        >>> # Create V1 (lighter version)
        >>> model = create_squeezenodule_net_v2("v1", num_classes=2,
        >>>                                     input_shape=(50, 50, 1))
        >>>
        >>> # Create 3D version for CT volumes (48 voxels per axis; minimum is 35)
        >>> model = create_squeezenodule_net_v2("v2_3d", num_classes=2,
        >>>                                     input_shape=(48, 48, 48, 1))
    """
    return SqueezeNoduleNetV2.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        weights=weights,
        **kwargs
    )

# ---------------------------------------------------------------------
