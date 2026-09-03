"""``SqueezeNoduleNetV2``, a SqueezeNet variant for medical imaging (lung nodule classification), plus the ``create_squeezenodule_net_v2`` factory.

It keeps SqueezeNet's stem, eight-Fire-module body, and classification
head, but replaces the Fire module with `SimplifiedFireModule`, which
drops the 1x1 expand path and keeps only the 3x3 path, forcing every
feature to come from local spatial context. It also raises the squeeze
ratio (squeeze filters / expand filters) well above SqueezeNet's
aggressive 0.125, to 0.25 or 0.50, widening the information bottleneck for
richer per-stage features at the cost of some parameter efficiency.

References:
    -   Tsivgoulis et al., "An improved SqueezeNet model for the diagnosis
        of lung cancer in CT scans" (2022).
        https://doi.org/10.1016/j.mlwa.2022.100399
    -   Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer
        parameters and <0.5MB model size" (2016).
        https://arxiv.org/abs/1602.07360
    -   Official SqueezeNet Caffe prototxts. Tsivgoulis et al. 2022 specify no
        initialization; this model inherits SqueezeNet's macro-architecture, so
        it inherits SqueezeNet's published fillers (`xavier` on conv1 and every
        fire convolution, gaussian std=0.01 on conv10). Caffe's `xavier`
        normalizes by fan_in, so its Keras equivalent is `lecun_uniform`, NOT
        `glorot_uniform` -- see `caffe_reference_init.py`.
        https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt
"""

import keras
from keras import layers, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .spatial_guard import validate_spatial_extent
from .caffe_reference_init import (
    CAFFE_HEAD_INITIALIZER,
    CAFFE_XAVIER_INITIALIZER,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.squeezenet.squeezenet_v2")
class SimplifiedFireModule(keras.layers.Layer):
    """The core building block of SqueezeNoduleNetV2: squeeze, then a 3x3-only expand.

    Architecture:

    .. code-block:: text

        input  [B, H, W, C]
           |
           v
        Conv2D 1x1 -> ReLU   (squeeze, s1x1 filters)
           |
           v
        Conv2D 3x3 -> ReLU   (expand, e3x3 filters)
           |
           v
        output  [B, H, W, e3x3]

    :param s1x1: Number of 1x1 filters in the squeeze layer.
    :param e3x3: Number of 3x3 filters in the expand layer.
    :param kernel_regularizer: Regularizer for convolution kernels.
    :param kernel_initializer: Initializer for convolution kernels.
    :param kwargs: Passthrough to `keras.layers.Layer`.

    Input shape:
        4D tensor `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor `(batch_size, height, width, e3x3)`.

    Note:
        The squeeze ratio is `s1x1 / e3x3`. Unlike the standard Fire
        module, there is no 1x1 expand path.
    """

    def __init__(
            self,
            s1x1: int,
            e3x3: int,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = CAFFE_XAVIER_INITIALIZER,
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

        self.expand_3x3 = layers.Conv2D(
            filters=e3x3,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name='expand_3x3'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the squeeze and expand sub-layers."""
        self.squeeze.build(input_shape)

        squeeze_output_shape = self.squeeze.compute_output_shape(input_shape)
        self.expand_3x3.build(squeeze_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Squeeze, then expand.

        :param inputs: Input tensor, shape `(B, H, W, C)`.
        :param training: Passed to the squeeze and expand convolutions.
        :return: Output tensor, shape `(B, H, W, e3x3)`.
        """
        squeezed = self.squeeze(inputs, training=training)
        output = self.expand_3x3(squeezed, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, channel count replaced by `e3x3`."""
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

@register_dl_technique("dl_techniques.models.squeezenet.squeezenet_v2")
class SqueezeNoduleNetV2(keras.Model):
    """SqueezeNoduleNetV2: a stem, eight SimplifiedFireModules, and a classification head, in 2D or 3D.

    Architecture:

    .. code-block:: text

        image (or volume)  [B, H, W, (D,) C]
           |
           v
        Conv2D/3D -> ReLU -> MaxPool   (stem)
           |
           v
        SimplifiedFireModule x 8   (pooled after fire4 & fire8)
           |
           v
        Dropout -> Conv2D/3D 1x1 -> ReLU -> GlobalAvgPool
           |
           v
        class probabilities  [num_classes]

    :param num_classes: Number of output classes.
    :param variant_config: A `MODEL_VARIANTS` entry defining the Fire module configs.
    :param dropout_rate: Dropout rate after the final Fire module.
    :param kernel_regularizer: Regularizer for all convolution kernels.
    :param kernel_initializer: Initializer for all convolution kernels.
    :param include_top: Whether to include the classification head.
    :param use_3d: Use 3D convolutions for volumetric data.
    :param input_shape: `(height, width, channels)`, or `(depth, height,
        width, channels)` for 3D.
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If the configuration is invalid, or the input's
        spatial extent is below the shared minimum of 35 on every axis
        (2D and 3D alike, since all four variants share one stem and
        pooling schedule) — every downsampling stage uses
        `padding='valid'`, and a collapsed axis would otherwise yield an
        all-NaN output of the correct shape.

    Example::

        model = SqueezeNoduleNetV2.from_variant("v2", num_classes=2, input_shape=(50, 50, 1))
        # every axis needs at least 35; a 32-voxel cube collapses the last pooling stage.
        ct = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2, input_shape=(48, 48, 48, 1))
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

    # DECISION plan-2026-08-23T091307-9a110062/D-481: keep STEM_INITIALIZER and HEAD_INITIALIZER distinct; do not collapse to one value or glorot_uniform.
    # They transcribe different Caffe fillers (25 xavier convs vs conv10's gaussian); see caffe_reference_init.py. HEAD_INITIALIZER stays a serialized config so consumers get a fresh instance. See decisions.md.
    STEM_INITIALIZER = CAFFE_XAVIER_INITIALIZER
    HEAD_INITIALIZER = CAFFE_HEAD_INITIALIZER

    def __init__(
            self,
            num_classes: int = 1000,
            variant_config: Optional[Dict[str, Any]] = None,
            dropout_rate: float = 0.5,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = CAFFE_XAVIER_INITIALIZER,
            include_top: bool = True,
            use_3d: bool = False,
            input_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (224, 224, 3),
            **kwargs: Any
    ) -> None:
        if variant_config is None:
            variant_config = self.MODEL_VARIANTS["v2"]

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in range [0, 1)")

        # DECISION plan-2026-08-17T183311-79c63e38/D-020: validate here, in __init__, not in build().
        # By the time a functional Model's build() would run, the all-NaN graph is already assembled from super().__init__(inputs=..., outputs=...). Covers 3D variants too. See decisions.md.
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
            fire_name = f'simpfire{idx + 2}'

            if self.use_3d:
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

            fire_number = idx + 2
            if fire_number in self.pool_indices:
                pool_layer = MaxPool(
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    name=f'pool{len(self.pool_layers) + 1}'
                )
                x = pool_layer(x)
                self.pool_layers.append(pool_layer)

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
            kernel_initializer=dict(self.HEAD_INITIALIZER),
            name='conv10'
        )
        x = conv10(x)
        self.head_layers.append(conv10)

        globalpool = GlobalPool(name='globalpool')
        x = globalpool(x)
        self.head_layers.append(globalpool)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-063: softmax at every num_classes, including 2; do not restore a sigmoid special case for 2.
        # A 2-way sigmoid head would not sum to 1, while this package's num_classes=2 examples compile with categorical_crossentropy. See decisions.md.
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
        """Create a SqueezeNodule-Net model from a predefined variant.

        :param variant: One of `"v1"`, `"v2"`, `"v1_3d"`, `"v2_3d"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape.
        :param kwargs: Passthrough to the constructor. A non-`None` `weights`
            here raises `NotImplementedError`.
        :return: A configured `SqueezeNoduleNetV2` instance.
        :raises NotImplementedError: If a non-`None` `weights` is passed.
        :raises ValueError: If `variant` is not recognized, or `input_shape`'s
            spatial extent is below the computed minimum of 35 (shared by
            all four variants, which share one stem and pooling schedule).

        Example::

            model = SqueezeNoduleNetV2.from_variant("v2", num_classes=2, input_shape=(50, 50, 1))
            # 48 voxels per axis: the minimum is 35.
            ct = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2, input_shape=(48, 48, 48, 1))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        if kwargs.pop("weights", None) is not None:
            # from_variant is the chokepoint both public entry points reach; **kwargs would otherwise swallow weights silently.
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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-129: drop only the Functional-graph keys __init__ rebuilds; never add 'name' to this list.
        # Dropping 'name' too renamed a nested backbone on reload, so weight_transfer.py's name-keyed layer map silently left it at random init. See decisions.md.
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
    """Create a SqueezeNodule-Net V2 model.

    :param variant: Model variant: `"v1"`, `"v2"`, `"v1_3d"`, `"v2_3d"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape.
    :param weights: Unsupported; any non-`None` value raises `NotImplementedError`.
    :param kwargs: Passthrough to the model constructor.
    :return: A configured `SqueezeNoduleNetV2` instance.
    :raises NotImplementedError: If `weights` is not `None`.
    :raises ValueError: If `input_shape`'s spatial extent is below 35 on any axis.

    Example::

        model = create_squeezenodule_net_v2("v2", num_classes=2, input_shape=(50, 50, 1))
        # 48 voxels per axis; minimum is 35.
        ct = create_squeezenodule_net_v2("v2_3d", num_classes=2, input_shape=(48, 48, 48, 1))
    """
    return SqueezeNoduleNetV2.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        weights=weights,
        **kwargs
    )

# ---------------------------------------------------------------------
