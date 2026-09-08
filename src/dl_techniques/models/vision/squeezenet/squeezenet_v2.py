"""SqueezeNoduleNetV2, a SqueezeNet variant for lung nodule classification, with the ``create_squeezenodule_net_v2`` factory.

This file holds `SimplifiedFireModule`, the `SqueezeNoduleNetV2` functional
model with its `MODEL_VARIANTS` table, and the `create_squeezenodule_net_v2`
factory. The stem, eight-block body and classification head follow
SqueezeNet, with the Fire module replaced by `SimplifiedFireModule`, which
drops the 1x1 expand path and keeps only the 3x3 one, so every expanded
feature comes from a 3x3 neighbourhood. The squeeze ratio (squeeze filters
over expand filters) is 0.25 or 0.50 rather than SqueezeNet's 0.125, so the
bottleneck between stages is wider. Callers should know that the head applies
softmax at every `num_classes`, that the `_3d` variants force `use_3d` on and
need a four-element `input_shape`, that every downsampling stage uses
`padding='valid'` and so all axes must be at least 35, and that no pretrained
weights ship with this port.

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
        normalizes by fan_in, so its Keras equivalent is `lecun_uniform` rather
        than `glorot_uniform`; see `caffe_reference_init.py`.
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
    """Squeeze to a 1x1 bottleneck, then expand through a 3x3 convolution only.

    The squeeze convolution sets the width the expand convolution reads. Both
    carry a ReLU, and the 3x3 uses ``padding='same'``, so the spatial dimensions
    are preserved. This layer is used by the 2D variants; the 3D variants build
    the same two convolutions as a ``keras.Sequential`` instead.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ conv 1x1, relu               │  squeeze, s1x1
        └──────────────────────────────┘
              │ [B, H, W, s1x1]
              ▼
        ┌──────────────────────────────┐
        │ conv 3x3, same, relu         │  expand, e3x3
        └──────────────────────────────┘
              │
              ▼
        output [B, H, W, e3x3]

    Input shape:
        4D tensor `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor `(batch_size, height, width, e3x3)`.

    :param s1x1: Number of 1x1 filters in the squeeze layer. Must be positive and
        smaller than ``e3x3``.
    :type s1x1: int
    :param e3x3: Number of 3x3 filters in the expand layer. Must be positive.
    :type e3x3: int
    :param kernel_regularizer: Regularizer for both convolution kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for both convolution kernels. Defaults
        to the Caffe ``xavier`` transcription.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Passthrough to `keras.layers.Layer`.
    :raises ValueError: If ``s1x1`` or ``e3x3`` is not positive, or if ``s1x1`` is
        not smaller than ``e3x3``.

    Note:
        The squeeze ratio is `s1x1 / e3x3`, so the constructor's check holds it
        below 1. There is no 1x1 expand path, unlike the standard Fire module.
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
        """Build the squeeze and expand sub-layers.

        :param input_shape: Shape of the input tensor ``(batch, H, W, C)``.
        :type input_shape: tuple of int or None
        """
        self.squeeze.build(input_shape)

        # The expand convolution reads the squeeze output, not the module input.
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
        :type inputs: keras.KerasTensor
        :param training: Passed to the squeeze and expand convolutions.
        :type training: bool or None
        :return: Output tensor, shape `(B, H, W, e3x3)`.
        :rtype: keras.KerasTensor
        """
        squeezed = self.squeeze(inputs, training=training)
        output = self.expand_3x3(squeezed, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the module.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple of int or None
        :return: Input shape with the channel axis replaced by ``e3x3``.
        :rtype: tuple of int or None
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.e3x3
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        :return: Configuration dictionary containing every constructor parameter.
        :rtype: dict
        """
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
    """Assemble a stem, eight simplified Fire blocks, and a head, in 2D or 3D.

    The graph is built in ``__init__`` and handed to ``keras.Model``, so the
    instance is a functional model. The head applies softmax, so its output is a
    probability vector rather than logits.

    Architecture:

    .. code-block:: text

        image or volume [B, (D,) H, W, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ conv1, relu                  │  96 filters, stride 2
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ maxpool 3x3, stride 2        │  (pool index 1)
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ simpfire2 .. simpfire9       │  maxpool after
        │                              │  simpfire4, simpfire8
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dropout, drop9               │
        └──────────────────────────────┘
              │ [B, (d,) h, w, 256]
              ├──► output [B, (d,) h, w, 256]   (include_top=False)
              ▼
        ┌──────────────────────────────┐
        │ conv10 1x1, relu             │
        └──────────────────────────────┘
              │ [B, (d,) h, w, num_classes]
              ▼
        ┌──────────────────────────────┐
        │ global average pool          │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ softmax                      │
        └──────────────────────────────┘
              │
              ▼
        output [B, num_classes]

    Dropout sits before the fork, so it is applied on both leaves.

    2D and 3D paths:

    .. code-block:: text

        use_3d = False               use_3d = True

        input [H, W, C]              input [D, H, W, C]
        Conv2D, MaxPooling2D         Conv3D, MaxPooling3D
        SimplifiedFireModule         Sequential of two Conv3D
        GlobalAveragePooling2D       GlobalAveragePooling3D

    The 3D blocks are plain Sequentials, so they do not run
    ``SimplifiedFireModule``'s filter-count checks.

    Variants:

    .. code-block:: text

        variant  squeeze ratio fire2-7  fire8-9  convolutions
        v1                        0.25     0.25  2D
        v2                        0.50     0.25  2D
        v1_3d                     0.25     0.25  3D
        v2_3d                     0.50     0.25  3D

    All four share one stem (96 filters, kernel 7, stride 2), one pooling
    schedule (after conv1, simpfire4 and simpfire8) and the same expand widths
    (64, 64, 128, 128, 192, 192, 256, 256).

    :param num_classes: Number of output classes. Must be positive.
    :type num_classes: int
    :param variant_config: A `MODEL_VARIANTS` entry defining the Fire block
        configs, stem and pooling. Defaults to the `"v2"` entry.
    :type variant_config: dict or None
    :param dropout_rate: Dropout rate after the final Fire block. Must be in
        `[0, 1)`.
    :type dropout_rate: float
    :param kernel_regularizer: Regularizer for all convolution kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for the Fire block convolutions. The
        stem and head use `STEM_INITIALIZER` and `HEAD_INITIALIZER` instead.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param use_3d: Use 3D convolutions for volumetric data. It is combined with
        the variant's own setting by ``or``, so a `_3d` variant stays 3D even
        when this is `False`.
    :type use_3d: bool
    :param input_shape: `(height, width, channels)`, or `(depth, height,
        width, channels)` for 3D. The default is the 2D shape, so a 3D variant
        needs an explicit four-element value.
    :type input_shape: tuple of 3 or 4 ints
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If `num_classes` is not positive, `dropout_rate` is
        outside `[0, 1)`, or the input's spatial extent is below the shared
        minimum of 35 on every axis (2D and 3D alike, since all four variants
        share one stem and pooling schedule). Every downsampling stage uses
        `padding='valid'`, so a collapsed axis would otherwise yield an all-NaN
        output of the correct shape.

    Example::

        model = SqueezeNoduleNetV2.from_variant("v2", num_classes=2, input_shape=(50, 50, 1))
        # every axis needs at least 35; a 32-voxel cube collapses the last pooling stage.
        ct = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=2, input_shape=(48, 48, 48, 1))
    """

    MODEL_VARIANTS = {
        "v1": {
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
            "pool_indices": [1, 4, 8]
        },
        "v2": {
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

    # DECISION plan-2026-08-23T091307-9a110062/D-481: keep these two distinct; they
    # transcribe different Caffe fillers. See decisions.md.
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

        # DECISION plan-2026-08-17T183311-79c63e38/D-020: validate here, not in
        # build(); by then the all-NaN graph is already assembled. See decisions.md.
        validate_spatial_extent(input_shape[:-1], variant_config, type(self).__name__)

        self.num_classes = num_classes
        self.variant_config = variant_config
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        # A `_3d` variant wins over use_3d=False; the argument can only turn 3D on.
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
        """Build the stem, the Fire stack and, if requested, the head.

        :param inputs: The model's input tensor.
        :type inputs: keras.KerasTensor
        :return: The model's output tensor.
        :rtype: keras.KerasTensor
        """
        x = inputs

        x = self._build_stem(x)

        x = self._build_fire_modules(x)

        if self.include_top:
            x = self._build_head(x)

        return x

    def _build_stem(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the initial convolution and its pooling.

        :param x: The model's input tensor.
        :type x: keras.KerasTensor
        :return: Tensor after conv1 and, when pool index 1 is present, the pool.
        :rtype: keras.KerasTensor
        """
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
        """Build the eight Fire blocks, their pooling, and the dropout.

        :param x: Tensor coming out of the stem.
        :type x: keras.KerasTensor
        :return: Tensor after the last Fire block, its pooling and dropout.
        :rtype: keras.KerasTensor
        """
        if self.use_3d:
            MaxPool = layers.MaxPooling3D
        else:
            MaxPool = layers.MaxPooling2D

        for idx, fire_config in enumerate(self.fire_configs):
            # The stem's conv1 is layer 1, so the Fire blocks start at simpfire2.
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

            # pool_indices counts layers, so it names Fire blocks 2..9.
            fire_number = idx + 2
            if fire_number in self.pool_indices:
                pool_layer = MaxPool(
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    # Numbered by position in pool_layers, not by fire number.
                    name=f'pool{len(self.pool_layers) + 1}'
                )
                x = pool_layer(x)
                self.pool_layers.append(pool_layer)

        # Outside the loop and before the head branch, so include_top=False keeps it.
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
        """Build the squeeze and expand convolutions as a 3D Sequential block.

        This is the 3D stand-in for :class:`SimplifiedFireModule`, with the same
        two convolutions and activations. Being a bare Sequential, it applies none
        of that class's filter-count checks, and its inner layers are named
        ``<name>_squeeze`` and ``<name>_expand``.

        :param s1x1: Number of 1x1 filters in the squeeze convolution.
        :type s1x1: int
        :param e3x3: Number of 3x3 filters in the expand convolution.
        :type e3x3: int
        :param name: Name for the Sequential and the prefix for its sub-layers.
        :type name: str
        :return: A Sequential holding the two Conv3D layers.
        :rtype: keras.Sequential
        """
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
        """Build the classification head.

        The 1x1 convolution carries a ReLU before the pooling and softmax, as the
        reference prototxts do.

        :param x: Tensor coming out of the Fire stack.
        :type x: keras.KerasTensor
        :return: Class probabilities ``(batch, num_classes)``.
        :rtype: keras.KerasTensor
        """
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

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-063: softmax at every
        # num_classes; a 2-way sigmoid head would not sum to 1. See decisions.md.
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

        The chokepoint both public entry points reach, so the ``weights`` guard
        lives here.

        :param variant: One of `"v1"`, `"v2"`, `"v1_3d"`, `"v2_3d"`.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input shape. The `_3d` variants need four elements;
            the default is the 2D shape.
        :type input_shape: tuple of 3 or 4 ints
        :param kwargs: Passthrough to the constructor. A non-`None` `weights`
            here raises `NotImplementedError`.
        :return: A configured `SqueezeNoduleNetV2` instance.
        :rtype: SqueezeNoduleNetV2
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
            # Without the pop, `**kwargs` swallows `weights` and returns a random
            # model.
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
        """Return the model configuration for serialization.

        :return: Configuration dictionary containing every constructor parameter.
        :rtype: dict
        """
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
        """Rebuild the model from a configuration dictionary.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: dict
        :return: Reconstructed model instance.
        :rtype: SqueezeNoduleNetV2
        """
        if config.get('kernel_regularizer'):
            config['kernel_regularizer'] = regularizers.deserialize(
                config['kernel_regularizer']
            )
        if config.get('kernel_initializer'):
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-129: drop only the Functional
        # keys __init__ rebuilds; adding 'name' renames a nested backbone on reload
        # and weight_transfer.py's name-keyed map then misses it. See decisions.md.
        for key in ('layers', 'input_layers', 'output_layers'):
            config.pop(key, None)

        return cls(**config)

    def summary_with_details(self) -> None:
        """Print the Keras summary, then log the resolved configuration.

        The reduction line compares against a fixed 1000-class SqueezeNet count,
        so it is only a like-for-like comparison at ``num_classes=1000``.

        :return: None.
        :rtype: None
        """
        self.summary()

        logger.info("\nSqueezeNodule-Net V2 Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - 3D mode: {self.use_3d}")
        logger.info(f"  - Number of Fire modules: {len(self.fire_configs)}")

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

        total_params = self.count_params()
        logger.info(f"  - Total parameters: {total_params:,}")

        # The published count for the 1000-class SqueezeNet.
        squeezenet_params = 1_248_424
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
    """Create a SqueezeNodule-Net V2 model from a variant name.

    :param variant: Model variant: `"v1"`, `"v2"`, `"v1_3d"`, `"v2_3d"`.
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Input shape. The `_3d` variants need four elements; the
        default is the 2D shape.
    :type input_shape: tuple of 3 or 4 ints
    :param weights: Unsupported; any non-`None` value raises `NotImplementedError`.
    :type weights: str or None
    :param kwargs: Passthrough to the model constructor.
    :return: A configured `SqueezeNoduleNetV2` instance.
    :rtype: SqueezeNoduleNetV2
    :raises NotImplementedError: If `weights` is not `None`.
    :raises ValueError: If `variant` is not recognized, or `input_shape`'s
        spatial extent is below 35 on any axis.

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