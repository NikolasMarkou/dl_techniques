"""SqueezeNet v1.0/v1.1, a CNN built from Fire modules, with the ``create_squeezenet_v1`` factory.

This file holds `FireModule`, the `SqueezeNetV1` functional model with its
`MODEL_VARIANTS` table, and the `create_squeezenet_v1` factory. A Fire module
squeezes its input to a narrow 1x1 bottleneck, then expands through parallel
1x1 and 3x3 convolutions concatenated on the channel axis, so the 3x3
convolution runs on far fewer input channels than the block's own width.
Downsampling is delayed to later layers, and ResNet-style bypass connections
can be added around Fire modules. Callers should know that the head applies
softmax itself, that every downsampling stage uses `padding='valid'` and so
each variant has a minimum input extent (35 for the 1.0 stem family, 31 for
1.1) checked in the constructor, that no pretrained weights ship with this
port, and that the kernel initializers are transcribed from the official
Caffe prototxts rather than taken from Keras defaults.

References:
    -   Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer
        parameters and <0.5MB model size" (2016).
        https://arxiv.org/abs/1602.07360
    -   He et al., "Deep Residual Learning for Image Recognition" (2015)
        (for the bypass connection concept).
        https://arxiv.org/abs/1512.03385
    -   Official SqueezeNet Caffe prototxts, the source of this port's kernel
        initializers: `xavier` on conv1 and every fire convolution, gaussian
        std=0.01 on conv10. Caffe's `xavier` normalizes by fan_in, so its
        Keras equivalent is `lecun_uniform` rather than `glorot_uniform`; see
        `caffe_reference_init.py`.
        https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/train_val.prototxt
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

@register_dl_technique("dl_techniques.models.squeezenet.squeezenet_v1")
class FireModule(keras.layers.Layer):
    """Squeeze to a 1x1 bottleneck, then expand and concatenate.

    The squeeze convolution sets the width the two expand convolutions read, so
    the 3x3 convolution sees ``s1x1`` channels rather than the module's own
    output width. All three convolutions carry a ReLU. Spatial dimensions are
    preserved: the 1x1 kernels cannot shrink them and the 3x3 uses
    ``padding='same'``.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ conv 1x1, relu               │  squeeze, s1x1
        └──────────────────────────────┘
              │ [B, H, W, s1x1]
        ┌─────┴─────┐
        ▼           ▼
        ┌────────┐  ┌────────────────┐
        │ 1x1    │  │ 3x3, same      │  expand, e1x1 / e3x3
        │ relu   │  │ relu           │
        └────────┘  └────────────────┘
        │           │
        └─────┬─────┘
              ▼
        ┌──────────────────────────────┐
        │ concatenate, channel axis    │
        └──────────────────────────────┘
              │
              ▼
        output [B, H, W, e1x1 + e3x3]

    Input shape:
        4D tensor `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor `(batch_size, height, width, e1x1 + e3x3)`.

    :param s1x1: Number of 1x1 filters in the squeeze layer.
    :type s1x1: int
    :param e1x1: Number of 1x1 filters in the expand layer.
    :type e1x1: int
    :param e3x3: Number of 3x3 filters in the expand layer.
    :type e3x3: int
    :param kernel_regularizer: Regularizer for convolution kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for all three convolution kernels.
        Defaults to the Caffe ``xavier`` transcription.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Passthrough to `keras.layers.Layer`.
    :raises ValueError: If any of ``s1x1``, ``e1x1`` or ``e3x3`` is not positive.
    """

    def __init__(
            self,
            s1x1: int,
            e1x1: int,
            e3x3: int,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = CAFFE_XAVIER_INITIALIZER,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if s1x1 <= 0 or e1x1 <= 0 or e3x3 <= 0:
            raise ValueError("All filter counts must be positive integers")

        self.s1x1 = s1x1
        self.e1x1 = e1x1
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
            padding='same',
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name='expand_3x3'
        )

        self.concat = layers.Concatenate(axis=-1)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the squeeze, expand, and concatenate sub-layers.

        :param input_shape: Shape of the input tensor ``(batch, H, W, C)``.
        :type input_shape: tuple of int or None
        """
        self.squeeze.build(input_shape)

        # Both expand convolutions read the squeeze output, not the module input.
        squeeze_output_shape = self.squeeze.compute_output_shape(input_shape)

        self.expand_1x1.build(squeeze_output_shape)
        self.expand_3x3.build(squeeze_output_shape)

        expand_1x1_shape = self.expand_1x1.compute_output_shape(squeeze_output_shape)
        expand_3x3_shape = self.expand_3x3.compute_output_shape(squeeze_output_shape)
        self.concat.build([expand_1x1_shape, expand_3x3_shape])

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the forward pass through the Fire module.

        :param inputs: Input tensor ``(batch, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the three convolutions.
        :type training: bool or None
        :return: Output tensor ``(batch, H, W, e1x1 + e3x3)``.
        :rtype: keras.KerasTensor
        """
        squeezed = self.squeeze(inputs, training=training)

        expanded_1x1 = self.expand_1x1(squeezed, training=training)
        expanded_3x3 = self.expand_3x3(squeezed, training=training)

        output = self.concat([expanded_1x1, expanded_3x3])

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the Fire module.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple of int or None
        :return: Input shape with the channel axis replaced by ``e1x1 + e3x3``.
        :rtype: tuple of int or None
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.e1x1 + self.e3x3
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        :return: Configuration dictionary containing every constructor parameter.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            's1x1': self.s1x1,
            'e1x1': self.e1x1,
            'e3x3': self.e3x3,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.squeezenet.squeezenet_v1")
class SqueezeNetV1(keras.Model):
    """Assemble a stem, eight Fire modules, and a classification head.

    The graph is built in ``__init__`` and handed to ``keras.Model``, so the
    instance is a functional model. The head applies softmax, so its output is a
    probability vector rather than logits.

    Architecture:

    .. code-block:: text

        image [B, H, W, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ conv1, relu                  │  conv1_filters, stride 2
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ maxpool 3x3, stride 2        │  (pool index 1)
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ fire2 .. fire9               │  maxpool after the
        │ bypass adds optional         │  variant's fire indices
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dropout                      │
        └──────────────────────────────┘
              │ [B, h, w, 512]
              ├──► output [B, h, w, 512]     (include_top=False)
              ▼
        ┌──────────────────────────────┐
        │ conv10 1x1, relu             │
        └──────────────────────────────┘
              │ [B, h, w, num_classes]
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

    Bypass wiring:

    .. code-block:: text

        simple, at fire3, fire5, fire7, fire9

            x ──┬──► fire_i ──► (+) ──► out
                └───────────────►
            the add is made only when the widths already match

        complex, at every fire module

            x ──┬──► fire_i ────────► (+) ──► out
                └──► 1x1 conv ───────►
            the 1x1 conv appears only when the widths differ

    Variants:

    .. code-block:: text

        variant      conv1  kernel  stride  pool indices  bypass
        1.0             96       7       2  1, 4, 8       none
        1.1             64       3       2  1, 3, 5       none
        1.0_bypass      96       7       2  1, 4, 8       simple

    Index 1 pools after conv1; the others pool after that fire module. All three
    variants share the same eight Fire module widths.

    :param num_classes: Number of output classes. Must be positive.
    :type num_classes: int
    :param variant_config: A `MODEL_VARIANTS` entry defining the Fire module
        configs, stem and pooling. Defaults to the `"1.0"` entry.
    :type variant_config: dict or None
    :param use_bypass: `'simple'`, `'complex'`, `False`, or `None` to defer to
        the variant's own setting. An explicit `False` disables bypass even for a
        bypass variant. Only the two strings add connections; any other truthy
        value leaves the graph unchanged.
    :type use_bypass: bool or str or None
    :param dropout_rate: Dropout rate after the final Fire module. Must be in
        `[0, 1)`.
    :type dropout_rate: float
    :param kernel_regularizer: Regularizer for all convolution kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for the Fire module and bypass
        convolutions. The stem and head use `STEM_INITIALIZER` and
        `HEAD_INITIALIZER` instead.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param input_shape: Input shape `(height, width, channels)`.
    :type input_shape: tuple of 3 ints
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If `num_classes` is not positive, `dropout_rate` is
        outside `[0, 1)`, or the input's spatial extent is below the variant's
        minimum (35 for the "1.0" stem family, 31 for "1.1", computed from the
        variant rather than hard-coded). Every downsampling stage uses
        `padding='valid'`, so a collapsed axis would otherwise yield an all-NaN
        output of the correct shape.

    Example::

        model = SqueezeNetV1.from_variant("1.0", num_classes=1000)
        # 64px clears the "1.0" stem family's floor of 35; 32px does not.
        small = SqueezeNetV1.from_variant("1.0_bypass", num_classes=10, input_shape=(64, 64, 3))
    """

    MODEL_VARIANTS = {
        "1.0": {
            "fire_configs": [
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},
            ],
            "use_bypass": False,
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8]
        },
        "1.1": {
            "fire_configs": [
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},
            ],
            "use_bypass": False,
            "conv1_filters": 64,
            "conv1_kernel": 3,
            "conv1_stride": 2,
            "pool_indices": [1, 3, 5]
        },
        "1.0_bypass": {
            "fire_configs": [
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},
                {'s1x1': 16, 'e1x1': 64, 'e3x3': 64},
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},
                {'s1x1': 32, 'e1x1': 128, 'e3x3': 128},
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},
                {'s1x1': 48, 'e1x1': 192, 'e3x3': 192},
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},
                {'s1x1': 64, 'e1x1': 256, 'e3x3': 256},
            ],
            "use_bypass": "simple",
            "conv1_filters": 96,
            "conv1_kernel": 7,
            "conv1_stride": 2,
            "pool_indices": [1, 4, 8]
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
            use_bypass: Optional[Union[bool, str]] = None,
            dropout_rate: float = 0.5,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = CAFFE_XAVIER_INITIALIZER,
            include_top: bool = True,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            **kwargs: Any
    ) -> None:
        if variant_config is None:
            variant_config = self.MODEL_VARIANTS["1.0"]

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in range [0, 1)")

        # DECISION plan-2026-08-17T183311-79c63e38/D-020: validate here, not in
        # build(); by then the all-NaN graph is already assembled. See decisions.md.
        validate_spatial_extent(input_shape[:-1], variant_config, type(self).__name__)

        self.num_classes = num_classes
        self.variant_config = variant_config
        # DECISION plan-2026-08-17T183311-79c63e38/D-020: test `is None`, not
        # truthiness; truthiness ignores an explicit False. See decisions.md.
        self.use_bypass = (
            variant_config.get("use_bypass", False) if use_bypass is None else use_bypass
        )
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self._input_shape = input_shape

        self.fire_configs = variant_config["fire_configs"]
        self.conv1_filters = variant_config["conv1_filters"]
        self.conv1_kernel = variant_config["conv1_kernel"]
        self.conv1_stride = variant_config["conv1_stride"]
        self.pool_indices = variant_config["pool_indices"]

        self.stem_layers = []
        self.fire_modules = []
        self.pool_layers = []
        self.bypass_layers = []
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
        """Build the eight Fire modules with their pooling, bypass and dropout.

        :param x: Tensor coming out of the stem.
        :type x: keras.KerasTensor
        :return: Tensor after the last Fire module, its pooling and dropout.
        :rtype: keras.KerasTensor
        """
        bypass_indices = []
        if self.use_bypass == "simple":
            # The paper's fire3/5/7/9 sites are idx 1, 3, 5, 7, which are also the
            # only positions whose input and output widths match, as the Add needs.
            bypass_indices = [1, 3, 5, 7]
        elif self.use_bypass == "complex":
            bypass_indices = list(range(len(self.fire_configs)))

        for idx, fire_config in enumerate(self.fire_configs):
            # The stem's conv1 is layer 1, so the Fire modules start at fire2.
            fire_name = f'fire{idx + 2}'

            identity = x

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

            if idx in bypass_indices:
                if self.use_bypass == "simple" and identity.shape[-1] == x.shape[-1]:
                    add_layer = layers.Add(name=f'add_{fire_name}')
                    x = add_layer([x, identity])
                    self.bypass_layers.append(add_layer)
                elif self.use_bypass == "complex":
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

            # pool_indices counts layers, so it names fire modules 2..9.
            fire_number = idx + 2
            if fire_number in self.pool_indices:
                pool_layer = layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    name=f'maxpool{fire_number}'
                )
                x = pool_layer(x)
                self.pool_layers.append(pool_layer)

            # Dropout lands before the head fork, so include_top=False keeps it.
            if idx == len(self.fire_configs) - 1:
                dropout = layers.Dropout(
                    rate=self.dropout_rate,
                    name='dropout'
                )
                x = dropout(x)
                self.head_layers.append(dropout)

        return x

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the classification head.

        The 1x1 convolution carries a ReLU before the pooling and softmax, as the
        reference prototxts do.

        :param x: Tensor coming out of the Fire stack.
        :type x: keras.KerasTensor
        :return: Class probabilities ``(batch, num_classes)``.
        :rtype: keras.KerasTensor
        """
        conv10 = layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            activation='relu',
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=dict(self.HEAD_INITIALIZER),
            name='conv10'
        )
        x = conv10(x)
        self.head_layers.append(conv10)

        avgpool = layers.GlobalAveragePooling2D(name='avgpool')
        x = avgpool(x)
        self.head_layers.append(avgpool)

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
        """Create a SqueezeNet model from a predefined variant.

        The chokepoint both public entry points reach, so the ``weights`` guard
        lives here.

        :param variant: One of `"1.0"`, `"1.1"`, `"1.0_bypass"`.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input shape `(height, width, channels)`.
        :type input_shape: tuple of 3 ints
        :param kwargs: Passthrough to the constructor. A non-`None` `weights`
            here raises `NotImplementedError`.
        :return: A configured `SqueezeNetV1` instance.
        :rtype: SqueezeNetV1
        :raises NotImplementedError: If a non-`None` `weights` is passed.
        :raises ValueError: If `variant` is not recognized, or `input_shape`'s
            spatial extent is below the variant's computed minimum (35 for
            "1.0"/"1.0_bypass", 31 for "1.1").

        Example::

            model = SqueezeNetV1.from_variant("1.0", num_classes=1000)
            # 32px clears "1.1"'s floor of 31; the "1.0" stem family's floor is 35.
            small = SqueezeNetV1.from_variant("1.1", num_classes=10, input_shape=(32, 32, 3))
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
                f"No pretrained SqueezeNet weights are distributed with dl_techniques. "
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
        """Rebuild the model from a configuration dictionary.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: dict
        :return: Reconstructed model instance.
        :rtype: SqueezeNetV1
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

        The parameter-reduction line is the paper's headline figure, not a value
        derived from this configuration.

        :return: None.
        :rtype: None
        """
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

        # Kernel-weight count per Fire module, which the line below does not read.
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
    """Create a SqueezeNet V1 model from a variant name.

    :param variant: Model variant: `"1.0"`, `"1.1"`, `"1.0_bypass"`.
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Input shape `(height, width, channels)`.
    :type input_shape: tuple of 3 ints
    :param weights: Unsupported; any non-`None` value raises `NotImplementedError`.
    :type weights: str or None
    :param kwargs: Passthrough to the model constructor.
    :return: A configured `SqueezeNetV1` instance.
    :rtype: SqueezeNetV1
    :raises NotImplementedError: If `weights` is not `None`.
    :raises ValueError: If `variant` is not recognized, or `input_shape`'s
        spatial extent is below the variant's computed minimum (35 for
        "1.0"/"1.0_bypass", 31 for "1.1").

    Example::

        model = create_squeezenet_v1("1.0", num_classes=1000)
        # 32px clears "1.1"'s floor of 31; "1.0"/"1.0_bypass" need at least 35.
        cifar = create_squeezenet_v1("1.1", num_classes=10, input_shape=(32, 32, 3))
    """
    return SqueezeNetV1.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        weights=weights,
        **kwargs
    )

# ---------------------------------------------------------------------