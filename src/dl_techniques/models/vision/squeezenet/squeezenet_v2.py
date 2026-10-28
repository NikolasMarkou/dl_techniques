"""SqueezeNoduleNetV2, a SqueezeNet variant for lung nodule classification, with the ``create_squeezenodule_net_v2`` factory.

This file holds `SimplifiedFireModule`, the `SqueezeNoduleNetV2` functional
model with its `MODEL_VARIANTS` table, and the `create_squeezenodule_net_v2`
factory. The stem, eight-block body and classification head follow
SqueezeNet, with the Fire module replaced by `SimplifiedFireModule`, which
drops the 1x1 expand path and keeps only the 3x3 one, so every expanded
feature comes from a 3x3 neighbourhood. The squeeze ratio (squeeze filters
over expand filters) is 0.25 or 0.50 rather than SqueezeNet's 0.125, so the
bottleneck between stages is wider. Callers should know that the model returns
logits in 2D and 3D alike, SqueezeNet's ReLU'd `conv10` averaged over space
with no softmax (compile with `SparseCategoricalCrossentropy(from_logits=True)`),
that the `_3d` variants force `use_3d` on and
need a four-element `input_shape`, that every downsampling stage uses
`padding='valid'` and so all axes must be at least 35, and that no pretrained
weights ship with this port (``pretrained=<path>`` loads a local checkpoint).

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
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.utils.logger import logger
from .spatial_guard import validate_spatial_extent
from .caffe_reference_init import (
    CAFFE_HEAD_INITIALIZER,
    CAFFE_XAVIER_INITIALIZER,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.model_build import (
    coerce_include_top,
    refuse_unknown_model_kwargs,
    require_channel_count,
)
from dl_techniques.utils.weight_transfer import (
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (initializers and the
# regularizer act on the starting weights and the loss, dropout in training only, the input
# size is no weight shape). `load_pretrained_checkpoint` compares every other key the
# checkpoint and the model both hold, so another `variant` or `use_3d` is refused.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "input_shape",
    "kernel_initializer",
    "kernel_regularizer",
    "dropout_rate",
)

# The one weighted layer of the head is ``conv10``; the pool after it holds no weight.
_HEAD_PATTERN = "conv10"

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
        to the Caffe ``xavier`` transcription. An ``Initializer`` instance is
        stored as its serialized spec, so each convolution draws its own kernel.
    :type kernel_initializer: str or dict or keras.initializers.Initializer
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
            kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = CAFFE_XAVIER_INITIALIZER,
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
        self.kernel_initializer = initializer_spec(kernel_initializer)

        self.squeeze = keras.layers.Conv2D(
            filters=s1x1,
            kernel_size=1,
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            name='squeeze'
        )

        self.expand_3x3 = keras.layers.Conv2D(
            filters=e3x3,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
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
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': self.kernel_initializer
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.squeezenet.squeezenet_v2")
class SqueezeNoduleNetV2(keras.Model):
    """Assemble a stem, eight simplified Fire blocks, and a head, in 2D or 3D.

    The graph is built in ``__init__`` and handed to ``keras.Model``, so the
    instance is a functional model. The output is logits, in 2D and 3D alike:
    the head is SqueezeNet's ``conv10`` (1x1, ``num_classes`` filters, ReLU)
    averaged over space, with no softmax.

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
        │ dropout                      │
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
        logits [B, num_classes]

    Dropout sits before the fork, so it is applied on both leaves.

    The head is SqueezeNet's (official v1.1 prototxt ``conv10``,
    ``relu_conv10``, global ``pool10``), which Tsivgoulis et al. 2022 keep.
    It is not the shared ``ClassificationHead``, which cannot express it: that
    head pools first and classifies after the pool, while here the classifier
    runs per position with a ReLU between it and the pool, and a
    ``layers/heads`` class-score head applies no activation. Because of the
    ReLU every logit is non-negative.

    2D and 3D paths:

    .. code-block:: text

        use_3d = False               use_3d = True

        input [H, W, C]              input [D, H, W, C]
        ConvStem, rank 2             ConvStem, rank 3
        MaxPooling2D                 MaxPooling3D
        SimplifiedFireModule         Sequential of two Conv3D
        conv10 Conv2D                conv10 Conv3D
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
    :param variant: A key of `MODEL_VARIANTS` (`"v1"`, `"v2"`, `"v1_3d"` or
        `"v2_3d"`) defining the Fire block configs, stem and pooling.
    :type variant: str
    :param dropout_rate: Dropout rate after the final Fire block. Must be in
        `[0, 1)`.
    :type dropout_rate: float
    :param kernel_regularizer: Regularizer for all convolution kernels,
        ``conv10``'s included.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for the Fire block convolutions,
        stored as a string or a serialized spec (an ``Initializer`` instance is
        serialized once) and handed to every convolution, so no two of them
        replay one draw. The stem and head use `STEM_INITIALIZER` and
        `HEAD_INITIALIZER` instead.
    :type kernel_initializer: str or dict or keras.initializers.Initializer
    :param include_top: Whether to include the classification head
        (``conv10`` and the global pool). A ``bool`` or ``numpy.bool_``; anything
        else, a string included, raises ``TypeError``.
    :type include_top: bool
    :param use_3d: Use 3D convolutions for volumetric data. It is combined with
        the variant's own setting by ``or``, so a `_3d` variant stays 3D even
        when this is `False`.
    :type use_3d: bool
    :param input_shape: `(height, width, channels)`, or `(depth, height,
        width, channels)` for 3D. `None` is the 2D shape `(224, 224, 3)`, so a
        3D variant needs an explicit four-element value.
    :type input_shape: tuple of 3 or 4 ints or None
    :param kwargs: `name`, `trainable` or `dtype`, passed to `keras.Model`.
        Anything else raises `ValueError`, because Keras' functional
        constructor would drop it silently.
    :raises TypeError: If `include_top` is not a `bool` or `numpy.bool_`.
    :raises ValueError: If `variant` is not a key of `MODEL_VARIANTS`,
        `input_shape` has not three entries (four for a 3D model) or no channel
        count, `num_classes` is not positive, `dropout_rate` is outside
        `[0, 1)`, or the input's spatial extent is below the shared minimum of
        35 on every axis (2D and 3D alike, since all four variants share one
        stem and pooling schedule). Every downsampling stage uses
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
            variant: str = "v2",
            dropout_rate: float = 0.5,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = CAFFE_XAVIER_INITIALIZER,
            include_top: bool = True,
            use_3d: bool = False,
            input_shape: Optional[Union[Tuple[int, int, int], Tuple[int, int, int, int]]] = None,
            **kwargs: Any
    ) -> None:
        refuse_unknown_model_kwargs(kwargs, type(self).__name__)
        include_top = coerce_include_top(include_top)
        if variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(self.MODEL_VARIANTS.keys())}"
            )
        variant_config = self.MODEL_VARIANTS[variant]

        if num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in range [0, 1), got {dropout_rate}")

        # A `_3d` variant wins over use_3d=False; the argument can only turn 3D on.
        use_3d = use_3d or variant_config.get("use_3d", False)
        if input_shape is None:
            input_shape = (224, 224, 3)
        input_shape = tuple(input_shape)
        if len(input_shape) != (4 if use_3d else 3):
            raise ValueError(
                "input_shape must be (depth, height, width, channels) for a 3D model and "
                f"(height, width, channels) for a 2D one; use_3d={use_3d}, got {input_shape}"
            )
        require_channel_count(input_shape)

        # DECISION plan-2026-08-17T183311-79c63e38/D-020: validate here, not in
        # build(); by then the all-NaN graph is already assembled. See decisions.md.
        validate_spatial_extent(input_shape[:-1], variant_config, type(self).__name__)

        self.num_classes = num_classes
        self.variant = variant
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        # DECISION plan-2026-10-26T125959-7af851e5/D-033: store the initializer as a spec and hand that spec to every Fire block; never keep the caller's Initializer instance.
        # One seedless instance given to all of them replayed a single draw across 120 of 153 kernel pairs (SqueezeNoduleNet v2), and an instance in get_config is not JSON-safe. See decisions.md.
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.include_top = include_top
        self.use_3d = use_3d
        self.input_shape_config = input_shape

        self.fire_configs = variant_config["fire_configs"]
        self.conv1_filters = variant_config["conv1_filters"]
        self.conv1_kernel = variant_config["conv1_kernel"]
        self.conv1_stride = variant_config["conv1_stride"]
        self.pool_indices = variant_config["pool_indices"]

        self.stem: Optional[ConvStem] = None
        self.fire_modules = []
        self.pool_layers = []
        self.head_layers = []

        inputs = keras.Input(shape=input_shape)
        outputs = self._build_model(inputs)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created SqueezeNoduleNetV2 '{variant}' ({'3D' if self.use_3d else '2D'}) "
            f"for input {self.input_shape_config} with {len(self.fire_configs)} Fire blocks"
        )

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
        """Build the initial convolution and its pooling as one ``ConvStem``.

        The stem is ``ConvStem`` ``stem`` in 2D and 3D alike (``rank`` 2 or 3):
        ``stem_conv1`` (a ``Conv2D`` or ``Conv3D``; ReLU as the separate
        ``stem_act1`` layer, no normalization) and, when pool index 1 is
        present, ``stem_pool``. The pool is also appended to ``pool_layers``.

        :param x: The model's input tensor.
        :type x: keras.KerasTensor
        :return: Tensor after conv1 and, when pool index 1 is present, the pool.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-10-26T125959-7af851e5/D-004: ConvStem is 2D only,
        # so the 3D stem stays hand-rolled here. Do NOT add a `rank` option to
        # ConvStem to absorb it: that buys 2 of this model's 5 rank-dependent
        # sites (stem conv, stem pool, Fire-loop pool, 3D Fire, head GAP3D) for
        # one caller. See decisions.md.
        # DECISION plan-2026-10-26T125959-7af851e5/D-030: (2026-10-28) the D-004 text above
        # is superseded. ConvStem takes `rank` (2 or 3), so this one call builds the 2D stem
        # and the 3D stem; the Conv3D that was written out here is gone. Do NOT write the
        # 3D stem by hand again: the EQ cases `squeezenodule-v2-3d-*` hold it to the old
        # Conv3D at atol 0. See decisions.md.
        self.stem = ConvStem(
            filters=self.conv1_filters,
            kernel_size=self.conv1_kernel,
            strides=self.conv1_stride,
            padding='same' if self.conv1_stride == 1 else 'valid',
            use_bias=True,
            num_convs=1,
            normalization_type=None,
            activation_type='relu',
            use_pooling=1 in self.pool_indices,
            pool_size=3,
            pool_strides=2,
            pool_padding='valid',
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            rank=3 if self.use_3d else 2,
            name='stem'
        )
        x = self.stem(x)

        if self.stem.pool is not None:
            self.pool_layers.append(self.stem.pool)

        return x

    def _build_fire_modules(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the eight Fire blocks, their pooling, and the dropout.

        :param x: Tensor coming out of the stem.
        :type x: keras.KerasTensor
        :return: Tensor after the last Fire block, its pooling and dropout.
        :rtype: keras.KerasTensor
        """
        if self.use_3d:
            MaxPool = keras.layers.MaxPooling3D
        else:
            MaxPool = keras.layers.MaxPooling2D

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
                    name=f'maxpool{fire_number}'
                )
                x = pool_layer(x)
                self.pool_layers.append(pool_layer)

        # Outside the loop and before the head branch, so include_top=False keeps it.
        dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name='dropout'
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
            keras.layers.Conv3D(
                filters=s1x1,
                kernel_size=1,
                activation='relu',
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                name=f'{name}_squeeze'
            ),
            keras.layers.Conv3D(
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
        """Build SqueezeNet's head: ``conv10`` with its ReLU, then a global average pool.

        One path for 2D and 3D; only the convolution and pooling rank differ.
        Nothing follows the pool: the output is the logits.

        :param x: Tensor coming out of the Fire stack.
        :type x: keras.KerasTensor
        :return: Logits ``(batch, num_classes)``.
        :rtype: keras.KerasTensor
        """
        if self.use_3d:
            Conv = keras.layers.Conv3D
            GlobalPool = keras.layers.GlobalAveragePooling3D
        else:
            Conv = keras.layers.Conv2D
            GlobalPool = keras.layers.GlobalAveragePooling2D

        # DECISION plan-2026-10-14T062616-81938e81/D-017
        # conv10 + ReLU + global average pool is the paper's head, built here,
        # not by the shared ClassificationHead: that head classifies AFTER its
        # pool with no activation, and cannot put the ReLU between a per-position
        # classifier and the pool. Do NOT add a softmax after the pool: the
        # output is the logits. See decisions.md.
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

        avgpool = GlobalPool(name='avgpool')
        x = avgpool(x)
        self.head_layers.append(avgpool)

        return x

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load a local `.keras` checkpoint into the model, layer by layer.

        The transfer is `load_pretrained_checkpoint`: the model is built at
        construction, so its variables are kept and the checkpoint's are copied
        in, a headed checkpoint into a headless model included. Every layer
        outside ``conv10`` must be restored, so a checkpoint of another variant
        or of the other rank raises and copies no weight. Only a ``conv10`` of
        another `num_classes` is skipped: it keeps its initialization and a
        warning names it. The archive's own config is compared too: a
        constructor argument that both hold and that is not in
        `_NON_STRUCTURAL_CONFIG_KEYS` must be equal. An argument only one side
        holds is not compared, and an archive in which no single model of this
        class can be found is checked by layers only (the load warns).

        :param weights_path: Path to the weights file (`.keras` format).
        :type weights_path: str
        :param skip_mismatch: `True` (default) lets ``conv10`` differ; `False`
            allows no difference anywhere. The inverse of the transfer helper's
            `strict`.
        :type skip_mismatch: bool
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If a layer outside the head is not restored, or the
            weights cannot be loaded.
        """
        load_pretrained_checkpoint(
            self, weights_path, self.input_shape_config, skip_mismatch,
            head_pattern=_HEAD_PATTERN,
            ignore_config_keys=_NON_STRUCTURAL_CONFIG_KEYS,
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Union[Tuple[int, int, int], Tuple[int, int, int, int]]] = None,
            pretrained: Union[bool, str] = False,
            **kwargs: Any
    ) -> "SqueezeNoduleNetV2":
        """Create a SqueezeNodule-Net model from a predefined variant.

        :param variant: One of `"v1"`, `"v2"`, `"v1_3d"`, `"v2_3d"`.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input shape. The `_3d` variants need four elements;
            `None` is the 2D shape `(224, 224, 3)`.
        :type input_shape: tuple of 3 or 4 ints or None
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no SqueezeNodule-Net weights ship with
            `dl_techniques`), or `False` (default, random init).
        :type pretrained: bool or str
        :param kwargs: Passthrough to the constructor (`dropout_rate`,
            `kernel_regularizer`, ...). `weights` is no argument and is refused
            by name.
        :return: A configured `SqueezeNoduleNetV2` instance.
        :rtype: SqueezeNoduleNetV2
        :raises ValueError: If `variant` is not recognized, `input_shape`'s
            spatial extent is below the computed minimum of 35 (shared by
            all four variants, which share one stem and pooling schedule), or a
            keyword is not an argument.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

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

        weights_path = pretrained_path_or_raise(
            pretrained, f"{cls.__name__} variant '{variant}'"
        )

        model = cls(
            num_classes=num_classes,
            variant=variant,
            input_shape=input_shape,
            **kwargs
        )

        if weights_path:
            model.load_pretrained_weights(weights_path)

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Configuration dictionary containing every constructor parameter.
        :rtype: dict
        """
        config = {
            'num_classes': self.num_classes,
            'variant': self.variant,
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': self.kernel_initializer,
            'include_top': self.include_top,
            'use_3d': self.use_3d,
            'input_shape': self.input_shape_config,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SqueezeNoduleNetV2":
        """Rebuild the model from a configuration dictionary.

        :param config: Configuration dictionary produced by ``get_config``; it
            is not modified.
        :type config: dict
        :return: Reconstructed model instance.
        :rtype: SqueezeNoduleNetV2
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-129: drop only the Functional
        # keys __init__ rebuilds; adding 'name' renames a nested backbone on reload
        # and weight_transfer.py's name-keyed map then misses it. See decisions.md.
        config = {
            key: value for key, value in config.items()
            if key not in ('layers', 'input_layers', 'output_layers')
        }

        return cls(**config)

# ---------------------------------------------------------------------

def create_squeezenodule_net_v2(
        variant: str = "v2",
        num_classes: int = 1000,
        input_shape: Optional[Union[Tuple[int, int, int], Tuple[int, int, int, int]]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs: Any
) -> SqueezeNoduleNetV2:
    """Create a SqueezeNodule-Net V2 model from a variant name.

    :param variant: Model variant: `"v1"`, `"v2"`, `"v1_3d"`, `"v2_3d"`.
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Input shape. The `_3d` variants need four elements;
        `None` is the 2D shape `(224, 224, 3)`.
    :type input_shape: tuple of 3 or 4 ints or None
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :type pretrained: bool or str
    :param kwargs: Passthrough to the model constructor.
    :return: A configured `SqueezeNoduleNetV2` instance.
    :rtype: SqueezeNoduleNetV2
    :raises ValueError: If `variant` is not recognized, or `input_shape`'s
        spatial extent is below 35 on any axis.
    :raises NotImplementedError: If `pretrained` is `True`.
    :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

    Example::

        model = create_squeezenodule_net_v2("v2", num_classes=2, input_shape=(50, 50, 1))
        # 48 voxels per axis; minimum is 35.
        ct = create_squeezenodule_net_v2("v2_3d", num_classes=2, input_shape=(48, 48, 48, 1))
    """
    return SqueezeNoduleNetV2.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )

# ---------------------------------------------------------------------
