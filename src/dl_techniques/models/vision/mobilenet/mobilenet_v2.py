"""``MobileNetV2``, an inverted-residual classifier built from `UniversalInvertedBottleneck` blocks, plus the ``create_mobilenetv2`` factory.

Unlike a ResNet bottleneck (wide-narrow-wide, residual on the wide
tensors), V2 is narrow-wide-narrow: each block expands `C -> t*C`, filters
depthwise in that expanded space, projects back to a narrow `C_out`, and
runs the residual on the narrow tensors. Only the thin block boundaries
need to persist between blocks. The final projection of every block has
no activation, since a ReLU there would collapse the low-dimensional
representation the projection produces.

Channel counts round through `_make_divisible` (nearest multiple of 8, no
more than 10% below `width_multiplier * c`), so `width_multiplier=0.75`
does not give exactly `0.75 * c`. The final 1280-channel convolution is
not scaled down for `width_multiplier <= 1.0`, only widened above it. The
paper's first stage uses expansion factor 1 and skips the expansion
convolution; this implementation reuses one universal block that always
expands, so that stage carries a small extra `C -> C` projection. The
classifier ends in softmax, so this model outputs probabilities, not
logits. No pretrained checkpoints ship with this package;
`pretrained=True` raises `NotImplementedError`.

References:
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear Bottlenecks.
      (https://arxiv.org/abs/1801.04381)
    - Howard et al., 2017. MobileNets: Efficient Convolutional Neural Networks for
      Mobile Vision Applications. (https://arxiv.org/abs/1704.04861)
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
    - Qin et al., 2024. MobileNetV4: Universal Models for the Mobile Ecosystem.
      (https://arxiv.org/abs/2404.10518) — source of the Universal Inverted
      Bottleneck this module is built from.
"""

import keras
from keras import layers, regularizers
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.vision.mobilenet.common import (
    REFERENCE_BN_EPSILON,
    REFERENCE_BN_MOMENTUM,
    materialize_for_summary,
)
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mobilenet.mobilenet_v2")
class MobileNetV2(keras.Model):
    """Inverted-residual classifier built from Universal Inverted Bottleneck blocks.

    Architecture:

    .. code-block:: text

        image  [H, W, 3]
           |
           v
        Conv2D 3x3 s2 -> BN -> ReLU6   (32*alpha channels)
           |
           v
        17x UniversalInvertedBottleneck  (7 stages, ARCHITECTURE table)
           |
           v
        Conv2D 1x1 -> BN -> ReLU6   (1280 channels, or wider above alpha=1.0)
           |
        include_top=False --------+
           |                      |
           v                      v
        GlobalAvgPool         feature map (returned as-is)
           |
        Dropout -> Dense (softmax)
           |
           v
        class probabilities  [num_classes]

    :param num_classes: Number of output classes.
    :param width_multiplier: Channel-count multiplier (alpha).
    :param dropout_rate: Dropout rate before the classifier.
    :param weight_decay: L2 regularization factor for conv and dense layers.
    :param kernel_initializer: Weight initialization strategy.
    :param include_top: Whether to include the pooling and classification head.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param kwargs: Passthrough to `keras.Model`.

    Input shape:
        3D tensor `(height, width, channels)`, e.g. `(224, 224, 3)`.

    Output shape:
        `(batch_size, num_classes)` if `include_top=True`, else a 4-D feature map.

    Example::

        model = MobileNetV2(num_classes=1000, width_multiplier=1.0)
        small = MobileNetV2(num_classes=10, width_multiplier=0.75, input_shape=(32, 32, 3))
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "large": {"width_multiplier": 1.4},
        "medium": {"width_multiplier": 1.0},
        "small": {"width_multiplier": 0.75},
        "nano": {"width_multiplier": 0.5},
        "pico": {"width_multiplier": 0.35},
    }

    # Architecture definition from Table 2 of the paper: (t, c, n, s)
    # t: expansion factor, c: output channels, n: repetitions, s: stride
    ARCHITECTURE = [
        (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
        (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
    ]

    def __init__(
            self,
            num_classes: int = 1000,
            width_multiplier: float = 1.0,
            dropout_rate: float = 0.2,
            weight_decay: float = 4e-5,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if width_multiplier <= 0:
            raise ValueError(f"width_multiplier must be positive, got {width_multiplier}")
        self.input_shape_config = input_shape or (224, 224, 3)
        if len(self.input_shape_config) != 3:
            raise ValueError(f"input_shape must be a 3D tuple, got {self.input_shape_config}")

        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        self._build_model_layers()

    def _make_divisible(self, v: float, divisor: int = 8) -> int:
        """Round `v` to the nearest multiple of `divisor`, without dropping more than 10% below it."""
        new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _build_model_layers(self) -> None:
        """Create every sub-layer from the stored configuration."""
        first_channels = self._make_divisible(32 * self.width_multiplier)
        self.initial_conv = layers.Conv2D(
            first_channels, 3, strides=2, padding='same', use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name='conv1'
        )
        self.initial_bn = layers.BatchNormalization(
            momentum=REFERENCE_BN_MOMENTUM,
            epsilon=REFERENCE_BN_EPSILON,
            name='conv1_bn',
        )
        self.initial_relu = layers.ReLU(max_value=6, name='conv1_relu6')

        self.blocks = []
        block_id = 0
        for t, c, n, s in self.ARCHITECTURE:
            output_channels = self._make_divisible(c * self.width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(UniversalInvertedBottleneck(
                    filters=output_channels,
                    expansion_factor=t,
                    stride=stride,
                    kernel_size=3,
                    use_dw1=True,
                    use_dw2=False,
                    activation_type='relu',
                    activation_args={'max_value': 6},
                    normalization_type='batch_norm',
                    # DECISION plan-2026-08-22T035419-a11304c8/D-203: pass epsilon explicitly; see mobilenet_v1.py for the measured impact. See decisions.md.
                    normalization_args={'epsilon': REFERENCE_BN_EPSILON},
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'block_{block_id}'
                ))
                block_id += 1

        # Matches the paper: the last convolution's channels are not scaled down for width_multiplier <= 1.0.
        if self.width_multiplier > 1.0:
            last_channels = self._make_divisible(1280 * self.width_multiplier)
        else:
            last_channels = 1280
        self.last_conv = layers.Conv2D(
            last_channels, 1, padding='same', use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name='conv_last'
        )
        self.last_bn = layers.BatchNormalization(
            momentum=REFERENCE_BN_MOMENTUM,
            epsilon=REFERENCE_BN_EPSILON,
            name='conv_last_bn',
        )
        self.last_relu = layers.ReLU(max_value=6, name='conv_last_relu6')

        if self.include_top:
            self.global_avg_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')
            if self.dropout_rate > 0:
                self.dropout = layers.Dropout(self.dropout_rate, name='dropout')
            else:
                self.dropout = None
            self.classifier = layers.Dense(
                self.num_classes, activation='softmax', name='classifier',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from `input_shape` by tracing `call`.

        `materialize_sublayers` traces `call` on symbolic inputs, so what
        gets built cannot drift from what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to `call`.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the stem, bottleneck blocks, final conv, and optional classification head.

        :param inputs: Input images, shape `(B, H, W, C)`.
        :param training: Passed to batch norm and dropout.
        :return: Class probabilities, or the 4-D feature map if `include_top=False`.
        """
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.last_conv(x)
        x = self.last_bn(x, training=training)
        x = self.last_relu(x)

        if self.include_top:
            x = self.global_avg_pool(x)
            if self.dropout:
                x = self.dropout(x, training=training)
            x = self.classifier(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            width_multiplier: float = 1.0,
            **kwargs: Any
    ) -> "MobileNetV2":
        """Create a MobileNetV2 model from a predefined variant.

        :param variant: One of `"large"`, `"medium"`, `"small"`, `"nano"`, `"pico"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param width_multiplier: Extra multiplier applied on top of the variant default.
        :param kwargs: Passthrough to the constructor.
        :return: A configured `MobileNetV2` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}")

        config = cls.MODEL_VARIANTS[variant]
        effective_width = config["width_multiplier"] * width_multiplier
        logger.info(f"Creating MobileNetV2-{variant} (α={effective_width})")

        return cls(
            num_classes=num_classes,
            width_multiplier=effective_width,
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
        })
        return config

    def summary(self, **kwargs):
        """Print the model summary, plus configuration and parameter count."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: materialize with a real forward pass.
        # The keras.Input + build(shape) route marked the model built without creating any sub-layer weights. See decisions.md.
        materialize_for_summary(self, self.input_shape_config)

        super().summary(**kwargs)

        total_blocks = len(self.blocks)
        total_params = self.count_params()

        logger.info("MobileNetV2 Configuration:")
        logger.info(f"  - Input shape: {self.input_shape_config}")
        logger.info(f"  - Width multiplier (α): {self.width_multiplier}")
        logger.info(f"  - Number of bottleneck blocks: {total_blocks}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Weight decay: {self.weight_decay}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv2(
        variant: str = "medium",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        width_multiplier: float = 1.0,
        pretrained: bool = False,
        **kwargs: Any
) -> MobileNetV2:
    """Create a MobileNetV2 model.

    :param variant: Model variant: `"large"`, `"medium"`, `"small"`, `"nano"`, `"pico"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param width_multiplier: Extra multiplier applied on top of the variant default.
    :param pretrained: Must be `False`; `True` raises `NotImplementedError`
        since no MobileNetV2 checkpoints ship with this package.
    :param kwargs: Passthrough to the constructor.
    :return: A configured `MobileNetV2` instance.

    Example::

        model = create_mobilenetv2("medium", num_classes=1000)
        small = create_mobilenetv2("nano", num_classes=10, input_shape=(32, 32, 3))
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise on pretrained=True, do not warn and return random weights.
    # See decisions.md.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained MobileNetV2 weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_mobilenetv2('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )

    model = MobileNetV2.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=width_multiplier,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------
