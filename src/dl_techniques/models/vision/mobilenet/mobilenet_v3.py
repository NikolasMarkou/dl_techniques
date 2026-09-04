"""``MobileNetV3``, the NAS-searched Large/Small layer tables with squeeze-and-excitation and hard-swish, plus the ``create_mobilenetv3`` factory.

The block is still V2's inverted residual with a linear bottleneck; what
changed is that per-layer expansion factors, kernel sizes, widths, and
strides come from platform-aware NAS (MnasNet-style search, then NetAdapt
trimming against measured latency) rather than hand design. `LARGE_CONFIG`
and `SMALL_CONFIG` transcribe the paper's searched tables row for row.
Squeeze-and-excitation reweights channels using a pooled global context,
enabled only on the rows the search selected. Hard-swish,
`x * ReLU6(x + 3) / 6`, replaces swish in the lower-resolution half of the
network, where its lower per-position cost matters most. The head pools
first, then runs the wide projection on a `1x1` tensor, instead of
projecting at full spatial resolution before pooling.

The squeeze-and-excitation gate here is a plain sigmoid; the paper
specifies hard-sigmoid. The shared universal block always builds its
expansion convolution, so Large's first row (expansion 16 into 16
channels, no expansion in the paper) carries an extra `C -> C` projection.
The classifier ends in softmax, so this model outputs probabilities, not
logits. No pretrained checkpoints ship with this package;
`pretrained=True` raises `NotImplementedError`.

References:
    - Howard et al., 2019. Searching for MobileNetV3.
      (https://arxiv.org/abs/1905.02244)
    - Tan et al., 2019. MnasNet: Platform-Aware Neural Architecture Search for
      Mobile. (https://arxiv.org/abs/1807.11626)
    - Yang et al., 2018. NetAdapt: Platform-Aware Neural Network Adaptation for
      Mobile Applications. (https://arxiv.org/abs/1804.03230)
    - Hu et al., 2017. Squeeze-and-Excitation Networks.
      (https://arxiv.org/abs/1709.01507)
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear Bottlenecks.
      (https://arxiv.org/abs/1801.04381)
"""

import keras
from keras import layers, regularizers, initializers
from typing import Tuple, Optional, Dict, Any, Literal, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.hard_swish import HardSwish
from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.vision.mobilenet.common import (
    REFERENCE_BN_EPSILON,
    REFERENCE_BN_MOMENTUM,
    materialize_for_summary,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# MobileNetV3 Model
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mobilenet.mobilenet_v3")
class MobileNetV3(keras.Model):
    """NAS-searched mobile classifier with squeeze-and-excitation and hard-swish.

    Architecture:

    .. code-block:: text

        image  [H, W, 3]
           |
           v
        Conv2D 3x3 s2 -> BN -> hard-swish   (16*alpha channels)
           |
           v
        UniversalInvertedBottleneck x N   (LARGE_CONFIG or SMALL_CONFIG row per block)
           |
           v
        Conv2D 1x1 -> BN -> hard-swish   (960 Large / 576 Small, alpha-scaled)
           |
        include_top=False --------+
           |                      |
           v                      v
        GlobalAvgPool         feature map (returned as-is)
           |
        Dense -> hard-swish -> Dropout -> Dense (softmax)
           |
           v
        class probabilities  [num_classes]

    :param num_classes: Number of output classes.
    :param variant: `"large"` or `"small"`.
    :param width_multiplier: Channel-count multiplier.
    :param dropout_rate: Dropout rate before the classifier.
    :param weight_decay: L2 regularization factor.
    :param kernel_initializer: Weight initialization strategy.
    :param include_top: Whether to include the classification head.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param kwargs: Passthrough to `keras.Model`.

    Example::

        model = MobileNetV3(num_classes=1000, variant="large")
        small = MobileNetV3(num_classes=10, variant="small", input_shape=(32, 32, 3))
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "large": {"variant": "large"},
        "small": {"variant": "small"},
    }

    # Model configurations from the paper
    LARGE_CONFIG = [
        # exp_size, out_channels, kernel_size, stride, use_se, activation
        # Stage 1
        (16, 16, 3, 1, False, "relu"),  # 112x112
        # Stage 2
        (64, 24, 3, 2, False, "relu"),  # 56x56
        (72, 24, 3, 1, False, "relu"),
        # Stage 3
        (72, 40, 5, 2, True, "relu"),  # 28x28
        (120, 40, 5, 1, True, "relu"),
        (120, 40, 5, 1, True, "relu"),
        # Stage 4
        (240, 80, 3, 2, False, "hard_swish"),  # 14x14
        (200, 80, 3, 1, False, "hard_swish"),
        (184, 80, 3, 1, False, "hard_swish"),
        (184, 80, 3, 1, False, "hard_swish"),
        (480, 112, 3, 1, True, "hard_swish"),
        (672, 112, 3, 1, True, "hard_swish"),
        # Stage 5
        (672, 160, 5, 2, True, "hard_swish"),  # 7x7
        (960, 160, 5, 1, True, "hard_swish"),
        (960, 160, 5, 1, True, "hard_swish"),
    ]

    SMALL_CONFIG = [
        # exp_size, out_channels, kernel_size, stride, use_se, activation
        # Stage 1
        (16, 16, 3, 2, True, "relu"),  # 56x56
        # Stage 2
        (72, 24, 3, 2, False, "relu"),  # 28x28
        (88, 24, 3, 1, False, "relu"),
        # Stage 3
        (96, 40, 5, 2, True, "hard_swish"),  # 14x14
        (240, 40, 5, 1, True, "hard_swish"),
        (240, 40, 5, 1, True, "hard_swish"),
        (120, 48, 5, 1, True, "hard_swish"),
        (144, 48, 5, 1, True, "hard_swish"),
        # Stage 4
        (288, 96, 5, 2, True, "hard_swish"),  # 7x7
        (576, 96, 5, 1, True, "hard_swish"),
        (576, 96, 5, 1, True, "hard_swish"),
    ]

    def __init__(
            self,
            num_classes: int = 1000,
            variant: Literal["large", "small"] = "large",
            width_multiplier: float = 1.0,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-5,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, int, int]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if variant not in ["large", "small"]:
            raise ValueError(f"Unknown variant '{variant}'. Must be 'large' or 'small'.")

        if input_shape and len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.variant = variant
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.include_top = include_top
        self.input_shape_config = input_shape or (224, 224, 3)

        if variant == "large":
            self.block_configs = self.LARGE_CONFIG
            self.last_block_filters = 960
            self.last_conv_filters = 1280
        else:
            self.block_configs = self.SMALL_CONFIG
            self.last_block_filters = 576
            self.last_conv_filters = 1024

        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        def make_divisible(value: float, divisor: int = 8) -> int:
            """Round `value` to the nearest multiple of `divisor`, without dropping more than 10% below it."""
            new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
            if new_value < 0.9 * value:
                new_value += divisor
            return new_value

        first_filter = make_divisible(16 * width_multiplier)
        self.stem_conv = layers.Conv2D(
            first_filter,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        self.stem_bn = layers.BatchNormalization(
            momentum=REFERENCE_BN_MOMENTUM,
            epsilon=REFERENCE_BN_EPSILON,
            name="stem_bn",
        )
        self.stem_activation = HardSwish(name="stem_hard_swish")

        self.blocks = []
        in_channels = first_filter

        for i, (exp_size, out_size, kernel, stride, use_se, activation) in enumerate(self.block_configs):
            exp_channels = make_divisible(exp_size * self.width_multiplier)
            out_channels = make_divisible(out_size * self.width_multiplier)

            # expanded_channels is passed explicitly since expansion is not always an integer multiple of input channels.
            block = UniversalInvertedBottleneck(
                filters=out_channels,
                expanded_channels=exp_channels,
                kernel_size=kernel,
                stride=stride,
                use_squeeze_excitation=use_se,
                activation_type=activation,
                normalization_type='batch_norm',
                # DECISION plan-2026-08-22T035419-a11304c8/D-203: pass epsilon explicitly; see mobilenet_v1.py for the measured impact. See decisions.md.
                normalization_args={'epsilon': REFERENCE_BN_EPSILON},
                use_bias=False,
                use_dw1=True,
                use_dw2=False,
                se_ratio=0.25,
                se_activation='relu',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"block_{i}"
            )
            self.blocks.append(block)
            in_channels = out_channels

        last_block_filters = make_divisible(self.last_block_filters * width_multiplier)
        self.last_conv = layers.Conv2D(
            last_block_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="last_conv"
        )
        self.last_bn = layers.BatchNormalization(
            momentum=REFERENCE_BN_MOMENTUM,
            epsilon=REFERENCE_BN_EPSILON,
            name="last_bn",
        )
        self.last_activation = HardSwish(name="last_hard_swish")

        if self.include_top:
            self.global_pool = layers.GlobalAveragePooling2D(name="global_pool")

            last_conv_filters = make_divisible(self.last_conv_filters * width_multiplier)
            self.head_conv = layers.Dense(
                last_conv_filters,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="head_conv"
            )
            self.head_activation = HardSwish(name="head_hard_swish")
            self.dropout = layers.Dropout(dropout_rate, name="dropout")

            self.classifier = layers.Dense(
                num_classes,
                activation='softmax',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the stem, inverted residual blocks, last conv, and optional head.

        :param x: Input images, shape `(B, H, W, C)`.
        :param training: Passed to batch norm and dropout.
        :return: Class probabilities, or the 4-D feature map if `include_top=False`.
        """
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_activation(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.last_conv(x)
        x = self.last_bn(x, training=training)
        x = self.last_activation(x)

        if self.include_top:
            x = self.global_pool(x)
            x = self.head_conv(x)
            x = self.head_activation(x)
            x = self.dropout(x, training=training)
            x = self.classifier(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: Literal["large", "small"],
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, int, int]] = None,
            width_multiplier: float = 1.0,
            **kwargs
    ) -> "MobileNetV3":
        """Create a MobileNetV3 model from a predefined variant.

        :param variant: `"large"` or `"small"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param width_multiplier: Channel-count multiplier.
        :param kwargs: Passthrough to the constructor.
        :return: A configured `MobileNetV3` instance.

        Example::

            model = MobileNetV3.from_variant("large", num_classes=1000)
            small = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        logger.info(f"Creating MobileNetV3-{variant.capitalize()} model")

        return cls(
            num_classes=num_classes,
            variant=variant,
            width_multiplier=width_multiplier,
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "variant": self.variant,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
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

        total_params = self.count_params()

        logger.info("MobileNetV3 Configuration:")
        logger.info(f"  - Variant: {self.variant}")
        logger.info(f"  - Input shape: {self.input_shape_config}")
        logger.info(f"  - Width multiplier: {self.width_multiplier}")
        logger.info(f"  - Number of blocks: {len(self.blocks)}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv3(
        variant: Literal["large", "small"] = "large",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, int, int]] = None,
        width_multiplier: float = 1.0,
        pretrained: bool = False,
        **kwargs
) -> MobileNetV3:
    """Create a MobileNetV3 model.

    :param variant: Model variant: `"large"` or `"small"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param width_multiplier: Channel-count multiplier.
    :param pretrained: Must be `False`; `True` raises `NotImplementedError`
        since no MobileNetV3 checkpoints ship with this package.
    :param kwargs: Passthrough to the constructor.
    :return: A configured `MobileNetV3` instance.

    Example::

        model = create_mobilenetv3("large", num_classes=1000)
        small = create_mobilenetv3("small", num_classes=10, input_shape=(32, 32, 3))
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise on pretrained=True, do not warn and return random weights.
    # See decisions.md.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained MobileNetV3 weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_mobilenetv3('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )

    model = MobileNetV3.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=width_multiplier,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------
