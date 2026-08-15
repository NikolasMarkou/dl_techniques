"""
MobileNetV3 image classifier: the searched Large/Small layer tables, with
squeeze-and-excitation and hard-swish, assembled from `UniversalInvertedBottleneck`.

Where V2 contributed a block, V3 contributes a *layout*. The block is still V2's
inverted residual with a linear bottleneck; what changed is that the per-layer
expansion factors, kernel sizes, widths and strides stopped being hand-designed and
became the output of a search — platform-aware NAS over a MnasNet-style
factorized space to fix the block structure, then NetAdapt to trim each layer's
channel count against measured on-device latency rather than against FLOPs. The
resulting tables are irregular in a way no human schedule is: expansion sizes like
200, 184, 480, 672 that are not integer multiples of their input width, `5x5`
depthwise kernels in some stages and `3x3` in others, and ReLU in the early
high-resolution stages switching to hard-swish only once resolution has dropped.

`LARGE_CONFIG` and `SMALL_CONFIG` here are those tables transcribed
row-for-row from the paper (Table 1 and Table 2 of *Searching for MobileNetV3*),
`(exp_size, out_channels, kernel, stride, use_se, activation)` per row, and they do
match it — including the details a re-derivation usually gets wrong: Large's first
block having expansion 16 into 16 channels, Small starting with a stride-2 SE block,
and the ReLU-to-hard-swish switch landing at the 240/80 block in Large and at the
96/40 block in Small.

Two mechanisms are grafted onto the block. Squeeze-and-excitation, applied inside
the expanded space after the depthwise convolution and before the projection,
pools each channel to a scalar and predicts a per-channel gate from it, which lets
the block reweight channels using global context that a depthwise filter can never
see; V3 enables it only in the rows the search selected, at a reduction of `1/4` of
the *expanded* channels. Hard-swish replaces swish/SiLU with

`h-swish(x) = x * ReLU6(x + 3) / 6`

which keeps swish's smooth non-monotonic shape but needs no sigmoid or exponential,
so it costs a few piecewise-linear ops and quantizes cleanly. It is used only in
the later, lower-resolution half of the network, because the activation's cost is
paid per spatial position and the early stages have the most of those.

The head is the third contribution and is a pure latency optimization: rather than
running the final `1x1` expansion at `7x7` and then pooling, V3 pools first and
runs the expensive 1280-wide projection on a `1x1` tensor, deleting the previous
generation's bottleneck-and-projection pair entirely. This module implements that
head as `GlobalAveragePooling -> Dense(1280 or 1024) -> hard-swish -> dropout ->
Dense(num_classes)`, which is arithmetically the paper's post-pool `1x1`
convolutions; the last block convolution (960 for Large, 576 for Small) still runs
pre-pool with batch norm and hard-swish, as in the paper.

Three deviations follow from building on the shared universal block rather than a
bespoke one, and none of them is visible from the variant tables. The
squeeze-and-excitation gate in `UniversalInvertedBottleneck` is a plain **sigmoid**,
where the paper specifies hard-sigmoid — same shape, different numerics and a
different quantization story. The universal block always constructs its expansion
`1x1` + norm + activation, so Large's first row (expansion 16 into 16 channels,
where the paper omits the expansion convolution) carries an extra `C -> C`
projection here. And the classifier ends in softmax, so this model emits
probabilities rather than logits: compile with `from_logits=False`.

`width_multiplier` scales both the expansion sizes and the output widths through
the same round-to-multiple-of-8 rule V2 uses, so it reproduces the paper's
`alpha` family; it does not scale the head's 1280/1024 projection independently of
that rule.

`pretrained=True` on `create_mobilenetv3` raises `NotImplementedError`. No
checkpoints ship with this package. It used to succeed with random weights, with
only a log line distinguishing that from a real load; the contract in
`resnet/model.py` now holds here too. Warm-start from a local file with
`model.load_weights(path)`.

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
from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.mobilenet.common import materialize_for_summary

# ---------------------------------------------------------------------
# MobileNetV3 Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MobileNetV3(keras.Model):
    """MobileNetV3 model implemented with Universal Inverted Bottleneck blocks.

    A highly efficient mobile architecture discovered through platform-aware NAS
    and optimized with NetAdapt, featuring hard-swish activation and an efficient
    last stage design. This implementation leverages the `UniversalInvertedBottleneck`
    layer for its core building blocks.

    Args:
        num_classes: Integer, number of output classes for classification.
        variant: String, model variant ("large" or "small").
        width_multiplier: Float, multiplier for the number of filters.
        dropout_rate: Float, dropout rate for regularization.
        weight_decay: Float, L2 regularization factor.
        kernel_initializer: String or initializer, weight initialization.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape. If None, defaults to (224, 224, 3).
        **kwargs: Additional keyword arguments for Model base class.

    Example:
        >>> # Create MobileNetV3-Large for ImageNet
        >>> model = MobileNetV3(num_classes=1000, variant="large")
        >>>
        >>> # Create MobileNetV3-Small for CIFAR-10
        >>> model = MobileNetV3(
        ...     num_classes=10,
        ...     variant="small",
        ...     input_shape=(32, 32, 3)
        ... )
        >>>
        >>> # With custom width multiplier
        >>> model = MobileNetV3(
        ...     num_classes=100,
        ...     variant="large",
        ...     width_multiplier=0.75
        ... )
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

        # Validate variant
        if variant not in ["large", "small"]:
            raise ValueError(f"Unknown variant '{variant}'. Must be 'large' or 'small'.")

        # Validate input shape
        if input_shape and len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        # Store configuration
        self.num_classes = num_classes
        self.variant = variant
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.include_top = include_top
        self.input_shape_config = input_shape or (224, 224, 3)

        # Get configuration for the variant
        if variant == "large":
            self.block_configs = self.LARGE_CONFIG
            self.last_block_filters = 960
            self.last_conv_filters = 1280
        else:  # small
            self.block_configs = self.SMALL_CONFIG
            self.last_block_filters = 576
            self.last_conv_filters = 1024

        # Create regularizer
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        # Helper function to make divisible
        def make_divisible(value: float, divisor: int = 8) -> int:
            """Make value divisible by divisor."""
            new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_value < 0.9 * value:
                new_value += divisor
            return new_value

        # --- Build the model layers ---

        # Initial stem
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
        self.stem_bn = layers.BatchNormalization(name="stem_bn")
        self.stem_activation = HardSwish(name="stem_hard_swish")

        # Build inverted residual blocks using UniversalInvertedBottleneck
        self.blocks = []
        in_channels = first_filter

        for i, (exp_size, out_size, kernel, stride, use_se, activation) in enumerate(self.block_configs):
            # Apply width multiplier
            exp_channels = make_divisible(exp_size * self.width_multiplier)
            out_channels = make_divisible(out_size * self.width_multiplier)

            # We pass the absolute number of expansion channels to UIB to handle cases
            # where the expansion is not a clean integer multiple of input channels.
            # The previous logic using integer division was flawed.
            block = UniversalInvertedBottleneck(
                filters=out_channels,
                expanded_channels=exp_channels,
                kernel_size=kernel,
                stride=stride,
                use_squeeze_excitation=use_se,
                activation_type=activation,
                normalization_type='batch_norm',
                use_bias=False,
                use_dw1=True,  # Standard inverted bottleneck structure
                use_dw2=False,
                se_ratio=0.25,  # MobileNetV3 uses a SE reduction of 4 (ratio=1/4)
                se_activation='relu',  # Activation before expansion in SE
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"block_{i}"
            )
            self.blocks.append(block)
            in_channels = out_channels

        # Last convolution block (efficient last stage)
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
        self.last_bn = layers.BatchNormalization(name="last_bn")
        self.last_activation = HardSwish(name="last_hard_swish")

        # Head
        if self.include_top:
            self.global_pool = layers.GlobalAveragePooling2D(name="global_pool")

            # Final convolution (acts as FC after global pooling)
            last_conv_filters = make_divisible(self.last_conv_filters * width_multiplier)
            self.head_conv = layers.Dense(
                last_conv_filters,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="head_conv"
            )
            self.head_activation = HardSwish(name="head_hard_swish")
            self.dropout = layers.Dropout(dropout_rate, name="dropout")

            # Classifier
            self.classifier = layers.Dense(
                num_classes,
                activation='softmax',  # Add softmax for classification
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the model."""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_activation(x)

        # Inverted residual blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Last convolution block
        x = self.last_conv(x)
        x = self.last_bn(x, training=training)
        x = self.last_activation(x)

        # Head
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

        Args:
            variant: String, "large" or "small"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (224, 224, 3)
            width_multiplier: Float, multiplier for filter dimensions
            **kwargs: Additional arguments passed to the constructor

        Returns:
            MobileNetV3 model instance

        Example:
            >>> model = MobileNetV3.from_variant("large", num_classes=1000)
            >>> model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
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
        """Print model summary with additional information."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: a real forward pass; the
        # keras.Input + build(shape) route marked the model built without creating
        # any sub-layer weights. See decisions.md D-065.
        materialize_for_summary(self, self.input_shape_config)

        super().summary(**kwargs)

        # Print additional model information
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
    """Convenience function to create MobileNetV3 models.

    Args:
        variant: String, model variant ("large" or "small")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        width_multiplier: Float, multiplier for filter dimensions
        pretrained: Boolean, must be False. `True` raises `NotImplementedError` —
            no MobileNetV3 checkpoints ship with this package.
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV3 model instance

    Example:
        >>> # Create MobileNetV3-Large for ImageNet
        >>> model = create_mobilenetv3("large", num_classes=1000)
        >>>
        >>> # Create MobileNetV3-Small for CIFAR-10
        >>> model = create_mobilenetv3("small", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create with custom width multiplier
        >>> model = create_mobilenetv3("large", width_multiplier=0.75)
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained MobileNetV3 weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_mobilenetv3('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras')."
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
