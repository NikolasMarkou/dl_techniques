"""
The original MobileNet: a plain stack of depthwise separable convolutions with a
global width multiplier.

The generation's whole contribution is a factorization. A standard `KxK`
convolution does two jobs at once — it filters space and it mixes channels — and
pays for their product: `K*K*C_in*C_out*H*W` multiply-adds. Depthwise separable
convolution splits the two jobs into consecutive layers, a depthwise `KxK`
convolution that filters each input channel independently followed by a `1x1`
pointwise convolution that mixes the filtered channels:

`K*K*C_in*H*W  +  C_in*C_out*H*W`

The cost ratio against the standard convolution is `1/C_out + 1/K^2`, so for `K=3`
and any realistic channel count the block is roughly 8-9x cheaper, and the paper
reports it costs about one point of ImageNet top-1. Nothing else in this
generation is new: there are no residuals, no bottlenecks, no attention. The
network is 28 layers of that one block repeated, which is precisely why it is the
right place to read the factorization argument in isolation.

Two multipliers scale the result rather than retraining a different topology. The
width multiplier `alpha` thins every layer to `alpha * C` channels, which reduces
both parameters and compute quadratically. The resolution multiplier `rho` in the
paper shrinks the input, cutting compute quadratically at zero parameter cost.
Here `alpha` is the `width_multiplier` argument; `rho` has no argument of its own
and is expressed simply by passing a smaller `input_shape`, since the model is
fully convolutional up to the global pooling.

Architecturally this is a `3x3` stride-2 stem into `32*alpha` channels, then the
paper's thirteen separable blocks — 64; 128/s2, 128; 256/s2, 256; 512/s2 then five
at 512; 1024/s2, 1024 — with all downsampling done by the depthwise stride rather
than by pooling. The head pools globally, reshapes to `1x1xC`, applies dropout and
then a `1x1` convolution as the classifier, which is the paper's form and is
arithmetically a dense layer. The global pooling belongs to that head: with
`include_top=False` the model returns the 4-D feature map of the last block, matching
V2/V3/V4, and a caller who wants the pooled vector adds the pooling layer.

Two implementation details are easy to get wrong. The head's `Reshape` target is
hardcoded to `int(1024 * width_multiplier)`, so `include_top=True` is bound to the
final block being 1024-wide: editing `ARCHITECTURE`'s last entry breaks the head
rather than the body. And widths are scaled with a bare `int()` truncation, not the
round-to-multiple-of-8 `_make_divisible` rule that V2 and V3 use, so channel counts
here follow the paper's `alpha` table exactly but will not match TF-slim
checkpoints for fractional `alpha`.

Two deliberate deviations from the reference implementation. The blocks use plain
unbounded ReLU (the `DepthwiseSeparableBlock` factory default), where the paper's
released model uses ReLU6, which matters if low-precision quantization is the goal.
And the classifier ends in a softmax, so this model emits probabilities, not
logits: compile it with `from_logits=False`.

`pretrained=True` on `create_mobilenetv1` logs a warning and returns a randomly
initialized model. No checkpoints are distributed with this package, and the
warning is easy to miss, so pretrained weights should be treated as unsupported
here — the house contract elsewhere in `models/` (see `resnet/model.py`) is to
raise instead of silently handing back random weights, and this module predates it.

References:
    - Howard et al., 2017. MobileNets: Efficient Convolutional Neural Networks for
      Mobile Vision Applications. (https://arxiv.org/abs/1704.04861)
    - Sifre and Mallat, 2014. Rigid-Motion Scattering for Image Classification.
      (the origin of the depthwise separable factorization)
    - Chollet, 2017. Xception: Deep Learning with Depthwise Separable Convolutions.
      (https://arxiv.org/abs/1610.02357)
"""

import keras
from keras import layers, regularizers
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.depthwise_separable_block import DepthwiseSeparableBlock
from dl_techniques.models.mobilenet.common import materialize_for_summary

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileNetV1(keras.Model):
    """MobileNetV1 model implementation with depthwise separable convolutions.

    The original efficient architecture for mobile and embedded vision_heads applications
    using depthwise separable convolutions to drastically reduce computation and
    model size while maintaining good accuracy.

    Args:
        num_classes: Integer, number of output classes for classification
        width_multiplier: Float, multiplier for the number of filters (α).
            Controls model width. Common values: 1.0, 0.75, 0.5, 0.25
        dropout_rate: Float, dropout rate before the classifier
        weight_decay: Float, L2 regularization factor for all layers
        kernel_initializer: String or initializer, weight initialization strategy
        include_top: Boolean, whether to include the classification head
        input_shape: Tuple, input shape. If None, defaults to (224, 224, 3)
        **kwargs: Additional keyword arguments for Model base class

    Raises:
        ValueError: If width_multiplier is not positive
        ValueError: If input_shape is invalid

    Example:
        >>> # Create standard MobileNetV1 for ImageNet
        >>> model = MobileNetV1(num_classes=1000, width_multiplier=1.0)
        >>>
        >>> # Create smaller model for CIFAR-10
        >>> model = MobileNetV1(num_classes=10, width_multiplier=0.5, input_shape=(32, 32, 3))
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "large": {"width_multiplier": 1.0},
        "medium": {"width_multiplier": 0.75},
        "small": {"width_multiplier": 0.5},
        "pico": {"width_multiplier": 0.25},
    }

    # Architecture definition (filters and strides for each block)
    # Format: (filters, stride)
    ARCHITECTURE = [
        # Initial standard conv is handled separately
        (64, 1),  # Block 1
        (128, 2),  # Block 2
        (128, 1),  # Block 3
        (256, 2),  # Block 4
        (256, 1),  # Block 5
        (512, 2),  # Block 6
        (512, 1),  # Block 7 (repeated 5 times)
        (512, 1),  # Block 8
        (512, 1),  # Block 9
        (512, 1),  # Block 10
        (512, 1),  # Block 11
        (1024, 2),  # Block 12
        (1024, 1),  # Block 13
    ]

    def __init__(
            self,
            num_classes: int = 1000,
            width_multiplier: float = 1.0,
            dropout_rate: float = 0.001,  # MobileNetV1 uses very light dropout
            weight_decay: float = 0.00004,  # Standard weight decay for MobileNet
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if width_multiplier <= 0:
            raise ValueError(f"width_multiplier must be positive, got {width_multiplier}")

        if input_shape is None:
            input_shape = (224, 224, 3)

        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        height, width, channels = input_shape
        if channels not in [1, 3]:
            logger.warning(f"Unusual number of channels: {channels}")

        # Store configuration
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self._input_shape = input_shape

        # Create regularizer
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        # Build the model layers
        self._build_layers()

    def _build_layers(self) -> None:
        """Build all layers of the model."""
        # Initial standard convolution (not depthwise separable)
        self.initial_conv = layers.Conv2D(
            filters=int(32 * self.width_multiplier),
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='conv1'
        )
        self.initial_bn = layers.BatchNormalization(name='conv1_bn')
        self.initial_relu = layers.ReLU(name='conv1_relu')

        # Build depthwise separable blocks
        self.depthwise_blocks = []
        for block_id, (filters, stride) in enumerate(self.ARCHITECTURE, start=1):
            # Apply width multiplier to filter count
            actual_filters = int(filters * self.width_multiplier)

            block = DepthwiseSeparableBlock(
                filters=actual_filters,
                stride=stride,
                block_id=block_id,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'block_{block_id}'
            )
            self.depthwise_blocks.append(block)

        # Classification head (the global pool is part of it -- D-066)
        if self.include_top:
            self.global_avg_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')

            # Shape layer to ensure correct dimensions
            self.reshape = layers.Reshape((1, 1, int(1024 * self.width_multiplier)), name='reshape')

            # Dropout for regularization
            self.dropout = layers.Dropout(self.dropout_rate, name='dropout')

            # Final convolution as FC layer (MobileNetV1 uses Conv instead of Dense)
            self.classifier_conv = layers.Conv2D(
                filters=self.num_classes,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='conv_preds'
            )

            # Reshape to get proper output shape
            self.output_reshape = layers.Reshape((self.num_classes,), name='output_reshape')

            # Softmax activation
            self.softmax = layers.Activation('softmax', name='act_softmax')

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the MobileNetV1 model."""
        # Initial convolution
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)

        # Depthwise separable blocks
        for block in self.depthwise_blocks:
            x = block(x, training=training)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-066: pooling belongs to the
        # HEAD. Do NOT move `global_avg_pool` back outside this branch: applied
        # unconditionally it made `include_top=False` return a 2-D pooled vector,
        # where V2/V3/V4 all return the 4-D feature map, so a detection or
        # segmentation head silently received the wrong rank. A caller who wants
        # the pooled vector adds one GlobalAveragePooling2D. See decisions.md D-066.
        if self.include_top:
            # Global average pooling
            x = self.global_avg_pool(x)

            # Classification head
            x = self.reshape(x)
            x = self.dropout(x, training=training)
            x = self.classifier_conv(x)
            x = self.output_reshape(x)
            x = self.softmax(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            width_multiplier: float = 1.0,
            **kwargs: Any
    ) -> "MobileNetV1":
        """Create a MobileNetV1 model from a predefined variant.

        Args:
            variant: String, one of "large", "medium", "small", "pico"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (224, 224, 3)
            width_multiplier: Float, additional multiplier applied on top of variant default
            **kwargs: Additional arguments passed to the constructor

        Returns:
            MobileNetV1 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> model = MobileNetV1.from_variant("large", num_classes=1000)
            >>> model = MobileNetV1.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]
        effective_width = config["width_multiplier"] * width_multiplier

        # Set default input shape if not provided
        if input_shape is None:
            input_shape = (224, 224, 3)

        logger.info(f"Creating MobileNetV1-{variant} model")
        logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

        return cls(
            num_classes=num_classes,
            width_multiplier=effective_width,
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileNetV1":
        """Create model from configuration."""
        # Deserialize the initializer if needed
        if isinstance(config.get("kernel_initializer"), dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: a real forward pass.
        # Do NOT go back to `self.build((None, *self._input_shape))`: for a
        # subclassed model that only MARKS the model built and materializes no
        # sub-layer weights, so the summary and the count_params() line below both
        # reported exactly 0. See decisions.md D-065.
        materialize_for_summary(self, self._input_shape)

        super().summary(**kwargs)

        # Print additional model information
        total_blocks = len(self.depthwise_blocks)
        total_params = self.count_params()

        logger.info("MobileNetV1 Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Width multiplier (α): {self.width_multiplier}")
        logger.info(f"  - Number of depthwise blocks: {total_blocks}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Weight decay: {self.weight_decay}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv1(
        variant: str = "large",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        width_multiplier: float = 1.0,
        pretrained: bool = False,
        **kwargs: Any
) -> MobileNetV1:
    """Convenience function to create MobileNetV1 models.

    Args:
        variant: String, model variant ("large", "medium", "small", "pico")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        width_multiplier: Float, additional multiplier applied on top of variant default
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV1 model instance

    Example:
        >>> model = create_mobilenetv1("large", num_classes=1000)
        >>> model = create_mobilenetv1("small", num_classes=10, input_shape=(32, 32, 3))
        >>> model = create_mobilenetv1("pico", num_classes=100)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented for MobileNetV1")

    model = MobileNetV1.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=width_multiplier,
        **kwargs
    )

    return model

# ------------------------------------------------------------------------