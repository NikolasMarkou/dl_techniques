"""
MobileNetV2 image classifier: inverted residual bottlenecks with linear
projections, assembled here from `UniversalInvertedBottleneck` blocks.

What this generation adds over V1 is a shape and a missing activation. V1 was a
flat stack of separable blocks with no shortcuts and no bottleneck; V2 wraps the
same depthwise convolution in a residual block whose channel profile is inverted
relative to a ResNet bottleneck. A ResNet block is wide-narrow-wide: it squeezes
channels, does the expensive spatial work in the narrow interior, then restores
width, and the residual runs along the *wide* tensors. V2 is narrow-wide-narrow:
each block expands `C -> t*C` with a `1x1` convolution, does its `3x3` depthwise
filtering in that expanded space where a per-channel filter has enough channels to
be expressive, projects back to a narrow `C_out`, and runs the residual along the
*narrow* tensors. The inversion is what makes it memory-efficient: only the thin
block boundaries have to be materialized between blocks, so the expanded tensor
never has to persist, and the same argument makes the block a natural fit for
memory-limited inference.

The linear bottleneck is the second half of the idea and the part that is easy to
misread as an oversight. The projection `1x1` at the end of every block has **no
activation** — no ReLU, no ReLU6 — only normalization. The paper's reasoning is
that the information a layer carries is assumed to lie on a low-dimensional
manifold embedded in the activation space; ReLU is only information-preserving on
such a manifold when the space it lives in is high-dimensional enough that the
zeroed half-space can be recovered from the surviving channels. Applying ReLU
directly to the narrow output of the projection collapses exactly the
low-dimensional representation the block just produced, and the collapse is not
recoverable downstream. So the nonlinearity is kept where the tensor is wide (after
expansion, after depthwise) and dropped where it is narrow. Empirically the paper
measures the linear projection as worth several points of top-1.

Architecturally: a `3x3` stride-2 stem into 32 channels, the paper's Table 2
`(t, c, n, s)` schedule of seven stages totalling 17 bottleneck blocks, a final
`1x1` expansion to 1280 channels, global pooling, dropout and a dense classifier.
Stride is applied only to the first block of each stage. The residual is added
inside `UniversalInvertedBottleneck` and only when `stride == 1` and the input
channel count already equals `filters`, so the first block of every stage — which
either strides or changes width — is a plain feed-forward block with no shortcut,
exactly as in the paper.

Three code-level details worth stating. Channel counts pass through
`_make_divisible`, which rounds to a multiple of 8 and refuses to round down by
more than 10%; this is why `width_multiplier=0.75` does not simply give
`0.75 * c`. The final 1280-channel convolution is deliberately *not* scaled for
`width_multiplier <= 1.0` and only widens above 1.0, matching the reference
implementation — thinning the classifier's input hurts far more than it saves.
And `MODEL_VARIANTS` includes a `1.4` width ("large") alongside the usual
1.0/0.75/0.5/0.35 ladder; the variants are named by size, not by their `alpha`
value, so `from_variant` takes `"medium"`, not `"1.0"`.

One deviation from the paper follows from reusing the universal block. The paper's
first stage has expansion factor `t=1` and therefore omits the expansion
convolution entirely; `UniversalInvertedBottleneck` always builds an expansion
`1x1` + norm + activation, so that stage here carries an extra `C -> C` projection
the reference does not have. It is a small parameter cost and a structural
difference, not a numerical equivalence. The classifier also ends in softmax, so
this model emits probabilities: compile with `from_logits=False`.

`pretrained=True` on `create_mobilenetv2` raises `NotImplementedError` — no
checkpoints ship with this package. It used to return an untrained model plus a
log line, so a caller who asked for pretrained weights got random ones; the house
contract in `resnet/model.py` now holds here too. Warm-start from a local file
with `model.load_weights(path)`.

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
from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.mobilenet.common import materialize_for_summary
from dl_techniques.utils.model_build import materialize_sublayers

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MobileNetV2(keras.Model):
    """MobileNetV2 classification model built with Universal Inverted Bottleneck blocks.

    This class implements the full MobileNetV2 architecture, an efficient
    convolutional neural network designed for mobile and embedded vision_heads
    applications. It utilizes `UniversalInvertedBottleneck` (UIB) layers configured
    to replicate the original's inverted residuals and linear bottlenecks.

    **Intent**: To provide a production-ready, configurable, and easily
    serializable implementation of the MobileNetV2 model. This serves as a
    best-practice example for building complex custom models in Keras 3,
    leveraging a flexible and unified building block (UIB).

    **Architecture**:
    ```
    Input(shape=[H, W, 3])
           ↓
    Initial Conv: 3x3 Conv2D(32, stride=2) -> BN -> ReLU6
           ↓
    Bottleneck Blocks: Sequence of 17 UniversalInvertedBottleneck layers
           ↓
    Final Conv: 1x1 Conv2D(1280) -> BN -> ReLU6
           ↓
    Pooling: GlobalAveragePooling2D
           ↓
    Classifier: Dropout -> Dense(num_classes, 'softmax')
           ↓
    Output(shape=[num_classes])
    ```

    **Data Flow**:
    1. An initial convolution layer performs downsampling and feature extraction.
    2. A series of 7 bottleneck stages, built from `UniversalInvertedBottleneck`
       layers, progressively extracts features and reduces spatial dimensions.
    3. A final 1x1 convolution expands the feature map to a high-dimensional space.
    4. Global average pooling converts the feature map into a single feature vector.
    5. A fully-connected classifier with softmax activation produces class probabilities.

    Args:
        num_classes: Integer, number of output classes for classification.
        width_multiplier: Float (α), scales the number of channels in each layer,
            allowing for control over model size and complexity.
        dropout_rate: Float, dropout rate applied before the final classifier.
        weight_decay: Float, L2 regularization factor for convolutional and dense layers.
        kernel_initializer: String or Initializer for kernel weight initialization.
        include_top: Boolean, whether to include the final pooling and classification layers.
        input_shape: Optional Tuple, the shape of the input tensor. Defaults to (224, 224, 3).
        **kwargs: Additional keyword arguments for the `keras.Model` base class.

    Input shape:
        3D tensor with shape `(height, width, channels)`, e.g., `(224, 224, 3)`.

    Output shape:
        2D tensor with shape `(batch_size, num_classes)` if `include_top=True`.
        4D feature map otherwise.

    Attributes:
        initial_conv, initial_bn, initial_relu: Layers for the first block.
        blocks: A list of `UniversalInvertedBottleneck` instances.
        last_conv, last_bn, last_relu: Layers for the final feature extraction.
        global_avg_pool: Global average pooling layer.
        dropout: Dropout layer (if used).
        classifier: Final Dense classification layer.

    Example:
        ```python
        # Standard MobileNetV2 (α=1.0) for ImageNet
        model = MobileNetV2(num_classes=1000, width_multiplier=1.0)

        # Smaller model (α=0.75) for CIFAR-10
        model = MobileNetV2(
            num_classes=10,
            width_multiplier=0.75,
            input_shape=(32, 32, 3)
        )
        ```

    References:
        - MobileNetV2 Paper: https://arxiv.org/abs/1801.04381
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

        # --- Configuration Validation and Storage ---
        if width_multiplier <= 0:
            raise ValueError(f"width_multiplier must be positive, got {width_multiplier}")
        self.input_shape_config = input_shape or (224, 224, 3)
        if len(self.input_shape_config) != 3:
            raise ValueError(f"input_shape must be a 3D tuple, got {self.input_shape_config}")

        # Store all configuration parameters for serialization
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        # --- CREATE all sub-layers in __init__ ---
        # For a keras.Model, all sub-layers should be created here. Keras will
        # handle calling their `build` methods automatically.
        self._build_model_layers()

    def _make_divisible(self, v: float, divisor: int = 8) -> int:
        """Ensures that layer channel counts are divisible by 8."""
        new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _build_model_layers(self) -> None:
        """Create all layers of the model based on the configuration."""
        # Initial Convolution
        first_channels = self._make_divisible(32 * self.width_multiplier)
        self.initial_conv = layers.Conv2D(
            first_channels, 3, strides=2, padding='same', use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name='conv1'
        )
        self.initial_bn = layers.BatchNormalization(name='conv1_bn')
        self.initial_relu = layers.ReLU(max_value=6, name='conv1_relu6')

        # Bottleneck Blocks (using UniversalInvertedBottleneck)
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
                    kernel_size=3,              # Standard for MobileNetV2
                    use_dw1=True,               # Emulates MobileNetV2 block
                    use_dw2=False,              # Emulates MobileNetV2 block
                    activation_type='relu',     # Use ReLU...
                    activation_args={'max_value': 6}, # ...with max_value=6 (ReLU6)
                    normalization_type='batch_norm',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'block_{block_id}'
                ))
                block_id += 1

        # Final Convolution
        # Per the original paper and official implementations, the last convolution
        # layer's channels are not scaled down for width multipliers <= 1.0.
        if self.width_multiplier > 1.0:
            last_channels = self._make_divisible(1280 * self.width_multiplier)
        else:
            last_channels = 1280
        self.last_conv = layers.Conv2D(
            last_channels, 1, padding='same', use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name='conv_last'
        )
        self.last_bn = layers.BatchNormalization(name='conv_last_bn')
        self.last_relu = layers.ReLU(max_value=6, name='conv_last_relu6')

        # Top (Classification Head)
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
        """Materialize every sub-layer from ``input_shape``.

        Without this method MobileNetV2 inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        Args:
            input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the MobileNetV2 model."""
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

        Args:
            variant: String, one of "large", "medium", "small", "nano", "pico"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (224, 224, 3)
            width_multiplier: Float, additional multiplier applied on top of variant default
            **kwargs: Additional arguments passed to the constructor

        Returns:
            MobileNetV2 model instance
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
        """Print model summary with additional information."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: a real forward pass; the
        # keras.Input + build(shape) route marked the model built without creating
        # any sub-layer weights. See decisions.md D-065.
        materialize_for_summary(self, self.input_shape_config)

        super().summary(**kwargs)

        # Print additional model information
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
    """Convenience function to create MobileNetV2 models.

    Args:
        variant: String, model variant ("large", "medium", "small", "nano", "pico")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        width_multiplier: Float, additional multiplier applied on top of variant default
        pretrained: Boolean, must be False. `True` raises `NotImplementedError` —
            no MobileNetV2 checkpoints ship with this package.
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV2 model instance

    Example:
        >>> model = create_mobilenetv2("medium", num_classes=1000)
        >>> model = create_mobilenetv2("nano", num_classes=10, input_shape=(32, 32, 3))
        >>> model = create_mobilenetv2("pico", num_classes=100)
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
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
