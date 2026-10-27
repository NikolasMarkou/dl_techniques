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

Every width follows one reference, Keras Applications `MobileNetV3`: the
stem is 16 channels at every `width_multiplier`; a block's expanded width
is `make_divisible(C_in * exp / in)`, its actual input width times the
table's expansion ratio; the first row (expansion 1) has no expansion
convolution; the last conv is `make_divisible(6 * C_last)`; the head widens
only above 1.0. `count_params()` equals `keras.applications.MobileNetV3Small`
/ `Large` (`weights=None`) at every width. The squeeze-and-excitation gate is
the reference's: biased 1x1 convs, a bottleneck of
`make_divisible(expanded / 4, 8)` and a hard-sigmoid gate (Keras Applications
`_se_block`, torchvision `SqueezeExcitation`).
The classifier is linear, so this model outputs logits: compile with
`from_logits=True`. No pretrained checkpoints ship with this package;
`pretrained=True` raises `NotImplementedError`, and a local `.keras` path
loads by layer.

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

import copy
import keras
from typing import Tuple, Optional, Dict, Any, Literal, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.tensors import make_divisible
from dl_techniques.utils.weight_transfer import (
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.layers.activations.hard_swish import HardSwish
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.vision.mobilenet.common import (
    materialize_for_summary,
    reference_bn_kwargs,
)
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.initializers.clone import clone_initializer

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
        Conv2D 3x3 s2 -> BN -> hard-swish   (16 channels at every alpha)
           |
           v
        UniversalInvertedBottleneck x N   (LARGE_CONFIG or SMALL_CONFIG row per block)
           |
           v
        Conv2D 1x1 -> BN -> hard-swish   (6x the last block: 960 Large / 576 Small at 1.0)
           |
        include_top=False --------+
           |                      |
           v                      v
        GlobalAvgPool         feature map (returned as-is)
           |
        Dense(1280 Large / 1024 Small) -> hard-swish -> Dropout -> Dense (linear)
           |
           v
        logits  [num_classes]

    :param num_classes: Number of output classes; used only if `include_top=True`,
        where it must be positive.
    :param variant: `"large"` or `"small"`.
    :param width_multiplier: Channel-count multiplier. Must be positive.
    :param dropout_rate: Dropout rate before the classifier, in `[0, 1)`.
    :param weight_decay: L2 regularization factor, the head's included.
    :param kernel_initializer: Weight initialization strategy, the head's
        included.
    :param include_top: Whether to include the classification head.
    :param input_shape: Input shape `(height, width, channels)`; `None` means
        `(224, 224, 3)`.
    :param head_config_overrides: Optional dict merged into the head
        configuration. `name`, `num_classes` and `task_type` raise
        `ValueError`.
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If `variant` is not `"large"` or `"small"`,
        `num_classes` is not positive with a head, `width_multiplier` is not
        positive, `dropout_rate` is outside `[0, 1)`, or `input_shape` is not
        3-D.

    Input shape:
        4-D tensor `(batch_size, height, width, channels)`.

    Output shape:
        - `include_top=True`: logits `(batch_size, num_classes)`.
        - `include_top=False`: the last conv's feature map
          `(batch_size, H', W', last_conv.filters)`.

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

    # Keras Applications' stem is Conv2D(16) at every alpha.
    STEM_FILTERS = 16

    def __init__(
            self,
            num_classes: int = 1000,
            variant: Literal["large", "small"] = "large",
            width_multiplier: float = 1.0,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-5,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, int, int]] = None,
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(self.MODEL_VARIANTS.keys())}"
            )

        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )

        if width_multiplier <= 0:
            raise ValueError(
                f"width_multiplier must be positive, got {width_multiplier}"
            )

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )

        if input_shape and len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.variant = variant
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self.input_shape_config = input_shape or (224, 224, 3)

        # Head configuration
        self._head_config_overrides = dict(head_config_overrides) if head_config_overrides else None

        if variant == "large":
            self.block_configs = self.LARGE_CONFIG
            last_point = 1280
        else:
            self.block_configs = self.SMALL_CONFIG
            last_point = 1024

        self.kernel_regularizer = keras.regularizers.L2(weight_decay) if weight_decay > 0 else None

        # The head's hidden width ("last point"): widened only above
        # width_multiplier 1.0, as Keras Applications MobileNetV3 does.
        self.last_point_filters = (
            make_divisible(last_point * width_multiplier)
            if width_multiplier > 1.0 else last_point
        )

        self._build_stem()
        self._build_blocks()
        self._build_last_block()

        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created MobileNetV3-{variant} with {len(self.blocks)} blocks "
            f"for input {self.input_shape_config}"
        )

    def _build_stem(self) -> None:
        """Build the stem: conv, batch norm and hard-swish in one :class:`ConvStem`."""
        self.stem = ConvStem(
            filters=self.STEM_FILTERS,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            num_convs=1,
            normalization_type='batch_norm',
            normalization_kwargs=reference_bn_kwargs(),
            activation_type='hard_swish',
            use_pooling=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name='stem',
        )

    def _build_blocks(self) -> None:
        """Build one :class:`UniversalInvertedBottleneck` per row of the variant's table.

        The blocks are named ``block_{i}``, counted from 0.
        """
        self.blocks = []
        in_channels = self.STEM_FILTERS
        # The table's input width at width_multiplier 1.0 (the stem's 16 for
        # the first row): exp_size / in_size is the row's expansion ratio.
        in_size = 16

        for i, (exp_size, out_size, kernel, stride, use_se, activation) in enumerate(self.block_configs):
            # DECISION plan-2026-10-14T062616-81938e81/D-232: Keras Applications'
            # _depth(infilters * expansion), the ACTUAL input width times the ratio.
            # Do NOT use make_divisible(exp_size * width_multiplier) (torchvision's
            # adjust_channels): it differs at every width but 1.0 (65,584 parameters
            # short for Large at 0.75), and the head already follows Keras. See decisions.md.
            exp_channels = make_divisible(in_channels * (exp_size / in_size))
            out_channels = make_divisible(out_size * self.width_multiplier)

            # expanded_channels is passed explicitly since expansion is not always an integer multiple of input channels.
            block = UniversalInvertedBottleneck(
                filters=out_channels,
                expanded_channels=exp_channels,
                # The first row (expansion 1) has no expansion conv, as Keras
                # Applications' `if block_id:` and torchvision's ratio check.
                use_expansion=i > 0,
                kernel_size=kernel,
                stride=stride,
                use_squeeze_excitation=use_se,
                activation_type=activation,
                normalization_type='batch_norm',
                # DECISION plan-2026-08-22T035419-a11304c8/D-203: pass epsilon explicitly; see mobilenet_v1.py for the measured impact. See decisions.md.
                normalization_args=reference_bn_kwargs(),
                use_bias=False,
                use_dw1=True,
                use_dw2=False,
                se_ratio=0.25,
                # The reference SE width: Keras Applications _depth(c * 0.25) and
                # torchvision _make_divisible(c // 4, 8), equal here (c % 8 == 0).
                se_divisor=8,
                se_activation='relu',
                # Keras Applications' SE convs keep Conv2D's default bias and
                # gate with hard_sigmoid; torchvision: nn.Conv2d (bias) and
                # nn.Hardsigmoid; the paper (Sec. 5.2): hard-sigmoid.
                se_use_bias=True,
                se_gate_activation='hard_sigmoid',
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name=f"block_{i}"
            )
            self.blocks.append(block)
            in_channels = out_channels
            in_size = out_size

    def _build_last_block(self) -> None:
        """Build the last conv, its batch norm and its hard-swish.

        The conv is ``make_divisible(6 * C_last)`` wide (Keras Applications:
        ``last_conv_ch = _depth(x.shape[-1] * 6)``).
        """
        last_block_filters = make_divisible(self.blocks[-1].filters * 6)
        self.last_conv = keras.layers.Conv2D(
            last_block_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="last_conv"
        )
        self.last_bn = keras.layers.BatchNormalization(
            **reference_bn_kwargs(),
            name="last_bn",
        )
        self.last_activation = HardSwish(name="last_hard_swish")

    def _build_head(self) -> None:
        """Build the reference classifier on the shared ``ClassificationHead``.

        The paper's classifier (Howard et al. 2019, Table 1/2: ``pool 7x7``,
        ``conv2d 1x1, NBN, 1280`` (Large) or ``1024`` (Small) with
        hard-swish, ``conv2d 1x1, NBN, k``), as Keras Applications
        ``MobileNetV3`` builds it: global average pool, a biased ``Dense`` of
        ``last_point_filters`` with no normalization, hard-swish,
        ``Dropout(dropout_rate)``, one linear ``Dense``. The model returns
        logits.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                'num_classes': self.num_classes,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
                # One hidden block: Dense(last point) -> hard-swish -> dropout,
                # no normalization ("NBN" in the paper's tables).
                'block_dims': (self.last_point_filters,),
                'normalization_type': None,
                'activation_type': 'hard_swish',
                'dropout_rate': self.dropout_rate,
                'kernel_initializer': clone_initializer(self.kernel_initializer),
                'kernel_regularizer': self.kernel_regularizer,
                # Deterministic layer name: an unnamed head takes Keras'
                # auto suffix in process-creation order, which breaks the
                # by-name matching that weight transfer relies on.
                'name': 'classification_head',
            },
            self._head_config_overrides,
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

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the stem, inverted residual blocks, last conv, and optional head.

        :param x: Input images, shape `(B, H, W, C)`.
        :param training: Passed to batch norm and dropout.
        :return: Logits, or the 4-D feature map if `include_top=False`.
        """
        x = self.stem(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.last_conv(x)
        x = self.last_bn(x, training=training)
        x = self.last_activation(x)

        if self.include_top:
            return self.classification_head(x, training=training)['logits']

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Report the shape `call` returns, without building the model.

        :param input_shape: `(batch_size, height, width, channels)`; the
            spatial axes may be `None`.
        :return: The logits shape `(batch_size, num_classes)`
            (`include_top=True`) or the last conv's feature map shape.
        """
        shape = self.stem.compute_output_shape(input_shape)

        for block in self.blocks:
            shape = block.compute_output_shape(shape)

        shape = self.last_conv.compute_output_shape(shape)
        shape = self.last_bn.compute_output_shape(shape)
        shape = self.last_activation.compute_output_shape(shape)

        if not self.include_top:
            return shape

        # The head reports both keys; the model returns the logits.
        return self.classification_head.compute_output_shape(shape)['logits']

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load a local `.keras` checkpoint into the model, layer by layer.

        The transfer is `load_pretrained_checkpoint`: it builds the model at
        `input_shape` if it is unbuilt, takes a headed checkpoint into a
        headless model, and refuses a checkpoint that restores no layer.

        :param weights_path: Path to the weights file (`.keras` format).
        :param skip_mismatch: Whether to skip layers with mismatched shapes;
            the inverse of the transfer helper's `strict`.
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If weights cannot be loaded or none are restored.
        """
        load_pretrained_checkpoint(
            self, weights_path, self.input_shape_config, skip_mismatch
        )

    @classmethod
    def from_variant(
            cls,
            variant: Literal["large", "small"],
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, int, int]] = None,
            pretrained: Union[bool, str] = False,
            **kwargs
    ) -> "MobileNetV3":
        """Create a MobileNetV3 model from a predefined variant.

        :param variant: `"large"` or `"small"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no MobileNetV3 weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; a field of the variant
            (e.g. `width_multiplier`) is replaced by the value given here.
        :return: A configured `MobileNetV3` instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

        Example::

            model = MobileNetV3.from_variant("large", num_classes=1000)
            small = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise on pretrained=True, do not warn and return random weights.
        # See decisions.md.
        weights_path = pretrained_path_or_raise(
            pretrained, f"{cls.__name__} variant '{variant}'"
        )

        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)

        model = cls(num_classes=num_classes, input_shape=input_shape, **config)

        if weights_path:
            model.load_pretrained_weights(weights_path)

        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "variant": self.variant,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
            "head_config_overrides": self._head_config_overrides,
        })
        return config

    def summary(self, **kwargs):
        """Print the model summary; an unbuilt model is materialized first."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: materialize with a real forward pass.
        # The keras.Input + build(shape) route marked the model built without creating any sub-layer weights. See decisions.md.
        materialize_for_summary(self, self.input_shape_config)

        super().summary(**kwargs)


# ---------------------------------------------------------------------

def create_mobilenetv3(
        variant: Literal["large", "small"] = "large",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, int, int]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs
) -> MobileNetV3:
    """Create a MobileNetV3 model.

    :param variant: Model variant: `"large"` or `"small"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param kwargs: Passthrough to `MobileNetV3.from_variant`, e.g.
        `width_multiplier`, `head_config_overrides`, `include_top`.
    :return: A configured `MobileNetV3` instance.
    :raises NotImplementedError: If `pretrained` is `True`.

    Example::

        model = create_mobilenetv3("large", num_classes=1000)
        small = create_mobilenetv3("small", num_classes=10, input_shape=(32, 32, 3))
    """
    return MobileNetV3.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )

# ---------------------------------------------------------------------
