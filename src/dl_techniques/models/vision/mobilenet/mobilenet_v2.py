"""``MobileNetV2``, an inverted-residual classifier built from `UniversalInvertedBottleneck` blocks, plus the ``create_mobilenetv2`` factory.

Unlike a ResNet bottleneck (wide-narrow-wide, residual on the wide
tensors), V2 is narrow-wide-narrow: each block expands `C -> t*C`, filters
depthwise in that expanded space, projects back to a narrow `C_out`, and
runs the residual on the narrow tensors. Only the thin block boundaries
need to persist between blocks. The final projection of every block has
no activation, since a ReLU there would collapse the low-dimensional
representation the projection produces.

Channel counts round through `utils.tensors.make_divisible` (nearest multiple of 8, no
more than 10% below `width_multiplier * c`), so `width_multiplier=0.75`
does not give exactly `0.75 * c`. The final 1280-channel convolution is
not scaled down for `width_multiplier <= 1.0`, only widened above it. The
first stage (expansion factor 1) has no expansion convolution, as the paper,
Keras Applications and torchvision build it, so `count_params()` equals
`keras.applications.MobileNetV2` at each variant's `alpha`. The
classifier is linear, so this model outputs logits: compile with
`from_logits=True`. No pretrained checkpoints ship with this package;
`pretrained=True` raises `NotImplementedError`, and a local `.keras` path
loads by layer.

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

import copy
import keras
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import (
    coerce_include_top,
    materialize_sublayers,
    plain_head_config_overrides,
)
from dl_techniques.utils.tensors import make_divisible
from dl_techniques.utils.weight_transfer import (
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.vision.mobilenet.common import (
    materialize_for_summary,
    reference_bn_kwargs,
)
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.initializers.clone import clone_initializer, initializer_spec

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (the initializer, the
# dropout rate, the regularizer, the input size). `load_pretrained_checkpoint` compares
# every other key the checkpoint and the model both hold, so a changed width multiplier
# is refused even when every weight shape still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "input_shape",
    "kernel_initializer",
    "dropout_rate",
    "kernel_regularizer",
)

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
        Dropout -> Dense (linear)   (ClassificationHead)
           |
           v
        logits  [num_classes]

    :param num_classes: Number of output classes; used only if `include_top=True`,
        where it must be positive.
    :param width_multiplier: Channel-count multiplier (alpha). Must be positive.
    :param dropout_rate: Dropout rate before the classifier, in `[0, 1)`.
    :param kernel_regularizer: Optional regularizer applied to the standard
        convolution kernels and the classifier kernel (the depthwise kernels
        carry none): a regularizer, its serialized config or a name. `None`
        (the default) adds no loss, so the optimizer's decoupled weight decay is
        the only decay.
    :param kernel_initializer: Weight initialization strategy, the
        classifier's included: a name, a serialized config dict or an
        `Initializer` instance. Each site draws from its own clone of it, so a
        name, a dict or a seedless instance gives every kernel its own draw; an
        instance or dict carrying an explicit `seed` reuses that seed at every
        site by contract, so same-shaped kernels start equal. `get_config`
        reports its serialized spec.
    :param include_top: Whether to include the pooling and classification head.
        A `bool` or `numpy.bool_`; anything else, a string included, raises `TypeError`.
    :param input_shape: Input shape `(height, width, channels)`; `None` means
        `(224, 224, 3)`.
    :param head_config_overrides: Optional dict merged into the head
        configuration. `name`, `num_classes` and `task_type` raise
        `ValueError`.
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If `num_classes` is not positive with a head,
        `width_multiplier` is not positive, `dropout_rate` is outside `[0, 1)`,
        or `input_shape` is not 3-D.
    :raises TypeError: If `include_top` is not a `bool` or `numpy.bool_`, or if
        `kernel_initializer` is not a string, a dict or an `Initializer` instance.
    :raises NotImplementedError: If `kernel_initializer` is an `Initializer`
        instance that cannot serialize itself.

    Input shape:
        4-D tensor `(batch_size, height, width, channels)`.

    Output shape:
        - `include_top=True`: logits `(batch_size, num_classes)`.
        - `include_top=False`: the last conv's feature map
          `(batch_size, H/32, W/32, last_conv.filters)`.

    Example::

        model = MobileNetV2.from_variant("medium", num_classes=1000)
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
            kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
            kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = "he_normal",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        include_top = coerce_include_top(include_top)
        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )

        if width_multiplier <= 0:
            raise ValueError(f"width_multiplier must be positive, got {width_multiplier}")

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        # DECISION plan-2026-10-26T125959-7af851e5/D-039: keep the caller's Initializer and hand a clone of it to each site; compute the spec only to refuse a bad initializer here and to report it in get_config. Do NOT store the spec and hand it to the sites, as MobileNetV4 does.
        # A stored spec re-draws every kernel from a fresh default seed, so a seeded V2 kernel would no longer equal the base's (0 of 53 kernels at `small` stay equal); the clone keeps the seeded draw unchanged. See decisions.md.
        self._kernel_initializer_spec = initializer_spec(kernel_initializer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self.input_shape_config = input_shape
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self._head_config_overrides = plain_head_config_overrides(
            head_config_overrides, "MobileNetV2"
        )

        self._build_stem()

        self.blocks = []
        self._build_blocks()
        self._build_last_block()

        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created MobileNetV2 (alpha={width_multiplier}) with {len(self.blocks)} blocks "
            f"for input {self.input_shape_config}"
        )

    def _build_stem(self) -> None:
        """Build the stem: conv, batch norm and ReLU6 in one :class:`ConvStem`."""
        self.stem = ConvStem(
            filters=make_divisible(32 * self.width_multiplier),
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            num_convs=1,
            normalization_type='batch_norm',
            normalization_kwargs=reference_bn_kwargs(),
            activation_type='relu',
            activation_kwargs={'max_value': 6},
            use_pooling=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name='stem',
        )

    def _build_blocks(self) -> None:
        """Build one :class:`UniversalInvertedBottleneck` per row of ``ARCHITECTURE``, repeated ``n`` times.

        The blocks are named ``block_{i}``, counted from 0.
        """
        block_id = 0
        for t, c, n, s in self.ARCHITECTURE:
            output_channels = make_divisible(c * self.width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(UniversalInvertedBottleneck(
                    filters=output_channels,
                    expansion_factor=t,
                    # t=1 has no expansion conv (Keras Applications
                    # `if block_id:`, torchvision `if expand_ratio != 1`).
                    use_expansion=t != 1,
                    stride=stride,
                    kernel_size=3,
                    use_dw1=True,
                    use_dw2=False,
                    activation_type='relu',
                    activation_args={'max_value': 6},
                    normalization_type='batch_norm',
                    # DECISION plan-2026-08-22T035419-a11304c8/D-203: pass epsilon explicitly; see mobilenet_v1.py for the measured impact. See decisions.md.
                    normalization_args=reference_bn_kwargs(),
                    use_bias=False,
                    kernel_initializer=clone_initializer(self.kernel_initializer),
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'block_{block_id}'
                ))
                block_id += 1

    def _build_last_block(self) -> None:
        """Build the last 1x1 conv, its batch norm and its ReLU6.

        The conv is not scaled down for `width_multiplier <= 1.0`, as in the
        paper, and widened above it.
        """
        if self.width_multiplier > 1.0:
            last_channels = make_divisible(1280 * self.width_multiplier)
        else:
            last_channels = 1280
        self.last_conv = keras.layers.Conv2D(
            last_channels, 1, padding='same', use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer, name='conv_last'
        )
        self.last_bn = keras.layers.BatchNormalization(
            **reference_bn_kwargs(),
            name='conv_last_bn',
        )
        self.last_relu = keras.layers.ReLU(max_value=6, name='conv_last_relu6')

    def _build_head(self) -> None:
        """Build the reference classifier on the shared ``ClassificationHead``.

        The paper's classifier (Sandler et al. 2018, Table 2: ``avgpool 7x7``,
        then ``conv2d 1x1 k``) as torchvision ``mobilenet_v2`` builds it:
        global average pool, ``Dropout(dropout_rate)`` (0.2 there and here),
        one linear ``Dense``. The model returns logits.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                'num_classes': self.num_classes,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
                # No hidden block: the pooled vector, a dropout, then the
                # classifier.
                'block_dims': (),
                'classifier_dropout_rate': self.dropout_rate,
                # No hidden block, so no block dropout either.
                'dropout_rate': 0.0,
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

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the stem, bottleneck blocks, final conv, and optional classification head.

        :param inputs: Input images, shape `(B, H, W, C)`.
        :param training: Passed to batch norm and dropout.
        :return: Logits, or the 4-D feature map if `include_top=False`.
        """
        x = self.stem(inputs, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.last_conv(x)
        x = self.last_bn(x, training=training)
        x = self.last_relu(x)

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
        shape = self.last_relu.compute_output_shape(shape)

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
        `input_shape` if it is unbuilt and takes a headed checkpoint into a
        headless model. Every layer outside the classification head must be
        restored, so a checkpoint of another width raises and copies no weight.
        Only a head of another `num_classes` is skipped: it keeps its
        initialization and a warning names it. The archive's own config is
        compared too: a constructor argument that both hold and that is not in
        `_NON_STRUCTURAL_CONFIG_KEYS` must be equal, so a checkpoint of another
        `width_multiplier` is refused even when every weight shape fits. An
        argument only one side holds is not compared, and an archive in which
        no single model of this class can be found is checked by layers only
        (the load warns). Where the head is copied, its `head_config_overrides`
        must be equal too (dropout, initializers and regularizers aside).

        :param weights_path: Path to the weights file (`.keras` format).
        :param skip_mismatch: `True` (default) lets the head layers differ;
            `False` allows no difference anywhere. The inverse of the transfer
            helper's `strict`.
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If a layer outside the head is not restored, or
            the weights cannot be loaded.
        """
        load_pretrained_checkpoint(
            self, weights_path, self.input_shape_config, skip_mismatch,
            ignore_config_keys=_NON_STRUCTURAL_CONFIG_KEYS,
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: Union[bool, str] = False,
            **kwargs: Any
    ) -> "MobileNetV2":
        """Create a MobileNetV2 model from a predefined variant.

        :param variant: One of `"large"`, `"medium"`, `"small"`, `"nano"`, `"pico"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no MobileNetV2 weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; a field of the variant
            (`width_multiplier`) is replaced by the value given here.
        :return: A configured `MobileNetV2` instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

        Example::

            model = MobileNetV2.from_variant("medium", num_classes=1000)
            small = MobileNetV2.from_variant("nano", num_classes=10, input_shape=(32, 32, 3))
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
        """Return configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_initializer": self._kernel_initializer_spec,
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self, **kwargs):
        """Print the model summary; an unbuilt model is materialized first."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: materialize with a real forward pass.
        # The keras.Input + build(shape) route marked the model built without creating any sub-layer weights. See decisions.md.
        materialize_for_summary(self, self.input_shape_config)

        super().summary(**kwargs)


# ---------------------------------------------------------------------

def create_mobilenetv2(
        variant: str = "medium",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs: Any
) -> MobileNetV2:
    """Create a MobileNetV2 model.

    :param variant: Model variant: `"large"`, `"medium"`, `"small"`, `"nano"`, `"pico"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param kwargs: Passthrough to `MobileNetV2.from_variant`, e.g.
        `width_multiplier`, `head_config_overrides`, `include_top`.
    :return: A configured `MobileNetV2` instance.
    :raises NotImplementedError: If `pretrained` is `True`.

    Example::

        model = create_mobilenetv2("medium", num_classes=1000)
        small = create_mobilenetv2("nano", num_classes=10, input_shape=(32, 32, 3))
    """
    return MobileNetV2.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )

# ---------------------------------------------------------------------
