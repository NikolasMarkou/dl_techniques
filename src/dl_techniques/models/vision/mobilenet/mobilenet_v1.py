"""``MobileNetV1``, the original depthwise-separable-convolution backbone, plus the ``create_mobilenetv1`` factory.

A standard `KxK` convolution filters space and mixes channels in one step,
costing `K*K*C_in*C_out*H*W` multiply-adds. This splits that into a
depthwise `KxK` convolution (filters each channel) followed by a `1x1`
pointwise convolution (mixes channels), roughly 8-9x cheaper for `K=3`.
The network is 28 layers of that one block, with no residuals, bottlenecks,
or attention. A width multiplier `alpha` thins every layer; a resolution
multiplier is not a separate argument, just a smaller `input_shape`.

Blocks use plain ReLU (the paper's released model uses ReLU6, which
matters for quantization). The classifier is linear, so this model
outputs logits: compile with `from_logits=True`. With `include_top=False` it returns the
4-D feature map of the last block, matching V2/V3/V4. Widths use plain
`int()` truncation, not V2/V3's round-to-multiple-of-8 rule, so channel
counts follow the paper's alpha table but will not match TF-Slim
checkpoints at fractional alpha. No pretrained checkpoints ship with this
package; `pretrained=True` raises `NotImplementedError`, and a local `.keras`
path loads by layer.

References:
    - Howard et al., 2017. MobileNets: Efficient Convolutional Neural Networks for
      Mobile Vision Applications. (https://arxiv.org/abs/1704.04861)
    - Sifre and Mallat, 2014. Rigid-Motion Scattering for Image Classification.
      (the origin of the depthwise separable factorization)
    - Chollet, 2017. Xception: Deep Learning with Depthwise Separable Convolutions.
      (https://arxiv.org/abs/1610.02357)
"""

import copy
import keras
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import (
    materialize_sublayers,
    plain_head_config_overrides,
)
from dl_techniques.utils.weight_transfer import (
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.layers.conv_blocks.depthwise_separable_block import DepthwiseSeparableBlock
from dl_techniques.layers.conv_blocks.stem import ConvStem
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
# dropout rate, the weight decay, the input size). `load_pretrained_checkpoint` compares
# every other key the checkpoint and the model both hold, so a changed width multiplier
# is refused even when every weight shape still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "input_shape",
    "kernel_initializer",
    "dropout_rate",
    "weight_decay",
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mobilenet.mobilenet_v1")
class MobileNetV1(keras.Model):
    """Depthwise-separable-convolution backbone for image classification.

    Architecture:

    .. code-block:: text

        image  [B, H, W, C]
           |
           v
        Conv2D 3x3 s2 -> BN -> ReLU   (32*alpha channels)
           |
           v
        13x DepthwiseSeparableBlock   (ARCHITECTURE table, alpha-scaled)
           |  [B, H/32, W/32, 1024*alpha]
           v
        include_top=False --------+
           |                      |
           v                      v
        GlobalAvgPool         feature map (returned as-is)
           |
        Dropout -> Dense (linear)       (ClassificationHead)
           |
           v
        logits  [B, num_classes]

    :param num_classes: Number of output classes; used only if `include_top=True`,
        where it must be positive.
    :param width_multiplier: Filter-count multiplier (alpha). Must be positive.
        Common values: 1.0, 0.75, 0.5, 0.25.
    :param dropout_rate: Dropout rate before the classifier, in `[0, 1)`.
    :param weight_decay: L2 regularization factor for all layers, the
        classifier included.
    :param kernel_initializer: Weight initialization strategy, the
        classifier's included: a name, a serialized config dict or an
        `Initializer` instance. Each site draws from its own clone of it, so a
        name, a dict or a seedless instance gives every kernel its own draw; an
        instance or dict carrying an explicit `seed` reuses that seed at every
        site by contract, so same-shaped kernels start equal. `get_config`
        reports its serialized spec.
    :param include_top: Whether to include the classification head.
    :param input_shape: Input shape `(height, width, channels)`; `None` means
        `(224, 224, 3)`.
    :param head_config_overrides: Optional dict merged into the head
        configuration. `name`, `num_classes` and `task_type` raise
        `ValueError`.
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If `num_classes` is not positive with a head,
        `width_multiplier` is not positive, `dropout_rate` is outside `[0, 1)`,
        or `input_shape` is not 3-D.
    :raises TypeError: If `kernel_initializer` is not a string, a dict or an
        `Initializer` instance.
    :raises NotImplementedError: If `kernel_initializer` is an `Initializer`
        instance that cannot serialize itself.

    Input shape:
        4-D tensor `(batch_size, height, width, channels)`.

    Output shape:
        - `include_top=True`: logits `(batch_size, num_classes)`.
        - `include_top=False`: the last block's feature map
          `(batch_size, H/32, W/32, int(1024 * width_multiplier))`.

    Example::

        model = MobileNetV1(num_classes=1000, width_multiplier=1.0)
        small = MobileNetV1(num_classes=10, width_multiplier=0.5, input_shape=(32, 32, 3))
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
            kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = "he_normal",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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

        channels = input_shape[2]
        if channels not in [1, 3]:
            logger.warning(f"Unusual number of channels: {channels}")

        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        # DECISION plan-2026-10-26T125959-7af851e5/D-039: keep the caller's Initializer and hand a clone of it to each site; compute the spec only to refuse a bad initializer here and to report it in get_config. Do NOT store the spec and hand it to the sites, as MobileNetV4 does.
        # A stored spec re-draws every kernel from a fresh default seed, so a seeded V1 kernel would no longer equal the base's (0 of 28 kernels at `small` stay equal); the clone keeps the seeded draw unchanged. See decisions.md.
        self._kernel_initializer_spec = initializer_spec(kernel_initializer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self.input_shape_config = input_shape
        self.kernel_regularizer = keras.regularizers.L2(weight_decay) if weight_decay > 0 else None

        self._head_config_overrides = plain_head_config_overrides(
            head_config_overrides, "MobileNetV1"
        )

        self._build_stem()

        self.blocks = []
        self._build_blocks()

        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created MobileNetV1 (alpha={width_multiplier}) with {len(self.blocks)} blocks "
            f"for input {self.input_shape_config}"
        )

    def _build_stem(self) -> None:
        """Build the stem: a standard convolution, batch norm and ReLU in one :class:`ConvStem`.

        The width is ``int(32 * width_multiplier)``: V1 truncates, V2 to V4
        round to a multiple of 8.
        """
        self.stem = ConvStem(
            filters=int(32 * self.width_multiplier),
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            num_convs=1,
            normalization_type='batch_norm',
            normalization_kwargs=reference_bn_kwargs(),
            activation_type='relu',
            use_pooling=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name='stem',
        )

    def _build_blocks(self) -> None:
        """Build one :class:`DepthwiseSeparableBlock` per row of ``ARCHITECTURE``.

        The blocks are named ``block_{i}``, counted from 0; ``i`` is also the
        ``block_id`` that names the block's own layers (``conv_dw_{i}``).
        Filter counts are ``int(filters * width_multiplier)``.
        """
        for i, (filters, stride) in enumerate(self.ARCHITECTURE):
            actual_filters = int(filters * self.width_multiplier)

            block = DepthwiseSeparableBlock(
                filters=actual_filters,
                stride=stride,
                block_id=i,
                # DECISION plan-2026-08-22T035419-a11304c8/D-203: pass epsilon explicitly; create_normalization_layer defaults to 1e-6, not the reference's 1e-3.
                # Dropping this kwarg silently ran 183 of 189 BatchNorm layers in this family at the wrong epsilon. See decisions.md.
                normalization_kwargs=reference_bn_kwargs(),
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name=f'block_{i}'
            )
            self.blocks.append(block)

    def _build_head(self) -> None:
        """Build the reference classifier on the shared ``ClassificationHead``.

        The paper's classifier (Howard et al. 2017, Table 1: ``Avg Pool 7x7``,
        ``FC 1024 x 1000``, ``Softmax``) as Keras Applications ``MobileNet``
        builds it: global average pool, ``Dropout(dropout_rate)`` (``1e-3``
        there and here), one linear ``Dense``. The softmax belongs in the
        loss, so the model returns logits.
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
        """Run the stem, the depthwise separable blocks and the optional head.

        :param inputs: Input images, shape `(B, H, W, C)`.
        :param training: Passed to batch norm and dropout.
        :return: Logits, or the 4-D feature map if `include_top=False`.
        """
        x = self.stem(inputs, training=training)

        for block in self.blocks:
            x = block(x, training=training)

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
            (`include_top=True`) or the last block's feature map shape.
        """
        shape = self.stem.compute_output_shape(input_shape)

        for block in self.blocks:
            shape = block.compute_output_shape(shape)

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
        restored, so a checkpoint of another width raises and leaves the model
        untouched. Only a head of another `num_classes` is skipped: it keeps
        its initialization and a warning names it. The archive's own config is
        compared too: a constructor argument that both hold and that is not in
        `_NON_STRUCTURAL_CONFIG_KEYS` must be equal, so a checkpoint of another
        `width_multiplier` is refused even when every weight shape fits. An
        argument only one side holds is not compared, and an archive without a
        config of this class is checked by layers only.

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
    ) -> "MobileNetV1":
        """Create a MobileNetV1 model from a predefined variant.

        :param variant: One of `"large"`, `"medium"`, `"small"`, `"pico"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no MobileNetV1 weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; a field of the variant
            (`width_multiplier`) is replaced by the value given here.
        :return: A configured `MobileNetV1` instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

        Example::

            model = MobileNetV1.from_variant("large", num_classes=1000)
            small = MobileNetV1.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
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
        config = {
            "num_classes": self.num_classes,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": self._kernel_initializer_spec,
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self, **kwargs):
        """Print the model summary; an unbuilt model is materialized first."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: materialize with a real forward pass, not self.build(...).
        # build() only marks a subclassed model built without materializing sub-layer weights, so count_params() would read 0. See decisions.md.
        materialize_for_summary(self, self.input_shape_config)

        super().summary(**kwargs)


# ---------------------------------------------------------------------

def create_mobilenetv1(
        variant: str = "large",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs: Any
) -> MobileNetV1:
    """Create a MobileNetV1 model.

    :param variant: Model variant: `"large"`, `"medium"`, `"small"`, `"pico"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param kwargs: Passthrough to `MobileNetV1.from_variant`, e.g.
        `width_multiplier`, `head_config_overrides`, `include_top`.
    :return: A configured `MobileNetV1` instance.
    :raises NotImplementedError: If `pretrained` is `True`.

    Example::

        model = create_mobilenetv1("large", num_classes=1000)
        small = create_mobilenetv1("small", num_classes=10, input_shape=(32, 32, 3))
    """
    return MobileNetV1.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )

# ------------------------------------------------------------------------
