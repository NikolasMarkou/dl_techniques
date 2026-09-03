"""ASCII ConvNeXt BERT: the study's convolutional arm.

Same skeleton as :mod:`...ascii_bert` -- same ASCII embeddings, depth/width
ladder, pooling, output contract -- with self-attention replaced by a
ConvNeXt V1 block applied along the sequence axis: depthwise convolution,
normalization, pointwise expansion by 4, activation, pointwise contraction,
LayerScale.

This is the third arm of a three-way comparison. The Clifford arm differs
from the transformer baseline in two ways at once: convolutional instead of
attentional, and mixing channels through a geometric product. This arm is
convolutional without the geometric product, so the three together separate
"convolution instead of attention" from "the geometric product on top of
convolution".

Cost is linear in sequence length: no ``S x S`` matrix is built. Token
mixing is local, and the span is a design parameter. A ConvNeXt block
applies one depthwise convolution where ``CliffordNetBlock`` applies two, so
at equal depth and kernel this arm's receptive field is half the Clifford
arm's: ``num_layers * (K - 1) + 1`` against ``num_layers * 2 * (K - 1) + 1``.
Use :func:`~...shared.blocks.conv_receptive_field` for this arm. Padding is
not neutral here either, for the same reason as the Clifford arm, and
mitigated the same way, by zeroing masked positions before the block.
LayerScale starts at 1.0 here, not 1e-5, so the block contributes at full
magnitude from the first step.

References:
    - Liu et al., 2022. A ConvNet for the 2020s (ConvNeXt).
      (https://arxiv.org/abs/2201.03545)
    - Devlin et al., 2019. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Clark et al., 2022. CANINE: Pre-training an Efficient
      Tokenization-Free Encoder for Language Representation.
      (https://arxiv.org/abs/2103.06874)
    - Touvron et al., 2021. Going Deeper with Image Transformers (CaiT).
      (LayerScale) (https://arxiv.org/abs/2103.17239)
"""

from typing import Any, Dict, Optional

import keras

from dl_techniques.utils.logger import logger

from ..shared.blocks import conv_receptive_field
from ..shared.encoder import EmbeddingEncoder
from dl_techniques.utils.keras_registration import register_dl_technique

__all__ = ["AsciiConvNextBert", "create_ascii_convnext_bert"]


@register_dl_technique("dl_techniques.models.ascii_convnext_bert.model")
class AsciiConvNextBert(EmbeddingEncoder):
    """
    ConvNeXt-block embedding encoder over the ASCII character vocabulary.

    Architecture:

    .. code-block:: text

        ascii ids [B, S] ──► ASCII embedding table [101, H]
                                        │
                                        ▼
                    ┌───────── convnext block ─────────┐  x num_layers
                    │ depthwise conv (sequence axis)    │
                    │ + norm + pointwise 4x + act        │
                    │ + pointwise contract + LayerScale  │
                    └──────────────────┬─────────────────┘
                                        ▼
                                  pooling head
                                        │
                                        ▼
                              sequence vector [B, H]

    :param hidden_size: Model width.
    :type hidden_size: int
    :param num_layers: Number of ConvNeXt blocks.
    :type num_layers: int
    :param kernel_size: Depthwise kernel width along the sequence axis.
    :type kernel_size: int
    :param block_activation: Activation between the pointwise convolutions.
    :type block_activation: str
    :param gamma_initial_value: Initial LayerScale value.
    :type gamma_initial_value: float
    :param use_gamma: Whether to apply LayerScale at all.
    :type use_gamma: bool
    :param block_normalization_type: Normalization inside the block.
    :type block_normalization_type: str
    :param kwargs: Forwarded to :class:`EmbeddingEncoder`.

    Variants:

    .. code-block:: text

        variant   hidden_size   num_layers   kernel_size
        tiny      128           4            7
        small     256           6            7
        base      512           8            7
    """

    #: Public variant registry, depth- and width-matched to
    #: :attr:`...ascii_bert.AsciiBert.MODEL_VARIANTS` so the size axis lines up.
    #: The FFN width is not a knob here: ``ConvNextV1Block`` fixes the expansion
    #: at 4x, which happens to match the baseline arm's 4x intermediate size.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "hidden_size": 128,
            "num_layers": 4,
            "kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-tiny.",
        },
        "small": {
            "hidden_size": 256,
            "num_layers": 6,
            "kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-small.",
        },
        "base": {
            "hidden_size": 512,
            "num_layers": 8,
            "kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-base.",
        },
    }

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        kernel_size: int = 7,
        block_activation: str = "gelu",
        gamma_initial_value: float = 1.0,
        use_gamma: bool = True,
        block_normalization_type: str = "layernorm",
        **kwargs: Any,
    ) -> None:
        self.kernel_size = kernel_size
        self.block_activation = block_activation
        self.gamma_initial_value = gamma_initial_value
        self.use_gamma = use_gamma
        self.block_normalization_type = block_normalization_type

        block_config = {
            "kernel_size": kernel_size,
            "activation": block_activation,
            "gamma_initial_value": gamma_initial_value,
            "use_gamma": use_gamma,
            "normalization_type": block_normalization_type,
            "dropout_rate": kwargs.get("hidden_dropout_rate", 0.1),
        }

        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            block_type="convnext",
            block_config=block_config,
            **kwargs,
        )

        span = conv_receptive_field(num_layers, kernel_size)
        logger.info(
            f"AsciiConvNextBert token-mixing span: {span} characters "
            f"({num_layers} blocks x 1 depthwise conv of width {kernel_size})"
        )
        if span < self.max_position_embeddings:
            logger.warning(
                f"Token-mixing span ({span}) is shorter than "
                f"max_position_embeddings ({self.max_position_embeddings}): "
                "positions further apart than the span cannot interact. Raise "
                "kernel_size or num_layers. This arm applies one conv per "
                "block, so its span is half a Clifford stack's at equal depth "
                "and kernel."
            )

    @property
    def receptive_field(self) -> int:
        """Token-mixing span of the stack, in characters.

        :return: ``num_layers * (kernel_size - 1) + 1``.
        :rtype: int
        """
        return conv_receptive_field(self.num_layers, self.kernel_size)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> "AsciiConvNextBert":
        """Construct a named variant.

        :param variant: A key of :data:`MODEL_VARIANTS`.
        :type variant: str
        :param pretrained: Must be ``False``. No weights are distributed for
            this experimental family. The refusal lives here rather than in the
            module factory so a caller reaching the class directly cannot get
            silently-random weights.
        :type pretrained: bool
        :param kwargs: Overrides applied on top of the variant.
        :type kwargs: Any
        :return: The configured model.
        :rtype: AsciiConvNextBert
        :raises ValueError: If ``variant`` is unknown.
        :raises NotImplementedError: If ``pretrained`` is truthy.
        """
        if pretrained:
            raise NotImplementedError(
                "No pretrained weights are distributed for AsciiConvNextBert. "
                "Train one with `python -m "
                "train.embeddings_experimental.train_embeddings --model "
                "ascii_convnext_bert`, then load the checkpoint path directly "
                "with keras.models.load_model()."
            )
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Available: {sorted(cls.MODEL_VARIANTS)}"
            )
        config = dict(cls.MODEL_VARIANTS[variant])
        config.pop("description", None)
        config.update(kwargs)
        logger.info(f"Creating AsciiConvNextBert-{variant.upper()}")
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        for key in ("block_type", "block_config"):
            config.pop(key, None)
        config.update(
            {
                "kernel_size": self.kernel_size,
                "block_activation": self.block_activation,
                "gamma_initial_value": self.gamma_initial_value,
                "use_gamma": self.use_gamma,
                "block_normalization_type": self.block_normalization_type,
            }
        )
        return config


def create_ascii_convnext_bert(
    variant: str = "small",
    pretrained: bool = False,
    **kwargs: Any,
) -> AsciiConvNextBert:
    """Create an :class:`AsciiConvNextBert` from a variant name.

    :param variant: A key of :attr:`AsciiConvNextBert.MODEL_VARIANTS`.
    :type variant: str
    :param pretrained: Must be ``False``. No weights are distributed for this
        experimental family.
    :type pretrained: bool
    :param kwargs: Overrides applied on top of the variant.
    :type kwargs: Any
    :return: The configured model.
    :rtype: AsciiConvNextBert
    :raises NotImplementedError: If ``pretrained`` is truthy; the refusal is
        raised by :meth:`AsciiConvNextBert.from_variant`.
    """
    return AsciiConvNextBert.from_variant(
        variant, pretrained=pretrained, **kwargs
    )
