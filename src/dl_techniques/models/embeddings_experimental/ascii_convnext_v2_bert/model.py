"""ASCII ConvNeXt V2 BERT: the convolutional arm with Global Response Normalization.

The same BERT skeleton as :mod:`...ascii_bert`, and the same block as
:mod:`...ascii_convnext_bert` with ONE addition: Global Response Normalization
after the activation. That single difference is the point of this arm --
paired against the V1 arm it isolates what GRN does, with depth, width,
kernel, expansion and every other choice held fixed.

GRN scores each channel by its L2 magnitude over the sequence, divides that by
the mean score across channels, and reweights -- a channel-competition mechanism
with no parameters beyond a gamma and a beta. It is the whole of what separates
ConvNeXt V2 from V1, so this arm and :mod:`...ascii_convnext_bert` form a
matched pair around exactly one variable.

**GRN makes this arm sensitive to padding LENGTH, and only after training.**
GRN reduces over the sequence, so what sits in the padded region enters every
real position's normalizer. The wrapper zeroes masked positions, and GRN's
score is an L2 SUM -- ``sqrt(sum(x**2) + eps)`` -- so exact zeros contribute
nothing. At initialization every bias is zero, the padded region stays exactly
zero through the whole block, and pad length is exactly inert. Measured on a
6-token prefix, ``hidden_size=32``, 2 blocks, ``K=3``:

    arm          pad-8 vs pad-12 at init    with non-zero biases
    convnext     0.000e+00                  0.000e+00
    convnext_v2  0.000e+00                  3.215e-03

Once training moves the biases off zero the padded region is no longer zero,
GRN's sum picks it up, and batch composition changes every sequence's
embedding. **A smoke test at initialization would report this arm as
padding-safe.** It is not; it is the same shape of trap as LayerScale hiding
the Clifford arm's boundary effect, and the same answer applies -- stage 1 of
the study trains on packed sequences carrying no padding at all.

Properties it shares with the Clifford arm, and one it does not:

- **Cost is linear in sequence length.** No ``S x S`` matrix is built.
- **Token mixing is local**, and the span is a design parameter. But a ConvNeXt
  block applies a SINGLE depthwise convolution where ``CliffordNetBlock``
  applies two, so at equal depth and kernel this arm's receptive field is
  **half** the Clifford arm's: ``num_layers * (K - 1) + 1`` against
  ``num_layers * 2 * (K - 1) + 1``. Matching the two on ``K`` does NOT match
  them on span. :func:`~...shared.blocks.conv_receptive_field` is the one for
  this arm.
- **Padding is not neutral**: a same-padded depthwise convolution pulls zero
  padding into the receptive field of real positions near the boundary, exactly
  as in the Clifford arm. Masked positions are zeroed before the block, which
  bounds the effect without removing it; the study trains stage 1 on packed
  sequences carrying no padding.
- **Unlike the Clifford arm, LayerScale starts at 1.0**, not 1e-5, so the
  block contributes at full magnitude from the first step rather than easing in.
- **Unlike the V1 arm, padding LENGTH is not inert after training** (above).

References:
    - Woo et al., 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with
      Masked Autoencoders. (https://arxiv.org/abs/2301.00808)
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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from ..shared.blocks import conv_receptive_field
from ..shared.encoder import EmbeddingEncoder
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = ["AsciiConvNextV2Bert", "create_ascii_convnext_v2_bert"]


@register_dl_technique("dl_techniques.models.ascii_convnext_v2_bert.model")
class AsciiConvNextV2Bert(EmbeddingEncoder):
    """
    ConvNeXt V2 embedding encoder over the ASCII character vocabulary.

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
    """

    #: Public variant registry, depth- and width-matched to
    #: :attr:`...ascii_bert.AsciiBert.MODEL_VARIANTS` so the size axis lines up.
    #: The FFN width is not a knob here: ``ConvNextV2Block`` fixes the expansion
    #: at 4x, which happens to match the baseline arm's 4x intermediate size.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "hidden_size": 128,
            "num_layers": 4,
            "kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-tiny; GRN enabled.",
        },
        "small": {
            "hidden_size": 256,
            "num_layers": 6,
            "kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-small; GRN enabled.",
        },
        "base": {
            "hidden_size": 512,
            "num_layers": 8,
            "kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-base; GRN enabled.",
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
            block_type="convnext_v2",
            block_config=block_config,
            **kwargs,
        )

        span = conv_receptive_field(num_layers, kernel_size)
        logger.info(
            f"AsciiConvNextV2Bert token-mixing span: {span} characters "
            f"({num_layers} blocks x 1 depthwise conv of width {kernel_size})"
        )
        if span < self.max_position_embeddings:
            logger.warning(
                f"Token-mixing span ({span}) is shorter than "
                f"max_position_embeddings ({self.max_position_embeddings}): "
                "positions further apart than the span cannot interact. Raise "
                "kernel_size or num_layers. Note this arm applies ONE conv per "
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
    ) -> "AsciiConvNextV2Bert":
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
        :rtype: AsciiConvNextV2Bert
        :raises ValueError: If ``variant`` is unknown.
        :raises NotImplementedError: If ``pretrained`` is truthy.
        """
        if pretrained:
            raise NotImplementedError(
                "No pretrained weights are distributed for AsciiConvNextV2Bert. "
                "Train one with `python -m "
                "train.embeddings_experimental.train_embeddings --model "
                "ascii_convnext_v2_bert`, then load the checkpoint path directly "
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
        logger.info(f"Creating AsciiConvNextV2Bert-{variant.upper()}")
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


def create_ascii_convnext_v2_bert(
    variant: str = "small",
    pretrained: bool = False,
    **kwargs: Any,
) -> AsciiConvNextV2Bert:
    """Create an :class:`AsciiConvNextV2Bert` from a variant name.

    :param variant: A key of :attr:`AsciiConvNextV2Bert.MODEL_VARIANTS`.
    :type variant: str
    :param pretrained: Must be ``False``. No weights are distributed for this
        experimental family.
    :type pretrained: bool
    :param kwargs: Overrides applied on top of the variant.
    :type kwargs: Any
    :return: The configured model.
    :rtype: AsciiConvNextV2Bert
    :raises NotImplementedError: If ``pretrained`` is truthy; the refusal is
        raised by :meth:`AsciiConvNextV2Bert.from_variant`.
    """
    return AsciiConvNextV2Bert.from_variant(
        variant, pretrained=pretrained, **kwargs
    )
