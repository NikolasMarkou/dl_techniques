"""ASCII Clifford BERT: the study's attention-free arm.

The same BERT skeleton as :mod:`...ascii_bert` -- same ASCII embeddings, same
depth and width ladder, same pooling, same output contract -- with self-attention
replaced by bidirectional sequence-mode
:class:`~dl_techniques.layers.geometric.clifford_block.CliffordNetBlock`
mixing: a shifted geometric product along the channel axis plus a depthwise
convolutional context branch. There is no attention anywhere in this arm.

Two properties separate it from the baseline, and both must be read before its
numbers are compared to anything.

**Cost is linear in sequence length, not quadratic.** The geometric product
rolls along the CHANNEL axis and the context branch is a depthwise convolution,
so nothing here builds an ``S x S`` matrix. At character granularity, where
sequences are roughly five times longer than their sub-word equivalents, that
is the whole reason this arm is interesting.

**Token mixing is local, and its span is a design parameter.** Because the
geometric product mixes channels rather than positions, ALL cross-token mixing
comes from the two stacked depthwise convolutions inside each block. A stack of
``num_layers`` blocks at ``context_kernel_size = K`` reaches
``num_layers * 2 * (K - 1) + 1`` tokens -- 17 characters for the layer's default
``K = 3`` at four layers, which is a couple of words. This arm therefore
defaults to ``context_kernel_size = 7`` rather than inheriting 3, and exposes
``use_global_context`` (a cumulative mean, unbounded reach) as an opt-in. Use
:func:`~...shared.blocks.clifford_receptive_field` to check any configuration
before training it.

**Padding is not neutral for this arm.** ``CliffordNetBlock`` has
``supports_masking = False`` and its module docstring records the measured
effects: zero padding entering the convolutional receptive field moves the last
real position by 1.183 on a ~2.4-scale output, and with ``use_global_context``
enabled the pad LENGTH shifts every real position by up to 0.449. The wrapper
zeroes masked positions as a partial mitigation, but the study's real answer is
to pretrain on PACKED fixed-length sequences carrying no padding, and to bucket
by length wherever padding cannot be avoided. ``use_global_context`` defaults to
``False`` for the same reason.

References:
    - Ji, Z., 2026. CliffordNet: All You Need is Geometric Algebra.
      (https://arxiv.org/abs/2601.06793)
    - Brandstetter et al., 2023. Clifford Neural Layers for PDE Modeling.
      (https://arxiv.org/abs/2209.04934)
    - Ruhe et al., 2023. Geometric Clifford Algebra Networks.
      (https://arxiv.org/abs/2302.06594)
    - Clark et al., 2022. CANINE: Pre-training an Efficient
      Tokenization-Free Encoder for Language Representation.
      (https://arxiv.org/abs/2103.06874)
    - Touvron et al., 2021. Going Deeper with Image Transformers (CaiT).
      (LayerScale) (https://arxiv.org/abs/2103.17239)
"""

from typing import Any, Dict, List, Optional, Sequence

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from ..shared.blocks import clifford_receptive_field
from ..shared.encoder import EmbeddingEncoder
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = ["AsciiCliffordBert", "create_ascii_clifford_bert"]


@register_dl_technique("dl_techniques.models.ascii_clifford_bert.model")
class AsciiCliffordBert(EmbeddingEncoder):
    """
    Clifford-block embedding encoder over the ASCII character vocabulary.

    :param hidden_size: Model width.
    :type hidden_size: int
    :param num_layers: Number of Clifford blocks.
    :type num_layers: int
    :param shifts: Channel-axis roll offsets for the geometric product.
    :type shifts: Sequence[int] | None
    :param cli_mode: Clifford components used -- ``"inner"`` (symmetric),
        ``"wedge"`` (antisymmetric) or ``"full"``.
    :type cli_mode: str
    :param ctx_mode: Context mode, ``"diff"`` or ``"abs"``.
    :type ctx_mode: str
    :param use_global_context: Whether to add the cumulative-mean global branch.
        Off by default; see the module docstring's padding note.
    :type use_global_context: bool
    :param context_kernel_size: Depthwise kernel width, the sole lever on token
        mixing span. Defaults to 7, not the layer's own 3.
    :type context_kernel_size: int
    :param layer_scale_init: Initial LayerScale gamma.
    :type layer_scale_init: float
    :param block_normalization_type: Normalization inside the block, or ``None``
        for the layer's sequence-mode default.
    :type block_normalization_type: str | None
    :param kwargs: Forwarded to :class:`EmbeddingEncoder`.
    """

    #: Public variant registry, depth- and width-matched to
    #: :attr:`...ascii_bert.AsciiBert.MODEL_VARIANTS` so the two arms line up
    #: on the size axis. ``shifts`` widens with the model, as in the other
    #: Clifford packages in this repo.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "hidden_size": 128,
            "num_layers": 4,
            "shifts": [1, 2],
            "context_kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-tiny.",
        },
        "small": {
            "hidden_size": 256,
            "num_layers": 6,
            "shifts": [1, 2, 4],
            "context_kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-small.",
        },
        "base": {
            "hidden_size": 512,
            "num_layers": 8,
            "shifts": [1, 2, 4, 8],
            "context_kernel_size": 7,
            "description": "Depth/width-matched to AsciiBert-base.",
        },
    }

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        shifts: Optional[Sequence[int]] = None,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_global_context: bool = False,
        context_kernel_size: int = 7,
        layer_scale_init: float = 1e-5,
        block_normalization_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.shifts = list(shifts) if shifts is not None else [1, 2, 4]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.context_kernel_size = context_kernel_size
        self.layer_scale_init = layer_scale_init
        self.block_normalization_type = block_normalization_type

        block_config = {
            "shifts": self.shifts,
            "cli_mode": cli_mode,
            "ctx_mode": ctx_mode,
            "use_global_context": use_global_context,
            "context_kernel_size": context_kernel_size,
            "layer_scale_init": layer_scale_init,
            "normalization_type": block_normalization_type,
        }

        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            block_type="clifford",
            block_config=block_config,
            **kwargs,
        )

        span = clifford_receptive_field(num_layers, context_kernel_size)
        logger.info(
            f"AsciiCliffordBert token-mixing span: {span} characters "
            f"({num_layers} blocks x 2 depthwise convs of width "
            f"{context_kernel_size}), global_context={use_global_context}"
        )
        if not use_global_context and span < self.max_position_embeddings:
            logger.warning(
                f"Token-mixing span ({span}) is shorter than "
                f"max_position_embeddings ({self.max_position_embeddings}): "
                "positions further apart than the span cannot interact. Raise "
                "context_kernel_size or num_layers, or set "
                "use_global_context=True."
            )

    @property
    def receptive_field(self) -> int:
        """Token-mixing span of the stack, in characters.

        :return: ``num_layers * 2 * (context_kernel_size - 1) + 1``.
        :rtype: int
        """
        return clifford_receptive_field(self.num_layers, self.context_kernel_size)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> "AsciiCliffordBert":
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
        :rtype: AsciiCliffordBert
        :raises ValueError: If ``variant`` is unknown.
        :raises NotImplementedError: If ``pretrained`` is truthy.
        """
        if pretrained:
            raise NotImplementedError(
                "No pretrained weights are distributed for AsciiCliffordBert. Train one with "
                "`python -m train.embeddings_experimental.train_embeddings "
                "--model ascii_clifford_bert`, then load the checkpoint path directly "
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
        logger.info(f"Creating AsciiCliffordBert-{variant.upper()}")
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
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "context_kernel_size": self.context_kernel_size,
                "layer_scale_init": self.layer_scale_init,
                "block_normalization_type": self.block_normalization_type,
            }
        )
        return config


def create_ascii_clifford_bert(
    variant: str = "small",
    pretrained: bool = False,
    **kwargs: Any,
) -> AsciiCliffordBert:
    """Create an :class:`AsciiCliffordBert` from a variant name.

    :param variant: A key of :attr:`AsciiCliffordBert.MODEL_VARIANTS`.
    :type variant: str
    :param pretrained: Must be ``False``. No weights are distributed for this
        experimental family.
    :type pretrained: bool
    :param kwargs: Overrides applied on top of the variant.
    :type kwargs: Any
    :return: The configured model.
    :rtype: AsciiCliffordBert
    :raises NotImplementedError: If ``pretrained`` is truthy; the refusal
        is raised by :meth:`AsciiCliffordBert.from_variant`.
    """
    return AsciiCliffordBert.from_variant(variant, pretrained=pretrained, **kwargs)
