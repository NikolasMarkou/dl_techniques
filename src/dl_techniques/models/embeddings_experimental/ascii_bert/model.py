"""ASCII BERT: the study's baseline arm.

A bidirectional transformer encoder over the character-level ASCII vocabulary.
Architecturally this is ``models/language/bert`` with two changes the study
requires: the 101-id ASCII vocabulary in place of a 30 522-id WordPiece one,
and a pooling head, since an embedding model must emit one vector per sequence
and upstream BERT owns no pooler.

Everything else -- the embeddings, the block stack, the depth and width ladder,
the output contract -- is shared with every other arm through
:class:`~...shared.encoder.EmbeddingEncoder`, which is what makes a measured
difference attributable to the block rather than to the plumbing.

Two consequences of the ASCII vocabulary are worth stating before any number
from this arm is compared to a sub-word baseline:

1. **The embedding table almost vanishes.** At ``hidden_size=256`` a 30 522-id
   WordPiece table is 7.8 M parameters; the 101-id ASCII table is 25 856. The
   freed budget moves into the blocks, so an arm with the same *name* as a
   published BERT variant is not the same model.
2. **Sequences get roughly five times longer** for the same text, because a
   character is not a sub-word piece. Self-attention is quadratic in sequence
   length, so this arm's cost grows as ``S**2`` where the Clifford arm's grows
   linearly. A like-for-like comparison must say which budget is being held
   fixed; the study's ``param_matched`` mode exists for exactly this.

References:
    - Devlin et al., 2019. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Clark et al., 2022. CANINE: Pre-training an Efficient
      Tokenization-Free Encoder for Language Representation.
      (https://arxiv.org/abs/2103.06874)
    - Reimers and Gurevych, 2019. Sentence-BERT: Sentence Embeddings
      using Siamese BERT-Networks. (https://arxiv.org/abs/1908.10084)
"""

from typing import Any, Dict, Optional

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from ..shared.encoder import EmbeddingEncoder
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = ["AsciiBert", "create_ascii_bert"]


@register_dl_technique("dl_techniques.models.ascii_bert.model")
class AsciiBert(EmbeddingEncoder):
    """
    Transformer-block embedding encoder over the ASCII character vocabulary.

    :param hidden_size: Model width.
    :type hidden_size: int
    :param num_layers: Number of transformer blocks.
    :type num_layers: int
    :param num_heads: Attention heads per block.
    :type num_heads: int
    :param intermediate_size: FFN inner width.
    :type intermediate_size: int
    :param attention_type: Attention registry key.
    :type attention_type: str
    :param ffn_type: FFN registry key.
    :type ffn_type: str
    :param normalization_position: ``"pre"`` or ``"post"``.
    :type normalization_position: str
    :param attention_probs_dropout_rate: Dropout on attention probabilities.
    :type attention_probs_dropout_rate: float
    :param hidden_act: FFN activation.
    :type hidden_act: str
    :param kwargs: Forwarded to :class:`EmbeddingEncoder` (``vocab_size``,
        ``pooling_strategy``, ``max_position_embeddings``, and so on).
    :raises ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
    """

    #: Public variant registry. The ladder is character-level: shallower and
    #: narrower than a sub-word BERT of the same name, because the sequences
    #: are longer and attention is quadratic in their length.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "hidden_size": 128,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 512,
            "description": "Smallest arm; the sweep's default smoke size.",
        },
        "small": {
            "hidden_size": 256,
            "num_layers": 6,
            "num_heads": 8,
            "intermediate_size": 1024,
            "description": "Mid ladder rung; the study's headline size.",
        },
        "base": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "Largest arm that trains comfortably on one GPU.",
        },
    }

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 1024,
        attention_type: str = "multi_head",
        ffn_type: str = "mlp",
        normalization_position: str = "post",
        attention_probs_dropout_rate: Optional[float] = None,
        hidden_act: str = "gelu",
        **kwargs: Any,
    ) -> None:
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_position = normalization_position
        # `None` means "follow hidden_dropout_rate". It previously defaulted to
        # a hard 0.1, which made the study's `hidden_dropout_rate` knob a
        # PARTIAL control of this arm: setting it to 0.0 left attention dropout
        # at 0.1 and still perturbed two training passes by 3.50e-03. A config
        # field that silently governs only part of a model is the defect class
        # `tests/test_train/test_config_fields_are_live.py` exists to catch.
        # Pass a float to override the two independently.
        if attention_probs_dropout_rate is None:
            attention_probs_dropout_rate = kwargs.get("hidden_dropout_rate", 0.1)
        self.attention_probs_dropout_rate = attention_probs_dropout_rate
        self.hidden_act = hidden_act

        block_config = {
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "attention_type": attention_type,
            "ffn_type": ffn_type,
            "normalization_position": normalization_position,
            "attention_dropout_rate": attention_probs_dropout_rate,
            "activation": hidden_act,
            "dropout_rate": kwargs.get("hidden_dropout_rate", 0.1),
            "normalization_type": kwargs.get("normalization_type", "layer_norm"),
            "layer_norm_eps": kwargs.get("layer_norm_eps", 1e-12),
        }

        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            block_type="transformer",
            block_config=block_config,
            **kwargs,
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> "AsciiBert":
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
        :rtype: AsciiBert
        :raises ValueError: If ``variant`` is unknown.
        :raises NotImplementedError: If ``pretrained`` is truthy.
        """
        if pretrained:
            raise NotImplementedError(
                "No pretrained weights are distributed for AsciiBert. Train one with "
                "`python -m train.embeddings_experimental.train_embeddings "
                "--model ascii_bert`, then load the checkpoint path directly "
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
        logger.info(f"Creating AsciiBert-{variant.upper()}")
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        The generic ``block_type`` / ``block_config`` pair is removed, because
        this subclass derives both from its own explicit arguments and does not
        accept them.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        for key in ("block_type", "block_config"):
            config.pop(key, None)
        config.update(
            {
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "attention_type": self.attention_type,
                "ffn_type": self.ffn_type,
                "normalization_position": self.normalization_position,
                "attention_probs_dropout_rate": self.attention_probs_dropout_rate,
                "hidden_act": self.hidden_act,
            }
        )
        return config


def create_ascii_bert(
    variant: str = "small",
    pretrained: bool = False,
    **kwargs: Any,
) -> AsciiBert:
    """Create an :class:`AsciiBert` from a variant name.

    :param variant: A key of :attr:`AsciiBert.MODEL_VARIANTS`.
    :type variant: str
    :param pretrained: Must be ``False``. No weights are distributed for this
        experimental family.
    :type pretrained: bool
    :param kwargs: Overrides applied on top of the variant.
    :type kwargs: Any
    :return: The configured model.
    :rtype: AsciiBert
    :raises NotImplementedError: If ``pretrained`` is truthy; the refusal
        is raised by :meth:`AsciiBert.from_variant`.
    """
    return AsciiBert.from_variant(variant, pretrained=pretrained, **kwargs)
