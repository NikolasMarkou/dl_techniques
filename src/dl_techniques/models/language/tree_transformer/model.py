"""
Tree Transformer: an encoder that induces a soft constituency tree and uses it to
constrain its own attention.

Defines :class:`TreeTransformer` and the factories that build it. Standard
self-attention scores every token pair on its own, with nothing preferring "the old
man" as a unit over "man who is". Tree Transformer computes, for each adjacent token
pair, the probability that a constituent boundary falls between them, then turns those
neighbour scores into the probability that each span is one constituent through a
dynamic-programming recurrence written as matrix products rather than an enumeration
over spans. That span matrix multiplies the ordinary attention weights element-wise,
so a pair the model reads as straddling a boundary is attenuated instead of
hard-masked. Each block hands its group probabilities to the next as a prior, so
boundaries persist and constituents grow with depth, and every block's break
probabilities come back stacked as the induced grammar. Blocks are Pre-LN, ordered
GroupAttention, TreeMHA, then FFN. ``lm_head`` is always built, and no pretrained
weights ship with this package: ``pretrained=True`` raises ``NotImplementedError``,
so pass a local ``.keras`` path instead.

References:
    - Wang et al., 2019. Tree Transformer: Integrating Tree Structures into
      Self-Attention. EMNLP-IJCNLP. (https://arxiv.org/abs/1909.06639)
    - Shen et al., 2019. Ordered Neurons: Integrating Tree Structures into
      Recurrent Neural Networks. (https://arxiv.org/abs/1810.09536)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
"""


import os
import keras
from keras import ops
from typing import Optional, Union, Dict, Any, List, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import FFNType
from dl_techniques.layers.norms import (
    create_normalization_layer,
    NormalizationType,
)
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig
from dl_techniques.utils.model_build import materialize_sublayers

from .components import (  # noqa: F401
    PositionalEncoding,
    GroupAttention,
    TreeMHA,
    TreeTransformerBlock,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.tree_transformer.model")
class TreeTransformer(keras.Model):
    """Encode tokens and induce break probabilities in the same forward pass.

    A pure encoder with no task-specific layers beyond the language-modelling head.
    ``call`` accepts either a tensor of token ids or a dictionary with ``input_ids``
    and an optional ``attention_mask``, and returns a dictionary with
    ``last_hidden_state``, ``logits`` and the per-layer ``break_probs``.

    Architecture:

    .. code-block:: text

        {"input_ids": [B, L]}   (+ optional "attention_mask")
                 │
                 ▼
        ┌───────────────────┐
        │ embedding         │  x sqrt(hidden_size)
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ pos_encoding      │
        └───────────────────┘
                 │  [B, L, H]
                 ▼
        ┌───────────────────┐
        │ block_0           │──► break_prob
        └───────────────────┘
                 │  group_prob carried forward
                 ▼
                ...
                 │
                 ▼
        ┌───────────────────┐
        │ block_{n-1}       │──► break_prob
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ final_norm        │──► "last_hidden_state"  [B, L, H]
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ lm_head           │──► "logits"  [B, L, V]
        └───────────────────┘

        every break_prob stacked ──► "break_probs"  [B, N, L, L]

    Each block takes the previous block's group_prob as its prior; the first gets 0.

    Mask:

    .. code-block:: text

        dict with "attention_mask"?
                    │
            ┌───────┴───────┐
            ▼               ▼
           yes              no
        cast to int32   input_ids != pad_token_id
            │               │
            └───────┬───────┘
                    ▼
           expand_dims axis 1 ──► [B, 1, L]

    Variants:

    .. code-block:: text

        variant  hidden  layers  heads  intermediate
        tiny        128       4      4           512
        small       256       6      4          1024
        base        512      10      8          2048
        large      1024      16     16          4096

    base follows the original paper.

    :param vocab_size: Size of the vocabulary. Defaults to 30000.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Must be divisible by
        ``num_heads``. Defaults to 512.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 10.
    :type num_layers: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the FFN layer. Defaults to 2048.
    :type intermediate_size: int
    :param hidden_act: Activation function in the encoder. Defaults to "gelu".
    :type hidden_act: str
    :param hidden_dropout_rate: Dropout for embeddings/encoder, in [0, 1]. Defaults to 0.1.
    :type hidden_dropout_rate: float
    :param attention_dropout_rate: Dropout for attention scores, in [0, 1]. Defaults to 0.1.
    :type attention_dropout_rate: float
    :param max_len: Maximum sequence length the positional encoding covers. Defaults to 256.
    :type max_len: int
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-6.
    :type layer_norm_eps: float
    :param pad_token_id: ID of the padding token, used to derive the mask when the
        caller passes none. Defaults to 0.
    :type pad_token_id: int
    :param normalization_type: Type of normalization layer. Defaults to "layer_norm".
    :type normalization_type: str
    :param ffn_type: Type of feed-forward network. Defaults to "mlp".
    :type ffn_type: str
    :param **kwargs: Additional keyword arguments for the `keras.Model`.

    :raises ValueError: If ``vocab_size``, ``hidden_size``, ``num_layers`` or
        ``num_heads`` is not positive, if ``hidden_size`` is not divisible by
        ``num_heads``, or if either dropout rate falls outside [0, 1].
    """

    MODEL_VARIANTS = {
        "large": {
            "hidden_size": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "intermediate_size": 4096,
            "description": "TreeTransformer-Large: High capacity for large datasets",
        },
        "base": {
            "hidden_size": 512,
            "num_layers": 10,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "TreeTransformer-Base: Balanced performance, based on original paper",
        },
        "small": {
            "hidden_size": 256,
            "num_layers": 6,
            "num_heads": 4,
            "intermediate_size": 1024,
            "description": "TreeTransformer-Small: Lightweight for faster training",
        },
        "tiny": {
            "hidden_size": 128,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 512,
            "description": "TreeTransformer-Tiny: Ultra-lightweight for research",
        },
    }

    DEFAULT_VOCAB_SIZE = 30000
    DEFAULT_MAX_LEN = 256
    DEFAULT_LAYER_NORM_EPSILON = 1e-6
    DEFAULT_HIDDEN_ACT = "gelu"
    DEFAULT_PAD_TOKEN_ID = 0

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_act: str = DEFAULT_HIDDEN_ACT,
        hidden_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        max_len: int = DEFAULT_MAX_LEN,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        normalization_type: NormalizationType = "layer_norm",
        ffn_type: FFNType = "mlp",
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

        Arguments are documented on the class.
        """
        super().__init__(**kwargs)

        self._validate_config(
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            hidden_dropout_rate,
            attention_dropout_rate,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.max_len = max_len
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type

        self._build_architecture()
        logger.info(
            f"Created Tree Transformer foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dropout_rate: float,
        attention_dropout_rate: float,
    ) -> None:
        """Check the size arguments and the two dropout rates.

        :raises ValueError: If a size is not positive, ``hidden_size`` is not
            divisible by ``num_heads``, or a dropout rate is outside [0, 1].
        """
        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {vocab_size}"
            )
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= hidden_dropout_rate <= 1.0):
            raise ValueError(
                f"hidden_dropout_rate must be in [0, 1], got {hidden_dropout_rate}"
            )
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1], got {attention_dropout_rate}"
            )

    def _build_architecture(self) -> None:
        """Create the embedding, positional encoding, blocks, final norm and LM head."""
        self.embedding = keras.layers.Embedding(
            self.vocab_size, self.hidden_size, name="embedding"
        )
        self.pos_encoding = PositionalEncoding(
            hidden_size=self.hidden_size,
            dropout_rate=self.hidden_dropout_rate,
            max_len=self.max_len,
            name="pos_encoding",
        )
        self.blocks: List[TreeTransformerBlock] = [
            TreeTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                hidden_dropout_rate=self.hidden_dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                normalization_type=self.normalization_type,
                ffn_type=self.ffn_type,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            epsilon=self.layer_norm_eps,
            name="final_norm",
        )
        self.lm_head = keras.layers.Dense(
            self.vocab_size, name="lm_head_projection"
        )

    def compute_output_shape(
        self, input_shape: Any
    ) -> Dict[str, Any]:
        """Compute the shapes of all three outputs.

        :param input_shape: A ``(batch, seq_len)`` shape, or a dict holding one under
            ``'input_ids'``.
        :type input_shape: Any
        :return: Shapes for ``last_hidden_state``, ``logits`` and ``break_probs``.
        :rtype: Dict[str, Any]
        """
        if isinstance(input_shape, dict):
            input_shape = input_shape["input_ids"]

        batch_size, seq_len = input_shape

        return {
            "last_hidden_state": (batch_size, seq_len, self.hidden_size),
            "logits": (batch_size, seq_len, self.vocab_size),
            "break_probs": (batch_size, self.num_layers, seq_len, seq_len),
        }

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method TreeTransformer inherits ``Layer.build``, which marks the
        model built while its sub-layers are still unbuilt, and Keras warns about it.
        The shared helper traces ``call()`` on symbolic inputs, so what gets built
        matches what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Embed the ids, run every block, and return hidden states, logits and breaks.

        :param inputs: Token ids ``(B, L)``, or a dictionary with ``'input_ids'`` and
            optionally ``'attention_mask'``. Without a mask, one is derived from
            ``input_ids != pad_token_id``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call runs in training mode.
        :type training: Optional[bool]
        :return: ``last_hidden_state`` ``(B, L, hidden_size)``, ``logits``
            ``(B, L, vocab_size)`` and ``break_probs``
            ``(B, num_layers, L, L)``.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If a dictionary input has no ``'input_ids'`` key.
        """
        explicit_attention_mask = None
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
            # An explicit attention_mask wins, so a caller passing only input_ids
            # still gets the pad-derived mask.
            explicit_attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs

        if explicit_attention_mask is not None:
            mask = ops.cast(explicit_attention_mask, "int32")
        else:
            mask = ops.cast(
                ops.not_equal(input_ids, self.pad_token_id), "int32"
            )
        mask = ops.expand_dims(mask, axis=1)

        x = self.embedding(input_ids)
        x *= ops.cast(self.hidden_size, x.dtype) ** 0.5
        x = self.pos_encoding(x, training=training)

        # A tensor, not a float, so Keras can trace the first block.
        group_prob: keras.KerasTensor = ops.convert_to_tensor(
            0.0, dtype=self.compute_dtype
        )
        all_break_probs = []

        for block in self.blocks:
            x, group_prob, break_prob = block(
                (x, mask, group_prob), training=training
            )
            all_break_probs.append(break_prob)

        last_hidden_state = self.final_norm(x)
        logits = self.lm_head(last_hidden_state)
        stacked_break_probs = ops.stack(all_break_probs, axis=1)

        return {
            "last_hidden_state": last_hidden_state,
            "logits": logits,
            "break_probs": stacked_break_probs,
        }

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_prefixes: Sequence[str] = (),
        strict: bool = False,
    ) -> None:
        """Load weights from a ``.keras`` checkpoint by layer name.

        Uses :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
        which walks layers by name and calls ``set_weights`` when shapes match. An
        unbuilt model is first called on a dummy batch, since name matching needs the
        layers to exist.

        :param weights_path: Path to a ``.keras`` checkpoint.
        :param skip_prefixes: Layer-name prefixes to skip during transfer.
        :param strict: If True, raise on any shape mismatch.
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
        if not self.built:
            seq_len = min(64, self.max_len)
            dummy_input = {
                "input_ids": ops.cast(
                    keras.random.uniform(
                        (1, seq_len), 0, self.vocab_size
                    ),
                    "int32",
                )
            }
            self(dummy_input, training=False)
        logger.info(f"Loading pretrained weights from {weights_path}")
        report = load_weights_from_checkpoint(
            target=self,
            ckpt_path=weights_path,
            skip_prefixes=skip_prefixes,
            strict=strict,
        )
        logger.info(report.summary_string())

    @staticmethod
    def _download_weights(
        variant: str, dataset: str = "uncased", cache_dir: Optional[str] = None
    ) -> str:
        """No public pretrained weights exist for TreeTransformer; always raises.

        Pass ``pretrained=<path/to/checkpoint.keras>`` to ``from_variant`` to
        load local weights, or omit ``pretrained`` to initialize randomly.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No public pretrained TreeTransformer weights are distributed with "
            f"dl_techniques (requested variant '{variant}', dataset '{dataset}'). "
            f"Pass a local checkpoint instead: TreeTransformer.from_variant("
            f"'{variant}', pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "TreeTransformer":
        """Create a TreeTransformer from a predefined variant.

        A download that fails with an I/O or value error logs a warning and leaves the
        model randomly initialized; a missing local file or a failed transfer raises.

        :param variant: One of ``cls.MODEL_VARIANTS`` (e.g. ``"tiny"``, ``"small"``, ``"base"``, ``"large"``).
        :param pretrained: ``False`` (default) for random init. ``True`` to attempt downloading hosted weights — this currently raises :class:`NotImplementedError` because no public Tree Transformer weights are hosted. A string path is treated as a local ``.keras`` / ``.weights.h5`` file to load.
        :param weights_dataset: Dataset key for hosted weights (kept for API parity with BERT / DistilBERT / ResNet — currently unused since no public weights are hosted).
        :param cache_dir: Optional cache directory for downloaded weights.
        :param **kwargs: Forwarded to ``TreeTransformer.__init__``. A ``vocab_size``
            other than :attr:`DEFAULT_VOCAB_SIZE` skips the embedding and LM head
            during a pretrained transfer.

        :return: The model, with weights loaded if a path was given.
        :rtype: TreeTransformer
        :raises ValueError: If ``variant`` is not a known name.
        :raises NotImplementedError: If ``pretrained=True``. Use ``pretrained="path/to/weights.keras"`` to load local weights.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        logger.info(
            f"Creating TreeTransformer-{variant.upper()} model: {description}"
        )
        load_weights_path, skip_mismatch = None, False
        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
            else:
                # DECISION plan_2026-05-11_0a5779e8/D-001: catch I/O errors only; a bare
                # Exception swallows NotImplementedError into a silent random init.
                # See decisions.md.
                try:
                    load_weights_path = cls._download_weights(
                        variant, weights_dataset, cache_dir
                    )
                except (IOError, OSError, ValueError) as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {e}. "
                        "Continuing with random initialization."
                    )
            if (
                "vocab_size" in kwargs
                and kwargs["vocab_size"] != cls.DEFAULT_VOCAB_SIZE
            ):
                skip_mismatch = True
                logger.info(
                    "Custom vocab_size differs from pretrained, will skip "
                    "embedding and LM head weights."
                )
        config.update(kwargs)
        model = cls(**config)
        if load_weights_path:
            try:
                # The vocab-dependent layers are the ones a changed vocab_size
                # invalidates, so they become the loader's skip prefixes.
                skip_prefixes = (
                    ("embedding", "lm_head") if skip_mismatch else ()
                )
                model.load_pretrained_weights(
                    load_weights_path,
                    skip_prefixes=skip_prefixes,
                    strict=False,
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                raise
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model's configuration for serialization.

        :return: Dict holding every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "max_len": self.max_len,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "normalization_type": self.normalization_type,
                "ffn_type": self.ffn_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TreeTransformer":
        """Create a model instance from its configuration.

        :param config: Dict as returned by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new model.
        :rtype: TreeTransformer
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the Keras summary, then log the architecture settings.

        :param **kwargs: Forwarded to ``keras.Model.summary``.
        """
        super().summary(**kwargs)
        logger.info("Tree Transformer Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size"
        )
        logger.info(f"  - Attention: {self.num_heads} heads")
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Max sequence length: {self.max_len}")
        logger.info(
            f"  - Normalization: {self.normalization_type} (Pre-LN in blocks)"
        )
        logger.info(
            f"  - Feed-forward: {self.ffn_type}, {self.intermediate_size} intermediate size"
        )


def create_tree_transformer(
        variant: str = "base",
        vocab_size: Optional[int] = None,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
) -> "TreeTransformer":
    """Create a Tree Transformer encoder from a variant name.

    A thin module-level factory that delegates to
    :meth:`TreeTransformer.from_variant`, mirroring
    :func:`dl_techniques.models.vision.resnet.model.create_resnet`.

    :param variant: String, model variant ("tiny", "small", "base", "large").
    :param vocab_size: Optional integer; override the variant default vocabulary size. Passing a value different from :attr:`TreeTransformer.DEFAULT_VOCAB_SIZE` while ``pretrained=True`` will skip loading vocab-dependent layers (embeddings, LM head).
    :param pretrained: Boolean or string. If ``True``, attempts to load pretrained weights for the chosen ``weights_dataset`` (currently raises :class:`NotImplementedError` — no public Tree Transformer weights are hosted). If a string, treated as a path to a local ``.keras`` / ``.weights.h5`` file.
    :param weights_dataset: String, dataset key for pretrained weights (kept for API parity with other foundation models).
    :param cache_dir: Optional string, directory to cache downloaded weights.
    :param **kwargs: Additional arguments forwarded to ``TreeTransformer.__init__`` (e.g. ``hidden_dropout_rate``, ``max_len``, ``pad_token_id``).

    :return: TreeTransformer encoder instance.
    :raises ValueError: If ``variant`` is unknown, or a forwarded argument is invalid.
    :raises NotImplementedError: If ``pretrained=True``.

    Example:
        >>> # Create a Tree Transformer base encoder with random init
        >>> model = create_tree_transformer("base")
        >>>
        >>> # Smaller variant with a custom vocabulary
        >>> model = create_tree_transformer("tiny", vocab_size=8000)
        >>>
        >>> # Load from local weights file
        >>> model = create_tree_transformer("base", pretrained="path/to/weights.keras")
    """
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size
    return TreeTransformer.from_variant(
        variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------
# Integration with NLP Task Heads
# ---------------------------------------------------------------------


def create_tree_transformer_with_head(
    tree_transformer_variant: str,
    task_config: NLPTaskConfig,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "uncased",
    cache_dir: Optional[str] = None,
    encoder_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Build an end-to-end model: a Tree Transformer encoder plus a task head.

    The functional model takes ``input_ids`` only, and the head receives only
    ``hidden_states``; the encoder's ``logits`` and ``break_probs`` are not wired
    through, and the head gets no attention mask.

    .. code-block:: text

        {"input_ids": [B, L]}
                 │
                 ▼
        ┌───────────────────────┐
        │ TreeTransformer       │
        └───────────────────────┘
                 │  last_hidden_state
                 ▼
        ┌───────────────────────┐
        │ nlp task head         │
        └───────────────────────┘
                 │
                 ▼
            task outputs

    :param tree_transformer_variant: The Tree Transformer variant (e.g., "base").
    :type tree_transformer_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: If ``True``, attempts to load hosted weights — this
        currently raises ``NotImplementedError`` (no public Tree Transformer
        weights are hosted). Pass a string path to a local
        ``.keras`` / ``.weights.h5`` file to load weights instead.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset key for hosted weights ("uncased"). Kept
        for API parity; currently unused since no public weights are hosted.
    :type weights_dataset: str
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param encoder_config_overrides: Dict to override default encoder config.
    :type encoder_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Dict to override default head config.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete `keras.Model` ready for the specified task.
    :rtype: keras.Model
    :raises ValueError: If the variant is unknown, or an override is invalid.
    :raises NotImplementedError: If ``pretrained=True``.
    """
    encoder_config_overrides = encoder_config_overrides or {}
    head_config_overrides = head_config_overrides or {}
    logger.info(
        f"Creating TreeTransformer-{tree_transformer_variant} with a '{task_config.name}' head."
    )

    tree_encoder = TreeTransformer.from_variant(
        tree_transformer_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **encoder_config_overrides,
    )

    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=tree_encoder.hidden_size,
        **head_config_overrides,
    )

    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        )
    }
    encoder_outputs = tree_encoder(inputs)

    head_inputs = {"hidden_states": encoder_outputs["last_hidden_state"]}
    task_outputs = task_head(head_inputs)

    model_name = (
        f"tree_transformer_{tree_transformer_variant}_with_{task_config.name}_head"
    )
    model = keras.Model(inputs=inputs, outputs=task_outputs, name=model_name)
    logger.info(
        f"Successfully created model with {model.count_params():,} parameters."
    )
    return model

# ---------------------------------------------------------------------
