"""
Tree Transformer, a transformer encoder that induces a soft constituency tree
and uses it to constrain its own attention.

This model embodies the principle that language is hierarchical and a flat
attention matrix has no way to say so. Standard self-attention scores every
token pair independently; nothing in the architecture prefers "the old man" to
"man who is" as a unit, so the model must rediscover phrase structure from data
at every layer, from scratch, with no inductive bias helping it. Supervised
parsers do have that bias but need treebanks, which exist for a handful of
languages. Tree Transformer takes the third route: it makes constituency an
unsupervised byproduct of ordinary language-model training by giving attention
a structural prior it must learn to satisfy.

The mechanism is Group Attention. For each adjacent token pair the model
computes a *break probability* -- how likely a constituent boundary falls
between them -- and from those neighbour scores derives, for every span, the
probability that the span is a single constituent. The derivation is the
non-obvious part: rather than enumerating `O(L^2)` spans explicitly, the
constituent probability of a span is obtained from the break probabilities
inside it by a dynamic-programming recurrence expressed as matrix products, so
the whole span table costs a few matmuls. The resulting matrix multiplies the
standard scaled-dot-product attention weights element-wise, so a token pair
that the model believes to straddle a boundary has its attention attenuated.
Attention is thus biased toward learned constituents without ever being
hard-masked by them.

The tree is built jointly across depth, not per layer. Each block passes its
computed group probabilities to the next as a prior, so boundaries agreed on by
lower layers persist and constituents grow monotonically with depth -- the
hierarchy emerges from the layer stack itself rather than being decoded at the
end. Every block's break probabilities are also collected and returned, shape
`[batch, num_layers, seq_len, seq_len]`, which is the induced grammar and the
artifact a grammar-induction evaluation consumes.

Each block is Pre-LN for training stability, ordering
`GroupAttention -> TreeMHA -> FFN` with residual connections throughout. The
`lm_head` is part of the foundation model and is always built; for pure-encoder
downstream use that is wasted weight, and the pretraining script saves a
separate encoder-only artifact. Four preset variants span tiny through large.

No pretrained weights are distributed with this package. `pretrained=True`
raises `NotImplementedError` rather than warning and returning a randomly
initialized model, which is a deliberate choice: the previous behaviour held a
table of unreachable weight URLs and swallowed the download failure, making an
unavailable checkpoint silently indistinguishable from a successful load. Pass
a local `.keras` path to `pretrained` instead.

References:
    - Shen et al., 2019. Tree Transformer: Integrating Tree Structures into
      Self-Attention. (https://arxiv.org/abs/1904.00035)
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

# ---------------------------------------------------------------------

from .components import (  # noqa: F401
    PositionalEncoding,
    GroupAttention,
    TreeMHA,
    TreeTransformerBlock,
)

# ---------------------------------------------------------------------
# Main TreeTransformer Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TreeTransformer(keras.Model):
    """Tree Transformer model for grammar induction and language modeling.

    This is a pure encoder implementation with pretrained weights support,
    designed to produce contextual token representations alongside learned
    syntactic structures (break probabilities). It separates the core transformer
    architecture from any task-specific layers, making it highly flexible.

    The model expects input as a dictionary containing 'input_ids'. It outputs a
    dictionary containing 'last_hidden_state', 'logits' (from LM head), and
    'break_probs' from all layers.

    **Architecture Overview**:
    .. code-block:: text

        Input({"input_ids": [B, L]})
               │
               ▼
        Embedding & PositionalEncoding
               │
               ▼
        TreeTransformerBlock₁ (passes `group_prob` forward)
               │
               ▼
              ...
               │
               ▼
        TreeTransformerBlockₙ
               │
               ▼
        Final LayerNorm ───► LM Head ───► "logits": [B, L, V]
               │
               ▼
        Output Dictionary {
            "last_hidden_state": [B, L, H],
            "logits": [B, L, V],
            "break_probs": [B, N, L, L]
        }

    :param vocab_size: Size of the vocabulary. Defaults to 30000.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Defaults to 512.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 10.
    :type num_layers: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the FFN layer. Defaults to 2048.
    :type intermediate_size: int
    :param hidden_act: Activation function in the encoder. Defaults to "gelu".
    :type hidden_act: str
    :param hidden_dropout_rate: Dropout for embeddings/encoder. Defaults to 0.1.
    :type hidden_dropout_rate: float
    :param attention_dropout_rate: Dropout for attention scores. Defaults to 0.1.
    :type attention_dropout_rate: float
    :param max_len: Maximum sequence length. Defaults to 256.
    :type max_len: int
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-6.
    :type layer_norm_eps: float
    :param pad_token_id: ID of the padding token. Defaults to 0.
    :type pad_token_id: int
    :param normalization_type: Type of normalization layer. Defaults to "layer_norm".
    :type normalization_type: str
    :param ffn_type: Type of feed-forward network. Defaults to "mlp".
    :type ffn_type: str
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :raises ValueError: If invalid configuration parameters are provided.
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
        """Initializes the TreeTransformer model instance."""
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

        # Create all sub-layers
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
        """Validates model configuration parameters."""
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
        """Builds all model components."""
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
        """Computes the output shape of the layer."""
        if isinstance(input_shape, dict):
            input_shape = input_shape["input_ids"]

        batch_size, seq_len = input_shape

        return {
            "last_hidden_state": (batch_size, seq_len, self.hidden_size),
            "logits": (batch_size, seq_len, self.vocab_size),
            "break_probs": (batch_size, self.num_layers, seq_len, seq_len),
        }

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the TreeTransformer model."""
        explicit_attention_mask = None
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
            # B-3 fix: honor an explicitly-provided attention_mask when the
            # caller passes a dict (BERT-style API). When absent, fall back to
            # deriving the mask from `input_ids != pad_token_id` so callers
            # that only pass `input_ids` keep working unchanged.
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

        # Initialize group probability for the first layer as a scalar tensor.
        # This must be a tensor, not a float, for Keras tracing to work correctly.
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
        """Loads pretrained weights into the model from a ``.keras`` checkpoint.

        Uses :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`
        which walks layers by name and calls ``set_weights`` when shapes match.

        B-4 fix: replaces the previous ``self.load_weights(..., by_name=True)``
        path which is broken on Keras 3.8 ``.keras`` files.

        :param weights_path: Path to a ``.keras`` checkpoint.
        :param skip_prefixes: Layer-name prefixes to skip during transfer.
        :param strict: If True, raise on any shape mismatch.
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

    # `_download_weights` raises instead of falling back to random init. This
    # class used to carry a `PRETRAINED_WEIGHTS` table of placeholder URLs on a
    # non-existent host; `from_variant` caught the resulting download failure,
    # logged a warning and returned a randomly-initialized model, so
    # `pretrained=True` silently produced untrained weights. The B-5 fix emptied
    # the table and this method has raised since; the empty dict itself was then
    # dead machinery that could never succeed, so it is gone too. Do NOT
    # reinstate either the table or a warn-and-return branch -- the except clause
    # in `from_variant` is deliberately narrow (see the D-001 anchor) so this
    # NotImplementedError reaches the caller.
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
        """Creates a TreeTransformer model from a predefined variant.

        Args:
            variant: One of ``cls.MODEL_VARIANTS`` (e.g. ``"tiny"``, ``"small"``,
                ``"base"``, ``"large"``).
            pretrained: ``False`` (default) for random init. ``True`` to attempt
                downloading hosted weights — this currently raises
                :class:`NotImplementedError` because no public Tree Transformer
                weights are hosted. A string path is treated as a local
                ``.keras`` / ``.weights.h5`` file to load.
            weights_dataset: Dataset key for hosted weights (kept for API
                parity with BERT / DistilBERT / ResNet — currently unused since
                no public weights are hosted).
            cache_dir: Optional cache directory for downloaded weights.
            **kwargs: Forwarded to ``TreeTransformer.__init__``.

        Raises:
            NotImplementedError: If ``pretrained=True``. Use
                ``pretrained="path/to/weights.keras"`` to load local weights.
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
                # DECISION plan_2026-05-11_0a5779e8/D-001
                # Narrow the except clause so NotImplementedError raised by
                # _download_weights (no public Tree Transformer weights are
                # hosted) propagates to the caller. Previously this was caught
                # by `except Exception`, silently random-initializing the
                # model and misleading users into believing they got
                # pretrained weights. Only catch genuine network/disk errors.
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
                # skip_mismatch (legacy local flag) controls whether vocab-dependent
                # layers are excluded from transfer. Map to skip_prefixes for the
                # new weight_transfer-based loader.
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
        """Returns the model's configuration for serialization."""
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
        """Creates a model instance from its configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Prints the model summary with additional configuration details."""
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
    """Convenience function to create Tree Transformer encoder models.

    Mirrors :func:`dl_techniques.models.resnet.model.create_resnet` for consistency
    across the model zoo: a thin module-level factory that delegates to
    :meth:`TreeTransformer.from_variant`.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large").
        vocab_size: Optional integer; override the variant default vocabulary
            size. Passing a value different from
            :attr:`TreeTransformer.DEFAULT_VOCAB_SIZE` while ``pretrained=True``
            will skip loading vocab-dependent layers (embeddings, LM head).
        pretrained: Boolean or string. If ``True``, attempts to load pretrained
            weights for the chosen ``weights_dataset`` (currently raises
            :class:`NotImplementedError` — no public Tree Transformer weights
            are hosted). If a string, treated as a path to a local
            ``.keras`` / ``.weights.h5`` file.
        weights_dataset: String, dataset key for pretrained weights (kept for
            API parity with other foundation models).
        cache_dir: Optional string, directory to cache downloaded weights.
        **kwargs: Additional arguments forwarded to ``TreeTransformer.__init__``
            (e.g. ``hidden_dropout_rate``, ``max_len``, ``pad_token_id``).

    Returns:
        TreeTransformer encoder instance.

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
    """Factory function to create a Tree Transformer model with a task-specific head.

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
    """
    encoder_config_overrides = encoder_config_overrides or {}
    head_config_overrides = head_config_overrides or {}
    logger.info(
        f"Creating TreeTransformer-{tree_transformer_variant} with a '{task_config.name}' head."
    )

    # 1. Create the foundational TreeTransformer model
    tree_encoder = TreeTransformer.from_variant(
        tree_transformer_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **encoder_config_overrides,
    )

    # 2. Create the task-specific head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=tree_encoder.hidden_size,
        **head_config_overrides,
    )

    # 3. Define inputs and build the end-to-end model
    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        )
    }
    encoder_outputs = tree_encoder(inputs)

    # Pass encoder outputs to the task head.
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
