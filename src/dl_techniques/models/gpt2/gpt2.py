"""Decoder-only GPT-2 language model with an optionally weight-tied output head.

GPT-2 makes one claim about language: every supervised NLP task is a subset of
next-token prediction, so a single objective `p(x) = prod_i p(x_i | x_<i)` trained
at sufficient scale on sufficiently varied text acquires the tasks as a side
effect. Nothing in the architecture encodes that claim; what the architecture must
do is make the factorization *exact*. Each position may condition on everything to
its left and on nothing to its right, because a single leaked future token turns
the training loss into a copying shortcut and the model's perplexity into a
fiction. Causality is therefore not an option of this model, it is the model.

The causal constraint is enforced by masking rather than by structure. This
implementation delegates the whole transformer stack to the library's
``TextDecoder``, which rebuilds the mask on every forward pass: a lower-triangular
boolean `causal_mask` in *block* semantics (`True` = suppress), OR-combined with
the padding mask derived from `attention_mask == 0`, then logically inverted once
at the end because the attention layers expect *attend* semantics (`True` = allow).
The inversion is done a single time on the combined mask rather than per component,
which is why a caller supplying an already-inverted padding mask silently unmasks
the future — the polarity convention at the model boundary is 1 = attend, 0 = pad.
The same mask tensor is passed to every layer; there is no per-layer state and no
KV cache, so autoregressive decoding through this class recomputes the full prefix
at each step and costs `O(n^2)` per generated token rather than `O(n)`.

Positions are learned absolute embeddings of size ``max_seq_len``, added to token
embeddings before the stack. Extrapolation beyond ``max_seq_len`` is not merely
degraded but undefined, since those rows do not exist; ``TextDecoder.build`` raises
when the *static* sequence length exceeds the budget. A dynamic (unknown at build
time) sequence length slips past that guard, so the check is a development aid, not
a runtime invariant.

Normalization is pre-norm: each sublayer applies its LayerNorm to the branch input
and leaves the residual stream unnormalized end to end, with one final LayerNorm
before the head. Post-norm — the original 2017 encoder-decoder placement, which
GPT-2 abandoned — puts a normalization on the residual path itself, so gradient
magnitude at initialization grows with depth and training needs a warmup schedule
to survive. Pre-norm gives every layer an unobstructed additive path to the loss,
which is what makes the 48-layer XL variant trainable with a flat schedule. Note
that ``TextDecoder`` additionally normalizes the embedding sum before the first
block; the original GPT-2 applies dropout there but no normalization, so this is a
small deliberate divergence from the reference recipe.

The head is the transposed token embedding matrix by default:
`logits = h @ E^T`. Tying is not only a parameter saving, though at this
configuration it is a large one — a 100277-row by 768-column embedding is roughly
77M weights against the ~85M in the small variant's blocks, so untying nearly
doubles the model. It also couples the input and output representations of a token,
which regularizes rare tokens whose output row would otherwise receive gradient
only from the handful of times they are the target. ``tie_word_embeddings=False``
substitutes an independent bias-free ``Dense``; that is the modern preference at
multi-billion-parameter scale, where the embedding is a small fraction of the total
and the coupling costs more capacity than it saves.

Two departures from the published model are worth stating plainly. The default
vocabulary is 100277 (Tiktoken ``cl100k_base``), not GPT-2's own 50257-entry BPE
vocabulary, so a checkpoint from OpenAI will not load against the defaults; pass
``vocab_size=50257`` for shape compatibility with the original. And ``attention_type``
and ``ffn_type`` are factory keys rather than fixed choices, so the class describes a
GPT-2-*shaped* decoder whose mixer and MLP can be swapped; only the defaults
(``'multi_head'``, ``'mlp'``) reproduce the paper.

No pretrained weights are distributed with this package. ``pretrained=True`` routes
to ``_download_weights``, which raises ``NotImplementedError`` rather than logging a
warning and returning a randomly initialized model — the earlier fallback made an
unavailable download indistinguishable from a successful one at the call site.
Local checkpoints load by path, with a dummy forward pass first to materialize the
lazily-built sublayers.

References:
    - Radford et al., 2019. Language Models are Unsupervised Multitask Learners.
      OpenAI technical report.
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Press and Wolf, 2017. Using the Output Embedding to Improve Language Models.
      (https://arxiv.org/abs/1608.05859)
    - Xiong et al., 2020. On Layer Normalization in the Transformer Architecture.
      (https://arxiv.org/abs/2002.04745)
    - Kaplan et al., 2020. Scaling Laws for Neural Language Models.
      (https://arxiv.org/abs/2001.08361)
"""

import os
import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers.text_decoder import TextDecoder

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GPT2(keras.Model):
    """GPT-2 language model with weight-tied LM head.

    Wraps a ``TextDecoder`` to produce contextual representations and projects
    them to vocabulary logits using the transposed token embedding matrix
    (weight tying).

    :param vocab_size: Vocabulary size. Default: 100277 (Tiktoken cl100k_base).
    :type vocab_size: int
    :param embed_dim: Token embedding / hidden dimension. Default: 768.
    :type embed_dim: int
    :param depth: Number of transformer decoder layers. Default: 12.
    :type depth: int
    :param num_heads: Number of attention heads. Default: 12.
    :type num_heads: int
    :param max_seq_len: Maximum sequence length. Default: 1024.
    :type max_seq_len: int
    :param dropout_rate: Dropout rate for embeddings and residual paths.
        Default: 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate for attention weights.
        Default: 0.0.
    :type attention_dropout_rate: float
    :param initializer_range: Stddev for TruncatedNormal weight init.
        Default: 0.02.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon for LayerNorm. Default: 1e-5.
    :type layer_norm_eps: float
    :param attention_type: Attention mechanism type. Default: ``'multi_head'``.
    :type attention_type: str
    :param ffn_type: FFN architecture type. Default: ``'mlp'``.
    :type ffn_type: str
    :param tie_word_embeddings: If True, the LM head reuses the (transposed)
        token embedding matrix. If False, an independent Dense projection is
        used. Tying saves ``vocab_size * embed_dim`` parameters and matches
        the original GPT-2 recipe; untying is the modern preference at
        multi-billion-parameter scale (Llama 3, OLMo 2, DeepSeek-V3, large
        Qwen3 variants). Default: True.
    :type tie_word_embeddings: bool
    :param kwargs: Additional keyword arguments for ``keras.Model``.

    Example:
        .. code-block:: python

            # Create GPT-2 small from variant
            model = GPT2.from_variant("small")

            # Forward pass
            input_ids = keras.random.uniform((2, 128), 0, 100277, dtype="int32")
            outputs = model(input_ids)
            print(outputs["logits"].shape)  # (2, 128, 100277)

            # Create custom configuration
            model = GPT2(vocab_size=50257, embed_dim=512, depth=6, num_heads=8)
    """

    MODEL_VARIANTS = {
        "xl": {
            "embed_dim": 1600,
            "depth": 48,
            "num_heads": 25,
            "max_seq_len": 1024,
            "description": "GPT-2 XL: ~1558M parameters",
        },
        "large": {
            "embed_dim": 1280,
            "depth": 36,
            "num_heads": 20,
            "max_seq_len": 1024,
            "description": "GPT-2 Large: ~774M parameters",
        },
        "medium": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "max_seq_len": 1024,
            "description": "GPT-2 Medium: ~355M parameters",
        },
        "small": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "max_seq_len": 1024,
            "description": "GPT-2 Small: ~124M parameters",
        },
        "tiny": {
            "embed_dim": 256,
            "depth": 4,
            "num_heads": 4,
            "max_seq_len": 512,
            "description": "GPT-2 Tiny: lightweight for testing and mobile",
        },
    }

    DEFAULT_VOCAB_SIZE = 100277  # Tiktoken cl100k_base
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPS = 1e-5

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPS,
        attention_type: str = "multi_head",
        ffn_type: str = "mlp",
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._validate_config(
            vocab_size, embed_dim, depth, num_heads,
            dropout_rate, attention_dropout_rate,
        )

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.tie_word_embeddings = tie_word_embeddings

        # Build architecture
        self._build_architecture()

        logger.info(
            f"Created GPT-2: {self.depth} layers, "
            f"embed_dim={self.embed_dim}, heads={self.num_heads}, "
            f"max_seq_len={self.max_seq_len}, "
            f"tie_word_embeddings={self.tie_word_embeddings}"
        )

    @staticmethod
    def _validate_config(
        vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ) -> None:
        """Validate model configuration parameters.

        :raises ValueError: If any configuration value is invalid.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )
        if not 0.0 <= attention_dropout_rate <= 1.0:
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, "
                f"got {attention_dropout_rate}"
            )

    def _build_architecture(self) -> None:
        """Build all model components."""
        self.decoder = TextDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            embedding_type="learned",
            positional_type="learned",
            attention_type=self.attention_type,
            normalization_type="layer_norm",
            normalization_position="pre",  # GPT-2 uses pre-layer normalization
            ffn_type=self.ffn_type,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            name="decoder",
        )

        if not self.tie_word_embeddings:
            self.lm_head = keras.layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range,
                ),
                name="lm_head",
            )
        else:
            self.lm_head = None

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of GPT-2.

        :param inputs: Token IDs ``(B, seq_len)`` or a dictionary with
            ``'input_ids'`` and optionally ``'attention_mask'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param attention_mask: Optional padding mask ``(B, seq_len)``.
            1 = attend, 0 = mask. Overridden by dict input if present.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Dictionary with:
            - ``logits``: LM logits ``(B, seq_len, vocab_size)``
            - ``last_hidden_state``: Final hidden states ``(B, seq_len, embed_dim)``
        :rtype: Dict[str, keras.KerasTensor]
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        # Transformer decoder: embeddings → causal attention blocks → final norm
        hidden_states = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            training=training,
        )

        if self.tie_word_embeddings:
            # Weight-tied LM head: logits = hidden_states @ embedding_weights.T
            embedding_weights = self.decoder.word_embeddings.embeddings
            logits = ops.matmul(
                hidden_states, ops.transpose(embedding_weights),
            )
        else:
            logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "last_hidden_state": hidden_states,
        }

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute output shapes given input shape.

        :param input_shape: Input shape ``(batch, seq_len)``.
        :return: Dictionary of output shapes.
        """
        return {
            "logits": (*input_shape, self.vocab_size),
            "last_hidden_state": (*input_shape, self.embed_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "tie_word_embeddings": self.tie_word_embeddings,
        })
        return config

    # DECISION plan_2026-05-11_a9e8e6f6/D-001
    # _download_weights raises NotImplementedError instead of falling back to
    # random-init. The prior silent random-init (logger.warning + return model)
    # misled callers into thinking they had pretrained weights. No public GPT-2
    # weights are distributed with dl_techniques; users must pass a local path
    # via pretrained="/path/to/file.keras" or pretrained=False (default).
    @staticmethod
    def _download_weights(
        variant: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Resolve a local path for pretrained weights of ``variant``.

        Not implemented: no public GPT-2 weights are distributed with
        ``dl_techniques``. Always raises ``NotImplementedError``. This method
        exists to mirror the BERT / tree_transformer / ResNet factory recipe
        and to provide an explicit failure mode in place of a silent
        random-init fallback.

        :param variant: Variant name (unused).
        :type variant: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Pretrained GPT-2 weights are not distributed with dl_techniques. "
            "Pass pretrained=<local_path> to load a local checkpoint, or "
            "pretrained=False to random-init."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "GPT2":
        """Create a GPT-2 model from a predefined variant.

        :param variant: Variant name: ``'tiny'``, ``'small'``, ``'medium'``,
            ``'large'``, ``'xl'``.
        :type variant: str
        :param pretrained: If ``True``, raises ``NotImplementedError`` (no
            public GPT-2 weights are distributed by this library). If a string
            path, loads weights from that local ``.keras`` file. If ``False``
            (default), returns a random-initialized model.
        :type pretrained: Union[bool, str]
        :param kwargs: Override any variant parameter.
        :return: Configured GPT-2 model instance.
        :rtype: GPT2
        :raises ValueError: If the variant name is not recognized.

        Example:
            .. code-block:: python

                model = GPT2.from_variant("small")
                model = GPT2.from_variant("tiny", dropout_rate=0.2)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        config.update(kwargs)

        model = cls(**config)

        if pretrained:
            weights_path = pretrained if isinstance(pretrained, str) else None
            if weights_path is not None:
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(
                        f"Weights file not found: {weights_path}"
                    )
                # Build model before loading weights
                if not model.built:
                    import numpy as np
                    dummy = np.random.randint(
                        0, model.vocab_size, (1, 32)
                    ).astype(np.int32)
                    model(dummy, training=False)
                model.load_weights(weights_path, skip_mismatch=True)
                logger.info(f"Loaded weights from {weights_path}")
            else:
                # DECISION plan_2026-05-11_a9e8e6f6/D-001
                # pretrained=True (boolean) routes through _download_weights,
                # which raises NotImplementedError. Replaces the prior silent
                # random-init fallback (logger.warning + return model) that
                # misled callers (I-01).
                cls._download_weights(variant)

        return model

# ---------------------------------------------------------------------
# Module-level Factory
# ---------------------------------------------------------------------


def create_gpt2(
    variant: str = "small",
    vocab_size: Optional[int] = None,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> "GPT2":
    """Convenience factory that mirrors ``create_bert`` / ``create_resnet`` / ``create_tree_transformer``.

    Thin wrapper around :meth:`GPT2.from_variant` exposing the most common
    construction arguments at module level. Behaves identically to calling
    ``GPT2.from_variant(...)`` directly.

    :param variant: GPT-2 variant name (``"tiny"``, ``"small"``, ``"medium"``,
        ``"large"``, ``"xl"``). Defaults to ``"small"``.
    :type variant: str
    :param vocab_size: Optional vocabulary size override. If ``None`` (default),
        the model's default vocab size (100277, Tiktoken cl100k_base) is used.
        If provided, forwarded as ``vocab_size=...`` in ``kwargs``.
    :type vocab_size: Optional[int]
    :param pretrained: If ``True``, attempts to load pretrained weights — note
        that no public GPT-2 weights are distributed by this library, so
        ``True`` will raise ``NotImplementedError``. If a string path, loads
        local weights from that path. If ``False`` (default), random init.
    :type pretrained: Union[bool, str]
    :param kwargs: Additional keyword arguments forwarded to
        :meth:`GPT2.from_variant` (e.g. ``dropout_rate``, ``tie_word_embeddings``).
    :type kwargs: Any

    :returns: Configured ``GPT2`` instance.
    :rtype: GPT2

    :raises NotImplementedError: If ``pretrained=True`` (no public weights).
    :raises FileNotFoundError: If ``pretrained`` is a string path that does
        not exist.
    :raises ValueError: If ``variant`` is not a recognized GPT-2 variant.

    Example:
        >>> gpt = create_gpt2("small")
        >>> gpt = create_gpt2("tiny", vocab_size=200)
    """
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size
    return GPT2.from_variant(
        variant,
        pretrained=pretrained,
        **kwargs,
    )


# ---------------------------------------------------------------------