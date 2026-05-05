"""CliffordNet causal language model with hierarchical routing head.

A variant of :class:`CliffordNetLM` that replaces the final
``Dense(vocab_size)`` projection with a
:class:`RoutingProbabilitiesLayer`, producing class probabilities directly
from a hierarchical binary routing tree.

The body is identical to ``CliffordNetLM`` (token + positional embeddings,
``CausalCliffordNetBlock`` stack, head LayerNorm + Dropout). Only the head
projection differs.

Two modes are exposed via the ``routing_mode`` argument:

- ``"trainable"`` (default): learnable affine projection ``W x + b`` to
  ``log2(padded_vocab_size)`` decisions. ~3000x fewer params than the Dense
  head at vocab=50k.
- ``"deterministic"``: parameter-free fixed cosine-basis projection. Useful
  for ablation; expressive ceiling of 16 decisions for ~50k-vocab is tight.

Architecture:

.. code-block:: text

    Input IDs (B, seq_len)
         │
         ▼
    Token Embedding + Positional Embedding
    ─► LayerNorm ─► Dropout
         │
         ▼
    Reshape to (B, 1, seq_len, D)
         │
         ▼
    CausalCliffordNetBlock × depth
         │
         ▼
    Reshape to (B, seq_len, D)
         │
         ▼
    LayerNorm ─► Dropout ─► RoutingProbabilitiesLayer(vocab_size)
         │
         ▼
    Probabilities (B, seq_len, vocab_size), sum(-1)≈1, values in [eps, 1-eps]

References:
    Brandstetter, J., et al. (2025). CliffordNet: All You Need is
    Geometric Algebra. arXiv:2601.06793v2.

    Zhang, Z., et al. (2024). Softmax-free Large-scale Language Modeling.
    arXiv:2402.01258.
"""

import keras
from keras import initializers, regularizers
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CausalCliffordNetBlock,
)
from dl_techniques.layers.activations.routing_probabilities import (
    RoutingProbabilitiesLayer,
)
from dl_techniques.layers.embedding.hierarchical_codebook_embedding import (
    HierarchicalCodebookEmbedding,
)

# ---------------------------------------------------------------------------

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)

_VALID_ROUTING_MODES = ("trainable", "deterministic")

# Token embedding strategies. See ``_build_token_embedding`` for descriptions.
_VALID_INPUT_EMBEDDINGS = ("hce", "dense", "albert")


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Linearly spaced drop-path rates from 0 to ``max_rate``."""
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


@keras.saving.register_keras_serializable()
class CliffordNetLMRouting(keras.Model):
    """CliffordNet causal language model with hierarchical routing head.

    Mirrors :class:`CliffordNetLM` exactly except for the final vocab
    projection, which is a :class:`RoutingProbabilitiesLayer` producing
    probabilities (sum-to-1, in ``[eps, 1-eps]``) instead of unnormalized
    logits.

    The output dict still uses the key ``"logits"`` (despite values being
    probabilities) — this keeps the existing training pipeline's data
    wrapper (``(x, y) -> (x, {"logits": y})``) and compile spec
    (``loss={"logits": ...}, metrics={"logits": [...]}``) reusable. Loss
    must be configured with ``from_logits=False``.

    :param vocab_size: Vocabulary size (including special tokens).
    :param max_seq_length: Maximum sequence length for positional embeddings.
    :param channels: Feature dimensionality D (constant throughout blocks).
    :param depth: Number of CausalCliffordNetBlock layers.
    :param shifts: Channel-shift offsets for sparse rolling product.
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``, ``"full"``).
    :param ctx_mode: Context calculation mode (``"diff"`` or ``"abs"``).
    :param use_global_context: Add causal cumulative-mean context branch.
    :param layer_scale_init: Initial LayerScale gamma value.
    :param stochastic_depth_rate: Maximum DropPath rate (linear schedule).
    :param dropout_rate: Embedding and pre-output dropout rate.
    :param use_bias: Whether Dense/projection layers use bias.
    :param kernel_initializer: Kernel initializer for all dense layers.
    :param bias_initializer: Bias initializer for all dense layers.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    :param routing_mode: ``"trainable"`` (default) or ``"deterministic"``.

    Example:
        .. code-block:: python

            model = CliffordNetLMRouting.from_variant(
                "nano", vocab_size=50261, routing_mode="trainable",
            )
            ids = keras.random.uniform((2, 64), 0, 50261, dtype="int32")
            out = model(ids)
            print(out["logits"].shape)  # (2, 64, 50261)  -- values are probs
    """

    LAYERNORM_EPSILON: float = 1e-6

    # Reused 1:1 from CliffordNetLM.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            channels=128,
            depth=12,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.05,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "mini": dict(
            channels=192,
            depth=12,
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            channels=384,
            depth=18,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.15,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "large": dict(
            channels=512,
            depth=20,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.2,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "xl": dict(
            channels=768,
            depth=28,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.25,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        channels: int = 128,
        depth: int = 12,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.1,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        routing_mode: str = "trainable",
        input_embedding: str = "hce",
        embedding_bottleneck_dim: Optional[int] = None,
        hce_num_chunks: int = 2,
        hce_chunk_bits: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if routing_mode not in _VALID_ROUTING_MODES:
            raise ValueError(
                f"routing_mode must be one of {_VALID_ROUTING_MODES}, "
                f"got: {routing_mode!r}"
            )
        if input_embedding not in _VALID_INPUT_EMBEDDINGS:
            raise ValueError(
                f"input_embedding must be one of {_VALID_INPUT_EMBEDDINGS}, "
                f"got: {input_embedding!r}"
            )

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.channels = channels
        self.depth = depth
        self.shifts = shifts if shifts is not None else [1, 2]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.routing_mode = routing_mode
        self.input_embedding = input_embedding
        self.embedding_bottleneck_dim = embedding_bottleneck_dim
        self.hce_num_chunks = hce_num_chunks
        self.hce_chunk_bits = hce_chunk_bits

        # --- Embeddings ---
        # Token embedding strategy. The token embedding is the single
        # largest parameter sink (vocab * channels) at typical LM scales,
        # so we expose three strategies:
        #
        # - "hce" (default): HierarchicalCodebookEmbedding. Additive
        #   sum of K small codebooks indexed by chunks of the token
        #   ID's bit pattern. Gives ~100x parameter reduction at K=2,
        #   M=2^8=256 for 50K vocab. Embedding manifold is restricted
        #   (Minkowski sum of K finite codebook sets); pairs naturally
        #   with a Huffman/spectral vocab permutation.
        # - "albert": Embedding(vocab, k) -> Dense(channels). ALBERT-
        #   style factorization. Full-rank per-token embedding manifold;
        #   compression ~ channels / k.
        # - "dense": standard keras.layers.Embedding (legacy default).
        #   No compression. Use as baseline.
        #
        # All three return tensors of shape (B, seq, channels), so
        # downstream layers are unaffected. HCE's internal LayerNorm
        # is disabled here because the model already applies
        # ``embed_norm`` LayerNorm after the token+position sum.
        self.token_embedding = self._build_token_embedding()
        self.position_embedding = keras.layers.Embedding(
            max_seq_length, channels, name="position_embedding",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            dropout_rate, name="embed_dropout",
        )

        # --- CliffordNet blocks ---
        drop_rates = _linear_drop_path_rates(depth, stochastic_depth_rate)
        _block_kw: Dict[str, Any] = dict(
            channels=channels,
            shifts=self.shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            layer_scale_init=layer_scale_init,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.clifford_blocks = [
            CausalCliffordNetBlock(
                drop_path_rate=drop_rates[i],
                name=f"clifford_block_{i}",
                **_block_kw,
            )
            for i in range(depth)
        ]

        # --- Output head ---
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="head_norm",
        )
        self.head_dropout = (
            keras.layers.Dropout(dropout_rate, name="head_dropout")
            if dropout_rate > 0.0
            else None
        )
        # DECISION D-002: routing layer replaces Dense(vocab_size). Default
        # routing_mode="trainable" — deterministic mode is too tight for a
        # 50K vocabulary (padded to 65536, only 16 cosine projections to
        # discriminate 50K classes is at the info-theoretic floor).
        # Downstream loss MUST use from_logits=False because this layer
        # outputs probabilities in [eps, 1-eps] summing to 1, not logits.
        # NOTE: RoutingProbabilitiesLayer accepts kernel_initializer,
        # kernel_regularizer, use_bias, bias_initializer (no
        # bias_regularizer — bias_regularizer is intentionally omitted to
        # match the layer's API).
        self.output_routing = RoutingProbabilitiesLayer(
            output_dim=vocab_size,
            mode=routing_mode,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
            name="output_routing",
        )

        logger.info(
            f"Created CliffordNetLMRouting (vocab_size={vocab_size}, "
            f"max_seq_length={max_seq_length}, channels={channels}, "
            f"depth={depth}, shifts={self.shifts}, cli_mode={cli_mode}, "
            f"ctx_mode={ctx_mode}, global_ctx={use_global_context}, "
            f"routing_mode={routing_mode})"
        )

    def _build_token_embedding(self) -> keras.layers.Layer:
        """Construct the token embedding sub-layer per ``input_embedding``."""
        if self.input_embedding == "hce":
            return HierarchicalCodebookEmbedding(
                vocab_size=self.vocab_size,
                output_dim=self.channels,
                num_chunks=self.hce_num_chunks,
                chunk_bits=self.hce_chunk_bits,
                # Disabled: ``embed_norm`` LN is applied right after the
                # token+position sum, so an internal LN here would be
                # redundant.
                use_layer_norm=False,
                embeddings_initializer=self.kernel_initializer,
                name="token_embedding_hce",
            )
        if self.input_embedding == "albert":
            # ALBERT-style factorized embedding:
            # Embedding(vocab, k) -> Dense(channels). Default k =
            # min(channels // 2, 128), giving at least 2x compression at
            # all variant scales.
            k = (
                self.embedding_bottleneck_dim
                if self.embedding_bottleneck_dim is not None
                else max(8, min(self.channels // 2, 128))
            )
            return _AlbertFactorizedEmbedding(
                vocab_size=self.vocab_size,
                bottleneck_dim=k,
                output_dim=self.channels,
                embeddings_initializer=self.kernel_initializer,
                name="token_embedding_albert",
            )
        # "dense" — original behavior.
        return keras.layers.Embedding(
            self.vocab_size,
            self.channels,
            embeddings_initializer=self.kernel_initializer,
            name="token_embedding",
        )

    def call(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        :param input_ids: Token IDs ``(B, seq_len)``.
        :param training: Whether in training mode.
        :return: Dict with ``"logits"`` key: ``(B, seq_len, vocab_size)``.
            Values are probabilities (sum to 1 along last axis), not logits.
        """
        seq_len = keras.ops.shape(input_ids)[1]
        positions = keras.ops.arange(seq_len)

        # Embed tokens + positions
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_norm(x, training=training)
        x = self.embed_dropout(x, training=training)

        # Reshape to 4D: (B, seq_len, D) -> (B, 1, seq_len, D)
        x = keras.ops.expand_dims(x, axis=1)

        # Apply CausalCliffordNet blocks
        for block in self.clifford_blocks:
            x = block(x, training=training)

        # Reshape back to 3D: (B, 1, seq_len, D) -> (B, seq_len, D)
        x = keras.ops.squeeze(x, axis=1)

        # Output projection via hierarchical routing
        x = self.head_norm(x, training=training)
        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)
        # DECISION D-001: output dict key remains "logits" even though
        # values are probabilities. This preserves compatibility with the
        # existing training pipeline data wrapper
        # ((x, y) -> (x, {"logits": y})) and compile spec
        # (loss={"logits": ...}, metrics={"logits": ["accuracy"]}) without
        # forking them. Downstream loss must use from_logits=False.
        logits = self.output_routing(x, training=training)

        return {"logits": logits}

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        return {"logits": (input_shape[0], input_shape[1], self.vocab_size)}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "channels": self.channels,
            "depth": self.depth,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "routing_mode": self.routing_mode,
            "input_embedding": self.input_embedding,
            "embedding_bottleneck_dim": self.embedding_bottleneck_dim,
            "hce_num_chunks": self.hce_num_chunks,
            "hce_chunk_bits": self.hce_chunk_bits,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordNetLMRouting":
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        for key in ("kernel_initializer", "bias_initializer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = initializers.deserialize(config[key])
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        max_seq_length: int = 512,
        routing_mode: str = "trainable",
        **kwargs: Any,
    ) -> "CliffordNetLMRouting":
        """Create a CliffordNetLMRouting from a predefined variant.

        :param variant: One of ``"nano"``, ``"mini"``, ``"base"``,
            ``"large"``, ``"xl"``.
        :param vocab_size: Vocabulary size.
        :param max_seq_length: Maximum sequence length.
        :param routing_mode: ``"trainable"`` (default) or ``"deterministic"``.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNetLMRouting` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(
            f"Creating CliffordNetLMRouting-{variant.upper()} "
            f"(routing_mode={routing_mode})"
        )
        return cls(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            routing_mode=routing_mode,
            **defaults,
        )


# ---------------------------------------------------------------------------
# ALBERT-style factorized token embedding
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class _AlbertFactorizedEmbedding(keras.layers.Layer):
    """ALBERT-style factorized embedding: ``Embedding(vocab, k) @ W``.

    Used internally by :class:`CliffordNetLMRouting` when
    ``input_embedding="albert"``. Reduces token-embedding parameter count
    from ``vocab * channels`` to ``vocab * k + k * channels``. Each
    token's embedding can independently occupy any direction in the
    k-dim subspace projected to channels — full rank per token, unlike
    HCE's restricted Minkowski-sum manifold.

    Reference:
        Lan, Z., et al. (2019). ALBERT: A Lite BERT for Self-supervised
        Learning of Language Representations. arXiv:1909.11942.
    """

    def __init__(
        self,
        vocab_size: int,
        bottleneck_dim: int,
        output_dim: int,
        embeddings_initializer: Any = "uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if vocab_size <= 1:
            raise ValueError(f"vocab_size must be > 1, got {vocab_size}")
        if bottleneck_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"bottleneck_dim and output_dim must be positive, "
                f"got {bottleneck_dim}, {output_dim}"
            )
        self.vocab_size = vocab_size
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)

        self.inner_embedding = keras.layers.Embedding(
            vocab_size,
            bottleneck_dim,
            embeddings_initializer=self.embeddings_initializer,
            name="inner",
        )
        self.proj = keras.layers.Dense(
            output_dim,
            use_bias=False,
            kernel_initializer=self.embeddings_initializer,
            name="proj",
        )

        n_params = vocab_size * bottleneck_dim + bottleneck_dim * output_dim
        n_dense = vocab_size * output_dim
        logger.info(
            f"_AlbertFactorizedEmbedding(vocab={vocab_size}, "
            f"k={bottleneck_dim}, D={output_dim}): {n_params:,} params "
            f"(~{n_dense / max(1, n_params):.1f}x smaller than "
            f"Embedding({vocab_size},{output_dim})={n_dense:,} params)"
        )

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        return self.proj(self.inner_embedding(inputs))

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "bottleneck_dim": self.bottleneck_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": initializers.serialize(
                self.embeddings_initializer,
            ),
        })
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any],
    ) -> "_AlbertFactorizedEmbedding":
        if config.get("embeddings_initializer") and isinstance(
            config["embeddings_initializer"], dict,
        ):
            config["embeddings_initializer"] = initializers.deserialize(
                config["embeddings_initializer"],
            )
        return cls(**config)
