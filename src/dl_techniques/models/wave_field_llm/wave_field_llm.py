"""WaveFieldLLM — decoder-only LM with WaveFieldAttention.

A decoder-only causal language model that drops :class:`WaveFieldAttention`
into a GPT-2-style pre-norm transformer stack in place of dot-product
multi-head attention. The architecture mirrors :class:`GPT2` (token + learned
positional embeddings, pre-norm transformer blocks, weight-tied LM head) but
assembles its own decoder block locally (``WaveFieldDecoderBlock``) so it can
forward an attention mask of shape ``(B, N)`` — the shape WaveFieldAttention
expects — without touching the shared attention factory or
``TransformerLayer`` (which expect ``(B, N, N)``).

Architecture::

    Input IDs (B, N)
         │
         ▼
    Token Embedding + Positional Embedding
         │
         ▼
    Embed LayerNorm + Dropout
         │
         ▼
    WaveFieldDecoderBlock × depth
       │
       ├─ pre-norm
       ├─ WaveFieldAttention   (causal by construction)
       ├─ residual
       ├─ pre-norm
       ├─ FFN: Dense(4D, gelu) -> Dense(D) -> Dropout
       └─ residual
         │
         ▼
    Final LayerNorm
         │
         ▼
    LM Head: tied -> hidden @ E.T  |  untied -> Dense(vocab, no bias)
         │
         ▼
    {"logits": (B, N, V), "last_hidden_state": (B, N, D)}

Causality is provided by :class:`WaveFieldAttention` (its left-aligned damped
wave kernel); no explicit causal mask is constructed in this module.

References:
    - Radford et al., "Language Models are Unsupervised Multitask Learners",
      2019 (GPT-2 reference architecture).
    - Internal: ``src/dl_techniques/layers/attention/wave_field_attention.py``.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention.wave_field_attention import (
    WaveFieldAttention,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WaveFieldDecoderBlock(keras.layers.Layer):
    """Pre-norm transformer decoder block with :class:`WaveFieldAttention`.

    Sub-layers (pre-norm GPT-2 style):

    1. ``attn_norm`` -> :class:`WaveFieldAttention` -> residual
    2. ``ffn_norm``  -> Dense(4D, gelu) -> Dense(D) -> Dropout -> residual

    Causality is provided by the wave-field kernel (left-aligned, damped),
    so no explicit causal mask is built. Only an optional padding mask
    ``(B, N)`` is forwarded to attention.

    :param embed_dim: Hidden dim (must be divisible by ``num_heads``).
    :param num_heads: Number of attention heads.
    :param ffn_intermediate_size: FFN hidden width (default ``4 * embed_dim``).
    :param max_seq_len: Maximum sequence length (used by attention to map
        token indices to field cells).
    :param field_size: Wave field grid resolution.
    :param dropout_rate: Dropout on FFN output and embedding pipeline.
    :param attention_dropout_rate: Dropout on attention output.
    :param layer_norm_eps: LayerNorm epsilon.
    :param initializer_range: Stddev for TruncatedNormal weight init.
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        field_size: int,
        ffn_intermediate_size: Optional[int] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be positive, got "
                f"embed_dim={embed_dim}, num_heads={num_heads}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if max_seq_len <= 0 or field_size <= 0:
            raise ValueError(
                f"max_seq_len and field_size must be positive, got "
                f"max_seq_len={max_seq_len}, field_size={field_size}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.field_size = field_size
        self.ffn_intermediate_size = (
            ffn_intermediate_size
            if ffn_intermediate_size is not None
            else 4 * embed_dim
        )
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        kernel_init = keras.initializers.TruncatedNormal(
            stddev=initializer_range,
        )

        self.attn_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="attn_norm",
        )
        self.attention = WaveFieldAttention(
            dim=embed_dim,
            num_heads=num_heads,
            field_size=field_size,
            max_seq_len=max_seq_len,
            dropout_rate=attention_dropout_rate,
            kernel_initializer=kernel_init,
            name="attention",
        )

        self.ffn_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="ffn_norm",
        )
        self.ffn_dense_1 = keras.layers.Dense(
            self.ffn_intermediate_size,
            activation="gelu",
            kernel_initializer=kernel_init,
            name="ffn_dense_1",
        )
        self.ffn_dense_2 = keras.layers.Dense(
            embed_dim,
            kernel_initializer=kernel_init,
            name="ffn_dense_2",
        )
        self.ffn_dropout = keras.layers.Dropout(
            dropout_rate, name="ffn_dropout",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer so a ``.keras`` reload restores
        weights onto already-built sub-layers (H5).

        Explicit build of WaveFieldAttention is especially required: when the
        block is invoked, Keras 3's ``__call__`` wrapper triggers build for the
        block but does not always reach nested sub-layer build paths before the
        inner call is traced -- so ``add_weight`` inside the attention layer can
        fail with "'NoneType' object has no attribute 'assign'". Building it
        explicitly here pins variable creation to the block's build phase.

        Args:
            input_shape: Shape of the block input ``(B, seq, embed_dim)``.
        """
        # Attention block: pre-norm -> WaveFieldAttention.
        self.attn_norm.build(input_shape)
        self.attention.build(input_shape)

        # FFN block: pre-norm -> dense_1 -> dense_2 -> dropout.
        self.ffn_norm.build(input_shape)
        self.ffn_dense_1.build(input_shape)
        ffn_hidden_shape = tuple(input_shape[:-1]) + (self.ffn_intermediate_size,)
        self.ffn_dense_2.build(ffn_hidden_shape)
        self.ffn_dropout.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # Block 1: pre-norm + WaveFieldAttention + residual.
        h = self.attn_norm(inputs)
        h = self.attention(
            h, attention_mask=attention_mask, training=training,
        )
        x = inputs + h

        # Block 2: pre-norm + FFN + residual.
        h = self.ffn_norm(x)
        h = self.ffn_dense_1(h)
        h = self.ffn_dense_2(h)
        h = self.ffn_dropout(h, training=training)
        return x + h

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "field_size": self.field_size,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WaveFieldLLM(keras.Model):
    """Decoder-only language model with WaveFieldAttention blocks.

    Mirrors the public surface of :class:`GPT2` so it slots into the same
    training pipeline. The two notable differences are:

    1. Attention is :class:`WaveFieldAttention` (FFT damped-wave field) which
       is causal-by-construction; no explicit causal mask is required.
    2. A new hyperparameter ``field_size`` (defaults to ``2 * max_seq_len``,
       see ``DECISION plan_2026-05-07_1519e34f/D-002``).

    Output is a dict ``{"logits", "last_hidden_state"}`` so that
    :class:`MaskedCausalLMLoss` and the standard CLM data-wrapper that keys
    on ``"logits"`` work unchanged.

    :param vocab_size: Vocabulary size. Default 50261 (Tiktoken ``gpt2``
        + 4 special tokens — see DECISION ``D-005``).
    :param embed_dim: Hidden dim. Default 768.
    :param depth: Number of decoder blocks. Default 12.
    :param num_heads: Number of attention heads. Default 12.
    :param max_seq_len: Maximum sequence length. Default 1024.
    :param field_size: Wave field grid resolution. ``None`` -> ``2 * max_seq_len``
        (see DECISION ``D-002``).
    :param dropout_rate: Dropout for embedding and FFN paths. Default 0.0.
    :param attention_dropout_rate: Dropout on attention output. Default 0.0.
    :param initializer_range: Stddev for TruncatedNormal weight init.
        Default 0.02.
    :param layer_norm_eps: LayerNorm epsilon. Default 1e-5.
    :param tie_word_embeddings: Reuse transposed token embedding as LM head
        (DECISION ``D-003``). Default True.
    :param kwargs: Forwarded to ``keras.Model``.
    """

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "xl": {
            "embed_dim": 1600,
            "depth": 48,
            "num_heads": 25,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM XL: ~1.5B parameter class",
        },
        "large": {
            "embed_dim": 1280,
            "depth": 36,
            "num_heads": 20,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM Large: ~774M parameter class",
        },
        "medium": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM Medium: ~355M parameter class",
        },
        "small": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM Small: ~124M parameter class",
        },
        "tiny": {
            "embed_dim": 256,
            "depth": 4,
            "num_heads": 4,
            "max_seq_len": 512,
            "field_size": 1024,
            "description": "WaveFieldLLM Tiny: lightweight for testing",
        },
    }

    # DECISION plan_2026-05-07_1519e34f/D-005 — class default vocab matches
    # train script default (tiktoken `gpt2` 50257 base + 4 special) so no
    # silent vocab mismatch when a user instantiates the class directly.
    DEFAULT_VOCAB_SIZE = 50261
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPS = 1e-5

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024,
        field_size: Optional[int] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPS,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # DECISION plan_2026-05-07_1519e34f/D-002 — default field_size to
        # 2 * max_seq_len: sub-cell bilinear precision at modest FFT cost.
        if field_size is None:
            field_size = 2 * max_seq_len

        self._validate_config(
            vocab_size, embed_dim, depth, num_heads,
            field_size, max_seq_len,
            dropout_rate, attention_dropout_rate,
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.field_size = field_size
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings

        self._build_architecture()

        logger.info(
            f"Created WaveFieldLLM: depth={self.depth}, "
            f"embed_dim={self.embed_dim}, heads={self.num_heads}, "
            f"max_seq_len={self.max_seq_len}, field_size={self.field_size}, "
            f"tie_word_embeddings={self.tie_word_embeddings}"
        )

    @staticmethod
    def _validate_config(
        vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        field_size: int,
        max_seq_len: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ) -> None:
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
        if field_size <= 1:
            raise ValueError(
                f"field_size must be > 1, got {field_size}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {max_seq_len}"
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
        kernel_init = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range,
        )

        self.token_embeddings = keras.layers.Embedding(
            self.vocab_size,
            self.embed_dim,
            embeddings_initializer=kernel_init,
            name="token_embeddings",
        )
        self.position_embeddings = keras.layers.Embedding(
            self.max_seq_len,
            self.embed_dim,
            embeddings_initializer=kernel_init,
            name="position_embeddings",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            self.dropout_rate, name="embed_dropout",
        )

        self.blocks = [
            WaveFieldDecoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                max_seq_len=self.max_seq_len,
                field_size=self.field_size,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                initializer_range=self.initializer_range,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="final_norm",
        )

        if not self.tie_word_embeddings:
            self.lm_head = keras.layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=kernel_init,
                name="lm_head",
            )
        else:
            self.lm_head = None

        # Eagerly build the embedding tables and decoder blocks. The
        # WaveFieldAttention layer constructs its weights via
        # `_IdentityPlusNoise` which calls `keras.random.normal` at build
        # time. Under Keras 3's symbolic call tracing for `keras.Model`
        # subclasses, nested-layer build can be skipped, leading to the
        # initializer firing on every forward pass with no backing
        # variable. Eagerly building here pins variable creation to model
        # construction time, before any tracing occurs.
        block_input_shape: Tuple[Optional[int], ...] = (
            None, self.max_seq_len, self.embed_dim,
        )
        self.token_embeddings.build((None, self.max_seq_len))
        self.position_embeddings.build((self.max_seq_len,))
        self.embed_norm.build(block_input_shape)
        for block in self.blocks:
            block.build(block_input_shape)
        self.final_norm.build(block_input_shape)
        if self.lm_head is not None:
            self.lm_head.build(block_input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        seq_len = ops.shape(input_ids)[1]
        positions = ops.arange(seq_len, dtype="int32")

        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)
        # Broadcast (N, D) over batch dim of (B, N, D).
        x = token_emb + pos_emb

        x = self.embed_norm(x)
        x = self.embed_dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x, attention_mask=attention_mask, training=training,
            )

        x = self.final_norm(x)

        if self.tie_word_embeddings:
            embedding_weights = self.token_embeddings.embeddings
            logits = ops.matmul(x, ops.transpose(embedding_weights))
        else:
            logits = self.lm_head(x)

        return {
            "logits": logits,
            "last_hidden_state": x,
        }

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        return {
            "logits": (*input_shape, self.vocab_size),
            "last_hidden_state": (*input_shape, self.embed_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "field_size": self.field_size,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
        })
        return config

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "WaveFieldLLM":
        """Instantiate from a named variant in :data:`MODEL_VARIANTS`.

        :param variant: Variant name: ``'tiny'``, ``'small'``, ``'medium'``,
            ``'large'``, ``'xl'``.
        :param pretrained: Reserved for parity with :class:`GPT2`; if a
            string path is supplied, the model is built (with a dummy
            forward pass) and weights are loaded with ``skip_mismatch=True``.
        :param kwargs: Override any variant parameter.
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
            import os
            weights_path = pretrained if isinstance(pretrained, str) else None
            if weights_path is not None:
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(
                        f"Weights file not found: {weights_path}"
                    )
                if not model.built:
                    import numpy as np
                    dummy = np.random.randint(
                        0, model.vocab_size, (1, 32),
                    ).astype("int32")
                    model(dummy, training=False)
                model.load_weights(weights_path, skip_mismatch=True)
                logger.info(f"Loaded weights from {weights_path}")
            else:
                logger.warning(
                    "pretrained=True but no weights URL configured. "
                    "Model initialized with random weights."
                )

        return model

# ---------------------------------------------------------------------
