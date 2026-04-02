"""GPT-2 (Generative Pre-trained Transformer 2) model.

A decoder-only transformer language model based on the architecture from
"Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).

This implementation reuses the library's ``TextDecoder`` layer for the core
transformer stack (token/positional embeddings, causal self-attention blocks,
and final normalization) and adds a weight-tied language modeling head on top.

Architecture:

.. code-block:: text

    Input IDs (B, seq_len)
         │
         ▼
    TextDecoder
    ├─ Token Embedding + Positional Embedding
    ├─ Embed Norm → Embed Dropout
    ├─ TransformerLayer × depth (pre-norm, causal self-attention + MLP)
    └─ Final LayerNorm
         │
         ▼
    Hidden States (B, seq_len, embed_dim)
         │
         ▼
    LM Head: logits = hidden_states @ token_embedding.T  (weight tying)
         │
         ▼
    Logits (B, seq_len, vocab_size)

Key differences from BERT:
- Pre-layer normalization (not post-norm)
- Causal (autoregressive) masking
- Weight tying between token embeddings and output projection
- No token type embeddings

References:
    Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I.
    (2019). Language Models are Unsupervised Multitask Learners.
"""

import os
import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.layers.transformers.text_decoder import TextDecoder
from dl_techniques.utils.logger import logger


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
        Default: 0.1.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate for attention weights.
        Default: 0.1.
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
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPS,
        attention_type: str = "multi_head",
        ffn_type: str = "mlp",
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

        # Build architecture
        self._build_architecture()

        logger.info(
            f"Created GPT-2: {self.depth} layers, "
            f"embed_dim={self.embed_dim}, heads={self.num_heads}, "
            f"max_seq_len={self.max_seq_len}"
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

        # Weight-tied LM head: logits = hidden_states @ embedding_weights.T
        embedding_weights = self.decoder.word_embeddings.embeddings
        logits = ops.matmul(hidden_states, ops.transpose(embedding_weights))

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
        })
        return config

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
        :param pretrained: If True, loads pretrained weights from default URL.
            If string, treats it as a path to local weights file.
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
                logger.warning(
                    "pretrained=True but no weights URL configured. "
                    "Model initialized with random weights."
                )

        return model
