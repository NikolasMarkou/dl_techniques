"""`EntropyModel`, the causal transformer that scores each byte's next-byte
surprise for the Byte Latent Transformer (BLT), using the shared
`create_causal_attend_mask` helper from `dl_techniques.utils.masking`.

Its `call()` returns next-byte logits; `compute_entropy` turns those logits
into a per-position Shannon entropy in nats, which `DynamicPatcher` thresholds
to place patch boundaries. Every call site hands it an explicit
`create_causal_attend_mask` — the attention layer masks only with what it is
given.

References:
    - Pagnoni et al., 2024. Byte Latent Transformer: Patches Scale Better
      Than Tokens. (https://arxiv.org/abs/2412.09871)
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

from ..transformers.causal_stack import (
    CausalTransformerStackMixin,
    build_causal_stack_norm,
    build_causal_transformer_stack,
)
from ..embedding.positional_embedding import PositionalEmbedding

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.entropy_model")
class EntropyModel(CausalTransformerStackMixin, keras.layers.Layer):
    """Predict next-byte logits with a small causal transformer.

    ``call`` returns logits. Their Shannon entropy comes from a separate
    ``compute_entropy`` call, and that entropy is what ``DynamicPatcher``
    thresholds to place patch boundaries.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]
              │
              ▼
        ┌──────────────────────────────┐
        │ token embedding              │
        └──────────────────────────────┘
              │ [B, S, H]
              ▼
        ┌──────────────────────────────┐
        │ positional embedding         │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ transformer layer x N        │◄── causal attend mask
        │ ffn width 4H                 │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dense to vocab_size          │
        └──────────────────────────────┘
              │
              ▼
        logits [B, S, V]
              │
              ▼
        compute_entropy ──► [B, S] in nats

    :param vocab_size: Size of the byte vocabulary.
    :type vocab_size: int
    :param hidden_dim: Hidden dimension of the transformer.
    :type hidden_dim: int
    :param num_layers: Number of transformer layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param dropout_rate: Dropout rate, shared by the positional embedding and
        the transformer layers.
    :type dropout_rate: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            hidden_dim: int = 256,
            num_layers: int = 6,
            num_heads: int = 8,
            max_seq_len: int = 2048,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_dim,
            name='token_embedding'
        )

        self.bind_causal_stack(
            positional_embedding=PositionalEmbedding(
                max_seq_len=self.max_seq_len,
                dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                name='positional_embedding'
            ),
            layers=build_causal_transformer_stack(
                hidden_size=self.hidden_dim,
                num_heads=self.num_heads,
                depth=self.num_layers,
                dropout_rate=self.dropout_rate,
                name_prefix='transformer_layer',
            ),
            final_norm=build_causal_stack_norm('final_layer_norm'),
        )

        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name='output_projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the entropy model layers.

        :param input_shape: Token shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Explicit builds, because a lazy first-call build leaves the weights
        # unloadable on a .keras reload.
        self.embedding.build(input_shape)
        norm_shape = self.build_causal_stack(
            self.embedding.compute_output_shape(input_shape)
        )
        self.output_projection.build(norm_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the entropy model forward.

        :param inputs: Input token tensor, shape ``(batch_size, seq_len)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Logits, shape ``(batch_size, seq_len, vocab_size)``.
        :rtype: keras.KerasTensor
        """
        x = self.embedding(inputs)

        # Without the mask, the surprise at position i is computed from a state
        # that has already read byte i+1.
        x = self.run_causal_stack(x, training=training)

        return self.output_projection(x)

    def compute_entropy(self, logits: keras.KerasTensor) -> keras.KerasTensor:
        """Compute Shannon entropy ``H = -sum(p * log(p))`` from logits.

        Probabilities are floored at 1e-12 before the log, so the result stays
        finite and its ceiling is ``ln(vocab_size)``.

        :param logits: Logits, shape ``(batch_size, seq_len, vocab_size)``.
        :type logits: keras.KerasTensor
        :return: Entropy in nats, shape ``(batch_size, seq_len)``.
        :rtype: keras.KerasTensor
        """
        probs = keras.activations.softmax(logits, axis=-1)

        log_probs = ops.log(ops.maximum(probs, 1e-12))

        entropy = -ops.sum(probs * log_probs, axis=-1)

        return entropy

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Token shape ``(batch, seq_len)`` as a tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Input shape with ``vocab_size`` appended.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape + (self.vocab_size,)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate
        })
        return config
