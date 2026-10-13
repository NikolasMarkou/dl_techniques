"""
The sequence encoder of the Neural HMM: one layer, :class:`NeuralHMMEncoder`, that maps an
observation sequence ``(B, T, F)`` to a per-step context ``h`` of shape ``(B, T, hidden_dim)``.
The transition and emission layers read ``h``.

Architecture:

.. code-block:: text

    inputs (B, T, F)          mask (B, T), optional; True / nonzero = valid step
         │                          │
         ▼                          │
    padded steps set to 0 ◄─────────┤   (padding content can never reach the output)
         │                          │
         ▼                          │
    Dense(hidden_dim)  "input_projection"
         │                          │
         ▼                          │
    one path, chosen by encoder_type ◄──┘
      mlp          num_layers x create_ffn_layer("residual")       per step
      transformer  positional_learned embedding
                   + num_layers x TransformerLayer (pre-norm, multi_head attention,
                     causal: create_causal_attend_mask(x, mask); else the (B, T) keep-mask)
      lstm / gru   num_layers x [Dropout -> keras LSTM / GRU, with the boolean mask]
                   (wrapped in Bidirectional(merge_mode="sum") when not causal)
      conv         num_layers x [padded steps set to 0 -> Conv1D(padding="causal" | "same")
                     -> Dropout] + residual
         │
         ▼
    create_normalization_layer(normalization_type)  "output_norm"
         │
         ▼
    h (B, T, hidden_dim), in the compute dtype

**Causality.** With ``causal=True`` the output at step ``t`` depends on ``x_{<=t}`` only.
The model shifts its input right by one step before the encoder, so the emission of ``x_t``
reads a context built from ``x_{<t}`` (Tran et al., 2016). The ``mlp`` path is per-step
and therefore causal under either setting.

**Padding.** A mask marks a contiguous valid prefix (right padding). Padded inputs are
replaced by zeros before anything reads them, so the content of a padded step never reaches
any output. The valid outputs of a padded sequence also equal those of the truncated
sequence: the recurrent paths skip masked steps, the attention paths never attend to a
padded key, and the ``conv`` path zeroes padded steps before every convolution, so a
``"same"`` kernel sees there the same zeros it sees beyond the end of the sequence. Outputs
at padded steps are unspecified.

**Length limit.** The ``transformer`` path adds a learned position table of ``max_seq_len``
rows: a statically known ``T > max_seq_len`` is refused at build time, and a longer dynamic
``T`` fails inside the table slice. The other paths have no length limit.

References:
    - Tran, Bisk, Vaswani, Marcu and Knight, 2016. Unsupervised Neural Hidden Markov
      Models. Proceedings of the Workshop on Structured Prediction for NLP.
      arXiv:1609.09007.
    - Vaswani et al., 2017. Attention Is All You Need. NeurIPS. arXiv:1706.03762.
"""

from typing import Any, List, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.embedding.factory import create_embedding_layer
from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.transformers.transformer import TransformerLayer
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.masking.factory import create_causal_attend_mask

# ---------------------------------------------------------------------

#: Encoder types :class:`NeuralHMMEncoder` accepts.
ENCODER_TYPES: Tuple[str, ...] = ("mlp", "transformer", "lstm", "gru", "conv")

#: Activation of each ``conv`` block.
_CONV_ACTIVATION: str = "gelu"

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.neural_hmm.encoders")
class NeuralHMMEncoder(keras.layers.Layer):
    """Sequence encoder ``(B, T, F) -> (B, T, hidden_dim)`` with a padding mask.

    Each argument is read by the paths named beside it; the other paths ignore it.

    :param encoder_type: One of ``mlp``, ``transformer``, ``lstm``, ``gru``, ``conv``.
    :param hidden_dim: Output width, and the width of every path; positive. A multiple of
        ``num_heads`` for ``transformer``.
    :param num_layers: Number of blocks (``mlp``, ``transformer``, ``conv``) or stacked
        recurrent layers (``lstm``, ``gru``); positive.
    :param causal: When ``True``, the output at step ``t`` reads ``x_{<=t}`` only
        (``transformer``, ``lstm``, ``gru``, ``conv``; ``mlp`` is always per-step).
    :param num_heads: Attention heads (``transformer``); positive.
    :param intermediate_size: Hidden width of each feed-forward block (``mlp``,
        ``transformer``); positive.
    :param max_seq_len: Rows of the learned position table (``transformer``); positive.
    :param kernel_size: Convolution width (``conv``); positive.
    :param ffn_type: :func:`create_ffn_layer` type of the transformer feed-forward block
        (``transformer``).
    :param normalization_type: :func:`create_normalization_layer` type of the output norm
        (every path) and of the pre-norms (``transformer``).
    :param dropout_rate: Dropout in ``[0, 1)``: inside each residual block (``mlp``), after
        the position embedding and each feed-forward block (``transformer``), on the inputs of each recurrent layer,
        one mask per sequence (``lstm``, ``gru``), after each convolution (``conv``).
    :param attention_dropout_rate: Dropout in ``[0, 1)`` on the attention probabilities
        (``transformer``).
    :param kwargs: Base ``keras.layers.Layer`` arguments.
    :raises ValueError: If an argument is out of range, or ``hidden_dim`` is not a multiple
        of ``num_heads`` for ``transformer``.
    """

    def __init__(
        self,
        encoder_type: str,
        hidden_dim: int = 64,
        num_layers: int = 2,
        causal: bool = True,
        num_heads: int = 4,
        intermediate_size: int = 128,
        max_seq_len: int = 512,
        kernel_size: int = 3,
        ffn_type: str = "mlp",
        normalization_type: str = "layer_norm",
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if encoder_type not in ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type '{encoder_type}'. Available: {list(ENCODER_TYPES)}.")
        for name, value in (
            ("hidden_dim", hidden_dim), ("num_layers", num_layers), ("num_heads", num_heads),
            ("intermediate_size", intermediate_size), ("max_seq_len", max_seq_len),
            ("kernel_size", kernel_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        for name, value in (("dropout_rate", dropout_rate), ("attention_dropout_rate", attention_dropout_rate)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")
        if encoder_type == "transformer" and hidden_dim % num_heads:
            raise ValueError(
                f"encoder_type='transformer' needs hidden_dim divisible by num_heads, "
                f"got hidden_dim={hidden_dim}, num_heads={num_heads}."
            )

        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.causal = causal
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.kernel_size = kernel_size
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.input_projection = keras.layers.Dense(hidden_dim, name="input_projection")
        self.positional_embedding = None
        self.blocks: List[keras.layers.Layer] = []
        self.block_dropouts: List[keras.layers.Layer] = []

        if encoder_type == "mlp":
            self.blocks = [
                create_ffn_layer(
                    "residual",
                    name=f"mlp_block_{i}",
                    hidden_dim=intermediate_size,
                    output_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                )
                for i in range(num_layers)
            ]
        elif encoder_type == "transformer":
            self.positional_embedding = create_embedding_layer(
                "positional_learned",
                name="positional_embedding",
                max_seq_len=max_seq_len,
                dim=hidden_dim,
                dropout_rate=dropout_rate,
            )
            self.blocks = [
                TransformerLayer(
                    hidden_size=hidden_dim,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    attention_type="multi_head",
                    normalization_type=normalization_type,
                    normalization_position="pre",
                    ffn_type=ffn_type,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    name=f"transformer_layer_{i}",
                )
                for i in range(num_layers)
            ]
        elif encoder_type in ("lstm", "gru"):
            rnn_class = keras.layers.LSTM if encoder_type == "lstm" else keras.layers.GRU
            for i in range(num_layers):
                # DECISION plan-2026-10-13T192257-1bb3bf1f/D-016: the input dropout is a
                # separate Dropout layer. Do NOT pass dropout= to the LSTM/GRU: Keras 3.8's
                # cuDNN path applies that mask whatever `training` is, so on a GPU the
                # recurrent layer drops inputs at inference (measured on GPU 1: 0.55 for
                # LSTM, 1.15 for GRU, max |out(dropout=0.5) - out(dropout=0)| at
                # training=False; 0.0 on CPU). The noise shape keeps one mask per sequence,
                # as the recurrent layer's own dropout does.
                rnn = rnn_class(hidden_dim, return_sequences=True, name=f"{encoder_type}_{i}")
                if not causal:
                    rnn = keras.layers.Bidirectional(rnn, merge_mode="sum", name=f"bidirectional_{encoder_type}_{i}")
                self.blocks.append(rnn)
            self.block_dropouts = [
                keras.layers.Dropout(dropout_rate, noise_shape=(None, 1, hidden_dim), name=f"{encoder_type}_dropout_{i}")
                for i in range(num_layers)
            ]
        else:
            self.blocks = [
                keras.layers.Conv1D(
                    hidden_dim,
                    kernel_size,
                    padding="causal" if causal else "same",
                    activation=_CONV_ACTIVATION,
                    name=f"conv_{i}",
                )
                for i in range(num_layers)
            ]
            self.block_dropouts = [keras.layers.Dropout(dropout_rate, name=f"conv_dropout_{i}") for i in range(num_layers)]

        self.output_norm = create_normalization_layer(normalization_type, name="output_norm")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sublayer in computational order.

        :param input_shape: ``(B, T, F)``.
        :raises ValueError: If the input is not rank 3, or (``transformer``) its static
            ``T`` exceeds ``max_seq_len``.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected inputs of shape (batch, time, features), got {tuple(input_shape)}.")
        self.input_projection.build(input_shape)
        hidden_shape = (input_shape[0], input_shape[1], self.hidden_dim)
        if self.positional_embedding is not None:
            self.positional_embedding.build(hidden_shape)
        for block in self.blocks:
            block.build(hidden_shape)
        for dropout in self.block_dropouts:
            dropout.build(hidden_shape)
        self.output_norm.build(hidden_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode the sequence.

        :param inputs: ``(B, T, F)`` features.
        :param mask: ``(B, T)`` valid-step mask (bool, or numeric with nonzero = valid), a
            contiguous valid prefix; ``None`` when every step is valid.
        :param training: Forwarded to every sublayer.
        :return: ``(B, T, hidden_dim)`` in the compute dtype.
        """
        keep = None if mask is None else keras.ops.cast(mask, "bool")
        x = self._zero_padded(inputs, keep)
        x = self.input_projection(x, training=training)

        if self.encoder_type == "mlp":
            for block in self.blocks:
                x = block(x, training=training)
        elif self.encoder_type == "transformer":
            x = self.positional_embedding(x, training=training)
            if self.causal:
                attention_mask = create_causal_attend_mask(
                    x, None if keep is None else keras.ops.cast(keep, "int32")
                )
            else:
                attention_mask = None if keep is None else keras.ops.cast(keep, "int32")
            for block in self.blocks:
                x = block(x, attention_mask=attention_mask, training=training)
        elif self.encoder_type in ("lstm", "gru"):
            rnn_keep = self._without_empty_rows(keep)
            for rnn, dropout in zip(self.blocks, self.block_dropouts):
                x = rnn(dropout(x, training=training), mask=rnn_keep, training=training)
        else:
            # DECISION plan-2026-10-13T192257-1bb3bf1f/D-017: zero padded steps before EVERY
            # convolution, not only at the input. Do NOT drop this as redundant: after the
            # input projection a padded step holds the Dense bias, and a "same" kernel at the
            # last valid steps would read it, so valid outputs would depend on the padding
            # length and differ from the truncated sequence (arm e fails for conv-noncausal).
            for conv, dropout in zip(self.blocks, self.block_dropouts):
                x = x + dropout(conv(self._zero_padded(x, keep)), training=training)

        return self.output_norm(x, training=training)

    @staticmethod
    def _without_empty_rows(keep: Optional[keras.KerasTensor]) -> Optional[keras.KerasTensor]:
        """Mark step 0 valid in every row that has no valid step.

        :param keep: ``(B, T)`` boolean valid-step mask, or ``None``.
        :return: The mask with each all-False row given a valid first step.
        """
        if keep is None:
            return None
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-016: a row with no valid step runs one
        # step over its zeroed input. Do NOT pass the empty row through: Keras 3.8's cuDNN
        # LSTM/GRU asserts that no row is fully masked, which fails a graph-mode call
        # (fit, predict, tf.function) on a GPU with InvalidArgumentError (eager mode falls
        # back silently; CPU never checks). Every output of such a row is padding.
        first_step = keras.ops.arange(keras.ops.shape(keep)[1]) == 0
        empty = keras.ops.logical_not(keras.ops.any(keep, axis=1, keepdims=True))
        return keras.ops.logical_or(keep, keras.ops.logical_and(empty, first_step[None, :]))

    @staticmethod
    def _zero_padded(x: keras.KerasTensor, keep: Optional[keras.KerasTensor]) -> keras.KerasTensor:
        """Replace padded steps by zeros.

        :param x: ``(B, T, C)``.
        :param keep: ``(B, T)`` boolean valid-step mask, or ``None``.
        :return: ``x`` with every padded step set to 0.
        """
        if keep is None:
            return x
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-017: select, do NOT multiply by the
        # mask: 0 * nan is nan, and a NaN in padded content then reaches VALID transformer
        # outputs through the attention-weighted values (0 weight * nan value).
        return keras.ops.where(keep[:, :, None], x, keras.ops.zeros_like(x))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return ``(B, T, hidden_dim)`` from the stored config.

        :param input_shape: ``(B, T, F)``.
        :return: ``(B, T, hidden_dim)``.
        """
        return input_shape[0], input_shape[1], self.hidden_dim

    def get_config(self) -> dict:
        """Return every constructor argument.

        :return: The layer config.
        """
        config = super().get_config()
        config.update(
            {
                "encoder_type": self.encoder_type,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "causal": self.causal,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "max_seq_len": self.max_seq_len,
                "kernel_size": self.kernel_size,
                "ffn_type": self.ffn_type,
                "normalization_type": self.normalization_type,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
            }
        )
        return config
