"""
WaveFieldLLM, a decoder-only causal language model with wave-field attention.

This file holds ``WaveFieldDecoderBlock``, the ``WaveFieldLLM`` model with its
``MODEL_VARIANTS`` table, and the ``create_wave_field_llm`` factory. In place
of an N x N attention matrix, each token deposits its value onto a 1-D field
of ``field_size`` cells, weighted by its key magnitude and split bilinearly
across the two nearest cells. The field is convolved by FFT with a per-head
damped-wave kernel

    k(t) = exp(-alpha*t) * cos(omega*t + phi)

then a learned matrix mixes heads at each grid position and every token
gathers back from its own position, so cost is ``O(N*D + G log G * H * D_h)``
rather than ``O(N^2 * D)``. Two gates restore content dependence: a sigmoid on
the query scales the gathered field per token, and a projection of the block
input gates the block output. Around that sits a standard GPT-2-style pre-norm
stack with learned token and position embeddings and a weight-tied LM head by
default. ``call`` returns ``{"logits", "last_hidden_state"}``. No causal mask
is built anywhere in this module, so causality is a measured property of the
``(field_size, max_seq_len)`` pair rather than a guarantee; see
``WaveFieldLLM`` for the numbers. No pretrained weights ship with this
package, and ``pretrained=True`` raises ``NotImplementedError``.

References:
    - Radford et al., 2019. Language Models are Unsupervised Multitask Learners.
      (GPT-2 reference architecture; OpenAI technical report, no arXiv id)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Gu et al., 2022. Efficiently Modeling Long Sequences with Structured State
      Spaces. (https://arxiv.org/abs/2111.00396)
    - Poli et al., 2023. Hyena Hierarchy: Towards Larger Convolutional Language
      Models. (https://arxiv.org/abs/2302.10866)
    - Xiong et al., 2020. On Layer Normalization in the Transformer Architecture.
      (https://arxiv.org/abs/2002.04745)
    - Press & Wolf, 2017. Using the Output Embedding to Improve Language Models.
      (https://arxiv.org/abs/1608.05859)
    - Cooley & Tukey, 1965. An Algorithm for the Machine Calculation of Complex
      Fourier Series. Mathematics of Computation 19(90).
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_or_raise
from dl_techniques.utils.tied_embeddings import tied_embedding_logits
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.attention.wave_field_attention import (
    WaveFieldAttention,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.wave_field.model")
class WaveFieldDecoderBlock(keras.layers.Layer):
    """Apply pre-norm wave-field attention and an FFN, each with a residual.

    Block internals:

    .. code-block:: text

        input [B, N, D]
              │
              ├────────────────────────────────┐
              ▼                                │
        ┌──────────────────────────────┐       │
        │ layer norm                   │       │
        └──────────────────────────────┘       │
              │                                │
              ▼                                │
        ┌──────────────────────────────┐       │
        │ wave field attention         │◄── padding mask (optional)
        └──────────────────────────────┘       │
              │                                │
             (+)◄──────────────────────────────┘
              │
              ├────────────────────────────────┐
              ▼                                │
        ┌──────────────────────────────┐       │
        │ layer norm                   │       │
        └──────────────────────────────┘       │
              │                                │
              ▼                                │
        ┌──────────────────────────────┐       │
        │ dense to ffn width, gelu     │       │
        │ dense back to D              │       │
        │ dropout                      │       │
        └──────────────────────────────┘       │
              │                                │
             (+)◄──────────────────────────────┘
              │
              ▼
        output [B, N, D]

    ``attention_dropout_rate`` applies inside the attention sub-layer, so it
    does not appear as a stage here.

    No causal mask is built in this block. The only mask it forwards to
    attention is the optional padding mask ``(B, N)``, so token-level
    causality is whatever :class:`WaveFieldAttention` provides, which is a
    measured property of ``(field_size, max_seq_len)`` rather than a
    guarantee — see :class:`WaveFieldLLM` for the measured table.

    :param embed_dim: Hidden dim (must be positive and divisible by
        ``num_heads``).
    :param num_heads: Number of attention heads. Must be positive.
    :param max_seq_len: Maximum sequence length, used by attention to map
        token indices to field cells. Required, and must be positive.
    :param field_size: Wave field grid resolution. Required, and must be
        positive.
    :param ffn_intermediate_size: FFN hidden width. ``None`` resolves to
        ``4 * embed_dim``.
    :param dropout_rate: Dropout on the FFN output.
    :param attention_dropout_rate: Dropout inside the attention sub-layer.
    :param layer_norm_eps: LayerNorm epsilon.
    :param initializer_range: Stddev for TruncatedNormal weight init, shared
        by the attention and FFN kernels.
    :param kwargs: Forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``embed_dim`` or ``num_heads`` is not positive, if
        ``embed_dim`` is not divisible by ``num_heads``, or if ``max_seq_len``
        or ``field_size`` is not positive.
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
        """Build every sub-layer explicitly.

        Keras 3's ``__call__`` wrapper can trace a block's inner call before
        every nested sub-layer has built, and ``WaveFieldAttention`` then
        fails ``add_weight`` with "'NoneType' object has no attribute
        'assign'". Building each sub-layer here, before any call is traced,
        avoids that.

        :param input_shape: Shape of the block input ``(B, seq, embed_dim)``.
        :return: Nothing.
        :rtype: None
        """
        self.attn_norm.build(input_shape)
        self.attention.build(input_shape)

        self.ffn_norm.build(input_shape)
        self.ffn_dense_1.build(input_shape)
        # The second dense reads the widened FFN hidden state.
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
        """Run the attention and FFN halves, each around its residual.

        :param inputs: Block input ``(B, N, embed_dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional padding mask ``(B, N)``, forwarded to
            the attention sub-layer. Not a causal mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: Output tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        h = self.attn_norm(inputs)
        h = self.attention(
            h, attention_mask=attention_mask, training=training,
        )
        x = inputs + h

        h = self.ffn_norm(x)
        h = self.ffn_dense_1(h)
        h = self.ffn_dense_2(h)
        h = self.ffn_dropout(h, training=training)
        return x + h

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Return the input shape, since the block preserves it.

        :param input_shape: Shape of the block input.
        :return: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Config dict holding every constructor argument, with
            ``ffn_intermediate_size`` already resolved.
        :rtype: Dict[str, Any]
        """
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


@register_dl_technique("dl_techniques.models.wave_field.model")
class WaveFieldLLM(keras.Model):
    """Predict next-token logits from a stack of ``WaveFieldDecoderBlock`` layers.

    Mirrors the public surface of :class:`GPT2` so it slots into the same
    training pipeline. It differs in two ways: attention is
    :class:`WaveFieldAttention` (see the module docstring), and it takes a
    ``field_size`` hyperparameter that defaults to ``2 * max_seq_len``.

    Architecture:

    .. code-block:: text

        input_ids [B, N]
              │
              ▼
        ┌──────────────────────────────┐
        │ token embedding              │
        └──────────────────────────────┘
              │ [B, N, D]
              ▼
        ┌──────────────────────────────┐
        │ add learned position table   │
        │ sliced to N                  │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm, then dropout     │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ decoder block x depth        │◄── padding mask (optional)
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ final layer norm             │
        └──────────────────────────────┘
              │
              ├──────────► last_hidden_state [B, N, D]
              ▼
        ┌──────────────────────────────┐
        │ tied: matmul token emb. T    │
        │ else: lm_head dense, no bias │
        └──────────────────────────────┘
              │
              ▼
        logits [B, N, V]

    Both leaves are returned together as
    ``{"logits", "last_hidden_state"}``, so :class:`MaskedCausalLMLoss` and
    the standard CLM data-wrapper that keys on ``"logits"`` work unchanged.

    Causality is not guaranteed. No explicit causal mask is built anywhere
    in this module, and whatever token-level causality exists comes only
    from ``WaveFieldAttention``'s damped-wave kernel, which is causal on
    the field grid but not on tokens: the bilinear scatter and gather can let
    a later token deposit into a cell an earlier token reads from. Whether
    this leaks is a property of the exact ``(field_size, max_seq_len)``
    pair, measured end-to-end (``max_seq_len=32``, ``embed_dim=64``,
    ``depth=2``, one token substituted, worst absolute logit change over
    all positions, against logits of magnitude ~1.1):

    .. code-block:: text

        ratio  field_size  stride   worst leak (CPU / GPU)
        0.50    16         0.4839   5.46e-04 / 5.50e-04    leaks
        0.75    24         0.7419   3.77e-04 / 4.05e-04    leaks
        1.00    32         1.0000   6.71e-08 / below 1e-5  clean
        1.50    48         1.5161   4.96e-05 / 1.24e-04    leaks
        2.00    64         2.0323   5.96e-08 / below 1e-5  clean  (default)
        4.00   128         4.0968   8.94e-08 / below 1e-5  clean

    A "clean" row is float32 noise rather than an exact zero, roughly 40x
    below a "leaks" row. The property is not monotone in the ratio (1.50
    leaks, 1.00 does not), so re-measure after any change to ``field_size``
    or ``max_seq_len`` instead of inferring from the ratio. The default,
    ``field_size = 2 * max_seq_len``, measured clean at every configuration
    tested. The attention layer's own docstring
    (``layers/attention/wave_field_attention.py``) says not to rely on this
    for autoregressive decoding.

    Variants (:data:`MODEL_VARIANTS`):

    .. code-block:: text

        variant  embed_dim  depth  heads  max_seq_len  field_size
        xl       1600       48     25     1024         2048
        large    1280       36     20     1024         2048
        medium   1024       24     16     1024         2048
        small    768        12     12     1024         2048
        tiny     256        4      4      512          1024

    No entry sets ``vocab_size``, so every variant takes the class default.

    :param vocab_size: Vocabulary size. Default 50261 (Tiktoken ``gpt2``
        + 4 special tokens — see DECISION ``D-005``).
    :param embed_dim: Hidden dim. Default 768. Must be divisible by
        ``num_heads``.
    :param depth: Number of decoder blocks. Default 12.
    :param num_heads: Number of attention heads. Default 12.
    :param max_seq_len: Maximum sequence length. Default 1024. Also the
        sequence length every sub-layer is built at.
    :param field_size: Wave field grid resolution, which must exceed 1.
        ``None`` -> ``2 * max_seq_len`` (see DECISION ``D-002``). This value
        and ``max_seq_len`` jointly decide whether the stack leaks future
        tokens; do not change either without re-running the ratio sweep above.
    :param dropout_rate: Dropout for embedding and FFN paths. Default 0.0.
    :param attention_dropout_rate: Dropout on attention output. Default 0.0.
    :param initializer_range: Stddev for TruncatedNormal weight init.
        Default 0.02.
    :param layer_norm_eps: LayerNorm epsilon. Default 1e-5.
    :param tie_word_embeddings: Reuse transposed token embedding as LM head
        (DECISION ``D-003``). Default True, which builds no ``lm_head``.
    :param kwargs: Forwarded to ``keras.Model``.

    :raises ValueError: If ``vocab_size``, ``embed_dim``, ``depth``,
        ``num_heads`` or ``max_seq_len`` is not positive, if ``embed_dim`` is
        not divisible by ``num_heads``, if ``field_size`` is not above 1, or if
        either dropout rate falls outside ``[0, 1]``.
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

    # DECISION plan_2026-05-07_1519e34f/D-005: the class default matches the train
    # script's vocab (tiktoken gpt2 50257 + 4 special), so direct instantiation
    # cannot mismatch it. See decisions.md.
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

        # DECISION plan_2026-05-07_1519e34f/D-002: default to 2 * max_seq_len for
        # sub-cell bilinear precision at modest FFT cost. See decisions.md.
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
        """Check the constructor arguments, raising ``ValueError`` on any failure.

        Takes ``field_size`` already resolved, so the ``None`` default is
        never seen here.

        :return: Nothing.
        :rtype: None
        :raises ValueError: As listed in the class docstring.
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
        """Create every sub-layer and build it at ``max_seq_len``.

        The LM head exists only when ``tie_word_embeddings`` is False.

        :return: Nothing.
        :rtype: None
        """
        kernel_init = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range,
        )

        self.token_embeddings = keras.layers.Embedding(
            self.vocab_size,
            self.embed_dim,
            embeddings_initializer=kernel_init,
            name="token_embeddings",
        )
        # DECISION plan-2026-08-13T091555-230c101d/D-006: keep dropout_rate=0.0
        # here and embed_dropout as a separate post-norm layer; folding them
        # moves the output by up to 38% of signal RMS. See decisions.md.
        self.position_embeddings = create_embedding_layer(
            'positional_learned',
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout_rate=0.0,
            scale=self.initializer_range,
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

        # Built here because WaveFieldAttention's initializer calls
        # keras.random.normal at build time, which symbolic tracing can skip.
        block_input_shape: Tuple[Optional[int], ...] = (
            None, self.max_seq_len, self.embed_dim,
        )
        self.token_embeddings.build((None, self.max_seq_len))
        self.position_embeddings.build(block_input_shape)
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
        """Run the forward pass and return both heads in one dict.

        :param inputs: Token ids ``(B, N)``, or a dict holding ``"input_ids"``
            and optionally ``"attention_mask"``. A mask inside the dict wins
            over the ``attention_mask`` argument.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param attention_mask: Optional padding mask ``(B, N)``, forwarded to
            every block. Nothing here adds a causal mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: ``{"logits": (B, N, vocab_size), "last_hidden_state":
            (B, N, embed_dim)}``, the logits with no softmax applied.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``inputs`` is a dict without ``"input_ids"``.
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

        # The position layer slices its table to the incoming seq_len and adds
        # it to the token embeddings; its own dropout is off, see D-006.
        x = self.position_embeddings(
            self.token_embeddings(input_ids), training=training,
        )

        x = self.embed_norm(x)
        x = self.embed_dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x, attention_mask=attention_mask, training=training,
            )

        x = self.final_norm(x)

        if self.tie_word_embeddings:
            embedding_weights = self.token_embeddings.embeddings
            logits = tied_embedding_logits(x, embedding_weights)
        else:
            logits = self.lm_head(x)

        return {
            "logits": logits,
            "last_hidden_state": x,
        }

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute both output shapes from the token-id shape.

        :param input_shape: Token id shape ``(B, N)``.
        :return: A dict mirroring ``call``'s keys.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        return {
            "logits": (*input_shape, self.vocab_size),
            "last_hidden_state": (*input_shape, self.embed_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Config dict holding every constructor argument, with
            ``field_size`` already resolved.
        :rtype: Dict[str, Any]
        """
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

    # DECISION plan-2026-08-13T091555-230c101d/D-005: raise here; never warn and
    # return a random-init model for pretrained=True. See decisions.md.
    @staticmethod
    def _download_weights(
        variant: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Refuse to resolve pretrained weights for ``variant``.

        No public WaveFieldLLM weights are distributed with ``dl_techniques``,
        so this always raises. It mirrors the BERT and GPT-2 factory recipe and
        gives an explicit failure in place of a silent random-init fallback.

        :param variant: Variant name (unused).
        :type variant: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :return: Never returns, despite the return annotation.
        :rtype: str
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Pretrained WaveFieldLLM weights are not distributed with "
            "dl_techniques. Pass pretrained=<local_path> to load a local "
            "checkpoint, or pretrained=False to random-init."
        )

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
        :param pretrained: If ``True``, raises ``NotImplementedError`` (no
            public WaveFieldLLM weights are distributed by this library). If a
            string path is supplied, the model is built by a dummy forward pass
            of 32 random token ids and weights are loaded with
            ``skip_mismatch=True``. If ``False`` (default), returns a
            random-initialized model.
        :param kwargs: Override any variant parameter.
        :return: Configured ``WaveFieldLLM`` instance.
        :rtype: WaveFieldLLM
        :raises ValueError: If the variant name is not recognized.
        :raises NotImplementedError: If ``pretrained=True``.
        :raises FileNotFoundError: If ``pretrained`` is a path that does not
            exist.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        # Copied before the update, so an override cannot poison the table.
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
                # DECISION plan-2026-08-14T233721-d4f9beb2/D-070: load_weights_or_raise,
                # since load_weights no-ops on a non-matching checkpoint.
                # See decisions.md.
                load_weights_or_raise(model, weights_path, skip_mismatch=True)
            else:
                # DECISION plan-2026-08-13T091555-230c101d/D-005: catch I/O errors
                # only; Exception would swallow the NotImplementedError below.
                # See decisions.md.
                try:
                    resolved_path = cls._download_weights(variant)
                except (IOError, OSError, ValueError) as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {e}. "
                        f"Continuing with random initialization."
                    )
                    resolved_path = None
                if resolved_path is not None:
                    if not model.built:
                        import numpy as np
                        dummy = np.random.randint(
                            0, model.vocab_size, (1, 32),
                        ).astype("int32")
                        model(dummy, training=False)
                    # DECISION plan-2026-08-14T233721-d4f9beb2/D-070: same guard.
                    load_weights_or_raise(model, resolved_path, skip_mismatch=True)

        return model

# ---------------------------------------------------------------------
# Module-level Factory
# ---------------------------------------------------------------------


def create_wave_field_llm(
    variant: str = "small",
    vocab_size: Optional[int] = None,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> "WaveFieldLLM":
    """Convenience factory that mirrors ``create_bert`` / ``create_gpt2``.

    Thin wrapper around :meth:`WaveFieldLLM.from_variant` exposing the most
    common construction arguments at module level. Behaves identically to
    calling ``WaveFieldLLM.from_variant(...)`` directly.

    :param variant: Variant name (``"tiny"``, ``"small"``, ``"medium"``,
        ``"large"``, ``"xl"``). Defaults to ``"small"``.
    :type variant: str
    :param vocab_size: Optional vocabulary size override. No variant entry
        sets one, so ``None`` (the default) leaves the class default of
        ``DEFAULT_VOCAB_SIZE`` in place. A value is forwarded as
        ``vocab_size=...`` in ``kwargs``.
    :type vocab_size: Optional[int]
    :param pretrained: If ``True``, raises ``NotImplementedError`` — no public
        WaveFieldLLM weights are distributed by this library. If a string path,
        loads local weights from that path. If ``False`` (default), random
        init.
    :type pretrained: Union[bool, str]
    :param kwargs: Additional keyword arguments forwarded to
        :meth:`WaveFieldLLM.from_variant` (e.g. ``dropout_rate``,
        ``tie_word_embeddings``).
    :type kwargs: Any

    :returns: Configured ``WaveFieldLLM`` instance.
    :rtype: WaveFieldLLM

    :raises NotImplementedError: If ``pretrained=True`` (no public weights).
    :raises FileNotFoundError: If ``pretrained`` is a string path that does
        not exist.
    :raises ValueError: If ``variant`` is not a recognized variant.

    Example:
        >>> model = create_wave_field_llm("small")
        >>> model = create_wave_field_llm("tiny", vocab_size=200)
    """
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size
    return WaveFieldLLM.from_variant(
        variant,
        pretrained=pretrained,
        **kwargs,
    )

# ---------------------------------------------------------------------
