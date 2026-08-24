"""
The Gemma 3 transformer block: sandwich-normalized grouped-query attention and a
GeGLU feed-forward network, with the attention window selected per block.

The block's distinguishing feature is that normalization brackets each sublayer
rather than merely preceding it — `x = x + PostNorm(Attn(PreNorm(x)))`, and likewise
for the FFN, four RMSNorm layers in all. The pre-norm half is the familiar
conditioning fix. The post-norm half addresses something pre-norm alone does not: in
a pure pre-norm transformer nothing ever rescales what a branch contributes, so the
residual stream's variance accumulates monotonically with depth. Normalizing the
branch *output* before the addition bounds each block's contribution while leaving
the residual path itself free of any normalization, so gradients still reach layer
zero unattenuated.

Masking is built here rather than at the model, because it depends on this block's
own `attention_type`. `_create_attention_mask` works in *block* semantics
(`True` = suppress): `j > i` for causality, OR-ed with `(i - j) >= sliding_window_size`
when the block is windowed, so a windowed block sees a band rather than a full
triangle. The result is inverted once to the *attend* semantics the attention layer
expects, and then explicitly expanded to `(1, q, k)`. That leading axis is
load-bearing and not decorative broadcasting: a rank-2 mask is interpreted
downstream as a padding mask rather than a full attention bias, which would
silently discard causality. A caller-supplied `attention_mask` (1 = attend,
0 = pad) is cast to boolean and AND-ed in as `(batch, 1, k)`, masking padded *keys*
only — padded query rows still produce output, which the loss is expected to ignore.

Attention and the FFN come from the library factories (`group_query` and `geglu`)
rather than being implemented here. Two consequences follow that a reader comparing
against the published model should know. The grouped-query layer supports
`qk_norm_type`, but this block does not pass it, so there is no QK normalization.
And `rope_theta` is left at the attention layer's default for every block, whereas
the report uses a much larger RoPE base in the global-attention layers specifically
so they stay usable at long context; the interleaved pattern here therefore does not
reproduce the paper's long-context behaviour.

Sub-layers are created in `__init__` and built explicitly in `build`, the Modern
Keras 3 composite-layer pattern: every weight variable exists before any weight
restoration runs, so a saved block reloads into the same variable tree it was
serialized from rather than into a half-materialized one.

References:
    - Gemma Team, 2025. Gemma 3 Technical Report.
      (https://arxiv.org/abs/2503.19786)
    - Gemma Team, 2024. Gemma 2: Improving Open Language Models at a Practical Size.
      (https://arxiv.org/abs/2408.00118)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Beltagy et al., 2020. Longformer: The Long-Document Transformer.
      (https://arxiv.org/abs/2004.05150)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
"""



import keras
from typing import Any, Dict, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.activations import gelu_tanh
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Gemma3TransformerBlock(keras.layers.Layer):
    """
    Gemma 3 transformer block with sandwich normalization.

    Each sublayer is bracketed by RMSNorm rather than merely preceded by it --
    ``x = x + PostNorm(Attn(PreNorm(x)))`` and ``x = x + PostNorm(FFN(PreNorm(x)))``,
    four norms per block -- so every branch's contribution to the residual stream
    is bounded while the residual path itself stays free of normalization.
    Attention is grouped-query (``group_query`` factory) and the feed-forward
    network is GeGLU (``geglu`` factory) with the tanh-approximate GELU. The
    causal (and, for ``attention_type='sliding_window'``, banded) mask is built
    inside the block because it depends on this block's own attention type, then
    inverted to the attend-semantics the attention layer expects and expanded to
    rank 3 so it is not mistaken for a padding mask.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │   Input [B, T, hidden_size]          │
        └───────────────┬──────────────────────┘
                        ├──────────────────────────────┐
                        ▼                              │ residual
        ┌──────────────────────────────────────┐       │
        │  input_layernorm (RMSNorm)           │       │
        └───────────────┬──────────────────────┘       │
                        ▼                              │
        ┌──────────────────────────────────────┐       │
        │  attention (grouped-query)           │◄── mask (1, T, T)
        │  heads=H, kv_heads=G                 │       │
        └───────────────┬──────────────────────┘       │
                        ▼                              │
        ┌──────────────────────────────────────┐       │
        │  post_attention_layernorm (RMSNorm)  │       │
        └───────────────┬──────────────────────┘       │
                        ▼                              │
                       (+)◄────────────────────────────┘
                        ├──────────────────────────────┐
                        ▼                              │ residual
        ┌──────────────────────────────────────┐       │
        │  pre_feedforward_layernorm (RMSNorm) │       │
        └───────────────┬──────────────────────┘       │
                        ▼                              │
        ┌──────────────────────────────────────┐       │
        │  ffn (GeGLU, gelu_tanh)              │       │
        └───────────────┬──────────────────────┘       │
                        ▼                              │
        ┌──────────────────────────────────────┐       │
        │  post_feedforward_layernorm (RMSNorm)│       │
        └───────────────┬──────────────────────┘       │
                        ▼                              │
                       (+)◄────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │   Output [B, T, hidden_size]         │
        └──────────────────────────────────────┘

    **Masking:**

    .. code-block:: text

        internal (True = SUPPRESS)          full_attention   sliding_window
          causal:      j > i                    ■ □ □ □          ■ □ □ □
          far past:    (i - j) >= W             ■ ■ □ □          ■ ■ □ □
          (OR-ed when windowed)                 ■ ■ ■ □          □ ■ ■ □
                                                ■ ■ ■ ■          □ □ ■ ■
                                                             (W = 2, ■ = attend)

        logical_not  →  attend semantics
        [None, :, :] →  (1, q, k)   rank 3 is REQUIRED: a rank-2 mask is read
                                    downstream as a padding mask, dropping causality
        AND padding  →  (batch, q, k) from attention_mask[:, None, :]

    **Mathematical Operations:**

    .. code-block:: text

        1. Attention path: x = x + PostAttnNorm(Attention(InputNorm(x)))
        2. FFN path:       y = x + PostFFNNorm(FFN(PreFFNNorm(x)))

    :param hidden_size: Width of the residual stream. Positive, and divisible by
        ``num_attention_heads``.
    :type hidden_size: int
    :param num_attention_heads: Number of query heads. Positive.
    :type num_attention_heads: int
    :param num_key_value_heads: Number of key/value heads for grouped-query
        attention. Positive, and must divide ``num_attention_heads`` evenly.
    :type num_key_value_heads: int
    :param ffn_hidden_size: GeGLU intermediate width. Positive.
    :type ffn_hidden_size: int
    :param max_seq_len: Maximum sequence length the attention layer is
        configured for. Defaults to ``32768``.
    :type max_seq_len: int
    :param attention_type: ``'full_attention'`` (default) gives a plain causal
        triangle; ``'sliding_window'`` additionally suppresses keys further than
        ``sliding_window_size`` in the past, so the block sees a band. This is a
        per-block choice: the interleaved local/global pattern is composed by the
        model that stacks these blocks.
    :type attention_type: Literal['sliding_window', 'full_attention']
    :param sliding_window_size: Window width, in tokens, used only when
        ``attention_type='sliding_window'``. Defaults to ``512``.
    :type sliding_window_size: int
    :param dropout_rate: Dropout rate forwarded to both the attention layer and
        the FFN. In ``[0, 1]``. Defaults to ``0.0``.
    :type dropout_rate: float
    :param use_bias: Whether the attention and FFN projections carry biases.
        Defaults to ``False``.
    :type use_bias: bool
    :param norm_eps: Epsilon for all four RMSNorm layers. Defaults to ``1e-6``.
    :type norm_eps: float
    :param kernel_initializer: Initializer for the attention and FFN kernels.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the ``keras.layers.Layer``
        base class (``name``, ``trainable``, ...).

    Input shape:
        3D tensor with shape ``(batch_size, sequence_length, hidden_size)``.

    Output shape:
        3D tensor with shape ``(batch_size, sequence_length, hidden_size)``.
        The residual connections preserve the input shape exactly.

    Example:
        >>> # A global-attention block
        >>> block = Gemma3TransformerBlock(hidden_size=1152,
        ...                                num_attention_heads=4,
        ...                                num_key_value_heads=1,
        ...                                ffn_hidden_size=6912)
        >>>
        >>> # A local (windowed) block, as used in the interleaved pattern
        >>> block = Gemma3TransformerBlock(hidden_size=1152,
        ...                                num_attention_heads=4,
        ...                                num_key_value_heads=1,
        ...                                ffn_hidden_size=6912,
        ...                                attention_type="sliding_window",
        ...                                sliding_window_size=512)
        >>>
        >>> # With a tokenizer padding mask (1 = attend, 0 = pad)
        >>> y = block(x, attention_mask=padding_mask)

    Note:
        This block does not pass ``qk_norm_type`` to the grouped-query factory,
        so there is no QK normalization, and it leaves ``rope_theta`` at the
        attention layer's default for every block rather than raising it in the
        global-attention layers. Long-context behaviour therefore differs from
        the published model.

    Attributes:
        input_layernorm: RMSNorm applied before attention.
        post_attention_layernorm: RMSNorm applied to the attention branch output.
        pre_feedforward_layernorm: RMSNorm applied before the FFN.
        post_feedforward_layernorm: RMSNorm applied to the FFN branch output.
        attention: Grouped-query attention layer from the framework factory.
        ffn: GeGLU feed-forward network from the framework factory.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        ffn_hidden_size: int,
        max_seq_len: int = 32768,
        attention_type: Literal[
            "sliding_window", "full_attention"
        ] = "full_attention",
        sliding_window_size: int = 512,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        norm_eps: float = 1e-6,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Store ALL configuration parameters for serialization
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.max_seq_len = max_seq_len
        self.attention_type = attention_type
        self.sliding_window_size = sliding_window_size
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.norm_eps = norm_eps
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE all sub-layers in __init__ (Modern Keras 3 pattern)
        self.input_layernorm = create_normalization_layer(
            "rms_norm", epsilon=norm_eps, name="input_layernorm"
        )
        self.post_attention_layernorm = create_normalization_layer(
            "rms_norm", epsilon=norm_eps, name="post_attention_layernorm"
        )
        self.pre_feedforward_layernorm = create_normalization_layer(
            "rms_norm", epsilon=norm_eps, name="pre_feedforward_layernorm"
        )
        self.post_feedforward_layernorm = create_normalization_layer(
            "rms_norm", epsilon=norm_eps, name="post_feedforward_layernorm"
        )

        # Create attention layer using correct parameter names for the factory
        self.attention = create_attention_layer(
            "group_query",
            dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            max_seq_len=self.max_seq_len,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="attention",
        )

        # Create GeGLU FFN using framework factory
        # DECISION plan-2026-08-23T091307-9a110062/D-501
        # The activation is the TANH APPROXIMATION, not the string "gelu".
        # HuggingFace's Gemma3TextConfig.hidden_activation defaults to
        # "gelu_pytorch_tanh" (= functools.partial(F.gelu, approximate="tanh")):
        #   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py
        # Keras' "gelu" string is approximate=False (exact/erf), so the bare
        # string was silently the WRONG form. Do NOT revert to "gelu": that is
        # inference-changing (max|erf - tanh| = 4.732e-04 per call), not a
        # cosmetic difference. The callable is a registered serializable, so
        # GeGLUFFN.get_config()'s keras.activations.serialize round-trips it.
        # See decisions.md D-501.
        self.ffn = create_ffn_layer(
            "geglu",
            hidden_dim=self.ffn_hidden_size,
            output_dim=self.hidden_size,
            activation=gelu_tanh,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="ffn",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build this layer and, explicitly, every sub-layer.

        Each sub-layer is built by hand rather than being left to materialize on
        first call. Weight restoration during deserialization requires the
        variables to already exist, so a block that is only implicitly built
        reloads into an incomplete variable tree.

        :param input_shape: Shape of the input to ``call``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build all sub-layers, ensuring weights are created
        self.input_layernorm.build(input_shape)
        self.post_attention_layernorm.build(input_shape)
        self.pre_feedforward_layernorm.build(input_shape)
        self.post_feedforward_layernorm.build(input_shape)
        self.attention.build(input_shape)
        self.ffn.build(input_shape)

        # ALWAYS call parent build at the end
        super().build(input_shape)

    def compute_output_spec(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Infer the output shape and dtype for the functional API.

        The parameter names must match ``call``'s, which is why the unused
        ``attention_mask`` and ``training`` are repeated here. The block changes
        neither shape nor dtype.

        :param inputs: Symbolic input tensor.
        :type inputs: keras.KerasTensor
        :param attention_mask: Unused; present for signature parity with ``call``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Unused; present for signature parity with ``call``.
        :type training: Optional[bool]
        :return: A symbolic tensor with the input's shape and dtype.
        :rtype: keras.KerasTensor
        """
        # The arguments must have the same names as in the `call` method.
        # This layer does not change the shape or dtype of the input tensor.
        return keras.KerasTensor(shape=inputs.shape, dtype=inputs.dtype)

    def _create_attention_mask(self, seq_len: int) -> keras.KerasTensor:
        """Build this block's attention mask in SUPPRESS semantics.

        ``True`` means MASK, the inverse of what the attention layer consumes;
        ``call`` performs the single inversion. Causality is ``j > i``; a
        windowed block OR-s in ``(i - j) >= sliding_window_size`` so keys further
        back than the window are suppressed as well, turning the triangle into a
        band.

        :param seq_len: Query and key length.
        :type seq_len: int
        :return: Boolean mask of shape ``(seq_len, seq_len)``, ``True`` = suppress.
        :rtype: keras.KerasTensor
        """
        i = keras.ops.arange(seq_len)[:, None]
        j = keras.ops.arange(seq_len)
        causal_mask = j > i

        if self.attention_type == "sliding_window":
            far_past_mask = (i - j) >= self.sliding_window_size
            return keras.ops.logical_or(causal_mask, far_past_mask)
        # 'full_attention'
        return causal_mask

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass through the transformer block.

        :param inputs: Input tensor of shape ``(batch, seq_len, hidden_size)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional padding mask in the tokenizer
            convention, ``1`` = attend and ``0`` = pad, of shape
            ``(batch, seq_len)``. It masks padded KEYS only; padded query rows
            still produce output, which the loss is expected to ignore.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        residual = inputs
        x = self.input_layernorm(inputs)

        seq_len = keras.ops.shape(inputs)[1]

        # The internal mask generation creates a boolean mask where True means
        # MASK. The underlying attention layer expects a mask where True
        # means ATTEND. So, we create our internal mask and then invert it.
        internal_mask_to_hide = self._create_attention_mask(seq_len)
        final_mask_to_attend = keras.ops.logical_not(internal_mask_to_hide)

        # Expand dims to make the mask shape unambiguous for the attention
        # layer. It must be at least 3D to avoid being misinterpreted as a
        # padding mask. Shape becomes (1, q_len, k_len) for broadcasting
        # across the batch dim.
        final_mask_to_attend = final_mask_to_attend[None, :, :]

        # The `attention_mask` argument is a padding mask (e.g., from a
        # tokenizer). Conventionally, it's 1 for tokens to attend to,
        # 0 for padding (mask).
        if attention_mask is not None:
            # Cast to boolean where True means ATTEND.
            padding_mask_to_attend = keras.ops.cast(attention_mask, "bool")

            # Combine masks. A position is attended if it's not a future/
            # sliding token AND it's not a padding token.
            # Broadcasting:
            # final_mask_to_attend:   (1,     q_len, k_len)
            # padding_mask_to_attend: (batch, 1,     k_len)
            # Result:                 (batch, q_len, k_len)
            final_mask_to_attend = keras.ops.logical_and(
                final_mask_to_attend, padding_mask_to_attend[:, None, :]
            )

        attn_output = self.attention(
            x, attention_mask=final_mask_to_attend, training=training
        )
        attn_output = self.post_attention_layernorm(attn_output)
        x = residual + attn_output

        residual = x
        x_ffn = self.pre_feedforward_layernorm(x)
        ffn_output = self.ffn(x_ffn, training=training)
        ffn_output = self.post_feedforward_layernorm(ffn_output)

        return residual + ffn_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        Both sublayers are residual, so neither can change the width or length.

        :param input_shape: Shape of the input to ``call``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Every constructor argument is stored on the instance and returned here,
        so the sub-layers can be reconstructed from the config alone rather than
        being serialized individually.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "ffn_hidden_size": self.ffn_hidden_size,
                "max_seq_len": self.max_seq_len,
                "attention_type": self.attention_type,
                "sliding_window_size": self.sliding_window_size,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "norm_eps": self.norm_eps,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------