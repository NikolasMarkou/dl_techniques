"""
This module implements the Gemma 3 Transformer Block, a fundamental component of
the Gemma 3 language model architecture. It encapsulates the dual normalization
pattern, attention mechanism, and feed-forward network into a reusable Keras
layer.
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

import keras
from keras import initializers, ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Gemma3TransformerBlock(keras.layers.Layer):
    """
    Gemma 3 Transformer Block with a dual normalization pattern.

    This block implements Gemma 3's specific dual normalization architecture,
    where normalization is applied both before and after the attention and FFN
    sub-layers. It follows the Modern Keras 3 composite layer pattern for
    robust serialization.

    **Intent**: To provide a faithful and robust implementation of the Gemma 3
    transformer block, leveraging framework components for consistency and
    maintainability while ensuring correct serialization behavior.

    **Architecture**:
    ```
    Input(shape=[..., hidden_size])
           ↓
    InputLayerNorm(x) → Attention → PostAttnLayerNorm → + Residual
           ↓                                              ↑
           x ──────────────────────────────────────────────┘
           ↓
    PreFFNLayerNorm(x) → FFN → PostFFNNorm → + Residual
           ↓                                        ↑
           x ────────────────────────────────────────┘
           ↓
    Output(shape=[..., hidden_size])
    ```

    **Mathematical Operations**:
    1. **Attention Path**: attn_out = PostAttnNorm(Attention(InputNorm(x)))
       x = x + attn_out
    2. **FFN Path**: ffn_out = PostFFNNorm(FFN(PreFFNNorm(x)))
       output = x + ffn_out

    Args:
        hidden_size: Integer, hidden size of the layer. Must be positive and
            divisible by num_attention_heads.
        num_attention_heads: Integer, number of attention heads. Must be
            positive.
        num_key_value_heads: Integer, number of key-value heads for GQA. Must
            be positive and divide evenly into num_attention_heads.
        ffn_hidden_size: Integer, FFN intermediate size. Must be positive.
        max_seq_len: Integer, maximum sequence length for attention.
        attention_type: String, either 'sliding_window' or 'full_attention'.
        sliding_window_size: Integer, window size for local attention.
        dropout_rate: Float, dropout rate for regularization, in [0, 1].
        use_bias: Boolean, whether to use bias in linear layers.
        norm_eps: Float, epsilon for normalization layers.
        kernel_initializer: Initializer for kernel weights.
        **kwargs: Additional Layer arguments (name, trainable, etc.).

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    Attributes:
        input_layernorm: RMSNorm layer before attention.
        post_attention_layernorm: RMSNorm layer after attention.
        pre_feedforward_layernorm: RMSNorm layer before FFN.
        post_feedforward_layernorm: RMSNorm layer after FFN.
        attention: GroupedQueryAttention layer from the framework factory.
        ffn: GeGLU feed-forward network layer from the framework factory.
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
            str, initializers.Initializer
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
        self.kernel_initializer = initializers.get(kernel_initializer)

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
            name="attention",
        )

        # Create GeGLU FFN using framework factory
        self.ffn = create_ffn_layer(
            "geglu",
            hidden_dim=self.ffn_hidden_size,
            output_dim=self.hidden_size,
            activation="gelu",
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="ffn",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.
        CRITICAL: This method explicitly builds each sub-layer to ensure
        proper weight variable creation before weight restoration during
        serialization.
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

    def compute_output_spec(self, inputs, attention_mask=None, training=None):
        """Infers the output shape and dtype for the functional API."""
        # The arguments must have the same names as in the `call` method.
        # This layer does not change the shape or dtype of the input tensor.
        return keras.KerasTensor(shape=inputs.shape, dtype=inputs.dtype)

    def _create_attention_mask(self, seq_len: int) -> keras.KerasTensor:
        """Create attention mask based on attention type. True means MASK."""
        i = ops.arange(seq_len)[:, None]
        j = ops.arange(seq_len)
        causal_mask = j > i

        if self.attention_type == "sliding_window":
            far_past_mask = (i - j) >= self.sliding_window_size
            return ops.logical_or(causal_mask, far_past_mask)
        # 'full_attention'
        return causal_mask

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass through the transformer block."""
        residual = inputs
        x = self.input_layernorm(inputs)

        seq_len = ops.shape(inputs)[1]

        # The internal mask generation creates a boolean mask where True means
        # MASK. The underlying attention layer expects a mask where True
        # means ATTEND. So, we create our internal mask and then invert it.
        internal_mask_to_hide = self._create_attention_mask(seq_len)
        final_mask_to_attend = ops.logical_not(internal_mask_to_hide)

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
            padding_mask_to_attend = ops.cast(attention_mask, "bool")

            # Combine masks. A position is attended if it's not a future/
            # sliding token AND it's not a padding token.
            # Broadcasting:
            # final_mask_to_attend:   (1,     q_len, k_len)
            # padding_mask_to_attend: (batch, 1,     k_len)
            # Result:                 (batch, q_len, k_len)
            final_mask_to_attend = ops.logical_and(
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

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
