"""
A gated multi-head attention with rotary position embeddings.

This layer provides a sophisticated and high-performance implementation of
multi-head attention, integrating several modern architectural enhancements
commonly found in state-of-the-art large language models. It is designed
for improved training stability, expressiveness, and the effective encoding of
sequential information.

Architecture and Foundational Mathematics:
The layer's architecture augments the standard scaled dot-product attention
with three key concepts: Rotary Position Embedding (RoPE) for relative
positional awareness, RMSNorm for efficient normalization, and an output
gating mechanism for dynamic information flow control.

1.  **QKV Projection and Normalization:** Input tokens are first projected
    into Query (Q), Key (K), and Value (V) representations. Unlike standard
    Transformer architectures that apply Layer Normalization pre-attention,
    this layer normalizes Q, K, and V vectors independently using Root Mean
    Square Normalization (RMSNorm). RMSNorm is a simpler, computationally
    cheaper alternative that stabilizes training by re-scaling activations
    based on their root mean square magnitude, without re-centering them.

2.  **Rotary Position Embedding (RoPE):** To inject relative positional
    information, RoPE is applied to the normalized Q and K vectors. Instead
    of adding a positional encoding, RoPE rotates the embedding vectors in a
    high-dimensional space. The angle of rotation for a token at position `m`
    is determined by `m * θ`, where `θ` is a predefined frequency.

    The key insight is that the dot product of two rotated vectors, one at
    position `m` and another at `n`, depends only on their content and their
    relative distance `m-n`. This is because the rotational matrices are
    orthogonal, preserving vector norms, and the dot product `q_m^T k_n`
    becomes a function of the relative position `m-n`, allowing the
    self-attention mechanism to naturally capture relative spatial context.
    This implementation uses "partial RoPE," applying rotation only to a
    subset of the head dimensions, which may help disentangle positional
    information from content representation.

3.  **Scaled Dot-Product Attention:** After RoPE is applied, the standard
    attention mechanism is computed:

        Attention(Q, K, V) = softmax( (Q_rope * K_rope^T) / sqrt(d_k) ) * V

    The output is a weighted sum of the Value vectors, where the weights are
    determined by the similarity of the positionally-aware Query and Key
    vectors.

4.  **Output Gating:** The final attention output is modulated by a gating
    mechanism inspired by Gated Linear Units (GLU). The attention output `A`
    is used to compute a gate `g = sigmoid(Linear(A))`. The final layer
    output is the element-wise product `g ⊗ A`. This allows the model to
    learn to dynamically control the flow of information through the layer,
    selectively amplifying or attenuating features from the attention output.

References:
  - "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762
  - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021) https://arxiv.org/abs/2104.09864
  - "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
  - "LLaMA: Open and Efficient Foundation Language Models"
    (Touvron et al., 2023) https://arxiv.org/abs/2302.13971

"""

import keras
from keras import layers, initializers, regularizers, ops
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GatedAttention(keras.layers.Layer):
    """
    Gated multi-head attention with Zero-Centered RMSNorm, partial RoPE, and sigmoid output gating.

    Combines input linear projection, separate Q/K/V projections normalized
    with Zero-Centered RMSNorm, partial Rotary Position Embedding on Q and K,
    scaled dot-product attention, and a sigmoid gating mechanism. The forward
    pass computes ``output = sigma(W_gate(A')) * A'`` where
    ``A' = Attention(RoPE(RMSNorm(Q)), RoPE(RMSNorm(K)), RMSNorm(V))``.

    **Architecture Overview:**

    .. code-block:: text

        Input [B, S, dim]
              │
              ▼
        ┌──────────────┐
        │ Input Linear │
        └──────┬───────┘
               │
        ┌──────┼──────────────────────┐
        ▼      ▼                      ▼
      ┌────┐ ┌────┐               ┌────┐
      │W_q │ │W_k │               │W_v │
      └──┬─┘ └──┬─┘               └──┬─┘
         ▼      ▼                    ▼
      ┌──────┐┌──────┐          ┌──────┐
      │RMSNrm││RMSNrm│          │RMSNrm│
      └──┬───┘└──┬───┘          └──┬───┘
         ▼       ▼                 │
      ┌──────┐┌──────┐             │
      │ RoPE ││ RoPE │             │
      └──┬───┘└──┬───┘             │
         ▼       ▼                 ▼
        ┌─────────────────────────────┐
        │  Scaled Dot-Product Attn    │
        │  softmax(QK^T/√d_k) · V     │
        └──────────────┬──────────────┘
                       ▼
              (Optional Output Proj)
                       │
              ┌────────┼────────┐
              ▼                 ▼
        ┌───────────┐    ┌──────────┐
        │ W_gate    │    │ Identity │
        │ + Sigmoid │    │          │
        └─────┬─────┘    └────┬─────┘
              │               │
              └───── ⊗ ───────┘
                     ▼
              Output [B, S, dim]

    :param dim: Model dimension size. Must be positive and divisible by
        ``num_heads`` if ``head_dim`` is not specified.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param head_dim: Optional dimension per attention head. If ``None``,
        defaults to ``dim // num_heads``.
    :type head_dim: int or None
    :param max_seq_len: Maximum sequence length for RoPE precomputation.
        Defaults to 4096.
    :type max_seq_len: int
    :param rope_percentage: Fraction of head dimensions to apply RoPE to
        (partial RoPE). Must be in ``(0, 1]``. Defaults to 0.5.
    :type rope_percentage: float
    :param dropout_rate: Dropout rate for attention weights. Must be in
        ``[0, 1]``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias terms in linear layers.
        Defaults to ``False``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for linear layer weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias weights (if used).
        Defaults to ``'zeros'``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for linear layer weights.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim`` is not positive or not divisible by
        ``num_heads`` (when ``head_dim`` is ``None``).
    :raises ValueError: If ``num_heads`` is not positive.
    :raises ValueError: If ``head_dim`` is not positive (when specified).
    :raises ValueError: If ``rope_percentage`` is not in ``(0, 1]``.
    :raises ValueError: If ``dropout_rate`` is not in ``[0, 1]``.
    :raises ValueError: If ``max_seq_len`` is not positive.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            max_seq_len: int = 4096,
            rope_percentage: float = 0.5,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(dim, num_heads, head_dim, max_seq_len,
                            rope_percentage, dropout_rate)

        # Store ALL configuration parameters for serialization
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.max_seq_len = max_seq_len
        self.rope_percentage = rope_percentage
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # TRICKY POINT: When using custom head_dim, attention_dim may not equal dim
        # This requires an additional projection layer to match dimensions
        self.attention_dim = self.num_heads * self.head_dim

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Following the Golden Rule: Create in __init__, Build in build()

        # Input linear projection
        self.input_linear = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="input_linear"
        )

        # QKV projections - project to attention_dim, not dim
        self.q_linear = layers.Dense(
            self.attention_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="q_linear"
        )
        self.k_linear = layers.Dense(
            self.attention_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="k_linear"
        )
        self.v_linear = layers.Dense(
            self.attention_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="v_linear"
        )

        # Zero-Centered RMS Normalization layers
        self.q_norm = create_normalization_layer(
            'zero_centered_rms_norm',
            epsilon=1e-6,
            use_scale=True,
            name='q_norm'
        )
        self.k_norm = create_normalization_layer(
            'zero_centered_rms_norm',
            epsilon=1e-6,
            use_scale=True,
            name='k_norm'
        )
        self.v_norm = create_normalization_layer(
            'zero_centered_rms_norm',
            epsilon=1e-6,
            use_scale=True,
            name='v_norm'
        )

        # Rotary Position Embedding (Partial RoPE)
        # TRICKY POINT: RoPE doesn't have trainable parameters, just precomputed sin/cos
        self.rope = create_embedding_layer(
            'rope',
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_percentage=rope_percentage,
            name='rope'
        )

        # Dropout for attention weights (conditional creation)
        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(dropout_rate, name="attention_dropout")
        else:
            self.dropout = None

        # TRICKY POINT: Output projection needed when attention_dim != dim
        # This happens when using custom head_dim that doesn't divide evenly into dim
        if self.attention_dim != self.dim:
            self.output_proj = layers.Dense(
                self.dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="output_proj"
            )
        else:
            self.output_proj = None

        # Output gate - always projects to dim
        self.output_gate_linear = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_gate_linear"
        )

        logger.info(f"GatedAttention initialized: dim={dim}, "
                   f"num_heads={num_heads}, head_dim={self.head_dim}, "
                   f"attention_dim={self.attention_dim}")

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int],
        max_seq_len: int,
        rope_percentage: float,
        dropout_rate: float
    ) -> None:
        """
        Validate initialization parameters.

        :param dim: Model dimension to validate.
        :type dim: int
        :param num_heads: Number of attention heads to validate.
        :type num_heads: int
        :param head_dim: Head dimension to validate (can be ``None``).
        :type head_dim: int or None
        :param max_seq_len: Maximum sequence length to validate.
        :type max_seq_len: int
        :param rope_percentage: RoPE percentage to validate.
        :type rope_percentage: float
        :param dropout_rate: Dropout rate to validate.
        :type dropout_rate: float

        :raises ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim is None and dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}) when head_dim is None")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization support,
        ensuring all weight variables exist before weight restoration.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")

        batch_size, seq_len, features = input_shape

        if features != self.dim:
            raise ValueError(
                f"Input feature dimension ({features}) must match dim ({self.dim})"
            )

        # Build input linear
        self.input_linear.build(input_shape)

        # Compute intermediate shapes
        linear_output_shape = (batch_size, seq_len, self.dim)
        qkv_shape = (batch_size, seq_len, self.attention_dim)

        # Build QKV projections - MUST be built explicitly
        self.q_linear.build(linear_output_shape)
        self.k_linear.build(linear_output_shape)
        self.v_linear.build(linear_output_shape)

        # Build normalization layers
        self.q_norm.build(qkv_shape)
        self.k_norm.build(qkv_shape)
        self.v_norm.build(qkv_shape)

        # Build RoPE (it expects reshaped input: [batch, seq, heads, head_dim])
        rope_input_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        self.rope.build(rope_input_shape)

        # Build dropout if used
        if self.dropout is not None:
            self.dropout.build((batch_size, self.num_heads, seq_len, seq_len))

        # TRICKY POINT: Build output projection only if it exists
        if self.output_proj is not None:
            self.output_proj.build((batch_size, seq_len, self.attention_dim))

        # Build output gate
        self.output_gate_linear.build((batch_size, seq_len, self.dim))

        # Always call parent build at the end
        super().build(input_shape)

    def scaled_dot_product_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute scaled dot-product attention.

        :param q: Query tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :type q: keras.KerasTensor
        :param k: Key tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :type v: keras.KerasTensor
        :param attention_mask: Optional attention mask of shape
            ``[batch, seq_len]`` or ``[batch, seq_len, seq_len]``.
        :type attention_mask: keras.KerasTensor or None
        :param training: Training mode flag.
        :type training: bool or None
        :return: Attention output tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :rtype: keras.KerasTensor
        """
        # Transpose to [batch, num_heads, seq_len, head_dim] for attention computation
        q = ops.transpose(q, axes=[0, 2, 1, 3])
        k = ops.transpose(k, axes=[0, 2, 1, 3])
        v = ops.transpose(v, axes=[0, 2, 1, 3])

        # Compute attention scores
        matmul_qk = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))

        # Scale by sqrt(head_dim) for numerical stability
        dk = ops.cast(self.head_dim, keras.backend.floatx())
        scaled_attention_logits = matmul_qk / ops.sqrt(dk)

        if attention_mask is not None:
            # The mask can be (batch, seq_len) for padding or (batch, seq_len, seq_len) for causal.
            # We must broadcast it to (batch, num_heads, seq_len, seq_len).
            mask_ndim = ops.ndim(attention_mask)
            if mask_ndim == 2:
                # Padding mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
            elif mask_ndim == 3:
                # Causal/Combined mask: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = ops.expand_dims(attention_mask, 1)
            else:
                mask = attention_mask  # Assume it's already broadcastable

            # Create additive mask (masked positions get -inf)
            additive_mask = (1.0 - ops.cast(mask, scaled_attention_logits.dtype)) * -1e9
            scaled_attention_logits = scaled_attention_logits + additive_mask

        # Softmax over the last axis (key dimension)
        attention_weights = ops.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout during training
        if training and self.dropout is not None:
            attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        output = ops.matmul(attention_weights, v)

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        output = ops.transpose(output, axes=[0, 2, 1, 3])

        return output

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the gated attention layer.

        :param inputs: Input tensor of shape
            ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask of shape
            ``(batch_size, seq_len)`` or ``(batch_size, seq_len, seq_len)``.
        :type attention_mask: keras.KerasTensor or None
        :param training: Whether in training or inference mode.
        :type training: bool or None
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor
        """
        # Input linear projection
        x = self.input_linear(inputs, training=training)

        # Get batch and sequence dimensions dynamically
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # Generate Q, K, V projections
        q = self.q_linear(x, training=training)  # [batch, seq, attention_dim]
        k = self.k_linear(x, training=training)  # [batch, seq, attention_dim]
        v = self.v_linear(x, training=training)  # [batch, seq, attention_dim]

        # Apply Zero-Centered RMS Normalization
        q_norm = self.q_norm(q, training=training)
        k_norm = self.k_norm(k, training=training)
        v_norm = self.v_norm(v, training=training)

        # Reshape for multi-head attention and RoPE
        # [batch, seq, attention_dim] -> [batch, seq, num_heads, head_dim]
        q_reshaped = ops.reshape(q_norm, (batch_size, seq_len, self.num_heads, self.head_dim))
        k_reshaped = ops.reshape(k_norm, (batch_size, seq_len, self.num_heads, self.head_dim))
        v_reshaped = ops.reshape(v_norm, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Apply Partial RoPE to Q and K (not V)
        q_rope = self.rope(q_reshaped, training=training)
        k_rope = self.rope(k_reshaped, training=training)

        # Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(
            q_rope, k_rope, v_reshaped, attention_mask=attention_mask, training=training
        )

        # Reshape back to [batch, seq, attention_dim]
        attention_output = ops.reshape(
            attention_output, (batch_size, seq_len, self.attention_dim)
        )

        # TRICKY POINT: Project to original dim if needed
        # This is necessary when attention_dim != dim (custom head_dim case)
        if self.output_proj is not None:
            attention_output = self.output_proj(attention_output, training=training)

        # Output gating mechanism
        gate = ops.sigmoid(self.output_gate_linear(attention_output, training=training))
        gated_output = gate * attention_output

        return gated_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the complete layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'rope_percentage': self.rope_percentage,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config
