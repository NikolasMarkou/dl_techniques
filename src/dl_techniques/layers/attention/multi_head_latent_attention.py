"""
Multi-Head Latent Attention (MLA) Layer.

This module provides a Keras 3 implementation of the Multi-Head Latent Attention
mechanism as proposed in the DeepSeek-V2 architecture.

MLA significantly reduces Key-Value (KV) cache memory usage during inference
through low-rank compression, while maintaining performance comparable to
standard Multi-Head Attention (MHA). The core idea is to compress KV
representations into a low-dimensional latent space (``kv_latent_dim``) before
expanding them back for attention computation. Combined with a decoupled
Rotary Position Embedding (RoPE) strategy that separates content and
positional components, MLA achieves up to 93% KV cache reduction.

The attention score is computed as:
``scores = (Q_nope @ K_nope^T + Q_pe @ K_pe^T) * scale``
where ``Q_nope/K_nope`` carry content information and ``Q_pe/K_pe`` carry
positional information via RoPE. ``K_pe`` is shared across all heads for
additional memory savings.

References:
    - DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model
    - arXiv:2405.04434
"""

import math
import keras
from typing import Optional, Dict, Any, Tuple, Union
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiHeadLatentAttention(keras.layers.Layer):
    """
    Multi-Head Latent Attention (MLA) as proposed in DeepSeek-V2.

    MLA reduces KV cache memory from ``O(batch * seq * num_heads * head_dim)`` to
    ``O(batch * seq * kv_latent_dim)`` through low-rank compression of key-value
    representations, achieving up to 93% smaller KV cache while maintaining
    performance comparable to standard Multi-Head Attention.

    The layer uses a decoupled RoPE strategy that separates each query/key head into
    content (``nope``) and positional (``pe``) components. Content components carry
    semantic information through the latent bottleneck, while positional components
    bypass the bottleneck via a separate projection with RoPE applied. The positional
    key (``K_pe``) is shared across all heads for additional memory savings.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────────────┐
        │                   MULTI-HEAD LATENT ATTENTION (MLA)                 │
        │                                                                     │
        │                          Input (B, S, D)                            │
        │                                │                                    │
        │              ┌─────────────────┴─────────────────────┐              │
        │              │                                       │              │
        │              ▼                                       ▼              │
        │      ┌───────────────┐                       ┌───────────────┐     │
        │      │  Query Path   │                       │   KV Path     │     │
        │      └───────┬───────┘                       └───────┬───────┘     │
        │              │                                       │              │
        │              ▼                                       ▼              │
        │      ┌───────────────┐                       ┌───────────────┐     │
        │      │ Down-Project  │ (optional)             │ Down-Project  │     │
        │      │   D ──► c_q   │                       │   D ──► c_kv  │     │
        │      └───────┬───────┘                       └───────┬───────┘     │
        │              │                                       │              │
        │              ▼                                       ▼              │
        │      ┌───────────────┐                       ┌───────────────┐     │
        │      │   RMS Norm    │                       │   RMS Norm    │     │
        │      └───────┬───────┘                       └───────┬───────┘     │
        │              │                                       │              │
        │              ▼                                       ▼              │
        │      ┌───────────────┐                       ┌───────────────┐     │
        │      │  Up-Project   │                       │  Up-Project   │     │
        │      │ c_q ──► Q     │                       │c_kv ──► K,V   │     │
        │      └───────┬───────┘                       └───────┬───────┘     │
        │              │                                       │              │
        │              ▼                                 ┌─────┴─────┐        │
        │      ┌───────────────┐                         │           │        │
        │      │ Split Q into  │                         ▼           ▼        │
        │      │ Q_nope, Q_pe  │                    K_nope, V    K_pe via     │
        │      └───────┬───────┘                         │     separate proj  │
        │              │                                 │        + RoPE      │
        │              ▼                                 ▼           │        │
        │      ┌──────────────────────────────────────────────────────┐       │
        │      │              ATTENTION COMPUTATION                    │       │
        │      │                                                      │       │
        │      │  scores = (Q_nope @ K_nope^T) + (Q_pe @ K_pe^T)     │       │
        │      │  scores = scores * scale                             │       │
        │      │  weights = softmax(scores + mask)                    │       │
        │      │  output = weights @ V                                │       │
        │      └──────────────────────────┬───────────────────────────┘       │
        │                                 │                                   │
        │                                 ▼                                   │
        │                         ┌───────────────┐                           │
        │                         │ Output Proj   │                           │
        │                         │  H*v ──► D    │                           │
        │                         └───────┬───────┘                           │
        │                                 │                                   │
        │                                 ▼                                   │
        │                          Output (B, S, D)                           │
        └─────────────────────────────────────────────────────────────────────┘

    :param dim: Model dimension (hidden size). Must be positive.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param kv_latent_dim: Dimension of the compressed KV latent vector. Must be positive.
    :type kv_latent_dim: int
    :param qk_nope_head_dim: Dimension per head for non-positional content
        (query/key). Defaults to 128.
    :type qk_nope_head_dim: int
    :param qk_rope_head_dim: Dimension per head for rotary positional embeddings.
        Defaults to 64.
    :type qk_rope_head_dim: int
    :param v_head_dim: Dimension per head for values. Defaults to 128.
    :type v_head_dim: int
    :param q_latent_dim: Dimension of the compressed Query latent vector.
        If None, Query compression is disabled (DeepSeek-V2 Lite style).
        Defaults to None.
    :type q_latent_dim: Optional[int]
    :param dropout_rate: Dropout rate applied to attention weights.
        Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in dense projections. Defaults to False.
    :type use_bias: bool
    :param max_seq_len: Maximum sequence length for RoPE. Defaults to 4096.
    :type max_seq_len: int
    :param rope_theta: Base frequency for RoPE. Defaults to 10000.0.
    :type rope_theta: float
    :param rope_percentage: Percentage of dimensions to apply RoPE. Defaults to 1.0.
    :type rope_percentage: float
    :param normalization_type: Type of normalization for latent vectors.
        Defaults to 'rms_norm'.
    :type normalization_type: str
    :param kernel_initializer: Initializer for dense layer kernels.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernels.
        Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments passed to the parent class.

    :raises ValueError: If dim, num_heads, or kv_latent_dim are not positive.
    :raises ValueError: If dropout_rate is not in [0, 1].
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_latent_dim: int,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        q_latent_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_percentage: float = 1.0,
        normalization_type: str = "rms_norm",
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Multi-Head Latent Attention layer."""
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if kv_latent_dim <= 0:
            raise ValueError(f"kv_latent_dim must be positive, got {kv_latent_dim}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage
        self.normalization_type = normalization_type
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Scaling factor for attention scores
        # Scale by sqrt(total_qk_dim) for numerical stability
        self._scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

        # ─────────────────────────────────────────────────────────────────────
        # Create Sub-layers in __init__ (Keras 3 Pattern)
        # All sub-layers instantiated here, built in build()
        # ─────────────────────────────────────────────────────────────────────

        # 1. Query Path: Optional compression via down-project -> norm -> up-project
        if self.q_latent_dim is not None:
            self.q_down_proj = layers.Dense(
                q_latent_dim,
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="q_down_proj"
            )
            self.q_norm = create_normalization_layer(
                normalization_type,
                name="q_norm"
            )
            self.q_up_proj = layers.Dense(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="q_up_proj"
            )
        else:
            # Direct projection if no compression (DeepSeek-V2 Lite style)
            self.query_proj = layers.Dense(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="query_proj"
            )

        # 2. KV Compression Path
        self.kv_down_proj = layers.Dense(
            kv_latent_dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="kv_down_proj"
        )
        self.kv_norm = create_normalization_layer(
            normalization_type,
            name="kv_norm"
        )

        # 3. KV Up-Projection: Generates K_nope and V from latent
        self.kv_up_proj = layers.Dense(
            num_heads * (qk_nope_head_dim + v_head_dim),
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="kv_up_proj"
        )

        # 4. Decoupled RoPE Key projection (shared across heads)
        #    This generates positional keys directly from input, NOT from latent
        self.k_rope_proj = layers.Dense(
            qk_rope_head_dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="k_rope_proj"
        )

        # 5. RoPE Embeddings for Q_pe and K_pe
        #    Uses framework factory for consistent RoPE implementation
        self.rope = create_embedding_layer(
            "rope",
            head_dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_percentage=rope_percentage
        )

        # 6. Output Projection: Combines all heads back to model dimension
        self.output_proj = layers.Dense(
            dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="output_proj"
        )

        # 7. Optional Dropout on attention weights
        if dropout_rate > 0.0:
            self.dropout_layer = layers.Dropout(dropout_rate, name="attn_dropout")
        else:
            self.dropout_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers.

        Explicitly builds all sub-layers for robust serialization
        as required by Keras 3 patterns.

        :param input_shape: Shape of the input tensor. Can be a single tuple for
            self-attention or a list of tuples for cross-attention.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Handle input_shape being a list (cross-attention) or single tuple
        is_list_of_shapes = isinstance(input_shape, list) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple))

        if is_list_of_shapes:
            q_shape = input_shape[0]
            kv_shape = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        else:
            q_shape = kv_shape = input_shape

        # Validate input shape
        if len(q_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape (batch, seq_len, dim), got {q_shape}"
            )

        # Build Query path
        if self.q_latent_dim is not None:
            self.q_down_proj.build(q_shape)
            q_latent_shape = (q_shape[0], q_shape[1], self.q_latent_dim)
            self.q_norm.build(q_latent_shape)
            self.q_up_proj.build(q_latent_shape)
        else:
            self.query_proj.build(q_shape)

        # Build KV path (Content)
        self.kv_down_proj.build(kv_shape)
        kv_latent_shape = (kv_shape[0], kv_shape[1], self.kv_latent_dim)
        self.kv_norm.build(kv_latent_shape)
        self.kv_up_proj.build(kv_latent_shape)

        # Build KV path (RoPE - Shared Key)
        self.k_rope_proj.build(kv_shape)

        # Build RoPE embedding
        rope_input_shape = (
            q_shape[0], q_shape[1], self.num_heads, self.qk_rope_head_dim
        )
        self.rope.build(rope_input_shape)

        # Build Output projection
        output_input_shape = (
            q_shape[0], q_shape[1], self.num_heads * self.v_head_dim
        )
        self.output_proj.build(output_input_shape)

        # Build dropout if present
        if self.dropout_layer is not None:
            attn_shape = (q_shape[0], self.num_heads, q_shape[1], kv_shape[1])
            self.dropout_layer.build(attn_shape)

        super().build(input_shape)

    def call(
        self,
        query_input: keras.KerasTensor,
        kv_input: Optional[keras.KerasTensor] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Multi-Head Latent Attention layer.

        Computes attention via low-rank KV compression with decoupled RoPE:
        content scores ``(Q_nope @ K_nope^T)`` are combined with positional
        scores ``(Q_pe @ K_pe^T)`` before softmax normalization.

        :param query_input: Query tensor of shape ``(batch, seq_len_q, dim)``.
        :type query_input: keras.KerasTensor
        :param kv_input: Key-Value tensor of shape ``(batch, seq_len_kv, dim)``.
            If None, uses query_input for self-attention. Defaults to None.
        :type kv_input: Optional[keras.KerasTensor]
        :param attention_mask: Optional attention mask. Supports shapes:
            ``(batch, seq_len_kv)`` for padding mask,
            ``(batch, seq_len_q, seq_len_kv)`` for full attention mask,
            ``(batch, 1, seq_len_q, seq_len_kv)`` for broadcasted mask.
            Values of 1 indicate positions to attend to, 0 for masked.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether the layer is in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch, seq_len_q, dim)``.
        :rtype: keras.KerasTensor
        """
        # Default to self-attention if kv_input not provided
        if kv_input is None:
            kv_input = query_input

        # Get dynamic shapes
        batch_size = ops.shape(query_input)[0]
        seq_len_q = ops.shape(query_input)[1]
        seq_len_kv = ops.shape(kv_input)[1]

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: QUERY GENERATION
        # ═══════════════════════════════════════════════════════════════════
        if self.q_latent_dim is not None:
            # Compressed Query Path (DeepSeek-V2 Standard)
            c_q = self.q_down_proj(query_input)
            c_q = self.q_norm(c_q)
            q = self.q_up_proj(c_q)
        else:
            # Standard Query Path (DeepSeek-V2 Lite)
            q = self.query_proj(query_input)

        # Reshape Q -> (B, S_q, H, nope_dim + rope_dim)
        q = ops.reshape(
            q,
            (batch_size, seq_len_q, self.num_heads,
             self.qk_nope_head_dim + self.qk_rope_head_dim)
        )

        # Split Q into Content (nope) and RoPE (pe) parts
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim:]

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: KEY-VALUE GENERATION (MLA Core)
        # ═══════════════════════════════════════════════════════════════════

        # a. Latent Compression for K_nope and V
        c_kv = self.kv_down_proj(kv_input)
        c_kv = self.kv_norm(c_kv)

        # b. Up-Projection for K_nope and V
        kv_up = self.kv_up_proj(c_kv)
        kv_up = ops.reshape(
            kv_up,
            (batch_size, seq_len_kv, self.num_heads,
             self.qk_nope_head_dim + self.v_head_dim)
        )

        # Split into K_nope and V
        k_nope = kv_up[..., :self.qk_nope_head_dim]
        v = kv_up[..., self.qk_nope_head_dim:]

        # c. Decoupled RoPE Key (Shared)
        # K_pe is generated from original input, NOT latent vector
        k_pe = self.k_rope_proj(kv_input)  # (B, S_kv, rope_dim)
        # Expand dims for heads to broadcast: (B, S_kv, 1, rope_dim)
        k_pe = ops.expand_dims(k_pe, axis=2)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: ROPE APPLICATION
        # ═══════════════════════════════════════════════════════════════════
        q_pe = self.rope(q_pe)
        k_pe = self.rope(k_pe)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: ATTENTION SCORE CALCULATION
        # ═══════════════════════════════════════════════════════════════════

        # Transpose for matmul: (B, H, S, D)
        q_nope = ops.transpose(q_nope, (0, 2, 1, 3))
        q_pe = ops.transpose(q_pe, (0, 2, 1, 3))
        k_nope = ops.transpose(k_nope, (0, 2, 1, 3))
        k_pe = ops.transpose(k_pe, (0, 2, 1, 3))  # (B, 1, S_kv, rope_dim)

        # Content Score: (B, H, S_q, S_kv)
        score_content = ops.matmul(q_nope, ops.transpose(k_nope, (0, 1, 3, 2)))

        # Positional Score: (B, H, S_q, S_kv)
        # K_pe broadcasts along Head dimension because shape is (B, 1, S_kv, D)
        score_pos = ops.matmul(q_pe, ops.transpose(k_pe, (0, 1, 3, 2)))

        # Combine and Scale
        scores = (score_content + score_pos) * self._scale

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: MASKING & SOFTMAX
        # ═══════════════════════════════════════════════════════════════════
        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        attn_weights = ops.softmax(scores, axis=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: OUTPUT COMPUTATION
        # ═══════════════════════════════════════════════════════════════════

        # V shape: (B, S_kv, H, v_dim) -> (B, H, S_kv, v_dim)
        v = ops.transpose(v, (0, 2, 1, 3))

        # (B, H, S_q, S_kv) @ (B, H, S_kv, v_dim) -> (B, H, S_q, v_dim)
        out = ops.matmul(attn_weights, v)

        # Reshape for output projection
        # (B, H, S_q, v_dim) -> (B, S_q, H, v_dim) -> (B, S_q, H*v_dim)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(
            out, (batch_size, seq_len_q, self.num_heads * self.v_head_dim)
        )

        return self.output_proj(out)

    def _apply_attention_mask(
        self,
        scores: keras.KerasTensor,
        attention_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply attention mask to scores.

        :param scores: Attention scores of shape ``(B, H, S_q, S_kv)``.
        :type scores: keras.KerasTensor
        :param attention_mask: Mask tensor with values 1 for positions to attend to.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores tensor.
        :rtype: keras.KerasTensor
        """
        # Get mask dimensions
        mask_ndim = len(ops.shape(attention_mask))

        # Expand mask for broadcasting if needed
        if mask_ndim == 2:
            # (B, S_kv) -> (B, 1, 1, S_kv)
            attention_mask = ops.expand_dims(
                ops.expand_dims(attention_mask, axis=1), axis=1
            )
        elif mask_ndim == 3:
            # (B, S_q, S_kv) -> (B, 1, S_q, S_kv)
            attention_mask = ops.expand_dims(attention_mask, axis=1)

        # Cast and apply additive mask
        attention_mask = ops.cast(attention_mask, scores.dtype)
        scores = scores + (1.0 - attention_mask) * -1e9

        return scores

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape of the layer.

        :param input_shape: Input shape tuple or list of tuples.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple ``(batch, seq_len, dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        is_list_of_shapes = (
                isinstance(input_shape, list) and
                len(input_shape) > 0 and
                isinstance(input_shape[0],
                           (list, tuple))
        )

        if is_list_of_shapes:
            q_shape = input_shape[0]
        else:
            q_shape = input_shape

        return (q_shape[0], q_shape[1], self.dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "kv_latent_dim": self.kv_latent_dim,
            "q_latent_dim": self.q_latent_dim,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "rope_percentage": self.rope_percentage,
            "normalization_type": self.normalization_type,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
