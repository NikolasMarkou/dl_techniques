"""
Multi-Head Latent Attention (MLA) Layer.

This module provides a Keras 3 implementation of the Multi-Head Latent Attention
mechanism as proposed in DeepSeek-V2 architecture.

MLA significantly reduces Key-Value (KV) cache memory usage during inference
through low-rank compression, while maintaining performance comparable to
standard Multi-Head Attention (MHA).

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

    MLA significantly reduces Key-Value (KV) cache memory usage during inference
    through low-rank compression, while maintaining performance comparable to
    standard Multi-Head Attention (MHA).

    Architecture Overview::

        ┌─────────────────────────────────────────────────────────────────────┐
        │                   MULTI-HEAD LATENT ATTENTION (MLA)                 │
        │                                                                     │
        │  Standard MHA KV Cache: O(batch × seq × num_heads × head_dim)       │
        │  MLA KV Cache:          O(batch × seq × kv_latent_dim)              │
        │  Memory Reduction:      Up to 93% smaller KV cache!                 │
        └─────────────────────────────────────────────────────────────────────┘

    High-Level Data Flow::

                                    Input (B, S, D)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
            ┌───────────────┐                           ┌───────────────┐
            │  Query Path   │                           │   KV Path     │
            └───────┬───────┘                           └───────┬───────┘
                    │                                           │
                    ▼                                           ▼
            ┌───────────────┐                           ┌───────────────┐
            │ Down-Project  │ (optional)                │ Down-Project  │
            │   D → c_q     │                           │   D → c_kv    │
            └───────┬───────┘                           └───────┬───────┘
                    │                                           │
                    ▼                                           ▼
            ┌───────────────┐                           ┌───────────────┐
            │   RMS Norm    │                           │   RMS Norm    │
            └───────┬───────┘                           └───────┬───────┘
                    │                                           │
                    ▼                                           ▼
            ┌───────────────┐                           ┌───────────────┐
            │  Up-Project   │                           │  Up-Project   │
            │ c_q → Q_nope  │                           │c_kv → K_nope  │
            │      + Q_pe   │                           │       + V     │
            └───────┬───────┘                           └───────┬───────┘
                    │                                           │
                    │                               ┌───────────┴───────────┐
                    │                               │                       │
                    ▼                               ▼                       ▼
            ┌───────────────┐               ┌───────────────┐       ┌───────────┐
            │  Split Q into │               │  Split into   │       │  K_pe via │
            │ Q_nope, Q_pe  │               │ K_nope, V     │       │ separate  │
            └───────┬───────┘               └───────┬───────┘       │ projection│
                    │                               │               └─────┬─────┘
                    │                               │                     │
                    ▼                               ▼                     ▼
            ┌───────────────────────────────────────────────────────────────┐
            │                      ATTENTION COMPUTATION                    │
            │                                                               │
            │   scores = (Q_nope @ K_nope.T) + (Q_pe @ K_pe.T)              │
            │   scores = scores * scale                                     │
            │   weights = softmax(scores + mask)                            │
            │   output = weights @ V                                        │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ Output Proj   │
                                  │  H*v → D      │
                                  └───────┬───────┘
                                          │
                                          ▼
                                   Output (B, S, D)

    Decoupled RoPE Strategy::

        ┌─────────────────────────────────────────────────────────────────────┐
        │                    DECOUPLED ROPE MECHANISM                         │
        │                                                                     │
        │  Traditional RoPE: Apply rotation to full Q and K                   │
        │  Decoupled RoPE:   Separate content (nope) and position (pe)        │
        │                                                                     │
        │  Query Head Structure:                                              │
        │  ┌──────────────────────────────────────────────┐                   │
        │  │    Q_nope (content)     │    Q_pe (position) │                   │
        │  │   [128 dims/head]       │   [64 dims/head]   │                   │
        │  └──────────────────────────────────────────────┘                   │
        │           │                          │                              │
        │           │                          ▼                              │
        │           │                    ┌──────────┐                         │
        │           │                    │  RoPE    │                         │
        │           │                    │ rotation │                         │
        │           │                    └──────────┘                         │
        │           │                          │                              │
        │           ▼                          ▼                              │
        │     Content Score            Position Score                         │
        │    (Q_nope @ K_nope.T)      (Q_pe @ K_pe.T)                         │
        │           │                          │                              │
        │           └──────────┬───────────────┘                              │
        │                      ▼                                              │
        │              Combined Score                                         │
        │                                                                     │
        │  Key Insight: K_pe is SHARED across heads (1 head broadcasted)      │
        │  This further reduces memory while preserving positional info.      │
        └─────────────────────────────────────────────────────────────────────┘

    KV Compression Detail::

        ┌─────────────────────────────────────────────────────────────────────┐
        │                     KV LATENT COMPRESSION                           │
        │                                                                     │
        │  Standard MHA:                                                      │
        │  ┌─────────┐                                                        │
        │  │  Input  │ ──► W_k ──► K (H heads × head_dim)                     │
        │  │ (D=2048)│ ──► W_v ──► V (H heads × head_dim)                     │
        │  └─────────┘                                                        │
        │  Cache Size: 2 × H × head_dim = 2 × 16 × 128 = 4096 per token       │
        │                                                                     │
        │  MLA:                                                               │
        │  ┌─────────┐     ┌─────────┐     ┌─────────┐                        │
        │  │  Input  │ ──► │  Down   │ ──► │ Latent  │ (c_kv = 512)           │
        │  │ (D=2048)│     │  Proj   │     │  + Norm │                        │
        │  └─────────┘     └─────────┘     └────┬────┘                        │
        │                                       │                             │
        │                         ┌─────────────┼─────────────┐               │
        │                         ▼             │             ▼               │
        │                    ┌─────────┐        │        ┌─────────┐          │
        │                    │ Up Proj │        │        │ K_pe    │          │
        │                    │ K_nope  │        │        │ Proj    │          │
        │                    │   + V   │        │        │ (shared)│          │
        │                    └─────────┘        │        └─────────┘          │
        │                                       │                             │
        │  Cache Size: c_kv + rope_dim = 512 + 64 = 576 per token             │
        │  Reduction: (4096 - 576) / 4096 = 86% smaller!                      │
        └─────────────────────────────────────────────────────────────────────┘

    Attributes:
        dim: Model dimension (hidden size).
        num_heads: Number of attention heads.
        kv_latent_dim: Dimension of the compressed KV latent vector.
        q_latent_dim: Dimension of the compressed Query latent vector.
        qk_nope_head_dim: Dimension per head for non-positional content.
        qk_rope_head_dim: Dimension per head for rotary positional embeddings.
        v_head_dim: Dimension per head for values.
        dropout_rate: Dropout rate applied to attention weights.
        use_bias: Whether to use bias in projections.
        max_seq_len: Maximum sequence length for RoPE.
        rope_theta: Base frequency for RoPE.
        rope_percentage: Percentage of head dimensions to apply RoPE.

    Example:
        >>> mla = MultiHeadLatentAttention(
        ...     dim=2048,
        ...     num_heads=16,
        ...     kv_latent_dim=512,
        ...     q_latent_dim=1536,
        ...     qk_nope_head_dim=128,
        ...     qk_rope_head_dim=64,
        ...     v_head_dim=128,
        ... )
        >>> x = keras.random.normal((2, 128, 2048))
        >>> output = mla(x)
        >>> output.shape
        TensorShape([2, 128, 2048])
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
        """
        Initialize the Multi-Head Latent Attention layer.

        Parameter Relationships::

            ┌─────────────────────────────────────────────────────────────────┐
            │                    DIMENSION RELATIONSHIPS                      │
            │                                                                 │
            │  dim (D)                    Model hidden dimension              │
            │   │                                                             │
            │   ├──► q_latent_dim         Query compression (optional)        │
            │   │     │                                                       │
            │   │     └──► num_heads × (qk_nope_head_dim + qk_rope_head_dim)  │
            │   │                                                             │
            │   └──► kv_latent_dim        KV compression                      │
            │         │                                                       │
            │         └──► num_heads × (qk_nope_head_dim + v_head_dim)        │
            │                                                                 │
            │  Output: num_heads × v_head_dim ──► dim                         │
            └─────────────────────────────────────────────────────────────────┘

        Args:
            dim: Model dimension (hidden size).
            num_heads: Number of attention heads.
            kv_latent_dim: Dimension of the compressed KV latent vector.
            qk_nope_head_dim: Dimension per head for non-positional content
                (query/key). Defaults to 128.
            qk_rope_head_dim: Dimension per head for rotary positional embeddings.
                Defaults to 64.
            v_head_dim: Dimension per head for values. Defaults to 128.
            q_latent_dim: Dimension of the compressed Query latent vector.
                If None, Query compression is disabled (DeepSeek-V2 Lite style).
                Defaults to None.
            dropout_rate: Dropout rate applied to attention weights.
                Defaults to 0.0.
            use_bias: Whether to use bias in dense projections. Defaults to False.
            max_seq_len: Maximum sequence length for RoPE. Defaults to 4096.
            rope_theta: Base frequency for RoPE. Defaults to 10000.0.
            rope_percentage: Percentage of dimensions to apply RoPE. Defaults to 1.0.
            normalization_type: Type of normalization for latent vectors.
                Defaults to 'rms_norm'.
            kernel_initializer: Initializer for dense layer kernels.
                Defaults to 'glorot_uniform'.
            kernel_regularizer: Optional regularizer for kernels.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If dim, num_heads, or kv_latent_dim are not positive.
            ValueError: If dropout_rate is not in [0, 1].
        """
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

        # 1. Query Path: Optional compression via down-project → norm → up-project
        #
        #    With compression (q_latent_dim is set):
        #    Input ──► q_down_proj ──► q_norm ──► q_up_proj ──► Q
        #
        #    Without compression (q_latent_dim is None):
        #    Input ──► query_proj ──► Q
        #
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
        #
        #    Input ──► kv_down_proj ──► kv_norm ──► kv_up_proj ──► [K_nope, V]
        #          │
        #          └──► k_rope_proj ──► K_pe (shared across heads)
        #
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
        #
        #    Attention Output (H × v_head_dim) ──► output_proj ──► (D)
        #
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

        This method explicitly builds all sub-layers for robust serialization
        as required by Keras 3 patterns.

        Build Order::

            ┌─────────────────────────────────────────────────────────────────┐
            │                      BUILD SEQUENCE                             │
            │                                                                 │
            │  1. Query Path:                                                 │
            │     q_down_proj.build(input_shape)                              │
            │     q_norm.build(latent_shape)                                  │
            │     q_up_proj.build(latent_shape)                               │
            │     OR                                                          │
            │     query_proj.build(input_shape)                               │
            │                                                                 │
            │  2. KV Path:                                                    │
            │     kv_down_proj.build(kv_shape)                                │
            │     kv_norm.build(kv_latent_shape)                              │
            │     kv_up_proj.build(kv_latent_shape)                           │
            │     k_rope_proj.build(kv_shape)                                 │
            │                                                                 │
            │  3. RoPE:                                                       │
            │     rope.build(rope_input_shape)                                │
            │                                                                 │
            │  4. Output:                                                     │
            │     output_proj.build(concat_shape)                             │
            │                                                                 │
            │  5. Dropout (if present):                                       │
            │     dropout_layer.build(attn_weights_shape)                     │
            └─────────────────────────────────────────────────────────────────┘

        Args:
            input_shape: Shape of the input tensor. Can be a single tuple for
                self-attention or a list of tuples for cross-attention.
        """
        # Handle input_shape being a list (cross-attention) or single tuple
        # Check if input_shape is a list of shapes (cross-attn) or a single shape
        # In Keras serialization, a single shape can be a list [None, 16, 256]
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
        # RoPE expects shape (batch, seq_len, num_heads, head_dim)
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
            # Dropout expects attention weights shape (B, H, S_q, S_kv)
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

        Computation Flow::

            ┌─────────────────────────────────────────────────────────────────┐
            │                      FORWARD PASS STEPS                         │
            │                                                                 │
            │  Step 1: QUERY GENERATION                                       │
            │  ─────────────────────────────────────────                      │
            │  query_input ──► [compress] ──► Q ──► split ──► Q_nope, Q_pe    │
            │                                                                 │
            │  Step 2: KEY-VALUE GENERATION                                   │
            │  ─────────────────────────────────────────                      │
            │  kv_input ──► kv_down ──► c_kv ──► kv_up ──► K_nope, V          │
            │          └──► k_rope_proj ──────────────────► K_pe (shared)     │
            │                                                                 │
            │  Step 3: ROPE APPLICATION                                       │
            │  ─────────────────────────────────────────                      │
            │  Q_pe ──► RoPE ──► Q_pe_rotated                                 │
            │  K_pe ──► RoPE ──► K_pe_rotated                                 │
            │                                                                 │
            │  Step 4: ATTENTION SCORES                                       │
            │  ─────────────────────────────────────────                      │
            │  score = (Q_nope @ K_nope.T + Q_pe @ K_pe.T) × scale            │
            │                                                                 │
            │  Step 5: MASKING & SOFTMAX                                      │
            │  ─────────────────────────────────────────                      │
            │  score ──► [+ mask] ──► softmax ──► [dropout] ──► weights       │
            │                                                                 │
            │  Step 6: OUTPUT                                                 │
            │  ─────────────────────────────────────────                      │
            │  weights @ V ──► reshape ──► output_proj ──► output             │
            └─────────────────────────────────────────────────────────────────┘

        Args:
            query_input: Query tensor of shape (batch, seq_len_q, dim).
            kv_input: Key-Value tensor of shape (batch, seq_len_kv, dim).
                If None, uses query_input for self-attention. Defaults to None.
            attention_mask: Optional attention mask. Supports shapes:
                - (batch, seq_len_kv): Padding mask
                - (batch, seq_len_q, seq_len_kv): Full attention mask
                - (batch, 1, seq_len_q, seq_len_kv): Broadcasted mask
                Values of 1 indicate positions to attend to, 0 indicates masked.
                Defaults to None.
            training: Whether the layer is in training mode. Defaults to None.

        Returns:
            Output tensor of shape (batch, seq_len_q, dim).
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
        #
        #   With compression:     Input → Down → Norm → Up → Q
        #   Without compression:  Input → Proj → Q
        #
        #   Q shape: (B, S_q, H, nope_dim + rope_dim)
        # ═══════════════════════════════════════════════════════════════════
        if self.q_latent_dim is not None:
            # Compressed Query Path (DeepSeek-V2 Standard)
            c_q = self.q_down_proj(query_input)
            c_q = self.q_norm(c_q)
            q = self.q_up_proj(c_q)
        else:
            # Standard Query Path (DeepSeek-V2 Lite)
            q = self.query_proj(query_input)

        # Reshape Q → (B, S_q, H, nope_dim + rope_dim)
        q = ops.reshape(
            q,
            (batch_size, seq_len_q, self.num_heads,
             self.qk_nope_head_dim + self.qk_rope_head_dim)
        )

        # Split Q into Content (nope) and RoPE (pe) parts
        # Q_nope: (B, S_q, H, nope_dim) - content/semantic information
        # Q_pe:   (B, S_q, H, rope_dim) - positional information
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim:]

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: KEY-VALUE GENERATION (MLA Core)
        #
        #   KV Latent Path:
        #   Input → kv_down → c_kv → kv_norm → kv_up → [K_nope, V]
        #
        #   Decoupled RoPE Key (separate path, shared across heads):
        #   Input → k_rope_proj → K_pe
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
        # K_nope: (B, S_kv, H, nope_dim)
        # V:      (B, S_kv, H, v_dim)
        k_nope = kv_up[..., :self.qk_nope_head_dim]
        v = kv_up[..., self.qk_nope_head_dim:]

        # c. Decoupled RoPE Key (Shared)
        # K_pe is generated from original input, NOT latent vector
        # This is key to MLA's efficiency - K_pe is shared across heads
        k_pe = self.k_rope_proj(kv_input)  # (B, S_kv, rope_dim)
        # Expand dims for heads to broadcast: (B, S_kv, 1, rope_dim)
        k_pe = ops.expand_dims(k_pe, axis=2)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: ROPE APPLICATION
        #
        #   Apply rotary position embeddings to positional components only
        #   Q_pe and K_pe get rotated, Q_nope and K_nope remain unchanged
        # ═══════════════════════════════════════════════════════════════════
        q_pe = self.rope(q_pe)
        k_pe = self.rope(k_pe)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: ATTENTION SCORE CALCULATION
        #
        #   Total Score = Content Score + Positional Score
        #
        #   Content:   Q_nope @ K_nope.T  (semantic similarity)
        #   Position:  Q_pe @ K_pe.T      (relative position)
        #
        #   Note: K_pe broadcasts along Head dimension (shape B,1,S,D)
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
        #
        #   Apply attention mask (if provided) and compute attention weights
        # ═══════════════════════════════════════════════════════════════════
        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        attn_weights = ops.softmax(scores, axis=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: OUTPUT COMPUTATION
        #
        #   Weighted sum of values, reshape, and project back to model dim
        #
        #   weights @ V → (B, H, S_q, v_dim)
        #   reshape   → (B, S_q, H * v_dim)
        #   project   → (B, S_q, dim)
        # ═══════════════════════════════════════════════════════════════════

        # V shape: (B, S_kv, H, v_dim) → (B, H, S_kv, v_dim)
        v = ops.transpose(v, (0, 2, 1, 3))

        # (B, H, S_q, S_kv) @ (B, H, S_kv, v_dim) → (B, H, S_q, v_dim)
        out = ops.matmul(attn_weights, v)

        # Reshape for output projection
        # (B, H, S_q, v_dim) → (B, S_q, H, v_dim) → (B, S_q, H*v_dim)
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

        Mask Broadcasting::

            ┌─────────────────────────────────────────────────────────────────┐
            │                    MASK SHAPE HANDLING                          │
            │                                                                 │
            │  Input Mask Shape          →  Broadcast To                      │
            │  ─────────────────────────────────────────                      │
            │  (B, S_kv)                 →  (B, 1, 1, S_kv)   Padding mask    │
            │  (B, S_q, S_kv)            →  (B, 1, S_q, S_kv) Full mask       │
            │  (B, 1, S_q, S_kv)         →  (B, 1, S_q, S_kv) Already correct │
            │  (B, H, S_q, S_kv)         →  No change needed                  │
            │                                                                 │
            │  Mask Values:                                                   │
            │  1 = attend to this position                                    │
            │  0 = mask out (add -1e9 to scores)                              │
            └─────────────────────────────────────────────────────────────────┘

        Args:
            scores: Attention scores of shape (B, H, S_q, S_kv).
            attention_mask: Mask tensor with values 1 for positions to attend to.

        Returns:
            Masked scores tensor.
        """
        # Get mask dimensions
        mask_ndim = len(ops.shape(attention_mask))

        # Expand mask for broadcasting if needed
        if mask_ndim == 2:
            # (B, S_kv) → (B, 1, 1, S_kv)
            attention_mask = ops.expand_dims(
                ops.expand_dims(attention_mask, axis=1), axis=1
            )
        elif mask_ndim == 3:
            # (B, S_q, S_kv) → (B, 1, S_q, S_kv)
            attention_mask = ops.expand_dims(attention_mask, axis=1)

        # Cast and apply additive mask
        attention_mask = ops.cast(attention_mask, scores.dtype)
        scores = scores + (1.0 - attention_mask) * -1e9

        return scores

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Input shape tuple or list of tuples.

        Returns:
            Output shape tuple (batch, seq_len, dim).
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
        """
        Return the configuration of the layer for serialization.

        Returns:
            Configuration dictionary containing all constructor arguments.
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
