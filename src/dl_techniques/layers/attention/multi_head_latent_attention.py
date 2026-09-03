"""Multi-Head Latent Attention (MLA), built by ``MultiHeadLatentAttention``.

MLA cuts key-value cache memory by compressing K and V into a low-dimensional
latent space (``kv_latent_dim``) before expanding them back for attention,
cutting KV cache size by up to 93% versus standard multi-head attention. A
decoupled RoPE strategy keeps positional information out of that bottleneck:
each query/key head splits into a content part (compressed) and a positional
part (a separate, uncompressed projection with RoPE applied). The positional
key is shared across all heads, so it broadcasts rather than being
materialized per head. Attention scores are the sum of the content and
positional dot products, scaled by ``1/sqrt(qk_nope_head_dim +
qk_rope_head_dim)``.

References:
    - DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model.
      (https://arxiv.org/abs/2405.04434)
"""

import keras
from typing import Optional, Dict, Any, Tuple, Union

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.utils.keras_registration import register_dl_technique

from .common import apply_attention_mask, compute_attention_scale


@register_dl_technique("dl_techniques.layers.attention.multi_head_latent_attention")
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

    Three responsibilities are delegated rather than reimplemented: the two
    latent norms (``q_norm``, ``kv_norm``) come from
    :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`,
    so ``qk_norm_type`` selects from all registered norm types; RoPE comes
    from :func:`~dl_techniques.layers.embedding.create_embedding_layer`
    (``"rope"``), the same implementation ``group_query_attention.py`` uses
    (the decoupled-RoPE trick is about which tensors get RoPE applied, not a
    different RoPE); score normalization goes through the shared
    :class:`~dl_techniques.layers.activations.ProbabilityOutput` layer.

    Symbols used below: ``D`` = dim, ``H`` = num_heads, ``n`` =
    qk_nope_head_dim, ``r`` = qk_rope_head_dim, ``v`` = v_head_dim,
    ``q_lat`` = q_latent_dim, ``kv_lat`` = kv_latent_dim.

    Architecture, building Q, K, V and the rotary parts:

    .. code-block:: text

        query_input [B, S_q, D]      kv_input [B, S_kv, D]
        (kv_input defaults to query_input for self-attention)
                │                        │
                │                        ├───────────────────┐
                ▼                        ▼                   │
        ┌────────────────┐      ┌────────────────┐           │
        │ q_down_proj    │      │ kv_down_proj   │           │
        │  D -> q_lat    │      │  D -> kv_lat   │           │
        └───────┬────────┘      └───────┬────────┘           │
                ▼                       ▼                    │
        ┌────────────────┐      ┌────────────────┐           │
        │ q_norm         │      │ kv_norm        │           │
        └───────┬────────┘      └───────┬────────┘           │
            c_q │ [B,S_q,q_lat]    c_kv │ [B,S_kv,kv_lat]    │
                ▼                       ▼                    │
        ┌────────────────┐      ┌────────────────┐           │
        │ q_up_proj      │      │ kv_up_proj     │           │
        │  -> H*(n + r)  │      │  -> H*(n + v)  │           │
        └───────┬────────┘      └───────┬────────┘           ▼
                ▼                       ▼           ┌────────────────┐
        split last axis         split last axis     │ k_rope_proj    │
        Q_nope [B,S_q,H,n]      K_nope [B,S_kv,H,n] │  D -> r        │
        Q_pe   [B,S_q,H,r]      V      [B,S_kv,H,v] └───────┬────────┘
                                                    [B,S_kv,r], then
                                                    expand_dims axis 2
                                                    -> [B,S_kv,1,r]

        The query tower is optional. With q_latent_dim=None the three
        query boxes collapse into one query_proj, D -> H*(n + r).
        k_rope_proj reads kv_input directly, never c_kv, which is why an
        inference cache only has to hold c_kv and k_pe instead of full
        per-head K and V. This layer itself is stateless and keeps no cache.

    Score computation and output:

    .. code-block:: text

        Q_pe and K_pe are transposed into the (B, H, S, D) frame and
        rotated there first, because RoPE reads its sequence length from
        axis 2.

            rope(Q_pe)  [B, H, S_q,  r]
            rope(K_pe)  [B, 1, S_kv, r]   one shared rotary head
                                │
          Q_nope @ K_nopeᵀ  ──┐ │
            [B, H, S_q, S_kv] │ │
                              ▼ ▼
                     scores = content + positional
                              │   K_pe broadcasts over H
                              ▼
                     scores * scale, scale = 1/sqrt(n + r)
                              ▼
                  _apply_attention_mask   (only if a mask is passed)
                              ▼
                     attn_prob  ►  dropout (only if dropout_rate > 0)
                              ▼
                     weights @ V   [B, H, S_q, v]
                              ▼
                    merge heads   [B, S_q, H*v]
                              ▼
                    ┌────────────────────┐
                    │ output_proj -> D   │
                    └─────────┬──────────┘
                              ▼
                      output  [B, S_q, D]

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
    :param qk_norm_type: Type of normalization for latent vectors (Q and KV).
        Forwarded to ``create_normalization_layer``. Defaults to 'rms_norm'.
    :type qk_norm_type: str
    :param qk_norm_kwargs: Optional extra keyword arguments forwarded to the
        normalization factory for both ``q_norm`` and ``kv_norm``. Defaults to None.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param probability_type: Strategy used to normalize attention scores into a
        probability distribution. Forwarded to :class:`ProbabilityOutput`.
        Defaults to ``"softmax"``. Routing/hierarchical variants are not
        supported in this layer.
    :type probability_type: str
    :param probability_config: Optional configuration dict forwarded to
        :class:`ProbabilityOutput` as ``type_config``. Defaults to None.
    :type probability_config: Optional[Dict[str, Any]]
    :param kernel_initializer: Initializer for dense layer kernels.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernels.
        Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments passed to the parent class.

    :raises ValueError: If dim, num_heads, or kv_latent_dim are not positive.
    :raises ValueError: If dropout_rate is not in [0, 1].
    :raises ValueError: If ``probability_type`` is a routing/hierarchical variant.
        Those strategies consume features and require a fixed ``output_dim``, which
        is incompatible with the ``(B, H, S_q, S_kv)`` score shape whose last axis
        is a runtime sequence length.
    :raises ValueError: From ``build()``, if the (query) input shape is not 3D.
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
        qk_norm_type: str = "rms_norm",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
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
        if probability_type in (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type '{probability_type}' is not supported by "
                "MultiHeadLatentAttention (routing/hierarchical variants are "
                "incompatible with the (B, H, S_q, S_kv) score shape)."
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
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # The per-head score width is the sum of the content and RoPE head
        # dims (the DeepSeek-V2 formulation), not dim // num_heads.
        self._scale = compute_attention_scale(qk_nope_head_dim + qk_rope_head_dim)


        # 1. Query Path: Optional compression via down-project -> norm -> up-project
        if self.q_latent_dim is not None:
            self.q_down_proj = keras.layers.Dense(
                q_latent_dim,
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="q_down_proj"
            )
            self.q_norm = create_normalization_layer(
                qk_norm_type,
                name="q_norm",
                **(self.qk_norm_kwargs or {})
            )
            self.q_up_proj = keras.layers.Dense(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="q_up_proj"
            )
        else:
            # Direct projection if no compression (DeepSeek-V2 Lite style)
            self.query_proj = keras.layers.Dense(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="query_proj"
            )

        # 2. KV Compression Path
        self.kv_down_proj = keras.layers.Dense(
            kv_latent_dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="kv_down_proj"
        )
        self.kv_norm = create_normalization_layer(
            qk_norm_type,
            name="kv_norm",
            **(self.qk_norm_kwargs or {})
        )

        # 3. KV Up-Projection: Generates K_nope and V from latent
        self.kv_up_proj = keras.layers.Dense(
            num_heads * (qk_nope_head_dim + v_head_dim),
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="kv_up_proj"
        )

        # Decoupled RoPE key projection, shared across heads, reads the
        # input directly rather than the latent vector.
        self.k_rope_proj = keras.layers.Dense(
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
        self.output_proj = keras.layers.Dense(
            dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="output_proj"
        )

        # 7. Attention probability layer (replaces direct softmax)
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # 8. Optional Dropout on attention weights
        if dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(dropout_rate, name="attn_dropout")
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
        if self.built:
            return

        # Positive test: container is a list whose first element is itself a
        # list/tuple. multi_head_cross_attention.py and perceiver_attention.py
        # each use a differently-shaped variant; not interchangeable.
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

        # Build RoPE embedding in the frame `call()` hands it: (B, H, S, D).
        # Axis 2 must be the SEQUENCE axis, because that is the axis RoPE reads
        # its length from. Build it in any other frame and the rotation is
        # applied against the wrong index.
        rope_input_shape = (
            q_shape[0], self.num_heads, q_shape[1], self.qk_rope_head_dim
        )
        self.rope.build(rope_input_shape)

        # Build Output projection
        output_input_shape = (
            q_shape[0], q_shape[1], self.num_heads * self.v_head_dim
        )
        self.output_proj.build(output_input_shape)

        # Build attention probability layer with score shape
        attn_shape = (q_shape[0], self.num_heads, q_shape[1], kv_shape[1])
        self.attn_prob.build(attn_shape)

        # Build dropout if present
        if self.dropout_layer is not None:
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
        batch_size = keras.ops.shape(query_input)[0]
        seq_len_q = keras.ops.shape(query_input)[1]
        seq_len_kv = keras.ops.shape(kv_input)[1]

        if self.q_latent_dim is not None:
            # Compressed query path (DeepSeek-V2 standard).
            c_q = self.q_down_proj(query_input)
            c_q = self.q_norm(c_q)
            q = self.q_up_proj(c_q)
        else:
            # Direct query path (DeepSeek-V2 Lite).
            q = self.query_proj(query_input)

        # Reshape Q -> (B, S_q, H, nope_dim + rope_dim)
        q = keras.ops.reshape(
            q,
            (batch_size, seq_len_q, self.num_heads,
             self.qk_nope_head_dim + self.qk_rope_head_dim)
        )

        # Split Q into Content (nope) and RoPE (pe) parts
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim:]

        c_kv = self.kv_down_proj(kv_input)
        c_kv = self.kv_norm(c_kv)

        kv_up = self.kv_up_proj(c_kv)
        kv_up = keras.ops.reshape(
            kv_up,
            (batch_size, seq_len_kv, self.num_heads,
             self.qk_nope_head_dim + self.v_head_dim)
        )

        # Split into K_nope and V
        k_nope = kv_up[..., :self.qk_nope_head_dim]
        v = kv_up[..., self.qk_nope_head_dim:]

        # Decoupled RoPE key, shared by every head, from the original input,
        # not the latent vector: (B, S_kv, rope_dim).
        k_pe = self.k_rope_proj(kv_input)
        # Expand dims for heads to broadcast: (B, S_kv, 1, rope_dim)
        k_pe = keras.ops.expand_dims(k_pe, axis=2)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-083: RoPE is applied in the
        # (B, H, S, D) frame -- transpose first and leave the tensors transposed. See decisions.md.
        q_pe = self.rope(keras.ops.transpose(q_pe, (0, 2, 1, 3)))
        k_pe = self.rope(keras.ops.transpose(k_pe, (0, 2, 1, 3)))

        # Only the two nope tensors still need transposing to (B, H, S, D);
        # q_pe/k_pe are already in that frame from the RoPE step above.
        q_nope = keras.ops.transpose(q_nope, (0, 2, 1, 3))
        k_nope = keras.ops.transpose(k_nope, (0, 2, 1, 3))

        # Content score: (B, H, S_q, S_kv)
        score_content = keras.ops.matmul(q_nope, keras.ops.transpose(k_nope, (0, 1, 3, 2)))

        # Positional score, K_pe broadcasts over the head dimension: (B, H, S_q, S_kv)
        score_pos = keras.ops.matmul(q_pe, keras.ops.transpose(k_pe, (0, 1, 3, 2)))

        scores = (score_content + score_pos) * self._scale

        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        # `training` is forwarded so the probability layer can honour it.
        attn_weights = self.attn_prob(scores, training=training)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # V shape: (B, S_kv, H, v_dim) -> (B, H, S_kv, v_dim)
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # (B, H, S_q, S_kv) @ (B, H, S_kv, v_dim) -> (B, H, S_q, v_dim)
        out = keras.ops.matmul(attn_weights, v)

        # Reshape for output projection
        # (B, H, S_q, v_dim) -> (B, S_q, H, v_dim) -> (B, S_q, H*v_dim)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(
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
        # Not shared with the near-identical helpers in
        # MultiHeadCrossAttention and GroupedQueryAttention -- each casts and
        # probes rank in its own order, and unifying them would change the
        # traced graph of the other two layers for no behavioural gain.
        mask_ndim = len(keras.ops.shape(attention_mask))

        if mask_ndim == 2:
            # (B, S_kv) -> (B, 1, 1, S_kv)
            attention_mask = keras.ops.expand_dims(
                keras.ops.expand_dims(attention_mask, axis=1), axis=1
            )
        elif mask_ndim == 3:
            # (B, S_q, S_kv) -> (B, 1, S_q, S_kv)
            attention_mask = keras.ops.expand_dims(attention_mask, axis=1)

        # attention_mask is a 1-means-keep predicate; passed through as-is.
        attention_mask = keras.ops.cast(attention_mask, scores.dtype)
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-007: out_dtype pinned to
        # scores' own dtype -- the additive form this replaces gives 0*-inf=NaN in float16. See decisions.md.
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-009: use the helper's
        # default rescue axis so a fully-masked row never forms a NaN gradient. See decisions.md.
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-017: rescue axis is
        # derived from probability_config, not a bare -1 default. See decisions.md.
        scores = apply_attention_mask(
            scores,
            attention_mask,
            out_dtype=(getattr(scores.dtype, "name", None) or str(scores.dtype)),
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

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
            "qk_norm_type": self.qk_norm_type,
            "qk_norm_kwargs": self.qk_norm_kwargs,
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config
