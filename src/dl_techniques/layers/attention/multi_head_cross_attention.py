"""
A unified multi-head attention with adaptive temperature and optional hierarchical routing.

This layer provides a versatile implementation of multi-head attention that
can operate in both self-attention and cross-attention modes. It extends
the standard mechanism with an optional adaptive temperature softmax, which
dynamically adjusts the sharpness of the attention distribution based on
the input, potentially improving model calibration and performance.

The layer's architecture is designed for flexibility. It can function in
two primary configurations determined by the inputs:

1.  **Cross-Attention:** When provided with distinct ``query`` and ``kv_input``
    tensors, it performs cross-attention. This is an asymmetric setup
    where a set of query vectors attends to a separate set of key-value
    pairs. This mode is fundamental to encoder-decoder models and
    architectures like Perceiver, where a small set of latent queries
    attends to a large set of input features.

2.  **Self-Attention:** When only a single input tensor is provided, it
    performs self-attention. This is a symmetric setup where all
    tokens in a sequence attend to all other tokens. The layer offers a
    ``shared_qk_projections`` option for this mode, which uses a single
    projection matrix to generate Q, K, and V. This is a parameter-
    efficient variant suitable for standard transformer blocks.

The core of this layer is the scaled dot-product attention mechanism
with an optional adaptive temperature ``T`` that is a function of the input:

    ``Attention(Q, K, V) = softmax( (Q @ K.T) / (sqrt(d_k) * T) ) @ V``

The adaptive temperature ``T`` is determined dynamically based on the
entropy of the pre-softmax attention scores for each query. High entropy
(uniform scores) yields low temperature to sharpen the distribution, while
low entropy (peaked scores) yields high temperature to soften it.

References:
    - The scaled dot-product attention mechanism was introduced in:
      Vaswani, A., et al. (2017). "Attention Is All You Need".

    - The use of temperature to control the sharpness of a softmax is a
      well-established technique, famously used in knowledge distillation:
      Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the
      Knowledge in a Neural Network".
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations import ProbabilityOutput
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiHeadCrossAttention(keras.layers.Layer):
    """
    Unified, highly configurable multi-head attention layer with advanced features.

    This layer serves as a versatile attention mechanism supporting both cross-attention
    and self-attention modes with flexible projection strategies, comprehensive masking,
    and optional adaptive temperature softmax for enhanced attention normalization.

    In cross-attention mode, separate projection matrices generate Q from the query input
    and K, V from a distinct key-value input. In self-attention mode with shared projections,
    a single dense layer generates Q, K, and V from the same input. The core computation
    follows: ``Attention(Q, K, V) = Normalize(Q @ K^T / sqrt(d_k)) @ V`` where
    Normalize is either standard softmax, adaptive temperature softmax, or hierarchical
    routing probabilities.

    **Architecture Overview:**

    .. code-block:: text

        Cross-Attention Mode (separate projections):

        ┌───────────────────────────────────────────────────────────────┐
        │                                                               │
        │  Query Input [B, Q_seq, D] ──► Q_proj ──► Q [B, H, Q_seq, D_h]│
        │                                             │                 │
        │  KV Input [B, KV_seq, D] ──► KV_proj ──► K, V [B,H,KV_seq,D_h]│
        │                                             │                 │
        │                                             ▼                 │
        │                               scores = Q @ K^T / sqrt(d_k)    │
        │                                             │                 │
        │  Mask [B, Q_seq, KV_seq] ──────────────► [+ mask]             │
        │                                             │                 │
        │                                             ▼                 │
        │                               AdaptiveSoftmax / Softmax       │
        │                                             │                 │
        │                                             ▼                 │
        │                                    weights @ V                │
        │                                             │                 │
        │                                             ▼                 │
        │                                     Output Projection         │
        │                                             │                 │
        │                                             ▼                 │
        │                                  Output [B, Q_seq, D]         │
        └───────────────────────────────────────────────────────────────┘

        Self-Attention Mode (shared projections):

        ┌───────────────────────────────────────────────────────────────┐
        │                                                               │
        │  Input [B, seq, D] ──► QKV_proj ──► Q, K, V [B, H, seq, D_h]  │
        │                                        │                      │
        │                                        ▼                      │
        │                          scores = Q @ K^T / sqrt(d_k)         │
        │                                        │                      │
        │  Mask [B, seq, seq] ──────────────► [+ mask]                  │
        │                                        │                      │
        │                                        ▼                      │
        │                          AdaptiveSoftmax / Softmax            │
        │                                        │                      │
        │                                        ▼                      │
        │                               weights @ V                     │
        │                                        │                      │
        │                                        ▼                      │
        │                                Output Projection              │
        │                                        │                      │
        │                                        ▼                      │
        │                              Output [B, seq, D]               │
        └───────────────────────────────────────────────────────────────┘

    :param dim: Integer, input/output dimension. Must be positive and divisible
        by num_heads.
    :type dim: int
    :param num_heads: Integer, number of attention heads. Must be positive.
        Defaults to 8.
    :type num_heads: int
    :param dropout_rate: Float, dropout rate for attention weights. Must be between
        0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param shared_qk_projections: Boolean, if True, uses a single dense layer for
        Q, K, and V. Only valid for self-attention. Defaults to False.
    :type shared_qk_projections: bool
    :param use_bias: Boolean, whether to use bias in linear projections.
        Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: String or Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: String or Initializer for bias vectors.
        Defaults to "zeros".
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param probability_type: String identifier for the attention-score normalization
        strategy. Forwarded to :class:`ProbabilityOutput`. One of ``"softmax"``,
        ``"sparsemax"``, ``"threshmax"``, ``"adaptive"`` (and their aliases).
        Defaults to ``"softmax"`` (standard scaled dot-product attention).
        ``"routing"`` and ``"hierarchical"`` are rejected with a ``ValueError``
        because they require a fixed ``output_dim`` and consume features rather
        than logits.
    :type probability_type: str
    :param probability_config: Optional dictionary of arguments forwarded to the
        underlying :class:`ProbabilityOutput` strategy. For ``"adaptive"`` accepts
        keys such as ``min_temp``, ``max_temp``, ``entropy_threshold``,
        ``polynomial_coeffs``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to Q and K projections
        before computing attention scores (QK-norm). Forwarded to
        :func:`create_normalization_layer`. ``None`` disables QK-norm.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when constructing Q/K norms.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``dim`` is not divisible by ``num_heads``, or if
        parameters are invalid.
    :raises ValueError: If ``shared_qk_projections=True`` is used with ``kv_input``.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_rate: float = 0.0,
            shared_qk_projections: bool = False,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.dropout_rate = dropout_rate
        self.shared_qk_projections = shared_qk_projections
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Scale factor for attention scores
        self.scale = 1.0 / ops.sqrt(float(self.head_dim))

        # CREATE sub-layers based on projection strategy
        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer
        }

        if self.shared_qk_projections:
            self.qkv_dense = keras.layers.Dense(
                self.dim * 3, name="qkv", **dense_kwargs
            )
            self.q_dense, self.kv_dense = None, None
        else:
            self.q_dense = keras.layers.Dense(self.dim, name="q", **dense_kwargs)
            self.kv_dense = keras.layers.Dense(self.dim * 2, name="kv", **dense_kwargs)
            self.qkv_dense = None

        self.proj_dense = keras.layers.Dense(self.dim, name="proj", **dense_kwargs)
        self.dropout_layer = keras.layers.Dropout(
            self.dropout_rate, name="dropout"
        ) if self.dropout_rate > 0.0 else None

        # Reject routing/hierarchical probability types: they require an
        # ``output_dim`` (and perform their own projection on FEATURES), which
        # is incompatible with operating on attention scores whose last
        # dimension is the dynamic kv sequence length.
        _ptype_lower = self.probability_type.lower()
        if _ptype_lower in (
                "routing",
                "deterministic_routing",
                "hierarchical",
                "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type='{self.probability_type}' is not supported "
                "in MultiHeadCrossAttention: routing/hierarchical strategies "
                "require a fixed output_dim and consume features rather than "
                "score logits. Use one of: 'softmax', 'sparsemax', 'threshmax', "
                "'adaptive'."
            )

        # CREATE unified probability output layer for attention-score normalization.
        # ProbabilityOutput validates probability_type internally.
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # CREATE optional QK-norm layers (applied to Q and K projections).
        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type,
                name="q_norm",
                **(self.qk_norm_kwargs or {}),
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type,
                name="k_norm",
                **(self.qk_norm_kwargs or {}),
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build the layer by creating weight variables and building sub-layers.

        Explicitly builds each sub-layer for robust serialization, ensuring
        weight variables exist before weight restoration during loading.

        :param input_shape: Shape tuple of the input tensor(s). A single tuple for
            self-attention or a list of two tuples for cross-attention.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        # Robustly determine if input_shape is a list of shapes (cross-attention)
        # or a single shape (self-attention). This works across backends.
        is_list_of_shapes = (
            isinstance(input_shape, list) and
            len(input_shape) > 0 and
            not isinstance(input_shape[0], (int, type(None)))
        )

        if is_list_of_shapes:
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs for cross-attention, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            query_shape = kv_shape = input_shape

        # Validate input shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if query_shape[-1] is not None and query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) must match dim ({self.dim})")

        # Build projection layers explicitly for serialization
        if self.shared_qk_projections:
            self.qkv_dense.build(query_shape)
        else:
            self.q_dense.build(query_shape)
            self.kv_dense.build(kv_shape)

        # Build output projection layer
        proj_input_shape = (query_shape[0], query_shape[1], self.dim)
        self.proj_dense.build(proj_input_shape)

        # Build dropout layer if exists
        if self.dropout_layer is not None:
            # Dropout doesn't change shape, use attention weight shape for building
            attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
            self.dropout_layer.build(attn_shape)

        # Build the unified probability output layer with the attention score
        # shape (B, H, Q_seq, KV_seq). Note: for routing/hierarchical strategies
        # the last-dim may vary at runtime; rebuild semantics follow
        # ProbabilityOutput's behavior.
        attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
        self.attn_prob.build(attn_shape)

        # Build QK-norm layers (operate on per-head Q/K projections of shape
        # (B, H, seq, D_h)).
        if self.q_norm is not None:
            q_norm_shape = (query_shape[0], self.num_heads, query_shape[1], self.head_dim)
            self.q_norm.build(q_norm_shape)
        if self.k_norm is not None:
            k_norm_shape = (query_shape[0], self.num_heads, kv_shape[1], self.head_dim)
            self.k_norm.build(k_norm_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _apply_attention_mask(
            self,
            scores: keras.KerasTensor,
            attention_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply attention mask to scores tensor.

        :param scores: Attention scores of shape ``(batch, num_heads, query_seq, kv_seq)``.
        :type scores: keras.KerasTensor
        :param attention_mask: Attention mask with supported shapes:
            ``(batch, kv_seq)`` for padding mask, ``(batch, query_seq, kv_seq)``
            for full attention mask, or other broadcastable shapes.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores tensor with same shape as input scores.
        :rtype: keras.KerasTensor
        """
        attention_mask = ops.cast(attention_mask, scores.dtype)

        # Expand mask dimensions to match scores shape (batch, num_heads, query_seq, kv_seq)
        if len(attention_mask.shape) == 2:  # Padding mask (batch, kv_seq)
            attention_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)  # (batch, 1, 1, kv_seq)
        elif len(attention_mask.shape) == 3:  # Full mask (batch, query_seq, kv_seq)
            attention_mask = ops.expand_dims(attention_mask, 1)  # (batch, 1, query_seq, kv_seq)

        # Apply mask: set masked positions to large negative value
        mask_value = -1e9
        return scores + (1.0 - attention_mask) * mask_value

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through multi-head attention with optional masking and adaptive softmax.

        Computes scaled dot-product attention with optional adaptive temperature
        softmax or hierarchical routing for attention weight normalization.

        :param query_input: Query tensor of shape ``(batch, query_seq_len, dim)``.
        :type query_input: keras.KerasTensor
        :param kv_input: Optional Key-Value tensor of shape ``(batch, kv_seq_len, dim)``.
            If ``None``, self-attention is performed on ``query_input``.
        :type kv_input: Optional[keras.KerasTensor]
        :param attention_mask: Optional mask to prevent attention to certain positions.
            Supports shapes: ``(batch, kv_seq_len)`` for padding mask,
            ``(batch, query_seq_len, kv_seq_len)`` for full mask, or broadcastable shapes.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating training or inference mode.
        :type training: Optional[bool]
        :return: Output tensor with shape ``(batch_size, query_seq_len, dim)``.
        :rtype: keras.KerasTensor
        """
        # --- 1. Initial Setup and Shape Extraction ---
        # We extract the batch size and query sequence length from the query input.
        # These values will be used repeatedly for reshaping tensors throughout the process.
        # query_input shape: (B, Q_seq, D)
        batch_size = ops.shape(query_input)[0]
        query_seq_len = ops.shape(query_input)[1]

        # --- 2. Project Inputs to Query, Key, and Value Tensors ---
        # This is the core projection step. Depending on the `shared_qk_projections`
        # flag, we use either a single large dense layer for self-attention or
        # separate dense layers for query and key-value pairs.

        if self.shared_qk_projections:
            # --- 2a. Shared Projection (Self-Attention Only) ---
            # This mode is parameter-efficient and only applicable for self-attention,
            # where query, key, and value all originate from the same input tensor.
            if kv_input is not None:
                raise ValueError(
                    "When `shared_qk_projections=True`, `kv_input` must be None "
                    "(self-attention mode only)."
                )

            # Project the single input into a combined Q, K, V tensor.
            # input shape: (B, Q_seq, D)
            # qkv_dense projects to 3 * D to hold Q, K, and V data.
            # qkv shape: (B, Q_seq, 3 * D)
            qkv = self.qkv_dense(query_input)

            # Reshape to separate Q, K, V and split the model dimension into heads.
            # Shape: (B, Q_seq, 3, H, D_h)
            qkv = ops.reshape(qkv, (batch_size, query_seq_len, 3, self.num_heads, self.head_dim))

            # Transpose to bring the head dimension forward, which is the standard
            # format for multi-head attention computation: (3, B, H, Q_seq, D_h)
            qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))

            # Unpack the first dimension to get separate Q, K, V tensors.
            # Each tensor shape: (B, H, Q_seq, D_h)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Optional QK-norm on per-head Q and K.
            if self.q_norm is not None:
                q = self.q_norm(q, training=training)
            if self.k_norm is not None:
                k = self.k_norm(k, training=training)

        else:
            # --- 2b. Separate Projections (Cross-Attention or Self-Attention) ---
            # This is the more general case. If `kv_input` is provided, we perform
            # cross-attention. Otherwise, we perform self-attention on `query_input`.
            kv_source = kv_input if kv_input is not None else query_input

            # --- Project Query ---
            # q_dense projects query_input to the model dimension.
            # query_input shape: (B, Q_seq, D)
            # q shape (after dense): (B, Q_seq, D)
            q = self.q_dense(query_input)
            # Reshape and transpose to multi-head format.
            # Shape (after reshape): (B, Q_seq, H, D_h)
            q = ops.reshape(q, (batch_size, query_seq_len, self.num_heads, self.head_dim))
            # Shape (after transpose): (B, H, Q_seq, D_h)
            q = ops.transpose(q, (0, 2, 1, 3))

            # --- Project Key and Value ---
            # kv_source shape: (B, KV_seq, D)
            kv_seq_len = ops.shape(kv_source)[1]
            # kv_dense projects to 2 * D to hold both K and V data.
            # kv shape (after dense): (B, KV_seq, 2 * D)
            kv = self.kv_dense(kv_source)
            # Reshape to separate K, V and split into heads.
            # Shape (after reshape): (B, KV_seq, 2, H, D_h)
            kv = ops.reshape(kv, (batch_size, kv_seq_len, 2, self.num_heads, self.head_dim))
            # Transpose to standard multi-head format.
            # Shape (after transpose): (2, B, H, KV_seq, D_h)
            kv = ops.transpose(kv, (2, 0, 3, 1, 4))
            # Unpack to get separate K, V tensors.
            # Each tensor shape: (B, H, KV_seq, D_h)
            k, v = kv[0], kv[1]

            # Optional QK-norm on per-head Q and K.
            if self.q_norm is not None:
                q = self.q_norm(q, training=training)
            if self.k_norm is not None:
                k = self.k_norm(k, training=training)

        # --- 3. Scaled Dot-Product Attention ---
        # Now that we have Q, K, and V, we compute the attention scores.
        # This involves a matrix multiplication between Q and K^T, followed by scaling.
        # q shape:      (B, H, Q_seq, D_h)
        # k shape:      (B, H, KV_seq, D_h)
        # k transposed: (B, H, D_h, KV_seq)
        # scores shape: (B, H, Q_seq, KV_seq)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        # Scale scores by the inverse square root of head dimension to prevent gradients
        # from becoming too small. The cast ensures type compatibility.
        scores = scores * ops.cast(self.scale, q.dtype)

        # --- 4. Apply Attention Mask (Optional) ---
        # If a mask is provided, we apply it to the scores. This sets the scores
        # for masked positions to a very large negative number, so they become
        # zero after the softmax normalization.
        if attention_mask is not None:
            # _apply_attention_mask handles broadcasting the mask to the scores' shape.
            # scores shape remains: (B, H, Q_seq, KV_seq)
            scores = self._apply_attention_mask(scores, attention_mask)

        # --- 5. Normalize Scores to get Attention Weights ---
        # Delegate to the unified ProbabilityOutput layer (softmax / sparsemax /
        # threshmax / adaptive / routing / hierarchical).
        # attn_weights shape: (B, H, Q_seq, KV_seq)
        attn_weights = self.attn_prob(scores, training=training)

        # --- 6. Apply Dropout to Attention Weights (Optional) ---
        # During training, dropout is applied to the attention weights to prevent
        # the model from becoming over-reliant on a few key-value pairs.
        if self.dropout_layer is not None:
            # attn_weights shape remains: (B, H, Q_seq, KV_seq)
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # --- 7. Compute Output by Attending to Values ---
        # The attention weights are used to compute a weighted sum of the value vectors.
        # attn_weights shape: (B, H, Q_seq, KV_seq)
        # v shape:            (B, H, KV_seq, D_h)
        # out shape (context vectors): (B, H, Q_seq, D_h)
        out = ops.matmul(attn_weights, v)

        # --- 8. Reshape and Project Final Output ---
        # The outputs from all heads are concatenated and passed through a final
        # linear projection layer.

        # First, transpose to bring the sequence and head dimensions together.
        # Shape (after transpose): (B, Q_seq, H, D_h)
        out = ops.transpose(out, (0, 2, 1, 3))

        # Reshape to concatenate the head outputs, effectively merging the heads.
        # Shape (after reshape): (B, Q_seq, H * D_h) -> (B, Q_seq, D)
        out = ops.reshape(out, (batch_size, query_seq_len, self.dim))

        # Apply the final linear projection. This allows the model to mix information
        # learned from the different attention heads.
        # out shape remains: (B, Q_seq, D)
        return self.proj_dense(out)

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """Compute output shape, returns query input shape.

        :param input_shape: Shape tuple or list of shape tuples.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: Output shape tuple.
        :rtype: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        is_list_of_shapes = (
            isinstance(input_shape, list) and
            len(input_shape) > 0 and
            not isinstance(input_shape[0], (int, type(None)))
        )

        if is_list_of_shapes:
            return input_shape[0]
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "shared_qk_projections": self.shared_qk_projections,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "qk_norm_type": self.qk_norm_type,
            "qk_norm_kwargs": self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------
