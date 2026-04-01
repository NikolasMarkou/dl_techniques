"""
A parameter-efficient, bidirectional cross-attention mechanism.

This layer facilitates information exchange between two distinct sets of
tokens (modalities) through a cross-attention pattern where the projection
weights for queries, keys, and values are shared. Given modalities ``X_A``
and ``X_B``, shared projections compute ``Q_A, K_A, V_A = f_q(X_A), f_k(X_A),
f_v(X_A)`` and ``Q_B, K_B, V_B = f_q(X_B), f_k(X_B), f_v(X_B)``, then
cross-attend: ``Out_A = Attention(Q_A, K_B, V_B)`` and
``Out_B = Attention(Q_B, K_A, V_A)``. This weight sharing forces both
modalities into a common semantic space, similar to Siamese networks.

References:
    - Vaswani, A., et al. (2017). "Attention Is All You Need".
    - Bromley, J., et al. (1994). "Signature verification using a Siamese
      time delay neural network".
    - Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The
      Long-Document Transformer".
"""

import keras
from keras import ops
from typing import Any, List, Union, Tuple, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SharedWeightsCrossAttention(keras.layers.Layer):
    """Bidirectional cross-attention between modalities with shared QKV projections.

    Implements parameter-efficient cross-attention where two modalities exchange
    information through shared projection weights. The concatenated input is
    projected via a single Dense layer to produce Q, K, V for all tokens, then
    split by modality for bidirectional cross-attention:
    ``scores_A = Q_A K_B^T / sqrt(d_k)``, ``Out_A = softmax(scores_A) V_B`` and
    vice versa. Optionally supports an anchor-query hierarchy where each modality
    is split into anchors and queries, and all tokens attend only to the opposing
    modality's anchors for efficient bottleneck communication.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────────┐
        │ Input [B, total_seq, dim]                      │
        │ (concatenated Modality A + Modality B tokens)  │
        └───────────────────────┬────────────────────────┘
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Shared QKV Dense(dim * 3)                      │
        │ → Q, K, V [B, heads, total_seq, head_dim]     │
        └───────────────────────┬────────────────────────┘
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Split by modality (using split_sizes)          │
        ├──────────────────┬─────────────────────────────┤
        │ Q_A, K_A, V_A   │  Q_B, K_B, V_B              │
        └────────┬─────────┴────────────┬────────────────┘
                 │                      │
                 ▼                      ▼
        ┌────────────────┐    ┌─────────────────┐
        │ Cross-Attend   │    │ Cross-Attend     │
        │ Q_A → K_B, V_B │    │ Q_B → K_A, V_A  │
        └───────┬────────┘    └────────┬─────────┘
                │                      │
                ▼                      ▼
        ┌────────────────────────────────────────────────┐
        │ Concatenate → [B, total_seq, dim]              │
        └───────────────────────┬────────────────────────┘
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Output Projection Dense(dim)                   │
        └───────────────────────┬────────────────────────┘
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Output [B, total_seq, dim]                     │
        └────────────────────────────────────────────────┘

    :param dim: Input/output dimension. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dropout_rate: Dropout rate for attention weights, between 0 and 1.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If dim is not divisible by num_heads.
    :raises ValueError: If dim or num_heads are not positive.
    :raises ValueError: If dropout_rate is not between 0 and 1.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
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

        # Store all configuration
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Scale factor for attention scores
        self.scale = 1.0 / ops.sqrt(float(self.head_dim))

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.qkv_dense = keras.layers.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="qkv"
        )

        self.proj_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj"
        )

        # Conditionally create dropout layer
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for robust serialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {input_shape}")

        if input_shape[-1] != self.dim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match dim ({self.dim})")

        # Build sub-layers explicitly for robust serialization
        self.qkv_dense.build(input_shape)
        self.proj_dense.build(input_shape)

        if self.dropout_layer is not None:
            # Dropout layer needs to be built with attention weights shape
            # For simplicity, we'll let it build automatically during call
            pass

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        split_sizes: Union[List[int], Tuple[int, ...]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply shared-weights bidirectional cross-attention.

        :param inputs: Input tensor of shape ``(batch_size, total_seq_len, dim)``
            containing concatenated sequences from different modalities.
        :type inputs: keras.KerasTensor
        :param split_sizes: How to split the input. Length 2 for anchor-only mode
            ``[mod_a_len, mod_b_len]`` or length 4 for anchor-query mode
            ``[mod_a_anchor, mod_a_query, mod_b_anchor, mod_b_query]``.
        :type split_sizes: Union[List[int], Tuple[int, ...]]
        :param attention_mask: Optional attention mask tensor.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Output tensor with cross-attended features, same shape as input.
        :rtype: keras.KerasTensor
        """
        if not isinstance(split_sizes, (list, tuple)):
            raise ValueError("split_sizes must be a list or tuple")

        if len(split_sizes) not in [2, 4]:
            raise ValueError("split_sizes must have length 2 or 4")

        _, total_seq_len, _ = inputs.shape

        # Verify split sizes sum to total sequence length
        if sum(split_sizes) != total_seq_len:
            raise ValueError(f"Sum of split_sizes ({sum(split_sizes)}) "
                             f"must equal total sequence length ({total_seq_len})")

        # Compute Q, K, V for all tokens
        qkv = self.qkv_dense(inputs)  # (batch_size, total_seq_len, dim * 3)
        # Use -1 for the batch dimension to be compatible with symbolic Keras tensors
        qkv = ops.reshape(qkv, (-1, total_seq_len, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch_size, num_heads, total_seq_len, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if len(split_sizes) == 2:
            # Two modalities: A and B cross-attend
            return self._two_modality_attention(q, k, v, split_sizes, training)
        else:
            # Four splits: anchors and queries for two modalities
            return self._anchor_query_attention(q, k, v, split_sizes, training)

    def _two_modality_attention(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        split_sizes: Union[List[int], Tuple[int, ...]],
        training: Optional[bool]
    ) -> keras.KerasTensor:
        """Cross-attention between two modalities.

        :param q: Query tensor of shape ``(batch, heads, total_seq, head_dim)``.
        :type q: keras.KerasTensor
        :param k: Key tensor of shape ``(batch, heads, total_seq, head_dim)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, heads, total_seq, head_dim)``.
        :type v: keras.KerasTensor
        :param split_sizes: Split sizes ``[mod_a_len, mod_b_len]``.
        :type split_sizes: Union[List[int], Tuple[int, ...]]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Cross-attended output tensor.
        :rtype: keras.KerasTensor
        """
        mod_a_len, mod_b_len = split_sizes

        # Split Q, K, V by modality
        q_splits = [q[:, :, :mod_a_len, :], q[:, :, mod_a_len:, :]]
        k_splits = [k[:, :, :mod_a_len, :], k[:, :, mod_a_len:, :]]
        v_splits = [v[:, :, :mod_a_len, :], v[:, :, mod_a_len:, :]]

        # Check for equal-sized modalities for optimization
        if mod_a_len == mod_b_len:
            # Optimized path for equal sizes
            q_combined = ops.concatenate(q_splits, axis=0)  # Stack batch dims
            k_swapped = ops.concatenate([k_splits[1], k_splits[0]], axis=0)
            v_swapped = ops.concatenate([v_splits[1], v_splits[0]], axis=0)

            # Compute attention
            scores = ops.matmul(q_combined, ops.transpose(k_swapped, (0, 1, 3, 2))) * self.scale
            attn_weights = ops.softmax(scores, axis=-1)

            if self.dropout_layer is not None:
                attn_weights = self.dropout_layer(attn_weights, training=training)

            attn_out = ops.matmul(attn_weights, v_swapped)

            # Split back and concatenate
            attn_a, attn_b = ops.split(attn_out, 2, axis=0)
            combined_out = ops.concatenate([attn_a, attn_b], axis=2)
        else:
            # General case for different sizes
            # Modality A attends to Modality B
            scores_a = ops.matmul(q_splits[0], ops.transpose(k_splits[1], (0, 1, 3, 2))) * self.scale
            attn_weights_a = ops.softmax(scores_a, axis=-1)
            if self.dropout_layer is not None:
                attn_weights_a = self.dropout_layer(attn_weights_a, training=training)
            attn_out_a = ops.matmul(attn_weights_a, v_splits[1])

            # Modality B attends to Modality A
            scores_b = ops.matmul(q_splits[1], ops.transpose(k_splits[0], (0, 1, 3, 2))) * self.scale
            attn_weights_b = ops.softmax(scores_b, axis=-1)
            if self.dropout_layer is not None:
                attn_weights_b = self.dropout_layer(attn_weights_b, training=training)
            attn_out_b = ops.matmul(attn_weights_b, v_splits[0])

            # Combine outputs
            combined_out = ops.concatenate([attn_out_a, attn_out_b], axis=2)

        # Reshape and project
        _, _, total_seq_len, _ = combined_out.shape
        combined_out = ops.transpose(combined_out, (0, 2, 1, 3))
        # Use -1 for the batch dimension to be compatible with symbolic Keras tensors
        combined_out = ops.reshape(combined_out, (-1, total_seq_len, self.dim))

        return self.proj_dense(combined_out)

    def _anchor_query_attention(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        split_sizes: Union[List[int], Tuple[int, ...]],
        training: Optional[bool]
    ) -> keras.KerasTensor:
        """Attention with anchor-query structure for two modalities.

        :param q: Query tensor of shape ``(batch, heads, total_seq, head_dim)``.
        :type q: keras.KerasTensor
        :param k: Key tensor of shape ``(batch, heads, total_seq, head_dim)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, heads, total_seq, head_dim)``.
        :type v: keras.KerasTensor
        :param split_sizes: Split sizes
            ``[mod_a_anchor, mod_a_query, mod_b_anchor, mod_b_query]``.
        :type split_sizes: Union[List[int], Tuple[int, ...]]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Cross-attended output tensor.
        :rtype: keras.KerasTensor
        """
        mod_a_anchor, mod_a_query, mod_b_anchor, mod_b_query = split_sizes

        # Combine modality A (anchors + queries) and modality B (anchors + queries)
        mod_a_total = mod_a_anchor + mod_a_query
        mod_b_total = mod_b_anchor + mod_b_query

        # Split by combined modalities
        q_mod_a = q[:, :, :mod_a_total, :]  # All of modality A
        q_mod_b = q[:, :, mod_a_total:, :]  # All of modality B

        k_mod_a_anchor = k[:, :, :mod_a_anchor, :]  # Only A anchors for keys
        k_mod_b_anchor = k[:, :, mod_a_total:mod_a_total + mod_b_anchor, :]  # Only B anchors

        v_mod_a_anchor = v[:, :, :mod_a_anchor, :]  # Only A anchors for values
        v_mod_b_anchor = v[:, :, mod_a_total:mod_a_total + mod_b_anchor, :]  # Only B anchors

        # Modality A (anchors + queries) attends to Modality B anchors
        scores_a = ops.matmul(q_mod_a, ops.transpose(k_mod_b_anchor, (0, 1, 3, 2))) * self.scale
        attn_weights_a = ops.softmax(scores_a, axis=-1)
        if self.dropout_layer is not None:
            attn_weights_a = self.dropout_layer(attn_weights_a, training=training)
        attn_out_a = ops.matmul(attn_weights_a, v_mod_b_anchor)

        # Modality B (anchors + queries) attends to Modality A anchors
        scores_b = ops.matmul(q_mod_b, ops.transpose(k_mod_a_anchor, (0, 1, 3, 2))) * self.scale
        attn_weights_b = ops.softmax(scores_b, axis=-1)
        if self.dropout_layer is not None:
            attn_weights_b = self.dropout_layer(attn_weights_b, training=training)
        attn_out_b = ops.matmul(attn_weights_b, v_mod_a_anchor)

        # Combine outputs
        combined_out = ops.concatenate([attn_out_a, attn_out_b], axis=2)

        # Reshape and project
        _, _, total_seq_len, _ = combined_out.shape
        combined_out = ops.transpose(combined_out, (0, 2, 1, 3))
        # Use -1 for the batch dimension to be compatible with symbolic Keras tensors
        combined_out = ops.reshape(combined_out, (-1, total_seq_len, self.dim))

        return self.proj_dense(combined_out)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple, same as input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing all configuration parameters.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
