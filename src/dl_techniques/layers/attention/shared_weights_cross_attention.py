"""
Implements a parameter-efficient, bidirectional cross-attention mechanism.

This layer facilitates information exchange between two distinct sets of
tokens (modalities) through a cross-attention pattern where the projection
weights for queries, keys, and values are shared. This design is highly
parameter-efficient and encourages the model to learn a common, modality-
invariant representation space for interaction.

Architecture:
    The core architectural principle is to force two different input
    modalities (e.g., surface features and volume features) to communicate
    using a single, shared set of transformation rules. The process is as
    follows:

    1.  **Shared Projection:** The input, a concatenation of tokens from
        modality A and modality B, is passed through a single, shared projection
        layer to generate queries (Q), keys (K), and values (V) for all
        tokens simultaneously.

    2.  **Bidirectional Cross-Attention:** The resulting Q, K, and V tensors
        are split according to the original modalities. A bidirectional
        attention is then computed:
        -   The queries of modality A attend to the keys and values of
            modality B.
        -   Simultaneously, the queries of modality B attend to the keys
            and values of modality A.

    3.  **Optional Anchor-Query Hierarchy:** The layer also supports a more
        complex, sparse attention pattern. In this mode, each modality is
        further divided into "anchors" and "queries." The anchors are
        intended to act as compressed summaries of their respective
        modalities. The attention flow is then modified such that all tokens
        from one modality (both its anchors and queries) attend *only* to the
        anchors of the opposing modality. This creates an efficient, two-
        tiered information exchange through a compact bottleneck.

Foundational Mathematics:
    The defining characteristic of this layer is its weight sharing. Let
    `f_q`, `f_k`, and `f_v` be the shared linear projection functions. For
    input sequences `X_A` and `X_B` from two modalities, the Q, K, and V
    vectors are computed using these same functions:

        Q_A, K_A, V_A = f_q(X_A), f_k(X_A), f_v(X_A)
        Q_B, K_B, V_B = f_q(X_B), f_k(X_B), f_v(X_B)

    The cross-attention outputs are then calculated as:

        Output_A = Attention(Q_A, K_B, V_B)
        Output_B = Attention(Q_B, K_A, V_A)

    This shared projection acts as a strong inductive bias, forcing the model
    to map both modalities into a common semantic space where their features
    can be meaningfully compared and combined. This is a form of parameter
    tying reminiscent of Siamese networks, which promotes generalization and
    is highly efficient when the modalities share underlying structural or
    semantic properties.

References:
    - The cross-attention mechanism is a cornerstone of encoder-decoder
      architectures:
      Vaswani, A., et al. (2017). "Attention Is All You Need".

    - The concept of weight sharing across different inputs to learn a
      common feature space is fundamental to Siamese Networks:
      Bromley, J., et al. (1994). "Signature verification using a Siamese
      time delay neural network".

    - The anchor-query structure is a form of sparse attention, related to
      models that use global tokens as information hubs:
      Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The
      Long-Document Transformer".
"""

import keras
from keras import ops
from typing import Any, List, Union, Tuple, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SharedWeightsCrossAttention(keras.layers.Layer):
    """Cross-attention between different modalities with shared weights.

    This layer implements efficient cross-attention where:
    - Different modalities (e.g., surface and volume data) cross-attend to each other
    - Weights are shared across modalities for parameter efficiency
    - Supports both anchor-only and anchor-query configurations

    The attention pattern is:
    - Modality A tokens attend to Modality B tokens
    - Modality B tokens attend to Modality A tokens
    - Optional query tokens attend to opposite modality anchors

    This is particularly useful for multi-modal learning where different
    data types need to exchange information efficiently.

    Args:
        dim: Integer, input/output dimension of the attention layer. Must be positive
            and divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive. Defaults to 8.
        dropout_rate: Float, dropout rate for attention weights. Must be between 0 and 1.
            Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
        kernel_initializer: String or initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer instance for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights. Defaults to None.
        bias_regularizer: Optional regularizer for bias weights. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, total_sequence_length, dim)`
        where total_sequence_length is the sum of all modality sequences

    Output shape:
        3D tensor with same shape as input.

    Call arguments:
        x: Input tensor of shape (batch_size, total_seq_len, dim) containing
           concatenated sequences from different modalities.
        split_sizes: List of integers specifying how to split the input:
            - Length 2: [modality_a_len, modality_b_len] for anchor-only mode
            - Length 4: [mod_a_anchor, mod_a_query, mod_b_anchor, mod_b_query]
        training: Boolean indicating training mode.

    Returns:
        Output tensor with cross-attended features, same shape as input.

    Raises:
        ValueError: If dim is not divisible by num_heads.
        ValueError: If dim or num_heads are not positive.
        ValueError: If dropout_rate is not between 0 and 1.

    Example:
        ```python
        # Multi-modal cross-attention (surface and volume data)
        surface_features = keras.random.normal((2, 100, 256))  # Surface data
        volume_features = keras.random.normal((2, 150, 256))   # Volume data

        # Concatenate modalities
        combined = ops.concatenate([surface_features, volume_features], axis=1)

        cross_attn = SharedWeightsCrossAttention(dim=256, num_heads=8)

        # Cross-attention between modalities
        output = cross_attn(combined, split_sizes=[100, 150])
        print(output.shape)  # (2, 250, 256)

        # With anchor-query structure
        surface_anchors = keras.random.normal((2, 50, 256))
        surface_queries = keras.random.normal((2, 50, 256))
        volume_anchors = keras.random.normal((2, 75, 256))
        volume_queries = keras.random.normal((2, 75, 256))

        combined_aq = ops.concatenate([
            surface_anchors, surface_queries,
            volume_anchors, volume_queries
        ], axis=1)

        output_aq = cross_attn(combined_aq, split_sizes=[50, 50, 75, 75])
        print(output_aq.shape)  # (2, 250, 256)
        ```
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
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
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
        """Apply shared weights cross-attention.

        Args:
            inputs: Input tensor of shape (batch_size, total_seq_len, dim).
            split_sizes: List specifying how to split input into modalities.
            attention_mask: Optional attention mask tensor.
            training: Boolean indicating training mode.

        Returns:
            Output tensor with cross-attended features.
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
        """Cross-attention between two modalities."""
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
        """Attention with anchor-query structure for two modalities."""
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
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Returns the layer configuration for serialization."""
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