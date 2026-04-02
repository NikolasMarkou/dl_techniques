"""
Capsule-based dynamic routing mechanism for attention.

This layer replaces the standard softmax normalization in multi-head
attention with a structured, iterative agreement process inspired by Capsule
Networks. The core motivation is to move beyond the simple, independent
normalization of attention scores and instead model higher-order relationships
between different attention components. By doing so, it aims to produce more
robust and contextually-aware attention distributions through a "routing-by-
agreement" consensus.

Architecturally, the layer first computes the standard scaled dot-product
attention scores, which serve as initial "votes." These votes are then refined
through two concurrent routing mechanisms:

1.  **Vertical Routing (Head-wise):** For a single query token, the attention
    distributions from all ``H`` heads are treated as low-level capsules.
    Dynamic routing is then applied across these capsules, allowing different
    "perspectives" captured by each head to influence one another and form
    a consensus.

2.  **Horizontal Routing (Token-wise):** For a given query token, the
    attention scores from all source tokens are treated as input capsules.
    The routing mechanism allows these source-token "perspectives" to agree
    on a final attention distribution.

The **Dynamic Routing** algorithm iteratively refines coupling coefficients
``c = softmax(b)`` between lower-level (initial scores) and higher-level
(refined scores) capsules. Each iteration: (1) compute weighted sum ``s``,
(2) squash via ``v = squash(s) = ||s||^2 / (1 + ||s||^2) * s / ||s||``,
(3) update log-priors ``b`` by agreement (dot product of ``v`` and votes).

References:
    - Sabour, Frosst, & Hinton, 2017. Dynamic Routing Between Capsules.
      (https://arxiv.org/abs/1710.09829)
    - Duan, et al., 2019. Capsule-Transformer for Neural Machine Translation.
      (https://arxiv.org/abs/1909.04321)

"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CapsuleRoutingSelfAttention(keras.layers.Layer):
    """
    Capsule Routing Self-Attention mechanism from Capsule-Transformer.

    This layer extends standard multi-head self-attention by organizing attention weights
    into capsules and applying dynamic routing algorithms to obtain better contextualized
    attention distributions. The implementation includes both vertical (head-wise) and
    horizontal (token-wise) capsule routing with optional positional constraints.

    Standard attention computes ``A = softmax(QK^T / sqrt(d)) @ V``. This layer
    enhances the attention logits via capsule routing before the final softmax:
    ``A_enhanced = logits + CapsuleRoute_vertical(logits) + CapsuleRoute_horizontal(logits)``.
    The squashing function from capsule networks is used:
    ``squash(s) = ||s||^2 / (1 + ||s||^2) * s / ||s||``.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────┐
        │              CapsuleRoutingSelfAttention                    │
        │                                                             │
        │  Input [B, seq, embed_dim]                                  │
        │         │                                                   │
        │         ├──────────────┬──────────────┐                     │
        │         ▼              ▼              ▼                     │
        │    ┌────────┐    ┌────────┐    ┌────────┐                   │
        │    │ Q Proj │    │ K Proj │    │ V Proj │                   │
        │    └───┬────┘    └───┬────┘    └───┬────┘                   │
        │        │             │             │                        │
        │        ▼             ▼             │                        │
        │     scores = Q @ K^T / sqrt(d_k)   │                        │
        │        │                           │                        │
        │        ├───────────────────┐       │                        │
        │        ▼                   ▼       │                        │
        │  ┌──────────────┐  ┌──────────────┐│                        │
        │  │  Vertical    │  │ Horizontal   ││                        │
        │  │  Routing     │  │ Routing      ││                        │
        │  │ (head-wise)  │  │ (token-wise) ││                        │
        │  └──────┬───────┘  └──────┬───────┘│                        │
        │         │                 │        │                        │
        │         └────────┬────────┘        │                        │
        │                  ▼                 │                        │
        │        scores + routing_output     │                        │
        │                  │                 │                        │
        │                  ▼                 │                        │
        │           [+ attention_mask]       │                        │
        │                  │                 │                        │
        │                  ▼                 │                        │
        │           softmax ──► dropout      │                        │
        │                  │                 │                        │
        │                  ▼                 ▼                        │
        │              weights @ V                                    │
        │                  │                                          │
        │                  ▼                                          │
        │           Output Projection                                 │
        │                  │                                          │
        │                  ▼                                          │
        │        Output [B, seq, embed_dim]                           │
        └─────────────────────────────────────────────────────────────┘

    :param num_heads: Integer, number of attention heads. Must be positive and should
        divide embed_dim evenly for optimal performance.
    :type num_heads: int
    :param key_dim: Optional integer, size of each attention head for query and key.
        If None, defaults to ``embed_dim // num_heads``. Must be positive.
    :type key_dim: Optional[int]
    :param value_dim: Optional integer, size of each attention head for value.
        If None, defaults to key_dim. Must be positive.
    :type value_dim: Optional[int]
    :param dropout_rate: Float, dropout rate applied to attention weights. Must be
        in range [0, 1]. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: String or Initializer instance for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: String or Initializer instance for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights. Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights. Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param activity_regularizer: Optional regularizer for layer output. Defaults to None.
    :type activity_regularizer: Optional[regularizers.Regularizer]
    :param routing_iterations: Integer, number of dynamic routing iterations. Must be
        positive. Higher values allow more sophisticated routing but increase
        computational cost. Defaults to 3.
    :type routing_iterations: int
    :param use_vertical_routing: Boolean, whether to apply vertical (head-wise) capsule
        routing. Enables information aggregation across attention heads.
        Defaults to True.
    :type use_vertical_routing: bool
    :param use_horizontal_routing: Boolean, whether to apply horizontal (token-wise)
        capsule routing. Enables information aggregation across sequence positions.
        Defaults to True.
    :type use_horizontal_routing: bool
    :param use_positional_routing: Boolean, whether to use positional routing constraints
        for horizontal capsules. When True, tokens can only route information from
        previous positions, preserving sequential order. Defaults to True.
    :type use_positional_routing: bool
    :param epsilon: Float, small constant for numerical stability in norm calculations.
        Must be positive. Defaults to 1e-8.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If num_heads, key_dim, or value_dim is not positive.
    :raises ValueError: If dropout_rate is not in range [0, 1].
    :raises ValueError: If routing_iterations is not positive.
    :raises ValueError: If epsilon is not positive.
    :raises ValueError: If embed_dim is not divisible by num_heads (when key_dim is None).
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        routing_iterations: int = 3,
        use_vertical_routing: bool = True,
        use_horizontal_routing: bool = True,
        use_positional_routing: bool = True,
        epsilon: float = 1e-8,
        **kwargs: Any
    ) -> None:
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Validate inputs
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if key_dim is not None and key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if value_dim is not None and value_dim <= 0:
            raise ValueError(f"value_dim must be positive, got {value_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if routing_iterations <= 0:
            raise ValueError(f"routing_iterations must be positive, got {routing_iterations}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store ALL configuration parameters for complete serialization
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.routing_iterations = routing_iterations
        self.use_vertical_routing = use_vertical_routing
        self.use_horizontal_routing = use_horizontal_routing
        self.use_positional_routing = use_positional_routing
        self.epsilon = epsilon

        # These will be set in build() based on input shape
        self.embed_dim = None
        self.actual_key_dim = None
        self.actual_value_dim = None

        # Create ALL sub-layers in __init__ (modern Keras 3 pattern)
        # Note: Dense layers will be properly configured in build() when we know embed_dim
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.output_dense = None
        self.dropout_layer = layers.Dropout(self.dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create all sub-components.

        Creates weight variables for both the layer and its sub-layers, ensuring
        proper serialization compatibility by explicitly building each sub-layer
        in computational order.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, embed_dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If input is not 3D or dimensions are incompatible.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {input_shape}")

        self.embed_dim = input_shape[-1]
        if self.embed_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Set actual dimensions based on configuration
        self.actual_key_dim = self.key_dim if self.key_dim is not None else self.embed_dim // self.num_heads
        self.actual_value_dim = self.value_dim if self.value_dim is not None else self.actual_key_dim

        # Validate dimension compatibility
        if self.key_dim is None and self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}) "
                f"when key_dim is None"
            )

        # Create projection layers now that we know dimensions
        self.query_dense = layers.Dense(
            self.num_heads * self.actual_key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="query"
        )

        self.key_dense = layers.Dense(
            self.num_heads * self.actual_key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="key"
        )

        self.value_dense = layers.Dense(
            self.num_heads * self.actual_value_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="value"
        )

        self.output_dense = layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output"
        )

        # Build sub-layers explicitly in computational order for robust serialization
        batch_size, seq_len, _ = input_shape

        # Build projection layers
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)

        # Output dense receives concatenated multi-head values
        output_input_shape = (batch_size, seq_len, self.num_heads * self.actual_value_dim)
        self.output_dense.build(output_input_shape)

        # Dropout operates on attention weights: (batch, num_heads, seq_len, seq_len)
        dropout_input_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.dropout_layer.build(dropout_input_shape)

        # Create vertical routing parameters if enabled
        if self.use_vertical_routing:
            self.vertical_aggregation_weights = self.add_weight(
                name="vertical_aggregation_weights",
                shape=(self.num_heads, self.num_heads),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )

            if self.use_bias:
                self.vertical_aggregation_bias = self.add_weight(
                    name="vertical_aggregation_bias",
                    shape=(self.num_heads,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=True
                )
            else:
                self.vertical_aggregation_bias = None
        else:
            self.vertical_aggregation_weights = None
            self.vertical_aggregation_bias = None

        # Always call parent build at the END
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of capsule routing self-attention.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, embed_dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor. Can be
            ``(batch_size, seq_len)`` for padding mask or
            ``(batch_size, seq_len, seq_len)`` for causal/custom mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating training mode for dropout.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, embed_dim)`` with
            contextualized representations enhanced by capsule routing.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Linear projections for Q, K, V
        query = self.query_dense(inputs)  # (batch, seq_len, num_heads * key_dim)
        key = self.key_dense(inputs)      # (batch, seq_len, num_heads * key_dim)
        value = self.value_dense(inputs)  # (batch, seq_len, num_heads * value_dim)

        # Reshape to multi-head format
        query = ops.reshape(query, (batch_size, seq_len, self.num_heads, self.actual_key_dim))
        key = ops.reshape(key, (batch_size, seq_len, self.num_heads, self.actual_key_dim))
        value = ops.reshape(value, (batch_size, seq_len, self.num_heads, self.actual_value_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim)
        query = ops.transpose(query, [0, 2, 1, 3])
        key = ops.transpose(key, [0, 2, 1, 3])
        value = ops.transpose(value, [0, 2, 1, 3])

        # Compute scaled dot-product attention logits
        attention_logits = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2]))
        attention_logits = attention_logits / ops.sqrt(
            ops.cast(self.actual_key_dim, attention_logits.dtype)
        )

        # Apply capsule routing enhancements
        routing_output = attention_logits

        if self.use_vertical_routing:
            vertical_output = self._vertical_routing(attention_logits)
            routing_output = routing_output + vertical_output

        if self.use_horizontal_routing:
            horizontal_output = self._horizontal_routing(attention_logits)
            routing_output = routing_output + horizontal_output

        # Apply attention mask if provided
        if attention_mask is not None:
            routing_output = self._apply_attention_mask(routing_output, attention_mask)

        # Convert to attention weights and apply dropout
        attention_weights = ops.softmax(routing_output, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)

        # Apply attention to values
        attended_values = ops.matmul(attention_weights, value)
        # Shape: (batch, num_heads, seq_len, value_dim)

        # Transpose and reshape to concatenate heads
        attended_values = ops.transpose(attended_values, [0, 2, 1, 3])
        # Shape: (batch, seq_len, num_heads, value_dim)

        concatenated = ops.reshape(
            attended_values, (batch_size, seq_len, self.num_heads * self.actual_value_dim)
        )

        # Final linear projection
        output = self.output_dense(concatenated)
        return output

    def _apply_attention_mask(
        self,
        attention_logits: keras.KerasTensor,
        attention_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply attention mask to logits.

        :param attention_logits: Attention logits of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_logits: keras.KerasTensor
        :param attention_mask: Attention mask tensor.
        :type attention_mask: keras.KerasTensor
        :return: Masked attention logits.
        :rtype: keras.KerasTensor
        """
        # Expand mask to match attention shape
        if len(attention_mask.shape) == 2:
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
        elif len(attention_mask.shape) == 3:
            # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            attention_mask = ops.expand_dims(attention_mask, 1)

        # Apply mask (set masked positions to large negative value)
        mask_value = ops.cast(-1e9, attention_logits.dtype)
        return ops.where(attention_mask, attention_logits, mask_value)

    def _squash(self, vectors: keras.KerasTensor) -> keras.KerasTensor:
        """
        Squashing function from capsule networks.

        Applies the non-linearity: ``v = ||s||^2 / (1 + ||s||^2) * s / ||s||``.

        :param vectors: Input vectors to squash.
        :type vectors: keras.KerasTensor
        :return: Squashed vectors with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Calculate squared norm along last axis
        squared_norm = ops.sum(ops.square(vectors), axis=-1, keepdims=True)
        norm = ops.sqrt(squared_norm + self.epsilon)

        # Apply squashing transformation
        scale = squared_norm / (1 + squared_norm)
        return scale * vectors / norm

    def _dynamic_routing(
        self,
        vote_vectors: keras.KerasTensor,
        num_output_capsules: int
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply dynamic routing algorithm between capsules.

        Implements the iterative routing-by-agreement algorithm that computes
        coupling coefficients based on agreement between prediction vectors
        and output capsules.

        :param vote_vectors: Vote vectors of shape
            ``(..., num_input, num_output, capsule_dim)``.
        :type vote_vectors: keras.KerasTensor
        :param num_output_capsules: Number of output capsules.
        :type num_output_capsules: int
        :return: Tuple of ``(output_capsules, routing_weights)`` where
            output_capsules are final capsule outputs after routing and
            routing_weights are final routing coefficients.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Get input dimensions for routing logits initialization
        input_shape = ops.shape(vote_vectors)
        batch_dims = input_shape[:-3]
        num_input_capsules = input_shape[-3]

        # Initialize routing logits to zero (uniform initial routing)
        # FIX: The original code used ops.concatenate, which fails if ops.shape
        # returns a Python tuple (e.g., in eager execution or with static shapes).
        # Using standard tuple concatenation is robust for shape construction.
        routing_logits_shape = batch_dims + (num_input_capsules, num_output_capsules)
        routing_logits = ops.zeros(shape=routing_logits_shape, dtype=vote_vectors.dtype)

        # Iterative routing algorithm
        for iteration in range(self.routing_iterations):
            # Compute coupling coefficients via softmax over input capsules
            routing_weights = ops.softmax(routing_logits, axis=-2)

            # Expand routing weights for broadcasting with vote vectors
            routing_weights_expanded = ops.expand_dims(routing_weights, axis=-1)

            # Compute weighted sum of vote vectors (s_j = sum_i c_ij * u_j|i)
            weighted_votes = routing_weights_expanded * vote_vectors
            output_capsules = ops.sum(weighted_votes, axis=-3)

            # Apply squashing function to get final capsule outputs
            output_capsules = self._squash(output_capsules)

            # Update routing logits based on agreement (except on last iteration)
            if iteration < self.routing_iterations - 1:
                # Expand output capsules for agreement calculation
                output_expanded = ops.expand_dims(output_capsules, axis=-3)

                # Calculate agreement: dot product between votes and outputs
                agreement = ops.sum(vote_vectors * output_expanded, axis=-1)
                routing_logits = routing_logits + agreement

        return output_capsules, routing_weights

    def _vertical_routing(self, attention_weights: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply vertical (head-wise) capsule routing.

        Treats attention heads as capsules and applies dynamic routing to aggregate
        information across different attention perspectives for each query position.

        :param attention_weights: Attention weights of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_weights: keras.KerasTensor
        :return: Vertical routing output of same shape as input.
        :rtype: keras.KerasTensor
        """
        # Reshape for routing: treat each query position independently
        # (batch, num_heads, seq_len_q, seq_len_k) -> (batch, seq_len_q, num_heads, seq_len_k)
        attention_reshaped = ops.transpose(attention_weights, [0, 2, 1, 3])

        # Create vote vectors: each input head votes for each output head
        # Shape: (batch, seq_len_q, num_heads_in, num_heads_out, seq_len_k)
        vote_vectors = ops.expand_dims(attention_reshaped, axis=3)
        vote_vectors = ops.repeat(vote_vectors, self.num_heads, axis=3)

        # Apply dynamic routing over heads
        output_capsules, _ = self._dynamic_routing(vote_vectors, self.num_heads)
        # output_capsules shape: (batch, seq_len_q, num_heads_out, seq_len_k)

        # Apply learned aggregation weights if available
        if self.vertical_aggregation_weights is not None:
            # Transpose for matrix multiplication: (batch, seq_len_q, seq_len_k, num_heads)
            output_transposed = ops.transpose(output_capsules, [0, 1, 3, 2])

            # Apply linear transformation: (..., num_heads) @ (num_heads, num_heads)
            aggregated = ops.matmul(output_transposed, self.vertical_aggregation_weights)

            if self.vertical_aggregation_bias is not None:
                aggregated = aggregated + self.vertical_aggregation_bias

            # Apply softmax to get importance weights
            importance_weights = ops.softmax(aggregated, axis=-1)

            # Weight the output capsules and transpose back
            weighted_output = importance_weights * output_transposed
            vertical_output = ops.transpose(weighted_output, [0, 3, 1, 2])
        else:
            # Reshape back to original attention format
            vertical_output = ops.transpose(output_capsules, [0, 2, 1, 3])

        return vertical_output

    def _horizontal_routing(self, attention_weights: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply horizontal (token-wise) capsule routing with optional positional constraints.

        Treats sequence tokens as capsules and applies dynamic routing to aggregate
        information across token positions, with optional causal masking to preserve
        sequential information flow.

        :param attention_weights: Attention weights of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_weights: keras.KerasTensor
        :return: Horizontal routing output of same shape as input.
        :rtype: keras.KerasTensor
        """
        seq_len = ops.shape(attention_weights)[2]

        if self.use_positional_routing:
            # Apply positional constraints: each position can only route from previous positions
            routed_rows = []

            for l in range(seq_len):
                if l == 0:
                    # First position: no routing needed
                    routed_row = attention_weights[:, :, :1, :]
                else:
                    # Extract attention for positions up to l (including l)
                    pos_attention = attention_weights[:, :, :l + 1, :]
                    # Shape: (batch, num_heads, l + 1, seq_len)

                    # Create vote vectors: each token's attention is a vote
                    vote_vectors = ops.expand_dims(pos_attention, axis=-2)
                    # Shape: (batch, num_heads, l + 1, 1, seq_len)

                    # Apply routing to aggregate information from tokens <= l
                    output_capsules, _ = self._dynamic_routing(vote_vectors, 1)
                    routed_row = output_capsules

                routed_rows.append(routed_row)

            # Reconstruct full attention matrix
            horizontal_output = ops.concatenate(routed_rows, axis=2)
        else:
            # Standard horizontal routing without positional constraints
            # Reshape: (batch, seq_len, num_heads, seq_len)
            attention_reshaped = ops.transpose(attention_weights, [0, 2, 1, 3])

            # Create vote vectors: (batch, seq_len, num_heads, num_heads, seq_len)
            vote_vectors = ops.expand_dims(attention_reshaped, axis=-2)
            vote_vectors = ops.repeat(vote_vectors, self.num_heads, axis=-2)

            # Apply routing
            output_capsules, _ = self._dynamic_routing(vote_vectors, self.num_heads)

            # Reshape back: (batch, num_heads, seq_len, seq_len)
            horizontal_output = ops.transpose(output_capsules, [0, 2, 1, 3])

        return horizontal_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, identical to input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all initialization parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'routing_iterations': self.routing_iterations,
            'use_vertical_routing': self.use_vertical_routing,
            'use_horizontal_routing': self.use_horizontal_routing,
            'use_positional_routing': self.use_positional_routing,
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
