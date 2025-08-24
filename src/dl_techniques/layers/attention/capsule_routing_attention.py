"""
Capsule Routing Self-Attention Layer for Capsule-Transformer

Implementation based on "Capsule-Transformer for Neural Machine Translation" by Duan et al.
This layer extends standard multi-head self-attention with capsule routing mechanisms,
organizing attention weights into vertical (head-wise) and horizontal (token-wise) capsules
and applying dynamic routing to obtain better contextualized representations.

Key innovations:
- Vertical capsules: Head-wise attention weight organization
- Horizontal capsules: Token-wise attention weight organization
- Dynamic routing: Iterative capsule aggregation algorithm
- Positional routing: Sequential information preservation for horizontal capsules
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
    horizontal (token-wise) capsule routing.

    The layer follows the architecture described in "Capsule-Transformer for Neural Machine
    Translation" where attention weights are viewed as capsules and the dynamic routing
    algorithm from capsule networks is applied to aggregate information across different
    perspectives.

    Args:
        num_heads: Integer, number of attention heads. Must be positive and divide embed_dim evenly.
        key_dim: Integer, size of each attention head for query and key. If None, defaults to embed_dim // num_heads.
        value_dim: Optional integer, size of each attention head for value. If None, defaults to key_dim.
        dropout: Float, dropout rate for attention weights. Must be between 0 and 1. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
        kernel_initializer: String or Initializer, initializer for kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        activity_regularizer: Optional regularizer for layer output.
        routing_iterations: Integer, number of routing iterations. Must be positive. Defaults to 3.
        use_vertical_routing: Boolean, whether to use vertical (head-wise) routing. Defaults to True.
        use_horizontal_routing: Boolean, whether to use horizontal (token-wise) routing. Defaults to True.
        use_positional_routing: Boolean, whether to use positional routing for horizontal capsules. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-8.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape (batch_size, sequence_length, embed_dim).

    Output shape:
        3D tensor with shape (batch_size, sequence_length, embed_dim).

    Example:
        ```python
        # Basic usage
        attention = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=64
        )

        # Advanced configuration
        attention = CapsuleRoutingSelfAttention(
            num_heads=12,
            key_dim=64,
            dropout=0.1,
            routing_iterations=5,
            use_vertical_routing=True,
            use_horizontal_routing=True,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer block
        inputs = keras.Input(shape=(seq_len, embed_dim))
        attended = attention(inputs)
        ```

    Note:
        The layer implements the capsule routing mechanism as described in the paper,
        where attention weights are organized into vertical and horizontal capsules,
        and dynamic routing is applied to aggregate information. The routing process
        uses the squashing function from the original capsule network paper.

    References:
        - Capsule-Transformer for Neural Machine Translation (Duan et al., 2019)
        - Dynamic Routing Between Capsules (Sabour et al., 2017)
    """

    def __init__(
            self,
            num_heads: int,
            key_dim: Optional[int] = None,
            value_dim: Optional[int] = None,
            dropout: float = 0.0,
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
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if routing_iterations <= 0:
            raise ValueError(f"routing_iterations must be positive, got {routing_iterations}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout
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

        # Will be set in build()
        self.embed_dim = None
        self.actual_key_dim = None
        self.actual_value_dim = None

        # Projection layers - created in build()
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.output_dense = None
        self.dropout_layer = None

        # Routing parameters - created in build()
        self.vertical_aggregation_weights = None
        self.vertical_aggregation_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and create all sub-components."""
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {input_shape}")

        self.embed_dim = input_shape[-1]
        if self.embed_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Set actual dimensions
        self.actual_key_dim = self.key_dim if self.key_dim is not None else self.embed_dim // self.num_heads
        self.actual_value_dim = self.value_dim if self.value_dim is not None else self.actual_key_dim

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")

        # Create projection layers
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

        self.dropout_layer = layers.Dropout(self.dropout)

        # Build projection layers explicitly
        batch_size, seq_len, _ = input_shape
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        self.output_dense.build((batch_size, seq_len, self.num_heads * self.actual_value_dim))
        self.dropout_layer.build((batch_size, self.num_heads, seq_len, seq_len))

        # Create vertical routing parameters if needed
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

        super().build(input_shape)

    def _squash(self, vectors: keras.KerasTensor) -> keras.KerasTensor:
        """
        Squashing function from capsule networks.

        Args:
            vectors: Input vectors to squash.

        Returns:
            Squashed vectors with same shape as input.
        """
        # Calculate squared norm along last axis
        squared_norm = ops.sum(ops.square(vectors), axis=-1, keepdims=True)
        norm = ops.sqrt(squared_norm + self.epsilon)

        # Apply squashing: v * ||v||^2 / (1 + ||v||^2) / ||v||
        scale = squared_norm / (1 + squared_norm)
        return scale * vectors / norm

    def _dynamic_routing(
            self,
            vote_vectors: keras.KerasTensor,
            num_output_capsules: int
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply dynamic routing algorithm.

        Args:
            vote_vectors: Vote vectors of shape (..., num_input, num_output, capsule_dim).
            num_output_capsules: Number of output capsules.

        Returns:
            Tuple of (output_capsules, routing_weights).
        """
        # Get input dimensions
        input_shape = ops.shape(vote_vectors)
        batch_dims = input_shape[:-3]
        num_input_capsules = input_shape[-3]

        # Initialize routing logits to zero
        routing_logits_shape = batch_dims + (num_input_capsules, num_output_capsules)
        routing_logits = ops.zeros(shape=routing_logits_shape, dtype="float32")

        # Iterative routing
        for iteration in range(self.routing_iterations):
            # To create aggregation weights, softmax should be over the input capsules.
            softmax_axis = -2  # num_input_capsules dimension
            routing_weights = ops.softmax(routing_logits, axis=softmax_axis)

            # Expand routing weights for broadcasting
            routing_weights_expanded = ops.expand_dims(routing_weights, axis=-1)

            # Compute weighted sum of vote vectors
            weighted_votes = routing_weights_expanded * vote_vectors
            output_capsules = ops.sum(weighted_votes, axis=-3)

            # Apply squashing function
            output_capsules = self._squash(output_capsules)

            # Update routing logits (except on last iteration)
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

        Args:
            attention_weights: Attention weights of shape (batch, num_heads, seq_len, seq_len).

        Returns:
            Vertical routing output of same shape as input.
        """
        # attention_weights shape: (batch, num_heads, seq_len_q, seq_len_k)

        # Reshape for routing: treat each query position independently.
        # This treats heads as input capsules for each query position.
        # New shape: (batch, seq_len_q, num_heads, seq_len_k)
        attention_reshaped = ops.transpose(attention_weights, [0, 2, 1, 3])

        # Create vote vectors. Each input capsule (head) votes for each output capsule (head).
        # Since the paper implies u_j|i = u_j, we just replicate the input capsules.
        # Shape: (batch, seq_len_q, num_heads_in, 1, seq_len_k)
        vote_vectors = ops.expand_dims(attention_reshaped, axis=3)
        # Shape: (batch, seq_len_q, num_heads_in, num_heads_out, seq_len_k)
        vote_vectors = ops.repeat(vote_vectors, self.num_heads, axis=3)

        # Apply dynamic routing over the heads.
        # num_output_capsules is self.num_heads.
        output_capsules, _ = self._dynamic_routing(vote_vectors, self.num_heads)
        # output_capsules shape: (batch, seq_len_q, num_heads_out, seq_len_k)

        # Apply learned aggregation weights
        if self.vertical_aggregation_weights is not None:
            # Transpose to (batch, seq_len_q, seq_len_k, num_heads_out) for aggregation
            output_transposed = ops.transpose(output_capsules, [0, 1, 3, 2])

            # Apply linear transformation: (..., seq_len_k, num_heads) @ (num_heads, num_heads)
            aggregated = ops.matmul(output_transposed, self.vertical_aggregation_weights)

            if self.vertical_aggregation_bias is not None:
                aggregated = aggregated + self.vertical_aggregation_bias

            # Apply softmax to get importance weights
            # Shape: (batch, seq_len_q, seq_len_k, num_heads)
            importance_weights = ops.softmax(aggregated, axis=-1)

            # Weight the output capsules
            weighted_output = importance_weights * output_transposed

            # Reshape back to original attention format: (batch, num_heads, seq_len_q, seq_len_k)
            vertical_output = ops.transpose(weighted_output, [0, 3, 1, 2])
        else:
            # Reshape back to original attention format: (batch, num_heads, seq_len_q, seq_len_k)
            vertical_output = ops.transpose(output_capsules, [0, 2, 1, 3])

        return vertical_output

    def _horizontal_routing(self, attention_weights: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply horizontal (token-wise) capsule routing with positional constraints.

        Args:
            attention_weights: Attention weights of shape (batch, num_heads, seq_len, seq_len).

        Returns:
            Horizontal routing output of same shape as input.
        """
        seq_len = ops.shape(attention_weights)[2]

        if self.use_positional_routing:
            # For each position l, only use information from positions <= l
            routed_rows = []
            for l in range(seq_len):
                if l == 0:
                    # First position: no routing needed, just use original weights for the first query row
                    # Shape: (batch, num_heads, 1, seq_len)
                    routed_row = attention_weights[:, :, :1, :]
                else:
                    # Extract relevant attention scores for queries up to position l
                    # Shape: (batch, num_heads, l + 1, seq_len)
                    pos_attention = attention_weights[:, :, :l + 1, :]

                    # Create vote vectors from tokens <= l. Each token's attention is a vote.
                    # Shape: (batch, num_heads, l + 1, 1, seq_len)
                    vote_vectors = ops.expand_dims(pos_attention, axis=-2)

                    # Apply routing to aggregate information from tokens <= l for each head independently.
                    # Routing is over the l+1 input capsules (tokens).
                    # `_dynamic_routing` expects (..., num_input, num_output, capsule_dim)
                    # vote_vectors shape is (batch, num_heads, l+1, 1, seq_len)
                    # This means batch_dims=(batch, num_heads), num_input=l+1, num_output=1, capsule_dim=seq_len.
                    # This correctly routes over the l+1 positions to get one output capsule.
                    output_capsules, _ = self._dynamic_routing(vote_vectors, 1)
                    routed_row = output_capsules

                routed_rows.append(routed_row)

            # Reconstruct full attention matrix by stacking the routed rows
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

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of capsule routing self-attention.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Optional attention mask.
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Linear projections
        query = self.query_dense(inputs)  # (batch, seq_len, num_heads * key_dim)
        key = self.key_dense(inputs)  # (batch, seq_len, num_heads * key_dim)
        value = self.value_dense(inputs)  # (batch, seq_len, num_heads * value_dim)

        # Reshape to multi-head format
        query = ops.reshape(query, (batch_size, seq_len, self.num_heads, self.actual_key_dim))
        key = ops.reshape(key, (batch_size, seq_len, self.num_heads, self.actual_key_dim))
        value = ops.reshape(value, (batch_size, seq_len, self.num_heads, self.actual_value_dim))

        # Transpose to (batch, num_heads, seq_len, key_dim)
        query = ops.transpose(query, [0, 2, 1, 3])
        key = ops.transpose(key, [0, 2, 1, 3])
        value = ops.transpose(value, [0, 2, 1, 3])

        # Compute scaled dot-product attention weights
        # (batch, num_heads, seq_len, seq_len)
        attention_logits = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2]))
        attention_logits = attention_logits / ops.sqrt(ops.cast(self.actual_key_dim, attention_logits.dtype))

        # Apply capsule routing to attention weights
        routing_output = attention_logits

        if self.use_vertical_routing:
            vertical_output = self._vertical_routing(attention_logits)
            routing_output = routing_output + vertical_output

        if self.use_horizontal_routing:
            horizontal_output = self._horizontal_routing(attention_logits)
            routing_output = routing_output + horizontal_output

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention shape
            if len(attention_mask.shape) == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                attention_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
            elif len(attention_mask.shape) == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                attention_mask = ops.expand_dims(attention_mask, 1)

            # Apply mask (set masked positions to large negative value)
            mask_value = -1e9
            routing_output = ops.where(attention_mask, routing_output, mask_value)

        # Apply softmax to get attention weights
        attention_weights = ops.softmax(routing_output, axis=-1)

        # Apply dropout
        attention_weights = self.dropout_layer(attention_weights, training=training)

        # Apply attention to values
        # (batch, num_heads, seq_len, value_dim)
        attended_values = ops.matmul(attention_weights, value)

        # Transpose and reshape to concatenate heads
        # (batch, seq_len, num_heads, value_dim)
        attended_values = ops.transpose(attended_values, [0, 2, 1, 3])

        # Concatenate heads: (batch, seq_len, num_heads * value_dim)
        concatenated = ops.reshape(attended_values, (batch_size, seq_len, self.num_heads * self.actual_value_dim))

        # Final linear projection
        output = self.output_dense(concatenated)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'dropout': self.dropout,
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