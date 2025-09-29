"""Modern Hopfield Network Layer with Iterative Updates.

This layer implements a Modern Hopfield Network, as described in the paper
"Hopfield Networks is All You Need" [1]. It functions as a content-addressable
associative memory, capable of storing and retrieving a large number of patterns.

The core of the layer is the scaled dot-product attention mechanism from the
Transformer architecture, which serves as the update rule for the network's
state. Unlike a standard attention layer, which performs a single, feed-forward
computation, this layer can apply the attention mechanism iteratively. An
initial query (a "probe" or noisy pattern) is repeatedly refined until it
converges to one of the stored patterns (the "memories"), which are represented
by the Key-Value pairs.

Difference from Standard Transformer Attention:
---------------------------------------------
The fundamental difference is the computational flow:

| Feature                | Standard Transformer Attention      | Modern Hopfield Network (This Layer)    |
|:-----------------------|:------------------------------------|:----------------------------------------|
| **Computational Flow** | Single-step, feed-forward           | Iterative, recurrent until convergence  |
| **Primary Goal**       | Contextual information weighting    | Associative pattern retrieval & cleaning|
| **Mechanism**          | `output = attention(Q, K, V)`       | `state_t+1 = attention(state_t, K, V)`  |
| **Query (Q)**          | Static input representation         | Dynamic state vector that evolves       |

In essence, a standard transformer attention layer performs a single update
step (`update_steps_max=0` in this implementation) of a modern Hopfield network.
This layer generalizes that concept by allowing for multiple (`update_steps_max > 0`)
iterative updates, enabling it to function as a true associative memory that
converges to stable fixed-points (attractors).

Convergence Criteria:
---------------------
The iterative update process, which runs when `update_steps_max > 0`,
terminates based on one of the following conditions:

1.  **Maximum Steps Reached:** The loop hard-stops after completing the number
    of iterations specified by `update_steps_max`. This acts as a safeguard
    to control computation time and prevent infinite loops.

2.  **State Convergence:** The network is considered to have converged if the
    change between consecutive states becomes negligibly small. This is
    measured by calculating the **Frobenius norm** of the difference between
    the attention matrix of the current step and the previous step. If this
    norm falls below the threshold defined by `update_steps_eps`, the
    iteration stops. This indicates that the system has settled into a
    stable fixed-point attractor.

References:
    [1] Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need".
        arXiv:2008.02217.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Any, List, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HopfieldAttention(keras.layers.Layer):
    """
    Modern Hopfield layer implementation as described in 'Hopfield Networks is All You Need'.

    This layer implements a modern Hopfield network that can store exponentially many patterns
    and converges with one update. It uses a transformer-like attention mechanism as its core
    operation. For detailed theoretical background and implementation details, please refer to
    the accompanying documentation file.

    Args:
        num_heads: Integer, number of attention heads. Must be positive.
        key_dim: Integer, size of each attention head for key and query. Must be positive.
        value_dim: Optional integer, size of each attention head for value.
            If None, defaults to key_dim.
        dropout_rate: Float, dropout probability for attention weights.
            Must be between 0.0 and 1.0. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in the attention projections.
            Defaults to True.
        kernel_initializer: String or initializer instance for projection matrices.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer instance for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for projection matrices.
        bias_regularizer: Optional regularizer for bias vectors.
        activity_regularizer: Optional regularizer for layer output.
        normalize_patterns: Boolean, whether to apply layer normalization to patterns.
            Defaults to True.
        update_steps_max: Integer, maximum number of association update steps.
            0 means single-step (standard attention). Must be non-negative.
            Defaults to 0.
        update_steps_eps: Float, minimum difference threshold between update steps
            for convergence detection. Must be positive. Defaults to 1e-4.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - Single tensor: (batch_size, seq_len, input_dim) for self-attention
        - List of tensors: [query, key, value] where each has shape
          (batch_size, seq_len, input_dim)

    Output shape:
        Tensor with shape (batch_size, seq_len_query, input_dim).

    Attributes:
        query_dense: Dense layer for query projection.
        key_dense: Dense layer for key projection.
        value_dense: Dense layer for value projection.
        output_dense: Dense layer for final output projection.
        dropout_layer: Dropout layer for attention weights (if dropout_rate > 0).
        layernorm: LayerNormalization for pattern normalization (if normalize_patterns=True).

    Example:
        ```python
        # Self-attention with iterative updates (Hopfield dynamics)
        layer = HopfieldAttention(
            num_heads=8,
            key_dim=64,
            update_steps_max=3,
            normalize_patterns=True
        )
        inputs = keras.Input(shape=(128, 512))
        outputs = layer(inputs)

        # Cross-attention (standard attention behavior)
        layer = HopfieldAttention(
            num_heads=8,
            key_dim=64,
            update_steps_max=0  # Single step
        )
        query = keras.Input(shape=(32, 512))
        key_value = keras.Input(shape=(64, 512))
        outputs = layer([query, key_value, key_value])

        # With custom regularization
        layer = HopfieldAttention(
            num_heads=12,
            key_dim=64,
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            update_steps_max=5,
            update_steps_eps=1e-5
        )
        ```

    Raises:
        ValueError: If num_heads <= 0 or key_dim <= 0.
        ValueError: If dropout_rate is not in [0, 1].
        ValueError: If update_steps_max < 0.
        ValueError: If update_steps_eps <= 0.

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and explicitly built in build() for robust serialization.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        normalize_patterns: bool = True,
        update_steps_max: int = 0,
        update_steps_eps: float = 1e-4,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if update_steps_max < 0:
            raise ValueError(f"update_steps_max must be non-negative, got {update_steps_max}")
        if update_steps_eps <= 0:
            raise ValueError(f"update_steps_eps must be positive, got {update_steps_eps}")

        # Store ALL configuration parameters
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.normalize_patterns = normalize_patterns
        self.update_steps_max = update_steps_max
        self.update_steps_eps = update_steps_eps

        # CREATE all sub-layers in __init__ (they are unbuilt at this point)
        # Following Pattern 2: Composite Layer from the Modern Keras 3 guide

        # Projection layers - will be properly dimensioned in build()
        self.query_dense = keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="query_dense"
        )

        self.key_dense = keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="key_dense"
        )

        self.value_dense = keras.layers.Dense(
            self.num_heads * self.value_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="value_dense"
        )

        # Output projection - input_dim will be determined in build()
        # For now, we create it but it will be properly built later
        self.output_dense = keras.layers.Dense(
            0,  # Will be set correctly in build()
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="output_dense"
        )

        # Dropout layer (conditional creation)
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(
                self.dropout_rate,
                name="attention_dropout"
            )
        else:
            self.dropout_layer = None

        # Layer normalization (conditional creation)
        if self.normalize_patterns:
            self.layernorm = keras.layers.LayerNormalization(
                epsilon=1e-5,
                name="pattern_norm"
            )
        else:
            self.layernorm = None

        logger.info(f"Initialized HopfieldAttention with {num_heads} heads, "
                   f"key_dim={key_dim}, value_dim={self.value_dim}")

    def build(self, input_shape: Union[Tuple, List]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization
        following the Modern Keras 3 pattern.

        Args:
            input_shape: Shape of input tensor or list of shapes for [query, key, value].
        """
        # Handle different input formats to extract query shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
            # Check if this is a list of shapes or a single shape
            if isinstance(input_shape[0], (list, tuple)):
                # This is a list of shapes [query_shape, key_shape, value_shape]
                query_shape = input_shape[0]
            else:
                # This is a single shape tuple (None, 32, 512)
                query_shape = input_shape
        else:
            # Single input shape provided
            query_shape = input_shape

        input_dim = query_shape[-1]
        logger.debug(f"Building HopfieldAttention with input_dim={input_dim}")

        # Update output_dense to have correct units
        # We need to recreate it with the correct output dimension
        self.output_dense = keras.layers.Dense(
            input_dim,  # Output same dimension as input
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="output_dense"
        )

        # Build sub-layers explicitly in computational order
        # This ensures all weight variables exist before weight restoration

        # Build projection layers with input shape
        self.query_dense.build(query_shape)
        self.key_dense.build(query_shape)  # Assuming key has same shape as query for self-attention
        self.value_dense.build(query_shape)  # Assuming value has same shape as query for self-attention

        # Calculate intermediate shape for output projection
        projected_shape = list(query_shape)
        projected_shape[-1] = self.num_heads * self.value_dim
        self.output_dense.build(tuple(projected_shape))

        # Build conditional layers
        if self.dropout_layer is not None:
            # Dropout layer doesn't need explicit build as it doesn't have weights
            pass

        if self.layernorm is not None:
            # Build layer norm with the shape it will receive
            # LayerNorm will receive (batch, num_heads, seq_len, head_dim)
            norm_shape = (None, None, None, self.key_dim)
            self.layernorm.build(norm_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug("HopfieldAttention build completed")

    def _hopfield_update_step(
        self,
        query: keras.KerasTensor,
        key: keras.KerasTensor,
        value: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Perform one Hopfield update step using scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim).
            key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim).
            value: Value tensor of shape (batch, num_heads, seq_len_v, value_dim).
            attention_mask: Optional attention mask.
            training: Whether in training mode.

        Returns:
            Tuple of (updated_output, attention_weights).
        """
        # Scaled dot-product attention
        scale = ops.sqrt(ops.cast(self.key_dim, query.dtype))
        attention_scores = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2])) / scale

        # Handle mask processing
        actual_mask = None
        if isinstance(attention_mask, (list, tuple)):
            # If Keras passes a list, find the first non-None mask.
            for m in attention_mask:
                if m is not None:
                    actual_mask = m
                    break
        else:
            actual_mask = attention_mask

        if actual_mask is not None:
            mask_tensor = ops.cast(actual_mask, attention_scores.dtype)
            # Add heads dimension if missing for broadcasting.
            # attention_scores shape: (batch, num_heads, seq_len_q, seq_len_k)
            # A common mask shape is (batch, seq_len_q, seq_len_k).
            if len(ops.shape(mask_tensor)) == 3:
                mask_tensor = ops.expand_dims(mask_tensor, axis=1)

            # Add large negative values to masked positions
            attention_scores = attention_scores + (1.0 - mask_tensor) * -1e9

        attention_weights = ops.softmax(attention_scores, axis=-1)

        # Apply dropout if configured
        if self.dropout_layer is not None:
            attention_weights = self.dropout_layer(attention_weights, training=training)

        output = ops.matmul(attention_weights, value)
        return output, attention_weights

    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_scores: bool = False,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass of the Hopfield attention layer.

        Args:
            inputs: Input tensor or list of tensors [query, key, value].
                For self-attention, pass a single tensor.
                For cross-attention, pass [query, key, value] or [query, key_value].
            attention_mask: Optional attention mask tensor.
            return_attention_scores: Boolean, whether to return attention scores along with output.
            training: Optional boolean indicating training mode.

        Returns:
            If return_attention_scores=False: Output tensor of shape (batch_size, seq_len_query, input_dim).
            If return_attention_scores=True: Tuple of (output tensor, attention_weights tensor).
        """
        # Handle input formats
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3:
                query, key, value = inputs
            elif len(inputs) == 2:
                query, key = inputs
                value = key
            else:
                raise ValueError(f"Expected 2 or 3 inputs, got {len(inputs)}")
        else:
            # Self-attention case
            query = key = value = inputs

        # Get shapes using Keras ops
        batch_size = ops.shape(query)[0]
        query_len = ops.shape(query)[1]
        key_len = ops.shape(key)[1]

        # Linear projections using built sub-layers
        query_proj = self.query_dense(query)
        key_proj = self.key_dense(key)
        value_proj = self.value_dense(value)

        # Reshape to multi-head attention format
        query_proj = ops.reshape(
            query_proj,
            (batch_size, query_len, self.num_heads, self.key_dim)
        )
        key_proj = ops.reshape(
            key_proj,
            (batch_size, key_len, self.num_heads, self.key_dim)
        )
        value_proj = ops.reshape(
            value_proj,
            (batch_size, key_len, self.num_heads, self.value_dim)
        )

        # Transpose to (batch, num_heads, seq_len, dim)
        query_proj = ops.transpose(query_proj, [0, 2, 1, 3])
        key_proj = ops.transpose(key_proj, [0, 2, 1, 3])
        value_proj = ops.transpose(value_proj, [0, 2, 1, 3])

        # Apply layer normalization if enabled
        if self.layernorm is not None:
            query_proj = self.layernorm(query_proj, training=training)
            key_proj = self.layernorm(key_proj, training=training)

        # Initialize Hopfield update loop
        current_query = query_proj
        prev_attention = None
        update_step = 0

        # Hopfield iterative updates
        while True:
            # Perform one Hopfield update step
            output, attention_weights = self._hopfield_update_step(
                current_query, key_proj, value_proj, attention_mask, training
            )

            # If update_steps_max is 0, only do one step (standard attention)
            if self.update_steps_max == 0:
                break

            # Check convergence if we have previous attention weights
            if prev_attention is not None and self.update_steps_eps > 0:
                # Compute Frobenius norm of the difference
                diff = attention_weights - prev_attention
                diff_norm = ops.sqrt(ops.sum(ops.square(diff)))
                if diff_norm < self.update_steps_eps:
                    logger.debug(f"Hopfield converged after {update_step} steps")
                    break

            # Check maximum steps
            if update_step >= self.update_steps_max:
                logger.debug(f"Hopfield stopped at max steps: {self.update_steps_max}")
                break

            # Update for next iteration
            prev_attention = attention_weights
            update_step += 1

            # Use attention weights to update the query for next iteration
            # This implements the iterative Hopfield dynamics
            current_query = ops.matmul(attention_weights, key_proj)

        # Reshape output back to original format
        output = ops.transpose(output, [0, 2, 1, 3])  # (batch, seq_len, num_heads, value_dim)
        output = ops.reshape(output, (batch_size, query_len, self.num_heads * self.value_dim))

        # Final output projection
        output = self.output_dense(output)

        if return_attention_scores:
            return output, attention_weights
        return output

    def compute_output_shape(self, input_shape: Union[Tuple, List]) -> Tuple:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input or list of input shapes.

        Returns:
            Output shape tuple. Same as query input shape for Hopfield attention.
        """
        # Handle different input formats
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
            # Check if this is a list of shapes or a single shape
            if isinstance(input_shape[0], (list, tuple)):
                # This is a list of shapes [query_shape, key_shape, value_shape]
                query_shape = input_shape[0]
            else:
                # This is a single shape tuple (None, 32, 512)
                query_shape = input_shape
        else:
            # Single input shape provided
            query_shape = input_shape

        # Output has same shape as query input
        return tuple(query_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        This method must include ALL parameters passed to __init__ for proper
        serialization and deserialization.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
            "normalize_patterns": self.normalize_patterns,
            "update_steps_max": self.update_steps_max,
            "update_steps_eps": self.update_steps_eps,
        })
        return config

# ---------------------------------------------------------------------