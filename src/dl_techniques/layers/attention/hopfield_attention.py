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

Example:
    >>> # Self-attention with iterative updates
    >>> x = keras.random.normal((4, 32, 128))
    >>> hopfield_layer = HopfieldAttention(
    ...     num_heads=8, key_dim=16, update_steps_max=3
    ... )
    >>> output = hopfield_layer(x)
    >>> print(output.shape)
    (4, 32, 128)

    >>> # Cross-attention (single step, like standard attention)
    >>> query = keras.random.normal((4, 32, 128))
    >>> key_value = keras.random.normal((4, 64, 128))
    >>> hopfield_layer = HopfieldAttention(
    ...     num_heads=8, key_dim=16, update_steps_max=0
    ... )
    >>> output = hopfield_layer([query, key_value])
    >>> print(output.shape)
    (4, 32, 128)

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
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for key and query.
        value_dim: Size of each attention head for value. If None, defaults to key_dim.
        dropout_rate: Dropout probability for attention weights.
        use_bias: Whether to use bias in the attention projections.
        kernel_initializer: Initializer for the projection matrices.
        bias_initializer: Initializer for the bias vectors.
        kernel_regularizer: Regularizer function for the projection matrices.
        bias_regularizer: Regularizer function for the bias vectors.
        activity_regularizer: Regularizer function for the output.
        normalize_patterns: Whether to apply layer normalization to patterns.
        update_steps_max: Maximum number of association update steps (0 means no limit).
        update_steps_eps: Minimum difference threshold between update steps.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Call arguments:
        inputs: Query tensor of shape `(batch_size, seq_len_q, dim)` or list of tensors
               [query, key, value].
        mask: Optional mask of shape `(batch_size, seq_len_q, seq_len_k)`.
        return_attention_scores: If True, returns attention scores with output.
        training: Boolean indicating whether the layer should behave in training mode.

    Returns:
        output: Hopfield-processed tensor of shape `(batch_size, seq_len_q, dim)`.
        attention_scores: Optional attention weight tensor if return_attention_scores=True.

    Raises:
        ValueError: If num_heads <= 0 or key_dim <= 0.
        ValueError: If dropout is not in [0, 1].
        ValueError: If update_steps_max < 0.
        ValueError: If update_steps_eps <= 0.

    Example:
        >>> # Self-attention case
        >>> x = keras.random.normal((4, 32, 128))
        >>> hopfield_layer = HopfieldAttention(num_heads=8, key_dim=64)
        >>> output = hopfield_layer(x)
        >>> print(output.shape)
        (4, 32, 128)

        >>> # Cross-attention case
        >>> query = keras.random.normal((4, 32, 128))
        >>> key = keras.random.normal((4, 64, 128))
        >>> value = keras.random.normal((4, 64, 128))
        >>> output = hopfield_layer([query, key, value])
        >>> print(output.shape)
        (4, 32, 128)
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

        # Store configuration parameters
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

        # Initialize sublayers to None - will be created in build()
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.output_dense = None
        self.dropout_layer = None
        self.layernorm = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(f"Initialized HopfieldAttention with {num_heads} heads, "
                   f"key_dim={key_dim}, value_dim={self.value_dim}")

    def build(self, input_shape: Union[Tuple, List]) -> None:
        """
        Build the layer by creating sublayers when first called.

        Args:
            input_shape: Shape of input tensor or list of shapes for [query, key, value].
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

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

        input_dim = query_shape[-1]
        logger.debug(f"Building HopfieldAttention with input_dim={input_dim}")

        # Create projection layers
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

        # Output projection
        self.output_dense = keras.layers.Dense(
            input_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="output_dense"
        )

        # Dropout layer
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="attention_dropout")

        # Layer normalization
        if self.normalize_patterns:
            self.layernorm = keras.layers.LayerNormalization(
                epsilon=1e-5,
                name="pattern_norm"
            )

        # Build sublayers explicitly
        self.query_dense.build(query_shape)
        self.key_dense.build(query_shape)
        self.value_dense.build(query_shape)

        # Calculate output shape for output_dense
        projected_shape = list(query_shape)
        projected_shape[-1] = self.num_heads * self.value_dim
        self.output_dense.build(tuple(projected_shape))

        if self.dropout_layer is not None:
            # Dropout doesn't need explicit build
            pass

        if self.layernorm is not None:
            norm_shape = (None, None, self.num_heads, self.key_dim)
            self.layernorm.build(norm_shape)

        super().build(input_shape)
        logger.debug("HopfieldAttention build completed")

    def _hopfield_update_step(
        self,
        query: keras.KerasTensor,
        key: keras.KerasTensor,
        value: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Perform one Hopfield update step using scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim).
            key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim).
            value: Value tensor of shape (batch, num_heads, seq_len_v, value_dim).
            mask: Optional attention mask.
            training: Whether in training mode.

        Returns:
            Tuple of (updated_output, attention_weights).
        """
        # Scaled dot-product attention
        scale = ops.sqrt(ops.cast(self.key_dim, query.dtype))
        attention_scores = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2])) / scale

        # The 'mask' argument can be a tensor, None, or a list of masks propagated
        # by Keras. We need to resolve this to a single tensor mask or None.
        actual_mask = None
        if isinstance(mask, (list, tuple)):
            # If Keras passes a list, find the first non-None mask.
            for m in mask:
                if m is not None:
                    actual_mask = m
                    break
        else:
            actual_mask = mask

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
        mask: Optional[keras.KerasTensor] = None,
        return_attention_scores: bool = False,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass of the layer.

        Args:
            inputs: A tensor or list of tensors [query, key, value].
            mask: Optional attention mask.
            return_attention_scores: Whether to return attention scores.
            training: Whether in training mode.

        Returns:
            Output tensor or tuple of (output tensor, attention scores).
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
            query = key = value = inputs

        # Get shapes using Keras ops
        batch_size = ops.shape(query)[0]
        query_len = ops.shape(query)[1]
        key_len = ops.shape(key)[1]

        # Linear projections
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
                current_query, key_proj, value_proj, mask, training
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
            Output shape tuple.
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

        # Convert to list for manipulation
        output_shape = list(query_shape)

        # Output has same shape as query input
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a configuration dictionary.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
