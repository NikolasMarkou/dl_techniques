"""
Modern Hopfield Networks - Theory and Implementation Guide
=======================================================

Theory and Implementation:
-------------------------
Modern Hopfield Networks combine the classic Hopfield Network concept with attention mechanisms
from transformer architectures. Here's a detailed breakdown:

1. Theoretical Foundation:
------------------------
Classic Hopfield Networks had limited storage capacity (0.138N patterns for N neurons) and could
only handle binary patterns. Modern Hopfield Networks overcome these limitations:

- Can store exponentially many patterns (exp(N/c) patterns for constant c)
- Handle continuous-valued patterns naturally
- Guaranteed convergence in a single update step under normal conditions
- No spurious minima unlike classic Hopfield networks
- Stable retrieval even with partial or noisy patterns

The key innovation is replacing the traditional energy function and Hebbian learning with a
softmax-based association mechanism, similar to attention in transformers.

2. Key Components:
----------------
a) Multi-Head Attention:
   - Input patterns are projected into three spaces:
     * Query (Q): represents the pattern to be retrieved/completed
     * Key (K): represents stored patterns for matching
     * Value (V): represents the actual content to be retrieved
   - Multiple attention heads allow parallel pattern association
   - Each head can focus on different aspects of the patterns
   - Heads are concatenated and projected to produce final output

b) Pattern Storage:
   - Patterns are stored implicitly in the Key-Value pairs
   - No separate weight matrix needed (unlike classic Hopfield)
   - Storage capacity scales exponentially with dimension
   - Can dynamically update stored patterns during inference
   - Continuous-valued patterns are handled naturally

c) Pattern Retrieval:
   - Uses scaled dot-product attention: softmax(QK^T/sqrt(d))V
   - The scaling factor sqrt(d) prevents saturation of softmax
   - Softmax operation ensures proper energy minimization
   - Retrieval is iterative but typically converges quickly
   - Can retrieve multiple patterns simultaneously

3. Update Dynamics:
-----------------
The network performs iterative updates until convergence. Each update step consists of:

1. Computing attention scores:
   score = softmax(QK^T/sqrt(d))

2. Retrieving patterns:
   output = score * V

3. Checking convergence:
   - Compute change in attention scores
   - Stop if change < update_steps_eps
   - Or if update_steps_max reached

The update rule can be interpreted as:
- High attention scores indicate pattern matches
- Softmax ensures competition between patterns
- The scaling factor controls retrieval sharpness

4. Implementation Details:
------------------------
a) Initialization:
   - Creates projection matrices for Q, K, V transformations
   - Supports different dimensions for key/query and value
   - Initializes using Glorot uniform for stability
   - Sets up layer normalization if enabled

b) Forward Pass:
   1. Project inputs to Q, K, V spaces
   2. Reshape for multi-head attention
   3. Apply layer norm if enabled
   4. Perform iterative Hopfield updates
   5. Concatenate heads and project output

c) Optimization Features:
   - Masking support for variable-length sequences
   - Dropout for regularization
   - Layer normalization for training stability
   - Multiple regularization options (kernel, bias, activity)

5. Mathematical Details:
----------------------
Core Update Rule:
xi_{t+1} = softmax(β * K^T * xi_t) * V

where:
- xi_t is the state at step t
- β is the scaling factor (inverse temperature)
- K is the key matrix
- V is the value matrix

Energy Function:
E(xi) = -1/β * log(sum_k exp(β * k_i^T * xi)) + 1/2 * ||xi||^2

Properties:
- Energy function is continuous and differentiable
- Global minimum is guaranteed to exist
- No spurious local minima
- Convergence typically in one step for β -> ∞

6. Usage Considerations:
----------------------
Performance Optimization:
- Pattern normalization improves convergence but adds compute
- Number of heads should divide input dimension evenly
- update_steps_max trades accuracy vs. computation
- Dropout important for preventing overfitting

Hyperparameter Guidelines:
- num_heads: typically 4-12 depending on input size
- key_dim: usually input_dim // num_heads
- dropout: 0.1-0.3 works well for most cases
- update_steps_max: 0-3 sufficient for most applications
- update_steps_eps: 1e-4 is a good default

Common Issues:
- Poor convergence: try adjusting layer normalization
- Slow training: reduce update_steps_max
- Overfitting: increase dropout or add regularization
- Memory issues: reduce num_heads or key_dim

7. Extensions and Variants:
-------------------------
Possible modifications include:
- Adaptive scaling factor (β)
- Gated update rule
- Sparse attention patterns
- Hierarchical pattern storage
- Continuous update dynamics

8. References:
-------------
[1] Ramsauer, H., et al. (2020).
    "Hopfield Networks is All You Need"
    arXiv:2008.02217

[2] Krotov, D., & Hopfield, J. J. (2016).
    "Dense Associative Memory for Pattern Recognition"
    arXiv:1606.01164

[3] Vaswani, A., et al. (2017).
    "Attention is All You Need"
    arXiv:1706.03762
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
        dropout: Dropout probability for attention weights.
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
        dropout: float = 0.0,
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
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if update_steps_max < 0:
            raise ValueError(f"update_steps_max must be non-negative, got {update_steps_max}")
        if update_steps_eps <= 0:
            raise ValueError(f"update_steps_eps must be positive, got {update_steps_eps}")

        # Store configuration parameters
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout = dropout
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
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            # [query, key, value] shapes provided
            query_shape = input_shape[0]
        elif isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
            # Single input shape provided as nested structure
            query_shape = input_shape[0]
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
        if self.dropout > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout, name="attention_dropout")

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

        if mask is not None:
            # Expand mask to match attention scores shape
            mask = ops.cast(mask, attention_scores.dtype)
            # Add large negative values to masked positions
            attention_scores = attention_scores + (1.0 - mask) * -1e9

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

            # Check convergence if we have previous attention weights
            if prev_attention is not None and self.update_steps_eps > 0:
                diff_norm = ops.norm(attention_weights - prev_attention, ord=2)
                if diff_norm < self.update_steps_eps:
                    logger.debug(f"Hopfield converged after {update_step} steps")
                    break

            # Check maximum steps
            if self.update_steps_max > 0 and update_step >= self.update_steps_max:
                logger.debug(f"Hopfield stopped at max steps: {self.update_steps_max}")
                break

            # Update for next iteration
            prev_attention = attention_weights
            update_step += 1

            # For next iteration, use the output to compute new queries
            # This implements the iterative Hopfield dynamics
            if update_step < self.update_steps_max or self.update_steps_max == 0:
                # Use attention weights to update the query for next iteration
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
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            # [query, key, value] shapes provided
            query_shape = input_shape[0]
        elif isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
            # Nested structure
            query_shape = input_shape[0]
        else:
            # Single input shape
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
            "dropout": self.dropout,
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

