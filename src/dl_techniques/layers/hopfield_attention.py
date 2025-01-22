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

from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union
from keras.api.layers import Layer, Dense, LayerNormalization


class HopfieldAttention(Layer):
    """
    Modern Hopfield layer implementation as described in 'Hopfield Networks is All You Need'.
    This layer implements a modern Hopfield network that can store exponentially many patterns
    and converges with one update. It uses a transformer-like attention mechanism as its core
    operation. For detailed theoretical background and implementation details, please refer to
    the accompanying documentation file.

    Args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for key and query.
        value_dim: Size of each attention head for value.
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

    Call arguments:
        query: Query tensor of shape `(batch_size, seq_len_q, dim)`.
        key: Key tensor of shape `(batch_size, seq_len_k, dim)`.
        value: Value tensor of shape `(batch_size, seq_len_v, dim)`.
        attention_mask: Optional mask of shape `(batch_size, seq_len_q, seq_len_k)`.
        return_attention_scores: If True, returns attention scores with output.

    Returns:
        output: Hopfield-processed tensor of shape `(batch_size, seq_len_q, dim)`.
        attention_scores: Optional attention weight tensor.
    """

    def __init__(
            self,
            num_heads: int,
            key_dim: int,
            value_dim: Optional[int] = None,
            dropout: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: str = "glorot_uniform",
            bias_initializer: str = "zeros",
            kernel_regularizer: Optional[str] = None,
            bias_regularizer: Optional[str] = None,
            activity_regularizer: Optional[Union[str, tf.keras.regularizers.Regularizer]] = None,
            normalize_patterns: bool = True,
            update_steps_max: int = 0,
            update_steps_eps: float = 1e-4,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim else key_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.normalize_patterns = normalize_patterns
        self.update_steps_max = update_steps_max
        self.update_steps_eps = update_steps_eps

        self._built = False

    def build(self, input_shape: Union[Tuple, list]) -> None:
        """Builds the layer by creating weights when first called.

        Args:
            input_shape: Shape of input tensor or list of shapes for [query, key, value].
        """
        if isinstance(input_shape, (list, tuple)):
            query_shape = input_shape[0]
        else:
            query_shape = input_shape

        input_dim = query_shape[-1]
        head_dim = self.key_dim

        # Query, Key, Value projection matrices
        self.query_dense = Dense(
            self.num_heads * head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="query_dense"
        )

        self.key_dense = Dense(
            self.num_heads * head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="key_dense"
        )

        self.value_dense = Dense(
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
        self.output_dense = Dense(
            input_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="output_dense"
        )

        if self.normalize_patterns:
            self.layernorm = LayerNormalization(epsilon=1e-5, name="pattern_norm")

        self._built = True

    def _hopfield_update_step(
            self,
            query: tf.Tensor,
            key: tf.Tensor,
            value: tf.Tensor,
            mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Performs one Hopfield update step using scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
            key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim)
            value: Value tensor of shape (batch, num_heads, seq_len_v, value_dim)
            mask: Optional attention mask

        Returns:
            Updated query tensor and attention weights
        """
        # Scaled dot-product attention
        scale = tf.cast(tf.math.sqrt(tf.cast(self.key_dim, tf.float32)), query.dtype)
        attention_scores = ops.matmul(query, key, transpose_b=True) / scale

        if mask is not None:
            # Add large negative values to masked positions
            attention_scores += (1.0 - tf.cast(mask, attention_scores.dtype)) * -1e9

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, self.dropout)

        return ops.matmul(attention_weights, value), attention_weights

    def call(
            self,
            inputs: Union[tf.Tensor, list],
            mask: Optional[tf.Tensor] = None,
            return_attention_scores: bool = False,
            training: Optional[bool] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass of the layer.

        Args:
            inputs: A tensor or list of tensors [query, key, value]
            mask: Optional attention mask
            return_attention_scores: Whether to return attention scores
            training: Whether in training mode

        Returns:
            Output tensor or tuple of (output tensor, attention scores)
        """
        # Handle input formats
        if isinstance(inputs, (list, tuple)):
            query, key, value = inputs
        else:
            query = key = value = inputs

        # Get shapes
        batch_size = tf.shape(query)[0]
        query_len = tf.shape(query)[1]
        key_len = tf.shape(key)[1]

        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Reshape to multi-head attention format
        query = tf.reshape(query, (batch_size, query_len, self.num_heads, self.key_dim))
        key = tf.reshape(key, (batch_size, key_len, self.num_heads, self.key_dim))
        value = tf.reshape(value, (batch_size, key_len, self.num_heads, self.value_dim))

        # Transpose to (batch, num_heads, seq_len, dim)
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        if self.normalize_patterns:
            query = self.layernorm(query)
            key = self.layernorm(key)

        # Initialize Hopfield update loop
        xi = None
        xi_old = None
        update_step = 0

        while True:
            # Perform Hopfield update
            output, attention_weights = self._hopfield_update_step(query, key, value, mask)

            if xi is None:
                xi = attention_weights
            else:
                # Update xi based on new attention weights
                xi = attention_weights

            # Check convergence
            if xi_old is not None:
                diff_norm = tf.norm(xi - xi_old, ord=2)
                if diff_norm < self.update_steps_eps:
                    break

            if self.update_steps_max > 0 and update_step >= self.update_steps_max:
                break

            xi_old = xi
            update_step += 1

            # Update query for next iteration
            query = ops.matmul(xi, key)

        # Reshape output
        output = tf.transpose(output, [0, 2, 1, 3])  # (batch, seq_len, num_heads, value_dim)
        output = tf.reshape(output, (batch_size, query_len, self.num_heads * self.value_dim))

        # Final output projection
        output = self.output_dense(output)

        if return_attention_scores:
            return output, attention_weights
        return output

    def get_config(self) -> dict:
        """Returns the config of the layer."""
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "dropout": self.dropout,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "normalize_patterns": self.normalize_patterns,
            "update_steps_max": self.update_steps_max,
            "update_steps_eps": self.update_steps_eps,
        })
        return config
