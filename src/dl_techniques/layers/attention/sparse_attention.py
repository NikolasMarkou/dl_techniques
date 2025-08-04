"""
# SparseAttention Layer

A Keras layer implementing sparse attention mechanisms to improve efficiency and interpretability
of transformer models by reducing the number of non-zero attention weights.

## Conceptual Overview

Standard transformer attention creates a dense attention matrix where every query token attends to
every key token, leading to quadratic complexity with sequence length. Sparse attention introduces
controlled sparsity by keeping only the most important connections and pruning low-importance weights.

This implementation offers two sparsification strategies:

1. **Direct Thresholding**: Apply a single threshold to softmax outputs, preserving only weights
   above the threshold and renormalizing.
2. **Iterative Sparsification (Variant B)**: Progressively increase the threshold over iterations,
   converging toward a near-one-hot distribution.

### Mathematical Description:

#### Direct Thresholding:
* Apply softmax: `p = softmax(QK^T / sqrt(d_k))`
* Apply threshold: `p_sparse = where(p >= threshold, p, 0)`
* Renormalize: `p_final = p_sparse / sum(p_sparse)`

#### Iterative Sparsification (Variant B):
* **Iteration 0**:
  * Apply softmax: `p = softmax(QK^T / sqrt(d_k))`
  * Initial threshold τ₀ = 1/N (where N is sequence length)
  * Prune weights: `p = where(p >= τ₀, p, 0)`
  * Renormalize: `p = p / sum(p)`

* **Iterations 1...max_iterations**:
  * Increase threshold: τᵢ = 1/(α^i × N) where α ∈ (0,1)
  * Prune weights: `p = where(p >= τᵢ, p, 0)`
  * Ensure argmax fallback for all-zero rows
  * Renormalize: `p = p / sum(p)`
  * Stop if all rows have ≤ 1 non-zero element

### Key Benefits:

1. **Computational Efficiency**: Reduced number of attention connections may lead to computational savings.
2. **Improved Interpretability**: Sparser attention maps provide clearer visibility into which tokens are relevant.
3. **Better Representations**: By focusing on key relationships, sparse attention can produce cleaner feature representations.
4. **Automatic Statistical Threshold**: Using 1/N as threshold has a sound interpretation - values below
   uniform distribution are "no better than random".
"""

import keras
from keras import ops
from typing import Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SparseAttention(keras.layers.Layer):
    """Sparse Attention mechanism with iterative sparsification.

    This layer implements a multi-head attention mechanism with sparsification to
    improve efficiency and interpretability. It supports two sparsification strategies:

    1. Direct thresholding: Apply a single threshold to softmax outputs
    2. Iterative sparsification: Progressively increase the threshold over iterations

    The iterative approach (enabled by default) follows the algorithm:
    - Apply initial threshold of 1/N
    - Iteratively increase threshold according to (alpha^t * N)
    - Continue for max_iterations or until convergence to a one-hot distribution

    Example:
    ```python
    # Basic usage
    attention = SparseAttention(num_heads=8, key_dim=64)

    # Use with separate query, key, value tensors
    output = attention([query, key, value], mask=padding_mask)

    # Or with a self-attention setup
    output = attention(inputs)
    ```

    Args:
        num_heads: Integer, number of attention heads.
        key_dim: Integer, size of each attention head for key and query.
        value_dim: Integer, size of each attention head for value. Default is key_dim.
        threshold: Float between 0 and 1, the threshold that controls sparsity,
            or "auto" to use a statistically derived threshold of 1/sequence_length.
            Higher values create more sparsity. Default is 0.1.
        learnable_threshold: Boolean, whether to make the threshold learnable. Default is False.
            When threshold="auto", this is ignored.
        per_head_thresholds: Boolean, whether to use different thresholds for each attention head.
            Only used when learnable_threshold is True. Default is False.
        causal: Boolean, whether to apply causal masking (for autoregressive models). Default is False.
        iterative_sparsification: Boolean, whether to use the iterative sparsification algorithm
            that progressively increases the threshold. Default is True.
        alpha: Float in (0, 1), the shrink multiplier for iterative sparsification.
            Only used when iterative_sparsification is True. Default is 0.6.
        max_iterations: Integer, maximum number of iterations for iterative sparsification.
            Only used when iterative_sparsification is True. Default is 3.
        dropout: Float, dropout rate for attention weights. Default is 0.0.
        use_bias: Boolean, whether the layer uses a bias vector. Default is True.
        kernel_initializer: Initializer for the kernel weights. Default is "glorot_uniform".
        bias_initializer: Initializer for the bias vector. Default is "zeros".
        kernel_regularizer: Regularizer for the kernel weights. Default is None.
        bias_regularizer: Regularizer for the bias vector. Default is None.
        activity_regularizer: Regularizer for the output. Default is None.
        kernel_constraint: Constraint for the kernel weights. Default is None.
        bias_constraint: Constraint for the bias vector. Default is None.
    """

    def __init__(
            self,
            num_heads: int,
            key_dim: int,
            value_dim: Optional[int] = None,
            threshold: Union[float, str] = 0.1,
            learnable_threshold: bool = False,
            per_head_thresholds: bool = False,
            causal: bool = False,
            iterative_sparsification: bool = True,
            alpha: float = 0.6,
            max_iterations: int = 3,
            dropout: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            kernel_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            bias_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if threshold != "auto" and not isinstance(threshold, str) and not learnable_threshold:
            if threshold <= 0 or threshold >= 1:
                raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        if max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive, got {max_iterations}")

        # Store configuration parameters
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.threshold_value = threshold
        self.auto_threshold = threshold == "auto"
        # Only use learnable threshold if not in auto mode
        self.learnable_threshold = learnable_threshold and not self.auto_threshold
        self.per_head_thresholds = per_head_thresholds
        self.causal = causal
        self.iterative_sparsification = iterative_sparsification
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.dropout_rate = dropout
        self.use_bias = use_bias

        # Store initializers, regularizers and constraints
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        # Will be set up in build()
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.output_dense = None
        self.dropout = None
        self.threshold_var = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the layer weights based on input shape.

        Args:
            input_shape: Shape tuple (tuple of integers) or a list of shape tuples,
                indicating the input shape of the layer.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Check if input_shape is a list/tuple of length 3 (query, key, value)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            query_shape, key_shape, value_shape = input_shape
            last_dim_query = int(query_shape[-1])
            last_dim_key = int(key_shape[-1])
            last_dim_value = int(value_shape[-1])
        else:
            # If a single tensor is provided for Q, K, V
            last_dim_query = last_dim_key = last_dim_value = int(input_shape[-1])

        # Build projection layers
        self.query_dense = keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="query"
        )

        self.key_dense = keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="key"
        )

        self.value_dense = keras.layers.Dense(
            self.num_heads * self.value_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="value"
        )

        # Output dense shape depends on the reshaped attention output with dimensions:
        # [batch_size, seq_len_q, num_heads * value_dim]
        self.output_dense = keras.layers.Dense(
            last_dim_query,  # Reduce back to original query dimension
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="output"
        )

        # Create dropout layer
        self.dropout = keras.layers.Dropout(self.dropout_rate)

        # Initialize learnable threshold parameters if enabled
        if self.learnable_threshold:
            if self.per_head_thresholds:
                # Create a threshold parameter for each attention head
                threshold_shape = (self.num_heads,)
                threshold_name = "thresholds_per_head"
            else:
                # Create a single threshold parameter for all heads
                threshold_shape = (1,)
                threshold_name = "threshold"

            # Initialize with the provided threshold value
            threshold_initializer = keras.initializers.Constant(self.threshold_value)

            # Create the threshold variable with a serializable constraint
            self.threshold_var = self.add_weight(
                name=threshold_name,
                shape=threshold_shape,
                initializer=threshold_initializer,
                constraint=ValueRangeConstraint(min_value=0.0, max_value=0.9999),
                trainable=True
            )

        # Build sublayers explicitly
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            self.query_dense.build(query_shape)
            self.key_dense.build(key_shape)
            self.value_dense.build(value_shape)
        else:
            self.query_dense.build(input_shape)
            self.key_dense.build(input_shape)
            self.value_dense.build(input_shape)

        # Output dense shape depends on the reshaped attention output
        output_shape = list(query_shape if isinstance(input_shape, (list, tuple)) else input_shape)
        output_shape[-1] = self.num_heads * self.value_dim
        self.output_dense.build(tuple(output_shape))

        self.built = True

    def call(self, inputs, mask=None, training=None):
        """Forward computation with sparse attention mechanism.

        Args:
            inputs: Input tensor or list/tuple of input tensors (query, key, value).
            mask: Optional attention mask.
                - Float tensor with shape [batch, seq_len_k] or [batch, seq_len_q, seq_len_k]
                - In case of a 2D tensor, it will be reshaped to [batch, 1, 1, seq_len_k]
                - In case of a 3D tensor, it will be reshaped to [batch, 1, seq_len_q, seq_len_k]
                - A value of 1 indicates positions to attend to, 0 indicates masked positions
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            A tensor containing the computation result after sparse attention.
        """
        # Handle different input types
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3:
                query, key, value = inputs
            else:
                raise ValueError(f"Expected 3 input tensors (query, key, value), got {len(inputs)}")
        else:
            query = key = value = inputs

        # Get batch size and sequence length
        batch_size = ops.shape(query)[0]
        query_length = ops.shape(query)[1]

        # Project inputs to query, key, and value
        query = self.query_dense(query)  # [batch, seq_len_q, num_heads * key_dim]
        key = self.key_dense(key)  # [batch, seq_len_k, num_heads * key_dim]
        value = self.value_dense(value)  # [batch, seq_len_v, num_heads * value_dim]

        # Reshape to [batch, seq_len, num_heads, dim]
        query = ops.reshape(query, (batch_size, query_length, self.num_heads, self.key_dim))
        key = ops.reshape(key, (batch_size, -1, self.num_heads, self.key_dim))
        value = ops.reshape(value, (batch_size, -1, self.num_heads, self.value_dim))

        # Apply sparse attention mechanism
        attn_output = self._sparse_attention(query, key, value, mask=mask, training=training)

        # Reshape back to [batch, seq_len_q, num_heads * value_dim]
        attn_output = ops.reshape(
            attn_output, (batch_size, query_length, self.num_heads * self.value_dim)
        )

        # Apply output projection
        output = self.output_dense(attn_output)

        return output

    def _sparse_attention(self, query, key, value, mask=None, training=None):
        """Compute sparse attention using the selected sparsification strategy.

        Args:
            query: Query tensor of shape [batch, seq_len_q, num_heads, key_dim]
            key: Key tensor of shape [batch, seq_len_k, num_heads, key_dim]
            value: Value tensor of shape [batch, seq_len_v, num_heads, value_dim]
            mask: Optional attention mask
            training: Boolean indicating whether in training mode

        Returns:
            Output tensor of shape [batch, seq_len_q, num_heads, value_dim]
        """
        # Transpose to [batch, num_heads, seq_len, dim]
        query = ops.transpose(query, [0, 2, 1, 3])  # [batch, num_heads, seq_len_q, key_dim]
        key = ops.transpose(key, [0, 2, 3, 1])  # [batch, num_heads, key_dim, seq_len_k] (transposed for matmul)
        value = ops.transpose(value, [0, 2, 1, 3])  # [batch, num_heads, seq_len_v, value_dim]

        # Calculate attention scores: QK^T / sqrt(dk)
        scale = ops.cast(1.0 / ops.sqrt(ops.cast(self.key_dim, "float32") + keras.backend.epsilon()), query.dtype)
        attention_scores = ops.matmul(query, key) * scale  # [batch, num_heads, seq_len_q, seq_len_k]

        # Get sequence length for automatic threshold
        seq_len_k = ops.shape(key)[3]

        # Process masks and combine them into a single mask if needed
        if self.causal or mask is not None:
            # Convert mask to appropriate format and apply to attention scores
            attention_mask = self._prepare_masks(query, key, mask)

            # Apply large negative bias to masked positions
            neg_inf = ops.cast(-1e9, attention_scores.dtype)
            attention_scores = attention_scores + (1.0 - attention_mask) * neg_inf

        # Apply softmax to get the initial attention distribution
        attention_weights = keras.activations.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights if in training mode
        attention_weights = self.dropout(attention_weights, training=training)

        # Decide which sparsification method to use
        if self.iterative_sparsification:
            # Use the iterative sparsification algorithm
            normalized_weights = self._iterative_sparsify(attention_weights, seq_len_k)
        else:
            # Use direct thresholding
            normalized_weights = self._direct_threshold(attention_weights, seq_len_k)

        # Apply attention to values
        outputs = ops.matmul(normalized_weights, value)  # [batch, num_heads, seq_len_q, value_dim]

        # Transpose back to [batch, seq_len_q, num_heads, value_dim]
        outputs = ops.transpose(outputs, [0, 2, 1, 3])

        return outputs

    def _prepare_masks(self, query, key, mask):
        """Prepare and combine masks into a single attention mask.

        Uses optimized causal mask creation for XLA compatibility.

        Args:
            query: Query tensor of shape [batch, num_heads, seq_len_q, key_dim]
            key: Key tensor of shape [batch, num_heads, key_dim, seq_len_k]
            mask: Optional external mask

        Returns:
            Combined mask tensor of shape [batch, num_heads, seq_len_q, seq_len_k]
        """
        seq_len_q = ops.shape(query)[2]
        seq_len_k = ops.shape(key)[3]
        dtype = query.dtype

        # Generate causal mask if needed
        if self.causal:
            # Create a lower triangular mask using efficient band_part approach
            # This is more XLA-friendly than using triu
            ones = ops.ones((seq_len_q, seq_len_k), dtype=dtype)

            # Create lower triangular part (including diagonal)
            # -1 means "all rows", 0 means "only the diagonal and below"
            causal_mask = ops.linalg.band_part(ones, -1, 0)

            # Reshape for broadcasting
            causal_mask = ops.reshape(causal_mask, (1, 1, seq_len_q, seq_len_k))

            # If external mask was provided, combine with causal mask
            if mask is not None:
                # Check mask rank safely (without using len on a tensor)
                mask_rank = ops.rank(mask)

                # Handle different mask shapes
                mask_is_2d = ops.equal(mask_rank, 2)
                mask_is_3d = ops.equal(mask_rank, 3)
                mask_is_4d = ops.equal(mask_rank, 4)

                # Reshape 2D mask [batch, seq_len_k] -> [batch, 1, 1, seq_len_k]
                def handle_2d_mask():
                    return ops.reshape(mask, (ops.shape(mask)[0], 1, 1, ops.shape(mask)[1]))

                # Reshape 3D mask [batch, seq_len_q, seq_len_k] -> [batch, 1, seq_len_q, seq_len_k]
                def handle_3d_mask():
                    return ops.expand_dims(mask, axis=1)

                # Use 4D mask as is
                def handle_4d_mask():
                    return mask

                # Use safe conditional ops instead of Python control flow
                padding_mask = ops.cond(
                    mask_is_2d,
                    handle_2d_mask,
                    lambda: ops.cond(
                        mask_is_3d,
                        handle_3d_mask,
                        handle_4d_mask
                    )
                )

                # Combine masks (both must be 1 to attend)
                return causal_mask * ops.cast(padding_mask, dtype)

            return causal_mask
        elif mask is not None:
            # No causal mask, just prepare the provided mask
            # Check mask rank safely (without using len on a tensor)
            mask_rank = ops.rank(mask)

            # Handle different mask shapes
            mask_is_2d = ops.equal(mask_rank, 2)
            mask_is_3d = ops.equal(mask_rank, 3)

            # Reshape 2D mask [batch, seq_len_k] -> [batch, 1, 1, seq_len_k]
            def handle_2d_mask():
                return ops.reshape(ops.cast(mask, dtype),
                                   (ops.shape(mask)[0], 1, 1, ops.shape(mask)[1]))

            # Reshape 3D mask [batch, seq_len_q, seq_len_k] -> [batch, 1, seq_len_q, seq_len_k]
            def handle_3d_mask():
                return ops.expand_dims(ops.cast(mask, dtype), axis=1)

            # Use 4D mask as is
            def handle_4d_mask():
                return ops.cast(mask, dtype)

            # Use safe conditional ops instead of Python control flow
            return ops.cond(
                mask_is_2d,
                handle_2d_mask,
                lambda: ops.cond(
                    mask_is_3d,
                    handle_3d_mask,
                    handle_4d_mask
                )
            )

        # No masks needed
        return ops.ones((1, 1, seq_len_q, seq_len_k), dtype=dtype)

    def _direct_threshold(self, attention_weights, seq_len_k):
        """Apply direct thresholding to attention weights.

        Args:
            attention_weights: Attention weights from softmax
            seq_len_k: Sequence length of keys (for auto threshold)

        Returns:
            Normalized sparse attention weights
        """
        # Determine threshold to use
        if self.auto_threshold:
            # Calculate statistically meaningful threshold: 1/sequence_length
            # Any attention weight below this is no better than random
            threshold = 1.0 / (ops.cast(seq_len_k, attention_weights.dtype) + keras.backend.epsilon())
            threshold = ops.reshape(threshold, (1, 1, 1, 1))  # For broadcasting
        elif self.learnable_threshold:
            if self.per_head_thresholds:
                # Shape: [1, num_heads, 1, 1] for broadcasting
                threshold = ops.reshape(self.threshold_var, (1, self.num_heads, 1, 1))
            else:
                # Single threshold for all heads, shape: [1, 1, 1, 1]
                threshold = ops.reshape(self.threshold_var, (1, 1, 1, 1))
        else:
            # Cast the Python float to match the attention weights dtype
            threshold = ops.cast(self.threshold_value, attention_weights.dtype)

        # Apply thresholding directly to softmax outputs (using where for efficiency)
        sparse_weights = ops.where(
            attention_weights >= threshold,
            attention_weights,
            ops.zeros_like(attention_weights)
        )

        # Handle the case where all weights for a token are below threshold
        epsilon = keras.backend.epsilon()

        # Re-normalize to ensure weights sum to 1 (maintaining probability distribution)
        sum_weights = ops.sum(sparse_weights, axis=-1, keepdims=True) + epsilon

        # Where sum is near zero (all weights were below threshold), use original weights
        # to avoid division problems
        needs_original = ops.cast(ops.less_equal(sum_weights, 2 * epsilon), sparse_weights.dtype)
        normalized_weights = (
                (1.0 - needs_original) * (sparse_weights / sum_weights) +
                needs_original * attention_weights
        )

        return normalized_weights

    def _iterative_sparsify(self, attention_weights, seq_len_k):
        """Apply iterative sparsification to attention weights (Variant B).

        This implements a fully vectorized version of the iterative sparsification algorithm
        without any Python control flow for TPU/GPU compatibility and XLA optimization.

        Args:
            attention_weights: Attention weights from softmax
            seq_len_k: Sequence length of keys (for threshold calculation)

        Returns:
            Normalized sparse attention weights
        """
        # Get the data type and shape information
        dtype = attention_weights.dtype
        epsilon = keras.backend.epsilon()
        batch_size = ops.shape(attention_weights)[0]
        num_heads = ops.shape(attention_weights)[1]
        seq_len_q = ops.shape(attention_weights)[2]

        # Keep a copy of the original weights for fallback cases
        original_weights = attention_weights

        # Initialize result with first iteration
        p = attention_weights

        # Pre-compute all thresholds using integer range and proper casting
        iterations_int = ops.range(0, self.max_iterations)
        iterations_float = ops.cast(iterations_int, dtype)
        alphas = ops.power(ops.cast(self.alpha, dtype), iterations_float)
        n_effs = ops.cast(seq_len_k, dtype) * alphas + epsilon  # Effective N for each iteration
        thresholds = 1.0 / n_effs  # 1/N for each iteration

        # The fully vectorized approach would be to stack all iterations together
        # and process them in a single tensor operation, but for now we'll optimize
        # the inner loops to reduce computation and avoid unnecessary work.

        # Apply the initial threshold (t=0)
        mask = ops.cast(p >= thresholds[0], dtype)
        p = p * mask

        # Normalize
        sum_p = ops.sum(p, axis=-1, keepdims=True) + epsilon
        has_nonzeros = ops.cast(sum_p > 2 * epsilon, dtype)
        p = (p / sum_p) * has_nonzeros + (1.0 - has_nonzeros) * original_weights

        # Compute argmax indices only if needed, using a conditional
        def compute_argmax_onehot():
            # Reshape to [batch*heads*seq_q, seq_k] for efficiency
            flat_shape = (batch_size * num_heads * seq_len_q, seq_len_k)
            p_flat = ops.reshape(p, flat_shape)

            # Get argmax indices
            max_indices_flat = ops.argmax(p_flat, axis=-1)

            # Create one-hot encoding for the argmax indices
            one_hot_flat = ops.one_hot(max_indices_flat, seq_len_k, dtype=dtype)

            # Reshape back to match p's shape
            return ops.reshape(one_hot_flat, (batch_size, num_heads, seq_len_q, seq_len_k))

        # For each iteration, apply increasing thresholds
        result = p
        for t in range(1, self.max_iterations):
            # Apply threshold
            threshold = thresholds[t]
            iter_mask = ops.cast(result >= threshold, dtype)

            # Count survivors (non-zero entries)
            survivors = ops.sum(iter_mask, axis=-1, keepdims=True)
            all_zeros = ops.equal(survivors, 0)

            # Only compute one-hot if we have any all-zero rows
            # This saves computation when it's not needed
            has_any_zeros = ops.any(all_zeros)
            one_hot = ops.cond(
                has_any_zeros,
                compute_argmax_onehot,
                lambda: ops.zeros_like(result)  # Dummy value, won't be used
            )

            # Create a fallback mask for all-zero cases
            all_zeros_expanded = ops.expand_dims(all_zeros, axis=-1)

            # Combine masks: use one-hot where all entries are zeroed, otherwise use threshold mask
            combined_mask = ops.where(all_zeros_expanded, one_hot, iter_mask)

            # Apply mask
            masked_result = result * combined_mask

            # Normalize
            sum_masked_result = ops.sum(masked_result, axis=-1, keepdims=True) + epsilon
            iter_has_nonzeros = ops.cast(sum_masked_result > 2 * epsilon, dtype)
            normalized_result = (masked_result / sum_masked_result) * iter_has_nonzeros + (
                        1.0 - iter_has_nonzeros) * original_weights

            # Check if we've reached 1-hot (or close to it) per sample/head
            # Use reduce_all with specified axes for better efficiency
            survivors = ops.sum(combined_mask, axis=-1, keepdims=True)
            all_sparse_per_sample_head = ops.all(
                ops.less_equal(survivors, 1),
                axis=[2]  # Check across seq_len_q dimension
            )

            # Only update result if not all positions have reached sparsity
            # This allows per-sample/per-head early stopping in the graph
            should_continue = ops.logical_not(all_sparse_per_sample_head)
            should_continue = ops.reshape(should_continue, (batch_size, num_heads, 1, 1))
            should_continue = ops.cast(should_continue, dtype)

            # Update result conditionally
            result = should_continue * normalized_result + (1.0 - should_continue) * result

            # Early stopping check across the entire batch (for loop efficiency)
            if ops.all(ops.logical_not(should_continue)):
                break

        return result

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        The output shape matches the input query shape, preserving batch size and
        sequence length, but using the original query's last dimension as the
        output dimension after projection through the output_dense layer.

        Note: If you subclass this layer and modify output_dense.units or set
        use_bias=False, make sure to update this method accordingly.

        Args:
            input_shape: Shape of the input or a list of shapes for query, key, value.

        Returns:
            Output shape.
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            query_shape = input_shape[0]
        else:
            query_shape = input_shape

        # Convert to list for consistent manipulation
        query_shape_list = list(query_shape)

        # Output shape preserves batch and sequence length from query
        # but uses the last dimension from the original query shape
        return tuple(query_shape_list)

    def get_config(self):
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "threshold": self.threshold_value,
            "learnable_threshold": self.learnable_threshold,
            "per_head_thresholds": self.per_head_thresholds,
            "causal": self.causal,
            "iterative_sparsification": self.iterative_sparsification,
            "alpha": self.alpha,
            "max_iterations": self.max_iterations,
            "dropout": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
        })
        return config

    def get_build_config(self):
        """Get the config needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config):
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
