"""
A robust attention mechanism via Principal Component Pursuit.

This layer enhances the standard scaled dot-product attention by integrating
Principal Component Pursuit (PCP), a matrix decomposition technique. The
core idea is to make attention more robust to noise, adversarial
perturbations, and out-of-distribution data by separating the underlying
structure of the attention matrix from sparse, potentially disruptive
elements.

Architecture:
    The standard attention mechanism computes an attention matrix `A` from
    queries (Q) and keys (K). RPC-Attention intercepts this matrix `A`
    before the softmax operation and decomposes it into two distinct
    components:

    1.  A low-rank matrix `L`: This component captures the principal,
        globally smooth patterns and correlations within the sequence. It
        represents the broad, foundational relationships between tokens.

    2.  A sparse matrix `S`: This component captures localized, sharp, or
        outlier information. It isolates features that are highly specific,
        potentially representing noise, adversarial attacks, or uniquely
        important token-to-token interactions that deviate from the global
        pattern.

    The robust attention matrix is then reconstructed as `A_robust = L + S`
    before being passed to the final softmax function. This decomposition
    and reconstruction filters the attention mechanism, preserving the
    stable global structure while explicitly modeling sparse corruptions or
    salient details, leading to improved generalization and robustness.

Foundational Mathematics:
    The decomposition is achieved by solving the Principal Component
    Pursuit (PCP) convex optimization problem. Given the raw attention
    matrix `A`, PCP seeks to find `L` and `S` such that:

        min_{L,S} ||L||_* + lambda * ||S||_1   subject to   A = L + S

    -   `||L||_*` is the **nuclear norm** of `L` (the sum of its singular
        values). Minimizing the nuclear norm is a convex relaxation for
        minimizing the rank of the matrix, thus encouraging `L` to be
        low-rank.
    -   `||S||_1` is the **L1 norm** of `S` (the sum of the absolute values
        of its elements). Minimizing the L1 norm is a standard technique
        for inducing sparsity, encouraging most elements of `S` to be zero.
    -   `lambda` is a regularization parameter that balances the trade-off
        between the low-rankness of `L` and the sparsity of `S`.

    This problem is typically solved using iterative methods, such as the
    Alternating Direction Method of Multipliers (ADMM), which involves
    repeatedly applying singular value thresholding to update `L` and
    soft-thresholding to update `S`.

References:
    - The foundational theory for Principal Component Pursuit was
      introduced in:
      Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). "Robust
      Principal Component Analysis?". Journal of the ACM.
"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Any, Dict
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RPCAttention(keras.layers.Layer):
    """
    Robust Principal Components Attention layer.

    This layer implements RPC-Attention, which decomposes the attention matrix into
    low-rank and sparse components using Principal Component Pursuit (PCP). This
    decomposition provides robustness against adversarial attacks and improves
    generalization by separating global attention patterns (low-rank) from
    localized important features (sparse).

    **Mathematical Foundation**:

    Principal Component Pursuit optimization:

    .. math::
        \\min_{L,S} ||L||_* + \\lambda ||S||_1 \\text{ subject to } A = L + S

    Where:
    - A is the attention matrix
    - L is the low-rank component (captures global patterns)
    - S is the sparse component (captures outliers/details)
    - ||·||_* is the nuclear norm (sum of singular values)
    - ||·||_1 is the L1 norm
    - λ controls sparsity level

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, dim])
           ↓
    Linear Projection: Q, K, V = Linear(input)
           ↓
    Attention Matrix: A = softmax(QK^T/√d)
           ↓
    PCP Decomposition: A → L (low-rank) + S (sparse)
           ↓
    Robust Attention: Attn = softmax(L + S)
           ↓
    Output: Attn @ V
           ↓
    Linear Projection & Concatenate
           ↓
    Output(shape=[batch, seq_len, dim])
    ```

    **Key Benefits**:
    - **Robustness**: 15-25% better performance under adversarial attacks
    - **Accuracy**: >1% improvement on standard benchmarks
    - **Out-of-Distribution**: ~3 AUPR improvement on OOD detection
    - **Interpretability**: Separates global and local attention patterns

    Args:
        dim: Integer, dimensionality of the model. Must be positive and divisible
            by num_heads. This is the size of input and output embeddings.
        num_heads: Integer, number of attention heads. Must be positive and divide
            evenly into dim. Defaults to 8.
        lambda_sparse: Float, sparsity regularization parameter. Controls the
            sparsity of the S component. Higher values create sparser attention.
            Must be positive. Defaults to 0.1.
        max_pcp_iter: Integer, maximum iterations for PCP decomposition.
            More iterations give better decomposition but slower computation.
            Defaults to 10.
        svd_threshold: Float, threshold for singular value soft-thresholding
            in low-rank approximation. Defaults to 1.0.
        qkv_bias: Boolean, whether to use bias in Q, K, V projections.
            Defaults to False.
        dropout_rate: Float between 0 and 1, dropout rate for attention weights.
            Applied after the attention computation. Defaults to 0.0.
        kernel_initializer: Initializer for projection weight matrices.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for projection bias vectors.
            Only used when qkv_bias=True. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for projection weights.
        bias_regularizer: Optional regularizer for projection biases.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.
        Same shape as input.

    Attributes:
        to_qkv: Dense layer projecting input to Q, K, V.
        to_out: Dense layer for output projection.
        dropout: Dropout layer for regularization.
        attention_scale: Scale factor for attention scores.

    Example:
        ```python
        # Basic RPC attention
        rpc_attn = RPCAttention(
            dim=512,
            num_heads=8,
            lambda_sparse=0.1
        )

        # Process sequence with robust attention
        inputs = keras.random.normal((2, 100, 512))
        outputs = rpc_attn(inputs)  # Shape: (2, 100, 512)

        # More aggressive decomposition for adversarial robustness
        robust_attn = RPCAttention(
            dim=768,
            num_heads=12,
            lambda_sparse=0.2,  # Higher sparsity
            max_pcp_iter=20,    # More iterations
            svd_threshold=0.5   # Lower threshold
        )

        # Transformer block with RPC attention
        class RPCTransformerBlock(keras.layers.Layer):
            def __init__(self, dim, num_heads=8, **kwargs):
                super().__init__(**kwargs)
                self.attention = RPCAttention(
                    dim=dim,
                    num_heads=num_heads,
                    lambda_sparse=0.15
                )
                self.norm1 = keras.layers.LayerNormalization()
                self.norm2 = keras.layers.LayerNormalization()
                self.ffn = keras.Sequential([
                    keras.layers.Dense(dim * 4, activation='gelu'),
                    keras.layers.Dense(dim)
                ])

            def call(self, x):
                # Pre-norm architecture
                attn_out = self.attention(self.norm1(x))
                x = x + attn_out
                ffn_out = self.ffn(self.norm2(x))
                x = x + ffn_out
                return x
        ```

    References:
        - Robust Principal Component Analysis? (Candès et al., 2009)
        - RPC-Attention: Robust Attention via Principal Component Pursuit

    Note:
        The PCP decomposition is computed dynamically for each forward pass.
        For very long sequences, consider caching decompositions or using
        approximation methods to reduce computational cost.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            lambda_sparse: float = 0.1,
            max_pcp_iter: int = 10,
            svd_threshold: float = 1.0,
            qkv_bias: bool = False,
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
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
        if lambda_sparse <= 0:
            raise ValueError(f"lambda_sparse must be positive, got {lambda_sparse}")
        if max_pcp_iter <= 0:
            raise ValueError(f"max_pcp_iter must be positive, got {max_pcp_iter}")
        if svd_threshold <= 0:
            raise ValueError(f"svd_threshold must be positive, got {svd_threshold}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.lambda_sparse = lambda_sparse
        self.max_pcp_iter = max_pcp_iter
        self.svd_threshold = svd_threshold
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Computed attributes
        self.head_dim = dim // num_heads
        self.attention_scale = 1.0 / np.sqrt(self.head_dim)

        # Create sub-layers in __init__
        # Q, K, V projection layer
        self.to_qkv = layers.Dense(
            3 * dim,
            use_bias=qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='qkv_projection'
        )

        # Output projection layer
        self.to_out = layers.Dense(
            dim,
            use_bias=qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_projection'
        )

        # Dropout layer
        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and its sub-layers.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {input_shape}")
        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match dim ({self.dim})"
            )

        # Build sub-layers
        self.to_qkv.build(input_shape)
        self.to_out.build(input_shape)

        super().build(input_shape)

    def _soft_threshold(self, x: keras.KerasTensor, threshold: float) -> keras.KerasTensor:
        """
        Apply soft thresholding operator for sparse approximation.

        Soft thresholding is defined as:
        S_λ(x) = sign(x) * max(|x| - λ, 0)

        Args:
            x: Input tensor.
            threshold: Threshold value λ.

        Returns:
            Soft-thresholded tensor.
        """
        return ops.sign(x) * ops.maximum(ops.abs(x) - threshold, 0.0)

    def _nuclear_norm_minimization(
            self,
            matrix: keras.KerasTensor,
            threshold: float
    ) -> keras.KerasTensor:
        """
        Minimize nuclear norm via singular value thresholding.

        Performs SVD and applies soft thresholding to singular values.

        Args:
            matrix: Input matrix to be approximated.
            threshold: Threshold for singular values.

        Returns:
            Low-rank approximation of the input matrix.
        """
        # Perform SVD
        # Note: keras.ops.svd returns (u, s, v)
        # For TF backend, v corresponds to V^H (adjoint of right singular vectors)
        u, s, v = ops.svd(matrix, full_matrices=False)

        # Apply soft thresholding to singular values
        s_thresholded = ops.maximum(s - threshold, 0.0)

        # Reconstruct low-rank matrix
        # Need to construct a batch of diagonal matrices from s_thresholded
        # s_thresholded is (B, K)
        # We need s_diag to be (B, K, K)

        k = ops.shape(s)[-1]
        eye = ops.eye(k, dtype=s.dtype)

        # Broadcasting: (B, K, 1) * (1, K, K) -> (B, K, K)
        s_diag = ops.expand_dims(s_thresholded, axis=-1) * ops.expand_dims(eye, axis=0)

        # Reconstruct: U @ S @ V^H
        # Since v is already V^H in Keras/TF, we use it directly
        low_rank = ops.matmul(ops.matmul(u, s_diag), v)

        return low_rank

    def _pcp_decomposition(
            self,
            attention_matrix: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Perform Principal Component Pursuit decomposition.

        Decomposes attention matrix A into low-rank L and sparse S components
        using alternating minimization.

        Args:
            attention_matrix: Input attention matrix of shape
                (batch, num_heads, seq_len, seq_len).

        Returns:
            Tuple of (L, S) where:
            - L: Low-rank component (global patterns)
            - S: Sparse component (local details)
        """
        # Get shape info
        shape = ops.shape(attention_matrix)
        seq_len = shape[2]

        # Reshape to (Batch * Heads, Seq, Seq) for vectorized SVD
        # We use -1 for the batch dimension to handle symbolic shapes
        flat_matrix = ops.reshape(attention_matrix, (-1, seq_len, seq_len))

        # Initialize components
        L_flat = ops.zeros_like(flat_matrix)
        S_flat = ops.zeros_like(flat_matrix)

        # Alternating minimization
        for _ in range(self.max_pcp_iter):
            # Update L (low-rank component) via nuclear norm minimization
            # L = argmin ||L||_* s.t. A = L + S
            # Solution: L = SVT(A - S) where SVT is singular value thresholding
            residual = flat_matrix - S_flat

            # _nuclear_norm_minimization handles batched inputs correctly
            L_flat = self._nuclear_norm_minimization(residual, self.svd_threshold)

            # Update S (sparse component) via soft thresholding
            # S = argmin ||S||_1 s.t. A = L + S
            # Solution: S = soft_threshold(A - L, λ)
            S_flat = self._soft_threshold(flat_matrix - L_flat, self.lambda_sparse)

            # Note: Early stopping removed for Graph mode compatibility

        # Reshape back to (Batch, Num_Heads, Seq, Seq)
        L = ops.reshape(L_flat, shape)
        S = ops.reshape(S_flat, shape)

        return L, S

    def _compute_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Compute robust attention using PCP decomposition.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim).
            v: Value tensor of shape (batch, num_heads, seq_len, head_dim).
            mask: Optional attention mask.

        Returns:
            Attention output of shape (batch, num_heads, seq_len, head_dim).
        """
        # Compute attention scores
        # Shape: (batch, num_heads, seq_len, seq_len)
        attention_scores = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
        attention_scores = attention_scores * self.attention_scale

        # Apply mask if provided
        if mask is not None:
            # Broadcast mask to (batch, num_heads, seq_len, seq_len)
            mask_shape = ops.shape(mask)

            # Case 1: Mask is (batch, seq_len) -> Expand to (batch, 1, 1, seq_len)
            if len(mask.shape) == 2:
                mask = ops.expand_dims(mask, axis=1)
                mask = ops.expand_dims(mask, axis=1)
            # Case 2: Mask is (batch, seq_len, seq_len) -> Expand to (batch, 1, seq_len, seq_len)
            elif len(mask.shape) == 3:
                mask = ops.expand_dims(mask, axis=1)

            attention_scores = ops.where(
                mask == 0,
                ops.cast(-1e9, dtype=attention_scores.dtype),
                attention_scores
            )

        # Perform PCP decomposition
        L, S = self._pcp_decomposition(attention_scores)

        # Combine low-rank and sparse components
        robust_attention_scores = L + S

        # Apply softmax to get attention weights
        attention_weights = ops.softmax(robust_attention_scores, axis=-1)

        # Apply attention weights to values
        # Shape: (batch, num_heads, seq_len, head_dim)
        attention_output = ops.matmul(attention_weights, v)

        return attention_output

    def call(
            self,
            inputs: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            return_attention_scores: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Apply RPC attention to inputs.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, dim).
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len).
            training: Boolean indicating training mode.
            return_attention_scores: If True, also returns the attention weights.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
            If return_attention_scores=True, returns (output, attention_weights).
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q, K, V
        # Shape: (batch, seq_len, 3*dim)
        qkv = self.to_qkv(inputs)

        # Split into Q, K, V
        # Each has shape: (batch, seq_len, dim)
        q, k, v = ops.split(qkv, 3, axis=-1)

        # Reshape to multi-head format
        # Shape: (batch, num_heads, seq_len, head_dim)
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))

        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Compute robust attention with PCP decomposition
        attention_output = self._compute_attention(q, k, v, mask)

        # For returning attention scores if requested
        attention_weights = None
        if return_attention_scores:
            # Recompute attention scores for output
            attention_scores = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
            attention_scores = attention_scores * self.attention_scale
            L, S = self._pcp_decomposition(attention_scores)
            attention_weights = ops.softmax(L + S, axis=-1)

        # Reshape back to (batch, seq_len, dim)
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(attention_output, (batch_size, seq_len, self.dim))

        # Apply output projection
        output = self.to_out(attention_output)

        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        if return_attention_scores:
            return output, attention_weights

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Shape tuple of the output (same as input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing all configuration parameters.
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'lambda_sparse': self.lambda_sparse,
            'max_pcp_iter': self.max_pcp_iter,
            'svd_threshold': self.svd_threshold,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------