"""
A robust attention mechanism via Principal Component Pursuit.

This layer enhances standard scaled dot-product attention by integrating
Principal Component Pursuit (PCP), a matrix decomposition technique that
separates the attention matrix into a low-rank component ``L`` (global
patterns) and a sparse component ``S`` (localized details/outliers) by
solving ``min_{L,S} ||L||_* + lambda ||S||_1  s.t.  A = L + S``. The
robust attention is then ``softmax(L + S) V``, providing resilience
against noise and adversarial perturbations.

References:
    - Candes, E. J., Li, X., Ma, Y., & Wright, J. (2011). "Robust
      Principal Component Analysis?". Journal of the ACM.
"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Any, Dict
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RPCAttention(keras.layers.Layer):
    """Robust Principal Components Attention via PCP decomposition.

    Implements RPC-Attention which decomposes the raw attention score matrix
    ``A = Q K^T / sqrt(d_k)`` into low-rank ``L`` and sparse ``S`` components
    using iterative alternating minimization (ADMM-style). The low-rank component
    is obtained via singular value thresholding:
    ``L = U diag(max(sigma - tau, 0)) V^H`` where ``(U, sigma, V^H) = SVD(A - S)``.
    The sparse component uses soft thresholding:
    ``S = sign(A - L) max(|A - L| - lambda, 0)``. After ``max_pcp_iter``
    iterations, the robust attention output is ``softmax(L + S) V``.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────┐
        │ Input [B, seq_len, dim]       │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Dense(3*dim) → Q, K, V        │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Multi-Head Reshape            │
        │ [B, heads, seq_len, head_dim] │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ A = Q K^T * scale             │
        │ [B, heads, seq_len, seq_len]  │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │ PCP Decomposition (iterative):        │
        │ ┌───────────────────────────────────┐ │
        │ │ L = SVT(A - S, tau)   (low-rank)  │ │
        │ │ S = soft(A - L, lambda) (sparse)  │ │
        │ └───────────────────────────────────┘ │
        │ Repeat max_pcp_iter times             │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Robust Attn = softmax(L + S)  │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Output = Attn @ V             │
        │ Reshape → [B, seq_len, dim]   │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Dense(dim) output projection  │
        │ + Dropout (optional)          │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Output [B, seq_len, dim]      │
        └───────────────────────────────┘

    :param dim: Model dimensionality. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param lambda_sparse: Sparsity regularization parameter for the ``S`` component.
        Higher values create sparser attention.
    :type lambda_sparse: float
    :param max_pcp_iter: Maximum iterations for PCP decomposition.
    :type max_pcp_iter: int
    :param svd_threshold: Threshold for singular value soft-thresholding in low-rank
        approximation.
    :type svd_threshold: float
    :param qkv_bias: Whether to use bias in Q, K, V projections.
    :type qkv_bias: bool
    :param dropout_rate: Dropout rate for output, between 0 and 1.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for projection weight matrices.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for projection bias vectors.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for projection weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for projection biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any
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
        """Build the layer and its sub-layers.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
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
        """Apply soft thresholding: ``S_lambda(x) = sign(x) max(|x| - lambda, 0)``.

        :param x: Input tensor.
        :type x: keras.KerasTensor
        :param threshold: Threshold value lambda.
        :type threshold: float

        :return: Soft-thresholded tensor.
        :rtype: keras.KerasTensor
        """
        return ops.sign(x) * ops.maximum(ops.abs(x) - threshold, 0.0)

    def _nuclear_norm_minimization(
            self,
            matrix: keras.KerasTensor,
            threshold: float
    ) -> keras.KerasTensor:
        """Minimize nuclear norm via singular value thresholding.

        :param matrix: Input matrix to be approximated.
        :type matrix: keras.KerasTensor
        :param threshold: Threshold for singular values.
        :type threshold: float

        :return: Low-rank approximation of the input matrix.
        :rtype: keras.KerasTensor
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
        """Perform Principal Component Pursuit decomposition via alternating minimization.

        :param attention_matrix: Input attention matrix of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_matrix: keras.KerasTensor

        :return: Tuple of ``(L, S)`` where ``L`` is the low-rank component and
            ``S`` is the sparse component, both with the same shape as input.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
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
        """Compute robust attention using PCP decomposition.

        :param q: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type q: keras.KerasTensor
        :param k: Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type v: keras.KerasTensor
        :param mask: Optional attention mask.
        :type mask: Optional[keras.KerasTensor]

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
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
        """Apply RPC attention to inputs.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional attention mask of shape
            ``(batch_size, seq_len, seq_len)``.
        :type mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :param return_attention_scores: If ``True``, also returns attention weights.
        :type return_attention_scores: bool

        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
            If return_attention_scores is ``True``, returns
            ``(output, attention_weights)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
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
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Shape tuple of the output (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        :return: Dictionary containing all configuration parameters.
        :rtype: Dict[str, Any]
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
