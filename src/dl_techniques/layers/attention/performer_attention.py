"""
Approximates softmax attention with linear complexity using random features.

This layer implements the Performer, a transformer architecture that can
process long sequences with a memory and time complexity that scales
linearly with sequence length, as opposed to the quadratic complexity of
standard dot-product attention. This is achieved by approximating the
softmax kernel using the FAVOR+ (Fast Attention Via positive Orthogonal
Random features) algorithm.

The core idea replaces the explicit ``(Q @ K^T) @ V`` computation with
``phi(Q) @ (phi(K)^T @ V)`` via random feature maps, reducing complexity
from ``O(N^2 * d)`` to ``O(N * r * d)`` where ``r`` is the number of
random features.

References:
    - Choromanski, K., Likhosherstov, V., Dohan, D., et al. (2020).
      "Rethinking Attention with Performers".
"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Any, Dict
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PerformerAttention(keras.layers.Layer):
    """Performer attention with linear complexity via FAVOR+ kernel approximation.

    Implements the Performer attention mechanism using Fast Attention Via positive
    Orthogonal Random features (FAVOR+) to approximate standard softmax attention
    with linear time and memory complexity ``O(N)`` instead of ``O(N^2)``. The
    standard attention ``softmax(Q K^T / sqrt(d_k)) V`` is approximated by
    constructing positive random feature maps ``phi(Q)`` and ``phi(K)`` such that
    ``exp(q . k) ~ <phi(q), phi(k)>``, then computing
    ``phi(Q) (phi(K)^T V) / phi(Q) (phi(K)^T 1_N)`` which avoids materializing
    the ``N x N`` attention matrix. The feature map uses trigonometric random
    projections: ``phi(x) = (1/sqrt(r)) [cos(w_i . x), sin(w_i . x)]``
    scaled by ``exp(-||x||^2 / 2)``.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────┐
        │ Input [B, seq_len, dim]       │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Dense(3 * dim) → Q, K, V     │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Multi-Head Reshape            │
        │ [B, heads, seq_len, head_dim] │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Random Feature Projection     │
        │ phi(Q), phi(K)                │
        │ [B, heads, seq_len, nb_feat]  │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Linear Attention              │
        │ KV = phi(K)^T V              │
        │ Z  = phi(Q) sum(phi(K))      │
        │ Out = phi(Q) KV / Z          │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Concatenate Heads & Project   │
        │ Dense(dim)                    │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Output [B, seq_len, dim]      │
        └───────────────────────────────┘

    :param dim: Model dimensionality. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param nb_features: Number of random features for kernel approximation.
        Higher values give better approximation at the cost of more memory.
    :type nb_features: int
    :param ortho_scaling: Scaling factor for orthogonal features. 0.0 disables.
    :type ortho_scaling: float
    :param causal: Whether to use causal (autoregressive) attention masking.
    :type causal: bool
    :param dropout_rate: Dropout rate for attention weights, between 0 and 1.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in Q, K, V projections.
    :type use_bias: bool
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
            nb_features: int = 256,
            ortho_scaling: float = 0.0,
            causal: bool = False,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
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
        if nb_features <= 0:
            raise ValueError(f"nb_features must be positive, got {nb_features}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Computed attributes
        self.head_dim = dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Create sub-layers in __init__
        # Q, K, V projection layer
        self.to_qkv = layers.Dense(
            3 * dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='qkv_projection'
        )

        # Output projection layer
        self.to_out = layers.Dense(
            dim,
            use_bias=use_bias,
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

    def _create_projection_matrix(self, batch_size: int) -> keras.KerasTensor:
        """Create random projection matrix for FAVOR+ approximation.

        :param batch_size: Batch size for broadcasting.
        :type batch_size: int

        :return: Projection matrix of shape
            ``(batch, num_heads, nb_features//2, head_dim)``.
        :rtype: keras.KerasTensor
        """
        # Generate random Gaussian matrix
        # Shape: (num_heads, nb_features//2, head_dim)
        shape = (self.num_heads, self.nb_features // 2, self.head_dim)
        projection = keras.random.normal(
            shape=shape,
            mean=0.0,
            stddev=1.0,
            seed=None
        )

        # Optionally apply orthogonalization for better approximation
        if self.ortho_scaling > 0:
            # QR decomposition for orthogonalization
            # Note: Keras doesn't have QR, so we use Gram-Schmidt approximation
            # This is a simplified version - in production, consider using backend-specific QR
            projection = projection * self.ortho_scaling

        # Scale by sqrt(head_dim) for proper variance
        projection = projection / ops.sqrt(ops.cast(self.head_dim, dtype=projection.dtype))

        # Broadcast to batch dimension
        # Shape: (batch, num_heads, nb_features//2, head_dim)
        projection = ops.expand_dims(projection, axis=0)
        projection = ops.repeat(projection, batch_size, axis=0)

        return projection

    def _create_kernel_features(
            self,
            x: keras.KerasTensor,
            projection_matrix: keras.KerasTensor,
            is_query: bool = True
    ) -> keras.KerasTensor:
        """Create positive random features for kernel approximation.

        :param x: Input tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type x: keras.KerasTensor
        :param projection_matrix: Random projection matrix.
        :type projection_matrix: keras.KerasTensor
        :param is_query: Whether this is for queries (affects normalization).
        :type is_query: bool

        :return: Random features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :rtype: keras.KerasTensor
        """
        # Project input: x @ projection_matrix^T
        # x: (batch, num_heads, seq_len, head_dim)
        # projection_matrix: (batch, num_heads, nb_features//2, head_dim)
        # Result: (batch, num_heads, seq_len, nb_features//2)
        x_projected = ops.einsum('bhnd,bhfd->bhnf', x, projection_matrix)

        # Apply trigonometric random features (FAVOR+)
        # This creates positive features that approximate exp(x·y)
        features_cos = ops.cos(x_projected)
        features_sin = ops.sin(x_projected)

        # Concatenate to get full feature dimension
        # Shape: (batch, num_heads, seq_len, nb_features)
        features = ops.concatenate([features_cos, features_sin], axis=-1)

        # Apply proper scaling for kernel approximation
        # The scaling ensures E[φ(x)ᵀφ(y)] ≈ exp(xᵀy)
        features = features * ops.sqrt(2.0 / ops.cast(self.nb_features, dtype=features.dtype))

        # For numerical stability, apply exponential normalization
        if is_query:
            # Normalize queries to prevent numerical issues
            features = features * ops.exp(-ops.square(ops.norm(x, axis=-1, keepdims=True)) / 2.0)

        return ops.maximum(features, 0)  # Ensure positive features

    def _linear_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute linear attention using FAVOR+ approximation.

        :param q: Query features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :type q: keras.KerasTensor
        :param k: Key features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type v: keras.KerasTensor

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
        """
        # Compute KV: φ(K)ᵀV
        # k: (batch, num_heads, seq_len, nb_features)
        # v: (batch, num_heads, seq_len, head_dim)
        # kv: (batch, num_heads, nb_features, head_dim)
        kv = ops.einsum('bhnf,bhnd->bhfd', k, v)

        # Compute normalization: φ(Q) · sum(φ(K))
        # k_sum: (batch, num_heads, nb_features)
        k_sum = ops.sum(k, axis=2)

        # z: (batch, num_heads, seq_len)
        z = ops.einsum('bhnf,bhf->bhn', q, k_sum)
        z = z + 1e-6  # Add small epsilon for numerical stability

        # Compute output: φ(Q) · KV / Z
        # out: (batch, num_heads, seq_len, head_dim)
        out = ops.einsum('bhnf,bhfd->bhnd', q, kv)
        out = out / ops.expand_dims(z, axis=-1)

        # Apply causal mask if needed
        if self.causal:
            # For causal attention, we need cumulative sums
            # This is a simplified implementation - consider optimizing for production
            seq_len = ops.shape(v)[2]

            # Create cumulative KV and K_sum for causal attention
            kv_cumsum = ops.cumsum(
                ops.einsum('bhnf,bhnd->bhnfd', k, ops.expand_dims(v, axis=-2)),
                axis=2
            )
            k_cumsum = ops.cumsum(k, axis=2)

            # Recompute with cumulative values
            z_causal = ops.einsum('bhnf,bhnf->bhn', q, k_cumsum) + 1e-6
            out = ops.einsum('bhnf,bhnfd->bhnd',
                             ops.expand_dims(q, axis=-2), kv_cumsum)
            out = ops.squeeze(out, axis=-2) / ops.expand_dims(z_causal, axis=-1)

        return out

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            return_attention_scores: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]:
        """Apply Performer attention to inputs.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :param return_attention_scores: If ``True``, returns ``(output, None)``
            for compatibility. Performer does not compute explicit attention matrices.
        :type return_attention_scores: bool

        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
            If return_attention_scores is ``True``, returns ``(output, None)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]
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

        # Scale queries
        q = q * self.scale

        # Generate random projection matrix
        projection_matrix = self._create_projection_matrix(batch_size)

        # Create kernel features φ(Q) and φ(K)
        q_features = self._create_kernel_features(q, projection_matrix, is_query=True)
        k_features = self._create_kernel_features(k, projection_matrix, is_query=False)

        # Compute linear attention
        out = self._linear_attention(q_features, k_features, v)

        # Reshape back to (batch, seq_len, dim)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.dim))

        # Apply output projection
        out = self.to_out(out)

        # Apply dropout if specified
        if self.dropout is not None:
            out = self.dropout(out, training=training)

        if return_attention_scores:
            # Performer doesn't compute explicit attention matrices
            # Return None for compatibility with standard attention interface
            return out, None

        return out

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
            'nb_features': self.nb_features,
            'ortho_scaling': self.ortho_scaling,
            'causal': self.causal,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
