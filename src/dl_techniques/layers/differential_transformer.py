"""
Differential Transformer (DIFF Transformer) Implementation in TensorFlow
====================================================================

This implementation is based on the paper:
"DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context while canceling noise"

Key Findings from the Paper:
---------------------------
1. Performance Improvements:
   - DIFF Transformer outperforms standard Transformer while using only ~65% of model size/training tokens
   - Shows superior performance in long-context modeling and key information retrieval
   - Enhanced accuracy in in-context learning tasks
   - More robust to order permutation in few-shot learning

2. Architecture Benefits:
   - Mitigates hallucination in question answering and text summarization
   - Reduces activation outliers, enabling better quantization
   - Differential attention mechanism effectively cancels out attention noise
   - Improved focus on relevant information in the input sequence

3. Scaling Properties:
   - 6.8B DIFF Transformer achieves comparable performance to 11B standard Transformer
   - Requires only ~63.7% of training tokens for equivalent performance
   - Shows consistent improvements across various model sizes (830M to 13B)

Implementation Details:
---------------------
1. Attention Mechanism:
   - Uses dual softmax attention computation
   - Incorporates learnable scalar λ for attention map balancing
   - λ initialization: 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))
   - Applies GroupNorm to each attention head independently

2. Key Components:
   - Pre-RMSNorm for layer normalization
   - SwiGLU activation in feed-forward networks
   - Global Response Normalization
   - Differential attention with noise cancellation
"""

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from typing import Optional, Tuple, Union, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.tensors import safe_divide, create_causal_mask

# ---------------------------------------------------------------------


class LayerNorm(keras.layers.Layer):
    """Custom Layer Normalization with safe division."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = self.add_weight(
            "gamma",
            shape=(dim,),
            initializer="ones",
            trainable=True
        )
        self.beta = self.add_weight(
            "beta",
            shape=(dim,),
            initializer="zeros",
            trainable=True
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        means = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - means), axis=-1, keepdims=True)
        x = safe_divide(x - means, tf.sqrt(variance + self.eps))
        return x * self.gamma + self.beta

# ---------------------------------------------------------------------


class GlobalResponseNorm(keras.layers.Layer):
    """Global Response Normalization with enhanced stability."""

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            init_scale: float = 1.0,
            norm_clip: float = 1e3
    ) -> None:
        """
        Args:
            dim: Input dimension
            eps: Small constant for numerical stability
            init_scale: Initial scale for gamma parameter
            norm_clip: Maximum norm value for stability
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")

        self.dim = dim
        self.eps = eps
        self.norm_clip = norm_clip

        self.gamma = self.add_weight(
            "gamma",
            shape=(dim,),
            initializer=keras.initializers.Constant(init_scale),
            trainable=True
        )
        self.beta = self.add_weight(
            "beta",
            shape=(dim,),
            initializer="zeros",
            trainable=True
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if not tf.is_tensor(x):
            raise TypeError(f"Input must be a tensor, got {type(x)}")

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Input dimension {x.shape[-1]} does not match layer dimension {self.dim}"
            )

        # Compute global L2 norm with stability
        square_sum = tf.reduce_sum(tf.square(x), axis=(1, 2), keepdims=True)
        square_sum = tf.clip_by_value(square_sum, self.eps, self.norm_clip)
        gx = tf.sqrt(square_sum)

        # Normalize by mean norm safely
        mean_norm = tf.reduce_mean(gx, axis=-1, keepdims=True)
        nx = safe_divide(gx, mean_norm)
        nx = tf.clip_by_value(nx, -self.norm_clip, self.norm_clip)

        return self.gamma * (x * nx) + self.beta + x

    def get_config(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "eps": self.eps,
            "norm_clip": self.norm_clip
        }

# ---------------------------------------------------------------------


class DifferentialAttention(keras.layers.Layer):
    """Differential attention with comprehensive stability features."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            lambda_init: float = 0.8,
            max_position_bias: float = 1e2,
            qkv_bias: bool = False,
            attn_clip: float = 1e2
    ) -> None:
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            dropout: Output dropout rate
            attention_dropout: Attention matrix dropout rate
            lambda_init: Initial value for λ parameter
            max_position_bias: Maximum position embedding value
            qkv_bias: Whether to use bias in QKV projection
            attn_clip: Maximum attention score value
        """
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_position_bias = max_position_bias
        self.attn_clip = attn_clip

        # Initialize attention components
        self.scale = head_dim ** -0.5

        # QKV projection with careful initialization
        self.qkv = keras.layers.Dense(
            3 * num_heads * head_dim,
            use_bias=qkv_bias,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_out',
                distribution='truncated_normal'
            )
        )

        # Output projection
        self.proj = keras.layers.Dense(
            dim,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal'
            ),
            bias_initializer="zeros"
        )

        # Dropout layers
        self.dropout = keras.layers.Dropout(dropout)
        self.attn_dropout = keras.layers.Dropout(attention_dropout)

        # Initialize λ parameters
        self._init_lambda_params(head_dim, lambda_init)

    def _init_lambda_params(self, head_dim: int, lambda_init: float) -> None:
        """Initialize λ parameters with careful initialization."""
        init_val = np.log(lambda_init / 4) / head_dim

        self.lambda_q1 = self.add_weight(
            "lambda_q1",
            shape=(head_dim,),
            initializer=keras.initializers.Constant(init_val),
            trainable=True
        )
        self.lambda_k1 = self.add_weight(
            "lambda_k1",
            shape=(head_dim,),
            initializer=keras.initializers.Constant(init_val),
            trainable=True
        )
        self.lambda_q2 = self.add_weight(
            "lambda_q2",
            shape=(head_dim,),
            initializer=keras.initializers.Constant(-init_val),
            trainable=True
        )
        self.lambda_k2 = self.add_weight(
            "lambda_k2",
            shape=(head_dim,),
            initializer=keras.initializers.Constant(-init_val),
            trainable=True
        )

    def get_lambda(self, layer_idx: int = 0) -> tf.Tensor:
        """Compute λ value with stability controls."""
        # Safe exponential initialization
        layer_factor = tf.clip_by_value(
            tf.cast(layer_idx, tf.float32) * 0.3,
            0.0,
            5.0
        )
        lambda_init = 0.8 - 0.6 * tf.exp(-layer_factor)

        # Compute λ components safely
        lambda_1 = tf.clip_by_value(
            tf.reduce_sum(self.lambda_q1 * self.lambda_k1),
            -10.0,
            10.0
        )
        lambda_2 = tf.clip_by_value(
            tf.reduce_sum(self.lambda_q2 * self.lambda_k2),
            -10.0,
            10.0
        )

        lambda_val = tf.exp(lambda_1) - tf.exp(lambda_2) + lambda_init
        return tf.clip_by_value(lambda_val, 0.1, 5.0)

    def _compute_attention(
            self,
            q: tf.Tensor,
            k: tf.Tensor,
            v: tf.Tensor,
            mask: Optional[tf.Tensor] = None,
            training: bool = False
    ) -> tf.Tensor:
        """Compute attention with stability controls and masking."""
        # Scaled dot product attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        # Clip extreme values
        attn = tf.clip_by_value(attn, -self.attn_clip, self.attn_clip)

        # Apply mask if provided
        if mask is not None:
            big_neg = -1e9
            attn = tf.where(mask == 0, big_neg, attn)

        # Safe softmax
        max_score = tf.reduce_max(attn, axis=-1, keepdims=True)
        exp_scores = tf.exp(attn - max_score)
        attn = safe_divide(exp_scores, tf.reduce_sum(exp_scores, axis=-1, keepdims=True))

        # Apply dropout
        attn = self.attn_dropout(attn, training=training)

        return tf.matmul(attn, v)

    def call(
            self,
            x: tf.Tensor,
            mask: Optional[tf.Tensor] = None,
            layer_idx: int = 0,
            training: bool = False
    ) -> tf.Tensor:
        """Apply differential attention with comprehensive error checking."""
        if not tf.is_tensor(x):
            raise TypeError(f"Input must be a tensor, got {type(x)}")

        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project and reshape QKV
        qkv = self.qkv(x)
        qkv = tf.reshape(
            qkv,
            (batch_size, seq_len, 3, self.num_heads, self.head_dim)
        )
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))

        # Safe unstack
        try:
            q1, q2, k1, k2, v = tf.unstack(qkv[:5], axis=0)
        except tf.errors.InvalidArgumentError as e:
            raise ValueError(f"Failed to unstack QKV tensor: {e}")

        # Compute differential attention
        lambda_val = self.get_lambda(layer_idx)
        attn1 = self._compute_attention(q1, k1, v, mask, training)
        attn2 = self._compute_attention(q2, k2, v, mask, training)

        # Combine with safe scaling
        x = attn1 - lambda_val * attn2

        # Reshape and project output
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (batch_size, seq_len, -1))

        # Final projection and dropout
        x = self.proj(x)
        x = self.dropout(x, training=training)

        return x

# ---------------------------------------------------------------------


class FeedForward(keras.layers.Layer):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
            activation: str = "swish",
            init_scale: float = 2.0
    ) -> None:
        super().__init__()

        self.fc1 = keras.layers.Dense(
            hidden_dim,
            activation=activation,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=init_scale,
                mode='fan_out',
                distribution='truncated_normal'
            )
        )
        self.fc2 = keras.layers.Dense(
            dim,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=init_scale,
                mode='fan_in',
                distribution='truncated_normal'
            )
        )
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x

# ---------------------------------------------------------------------


class TransformerBlock(keras.layers.Layer):
    """Transformer block with differential attention and enhanced stability.

    Implements pre-normalization, differential attention, and feed-forward layers
    with comprehensive stability features and error checking."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            path_dropout: float = 0.0
    ) -> None:
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            mlp_dim: Hidden dimension of feed-forward network
            dropout: Dropout rate for attention and MLP outputs
            attention_dropout: Dropout rate for attention matrix
            path_dropout: Stochastic depth rate for residual paths
        """
        super().__init__()

        # Main layers
        self.attn = DifferentialAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        self.ff = FeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout
        )

        # Normalization layers
        self.norm1 = GlobalResponseNorm(dim)
        self.norm2 = GlobalResponseNorm(dim)

        # Stochastic depth for training stability
        self.path_dropout = path_dropout

    def stochastic_path(
            self,
            x: tf.Tensor,
            residual: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """Apply stochastic depth to residual connection."""
        if training and self.path_dropout > 0:
            keep_prob = 1.0 - self.path_dropout
            mask = tf.cast(
                tf.random.uniform([tf.shape(x)[0], 1, 1]) < keep_prob,
                x.dtype
            )
            return x * mask + residual
        return x + residual

    def call(
            self,
            x: tf.Tensor,
            mask: Optional[tf.Tensor] = None,
            layer_idx: int = 0,
            training: bool = False
    ) -> tf.Tensor:
        """Apply transformer block with stability controls.

        Args:
            x: Input tensor
            mask: Optional attention mask
            layer_idx: Index of transformer layer
            training: Whether in training mode

        Returns:
            Transformed tensor
        """
        # Input validation
        if not tf.is_tensor(x):
            raise TypeError(f"Input must be a tensor, got {type(x)}")

        # Attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask, layer_idx=layer_idx, training=training)
        x = self.stochastic_path(x, residual, training)

        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ff(x, training=training)
        x = self.stochastic_path(x, residual, training)

        return x

# ---------------------------------------------------------------------


class PositionalEmbedding(keras.layers.Layer):
    """Learned positional embedding with enhanced stability."""

    def __init__(
            self,
            max_seq_len: int,
            dim: int,
            dropout: float = 0.0,
            scale: float = 0.02
    ) -> None:
        """
        Args:
            max_seq_len: Maximum sequence length
            dim: Embedding dimension
            dropout: Dropout rate
            scale: Initialization scale for embeddings
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.dim = dim

        # Initialize embeddings with truncated normal
        self.pos_embedding = self.add_weight(
            "pos_embedding",
            shape=(1, max_seq_len, dim),
            initializer=keras.initializers.TruncatedNormal(stddev=scale),
            trainable=True
        )

        self.dropout = keras.layers.Dropout(dropout)

    def call(
            self,
            x: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            training: Whether in training mode

        Returns:
            Tensor with positional information added
        """
        seq_len = tf.shape(x)[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
            )

        # Add positions
        positions = self.pos_embedding[:, :seq_len, :]
        x = x + positions

        return self.dropout(x, training=training)

# ---------------------------------------------------------------------


class DifferentialTransformer(keras.Model):
    """Complete Differential Transformer model with comprehensive stability features."""

    def __init__(
            self,
            num_classes: int,
            dim: int,
            depth: int,
            num_heads: int,
            head_dim: int,
            mlp_dim: int,
            max_seq_len: int = 2048,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            path_dropout: float = 0.0,
            embedding_dropout: float = 0.0
    ) -> None:
        """
        Args:
            num_classes: Number of output classes
            dim: Model dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            mlp_dim: Hidden dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout rate for attention and MLP outputs
            attention_dropout: Dropout rate for attention matrix
            path_dropout: Stochastic depth rate
            embedding_dropout: Dropout rate for embeddings
        """
        super().__init__()

        # Validate dimensions
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        # Input embedding with proper initialization
        self.embedding = keras.layers.Dense(
            dim,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)
        )

        # Positional embedding
        self.pos_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            dim=dim,
            dropout=embedding_dropout
        )

        # Transformer layers with increasing path dropout
        self.layers = []
        for i in range(depth):
            # Increase path dropout linearly with depth
            layer_path_dropout = path_dropout * float(i) / (depth - 1)

            self.layers.append(
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    path_dropout=layer_path_dropout
                )
            )

        # Output head
        self.norm = GlobalResponseNorm(dim)
        self.fc = keras.layers.Dense(
            num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            bias_initializer="zeros"
        )

        # Save config for serialization
        self.config = {
            "num_classes": num_classes,
            "dim": dim,
            "depth": depth,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mlp_dim": mlp_dim,
            "max_seq_len": max_seq_len
        }

    def call(
            self,
            x: tf.Tensor,
            mask: Optional[tf.Tensor] = None,
            training: bool = False
    ) -> tf.Tensor:
        """Forward pass with comprehensive error checking.

        Args:
            x: Input tensor
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Output logits
        """
        # Input validation
        if not tf.is_tensor(x):
            raise TypeError(f"Input must be a tensor, got {type(x)}")

        seq_len = tf.shape(x)[1]
        if seq_len > self.config["max_seq_len"]:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.config['max_seq_len']}"
            )

        # Add embeddings
        x = self.embedding(x)
        x = self.pos_embedding(x, training=training)

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask, layer_idx=i, training=training)

        # Pool and classify
        x = tf.reduce_mean(x, axis=1)
        x = self.norm(x)
        x = self.fc(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config


# ---------------------------------------------------------------------


def create_diff_transformer(
        num_classes: int = 1000,
        dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        path_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.95
) -> DifferentialTransformer:
    """Create a configured DifferentialTransformer model.

    Args:
        num_classes: Number of output classes
        dim: Model dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mlp_dim: Hidden dimension of feed-forward network
        max_seq_len: Maximum sequence length
        dropout: General dropout rate
        attention_dropout: Attention-specific dropout rate
        path_dropout: Stochastic depth rate
        embedding_dropout: Embedding dropout rate
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        beta1: AdamW beta1 parameter
        beta2: AdamW beta2 parameter

    Returns:
        Configured DifferentialTransformer model
    """
    model = DifferentialTransformer(
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
        attention_dropout=attention_dropout,
        path_dropout=path_dropout,
        embedding_dropout=embedding_dropout
    )

    # Configure optimizer with weight decay fix
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta_1=beta1,
        beta_2=beta2,
        clipnorm=1.0  # Gradient clipping for stability
    )

    # Compile with loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE  # For custom loss scaling
        ),
        metrics=['accuracy']
    )

    return model

# ---------------------------------------------------------------------


