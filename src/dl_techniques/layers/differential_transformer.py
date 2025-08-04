"""
Differential Transformer (DIFF Transformer) Implementation in TensorFlow
====================================================================

This implementation is based on the paper:
"DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context while canceling noise"

Architecture Schematic:
--------------------
                        DIFFERENTIAL TRANSFORMER
                               ARCHITECTURE

    Input                                      Output
     │                                           ▲
     ▼                                           │
┌────────────┐                             ┌───────────┐
│  Embedding │                             │ Classifier │
└────────────┘                             └───────────┘
     │                                           ▲
     ▼                                           │
┌────────────┐                             ┌───────────┐
│ Positional │                             │ Final Norm │
│ Embedding  │                             └───────────┘
└────────────┘                                   ▲
     │                                           │
     ▼                                           │
┌─────────────────────────┐                ┌───────────┐
│   Transformer Block 1   │                │  Pooling  │
└─────────────────────────┘                └───────────┘
     │                                           ▲
     ▼                                           │
┌─────────────────────────┐                      │
│         ...             │                      │
└─────────────────────────┘                      │
     │                                           │
     ▼                                           │
┌─────────────────────────┐                      │
│   Transformer Block N   │─────────────────────┘
└─────────────────────────┘

TRANSFORMER BLOCK STRUCTURE:                    DIFFERENTIAL ATTENTION:

Input                                          MHA1        MHA2
  │                                             │           │
  ▼                                             ▼           ▼
┌──────────┐                                 ┌─────────┐ ┌─────────┐
│ PreNorm  │                                 │ Attn 1  │ │ Attn 2  │
└──────────┘                                 └─────────┘ └─────────┘
  │                                              │         │
  ▼                                              │   λ×    │
┌──────────────┐                                 ▼   ▼     ▼
│ Differential │                          Output = Attn1 - λ·Attn2
│  Attention   │                                     │
└──────────────┘                                     ▼
  │                                           ┌────────────┐
  ▼                                           │  Project   │
┌──────────┐                                  └────────────┘
│ Residual │                                        │
└──────────┘                                        ▼
  │                                              Output
  ▼
┌──────────┐
│ PreNorm  │
└──────────┘
  │
  ▼
┌──────────┐
│    FFN   │
└──────────┘
  │
  ▼
┌──────────┐
│ Residual │
└──────────┘
  │
  ▼
 Output

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
   - Uses dual MultiHeadAttention layers with separate projection matrices
   - Incorporates learnable scalar λ for attention map balancing
   - λ initialization: 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))
   - Uses standard Keras layers for improved compatibility and performance

2. Key Components:
   - Pre-LayerNormalization for layer normalization
   - SwiGLU activation in feed-forward networks
   - Global Response Normalization
   - Differential attention with noise cancellation
"""

import keras
import tensorflow as tf
from keras.api import Model
from keras.api import layers
from keras.api import losses
from keras.api import optimizers
from keras.api import initializers
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .stochastic_depth import StochasticDepth
from .positional_embedding import PositionalEmbedding
from .attention.differential_attention import DifferentialMultiHeadAttention

# ---------------------------------------------------------------------


class FeedForward(layers.Layer):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
            activation: str = "swish",
            init_scale: float = 2.0,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.init_scale = init_scale
        self.fc1 = None
        self.fc2 = None
        self.dropout = None

    def build(self, input_shape):
        self.fc1 = layers.Dense(
            self.hidden_dim,
            activation=self.activation,
            kernel_initializer=initializers.VarianceScaling(
                scale=self.init_scale,
                mode='fan_out',
                distribution='truncated_normal'
            )
        )
        self.fc2 = layers.Dense(
            self.dim,
            kernel_initializer=initializers.VarianceScaling(
                scale=self.init_scale,
                mode='fan_in',
                distribution='truncated_normal'
            )
        )
        self.dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "activation": self.activation,
            "init_scale": self.init_scale
        })
        return config

# ---------------------------------------------------------------------


class TransformerBlock(layers.Layer):
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
            path_dropout: float = 0.0,
            **kwargs
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
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.attention_dropout_rate = attention_dropout
        self.path_dropout_rate = path_dropout
        self.attn = None


    def build(self, input_shape):
        # Main layers
        self.attn = DifferentialMultiHeadAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout_rate,
            attention_dropout=self.attention_dropout_rate
        )
        self.ff = FeedForward(
            dim=self.dim,
            hidden_dim=self.mlp_dim,
            dropout=self.dropout_rate
        )

        # Normalization layers - using standard LayerNormalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        # Stochastic depth for training stability
        self.stochastic_depth = StochasticDepth(self.path_dropout_rate) if self.path_dropout_rate > 0 else None

        super().build(input_shape)

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
        if self.stochastic_depth is not None:
            x = self.stochastic_depth([x, residual], training=training)
        else:
            x = x + residual

        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ff(x, training=training)
        if self.stochastic_depth is not None:
            x = self.stochastic_depth([x, residual], training=training)
        else:
            x = x + residual

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
            "attention_dropout": self.attention_dropout_rate,
            "path_dropout": self.path_dropout_rate
        })
        return config


# ---------------------------------------------------------------------


class DifferentialTransformer(Model):
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
            embedding_dropout: float = 0.0,
            **kwargs
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
        super().__init__(**kwargs)

        # Validate dimensions
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        # Store configuration
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.attention_dropout_rate = attention_dropout
        self.path_dropout_rate = path_dropout
        self.embedding_dropout_rate = embedding_dropout

        # Save config for serialization
        self.config = {
            "num_classes": num_classes,
            "dim": dim,
            "depth": depth,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mlp_dim": mlp_dim,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "path_dropout": path_dropout,
            "embedding_dropout": embedding_dropout
        }

    def build(self, input_shape):
        """Build the model layers based on input shape."""
        # Input embedding with proper initialization
        self.embedding = layers.Dense(
            self.dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02)
        )

        # Positional embedding
        self.pos_embedding = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.dim,
            dropout=self.embedding_dropout_rate
        )

        # Transformer layers with increasing path dropout
        self.layers = []
        for i in range(self.depth):
            # Increase path dropout linearly with depth
            layer_path_dropout = self.path_dropout_rate * float(i) / max(1, self.depth - 1)

            self.layers.append(
                TransformerBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    mlp_dim=self.mlp_dim,
                    dropout=self.dropout_rate,
                    attention_dropout=self.attention_dropout_rate,
                    path_dropout=layer_path_dropout
                )
            )

        # Output head
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.fc = layers.Dense(
            self.num_classes,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            bias_initializer="zeros"
        )

        super().build(input_shape)

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
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
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
    # Create input to force model building
    dummy_input = keras.Input(shape=(None, dim))

    # Create model
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

    # Build model with dummy input
    _ = model(dummy_input)

    # Configure optimizer with weight decay fix
    optimizer = optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta_1=beta1,
        beta_2=beta2,
        clipnorm=1.0  # Gradient clipping for stability
    )

    # Compile with loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy',
                 losses.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy'),
                 'top_k_categorical_accuracy']
    )

    return model

# ---------------------------------------------------------------------

def create_diff_transformer_for_vision(
        input_shape=(32, 32, 3),
        patch_size=4,
        num_classes=1000,
        dim=768,
        depth=12,
        num_heads=12,
        head_dim=64,
        mlp_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        path_dropout=0.1,
        embedding_dropout=0.1,
):
    """
    Create a Vision Transformer using the Differential Transformer architecture.

    Args:
        input_shape: Input image shape (height, width, channels)
        patch_size: Size of image patches
        num_classes: Number of classification classes
        dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mlp_dim: Hidden dimension of feed-forward network
        dropout: Dropout rate
        attention_dropout: Attention-specific dropout rate
        path_dropout: Stochastic depth rate
        embedding_dropout: Embedding dropout rate

    Returns:
        A Keras Model for image classification
    """
    # Calculate number of patches
    h, w, c = input_shape
    num_patches = (h // patch_size) * (w // patch_size)

    # Create data augmentation layers
    data_augmentation = keras.Sequential([
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ], name="data_augmentation")

    # Create the model inputs
    inputs = layers.Input(shape=input_shape)

    # Augment data
    x = data_augmentation(inputs)

    # Create patches
    patches = keras.ops.image.extract_patches(x, size=patch_size)
    patches = keras.ops.reshape(
        patches,
        (-1, num_patches, patch_size * patch_size * c)
    )

    # Patch embedding
    x = layers.Dense(dim)(patches)

    # Add positional embeddings
    positions = layers.Embedding(
        input_dim=num_patches,
        output_dim=dim
    )(keras.ops.arange(start=0, stop=num_patches, step=1))
    positions = keras.ops.expand_dims(positions, axis=0)
    x = x + positions
    x = layers.Dropout(embedding_dropout)(x)

    # Create transformer blocks
    for i in range(depth):
        # Calculate path dropout rate based on depth
        layer_path_dropout = path_dropout * float(i) / max(1, depth - 1)

        # Create transformer block
        x = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            path_dropout=layer_path_dropout
        )(x, layer_idx=i)

    # Create output head
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    # MLP head
    x = layers.Dense(mlp_dim // 2, activation=keras.activations.gelu)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(mlp_dim // 4, activation=keras.activations.gelu)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# ---------------------------------------------------------------------