"""
Task-specific heads and utilities for Vision Language Models.

This module provides specialized heads for different vision-language tasks
such as image captioning, visual question answering, and contrastive learning.
Following modern Keras 3 best practices for robust serialization.
"""

import keras
from keras import ops
from typing import List, Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms.rms_norm import RMSNorm
from .ffn.swiglu_ffn import SwiGLUFFN
from .attention.multi_head_attention import MultiHeadAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CaptioningHead(keras.layers.Layer):
    """
    Captioning head for autoregressive text generation.

    This layer implements a multi-layer decoder for image captioning tasks
    using cross-attention between text and vision features. It follows modern
    Keras 3 patterns for robust serialization and building.

    The architecture consists of:
    - Self-attention layers for text sequence modeling
    - Cross-attention layers for attending to vision features
    - Feed-forward networks for transformation
    - RMS normalization for stability
    - Final vocabulary projection for token generation

    Args:
        vocab_size (int): Size of the vocabulary. Must be positive.
        embed_dim (int, optional): Embedding dimension. Defaults to 768.
        num_layers (int, optional): Number of decoder layers. Must be positive. Defaults to 6.
        num_heads (int, optional): Number of attention heads. Must be positive and divide embed_dim. Defaults to 12.
        intermediate_size (int, optional): Size of the intermediate layer in FFN. Defaults to 3072.
        dropout_rate (float, optional): Dropout rate between 0 and 1. Defaults to 0.1.
        max_length (int, optional): Maximum generation length. Must be positive. Defaults to 128.
        kernel_initializer (str or initializer, optional): Initializer for kernel weights. Defaults to 'glorot_normal'.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        - text_features: Tensor with shape (batch_size, text_seq_len, embed_dim)
        - vision_features: Tensor with shape (batch_size, vision_seq_len, embed_dim)

    Output shape:
        Tensor with shape (batch_size, text_seq_len, vocab_size) containing logits for next token prediction.

    Example:
        ```python
        # Basic usage
        captioning_head = CaptioningHead(vocab_size=50000)

        # Advanced configuration
        captioning_head = CaptioningHead(
            vocab_size=30522,
            embed_dim=768,
            num_layers=6,
            num_heads=12,
            dropout_rate=0.1
        )

        # In forward pass
        text_features = keras.Input(shape=(128, 768))
        vision_features = keras.Input(shape=(196, 768))
        logits = captioning_head(text_features, vision_features)
        ```

    Raises:
        ValueError: If vocab_size, num_layers, or num_heads is not positive.
        ValueError: If embed_dim is not divisible by num_heads.
        ValueError: If dropout_rate is not between 0 and 1.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        dropout_rate: float = 0.1,
        max_length: int = 128,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE all sub-layers in __init__ (they will be unbuilt)
        self.cross_attention_layers = []
        self.self_attention_layers = []
        self.ffn_layers = []
        self.norm_layers = []

        for i in range(self.num_layers):
            # Cross-attention to vision features
            cross_attn = MultiHeadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"cross_attention_{i}"
            )
            self.cross_attention_layers.append(cross_attn)

            # Self-attention for text
            self_attn = MultiHeadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"self_attention_{i}"
            )
            self.self_attention_layers.append(self_attn)

            # FFN
            ffn = SwiGLUFFN(
                d_model=self.embed_dim,
                name=f"ffn_{i}"
            )
            self.ffn_layers.append(ffn)

            # Layer norms
            norm1 = RMSNorm(name=f"norm1_{i}")
            norm2 = RMSNorm(name=f"norm2_{i}")
            norm3 = RMSNorm(name=f"norm3_{i}")
            self.norm_layers.extend([norm1, norm2, norm3])

        # Output projection to vocabulary
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name="output_projection",
            kernel_initializer=self.kernel_initializer
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and explicitly build sub-layers for robust serialization.

        Args:
            input_shape: Shape of the input tensors. Expected to be a tuple of shapes
                         for (text_features, vision_features).
        """
        # For layers with many sub-layers, explicit building ensures proper serialization
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            text_shape, vision_shape = input_shape
        else:
            # Fallback if single shape provided
            text_shape = input_shape
            vision_shape = input_shape

        # Build sub-layers in computational order
        for i in range(self.num_layers):
            # Self-attention layers
            self.self_attention_layers[i].build(text_shape)

            # Cross-attention layers - text queries, vision keys/values
            self.cross_attention_layers[i].build(text_shape)

            # Normalization layers
            self.norm_layers[i * 3].build(text_shape)
            self.norm_layers[i * 3 + 1].build(text_shape)
            self.norm_layers[i * 3 + 2].build(text_shape)

            # FFN layers
            self.ffn_layers[i].build(text_shape)

        # Build output projection
        self.output_projection.build(text_shape)

        super().build(input_shape)

    def call(
        self,
        text_features: keras.KerasTensor,
        vision_features: keras.KerasTensor,
        causal_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the captioning head.

        Args:
            text_features: Text features of shape (batch_size, text_seq_len, embed_dim).
            vision_features: Vision features of shape (batch_size, vision_seq_len, embed_dim).
            causal_mask: Optional causal mask for text generation.
            training: Whether in training mode.

        Returns:
            Logits for next token prediction of shape (batch_size, text_seq_len, vocab_size).
        """
        x = text_features

        for i in range(self.num_layers):
            # Self-attention with causal mask
            attn_output = self.self_attention_layers[i](
                x, attention_mask=causal_mask, training=training
            )
            x = self.norm_layers[i * 3](x + attn_output)

            # Cross-attention to vision features
            cross_attn_output = self.cross_attention_layers[i](
                x, context=vision_features, training=training
            )
            x = self.norm_layers[i * 3 + 1](x + cross_attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.norm_layers[i * 3 + 2](x + ffn_output)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            text_shape, _ = input_shape
        else:
            text_shape = input_shape

        # Output shape: (batch_size, text_seq_len, vocab_size)
        output_shape = list(text_shape)
        output_shape[-1] = self.vocab_size
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "dropout_rate": self.dropout_rate,
            "max_length": self.max_length,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VQAHead(keras.layers.Layer):
    """
    Visual Question Answering head for answer classification/generation.

    This layer combines vision and text features using various pooling strategies
    and processes them through a multi-layer classifier for VQA tasks.

    The architecture supports different pooling strategies:
    - Mean pooling: Simple average over sequence dimension
    - Max pooling: Maximum values over sequence dimension
    - Attention pooling: Cross-attention between modalities for better fusion

    Args:
        num_answers (int): Number of possible answers (for classification). Must be positive.
        embed_dim (int, optional): Embedding dimension. Defaults to 768.
        hidden_dims (List[int], optional): List of hidden layer dimensions. Defaults to [512, 256].
        dropout_rate (float, optional): Dropout rate between 0 and 1. Defaults to 0.1.
        pooling_strategy (str, optional): Strategy for pooling multimodal features.
            Options: "mean", "max", "attention". Defaults to "attention".
        kernel_initializer (str or initializer, optional): Initializer for kernel weights. Defaults to 'glorot_normal'.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        - vision_features: Tensor with shape (batch_size, vision_seq_len, embed_dim)
        - text_features: Tensor with shape (batch_size, text_seq_len, embed_dim)

    Output shape:
        Tensor with shape (batch_size, num_answers) containing answer logits.

    Example:
        ```python
        # Basic usage
        vqa_head = VQAHead(num_answers=1000)

        # With attention pooling
        vqa_head = VQAHead(
            num_answers=3000,
            embed_dim=768,
            hidden_dims=[512, 256],
            pooling_strategy="attention"
        )

        # Forward pass
        vision_feats = keras.Input(shape=(196, 768))
        text_feats = keras.Input(shape=(128, 768))
        logits = vqa_head(vision_feats, text_feats)
        ```

    Raises:
        ValueError: If num_answers is not positive.
        ValueError: If embed_dim is not positive.
        ValueError: If dropout_rate is not between 0 and 1.
        ValueError: If pooling_strategy is not supported.
    """

    def __init__(
        self,
        num_answers: int,
        embed_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.1,
        pooling_strategy: str = "attention",
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_answers <= 0:
            raise ValueError(f"num_answers must be positive, got {num_answers}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if pooling_strategy not in ["mean", "max", "attention"]:
            raise ValueError(f"pooling_strategy must be one of ['mean', 'max', 'attention'], got {pooling_strategy}")

        # Store configuration
        self.num_answers = num_answers
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.pooling_strategy = pooling_strategy
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE all sub-layers in __init__
        # Attention pooling if specified
        if self.pooling_strategy == "attention":
            self.attention_pooling = MultiHeadAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout_rate=self.dropout_rate,
                name="attention_pooling"
            )
        else:
            self.attention_pooling = None

        # Hidden layers
        self.hidden_layers = []
        self.dropout_layers = []

        prev_dim = self.embed_dim * 2  # Concatenated vision and text features
        for i, hidden_dim in enumerate(self.hidden_dims):
            if hidden_dim <= 0:
                raise ValueError(f"All hidden dimensions must be positive, got {hidden_dim} at index {i}")

            hidden_layer = keras.layers.Dense(
                hidden_dim,
                activation="gelu",
                kernel_initializer=self.kernel_initializer,
                name=f"hidden_{i}"
            )
            dropout_layer = keras.layers.Dropout(self.dropout_rate, name=f"dropout_{i}")

            self.hidden_layers.append(hidden_layer)
            self.dropout_layers.append(dropout_layer)
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = keras.layers.Dense(
            self.num_answers,
            kernel_initializer=self.kernel_initializer,
            name="output_layer"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and explicitly build sub-layers for robust serialization.

        Args:
            input_shape: Shape of the input tensors. Expected to be a tuple of shapes
                         for (vision_features, text_features).
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            vision_shape, text_shape = input_shape
        else:
            # Fallback if single shape provided
            vision_shape = input_shape
            text_shape = input_shape

        # Build attention pooling if used
        if self.attention_pooling is not None:
            self.attention_pooling.build(vision_shape)

        # Build hidden layers
        # Input to first hidden layer is concatenated features
        concat_dim = self.embed_dim * 2
        concat_shape = list(vision_shape)
        concat_shape[-1] = concat_dim
        concat_shape = tuple(concat_shape)

        current_shape = concat_shape
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            hidden_layer.build(current_shape)
            dropout_layer.build(hidden_layer.compute_output_shape(current_shape))
            current_shape = hidden_layer.compute_output_shape(current_shape)

        # Build output layer
        self.output_layer.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        vision_features: keras.KerasTensor,
        text_features: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the VQA head.

        Args:
            vision_features: Vision features of shape (batch_size, vision_seq_len, embed_dim).
            text_features: Text features of shape (batch_size, text_seq_len, embed_dim).
            training: Whether in training mode.

        Returns:
            Answer logits of shape (batch_size, num_answers).
        """
        # Pool features based on strategy
        if self.pooling_strategy == "mean":
            vision_pooled = ops.mean(vision_features, axis=1)
            text_pooled = ops.mean(text_features, axis=1)
        elif self.pooling_strategy == "max":
            vision_pooled = ops.max(vision_features, axis=1)
            text_pooled = ops.max(text_features, axis=1)
        elif self.pooling_strategy == "attention":
            # Use cross-attention between vision and text for pooling
            vision_attended = self.attention_pooling(
                vision_features, context=text_features, training=training
            )
            text_attended = self.attention_pooling(
                text_features, context=vision_features, training=training
            )
            vision_pooled = ops.mean(vision_attended, axis=1)
            text_pooled = ops.mean(text_attended, axis=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Concatenate features
        x = ops.concatenate([vision_pooled, text_pooled], axis=-1)

        # Pass through hidden layers
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x, training=training)

        # Output layer
        logits = self.output_layer(x)

        return logits

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            vision_shape, _ = input_shape
            batch_size = vision_shape[0]
        else:
            batch_size = input_shape[0]

        return (batch_size, self.num_answers)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_answers": self.num_answers,
            "embed_dim": self.embed_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "pooling_strategy": self.pooling_strategy,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ContrastiveHead(keras.layers.Layer):
    """
    Contrastive learning head for image-text matching.

    This layer projects vision and text features to a shared embedding space
    and computes similarity scores for contrastive learning. Features are
    L2-normalized and similarity is computed with temperature scaling.

    The contrastive head enables:
    - Cross-modal retrieval (image-to-text and text-to-image)
    - Similarity-based ranking
    - Contrastive pre-training objectives

    Args:
        embed_dim (int, optional): Input embedding dimension. Defaults to 768.
        projection_dim (int, optional): Projection dimension for contrastive learning.
            Must be positive. Defaults to 256.
        temperature (float, optional): Temperature parameter for contrastive loss.
            Must be positive. Defaults to 0.07.
        kernel_initializer (str or initializer, optional): Initializer for kernel weights.
            Defaults to 'glorot_normal'.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        - vision_features: Tensor with shape (batch_size, embed_dim)
        - text_features: Tensor with shape (batch_size, embed_dim)

    Output shape:
        Dictionary containing:
        - vision_projected: (batch_size, projection_dim)
        - text_projected: (batch_size, projection_dim)
        - similarity_matrix: (batch_size, batch_size)
        - logits: (batch_size, batch_size) scaled by temperature

    Example:
        ```python
        # Basic usage
        contrastive_head = ContrastiveHead()

        # Custom configuration
        contrastive_head = ContrastiveHead(
            embed_dim=768,
            projection_dim=256,
            temperature=0.07
        )

        # Forward pass
        vision_feats = keras.Input(shape=(768,))
        text_feats = keras.Input(shape=(768,))
        outputs = contrastive_head(vision_feats, text_feats)
        # outputs contains projected features and similarity logits
        ```

    Raises:
        ValueError: If embed_dim or projection_dim is not positive.
        ValueError: If temperature is not positive.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be positive, got {projection_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Store configuration
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE sub-layers in __init__
        self.vision_projection = keras.layers.Dense(
            self.projection_dim,
            kernel_initializer=self.kernel_initializer,
            name="vision_projection"
        )

        self.text_projection = keras.layers.Dense(
            self.projection_dim,
            kernel_initializer=self.kernel_initializer,
            name="text_projection"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and explicitly build sub-layers for robust serialization.

        Args:
            input_shape: Shape of the input tensors. Expected to be a tuple of shapes
                         for (vision_features, text_features).
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            vision_shape, text_shape = input_shape
        else:
            # Fallback - assume both have same shape
            vision_shape = input_shape
            text_shape = input_shape

        # Build projection layers
        self.vision_projection.build(vision_shape)
        self.text_projection.build(text_shape)

        super().build(input_shape)

    def call(
        self,
        vision_features: keras.KerasTensor,
        text_features: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass of the contrastive head.

        Args:
            vision_features: Vision features of shape (batch_size, embed_dim).
            text_features: Text features of shape (batch_size, embed_dim).
            training: Whether in training mode.

        Returns:
            Dictionary containing projected features and similarity logits:
            - vision_projected: L2-normalized vision projections
            - text_projected: L2-normalized text projections
            - similarity_matrix: Raw cosine similarities
            - logits: Temperature-scaled similarities for contrastive loss
        """
        # Project to contrastive space
        vision_projected = self.vision_projection(vision_features)
        text_projected = self.text_projection(text_features)

        # L2 normalize
        vision_projected = ops.l2_normalize(vision_projected, axis=-1)
        text_projected = ops.l2_normalize(text_projected, axis=-1)

        # Compute similarity matrix
        similarity_matrix = ops.matmul(vision_projected, ops.transpose(text_projected))
        logits = similarity_matrix / self.temperature

        return {
            "vision_projected": vision_projected,
            "text_projected": text_projected,
            "similarity_matrix": similarity_matrix,
            "logits": logits,
        }

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute output shapes for all returned tensors."""
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            vision_shape, _ = input_shape
            batch_size = vision_shape[0]
        else:
            batch_size = input_shape[0]

        return {
            "vision_projected": (batch_size, self.projection_dim),
            "text_projected": (batch_size, self.projection_dim),
            "similarity_matrix": (batch_size, batch_size),
            "logits": (batch_size, batch_size),
        }

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "projection_dim": self.projection_dim,
            "temperature": self.temperature,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config

# ---------------------------------------------------------------------