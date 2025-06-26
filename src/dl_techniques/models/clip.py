"""
CLIP (Contrastive Language-Image Pre-training) Model Implementation

This implementation follows modern transformer architecture best practices with:
- RMSNorm instead of LayerNorm for better stability
- Grouped Query Attention (GQA) for memory efficiency
- RoPE positional embeddings for better length extrapolation
- SwiGLU FFN for improved performance
- Stochastic Depth for regularization
- Pre-layer normalization for training stability

The model implements the contrastive learning framework that learns joint representations
of images and text by maximizing similarity between matching pairs while minimizing
similarity between non-matching pairs in a shared embedding space.

Mathematical Framework:
    1. Image encoder: f_I(image) → R^d (ViT with patches)
    2. Text encoder: f_T(text) → R^d (Transformer with tokens)
    3. Similarity: S = f_I(I) · f_T(T)^T / τ (temperature-scaled cosine similarity)
    4. Contrastive loss: symmetric cross-entropy on similarity matrix

Architecture Benefits:
    - Zero-shot transfer capabilities through shared embedding space
    - Scalable contrastive learning on large datasets
    - Rich multimodal representations without explicit supervision
    - Efficient inference through pre-computed embeddings

References:
    - Radford, A., et al. (2021). "Learning Transferable Visual Representations
      from Natural Language Supervision." https://arxiv.org/abs/2103.00020

    - Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models."
      https://arxiv.org/abs/2302.13971 (SwiGLU and RMSNorm usage)

    - Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints." https://arxiv.org/abs/2305.13245

Author: Deep Learning Techniques Library
License: MIT
"""

import keras
from keras import layers, ops
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn.swiglu_ffn import SwiGLUFFN
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.group_query_attention_layer import GroupedQueryAttention

# ---------------------------------------------------------------------

@dataclass
class CLIPConfig:
    """
    Configuration dataclass for CLIP model with modern architecture settings.

    This configuration supports both standard CLIP architectures and modern improvements
    like Grouped Query Attention, SwiGLU, and RMSNorm for better performance and efficiency.

    Attributes:
        Vision encoder configuration:
            image_size: Input image size (height and width)
            patch_size: Size of image patches for vision transformer
            vision_layers: Number of transformer layers in vision encoder
            vision_width: Hidden dimension of vision transformer
            vision_heads: Number of attention heads in vision transformer
            vision_kv_heads: Number of key-value heads for vision GQA

        Text encoder configuration:
            vocab_size: Size of text vocabulary
            context_length: Maximum text sequence length
            text_layers: Number of transformer layers in text encoder
            text_width: Hidden dimension of text transformer
            text_heads: Number of attention heads in text transformer
            text_kv_heads: Number of key-value heads for text attention

        Shared configuration:
            embed_dim: Dimension of shared embedding space

        FFN configuration:
            ffn_expansion_factor: Expansion factor for feed-forward networks
            ffn_multiple_of: Round FFN hidden dim to multiple of this value

        Regularization:
            dropout_prob: General dropout probability
            attention_dropout: Dropout probability for attention weights
            stochastic_depth_prob: Maximum stochastic depth drop rate

        Normalization:
            rms_norm_eps: Epsilon value for RMSNorm numerical stability

        Positional encoding:
            rope_percentage: Fraction of attention head dims to apply RoPE to

        Training specifics:
            logit_scale_init: Initial value for learnable temperature parameter
    """

    # Vision encoder configuration
    image_size: int = 224
    patch_size: int = 16
    vision_layers: int = 12
    vision_width: int = 768
    vision_heads: int = 12
    vision_kv_heads: int = 4

    # Text encoder configuration
    vocab_size: int = 49408
    context_length: int = 77
    text_layers: int = 12
    text_width: int = 512
    text_heads: int = 8
    text_kv_heads: int = 8  # Standard attention for text

    # Shared configuration
    embed_dim: int = 512

    # FFN configuration
    ffn_expansion_factor: int = 4
    ffn_multiple_of: int = 256

    # Regularization
    dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    stochastic_depth_prob: float = 0.1

    # Normalization
    rms_norm_eps: float = 1e-6

    # Positional encoding
    rope_percentage: float = 0.5

    # Training specifics
    logit_scale_init: float = 2.6592  # ln(1/0.07)

    def __post_init__(self) -> None:
        """Compute derived properties and validate configuration."""
        # Derived properties
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.vision_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validation
        self._validate_config()

        logger.info(f"CLIPConfig initialized: image_size={self.image_size}, "
                   f"vision_layers={self.vision_layers}, text_layers={self.text_layers}, "
                   f"num_patches={self.num_patches}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Image size and patch size validation
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

        # Vision transformer validation
        if self.vision_width % self.vision_heads != 0:
            raise ValueError(
                f"vision_width ({self.vision_width}) must be divisible by "
                f"vision_heads ({self.vision_heads})"
            )
        if self.vision_heads % self.vision_kv_heads != 0:
            raise ValueError(
                f"vision_heads ({self.vision_heads}) must be divisible by "
                f"vision_kv_heads ({self.vision_kv_heads})"
            )

        # Text transformer validation
        if self.text_width % self.text_heads != 0:
            raise ValueError(
                f"text_width ({self.text_width}) must be divisible by "
                f"text_heads ({self.text_heads})"
            )
        if self.text_heads % self.text_kv_heads != 0:
            raise ValueError(
                f"text_heads ({self.text_heads}) must be divisible by "
                f"text_kv_heads ({self.text_kv_heads})"
            )

        # Dropout validation
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")
        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1], got {self.attention_dropout}")
        if not 0.0 <= self.stochastic_depth_prob <= 1.0:
            raise ValueError(f"stochastic_depth_prob must be in [0, 1], got {self.stochastic_depth_prob}")

        # Other validation
        if self.rms_norm_eps <= 0.0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if not 0.0 <= self.rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {self.rope_percentage}")

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """
    Transformer block with modern architecture improvements.

    Features:
    - Pre-layer normalization with RMSNorm
    - Grouped Query Attention for memory efficiency
    - SwiGLU FFN for better performance
    - Stochastic depth for regularization
    - Residual connections around each sub-layer
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int,
        layer_idx: int,
        total_layers: int,
        ffn_expansion_factor: int = 4,
        ffn_multiple_of: int = 256,
        attention_dropout: float = 0.0,
        dropout_prob: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        rms_norm_eps: float = 1e-6,
        rope_percentage: float = 0.5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.attention_dropout = attention_dropout
        self.dropout_prob = dropout_prob
        self.stochastic_depth_prob = stochastic_depth_prob
        self.rms_norm_eps = rms_norm_eps
        self.rope_percentage = rope_percentage

        # Will be initialized in build()
        self.attention_norm = None
        self.attention = None
        self.ffn_norm = None
        self.ffn = None
        self.attn_stochastic_depth = None
        self.ffn_stochastic_depth = None
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build the transformer block components."""
        self._build_input_shape = input_shape

        # Linear scaling of stochastic depth rate
        depth_rate = self.stochastic_depth_prob * self.layer_idx / max(1, self.total_layers - 1)

        # Pre-attention normalization
        self.attention_norm = RMSNorm(
            epsilon=self.rms_norm_eps,
            name='attention_norm'
        )

        # Grouped Query Attention
        self.attention = GroupedQueryAttention(
            d_model=self.d_model,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            max_seq_len=self.max_seq_len,
            attention_dropout=self.attention_dropout,
            rope_percentage=self.rope_percentage,
            name='attention'
        )

        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(
            epsilon=self.rms_norm_eps,
            name='ffn_norm'
        )

        # SwiGLU Feed-Forward Network
        self.ffn = SwiGLUFFN(
            d_model=self.d_model,
            ffn_expansion_factor=self.ffn_expansion_factor,
            ffn_multiple_of=self.ffn_multiple_of,
            dropout_prob=self.dropout_prob,
            name='ffn'
        )

        # Stochastic depth for regularization
        self.attn_stochastic_depth = StochasticDepth(
            drop_rate=depth_rate,
            name='attn_stochastic_depth'
        )
        self.ffn_stochastic_depth = StochasticDepth(
            drop_rate=depth_rate,
            name='ffn_stochastic_depth'
        )

        # Build sublayers
        self.attention_norm.build(input_shape)
        self.attention.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.ffn.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Apply transformer block processing."""
        x = inputs

        # Pre-norm attention with residual connection and stochastic depth
        attn_input = self.attention_norm(x, training=training)
        attn_output = self.attention(attn_input, training=training, mask=mask)
        attn_output = self.attn_stochastic_depth(attn_output, training=training)
        x = x + attn_output

        # Pre-norm FFN with residual connection and stochastic depth
        ffn_input = self.ffn_norm(x, training=training)
        ffn_output = self.ffn(ffn_input, training=training)
        ffn_output = self.ffn_stochastic_depth(ffn_output, training=training)
        x = x + ffn_output

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "max_seq_len": self.max_seq_len,
            "layer_idx": self.layer_idx,
            "total_layers": self.total_layers,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "ffn_multiple_of": self.ffn_multiple_of,
            "attention_dropout": self.attention_dropout,
            "dropout_prob": self.dropout_prob,
            "stochastic_depth_prob": self.stochastic_depth_prob,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_percentage": self.rope_percentage,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisionTransformer(layers.Layer):
    """
    Vision Transformer encoder with patch embeddings.

    Converts input images to patch embeddings and processes them through
    a stack of transformer blocks with a learnable class token.
    """

    def __init__(
        self,
        config: CLIPConfig,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.vision_width = config.vision_width
        self.vision_layers = config.vision_layers
        self.vision_heads = config.vision_heads
        self.vision_kv_heads = config.vision_kv_heads

        # Derived properties
        self.num_patches = config.num_patches
        self.seq_len = config.vision_seq_len

        # Will be initialized in build()
        self.patch_embedding = None
        self.class_token = None
        self.transformer_blocks = None
        self.final_norm = None
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build the vision transformer components."""
        self._build_input_shape = input_shape

        # Calculate patch embedding input size
        patch_dim = self.patch_size * self.patch_size * input_shape[-1]

        # Patch embedding projection
        self.patch_embedding = layers.Dense(
            self.vision_width,
            use_bias=False,
            name='patch_embedding'
        )

        # Class token
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.vision_width),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(self.vision_layers):
            block = TransformerBlock(
                d_model=self.vision_width,
                n_head=self.vision_heads,
                n_kv_head=self.vision_kv_heads,
                max_seq_len=self.seq_len,
                layer_idx=i,
                total_layers=self.vision_layers,
                ffn_expansion_factor=self.config.ffn_expansion_factor,
                ffn_multiple_of=self.config.ffn_multiple_of,
                attention_dropout=self.config.attention_dropout,
                dropout_prob=self.config.dropout_prob,
                stochastic_depth_prob=self.config.stochastic_depth_prob,
                rms_norm_eps=self.config.rms_norm_eps,
                rope_percentage=self.config.rope_percentage,
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(block)

        # Final layer norm
        self.final_norm = RMSNorm(
            epsilon=self.config.rms_norm_eps,
            name='final_norm'
        )

        # Build sublayers
        # Patch embedding expects flattened patches
        patch_input_shape = (input_shape[0], self.num_patches, patch_dim)
        self.patch_embedding.build(patch_input_shape)

        # Transformer blocks expect (batch, seq_len, d_model)
        transformer_input_shape = (input_shape[0], self.seq_len, self.vision_width)
        for block in self.transformer_blocks:
            block.build(transformer_input_shape)

        self.final_norm.build(transformer_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Process images through vision transformer."""
        batch_size = ops.shape(inputs)[0]

        # Convert to patches and flatten
        # Use conv2d to extract patches more efficiently
        patches = layers.Conv2D(
            filters=self.vision_width,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            name='patch_conv'
        )(inputs)

        # Reshape to sequence format
        patches = ops.reshape(patches, (batch_size, self.num_patches, self.vision_width))

        # Add class token
        class_tokens = ops.broadcast_to(self.class_token, (batch_size, 1, self.vision_width))
        x = ops.concatenate([class_tokens, patches], axis=1)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Apply final normalization
        x = self.final_norm(x, training=training)

        # Return class token embedding
        return x[:, 0]  # (batch_size, vision_width)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "vision_width": self.vision_width,
            "vision_layers": self.vision_layers,
            "vision_heads": self.vision_heads,
            "vision_kv_heads": self.vision_kv_heads,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TextTransformer(layers.Layer):
    """
    Text Transformer encoder with token embeddings.

    Processes tokenized text through embedding layer and transformer blocks,
    extracting the final token representation for each sequence.
    """

    def __init__(
        self,
        config: CLIPConfig,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.text_width = config.text_width
        self.text_layers = config.text_layers
        self.text_heads = config.text_heads
        self.text_kv_heads = config.text_kv_heads

        # Will be initialized in build()
        self.token_embedding = None
        self.transformer_blocks = None
        self.final_norm = None
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build the text transformer components."""
        self._build_input_shape = input_shape

        # Token embeddings
        self.token_embedding = layers.Embedding(
            self.vocab_size,
            self.text_width,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name='token_embedding'
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(self.text_layers):
            block = TransformerBlock(
                d_model=self.text_width,
                n_head=self.text_heads,
                n_kv_head=self.text_kv_heads,
                max_seq_len=self.context_length,
                layer_idx=i,
                total_layers=self.text_layers,
                ffn_expansion_factor=self.config.ffn_expansion_factor,
                ffn_multiple_of=self.config.ffn_multiple_of,
                attention_dropout=self.config.attention_dropout,
                dropout_prob=self.config.dropout_prob,
                stochastic_depth_prob=self.config.stochastic_depth_prob,
                rms_norm_eps=self.config.rms_norm_eps,
                rope_percentage=self.config.rope_percentage,
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(block)

        # Final layer norm
        self.final_norm = RMSNorm(
            epsilon=self.config.rms_norm_eps,
            name='final_norm'
        )

        # Build sublayers
        self.token_embedding.build(input_shape)

        # Transformer input shape after embedding
        transformer_input_shape = (input_shape[0], input_shape[1], self.text_width)
        for block in self.transformer_blocks:
            block.build(transformer_input_shape)

        self.final_norm.build(transformer_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Process text tokens through text transformer."""
        seq_len = ops.shape(inputs)[1]

        # Token embeddings
        x = self.token_embedding(inputs)

        # Create causal mask for autoregressive attention
        mask = ops.triu(ops.ones((seq_len, seq_len), dtype='bool'), k=1)
        mask = ops.expand_dims(ops.expand_dims(mask, 0), 0)  # (1, 1, seq_len, seq_len)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training, mask=mask)

        # Apply final normalization
        x = self.final_norm(x, training=training)

        # Extract text features from the last non-padding token
        # Assumes padding token ID is 0
        sequence_lengths = ops.sum(ops.cast(inputs != 0, 'int32'), axis=1)
        sequence_lengths = ops.clip(sequence_lengths - 1, 0, seq_len - 1)

        # Gather embeddings at sequence end positions
        batch_indices = ops.arange(ops.shape(x)[0])
        text_features = x[batch_indices, sequence_lengths]

        return text_features

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "text_width": self.text_width,
            "text_layers": self.text_layers,
            "text_heads": self.text_heads,
            "text_kv_heads": self.text_kv_heads,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CLIPModel(keras.Model):
    """
    CLIP model with vision and text encoders.

    Implements the complete CLIP architecture with separate vision and text encoders
    that project to a shared embedding space for contrastive learning.
    """

    def __init__(
        self,
        config: CLIPConfig,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.embed_dim = config.embed_dim

        # Will be initialized in build()
        self.vision_encoder = None
        self.text_encoder = None
        self.vision_projection = None
        self.text_projection = None
        self.logit_scale = None

        logger.info(f"CLIPModel initialized with embed_dim={self.embed_dim}")

    def build(self, input_shape) -> None:
        """Build the CLIP model components."""
        # Vision encoder
        self.vision_encoder = VisionTransformer(self.config, name='vision_encoder')

        # Text encoder
        self.text_encoder = TextTransformer(self.config, name='text_encoder')

        # Projection layers to shared embedding space
        self.vision_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            name='vision_projection'
        )

        self.text_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            name='text_projection'
        )

        # Learnable temperature parameter for contrastive loss
        self.logit_scale = self.add_weight(
            name='logit_scale',
            shape=(),
            initializer=keras.initializers.Constant(self.config.logit_scale_init),
            trainable=True
        )

        # Build encoders with appropriate input shapes
        if isinstance(input_shape, dict):
            if 'image' in input_shape:
                self.vision_encoder.build(input_shape['image'])
                self.vision_projection.build((input_shape['image'][0], self.config.vision_width))
            if 'text' in input_shape:
                self.text_encoder.build(input_shape['text'])
                self.text_projection.build((input_shape['text'][0], self.config.text_width))
        else:
            # Assume tuple format (image_shape, text_shape)
            if len(input_shape) >= 1:
                self.vision_encoder.build(input_shape[0])
                self.vision_projection.build((input_shape[0][0], self.config.vision_width))
            if len(input_shape) >= 2:
                self.text_encoder.build(input_shape[1])
                self.text_projection.build((input_shape[1][0], self.config.text_width))

        super().build(input_shape)

    def encode_image(
        self,
        images: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Encode images to shared embedding space."""
        # Get image features from vision encoder
        image_features = self.vision_encoder(images, training=training)

        # Project to shared embedding space
        image_features = self.vision_projection(image_features)

        # L2 normalize features
        image_features = image_features / ops.norm(image_features, axis=-1, keepdims=True)

        return image_features

    def encode_text(
        self,
        text_ids: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Encode text to shared embedding space."""
        # Get text features from text encoder
        text_features = self.text_encoder(text_ids, training=training)

        # Project to shared embedding space
        text_features = self.text_projection(text_features)

        # L2 normalize features
        text_features = text_features / ops.norm(text_features, axis=-1, keepdims=True)

        return text_features

    def call(
        self,
        inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of CLIP model."""
        # Parse inputs
        if isinstance(inputs, dict):
            images = inputs.get('image', None)
            texts = inputs.get('text', None)
        else:
            images = inputs[0] if len(inputs) > 0 else None
            texts = inputs[1] if len(inputs) > 1 else None

        results = {}

        # Encode images if provided
        if images is not None:
            image_features = self.encode_image(images, training=training)
            results['image_features'] = image_features

        # Encode texts if provided
        if texts is not None:
            text_features = self.encode_text(texts, training=training)
            results['text_features'] = text_features

        # Compute similarity matrices if both modalities are provided
        if images is not None and texts is not None:
            # Temperature-scaled cosine similarity
            logit_scale = ops.exp(self.logit_scale)
            logits_per_image = logit_scale * ops.matmul(
                image_features, ops.transpose(text_features)
            )
            logits_per_text = ops.transpose(logits_per_image)

            results.update({
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text,
                'logit_scale': logit_scale
            })

        return results

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            'config': {
                'image_size': self.config.image_size,
                'patch_size': self.config.patch_size,
                'vision_layers': self.config.vision_layers,
                'vision_width': self.config.vision_width,
                'vision_heads': self.config.vision_heads,
                'vision_kv_heads': self.config.vision_kv_heads,
                'vocab_size': self.config.vocab_size,
                'context_length': self.config.context_length,
                'text_layers': self.config.text_layers,
                'text_width': self.config.text_width,
                'text_heads': self.config.text_heads,
                'text_kv_heads': self.config.text_kv_heads,
                'embed_dim': self.config.embed_dim,
                'ffn_expansion_factor': self.config.ffn_expansion_factor,
                'ffn_multiple_of': self.config.ffn_multiple_of,
                'dropout_prob': self.config.dropout_prob,
                'attention_dropout': self.config.attention_dropout,
                'stochastic_depth_prob': self.config.stochastic_depth_prob,
                'rms_norm_eps': self.config.rms_norm_eps,
                'rope_percentage': self.config.rope_percentage,
                'logit_scale_init': self.config.logit_scale_init,
            }
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CLIPModel':
        """Create model from configuration."""
        clip_config = CLIPConfig(**config['config'])
        return cls(clip_config)

# ---------------------------------------------------------------------

def create_clip_model(
    image_size: int = 224,
    patch_size: int = 16,
    vision_layers: int = 12,
    vision_width: int = 768,
    vision_heads: int = 12,
    vision_kv_heads: int = 4,
    vocab_size: int = 49408,
    context_length: int = 77,
    text_layers: int = 12,
    text_width: int = 512,
    text_heads: int = 8,
    embed_dim: int = 512,
    **kwargs: Any
) -> CLIPModel:
    """
    Create a CLIP model with specified configuration.

    Args:
        image_size: Size of input images
        patch_size: Size of image patches
        vision_layers: Number of vision transformer layers
        vision_width: Width of vision transformer
        vision_heads: Number of attention heads in vision transformer
        vision_kv_heads: Number of key-value heads in vision transformer (for GQA)
        vocab_size: Vocabulary size for text encoder
        context_length: Maximum text sequence length
        text_layers: Number of text transformer layers
        text_width: Width of text transformer
        text_heads: Number of attention heads in text transformer
        embed_dim: Dimension of shared embedding space
        **kwargs: Additional configuration parameters

    Returns:
        Configured CLIP model
    """
    config = CLIPConfig(
        image_size=image_size,
        patch_size=patch_size,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_heads=vision_heads,
        vision_kv_heads=vision_kv_heads,
        vocab_size=vocab_size,
        context_length=context_length,
        text_layers=text_layers,
        text_width=text_width,
        text_heads=text_heads,
        embed_dim=embed_dim,
        **kwargs
    )

    logger.info(f"Creating CLIP model with configuration")
    return CLIPModel(config)

# ---------------------------------------------------------------------

def create_clip_variant(variant: str = "ViT-B/16") -> CLIPModel:
    """
    Create predefined CLIP model variants with modern improvements.

    Args:
        variant: Model variant string. Options:
            - "ViT-B/32": Base model with 32x32 patches
            - "ViT-B/16": Base model with 16x16 patches
            - "ViT-L/14": Large model with 14x14 patches
            - "ViT-H/14": Huge model with 14x14 patches

    Returns:
        Configured CLIP model with modern architecture improvements
    """
    variant_configs = {
        "ViT-B/32": {
            "image_size": 224,
            "patch_size": 32,
            "vision_layers": 12,
            "vision_width": 768,
            "vision_heads": 12,
            "vision_kv_heads": 4,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "embed_dim": 512,
        },
        "ViT-B/16": {
            "image_size": 224,
            "patch_size": 16,
            "vision_layers": 12,
            "vision_width": 768,
            "vision_heads": 12,
            "vision_kv_heads": 4,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "embed_dim": 512,
        },
        "ViT-L/14": {
            "image_size": 224,
            "patch_size": 14,
            "vision_layers": 24,
            "vision_width": 1024,
            "vision_heads": 16,
            "vision_kv_heads": 4,
            "text_layers": 12,
            "text_width": 768,
            "text_heads": 12,
            "embed_dim": 768,
        },
        "ViT-H/14": {
            "image_size": 224,
            "patch_size": 14,
            "vision_layers": 32,
            "vision_width": 1280,
            "vision_heads": 16,
            "vision_kv_heads": 4,
            "text_layers": 12,
            "text_width": 1024,
            "text_headers": 16,
            "embed_dim": 1024,
        },
    }

    if variant not in variant_configs:
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {list(variant_configs.keys())}"
        )

    config_dict = variant_configs[variant]
    logger.info(f"Creating CLIP {variant} variant with modern improvements")

    return create_clip_model(**config_dict)

# ---------------------------------------------------------------------