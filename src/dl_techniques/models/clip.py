"""
CLIP (Contrastive Language-Image Pre-training) Model Implementation

This implementation follows modern transformer architecture best practices with:
- RMSNorm instead of LayerNorm for better stability
- Grouped Query Attention (GQA) for memory efficiency
- RoPE positional embeddings for better length extrapolation
- SwiGLU FFN for improved performance
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
"""

import keras
from keras import layers, ops, initializers
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer

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
            dropout_rate: General dropout probability
            attention_dropout_rate: Dropout probability for attention weights

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
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0

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
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")
        if not 0.0 <= self.attention_dropout_rate <= 1.0:
            raise ValueError(f"attention_dropout_rate must be in [0, 1], got {self.attention_dropout_rate}")

        # Other validation
        if self.rms_norm_eps <= 0.0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if not 0.0 <= self.rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {self.rope_percentage}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'vision_layers': self.vision_layers,
            'vision_width': self.vision_width,
            'vision_heads': self.vision_heads,
            'vision_kv_heads': self.vision_kv_heads,
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'text_layers': self.text_layers,
            'text_width': self.text_width,
            'text_heads': self.text_heads,
            'text_kv_heads': self.text_kv_heads,
            'embed_dim': self.embed_dim,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'ffn_multiple_of': self.ffn_multiple_of,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'rms_norm_eps': self.rms_norm_eps,
            'rope_percentage': self.rope_percentage,
            'logit_scale_init': self.logit_scale_init,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CLIPConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisionTransformer(layers.Layer):
    """
    Vision Transformer encoder with patch embeddings.

    Converts input images to patch embeddings and processes them through
    a stack of transformer blocks with a learnable class token.

    Args:
        config: CLIP configuration containing vision transformer parameters
        **kwargs: Additional keyword arguments for the Layer base class

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        2D tensor with shape: `(batch_size, embed_dim)`

    Example:
        ```python
        config = CLIPConfig(vision_width=768, vision_layers=12)
        vision_encoder = VisionTransformer(config)

        # Process batch of images
        images = keras.ops.random.normal((32, 224, 224, 3))
        features = vision_encoder(images)  # Shape: (32, 768)
        ```
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

        # Layers will be created in build()
        self.patch_conv = None
        self.class_token = None
        self.transformer_layers = []

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the vision transformer components."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Patch embedding using Conv2D - create in build(), not call()
        self.patch_conv = layers.Conv2D(
            filters=self.vision_width,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='patch_conv'
        )

        # Class token - create as managed weight
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.vision_width),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Transformer layers using dl-techniques TransformerLayer
        self.transformer_layers = []

        for i in range(self.vision_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.vision_width,
                num_heads=self.vision_heads,
                intermediate_size=int(self.vision_width * self.config.ffn_expansion_factor),
                attention_type='group_query_attention',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.config.dropout_rate,
                attention_dropout_rate=self.config.attention_dropout_rate,
                n_kv_head=self.vision_kv_heads,
                ffn_expansion_factor=self.config.ffn_expansion_factor,
                ffn_multiple_of=self.config.ffn_multiple_of,
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(transformer_layer)

        # Build patch convolution
        self.patch_conv.build(input_shape)

        # Build transformer layers
        transformer_input_shape = (input_shape[0], self.seq_len, self.vision_width)
        for transformer_layer in self.transformer_layers:
            transformer_layer.build(transformer_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Process images through vision transformer."""
        batch_size = ops.shape(inputs)[0]

        # Convert to patches using pre-built conv layer
        patches = self.patch_conv(inputs)

        # Reshape to sequence format
        patches = ops.reshape(patches, (batch_size, self.num_patches, self.vision_width))

        # Add class token
        class_tokens = ops.broadcast_to(self.class_token, (batch_size, 1, self.vision_width))
        x = ops.concatenate([class_tokens, patches], axis=1)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)

        # Return class token embedding (already normalized by final layer)
        return x[:, 0]  # Shape: (batch_size, vision_width)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return (input_shape[0], self.vision_width)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "config": self.config.to_dict(),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VisionTransformer':
        """Create layer from configuration."""
        clip_config = CLIPConfig.from_dict(config['config'])
        return cls(clip_config)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TextTransformer(layers.Layer):
    """
    Text Transformer encoder with token embeddings.

    Processes tokenized text through embedding layer and transformer blocks,
    extracting the final token representation for each sequence.

    Args:
        config: CLIP configuration containing text transformer parameters
        **kwargs: Additional keyword arguments for the Layer base class

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)`

    Output shape:
        2D tensor with shape: `(batch_size, text_width)`

    Example:
        ```python
        config = CLIPConfig(text_width=512, text_layers=12)
        text_encoder = TextTransformer(config)

        # Process batch of tokenized text
        text_tokens = keras.ops.random.uniform((32, 77), 0, 49408, dtype='int32')
        features = text_encoder(text_tokens)  # Shape: (32, 512)
        ```
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

        # Layers will be created in build()
        self.token_embedding = None
        self.transformer_layers = []

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the text transformer components."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Token embeddings
        self.token_embedding = layers.Embedding(
            self.vocab_size,
            self.text_width,
            embeddings_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='token_embedding'
        )

        # Transformer layers using dl-techniques TransformerLayer
        self.transformer_layers = []

        for i in range(self.text_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.text_width,
                num_heads=self.text_heads,
                intermediate_size=int(self.text_width * self.config.ffn_expansion_factor),
                attention_type='group_query_attention',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.config.dropout_rate,
                attention_dropout_rate=self.config.attention_dropout_rate,
                n_kv_head=self.text_kv_heads,
                ffn_expansion_factor=self.config.ffn_expansion_factor,
                ffn_multiple_of=self.config.ffn_multiple_of,
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(transformer_layer)

        # Build sublayers
        self.token_embedding.build(input_shape)

        # Transformer input shape after embedding
        transformer_input_shape = (input_shape[0], input_shape[1], self.text_width)
        for transformer_layer in self.transformer_layers:
            transformer_layer.build(transformer_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Process text tokens through text transformer."""
        # Token embeddings
        x = self.token_embedding(inputs)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)

        # Extract text features from the last non-padding token
        # Assumes padding token ID is 0
        sequence_lengths = ops.sum(ops.cast(inputs != 0, 'int32'), axis=1)

        # Get the actual sequence length from the input tensor's shape
        current_seq_len = ops.shape(x)[1]
        last_token_indices = ops.clip(sequence_lengths - 1, 0, current_seq_len - 1)

        # Gather embeddings at sequence end positions. The standard advanced
        # indexing `x[ops.arange(batch_size), last_token_indices]` is not
        # supported by all Keras backends (e.g., TensorFlow).
        # A backend-agnostic way is to use `one_hot` and `matmul`.
        one_hot_indices = ops.one_hot(
            last_token_indices, num_classes=current_seq_len, dtype=x.dtype
        )
        # Reshape for matmul: (batch_size, 1, seq_len) @ (batch_size, seq_len, width)
        reshaped_indices = ops.expand_dims(one_hot_indices, axis=1)
        # Squeeze to get (batch_size, width)
        text_features = ops.squeeze(ops.matmul(reshaped_indices, x), axis=1)

        return text_features

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return (input_shape[0], self.text_width)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "config": self.config.to_dict(),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TextTransformer':
        """Create layer from configuration."""
        clip_config = CLIPConfig.from_dict(config['config'])
        return cls(clip_config)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CLIPModel(keras.Model):
    """
    CLIP model with vision and text encoders.

    Implements the complete CLIP architecture with separate vision and text encoders
    that project to a shared embedding space for contrastive learning.

    Args:
        config: CLIP configuration containing model parameters
        **kwargs: Additional keyword arguments for the Model base class

    Input shape:
        Dictionary with keys:
        - 'image': 4D tensor with shape `(batch_size, height, width, channels)`
        - 'text': 2D tensor with shape `(batch_size, sequence_length)`

    Output shape:
        Dictionary with keys:
        - 'image_features': 2D tensor with shape `(batch_size, embed_dim)`
        - 'text_features': 2D tensor with shape `(batch_size, embed_dim)`
        - 'logits_per_image': 2D tensor with shape `(batch_size, batch_size)`
        - 'logits_per_text': 2D tensor with shape `(batch_size, batch_size)`
        - 'logit_scale': Scalar tensor

    Example:
        ```python
        config = CLIPConfig()
        model = CLIPModel(config)

        # Prepare inputs
        images = keras.ops.random.normal((32, 224, 224, 3))
        text_tokens = keras.ops.random.uniform((32, 77), 0, 49408, dtype='int32')

        # Forward pass
        outputs = model({'image': images, 'text': text_tokens})

        # Extract features
        image_features = outputs['image_features']  # Shape: (32, 512)
        text_features = outputs['text_features']    # Shape: (32, 512)
        ```
    """

    def __init__(
        self,
        config: CLIPConfig,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.embed_dim = config.embed_dim

        # Components will be created in build()
        self.vision_encoder = None
        self.text_encoder = None
        self.vision_projection = None
        self.text_projection = None
        self.logit_scale = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(f"CLIPModel initialized with embed_dim={self.embed_dim}")

    def build(self, input_shape: Union[Dict[str, Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...], Tuple[Optional[int], ...]]) -> None:
        """Build the CLIP model components."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Create encoders
        self.vision_encoder = VisionTransformer(self.config, name='vision_encoder')
        self.text_encoder = TextTransformer(self.config, name='text_encoder')

        # Projection layers to shared embedding space
        self.vision_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='vision_projection'
        )

        self.text_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='text_projection'
        )

        # Learnable temperature parameter for contrastive loss
        self.logit_scale = self.add_weight(
            name='logit_scale',
            shape=(),
            initializer=initializers.Constant(self.config.logit_scale_init),
            trainable=True
        )

        # Build sub-components with appropriate input shapes
        if isinstance(input_shape, dict):
            if 'image' in input_shape:
                image_shape = input_shape['image']
                self.vision_encoder.build(image_shape)
                self.vision_projection.build((image_shape[0], self.config.vision_width))
            if 'text' in input_shape:
                text_shape = input_shape['text']
                self.text_encoder.build(text_shape)
                self.text_projection.build((text_shape[0], self.config.text_width))
        else:
            # Handle tuple format (image_shape, text_shape) for backwards compatibility
            if len(input_shape) >= 2:
                image_shape, text_shape = input_shape[0], input_shape[1]
                self.vision_encoder.build(image_shape)
                self.text_encoder.build(text_shape)
                self.vision_projection.build((image_shape[0], self.config.vision_width))
                self.text_projection.build((text_shape[0], self.config.text_width))

        super().build(input_shape)

    def encode_image(
        self,
        images: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode images to shared embedding space.

        Args:
            images: Input images tensor with shape (batch_size, height, width, channels)
            training: Whether in training mode

        Returns:
            L2-normalized image features with shape (batch_size, embed_dim)
        """
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
        """
        Encode text to shared embedding space.

        Args:
            text_ids: Input text token IDs with shape (batch_size, sequence_length)
            training: Whether in training mode

        Returns:
            L2-normalized text features with shape (batch_size, embed_dim)
        """
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
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build model from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CLIPModel':
        """Create model from configuration."""
        clip_config = CLIPConfig.from_dict(config['config'])
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

    Example:
        ```python
        # Create a standard CLIP model
        model = create_clip_model(
            image_size=224,
            vision_layers=12,
            vision_width=768,
            text_layers=12,
            text_width=512,
            embed_dim=512
        )

        # Build model with input shapes
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 77)
        })

        # Compile for training
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy'
        )
        ```
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

    Example:
        ```python
        # Create a ViT-B/16 variant
        model = create_clip_variant("ViT-B/16")

        # Build and compile
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 77)
        })
        ```
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
            "text_kv_heads": 8,
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
            "text_kv_heads": 8,
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
            "text_kv_heads": 12,
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
            "text_heads": 16,
            "text_kv_heads": 16,
            "embed_dim": 1024,
        },
    }

    if variant not in variant_configs:
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {list(variant_configs.keys())}"
        )

    config_dict = variant_configs[variant]
    # Ensure text_kv_heads is set for standard MHA by default
    config_dict.setdefault("text_kv_heads", config_dict["text_heads"])

    logger.info(f"Creating CLIP {variant} variant with modern improvements")

    return create_clip_model(**config_dict)

# ---------------------------------------------------------------------