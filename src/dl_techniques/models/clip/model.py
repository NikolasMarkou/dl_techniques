"""
CLIP (Contrastive Language-Image Pre-training) Model Implementation

The model implements the contrastive learning framework that learns joint representations
of images and text by maximizing similarity between matching pairs while minimizing
similarity between non-matching pairs in a shared embedding space.

Mathematical Framework:
    1. Image encoder: f_I(image) → R^d (ViT with patches)
    2. Text encoder: f_T(text) → R^d (Transformer with tokens)
    3. Similarity: S = f_I(I) · f_T(T)^T / τ (temperature-scaled cosine similarity)
    4. Contrastive loss: symmetric cross-entropy on similarity matrix

References:
    - Radford, A., et al. (2021). "Learning Transferable Visual Representations
      from Natural Language Supervision." https://arxiv.org/abs/2103.00020
"""

import keras
from keras import layers, ops, initializers
from typing import Optional, Any, Dict, Union, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CLIPModel(keras.Model):
    """
    CLIP model with integrated vision_heads and text encoders.

    This model implements the complete CLIP architecture with vision_heads and text processing
    integrated within a single model class. It uses modern transformer components from
    the dl-techniques framework including GroupedQueryAttention, RMSNorm, and SwiGLU.

    **Architecture**:
    ```
    Vision Path:
    Image → Patches → Vision Transformers → Vision Projection → L2 Norm

    Text Path:
    Tokens → Text Embedding → Text Transformers → Text Projection → L2 Norm

    Similarity:
    Vision Features × Text Features^T × Temperature → Logits
    ```

    Args:
        # Vision encoder configuration
        image_size: Input image size (height and width). Default 224.
        patch_size: Size of image patches for vision_heads transformer. Default 16.
        vision_layers: Number of transformer layers in vision_heads encoder. Default 12.
        vision_width: Hidden dimension of vision_heads transformer. Default 768.
        vision_heads: Number of attention heads in vision_heads transformer. Default 12.
        vision_kv_heads: Number of key-value heads for vision_heads GQA. Default 4.

        # Text encoder configuration
        vocab_size: Size of text vocabulary. Default 49408.
        context_length: Maximum text sequence length. Default 77.
        text_layers: Number of transformer layers in text encoder. Default 12.
        text_width: Hidden dimension of text transformer. Default 512.
        text_heads: Number of attention heads in text transformer. Default 8.
        text_kv_heads: Number of key-value heads for text GQA. Default 8.

        # Shared configuration
        embed_dim: Dimension of shared embedding space. Default 512.

        # FFN configuration
        ffn_expansion_factor: Expansion factor for feed-forward networks. Default 4.
        ffn_multiple_of: Round FFN hidden dim to multiple of this value. Default 256.

        # Regularization
        dropout_rate: General dropout probability. Default 0.0.
        attention_dropout_rate: Dropout probability for attention weights. Default 0.0.

        # Training specifics
        logit_scale_init: Initial value for learnable temperature parameter. Default 2.6592.

        **kwargs: Additional arguments for Model base class.

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
        # Create model
        model = CLIPModel(
            vision_layers=12,
            vision_width=768,
            text_layers=12,
            text_width=512,
            embed_dim=512
        )

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
        # Vision encoder configuration
        image_size: int = 224,
        patch_size: int = 16,
        vision_layers: int = 12,
        vision_width: int = 768,
        vision_heads: int = 12,
        vision_kv_heads: int = 4,
        # Text encoder configuration
        vocab_size: int = 49408,
        context_length: int = 77,
        text_layers: int = 12,
        text_width: int = 512,
        text_heads: int = 8,
        text_kv_heads: int = 8,
        # Shared configuration
        embed_dim: int = 512,
        # FFN configuration
        ffn_expansion_factor: int = 4,
        ffn_multiple_of: int = 256,
        # Regularization
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        # Training specifics
        logit_scale_init: float = 2.6592,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store all configuration parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_heads = vision_heads
        self.vision_kv_heads = vision_kv_heads

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.text_layers = text_layers
        self.text_width = text_width
        self.text_heads = text_heads
        self.text_kv_heads = text_kv_heads

        self.embed_dim = embed_dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.logit_scale_init = logit_scale_init

        # Validate configuration
        self._validate_config()

        # Derived properties
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.vision_seq_len = self.num_patches + 1  # +1 for CLS token

        # Vision components
        self.patch_conv = layers.Conv2D(
            filters=self.vision_width,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='patch_conv'
        )

        # Vision transformer layers
        self.vision_transformer_layers: List[TransformerLayer] = []
        for i in range(self.vision_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.vision_width,
                num_heads=self.vision_heads,
                intermediate_size=int(self.vision_width * self.ffn_expansion_factor),
                attention_type='group_query_attention',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                n_kv_head=self.vision_kv_heads,
                ffn_args={
                    "ffn_expansion_factor" :self.ffn_expansion_factor,
                    "ffn_multiple_of": self.ffn_multiple_of,
                },
                name=f'vision_transformer_{i}'
            )
            self.vision_transformer_layers.append(transformer_layer)

        # Vision projection
        self.vision_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='vision_projection'
        )

        # Text components
        self.token_embedding = layers.Embedding(
            self.vocab_size,
            self.text_width,
            embeddings_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='token_embedding'
        )

        # Text transformer layers
        self.text_transformer_layers: List[TransformerLayer] = []
        for i in range(self.text_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.text_width,
                num_heads=self.text_heads,
                intermediate_size=int(self.text_width * self.ffn_expansion_factor),
                attention_type='group_query_attention',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                n_kv_head=self.text_kv_heads,
                ffn_args={
                    "ffn_expansion_factor" :self.ffn_expansion_factor,
                    "ffn_multiple_of": self.ffn_multiple_of,
                },
                name=f'text_transformer_{i}'
            )
            self.text_transformer_layers.append(transformer_layer)

        # Text projection
        self.text_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='text_projection'
        )

        # Learnable temperature parameter and class token will be created in build()
        self.logit_scale = None
        self.class_token = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(f"CLIPModel initialized with embed_dim={self.embed_dim}, "
                   f"vision_layers={self.vision_layers}, text_layers={self.text_layers}")

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

    def build(self, input_shape: Union[Dict[str, Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]) -> None:
        """Build the CLIP model components."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Create learnable parameters that need specific shapes
        self.logit_scale = self.add_weight(
            name='logit_scale',
            shape=(),
            initializer=initializers.Constant(self.logit_scale_init),
            trainable=True
        )

        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.vision_width),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Build sub-components with appropriate input shapes
        if isinstance(input_shape, dict):
            if 'image' in input_shape:
                image_shape = input_shape['image']
                # Build vision_heads components
                self.patch_conv.build(image_shape)

                # Build vision_heads transformer layers
                vision_transformer_input_shape = (image_shape[0], self.vision_seq_len, self.vision_width)
                for transformer_layer in self.vision_transformer_layers:
                    transformer_layer.build(vision_transformer_input_shape)

                self.vision_projection.build((image_shape[0], self.vision_width))

            if 'text' in input_shape:
                text_shape = input_shape['text']
                # Build text components
                self.token_embedding.build(text_shape)

                # Build text transformer layers
                text_transformer_input_shape = (text_shape[0], text_shape[1], self.text_width)
                for transformer_layer in self.text_transformer_layers:
                    transformer_layer.build(text_transformer_input_shape)

                self.text_projection.build((text_shape[0], self.text_width))

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
        batch_size = ops.shape(images)[0]

        # Convert to patches
        patches = self.patch_conv(images)  # Shape: (batch, h_patches, w_patches, vision_width)
        patches = ops.reshape(patches, (batch_size, self.num_patches, self.vision_width))

        # Add class token
        class_tokens = ops.broadcast_to(self.class_token, (batch_size, 1, self.vision_width))
        x = ops.concatenate([class_tokens, patches], axis=1)

        # Apply vision_heads transformer layers
        for transformer_layer in self.vision_transformer_layers:
            x = transformer_layer(x, training=training)

        # Extract class token representation
        class_token_features = x[:, 0]  # Shape: (batch_size, vision_width)

        # Project to shared embedding space
        image_features = self.vision_projection(class_token_features)

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
        # Token embeddings
        x = self.token_embedding(text_ids)

        # Apply text transformer layers
        for transformer_layer in self.text_transformer_layers:
            x = transformer_layer(x, training=training)

        # Extract text features from the last non-padding token
        # Assumes padding token ID is 0
        sequence_lengths = ops.sum(ops.cast(text_ids != 0, 'int32'), axis=1)

        # Get the actual sequence length from the input tensor's shape
        current_seq_len = ops.shape(x)[1]
        last_token_indices = ops.clip(sequence_lengths - 1, 0, current_seq_len - 1)

        # Use one_hot + matmul for backend-agnostic advanced indexing
        one_hot_indices = ops.one_hot(
            last_token_indices, num_classes=current_seq_len, dtype=x.dtype
        )
        reshaped_indices = ops.expand_dims(one_hot_indices, axis=1)
        text_features = ops.squeeze(ops.matmul(reshaped_indices, x), axis=1)

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
            # Vision encoder configuration
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'vision_layers': self.vision_layers,
            'vision_width': self.vision_width,
            'vision_heads': self.vision_heads,
            'vision_kv_heads': self.vision_kv_heads,
            # Text encoder configuration
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'text_layers': self.text_layers,
            'text_width': self.text_width,
            'text_heads': self.text_heads,
            'text_kv_heads': self.text_kv_heads,
            # Shared configuration
            'embed_dim': self.embed_dim,
            # FFN configuration
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'ffn_multiple_of': self.ffn_multiple_of,
            # Regularization
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            # Training specifics
            'logit_scale_init': self.logit_scale_init,
        })
        return config


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
    text_kv_heads: int = 8,
    embed_dim: int = 512,
    **kwargs: Any
) -> CLIPModel:
    """
    Create a CLIP model with specified configuration.

    Args:
        image_size: Size of input images. Default 224.
        patch_size: Size of image patches. Default 16.
        vision_layers: Number of vision_heads transformer layers. Default 12.
        vision_width: Width of vision_heads transformer. Default 768.
        vision_heads: Number of attention heads in vision_heads transformer. Default 12.
        vision_kv_heads: Number of key-value heads in vision_heads transformer for GQA. Default 4.
        vocab_size: Vocabulary size for text encoder. Default 49408.
        context_length: Maximum text sequence length. Default 77.
        text_layers: Number of text transformer layers. Default 12.
        text_width: Width of text transformer. Default 512.
        text_heads: Number of attention heads in text transformer. Default 8.
        text_kv_heads: Number of key-value heads in text transformer for GQA. Default 8.
        embed_dim: Dimension of shared embedding space. Default 512.
        **kwargs: Additional configuration parameters.

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
    logger.info("Creating CLIP model with modern architecture")
    return CLIPModel(
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
        text_kv_heads=text_kv_heads,
        embed_dim=embed_dim,
        **kwargs
    )


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
    logger.info(f"Creating CLIP {variant} variant with modern improvements")

    return create_clip_model(**config_dict)


# ---------------------------------------------------------------------