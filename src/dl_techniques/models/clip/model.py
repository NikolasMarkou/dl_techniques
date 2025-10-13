"""
CLIP (Contrastive Language-Image Pre-training) Model Implementation
====================================================================

Based on: "Learning Transferable Visual Representations from Natural Language Supervision" (Radford et al., 2021)
https://arxiv.org/abs/2103.00020

Mathematical Framework:
    1. Image encoder: f_I(image) → R^d (ViT with patches)
    2. Text encoder: f_T(text) → R^d (Transformer with tokens)
    3. Similarity: S = f_I(I) · f_T(T)^T / τ (temperature-scaled cosine similarity)
    4. Contrastive loss: symmetric cross-entropy on similarity matrix

Model Variants:
---------------
- "ViT-B/32": Base model with 32x32 patches
- "ViT-B/16": Base model with 16x16 patches
- "ViT-L/14": Large model with 14x14 patches
- "ViT-H/14": Huge model with 14x14 patches

Usage Examples:
---------------
```python
# Create a model from a predefined variant
model = CLIPModel.from_variant("ViT-B/16")

# Build the model (e.g., before loading weights or calling summary)
model.build({
    'image': (None, 224, 224, 3),
    'text': (None, 77)
})
model.summary()

# Prepare inputs
images = keras.ops.random.normal((32, 224, 224, 3))
text_tokens = keras.ops.random.uniform((32, 77), 0, 49408, dtype='int32')

# Full forward pass for training
outputs = model({'image': images, 'text': text_tokens})

# Get features for a single modality (inference)
image_features = model.encode_image(images)
text_features = model.encode_text(text_tokens)
```
"""

import keras
from keras import layers, ops, initializers
from typing import Optional, Any, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CLIPModel(keras.Model):
    """
    CLIP model with integrated vision and text encoders.

    This model implements the complete CLIP architecture following modern Keras 3
    patterns. It uses a Vision Transformer (ViT) for image encoding and a
    standard Transformer for text encoding, projecting both into a shared
    embedding space for contrastive learning.

    Args:
        # Vision encoder configuration
        image_size: Input image size (height and width).
        patch_size: Size of image patches for vision transformer.
        vision_layers: Number of transformer layers in vision encoder.
        vision_width: Hidden dimension of vision transformer.
        vision_heads: Number of attention heads in vision transformer.
        vision_kv_heads: Number of key-value heads for vision GQA.

        # Text encoder configuration
        vocab_size: Size of text vocabulary.
        context_length: Maximum text sequence length.
        text_layers: Number of transformer layers in text encoder.
        text_width: Hidden dimension of text transformer.
        text_heads: Number of attention heads in text transformer.
        text_kv_heads: Number of key-value heads for text GQA.

        # Shared configuration
        embed_dim: Dimension of shared embedding space.

        # FFN configuration
        ffn_expansion_factor: Expansion factor for feed-forward networks.
        ffn_multiple_of: Round FFN hidden dim to multiple of this value.

        # Regularization
        dropout_rate: General dropout probability.
        attention_dropout_rate: Dropout probability for attention weights.

        # Training specifics
        logit_scale_init: Initial value for learnable temperature parameter.

        **kwargs: Additional arguments for Model base class.

    Input shape:
        A dictionary with keys:
        - 'image': 4D tensor of shape `(batch_size, height, width, channels)`
        - 'text': 2D tensor of shape `(batch_size, sequence_length)`

    Output shape:
        A dictionary with keys:
        - 'image_features': 2D tensor of shape `(batch_size, embed_dim)`
        - 'text_features': 2D tensor of shape `(batch_size, embed_dim)`
        - 'logits_per_image': 2D tensor of shape `(batch_size, batch_size)`
        - 'logits_per_text': 2D tensor of shape `(batch_size, batch_size)`
        - 'logit_scale': Scalar tensor (learnable temperature)
    """

    MODEL_VARIANTS = {
        "ViT-B/32": {
            "patch_size": 32, "vision_layers": 12, "vision_width": 768,
            "vision_heads": 12, "vision_kv_heads": 4, "text_layers": 12,
            "text_width": 512, "text_heads": 8, "text_kv_heads": 8,
            "embed_dim": 512,
        },
        "ViT-B/16": {
            "patch_size": 16, "vision_layers": 12, "vision_width": 768,
            "vision_heads": 12, "vision_kv_heads": 4, "text_layers": 12,
            "text_width": 512, "text_heads": 8, "text_kv_heads": 8,
            "embed_dim": 512,
        },
        "ViT-L/14": {
            "patch_size": 14, "vision_layers": 24, "vision_width": 1024,
            "vision_heads": 16, "vision_kv_heads": 4, "text_layers": 12,
            "text_width": 768, "text_heads": 12, "text_kv_heads": 12,
            "embed_dim": 768,
        },
        "ViT-H/14": {
            "patch_size": 14, "vision_layers": 32, "vision_width": 1280,
            "vision_heads": 16, "vision_kv_heads": 4, "text_layers": 12,
            "text_width": 1024, "text_heads": 16, "text_kv_heads": 16,
            "embed_dim": 1024,
        },
    }

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

        # --- Create sub-layers in __init__ ---
        self._build_vision_encoder()
        self._build_text_encoder()

        # Weight attributes (created in build())
        self.logit_scale = None
        self.class_token = None

        logger.info(f"CLIPModel initialized with embed_dim={self.embed_dim}, "
                    f"vision_layers={self.vision_layers}, "
                    f"text_layers={self.text_layers}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size ({self.image_size}) must be "
                             f"divisible by patch_size ({self.patch_size})")
        if self.vision_width % self.vision_heads != 0:
            raise ValueError(f"vision_width ({self.vision_width}) must be "
                             f"divisible by vision_heads ({self.vision_heads})")
        if self.vision_heads % self.vision_kv_heads != 0:
            raise ValueError(f"vision_heads ({self.vision_heads}) must be "
                             f"divisible by vision_kv_heads ({self.vision_kv_heads})")
        if self.text_width % self.text_heads != 0:
            raise ValueError(f"text_width ({self.text_width}) must be "
                             f"divisible by text_heads ({self.text_heads})")
        if self.text_heads % self.text_kv_heads != 0:
            raise ValueError(f"text_heads ({self.text_heads}) must be "
                             f"divisible by text_kv_heads ({self.text_kv_heads})")

    def _build_vision_encoder(self) -> None:
        """Create all layers for the vision encoder."""
        self.patch_conv = layers.Conv2D(
            filters=self.vision_width,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='patch_conv'
        )
        self.vision_transformer_layers = [
            TransformerLayer(
                hidden_size=self.vision_width,
                num_heads=self.vision_heads,
                intermediate_size=int(self.vision_width * self.ffn_expansion_factor),
                attention_type='group_query',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                n_kv_head=self.vision_kv_heads,
                ffn_args={
                    "ffn_expansion_factor": self.ffn_expansion_factor,
                    "ffn_multiple_of": self.ffn_multiple_of,
                },
                name=f'vision_transformer_{i}'
            ) for i in range(self.vision_layers)
        ]
        self.vision_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='vision_projection'
        )

    def _build_text_encoder(self) -> None:
        """Create all layers for the text encoder."""
        self.token_embedding = layers.Embedding(
            self.vocab_size,
            self.text_width,
            embeddings_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='token_embedding'
        )
        self.text_transformer_layers = [
            TransformerLayer(
                hidden_size=self.text_width,
                num_heads=self.text_heads,
                intermediate_size=int(self.text_width * self.ffn_expansion_factor),
                attention_type='group_query',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                n_kv_head=self.text_kv_heads,
                ffn_args={
                    "ffn_expansion_factor": self.ffn_expansion_factor,
                    "ffn_multiple_of": self.ffn_multiple_of,
                },
                name=f'text_transformer_{i}'
            ) for i in range(self.text_layers)
        ]
        self.text_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='text_projection'
        )

    def build(self, input_shape: Union[Dict[str, Any], Tuple[Any, ...]]) -> None:
        """Create the model's own weights."""
        if self.built:
            return

        # Create weights that do not depend on input shapes
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

        # Keras will automatically build sub-layers on the first call.
        # Calling super().build() here ensures the model is marked as built.
        super().build(input_shape)

    def encode_image(
        self,
        images: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode images to the shared embedding space.

        Args:
            images: Input images tensor with shape (batch_size, height, width, channels).
            training: Boolean, whether the model is in training mode.

        Returns:
            L2-normalized image features with shape (batch_size, embed_dim).
        """
        batch_size = ops.shape(images)[0]

        # Convert to patches
        patches = self.patch_conv(images, training=training)
        patches = ops.reshape(
            patches, (batch_size, self.num_patches, self.vision_width))

        # Add class token
        class_tokens = ops.broadcast_to(
            self.class_token, (batch_size, 1, self.vision_width))
        x = ops.concatenate([class_tokens, patches], axis=1)

        # Apply vision transformer layers
        for transformer_layer in self.vision_transformer_layers:
            x = transformer_layer(x, training=training)

        # Extract class token representation
        class_token_features = x[:, 0]

        # Project to shared embedding space
        image_features = self.vision_projection(
            class_token_features, training=training)

        # L2 normalize features
        image_features = image_features / ops.norm(
            image_features, axis=-1, keepdims=True)
        return image_features

    def encode_text(
        self,
        text_ids: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode text to the shared embedding space.

        Args:
            text_ids: Input text token IDs of shape (batch_size, sequence_length).
            training: Boolean, whether the model is in training mode.

        Returns:
            L2-normalized text features with shape (batch_size, embed_dim).
        """
        # Token embeddings
        x = self.token_embedding(text_ids, training=training)

        # Apply text transformer layers
        for transformer_layer in self.text_transformer_layers:
            x = transformer_layer(x, training=training)

        # Extract features from the last non-padding token
        sequence_lengths = ops.sum(ops.cast(text_ids != 0, 'int32'), axis=1)
        current_seq_len = ops.shape(x)[1]
        last_token_indices = ops.clip(sequence_lengths - 1, 0, current_seq_len - 1)

        # Gather last token features
        one_hot_indices = ops.one_hot(
            last_token_indices, num_classes=current_seq_len, dtype=x.dtype)
        reshaped_indices = ops.expand_dims(one_hot_indices, axis=1)
        text_features_raw = ops.squeeze(ops.matmul(reshaped_indices, x), axis=1)

        # Project to shared embedding space
        text_features = self.text_projection(text_features_raw, training=training)

        # L2 normalize features
        text_features = text_features / ops.norm(
            text_features, axis=-1, keepdims=True)
        return text_features

    def call(
        self,
        inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass of the CLIP model.

        This method handles both single-modality and multi-modality inputs. For
        training, both 'image' and 'text' should be provided. For inference,
        either one or both can be provided.
        """
        if isinstance(inputs, dict):
            images = inputs.get('image')
            texts = inputs.get('text')
        else:
            images = inputs[0] if len(inputs) > 0 else None
            texts = inputs[1] if len(inputs) > 1 else None

        results = {}
        image_features, text_features = None, None

        if images is not None:
            image_features = self.encode_image(images, training=training)
            results['image_features'] = image_features
        if texts is not None:
            text_features = self.encode_text(texts, training=training)
            results['text_features'] = text_features

        if image_features is not None and text_features is not None:
            logit_scale = ops.exp(self.logit_scale)
            logits_per_image = logit_scale * ops.matmul(
                image_features, ops.transpose(text_features))
            logits_per_text = ops.transpose(logits_per_image)
            results.update({
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text,
                'logit_scale': logit_scale
            })
        return results

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
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
            'logit_scale_init': self.logit_scale_init,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CLIPModel":
        """Create model from configuration."""
        return cls(**config)

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "CLIPModel":
        """
        Create a CLIP model from a predefined variant.

        Args:
            variant: String, one of "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-H/14".
            **kwargs: Additional arguments to override the variant's default config.

        Returns:
            A CLIPModel instance.

        Raises:
            ValueError: If the variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)  # Allow overriding defaults

        logger.info(f"Creating CLIP {variant} variant with modern improvements")
        return cls(**config)

# ---------------------------------------------------------------------

def create_clip_model(**kwargs: Any) -> CLIPModel:
    """
    Convenience function to create a CLIP model.

    This function is a simple wrapper around the CLIPModel constructor,
    allowing for easy creation with custom configurations.

    Args:
        **kwargs: Keyword arguments for the CLIPModel constructor.

    Returns:
        A configured CLIPModel instance.
    """
    logger.info("Creating custom CLIP model")
    return CLIPModel(**kwargs)

def create_clip_variant(variant: str, **kwargs: Any) -> CLIPModel:
    """
    Convenience function to create a CLIP model from a predefined variant.

    Args:
        variant: The model variant string (e.g., "ViT-B/16").
        **kwargs: Additional arguments to override the variant's default config.

    Returns:
        A configured CLIPModel instance.
    """
    return CLIPModel.from_variant(variant, **kwargs)

# ---------------------------------------------------------------------
