"""
Vision Transformer with Hierarchical MLP Stem for dl-techniques Framework
=======================================================================

This module implements a Vision Transformer (ViT) model with Hierarchical MLP (hMLP) stem,
designed for compatibility with the dl-techniques framework. The implementation follows
Keras 3 lifecycle patterns and leverages the framework's highly configurable TransformerLayer
for maximum flexibility and performance.

PAPER REFERENCE:
---------------
"Three things everyone should know about Vision Transformers"
Hugo Touvron, Matthieu Cord, Alaaeldin El-Hassany, Matthijs Douze, Armand Joulin, Hervé Jégou
https://arxiv.org/abs/2203.09795

HIERARCHICAL MLP STEM OVERVIEW:
------------------------------
The hMLP stem is a revolutionary patch preprocessing technique that addresses key limitations
of traditional Vision Transformers while maintaining compatibility with masked self-supervised
learning approaches. Unlike convolutional stems that cause information leakage between patches,
the hMLP stem processes each patch independently through a hierarchical structure.

STEM ARCHITECTURE:
- Progressive patch processing: 2×2 → 4×4 → 8×8 → 16×16 pixels
- Independent patch processing (no cross-patch information leakage)
- Linear projections with normalization and non-linearity at each stage
- Minimal computational overhead (<1% FLOPs increase vs standard ViT)
- Compatible with both BatchNorm (better performance) and LayerNorm (stable for small batches)

KEY ADVANTAGES:
--------------
1. **Masked Self-Supervised Learning Compatibility**:
   - Perfect compatibility with BeiT, MAE, and other mask-based approaches
   - Masking can be applied before or after stem with identical results
   - No information leakage between patches (unlike convolutional stems)

2. **Performance Benefits**:
   - Supervised learning: ~0.3% accuracy improvement over standard ViT
   - BeiT pre-training: +0.4% accuracy improvement over linear projection
   - Matches or exceeds convolutional stem performance for supervised learning

EXPERIMENTAL RESULTS FROM PAPER:
-------------------------------
- Supervised ViT-B with Linear stem: 82.2% top-1 accuracy on ImageNet
- Supervised ViT-B with hMLP stem: 82.5% top-1 accuracy (+0.3%)
- BeiT+FT ViT-B with Linear stem: 83.1% top-1 accuracy
- BeiT+FT ViT-B with hMLP stem: 83.4% top-1 accuracy (+0.3%)

When used with BeiT, existing convolutional stems show no improvement (83.0%)
while hMLP stem provides significant gains, demonstrating its unique effectiveness
for masked self-supervised learning.
"""

import keras
from dataclasses import dataclass
from keras import ops, initializers, layers
from typing import Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.transformer import TransformerLayer
from ..layers.hierarchical_mlp_stem import HierarchicalMLPStem

# ---------------------------------------------------------------------

@dataclass
class ViTConfig:
    """Configuration class for Vision Transformer with hMLP stem."""

    image_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple[int, int] = (16, 16)
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    norm_layer: str = "batch"

    # TransformerLayer specific configurations
    attention_type: str = 'multi_head_attention'
    normalization_type: str = 'layer_norm'
    normalization_position: str = 'pre'  # Pre-norm is more stable for deep networks
    ffn_type: str = 'mlp'
    use_stochastic_depth: bool = False
    stochastic_depth_rate: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'norm_layer': self.norm_layer,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_rate': self.stochastic_depth_rate,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ViTConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ViTHMLP(keras.Model):
    """
    Vision Transformer with Hierarchical MLP Stem using framework TransformerLayer.

    Args:
        config: ViTConfig containing model configuration parameters.
        **kwargs: Additional keyword arguments passed to the Model base class.

    Input shape:
        4D tensor with shape: (batch_size, height, width, channels)

    Output shape:
        2D tensor with shape: (batch_size, num_classes)

    Example:
        ```python
        config = ViTConfig(
            image_size=(224, 224),
            patch_size=(16, 16),
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12
        )
        model = ViTWithHMLPStem(config)

        # Build model with sample input
        model.build((None, 224, 224, 3))

        # Use for prediction
        x = keras.random.normal((2, 224, 224, 3))
        logits = model(x)
        ```
    """

    def __init__(
        self,
        config: ViTConfig,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.config = config

        # Validate configuration
        if config.embed_dim % config.num_heads != 0:
            raise ValueError(
                f"embed_dim ({config.embed_dim}) must be divisible by "
                f"num_heads ({config.num_heads})"
            )

        # Initialize components (not built yet)
        self.stem = None
        self.transformer_blocks = []
        self.pos_drop = None
        self.norm = None
        self.head = None

        # Weights will be created in build()
        self.pos_embed = None
        self.cls_token = None

        # Store build state
        self._build_input_shape = None
        self._num_patches = None

        logger.info(f"Created ViTWithHMLPStem with config: {config}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model components.

        Args:
            input_shape: Shape of input tensor (batch_size, height, width, channels).
        """
        if self.built:
            return

        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input, got {len(input_shape)}D: {input_shape}")

        if input_shape[1] != self.config.image_size[0] or input_shape[2] != self.config.image_size[1]:
            logger.warning(
                f"Input shape {input_shape[1:3]} doesn't match configured image_size "
                f"{self.config.image_size}. This may cause issues."
            )

        # Create the hierarchical MLP stem
        self.stem = HierarchicalMLPStem(
            embed_dim=self.config.embed_dim,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_channels=input_shape[-1],
            norm_layer=self.config.norm_layer,
            name='hierarchical_mlp_stem'
        )

        # Build stem and get number of patches
        self.stem.build(input_shape)
        self._num_patches = self.stem.num_patches

        # Create positional embedding weight
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self._num_patches + 1, self.config.embed_dim),
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        # Create class token weight
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.config.embed_dim),
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        # Create dropout layer
        self.pos_drop = layers.Dropout(
            self.config.drop_rate,
            name='pos_dropout'
        )

        # Create transformer blocks using framework TransformerLayer
        intermediate_size = int(self.config.embed_dim * self.config.mlp_ratio)

        for i in range(self.config.depth):
            # Calculate stochastic depth rate for this layer (linear scaling)
            layer_drop_rate = (
                self.config.stochastic_depth_rate * i / max(1, self.config.depth - 1)
                if self.config.use_stochastic_depth
                else 0.0
            )

            transformer_block = TransformerLayer(
                hidden_size=self.config.embed_dim,
                num_heads=self.config.num_heads,
                intermediate_size=intermediate_size,
                attention_type=self.config.attention_type,
                normalization_type=self.config.normalization_type,
                normalization_position=self.config.normalization_position,
                ffn_type=self.config.ffn_type,
                dropout_rate=self.config.drop_rate,
                attention_dropout_rate=self.config.attn_drop_rate,
                use_stochastic_depth=self.config.use_stochastic_depth,
                stochastic_depth_rate=layer_drop_rate,
                activation='gelu',
                use_bias=self.config.qkv_bias,
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(transformer_block)

        # Create final normalization layer
        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="encoder_norm"
        )

        # Create classification head
        self.head = layers.Dense(
            self.config.num_classes,
            kernel_initializer=initializers.RandomNormal(stddev=0.02),
            name="classification_head"
        )

        # Build all transformer blocks
        # After stem processing: (batch_size, num_patches, embed_dim)
        # After adding cls token: (batch_size, num_patches + 1, embed_dim)
        transformer_input_shape = (None, self._num_patches + 1, self.config.embed_dim)

        for block in self.transformer_blocks:
            block.build(transformer_input_shape)

        # Build normalization and head
        self.norm.build(transformer_input_shape)
        self.head.build((None, self.config.embed_dim))

        # Build pos_drop
        self.pos_drop.build(transformer_input_shape)

        super().build(input_shape)

        logger.info(f"Built ViTWithHMLPStem with {self._num_patches} patches")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether in training mode.

        Returns:
            Classification logits of shape (batch_size, num_classes).
        """
        batch_size = ops.shape(inputs)[0]

        # Process patches with hMLP stem
        x = self.stem(inputs, training=training)

        # Add class token
        cls_tokens = ops.repeat(self.cls_token, batch_size, axis=0)
        x = ops.concatenate([cls_tokens, x], axis=1)

        # Add position embedding and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Apply final normalization
        x = self.norm(x, training=training)

        # Use [CLS] token for classification
        x = x[:, 0]

        # Classification head
        x = self.head(x, training=training)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the model from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ViTHMLP':
        """Create model from configuration."""
        model_config = ViTConfig.from_dict(config.pop('config'))
        return cls(config=model_config, **config)

    @property
    def num_patches(self) -> Optional[int]:
        """Get number of patches."""
        return self._num_patches

# ---------------------------------------------------------------------

def create_vit_model(
    image_size: Tuple[int, int] = (224, 224),
    patch_size: Tuple[int, int] = (16, 16),
    num_classes: int = 1000,
    model_size: str = "base",
    use_stochastic_depth: bool = False,
    stochastic_depth_rate: float = 0.1,
    **kwargs: Any
) -> ViTHMLP:
    """
    Create a Vision Transformer with hierarchical MLP stem.

    Args:
        image_size: Input image dimensions (height, width).
        patch_size: Patch dimensions (height, width).
        num_classes: Number of output classes.
        model_size: Model size ('tiny', 'small', 'base', or 'large').
        use_stochastic_depth: Whether to use stochastic depth regularization.
        stochastic_depth_rate: Maximum drop path rate for stochastic depth.
        **kwargs: Additional configuration parameters.

    Returns:
        ViT model with hMLP stem using framework TransformerLayer.

    Raises:
        ValueError: If model_size is not supported.

    Example:
        ```python
        # Create base model with stochastic depth
        model = create_vit_model(
            image_size=(224, 224),
            patch_size=(16, 16),
            num_classes=1000,
            model_size='base',
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1
        )

        # Build and compile
        model.build((None, 224, 224, 3))
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    # Model configurations
    configs = {
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    }

    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}. Available: {list(configs.keys())}")

    config_dict = configs[model_size]

    # Create configuration with overrides
    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=config_dict["embed_dim"],
        depth=config_dict["depth"],
        num_heads=config_dict["num_heads"],
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer="batch",
        use_stochastic_depth=use_stochastic_depth,
        stochastic_depth_rate=stochastic_depth_rate,
        **kwargs
    )

    logger.info(f"Creating ViT-{model_size} model with hMLP stem")
    return ViTHMLP(config=config)

# ---------------------------------------------------------------------

def create_inputs_with_masking(
    batch_size: int = 8,
    image_size: Tuple[int, int] = (224, 224),
    patch_size: Tuple[int, int] = (16, 16),
    mask_ratio: float = 0.4,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Create masked input images and corresponding mask for self-supervised learning.

    This function demonstrates the key advantage of hMLP stem: it can be used with
    masked self-supervised learning approaches like MAE and BeiT.

    Args:
        batch_size: Batch size for generated data.
        image_size: Image dimensions (height, width).
        patch_size: Patch dimensions (height, width).
        mask_ratio: Ratio of patches to mask (0.0 to 1.0).

    Returns:
        Tuple of (images, mask) where mask is 1 for masked patches, 0 for visible.

    Example:
        ```python
        # Create masked inputs for MAE-style training
        images, mask = create_inputs_with_masking(
            batch_size=32,
            mask_ratio=0.75  # Typical MAE masking ratio
        )

        # Use with model
        model = create_vit_model()
        model.build(images.shape)

        # Apply masking after stem (key advantage of hMLP)
        masked_patches, _ = apply_mask_after_stem(model.stem, images, mask)
        ```
    """
    # Create random images
    images = keras.random.normal([batch_size, image_size[0], image_size[1], 3])

    # Calculate number of patches
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    num_mask = int(mask_ratio * num_patches)

    # Create mask for each sample in batch
    masks = []
    for _ in range(batch_size):
        # Create random mask for this sample
        indices = ops.random.shuffle(ops.arange(num_patches, dtype='int32'))[:num_mask]
        mask_sample = ops.zeros([num_patches], dtype='float32')

        # Set masked positions to 1
        mask_sample = ops.scatter(
            mask_sample,
            ops.expand_dims(indices, 1),
            ops.ones([num_mask], dtype='float32')
        )
        masks.append(mask_sample)

    mask = ops.stack(masks, axis=0)

    return images, mask

# ---------------------------------------------------------------------

def apply_mask_after_stem(
    stem: HierarchicalMLPStem,
    images: keras.KerasTensor,
    mask: keras.KerasTensor
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Process images with hMLP stem then apply masking.

    This function demonstrates the key advantage of hMLP stem: it can be used with
    masked self-supervised learning by applying masking after the stem processing.
    Unlike convolutional stems, hMLP doesn't cause information leakage between patches.

    Args:
        stem: Hierarchical MLP stem instance.
        images: Input images of shape (batch_size, height, width, channels).
        mask: Mask tensor of shape (batch_size, num_patches) where 1 = masked.

    Returns:
        Tuple of (masked_patches, mask) where masked patches are set to zero.

    Example:
        ```python
        # Create model and inputs
        model = create_vit_model()
        model.build((None, 224, 224, 3))

        images, mask = create_inputs_with_masking(batch_size=4, mask_ratio=0.75)

        # Apply masking after stem - key advantage for SSL
        masked_patches, mask = apply_mask_after_stem(model.stem, images, mask)

        # masked_patches can be used for MAE/BeiT training
        # where only visible patches are processed by the encoder
        ```
    """
    # Process images with stem
    patches = stem(images)  # Shape: (batch_size, num_patches, embed_dim)

    # Apply mask (set masked patches to zero)
    mask_expanded = ops.expand_dims(mask, -1)  # Shape: (batch_size, num_patches, 1)

    # Broadcast mask to match patch dimensions
    embed_dim = ops.shape(patches)[-1]
    mask_expanded = ops.repeat(mask_expanded, embed_dim, axis=-1)

    # Apply mask: multiply by (1 - mask) to zero out masked patches
    masked_patches = patches * (1 - mask_expanded)

    return masked_patches, mask

# ---------------------------------------------------------------------