"""
DinoV3 Model Implementation following Modern Keras 3 Patterns.

This module implements the DinoV3 (Data-efficient Image Transformers) architecture,
a self-supervised Vision Transformer model that learns visual representations without
labels through a distillation-based approach.

The implementation follows the modern Keras 3 patterns:
- All sub-layers are created in __init__()
- Explicit building of sub-layers in build() when needed
- Proper serialization support with get_config()

Based on: "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)

Key Features:
------------
- Modular design using TransformerLayer as building blocks
- Support for all standard DinoV3 variants (Tiny, Small, Base, Large, Giant)
- Smart patch embedding and positional encoding strategies
- Configurable stochastic depth (drop path) with linear scaling
- Flexible attention mechanisms and normalization strategies
- Complete serialization support
- Production-ready implementation

Model Variants:
--------------
- DinoV3-Tiny: 192 dim, 12 layers, 3 heads
- DinoV3-Small: 384 dim, 12 layers, 6 heads
- DinoV3-Base: 768 dim, 12 layers, 12 heads
- DinoV3-Large: 1024 dim, 24 layers, 16 heads
- DinoV3-Giant: 1536 dim, 40 layers, 24 heads

Usage Examples:
-------------
```python
# ImageNet model (224x224 input)
model = DinoV3Model.from_variant("base", num_classes=1000)

# CIFAR-10 model (32x32 input)
model = DinoV3Model.from_variant("small", image_size=(32, 32), num_classes=10)

# Feature extraction model
model = create_dinov3("large", include_top=False)

# Custom configuration
model = create_dinov3("tiny", stochastic_depth_rate=0.2)
```
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.transformer import TransformerLayer
from ..layers.patch_embedding import PatchEmbedding2D

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DinoV3Model(keras.Model):
    """
    DinoV3 (Data-efficient Image Transformers) Model Implementation.

    DinoV3 is a self-supervised Vision Transformer that learns visual representations
    through a teacher-student distillation framework. This implementation provides
    the core architecture that can be used for both teacher and student networks.

    The model consists of:
    - Patch embedding layer to tokenize input images
    - Learnable class token and positional embeddings
    - Stack of transformer encoder layers with configurable attention
    - Optional layer normalization
    - Flexible output heads for different tasks

    Key architectural features:
    - Pre-normalization transformer blocks for training stability
    - Stochastic depth with linear scaling across layers
    - Configurable attention mechanisms (standard, window, grouped-query)
    - Support for both classification and feature extraction modes
    - Adaptive to different input resolutions

    Args:
        image_size: Tuple of integers, input image size (height, width).
            Must be divisible by patch_size. Defaults to (224, 224).
        patch_size: Tuple of integers, patch size for tokenization.
            Defaults to (16, 16).
        num_classes: Integer, number of output classes. If 0, no classification
            head is added. Defaults to 1000.
        embed_dim: Integer, embedding dimension. Must be positive and typically
            divisible by num_heads. Defaults to 768.
        num_layers: Integer, number of transformer layers. Must be positive.
            Defaults to 12.
        num_heads: Integer, number of attention heads per layer. Must be positive
            and divide embed_dim evenly. Defaults to 12.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Defaults to 4.0.
        dropout_rate: Float, dropout rate for regularization. Must be between
            0 and 1. Defaults to 0.1.
        attention_dropout_rate: Float, dropout rate for attention layers.
            Must be between 0 and 1. Defaults to 0.1.
        stochastic_depth_rate: Float, maximum drop path rate for stochastic depth.
            Applied linearly across layers. Must be between 0 and 1. Defaults to 0.1.
        use_class_token: Boolean, whether to use a learnable class token.
            Defaults to True.
        use_mean_pooling: Boolean, whether to use mean pooling instead of class token
            for final representation. Only used if use_class_token=False. Defaults to False.
        normalization_type: String, type of normalization to use in transformer layers.
            Options: 'layer_norm', 'rms_norm'. Defaults to 'layer_norm'.
        attention_type: String, type of attention mechanism to use.
            Options: 'multi_head_attention', 'window_attention', etc. Defaults to 'multi_head_attention'.
        activation: String or callable, activation function for MLP layers.
            Defaults to 'gelu'.
        kernel_initializer: String or initializer, weight initialization method.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, bias initialization method.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        include_top: Boolean, whether to include the classification head.
            If False, returns features before classification. Defaults to True.
        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, channels).
        Height and width must match image_size.

    Output shape:
        If include_top=True: 2D tensor with shape (batch_size, num_classes).
        If include_top=False: 3D tensor with shape (batch_size, num_patches + 1, embed_dim)
            if use_class_token=True, or (batch_size, num_patches, embed_dim) otherwise.

    Raises:
        ValueError: If any parameter is invalid or incompatible.

    Example:
        >>> # Create DinoV3-Base for ImageNet
        >>> model = DinoV3Model.from_variant("base", num_classes=1000)
        >>>
        >>> # Create DinoV3-Small for CIFAR-10
        >>> model = DinoV3Model.from_variant("small", image_size=(32, 32), num_classes=10)
        >>>
        >>> # Create feature extraction model
        >>> model = DinoV3Model.from_variant("large", include_top=False)

    References:
        - Oquab, Maxime, et al. "DINOv2: Learning Robust Visual Features without Supervision"
        - Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale"
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 192,
            "num_layers": 12,
            "num_heads": 3,
            "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "small": {
            "embed_dim": 384,
            "num_layers": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "base": {
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "large": {
            "embed_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "giant": {
            "embed_dim": 1536,
            "num_layers": 40,
            "num_heads": 24,
            "mlp_ratio": 4.0,
            "patch_size": (14, 14),
            "stochastic_depth_rate": 0.4
        }
    }

    # Architecture constants
    LAYERNORM_EPSILON = 1e-6
    POSITIONAL_INITIALIZER = "truncated_normal"
    HEAD_INITIALIZER = "truncated_normal"

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        stochastic_depth_rate: float = 0.1,
        use_class_token: bool = True,
        use_mean_pooling: bool = False,
        normalization_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        attention_type: Literal['multi_head_attention', 'window_attention', 'group_query_attention'] = 'multi_head_attention',
        activation: Union[str, Callable] = 'gelu',
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        include_top: bool = True,
        **kwargs: Any
    ) -> None:
        # Validate inputs
        if len(image_size) != 2 or any(s <= 0 for s in image_size):
            raise ValueError(f"image_size must be a tuple of 2 positive integers, got {image_size}")
        if len(patch_size) != 2 or any(s <= 0 for s in patch_size):
            raise ValueError(f"patch_size must be a tuple of 2 positive integers, got {patch_size}")
        if any(img % patch != 0 for img, patch in zip(image_size, patch_size)):
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
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
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")
        if not (0.0 <= stochastic_depth_rate <= 1.0):
            raise ValueError(f"stochastic_depth_rate must be between 0 and 1, got {stochastic_depth_rate}")
        if num_classes < 0:
            raise ValueError(f"num_classes must be non-negative, got {num_classes}")

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_class_token = use_class_token
        self.use_mean_pooling = use_mean_pooling
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.include_top = include_top

        # Compute derived values
        self.num_patches_h = image_size[0] // patch_size[0]
        self.num_patches_w = image_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.sequence_length = self.num_patches + (1 if use_class_token else 0)

        # Initialize layer lists
        self.encoder_layers = []
        self.head_layers = []

        # Set input shape for the model
        inputs = keras.Input(shape=image_size + (3,))

        # Build the model
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created DinoV3 model for input {image_size + (3,)} "
            f"with {num_layers} layers, {embed_dim} embedding dim"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete DinoV3 model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        x = inputs

        # Build patch embedding
        x = self._build_patch_embedding(x)

        # Add positional embeddings and class token
        x = self._build_positional_encoding(x)

        # Build transformer encoder layers
        x = self._build_encoder_layers(x)

        # Build final normalization
        x = self._build_final_norm(x)

        # Build classification head if requested
        if self.include_top:
            x = self._build_head(x)

        return x

    def _build_patch_embedding(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build patch embedding layer.

        Args:
            x: Input tensor

        Returns:
            Patch embeddings tensor
        """
        self.patch_embedding = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='patch_embedding'
        )

        x = self.patch_embedding(x)
        return x

    def _build_positional_encoding(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build positional encoding with optional class token.

        Args:
            x: Patch embeddings tensor

        Returns:
            Tensor with positional encodings and class token
        """
        batch_size = ops.shape(x)[0]

        # Add class token if used
        if self.use_class_token:
            self.class_token = self.add_weight(
                name='class_token',
                shape=(1, 1, self.embed_dim),
                initializer=self.POSITIONAL_INITIALIZER,
                trainable=True,
            )

            # Expand class token to batch size
            class_tokens = ops.broadcast_to(
                self.class_token,
                (batch_size, 1, self.embed_dim)
            )
            # Concatenate class token with patch embeddings
            x = ops.concatenate([class_tokens, x], axis=1)

        # Add positional embeddings
        self.positional_embedding = self.add_weight(
            name='positional_embedding',
            shape=(1, self.sequence_length, self.embed_dim),
            initializer=self.POSITIONAL_INITIALIZER,
            trainable=True,
        )

        x = x + self.positional_embedding

        # Apply embedding dropout
        self.embedding_dropout = layers.Dropout(self.dropout_rate, name='embedding_dropout')
        x = self.embedding_dropout(x)

        return x

    def _build_encoder_layers(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build transformer encoder layers.

        Args:
            x: Input tensor with positional encodings

        Returns:
            Encoded tensor
        """
        for i in range(self.num_layers):
            # Calculate stochastic depth rate for this layer (linear scaling)
            layer_drop_rate = self.stochastic_depth_rate * i / (self.num_layers - 1) if self.num_layers > 1 else 0.0

            encoder_layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=int(self.embed_dim * self.mlp_ratio),
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                normalization_position='pre',  # DinoV3 uses pre-norm
                ffn_type='mlp',
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=layer_drop_rate > 0.0,
                stochastic_depth_rate=layer_drop_rate,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'encoder_layer_{i}'
            )

            x = encoder_layer(x)
            self.encoder_layers.append(encoder_layer)

        return x

    def _build_final_norm(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build final layer normalization.

        Args:
            x: Encoded tensor

        Returns:
            Normalized tensor
        """
        if self.normalization_type == 'layer_norm':
            self.final_norm = layers.LayerNormalization(
                epsilon=self.LAYERNORM_EPSILON,
                name='final_norm'
            )
        elif self.normalization_type == 'rms_norm':
            from dl_techniques.layers.norms.rms_norm import RMSNorm
            self.final_norm = RMSNorm(name='final_norm')
        else:
            raise ValueError(f"Unknown normalization_type: {self.normalization_type}")

        x = self.final_norm(x)
        return x

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build classification head.

        Args:
            x: Feature tensor

        Returns:
            Classification logits
        """
        # Extract features for classification
        if self.use_class_token:
            # Use class token (first token) for classification
            features = x[:, 0]  # (batch_size, embed_dim)
        elif self.use_mean_pooling:
            # Use global average pooling
            pooling = layers.GlobalAveragePooling1D(name='global_avg_pool')
            features = pooling(x)  # (batch_size, embed_dim)
            self.head_layers.append(pooling)
        else:
            # Use mean of all patch tokens
            features = ops.mean(x, axis=1)  # (batch_size, embed_dim)

        # Classification layer
        if self.num_classes > 0:
            classifier = layers.Dense(
                units=self.num_classes,
                kernel_initializer=self.HEAD_INITIALIZER,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='classifier'
            )
            x = classifier(features)
            self.head_layers.append(classifier)
        else:
            x = features

        return x

    @classmethod
    def from_variant(
        cls,
        variant: str,
        image_size: Tuple[int, int] = (224, 224),
        num_classes: int = 1000,
        include_top: bool = True,
        **kwargs: Any
    ) -> "DinoV3Model":
        """
        Create a DinoV3 model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large", "giant".
            image_size: Tuple of integers, input image size (height, width).
            num_classes: Integer, number of output classes.
            include_top: Boolean, whether to include the classification head.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            DinoV3Model instance configured for the specified variant.

        Raises:
            ValueError: If variant is not recognized.

        Example:
            >>> # Create DinoV3-Base for ImageNet
            >>> model = DinoV3Model.from_variant("base", num_classes=1000)
            >>> # Create DinoV3-Small for CIFAR-10
            >>> model = DinoV3Model.from_variant("small", image_size=(32, 32), num_classes=10)
            >>> # Feature extraction model
            >>> model = DinoV3Model.from_variant("large", include_top=False)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        logger.info(f"Creating DinoV3-{variant.upper()} model")
        logger.info(f"Configuration: {config}")

        return cls(
            image_size=image_size,
            num_classes=num_classes,
            include_top=include_top,
            **config,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'use_class_token': self.use_class_token,
            'use_mean_pooling': self.use_mean_pooling,
            'normalization_type': self.normalization_type,
            'attention_type': self.attention_type,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'include_top': self.include_top,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DinoV3Model":
        """
        Create model from configuration.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            DinoV3Model instance.
        """
        # Deserialize initializers and regularizers
        if config.get("kernel_initializer"):
            config["kernel_initializer"] = initializers.deserialize(
                config["kernel_initializer"]
            )
        if config.get("bias_initializer"):
            config["bias_initializer"] = initializers.deserialize(
                config["bias_initializer"]
            )
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if config.get("bias_regularizer"):
            config["bias_regularizer"] = regularizers.deserialize(
                config["bias_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional DinoV3-specific information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info("DinoV3 Model Configuration:")
        logger.info(f"  - Input shape: {self.image_size} + (3,)")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Number of patches: {self.num_patches} ({self.num_patches_h}x{self.num_patches_w})")
        logger.info(f"  - Sequence length: {self.sequence_length}")
        logger.info(f"  - Embedding dimension: {self.embed_dim}")
        logger.info(f"  - Number of layers: {self.num_layers}")
        logger.info(f"  - Number of heads: {self.num_heads}")
        logger.info(f"  - Head dimension: {self.embed_dim // self.num_heads}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Stochastic depth rate: {self.stochastic_depth_rate}")
        logger.info(f"  - Use class token: {self.use_class_token}")
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - Attention type: {self.attention_type}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        else:
            logger.info("  - Feature extraction mode (no classification head)")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_dinov3(
    variant: str = "base",
    image_size: Tuple[int, int] = (224, 224),
    num_classes: int = 1000,
    include_top: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> DinoV3Model:
    """
    Convenience function to create DinoV3 models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "giant").
        image_size: Tuple of integers, input image size (height, width).
        num_classes: Integer, number of output classes.
        include_top: Boolean, whether to include the classification head.
        pretrained: Boolean, whether to load pretrained weights (not implemented).
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        DinoV3Model instance.

    Example:
        >>> # Create DinoV3-Base for ImageNet
        >>> model = create_dinov3("base", num_classes=1000)
        >>> # Create DinoV3-Small for CIFAR-10
        >>> model = create_dinov3("small", image_size=(32, 32), num_classes=10)
        >>> # Create DinoV3-Large for feature extraction
        >>> model = create_dinov3("large", include_top=False)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    return DinoV3Model.from_variant(
        variant,
        image_size=image_size,
        num_classes=num_classes,
        include_top=include_top,
        **kwargs
    )

# ---------------------------------------------------------------------
