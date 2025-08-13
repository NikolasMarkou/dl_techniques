"""
Vision Encoder Layer for Vision-Language Models

This module provides `VisionEncoder`, a Keras Layer that wraps the SigLIP Vision
Transformer to provide patch-level image features for vision-language tasks.

The VisionEncoder serves as a specialized wrapper around the SigLIP Vision Transformer
that extracts patch tokens (excluding the CLS token) for use in multimodal models.
This is particularly useful for tasks requiring spatial understanding, such as:

- Vision-Language Models (VLMs)
- Visual Question Answering (VQA)
- Image Captioning
- Cross-modal retrieval
- Dense prediction tasks requiring spatial features

Key Features:

1. **Patch Token Extraction**: Extracts spatial patch tokens, excluding CLS token
2. **SigLIP Integration**: Uses SigLIP Vision Transformer for robust visual features
3. **Configurable Architecture**: Supports different model sizes and configurations
4. **Spatial Preservation**: Maintains spatial relationships in patch features
5. **Modern Keras 3**: Follows modern serialization and build patterns

Example:
    ```python
    # Basic usage
    encoder = VisionEncoder(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    )

    # Process images to get patch features
    images = keras.random.normal((8, 224, 224, 3))
    patch_features = encoder(images)  # Shape: (8, 196, 768)

    # Use factory functions for standard configurations
    encoder = create_vision_encoder_base()    # Base config
    encoder = create_vision_encoder_small()   # Small config
    encoder = create_vision_encoder_large()   # Large config
    ```
"""

import keras
from keras import initializers, regularizers
from typing import Optional, Dict, Any, Tuple, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from  .vit_siglip import SigLIPVisionTransformer


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisionEncoder(keras.layers.Layer):
    """
    Vision Encoder layer that extracts patch features from images.

    This layer wraps a SigLIP Vision Transformer to provide spatial patch features
    for vision-language models and other multimodal applications. It specifically
    extracts patch tokens (excluding the CLS token) to preserve spatial information.

    The encoder processes input images through a Vision Transformer and returns
    the patch-level features, which maintain spatial relationships and are suitable
    for tasks requiring fine-grained visual understanding.

    Args:
        img_size: Integer, input image size (square images). Must be positive and
            divisible by patch_size. Defaults to 224.
        patch_size: Integer, size of image patches. Must be positive and divide
            img_size evenly. Defaults to 16.
        embed_dim: Integer, embedding dimension for patch features. Must be positive
            and divisible by num_heads. Defaults to 768.
        depth: Integer, number of transformer layers. Must be positive. Defaults to 12.
        num_heads: Integer, number of attention heads. Must be positive and divide
            embed_dim evenly. Defaults to 12.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Must be positive. Defaults to 4.0.
        dropout: Float, dropout rate for transformer blocks. Must be between 0 and 1.
            Defaults to 0.0.
        activation: String or callable, activation function for MLP layers.
            Defaults to 'gelu'.
        attention_type: String, type of attention mechanism. Passed to underlying
            transformer layers. Defaults to 'multi_head_attention'.
        normalization_type: String, type of normalization. Passed to underlying
            transformer layers. Defaults to 'layer_norm'.
        normalization_position: String, position of normalization ('pre' or 'post').
            Defaults to 'post'.
        ffn_type: String, type of feed-forward network. Passed to underlying
            transformer layers. Defaults to 'mlp'.
        use_bias: Boolean, whether to use bias terms in linear layers. Defaults to True.
        kernel_initializer: String or initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, img_size, img_size, 3)`

    Output shape:
        3D tensor with shape: `(batch_size, num_patches, embed_dim)`
        Where num_patches = (img_size // patch_size) ** 2

    Raises:
        ValueError: If img_size is not divisible by patch_size.
        ValueError: If embed_dim is not divisible by num_heads.
        ValueError: If any parameter is invalid.

    Example:
        ```python
        # Basic usage for VQA
        vision_encoder = VisionEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            dropout=0.1
        )

        # Process a batch of images
        images = keras.random.normal((32, 224, 224, 3))
        patch_features = vision_encoder(images)  # Shape: (32, 196, 768)

        # Use with different image sizes
        large_encoder = VisionEncoder(
            img_size=384,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16
        )

        # For dense prediction tasks
        dense_encoder = VisionEncoder(
            img_size=224,
            patch_size=8,  # Smaller patches for finer granularity
            embed_dim=768,
            depth=12
        )
        ```

    Note:
        This layer extracts only the patch tokens from the Vision Transformer,
        excluding the CLS token. If you need the CLS token for classification,
        use the underlying SigLIPVisionTransformer directly.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            activation: str = 'gelu',
            attention_type: str = 'multi_head_attention',
            normalization_type: str = 'layer_norm',
            normalization_position: str = 'post',
            ffn_type: str = 'mlp',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(img_size, patch_size, embed_dim, depth, num_heads,
                              mlp_ratio, dropout)

        # Store configuration parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.activation = activation
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.use_bias = use_bias

        # Store initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Calculate derived parameters
        self.num_patches = (img_size // patch_size) ** 2

        # CREATE the underlying Vision Transformer in __init__ (Modern Keras 3 pattern)
        self.vision_transformer = SigLIPVisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            activation=self.activation,
            attention_type=self.attention_type,
            normalization_type=self.normalization_type,
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='vision_transformer'
        )

        logger.info(
            f"Created VisionEncoder: {self.num_patches} patches, "
            f"{embed_dim}D features, {depth} layers"
        )

    def _validate_config(
            self,
            img_size: int,
            patch_size: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            mlp_ratio: float,
            dropout: float
    ) -> None:
        """
        Validate configuration parameters.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            embed_dim: Embedding dimension.
            depth: Number of layers.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            dropout: Dropout rate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        This method validates the input shape and ensures the underlying
        Vision Transformer is properly configured. No additional weights
        are created as this layer is a wrapper.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is invalid.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        expected_height, expected_width = self.img_size, self.img_size
        actual_height, actual_width = input_shape[1], input_shape[2]

        if (actual_height != expected_height or actual_width != expected_width):
            if actual_height is not None and actual_width is not None:
                logger.warning(
                    f"Input image size ({actual_height}, {actual_width}) "
                    f"doesn't match configured img_size ({self.img_size}). "
                    f"This may cause runtime errors."
                )

        if input_shape[3] != 3:
            logger.warning(
                f"Expected 3 input channels (RGB), got {input_shape[3]}. "
                f"Model may not work correctly."
            )

        # Let Keras know the build is complete
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the Vision Encoder.

        This method processes input images through the Vision Transformer and
        extracts patch tokens, excluding the CLS token for spatial tasks.

        Args:
            inputs: Input tensor with shape (batch_size, img_size, img_size, 3).
            training: Boolean indicating whether the model is in training mode.

        Returns:
            Patch feature tensor with shape (batch_size, num_patches, embed_dim).
        """
        # Get full Vision Transformer features (CLS + patches)
        vit_features = self.vision_transformer(inputs, training=training)

        # Extract only patch tokens (exclude CLS token at index 0)
        patch_features = self.vision_transformer.get_patch_tokens(vit_features)

        return patch_features

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (batch_size, num_patches, embed_dim).
        """
        batch_size = input_shape[0]
        return (batch_size, self.num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed
            to recreate this layer.
        """
        config = super().get_config()
        config.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout,
            'activation': self.activation,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------


def create_vision_encoder_base(**kwargs: Any) -> VisionEncoder:
    """
    Create base vision encoder configuration.

    Base configuration provides a good balance of performance and efficiency:
    - 768 embedding dimensions
    - 12 transformer layers
    - 12 attention heads
    - ~86M parameters in the vision transformer
    - Suitable for most vision-language tasks

    Args:
        **kwargs: Additional arguments to override base configuration.
            Common overrides:
            - img_size: Change input image size
            - dropout: Add regularization
            - patch_size: Change patch granularity

    Returns:
        VisionEncoder with base configuration.

    Example:
        ```python
        # Standard base encoder
        encoder = create_vision_encoder_base()

        # Base encoder with regularization
        encoder = create_vision_encoder_base(dropout=0.1)

        # Base encoder with larger input size
        encoder = create_vision_encoder_base(img_size=384)

        # Base encoder with finer patches
        encoder = create_vision_encoder_base(patch_size=8)
        ```
    """
    config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.0,
    }
    config.update(kwargs)

    logger.info("Creating base vision encoder (ViT-Base equivalent)")
    return VisionEncoder(**config)


def create_vision_encoder_small(**kwargs: Any) -> VisionEncoder:
    """
    Create small vision encoder configuration.

    Small configuration is optimized for efficiency and smaller datasets:
    - 384 embedding dimensions
    - 6 transformer layers
    - 6 attention heads
    - ~22M parameters in the vision transformer
    - Faster training and inference
    - Good for resource-constrained environments

    Args:
        **kwargs: Additional arguments to override small configuration.

    Returns:
        VisionEncoder with small configuration.

    Example:
        ```python
        # Standard small encoder
        encoder = create_vision_encoder_small()

        # Small encoder with more regularization for small datasets
        encoder = create_vision_encoder_small(dropout=0.2)

        # Small encoder with different image size
        encoder = create_vision_encoder_small(img_size=128)
        ```
    """
    config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,  # Default dropout for small model
    }
    config.update(kwargs)

    logger.info("Creating small vision encoder (ViT-Small equivalent)")
    return VisionEncoder(**config)


def create_vision_encoder_large(**kwargs: Any) -> VisionEncoder:
    """
    Create large vision encoder configuration.

    Large configuration provides maximum capacity for complex tasks:
    - 1024 embedding dimensions
    - 24 transformer layers
    - 16 attention heads
    - ~307M parameters in the vision transformer
    - Best performance on challenging datasets
    - Requires significant computational resources

    Args:
        **kwargs: Additional arguments to override large configuration.

    Returns:
        VisionEncoder with large configuration.

    Example:
        ```python
        # Standard large encoder
        encoder = create_vision_encoder_large()

        # Large encoder with different normalization
        encoder = create_vision_encoder_large(
            normalization_position='pre'
        )

        # Large encoder with different attention mechanism
        encoder = create_vision_encoder_large(
            attention_type='window_attention'
        )
        ```
    """
    config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'dropout': 0.0,
    }
    config.update(kwargs)

    logger.info("Creating large vision encoder (ViT-Large equivalent)")
    return VisionEncoder(**config)


def create_vision_encoder_custom(
        size: str = 'base',
        img_size: int = 224,
        patch_size: int = 16,
        **kwargs: Any
) -> VisionEncoder:
    """
    Create custom vision encoder with flexible configuration.

    This function provides a convenient way to create encoders with custom
    image sizes and patch sizes while maintaining standard model sizes.

    Args:
        size: String, base model size ('small', 'base', 'large').
        img_size: Integer, input image size. Must be divisible by patch_size.
        patch_size: Integer, patch size. Must divide img_size evenly.
        **kwargs: Additional arguments to override configuration.

    Returns:
        VisionEncoder with custom configuration.

    Raises:
        ValueError: If size is not recognized.

    Example:
        ```python
        # High-resolution base encoder
        encoder = create_vision_encoder_custom(
            size='base',
            img_size=384,
            patch_size=16
        )

        # Fine-grained small encoder
        encoder = create_vision_encoder_custom(
            size='small',
            img_size=224,
            patch_size=8
        )

        # Custom large encoder with specific configuration
        encoder = create_vision_encoder_custom(
            size='large',
            img_size=512,
            patch_size=32,
            dropout=0.1,
            normalization_position='pre'
        )
        ```
    """
    # Select base configuration
    if size == 'small':
        factory_fn = create_vision_encoder_small
    elif size == 'base':
        factory_fn = create_vision_encoder_base
    elif size == 'large':
        factory_fn = create_vision_encoder_large
    else:
        raise ValueError(f"Unknown size '{size}'. Must be 'small', 'base', or 'large'")

    # Override with custom parameters
    custom_kwargs = {
        'img_size': img_size,
        'patch_size': patch_size,
        **kwargs
    }

    logger.info(
        f"Creating custom {size} vision encoder: "
        f"{img_size}px images, {patch_size}px patches"
    )

    return factory_fn(**custom_kwargs)