import keras
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .vit_siglip import ViTSigLIP


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VisionEncoder(keras.layers.Layer):
    """
    Vision encoder using ViTSigLIP architecture for vision-language models.

    This layer wraps a ViTSigLIP model to provide visual feature extraction
    for multimodal applications. It processes input images and returns
    patch-level features suitable for cross-modal fusion.

    The encoder extracts patch tokens from the ViT output (excluding the CLS token)
    to provide spatial visual features that can be aligned with text tokens
    in vision-language models.

    Args:
        img_size: Input image size. Defaults to 224.
        patch_size: Size of image patches. Defaults to 16.
        embed_dim: Embedding dimension. Defaults to 768.
        depth: Number of transformer blocks. Defaults to 12.
        num_heads: Number of attention heads. Defaults to 12.
        mlp_ratio: MLP expansion ratio. Defaults to 4.0.
        dropout: Dropout rate. Defaults to 0.0.
        activation: Activation function. Defaults to 'gelu'.
        use_bias: Whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Initializer for convolution kernels.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for convolution kernels.
        bias_regularizer: Regularizer for bias vectors.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor of shape (batch_size, height, width, channels)

    Output shape:
        3D tensor of shape (batch_size, num_patches, embed_dim)
        where num_patches = (img_size // patch_size) ** 2

    Example:
        >>> encoder = VisionEncoder(
        ...     img_size=224,
        ...     patch_size=16,
        ...     embed_dim=768,
        ...     depth=12
        ... )
        >>>
        >>> images = keras.ops.random.normal((2, 224, 224, 3))
        >>> features = encoder(images)  # Shape: (2, 196, 768)
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
            use_bias: bool = True,
            kernel_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            kernel_regularizer: Optional[str] = None,
            bias_regularizer: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Computed properties
        self.num_patches = (img_size // patch_size) ** 2

        # Initialize vision transformer (will be created in build())
        self.vision_transformer = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(f"Created VisionEncoder with {self.num_patches} patches, "
                    f"embed_dim={embed_dim}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the vision encoder by creating the ViTSigLIP model.

        Args:
            input_shape: Shape tuple of input images (batch, height, width, channels)

        Raises:
            ValueError: If input shape is invalid
        """
        if self.built:
            return

        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {input_shape}"
            )

        if input_shape[1] != self.img_size or input_shape[2] != self.img_size:
            raise ValueError(
                f"Input image size {input_shape[1:3]} must match "
                f"configured img_size ({self.img_size})"
            )

        logger.info(f"Building VisionEncoder with input_shape: {input_shape}")

        # Create ViTSigLIP model
        self.vision_transformer = ViTSigLIP(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='vit_siglip'
        )

        # Build the vision transformer
        self.vision_transformer.build(input_shape)

        super().build(input_shape)
        logger.info("VisionEncoder built successfully")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the vision encoder.

        Args:
            inputs: Input images of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Patch features of shape [batch_size, num_patches, embed_dim]
            Excludes the CLS token to provide spatial patch-level features
        """
        # Get full ViT features (includes CLS token + patch tokens)
        vit_features = self.vision_transformer(inputs, training=training)

        # Extract patch tokens (exclude CLS token at position 0)
        patch_features = self.vision_transformer.get_patch_tokens(vit_features)

        return patch_features

    def get_cls_token(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Extract CLS token for classification tasks.

        Args:
            inputs: Input images of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            CLS token features of shape [batch_size, embed_dim]
        """
        vit_features = self.vision_transformer(inputs, training=training)
        return self.vision_transformer.get_cls_token(vit_features)

    def get_spatial_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Get spatial features for dense prediction tasks.

        Args:
            inputs: Input images of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Spatial features of shape [batch_size, patch_height, patch_width, embed_dim]
        """
        vit_features = self.vision_transformer(inputs, training=training)
        return self.vision_transformer.get_spatial_features(vit_features)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple [batch_size, num_patches, embed_dim]
        """
        batch_size = input_shape[0]
        return (batch_size, self.num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
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
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build layer from configuration.

        Args:
            config: Build configuration dictionary
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VisionEncoder':
        """
        Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            VisionEncoder instance
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_vision_encoder_base(**kwargs) -> VisionEncoder:
    """
    Create base vision encoder configuration.

    Args:
        **kwargs: Additional arguments to override base configuration

    Returns:
        VisionEncoder with base configuration
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

    logger.info("Creating base vision encoder")
    return VisionEncoder(**config)


def create_vision_encoder_small(**kwargs) -> VisionEncoder:
    """
    Create small vision encoder configuration.

    Args:
        **kwargs: Additional arguments to override small configuration

    Returns:
        VisionEncoder with small configuration
    """
    config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    }
    config.update(kwargs)

    logger.info("Creating small vision encoder")
    return VisionEncoder(**config)


def create_vision_encoder_large(**kwargs) -> VisionEncoder:
    """
    Create large vision encoder configuration.

    Args:
        **kwargs: Additional arguments to override large configuration

    Returns:
        VisionEncoder with large configuration
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

    logger.info("Creating large vision encoder")
    return VisionEncoder(**config)