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

    Key features:
    - Patch-level feature extraction for spatial understanding
    - CLS token access for global image representation
    - Spatial feature extraction for dense prediction tasks
    - Configurable architecture parameters

    Mathematical formulation:
        features = ViTSigLIP(images)
        patch_features = features[:, 1:, :]  # Exclude CLS token

    Where patch_features contain spatial information for cross-modal alignment.

    Args:
        img_size: Integer, input image size. Must be positive and divisible by patch_size.
            Defaults to 224.
        patch_size: Integer, size of image patches. Must be positive and divide img_size.
            Defaults to 16.
        embed_dim: Integer, embedding dimension. Must be positive. Defaults to 768.
        depth: Integer, number of transformer blocks. Must be positive. Defaults to 12.
        num_heads: Integer, number of attention heads. Must be positive and divide embed_dim.
            Defaults to 12.
        mlp_ratio: Float, MLP expansion ratio. Must be positive. Defaults to 4.0.
        dropout: Float, dropout rate between 0 and 1. Defaults to 0.0.
        activation: String, activation function name. Accepts standard Keras activations
            like 'gelu', 'relu', 'swish'. Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for convolution kernels.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer, regularizer for convolution kernels.
            Defaults to None.
        bias_regularizer: Optional Regularizer, regularizer for bias vectors.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        - height and width should equal img_size
        - channels is typically 3 for RGB images

    Output shape:
        3D tensor with shape: `(batch_size, num_patches, embed_dim)`
        where num_patches = (img_size // patch_size) ** 2

    Attributes:
        vision_transformer: ViTSigLIP model for feature extraction.
        num_patches: Number of image patches computed from img_size and patch_size.

    Example:
        ```python
        # Basic usage
        encoder = VisionEncoder(img_size=224, embed_dim=768)

        # Small configuration
        encoder = VisionEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=6,
            num_heads=6
        )

        # With regularization
        encoder = VisionEncoder(
            img_size=224,
            embed_dim=768,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            dropout=0.1
        )

        # In a vision-language model
        images = keras.Input(shape=(224, 224, 3))
        patch_features = encoder(images)  # Shape: (batch, 196, 768)

        # Access CLS token for classification
        cls_features = encoder.get_cls_token(images)  # Shape: (batch, 768)
        ```

    Note:
        The layer excludes the CLS token from the default output to provide
        spatial patch-level features suitable for cross-modal alignment.
        Use get_cls_token() or get_spatial_features() for alternative outputs.

    Raises:
        ValueError: If img_size is not positive or not divisible by patch_size.
        ValueError: If embed_dim is not positive or not divisible by num_heads.
        ValueError: If any dimension parameter is not positive.
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
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        # Store ALL configuration parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Computed properties
        self.num_patches = (img_size // patch_size) ** 2

        # CREATE sub-layer in __init__ (following modern Keras 3 pattern)
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

        logger.info(f"Created VisionEncoder with {self.num_patches} patches, "
                    f"embed_dim={embed_dim}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the vision encoder and its sub-layers.

        This method validates input shape and explicitly builds the sub-layer
        for robust serialization following modern Keras 3 patterns.

        Args:
            input_shape: Shape tuple of input images (batch, height, width, channels)

        Raises:
            ValueError: If input shape is invalid or incompatible with configuration.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {input_shape}"
            )

        height, width = input_shape[1], input_shape[2]
        if height is not None and width is not None:
            if height != self.img_size or width != self.img_size:
                logger.warning(
                    f"Input image size ({height}, {width}) does not match "
                    f"configured img_size ({self.img_size}). This may lead to "
                    f"unexpected behavior if positional embeddings are not interpolated."
                )

        # Explicitly build sub-layer for robust serialization
        # This ensures weight variables exist before loading saved weights
        self.vision_transformer.build(input_shape)

        logger.info(f"Building VisionEncoder with input_shape: {input_shape}")

        # Always call parent build at the end
        super().build(input_shape)
        logger.info("VisionEncoder built successfully")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the vision encoder.

        Processes input images through the ViT and extracts patch-level features
        by excluding the CLS token, providing spatial features suitable for
        cross-modal alignment.

        Args:
            inputs: Input images tensor of shape [batch_size, height, width, channels]
            training: Optional boolean indicating training mode. If None, uses
                the default behavior of the sub-layers.

        Returns:
            Patch features tensor of shape [batch_size, num_patches, embed_dim]
            Contains spatial patch-level features excluding the CLS token.
        """
        # Get full ViT features (includes CLS token + patch tokens)
        vit_features = self.vision_transformer(inputs, training=training)

        # Extract patch tokens (exclude CLS token at position 0)
        # This provides spatial features for cross-modal alignment
        patch_features = self.vision_transformer.get_patch_tokens(vit_features)

        return patch_features

    def get_cls_token(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Extract CLS token for global image classification tasks.

        The CLS token provides a global representation of the entire image,
        suitable for image classification or global image understanding tasks.

        Args:
            inputs: Input images tensor of shape [batch_size, height, width, channels]
            training: Optional boolean indicating training mode.

        Returns:
            CLS token features tensor of shape [batch_size, embed_dim]
            Contains global image representation.
        """
        vit_features = self.vision_transformer(inputs, training=training)
        return self.vision_transformer.get_cls_token(vit_features)

    def get_spatial_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Get spatial features reshaped for dense prediction tasks.

        Reshapes patch features into spatial dimensions suitable for tasks
        like segmentation, object detection, or other dense prediction tasks.

        Args:
            inputs: Input images tensor of shape [batch_size, height, width, channels]
            training: Optional boolean indicating training mode.

        Returns:
            Spatial features tensor of shape [batch_size, patch_height, patch_width, embed_dim]
            where patch_height = patch_width = img_size // patch_size
        """
        vit_features = self.vision_transformer(inputs, training=training)
        return self.vision_transformer.get_spatial_features(vit_features)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Input shape tuple (batch_size, height, width, channels)

        Returns:
            Output shape tuple (batch_size, num_patches, embed_dim)
        """
        batch_size = input_shape[0]
        return (batch_size, self.num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL parameters passed to __init__ for complete reconstruction.

        Returns:
            Dictionary containing all layer configuration parameters.
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
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_vision_encoder_base(**kwargs: Any) -> VisionEncoder:
    """
    Create base vision encoder configuration.

    Provides a standard base configuration suitable for most vision-language
    applications with good balance of performance and computational efficiency.

    Args:
        **kwargs: Additional arguments to override base configuration

    Returns:
        VisionEncoder instance with base configuration

    Example:
        ```python
        encoder = create_vision_encoder_base()
        encoder = create_vision_encoder_base(dropout=0.1)  # With custom dropout
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

    logger.info("Creating base vision encoder")
    return VisionEncoder(**config)


def create_vision_encoder_small(**kwargs: Any) -> VisionEncoder:
    """
    Create small vision encoder configuration.

    Provides a smaller, more efficient configuration suitable for resource-
    constrained environments or when faster inference is required.

    Args:
        **kwargs: Additional arguments to override small configuration

    Returns:
        VisionEncoder instance with small configuration

    Example:
        ```python
        encoder = create_vision_encoder_small()
        encoder = create_vision_encoder_small(img_size=196)  # Custom image size
        ```
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


def create_vision_encoder_large(**kwargs: Any) -> VisionEncoder:
    """
    Create large vision encoder configuration.

    Provides a larger, more powerful configuration suitable for high-accuracy
    applications where computational resources are available.

    Args:
        **kwargs: Additional arguments to override large configuration

    Returns:
        VisionEncoder instance with large configuration

    Example:
        ```python
        encoder = create_vision_encoder_large()
        encoder = create_vision_encoder_large(depth=36)  # Even deeper model
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

    logger.info("Creating large vision encoder")
    return VisionEncoder(**config)