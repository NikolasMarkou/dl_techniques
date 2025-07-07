"""
Enhanced Vision Transformer Model with Register Tokens

This implementation incorporates the breakthrough findings from "Vision Transformers Need Registers"
paper, which discovered that ViTs develop high-norm artifact tokens that can be eliminated using
register tokens, leading to smoother feature maps and better performance on dense prediction tasks.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.positional_embedding import PositionalEmbedding
from dl_techniques.layers.vision_transformer_register import RegisterEnhancedVisionTransformerLayer

@keras.saving.register_keras_serializable()
class RegisterEnhancedViT(keras.Model):
    """
    Enhanced Vision Transformer with Register Tokens.

    This model implements the key findings from "Vision Transformers Need Registers" research:
    - Adds learnable register tokens to provide dedicated spaces for internal computations
    - Eliminates high-norm artifact tokens that appear in low-informative background areas
    - Results in smoother attention maps and improved dense prediction performance
    - Register tokens are processed through all layers but excluded from final outputs

    The register tokens act as dedicated "scratch space" for the model's internal computations,
    preventing the model from hijacking patch tokens for global information storage.

    Args:
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of output classes for classification.
        scale: Model scale configuration ('tiny', 'small', 'base', 'large', 'huge').
        patch_size: Size of patches to extract from input images.
        num_registers: Number of register tokens to add (typically 4).
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction ('cls', 'mean', 'max', None).
        dropout_rate: Dropout rate for regularization.
        attention_dropout_rate: Dropout rate for attention weights.
        pos_dropout_rate: Dropout rate for positional embeddings.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        norm_type: Type of normalization ('layer' or 'rms').
        name: Model name.

    Returns:
        Model outputs depend on include_top and pooling settings.
        Register tokens are automatically excluded from outputs.
    """

    # Scale configurations: [embed_dim, num_heads, num_layers, mlp_ratio]
    SCALE_CONFIGS = {
        "micro": [64, 3, 4, 4.0],  # ViT-Micro
        "tiny": [192, 3, 12, 4.0],  # ViT-Tiny
        "small": [384, 6, 12, 4.0],  # ViT-Small
        "base": [768, 12, 12, 4.0],  # ViT-Base
        "large": [1024, 16, 24, 4.0],  # ViT-Large
        "huge": [1280, 16, 32, 4.0],  # ViT-Huge
    }

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            num_classes: int = 1000,
            scale: str = "base",
            patch_size: Union[int, Tuple[int, int]] = 16,
            num_registers: int = 4,
            include_top: bool = True,
            pooling: Optional[str] = None,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            pos_dropout_rate: float = 0.0,
            kernel_initializer: str = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            norm_type: str = "layer",
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize Enhanced Vision Transformer with Register Tokens.
        """
        if name is None:
            name = f"register_enhanced_vit_{scale}"
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.input_shape_config = input_shape
        self.num_classes = num_classes
        self.scale = scale
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.norm_type = norm_type

        # Validate inputs
        if scale not in self.SCALE_CONFIGS:
            raise ValueError(
                f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}"
            )

        if pooling not in [None, "cls", "mean", "max"]:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Choose from [None, 'cls', 'mean', 'max']"
            )

        if not include_top and pooling is None:
            logger.warning(
                "include_top=False and pooling=None will return the raw transformer output excluding registers"
            )

        if num_registers < 0:
            raise ValueError(f"num_registers must be >= 0, got {num_registers}")

        # Get model configuration
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate sequence length for positional embeddings
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size

        img_h, img_w = input_shape[:2]
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        # CLS token + patches + register tokens
        self.max_seq_len = 1 + self.num_patches + self.num_registers

        # Initialize layers as None - they will be created in build()
        self.patch_embed = None
        self.cls_token = None
        self.register_tokens = None
        self.pos_embed = None
        self.transformer_layers = None
        self.norm = None
        self.head_dropout = None
        self.head = None
        self.global_pool = None

        # Store build state for serialization
        self._build_input_shape = None
        self._layers_built = False

        logger.info(
            f"Created RegisterEnhancedViT-{scale} with {self.embed_dim}d, {self.num_heads}h, "
            f"{self.num_layers}L, {self.num_registers} register tokens"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model layers."""
        if self._layers_built:
            return

        self._build_input_shape = input_shape
        self._build_layers()
        self._layers_built = True
        super().build(input_shape)

    def _build_layers(self) -> None:
        """Initialize all model layers."""
        # Patch embedding layer
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="patch_embed"
        )

        # Build patch embedding to get actual number of patches
        dummy_input_shape = (None,) + self.input_shape_config
        self.patch_embed.build(dummy_input_shape)

        # Get actual number of patches from built layer
        actual_num_patches = self.patch_embed.num_patches
        if actual_num_patches is not None:
            self.num_patches = actual_num_patches
            self.max_seq_len = 1 + self.num_patches + self.num_registers

        # CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )

        # Register tokens - key innovation from the research
        if self.num_registers > 0:
            self.register_tokens = self.add_weight(
                name="register_tokens",
                shape=(1, self.num_registers, self.embed_dim),
                initializer="zeros",
                trainable=True
            )
        else:
            self.register_tokens = None

        # Positional embedding
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout=self.pos_dropout_rate,
            name="pos_embed"
        )

        # Transformer layers - these will process both patches and register tokens
        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = RegisterEnhancedVisionTransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                norm_type=self.norm_type,
                process_registers=True,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Final normalization
        if self.norm_type == "rms":
            self.norm = RMSNorm(name="norm")
        else:
            self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        # Classification head (if include_top)
        if self.include_top:
            self.head_dropout = keras.layers.Dropout(self.dropout_rate, name="head_dropout")
            self.head = keras.layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="head"
            )

        # Global pooling layers (if needed)
        if self.pooling == "mean":
            self.global_pool = keras.layers.GlobalAveragePooling1D(name="global_avg_pool")
        elif self.pooling == "max":
            self.global_pool = keras.layers.GlobalMaxPooling1D(name="global_max_pool")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Model output tensor. Shape depends on include_top and pooling settings.
            Register tokens are automatically excluded from outputs.
        """
        # Convert image to patches
        x = self.patch_embed(inputs, training=training)  # (batch_size, num_patches, embed_dim)

        # Add CLS token
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))

        # Add register tokens if configured
        if self.num_registers > 0 and self.register_tokens is not None:
            register_tokens = ops.broadcast_to(
                self.register_tokens,
                (batch_size, self.num_registers, self.embed_dim)
            )
            # Sequence: [CLS, patches, registers]
            x = ops.concatenate([cls_tokens, x, register_tokens], axis=1)
        else:
            # Sequence: [CLS, patches]
            x = ops.concatenate([cls_tokens, x], axis=1)

        # Add positional embeddings (includes dropout)
        x = self.pos_embed(x, training=training)

        # Apply transformer layers (they process all tokens including registers)
        for layer in self.transformer_layers:
            x = layer(x, num_registers=self.num_registers, training=training)

        # Apply final normalization
        x = self.norm(x, training=training)

        # CRITICAL: Exclude register tokens from outputs
        # The register tokens served their purpose during computation
        if self.num_registers > 0:
            # Extract only [CLS, patches], exclude the last num_registers tokens
            x = x[:, :-(self.num_registers), :]

        # Handle different output modes
        if self.include_top:
            # Extract CLS token for classification
            cls_token = x[:, 0, :]  # (batch_size, embed_dim)
            x = self.head_dropout(cls_token, training=training)
            x = self.head(x)  # (batch_size, num_classes)
            return x
        else:
            # Feature extraction mode
            if self.pooling == "cls":
                # Return CLS token
                return x[:, 0, :]  # (batch_size, embed_dim)
            elif self.pooling == "mean":
                # Global average pooling over sequence (excluding CLS)
                patch_tokens = x[:, 1:, :]  # Exclude CLS token
                return self.global_pool(patch_tokens)  # (batch_size, embed_dim)
            elif self.pooling == "max":
                # Global max pooling over sequence (excluding CLS)
                patch_tokens = x[:, 1:, :]  # Exclude CLS token
                return self.global_pool(patch_tokens)  # (batch_size, embed_dim)
            else:
                # Return full transformer output (without registers)
                return x  # (batch_size, 1 + num_patches, embed_dim)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output shape.
        """
        batch_size = input_shape[0]

        if self.include_top:
            return (batch_size, self.num_classes)
        else:
            if self.pooling in ["cls", "mean", "max"]:
                return (batch_size, self.embed_dim)
            else:
                # Return shape excludes register tokens
                return (batch_size, 1 + self.num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes,
            "scale": self.scale,
            "patch_size": self.patch_size,
            "num_registers": self.num_registers,
            "include_top": self.include_top,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "pos_dropout_rate": self.pos_dropout_rate,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "norm_type": self.norm_type,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build model from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RegisterEnhancedViT":
        """Create model from configuration."""
        return cls(**config)

    def get_feature_extractor(self) -> "RegisterEnhancedViT":
        """
        Get a feature extractor version of this model.

        Returns:
            New RegisterEnhancedViT configured for feature extraction.
        """
        return RegisterEnhancedViT(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            num_registers=self.num_registers,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            pos_dropout_rate=self.pos_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            norm_type=self.norm_type,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Print detailed model summary including register token information."""
        logger.info(f"Enhanced Vision Transformer Model Summary")
        logger.info(f"Scale: {self.scale}")
        logger.info(f"Input Shape: {self.input_shape_config}")
        logger.info(f"Patch Size: {self.patch_size}")
        logger.info(f"Number of Patches: {self.num_patches}")
        logger.info(f"Number of Register Tokens: {self.num_registers}")
        logger.info(
            f"Total Sequence Length: {self.max_seq_len} (1 CLS + {self.num_patches} patches + {self.num_registers} registers)")
        logger.info(f"Embedding Dimension: {self.embed_dim}")
        logger.info(f"Number of Heads: {self.num_heads}")
        logger.info(f"Number of Layers: {self.num_layers}")
        logger.info(f"MLP Ratio: {self.mlp_ratio}")
        logger.info(f"Dropout Rate: {self.dropout_rate}")
        logger.info(f"Attention Dropout Rate: {self.attention_dropout_rate}")
        logger.info(f"Positional Dropout Rate: {self.pos_dropout_rate}")
        logger.info(f"Normalization Type: {self.norm_type}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self._layers_built:
            logger.info(f"Total Parameters: {self.count_params():,}")


def create_register_enhanced_vit(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 1000,
        scale: str = "base",
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_registers: int = 4,
        include_top: bool = True,
        pooling: Optional[str] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        pos_dropout_rate: float = 0.0,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        norm_type: str = "layer",
        **kwargs: Any
) -> RegisterEnhancedViT:
    """
    Create a Register-Enhanced Vision Transformer model with specified configuration.

    This function creates a ViT model that incorporates the latest research findings
    on register tokens, which eliminate artifacts in vision transformers and improve
    performance on dense prediction tasks.

    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes for classification.
        scale: Model scale ('tiny', 'small', 'base', 'large', 'huge').
        patch_size: Size of patches to extract from input images.
        num_registers: Number of register tokens (typically 4).
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction ('cls', 'mean', 'max', None).
        dropout_rate: Dropout rate for regularization.
        attention_dropout_rate: Dropout rate for attention weights.
        pos_dropout_rate: Dropout rate for positional embeddings.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        norm_type: Type of normalization ('layer' or 'rms').
        **kwargs: Additional arguments for RegisterEnhancedViT.

    Returns:
        RegisterEnhancedViT instance.

    Examples:
        >>> # Create a ViT-Base model with register tokens for ImageNet
        >>> model = create_register_enhanced_vit(
        ...     input_shape=(224, 224, 3),
        ...     num_classes=1000,
        ...     scale="base",
        ...     num_registers=4
        ... )

        >>> # Create a feature extractor with register tokens
        >>> feature_extractor = create_register_enhanced_vit(
        ...     input_shape=(224, 224, 3),
        ...     scale="small",
        ...     include_top=False,
        ...     pooling="cls",
        ...     num_registers=4
        ... )

        >>> # Create a ViT for CIFAR-10 with register tokens
        >>> cifar_model = create_register_enhanced_vit(
        ...     input_shape=(32, 32, 3),
        ...     num_classes=10,
        ...     scale="tiny",
        ...     patch_size=4,
        ...     num_registers=2,  # Fewer registers for smaller images
        ...     dropout_rate=0.1
        ... )
    """
    model = RegisterEnhancedViT(
        input_shape=input_shape,
        num_classes=num_classes,
        scale=scale,
        patch_size=patch_size,
        num_registers=num_registers,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        pos_dropout_rate=pos_dropout_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        norm_type=norm_type,
        **kwargs
    )

    logger.info(f"RegisterEnhancedViT-{scale} with {num_registers} register tokens created successfully")
    return model


# Convenience functions for different scales
def create_register_vit_tiny(**kwargs: Any) -> RegisterEnhancedViT:
    """Create ViT-Tiny model with register tokens."""
    return create_register_enhanced_vit(scale="tiny", **kwargs)


def create_register_vit_small(**kwargs: Any) -> RegisterEnhancedViT:
    """Create ViT-Small model with register tokens."""
    return create_register_enhanced_vit(scale="small", **kwargs)


def create_register_vit_base(**kwargs: Any) -> RegisterEnhancedViT:
    """Create ViT-Base model with register tokens."""
    return create_register_enhanced_vit(scale="base", **kwargs)


def create_register_vit_large(**kwargs: Any) -> RegisterEnhancedViT:
    """Create ViT-Large model with register tokens."""
    return create_register_enhanced_vit(scale="large", **kwargs)


def create_register_vit_huge(**kwargs: Any) -> RegisterEnhancedViT:
    """Create ViT-Huge model with register tokens."""
    return create_register_enhanced_vit(scale="huge", **kwargs)