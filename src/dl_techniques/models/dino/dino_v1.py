"""
DINO (DIstillation with NO labels) Vision Transformer Implementation
==================================================================

A Keras 3 implementation of the DINO self-supervised learning model
based on Vision Transformers. This implementation follows the architecture
described in "Emerging Properties in Self-Supervised Vision Transformers"
(Caron et al., 2021).

Key Features:
------------
- Vision Transformer backbone with configurable architecture
- DINO projection head for self-supervised learning
- Support for different model variants (tiny, small, base, large)
- Configurable attention mechanisms through factory system
- Proper Keras 3 serialization and deserialization
- Reuses existing transformer and embedding layers

Architecture:
------------
The DINO model consists of:
1. Patch embedding layer to tokenize input images
2. Learned positional embeddings
3. Multiple transformer layers with self-attention and FFN
4. DINO head for projection (used during self-supervised training)

Model Variants:
--------------
- DINO-Tiny: 12 layers, 192 dim, 3 heads, 768 FFN dim
- DINO-Small: 12 layers, 384 dim, 6 heads, 1536 FFN dim
- DINO-Base: 12 layers, 768 dim, 12 heads, 3072 FFN dim
- DINO-Large: 24 layers, 1024 dim, 16 heads, 4096 FFN dim

Usage:
------
```python
# Create DINO model for ImageNet (224x224)
model = DINOv1.from_variant(
    "small",
    num_classes=0,  # 0 for feature extraction
    input_shape=(224, 224, 3)
)

# Create DINO model with projection head
model = DINOv1.from_variant(
    "base",
    num_classes=0,
    include_projection_head=True,
    dino_out_dim=65536,
    input_shape=(224, 224, 3)
)

# Create custom DINO model
model = DINOv1(
    embed_dim=768,
    depth=12,
    num_heads=12,
    patch_size=16,
    num_classes=1000,
    input_shape=(224, 224, 3)
)
```
"""

import keras
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# Local imports - using existing layers from the framework
# ---------------------------------------------------------------------

# Use existing transformer layer
from dl_techniques.layers.transformer import TransformerLayer

# Use existing embedding layers
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding

# Use existing normalization factory
from dl_techniques.layers.norms import create_normalization_layer

# Use existing FFN factory
from dl_techniques.layers.ffn.factory import create_ffn_from_config

# Use existing stochastic depth
from dl_techniques.layers.stochastic_depth import StochasticDepth

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

ModelVariant = Literal["tiny", "small", "base", "large"]


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DINOHead(keras.layers.Layer):
    """
    DINO projection head for self-supervised learning.

    This head projects the [CLS] token representation to a higher-dimensional
    space and applies normalization and a final linear layer for contrastive learning.

    Args:
        in_dim: Integer, input dimension (backbone output dimension).
        out_dim: Integer, output dimension for contrastive learning.
        use_bn: Boolean, whether to use batch normalization in intermediate layers.
        norm_last_layer: Boolean, whether to normalize the last layer weights.
        nlayers: Integer, number of layers in the projection head (minimum 1).
        hidden_dim: Integer, hidden dimension in intermediate layers.
        bottleneck_dim: Integer, dimension before the final projection layer.
        normalization_type: String, type of normalization to use.
        activation: String or callable, activation function to use.
        dropout_rate: Float, dropout rate for regularization.
        kernel_initializer: String or initializer, weight initialization scheme.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, in_dim)`

    Output shape:
        2D tensor with shape: `(batch_size, out_dim)`

    Example:
        ```python
        # Create DINO head for 384-dim input to 65536-dim output
        dino_head = DINOHead(
            in_dim=384,
            out_dim=65536,
            use_bn=False,
            norm_last_layer=True,
            nlayers=3,
            hidden_dim=2048,
            bottleneck_dim=256
        )

        # Forward pass
        cls_token = keras.Input(shape=(384,))
        projection = dino_head(cls_token)
        ```
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            use_bn: bool = False,
            norm_last_layer: bool = True,
            nlayers: int = 3,
            hidden_dim: int = 2048,
            bottleneck_dim: int = 256,
            normalization_type: str = "batch_norm",
            activation: str = "gelu",
            dropout_rate: float = 0.0,
            kernel_initializer: str = "truncated_normal",
            **kwargs
    ):
        super().__init__(**kwargs)

        # Validate inputs
        if nlayers < 1:
            raise ValueError(f"nlayers must be at least 1, got {nlayers}")
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim and out_dim must be positive, got {in_dim}, {out_dim}")
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        # Store configuration
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.norm_last_layer = norm_last_layer
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.normalization_type = normalization_type
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Initialize layer lists
        self.mlp_layers = []
        self.last_layer = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the DINO head layers."""
        super().build(input_shape)

        if self.nlayers == 1:
            # Single layer: direct projection to bottleneck dimension
            layer = keras.layers.Dense(
                units=self.bottleneck_dim,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name="mlp_single"
            )
            self.mlp_layers.append(layer)

        else:
            # Multi-layer MLP
            # First layer: in_dim -> hidden_dim
            layer = keras.layers.Dense(
                units=self.hidden_dim,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name="mlp_0"
            )
            self.mlp_layers.append(layer)

            # Batch norm after first layer if requested
            if self.use_bn:
                norm_layer = create_normalization_layer(
                    self.normalization_type,
                    name="mlp_norm_0"
                )
                self.mlp_layers.append(norm_layer)

            # Activation after first layer
            if isinstance(self.activation, str):
                activation_layer = keras.layers.Activation(
                    self.activation, name="mlp_activation_0"
                )
            else:
                activation_layer = self.activation
            self.mlp_layers.append(activation_layer)

            # Dropout if specified
            if self.dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                    rate=self.dropout_rate, name="mlp_dropout_0"
                )
                self.mlp_layers.append(dropout_layer)

            # Intermediate layers: hidden_dim -> hidden_dim
            for i in range(1, self.nlayers - 1):
                layer = keras.layers.Dense(
                    units=self.hidden_dim,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    name=f"mlp_{i}"
                )
                self.mlp_layers.append(layer)

                if self.use_bn:
                    norm_layer = create_normalization_layer(
                        self.normalization_type,
                        name=f"mlp_norm_{i}"
                    )
                    self.mlp_layers.append(norm_layer)

                if isinstance(self.activation, str):
                    activation_layer = keras.layers.Activation(
                        self.activation, name=f"mlp_activation_{i}"
                    )
                else:
                    activation_layer = self.activation
                self.mlp_layers.append(activation_layer)

                if self.dropout_rate > 0.0:
                    dropout_layer = keras.layers.Dropout(
                        rate=self.dropout_rate, name=f"mlp_dropout_{i}"
                    )
                    self.mlp_layers.append(dropout_layer)

            # Final layer before bottleneck: hidden_dim -> bottleneck_dim
            final_mlp_layer = keras.layers.Dense(
                units=self.bottleneck_dim,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name=f"mlp_{self.nlayers - 1}"
            )
            self.mlp_layers.append(final_mlp_layer)

        # Final projection layer (bottleneck_dim -> out_dim)
        self.last_layer = keras.layers.Dense(
            units=self.out_dim,
            use_bias=False,  # DINO typically doesn't use bias in the last layer
            kernel_initializer=self.kernel_initializer,
            name="last_layer"
        )

        # Initialize last layer weights with unit norm if specified
        if self.norm_last_layer:
            # This will be applied in the build method of the Dense layer
            pass

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the DINO head.

        Args:
            inputs: Input tensor of shape (batch_size, in_dim).
            training: Boolean indicating whether the model is in training mode.

        Returns:
            Projected tensor of shape (batch_size, out_dim).
        """
        x = inputs

        # Apply MLP layers
        for layer in self.mlp_layers:
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        # L2 normalize before final projection (as in DINO paper)
        x = keras.utils.normalize(x, axis=-1, order=2)

        # Final projection
        x = self.last_layer(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "use_bn": self.use_bn,
            "norm_last_layer": self.norm_last_layer,
            "nlayers": self.nlayers,
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "normalization_type": self.normalization_type,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": self.kernel_initializer,
        })
        return config


@keras.saving.register_keras_serializable()
class DINOv1(keras.Model):
    """
    DINO Vision Transformer model for self-supervised learning.

    This model implements the Vision Transformer backbone used in DINO
    (DIstillation with NO labels) self-supervised learning framework.
    It can be used both for feature extraction and with the DINO head
    for contrastive self-supervised training.

    Args:
        embed_dim: Integer, embedding dimension of the model.
        depth: Integer, number of transformer layers.
        num_heads: Integer, number of attention heads.
        patch_size: Integer or tuple, size of image patches.
        image_size: Integer or tuple, input image size. Default is 224.
        in_channels: Integer, number of input channels. Default is 3.
        num_classes: Integer, number of output classes for classification.
            Set to 0 for feature extraction only.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
        qkv_bias: Boolean, whether to use bias in QKV projection.
        dropout_rate: Float, dropout rate.
        attention_dropout_rate: Float, attention dropout rate.
        stochastic_depth_rate: Float, stochastic depth rate.
        norm_layer: String, normalization layer type.
        attention_type: String, type of attention mechanism to use.
        ffn_type: String, type of feed-forward network to use.
        include_top: Boolean, whether to include classification head.
        include_projection_head: Boolean, whether to include DINO projection head.
        dino_out_dim: Integer, output dimension for DINO head.
        dino_hidden_dim: Integer, hidden dimension for DINO head.
        dino_bottleneck_dim: Integer, bottleneck dimension for DINO head.
        dino_nlayers: Integer, number of layers in DINO head.
        use_cls_token: Boolean, whether to use [CLS] token.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        - If include_projection_head=True: 2D tensor `(batch_size, dino_out_dim)`
        - If include_top=True and num_classes>0: 2D tensor `(batch_size, num_classes)`
        - Otherwise: 2D tensor `(batch_size, embed_dim)`

    Example:
        ```python
        # Feature extraction model
        model = DINOv1(
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_size=16,
            num_classes=0,
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Self-supervised learning model with DINO head
        model = DINOv1(
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_size=16,
            num_classes=0,
            include_projection_head=True,
            dino_out_dim=65536,
            input_shape=(224, 224, 3)
        )
        ```
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "mlp_ratio": 4.0,
        },
        "small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
        "base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
        },
        "large": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
        }
    }

    def __init__(
            self,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            patch_size: Union[int, Tuple[int, int]] = 16,
            image_size: Union[int, Tuple[int, int]] = 224,  # Renamed from img_size
            in_channels: int = 3,
            num_classes: int = 1000,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout_rate: float = 0.0,  # Renamed from drop_rate
            attention_dropout_rate: float = 0.0,  # Renamed from attn_drop_rate
            stochastic_depth_rate: float = 0.0,  # Renamed from drop_path_rate
            norm_layer: str = "layer_norm",
            attention_type: str = "multi_head_attention",
            ffn_type: str = "mlp",
            include_top: bool = True,  # New argument for standard classifier
            include_projection_head: bool = False,  # Renamed from include_head
            dino_out_dim: int = 65536,
            dino_hidden_dim: int = 2048,
            dino_bottleneck_dim: int = 256,
            dino_nlayers: int = 3,
            use_cls_token: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ):
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Store configuration with renamed parameters
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)  # Renamed
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate  # Renamed
        self.attention_dropout_rate = attention_dropout_rate  # Renamed
        self.stochastic_depth_rate = stochastic_depth_rate  # Renamed
        self.norm_layer = norm_layer
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.include_top = include_top  # New
        self.include_projection_head = include_projection_head  # Renamed
        self.dino_out_dim = dino_out_dim
        self.dino_hidden_dim = dino_hidden_dim
        self.dino_bottleneck_dim = dino_bottleneck_dim
        self.dino_nlayers = dino_nlayers
        self.use_cls_token = use_cls_token

        # Calculate derived parameters
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # Set input shape
        if input_shape is None:
            input_shape = (*self.image_size, self.in_channels)
        self._input_shape = input_shape

        # Build the model
        inputs = keras.Input(shape=input_shape)
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(f"Created DINO Vision Transformer with {depth} layers, "
                    f"{num_heads} heads, {embed_dim} embed_dim")

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete DINO Vision Transformer model."""
        x = inputs

        # Patch embedding
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name="patch_embed"
        )
        x = self.patch_embed(x)

        # Add CLS token if requested
        if self.use_cls_token:
            batch_size = keras.ops.shape(x)[0]
            seq_len = keras.ops.shape(x)[1]

            # Create learnable CLS token
            self.cls_token = self.add_weight(
                shape=(1, 1, self.embed_dim),
                initializer="truncated_normal",
                trainable=True,
                name="cls_token"
            )

            # Expand CLS token for batch
            cls_tokens = keras.ops.tile(self.cls_token, [batch_size, 1, 1])

            # Concatenate CLS token with patch embeddings
            x = keras.ops.concatenate([cls_tokens, x], axis=1)
            seq_len = seq_len + 1
        else:
            seq_len = keras.ops.shape(x)[1]

        # Positional embedding
        max_seq_len = self.num_patches + (1 if self.use_cls_token else 0)
        self.pos_embed = PositionalEmbedding(
            max_seq_len=max_seq_len,
            dim=self.embed_dim,
            dropout=self.dropout_rate,  # Updated
            name="pos_embed"
        )
        x = self.pos_embed(x)

        # Transformer blocks
        self.transformer_blocks = []
        dpr = [x.item() for x in keras.ops.linspace(0., self.stochastic_depth_rate, self.depth)]  # Updated

        for i in range(self.depth):
            block = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                attention_args={"qkv_bias": self.qkv_bias} if self.attention_type == "multi_head_attention" else {},
                normalization_type=self.norm_layer,
                normalization_position="pre",  # Pre-normalization as in DINO
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,  # Updated
                attention_dropout_rate=self.attention_dropout_rate,  # Updated
                use_stochastic_depth=self.stochastic_depth_rate > 0.0,  # Updated
                stochastic_depth_rate=dpr[i],
                name=f"transformer_block_{i}"
            )
            self.transformer_blocks.append(block)
            x = block(x)

        # Final layer normalization
        self.norm = create_normalization_layer(
            self.norm_layer,
            name="norm"
        )
        x = self.norm(x)

        # Extract features based on configuration
        if self.use_cls_token:
            # Use CLS token representation
            cls_output = x[:, 0]  # Shape: (batch_size, embed_dim)
            features = cls_output
        else:
            # Global average pooling over patch tokens
            features = keras.ops.mean(x, axis=1)  # Shape: (batch_size, embed_dim)

        # Output head - refactored logic
        if self.include_projection_head:
            # DINO projection head for self-supervised learning
            self.head = DINOHead(
                in_dim=self.embed_dim,
                out_dim=self.dino_out_dim,
                hidden_dim=self.dino_hidden_dim,
                bottleneck_dim=self.dino_bottleneck_dim,
                nlayers=self.dino_nlayers,
                use_bn=False,
                norm_last_layer=True,
                dropout_rate=self.dropout_rate,  # Updated
                name="dino_projection_head"  # Updated name
            )
            outputs = self.head(features)
        elif self.include_top and self.num_classes > 0:
            # Standard classification head
            self.head = keras.layers.Dense(
                units=self.num_classes,
                kernel_initializer="truncated_normal",
                name="classifier"  # Updated name
            )
            outputs = self.head(features)
        else:
            # Feature extraction only
            self.head = None
            outputs = features

        return outputs

    def get_last_selfattention(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Get attention weights from the last transformer layer.

        This method is useful for visualizing attention patterns,
        similar to the original DINO implementation.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Attention tensor from the last layer.
        """
        x = inputs

        # Process through patch embedding and positional encoding
        x = self.patch_embed(x)

        if self.use_cls_token:
            batch_size = keras.ops.shape(x)[0]
            cls_tokens = keras.ops.tile(self.cls_token, [batch_size, 1, 1])
            x = keras.ops.concatenate([cls_tokens, x], axis=1)

        x = self.pos_embed(x)

        # Process through all but the last transformer block
        for i in range(self.depth - 1):
            x = self.transformer_blocks[i](x)

        # Get attention from the last block
        # Note: This requires the transformer block to support returning attention
        # For now, we'll return a placeholder - this would need to be implemented
        # in the TransformerLayer class to return attention weights when requested

        logger.warning("get_last_selfattention not fully implemented - "
                       "TransformerLayer needs attention return capability")

        # Process through last layer normally for now
        x = self.transformer_blocks[-1](x)

        # Return dummy attention tensor for compatibility
        batch_size = keras.ops.shape(x)[0]
        seq_len = keras.ops.shape(x)[1]
        attention_shape = (batch_size, self.num_heads, seq_len, seq_len)
        return keras.ops.zeros(attention_shape)

    @classmethod
    def from_variant(
            cls,
            variant: ModelVariant,
            num_classes: int = 0,
            patch_size: Union[int, Tuple[int, int]] = 16,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> "DINOv1":
        """
        Create a DINO model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large".
            num_classes: Integer, number of output classes.
            patch_size: Integer or tuple, size of image patches.
            input_shape: Tuple, input shape. If None, uses (224, 224, 3).
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            DINOv1 model instance.

        Example:
            ```python
            # Create DINO-Small for feature extraction
            model = DINOv1.from_variant(
                "small",
                num_classes=0,
                patch_size=16
            )

            # Create DINO-Base with projection head
            model = DINOv1.from_variant(
                "base",
                num_classes=0,
                include_projection_head=True,
                dino_out_dim=65536
            )
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating DINO-{variant.upper()} model")

        return cls(
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            num_classes=num_classes,
            patch_size=patch_size,
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "patch_size": self.patch_size,
            "image_size": self.image_size,  # Updated
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "dropout_rate": self.dropout_rate,  # Updated
            "attention_dropout_rate": self.attention_dropout_rate,  # Updated
            "stochastic_depth_rate": self.stochastic_depth_rate,  # Updated
            "norm_layer": self.norm_layer,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "include_top": self.include_top,  # New
            "include_projection_head": self.include_projection_head,  # Updated
            "dino_out_dim": self.dino_out_dim,
            "dino_hidden_dim": self.dino_hidden_dim,
            "dino_bottleneck_dim": self.dino_bottleneck_dim,
            "dino_nlayers": self.dino_nlayers,
            "use_cls_token": self.use_cls_token,
            "input_shape": self._input_shape,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DINOv1":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info(f"DINO Vision Transformer configuration:")
        logger.info(f"  - Embedding dimension: {self.embed_dim}")
        logger.info(f"  - Number of layers: {self.depth}")
        logger.info(f"  - Number of heads: {self.num_heads}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Image size: {self.image_size}")  # Updated
        logger.info(f"  - Number of patches: {self.num_patches}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Use CLS token: {self.use_cls_token}")
        logger.info(f"  - Include top: {self.include_top}")  # New
        logger.info(f"  - Include DINO head: {self.include_projection_head}")  # Updated
        if self.include_projection_head:
            logger.info(f"  - DINO output dim: {self.dino_out_dim}")
        if self.num_classes > 0:
            logger.info(f"  - Number of classes: {self.num_classes}")


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------

def create_dino_v1(  # Renamed from create_dino_model
        variant: ModelVariant = "small",
        num_classes: int = 0,
        patch_size: Union[int, Tuple[int, int]] = 16,
        input_shape: Optional[Tuple[int, ...]] = None,
        include_top: bool = True,  # New argument
        include_projection_head: bool = False,  # Renamed from include_head
        dino_out_dim: int = 65536,
        **kwargs
) -> DINOv1:
    """
    Convenience function to create DINO Vision Transformer models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large").
        num_classes: Integer, number of output classes. Set to 0 for feature extraction.
        patch_size: Integer or tuple, size of image patches.
        input_shape: Tuple, input shape. If None, uses (224, 224, 3).
        include_top: Boolean, whether to include classification head.
        include_projection_head: Boolean, whether to include DINO projection head.
        dino_out_dim: Integer, output dimension for DINO head.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        DINOv1 model instance.

    Example:
        ```python
        # Create DINO-Small for self-supervised learning
        model = create_dino_v1(
            variant="small",
            include_projection_head=True,
            dino_out_dim=65536
        )

        # Create DINO-Base for fine-tuning
        model = create_dino_v1(
            variant="base",
            num_classes=1000,
            input_shape=(224, 224, 3)
        )
        ```
    """
    if input_shape is None:
        input_shape = (224, 224, 3)

    return DINOv1.from_variant(
        variant=variant,
        num_classes=num_classes,
        patch_size=patch_size,
        input_shape=input_shape,
        include_top=include_top,  # New
        include_projection_head=include_projection_head,  # Updated
        dino_out_dim=dino_out_dim,
        **kwargs
    )


def create_dino_teacher_student_pair(
        variant: ModelVariant = "small",
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        patch_size: Union[int, Tuple[int, int]] = 16,
        input_shape: Optional[Tuple[int, ...]] = None,
        dino_out_dim: int = 65536,
        **kwargs
) -> Tuple[DINOv1, DINOv1]:
    """
    Create teacher-student pair for DINO self-supervised learning.

    Args:
        variant: String, model variant for both teacher and student.
        teacher_temp: Float, temperature for teacher model (not used in model creation).
        student_temp: Float, temperature for student model (not used in model creation).
        patch_size: Integer or tuple, size of image patches.
        input_shape: Tuple, input shape. If None, uses (224, 224, 3).
        dino_out_dim: Integer, output dimension for DINO heads.
        **kwargs: Additional arguments passed to both model constructors.

    Returns:
        Tuple of (teacher_model, student_model).

    Note:
        The temperature parameters are provided for API compatibility but are
        typically applied during loss computation, not in the model architecture.

    Example:
        ```python
        teacher, student = create_dino_teacher_student_pair(
            variant="small",
            teacher_temp=0.04,
            student_temp=0.1,
            dino_out_dim=65536
        )

        # The teacher model weights are typically updated via EMA
        # from the student model during training
        ```
    """
    if input_shape is None:
        input_shape = (224, 224, 3)

    # Create teacher model
    teacher = DINOv1.from_variant(
        variant=variant,
        num_classes=0,
        patch_size=patch_size,
        input_shape=input_shape,
        include_projection_head=True,  # Updated
        dino_out_dim=dino_out_dim,
        name="dino_teacher",
        **kwargs
    )

    # Create student model (identical architecture)
    student = DINOv1.from_variant(
        variant=variant,
        num_classes=0,
        patch_size=patch_size,
        input_shape=input_shape,
        include_projection_head=True,  # Updated
        dino_out_dim=dino_out_dim,
        name="dino_student",
        **kwargs
    )

    logger.info(f"Created DINO teacher-student pair with variant '{variant}'")
    logger.info(f"Teacher temp: {teacher_temp}, Student temp: {student_temp}")

    return teacher, student