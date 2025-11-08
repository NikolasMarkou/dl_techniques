"""
DINOv3 Model Implementation

This module provides a refactored implementation of a Vision Transformer,
named DINOv3 for consistency, that aligns with the architecture of the original
DINO (self-DIstillation with NO labels) model and the coding style of the
provided DINOv1 and DINOv2 implementations.

The architecture is a standard Vision Transformer (ViT) with pre-normalization
transformer blocks, which is crucial for training stability in self-supervised
settings like DINO. This implementation re-uses existing, shared layers from the
`dl_techniques` library for better modularity and consistency.

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
# Create DinoV3-Base for ImageNet classification (224x224 input)
model = create_dino_v3("base", num_classes=1000)

# Create DinoV3-Small for CIFAR-10 (32x32 input)
model = create_dino_v3(
    "small",
    image_size=(32, 32),
    patch_size=(4, 4), # Smaller patch size for smaller images
    num_classes=10
)

# Create a feature extraction model (no classification head)
model = create_dino_v3("large", include_top=False)

# Create model for self-supervised pre-training using a DINO-style head
# This can be achieved by attaching a DINOHead layer from dino_v1.py
from dino_v1 import DINOHead
features = create_dino_v3("base", include_top=False).output
projection = DINOHead(in_dim=768, out_dim=65536)(features)
ssl_model = keras.Model(inputs=features.node.inputs, outputs=projection)
```
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DINOv3(keras.Model):
    """
    DINOv3 Vision Transformer Model Implementation.

    This class implements a standard Vision Transformer architecture, following the
    design of the original DINO paper which uses pre-normalization layers for
    improved training stability. It is designed to be a backbone for various
    downstream tasks, including classification and self-supervised learning.

    The model consists of:
    - A patch embedding layer to convert images into sequences of tokens.
    - A learnable [CLS] token for global image representation.
    - Learned positional embeddings to encode spatial information.
    - A stack of pre-normalized Transformer encoder layers.
    - A final normalization layer.
    - An optional classification head.

    Args:
        image_size: Tuple of integers (height, width) for the input image.
            Defaults to (224, 224).
        patch_size: Tuple of integers (height, width) for the image patches.
            Defaults to (16, 16).
        num_classes: Number of output classes for the classification head. If 0,
            no head is added. Defaults to 1000.
        embed_dim: The dimensionality of the token embeddings. Defaults to 768.
        depth: The number of transformer encoder layers. Defaults to 12.
        num_heads: The number of attention heads in each transformer layer.
            Defaults to 12.
        mlp_ratio: Ratio to determine the hidden dimension of the FFN in
            transformer layers (hidden_dim = embed_dim * mlp_ratio). Defaults to 4.0.
        qkv_bias: If True, add a learnable bias to the query, key, and value
            projections. Defaults to True.
        dropout_rate: Dropout rate for the embedding and FFN layers.
            Defaults to 0.0.
        attention_dropout_rate: Dropout rate for the attention weights.
            Defaults to 0.0.
        stochastic_depth_rate: Maximum drop rate for stochastic depth, which
            linearly increases across layers. Defaults to 0.0.
        normalization_type: The type of normalization to use ('layer_norm',
            'rms_norm'). Defaults to 'layer_norm'.
        activation: Activation function for the FFN layers. Defaults to 'gelu'.
        kernel_initializer: Initializer for kernel weights. Defaults to
            'glorot_uniform'.
        bias_initializer: Initializer for bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        include_top: If True, include the final classification head. If False,
            the model outputs features from the transformer. Defaults to True.
        **kwargs: Additional arguments for the `keras.Model` base class.

    Input shape:
        A 4D tensor of shape `(batch_size, height, width, channels)`, where
        height and width must match `image_size`.

    Output shape:
        - If `include_top=True`: A 2D tensor of shape `(batch_size, num_classes)`.
        - If `include_top=False`: A 2D tensor of shape `(batch_size, embed_dim)`
          representing the [CLS] token features.

    Raises:
        ValueError: If model parameters are invalid or incompatible.
    """

    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "small": {
            "embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "base": {
            "embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "large": {
            "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "giant": {
            "embed_dim": 1536, "depth": 40, "num_heads": 24, "mlp_ratio": 4.0,
            "patch_size": (14, 14), "stochastic_depth_rate": 0.4
        }
    }

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        normalization_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        activation: Union[str, Callable] = 'gelu',
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        include_top: bool = True,
        **kwargs: Any
    ) -> None:
        # Input validation
        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        super().__init__(**kwargs)

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.normalization_type = normalization_type
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.include_top = include_top

        # Compute derived values
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.sequence_length = self.num_patches + 1

        # Build the model using the functional API pattern
        inputs = keras.Input(shape=(*image_size, 3), name="input_image")
        outputs = self._build_model(inputs)

        # Finalize the Model
        super().__init__(inputs=inputs, outputs=outputs, name="DINOv3", **kwargs)

        logger.info(
            f"Created DINOv3 model with {depth} layers, {embed_dim} embedding dim for "
            f"input shape {image_size}"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Constructs the model architecture."""
        # 1. Patch Embedding
        x = self._build_patch_embedding(inputs)

        # 2. Add CLS Token and Positional Embedding
        x = self._build_token_processing(x)

        # 3. Transformer Encoder Layers
        x = self._build_encoder(x)

        # 4. Final Processing and Head
        x = self._build_head(x)

        return x

    def _build_patch_embedding(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the patch embedding layer."""
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='patch_embedding'
        )
        return self.patch_embed(inputs)

    def _build_token_processing(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Adds the [CLS] token and positional embeddings."""
        batch_size = ops.shape(x)[0]

        # Add CLS token
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer="truncated_normal",
            trainable=True,
            name="cls_token"
        )
        cls_tokens = ops.tile(self.cls_token, [batch_size, 1, 1])
        x = ops.concatenate([cls_tokens, x], axis=1)

        # Add positional embedding using the shared layer
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.sequence_length,
            dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            name="positional_embedding"
        )
        return self.pos_embed(x)

    def _build_encoder(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the stack of transformer encoder layers."""
        self.encoder_layers = []
        dpr = [r.item() for r in ops.linspace(0., self.stochastic_depth_rate, self.depth)]

        for i in range(self.depth):
            encoder_layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=int(self.embed_dim * self.mlp_ratio),
                attention_type="multi_head",
                attention_args={"use_bias": self.qkv_bias},
                normalization_type=self.normalization_type,
                normalization_position='pre',  # DINO uses pre-norm
                ffn_type='mlp',
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=dpr[i] > 0.0,
                stochastic_depth_rate=dpr[i],
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

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the final normalization and classification head."""
        # Final normalization using the shared factory
        self.norm = create_normalization_layer(
            self.normalization_type,
            name='final_norm'
        )
        x = self.norm(x)

        # Extract [CLS] token for classification
        features = x[:, 0]

        # Add classification head if requested
        if self.include_top:
            if self.num_classes > 0:
                self.classifier = layers.Dense(
                    units=self.num_classes,
                    kernel_initializer="truncated_normal",
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name='classifier'
                )
                outputs = self.classifier(features)
            else:
                # If include_top is True but num_classes is 0, return features
                outputs = features
        else:
            # If not including top, return features
            outputs = features

        return outputs

    def get_last_selfattention(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Extracts attention weights from the last transformer layer's [CLS] token.
        Useful for visualization as shown in the DINO paper.

        Note: This requires the `TransformerLayer` to have a mechanism to return
        attention scores, which may not be implemented. A warning is issued.

        Args:
            inputs: A batch of images.

        Returns:
            A tensor of attention weights from the last layer.
        """
        logger.warning(
            "get_last_selfattention() requires TransformerLayer to support "
            "returning attention weights. This may not be fully implemented."
        )
        x = self.patch_embed(inputs)
        x = self._build_token_processing(x)

        # Forward through all but the last transformer block
        for i in range(self.depth - 1):
            x = self.encoder_layers[i](x)

        # In a full implementation, the last layer would be called with a flag
        # to return attention. For now, we return a placeholder.
        # x, attention = self.encoder_layers[-1](x, return_attention=True)
        x = self.encoder_layers[-1](x)
        
        # Placeholder for attention weights
        batch_size = ops.shape(x)[0]
        attention_shape = (batch_size, self.num_heads, self.sequence_length, self.sequence_length)
        return ops.zeros(attention_shape)


    @classmethod
    def from_variant(
        cls,
        variant: str,
        image_size: Tuple[int, int] = (224, 224),
        num_classes: int = 1000,
        include_top: bool = True,
        **kwargs: Any
    ) -> "DINOv3":
        """
        Creates a DINOv3 model from a predefined variant.

        Args:
            variant: The model variant, one of "tiny", "small", "base", "large", "giant".
            image_size: The input image size (height, width).
            num_classes: Number of output classes.
            include_top: Whether to include the classification head.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            A DINOv3 model instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating DINOv3-{variant.upper()} model with config: {config}")

        return cls(
            image_size=image_size,
            num_classes=num_classes,
            include_top=include_top,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        config = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'normalization_type': self.normalization_type,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'include_top': self.include_top,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DINOv3":
        """Creates a model from its configuration."""
        return cls(**config)


def create_dino_v3(
    variant: str = "base",
    image_size: Tuple[int, int] = (224, 224),
    num_classes: int = 1000,
    include_top: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> DINOv3:
    """
    A factory function to create DINOv3 models.

    Args:
        variant: Model variant ("tiny", "small", "base", "large", "giant").
        image_size: Input image size (height, width).
        num_classes: Number of output classes.
        include_top: Whether to include the final classification layer.
        pretrained: If True, attempts to load pretrained weights (not implemented).
        **kwargs: Additional arguments for the model constructor.

    Returns:
        A DINOv3 model instance.
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented for DINOv3.")

    return DINOv3.from_variant(
        variant=variant,
        image_size=image_size,
        num_classes=num_classes,
        include_top=include_top,
        **kwargs
    )
