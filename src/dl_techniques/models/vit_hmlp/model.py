"""
Vision Transformer with Hierarchical MLP Stem - Modern Implementation

This module provides a Vision Transformer model with Hierarchical MLP (hMLP) stem,
designed for compatibility with the dl-techniques framework. The implementation follows
modern Keras 3 lifecycle patterns and leverages the framework's factory system for
consistent component creation.

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
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.hierarchical_mlp_stem import HierarchicalMLPStem
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

VitScale = Literal['tiny', 'small', 'base', 'large', 'huge']
PoolingMode = Literal['cls', 'mean', 'max']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
StemNormLayer = Literal['batch', 'layer']


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ViTHMLP(keras.Model):
    """
    Vision Transformer with Hierarchical MLP Stem using factory-based component creation.

    This model implements the complete Vision Transformer architecture with a hierarchical MLP stem
    that processes patches progressively through multiple scales. The implementation leverages the
    dl-techniques framework's factory system for consistent component creation and follows modern
    Keras 3 best practices for robust serialization and deployment.

    **Intent**: Provide a production-ready Vision Transformer implementation with hMLP stem that
    leverages the dl-techniques framework's modular components while following modern Keras 3 best
    practices for robust serialization and deployment. The hMLP stem provides superior performance
    for both supervised learning and masked self-supervised learning approaches.

    **Architecture**:
    ```
    Input Images (batch, height, width, channels)
           ↓
    HierarchicalMLPStem → Patches (batch, num_patches, embed_dim)
      ↓ 2×2 → 4×4 → 8×8 → 16×16 pixel processing
           ↓
    Add CLS Token → (batch, seq_len, embed_dim)
           ↓
    PositionalEmbedding + Dropout
           ↓
    TransformerLayer × num_layers
           ↓
    Final Normalization
           ↓
    [Classification Head] OR [Feature Extraction]
           ↓
    Output (shape depends on configuration)
    ```

    **hMLP Stem Processing**:
    1. **Progressive Resolution**: Patches processed at 2×2, 4×4, 8×8, 16×16 pixel scales
    2. **Independent Processing**: No cross-patch information leakage (SSL compatible)
    3. **Hierarchical Features**: Each scale contributes to final patch representation
    4. **Efficient Implementation**: <1% computational overhead vs standard linear projection

    **Scale Configurations**:
    - **Tiny**: 192d, 3h, 12L - Efficient for small datasets/mobile deployment
    - **Small**: 384d, 6h, 12L - Balanced performance and efficiency
    - **Base**: 768d, 12h, 12L - Standard configuration (original paper)
    - **Large**: 1024d, 16h, 24L - High performance for large datasets
    - **Huge**: 1280d, 16h, 32L - Maximum capacity for demanding tasks

    Args:
        input_shape: Tuple[int, int, int], input image shape (height, width, channels).
            Must have positive dimensions and be compatible with patch_size.
            Example: (224, 224, 3) for ImageNet.
        num_classes: Integer, number of output classes for classification.
            Must be positive. Only used when include_top=True.
        scale: VitScale, model scale configuration determining architecture size.
            Available: 'tiny', 'small', 'base', 'large', 'huge'. Defaults to 'base'.
        patch_size: Union[int, Tuple[int, int]], size of patches to extract from images.
            If int, uses square patches. Image dimensions must be divisible by patch size.
            Defaults to 16.
        include_top: Boolean, whether to include classification head.
            When False, model acts as feature extractor. Defaults to True.
        pooling: Optional[PoolingMode], pooling strategy for feature extraction.
            Only used when include_top=False:
            - 'cls': Use CLS token representation
            - 'mean': Global average pooling over sequence
            - 'max': Global max pooling over sequence
            - None: Return full sequence (batch, seq_len, embed_dim)
            Defaults to None.
        dropout_rate: Float, dropout rate for general regularization.
            Applied in transformer layers and classification head. Defaults to 0.0.
        attention_dropout_rate: Float, dropout rate for attention weights.
            Applied within attention mechanisms. Defaults to 0.0.
        pos_dropout_rate: Float, dropout rate after positional embeddings.
            Defaults to 0.0.
        stem_norm_layer: StemNormLayer, normalization type for hMLP stem.
            Available options:
            - 'batch': Batch normalization (better performance, default)
            - 'layer': Layer normalization (stable for small batches)
            Defaults to 'batch'.
        kernel_initializer: Union[str, Initializer], weight initializer for all layers.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional[Regularizer], weight regularizer for all layers.
            Defaults to None.
        bias_initializer: Union[str, Initializer], bias initializer for all layers.
            Defaults to 'zeros'.
        bias_regularizer: Optional[Regularizer], bias regularizer for all layers.
            Defaults to None.
        normalization_type: NormalizationType, normalization layer type.
            Uses factory for consistent creation. Available options:
            - 'layer_norm': Standard layer normalization (default)
            - 'rms_norm': Root Mean Square normalization
            - 'band_rms': Band-constrained RMS normalization
            - 'dynamic_tanh': Dynamic Tanh normalization
            Defaults to 'layer_norm'.
        normalization_position: Literal['pre', 'post'], normalization position in transformer.
            - 'post': Post-normalization (original Transformer)
            - 'pre': Pre-normalization (often more stable)
            Defaults to 'pre'.
        ffn_type: FFNType, feed-forward network type for transformer layers.
            Uses factory for consistent creation. Available options:
            - 'mlp': Standard MLP with intermediate expansion (default)
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'geglu': GELU-based Gated Linear Unit
            Defaults to 'mlp'.
        activation: Union[str, Callable], activation function for FFN.
            Defaults to 'gelu'.
        use_stochastic_depth: Boolean, whether to use stochastic depth regularization.
            Provides regularization for deep networks. Defaults to False.
        stochastic_depth_rate: Float, maximum drop path rate for stochastic depth.
            Only used when use_stochastic_depth=True. Defaults to 0.1.
        name: Optional[str], model name. Auto-generated if None.
        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

        Requirements:
        - height and width must be divisible by corresponding patch dimensions
        - All dimensions must be positive
        - Channels typically 1 (grayscale) or 3 (RGB)

    Output shape:
        Depends on configuration:

        **Classification mode** (include_top=True):
        - Shape: `(batch_size, num_classes)`
        - Values: Logits for each class (no softmax applied)

        **Feature extraction mode** (include_top=False):
        - pooling='cls': `(batch_size, embed_dim)` - CLS token features
        - pooling='mean': `(batch_size, embed_dim)` - Mean-pooled features
        - pooling='max': `(batch_size, embed_dim)` - Max-pooled features
        - pooling=None: `(batch_size, seq_len, embed_dim)` - Full sequence

    Attributes:
        embed_dim: Integer, embedding dimension determined by scale.
        num_heads: Integer, number of attention heads determined by scale.
        num_layers: Integer, number of transformer layers determined by scale.
        num_patches: Integer, total number of image patches.
        max_seq_len: Integer, maximum sequence length (num_patches + 1 for CLS).
        stem: HierarchicalMLPStem layer for progressive patch processing.
        pos_embed: PositionalEmbedding layer for sequence position encoding.
        transformer_layers: List of TransformerLayer instances.
        norm: Final normalization layer.
        head: Optional Dense layer for classification.

    Example:
        ```python
        # Standard ViT-Base with hMLP stem for ImageNet classification
        model = ViTHMLP(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base'
        )

        # Feature extractor with CLS token and modern components
        feature_model = ViTHMLP(
            input_shape=(224, 224, 3),
            scale='base',
            include_top=False,
            pooling='cls',
            normalization_type='rms_norm',
            ffn_type='swiglu'
        )

        # Model optimized for self-supervised learning
        ssl_model = ViTHMLP(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base',
            stem_norm_layer='batch',  # Better performance
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1,
            dropout_rate=0.0  # No dropout for SSL pre-training
        )

        # Compile for training
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```

    Note:
        This implementation follows modern Keras 3 patterns with proper serialization
        support. The hMLP stem provides superior performance for both supervised learning
        and masked self-supervised learning compared to standard patch embedding.
        All sub-components are created using dl_techniques factories for consistency
        and configurability.
    """

    # Scale configurations: [embed_dim, num_heads, num_layers, mlp_ratio]
    SCALE_CONFIGS: Dict[str, Tuple[int, int, int, float]] = {
        "tiny": (192, 3, 12, 4.0),  # ViT-Tiny
        "small": (384, 6, 12, 4.0),  # ViT-Small
        "base": (768, 12, 12, 4.0),  # ViT-Base
        "large": (1024, 16, 24, 4.0),  # ViT-Large
        "huge": (1280, 16, 32, 4.0),  # ViT-Huge
    }

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            num_classes: int = 1000,
            scale: VitScale = "base",
            patch_size: Union[int, Tuple[int, int]] = 16,
            include_top: bool = True,
            pooling: Optional[PoolingMode] = None,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            pos_dropout_rate: float = 0.0,
            stem_norm_layer: StemNormLayer = "batch",
            kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            normalization_type: NormalizationType = "layer_norm",
            normalization_position: Literal['pre', 'post'] = "pre",
            ffn_type: FFNType = "mlp",
            activation: Union[str, callable] = "gelu",
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize Vision Transformer model with Hierarchical MLP stem."""
        # Auto-generate name if not provided
        if name is None:
            name = f"vision_transformer_hmlp_{scale}"

        super().__init__(name=name, **kwargs)

        # Validate and store input_shape
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
            raise ValueError(f"input_shape must be a 3-tuple (height, width, channels), got {input_shape}")

        img_h, img_w, img_c = input_shape
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")

        # Validate and normalize patch_size
        if isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {patch_size}")
            patch_h = patch_w = patch_size
        else:
            if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
                raise ValueError(f"patch_size must be int or tuple of 2 ints, got {patch_size}")
            patch_h, patch_w = patch_size
            if patch_h <= 0 or patch_w <= 0:
                raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")

        # Validate divisibility for patch extraction
        if img_h % patch_h != 0:
            raise ValueError(f"Image height ({img_h}) must be divisible by patch height ({patch_h})")
        if img_w % patch_w != 0:
            raise ValueError(f"Image width ({img_w}) must be divisible by patch width ({patch_w})")

        # Validate other parameters
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        if scale not in self.SCALE_CONFIGS:
            raise ValueError(f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}")

        if pooling not in [None, "cls", "mean", "max"]:
            raise ValueError(f"Unsupported pooling: {pooling}. Choose from [None, 'cls', 'mean', 'max']")

        if stem_norm_layer not in ["batch", "layer"]:
            raise ValueError(f"Unsupported stem_norm_layer: {stem_norm_layer}. Choose from ['batch', 'layer']")

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(f"pos_dropout_rate must be between 0 and 1, got {pos_dropout_rate}")

        if not (0.0 <= stochastic_depth_rate <= 1.0):
            raise ValueError(f"stochastic_depth_rate must be between 0 and 1, got {stochastic_depth_rate}")

        # Store ALL configuration parameters for serialization
        self.input_shape_config = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.scale = str(scale)
        self.patch_size = (patch_h, patch_w)
        self.include_top = bool(include_top)
        self.pooling = pooling
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.pos_dropout_rate = float(pos_dropout_rate)
        self.stem_norm_layer = str(stem_norm_layer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = bias_regularizer
        self.normalization_type = str(normalization_type)
        self.normalization_position = str(normalization_position)
        self.ffn_type = str(ffn_type)
        self.activation = activation
        self.use_stochastic_depth = bool(use_stochastic_depth)
        self.stochastic_depth_rate = float(stochastic_depth_rate)

        # Get model configuration from scale
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate derived parameters
        self.intermediate_size = int(self.embed_dim * self.mlp_ratio)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate derived parameters
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Using factories for consistent component creation where available

        # Hierarchical MLP Stem (specialized component, direct instantiation)
        self.stem = HierarchicalMLPStem(
            embed_dim=self.embed_dim,
            img_size=self.input_shape_config[:2],
            patch_size=self.patch_size,
            in_channels=img_c,
            norm_layer=self.stem_norm_layer,
            name="hierarchical_mlp_stem"
        )

        # Positional embedding using factory
        self.pos_embed = create_embedding_layer(
            'positional_learned',
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout=self.pos_dropout_rate,
            name="pos_embed"
        )

        # Transformer layers using existing TransformerLayer
        self.transformer_layers = []
        for i in range(self.num_layers):
            # Calculate stochastic depth rate for this layer (linear scaling)
            layer_drop_rate = (
                self.stochastic_depth_rate * i / max(1, self.num_layers - 1)
                if self.use_stochastic_depth
                else 0.0
            )

            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type="multi_head",
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=layer_drop_rate,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Final normalization using factory
        self.norm = create_normalization_layer(
            self.normalization_type,
            name="norm"
        )

        # Classification components (if include_top)
        self.head_dropout = None
        self.head = None
        if self.include_top:
            if self.dropout_rate > 0.0:
                self.head_dropout = layers.Dropout(self.dropout_rate, name="head_dropout")

            self.head = layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="head"
            )

        # Global pooling layers (if needed for feature extraction)
        self.global_pool = None
        if self.pooling == "mean":
            self.global_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")
        elif self.pooling == "max":
            self.global_pool = layers.GlobalMaxPooling1D(name="global_max_pool")

        # CLS token weight (created in build())
        self.cls_token = None

        logger.info(f"Created VisionTransformer-hMLP-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")
        logger.info(
            f"Image shape: {self.input_shape_config}, Patch size: {self.patch_size}, Num patches: {self.num_patches}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        # Create CLS token weight
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        # Build all sub-layers in computational order
        # Hierarchical MLP Stem
        dummy_input_shape = (None,) + self.input_shape_config
        self.stem.build(dummy_input_shape)

        # Positional embedding
        pos_input_shape = (None, self.max_seq_len, self.embed_dim)
        self.pos_embed.build(pos_input_shape)

        # Transformer layers
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)

        # Final normalization
        self.norm.build(pos_input_shape)

        # Classification head components
        if self.include_top:
            head_input_shape = (None, self.embed_dim)
            if self.head_dropout is not None:
                self.head_dropout.build(head_input_shape)
            self.head.build(head_input_shape)

        # Global pooling
        if self.global_pool is not None:
            self.global_pool.build(pos_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(f"Built VisionTransformer-hMLP-{self.scale} with {self.num_patches} patches")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Vision Transformer with hMLP stem.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Model output tensor. Shape depends on include_top and pooling settings.
        """
        # Process patches with hierarchical MLP stem
        x = self.stem(inputs, training=training)  # (batch_size, num_patches, embed_dim)

        # Add CLS token to sequence
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)  # (batch_size, seq_len, embed_dim)

        # Add positional embeddings (includes dropout)
        x = self.pos_embed(x, training=training)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Apply final normalization
        x = self.norm(x, training=training)

        # Handle different output modes
        if self.include_top:
            # Extract CLS token for classification
            cls_token = x[:, 0, :]  # (batch_size, embed_dim)
            if self.head_dropout is not None:
                cls_token = self.head_dropout(cls_token, training=training)
            x = self.head(cls_token)  # (batch_size, num_classes)
            return x
        else:
            # Feature extraction mode
            if self.pooling == "cls":
                # Return CLS token representation
                return x[:, 0, :]  # (batch_size, embed_dim)
            elif self.pooling == "mean":
                # Global average pooling over sequence
                return self.global_pool(x)  # (batch_size, embed_dim)
            elif self.pooling == "max":
                # Global max pooling over sequence
                return self.global_pool(x)  # (batch_size, embed_dim)
            else:
                # Return full transformer output
                return x  # (batch_size, seq_len, embed_dim)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output shape tuple.
        """
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        batch_size = input_shape[0]

        if self.include_top:
            return (batch_size, self.num_classes)
        else:
            if self.pooling in ["cls", "mean", "max"]:
                return (batch_size, self.embed_dim)
            else:
                return (batch_size, self.max_seq_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        CRITICAL: Must include ALL __init__ parameters for proper serialization.
        """
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes,
            "scale": self.scale,
            "patch_size": self.patch_size,
            "include_top": self.include_top,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "pos_dropout_rate": self.pos_dropout_rate,
            "stem_norm_layer": self.stem_norm_layer,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            "activation": self.activation,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

    def get_feature_extractor(self) -> "ViTHMLP":
        """
        Get a feature extractor version of this model.

        Returns:
            New ViTHMLP instance configured for feature extraction.
        """
        if not hasattr(self, 'input_shape_config') or not self.input_shape_config:
            raise ValueError("Model must be properly initialized before creating feature extractor")

        return ViTHMLP(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            pos_dropout_rate=self.pos_dropout_rate,
            stem_norm_layer=self.stem_norm_layer,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            normalization_type=self.normalization_type,
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            activation=self.activation,
            use_stochastic_depth=self.use_stochastic_depth,
            stochastic_depth_rate=self.stochastic_depth_rate,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Print detailed model summary with architecture information."""
        logger.info("Vision Transformer with hMLP Stem Model Summary")
        logger.info(f"Scale: {self.scale}")
        logger.info(f"Input Shape: {self.input_shape_config}")
        logger.info(f"Patch Size: {self.patch_size}")
        logger.info(f"Number of Patches: {self.num_patches}")
        logger.info(f"Sequence Length: {self.max_seq_len}")
        logger.info(f"Embedding Dimension: {self.embed_dim}")
        logger.info(f"Number of Heads: {self.num_heads}")
        logger.info(f"Number of Layers: {self.num_layers}")
        logger.info(f"MLP Ratio: {self.mlp_ratio}")
        logger.info(f"Intermediate Size: {self.intermediate_size}")
        logger.info(f"Dropout Rate: {self.dropout_rate}")
        logger.info(f"Attention Dropout Rate: {self.attention_dropout_rate}")
        logger.info(f"Positional Dropout Rate: {self.pos_dropout_rate}")
        logger.info(f"Stem Normalization: {self.stem_norm_layer}")
        logger.info(f"Transformer Normalization Type: {self.normalization_type}")
        logger.info(f"Normalization Position: {self.normalization_position}")
        logger.info(f"FFN Type: {self.ffn_type}")
        logger.info(f"Activation: {self.activation}")
        logger.info(f"Use Stochastic Depth: {self.use_stochastic_depth}")
        logger.info(f"Stochastic Depth Rate: {self.stochastic_depth_rate}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self.built:
            logger.info(f"Total Parameters: {self.count_params():,}")

        # Additional architecture information
        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        logger.info(f"Patches per dimension: {img_h // patch_h} x {img_w // patch_w}")
        logger.info("hMLP Stem Processing: 2×2 → 4×4 → 8×8 → 16×16 pixels")


# ---------------------------------------------------------------------
# Factory Functions for Convenient Model Creation
# ---------------------------------------------------------------------


def create_vit_hmlp(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 1000,
        scale: VitScale = "base",
        patch_size: Union[int, Tuple[int, int]] = 16,
        include_top: bool = True,
        pooling: Optional[PoolingMode] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        pos_dropout_rate: float = 0.0,
        stem_norm_layer: StemNormLayer = "batch",
        kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: Literal['pre', 'post'] = "pre",
        ffn_type: FFNType = "mlp",
        activation: Union[str, callable] = "gelu",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any
) -> ViTHMLP:
    """
    Create a Vision Transformer with Hierarchical MLP stem.

    This factory function provides parameter validation and sensible defaults
    for creating Vision Transformer models with hMLP stem. The hMLP stem provides
    superior performance for both supervised learning and masked self-supervised
    learning compared to standard patch embedding.

    Args:
        input_shape: Input image shape (height, width, channels).
            Must have positive dimensions and be compatible with patch_size.
        num_classes: Number of output classes for classification.
            Must be positive. Only used when include_top=True.
        scale: Model scale determining architecture size.
            Available: 'tiny', 'small', 'base', 'large', 'huge'.
        patch_size: Size of patches to extract from input images.
            If int, uses square patches. Image dimensions must be divisible by patch size.
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction when include_top=False.
            Available: 'cls', 'mean', 'max', None.
        dropout_rate: Dropout rate for general regularization.
        attention_dropout_rate: Dropout rate for attention weights.
        pos_dropout_rate: Dropout rate for positional embeddings.
        stem_norm_layer: Normalization type for hMLP stem.
            'batch' provides better performance, 'layer' is stable for small batches.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        bias_initializer: Bias initializer for all layers.
        bias_regularizer: Bias regularizer for all layers.
        normalization_type: Type of normalization layer to use.
        normalization_position: Position of normalization in transformer layers.
        ffn_type: Type of feed-forward network for transformer layers.
        activation: Activation function for feed-forward networks.
        use_stochastic_depth: Whether to use stochastic depth regularization.
        stochastic_depth_rate: Maximum drop path rate for stochastic depth.
        **kwargs: Additional arguments for ViTHMLP constructor.

    Returns:
        ViTHMLP model instance.

    Raises:
        ValueError: If any parameter validation fails.

    Example:
        ```python
        # Create ViT-Base with hMLP stem for ImageNet
        model = create_vit_hmlp(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base',
            stem_norm_layer='batch'  # Better performance
        )

        # Create feature extractor with modern components
        feature_model = create_vit_hmlp(
            input_shape=(384, 384, 3),
            scale='small',
            include_top=False,
            pooling='cls',
            normalization_type='rms_norm',
            ffn_type='swiglu'
        )

        # Model optimized for self-supervised learning
        ssl_model = create_vit_hmlp(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base',
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1,
            dropout_rate=0.0,  # No dropout for SSL pre-training
            stem_norm_layer='batch'
        )
        ```
    """
    # Validate basic parameters before model creation
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3-element tuple/list, got {input_shape}")

    if any(dim <= 0 for dim in input_shape):
        raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")

    # Validate patch_size and ensure compatibility with input_shape
    if isinstance(patch_size, int):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        patch_h = patch_w = patch_size
    else:
        if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
            raise ValueError(f"patch_size must be int or 2-element tuple/list, got {patch_size}")
        patch_h, patch_w = patch_size
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")

    img_h, img_w = input_shape[:2]
    if img_h % patch_h != 0:
        raise ValueError(f"Image height ({img_h}) must be divisible by patch height ({patch_h})")
    if img_w % patch_w != 0:
        raise ValueError(f"Image width ({img_w}) must be divisible by patch width ({patch_w})")

    # Calculate and validate number of patches
    num_patches = (img_h // patch_h) * (img_w // patch_w)
    if num_patches <= 0:
        raise ValueError(f"Number of patches must be positive, got {num_patches}")
    if num_patches > 10000:  # Reasonable upper limit
        logger.warning(f"Large number of patches ({num_patches}) may cause memory issues")

    # Create model instance
    model = ViTHMLP(
        input_shape=input_shape,
        num_classes=num_classes,
        scale=scale,
        patch_size=patch_size,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        pos_dropout_rate=pos_dropout_rate,
        stem_norm_layer=stem_norm_layer,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        activation=activation,
        use_stochastic_depth=use_stochastic_depth,
        stochastic_depth_rate=stochastic_depth_rate,
        **kwargs
    )

    logger.info(f"VisionTransformer-hMLP-{scale} created successfully")
    logger.info(f"Configuration: {num_patches} patches ({img_h // patch_h}x{img_w // patch_w}), {num_classes} classes")
    logger.info(f"hMLP Stem: Progressive processing with {stem_norm_layer} normalization")
    return model


# ---------------------------------------------------------------------
# Utility Functions for Masked Self-Supervised Learning
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
    masked self-supervised learning approaches like MAE and BeiT. The hMLP stem
    processes patches independently without information leakage, making it perfect
    for masked learning scenarios.

    Args:
        batch_size: Batch size for generated data.
        image_size: Image dimensions (height, width).
        patch_size: Patch dimensions (height, width).
        mask_ratio: Ratio of patches to mask (0.0 to 1.0).

    Returns:
        Tuple of (images, mask) where mask is 1 for masked patches, 0 for visible.

    Example:
        ```python
        # Create masked inputs for MAE-style training with hMLP stem
        images, mask = create_inputs_with_masking(
            batch_size=32,
            mask_ratio=0.75  # Typical MAE masking ratio
        )

        # Create model with hMLP stem
        model = create_vit_hmlp(scale='base')
        model.build(images.shape)

        # Apply masking after stem (key advantage of hMLP)
        masked_patches, _ = apply_mask_after_stem(model.stem, images, mask)
        ```
    """
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")

    # Create random images
    images = keras.random.normal([batch_size, image_size[0], image_size[1], 3])

    # Calculate number of patches
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    num_mask = int(mask_ratio * num_patches)

    # Create mask for each sample in batch
    masks = []
    for _ in range(batch_size):
        # Create random mask for this sample
        indices = keras.random.shuffle(ops.arange(num_patches, dtype='int32'))[:num_mask]
        mask_sample = ops.zeros([num_patches], dtype='float32')

        # Set masked positions to 1
        mask_sample = ops.scatter_update(
            mask_sample,
            ops.expand_dims(indices, 1),
            ops.ones([num_mask], dtype='float32')
        )
        masks.append(mask_sample)

    mask = ops.stack(masks, axis=0)

    logger.info(f"Created masked inputs: {batch_size} samples, {mask_ratio:.1%} masking ratio")
    return images, mask


def apply_mask_after_stem(
        stem: HierarchicalMLPStem,
        images: keras.KerasTensor,
        mask: keras.KerasTensor
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Process images with hMLP stem then apply masking.

    This function demonstrates the key advantage of hMLP stem: it can be used with
    masked self-supervised learning by applying masking after the stem processing.
    Unlike convolutional stems, hMLP doesn't cause information leakage between patches,
    making it perfect for masked self-supervised learning scenarios.

    Args:
        stem: Hierarchical MLP stem instance.
        images: Input images of shape (batch_size, height, width, channels).
        mask: Mask tensor of shape (batch_size, num_patches) where 1 = masked.

    Returns:
        Tuple of (masked_patches, mask) where masked patches are set to zero.

    Example:
        ```python
        # Create model and inputs
        model = create_vit_hmlp(scale='base')
        model.build((None, 224, 224, 3))

        images, mask = create_inputs_with_masking(batch_size=4, mask_ratio=0.75)

        # Apply masking after stem - key advantage for SSL
        masked_patches, mask = apply_mask_after_stem(model.stem, images, mask)

        # masked_patches can be used for MAE/BeiT training
        # where only visible patches are processed by the encoder
        logger.info("Applied masking after hMLP stem processing")
        ```
    """
    # Process images with hierarchical MLP stem
    patches = stem(images)  # Shape: (batch_size, num_patches, embed_dim)

    # Apply mask (set masked patches to zero)
    mask_expanded = ops.expand_dims(mask, -1)  # Shape: (batch_size, num_patches, 1)

    # Broadcast mask to match patch dimensions
    embed_dim = ops.shape(patches)[-1]
    mask_expanded = ops.repeat(mask_expanded, embed_dim, axis=-1)

    # Apply mask: multiply by (1 - mask) to zero out masked patches
    masked_patches = patches * (1 - mask_expanded)

    logger.info("Applied mask to hMLP stem output - no information leakage between patches")
    return masked_patches, mask


# ---------------------------------------------------------------------