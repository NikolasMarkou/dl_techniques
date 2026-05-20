"""
Vision Transformer (ViT) Model Implementation

The implementation supports different scales and configurations similar to the original
"An Image is Worth 16x16 Words" paper and its variants, with enhanced flexibility
through factory-based component creation.
"""

import os
import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.deep_supervision import (
    create_inference_model_from_training_model,
)
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

PoolingMode = Literal['cls', 'mean', 'max']
VitScale = Literal['pico', 'tiny', 'small', 'base', 'large', 'huge']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ViT(keras.Model):
    """
    Vision Transformer model with factory-based component creation and modern Keras 3 patterns.

    This model implements the complete Vision Transformer architecture using the dl_techniques
    framework's factory system for consistent component creation. It supports different scales
    and configurations for various vision_heads tasks including classification, feature extraction,
    and transfer learning.

    **Intent**: Provide a production-ready Vision Transformer implementation that leverages
    the dl_techniques framework's modular components while following modern Keras 3 best
    practices for robust serialization and deployment.

    **Architecture**:
    ```
    Input Images (batch, height, width, channels)
           ↓
    PatchEmbedding2D → Patches (batch, num_patches, embed_dim)
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
            Defaults to 'post'.
        ffn_type: FFNType, feed-forward network type for transformer layers.
            Uses factory for consistent creation. Available options:
            - 'mlp': Standard MLP with intermediate expansion (default)
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'geglu': GELU-based Gated Linear Unit
            Defaults to 'mlp'.
        activation: Union[str, Callable], activation function for FFN.
            Defaults to 'gelu'.
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
        patch_embed: PatchEmbedding2D layer for image tokenization.
        pos_embed: PositionalEmbedding layer for sequence position encoding.
        transformer_layers: List of TransformerLayer instances.
        norm: Final normalization layer.
        head: Optional Dense layer for classification.

    Example:
        ```python
        # Standard ViT-Base for ImageNet classification
        model = ViT(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base'
        )

        # Feature extractor with CLS token
        feature_model = ViT(
            input_shape=(224, 224, 3),
            scale='base',
            include_top=False,
            pooling='cls'
        )

        # Custom configuration with modern components
        custom_model = ViT(
            input_shape=(384, 384, 3),
            num_classes=10,
            scale='small',
            patch_size=16,
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            dropout_rate=0.1,
            attention_dropout_rate=0.1
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
        support. All sub-components are created using dl_techniques factories for
        consistency and configurability.
    """

    # Scale configurations: [embed_dim, num_heads, num_layers, mlp_ratio]
    SCALE_CONFIGS: Dict[str, Tuple[int, int, int, float]] = {
        "pico": (192, 3, 6, 4.0),  # ViT-Pico
        "tiny": (192, 3, 12, 4.0),  # ViT-Tiny
        "small": (384, 6, 12, 4.0),  # ViT-Small
        "base": (768, 12, 12, 4.0),  # ViT-Base
        "large": (1024, 16, 24, 4.0),  # ViT-Large
        "huge": (1280, 16, 32, 4.0),  # ViT-Huge
    }

    # ResNet-template variant registry. Thin wrapper over SCALE_CONFIGS so
    # callers can use `ViT.from_variant("vit_pico", ...)` exactly like
    # `ResNet.from_variant("resnet50", ...)`.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "vit_pico":  {"scale": "pico"},
        "vit_tiny":  {"scale": "tiny"},
        "vit_small": {"scale": "small"},
        "vit_base":  {"scale": "base"},
        "vit_large": {"scale": "large"},
        "vit_huge":  {"scale": "huge"},
    }

    # Pretrained weights URLs (placeholder -- no public ViT weights are
    # distributed for this implementation; see `_download_weights`).
    PRETRAINED_WEIGHTS: Dict[str, Dict[str, str]] = {
        "vit_base":  {"imagenet": "https://example.com/vit_base_imagenet.keras"},
        "vit_large": {"imagenet": "https://example.com/vit_large_imagenet.keras"},
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
            kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            normalization_type: NormalizationType = "layer_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            normalization_position: Literal['pre', 'post'] = "post",
            ffn_type: FFNType = "mlp",
            activation: Union[str, callable] = "gelu",
            use_layer_scale: bool = False,
            layer_scale_init_value: float = 1e-5,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize Vision Transformer model."""
        # Auto-generate name if not provided
        if name is None:
            name = f"vision_transformer_{scale}"

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

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(f"pos_dropout_rate must be between 0 and 1, got {pos_dropout_rate}")

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
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = bias_regularizer
        self.normalization_type = str(normalization_type)
        # DECISION plan_2026-05-18_6776f8ba/D-003
        # Optional, additive `normalization_kwargs` plumbed into both the final
        # `self.norm` factory call and every TransformerLayer (as
        # `attention_norm_args` + `ffn_norm_args`). Default `None` -> `{}` -> the
        # factory call is byte-identical to the pre-plumbing version, preserving
        # bit-exactness for ALL existing serialized ViT checkpoints. This is the
        # multi-flag-plumbing pattern from LESSONS L72; the ViT path here mirrors
        # the ResNet path in `dl_techniques/models/resnet/model.py` for the
        # rms_variants_train Phase 3 `param_matched` mode (use_scale=False).
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}
        self.normalization_position = str(normalization_position)
        self.ffn_type = str(ffn_type)
        self.activation = activation
        self.use_layer_scale = bool(use_layer_scale)
        self.layer_scale_init_value = float(layer_scale_init_value)

        # Get model configuration from scale
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate derived parameters
        self.intermediate_size = int(self.embed_dim * self.mlp_ratio)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate derived parameters
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Using factories for consistent component creation

        # Patch embedding using factory
        self.patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name="patch_embed"
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
            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type="multi_head",
                normalization_type=self.normalization_type,
                attention_norm_args=dict(self.normalization_kwargs),
                ffn_norm_args=dict(self.normalization_kwargs),
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_layer_scale=self.use_layer_scale,
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Final normalization using factory
        self.norm = create_normalization_layer(
            self.normalization_type,
            name="norm",
            **self.normalization_kwargs,
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

        logger.info(f"Created VisionTransformer-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")
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
            initializer="zeros",
            trainable=True
        )

        # Build all sub-layers in computational order
        # Patch embedding
        dummy_input_shape = (None,) + self.input_shape_config
        self.patch_embed.build(dummy_input_shape)

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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Vision Transformer.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Model output tensor. Shape depends on include_top and pooling settings.
        """
        # 1. Convert image to a sequence of patch embeddings
        x = self.patch_embed(inputs, training=training)

        # 2. Prepend the CLS token
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)

        # 3. Add learned positional embeddings
        x = self.pos_embed(x, training=training)

        # 4. Process through the Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # 5. Apply final normalization to the entire sequence for architectural consistency
        x_norm = self.norm(x, training=training)

        # 6. Handle the output based on the model's configuration
        if self.include_top:
            # --- Classification Head Logic ---
            # Extract the CLS token from the *normalized* sequence.
            cls_token = x_norm[:, 0, :]

            # Pass through the final classification head
            if self.head_dropout is not None:
                cls_token = self.head_dropout(cls_token, training=training)

            return self.head(cls_token)
        else:
            # --- Feature Extraction Logic ---
            if self.pooling == "cls":
                # Return the normalized CLS token
                return x_norm[:, 0, :]
            elif self.pooling == "mean":
                # Pool over patch tokens only, excluding the CLS token
                return self.global_pool(x_norm[:, 1:, :])
            elif self.pooling == "max":
                # Pool over patch tokens only, excluding the CLS token
                return self.global_pool(x_norm[:, 1:, :])
            else:
                # Return the full, normalized sequence
                return x_norm

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
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            "activation": self.activation,
            "use_layer_scale": self.use_layer_scale,
            "layer_scale_init_value": self.layer_scale_init_value,
        })
        return config

    def get_feature_extractor(self) -> "ViT":
        """
        Get a feature extractor version of this model.

        Returns:
            New ViT instance configured for feature extraction.
        """
        if not hasattr(self, 'input_shape_config') or not self.input_shape_config:
            raise ValueError("Model must be properly initialized before creating feature extractor")

        return ViT(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            pos_dropout_rate=self.pos_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            normalization_type=self.normalization_type,
            normalization_kwargs=dict(self.normalization_kwargs),
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            activation=self.activation,
            use_layer_scale=self.use_layer_scale,
            layer_scale_init_value=self.layer_scale_init_value,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Print detailed model summary with architecture information."""
        logger.info("Vision Transformer Model Summary")
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
        logger.info(f"Normalization Type: {self.normalization_type}")
        logger.info(f"Normalization Position: {self.normalization_position}")
        logger.info(f"FFN Type: {self.ffn_type}")
        logger.info(f"Activation: {self.activation}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self.built:
            logger.info(f"Total Parameters: {self.count_params():,}")

        # Additional architecture information
        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        logger.info(f"Patches per dimension: {img_h // patch_h} x {img_w // patch_w}")

    # -----------------------------------------------------------------
    # Pretrained-weights API (resnet-template parity)
    # -----------------------------------------------------------------

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True,
    ) -> None:
        """Load pretrained weights from a local ``.keras`` file.

        # DECISION plan_2026-05-12_f2d29729/D-001
        Diverges from the resnet template's ``self.load_weights(..., by_name=True)``
        path because Keras 3.8 raises ``ValueError("Invalid keyword arguments:
        {'by_name': True}")`` when ``by_name=True`` is passed to
        ``.keras`` files (LESSONS L71). Route through
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`
        which does a full-model load + layer-by-layer ``set_weights``.

        Args:
            weights_path: Path to the ``.keras`` weights file.
            skip_mismatch: Whether to skip layers with mismatched shapes
                (forwarded as the inverse of ``strict``).

        Raises:
            FileNotFoundError: If ``weights_path`` does not exist.
            ValueError: If the file is not ``.keras`` or strict transfer fails.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Build the model if not already built (probe forward pass).
        if not self.built:
            dummy_input = keras.random.normal((1,) + tuple(self.input_shape_config))
            self(dummy_input, training=False)

        logger.info(f"Loading pretrained weights from {weights_path}")
        report = load_weights_from_checkpoint(
            target=self,
            ckpt_path=weights_path,
            skip_prefixes=("head",),
            strict=(not skip_mismatch),
        )
        logger.info(f"Weight transfer report: {report}")

    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None,
    ) -> str:
        """Download pretrained weights for ``variant``.

        # DECISION plan_2026-05-12_f2d29729/D-002
        Diverges from the resnet template which actually attempts to download
        from a placeholder URL. No public ViT checkpoints in the
        ``dl_techniques`` weight format are distributed at this time
        (LESSONS L53). Calling this method always raises
        :class:`NotImplementedError` so failures are loud, not silent
        404s that produce HTML payloads masquerading as ``.keras`` files.

        Args:
            variant: Model variant (e.g. ``"vit_base"``). Unused; reserved
                for the future signature parity with :class:`ResNet`.
            dataset: Pretraining dataset name. Unused; same reason.
            cache_dir: Cache directory. Unused; same reason.

        Raises:
            NotImplementedError: Always.
        """
        del variant, dataset, cache_dir  # silence unused-arg lint
        raise NotImplementedError(
            "No public ViT checkpoints are distributed for this implementation. "
            "Pass a local .keras path to `pretrained=` to load custom weights, "
            "e.g. `ViT.from_variant('vit_base', pretrained='/path/to/weights.keras')`."
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, int, int]] = None,
            pretrained: Union[bool, str] = False,
            weights_dataset: str = "imagenet",
            weights_input_shape: Optional[Tuple[int, int, int]] = None,
            cache_dir: Optional[str] = None,
            **kwargs: Any,
    ) -> "ViT":
        """Create a ViT model from a predefined variant.

        # DECISION plan_2026-05-12_f2d29729/D-003
        Resnet-template parity. The ``except (IOError, OSError, ValueError)``
        around ``_download_weights`` is narrow on purpose so a
        :class:`NotImplementedError` from ``_download_weights`` propagates
        (LESSONS L53 -- silent fallbacks hide bugs). A user who sets
        ``pretrained=True`` without providing a local checkpoint path
        gets a clear error.

        Args:
            variant: One of the keys in :attr:`MODEL_VARIANTS`
                (``"vit_pico".."vit_huge"``).
            num_classes: Number of output classes for the classification head.
            input_shape: Input image shape ``(H, W, C)``. Defaults to
                ``(224, 224, 3)``.
            pretrained: Either a boolean (``True`` -> attempt download, will
                raise) or a string path to a local ``.keras`` checkpoint.
            weights_dataset: Dataset name for pretrained weights routing.
            weights_input_shape: Optional input shape the pretrained weights
                were trained at. If different from ``input_shape``, mismatch
                skipping is enabled during transfer.
            cache_dir: Optional cache directory for downloads.
            **kwargs: Forwarded to the :class:`ViT` constructor.

        Returns:
            Configured (and possibly weight-loaded) :class:`ViT` instance.

        Raises:
            ValueError: If ``variant`` is not recognized.
            NotImplementedError: If ``pretrained=True`` (boolean) -- no public
                ViT weights are hosted.
            FileNotFoundError: If ``pretrained="/path"`` and the path is missing.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        variant_cfg = cls.MODEL_VARIANTS[variant]
        scale = variant_cfg["scale"]

        if input_shape is None:
            input_shape = (224, 224, 3)

        logger.info(f"Creating ViT model variant '{variant}' (scale='{scale}')")

        # Resolve pretrained source --------------------------------------
        load_weights_path: Optional[str] = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                # boolean True -- delegate to _download_weights, narrow except.
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir,
                    )
                except (IOError, OSError, ValueError) as e:
                    logger.warning(
                        f"Failed to acquire pretrained weights via download: {e}. "
                        f"Continuing with random initialization."
                    )
                    load_weights_path = None

            # Decide whether to enable shape-mismatch skipping.
            include_top = kwargs.get("include_top", True)
            if include_top and num_classes != 1000:
                skip_mismatch = True
                logger.info(
                    f"num_classes ({num_classes}) != pretrained head (1000); "
                    f"head weights will be skipped."
                )
            if weights_input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Pretrained weights trained on {weights_input_shape} but "
                    f"model uses {input_shape}; positional embeddings may differ."
                )
                skip_mismatch = True

        # Build model ----------------------------------------------------
        model = cls(
            input_shape=input_shape,
            num_classes=num_classes,
            scale=scale,
            **kwargs,
        )

        # Load weights if a path was resolved ----------------------------
        if load_weights_path:
            model.load_pretrained_weights(
                weights_path=load_weights_path,
                skip_mismatch=skip_mismatch,
            )

        return model


# ---------------------------------------------------------------------
# Factory Functions for Convenient Model Creation
# ---------------------------------------------------------------------


def create_vit(
        variant: str = "vit_base",
        num_classes: int = 1000,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        weights_input_shape: Optional[Tuple[int, int, int]] = None,
        cache_dir: Optional[str] = None,
        include_top: bool = True,
        pooling: Optional[PoolingMode] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        pos_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        normalization_type: NormalizationType = "layer_norm",
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        normalization_position: Literal['pre', 'post'] = "post",
        ffn_type: FFNType = "mlp",
        activation: Union[str, callable] = "gelu",
        **kwargs: Any
) -> ViT:
    """
    Create a Vision Transformer model with specified configuration.

    Resnet-template factory: ``variant`` is the canonical key and routes
    through :meth:`ViT.from_variant` so callers get the pretrained-weights
    + variant registry for free.

    Args:
        variant: Variant key (``"vit_pico".."vit_huge"``). Defaults to
            ``"vit_base"``.
        num_classes: Number of output classes for classification.
            Must be positive. Only used when include_top=True.
        input_shape: Input image shape (height, width, channels).
            Must have positive dimensions and be compatible with patch_size.
        patch_size: Size of patches to extract from input images.
            If int, uses square patches. Image dimensions must be divisible by patch size.
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction when include_top=False.
            Available: 'cls', 'mean', 'max', None.
        dropout_rate: Dropout rate for general regularization.
        attention_dropout_rate: Dropout rate for attention weights.
        pos_dropout_rate: Dropout rate for positional embeddings.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        bias_initializer: Bias initializer for all layers.
        bias_regularizer: Bias regularizer for all layers.
        normalization_type: Type of normalization layer to use.
        normalization_position: Position of normalization in transformer layers.
        ffn_type: Type of feed-forward network for transformer layers.
        activation: Activation function for feed-forward networks.
        **kwargs: Additional arguments for ViT constructor.

    Returns:
        ViT model instance.

    Raises:
        ValueError: If any parameter validation fails.

    Example:
        ```python
        # Create ViT-Base for ImageNet
        model = create_vision_transformer(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base'
        )

        # Create feature extractor with modern components
        feature_model = create_vision_transformer(
            input_shape=(384, 384, 3),
            scale='small',
            include_top=False,
            pooling='cls',
            normalization_type='rms_norm',
            ffn_type='swiglu'
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

    # Delegate to from_variant for unified pretrained-weights handling.
    model = ViT.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        patch_size=patch_size,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        pos_dropout_rate=pos_dropout_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        normalization_type=normalization_type,
        normalization_kwargs=normalization_kwargs,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        activation=activation,
        **kwargs,
    )

    logger.info(f"ViT variant '{variant}' created successfully")
    logger.info(f"Configuration: {num_patches} patches ({img_h // patch_h}x{img_w // patch_w}), {num_classes} classes")
    return model

# ---------------------------------------------------------------------
