"""
Vision Transformer (ViT) Model Implementation with Deep Supervision

The implementation supports different scales and configurations similar to the original
"An Image is Worth 16x16 Words" paper and its variants, with enhanced flexibility
through factory-based component creation.

Deep supervision provides multiple outputs during training:
- Output 0: Final inference output (highest resolution, primary output)
- Output 1-N: Intermediate supervision outputs at progressively earlier transformer layers

This allows for:
- Better gradient flow to earlier layers
- Multi-scale feature learning and supervision
- More stable training for very deep networks
- Curriculum learning capabilities through weight scheduling
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union, Literal, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
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
    Vision Transformer model with factory-based component creation and optional deep supervision.

    This model implements the complete Vision Transformer architecture using the dl_techniques
    framework's factory system for consistent component creation. It supports different scales
    and configurations for various vision tasks including classification, feature extraction,
    and transfer learning.

    **Deep Supervision**: When enabled, the model produces multiple outputs during training:
    - Output 0: Final inference output (primary classification head)
    - Output 1-N: Intermediate supervision outputs from earlier transformer layers

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
    TransformerLayer × num_layers [with optional intermediate outputs]
           ↓
    Final Normalization
           ↓
    [Classification Head] OR [Feature Extraction]
           ↓
    Output (shape depends on configuration)
    ```

    **Scale Configurations**:
    - **Pico**: 192d, 3h, 6L - Ultra-lightweight for quick experiments
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
            Available: 'pico', 'tiny', 'small', 'base', 'large', 'huge'. Defaults to 'base'.
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
        enable_deep_supervision: Boolean, whether to add deep supervision outputs.
            When True, model outputs multiple scales during training.
            Defaults to False.
        supervision_layer_indices: Optional[List[int]], indices of transformer layers
            to add supervision outputs after. If None, automatically selects evenly
            spaced layers. Defaults to None.
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
        - Without deep supervision: `(batch_size, num_classes)` - Logits for each class
        - With deep supervision: List of `[(batch_size, num_classes), ...]` where:
          - Output 0: Final output (primary)
          - Output 1-N: Intermediate supervision outputs

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
        supervision_heads: List of intermediate supervision heads (if deep supervision enabled).

    Example:
        ```python
        # Standard ViT-Base for ImageNet classification
        model = ViT(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base'
        )

        # ViT with deep supervision for better training
        model = ViT(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base',
            enable_deep_supervision=True
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
            attention_dropout_rate=0.1,
            enable_deep_supervision=True
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
            normalization_position: Literal['pre', 'post'] = "post",
            ffn_type: FFNType = "mlp",
            activation: Union[str, callable] = "gelu",
            enable_deep_supervision: bool = False,
            supervision_layer_indices: Optional[List[int]] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize Vision Transformer model with optional deep supervision."""
        # Auto-generate name if not provided
        if name is None:
            ds_suffix = '_ds' if enable_deep_supervision else ''
            name = f"vision_transformer_{scale}{ds_suffix}"

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
        self.normalization_position = str(normalization_position)
        self.ffn_type = str(ffn_type)
        self.activation = activation
        self.enable_deep_supervision = bool(enable_deep_supervision)
        self.supervision_layer_indices = supervision_layer_indices

        # Get model configuration from scale
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate derived parameters
        self.intermediate_size = int(self.embed_dim * self.mlp_ratio)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate derived parameters
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        # Determine supervision layer indices if not provided
        if self.enable_deep_supervision and self.supervision_layer_indices is None:
            # Automatically select evenly spaced layers (excluding the last one)
            # For example, with 12 layers, select layers at indices: [3, 7, 11] for 3 supervision outputs
            # This provides supervision at early, middle, and late stages
            if self.num_layers <= 3:
                # For very shallow models, supervise all intermediate layers
                self.supervision_layer_indices = list(range(self.num_layers - 1))
            else:
                # For deeper models, select 3-4 evenly spaced intermediate layers
                num_supervision = min(4, self.num_layers - 1)
                step = max(1, (self.num_layers - 1) // num_supervision)
                self.supervision_layer_indices = list(range(step - 1, self.num_layers - 1, step))
        elif not self.enable_deep_supervision:
            self.supervision_layer_indices = []

        # Validate supervision indices
        if self.supervision_layer_indices:
            for idx in self.supervision_layer_indices:
                if not (0 <= idx < self.num_layers - 1):
                    raise ValueError(
                        f"Supervision layer index {idx} out of range. "
                        f"Must be in [0, {self.num_layers - 1})"
                    )

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
                name="final_output"  # Named for multi-output metric tracking
            )

        # Deep supervision components
        self.supervision_dropouts = []
        self.supervision_heads = []

        if self.enable_deep_supervision and self.include_top:
            for i, layer_idx in enumerate(self.supervision_layer_indices):
                # Dropout for supervision output
                if self.dropout_rate > 0.0:
                    dropout = layers.Dropout(self.dropout_rate, name=f"supervision_dropout_{layer_idx}")
                    self.supervision_dropouts.append(dropout)
                else:
                    self.supervision_dropouts.append(None)

                # Classification head for supervision
                head = layers.Dense(
                    self.num_classes,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name=f"supervision_output_{layer_idx}"
                )
                self.supervision_heads.append(head)

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

        if self.enable_deep_supervision:
            logger.info(f"Deep supervision enabled with {len(self.supervision_layer_indices)} intermediate outputs")
            logger.info(f"Supervision at transformer layers: {self.supervision_layer_indices}")

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

        # Deep supervision components
        if self.enable_deep_supervision and self.include_top:
            for i in range(len(self.supervision_layer_indices)):
                if self.supervision_dropouts[i] is not None:
                    self.supervision_dropouts[i].build(head_input_shape)
                self.supervision_heads[i].build(head_input_shape)

        # Global pooling
        if self.global_pool is not None:
            self.global_pool.build(pos_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, List[keras.KerasTensor]]:
        """
        Forward pass through the Vision Transformer.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Model output tensor or list of tensors (if deep supervision enabled).
            - Single output: Shape depends on include_top and pooling settings.
            - Multi-output (deep supervision): List of [final_output, supervision_outputs...]
        """
        # 1. Convert image to a sequence of patch embeddings
        x = self.patch_embed(inputs, training=training)  # (batch_size, num_patches, embed_dim)

        # 2. Prepend the CLS token to the sequence of patches
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)  # (batch_size, seq_len, embed_dim)

        # 3. Add learned positional embeddings
        x = self.pos_embed(x, training=training)

        # 4. Process through transformer layers with optional intermediate supervision
        supervision_outputs = []

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, training=training)

            # Collect intermediate supervision outputs during training
            if self.enable_deep_supervision and self.include_top and i in self.supervision_layer_indices:
                # Find the index in supervision_layer_indices
                sup_idx = self.supervision_layer_indices.index(i)

                # The output 'x' is already normalized from the TransformerLayer (post-norm).
                # Extract the CLS token directly.
                cls_token_sup = x[:, 0, :]

                # Apply dropout
                if self.supervision_dropouts[sup_idx] is not None:
                    cls_token_sup = self.supervision_dropouts[sup_idx](cls_token_sup, training=training)

                # Generate supervision output
                sup_output = self.supervision_heads[sup_idx](cls_token_sup)
                supervision_outputs.append(sup_output)

        # 5. Handle the output based on the model's configuration
        if self.include_top:
            # --- Classification Head Logic ---
            # Apply final normalization to the entire sequence first
            x_norm = self.norm(x, training=training)

            # Now, extract the CLS token from the normalized sequence
            cls_token = x_norm[:, 0, :]

            # Apply dropout
            if self.head_dropout is not None:
                cls_token = self.head_dropout(cls_token, training=training)

            # Final classification
            final_output = self.head(cls_token)  # (batch_size, num_classes)

            # Return outputs based on deep supervision setting
            if self.enable_deep_supervision and supervision_outputs:
                # Multi-output: [final_output, supervision_outputs...]
                # Order is from shallowest intermediate layer to deepest
                all_outputs = [final_output] + supervision_outputs
                return all_outputs
            else:
                return final_output

        else:
            # --- Feature Extraction Logic ---
            x = self.norm(x, training=training)

            if self.pooling == "cls":
                return x[:, 0, :]  # (batch_size, embed_dim)
            elif self.pooling == "mean":
                # For mean/max pooling, we should exclude the CLS token
                return self.global_pool(x[:, 1:, :])  # (batch_size, embed_dim)
            elif self.pooling == "max":
                return self.global_pool(x[:, 1:, :])  # (batch_size, embed_dim)
            else:
                return x  # (batch_size, seq_len, embed_dim)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Union[
        Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """
        Compute output shape.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output shape tuple or list of tuples (if deep supervision enabled).
        """
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        batch_size = input_shape[0]

        if self.include_top:
            output_shape = (batch_size, self.num_classes)

            if self.enable_deep_supervision and self.supervision_layer_indices:
                # Return list of output shapes
                num_outputs = 1 + len(self.supervision_layer_indices)
                return [output_shape] * num_outputs
            else:
                return output_shape
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
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            "activation": self.activation,
            "enable_deep_supervision": self.enable_deep_supervision,
            "supervision_layer_indices": self.supervision_layer_indices,
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
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            activation=self.activation,
            enable_deep_supervision=False,  # Feature extractors don't use deep supervision
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
        logger.info(f"Deep Supervision: {self.enable_deep_supervision}")
        if self.enable_deep_supervision:
            logger.info(f"Supervision Layers: {self.supervision_layer_indices}")
        if self.built:
            logger.info(f"Total Parameters: {self.count_params():,}")

        # Additional architecture information
        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        logger.info(f"Patches per dimension: {img_h // patch_h} x {img_w // patch_w}")


# ---------------------------------------------------------------------
# Factory Functions for Convenient Model Creation
# ---------------------------------------------------------------------


def create_vision_transformer(
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
        normalization_position: Literal['pre', 'post'] = "post",
        ffn_type: FFNType = "mlp",
        activation: Union[str, callable] = "gelu",
        enable_deep_supervision: bool = False,
        supervision_layer_indices: Optional[List[int]] = None,
        **kwargs: Any
) -> ViT:
    """
    Create a Vision Transformer model with specified configuration.

    This factory function provides parameter validation and sensible defaults
    for creating Vision Transformer models with different scales and configurations.

    Args:
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of output classes for classification.
        scale: Model scale determining architecture size.
        patch_size: Size of patches to extract from input images.
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction when include_top=False.
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
        enable_deep_supervision: Whether to add deep supervision outputs.
        supervision_layer_indices: Optional indices for supervision layers.
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

        # Create ViT with deep supervision
        model = create_vision_transformer(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base',
            enable_deep_supervision=True
        )

        # Feature extractor with modern components
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

    # Create model instance
    model = ViT(
        input_shape=input_shape,
        num_classes=num_classes,
        scale=scale,
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
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        activation=activation,
        enable_deep_supervision=enable_deep_supervision,
        supervision_layer_indices=supervision_layer_indices,
        **kwargs
    )

    logger.info(f"VisionTransformer-{scale} created successfully")
    logger.info(f"Configuration: {num_patches} patches ({img_h // patch_h}x{img_w // patch_w}), {num_classes} classes")
    if enable_deep_supervision:
        logger.info("Deep supervision enabled")
    return model