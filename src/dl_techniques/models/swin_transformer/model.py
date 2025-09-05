"""
Swin Transformer Model Implementation
====================================

A complete implementation of the Swin Transformer architecture with hierarchical vision
transformer using shifted windows. This implementation follows the same patterns as ConvNeXtV2
for consistency and maintainability.

Based on: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021)
https://arxiv.org/abs/2103.14030

Model Variants:
--------------
- Swin-Tiny: [2,2,6,2] blocks, [96,192,384,768] dims, [3,6,12,24] heads (28.3M params)
- Swin-Small: [2,2,18,2] blocks, [96,192,384,768] dims, [3,6,12,24] heads (49.6M params)
- Swin-Base: [2,2,18,2] blocks, [128,256,512,1024] dims, [4,8,16,32] heads (87.8M params)
- Swin-Large: [2,2,18,2] blocks, [192,384,768,1536] dims, [6,12,24,48] heads (196.5M params)

Architecture Overview:
--------------------
1. Patch Embedding: Converts input image to non-overlapping patches
2. Stage 1-4: Each contains multiple Swin Transformer blocks
3. Patch Merging: Reduces spatial resolution and increases feature dimensions
4. Classification Head: Global average pooling + linear classifier

Usage Examples:
--------------
```python
# CIFAR-10 model (32x32 input)
model = SwinTransformer.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

# ImageNet model (224x224 input)
model = SwinTransformer.from_variant("base", num_classes=1000)

# Custom input size model
model = create_swin_transformer("large", num_classes=100, input_shape=(384, 384, 3))
```
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.swin_transformer_block import SwinTransformerBlock

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PatchMerging(keras.layers.Layer):
    """
    Patch merging layer for downsampling between Swin Transformer stages.

    This layer reduces the spatial resolution by a factor of 2 in both height and width
    while doubling the feature dimension. It implements the standard patch merging
    operation from the Swin Transformer paper.

    **Mathematical Operation**:
    - Input: (H, W, C)
    - Concatenate 2x2 neighborhoods: (H/2, W/2, 4C)
    - Linear projection: (H/2, W/2, 2C)

    Args:
        dim: Integer, input dimension (number of channels). Must be positive.
        use_bias: Boolean, whether to use bias in the linear projection. Defaults to False.
        kernel_initializer: Initializer for the projection kernel. Defaults to "glorot_uniform".
        bias_initializer: Initializer for bias if use_bias=True. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for the projection kernel.
        bias_regularizer: Optional regularizer for bias if use_bias=True.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor: (batch_size, height, width, dim)
        height and width should be even for proper merging.

    Output shape:
        4D tensor: (batch_size, height//2, width//2, dim*2)
    """

    def __init__(
            self,
            dim: int,
            use_bias: bool = False,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Layer normalization before merging
        self.norm = layers.LayerNormalization(
            epsilon=1e-5,
            name="norm"
        )

        # Linear projection to reduce dimension from 4C to 2C
        self.reduction = layers.Dense(
            units=2 * dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="reduction"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of patch merging.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, dim).
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, height//2, width//2, dim*2).
        """
        B, H, W, C = ops.shape(inputs)[0], ops.shape(inputs)[1], ops.shape(inputs)[2], ops.shape(inputs)[3]

        # Ensure dimensions are even for merging
        if H % 2 == 1 or W % 2 == 1:
            # Pad if dimensions are odd
            pad_h = H % 2
            pad_w = W % 2
            inputs = ops.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
            H, W = H + pad_h, W + pad_w

        # Extract 2x2 patches: (B, H//2, W//2, 4*C)
        x0 = inputs[:, 0::2, 0::2, :]  # Top-left
        x1 = inputs[:, 1::2, 0::2, :]  # Bottom-left
        x2 = inputs[:, 0::2, 1::2, :]  # Top-right
        x3 = inputs[:, 1::2, 1::2, :]  # Bottom-right

        # Concatenate the 4 patches along channel dimension
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)

        # Apply layer normalization
        x = self.norm(x, training=training)

        # Linear projection to reduce from 4C to 2C
        x = self.reduction(x, training=training)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size, height, width, channels = input_shape
        output_height = None if height is None else (height + 1) // 2
        output_width = None if width is None else (width + 1) // 2
        output_channels = self.dim * 2
        return (batch_size, output_height, output_width, output_channels)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


@keras.saving.register_keras_serializable()
class SwinTransformer(keras.Model):
    """
    Swin Transformer model for image classification with hierarchical vision transformer architecture.

    This model implements the Swin Transformer using shifted windows multi-head self-attention
    with linear computational complexity relative to image size. The architecture features
    hierarchical stages with patch merging for multi-scale feature learning.

    **Intent**: Provide a complete Swin Transformer implementation that efficiently processes
    images through windowed attention mechanisms while maintaining global receptive fields
    through shifted window operations and hierarchical design.

    **Architecture Overview**:
    ```
    Input Image (H, W, 3)
           ↓
    Patch Embedding (H/4, W/4, embed_dim)
           ↓
    Stage 1: depths[0] × SwinTransformerBlock (window_size=7, shift alternating)
           ↓
    Patch Merging (H/8, W/8, embed_dim×2)
           ↓
    Stage 2: depths[1] × SwinTransformerBlock
           ↓
    Patch Merging (H/16, W/16, embed_dim×4)
           ↓
    Stage 3: depths[2] × SwinTransformerBlock
           ↓
    Patch Merging (H/32, W/32, embed_dim×8)
           ↓
    Stage 4: depths[3] × SwinTransformerBlock
           ↓
    LayerNorm → GlobalAvgPool → Classifier (if include_top=True)
           ↓
    Output Logits (num_classes,)
    ```

    **Key Features**:
    - Hierarchical feature extraction with 4 stages
    - Window-based self-attention with O(H×W) complexity
    - Shifted window mechanism for cross-window connections
    - Patch merging for multi-scale representation learning
    - Configurable stochastic depth scheduling

    Args:
        num_classes: Integer, number of output classes. Must be positive.
            Only used if include_top=True. Defaults to 1000 for ImageNet.
        embed_dim: Integer, embedding dimension for patch embedding and first stage.
            Must be positive. Subsequent stages double this dimension. Defaults to 96.
        depths: List of integers, number of Swin blocks in each of the 4 stages.
            Must have exactly 4 elements with positive values. Defaults to [2,2,6,2].
        num_heads: List of integers, number of attention heads in each stage.
            Must have exactly 4 elements with positive values. Defaults to [3,6,12,24].
        window_size: Integer, window size for windowed attention. Must be positive.
            Typical values are 7 or 8. Defaults to 7.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Must be positive. Defaults to 4.0.
        qkv_bias: Boolean, whether to use bias in attention QKV projections.
            Defaults to True.
        dropout_rate: Float, dropout rate for attention projection and MLP.
            Must be in [0, 1). Defaults to 0.0.
        attn_dropout_rate: Float, dropout rate for attention weights.
            Must be in [0, 1). Defaults to 0.0.
        drop_path_rate: Float, maximum stochastic depth rate. Rates are linearly
            scheduled from 0 to this value across all blocks. Must be in [0, 1).
            Defaults to 0.1.
        patch_size: Integer, patch size for patch embedding. Must be positive.
            Determines the initial downsampling factor. Defaults to 4.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: String or Initializer, kernel weight initializer.
            Defaults to "glorot_uniform".
        bias_initializer: String or Initializer, bias initializer. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias terms.
        include_top: Boolean, whether to include the classification head.
            Set to False for feature extraction. Defaults to True.
        input_shape: Tuple, input tensor shape (height, width, channels).
            If None, defaults to (224, 224, 3) for ImageNet. For other datasets,
            specify the appropriate shape (e.g., (32, 32, 3) for CIFAR-10).
        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor: (batch_size, height, width, channels)
        Height and width should be divisible by patch_size * 8 for optimal performance.

    Output shape:
        - If include_top=True: (batch_size, num_classes) - classification logits
        - If include_top=False: (batch_size, H/32, W/32, embed_dim*8) - feature maps

    Attributes:
        patch_embed: Patch embedding layer for tokenizing input images.
        stages: List of lists containing SwinTransformerBlock layers for each stage.
        patch_merge_layers: List of PatchMerging layers for downsampling between stages.
        head_layers: List of layers in the classification head (if include_top=True).

    Example:
        ```python
        # ImageNet model
        model = SwinTransformer.from_variant("base", num_classes=1000)

        # CIFAR-10 model
        model = SwinTransformer.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

        # Feature extractor
        model = SwinTransformer.from_variant("small", include_top=False, input_shape=(224, 224, 3))

        # Custom configuration
        model = SwinTransformer(
            num_classes=100,
            embed_dim=128,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8,
            drop_path_rate=0.2,
            input_shape=(256, 256, 3)
        )
        ```

    Raises:
        ValueError: If configuration parameters are invalid (negative values,
            mismatched list lengths, etc.).

    References:
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          (Liu et al., 2021): https://arxiv.org/abs/2103.14030
        - Official implementation: https://github.com/microsoft/Swin-Transformer

    Note:
        This implementation follows the ConvNeXtV2 model patterns for consistency
        and integrates with the dl-techniques framework components.
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24]
        },
        "small": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24]
        },
        "base": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32]
        },
        "large": {
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48]
        },
    }

    # Architecture constants
    NUM_STAGES = 4
    LAYERNORM_EPSILON = 1e-5
    PATCH_EMBED_NORM = True

    def __init__(
            self,
            num_classes: int = 1000,
            embed_dim: int = 96,
            depths: List[int] = [2, 2, 6, 2],
            num_heads: List[int] = [3, 6, 12, 24],
            window_size: int = 7,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout_rate: float = 0.0,
            attn_dropout_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            patch_size: int = 4,
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:

        # Comprehensive validation
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if len(depths) != self.NUM_STAGES:
            raise ValueError(f"depths must have {self.NUM_STAGES} elements, got {len(depths)}")
        if len(num_heads) != self.NUM_STAGES:
            raise ValueError(f"num_heads must have {self.NUM_STAGES} elements, got {len(num_heads)}")
        if any(d <= 0 for d in depths):
            raise ValueError(f"All depths must be positive, got {depths}")
        if any(h <= 0 for h in num_heads):
            raise ValueError(f"All num_heads must be positive, got {num_heads}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not (0 <= attn_dropout_rate < 1):
            raise ValueError(f"attn_dropout_rate must be in [0, 1), got {attn_dropout_rate}")
        if not (0 <= drop_path_rate < 1):
            raise ValueError(f"drop_path_rate must be in [0, 1), got {drop_path_rate}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        # Set default input shape
        if input_shape is None:
            input_shape = (224, 224, 3)

        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        # Validate dimensions compatibility
        height, width, channels = input_shape
        if height is not None and height % (patch_size * 8) != 0:
            logger.warning(
                f"Input height {height} is not divisible by {patch_size * 8}. "
                f"This may cause issues in deeper stages."
            )
        if width is not None and width % (patch_size * 8) != 0:
            logger.warning(
                f"Input width {width} is not divisible by {patch_size * 8}. "
                f"This may cause issues in deeper stages."
            )

        # Store ALL configuration parameters
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.use_bias = use_bias
        self.include_top = include_top
        self._input_shape = input_shape

        # Store serializable initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Initialize layer lists
        self.stages = []
        self.patch_merge_layers = []
        self.head_layers = []

        # Build the model
        inputs = keras.Input(shape=input_shape, name="input")
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created Swin Transformer model: embed_dim={embed_dim}, "
            f"depths={depths}, num_heads={num_heads}, "
            f"total_blocks={sum(depths)}, input_shape={input_shape}"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Build the complete Swin Transformer model architecture.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor.
        """
        x = inputs

        # Build patch embedding (stem)
        x = self._build_patch_embedding(x)

        # Build hierarchical stages
        for stage_idx in range(self.NUM_STAGES):
            # Add patch merging (except for first stage)
            if stage_idx > 0:
                x = self._build_patch_merging(x, stage_idx)

            # Build stage with multiple Swin blocks
            x = self._build_stage(x, stage_idx)

        # Build classification head if requested
        if self.include_top:
            x = self._build_head(x)

        return x

    def _build_patch_embedding(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Build patch embedding layer to convert image to patches.

        Args:
            x: Input image tensor.

        Returns:
            Patch embedded tensor.
        """
        # Use the embedding factory to create patch embedding
        self.patch_embed = create_embedding_layer(
            embedding_type="patch_2d",
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            use_bias=self.use_bias,
            name="patch_embed"
        )

        x = self.patch_embed(x)

        # Optional normalization after patch embedding
        if self.PATCH_EMBED_NORM:
            self.patch_embed_norm = layers.LayerNormalization(
                epsilon=self.LAYERNORM_EPSILON,
                center=self.use_bias,
                scale=True,
                name="patch_embed_norm"
            )
            x = self.patch_embed_norm(x)

        return x

    def _build_patch_merging(
            self,
            x: keras.KerasTensor,
            stage_idx: int
    ) -> keras.KerasTensor:
        """
        Build patch merging layer for downsampling between stages.

        Args:
            x: Input tensor.
            stage_idx: Current stage index.

        Returns:
            Downsampled tensor.
        """
        # Calculate input dimension for this stage
        input_dim = self.embed_dim * (2 ** (stage_idx - 1))

        patch_merge = PatchMerging(
            dim=input_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name=f"patch_merge_{stage_idx}"
        )

        x = patch_merge(x)
        self.patch_merge_layers.append(patch_merge)

        return x

    def _build_stage(
            self,
            x: keras.KerasTensor,
            stage_idx: int
    ) -> keras.KerasTensor:
        """
        Build a stage with multiple Swin Transformer blocks.

        Args:
            x: Input tensor.
            stage_idx: Stage index (0-3).

        Returns:
            Processed tensor after the stage.
        """
        stage_blocks = []
        depth = self.depths[stage_idx]
        num_heads = self.num_heads[stage_idx]

        # Calculate dimension for this stage
        stage_dim = self.embed_dim * (2 ** stage_idx)

        # Calculate drop path rates for this stage (linear scheduling)
        total_blocks = sum(self.depths)
        block_start_idx = sum(self.depths[:stage_idx])

        for block_idx in range(depth):
            # Calculate current drop path rate
            current_block_idx = block_start_idx + block_idx
            if total_blocks > 1:
                current_drop_path_rate = (
                        self.drop_path_rate * current_block_idx / (total_blocks - 1)
                )
            else:
                current_drop_path_rate = 0.0

            # Alternate between regular and shifted window attention
            shift_size = 0 if block_idx % 2 == 0 else self.window_size // 2

            # Create Swin Transformer block
            block = SwinTransformerBlock(
                dim=stage_dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                dropout_rate=self.dropout_rate,
                attn_dropout_rate=self.attn_dropout_rate,
                drop_path=current_drop_path_rate,
                activation="gelu",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )

            x = block(x)
            stage_blocks.append(block)

        self.stages.append(stage_blocks)

        return x

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Build classification head.

        Args:
            x: Input feature tensor.

        Returns:
            Classification logits.
        """
        # Layer normalization before global pooling
        head_norm = layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        x = head_norm(x)

        # Global average pooling
        gap = layers.GlobalAveragePooling2D(name="global_avg_pool")
        x = gap(x)

        # Classification layer
        if self.num_classes > 0:
            classifier = layers.Dense(
                units=self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="classifier"
            )
            x = classifier(x)

            self.head_layers = [head_norm, gap, classifier]
        else:
            self.head_layers = [head_norm, gap]

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> "SwinTransformer":
        """
        Create a Swin Transformer model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large".
            num_classes: Integer, number of output classes.
            input_shape: Tuple, input shape. If None, uses (224, 224, 3).
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            SwinTransformer model instance.

        Raises:
            ValueError: If variant is not recognized.

        Example:
            ```python
            # CIFAR-10 model
            model = SwinTransformer.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

            # ImageNet model
            model = SwinTransformer.from_variant("base", num_classes=1000)

            # Feature extraction
            model = SwinTransformer.from_variant("large", include_top=False)
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating Swin Transformer-{variant.upper()} model")

        return cls(
            num_classes=num_classes,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = {
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "dropout_rate": self.dropout_rate,
            "attn_dropout_rate": self.attn_dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "patch_size": self.patch_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SwinTransformer":
        """
        Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            SwinTransformer model instance.
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

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        total_params = sum(layer.count_params() for layer in self.layers)

        logger.info("Swin Transformer configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Embedding dimension: {self.embed_dim}")
        logger.info(f"  - Window size: {self.window_size}")
        logger.info(f"  - Stages: {self.NUM_STAGES}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Num heads: {self.num_heads}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------

def create_swin_transformer(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: bool = False,
        **kwargs: Any
) -> SwinTransformer:
    """
    Convenience function to create Swin Transformer models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large").
        num_classes: Integer, number of output classes.
        input_shape: Tuple, input shape. If None, uses (224, 224, 3).
        pretrained: Boolean, whether to load pretrained weights (not implemented).
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        SwinTransformer model instance.

    Example:
        ```python
        # Create Swin-Tiny for CIFAR-10
        model = create_swin_transformer("tiny", num_classes=10, input_shape=(32, 32, 3))

        # Create Swin-Base for ImageNet
        model = create_swin_transformer("base", num_classes=1000)

        # Create feature extractor
        model = create_swin_transformer("large", include_top=False)
        ```
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    model = SwinTransformer.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------