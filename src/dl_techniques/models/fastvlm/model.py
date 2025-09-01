"""
FastVLM Model Implementation for dl_techniques Framework

This module provides the FastVLM (Fast Vision Language Model) architecture,
a hybrid vision model that efficiently combines convolutional and transformer-based
components for high-performance image classification and feature extraction.
"""

import keras
from keras import ops, layers, initializers
from typing import Optional, Union, Tuple, List, Dict, Any


from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.repmixer_block import RepMixerBlock, ConvolutionalStem
from dl_techniques.layers.layer_scale import LearnableMultiplier, MultiplierType


@keras.saving.register_keras_serializable()
class AttentionBlock(keras.layers.Layer):
    """
    Attention block using dl_techniques TransformerLayer with vision-specific adaptations.

    This layer wraps the TransformerLayer from dl_techniques framework with
    vision-specific configurations and preprocessing to work effectively with
    spatial feature maps from convolutional layers.

    **Intent**: Provide a vision-optimized attention mechanism that leverages
    the powerful TransformerLayer from dl_techniques while adding spatial
    awareness and efficient processing for vision tasks.

    **Architecture**:
    ```
    Input(shape=[H, W, C])
           ↓
    Spatial Flatten: [H*W, C]
           ↓
    TransformerLayer(attention + FFN)
           ↓
    Spatial Reshape: [H, W, C]
           ↓
    LayerScale (optional)
           ↓
    Output(shape=[H, W, C])
    ```

    **Design Features**:
    - Spatial-to-sequence conversion for transformer processing
    - Configurable attention mechanism (multi-head, window, etc.)
    - Optional layer scaling for training stability
    - Preserves spatial dimensions through reshape operations

    Args:
        dim: Integer, feature dimension. Must be positive and divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive and divide dim.
            Defaults to 8.
        mlp_ratio: Float, expansion ratio for the MLP in transformer.
            Must be positive. Defaults to 4.0.
        attention_type: String, type of attention mechanism to use.
            Options: 'multi_head_attention', 'window_attention', 'group_query_attention'.
            Defaults to 'multi_head_attention'.
        normalization_position: String, position of normalization layers.
            Options: 'pre', 'post'. Defaults to 'pre'.
        dropout_rate: Float, dropout rate. Must be between 0 and 1. Defaults to 0.0.
        use_layer_scale: Boolean, whether to apply learnable layer scaling.
            Defaults to True.
        layer_scale_init: Float, initial value for layer scale parameters.
            Must be positive. Defaults to 1e-4.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with same shape as input: `(batch_size, height, width, channels)`

    Attributes:
        transformer: TransformerLayer instance for attention computation.
        layer_scale: Optional LearnableMultiplier for output scaling.
        height: Height dimension extracted from input shape.
        width: Width dimension extracted from input shape.

    Example:
        ```python
        # Basic attention block
        attn = AttentionBlock(dim=256, num_heads=8)
        inputs = keras.Input(shape=(14, 14, 256))
        outputs = attn(inputs)  # Shape: (None, 14, 14, 256)

        # With window attention for efficiency
        attn = AttentionBlock(
            dim=512,
            num_heads=16,
            attention_type='window_attention',
            mlp_ratio=6.0
        )

        # With custom dropout and layer scaling
        attn = AttentionBlock(
            dim=768,
            num_heads=12,
            dropout_rate=0.1,
            use_layer_scale=True,
            layer_scale_init=1e-5
        )

        # In a vision model pipeline
        x = ConvolutionalStem(64)(image_input)  # [H/4, W/4, 64]
        x = RepMixerBlock(64)(x)                # Local feature mixing
        x = AttentionBlock(64, num_heads=4)(x)  # Global attention
        ```

    Note:
        The spatial flatten/reshape operations preserve the spatial structure
        while allowing transformer processing. This is more efficient than
        using 2D positional encodings for vision tasks.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            attention_type: str = 'multi_head_attention',
            normalization_position: str = 'pre',
            dropout_rate: float = 0.0,
            use_layer_scale: bool = True,
            layer_scale_init: float = 1e-4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if layer_scale_init <= 0:
            raise ValueError(f"layer_scale_init must be positive, got {layer_scale_init}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.normalization_position = normalization_position
        self.dropout_rate = dropout_rate
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init

        # Will be set in build
        self.height = None
        self.width = None

        # CREATE transformer layer with vision-optimized settings
        self.transformer = TransformerLayer(
            hidden_size=dim,
            num_heads=num_heads,
            intermediate_size=int(dim * mlp_ratio),
            attention_type=attention_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=dropout_rate,
            activation='gelu',
            name='vision_transformer'
        )

        # CREATE layer scale if requested
        if use_layer_scale:
            self.layer_scale = LearnableMultiplier(
                multiplier_type=MultiplierType.CHANNEL,
                initializer=keras.initializers.Constant(layer_scale_init),
                name='layer_scale'
            )
        else:
            self.layer_scale = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention block and extract spatial dimensions."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input, got {len(input_shape)}D")

        batch_size, height, width, channels = input_shape

        if channels != self.dim:
            raise ValueError(f"Input channels ({channels}) must match dim ({self.dim})")

        # Store spatial dimensions
        self.height = height
        self.width = width

        # Calculate sequence length for transformer
        if height is not None and width is not None:
            seq_length = height * width
        else:
            seq_length = None

        # Build transformer with flattened input shape
        transformer_input_shape = (batch_size, seq_length, channels)
        self.transformer.build(transformer_input_shape)

        # Build layer scale if present
        if self.layer_scale is not None:
            self.layer_scale.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through attention block."""
        # Get input shape
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1] if self.height is None else self.height
        width = input_shape[2] if self.width is None else self.width

        # Flatten spatial dimensions: [B, H, W, C] -> [B, H*W, C]
        x = ops.reshape(inputs, (batch_size, height * width, self.dim))

        # Apply transformer
        x = self.transformer(x, training=training)

        # Reshape back to spatial: [B, H*W, C] -> [B, H, W, C]
        x = ops.reshape(x, (batch_size, height, width, self.dim))

        # Apply layer scale if present
        if self.layer_scale is not None:
            x = self.layer_scale(x, training=training)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'attention_type': self.attention_type,
            'normalization_position': self.normalization_position,
            'dropout_rate': self.dropout_rate,
            'use_layer_scale': self.use_layer_scale,
            'layer_scale_init': self.layer_scale_init,
        })
        return config


@keras.saving.register_keras_serializable()
class FastVLM(keras.Model):
    """
    FastVLM: A fast hybrid vision model combining efficient convolutions and transformers.

    This model implements the FastVLM architecture that efficiently combines
    convolutional feature extraction, efficient mixing operations (RepMixer),
    and transformer-based attention for high-performance vision tasks while
    maintaining computational efficiency.

    **Intent**: Provide a state-of-the-art vision model that achieves excellent
    performance on image classification and other vision tasks while being
    suitable for both research and production deployment scenarios.

    **Architecture**:
    ```
    Input(Image: [H, W, 3])
            ↓
    ConvolutionalStem → [H/4, W/4, embed_dims[0]]
            ↓
    Stage 1: RepMixer Blocks → Downsample → [H/8, W/8, embed_dims[1]]
            ↓
    Stage 2: RepMixer Blocks → Downsample → [H/16, W/16, embed_dims[2]]
            ↓
    Stage 3: Attention Blocks (no downsampling)
            ↓
    Classification Head: GAP → Dense → Logits
    ```

    **Design Principles**:
    - **Hierarchical Feature Extraction**: Progressive spatial reduction with increasing channels
    - **Efficient Early Stages**: RepMixer blocks for efficient local feature mixing
    - **Global Context**: Transformer attention in later stages for global understanding
    - **Flexible Architecture**: Configurable depths and dimensions for different model sizes

    **Model Variants**:
    - **FastVLM-Tiny**: Minimal model for edge deployment
    - **FastVLM-Small**: Balanced model for mobile applications
    - **FastVLM-Base**: Standard model for general use
    - **FastVLM-Large**: High-capacity model for demanding tasks

    Args:
        num_classes: Integer, number of output classes for classification.
            Must be positive. Use 0 for feature extraction only.
        embed_dims: List of integers, feature dimensions for each stage.
            Must have 3 elements for the 3 stages. All values must be positive.
            Defaults to [64, 128, 256].
        depths: List of integers, number of blocks in each stage.
            Must have 3 elements. All values must be non-negative.
            Defaults to [3, 4, 6].
        use_se: Boolean, whether to use Squeeze-and-Excitation in MobileOne blocks.
            Defaults to False.
        attention_type: String, type of attention mechanism for attention blocks.
            Options: 'multi_head_attention', 'window_attention', 'group_query_attention'.
            Defaults to 'multi_head_attention'.
        num_heads: List of integers, number of attention heads for each stage.
            Must have 3 elements. Values must be positive and divide corresponding embed_dims.
            If None, defaults to [dim//32 for dim in embed_dims] with minimum of 1.
        mlp_ratio: Float, expansion ratio for MLP in transformer and RepMixer blocks.
            Must be positive. Defaults to 4.0.
        dropout_rate: Float, dropout rate applied throughout the model.
            Must be between 0 and 1. Defaults to 0.0.
        drop_path_rate: Float, stochastic depth rate for regularization.
            Must be between 0 and 1. Defaults to 0.1.
        use_layer_scale: Boolean, whether to use layer scaling in attention blocks.
            Defaults to True.
        activation: String or callable, activation function used throughout.
            Defaults to 'gelu'.
        kernel_initializer: String or initializer, initializer for conv kernels.
            Defaults to 'he_normal'.
        **kwargs: Additional keyword arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, 3)`
        Typically expects RGB images of size 224x224 or similar.

    Output shape:
        - If num_classes > 0: 2D tensor with shape `(batch_size, num_classes)`
        - If num_classes = 0: 4D tensor with shape `(batch_size, H/16, W/16, embed_dims[-1])`

    Attributes:
        stem: ConvolutionalStem for initial feature extraction.
        stages: List of stage blocks (RepMixer + Downsample or Attention).
        head: Classification head (GlobalAveragePooling + Dense) or None.
        downsample_layers: List of downsampling layers between stages.

    Methods:
        reparameterize(): Optimize the model for inference by fusing operations.
        reset_reparameterization(): Reset to training mode.
        extract_features(): Get intermediate feature maps from all stages.

    Example:
        ```python
        # Standard ImageNet classification
        model = FastVLM(num_classes=1000)
        model.compile(optimizer='adamw', loss='categorical_crossentropy')

        # Tiny model for mobile deployment
        model = FastVLM(
            num_classes=10,
            embed_dims=[32, 64, 128],
            depths=[2, 3, 4],
            dropout_rate=0.1
        )

        # Feature extraction backbone
        backbone = FastVLM(
            num_classes=0,  # No classification head
            embed_dims=[96, 192, 384],
            depths=[3, 6, 9]
        )
        features = backbone(images)  # Shape: [B, H/16, W/16, 384]

        # Custom attention and large model
        model = FastVLM(
            num_classes=1000,
            embed_dims=[128, 256, 512],
            depths=[4, 6, 8],
            attention_type='window_attention',
            num_heads=[4, 8, 16],
            mlp_ratio=6.0,
            use_se=True
        )

        # Reparameterize for inference
        model.reparameterize()  # Optimize for deployment
        predictions = model(test_images)
        ```

    Note:
        For best performance, call reparameterize() after training to fuse
        the MobileOne blocks for faster inference. The model supports both
        classification and feature extraction modes.

    References:
        FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization
        MobileOne: An Improved One millisecond Mobile Backbone
    """

    def __init__(
            self,
            num_classes: int,
            embed_dims: List[int] = [64, 128, 256],
            depths: List[int] = [3, 4, 6],
            use_se: bool = False,
            attention_type: str = 'multi_head_attention',
            num_heads: Optional[List[int]] = None,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            use_layer_scale: bool = True,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_classes < 0:
            raise ValueError(f"num_classes must be non-negative, got {num_classes}")
        if len(embed_dims) != 3:
            raise ValueError(f"embed_dims must have 3 elements, got {len(embed_dims)}")
        if len(depths) != 3:
            raise ValueError(f"depths must have 3 elements, got {len(depths)}")
        if any(dim <= 0 for dim in embed_dims):
            raise ValueError("All embed_dims must be positive")
        if any(depth < 0 for depth in depths):
            raise ValueError("All depths must be non-negative")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= drop_path_rate <= 1.0):
            raise ValueError(f"drop_path_rate must be between 0 and 1, got {drop_path_rate}")

        # Set default num_heads if not provided
        if num_heads is None:
            num_heads = [max(1, dim // 32) for dim in embed_dims]
        elif len(num_heads) != 3:
            raise ValueError(f"num_heads must have 3 elements, got {len(num_heads)}")

        # Validate num_heads divisibility
        for i, (dim, heads) in enumerate(zip(embed_dims, num_heads)):
            if heads <= 0:
                raise ValueError(f"All num_heads must be positive, got {heads} at index {i}")
            if dim % heads != 0:
                raise ValueError(
                    f"embed_dims[{i}] ({dim}) must be divisible by num_heads[{i}] ({heads})"
                )

        # Store configuration
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depths = depths
        self.use_se = use_se
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.use_layer_scale = use_layer_scale
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        # CREATE model components
        self._build_model()

    def _build_model(self) -> None:
        """Build the FastVLM model architecture."""
        # Import here to avoid circular imports

        # Convolutional stem
        self.stem = ConvolutionalStem(
            out_channels=self.embed_dims[0],
            use_se=self.use_se,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            name='stem'
        )

        # Create stages
        self.stages = []
        self.downsample_layers = []

        # Stage 1: RepMixer blocks + downsample
        stage1_blocks = []
        for i in range(self.depths[0]):
            stage1_blocks.append(
                RepMixerBlock(
                    dim=self.embed_dims[0],
                    expansion_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    name=f'stage1_block_{i}'
                )
            )
        self.stages.append(keras.Sequential(stage1_blocks, name='stage1'))

        # Downsample 1->2
        self.downsample_layers.append(
            layers.Conv2D(
                filters=self.embed_dims[1],
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                name='downsample_1_2'
            )
        )

        # Stage 2: RepMixer blocks + downsample
        stage2_blocks = []
        for i in range(self.depths[1]):
            stage2_blocks.append(
                RepMixerBlock(
                    dim=self.embed_dims[1],
                    expansion_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    name=f'stage2_block_{i}'
                )
            )
        self.stages.append(keras.Sequential(stage2_blocks, name='stage2'))

        # Downsample 2->3
        self.downsample_layers.append(
            layers.Conv2D(
                filters=self.embed_dims[2],
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                name='downsample_2_3'
            )
        )

        # Stage 3: Attention blocks (no downsampling)
        stage3_blocks = []
        for i in range(self.depths[2]):
            # Calculate stochastic depth rate for this block
            block_drop_rate = self.drop_path_rate * (
                    (sum(self.depths[:2]) + i) / (sum(self.depths) - 1)
            ) if sum(self.depths) > 1 else 0.0

            stage3_blocks.append(
                AttentionBlock(
                    dim=self.embed_dims[2],
                    num_heads=self.num_heads[2],
                    mlp_ratio=self.mlp_ratio,
                    attention_type=self.attention_type,
                    dropout_rate=max(self.dropout_rate, block_drop_rate),
                    use_layer_scale=self.use_layer_scale,
                    name=f'stage3_attention_{i}'
                )
            )
        self.stages.append(keras.Sequential(stage3_blocks, name='stage3'))

        # Classification head
        if self.num_classes > 0:
            head_layers = [
                layers.GlobalAveragePooling2D(name='gap'),
                layers.Dense(
                    self.num_classes,
                    kernel_initializer=self.kernel_initializer,
                    name='classifier'
                )
            ]

            if self.dropout_rate > 0.0:
                head_layers.insert(-1, layers.Dropout(self.dropout_rate, name='head_dropout'))

            self.head = keras.Sequential(head_layers, name='classification_head')
        else:
            self.head = None

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through FastVLM."""
        # Stem: [H, W, 3] -> [H/4, W/4, embed_dims[0]]
        x = self.stem(inputs, training=training)

        # Stage 1: RepMixer blocks
        x = self.stages[0](x, training=training)

        # Downsample 1->2: [H/4, W/4, embed_dims[0]] -> [H/8, W/8, embed_dims[1]]
        x = self.downsample_layers[0](x, training=training)

        # Stage 2: RepMixer blocks
        x = self.stages[1](x, training=training)

        # Downsample 2->3: [H/8, W/8, embed_dims[1]] -> [H/16, W/16, embed_dims[2]]
        x = self.downsample_layers[1](x, training=training)

        # Stage 3: Attention blocks
        x = self.stages[2](x, training=training)

        # Classification head (if present)
        if self.head is not None:
            x = self.head(x, training=training)

        return x

    def extract_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """
        Extract intermediate feature maps from all stages.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, 3].
            training: Whether in training mode.

        Returns:
            List of feature tensors from stem and each stage:
            - features[0]: Stem output [B, H/4, W/4, embed_dims[0]]
            - features[1]: Stage 1 output [B, H/4, W/4, embed_dims[0]]
            - features[2]: Stage 2 output [B, H/8, W/8, embed_dims[1]]
            - features[3]: Stage 3 output [B, H/16, W/16, embed_dims[2]]
        """
        features = []

        # Stem
        x = self.stem(inputs, training=training)
        features.append(x)

        # Stage 1
        x = self.stages[0](x, training=training)
        features.append(x)

        # Downsample and Stage 2
        x = self.downsample_layers[0](x, training=training)
        x = self.stages[1](x, training=training)
        features.append(x)

        # Downsample and Stage 3
        x = self.downsample_layers[1](x, training=training)
        x = self.stages[2](x, training=training)
        features.append(x)

        return features

    def reparameterize(self) -> None:
        """
        Reparameterize the model for efficient inference.

        This method optimizes all MobileOne blocks in the stem by fusing
        their training-time branches into single convolutions, significantly
        improving inference speed while maintaining identical outputs.
        """
        logger.info("Reparameterizing FastVLM model for inference")

        # Reparameterize stem
        if hasattr(self.stem, 'reparameterize'):
            self.stem.reparameterize()

        # Reparameterize MobileOne blocks in RepMixer stages (stages 1 and 2)
        for stage_idx in [0, 1]:  # Only RepMixer stages, not attention stage
            stage = self.stages[stage_idx]
            for layer in stage.layers:
                if hasattr(layer, 'reparameterize'):
                    layer.reparameterize()

        logger.info("FastVLM reparameterization complete")

    def reset_reparameterization(self) -> None:
        """Reset the model to training mode with multi-branch architecture."""
        logger.info("Resetting FastVLM reparameterization")

        # Reset stem
        if hasattr(self.stem, 'reset_reparameterization'):
            self.stem.reset_reparameterization()

        # Reset MobileOne blocks
        for stage_idx in [0, 1]:
            stage = self.stages[stage_idx]
            for layer in stage.layers:
                if hasattr(layer, 'reset_reparameterization'):
                    layer.reset_reparameterization()

        logger.info("FastVLM reparameterization reset complete")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'embed_dims': self.embed_dims,
            'depths': self.depths,
            'use_se': self.use_se,
            'attention_type': self.attention_type,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'use_layer_scale': self.use_layer_scale,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
        })
        return config


# Factory functions for common FastVLM variants
def create_fastvlm_tiny(num_classes: int = 1000, **kwargs) -> FastVLM:
    """Create FastVLM-Tiny model for edge deployment."""
    return FastVLM(
        num_classes=num_classes,
        embed_dims=[32, 64, 128],
        depths=[2, 3, 4],
        num_heads=[1, 2, 4],
        mlp_ratio=3.0,
        dropout_rate=0.0,
        **kwargs
    )


def create_fastvlm_small(num_classes: int = 1000, **kwargs) -> FastVLM:
    """Create FastVLM-Small model for mobile applications."""
    return FastVLM(
        num_classes=num_classes,
        embed_dims=[48, 96, 192],
        depths=[3, 4, 6],
        num_heads=[2, 3, 6],
        mlp_ratio=4.0,
        dropout_rate=0.1,
        **kwargs
    )


def create_fastvlm_base(num_classes: int = 1000, **kwargs) -> FastVLM:
    """Create FastVLM-Base model for general use."""
    return FastVLM(
        num_classes=num_classes,
        embed_dims=[64, 128, 256],
        depths=[3, 4, 6],
        num_heads=[2, 4, 8],
        mlp_ratio=4.0,
        dropout_rate=0.1,
        drop_path_rate=0.1,
        **kwargs
    )


def create_fastvlm_large(num_classes: int = 1000, **kwargs) -> FastVLM:
    """Create FastVLM-Large model for demanding tasks."""
    return FastVLM(
        num_classes=num_classes,
        embed_dims=[96, 192, 384],
        depths=[4, 6, 8],
        num_heads=[3, 6, 12],
        mlp_ratio=4.0,
        dropout_rate=0.1,
        drop_path_rate=0.2,
        use_se=True,
        **kwargs
    )