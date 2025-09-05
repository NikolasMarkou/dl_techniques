"""
FastVLM Model Implementation

This module provides the FastVLM (Fast Vision Language Model) architecture,
a hybrid vision model that efficiently combines convolutional and transformer-based
components for high-performance image classification and feature extraction.
"""

import keras
from keras import layers, initializers
from typing import Optional, Union, List, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .components import AttentionBlock
from dl_techniques.layers.repmixer_block import RepMixerBlock, ConvolutionalStem

# ---------------------------------------------------------------------

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
    - **FastVLM-Nano**: Ultra-lightweight model for IoT and edge devices
    - **FastVLM-Tiny**: Minimal model for mobile deployment
    - **FastVLM-Small**: Balanced model for mobile applications
    - **FastVLM-Base**: Standard model for general use
    - **FastVLM-Large**: High-capacity model for demanding tasks
    - **FastVLM-Huge**: Maximum performance model for research

    Args:
        num_classes: Integer, number of output classes for classification.
            Must be positive. Use 0 for feature extraction only.
        embed_dims: List of integers, feature dimensions for each stage.
            Must have 3 elements for the 3 stages. All values must be positive.
            Defaults to [64, 128, 256].
        depths: List of integers, number of blocks in each stage.
            Must have 3 elements. All values must be non-negative.
            Defaults to [3, 4, 6].
        num_heads: List of integers, number of attention heads for each stage.
            Must have 3 elements. Values must be positive and divide corresponding embed_dims.
            If None, defaults to [dim//32 for dim in embed_dims] with minimum of 1.
        mlp_ratio: Float, expansion ratio for MLP in transformer and RepMixer blocks.
            Must be positive. Defaults to 4.0.
        dropout_rate: Float, dropout rate applied throughout the model.
            Must be between 0 and 1. Defaults to 0.0.
        drop_path_rate: Float, stochastic depth rate for regularization.
            Must be between 0 and 1. Defaults to 0.1.
        use_se: Boolean, whether to use Squeeze-and-Excitation in MobileOne blocks.
            Defaults to False.
        attention_type: String, type of attention mechanism for attention blocks.
            Options: 'multi_head_attention', 'window_attention', 'group_query_attention'.
            Defaults to 'multi_head_attention'.
        use_layer_scale: Boolean, whether to use layer scaling in attention blocks.
            Defaults to True.
        activation: String or callable, activation function used throughout.
            Defaults to 'gelu'.
        kernel_initializer: String or initializer, initializer for conv kernels.
            Defaults to 'he_normal'.
        include_top: Boolean, whether to include the classification head.
            Defaults to True.
        input_shape: Tuple, input shape. If None, defaults to (224, 224, 3).
        **kwargs: Additional keyword arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, 3)`
        Typically expects RGB images of size 224x224 or similar.

    Output shape:
        - If include_top=True and num_classes > 0: 2D tensor with shape `(batch_size, num_classes)`
        - If include_top=False: 4D tensor with shape `(batch_size, H/16, W/16, embed_dims[-1])`

    Attributes:
        stem: ConvolutionalStem for initial feature extraction.
        stages: List of stage blocks (RepMixer + Downsample or Attention).
        head: Classification head (GlobalAveragePooling + Dense) or None.
        downsample_layers: List of downsampling layers between stages.

    Methods:
        extract_features(): Get intermediate feature maps from all stages.

    Example:
        ```python
        # Standard ImageNet classification
        model = FastVLM.from_variant("base", num_classes=1000)
        model.compile(optimizer='adamw', loss='categorical_crossentropy')

        # Tiny model for mobile deployment
        model = FastVLM.from_variant("tiny", num_classes=10, input_shape=(224, 224, 3))

        # Feature extraction backbone
        backbone = FastVLM.from_variant("base", include_top=False)
        features = backbone(images)  # Shape: [B, H/16, W/16, 256]

        # Custom configuration
        model = FastVLM(
            num_classes=1000,
            embed_dims=[128, 256, 512],
            depths=[4, 6, 8],
            attention_type='window_attention',
            num_heads=[4, 8, 16],
            mlp_ratio=6.0,
            use_se=True
        )
        ```

    References:
        FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization
        MobileOne: An Improved One millisecond Mobile Backbone
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "nano": {
            "embed_dims": [24, 48, 96],
            "depths": [1, 2, 3],
            "num_heads": [1, 2, 3],
            "mlp_ratio": 2.0,
            "dropout_rate": 0.0,
            "drop_path_rate": 0.0,
            "use_se": False
        },
        "tiny": {
            "embed_dims": [32, 64, 128],
            "depths": [2, 3, 4],
            "num_heads": [1, 2, 4],
            "mlp_ratio": 3.0,
            "dropout_rate": 0.0,
            "drop_path_rate": 0.05,
            "use_se": False
        },
        "small": {
            "embed_dims": [48, 96, 192],
            "depths": [3, 4, 6],
            "num_heads": [2, 3, 6],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.1,
            "use_se": False
        },
        "base": {
            "embed_dims": [64, 128, 256],
            "depths": [3, 4, 6],
            "num_heads": [2, 4, 8],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.1,
            "use_se": False
        },
        "large": {
            "embed_dims": [96, 192, 384],
            "depths": [4, 6, 8],
            "num_heads": [3, 6, 12],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.2,
            "use_se": True
        },
        "huge": {
            "embed_dims": [128, 256, 512],
            "depths": [6, 8, 12],
            "num_heads": [4, 8, 16],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.3,
            "use_se": True
        }
    }

    def __init__(
            self,
            num_classes: int = 1000,
            embed_dims: List[int] = [64, 128, 256],
            depths: List[int] = [3, 4, 6],
            num_heads: Optional[List[int]] = None,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            use_se: bool = False,
            attention_type: str = 'multi_head_attention',
            use_layer_scale: bool = True,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
        # Set default input shape
        if input_shape is None:
            input_shape = (224, 224, 3)

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
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

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
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.use_se = use_se
        self.attention_type = attention_type
        self.use_layer_scale = use_layer_scale
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self._input_shape = input_shape

        # Create inputs
        inputs = keras.Input(shape=input_shape)

        # Build model
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created FastVLM model for input {input_shape} "
            f"with {sum(depths)} blocks total"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the FastVLM model architecture."""
        x = inputs

        # Convolutional stem
        self.stem = ConvolutionalStem(
            out_channels=self.embed_dims[0],
            use_se=self.use_se,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            name='stem'
        )
        x = self.stem(x, training=None)

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
        stage1 = keras.Sequential(stage1_blocks, name='stage1')
        self.stages.append(stage1)
        x = stage1(x, training=None)

        # Downsample 1->2
        downsample_1_2 = layers.Conv2D(
            filters=self.embed_dims[1],
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='downsample_1_2'
        )
        self.downsample_layers.append(downsample_1_2)
        x = downsample_1_2(x, training=None)

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
        stage2 = keras.Sequential(stage2_blocks, name='stage2')
        self.stages.append(stage2)
        x = stage2(x, training=None)

        # Downsample 2->3
        downsample_2_3 = layers.Conv2D(
            filters=self.embed_dims[2],
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='downsample_2_3'
        )
        self.downsample_layers.append(downsample_2_3)
        x = downsample_2_3(x, training=None)

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
        stage3 = keras.Sequential(stage3_blocks, name='stage3')
        self.stages.append(stage3)
        x = stage3(x, training=None)

        # Classification head
        if self.include_top and self.num_classes > 0:
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
            x = self.head(x, training=None)
        else:
            self.head = None

        return x

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

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any
    ) -> "FastVLM":
        """
        Create a FastVLM model from a predefined variant.

        Args:
            variant: String, one of "nano", "tiny", "small", "base", "large", "huge".
            num_classes: Integer, number of output classes for classification.
            input_shape: Tuple, input shape. If None, defaults to (224, 224, 3).
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            FastVLM model instance.

        Raises:
            ValueError: If variant is not recognized.

        Example:
            ```python
            # Create FastVLM-Base for ImageNet
            model = FastVLM.from_variant("base", num_classes=1000)

            # Create FastVLM-Tiny for CIFAR-10
            model = FastVLM.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

            # Feature extraction backbone
            backbone = FastVLM.from_variant("base", include_top=False)
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        logger.info(f"Creating FastVLM-{variant.upper()} model")
        logger.info(f"Configuration: {config}")

        # Override with any user-provided arguments
        config.update(kwargs)

        return cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            'num_classes': self.num_classes,
            'embed_dims': self.embed_dims,
            'depths': self.depths,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'use_se': self.use_se,
            'attention_type': self.attention_type,
            'use_layer_scale': self.use_layer_scale,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'include_top': self.include_top,
            'input_shape': self._input_shape,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVLM":
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            FastVLM model instance.
        """
        # Deserialize initializer if present
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )

        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        logger.info(f"FastVLM configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Embed dimensions: {self.embed_dims}")
        logger.info(f"  - Stage depths: {self.depths}")
        logger.info(f"  - Attention heads: {self.num_heads}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Use SE: {self.use_se}")
        logger.info(f"  - Attention type: {self.attention_type}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top and self.num_classes > 0:
            logger.info(f"  - Number of classes: {self.num_classes}")

# ---------------------------------------------------------------------