import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.layer_scale import LearnableMultiplier, MultiplierType

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AttentionBlock(keras.layers.Layer):
    """
    Attention block with vision-specific adaptations.

    This layer wraps the TransformerLayer with
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

# ---------------------------------------------------------------------
