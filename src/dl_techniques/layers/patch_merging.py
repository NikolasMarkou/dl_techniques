"""
Downsamples feature maps by merging adjacent patches.

This layer implements the patch merging technique from the Swin Transformer
paper, which reduces spatial dimensions while increasing the channel dimension
to create hierarchical feature representations.

The layer reshapes an input tensor of shape `(B, H, W, C)` into a tensor
of shape `(B, H/2, W/2, 2*C)`. It does this by first concatenating the
features from non-overlapping 2x2 patches, which results in a shape of
`(B, H/2, W/2, 4*C)`. A `LayerNormalization` is applied, followed by a
`Dense` layer that projects the features from `4*C` to `2*C`. This layer
can automatically handle odd input resolutions by padding.
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PatchMerging(keras.layers.Layer):
    """
    Patch merging layer for hierarchical downsampling in Swin Transformer architectures.

    This layer performs spatial downsampling by merging 2x2 patches while doubling the
    feature dimension. It implements the standard patch merging operation from the
    Swin Transformer paper, combining adjacent patches and projecting them through
    a linear transformation for efficient multi-scale feature learning.

    **Intent**: Provide efficient spatial downsampling between Swin Transformer stages
    while maintaining feature information through dimension expansion. This enables
    hierarchical feature learning with progressively larger receptive fields.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C])
           ↓
    Extract 2x2 Patches: [x0, x1, x2, x3] from adjacent locations
           ↓
    Concatenate: (batch, H/2, W/2, 4*C)
           ↓
    LayerNormalization(4*C features)
           ↓
    Linear Projection: Dense(2*C)
           ↓
    Output(shape=[batch, H/2, W/2, 2*C])
    ```

    **Mathematical Operation**:
    1. **Patch Extraction**: Extract 2x2 neighborhoods from input feature map
    2. **Concatenation**: Combine patches along channel dimension (4C channels)
    3. **Normalization**: Apply layer normalization for training stability
    4. **Projection**: Linear transformation to reduce from 4C to 2C channels

    This creates a 2x spatial downsampling with 2x feature dimension increase.

    Args:
        dim: Integer, input dimension (number of channels). Must be positive.
            This represents the channel dimension of the input feature map.
        use_bias: Boolean, whether to use bias in the linear projection.
            When False, helps reduce parameters. Defaults to False.
        kernel_initializer: String or Initializer, initializer for projection kernel.
            Controls weight initialization strategy. Defaults to "glorot_uniform".
        bias_initializer: String or Initializer, bias initializer if use_bias=True.
            Only used when use_bias=True. Defaults to "zeros".
        kernel_regularizer: Optional Regularizer, regularization for projection weights.
            Helps prevent overfitting in the linear projection. Defaults to None.
        bias_regularizer: Optional Regularizer, regularization for bias if use_bias=True.
            Only applied when use_bias=True. Defaults to None.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        4D tensor: `(batch_size, height, width, dim)`
        Height and width should ideally be even for optimal patch merging,
        but odd dimensions are handled through padding.

    Output shape:
        4D tensor: `(batch_size, height//2, width//2, dim*2)`
        Spatial dimensions are halved, feature dimension is doubled.

    Attributes:
        norm: LayerNormalization layer for feature normalization after concatenation.
        reduction: Dense layer for projecting from 4*dim to 2*dim channels.

    Example:
        ```python
        # Standard patch merging for 96-channel input
        merge = PatchMerging(dim=96)
        inputs = keras.Input(shape=(56, 56, 96))
        outputs = merge(inputs)  # Shape: (batch, 28, 28, 192)

        # Patch merging without bias for parameter efficiency
        merge = PatchMerging(dim=192, use_bias=False)

        # With regularization for large models
        merge = PatchMerging(
            dim=384,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    Raises:
        ValueError: If dim is not positive.

    Note:
        For odd spatial dimensions, the layer automatically applies padding to
        ensure proper 2x2 patch extraction. This maintains compatibility with
        various input sizes while preserving the hierarchical structure.
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

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        # Store ALL configuration parameters
        self.dim = dim
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.norm = layers.LayerNormalization(
            epsilon=1e-5,
            name="norm"
        )

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
        Forward pass of patch merging operation.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, dim).
            training: Boolean indicating training mode for normalization and dropout.

        Returns:
            Output tensor of shape (batch_size, height//2, width//2, dim*2).
        """
        B, H, W, C = ops.shape(inputs)[0], ops.shape(inputs)[1], ops.shape(inputs)[2], ops.shape(inputs)[3]

        # Handle odd dimensions by padding
        if H % 2 == 1 or W % 2 == 1:
            pad_h = H % 2
            pad_w = W % 2
            inputs = ops.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
            H, W = H + pad_h, W + pad_w

        # Extract 2x2 patches and concatenate
        x0 = inputs[:, 0::2, 0::2, :]  # Top-left
        x1 = inputs[:, 1::2, 0::2, :]  # Bottom-left
        x2 = inputs[:, 0::2, 1::2, :]  # Top-right
        x3 = inputs[:, 1::2, 1::2, :]  # Bottom-right

        # Concatenate patches: (B, H//2, W//2, 4*C)
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)

        # Apply normalization and projection
        x = self.norm(x, training=training)
        x = self.reduction(x, training=training)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape for shape inference."""
        batch_size, height, width, channels = input_shape
        output_height = None if height is None else (height + 1) // 2
        output_width = None if width is None else (width + 1) // 2
        output_channels = self.dim * 2
        return (batch_size, output_height, output_width, output_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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

# ---------------------------------------------------------------------
