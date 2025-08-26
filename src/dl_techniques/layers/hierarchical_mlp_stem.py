"""
Hierarchical MLP (hMLP) Stem for Vision Transformers

This module implements the hierarchical MLP stem as described in
"Three things everyone should know about Vision Transformers" by Touvron et al.

The paper introduces three key insights about Vision Transformers:
1. Parallelizing ViT layers can improve efficiency without affecting accuracy
2. Fine-tuning only attention layers is sufficient for adaptation
3. Using hierarchical MLP stems improves compatibility with masked self-supervised learning

HMLP STEM DESIGN:
---------------
The hMLP stem is a patch pre-processing technique that:
- Processes each patch independently (no information leakage between patches)
- Progressively processes patches from 2×2 → 4×4 → 8×8 → 16×16
- Uses linear projections with normalization and non-linearity at each stage
- Has minimal computational overhead (<1% increase in FLOPs vs. standard ViT)

KEY ADVANTAGES:
-------------
1. Compatible with Masked Self-supervised Learning:
   - Unlike conventional convolutional stems which cause information leakage between patches
   - Works with BeiT, MAE, and other mask-based approaches
   - Masking can be applied either before or after the stem with identical results

2. Performance Benefits:
   - Supervised learning: ~0.3% accuracy improvement over standard ViT
   - BeiT pre-training: +0.4% accuracy improvement over linear projection
   - On par with the best convolutional stems for supervised learning

3. Implementation:
   - Uses convolutions with matching kernel size and stride for efficiency
   - Each patch is processed independently despite using convolutional layers
   - Works with both BatchNorm (better performance) and LayerNorm (stable for small batches)
"""

import keras
from keras import ops
from typing import Tuple, Optional, Union, Any, Dict, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalMLPStem(keras.layers.Layer):
    """
    Hierarchical MLP stem for Vision Transformers with patch-independent processing.

    This layer implements the hMLP stem that processes image patches through a sequence
    of hierarchical transformations without cross-patch information leakage. It progressively
    processes patches from 4×4 to 16×16 resolution using linear projections, normalization,
    and activation functions, making it compatible with masked self-supervised learning.

    **Intent**: Provide a patch embedding method that improves over standard linear projection
    while maintaining compatibility with masking-based pre-training methods like BeiT and MAE.
    The hierarchical processing enhances feature learning without computational overhead.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    Stage 1: Conv2D(dim1, kernel=4, stride=4) → Norm → Activation
           ↓ (processes 4×4 patches independently)
    Stage 2: Conv2D(dim1, kernel=2, stride=2) → Norm → Activation
           ↓ (processes 2×2 within each 4×4 patch)
    Stage 3: Conv2D(embed_dim, kernel=2, stride=2) → Norm
           ↓ (final 2×2 to complete 16×16 patches)
    Output(shape=[batch, num_patches, embed_dim])
    ```

    **Mathematical Operations**:
    1. **Stage 1**: x₁ = activation(norm(conv₁(x))) with kernel=4, stride=4
    2. **Stage 2**: x₂ = activation(norm(conv₂(x₁))) with kernel=2, stride=2
    3. **Stage 3**: x₃ = norm(conv₃(x₂)) with kernel=2, stride=2
    4. **Reshape**: output = reshape(x₃, [batch, H×W/256, embed_dim])

    Where each convolution processes patches independently, avoiding information leakage
    that would interfere with masked pre-training approaches.

    Args:
        embed_dim: Integer, final embedding dimension for each patch. Must be positive
            and divisible by 4 for hierarchical processing. Defaults to 768.
        img_size: Tuple[int, int], input image dimensions as (height, width).
            Both dimensions must be divisible by patch_size. Defaults to (224, 224).
        patch_size: Tuple[int, int], final patch dimensions as (height, width).
            Currently only supports (16, 16) for compatibility with standard ViT.
            Defaults to (16, 16).
        in_channels: Integer, number of input image channels. Must be positive.
            Typically 3 for RGB images. Defaults to 3.
        norm_layer: Literal['batch', 'layer'], type of normalization to apply.
            'batch' provides better performance, 'layer' is more stable for small batches
            and distributed training. Defaults to 'batch'.
        activation: Union[str, Callable], activation function for first two stages.
            Can be string name ('gelu', 'relu') or callable. Defaults to 'gelu'.
        use_bias: Boolean, whether convolution layers include bias parameters.
            Defaults to True.
        kernel_initializer: Union[str, keras.initializers.Initializer], initializer
            for convolution kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: Union[str, keras.initializers.Initializer], initializer
            for bias parameters when use_bias=True. Defaults to 'zeros'.
        kernel_regularizer: Optional[keras.regularizers.Regularizer], regularizer
            applied to convolution kernels for preventing overfitting. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        - height and width must be divisible by patch_size components
        - channels must match in_channels parameter

    Output shape:
        3D tensor with shape: `(batch_size, num_patches, embed_dim)`
        where num_patches = (height // 16) * (width // 16)

    Attributes:
        stage1_conv: First Conv2D layer processing 4×4 patches.
        stage2_conv: Second Conv2D layer processing 2×2 within 4×4.
        stage3_conv: Third Conv2D layer completing 16×16 patches.
        norm1: Normalization layer after first convolution.
        norm2: Normalization layer after second convolution.
        norm3: Normalization layer after third convolution.
        activation_fn: Activation function used in stages 1 and 2.
        num_patches: Total number of patches produced.
        dim1: Intermediate dimension (embed_dim // 4).

    Example:
        ```python
        # Standard ViT-B configuration for ImageNet
        stem = HierarchicalMLPStem(embed_dim=768)
        inputs = keras.Input(shape=(224, 224, 3))
        patches = stem(inputs)  # Shape: (batch, 196, 768)

        # Custom configuration with Layer Normalization
        stem = HierarchicalMLPStem(
            embed_dim=512,
            img_size=(256, 256),
            norm_layer='layer',
            activation='relu'
        )

        # Smaller model for mobile applications
        stem = HierarchicalMLPStem(
            embed_dim=384,
            img_size=(128, 128),
            in_channels=3,
            norm_layer='batch'
        )
        ```

    References:
        Three things everyone should know about Vision Transformers.
        Hugo Touvron et al., 2022.
        https://arxiv.org/abs/2203.09795

    Raises:
        ValueError: If embed_dim is not positive or not divisible by 4.
        ValueError: If patch_size is not (16, 16) - current implementation limitation.
        ValueError: If img_size dimensions are not divisible by patch_size.
        ValueError: If norm_layer is not 'batch' or 'layer'.
        ValueError: If in_channels is not positive.

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and built explicitly in build() for robust serialization.
        Currently optimized for 16×16 patches; other patch sizes may require modifications.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        norm_layer: Literal['batch', 'layer'] = 'batch',
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate all inputs with descriptive error messages
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 4 for hierarchical processing, got {embed_dim}"
            )
        if patch_size != (16, 16):
            raise ValueError(
                f"Current implementation only supports 16×16 patches, got {patch_size}"
            )
        if img_size[0] % patch_size[0] != 0 or img_size[1] % patch_size[1] != 0:
            raise ValueError(
                f"Image size {img_size} must be divisible by patch size {patch_size}"
            )
        if norm_layer not in ['batch', 'layer']:
            raise ValueError(f"norm_layer must be 'batch' or 'layer', got {norm_layer}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        # Store ALL configuration parameters for serialization
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm_layer = norm_layer
        self.activation_name = activation if isinstance(activation, str) else 'custom'
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Calculate derived values
        self.dim1 = embed_dim // 4  # Intermediate dimension
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        self.num_patches = h_patches * w_patches

        # Store activation function
        self.activation_fn = keras.activations.get(activation)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        # Stage 1: Initial 4×4 patch processing
        self.stage1_conv = keras.layers.Conv2D(
            filters=self.dim1,
            kernel_size=4,
            strides=4,
            padding='valid',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stage1_conv'
        )

        # Stage 2: 2×2 processing within 4×4 patches
        self.stage2_conv = keras.layers.Conv2D(
            filters=self.dim1,
            kernel_size=2,
            strides=2,
            padding='valid',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stage2_conv'
        )

        # Stage 3: Final 2×2 to complete 16×16 patches
        self.stage3_conv = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=2,
            strides=2,
            padding='valid',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stage3_conv'
        )

        # Create normalization layers based on configuration
        if self.norm_layer == 'batch':
            self.norm1 = keras.layers.BatchNormalization(name='norm1')
            self.norm2 = keras.layers.BatchNormalization(name='norm2')
            self.norm3 = keras.layers.BatchNormalization(name='norm3')
        else:  # layer normalization
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name='norm1')
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name='norm2')
            self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6, name='norm3')

        logger.info(
            f"Initialized HierarchicalMLPStem: embed_dim={embed_dim}, "
            f"img_size={img_size}, patch_size={patch_size}, "
            f"num_patches={self.num_patches}, dim1={self.dim1}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        This method explicitly builds each sub-layer for robust serialization,
        following the modern Keras 3 pattern. Each sub-layer is built with the
        appropriate input shape computed from the forward pass.

        Args:
            input_shape: Shape tuple of the input tensor as (batch, height, width, channels).
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channels {input_shape[-1]} don't match expected {self.in_channels}"
            )

        # Calculate intermediate shapes for building sub-layers
        batch_size = input_shape[0]

        # Stage 1: Input is original image shape
        stage1_input_shape = input_shape
        self.stage1_conv.build(stage1_input_shape)

        # Stage 1 output: [batch, height//4, width//4, dim1]
        stage1_output_shape = (
            batch_size,
            input_shape[1] // 4,
            input_shape[2] // 4,
            self.dim1
        )
        self.norm1.build(stage1_output_shape)

        # Stage 2: Uses stage 1 output
        self.stage2_conv.build(stage1_output_shape)

        # Stage 2 output: [batch, height//8, width//8, dim1]
        stage2_output_shape = (
            batch_size,
            input_shape[1] // 8,
            input_shape[2] // 8,
            self.dim1
        )
        self.norm2.build(stage2_output_shape)

        # Stage 3: Uses stage 2 output
        self.stage3_conv.build(stage2_output_shape)

        # Stage 3 output: [batch, height//16, width//16, embed_dim]
        stage3_output_shape = (
            batch_size,
            input_shape[1] // 16,
            input_shape[2] // 16,
            self.embed_dim
        )
        self.norm3.build(stage3_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply hierarchical MLP stem to input images.

        Performs three-stage hierarchical patch processing:
        1. Stage 1: 4×4 patch extraction with normalization and activation
        2. Stage 2: 2×2 sub-patch processing with normalization and activation
        3. Stage 3: Final 2×2 processing with normalization only

        Each stage processes patches independently to avoid information leakage
        that would interfere with masked pre-training methods.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating training mode for normalization layers.

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim] containing
            hierarchically processed patch embeddings.
        """
        x = inputs

        # Stage 1: Process 4×4 patches independently
        x = self.stage1_conv(x, training=training)
        x = self.norm1(x, training=training)
        x = self.activation_fn(x)

        # Stage 2: Process 2×2 sub-patches within each 4×4 patch
        x = self.stage2_conv(x, training=training)
        x = self.norm2(x, training=training)
        x = self.activation_fn(x)

        # Stage 3: Final 2×2 processing to complete 16×16 patches
        x = self.stage3_conv(x, training=training)
        x = self.norm3(x, training=training)
        # Note: No activation after final stage

        # Reshape from [batch, height, width, channels] to [batch, num_patches, embed_dim]
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]
        channels = ops.shape(x)[3]

        # Flatten spatial dimensions: num_patches = height * width
        x = ops.reshape(x, [batch_size, height * width, channels])

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple [batch_size, num_patches, embed_dim].
        """
        batch_size = input_shape[0]

        # Calculate number of patches after 16×16 patch extraction
        h_patches = input_shape[1] // self.patch_size[0]
        w_patches = input_shape[2] // self.patch_size[1]
        num_patches = h_patches * w_patches

        return (batch_size, num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns all constructor parameters needed to recreate the layer,
        ensuring complete serialization compatibility.

        Returns:
            Dictionary containing the complete layer configuration.
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'norm_layer': self.norm_layer,
            'activation': self.activation_name,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
