"""
Hierarchical MLP (hMLP) Stem for Vision Transformers
==================================================

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

EXPERIMENTAL RESULTS:
------------------
From the paper:
- Supervised ViT-B with Linear stem: 82.2% top-1 accuracy on ImageNet
- Supervised ViT-B with hMLP stem: 82.5% top-1 accuracy
- BeiT+FT ViT-B with Linear stem: 83.1% top-1 accuracy
- BeiT+FT ViT-B with hMLP stem: 83.4% top-1 accuracy

When used with BeiT, existing convolutional stems show no improvement (83.0%)
while hMLP stem provides significant gains, demonstrating its effectiveness
for masked self-supervised learning approaches.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Union, Any, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalMLPStem(keras.layers.Layer):
    """
    Hierarchical MLP Stem for Vision Transformers.

    This stem processes patches independently through a sequence of linear projections,
    normalizations, and activations, gradually increasing the patch size from
    2×2 to 16×16 without any cross-patch communication.

    The implementation follows the modern Keras 3 pattern where all sub-layers
    are created in __init__ and built explicitly in build() for robust serialization.

    Mathematical formulation:
        Stage 1: x1 = Activation(Norm(Conv2D(x, kernel=4, stride=4)))
        Stage 2: x2 = Activation(Norm(Conv2D(x1, kernel=2, stride=2)))
        Stage 3: x3 = Norm(Conv2D(x2, kernel=2, stride=2))
        Output: Reshape(x3, [batch_size, num_patches, embed_dim])

    Args:
        embed_dim: Integer, final embedding dimension. Must be positive and divisible by 4.
            Defaults to 768.
        img_size: Tuple[int, int], input image dimensions (height, width).
            Must be divisible by patch_size. Defaults to (224, 224).
        patch_size: Tuple[int, int], final patch dimensions (height, width).
            Currently only supports (16, 16). Defaults to (16, 16).
        in_channels: Integer, number of input channels. Must be positive.
            Defaults to 3 for RGB images.
        norm_layer: String, normalization type. Must be 'batch' or 'layer'.
            'batch' provides better performance, 'layer' is more stable for small batches.
            Defaults to 'batch'.
        activation: String or callable, activation function name or callable.
            Applied after first two stages. Defaults to 'gelu'.
        use_bias: Boolean, whether the convolution layers use bias vectors.
            Defaults to True.
        kernel_initializer: String or Initializer, initializer for convolution kernels.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer, regularizer for convolution kernels.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        - height and width must be divisible by patch_size
        - channels must match in_channels parameter

    Output shape:
        3D tensor with shape: `(batch_size, num_patches, embed_dim)`
        where num_patches = (height // patch_size[0]) * (width // patch_size[1])

    Attributes:
        stage1_conv: First Conv2D layer (4x4 kernel, stride 4).
        stage2_conv: Second Conv2D layer (2x2 kernel, stride 2).
        stage3_conv: Third Conv2D layer (2x2 kernel, stride 2).
        norm1: Normalization layer after first convolution.
        norm2: Normalization layer after second convolution.
        norm3: Normalization layer after third convolution.
        activation_fn: Activation function used in first two stages.

    Example:
        ```python
        # Basic usage for ImageNet ViT-B
        stem = HierarchicalMLPStem(embed_dim=768)

        # Custom configuration for smaller images
        stem = HierarchicalMLPStem(
            embed_dim=384,
            img_size=(128, 128),
            norm_layer='layer',
            activation='relu'
        )

        # In a Vision Transformer
        inputs = keras.Input(shape=(224, 224, 3))
        patches = HierarchicalMLPStem(embed_dim=768)(inputs)
        # patches shape: (batch_size, 196, 768) for 224x224 input
        ```

    References:
        Three things everyone should know about Vision Transformers.
        Hugo Touvron et al., 2022.
        https://arxiv.org/abs/2203.09795

    Raises:
        ValueError: If embed_dim is not positive or not divisible by 4.
        ValueError: If patch_size is not (16, 16) - current limitation.
        ValueError: If img_size is not divisible by patch_size.
        ValueError: If norm_layer is not 'batch' or 'layer'.
        ValueError: If in_channels is not positive.

    Note:
        This implementation currently only supports 16x16 patches. The hierarchical
        processing creates patches through three stages: 4x4 → 8x8 → 16x16.
        Future versions may support other patch sizes.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            img_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            in_channels: int = 3,
            norm_layer: str = "batch",
            activation: Union[str, Callable] = "gelu",
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4 for proper hierarchical processing, got {embed_dim}")
        if patch_size != (16, 16):
            raise ValueError(f"Current implementation only supports 16x16 patches, got {patch_size}")
        if img_size[0] % patch_size[0] != 0 or img_size[1] % patch_size[1] != 0:
            raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")
        if norm_layer not in ["batch", "layer"]:
            raise ValueError(f"norm_layer must be 'batch' or 'layer', got {norm_layer}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        # Store ALL configuration parameters
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm_layer = norm_layer
        self.activation_name = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Calculate derived values
        self.dim1 = embed_dim // 4  # Dimension after first stage
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        self.num_patches = h_patches * w_patches

        # Get activation function
        self.activation_fn = keras.activations.get(activation)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        # Stage 1: Process 4x4 patches (kernel_size=4, stride=4)
        self.stage1_conv = keras.layers.Conv2D(
            filters=self.dim1,
            kernel_size=4,
            strides=4,
            padding="valid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stage1_conv"
        )

        # Stage 2: Process 2x2 patches within the previous patches (kernel_size=2, stride=2)
        self.stage2_conv = keras.layers.Conv2D(
            filters=self.dim1,
            kernel_size=2,
            strides=2,
            padding="valid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stage2_conv"
        )

        # Stage 3: Final 2x2 patches to get 16x16 total patch size (kernel_size=2, stride=2)
        self.stage3_conv = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=2,
            strides=2,
            padding="valid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stage3_conv"
        )

        # Create normalization layers
        if self.norm_layer == "batch":
            self.norm1 = keras.layers.BatchNormalization(name="norm1")
            self.norm2 = keras.layers.BatchNormalization(name="norm2")
            self.norm3 = keras.layers.BatchNormalization(name="norm3")
        else:  # layer norm
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")
            self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm3")

        logger.info(f"Initialized HierarchicalMLPStem with embed_dim={embed_dim}, "
                    f"img_size={img_size}, patch_size={patch_size}, num_patches={self.num_patches}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        This method explicitly builds each sub-layer for robust serialization,
        following the modern Keras 3 pattern.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Calculate intermediate shapes for building sub-layers
        batch_size = input_shape[0]

        # Stage 1: Conv2D input shape [B, H, W, C]
        stage1_input_shape = input_shape
        self.stage1_conv.build(stage1_input_shape)

        # Stage 1 output shape: [B, H//4, W//4, dim1]
        stage1_output_shape = (batch_size, input_shape[1] // 4, input_shape[2] // 4, self.dim1)
        self.norm1.build(stage1_output_shape)

        # Stage 2 processes the output of stage 1
        self.stage2_conv.build(stage1_output_shape)

        # Stage 2 output shape: [B, H//8, W//8, dim1]
        stage2_output_shape = (batch_size, input_shape[1] // 8, input_shape[2] // 8, self.dim1)
        self.norm2.build(stage2_output_shape)

        # Stage 3 processes the output of stage 2
        self.stage3_conv.build(stage2_output_shape)

        # Stage 3 output shape: [B, H//16, W//16, embed_dim]
        stage3_output_shape = (batch_size, input_shape[1] // 16, input_shape[2] // 16, self.embed_dim)
        self.norm3.build(stage3_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the hierarchical MLP stem to input images.

        Forward pass through three stages of hierarchical processing:
        1. 4x4 patches with normalization and activation
        2. 8x8 patches (2x2 within 4x4) with normalization and activation
        3. 16x16 patches (2x2 within 8x8) with normalization only

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating whether the layer should behave in training mode.

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """
        x = inputs

        # Stage 1: 4x4 patches → normalize → activate
        x = self.stage1_conv(x, training=training)
        x = self.norm1(x, training=training)
        x = self.activation_fn(x)

        # Stage 2: 8x8 patches (4x4 + 2x2 processing) → normalize → activate
        x = self.stage2_conv(x, training=training)
        x = self.norm2(x, training=training)
        x = self.activation_fn(x)

        # Stage 3: 16x16 patches (8x8 + 2x2 processing) → normalize only
        x = self.stage3_conv(x, training=training)
        x = self.norm3(x, training=training)

        # Reshape from [B, H, W, C] to [B, HW, C] for transformer input
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]
        channels = ops.shape(x)[3]

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
        input_shape_list = list(input_shape)
        batch_size = input_shape_list[0]

        # Calculate number of patches
        h_patches = input_shape_list[1] // self.patch_size[0]
        w_patches = input_shape_list[2] // self.patch_size[1]
        num_patches = h_patches * w_patches

        return tuple([batch_size, num_patches, self.embed_dim])

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns all constructor parameters needed to recreate the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "norm_layer": self.norm_layer,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------