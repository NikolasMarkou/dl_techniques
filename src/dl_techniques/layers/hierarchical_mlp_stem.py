"""
Hierarchical MLP (hMLP) Stem for Vision Transformers
==================================================

This module implements the hierarchical MLP stem as described in
"Three things everyone should know about Vision Transformers" by Touvron et al.

PAPER OVERVIEW:
--------------
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
from typing import Tuple, Optional, Union, Any, Dict

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

    Args:
        embed_dim: Final embedding dimension.
        img_size: Input image dimensions (height, width).
        patch_size: Final patch dimensions (height, width).
        in_channels: Number of input channels (3 for RGB).
        norm_layer: Normalization type ('batch' or 'layer').
        activation: Activation function name or callable.
        use_bias: Boolean, whether the convolution layers use bias vectors.
        kernel_initializer: Initializer for the convolution kernels.
        bias_initializer: Initializer for the bias vectors.
        kernel_regularizer: Optional regularizer for the convolution kernels.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            img_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            in_channels: int = 3,
            norm_layer: str = "batch",
            activation: Union[str, callable] = "gelu",
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the Hierarchical MLP Stem.

        Args:
            embed_dim: Final embedding dimension.
            img_size: Input image dimensions (height, width).
            patch_size: Final patch dimensions (height, width).
            in_channels: Number of input channels (3 for RGB).
            norm_layer: Normalization type ('batch' or 'layer').
            activation: Activation function name or callable.
            use_bias: Boolean, whether the convolution layers use bias vectors.
            kernel_initializer: Initializer for the convolution kernels.
            bias_initializer: Initializer for the bias vectors.
            kernel_regularizer: Optional regularizer for the convolution kernels.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

        # Store configuration parameters
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

        # Validate inputs
        self._validate_inputs()

        # Calculate intermediate dimensions
        self.dim1 = embed_dim // 4  # Dimension after first stage

        # Calculate the number of patches
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        self.num_patches = h_patches * w_patches

        # These will be initialized in build()
        self.stage1_conv = None
        self.stage2_conv = None
        self.stage3_conv = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None
        self.activation_fn = None
        self._build_input_shape = None

        logger.info(f"Initialized HierarchicalMLPStem with embed_dim={embed_dim}, "
                    f"img_size={img_size}, patch_size={patch_size}, num_patches={self.num_patches}")

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")

        if self.embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 4 for proper hierarchical processing, got {self.embed_dim}")

        if self.patch_size != (16, 16):
            raise ValueError(f"Current implementation only supports 16x16 patches, got {self.patch_size}")

        if self.img_size[0] % self.patch_size[0] != 0 or self.img_size[1] % self.patch_size[1] != 0:
            raise ValueError(f"Image size {self.img_size} must be divisible by patch size {self.patch_size}")

        if self.norm_layer not in ["batch", "layer"]:
            raise ValueError(f"Unsupported normalization layer: {self.norm_layer}")

        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")

    def _get_activation_fn(self, activation: Union[str, callable]) -> callable:
        """Get activation function from name or callable."""
        if callable(activation):
            return activation
        return keras.activations.get(activation)

    def build(self, input_shape) -> None:
        """
        Build the layer's sublayers.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Get activation function
        self.activation_fn = self._get_activation_fn(self.activation_name)

        # Create normalization layers
        if self.norm_layer == "batch":
            self.norm1 = keras.layers.BatchNormalization(name=f"{self.name}_bn1" if self.name else None)
            self.norm2 = keras.layers.BatchNormalization(name=f"{self.name}_bn2" if self.name else None)
            self.norm3 = keras.layers.BatchNormalization(name=f"{self.name}_bn3" if self.name else None)
        else:  # layer norm
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln1" if self.name else None)
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln2" if self.name else None)
            self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln3" if self.name else None)

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
            name=f"{self.name}_conv1" if self.name else None
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
            name=f"{self.name}_conv2" if self.name else None
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
            name=f"{self.name}_conv3" if self.name else None
        )

        # Build sublayers with appropriate shapes
        # Calculate intermediate shapes for building sublayers
        batch_size = input_shape[0]

        # Stage 1 input shape
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

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> Any:
        """
        Apply the hierarchical MLP stem to input images.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating whether the layer should behave in training mode.

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """
        x = inputs

        # Stage 1: 4x4 patches
        x = self.stage1_conv(x)
        x = self.norm1(x, training=training)
        x = self.activation_fn(x)

        # Stage 2: 8x8 patches (4x4 + 2x2 processing)
        x = self.stage2_conv(x)
        x = self.norm2(x, training=training)
        x = self.activation_fn(x)

        # Stage 3: 16x16 patches (8x8 + 2x2 processing)
        x = self.stage3_conv(x)
        x = self.norm3(x, training=training)

        # Reshape from [B, H, W, C] to [B, HW, C] using keras ops
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]
        channels = ops.shape(x)[3]

        x = ops.reshape(x, [batch_size, height * width, channels])

        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple.
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
        Returns the layer configuration for serialization.

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

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
