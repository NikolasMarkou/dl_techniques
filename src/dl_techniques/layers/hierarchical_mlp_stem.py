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

import math
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

    This layer implements a flexible hMLP stem that processes image patches through a sequence
    of hierarchical transformations without cross-patch information leakage. It dynamically
    creates stages to support various patch sizes (e.g., 8, 16, 32), making it compatible
    with diverse Vision Transformer architectures and masked self-supervised learning.

    **Intent**: Provide a patch embedding method that improves over standard linear projection
    while maintaining compatibility with masking-based pre-training methods like BeiT and MAE.
    The hierarchical processing enhances feature learning without computational overhead.

    **Architecture**:
    The architecture is created dynamically. It starts with a 4x4 convolution and adds
    2x2 convolutions until the target patch size is reached. For a 16x16 patch:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    Stage 1: Conv2D(dim1, kernel=4, stride=4) → Norm → Activation
           ↓ (processes 4×4 patches independently)
    Stage 2: Conv2D(dim1, kernel=2, stride=2) → Norm → Activation
           ↓ (processes 8x8 patches hierarchically)
    Stage 3: Conv2D(embed_dim, kernel=2, stride=2) → Norm
           ↓ (processes 16x16 patches hierarchically)
    Output(shape=[batch, num_patches, embed_dim])
    ```

    Args:
        embed_dim: Integer, final embedding dimension for each patch. Must be positive
            and divisible by 4. Defaults to 768.
        img_size: Tuple[int, int], input image dimensions as (height, width).
            Both dimensions must be divisible by patch_size. Defaults to (224, 224).
        patch_size: Tuple[int, int], final patch dimensions. Both dimensions must be
            equal, a power of two, and >= 4. Defaults to (16, 16).
        in_channels: Integer, number of input image channels. Must be positive.
            Typically 3 for RGB images. Defaults to 3.
        norm_layer: Literal['batch', 'layer'], type of normalization to apply.
            'batch' provides better performance, 'layer' is more stable for small batches.
            Defaults to 'batch'.
        activation: Union[str, Callable], activation function for intermediate stages.
            Defaults to 'gelu'.
        use_bias: Boolean, whether convolution layers include bias parameters.
            Defaults to True.
        kernel_initializer: Union[str, keras.initializers.Initializer], initializer
            for convolution kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: Union[str, keras.initializers.Initializer], initializer
            for bias parameters. Defaults to 'zeros'.
        kernel_regularizer: Optional[keras.regularizers.Regularizer], regularizer
            applied to convolution kernels. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Raises:
        ValueError: If parameters are invalid (e.g., unsupported patch size).
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

        # Validate all inputs
        if embed_dim <= 0 or embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be positive and divisible by 4, got {embed_dim}")
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"Patch height and width must be equal, got {patch_size}")
        p_size = patch_size[0]
        if p_size < 4 or (p_size & (p_size - 1)) != 0:
            raise ValueError(f"patch_size must be a power of 2 and >= 4, got {p_size}")
        if img_size[0] % p_size != 0 or img_size[1] % p_size != 0:
            raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")
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
        self.dim1 = embed_dim // 4
        self.num_patches = (img_size[0] // p_size) * (img_size[1] // p_size)
        self.activation_fn = keras.activations.get(activation)

        # Dynamically create hierarchical stages
        self.conv_stages = []
        self.norm_stages = []
        current_stride = 1
        stage_idx = 0
        in_ch = self.in_channels

        # Start with a 4x4 convolution
        if p_size >= 4:
            stride = 4
            out_ch = self.dim1 if p_size > 4 else self.embed_dim
            self._add_stage(in_ch, out_ch, kernel_size=stride, name=f"stage_{stage_idx}")
            current_stride *= stride
            in_ch = out_ch
            stage_idx += 1

        # Add 2x2 convolutions until the target patch size is reached
        while current_stride < p_size:
            stride = 2
            out_ch = self.dim1 if (current_stride * stride) < p_size else self.embed_dim
            self._add_stage(in_ch, out_ch, kernel_size=stride, name=f"stage_{stage_idx}")
            current_stride *= stride
            in_ch = out_ch
            stage_idx += 1

        logger.info(
            f"Initialized HierarchicalMLPStem: embed_dim={embed_dim}, "
            f"img_size={img_size}, patch_size={patch_size}, "
            f"num_patches={self.num_patches}, stages={len(self.conv_stages)}"
        )

    def _add_stage(self, in_channels: int, out_channels: int, kernel_size: int, name: str):
        """Helper to create and append one stage of the hierarchy."""
        self.conv_stages.append(keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=kernel_size,
            padding='valid',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{name}_conv"
        ))
        if self.norm_layer == 'batch':
            self.norm_stages.append(keras.layers.BatchNormalization(name=f"{name}_norm"))
        else:
            self.norm_stages.append(keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm"))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers dynamically."""
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")
        if input_shape[-1] != self.in_channels:
            raise ValueError(f"Input channels {input_shape[-1]} don't match expected {self.in_channels}")

        current_shape = input_shape
        for conv_layer, norm_layer in zip(self.conv_stages, self.norm_stages):
            if not conv_layer.built:
                conv_layer.build(current_shape)
            current_shape = conv_layer.compute_output_shape(current_shape)
            if not norm_layer.built:
                norm_layer.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply hierarchical MLP stem to input images."""
        x = inputs
        num_stages = len(self.conv_stages)
        for i, (conv_layer, norm_layer) in enumerate(zip(self.conv_stages, self.norm_stages)):
            x = conv_layer(x, training=training)
            x = norm_layer(x, training=training)
            # No activation after the final stage
            if i < num_stages - 1:
                x = self.activation_fn(x)

        # Reshape from [batch, h, w, c] to [batch, num_patches, embed_dim]
        batch_size, height, width, channels = ops.shape(x)
        x = ops.reshape(x, [batch_size, height * width, channels])
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        h_patches = input_shape[1] // self.patch_size[0]
        w_patches = input_shape[2] // self.patch_size[1]
        num_patches = h_patches * w_patches
        return (batch_size, num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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