"""
A hierarchical, non-overlapping convolutional stem for ViTs.

This layer serves as a patch embedding module for Vision Transformers (ViTs),
converting an input image into a sequence of patch tokens. It replaces the
standard ViT stem's direct linear projection of flattened patches with a more
expressive, multi-stage convolutional architecture. The design enhances local
feature learning within each patch before the tokens are processed by the main
transformer body.

Architectural and Mathematical Foundations:
The stem is constructed dynamically as a series of non-overlapping 2D
convolutional stages. It begins with a `4x4` convolution with a stride of 4,
which processes the image into an initial set of `4x4` patch features.
Subsequently, `2x2` convolutions with a stride of 2 are progressively stacked.
Each additional stage doubles the effective patch size of the receptive field
(e.g., `4x4` -> `8x8` -> `16x16`), allowing features to be built
hierarchically within what will become the final patch token.

The core mathematical principle is the use of non-overlapping convolutions,
where `stride = kernel_size`. This ensures that the computation for each
output patch is exclusively dependent on the pixels within its corresponding
input region. There is no information leakage across patch boundaries within
the stem.

This property of **patch independence** is the primary motivation for this
design over a standard convolutional network stem (like that of a ResNet).
It makes the stem fully compatible with masked image modeling (MIM)
pre-training paradigms such as Masked Autoencoders (MAE) and BEiT. In MIM,
a subset of input patches is masked (e.g., zeroed out). With this stem, a
masked input patch maps directly to a predictable (e.g., zero) output token
without affecting the representations of any other visible patches. This clean
separation is critical for the reconstruction-based self-supervision task.

References:
    - Liu et al. "h-MLP: Vision MLP with Hierarchical Rearrangement". The
      hierarchical stem architecture using stacked non-overlapping convolutions
      is derived from this work.
      https://arxiv.org/abs/2203.09716

    - He et al. "Masked Autoencoders Are Scalable Vision Learners" (MAE). This
      paper exemplifies the masked image modeling paradigm for which the
      patch-independent property of this stem is essential.
      https://arxiv.org/abs/2111.06377
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
    """Hierarchical MLP stem for Vision Transformers with patch-independent processing.

    This layer implements a flexible hMLP stem that processes image patches through
    a sequence of hierarchical, non-overlapping convolutional stages without
    cross-patch information leakage. It dynamically creates stages to support
    various patch sizes (e.g., 8, 16, 32), making it compatible with diverse
    Vision Transformer architectures and masked self-supervised learning methods
    like MAE and BEiT. The non-overlapping property (``stride = kernel_size``)
    ensures that each output patch token depends exclusively on the pixels
    within its corresponding input region.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────┐
        │  Input [batch, H, W, in_channels]       │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  Stage 0: Conv2D(dim1, k=4, s=4) → Norm │
        │           → Activation                  │
        │  (processes 4x4 patches independently)  │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  Stage 1: Conv2D(dim1, k=2, s=2) → Norm │
        │           → Activation                  │
        │  (processes 8x8 patches hierarchically) │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  Stage N: Conv2D(embed_dim, k=2, s=2)   │
        │           → Norm  (final stage, no act) │
        └────────────────┬────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │  Reshape [batch, num_patches, embed_dim]│
        └─────────────────────────────────────────┘

    :param embed_dim: Final embedding dimension for each patch. Must be positive
        and divisible by 4. Defaults to 768.
    :type embed_dim: int
    :param img_size: Input image dimensions as ``(height, width)``.
        Both dimensions must be divisible by patch_size. Defaults to ``(224, 224)``.
    :type img_size: Tuple[int, int]
    :param patch_size: Final patch dimensions. Both dimensions must be equal,
        a power of two, and >= 4. Defaults to ``(16, 16)``.
    :type patch_size: Tuple[int, int]
    :param in_channels: Number of input image channels. Must be positive.
        Defaults to 3.
    :type in_channels: int
    :param norm_layer: Type of normalization to apply. ``'batch'`` provides
        better performance, ``'layer'`` is more stable for small batches.
        Defaults to ``'batch'``.
    :type norm_layer: Literal['batch', 'layer']
    :param activation: Activation function for intermediate stages.
        Defaults to ``'gelu'``.
    :type activation: Union[str, Callable]
    :param use_bias: Whether convolution layers include bias parameters.
        Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for convolution kernel weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias parameters.
        Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer applied to convolution kernels.
        Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If parameters are invalid (e.g., unsupported patch size).
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
        """Create and append one stage of the hierarchy.

        :param in_channels: Number of input channels for this stage.
        :type in_channels: int
        :param out_channels: Number of output channels for this stage.
        :type out_channels: int
        :param kernel_size: Kernel size (also used as stride).
        :type kernel_size: int
        :param name: Name prefix for the stage layers.
        :type name: str
        """
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
        """Build the layer and all its sub-layers dynamically.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
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
        """Apply hierarchical MLP stem to input images.

        :param inputs: Input tensor of shape ``(batch, height, width, in_channels)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Patch token tensor of shape ``(batch, num_patches, embed_dim)``.
        :rtype: keras.KerasTensor
        """
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
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        h_patches = input_shape[1] // self.patch_size[0]
        w_patches = input_shape[2] // self.patch_size[1]
        num_patches = h_patches * w_patches
        return (batch_size, num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
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