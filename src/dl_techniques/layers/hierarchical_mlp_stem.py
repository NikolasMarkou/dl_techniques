"""Hierarchical convolutional stem for Vision Transformers.

``HierarchicalMLPStem`` turns an image into patch tokens
``(batch, num_patches, embed_dim)``. In place of the standard ViT stem's single
linear projection of flattened patches, it stacks non-overlapping convolutions,
each with ``stride = kernel_size``: one 4x4 stride-4 stage, then 2x2 stride-2
stages until the accumulated stride reaches ``patch_size``. Since no stage
overlaps, an output token depends only on the pixels of its own patch, so a
masked input patch maps to a predictable token without touching any neighbour.
Intermediate stages carry ``embed_dim // 4`` channels; the last carries
``embed_dim`` and applies no activation.

``patch_size`` must be square, a power of two, and at least 4, and ``embed_dim``
must be divisible by 4. ``img_size`` only fixes the reported ``num_patches``;
the forward pass reads the actual input shape, so an image of a different size
still runs.

References:
    - Liu et al., 2022. h-MLP: Vision MLP with Hierarchical Rearrangement.
      (https://arxiv.org/abs/2203.09716)
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      (https://arxiv.org/abs/2111.06377)
"""

import keras
from keras import ops
from typing import Tuple, Optional, Union, Any, Dict, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.hierarchical_mlp_stem")
class HierarchicalMLPStem(keras.layers.Layer):
    """Embed image patches through non-overlapping convolutional stages.

    The stage list is built in ``__init__`` from ``patch_size``: a 4x4 stride-4
    convolution, then 2x2 stride-2 convolutions until the accumulated stride
    equals the patch size. Every convolution uses ``valid`` padding with
    ``stride = kernel_size``, so no receptive field crosses a patch boundary.
    The final feature map is flattened to one token per patch.

    Architecture:

    .. code-block:: text

              input [B, H, W, in_channels]
                                ▼
                ┌───────────────────────────────┐
                │ Conv2D(dim1, k=4, s=4)        │
                │  Norm, activation             │
                └───────────────┬───────────────┘
                     [B, H/4, W/4, dim1]
                                ▼
                ┌───────────────────────────────┐
                │ Conv2D(dim1, k=2, s=2)        │
                │  Norm, activation             │
                │  repeated until patch_size    │
                └───────────────┬───────────────┘
                     [B, H/s, W/s, dim1]
                                ▼
                ┌───────────────────────────────┐
                │ Conv2D(embed_dim, k=2, s=2)   │
                │  Norm, no activation          │
                └───────────────┬───────────────┘
                    [B, h, w, embed_dim]
                                ▼
                ┌───────────────────────────────┐
                │ reshape to tokens             │
                └───────────────┬───────────────┘
                                ▼
                     [B, h*w, embed_dim]

    One stage:

    .. code-block:: text

                        x
                        ▼
              ┌───────────────────┐
              │ Conv2D k=s, valid │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ BatchNorm or      │
              │  LayerNorm        │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ activation        │
              │  (not last stage) │
              └─────────┬─────────┘
                        ▼
                        y

    Stages by patch size:

    .. code-block:: text

        patch_size   stages                      output channels
        ──────────   ─────────────────────────   ───────────────────────
        4            4x4/4                       embed_dim
        8            4x4/4, 2x2/2                dim1, embed_dim
        16           4x4/4, 2x2/2, 2x2/2         dim1, dim1, embed_dim
        32           4x4/4, three 2x2/2          dim1 x 3, embed_dim

    ``dim1`` is ``embed_dim // 4``. At ``patch_size=4`` there is a single stage,
    which is therefore the last one and carries no activation.

    :param embed_dim: Final embedding dimension per patch. Must be positive and
        divisible by 4. Defaults to 768.
    :type embed_dim: int
    :param img_size: Input image dimensions as ``(height, width)``, both
        divisible by ``patch_size``. Used to report ``num_patches``; the
        forward pass reads the actual input shape. Defaults to ``(224, 224)``.
    :type img_size: Tuple[int, int]
    :param patch_size: Final patch dimensions. The two entries must be equal, a
        power of two, and at least 4. Defaults to ``(16, 16)``.
    :type patch_size: Tuple[int, int]
    :param in_channels: Number of input image channels. Must be positive, and
        the input's channel axis must match it. Defaults to 3.
    :type in_channels: int
    :param norm_layer: ``'batch'`` for ``BatchNormalization`` or ``'layer'`` for
        ``LayerNormalization(epsilon=1e-6)``. Defaults to ``'batch'``.
    :type norm_layer: Literal['batch', 'layer']
    :param activation: Activation applied after every stage except the last.
        Defaults to ``'gelu'``.
    :type activation: Union[str, Callable]
    :param use_bias: Give the convolutions a bias. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for convolution kernels. Defaults to
        ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for biases. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for convolution kernels. Defaults to
        None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor ``(batch, height, width, in_channels)``.

    Output shape:
        3D tensor ``(batch, num_patches, embed_dim)``.

    :raises ValueError: From ``__init__``, if ``embed_dim`` is not positive and
        divisible by 4, if the two ``patch_size`` entries differ or are not a
        power of two of at least 4, if ``img_size`` is not divisible by the
        patch size, if ``norm_layer`` is neither name, or if ``in_channels`` is
        not positive. From ``build``, if the input is not rank 4 or its channel
        axis does not equal ``in_channels``.
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

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm_layer = norm_layer
        # DECISION D-401: store the resolved activation, never an
        # 'activation_name' placeholder, which drops callables. See decisions.md.
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.dim1 = embed_dim // 4
        self.num_patches = (img_size[0] // p_size) * (img_size[1] // p_size)
        self.activation_fn = keras.activations.get(self.activation)

        self.conv_stages = []
        self.norm_stages = []
        current_stride = 1
        stage_idx = 0
        in_ch = self.in_channels

        if p_size >= 4:
            stride = 4
            out_ch = self.dim1 if p_size > 4 else self.embed_dim
            self._add_stage(in_ch, out_ch, kernel_size=stride, name=f"stage_{stage_idx}")
            current_stride *= stride
            in_ch = out_ch
            stage_idx += 1

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
        :param kernel_size: Kernel size, also used as the stride.
        :type kernel_size: int
        :param name: Name prefix for the stage's layers.
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
        """Build every stage, threading each stage's output shape to the next.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4, or its channel axis does
            not equal ``in_channels``.
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
        """Embed the input image into patch tokens.

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
            # The last stage stays linear, so the tokens reach the body
            # unsquashed.
            if i < num_stages - 1:
                x = self.activation_fn(x)

        batch_size, height, width, channels = ops.shape(x)
        x = ops.reshape(x, [batch_size, height * width, channels])
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape from the input's own spatial size.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, num_patches, embed_dim)``.
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
            'activation': serialize_activation(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    # ---------------------------------------------------------------------