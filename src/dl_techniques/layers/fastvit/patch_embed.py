"""
Patch embedding / downsampling block of the FastViT / MobileCLIP2 MCi backbone.

This module transcribes timm's FastViT ``PatchEmbed``, the block that sits at the
head of every downsampling stage of the MCi image tower.

Despite the name, this is NOT a ViT-style non-overlapping patchifier: there is no
reshape into tokens and no flattening. It is a purely convolutional ``/2``
downsample that also changes the channel width, built from two reparameterizable
primitives:

.. code-block:: text

    ReparamLargeKernelConv(k=7, stride=2, group_size=1, small_kernel=3)  # spatial
    MobileOneBlock(k=1, stride=1, use_se=False, act='gelu')              # channel

The first stage does the spatial work with a large depthwise kernel; the second is
a pointwise MobileOne block that mixes channels and applies the block's only
mandatory activation. Both collapse to a single convolution at inference time.

Two details are load-bearing and easy to get silently wrong:

1. **``group_size=1`` on the large-kernel conv means ``groups = in_channels``**
   (timm's ``num_groups`` semantics — see
   :func:`~dl_techniques.layers.mobile_one_block.resolve_num_groups`). A grouped
   convolution partitions the output channel axis too, so ``embed_dim`` MUST be an
   exact multiple of the incoming channel count. Every MCi variant satisfies this
   because the channel width exactly doubles from stage to stage, but a variant
   table typo would otherwise produce a confusing low-level Conv2D error instead
   of a diagnosable one. :class:`ReparamLargeKernelConv` raises loudly here.
2. **``lkc_use_act`` gates the large-kernel conv's activation only.** When it is
   False (the reference default for most stages) the large-kernel conv is purely
   affine and the block's first nonlinearity is the trailing MobileOne block's.
   Dropping the flag entirely is invisible to every shape assertion.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2024. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. (https://arxiv.org/abs/2311.17049)
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import initializers, regularizers, activations

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .reparam_large_kernel_conv import ReparamLargeKernelConv
from ..mobile_one_block import MobileOneBlock
from .reference import REFERENCE_NORM_EPSILON, REFERENCE_PADDING_MODE

# ---------------------------------------------------------------------

#: Kernel size of the trailing pointwise MobileOne block (reference: 1x1).
_POINTWISE_KERNEL_SIZE = 1

#: ``group_size`` passed to the large-kernel conv. In timm's ``num_groups``
#: semantics this means ``groups = in_channels``, i.e. DEPTHWISE.
_LKC_GROUP_SIZE = 1

#: Small parallel kernel of the large-kernel conv (reference: 3x3).
_LKC_SMALL_KERNEL = 3


@keras.saving.register_keras_serializable()
class FastVitPatchEmbed(keras.layers.Layer):
    """FastViT patch embedding: a convolutional ``/stride`` downsample + rewidening.

    Channels-last transcription of timm's FastViT ``PatchEmbed``. Sequential
    composition of a :class:`ReparamLargeKernelConv` (large depthwise kernel,
    strided, with a parallel small-kernel branch) and a pointwise
    :class:`MobileOneBlock`.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │           Input [B, H, W, C_in]              │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │ proj_lkc: ReparamLargeKernelConv             │
        │   k=patch_size, stride=stride                │
        │   group_size=1  ->  groups = C_in (depthwise)│
        │   small_kernel=3, optional SE                │
        │   activation only when lkc_use_act           │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │ proj_mobileone: MobileOneBlock               │
        │   k=1, stride=1, use_se=False, act=activation│
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │      Output [B, H', W', embed_dim]           │
        │      H' = ceil(H / stride)                   │
        └──────────────────────────────────────────────┘

    .. warning::
       Because the large-kernel convolution is depthwise
       (``groups = in_channels``), ``embed_dim`` must be an exact multiple of the
       incoming channel count. A violation raises a :class:`ValueError` naming both
       numbers at build time — it is never silently reshaped.

    :param embed_dim: Output channel count. Must be positive.
    :type embed_dim: int
    :param patch_size: Kernel size of the large-kernel convolution. Must be
        positive. Defaults to 7.
    :type patch_size: int
    :param stride: Spatial stride of the downsample. Must be positive. Defaults
        to 2.
    :type stride: int
    :param use_se: Whether the large-kernel convolution applies
        Squeeze-and-Excitation. Defaults to False.
    :type use_se: bool
    :param lkc_use_act: Whether the large-kernel convolution applies
        ``activation``. When False that stage is purely affine, matching the
        reference default. Defaults to False.
    :type lkc_use_act: bool
    :param activation: Activation used by the trailing MobileOne block always, and
        by the large-kernel convolution when ``lkc_use_act`` is True. Defaults to
        ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for every convolution kernel. Defaults
        to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every convolution kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``embed_dim``, ``patch_size`` or ``stride`` are not
        positive. At build time, also if the input is not rank 4, if the channel
        count is undefined, or if ``embed_dim`` is not a multiple of the input
        channel count (the depthwise divisibility precondition).

    Example:
        >>> import numpy as np
        >>> layer = FastVitPatchEmbed(embed_dim=64)
        >>> y = layer(np.zeros((2, 64, 64, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 32, 32, 64)
    """

    def __init__(
            self,
            embed_dim: int,
            patch_size: int = 7,
            stride: int = 2,
            use_se: bool = False,
            lkc_use_act: bool = False,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        # ---- store configuration ---------------------------------------
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride
        self.use_se = use_se
        self.lkc_use_act = lkc_use_act
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        # `lkc_use_act` selects between passing the activation and passing None
        # (the reference's `act_layer=GELU if lkc_use_act else None`). Passing the
        # activation unconditionally would insert a nonlinearity the reference
        # does not have, and every shape assertion would still pass.
        self.proj_lkc = ReparamLargeKernelConv(
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            group_size=_LKC_GROUP_SIZE,
            small_kernel=_LKC_SMALL_KERNEL,
            use_se=self.use_se,
            activation=self.activation if self.lkc_use_act else None,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='proj_lkc',
        )
        self.proj_mobileone = MobileOneBlock(
            out_channels=self.embed_dim,
            kernel_size=_POINTWISE_KERNEL_SIZE,
            stride=1,
            use_se=False,
            activation=self.activation,
            norm_epsilon=REFERENCE_NORM_EPSILON,
            padding_mode=REFERENCE_PADDING_MODE,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='proj_mobileone',
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, C_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4, if the channel count is
            undefined, or if ``embed_dim`` is not a multiple of the input channel
            count.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitPatchEmbed expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        in_channels = input_shape[-1]
        if in_channels is None:
            raise ValueError("Input channels dimension must be defined")

        # The large-kernel conv is DEPTHWISE (group_size=1 -> groups=in_channels),
        # so it can only map in_channels -> embed_dim when embed_dim is a multiple
        # of in_channels. ReparamLargeKernelConv also raises, but doing it here
        # names the layer the caller actually constructed and its own kwarg.
        if self.embed_dim % in_channels != 0:
            raise ValueError(
                f"FastVitPatchEmbed uses a DEPTHWISE large-kernel convolution "
                f"(group_size=1 -> groups = in_channels), so embed_dim must be an "
                f"exact multiple of the input channel count: "
                f"embed_dim={self.embed_dim}, in_channels={in_channels} "
                f"({self.embed_dim} % {in_channels} = "
                f"{self.embed_dim % in_channels})"
            )

        self.proj_lkc.build(input_shape)
        lkc_shape = self.proj_lkc.compute_output_shape(input_shape)

        self.proj_mobileone.build(lkc_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the patch embedding.

        :param inputs: Input tensor of shape ``(B, H, W, C_in)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — the BatchNormalizations inside both
            sub-blocks update their moving statistics otherwise.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H', W', embed_dim)``.
        """
        x = self.proj_lkc(inputs, training=training)
        x = self.proj_mobileone(x, training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        The trailing MobileOne block is stride-1 and shape-preserving, so the whole
        block reduces each spatial dimension by ``ceil(size / stride)``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple
            ``(B, ceil(H/stride), ceil(W/stride), embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        height = (
            None if input_shape[1] is None
            else (input_shape[1] + self.stride - 1) // self.stride
        )
        width = (
            None if input_shape[2] is None
            else (input_shape[2] + self.stride - 1) // self.stride
        )
        return (input_shape[0], height, width, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'use_se': self.use_se,
            'lkc_use_act': self.lkc_use_act,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitPatchEmbed":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitPatchEmbed` instance.
        :rtype: FastVitPatchEmbed
        """
        config = dict(config)
        config['activation'] = activations.deserialize(config['activation'])
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)

# ---------------------------------------------------------------------
