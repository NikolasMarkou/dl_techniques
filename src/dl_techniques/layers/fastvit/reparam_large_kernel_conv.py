"""
Reparameterizable large-kernel convolution of the FastViT / MobileCLIP2 MCi backbone.

This module transcribes timm's ``ReparamLargeKernelConv``, the downsampling
primitive that FastViT uses inside every ``PatchEmbed``.

The design idea is the same structural-reparameterization trick that
:class:`~dl_techniques.layers.mobile_one_block.MobileOneBlock` uses, applied to a
*large* kernel: a ``k x k`` Conv-BN branch (``k = 7`` at every MCi call site) is
summed with a parallel ``small_kernel x small_kernel`` Conv-BN branch (``3 x 3``).
Both branches are affine at inference, share a stride and a group count, and are
padded to the same output resolution, so the pair collapses into a single ``k x k``
convolution once the BatchNormalizations are folded in. The small branch exists
purely to improve optimization: it gives the block a well-conditioned short-range
path during training that costs nothing at deployment.

.. code-block:: text

    out = large_conv(x) + small_conv(x)     # both Conv-BN, NO activation
    out = se(out)                            # optional
    out = act(out)                           # optional (Identity by default)

This port implements the **train-time** multi-branch form only; no structural
reparameterization (`reparameterize()` / branch fusion) is provided, matching the
reference weights shipped by MobileCLIP2 (always evaluated with
``inference_mode=False``).

Three details are load-bearing and easy to get silently wrong:

1. **``group_size`` follows timm's ``num_groups`` semantics**, NOT a literal group
   count: ``group_size=0`` means ``groups=1`` (dense) and ``group_size=k>0`` means
   ``groups = in_channels // k``, so ``group_size=1`` is DEPTHWISE. This matches
   :class:`MobileOneBlock`, which resolves the same way.
2. **A grouped convolution constrains BOTH channel counts.** With
   ``groups = in_channels`` (the ``group_size=1`` case used by
   :class:`~dl_techniques.layers.fastvit.patch_embed.FastVitPatchEmbed`), the
   layer can only map ``in_channels -> out_channels`` when ``out_channels`` is a
   multiple of ``in_channels``. That holds for every MCi variant because the
   channel width exactly doubles from stage to stage — but it is a real
   precondition, and it is raised loudly here rather than silently reshaped.
3. **The Squeeze-and-Excitation ratio here is 0.25 with biases**, which is NOT
   timm's ``SqueezeExcite`` default of ``1/16``. The reference passes
   ``rd_ratio=0.25`` explicitly at this call site (MEASURED to be reachable by
   this repo's :class:`SqueezeExcitation`).

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Ding et al., 2022. Scaling Up Your Kernels to 31x31: Revisiting Large Kernel
      Design in CNNs. (https://arxiv.org/abs/2203.06717)
    - Hu et al., 2018. Squeeze-and-Excitation Networks. (https://arxiv.org/abs/1709.01507)
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import layers, ops, initializers, regularizers, activations

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms.factory import create_normalization_layer
from ..squeeze_excitation import SqueezeExcitation
# Single definition of timm's `num_groups` mapping, shared with MobileOneBlock.
# Do NOT re-implement it here: the two layers must resolve `group_size`
# identically or a FastViT block and the MobileOneBlock beside it would disagree
# about what `group_size=1` means.
from ..mobile_one_block import (
    resolve_num_groups,
    resolve_conv_padding,
    conv_output_size,
)
from .reference import REFERENCE_NORM_EPSILON, REFERENCE_PADDING_MODE

# ---------------------------------------------------------------------

#: Single definition of the reference epsilon lives in :mod:`.reference`.
_REFERENCE_BN_EPSILON = REFERENCE_NORM_EPSILON

#: Squeeze-and-Excitation bottleneck ratio used by the reference AT THIS CALL SITE
#: (``rd_ratio=0.25``). Deliberately different from timm's ``SqueezeExcite``
#: default of ``1/16`` and from MobileOne's usage.
_REFERENCE_SE_REDUCTION_RATIO = 0.25

#: The reference's ``SqueezeExcite`` uses biased 1x1 convolutions.
_REFERENCE_SE_USE_BIAS = True


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ReparamLargeKernelConv(keras.layers.Layer):
    """FastViT reparameterizable large-kernel convolution.

    Channels-last transcription of timm's ``ReparamLargeKernelConv``. A large
    ``kernel_size x kernel_size`` Conv-BN branch is summed with an optional small
    ``small_kernel x small_kernel`` Conv-BN branch at the same stride, padding and
    group count. Neither branch applies an activation; an optional
    Squeeze-and-Excitation block and an optional activation follow the sum.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │          Input [B, H, W, C_in]               │
        └───────┬──────────────────────────┬───────────┘
                │                          │
                ▼                          ▼
        ┌───────────────────┐  ┌───────────────────────┐
        │ large_conv        │  │ small_conv (optional) │
        │  Conv k×k, stride │  │  Conv s×s, stride     │
        │  groups, no bias  │  │  groups, no bias      │
        │  + BatchNorm      │  │  + BatchNorm          │
        │  NO activation    │  │  NO activation        │
        └─────────┬─────────┘  └───────────┬───────────┘
                  │                        │
                  └────────── + ───────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────┐
        │  se: SqueezeExcitation (optional)            │
        │      reduction_ratio=0.25, use_bias=True     │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │  activation (optional; Identity when None)   │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │      Output [B, H', W', out_channels]        │
        │      H' = ceil(H / stride)  (padding='same') │
        └──────────────────────────────────────────────┘

    .. warning::
       ``group_size`` uses timm's ``num_groups`` semantics, not a literal group
       count: ``0 -> groups=1`` (dense) and ``k>0 -> groups = in_channels // k``,
       so ``group_size=1`` is DEPTHWISE. Because ``in_channels`` is unknown until
       :meth:`build`, the convolution branches are constructed there for the
       grouped case — the same deliberate exception
       :class:`MobileOneBlock` makes, for the same reason.

    :param out_channels: Number of output channels. Must be positive.
    :type out_channels: int
    :param kernel_size: Spatial size of the large convolution. Must be positive.
    :type kernel_size: int
    :param stride: Stride of BOTH convolution branches. Must be positive.
    :type stride: int
    :param group_size: Grouped-convolution control in timm's ``num_groups``
        semantics (see the warning above). Must be non-negative.
    :type group_size: int
    :param small_kernel: Spatial size of the parallel small convolution, or
        ``None`` to omit that branch entirely. When given it must be positive and
        no larger than ``kernel_size``. Defaults to ``None``.
    :type small_kernel: Optional[int]
    :param use_se: Whether to apply a Squeeze-and-Excitation block to the summed
        branches. Defaults to False.
    :type use_se: bool
    :param activation: Activation applied last, or ``None`` for the identity (the
        reference's default — ``act_layer`` is only supplied when
        ``lkc_use_act`` is set). Defaults to ``None``.
    :type activation: Optional[Union[str, callable]]
    :param kernel_initializer: Initializer for both convolution kernels.
        Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for both convolution kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``out_channels``, ``kernel_size`` or ``stride`` are not
        positive, if ``group_size`` is negative, if ``small_kernel`` is not
        positive, or if ``small_kernel > kernel_size``. At build time, also if the
        input is not rank 4, if the channel count is undefined, or if the resolved
        group count does not divide both the input and the output channel counts.

    Example:
        >>> import numpy as np
        >>> layer = ReparamLargeKernelConv(
        ...     out_channels=32, kernel_size=7, stride=2, group_size=1,
        ...     small_kernel=3)
        >>> y = layer(np.zeros((2, 16, 16, 16), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 8, 32)
    """

    def __init__(
            self,
            out_channels: int,
            kernel_size: int,
            stride: int,
            group_size: int,
            small_kernel: Optional[int] = None,
            use_se: bool = False,
            activation: Optional[Union[str, callable]] = None,
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if group_size < 0:
            raise ValueError(f"group_size must be non-negative, got {group_size}")
        if small_kernel is not None:
            if small_kernel <= 0:
                raise ValueError(
                    f"small_kernel must be positive when given, got {small_kernel}")
            if small_kernel > kernel_size:
                raise ValueError(
                    f"small_kernel must not exceed kernel_size: "
                    f"small_kernel={small_kernel}, kernel_size={kernel_size}"
                )
            if small_kernel % 2 != kernel_size % 2:
                raise ValueError(
                    f"kernel_size and small_kernel must have the same parity: the "
                    f"branches pad symmetrically by their own kernel_size // 2 "
                    f"(the reference convention), so an odd and an even kernel "
                    f"produce output maps that differ by one pixel and cannot be "
                    f"summed. Got kernel_size={kernel_size}, "
                    f"small_kernel={small_kernel}"
                )

        # ---- store configuration ---------------------------------------
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.group_size = group_size
        self.small_kernel = small_kernel
        self.use_se = use_se
        self.activation = (
            None if activation is None else activations.get(activation)
        )
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Resolved group count. `group_size == 0` is knowable up front; anything
        # else depends on `in_channels` and is finalised in build().
        self.groups: Optional[int] = 1 if self.group_size == 0 else None

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        #
        # Deliberate exception, mirroring MobileOneBlock: `groups` must be passed
        # to Conv2D at CONSTRUCTION time, but timm's `num_groups` semantics derive
        # it from `in_channels`, which is unknown until build(). The convolution
        # branches are therefore constructed here ONLY for the `group_size == 0`
        # (groups == 1) case and otherwise in build(). Construction is still
        # driven purely by config plus the channel count and sub-layer names are
        # deterministic, so `.keras` round-tripping is unaffected (Keras calls
        # build() from `build_from_config` before restoring weights).
        self.large_conv: Optional[keras.Sequential] = None
        self.small_conv: Optional[keras.Sequential] = None
        if self.groups is not None:
            self.large_conv = self._create_branch(
                self.kernel_size, self.groups, 'large_conv')
            self.small_conv = self._create_small_branch(self.groups)

        self.se = None
        if self.use_se:
            self.se = SqueezeExcitation(
                reduction_ratio=_REFERENCE_SE_REDUCTION_RATIO,
                use_bias=_REFERENCE_SE_USE_BIAS,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='se',
            )

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _create_branch(
            self,
            kernel_size: int,
            groups: int,
            name: str,
    ) -> keras.Sequential:
        """Build one Conv-BN branch (no activation).

        :param kernel_size: Spatial size of this branch's convolution.
        :type kernel_size: int
        :param groups: Resolved convolution group count.
        :type groups: int
        :param name: Sub-layer name; the inner conv/BN names are derived from it.
        :type name: str
        :return: A ``Sequential`` of an optional symmetric ``ZeroPadding2D``, a
            ``Conv2D(use_bias=False)`` and ``BatchNormalization(epsilon=1e-5)``.
        :rtype: keras.Sequential
        """
        # The reference pads SYMMETRICALLY by `kernel_size // 2`. Keras'
        # `padding='same'` pads asymmetrically, so at stride > 1 the k=7 and k=3
        # branches summed below would sample a grid whose offset depends on the
        # kernel size — MEASURED, a one-pixel shift that no shape assertion sees.
        pad, keras_padding = resolve_conv_padding(
            kernel_size, 'same', REFERENCE_PADDING_MODE)
        padding_layers = (
            [] if pad == 0
            else [layers.ZeroPadding2D(padding=pad, name=f'{name}_pad')]
        )
        return keras.Sequential(padding_layers + [
            layers.Conv2D(
                filters=self.out_channels,
                kernel_size=kernel_size,
                strides=self.stride,
                padding=keras_padding,
                use_bias=False,
                groups=groups,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'{name}_conv',
            ),
            create_normalization_layer(
                'batch_norm',
                epsilon=_REFERENCE_BN_EPSILON,
                name=f'{name}_bn',
            ),
        ], name=name)

    def _create_small_branch(self, groups: int) -> Optional[keras.Sequential]:
        """Build the optional small-kernel Conv-BN branch.

        :param groups: Resolved convolution group count.
        :type groups: int
        :return: The small branch, or ``None`` when ``small_kernel is None``.
        :rtype: Optional[keras.Sequential]
        """
        if self.small_kernel is None:
            return None
        return self._create_branch(self.small_kernel, groups, 'small_conv')

    # ------------------------------------------------------------------
    # build / call
    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Resolve the group count, create the grouped branches, build everything.

        :param input_shape: Shape of the input tensor, ``(B, H, W, C_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4, if the channel count is
            undefined, if ``group_size`` does not divide the input channels, or if
            the resolved group count does not divide both the input and the output
            channel counts.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"ReparamLargeKernelConv expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        in_channels = input_shape[-1]
        if in_channels is None:
            raise ValueError("Input channels dimension must be defined")

        resolved_groups = resolve_num_groups(self.group_size, in_channels)

        # A grouped convolution partitions BOTH the input and the output channel
        # axis, so the group count must divide both. For the group_size=1
        # (depthwise) case used by FastVitPatchEmbed this reduces to
        # `out_channels % in_channels == 0` — legal for every MCi variant only
        # because the channel width exactly doubles from stage to stage. Raise
        # loudly rather than silently reshaping.
        if in_channels % resolved_groups != 0 or self.out_channels % resolved_groups != 0:
            raise ValueError(
                f"resolved groups={resolved_groups} must divide both "
                f"in_channels={in_channels} and out_channels={self.out_channels} "
                f"(group_size={self.group_size}); a grouped convolution cannot "
                f"map {in_channels} channels to {self.out_channels} channels here"
            )

        if self.groups is None:
            self.groups = resolved_groups
            self.large_conv = self._create_branch(
                self.kernel_size, resolved_groups, 'large_conv')
            self.small_conv = self._create_small_branch(resolved_groups)

        self.large_conv.build(input_shape)
        if self.small_conv is not None:
            self.small_conv.build(input_shape)

        if self.se is not None:
            self.se.build(self.compute_output_shape(input_shape))

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the large-kernel convolution.

        :param inputs: Input tensor of shape ``(B, H, W, C_in)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — the branch BatchNormalizations update their
            moving statistics otherwise.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H', W', out_channels)``.
        """
        x = self.large_conv(inputs, training=training)

        if self.small_conv is not None:
            x = ops.add(x, self.small_conv(inputs, training=training))

        if self.se is not None:
            x = self.se(x, training=training)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        Both branches pad symmetrically by their own ``kernel_size // 2`` and run
        at the same stride, so for ODD kernels the spatial reduction is
        ``ceil(size / stride)`` independently of either kernel size — the same
        figure Keras' ``'same'`` would give. For an EVEN kernel the reference
        convention loses one pixel, which is why the size is derived rather than
        assumed.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple ``(B, H', W', out_channels)``.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        height = conv_output_size(
            input_shape[1], self.kernel_size, self.stride,
            'same', REFERENCE_PADDING_MODE)
        width = conv_output_size(
            input_shape[2], self.kernel_size, self.stride,
            'same', REFERENCE_PADDING_MODE)
        return (input_shape[0], height, width, self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'group_size': self.group_size,
            'small_kernel': self.small_kernel,
            'use_se': self.use_se,
            'activation': (
                None if self.activation is None
                else activations.serialize(self.activation)
            ),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReparamLargeKernelConv":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`ReparamLargeKernelConv` instance.
        :rtype: ReparamLargeKernelConv
        """
        config = dict(config)
        if config.get('activation') is not None:
            config['activation'] = activations.deserialize(config['activation'])
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)

# ---------------------------------------------------------------------
