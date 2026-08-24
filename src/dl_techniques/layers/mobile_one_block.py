"""
MobileOne block using structural reparameterization.

This layer embodies the principle of structural reparameterization, a design
paradigm that decouples the training-time architecture from the
inference-time architecture. The core idea is to use a more complex,
over-parameterized, multi-branch structure during training to enhance model
representation and ease optimization, and then mathematically fuse these
branches into a single, computationally efficient layer for fast inference.

Architecturally, during training, this block consists of multiple parallel
branches whose outputs are summed. These typically include:
1.  One or more main branches, each a `k x k` convolution followed by a
    Batch Normalization layer.
2.  A 1x1 convolution branch, also followed by Batch Normalization, acting
    as a "scale" branch.
3.  An optional identity skip-connection, also passed through Batch
    Normalization if the input and output dimensions match.

This over-parameterization creates a richer gradient landscape, which can
lead to better model convergence and final accuracy.

For inference, these parallel affine operations are fused into a single
`Conv2D` operation. This fusion is possible due to the linear properties of
convolution and batch normalization. The fusion process relies on two key
mathematical principles:

First, a `Conv2D` layer followed by a `BatchNormalization` layer can be
converted into a single `Conv2D` layer with a new kernel and bias. Given a
convolution kernel `W` and a batch norm with mean `μ`, variance `σ²`, scale
`γ`, and shift `β`, the fused kernel `W'` and bias `b'` are:

`W' = (γ / sqrt(σ² + ε)) * W`
`b' = β - (γ * μ / sqrt(σ² + ε))`

Second, the sum of outputs from parallel convolutions (with identical stride
and padding) is equivalent to a single convolution whose kernel and bias are
the sum of the individual fused kernels and biases. The 1x1 and identity
branches are first converted to equivalent `k x k` convolutions (by centering
their kernels in a padded `k x k` tensor) before this summation. The result
is a standard, hardware-friendly `Conv2D` layer that is mathematically
equivalent to the complex training-time block, but with significantly lower
latency and memory access costs.

References:
    - Vasu et al., 2022. MobileOne: An Improved One millisecond Mobile
      Backbone. (https://arxiv.org/abs/2206.04040)
    - Ding et al., 2021. RepVGG: Making VGG-style ConvNets Great Again.
      (https://arxiv.org/abs/2101.03697)

"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import layers, initializers, regularizers, activations

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .squeeze_excitation import SqueezeExcitation

# ---------------------------------------------------------------------


def resolve_num_groups(group_size: int, in_channels: int) -> int:
    """Resolve timm's ``num_groups(group_size, in_chs)`` mapping.

    Single definition of the grouped-convolution convention shared by
    :class:`MobileOneBlock` and by the FastViT blocks that must resolve groups the
    same way (``layers/fastvit/reparam_large_kernel_conv.py``). Callers are
    responsible for their own ``out_channels`` divisibility check, whose error
    message differs per layer.

    :param group_size: ``0`` for a dense convolution (``groups = 1``); ``k > 0``
        for ``groups = in_channels // k``, so ``1`` is DEPTHWISE.
    :type group_size: int
    :param in_channels: Number of input channels; only known at build time.
    :type in_channels: int
    :return: The convolution group count (always ``>= 1``).
    :rtype: int
    :raises ValueError: If ``group_size > 0`` does not divide ``in_channels``.
    """
    if group_size == 0:
        return 1
    if in_channels % group_size != 0:
        raise ValueError(
            f"group_size must divide the input channels: "
            f"in_channels={in_channels}, group_size={group_size}"
        )
    return in_channels // group_size


# DECISION plan-2026-08-13T183738-24486492/D-007
# `norm_epsilon` and `padding_mode` both default to TODAY'S KERAS BEHAVIOUR
# (1e-3 and asymmetric `'same'`), which is NOT the MobileOne/FastViT reference
# (1e-5 and PyTorch's symmetric `padding = k // 2`). Do NOT "fix" the defaults:
# `models/fastvlm/` consumes this block through `layers/repmixer_block.py` and
# ships numerics that depend on both, and a defaults-unchanged value-identity
# test pins it. The faithful port passes both explicitly from
# `layers/fastvit/reference.py`. See decisions.md D-007.
#: Accepted values for the ``padding_mode`` knob shared by :class:`MobileOneBlock`
#: and the FastViT blocks.
PADDING_MODES = ('keras_same', 'reference')


def resolve_conv_padding(
        kernel_size: int,
        padding: str,
        padding_mode: str,
) -> Tuple[int, str]:
    """Map ``(padding, padding_mode)`` onto an explicit pad amount + Keras padding.

    Single definition of the two padding conventions, shared by
    :class:`MobileOneBlock` and ``layers/fastvit/reparam_large_kernel_conv.py``.
    Do NOT re-implement it: two branches summed inside one block must resolve the
    convention identically or they sample different pixels.

    ``'keras_same'`` is Keras' native ``padding='same'``, which pads
    ASYMMETRICALLY (the extra row/column goes to the bottom/right). At stride > 1
    that makes the sampled grid depend on the kernel size, so a ``k x k`` branch
    and a ``1 x 1`` branch summed in the same block read DIFFERENT input pixels.
    ``'reference'`` reproduces PyTorch's ``padding=kernel_size // 2``: a symmetric
    explicit pad followed by a ``'valid'`` convolution, which puts every kernel
    size on the same grid (output pixel ``i`` is centred on input pixel
    ``i * stride``). For an ODD kernel at stride 1 the two are identical.

    :param kernel_size: Spatial size of the convolution kernel.
    :type kernel_size: int
    :param padding: The layer's ``padding`` setting, ``'same'`` or ``'valid'``.
    :type padding: str
    :param padding_mode: One of :data:`PADDING_MODES`.
    :type padding_mode: str
    :return: ``(pad_amount, keras_padding)``. ``pad_amount`` is the symmetric
        :class:`keras.layers.ZeroPadding2D` amount to apply BEFORE the convolution
        (``0`` means no padding layer at all), and ``keras_padding`` is the value
        to pass to ``Conv2D(padding=...)``.
    :rtype: Tuple[int, str]
    :raises ValueError: If ``padding_mode`` is not a recognised mode.
    """
    if padding_mode not in PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}"
        )
    if padding_mode == 'reference' and padding == 'same':
        return kernel_size // 2, 'valid'
    return 0, padding


def conv_output_size(
        size: Optional[int],
        kernel_size: int,
        stride: int,
        padding: str,
        padding_mode: str,
) -> Optional[int]:
    """Compute one spatial output dimension of a convolution.

    Companion to :func:`resolve_conv_padding` — it must agree with it, so both
    live together.

    :param size: Input spatial size, or ``None`` when undefined.
    :type size: Optional[int]
    :param kernel_size: Spatial size of the convolution kernel.
    :type kernel_size: int
    :param stride: Convolution stride.
    :type stride: int
    :param padding: ``'same'`` or ``'valid'``.
    :type padding: str
    :param padding_mode: One of :data:`PADDING_MODES`.
    :type padding_mode: str
    :return: The output spatial size, or ``None`` when ``size`` is ``None``.
    :rtype: Optional[int]
    """
    if size is None:
        return None
    pad, keras_padding = resolve_conv_padding(kernel_size, padding, padding_mode)
    if keras_padding == 'same':
        return (size + stride - 1) // stride
    return (size + 2 * pad - kernel_size) // stride + 1


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileOneBlock(keras.layers.Layer):
    """MobileOne building block with structural reparameterization.

    This layer implements the multi-branched MobileOne architecture which can
    be fused into a single, efficient convolutional layer at inference time.
    During training, multiple parallel Conv-BN branches plus an optional 1x1
    scale branch and identity skip connection are summed:
    ``output = activation(sum(branch_i(x)) + SE(x))``. At inference time,
    all branches are fused into a single convolution by exploiting the linear
    properties of convolution and batch normalization:
    ``W' = (gamma / sqrt(var + eps)) * W``, ``b' = beta - gamma * mu / sqrt(var + eps)``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │       Input [B, H, W, C_in]          │
        └───┬────────┬────────┬────────┬───────┘
            │        │        │        │
            ▼        ▼        ▼        ▼
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
        │Conv  │ │Conv  │ │1x1   │ │Skip  │
        │Branch│ │Branch│ │Scale │ │(BN)  │
        │ k×k  │ │ k×k  │ │Branch│ │      │
        │+ BN  │ │+ BN  │ │+ BN  │ │      │
        └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
           │        │        │        │
           └────────┴────┬───┴────────┘
                         │
                         ▼
        ┌──────────────────────────────────────┐
        │  Sum → Activation                    │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Squeeze-and-Excitation (optional)   │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │     Output [B, H', W', out_channels] │
        └──────────────────────────────────────┘

    :param out_channels: Number of output channels. Must be positive.
    :type out_channels: int
    :param kernel_size: Size of the main convolution kernel. Must be positive.
    :type kernel_size: int
    :param stride: Stride of the convolution. Must be positive. Defaults to 1.
    :type stride: int
    :param padding: Padding mode: ``'same'`` or ``'valid'``. Defaults to ``'same'``.
    :type padding: str
    :param padding_mode: How ``padding='same'`` is realised — see
        :func:`resolve_conv_padding`. ``'keras_same'`` (the default, and this
        layer's historical behaviour) uses Keras' ASYMMETRIC ``'same'``, under
        which a strided ``k x k`` branch and the strided ``1 x 1`` scale branch
        sample DIFFERENT input pixels and are therefore not fusible.
        ``'reference'`` uses PyTorch's symmetric ``padding = kernel_size // 2``
        (explicit :class:`keras.layers.ZeroPadding2D` + a ``'valid'``
        convolution), which puts every branch on the same grid. Ignored when
        ``padding='valid'``. For an ODD kernel at ``stride=1`` the two modes are
        value-identical. Defaults to ``'keras_same'``.
    :type padding_mode: str
    :param use_se: Whether to include Squeeze-and-Excitation. Defaults to False.
    :type use_se: bool
    :param num_conv_branches: Number of Conv-BN branches. Must be non-negative.
        ``0`` creates no ``k x k`` branch at all, which — combined with
        ``use_scale_branch=False``, ``stride=1`` and ``out_channels == in_channels`` —
        reduces the block to exactly the identity BatchNormalization. Defaults to 1.
    :type num_conv_branches: int
    :param group_size: Grouped-convolution control using timm's ``num_groups``
        semantics: ``0`` means ``groups = 1`` (a dense convolution, the default);
        ``k > 0`` means ``groups = in_channels // k``, so ``group_size=1`` is a
        DEPTHWISE convolution. ``in_channels`` is only known at build time, so the
        resolved group count is computed in :meth:`build` and exposed as ``self.groups``.
        The group count is applied to the ``k x k`` branches AND to the 1x1 scale
        branch. Defaults to 0.
    :type group_size: int
    :param use_act: Whether to apply ``activation`` in :meth:`call`. Defaults to True.
    :type use_act: bool
    :param use_scale_branch: Whether to create the 1x1 scale branch when
        ``kernel_size > 1``. When False no scale branch is created regardless of
        kernel size. Defaults to True.
    :type use_scale_branch: bool
    :param se_reduction_ratio: Bottleneck ratio forwarded to
        :class:`SqueezeExcitation`. Defaults to 0.25.
    :type se_reduction_ratio: float
    :param se_use_bias: Whether the Squeeze-and-Excitation convolutions use bias
        vectors. Forwarded to :class:`SqueezeExcitation`. Defaults to False.
    :type se_use_bias: bool
    :param norm_epsilon: Variance epsilon for EVERY BatchNormalization the block
        creates (the ``k x k`` branches, the ``1 x 1`` scale branch and the
        identity skip branch). Defaults to ``1e-3`` — Keras' own default, i.e.
        this layer's historical behaviour. The FastViT / MobileOne reference uses
        ``1e-5``; pass it explicitly for a faithful port.
    :type norm_epsilon: float
    :param se_position: Where the Squeeze-and-Excitation block sits relative to the
        activation. ``'post_act'`` (the default, and this layer's historical
        behaviour) computes ``se(act(x))``; ``'pre_act'`` computes ``act(se(x))``,
        which is the FastViT/timm reference order. Defaults to ``'post_act'``.
    :type se_position: str
    :param activation: Activation function to use. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for conv kernels. Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias terms. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for conv kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias terms.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for Layer base class.

    :raises ValueError: If out_channels, kernel_size or stride are not positive, if
        num_conv_branches or group_size are negative, if padding is invalid, or if
        se_position is not one of ``'post_act'`` / ``'pre_act'``. At build time, also
        if the resolved group count does not divide both the input and the output
        channel counts.
    """

    #: Accepted values for ``se_position``.
    _SE_POSITIONS = ('post_act', 'pre_act')

    def __init__(
            self,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: str = 'same',
            padding_mode: str = 'keras_same',
            use_se: bool = False,
            num_conv_branches: int = 1,
            group_size: int = 0,
            use_act: bool = True,
            use_scale_branch: bool = True,
            se_reduction_ratio: float = 0.25,
            se_use_bias: bool = False,
            norm_epsilon: float = 1e-3,
            se_position: str = 'post_act',
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if num_conv_branches < 0:
            raise ValueError(f"num_conv_branches must be non-negative, got {num_conv_branches}")
        if group_size < 0:
            raise ValueError(f"group_size must be non-negative, got {group_size}")
        if padding not in ['same', 'valid']:
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")
        if padding_mode not in PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}"
            )
        if norm_epsilon <= 0:
            raise ValueError(f"norm_epsilon must be positive, got {norm_epsilon}")
        if se_position not in self._SE_POSITIONS:
            raise ValueError(
                f"se_position must be 'post_act' or 'pre_act', got {se_position!r}"
            )

        # Store configuration
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.norm_epsilon = float(norm_epsilon)
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches
        self.group_size = group_size
        self.use_act = use_act
        self.use_scale_branch = use_scale_branch
        self.se_reduction_ratio = se_reduction_ratio
        self.se_use_bias = se_use_bias
        self.se_position = se_position
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # State management
        self.inference_mode = False

        # Resolved group count. Depends on the input channel count, so it can only be
        # finalised in build(); `group_size == 0` is knowable up front and is the
        # historical dense path.
        self.groups = 1 if self.group_size == 0 else None

        # CREATE all sub-layers in __init__ (unbuilt).
        #
        # Exception, deliberate: `groups` must be passed to Conv2D at CONSTRUCTION
        # time, but timm's `num_groups` semantics derive it from `in_channels`, which
        # is unknown until build(). So the convolutional branches are constructed here
        # ONLY for the `group_size == 0` (groups == 1) case — the historical default
        # path, which is left bit-for-bit as it was — and are otherwise constructed in
        # build() once `in_channels` is known. Construction is still driven purely by
        # config flags plus the channel count, and sub-layer names are deterministic,
        # so `.keras` round-tripping is unaffected (Keras calls build() from
        # `build_from_config` before restoring weights).
        self.conv_branches = []
        self.scale_branch = None
        if self.groups is not None:
            self._create_conv_branches(self.groups)
            self.scale_branch = self._create_scale_branch(self.groups)

        # Skip branch (will be created in build if applicable)
        self.skip_branch = None

        # SE block if requested - reuse dl_techniques implementation
        if use_se:
            self.se_block = SqueezeExcitation(
                reduction_ratio=self.se_reduction_ratio,
                use_bias=self.se_use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='se_block'
            )
        else:
            self.se_block = None

    def _create_conv_branches(self, groups: int) -> None:
        """Populate ``self.conv_branches`` with ``num_conv_branches`` Conv-BN branches.

        :param groups: Number of convolution groups to use.
        :type groups: int
        """
        self.conv_branches = []
        for i in range(self.num_conv_branches):
            conv_branch = keras.Sequential(
                self._padding_layers(self.kernel_size, f'conv_branch_{i}_pad') + [
                    layers.Conv2D(
                        filters=self.out_channels,
                        kernel_size=self.kernel_size,
                        strides=self.stride,
                        padding=self._keras_padding(self.kernel_size),
                        use_bias=False,
                        groups=groups,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name=f'conv_branch_{i}_conv'
                    ),
                    layers.BatchNormalization(
                        epsilon=self.norm_epsilon, name=f'conv_branch_{i}_bn')
                ], name=f'conv_branch_{i}')
            self.conv_branches.append(conv_branch)

    def _padding_layers(self, kernel_size: int, name: str) -> list:
        """Return the explicit padding layers preceding a convolution, if any.

        :param kernel_size: Kernel size of the convolution being padded for.
        :type kernel_size: int
        :param name: Name for the :class:`keras.layers.ZeroPadding2D` layer.
        :type name: str
        :return: ``[ZeroPadding2D(p)]`` under ``padding_mode='reference'`` with
            ``p > 0``, otherwise the empty list (Keras' own padding does the job).
        :rtype: list
        """
        pad, _ = resolve_conv_padding(kernel_size, self.padding, self.padding_mode)
        if pad == 0:
            return []
        return [layers.ZeroPadding2D(padding=pad, name=name)]

    def _keras_padding(self, kernel_size: int) -> str:
        """Return the ``padding`` value to pass to ``Conv2D`` for this kernel size.

        :param kernel_size: Kernel size of the convolution.
        :type kernel_size: int
        :return: ``'same'`` or ``'valid'``.
        :rtype: str
        """
        return resolve_conv_padding(
            kernel_size, self.padding, self.padding_mode)[1]

    def _create_scale_branch(self, groups: int) -> Optional[keras.Sequential]:
        """Build the optional 1x1 scale branch.

        :param groups: Number of convolution groups to use.
        :type groups: int
        :return: The scale branch, or ``None`` when it is not applicable.
        :rtype: Optional[keras.Sequential]
        """
        if not (self.use_scale_branch and self.kernel_size > 1):
            return None
        # The scale branch's kernel is 1x1, so under `padding_mode='reference'` its
        # symmetric pad is `1 // 2 == 0` — no padding layer at all. That is exactly
        # the point: with the reference convention the k x k branch also lands on
        # the `i * stride` grid, so the two branches sum the SAME input pixels.
        return keras.Sequential(
            self._padding_layers(1, 'scale_branch_pad') + [
                layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=1,
                    strides=self.stride,
                    padding=self._keras_padding(1),
                    use_bias=False,
                    groups=groups,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name='scale_branch_conv'
                ),
                layers.BatchNormalization(
                    epsilon=self.norm_epsilon, name='scale_branch_bn')
            ], name='scale_branch')

    def _resolve_groups(self, input_channels: int) -> int:
        """Resolve timm's ``num_groups(group_size, in_chs)`` at build time.

        :param input_channels: Number of input channels.
        :type input_channels: int
        :return: The convolution group count.
        :rtype: int
        :raises ValueError: If the group count does not divide both the input and the
            output channel counts.
        """
        groups = resolve_num_groups(self.group_size, input_channels)

        if input_channels % groups != 0 or self.out_channels % groups != 0:
            raise ValueError(
                f"resolved groups={groups} must divide both in_channels="
                f"{input_channels} and out_channels={self.out_channels}"
            )
        return groups

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create weights and build sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Input channels dimension must be defined")

        # Resolve the grouped-convolution count now that in_channels is known, and
        # create the grouped branches if __init__ could not (see the note there).
        resolved_groups = self._resolve_groups(input_channels)
        if self.groups is None:
            self.groups = resolved_groups
            self._create_conv_branches(resolved_groups)
            self.scale_branch = self._create_scale_branch(resolved_groups)

        # Create skip branch if input/output channels match and stride is 1
        if input_channels == self.out_channels and self.stride == 1:
            self.skip_branch = layers.BatchNormalization(
                epsilon=self.norm_epsilon, name='skip_branch_bn')

        # Build all sub-layers explicitly
        for branch in self.conv_branches:
            branch.build(input_shape)

        if self.scale_branch is not None:
            self.scale_branch.build(input_shape)

        if self.skip_branch is not None:
            self.skip_branch.build(input_shape)

        if self.se_block is not None:
            # SE block needs output shape after conv. Resolved via
            # compute_output_shape so it also works with zero conv branches.
            conv_output_shape = self.compute_output_shape(input_shape)
            self.se_block.build(conv_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the block.

        :param inputs: Input tensor of shape ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor.
        :rtype: keras.KerasTensor
        :raises ValueError: If the configuration leaves the block with no branch at
            all (no conv branches, no scale branch and no identity skip).
        """
        x = None

        # Conv branches. `x` stays None when num_conv_branches == 0, so every
        # subsequent branch must handle the "first contribution" case too.
        for branch in self.conv_branches:
            branch_out = branch(inputs, training=training)
            x = branch_out if x is None else x + branch_out

        # Scale branch
        if self.scale_branch is not None:
            scale_out = self.scale_branch(inputs, training=training)
            x = scale_out if x is None else x + scale_out

        # Skip branch
        if self.skip_branch is not None:
            skip_out = self.skip_branch(inputs, training=training)
            x = skip_out if x is None else x + skip_out

        if x is None:
            raise ValueError(
                "MobileOneBlock has no active branch: num_conv_branches=0 with no "
                "scale branch and no identity skip branch produces no output. "
                "Set num_conv_branches > 0, use_scale_branch=True, or use a "
                "configuration where stride == 1 and out_channels == in_channels."
            )

        # DECISION plan-2026-08-13T183738-24486492/D-002
        # SE ordering. `'post_act'` — se(act(x)) — is this layer's HISTORICAL order and
        # MUST remain the default: `models/fastvlm/` (via layers/repmixer_block.py's
        # ConvolutionalStem) ships trained-against numerics that depend on it. Do NOT
        # "fix" the default to match timm. `'pre_act'` — act(se(x)) — is the
        # FastViT/timm reference order and is available opt-in for the faithful port.
        # See decisions.md D-002 (the divergence is DISCLOSED, not repaired in place).
        if self.se_position == 'pre_act':
            if self.se_block is not None:
                x = self.se_block(x, training=training)
            if self.use_act:
                x = self.activation(x)
        else:
            if self.use_act:
                x = self.activation(x)
            if self.se_block is not None:
                x = self.se_block(x, training=training)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        if self.conv_branches:
            return self.conv_branches[0].compute_output_shape(input_shape)

        # Fallback calculation (no k x k branch was created). The scale branch, when
        # present, is a 1x1 convolution; otherwise only the identity skip survives,
        # which is stride 1 and shape-preserving by construction.
        has_scale_branch = self.use_scale_branch and self.kernel_size > 1
        kernel_size = 1 if has_scale_branch else self.kernel_size
        height = conv_output_size(
            input_shape[1], kernel_size, self.stride, self.padding, self.padding_mode)
        width = conv_output_size(
            input_shape[2], kernel_size, self.stride, self.padding, self.padding_mode)

        return (input_shape[0], height, width, self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'padding_mode': self.padding_mode,
            'norm_epsilon': self.norm_epsilon,
            'use_se': self.use_se,
            'num_conv_branches': self.num_conv_branches,
            'group_size': self.group_size,
            'use_act': self.use_act,
            'use_scale_branch': self.use_scale_branch,
            'se_reduction_ratio': self.se_reduction_ratio,
            'se_use_bias': self.se_use_bias,
            'se_position': self.se_position,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
