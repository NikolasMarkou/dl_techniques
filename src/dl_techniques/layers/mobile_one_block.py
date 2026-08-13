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
            use_se: bool = False,
            num_conv_branches: int = 1,
            group_size: int = 0,
            use_act: bool = True,
            use_scale_branch: bool = True,
            se_reduction_ratio: float = 0.25,
            se_use_bias: bool = False,
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
        if se_position not in self._SE_POSITIONS:
            raise ValueError(
                f"se_position must be 'post_act' or 'pre_act', got {se_position!r}"
            )

        # Store configuration
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
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
            conv_branch = keras.Sequential([
                layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    padding=self.padding,
                    use_bias=False,
                    groups=groups,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'conv_branch_{i}_conv'
                ),
                layers.BatchNormalization(name=f'conv_branch_{i}_bn')
            ], name=f'conv_branch_{i}')
            self.conv_branches.append(conv_branch)

    def _create_scale_branch(self, groups: int) -> Optional[keras.Sequential]:
        """Build the optional 1x1 scale branch.

        :param groups: Number of convolution groups to use.
        :type groups: int
        :return: The scale branch, or ``None`` when it is not applicable.
        :rtype: Optional[keras.Sequential]
        """
        if not (self.use_scale_branch and self.kernel_size > 1):
            return None
        return keras.Sequential([
            layers.Conv2D(
                filters=self.out_channels,
                kernel_size=1,
                strides=self.stride,
                padding=self.padding,
                use_bias=False,
                groups=groups,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='scale_branch_conv'
            ),
            layers.BatchNormalization(name='scale_branch_bn')
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
        if self.group_size == 0:
            groups = 1
        else:
            if input_channels % self.group_size != 0:
                raise ValueError(
                    f"group_size must divide the input channels: "
                    f"in_channels={input_channels}, group_size={self.group_size}"
                )
            groups = input_channels // self.group_size

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
            self.skip_branch = layers.BatchNormalization(name='skip_branch_bn')

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

        # Fallback calculation
        if self.padding == 'same':
            height = (input_shape[1] + self.stride - 1) // self.stride if input_shape[1] is not None else None
            width = (input_shape[2] + self.stride - 1) // self.stride if input_shape[2] is not None else None
        else:  # valid padding
            height = (input_shape[1] - self.kernel_size + self.stride) // self.stride if input_shape[
                                                                                             1] is not None else None
            width = (input_shape[2] - self.kernel_size + self.stride) // self.stride if input_shape[
                                                                                            2] is not None else None

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
