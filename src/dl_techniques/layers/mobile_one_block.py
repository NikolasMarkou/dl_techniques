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
    :param num_conv_branches: Number of Conv-BN branches. Must be positive. Defaults to 1.
    :type num_conv_branches: int
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

    :raises ValueError: If out_channels, kernel_size, stride, or num_conv_branches
        are not positive, or padding is invalid.
    """

    def __init__(
            self,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: str = 'same',
            use_se: bool = False,
            num_conv_branches: int = 1,
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
        if num_conv_branches <= 0:
            raise ValueError(f"num_conv_branches must be positive, got {num_conv_branches}")
        if padding not in ['same', 'valid']:
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")

        # Store configuration
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # State management
        self.inference_mode = False

        # CREATE all sub-layers in __init__ (unbuilt)
        self.conv_branches = []
        for i in range(self.num_conv_branches):
            conv_branch = keras.Sequential([
                layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    use_bias=False,
                    groups=1,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'conv_branch_{i}_conv'
                ),
                layers.BatchNormalization(name=f'conv_branch_{i}_bn')
            ], name=f'conv_branch_{i}')
            self.conv_branches.append(conv_branch)

        # Scale branch (1x1 conv) if main kernel is larger
        if kernel_size > 1:
            self.scale_branch = keras.Sequential([
                layers.Conv2D(
                    filters=out_channels,
                    kernel_size=1,
                    strides=stride,
                    padding=padding,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name='scale_branch_conv'
                ),
                layers.BatchNormalization(name='scale_branch_bn')
            ], name='scale_branch')
        else:
            self.scale_branch = None

        # Skip branch (will be created in build if applicable)
        self.skip_branch = None

        # SE block if requested - reuse dl_techniques implementation
        if use_se:
            self.se_block = SqueezeExcitation(
                reduction_ratio=0.25,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='se_block'
            )
        else:
            self.se_block = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create weights and build sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Input channels dimension must be defined")

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
            # SE block needs output shape after conv
            conv_output_shape = self.conv_branches[0].compute_output_shape(input_shape)
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
        """
        x = None

        # Conv branches
        for branch in self.conv_branches:
            branch_out = branch(inputs, training=training)
            x = branch_out if x is None else x + branch_out

        # Scale branch
        if self.scale_branch is not None:
            x = x + self.scale_branch(inputs, training=training)

        # Skip branch
        if self.skip_branch is not None:
            x = x + self.skip_branch(inputs, training=training)

        # Apply activation
        x = self.activation(x)

        # Apply SE block if present
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
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
