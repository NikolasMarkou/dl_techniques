"""create_convunext, a functional builder for a U-Net with ConvNeXt-style blocks.

create_convunext returns a functional `keras.Model` shaped like U-Net, with
`ConvUNextStem`, `SpatialLinearAttention`, the `CONVUNEXT_CONFIGS` variant
table and `create_convunext_variant` alongside it. Each level's work is done
by `ConvNextV1Block` or `ConvNextV2Block` (depthwise 7x7, inverted
bottleneck, Global Response Normalization in V2, layer scale, optional
stochastic depth) applied as a residual branch. One builder covers two arms:
`use_bias=True` builds a bias-carrying network and `use_bias=False` builds
the bias-free denoiser that `bias_free_denoisers/bfconvunext.py` wraps, whose
output scales with its input. Optional paths replace the pooling and raw skip
at each encoder junction with a Laplacian low/high frequency split, and add
bias-free linear attention at the bottleneck. Callers should know that
`use_bias=False` triggers argument guardrails and does not by itself make the
graph degree-1 homogeneous, that `enable_deep_supervision` and
`expose_bottleneck` change the number of model outputs, and that no
pretrained weights are involved.

References:
    - Ronneberger et al., 2015. U-Net: Convolutional Networks for Biomedical
      Image Segmentation. (https://arxiv.org/abs/1505.04597)
    - Liu et al., 2022. A ConvNet for the 2020s (ConvNeXt V1).
      (https://arxiv.org/abs/2201.03545)
    - Woo et al., 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with
      Masked Autoencoders. (https://arxiv.org/abs/2301.00808)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Lee et al., 2015. Deeply-Supervised Nets. AISTATS 2015.
      (https://arxiv.org/abs/1409.5185)
    - Mohan et al., 2020. Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks. ICLR 2020.
      (https://arxiv.org/abs/1906.05478)
    - Burt and Adelson, 1983. The Laplacian Pyramid as a Compact Image Code.
      IEEE Trans. Communications 31(4)
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, List, Dict, Any, FrozenSet

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.conv_blocks.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.conv_blocks.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.conv_blocks.match_channels import MatchChannels
from dl_techniques.layers.pooling.downsample_and_skip import DownsampleAndSkip
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.initializers import (
    create_gabor_conv2d,
    create_gabor_depthwise_conv2d,
)
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.convunext.model")
class ConvUNextStem(keras.layers.Layer):
    """Extract initial features with a wide convolution, normalization and activation.

    Serves both ConvUNext arms: the normalization type and the bias flag are
    parameters, so ``'global_response_norm'`` with ``use_bias=False`` gives the
    bias-free denoiser stem and ``'layer_norm'`` with ``use_bias=True`` gives the
    standard ConvNeXt stem.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ conv2d filters, padding same │
        └──────────────────────────────┘
              │ [B, H, W, filters]
              ▼
        ┌──────────────────────────────┐
        │ stem_normalization           │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ activation                   │
        └──────────────────────────────┘
              │
              ▼
        output [B, H, W, filters]

    Stride is 1 and padding is ``'same'``, so the spatial dimensions are kept.

    Input shape:
        4D tensor ``(batch, height, width, channels)``.

    Output shape:
        4D tensor ``(batch, height, width, filters)``.

    :param filters: Number of output filters.
    :type filters: int
    :param kernel_size: Spatial size of the convolution kernel. Defaults to 7.
    :type kernel_size: int or tuple of 2 ints
    :param activation: Activation applied after the normalization. May be a string
        or a ``keras.layers.Layer`` instance (e.g. ``LeakyReLU(0.1)``). Defaults to
        ``'gelu'``. Pass ``'linear'`` for a stem with no activation.
    :type activation: str or keras.layers.Layer
    :param use_bias: Whether the stem convolution allocates a bias vector. Defaults
        to ``True``. Bias-free / Miyasawa denoisers pass ``False``, since degree-1
        homogeneity requires a zero additive offset.
    :type use_bias: bool
    :param stem_normalization: Registered normalization type built through
        ``create_normalization_layer``. Defaults to ``'global_response_norm'``
        (the ConvNeXt-V2 / bias-free choice); ``'layer_norm'`` reproduces the
        standard ConvNeXt stem.
    :type stem_normalization: str
    :param kernel_initializer: Initializer for the convolution kernel. Defaults to
        ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the convolution kernel.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 7,
            activation: Union[str, keras.layers.Layer] = 'gelu',
            use_bias: bool = True,
            stem_normalization: str = 'global_response_norm',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation_name = deserialize_activation(activation)
        self.use_bias = use_bias
        self.stem_normalization = stem_normalization
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # The sub-layers need the input shape, so build() creates them.
        self.conv = None
        self.norm = None
        self.activation_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the stem sub-layers.

        :param input_shape: Shape of the input tensor ``(batch, H, W, C)``.
        :type input_shape: tuple of int or None
        """
        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stem_conv'
        )

        # The factory keeps both arms expressible from one string argument.
        self.norm = create_normalization_layer(
            self.stem_normalization,
            name='stem_norm'
        )

        # Explicit builds, because lazy auto-build drops sub-layer state on reload.
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(conv_output_shape)

        # Normalization keeps the shape, so the activation takes conv_output_shape.
        self.activation_layer = keras.layers.Activation(
            self.activation_name, name='stem_activation'
        )
        self.activation_layer.build(conv_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the forward pass.

        :param inputs: Input tensor of shape ``(batch, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Accepted for API symmetry. None of the three sub-layers
            receives it, and none of them behaves differently in training.
        :type training: bool or None
        :return: Output tensor of shape ``(batch, H, W, filters)``.
        :rtype: keras.KerasTensor
        """
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation_layer(x)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple of int or None
        :return: Input shape with the channel axis replaced by ``filters``.
        :rtype: tuple of int or None
        """
        return tuple(input_shape)[:-1] + (self.filters,)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Configuration dictionary containing every constructor parameter.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            # DECISION plan_2026-06-21_eb7fd829/D-005: serialize a layer-instance stem
            # activation so LeakyReLU(alpha) round-trips. See decisions.md.
            'activation': serialize_activation(self.activation_name),
            'use_bias': self.use_bias,
            'stem_normalization': self.stem_normalization,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ConvUNextStem':
        """Rebuild the layer, reviving a layer-instance activation from its dict form.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: dict
        :return: Reconstructed layer instance.
        :rtype: ConvUNextStem
        """
        config = dict(config)
        if isinstance(config.get('activation'), dict):
            config['activation'] = keras.layers.deserialize(config['activation'])
        # keras.*.get(...) accepts a serialized dict in Keras 3, so these pass through.
        return cls(**config)

# ---------------------------------------------------------------------
# Spatial wrapper around bias-free LinearAttention (4D <-> 3D)
# ---------------------------------------------------------------------

# DECISION plan_2026-07-11_bb4b38b5/D-002: registered under this module's path.
@register_dl_technique("dl_techniques.models.convunext.model")
class SpatialLinearAttention(keras.layers.Layer):
    """Apply a bias-free LinearAttention over a 4D spatial feature map.

    ``LinearAttention`` accepts 3D sequence input ``(B, N, dim)`` and raises on 4D
    input. This wrapper flattens ``(B, H, W, C)`` to ``(B, H*W, C)`` with dynamic
    ``ops.shape``, so it also works when H and W are ``None`` at graph-build time,
    attends, and reshapes back. The inner attention is built with a fixed
    ``'linear'`` type, ``use_bias=False`` and the default ``feature_map='relu'``,
    which keeps it degree-1 homogeneous. That inner flag is not threaded from
    ``create_convunext``'s ``use_bias``.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
              │
              ▼
        reshape [B, H*W, dim]
              │
              ▼
        ┌──────────────────────────────┐
        │ linear attention, no bias    │
        │ heads = num_heads            │
        └──────────────────────────────┘
              │ [B, H*W, dim]
              ▼
        reshape [B, H, W, dim]
              │
              ▼
        output [B, H, W, dim]

    The reshapes use ``dim``, so the input channel count must equal ``dim``.

    Input shape:
        4D tensor ``(batch, height, width, dim)``.

    Output shape:
        Same as the input.

    :param dim: Channel count of the input feature map, also the attention
        embedding dim. Must be divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param name: Optional layer name.
    :type name: str or None
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.num_heads = num_heads

        # DECISION plan_2026-07-11_bb4b38b5/D-001: attention type stays fixed at
        # 'linear'; a type knob admits softmax and breaks Miyasawa. See decisions.md.
        self.attn = create_attention_layer(
            'linear', dim=self.dim, num_heads=self.num_heads,
            use_bias=False, name=f'{self.name}_linear'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the nested attention on the flattened sequence shape.

        ``input_shape`` is the 4D spatial shape, while the attention only ever sees
        ``(B, H*W, dim)``, so it is built with a dynamic sequence length. Building
        it here materializes its four Dense projections before a ``.keras`` load,
        which is what lets ``keras.models.load_model`` restore their weights.

        :param input_shape: Shape of the input tensor ``(B, H, W, C)``.
        :type input_shape: tuple of int or None
        """
        self.attn.build((input_shape[0], None, self.dim))
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Flatten the spatial axes, attend, and reshape back.

        :param inputs: Input tensor ``(B, H, W, dim)``.
        :param training: Forwarded to the attention sub-layer.
        :return: Output tensor of the same shape as the input.
        """
        shape = ops.shape(inputs)
        b, h, w = shape[0], shape[1], shape[2]
        seq = ops.reshape(inputs, [b, h * w, self.dim])
        attended = self.attn(seq, training=training)
        return ops.reshape(attended, [b, h, w, self.dim])

    def compute_output_shape(self, input_shape):
        """Return the input shape, since the layer preserves it."""
        return input_shape

    def get_config(self):
        """Return the constructor configuration.

        :return: Configuration dictionary. The attention sub-layer is rebuilt from
            these values in ``__init__``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
        })
        return config


# ---------------------------------------------------------------------
# ConvUNext Model Variant Configurations
# ---------------------------------------------------------------------

# One table for both arms, carrying no `block_normalization` key: adding one here
# would flip the bias-on variants too. See decisions.md D-003.
CONVUNEXT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'depth': 3,
        'initial_filters': 32,
        'blocks_per_level': 2,
        'convnext_version': 'v2',
        'drop_path_rate': 0.0,
        'description': 'Tiny ConvUNext (depth=3) for quick experiments.'
    },
    'small': {
        'depth': 3,
        'initial_filters': 48,
        'blocks_per_level': 2,
        'convnext_version': 'v2',
        'drop_path_rate': 0.1,
        'description': 'Small ConvUNext (depth=3) with minimal capacity.'
    },
    'base': {
        'depth': 4,
        'initial_filters': 64,
        'blocks_per_level': 3,
        'convnext_version': 'v2',
        'drop_path_rate': 0.1,
        'description': 'Base ConvUNext (depth=4) with standard configuration.'
    },
    'large': {
        'depth': 4,
        'initial_filters': 96,
        'blocks_per_level': 4,
        'convnext_version': 'v2',
        'drop_path_rate': 0.2,
        'description': 'Large ConvUNext (depth=4) with high capacity.'
    },
    'xlarge': {
        'depth': 5,
        'initial_filters': 128,
        'blocks_per_level': 5,
        'convnext_version': 'v2',
        'drop_path_rate': 0.3,
        'description': 'Extra-Large ConvUNext (depth=5) for maximum performance.'
    }
}

# ---------------------------------------------------------------------
# Residual ConvNeXt block application (with stochastic depth)
# ---------------------------------------------------------------------

def _apply_residual_convnext_block(
        x: keras.KerasTensor,
        block_cls: type,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        drop_path_rate: float,
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
        name: str,
        activation: Union[str, keras.layers.Layer] = 'gelu',
        depthwise_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
        depthwise_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        dropout_rate: float = 0.0,
        normalization_type: str = "layernorm",
        use_bias: bool = False,
) -> keras.KerasTensor:
    """Apply a ConvNeXt block as a residual branch with stochastic depth.

    ``ConvNextV1Block`` and ``ConvNextV2Block`` implement the residual branch
    alone: they end at the layer-scale multiply and add neither the skip nor
    drop-path. This helper supplies both, giving the canonical wiring

        x = x + StochasticDepth(drop_path_rate)(block(x))

    Architecture:

    .. code-block:: text

        x [B, H, W, filters]
        ├──────────────────────────────────────┐
        ▼                                      │
        ┌──────────────────────────────┐       │
        │ convnext branch, to gamma    │       │
        └──────────────────────────────┘       │
              │                                │
              ▼                                │
        ┌──────────────────────────────┐       │
        │ stochastic depth             │       │  (rate > 0 only)
        └──────────────────────────────┘       │
              │                                │
              ▼                                │
             (+)◄──────────────────────────────┘
              │
              ▼
        out [B, H, W, filters]

    Callers channel-adjust before the blocks, so the branch input and output both
    have ``filters`` channels and the add is always valid.

    :param x: Input tensor with ``filters`` channels.
    :type x: keras.KerasTensor
    :param block_cls: ``ConvNextV1Block`` or ``ConvNextV2Block``.
    :type block_cls: type
    :param filters: Channel count of the branch input and output.
    :type filters: int
    :param kernel_size: Depthwise kernel size inside the block.
    :type kernel_size: int or tuple of 2 ints
    :param drop_path_rate: Stochastic-depth probability for the whole branch. At
        ``0.0`` no ``StochasticDepth`` layer is added.
    :type drop_path_rate: float
    :param kernel_regularizer: Regularizer forwarded to the block.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param name: Name prefix for the block, its drop-path and its add.
    :type name: str
    :param activation: Activation inside the block's inverted bottleneck.
        Defaults to ``'gelu'``.
    :type activation: str or keras.layers.Layer
    :param depthwise_initializer: Initializer for the block's depthwise kernel.
    :type depthwise_initializer: str or keras.initializers.Initializer or None
    :param depthwise_regularizer: Regularizer for the block's depthwise kernel.
    :type depthwise_regularizer: str or keras.regularizers.Regularizer or None
    :param dropout_rate: Element-wise MLP dropout inside the block's inverted
        bottleneck, applied after the expansion activation in V1 and after GRN in
        V2. This is not stochastic depth. ``0.0`` adds no ``Dropout`` sub-layer.
        ``spatial_dropout_rate`` is fixed at ``0.0`` and not exposed.
    :type dropout_rate: float
    :param normalization_type: Pre-activation normalization inside the block,
        ``'layernorm'`` or ``'batchnorm'``.
    :type normalization_type: str
    :param use_bias: Whether the block's convolutions allocate biases. Defaults to
        ``False``, the bias-free denoiser value.
    :type use_bias: bool
    :return: The tensor after the residual add.
    :rtype: keras.KerasTensor

    Note:
        Layer-scale ``gamma`` starts at 1e-4, small enough for a near-identity
        prior and large enough to pass gradient from step 0, since the gradient
        into the branch weights is proportional to gamma.
        ``ConvNext*Block.GAMMA_MIN_VALUE`` floors it at 1e-6 so a branch cannot
        reach gamma 0 and stop learning.
    """
    residual = x
    # DECISION plan-2026-08-11T201945-91938f65/D-002 + D-004: the block is the branch
    # only; never collapse this to `x = block(x)`. See decisions.md.
    # DECISION plan_2026-06-21_eb7fd829/D-002: block activation threads through this
    # one choke-point; the default stays 'gelu'. See decisions.md D-002/D-005/D-006.
    y = block_cls(
        kernel_size=kernel_size,
        filters=filters,
        activation=activation,
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        spatial_dropout_rate=0.0,
        gamma_initial_value=1e-4,
        kernel_regularizer=kernel_regularizer,
        depthwise_initializer=depthwise_initializer,
        depthwise_regularizer=depthwise_regularizer,
        normalization_type=normalization_type,
        name=name,
    )(x)
    if drop_path_rate and drop_path_rate > 0.0:
        y = StochasticDepth(drop_path_rate, name=f'{name}_drop_path')(y)
    return keras.layers.Add(name=f'{name}_residual')([residual, y])


def _make_supervision_activation(activation, name):
    """Build a serialization-safe activation layer for the deep-supervision head.

    ``keras.layers.Activation(<layer instance>)`` does not round-trip through
    ``.keras`` in a functional graph, because the Functional ``from_config`` cannot
    deserialize a layer-instance activation. A string activation and a bare cloned
    activation layer both round-trip, so a layer instance is cloned under a fresh
    name and applied directly, and a string is wrapped in ``Activation``.

    :param activation: Activation string or layer instance.
    :type activation: str or keras.layers.Layer
    :param name: Name for the returned layer.
    :type name: str
    :return: A layer that applies the activation.
    :rtype: keras.layers.Layer
    """
    # DECISION plan_2026-06-21_eb7fd829/D-006: pass a string or a cloned bare layer;
    # never Activation(<live layer instance>). See decisions.md D-006.
    if isinstance(activation, keras.layers.Layer):
        cfg = keras.layers.serialize(activation)
        cfg = {**cfg, "config": {**cfg["config"], "name": name}}
        return keras.layers.deserialize(cfg)
    return keras.layers.Activation(activation, name=name)


# ---------------------------------------------------------------------
# Bias-free (use_bias=False) guardrails
# ---------------------------------------------------------------------

# DECISION plan-2026-08-14T092357-0e3d792d/D-012: an allowlist, not a denylist; a
# denylist admits unlisted activations. See decisions.md D-006/D-012.
POSITIVELY_HOMOGENEOUS_ACTIVATIONS: FrozenSet[Optional[str]] = frozenset(
    {None, 'linear', 'relu', 'leaky_relu'}
)


def _validate_bias_free_arguments(
        final_activation: Union[str, callable],
        gabor_activation: Optional[str],
        use_gabor_stem: bool,
        supervision_norm_center: bool,
        block_normalization: str,
) -> None:
    """Reject the arguments that break degree-1 homogeneity on the bias-free arm.

    Called from :func:`create_convunext` only when ``use_bias is False``. Under
    ``use_bias=True`` none of these arguments is a defect, so the whole function is
    inert on that arm.

    Three checks raise and one warns:

    - ``final_activation`` must name a member of
      :data:`POSITIVELY_HOMOGENEOUS_ACTIVATIONS`.
    - ``gabor_activation`` likewise, but only when ``use_gabor_stem`` is True. With
      the Gabor stem off the argument reaches no layer, so a configuration that is
      homogeneous would otherwise be rejected.
    - ``supervision_norm_center=True`` puts a trainable additive offset on the
      deep-supervision head LayerNorm, which is a bias by another name.
    - ``block_normalization='layernorm'`` warns and builds. Per-input LayerNorm is
      scale-invariant (degree 0) rather than degree 1, so it does break
      homogeneity, but it is the shipped default of both arms and raising would
      stop every existing bias-free caller.

    Two consequences of those rules:

    - A callable activation warns and never raises. Its homogeneity is a property
      of code this function cannot inspect, so neither raising nor staying silent
      fits.
    - ``supervision_norm_center=True`` raises even when
      ``enable_deep_supervision=False``, when no supervision head is built. The
      check is a pure function of its arguments, so it reports the stated intent.

    :param final_activation: The builder's ``final_activation`` argument.
    :type final_activation: str or callable
    :param gabor_activation: The builder's ``gabor_activation`` argument.
    :type gabor_activation: str or None
    :param use_gabor_stem: Whether the Gabor stem is built at all; scopes
        the ``gabor_activation`` clause.
    :type use_gabor_stem: bool
    :param supervision_norm_center: The builder's ``supervision_norm_center``.
    :type supervision_norm_center: bool
    :param block_normalization: The builder's ``block_normalization``.
    :type block_normalization: str
    :raises ValueError: If ``final_activation`` or (when the Gabor stem is on)
        ``gabor_activation`` names a non-positively-homogeneous activation, or if
        ``supervision_norm_center`` is True.
    :return: None. The function's whole effect is raising or warning.
    :rtype: None
    """
    allowed = sorted(a for a in POSITIVELY_HOMOGENEOUS_ACTIVATIONS if a is not None)
    allowed_msg = f"None or one of {allowed}"

    def _check_activation(arg_name: str, value: Any) -> None:
        if value is None or isinstance(value, str):
            if value not in POSITIVELY_HOMOGENEOUS_ACTIVATIONS:
                raise ValueError(
                    f"{arg_name}={value!r} is not positively homogeneous, and "
                    f"use_bias=False requires it to be. Allowed: {allowed_msg}. "
                    f"A non-homogeneous activation (gelu, elu, tanh, sigmoid, mish, "
                    f"swish, softmax, ...) breaks the degree-1 homogeneity "
                    f"f(a*x) = a*f(x) the bias-free stack rests on."
                )
            return
        # A callable or layer-instance activation cannot be checked statically.
        logger.warning(
            f"{arg_name} is a callable ({value!r}), not a string, so its "
            f"positive homogeneity cannot be checked statically under "
            f"use_bias=False. Building anyway. Verify f(a*x) = a*f(x) yourself; "
            f"the statically-checkable choices are {allowed_msg}."
        )

    _check_activation('final_activation', final_activation)

    if use_gabor_stem:
        _check_activation('gabor_activation', gabor_activation)

    if supervision_norm_center:
        raise ValueError(
            "supervision_norm_center=True is incompatible with use_bias=False: the "
            "deep-supervision head LayerNorm's `center` adds a trainable additive "
            "offset (beta), which is a bias by another name. Pass "
            "supervision_norm_center=False. NOTE: this raises regardless of "
            "enable_deep_supervision -- the guard is a pure function of its "
            "arguments, so a contradictory intent is reported even when no "
            "supervision head is built."
        )

    if block_normalization == 'layernorm':
        logger.warning(
            "block_normalization='layernorm' under use_bias=False: per-input "
            "LayerNorm divides by a per-sample std that itself scales with the "
            "input, so it is scale-INVARIANT (degree 0), NOT degree-1 "
            "f(a*x) = a*f(x). This WARNS rather than raises because 'layernorm' is "
            "the shipped default of both arms. Pass block_normalization='batchnorm' "
            "(variance-only BiasFreeBatchNorm) for a homogeneous bias-free stack."
        )


# ---------------------------------------------------------------------
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_convunext(
        input_shape: Tuple[int, int, int],
        use_bias: bool = True,
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: float = 2.0,
        blocks_per_level: int = 2,
        convnext_version: str = 'v2',
        stem_kernel_size: Union[int, Tuple[int, int]] = 7,
        stem_normalization: str = 'global_response_norm',
        use_gabor_stem: bool = False,
        gabor_filters: int = 32,
        gabor_filters_per_channel: Optional[int] = None,
        gabor_kernel_size: Union[int, Tuple[int, int]] = 11,
        gabor_activation: Optional[str] = None,
        gabor_stem_projection: bool = True,
        use_laplacian_pyramid: bool = False,
        laplacian_kernel_size: Tuple[int, int] = (5, 5),
        high_freq_blocks: int = 0,
        bottleneck_attention_blocks: int = 0,
        bottleneck_attention_heads: int = 8,
        zero_pad_channels: bool = False,
        extra_zero_output_channels: bool = False,
        final_projection_groups: int = 1,
        downsample_pool_type: str = "max",
        expose_bottleneck: bool = False,
        block_kernel_size: Union[int, Tuple[int, int]] = 7,
        block_activation: Union[str, keras.layers.Layer] = 'gelu',
        # DECISION plan_2026-07-01_8054f023/D-001: 'batchnorm' means the variance-only
        # BiasFreeBatchNorm; stock BatchNormalization and RMS norms measure
        # non-homogeneous. See decisions.md.
        block_normalization: str = "layernorm",
        stem_activation: Union[str, keras.layers.Layer] = 'gelu',
        drop_path_rate: float = 0.1,
        final_activation: Union[str, callable] = 'linear',
        # Orthogonal keeps the main-path structural convs norm-preserving; he_normal
        # compounds variance through the residual trunk and explodes a deep U-Net.
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'orthogonal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        depthwise_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
        depthwise_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        dropout_rate: float = 0.0,
        enable_deep_supervision: bool = False,
        supervision_norm_scale: bool = True,
        supervision_norm_center: bool = False,
        supervision_activation: Union[str, keras.layers.Layer] = 'gelu',
        include_top: bool = True,
        output_channels: Optional[int] = None,
        model_name: str = 'convunext'
) -> keras.Model:
    """Build a ConvUNext model as a Keras functional graph.

    One builder covers both arms. ``use_bias=True`` builds a bias-carrying
    network; ``use_bias=False`` builds the bias-free denoiser that
    ``bfconvunext.create_convunext_denoiser`` wraps, whose output scales with its
    input. The encoder-decoder shape with skips comes from U-Net, and every level's
    work is done by residual ConvNeXt V1 or V2 blocks.

    Architecture:

    .. code-block:: text

        input [B, H, W, C_in]
              │
              ▼
        ┌─────────────────────────────┐
        │ gabor stem, 1x1 projection  │  (use_gabor_stem)
        └─────────────────────────────┘
              │ [B, H, W, F_0]
              ▼
        ┌─────────────────────────────┐
        │ encoder level i, i=0..d-1   ├──► skip_i
        └─────────────────────────────┘
              │ [B, H/2^d, W/2^d, F_d]
              ▼
        ┌─────────────────────────────┐
        │ bottleneck attention x a    │  (optional)
        │ convnext block x n          │
        └─────────────────────────────┘
              ├──► bottleneck tap        (expose_bottleneck)
              ▼
        ┌─────────────────────────────┐
        │ decoder level i, i=d-1..0   │◄── skip_i
        └─────────────────────────────┘
              ├──► supervision head      (i > 0, optional)
              │ [B, H, W, F_0]
              ▼
        ┌─────────────────────────────┐
        │ output head                 │
        └─────────────────────────────┘
              │
              ▼
        output [B, H, W, output_channels]

    Without the Gabor stem, encoder level 0 starts with ``ConvUNextStem``.

    Encoder level i:

    .. code-block:: text

        in [B, h, w, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ convunext stem at level 0    │
        │ else 1x1 channel adjust      │
        └──────────────────────────────┘
              │ [B, h, w, F_i]
              ▼
        ┌──────────────────────────────┐
        │ convnext block x n           │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ downsample and skip          │
        └──────────────────────────────┘
              ├──► skip_i [B, h, w, F_i]
              │    convnext x hf         (pyramid only)
              ▼
        out [B, h/2, w/2, F_i]

    The channel adjust is skipped when the widths already match.

    Downsample junction:

    .. code-block:: text

        use_laplacian_pyramid=False    use_laplacian_pyramid=True

              x                              x
              ├──► skip: x itself            ├──► skip: high band
              ▼                              ▼
        pool or strided conv           blur, then subsample
              ▼                              ▼
        [h/2, w/2, F_i]                [h/2, w/2, F_i]

    Decoder level i:

    .. code-block:: text

        in [B, h, w, F_{i+1}]
              │
              ▼
        upsample x2, bilinear
              │
              ▼
        resize to skip_i              (only when the dims differ)
              │
              ▼
        ┌──────────────────────────────┐
        │ concat, then 1x1 adjust      │◄── skip_i
        │ or match, then add           │    (zero_pad_channels)
        └──────────────────────────────┘
              │ [B, H_i, W_i, F_i]
              ▼
        zero tail pad                 (level 0, extra tail)
              │
              ▼
        ┌──────────────────────────────┐
        │ convnext block x n           │
        └──────────────────────────────┘
              ├──► supervision head      (i > 0)
              ▼
        out [B, H_i, W_i, F_i]

    Supervision head and output head:

    .. code-block:: text

        supervision head
        x ──► 1x1 to F_i/2 ──► layer norm ──► activation
          ──► 1x1 to output_channels, final_activation

        output head
        include_top=False           ──► linear tap, decoder features
        extra_zero_output_channels  ──► tail slice, then final_activation
        otherwise                   ──► 1x1 conv, final_activation

    Drop-path schedule:

    .. code-block:: text

        encoder block j at level i    rate * (i*n + j) / (d*n)
        decoder block j at level i    0 at j=0, else as above
        bottleneck block j            rate * j / n
        high band block j             rate * j / high_freq_blocks
        attention block j             rate * j / attention_blocks

    Bias-free arm. ``use_bias=False`` drops the bias from the threaded
    convolutions. It does not make the graph degree-1 homogeneous on its own,
    because three sites sit outside the flag:

    1. ``block_activation``, ``stem_activation`` and ``supervision_activation``
       default to ``'gelu'``, which is not positively homogeneous, and are
       unguarded so that the shipped default configuration builds. Pass
       ``'relu'``, ``'leaky_relu'`` or a ``LeakyReLU`` instance for a homogeneous
       network.
    2. Two sites hardcode ``use_bias=False`` instead of reading this argument:
       ``SpatialLinearAttention``'s inner ``create_attention_layer('linear', ...)``
       and the trainable Gabor stem. Both stay bias-free even when
       ``use_bias=True``.
    3. ``GlobalResponseNormalization``'s ``use_beta`` is not threaded, so a
       trainable additive beta exists in every V2 block and in the
       ``'global_response_norm'`` stem on both arms (D-GAP-1).

    Guardrails, from :func:`_validate_bias_free_arguments`. On the bias-off arm
    ``final_activation``, and ``gabor_activation`` when ``use_gabor_stem=True``,
    must be in :data:`POSITIVELY_HOMOGENEOUS_ACTIVATIONS`, and
    ``supervision_norm_center=True`` is rejected; all three are inert on the
    bias-on arm. A callable activation warns instead of raising, since it cannot
    be checked statically, and ``supervision_norm_center=True`` raises even when
    ``enable_deep_supervision=False``. ``block_normalization='layernorm'``, the
    default on both arms, only warns. Given point 1 above, passing these checks is
    not a homogeneity certificate.

    :param input_shape: Shape of input images ``(height, width, channels)``.
    :type input_shape: tuple of 3 ints
    :param use_bias: Whether the threaded convolutions (stem, Gabor projection,
        channel adjusts, ``'strided_conv'`` downsample, ConvNeXt blocks,
        supervision heads, final projection) allocate a bias vector. Defaults to
        ``True``. Pass ``False`` for the bias-free / Miyasawa denoiser arm, and
        read the bias-free section above before treating ``False`` as a
        homogeneity guarantee.
    :type use_bias: bool
    :param depth: Depth of the U-Net (number of downsampling levels). Must be >= 2.
        Defaults to 4.
    :type depth: int
    :param initial_filters: Number of filters at the first level. Defaults to 64.
    :type initial_filters: int
    :param filter_multiplier: Per-encoder-level channel-growth multiplier (``>= 1``).
        Channels at level ``i`` are ``int(round(initial_filters * filter_multiplier
        ** i))``. Defaults to ``2.0``, which doubles per level.
    :type filter_multiplier: float
    :param blocks_per_level: Number of ConvNeXt blocks per level. Defaults to 2.
    :type blocks_per_level: int
    :param convnext_version: ``'v1'`` or ``'v2'``. Defaults to ``'v2'``.
    :type convnext_version: str
    :param stem_kernel_size: Size of the stem convolution kernel. Defaults to 7.
    :type stem_kernel_size: int or tuple of 2 ints
    :param stem_normalization: Registered normalization type for the stem, built
        through ``create_normalization_layer``. Defaults to
        ``'global_response_norm'`` (the ConvNeXt-V2 / bias-free choice);
        ``'layer_norm'`` reproduces the standard ConvNeXt stem. Only used when the
        standard stem is built, i.e. ``use_gabor_stem=False``.
    :type stem_normalization: str
    :param use_gabor_stem: If True prepend a trainable, bias-free cross-channel
        ``Conv2D`` Gabor stem, warm started with a Gabor filter bank per Ozbulak &
        Ekenel (SIU 2018), followed by a 1x1 projection to ``initial_filters``,
        instead of the ``ConvUNextStem``. Defaults to False.

        .. note::

           The stem is always built with ``trainable=True`` and there is no kwarg to
           freeze it. A caller who wants it frozen at its Gabor initialization flips
           ``trainable`` on the built, not-yet-compiled model;
           ``train.bfunet.common.freeze_gabor_stem_if_requested`` does that, and both
           bfunet U-Net trainers expose it as ``--freeze-gabor-stem``. Freezing and
           the depthwise stem are separate axes and compose:
           ``gabor_filters_per_channel=N`` plus a post-build freeze gives a frozen
           per-channel front end.
    :type use_gabor_stem: bool
    :param gabor_filters: Output channel count of the Gabor stem, passed as the
        ``Conv2D`` ``filters``, so the stem emits exactly ``gabor_filters`` channels
        and the 1x1 projection maps them to ``initial_filters``. Not a per-channel
        multiplier. Only used when ``use_gabor_stem=True`` and
        ``gabor_filters_per_channel`` is None, since the depthwise stem has no
        ``filters``. Defaults to 32.
    :type gabor_filters: int
    :param gabor_filters_per_channel: Opt-in switch to the depthwise Gabor stem: a
        ``DepthwiseConv2D`` with ``depth_multiplier = gabor_filters_per_channel``,
        built by
        :func:`~dl_techniques.initializers.gabor_filters_initializer.create_gabor_depthwise_conv2d`.
        ``None`` (default) keeps the cross-channel ``Conv2D`` stem. When set (must be
        >= 1) the bank is applied to each input channel independently with no
        cross-channel summing, so the stem emits ``input_channels *
        gabor_filters_per_channel`` channels and ``gabor_filters`` is not read.

        Output channel ``(c, j)`` is Gabor filter ``j``'s response to input channel
        ``c`` alone, so the stem is colour-selective at initialization rather than
        colour-blind (see the stem's ``standalone-2026-09-06-gabor-warm-start/D-001``
        anchor). The bank is built ``trainable=True``, which overrides
        ``create_gabor_depthwise_conv2d``'s own default, so that
        ``--freeze-gabor-stem`` stays the single knob that freezes a stem of either
        kind.

        Bias-free and therefore degree-1 homogeneous exactly as the ``Conv2D`` stem
        is: homogeneity comes from ``use_bias=False``, not from the depthwise form.

        Raises ``ValueError`` when set together with ``use_gabor_stem=False``, where
        it would reach no layer, or when ``< 1``.
    :type gabor_filters_per_channel: int or None
    :param gabor_kernel_size: Kernel size of the Gabor stem convolution. Defaults to 11.
    :type gabor_kernel_size: int or tuple of 2 ints
    :param gabor_activation: Optional activation on the Gabor stem. ``None``
        (default) is a linear passthrough. Under ``use_bias=False`` it must be
        positively homogeneous (relu, leaky_relu, linear); gelu, elu, tanh, sigmoid
        and mish break degree-1 homogeneity. Only used when ``use_gabor_stem=True``.
    :type gabor_activation: str or None
    :param gabor_stem_projection: If True (default) the Gabor stem is followed by a
        bias-free 1x1 projection to ``initial_filters``. If False the projection is
        dropped and the Gabor stem feeds the encoder directly, which is valid only
        when the stem already emits exactly ``initial_filters`` channels (it raises
        ``ValueError`` otherwise). The required equality depends on the stem kind:

        - cross-channel ``Conv2D`` stem (``gabor_filters_per_channel=None``):
          ``gabor_filters == initial_filters``.
        - depthwise stem (``gabor_filters_per_channel`` set): ``input_channels *
          gabor_filters_per_channel == initial_filters``.

        Dropping the projection also means something different on each arm. The
        ``Conv2D`` stem already mixes across input channels, so the encoder still
        receives mixed features. The depthwise stem does not mix at all, so the
        first ConvNeXt block's pointwise convolution becomes the model's first
        cross-channel mixer.
    :type gabor_stem_projection: bool
    :param use_laplacian_pyramid: If True replace each encoder downsample/skip
        junction with a bias-free ``LaplacianPyramidLevel`` split: the
        channel-preserving full-resolution high-frequency band becomes the skip
        connection and the half-resolution low-frequency band continues down the
        encoder. Defaults to False. Contributes zero trainable parameters.
    :type use_laplacian_pyramid: bool
    :param laplacian_kernel_size: Gaussian blur kernel size for the Laplacian pyramid
        split. Only used when ``use_laplacian_pyramid=True``. Defaults to ``(5, 5)``.
    :type laplacian_kernel_size: tuple of 2 ints
    :param high_freq_blocks: Number of ConvNeXt blocks applied to the Laplacian
        high-frequency skip band at each encoder level before it becomes the decoder
        skip. Ignored when ``use_laplacian_pyramid=False``, since the high band only
        exists under the pyramid split. Defaults to 0, which adds zero layers. Must
        be non-negative.
    :type high_freq_blocks: int
    :param bottleneck_attention_blocks: Number of bias-free LinearAttention blocks
        inserted at the bottleneck right after the channel-adjust and before the
        bottleneck ConvNeXt stack. Each block is a residual
        ``x + StochasticDepth(rate)(SpatialLinearAttention(x))`` with a local
        drop-path ramp starting at 0.0. The attention is degree-1 homogeneous
        (fixed ``'linear'`` type, ``use_bias=False``, ``feature_map='relu'``).
        Defaults to 0, which adds zero layers. Must be non-negative. When > 0, the
        bottleneck filter count must be divisible by ``bottleneck_attention_heads``.
    :type bottleneck_attention_blocks: int
    :param bottleneck_attention_heads: Number of attention heads per bottleneck
        attention block. Only used when ``bottleneck_attention_blocks > 0``.
        Defaults to 8. Must be >= 1 when attention blocks are enabled.
    :type bottleneck_attention_heads: int
    :param zero_pad_channels: If True replace every per-level channel-adjust 1x1
        convolution with a parameter-free channel match. Channel increases (encoder
        levels and the bottleneck) are done by zero-padding the channel axis.
        Channel decreases (the post-upsample decoder path) are done by slicing the
        upsampled branch to ``current_filters`` and adding the skip connection,
        since slicing the concatenation instead would discard the whole upsampled
        branch. The substitution is bias-free and homogeneous and removes all
        channel-adjust conv parameters. Defaults to False.
    :type zero_pad_channels: bool
    :param extra_zero_output_channels: If True, at decoder level 0 append
        ``output_channels`` zero-initialized feature channels before that level's
        ConvNeXt blocks (which are widened to ``initial_filters + output_channels``),
        and replace the final learned 1x1 output projection with a parameter-free
        slice that keeps the last ``output_channels`` channels. The residual blocks
        learn to write the output into the zero tail. Bias-free and homogeneous;
        default off. With ``include_top=False`` the widening still happens and no
        slice is built, so the two flags should not be combined.
    :type extra_zero_output_channels: bool
    :param final_projection_groups: Number of groups for the final 1x1
        ``final_output`` projection (``Conv2D(output_channels, 1, groups=...)``).
        Default 1 is a standard dense 1x1 conv. When > 1 the projection becomes a
        grouped conv: input feature channels and output channels are split into
        ``final_projection_groups`` groups and each output group is computed only
        from its own input group. Setting it to ``output_channels`` gives one group
        per output (e.g. colour) channel. The group count must divide both the
        projection's input channel count and ``output_channels`` (it raises
        ``ValueError`` otherwise). It is incompatible with
        ``extra_zero_output_channels``, which has no learned ``final_output`` conv
        to group, and with ``include_top=False``, which builds no projection.
    :type final_projection_groups: int
    :param downsample_pool_type: ``'max'``, ``'average'`` or ``'strided_conv'``.
        Downsample op for the non-Laplacian encoder junction. ``'max'`` (default) is
        MaxPooling2D, non-linear but positively homogeneous. ``'average'`` is
        AveragePooling2D, a linear operator that keeps the encoder path linear for
        the Miyasawa/Tweedie residual-as-score reading. ``'strided_conv'`` is a
        learned, channel-preserving ``Conv2D(kernel_size=2, strides=2)`` that
        threads ``use_bias``; with ``use_bias=False`` it is linear and degree-1
        homogeneous, so it is legal on the bias-free arm and is not guarded.
        Ignored when ``use_laplacian_pyramid=True``. The two pooling ops are
        weightless, so switching between them does not affect weight transfer;
        ``'strided_conv'`` adds parameters at every encoder junction.
    :type downsample_pool_type: str
    :param expose_bottleneck: If True expose the deepest-encoder bottleneck latent as
        an additional, trailing model output: ``[denoised, ...(supervision)...,
        bottleneck]``. A zero-parameter linear ``Activation('linear',
        name='bottleneck')`` tap is inserted after the bottleneck blocks. Defaults to
        False.
    :type expose_bottleneck: bool
    :param block_kernel_size: Depthwise kernel size inside every ConvNeXt block.
        Defaults to 7.
    :type block_kernel_size: int or tuple of 2 ints
    :param block_activation: Activation inside every ConvNeXt block's
        inverted-bottleneck MLP. Defaults to ``'gelu'``. Pass a
        ``keras.layers.LeakyReLU(negative_slope=0.1)`` instance for slope-0.1 leaky
        ReLU, since the ``'leaky_relu'`` string resolves to slope 0.2. A layer
        instance round-trips through ``.keras``, handled by
        ``ConvNext*Block.get_config``.
    :type block_activation: str or keras.layers.Layer
    :param block_normalization: The pre-activation normalization used inside every
        ConvNeXt block. One of:

        - ``'layernorm'`` (default): per-input ``LayerNormalization``
          (epsilon=1e-6, center=use_bias, scale=True). Per-input LayerNorm is
          scale-invariant (degree 0), not scale-homogeneous.
        - ``'batchnorm'``: the variance-only ``BiasFreeBatchNorm``. At inference
          (``training=False``) it divides by a frozen running_var constant, with no
          mean and no beta, which restores degree-1 homogeneity ``f(a*x) = a*f(x)``.
          It pairs best with a homogeneous activation such as LeakyReLU. That
          homogeneity is an inference-time property: during training the layer uses
          per-batch variance and is degree 0.

        Threaded to every encoder, bottleneck and decoder block. The stem
        normalization and the deep-supervision-head LayerNorm are outside this
        parameter.
    :type block_normalization: str
    :param stem_activation: Activation for the ``ConvUNextStem``; default ``'gelu'``.
        Only used when the standard stem is built, i.e. ``use_gabor_stem=False``.
    :type stem_activation: str or keras.layers.Layer
    :param drop_path_rate: Ceiling of the per-stack linear stochastic-depth ramps;
        see the schedule above, where no single block reaches the ceiling. Defaults
        to 0.1.
    :type drop_path_rate: float
    :param final_activation: Activation of the final output projection, and of every
        deep-supervision output. Defaults to ``'linear'``.
    :type final_activation: str or callable
    :param kernel_initializer: Initializer for the main-path structural convolutions
        (stem, channel adjusts, final projection, supervision heads). Defaults to
        ``'orthogonal'``, which is norm-preserving; ``'he_normal'`` compounds
        variance through the residual trunk and explodes a deep U-Net at init.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for those convolutions.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param depthwise_initializer: Applied to the depthwise conv kernel of every
        ConvNeXt block. Defaults to None, which reproduces the block's own
        ``TruncatedNormal(mean=0.0, stddev=0.02)``. For an orthonormal depthwise init
        pass keras ``Orthogonal(gain=1.0)``; a ``(K,K,C,1)`` depthwise kernel
        flattens to a single column, so orthonormal there means unit-norm. The repo
        ``OrthonormalInitializer`` / ``HeOrthonormalInitializer`` (2D only) and
        ``OrthogonalHypersphereInitializer`` (norm blow-up) are unsupported here.
    :type depthwise_initializer: str or keras.initializers.Initializer or None
    :param depthwise_regularizer: Applied to the depthwise conv kernel of every
        ConvNeXt block. Defaults to None, which reproduces the block's own behavior
        (a deepcopy of ``kernel_regularizer``).
    :type depthwise_regularizer: str or keras.regularizers.Regularizer or None
    :param dropout_rate: Float in ``[0, 1)``. Element-wise MLP dropout applied inside
        each ConvNeXt block's inverted bottleneck. This is not stochastic depth; see
        ``drop_path_rate``. Default ``0.0`` adds no ``Dropout`` sub-layer.
        ``spatial_dropout_rate`` stays ``0.0`` and is not exposed.
    :type dropout_rate: float
    :param enable_deep_supervision: Whether to add deep-supervision outputs. Defaults
        to False.
    :type enable_deep_supervision: bool
    :param supervision_norm_scale: Whether the deep-supervision head LayerNorm has a
        learnable scale (gamma). Defaults to True.
    :type supervision_norm_scale: bool
    :param supervision_norm_center: Whether the deep-supervision head LayerNorm has a
        learnable center (beta/bias). Defaults to False, which keeps the head free of
        an additive offset. ``True`` is rejected on the bias-free arm.
    :type supervision_norm_center: bool
    :param supervision_activation: Activation for the deep-supervision heads; default
        ``'gelu'``. Only used when ``enable_deep_supervision=True``.
    :type supervision_activation: str or keras.layers.Layer
    :param include_top: Whether to build the final ``output_channels`` projection.
        Defaults to True. With ``include_top=False`` the model's primary output is the
        full-resolution decoder feature map, exposed through a zero-parameter
        ``Activation('linear', name='decoder_features')`` tap, and the
        ``final_output`` projection is not constructed.

        ``keras.Model(inputs, outputs)`` keeps only the layers on a path to an
        output, so there is no weight-compatibility contract between the two
        settings: ``include_top=False`` yields a strictly smaller weight list, and
        ``set_weights`` between the two configurations raises. See decisions.md
        D-013.
    :type include_top: bool
    :param output_channels: Number of channels of the final projection and of every
        deep-supervision output. Defaults to ``None``, which means the input channel
        count (``input_shape[-1]``), the denoiser/autoencoder contract. Set it
        explicitly for a non-reconstruction head, e.g. ``1`` for a single-channel
        mask. Also sets the width of the zero tail appended by
        ``extra_zero_output_channels``. Inert when ``include_top=False``,
        ``enable_deep_supervision=False`` and ``extra_zero_output_channels=False``.
    :type output_channels: int or None
    :param model_name: Name of the returned model. Defaults to ``'convunext'``.
    :type model_name: str

    :return: A functional ``keras.Model``.

        - ``enable_deep_supervision=False``: a single output tensor.
        - ``enable_deep_supervision=True``: ``[final_output, supervision...]``,
          with the supervision outputs ordered from level 1 (highest resolution)
          down to level ``depth-1``.
        - ``expose_bottleneck=True``: the outputs list gains a trailing
          ``bottleneck`` output.

    :rtype: keras.Model
    :raises ValueError: If ``depth < 2``, ``initial_filters`` is non-positive,
        ``filter_multiplier < 1``, ``blocks_per_level`` is non-positive,
        ``high_freq_blocks`` or ``bottleneck_attention_blocks`` is negative,
        ``bottleneck_attention_heads < 1`` with attention enabled,
        ``convnext_version`` is not ``'v1'``/``'v2'``, ``downsample_pool_type`` is
        not ``'max'``/``'average'``/``'strided_conv'``, ``output_channels`` is not a
        positive integer or None, the bottleneck filter count is not divisible by
        ``bottleneck_attention_heads``, ``gabor_filters_per_channel`` is set without
        ``use_gabor_stem``, ``gabor_stem_projection=False`` with a stem width other
        than ``initial_filters``, ``final_projection_groups`` is misused, or a
        bias-free guardrail fires.
    :raises TypeError: If ``input_shape`` is not a tuple of length 3.

    Example::

        >>> # Standard (bias-carrying) ConvUNext with deep supervision
        >>> model = create_convunext(
        ...     input_shape=(256, 256, 3),
        ...     depth=4,
        ...     initial_filters=64,
        ...     enable_deep_supervision=True,
        ... )
        >>>
        >>> # Bias-free denoiser arm, flexible spatial dims, V1 blocks
        >>> denoiser = create_convunext(
        ...     input_shape=(None, None, 3),
        ...     use_bias=False,
        ...     convnext_version='v1',
        ... )
    """

    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise TypeError("input_shape must be a tuple of 3 integers (height, width, channels)")

    if depth < 2:
        raise ValueError(f"depth must be at least 2, got {depth}")

    if initial_filters <= 0:
        raise ValueError(f"initial_filters must be positive, got {initial_filters}")

    if filter_multiplier < 1:
        raise ValueError(f"filter_multiplier must be at least 1, got {filter_multiplier}")

    if blocks_per_level <= 0:
        raise ValueError(f"blocks_per_level must be positive, got {blocks_per_level}")

    if high_freq_blocks < 0:
        raise ValueError(f"high_freq_blocks must be non-negative, got {high_freq_blocks}")

    if bottleneck_attention_blocks < 0:
        raise ValueError(
            f"bottleneck_attention_blocks must be >= 0, got {bottleneck_attention_blocks}")

    if bottleneck_attention_blocks > 0 and bottleneck_attention_heads < 1:
        raise ValueError(
            f"bottleneck_attention_heads must be >= 1 when bottleneck_attention_blocks > 0, "
            f"got {bottleneck_attention_heads}")

    if convnext_version not in ['v1', 'v2']:
        raise ValueError(f"convnext_version must be 'v1' or 'v2', got {convnext_version}")

    if gabor_filters_per_channel is not None:
        if gabor_filters_per_channel < 1:
            raise ValueError(
                "gabor_filters_per_channel is a depthwise depth_multiplier and must be "
                f">= 1 when set, got {gabor_filters_per_channel}. Pass None (the "
                "default) for the cross-channel Conv2D Gabor stem."
            )
        if not use_gabor_stem:
            # Raise rather than ignore: a silently inert argument is this repo's
            # recorded defect class, so a contradictory intent is reported.
            raise ValueError(
                "gabor_filters_per_channel selects the DEPTHWISE Gabor stem, but "
                "use_gabor_stem=False builds no Gabor stem at all, so the argument "
                "would reach no layer. Pass use_gabor_stem=True, or drop "
                "gabor_filters_per_channel."
            )

    if downsample_pool_type not in ['max', 'average', 'strided_conv']:
        raise ValueError(
            "downsample_pool_type must be 'max', 'average' or 'strided_conv', got "
            f"{downsample_pool_type}"
        )

    if output_channels is None:
        output_channels = input_shape[-1]
    elif not isinstance(output_channels, int) or output_channels <= 0:
        raise ValueError(
            f"output_channels must be a positive integer or None, got {output_channels!r}"
        )

    if not include_top and final_projection_groups != 1:
        # The argument only parameterizes the final_output conv, which is not built
        # here; raise rather than let it sit inert.
        raise ValueError(
            "final_projection_groups is only meaningful with include_top=True (it "
            "groups the final_output projection, which include_top=False does not "
            f"build); got final_projection_groups={final_projection_groups}"
        )

    # DECISION plan-2026-08-14T092357-0e3d792d/D-012: guardrails fire only on the
    # bias-off arm; unconditional validation breaks bias-on configs. See decisions.md.
    if use_bias is False:
        _validate_bias_free_arguments(
            final_activation=final_activation,
            gabor_activation=gabor_activation,
            use_gabor_stem=use_gabor_stem,
            supervision_norm_center=supervision_norm_center,
            block_normalization=block_normalization,
        )

    ConvNextBlock = ConvNextV2Block if convnext_version == 'v2' else ConvNextV1Block

    inputs = keras.Input(shape=input_shape, name='input_images')

    if use_gabor_stem:
        if gabor_filters_per_channel is None:
            gabor = create_gabor_conv2d(
                filters=gabor_filters,
                kernel_size=gabor_kernel_size,
                activation=gabor_activation,
                strides=1,
                padding='same',
                # Bias-free whatever `use_bias` says: the denoiser arm needs
                # D(a*x) == a*D(x), which a bias breaks.
                use_bias=False,
                # The paper's warm start: Gabor-initialized, then refined.
                trainable=True,
                name='gabor_stem',
            )(inputs)
            stem_output_channels = gabor_filters
            stem_description = (
                f"cross-channel Conv2D, filters={gabor_filters}"
            )
        else:
            gabor = create_gabor_depthwise_conv2d(
                filters_per_channel=gabor_filters_per_channel,
                kernel_size=gabor_kernel_size,
                activation=gabor_activation,
                strides=1,
                padding='same',
                # Same bias-freedom as the Conv2D arm, for the same reason.
                use_bias=False,
                # Overrides the helper's own trainable=False default; freezing is
                # `--freeze-gabor-stem` alone. See D-001 above.
                trainable=True,
                name='gabor_stem',
            )(inputs)
            # A depthwise conv does not sum across input channels, so the widths
            # multiply here; downstream checks read stem_output_channels.
            stem_output_channels = input_shape[-1] * gabor_filters_per_channel
            stem_description = (
                f"per-channel DepthwiseConv2D, depth_multiplier="
                f"{gabor_filters_per_channel} x input_channels({input_shape[-1]})"
            )
        if gabor_stem_projection:
            stem_input = keras.layers.Conv2D(
                filters=initial_filters,
                kernel_size=1,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='gabor_stem_projection',
            )(gabor)
            logger.info(
                f"Trainable Gabor stem enabled ({stem_description}), "
                f"kernel_size={gabor_kernel_size} -> {stem_output_channels} channels "
                f"-> 1x1 projection to {initial_filters}"
            )
        else:
            # Without the projection there is no bias-free, parameter-free way to
            # reach initial_filters, so a width mismatch fails instead of padding.
            if stem_output_channels != initial_filters:
                if gabor_filters_per_channel is None:
                    detail = (
                        f"gabor_filters({gabor_filters}) != "
                        f"initial_filters({initial_filters}). NOTE: gabor_filters is the "
                        "stem's OUTPUT CHANNEL COUNT (a Conv2D `filters`), NOT a "
                        f"per-channel multiplier, so the old input_channels"
                        f"({input_shape[-1]}) * gabor_filters rule no longer applies to "
                        "this arm. Set gabor_filters == initial_filters, keep "
                        "gabor_stem_projection=True, or select the depthwise stem with "
                        "gabor_filters_per_channel."
                    )
                else:
                    detail = (
                        f"the depthwise stem emits input_channels({input_shape[-1]}) * "
                        f"gabor_filters_per_channel({gabor_filters_per_channel}) = "
                        f"{stem_output_channels} channels, != "
                        f"initial_filters({initial_filters}). Set "
                        "input_channels * gabor_filters_per_channel == initial_filters, "
                        "or keep gabor_stem_projection=True."
                    )
                raise ValueError(
                    "gabor_stem_projection=False requires the Gabor stem to emit exactly "
                    f"initial_filters channels, but {detail}"
                )
            stem_input = gabor
            logger.info(
                f"Trainable Gabor stem enabled (NO projection): {stem_description}, "
                f"kernel_size={gabor_kernel_size} -> {stem_output_channels} channels "
                f"feed the encoder directly (== initial_filters)"
            )
    else:
        stem_input = inputs

    # One entry per encoder level plus the bottleneck.
    filter_sizes = [int(round(initial_filters * (filter_multiplier ** i))) for i in range(depth + 1)]

    if use_laplacian_pyramid:
        logger.info(
            f"Laplacian pyramid downsample enabled: kernel_size={laplacian_kernel_size}, "
            f"split levels={depth} (high-band skips, low-band downsample; bias-free)"
        )
    else:
        _downsample_descriptions = {
            'average': 'AveragePooling2D — linear, Miyasawa-clean',
            'max': 'MaxPooling2D — non-linear but positively homogeneous',
            'strided_conv': (
                'Conv2D(k=2, s=2) — LEARNED, channel-preserving, '
                f"use_bias={use_bias}"
            ),
        }
        logger.info(
            f"Encoder downsample: {downsample_pool_type} "
            f"({_downsample_descriptions[downsample_pool_type]})"
        )

    if zero_pad_channels:
        logger.info(
            "Zero-pad channel matching ENABLED: per-level channel-adjust convs replaced by "
            "parameter-free pad/slice (encoder+bottleneck zero-pad; decoder slice-upsampled+add-skip; bias-free)"
        )

    skip_connections: List[keras.layers.Layer] = []
    deep_supervision_outputs: List[keras.layers.Layer] = []

    x = stem_input
    logger.info(f"Building ConvUNext encoder path with {depth} levels using ConvNeXt {convnext_version.upper()}")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"Encoder level {level}: {current_filters} filters")

        # The Gabor stem and its projection already did the initial feature
        # extraction and set the width to initial_filters, so this branch is only
        # for the level-0 case without it.
        if level == 0 and not use_gabor_stem:
            x = ConvUNextStem(
                filters=current_filters,
                kernel_size=stem_kernel_size,
                activation=stem_activation,
                use_bias=use_bias,
                stem_normalization=stem_normalization,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'encoder_level_{level}_stem'
            )(x)
        else:
            # Widths must match current_filters for the residual adds below.
            if x.shape[-1] != current_filters:
                if zero_pad_channels:
                    x = MatchChannels(current_filters, name=f'encoder_level_{level}_match_channels')(x)
                else:
                    x = keras.layers.Conv2D(
                        filters=current_filters,
                        kernel_size=1,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        name=f'encoder_level_{level}_channel_adjust'
                    )(x)

        for block_idx in range(blocks_per_level):
            # Ramp rises linearly with absolute block position across the encoder.
            current_drop_path = drop_path_rate * (level * blocks_per_level + block_idx) / (depth * blocks_per_level)
            x = _apply_residual_convnext_block(
                x, ConvNextBlock, current_filters, block_kernel_size,
                current_drop_path, kernel_regularizer,
                name=f'encoder_level_{level}_convnext_{convnext_version}_block_{block_idx}',
                activation=block_activation,
                depthwise_initializer=depthwise_initializer,
                depthwise_regularizer=depthwise_regularizer,
                dropout_rate=dropout_rate,
                normalization_type=block_normalization,
                use_bias=use_bias,
            )

        # The last encoder level's junction keeps the bottleneck name. The returned
        # order is (skip, downsampled) on both paths; both are rank 4, so a swap
        # would pass every shape check.
        junction_name = (
            f'encoder_downsample_{level}' if level < depth - 1 else 'bottleneck_downsample'
        )
        skip, x = DownsampleAndSkip(
            use_laplacian_pyramid=use_laplacian_pyramid,
            laplacian_kernel_size=laplacian_kernel_size,
            pool_type=downsample_pool_type,
            # Only the 'strided_conv' branch reads these three.
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=junction_name,
        )(x)

        # DECISION plan_2026-07-10_be906be8/D-002: keep both gates; dropping either
        # inserts layers into the raw-skip path. See decisions.md.
        if high_freq_blocks > 0 and use_laplacian_pyramid:
            for hf_idx in range(high_freq_blocks):
                # Local ramp, restarting at 0.0 for each level's high-band stack.
                current_drop_path = drop_path_rate * hf_idx / high_freq_blocks
                skip = _apply_residual_convnext_block(
                    skip, ConvNextBlock, current_filters, block_kernel_size,
                    current_drop_path,
                    kernel_regularizer,
                    name=f'skip_highfreq_block_{level}_{hf_idx}',
                    activation=block_activation,
                    depthwise_initializer=depthwise_initializer,
                    depthwise_regularizer=depthwise_regularizer,
                    dropout_rate=dropout_rate,
                    normalization_type=block_normalization,
                    use_bias=use_bias,
                )

        skip_connections.append(skip)

    bottleneck_filters = filter_sizes[depth]
    logger.info(f"Building ConvUNext bottleneck with {bottleneck_filters} filters")

    if x.shape[-1] != bottleneck_filters:
        if zero_pad_channels:
            x = MatchChannels(bottleneck_filters, name='bottleneck_match_channels')(x)
        else:
            x = keras.layers.Conv2D(
                filters=bottleneck_filters,
                kernel_size=1,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='bottleneck_channel_adjust'
            )(x)

    # Attention sits after the channel adjust and before the ConvNeXt stack.
    if bottleneck_attention_blocks > 0:
        if bottleneck_filters % bottleneck_attention_heads != 0:
            raise ValueError(
                f"bottleneck_filters ({bottleneck_filters}) must be divisible by "
                f"bottleneck_attention_heads ({bottleneck_attention_heads})")
        for attn_idx in range(bottleneck_attention_blocks):
            current_drop_path = drop_path_rate * attn_idx / bottleneck_attention_blocks
            residual = x
            y = SpatialLinearAttention(
                bottleneck_filters, bottleneck_attention_heads,
                name=f'bottleneck_attention_block_{attn_idx}')(x)
            if current_drop_path > 0:
                y = StochasticDepth(
                    current_drop_path, name=f'bottleneck_attention_sd_{attn_idx}')(y)
            x = keras.layers.Add(name=f'bottleneck_attention_add_{attn_idx}')([residual, y])

    for block_idx in range(blocks_per_level):
        # Local ramp, restarting at 0.0 in the bottleneck stack.
        current_drop_path = drop_path_rate * block_idx / blocks_per_level
        x = _apply_residual_convnext_block(
            x, ConvNextBlock, bottleneck_filters, block_kernel_size,
            current_drop_path, kernel_regularizer,
            name=f'bottleneck_convnext_{convnext_version}_block_{block_idx}',
            activation=block_activation,
            depthwise_initializer=depthwise_initializer,
            depthwise_regularizer=depthwise_regularizer,
            dropout_rate=dropout_rate,
            normalization_type=block_normalization,
            use_bias=use_bias,
        )

    # The tap sits on the main path, so the named layer survives a single-output
    # save and the latent can be extracted afterwards.
    if expose_bottleneck:
        x = keras.layers.Activation('linear', name='bottleneck')(x)
        bottleneck_output = x

    logger.info(f"Building ConvUNext decoder path with {depth} levels")

    for level in range(depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        logger.info(f"Decoder level {level}: {current_filters} filters")

        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation='bilinear',
            name=f'decoder_upsample_{level}'
        )(x)

        skip = skip_connections[level]

        # Odd input sizes make the upsampled dims miss the skip's.
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            target_height, target_width = skip.shape[1], skip.shape[2]
            x = keras.layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation='bilinear',
                name=f'decoder_resize_{level}'
            )(x)

        if zero_pad_channels:
            x = keras.layers.Add(name=f'decoder_level_{level}_match_add')(
                [skip, MatchChannels(current_filters, name=f'decoder_level_{level}_match_channels')(x)]
            )
        else:
            x = keras.layers.Concatenate(
                axis=-1,
                name=f'decoder_concat_{level}'
            )([skip, x])

            if x.shape[-1] != current_filters:
                x = keras.layers.Conv2D(
                    filters=current_filters,
                    kernel_size=1,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name=f'decoder_level_{level}_channel_adjust'
                )(x)

        # The zero tail the final slice will read is grown here, at level 0 only.
        block_filters = current_filters
        if extra_zero_output_channels and level == 0:
            block_filters = current_filters + output_channels
            x = MatchChannels(block_filters, name='extra_zero_output_pad')(x)

        for block_idx in range(blocks_per_level):
            # The first block of each decoder level carries no stochastic depth; the
            # rest follow the encoder's absolute-position ramp.
            if block_idx == 0:
                current_drop_path = 0.0
            else:
                current_drop_path = drop_path_rate * (level * blocks_per_level + block_idx) / (depth * blocks_per_level)
            x = _apply_residual_convnext_block(
                x, ConvNextBlock, block_filters, block_kernel_size,
                current_drop_path, kernel_regularizer,
                name=f'decoder_level_{level}_convnext_{convnext_version}_block_{block_idx}',
                activation=block_activation,
                depthwise_initializer=depthwise_initializer,
                depthwise_regularizer=depthwise_regularizer,
                dropout_rate=dropout_rate,
                normalization_type=block_normalization,
                use_bias=use_bias,
            )

        if enable_deep_supervision and level > 0:
            supervision_branch = keras.layers.Conv2D(
                filters=current_filters // 2,
                kernel_size=1,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'supervision_intermediate_level_{level}'
            )(x)

            # `center` reads supervision_norm_center, not use_bias: it is a separate,
            # separately guarded knob (plan invariant I-6).
            supervision_branch = keras.layers.LayerNormalization(
                center=supervision_norm_center,
                scale=supervision_norm_scale,
                name=f'supervision_layernorm_level_{level}'
            )(supervision_branch)

            supervision_branch = _make_supervision_activation(
                supervision_activation, f'supervision_activation_level_{level}'
            )(supervision_branch)

            supervision_output = keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=1,
                activation=final_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'supervision_output_level_{level}'
            )(supervision_branch)

            deep_supervision_outputs.append(supervision_output)

            logger.info(f"Added deep supervision output at level {level} "
                       f"with shape: {supervision_output.shape}")

    if extra_zero_output_channels and final_projection_groups != 1:
        raise ValueError(
            "final_projection_groups>1 is incompatible with extra_zero_output_channels: the "
            "latter drops the learned final_output Conv2D in favor of a parameter-free tail "
            "slice, so there is no projection to group. Use one or the other."
        )

    if not include_top:
        final_output = keras.layers.Activation(
            'linear', name='decoder_features'
        )(x)
        logger.info(
            f"include_top=False: no final projection; primary output is the decoder "
            f"feature map with {final_output.shape[-1]} channels"
        )
    elif extra_zero_output_channels:
        # DECISION plan_2026-06-26_0ec1a304/D-001: keep the zero-grown tail as the
        # output; do not restore the learned 1x1 projection. See decisions.md.
        final_output = MatchChannels(
            output_channels, slice_side='tail', name='final_output_tail_slice'
        )(x)
        if final_activation is not None and final_activation != 'linear':
            final_output = keras.layers.Activation(
                final_activation, name='final_output_activation'
            )(final_output)
    else:
        if final_projection_groups < 1:
            raise ValueError(
                f"final_projection_groups must be >= 1, got {final_projection_groups}"
            )
        in_ch = x.shape[-1]
        if final_projection_groups > 1 and (
            in_ch % final_projection_groups != 0
            or output_channels % final_projection_groups != 0
        ):
            raise ValueError(
                f"final_projection_groups={final_projection_groups} must divide BOTH the "
                f"final-projection input channels ({in_ch}, == initial_filters) and "
                f"output_channels ({output_channels}). Pick a group count dividing both, or "
                "use 1 (ungrouped)."
            )
        final_output = keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=1,
            groups=final_projection_groups,
            activation=final_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='final_output'
        )(x)

    if enable_deep_supervision and deep_supervision_outputs:
        # Reversing puts the highest-resolution supervision output first.
        ordered_supervision_outputs = list(reversed(deep_supervision_outputs))
        all_outputs = [final_output] + ordered_supervision_outputs
        if expose_bottleneck:
            all_outputs = all_outputs + [bottleneck_output]

        logger.info(f"Created ConvUNext deep supervision model with {len(all_outputs)} outputs:")
        logger.info(f"  - Final output (index 0): {final_output.shape}")
        for i, sup_output in enumerate(ordered_supervision_outputs):
            level = i + 1
            logger.info(f"  - Supervision output {i + 1} (index {i + 1}, level {level}): {sup_output.shape}")

        model = keras.Model(
            inputs=inputs,
            outputs=all_outputs,
            name=model_name
        )

    else:
        if expose_bottleneck:
            model = keras.Model(
                inputs=inputs,
                outputs=[final_output, bottleneck_output],
                name=model_name
            )
        else:
            model = keras.Model(
                inputs=inputs,
                outputs=final_output,
                name=model_name
            )

        logger.info(f"Created single-output ConvUNext model")

    logger.info(f"Created ConvUNext model '{model_name}' with depth {depth}")
    logger.info(f"ConvNeXt version: {convnext_version.upper()}")
    logger.info(f"Filter progression: {filter_sizes}")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")
    logger.info(f"Deep supervision enabled: {enable_deep_supervision}")
    logger.info(f"Drop path rate: {drop_path_rate}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model


# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_convunext_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        enable_deep_supervision: bool = False,
        **kwargs: Any
) -> keras.Model:
    """Build a ConvUNext model from a named variant configuration.

    The single expansion path for ``CONVUNEXT_CONFIGS``, shared by both arms:
    ``bfconvunext.create_convunext_variant`` forwards here with ``use_bias=False``.

    Variants:

    .. code-block:: text

        variant  depth  initial_filters  blocks_per_level  drop_path_rate
        tiny         3               32                 2             0.0
        small        3               48                 2             0.1
        base         4               64                 3             0.1
        large        4               96                 4             0.2
        xlarge       5              128                 5             0.3

    Every variant uses ``convnext_version='v2'``. The table sets no
    ``block_normalization``, so ``create_convunext``'s default applies unless a
    caller passes one.

    :param variant: One of ``'tiny'``, ``'small'``, ``'base'``, ``'large'``,
        ``'xlarge'``.
    :type variant: str
    :param input_shape: Shape of input images ``(height, width, channels)``.
    :type input_shape: tuple of 3 ints
    :param enable_deep_supervision: Whether to enable deep-supervision outputs.
        Defaults to False, matching ``create_convunext``. It also decides the
        ``_ds`` suffix on the generated model name.
    :type enable_deep_supervision: bool
    :param kwargs: Additional keyword arguments forwarded to ``create_convunext``,
        overriding the variant defaults (including ``use_bias`` and
        ``model_name``).
    :return: A functional ``keras.Model``.
    :rtype: keras.Model
    :raises ValueError: If ``variant`` is not a key of ``CONVUNEXT_CONFIGS``.

    Example::

        >>> model = create_convunext_variant('base', (256, 256, 3),
        ...                                  enable_deep_supervision=True)
    """
    if variant not in CONVUNEXT_CONFIGS:
        available_variants = list(CONVUNEXT_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = CONVUNEXT_CONFIGS[variant].copy()
    description = config.pop('description')

    config.update(kwargs)

    # A model_name passed through kwargs is already in config and wins.
    if 'model_name' not in config:
        ds_suffix = '_ds' if enable_deep_supervision else ''
        convnext_version = config.get('convnext_version', 'v2')
        config['model_name'] = f'convunext_{variant}_{convnext_version}{ds_suffix}'

    config['enable_deep_supervision'] = enable_deep_supervision

    logger.info(f"Creating ConvUNext variant '{variant}': {description}")
    logger.info(f"ConvNeXt version: {config.get('convnext_version', 'v2').upper()}")
    logger.info(f"Deep supervision: {'enabled' if enable_deep_supervision else 'disabled'}")

    return create_convunext(
        input_shape=input_shape,
        **config
    )

# ---------------------------------------------------------------------