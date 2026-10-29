"""
FastViT / MobileCLIP2 (MCi) backbone components.

Channels-last Keras 3 transcriptions of the blocks timm's ``fastvit.py`` builds the
FastViT **MCi** image tower from — the image encoder of Apple's MobileCLIP /
MobileCLIP2. They are specific to this architecture, so they live beside the model
(:class:`~dl_techniques.models.vision.fastvit.model.FastVitImageEncoder`) rather than
in ``layers/``. Their general-purpose parts come from ``layers/``:

* :class:`~dl_techniques.layers.conv_blocks.mobile_one_block.MobileOneBlock` — the
  RepMixer ``norm`` / ``mixer`` arms and the PatchEmbed pointwise projection.
* :class:`~dl_techniques.layers.conv_blocks.reparam_large_kernel_conv.ReparamLargeKernelConv`
  — the PatchEmbed large-kernel downsample.
* :class:`~dl_techniques.layers.embedding.rep_conditional_pos_enc.RepConditionalPosEnc`
  — the conditional positional encoding of the attention stages.
* :class:`~dl_techniques.layers.ffn.mlp.MLPBlock` — the position-wise half of the
  ConvMlp.
* :class:`~dl_techniques.layers.regularization.layer_scale.LayerScale`,
  :class:`~dl_techniques.layers.regularization.stochastic_depth.StochasticDepth`, and
  the attention and normalization factories.

.. code-block:: text

    FastVitStage
      ├─ (optional) FastVitPatchEmbed       ReparamLargeKernelConv -> MobileOneBlock(k=1)
      ├─ (optional) RepConditionalPosEnc    depthwise 7x7 + skip
      └─ depth x one of
           FastVitRepMixerBlock             FastVitRepMixer -> + LayerScale(FastVitConvMlp)
           FastVitAttentionBlock            + LayerScale(MHSA(norm(x))) -> + LayerScale(FastVitConvMlp)

Names whose reference spelling is generic (``ConvMlp``, ``PatchEmbed``, ``Stage``,
``AttentionBlock``, ``RepMixer``, ``RepMixerBlock``) carry a ``FastVit`` prefix: the
legacy ``Custom>ClassName`` alias namespace is keyed by bare class name, so a generic
name would collide with an unrelated class.

.. note::
   ``FastVitRepMixerBlock`` is NOT
   :class:`~dl_techniques.layers.conv_blocks.repmixer_block.RepMixerBlock`. The two
   are different architectures that happen to share a name; the ``conv_blocks`` one
   is consumed by ``models/vision_language/fastvlm/`` and is not interchangeable.

**No structural reparameterization — deliberately.** Every block here implements
the train-time multi-branch form only. The fused form is mathematically identical,
the reference evaluates with ``inference_mode=False`` everywhere, and no pretrained
weights are ported that a fusion path could be validated against. If you need the
fused form, write it as an explicit, separately tested conversion pass over a
trained model; do not change these blocks' ``call()`` paths.

**Reference-fidelity constants.** Two facts below are properties of the timm /
MobileCLIP2 reference, not of any one block, and BOTH are silent when wrong: a
normalization epsilon and a padding convention are invisible to every shape
assertion. They were each shipped wrong once (an undisclosed 100x epsilon divergence
in 86 of 114 BatchNormalizations, and a one-pixel sampling shift in every strided
convolution), so they are defined ONCE, here, and every consumer imports them —
including ``models/vision_language/mobile_clip/``. A second literal ``1e-5`` or
``'reference'`` at a FastViT construction site is a defect.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2024. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. (https://arxiv.org/abs/2311.17049)
    - Vasu et al., 2022. MobileOne: An Improved One millisecond Mobile Backbone.
      (https://arxiv.org/abs/2206.04040)
    - Touvron et al., 2021. Going Deeper with Image Transformers (LayerScale).
      (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from typing import Optional, Union, Tuple, List, Sequence, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.activations.exact_activations import exact_name
from dl_techniques.layers.activations.gelu_tanh import resolve_activation
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.conv_blocks.mobile_one_block import MobileOneBlock
from dl_techniques.layers.conv_blocks.reparam_large_kernel_conv import ReparamLargeKernelConv
from dl_techniques.layers.embedding.rep_conditional_pos_enc import RepConditionalPosEnc
from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.regularization.layer_scale import LayerScale
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# reference constants
# ---------------------------------------------------------------------

#: Variance epsilon for every normalization layer in the port. PyTorch's
#: ``BatchNorm2d`` and ``LayerNorm`` both default to ``1e-5``, while Keras'
#: ``BatchNormalization`` / ``LayerNormalization`` default to ``1e-3`` and this
#: repo's ``create_normalization_layer`` ``setdefault``s ``1e-6``. Every one of
#: those differs from the reference, so the value is passed EXPLICITLY at every
#: construction site.
REFERENCE_NORM_EPSILON = 1e-5

#: ``padding_mode`` passed to :class:`MobileOneBlock` and
#: :class:`ReparamLargeKernelConv`. PyTorch pads SYMMETRICALLY by
#: ``kernel_size // 2``; Keras' ``padding='same'`` pads asymmetrically, which at
#: stride > 1 makes the sampled grid depend on the kernel size. See
#: :func:`~dl_techniques.layers.conv_blocks.mobile_one_block.resolve_conv_padding`.
REFERENCE_PADDING_MODE = 'reference'

#: LayerScale initialization the reference uses on every FastViT residual branch.
_REFERENCE_LAYER_SCALE_INIT = 1e-5

#: timm's ``_init_weights`` uses ``trunc_normal_(std=.02)`` for every ConvMlp
#: kernel. Stored as a STDDEV, not as a built Initializer — see FastVitConvMlp.
_REFERENCE_KERNEL_INIT_STDDEV = 0.02

#: Reference per-head width. ``num_heads`` is derived as ``dim // head_dim``,
#: exactly as timm's ``Attention`` does.
_REFERENCE_HEAD_DIM = 32

#: Kernel size of the PatchEmbed's trailing pointwise MobileOne block (1x1).
_POINTWISE_KERNEL_SIZE = 1

#: ``group_size`` passed to the PatchEmbed's large-kernel conv. In timm's
#: ``num_groups`` semantics this means ``groups = in_channels``, i.e. DEPTHWISE.
_LKC_GROUP_SIZE = 1

#: Small parallel kernel of the PatchEmbed's large-kernel conv (3x3).
_LKC_SMALL_KERNEL = 3

#: The two token mixers a stage may be built from. The reference chooses one per
#: stage and never mixes them within a stage.
_TOKEN_MIXERS = ('repmixer', 'attention')

# ---------------------------------------------------------------------


def _create_layer_scale(
        layer_scale_init_value: Optional[float],
        name: str,
) -> Optional[LayerScale]:
    """Build a FastViT ``LayerScale2d`` equivalent, or ``None`` when disabled.

    Shared by every block in this module so that the ``constraint=None``
    requirement is stated in exactly one place.

    :param layer_scale_init_value: Constant value the per-channel gamma is
        initialized to, or ``None`` to disable LayerScale entirely (the branch is
        then admitted unscaled, which is the reference's ``layer_scale_init_value
        is None`` path).
    :type layer_scale_init_value: Optional[float]
    :param name: Sub-layer name.
    :type name: str
    :return: A per-channel :class:`LayerScale` with ``constraint=None``,
        or ``None`` when ``layer_scale_init_value is None``.
    :rtype: Optional[LayerScale]

    .. warning::
       ``constraint=None`` is REQUIRED, not cosmetic. The layer's own default
       is ``constraint='non_neg'``, which clamps gamma at zero; a LayerScale
       gamma must be free to take a negative value.
    """
    if layer_scale_init_value is None:
        return None
    return LayerScale(
        multiplier_type='CHANNEL',
        initializer=keras.initializers.Constant(layer_scale_init_value),
        constraint=None,
        name=name,
    )


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.fastvit.components")
class FastVitConvMlp(keras.layers.Layer):
    """FastViT convolutional MLP: depthwise ``k x k`` + BN, then a position-wise MLP.

    Channels-last transcription of timm's ``ConvMlp`` as instantiated by FastViT,
    the channel-mixing half of both :class:`FastVitRepMixerBlock` and
    :class:`FastVitAttentionBlock`. A depthwise convolution followed by
    BatchNormalization and **no** activation adds local spatial context at
    negligible cost; channel mixing is left entirely to the position-wise MLP that
    follows.

    The position-wise half is the library's
    :class:`~dl_techniques.layers.ffn.mlp.MLPBlock` (``fc1 -> act -> drop -> fc2``,
    both biased). A ``Dense`` over the channel axis of a channels-last tensor is
    exactly the reference's ``1 x 1`` ``Conv2d``. ``MLPBlock`` has no dropout after
    ``fc2`` and the reference does, so this layer adds that one ``Dropout``.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │           Input [B, H, W, dim]               │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │  DepthwiseConv2D k×k, 'same', use_bias=False │
        │  BatchNormalization (eps 1e-5)  — NO act     │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │  ffn: MLPBlock                               │
        │   Dense → hidden_dim (biased) → act → drop   │
        │   Dense → out_dim    (biased)                │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Dropout                                     │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │          Output [B, H, W, out_dim]           │
        └──────────────────────────────────────────────┘

    .. note::
       The reference builds the first convolution as ``Conv2d(in_chs, out_chs,
       k, groups=in_chs)``, i.e. a grouped convolution that is depthwise *only*
       when ``out_chs == in_chs``. Every FastViT/MCi call site satisfies that,
       and channels-last Keras expresses it exactly as a ``DepthwiseConv2D``
       with ``depth_multiplier=1`` (kernel shape ``(k, k, C, 1)``). Rather than
       silently building a different graph for the unreachable ``out_dim != dim``
       case, this layer raises :class:`ValueError`.

    :param dim: Number of input channels. Must be positive.
    :type dim: int
    :param hidden_dim: Width of the inverted bottleneck. Defaults to ``dim``
        when ``None``. Must be positive when given.
    :type hidden_dim: Optional[int]
    :param out_dim: Number of output channels. Defaults to ``dim`` when ``None``.
        Must equal ``dim`` (see the note above).
    :type out_dim: Optional[int]
    :param kernel_size: Spatial size of the leading depthwise convolution. Must
        be a positive odd integer (an even kernel under ``padding='same'`` would
        shift the feature map). Defaults to 7.
    :type kernel_size: int
    :param activation: Activation applied after the expanding projection.
        Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param dropout_rate: Dropout rate applied after the activation and again
        after the projecting layer. Must be in ``[0, 1)``. Defaults to 0.0.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for every kernel. ``None`` (the
        default) means a FRESH ``TruncatedNormal(stddev=0.02)`` per instance,
        matching timm's ``_init_weights``.
    :type kernel_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param bias_initializer: Initializer for the MLP biases. Defaults to
        ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for the MLP biases.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim``, ``hidden_dim`` or ``kernel_size`` are not
        positive, if ``kernel_size`` is even, if ``dropout_rate`` is outside
        ``[0, 1)``, or if ``out_dim`` differs from ``dim``.

    Example:
        >>> import numpy as np
        >>> layer = FastVitConvMlp(dim=64, hidden_dim=256)
        >>> y = layer(np.zeros((2, 16, 16, 64), dtype='float32'), training=False)
        >>> y.shape
        (2, 16, 16, 64)
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            kernel_size: int = 7,
            activation: Union[str, callable] = 'gelu',
            dropout_rate: float = 0.0,
            kernel_initializer: Optional[
                Union[str, keras.initializers.Initializer]] = None,
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim is not None and out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd (an even kernel with padding='same' "
                f"shifts the feature map), got {kernel_size}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        resolved_out_dim = dim if out_dim is None else out_dim
        if resolved_out_dim != dim:
            raise ValueError(
                f"out_dim must equal dim: the reference only ever uses the "
                f"out_dim == dim form of ConvMlp, whose leading grouped "
                f"convolution is exactly a channels-last DepthwiseConv2D. "
                f"Got dim={dim}, out_dim={out_dim}."
            )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        self.out_dim = resolved_out_dim
        self.kernel_size = kernel_size
        self.activation = resolve_activation(activation)  # D-277: float64-exact, bit-identical to stock at float32
        self.dropout_rate = dropout_rate
        # CONSTRUCTED PER INSTANCE, never as a default argument. A default
        # argument is evaluated ONCE at import time, so every FastVitConvMlp in
        # the process would share ONE Initializer object — harmless while
        # `seed=None`, silently correlating the moment a seed is added.
        self.kernel_initializer = keras.initializers.get(
            keras.initializers.TruncatedNormal(stddev=_REFERENCE_KERNEL_INIT_STDDEV)
            if kernel_initializer is None else kernel_initializer
        )
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            use_bias=False,
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name='dw_conv'
        )
        self.norm = create_normalization_layer(
            'batch_norm',
            epsilon=REFERENCE_NORM_EPSILON,
            name='norm'
        )
        # MLPBlock clones the kernel and bias initializers per projection, so
        # fc1 and fc2 never share one Initializer object.
        self.ffn = MLPBlock(
            hidden_dim=self.hidden_dim,
            output_dim=self.out_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='ffn'
        )
        # The reference's second dropout, after fc2. MLPBlock has none.
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='dropout')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is
            not ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitConvMlp expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        self.dw_conv.build(input_shape)
        dw_shape = self.dw_conv.compute_output_shape(input_shape)

        self.norm.build(dw_shape)

        self.ffn.build(dw_shape)
        ffn_shape = self.ffn.compute_output_shape(dw_shape)

        self.dropout.build(ffn_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the convolutional MLP.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour when ``dropout_rate > 0``.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, out_dim)``.
        """
        x = self.dw_conv(inputs)
        x = self.norm(x, training=training)
        x = self.ffn(x, training=training)
        x = self.dropout(x, training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, spatially identical to the input.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.out_dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'kernel_size': self.kernel_size,
            'activation': exact_name(self.activation) or keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitConvMlp":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitConvMlp` instance.
        :rtype: FastVitConvMlp
        """
        config = dict(config)
        # resolve, not keras.activations.deserialize: "gelu" must come back as exact_gelu (D-277)
        config['activation'] = resolve_activation(config['activation'])
        config['kernel_initializer'] = keras.initializers.deserialize(
            config['kernel_initializer'])
        config['bias_initializer'] = keras.initializers.deserialize(
            config['bias_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(
            config['kernel_regularizer'])
        config['bias_regularizer'] = keras.regularizers.deserialize(
            config['bias_regularizer'])
        return cls(**config)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.fastvit.components")
class FastVitRepMixer(keras.layers.Layer):
    """FastViT RepMixer token mixer: ``x + gamma * (mixer(x) - norm(x))``.

    Channels-last transcription of timm's ``RepMixer``. Both ``norm`` and
    ``mixer`` are :class:`MobileOneBlock` instances at ``stride=1`` with
    ``out_channels == dim``; ``norm`` is the degenerate single-BatchNormalization
    form (no conv branches, no scale branch), ``mixer`` additionally carries one
    depthwise ``k x k`` Conv-BN branch. Neither applies an activation.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │            Input [B, H, W, dim]              │
        └───────┬──────────────────────────┬───────────┘
                │                          │
                │              ┌───────────┴───────────┐
                │              │                       │
                │              ▼                       ▼
                │   ┌──────────────────────┐ ┌──────────────────────┐
                │   │ mixer: MobileOneBlock│ │ norm: MobileOneBlock │
                │   │  depthwise k×k + BN  │ │  identity BN ONLY    │
                │   │  + 1×1 scale + id BN │ │  (0 conv branches,   │
                │   │  use_act=False       │ │   no scale branch)   │
                │   └───────────┬──────────┘ └──────────┬───────────┘
                │               │                       │
                │               └────────── − ──────────┘
                │                           │
                │                           ▼
                │              ┌──────────────────────────┐
                │              │ LayerScale (per-channel) │
                │              │  gamma, constraint=None  │
                │              └────────────┬─────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │            Output [B, H, W, dim]             │
        └──────────────────────────────────────────────┘

    .. note::
       With ``layer_scale_init_value=0.0`` the layer is the EXACT identity at
       ``training=False``, and with a nonzero value it is not. Both halves matter:
       an identity-only assertion is also satisfied by a completely dead branch.

    :param dim: Number of channels. The layer preserves it. Must be positive.
    :type dim: int
    :param kernel_size: Spatial size of the mixer's depthwise convolution. Must be
        a positive odd integer (an even kernel under ``padding='same'`` would shift
        the feature map). Defaults to 3.
    :type kernel_size: int
    :param layer_scale_init_value: Constant initialization for the per-channel
        LayerScale gamma, or ``None`` to omit LayerScale altogether. Defaults to
        ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param kernel_initializer: Initializer for the MobileOne convolution kernels.
        Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the MobileOne convolution
        kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim`` or ``kernel_size`` are not positive, if
        ``kernel_size`` is even, or if ``layer_scale_init_value`` is not a real
        number or ``None``.

    Example:
        >>> import numpy as np
        >>> layer = FastVitRepMixer(dim=32)
        >>> y = layer(np.zeros((2, 8, 8, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 8, 32)
    """

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd (an even kernel with padding='same' "
                f"shifts the feature map), got {kernel_size}"
            )
        if layer_scale_init_value is not None:
            if isinstance(layer_scale_init_value, bool) or not isinstance(
                    layer_scale_init_value, (int, float)):
                raise ValueError(
                    f"layer_scale_init_value must be a real number or None, "
                    f"got {layer_scale_init_value!r}"
                )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.kernel_size = kernel_size
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        # `norm` is the DEGENERATE MobileOneBlock: num_conv_branches=0 and
        # use_scale_branch=False leave only the identity BatchNormalization alive
        # (stride=1 and out_channels == in_channels guarantee it exists). Giving it
        # a conv branch would silently change the architecture while every shape
        # assertion still passes.
        self.norm = MobileOneBlock(
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            group_size=1,
            use_act=False,
            use_scale_branch=False,
            num_conv_branches=0,
            norm_epsilon=REFERENCE_NORM_EPSILON,
            padding_mode=REFERENCE_PADDING_MODE,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='norm',
        )
        self.mixer = MobileOneBlock(
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            group_size=1,
            use_act=False,
            norm_epsilon=REFERENCE_NORM_EPSILON,
            padding_mode=REFERENCE_PADDING_MODE,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='mixer',
        )
        self.layer_scale = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is not
            ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitRepMixer expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        self.norm.build(input_shape)
        self.mixer.build(input_shape)
        if self.layer_scale is not None:
            self.layer_scale.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the RepMixer token mixer.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour (the BatchNormalizations inside the two
            MobileOne blocks update their moving statistics otherwise).
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, dim)``.
        """
        residual = keras.ops.subtract(
            self.mixer(inputs, training=training),
            self.norm(inputs, training=training),
        )
        if self.layer_scale is not None:
            residual = self.layer_scale(residual, training=training)
        return keras.ops.add(inputs, residual)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to the input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'kernel_size': self.kernel_size,
            'layer_scale_init_value': self.layer_scale_init_value,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitRepMixer":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitRepMixer` instance.
        :rtype: FastVitRepMixer
        """
        config = dict(config)
        config['kernel_initializer'] = keras.initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.fastvit.components")
class FastVitRepMixerBlock(keras.layers.Layer):
    """FastViT RepMixer block: a RepMixer token mixer followed by a residual ConvMlp.

    Channels-last transcription of timm's ``RepMixerBlock``. The token-mixing half
    carries its residual connection internally (inside :class:`FastVitRepMixer`);
    the channel-mixing half is an explicit residual around
    :class:`FastVitConvMlp`, gated by its own LayerScale and guarded by
    stochastic depth.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │            Input [B, H, W, dim]              │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │ token_mixer: FastVitRepMixer                 │
        │   x + gamma1 * (mixer(x) - norm(x))          │
        └───────┬──────────────────────────┬───────────┘
                │                          ▼
                │            ┌──────────────────────────────┐
                │            │ mlp: FastVitConvMlp          │
                │            │  dw 7×7 + BN → 1×1 → act →   │
                │            │  drop → 1×1 → drop           │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ LayerScale (per-channel)     │
                │            │  gamma2, constraint=None     │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ StochasticDepth (drop path)  │
                │            └──────────────┬───────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │            Output [B, H, W, dim]             │
        └──────────────────────────────────────────────┘

    .. note::
       ``StochasticDepth`` short-circuits to the identity only when ``training is
       False`` (or the rate is exactly 0.0); ``training=None`` runs the stochastic
       path. Deterministic tests must pass ``training=False`` EXPLICITLY.

    :param dim: Number of channels. The block preserves it. Must be positive.
    :type dim: int
    :param kernel_size: Spatial size of the token mixer's depthwise convolution.
        Must be a positive odd integer. Defaults to 3.
    :type kernel_size: int
    :param mlp_ratio: Expansion ratio of the ConvMlp bottleneck; the hidden width
        is ``int(dim * mlp_ratio)``. Must be positive and yield a positive hidden
        width. Defaults to 4.0.
    :type mlp_ratio: float
    :param dropout_rate: Dropout rate inside the ConvMlp. Must be in ``[0, 1)``.
        Defaults to 0.0.
    :type dropout_rate: float
    :param drop_path_rate: Per-sample stochastic-depth rate applied to the ConvMlp
        residual branch. Must be in ``[0, 1)`` — the value is validated by
        :class:`StochasticDepth`, which raises for ``>= 1.0``. Defaults to 0.0.
    :type drop_path_rate: float
    :param layer_scale_init_value: Constant initialization for BOTH LayerScale
        gammas (the token mixer's and the block's), or ``None`` to omit LayerScale.
        Defaults to ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param activation: Activation used inside the ConvMlp. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for the token mixer's MobileOne
        convolution kernels. Defaults to ``'he_normal'``. The ConvMlp keeps its own
        reference default (``TruncatedNormal(stddev=0.02)``, per timm's
        ``_init_weights``).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every convolution kernel in
        the block.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim`` or ``kernel_size`` are not positive, if
        ``kernel_size`` is even, if ``mlp_ratio`` is not positive or yields a
        zero-width bottleneck, if ``dropout_rate`` is outside ``[0, 1)``, if
        ``drop_path_rate`` is outside ``[0, 1)``, or if
        ``layer_scale_init_value`` is not a real number or ``None``.

    Example:
        >>> import numpy as np
        >>> block = FastVitRepMixerBlock(dim=32, mlp_ratio=3.0)
        >>> y = block(np.zeros((2, 8, 8, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 8, 32)
    """

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        # `dim` / `kernel_size` / `layer_scale_init_value` are re-validated by
        # FastVitRepMixer below, but validating here keeps the raise site on the
        # block the caller actually constructed.
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError(
                f"mlp_ratio={mlp_ratio} with dim={dim} yields a zero-width "
                f"bottleneck (int(dim * mlp_ratio) == {hidden_dim})"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if layer_scale_init_value is not None:
            if isinstance(layer_scale_init_value, bool) or not isinstance(
                    layer_scale_init_value, (int, float)):
                raise ValueError(
                    f"layer_scale_init_value must be a real number or None, "
                    f"got {layer_scale_init_value!r}"
                )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.kernel_size = kernel_size
        self.mlp_ratio = float(mlp_ratio)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        self.token_mixer = FastVitRepMixer(
            dim=self.dim,
            kernel_size=self.kernel_size,
            layer_scale_init_value=self.layer_scale_init_value,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='token_mixer',
        )
        self.mlp = FastVitConvMlp(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_regularizer=self.kernel_regularizer,
            name='mlp',
        )
        self.layer_scale = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale')
        # Created UNCONDITIONALLY: StochasticDepth accepts a rate of exactly 0.0
        # and short-circuits to the identity for it, so there is no reason to gate
        # construction on the rate. `drop_path_rate >= 1.0` raises from here.
        self.drop_path = StochasticDepth(
            drop_path_rate=self.drop_path_rate, name='drop_path')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is not
            ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitRepMixerBlock expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        self.token_mixer.build(input_shape)
        mixed_shape = self.token_mixer.compute_output_shape(input_shape)

        self.mlp.build(mixed_shape)
        mlp_shape = self.mlp.compute_output_shape(mixed_shape)

        if self.layer_scale is not None:
            self.layer_scale.build(mlp_shape)
        self.drop_path.build(mlp_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the RepMixer block.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — ``StochasticDepth`` treats ``None`` as
            training.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, dim)``.
        """
        x = self.token_mixer(inputs, training=training)

        residual = self.mlp(x, training=training)
        if self.layer_scale is not None:
            residual = self.layer_scale(residual, training=training)
        residual = self.drop_path(residual, training=training)

        return keras.ops.add(x, residual)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to the input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'kernel_size': self.kernel_size,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'layer_scale_init_value': self.layer_scale_init_value,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitRepMixerBlock":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitRepMixerBlock` instance.
        :rtype: FastVitRepMixerBlock
        """
        config = dict(config)
        config['activation'] = keras.activations.deserialize(config['activation'])
        config['kernel_initializer'] = keras.initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.fastvit.components")
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
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
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
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

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
            padding_mode=REFERENCE_PADDING_MODE,
            norm_epsilon=REFERENCE_NORM_EPSILON,
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
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
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
        config['activation'] = keras.activations.deserialize(config['activation'])
        config['kernel_initializer'] = keras.initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.fastvit.components")
class FastVitAttentionBlock(keras.layers.Layer):
    """FastViT attention block: pre-norm global self-attention + a convolutional MLP.

    Channels-last transcription of timm's ``AttentionBlock``. Both halves are
    residual, each gated by its own per-channel LayerScale and guarded by its own
    stochastic-depth branch. The block preserves its input shape exactly.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │            Input [B, H, W, dim]              │
        └───────┬──────────────────────────┬───────────┘
                │                          ▼
                │            ┌──────────────────────────────┐
                │            │ norm (batch_norm | layer_norm│
                │            │       eps = 1e-5)            │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ reshape (B,H,W,C)→(B,H*W,C)  │
                │            │ MultiHeadAttention, no bias  │
                │            │ reshape (B,H*W,C)→(B,H,W,C)  │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ layer_scale_1 (per-channel)  │
                │            │ drop_path_1                  │
                │            └──────────────┬───────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                ┌───────────────┴──────────┐
                │                          ▼
                │            ┌──────────────────────────────┐
                │            │ mlp: FastVitConvMlp          │
                │            │  dw 7×7 + BN → 1×1 → act →   │
                │            │  drop → 1×1 → drop           │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ layer_scale_2 (per-channel)  │
                │            │ drop_path_2                  │
                │            └──────────────┬───────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │            Output [B, H, W, dim]             │
        └──────────────────────────────────────────────┘

    .. note::
       ``StochasticDepth`` short-circuits to the identity only when ``training is
       False`` (or the rate is exactly 0.0); ``training=None`` runs the stochastic
       path. Deterministic tests must pass ``training=False`` EXPLICITLY.

    **The rank-4 <-> rank-3 conversion is load-bearing.** The tower is
    channels-last rank-4, but the shared attention layer accepts ONLY rank-3
    ``(B, N, C)`` and raises on a 4-D hand-off (MEASURED), so this block owns the
    flatten and the reshape back. The reshape MUST be the exact inverse of the
    H-major flatten: producing ``(B, W, H, C)`` silently transposes every feature
    map while every shape still matches whenever ``H == W``. Spatial dims may be
    dynamic under ``fit()``'s tracing, so the target shape is derived with
    ``keras.ops.shape`` and never coerced with ``int()``.

    .. warning::
       **Recorded deviation from the reference (X-2).** timm's ``Attention`` is
       ``qkv_bias=False`` with a BIASED output projection. The shared
       ``MultiHeadAttention`` has a single ``use_bias`` covering both, so
       ``use_bias=False`` is used: it costs exactly one missing bias vector of
       length ``dim`` per block, whereas ``use_bias=True`` would ADD a spurious
       ``3 * dim`` qkv bias the reference does not have. Pinned by
       ``test_attention_has_no_bias_weights``.

    :param dim: Number of channels. The block preserves it. Must be positive and
        divisible by ``head_dim``.
    :type dim: int
    :param mlp_ratio: Expansion ratio of the ConvMlp bottleneck; the hidden width
        is ``int(dim * mlp_ratio)``. Must be positive and yield a positive hidden
        width. Defaults to 4.0.
    :type mlp_ratio: float
    :param head_dim: Per-head width. ``num_heads`` is ``dim // head_dim``. Must be
        positive and divide ``dim`` exactly. Defaults to 32 (the reference value).
    :type head_dim: int
    :param normalization_type: Key passed to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.
        ``'batch_norm'`` reproduces the reference for mci0/mci1/mci2;
        ``'layer_norm'`` reproduces ``LayerNormChannel`` for mci3/mci4 (which, in
        channels-last, is exactly ``LayerNormalization(axis=-1, epsilon=1e-5)``).
        Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param dropout_rate: Dropout rate inside the ConvMlp. Must be in ``[0, 1)``.
        Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate on the attention probabilities.
        Must be in ``[0, 1)``. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param drop_path_rate: Per-sample stochastic-depth rate applied to BOTH
        residual branches. Must be in ``[0, 1)``. Defaults to 0.0.
    :type drop_path_rate: float
    :param layer_scale_init_value: Constant initialization for BOTH LayerScale
        gammas, or ``None`` to omit LayerScale on both branches. Defaults to
        ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param activation: Activation used inside the ConvMlp. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for the attention projections. Defaults
        to ``'he_normal'``. The ConvMlp keeps its own reference default
        (``TruncatedNormal(stddev=0.02)``, per timm's ``_init_weights``).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer applied to the attention
        projections and to every ConvMlp kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim`` or ``head_dim`` are not positive, if ``dim`` is
        not divisible by ``head_dim``, if ``mlp_ratio`` is not positive or yields a
        zero-width bottleneck, if ``dropout_rate`` / ``attention_dropout_rate`` /
        ``drop_path_rate`` fall outside ``[0, 1)``, if ``normalization_type`` is not
        a string, or if ``layer_scale_init_value`` is neither a real number nor
        ``None``.

    Example:
        >>> import numpy as np
        >>> block = FastVitAttentionBlock(dim=64, head_dim=32)
        >>> y = block(np.zeros((2, 8, 4, 64), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 4, 64)
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            head_dim: int = _REFERENCE_HEAD_DIM,
            normalization_type: str = 'batch_norm',
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if dim % head_dim != 0:
            raise ValueError(
                f"dim must be divisible by head_dim (num_heads is derived as "
                f"dim // head_dim), got dim={dim} and head_dim={head_dim} "
                f"(remainder {dim % head_dim})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError(
                f"mlp_ratio={mlp_ratio} with dim={dim} yields a zero-width "
                f"bottleneck (int(dim * mlp_ratio) == {hidden_dim})"
            )
        if not isinstance(normalization_type, str):
            raise ValueError(
                f"normalization_type must be a string key accepted by "
                f"create_normalization_layer, got {normalization_type!r}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not 0.0 <= attention_dropout_rate < 1.0:
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1), "
                f"got {attention_dropout_rate}"
            )
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}")
        if layer_scale_init_value is not None:
            if isinstance(layer_scale_init_value, bool) or not isinstance(
                    layer_scale_init_value, (int, float)):
                raise ValueError(
                    f"layer_scale_init_value must be a real number or None, "
                    f"got {layer_scale_init_value!r}"
                )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.mlp_ratio = float(mlp_ratio)
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.hidden_dim = hidden_dim
        self.normalization_type = normalization_type
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        # epsilon is EXPLICIT: the factory would otherwise setdefault 1e-6, while
        # the reference (BatchNorm2d / LayerNormChannel) uses 1e-5 (MEASURED).
        self.norm = create_normalization_layer(
            self.normalization_type,
            name='norm',
            epsilon=REFERENCE_NORM_EPSILON,
        )
        # The shared attention layer accepts ONLY rank-3 input, so this block owns
        # the flatten/reshape (see the class docstring). `use_bias=False` is the
        # recorded deviation X-2 (qkv unbiased matches, proj unbiased does not).
        self.attn = create_attention_layer(
            'multi_head',
            name='attn',
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.layer_scale_1 = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale_1')
        self.layer_scale_2 = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale_2')
        # Created UNCONDITIONALLY: StochasticDepth accepts a rate of exactly 0.0
        # and short-circuits to the identity for it.
        self.drop_path_1 = StochasticDepth(
            drop_path_rate=self.drop_path_rate, name='drop_path_1')
        self.drop_path_2 = StochasticDepth(
            drop_path_rate=self.drop_path_rate, name='drop_path_2')
        self.mlp = FastVitConvMlp(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_regularizer=self.kernel_regularizer,
            name='mlp',
        )

    # ------------------------------------------------------------------
    # rank-4 <-> rank-3 conversion
    # ------------------------------------------------------------------

    def _to_sequence(self, x):
        """Flatten ``(B, H, W, C)`` to ``(B, H*W, C)`` in H-major token order.

        :param x: Rank-4 channels-last tensor.
        :return: A ``(sequence, spatial_shape)`` pair, where ``spatial_shape`` is
            the argument to feed back to :meth:`_to_spatial` so the reshape is the
            EXACT inverse of this flatten.
        :rtype: Tuple[Any, Any]

        .. warning::
           The returned token order is H-major (row-major over ``(H, W)``). Any
           other order — e.g. flattening a ``(B, W, H, C)`` transpose — must be
           inverted by the matching inverse, or the feature map is silently
           transposed. See ``test_non_square_input_roundtrips_orientation``.
        """
        static = tuple(x.shape)
        height, width = static[1], static[2]
        if height is not None and width is not None:
            # Static fast path: both spatial dims are known at trace time.
            return (
                keras.ops.reshape(x, (-1, height * width, self.dim)),
                (-1, height, width, self.dim),
            )
        # Dynamic path: derive the dims as tensors. NEVER coerce a traced
        # dimension with int()/tuple() — that raises under @tf.function, which is
        # the regime fit() runs in.
        dynamic = keras.ops.shape(x)
        sequence_shape = keras.ops.stack(
            [dynamic[0], dynamic[1] * dynamic[2], dynamic[3]])
        spatial_shape = keras.ops.stack(
            [dynamic[0], dynamic[1], dynamic[2], dynamic[3]])
        return keras.ops.reshape(x, sequence_shape), spatial_shape

    def _to_spatial(self, sequence, spatial_shape):
        """Reshape ``(B, H*W, C)`` back to ``(B, H, W, C)``.

        :param sequence: Rank-3 tensor produced by :meth:`_to_sequence` (or by a
            shape-preserving op applied to it).
        :param spatial_shape: The second element returned by :meth:`_to_sequence`.
        :return: Rank-4 channels-last tensor with the original spatial layout.
        """
        return keras.ops.reshape(sequence, spatial_shape)

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is not
            ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitAttentionBlock expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        input_shape = tuple(input_shape)
        batch, height, width, channels = input_shape
        tokens = None if (height is None or width is None) else height * width
        # The attention sub-layer validates rank and raises on a 4-D shape, so it
        # must be built with the FLATTENED rank-3 shape (MEASURED, F-6 P-1).
        sequence_shape = (batch, tokens, channels)

        self.norm.build(input_shape)
        self.attn.build(sequence_shape)
        if self.layer_scale_1 is not None:
            self.layer_scale_1.build(input_shape)
        self.drop_path_1.build(input_shape)

        self.mlp.build(input_shape)
        mlp_shape = self.mlp.compute_output_shape(input_shape)
        if self.layer_scale_2 is not None:
            self.layer_scale_2.build(mlp_shape)
        self.drop_path_2.build(mlp_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the FastViT attention block.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — ``StochasticDepth`` treats ``None`` as
            training, and a BatchNormalization norm updates its moving statistics.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, dim)``.
        """
        normed = self.norm(inputs, training=training)
        sequence, spatial_shape = self._to_sequence(normed)
        attended = self.attn(sequence, training=training)
        residual = self._to_spatial(attended, spatial_shape)

        if self.layer_scale_1 is not None:
            residual = self.layer_scale_1(residual, training=training)
        residual = self.drop_path_1(residual, training=training)
        x = keras.ops.add(inputs, residual)

        residual = self.mlp(x, training=training)
        if self.layer_scale_2 is not None:
            residual = self.layer_scale_2(residual, training=training)
        residual = self.drop_path_2(residual, training=training)
        return keras.ops.add(x, residual)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to the input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'mlp_ratio': self.mlp_ratio,
            'head_dim': self.head_dim,
            'normalization_type': self.normalization_type,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'layer_scale_init_value': self.layer_scale_init_value,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitAttentionBlock":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitAttentionBlock` instance.
        :rtype: FastVitAttentionBlock
        """
        config = dict(config)
        config['activation'] = keras.activations.deserialize(config['activation'])
        config['kernel_initializer'] = keras.initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.fastvit.components")
class FastVitStage(keras.layers.Layer):
    """One FastViT stage: optional downsample, optional RepCPE, then ``depth`` blocks.

    Channels-last transcription of timm's ``FastVitStage``.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────────┐
        │            Input [B, H, W, C_in]                 │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │ downsample: FastVitPatchEmbed                    │  (if downsample)
        │   LKC(k=down_patch_size, s=down_stride, dw)      │
        │   + MobileOneBlock(k=1)                          │
        │   → [B, ceil(H/s), ceil(W/s), dim]               │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │ pos_emb: RepConditionalPosEnc                    │  (if use_pos_emb)
        │   dw conv(spatial_shape) + skip                  │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │ block_0   drop_path_rates[0]                     │
        │ block_1   drop_path_rates[1]                     │
        │   ...     (FastVitRepMixerBlock | FastVitAttn)   │
        │ block_D-1 drop_path_rates[D-1]                   │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │        Output [B, H', W', dim]                   │
        └──────────────────────────────────────────────────┘

    .. note::
       ``StochasticDepth`` short-circuits to the identity only when ``training is
       False`` (or the rate is exactly 0.0); ``training=None`` runs the stochastic
       path. Deterministic tests must pass ``training=False`` EXPLICITLY.

    **The stage does NOT compute its own drop-path schedule.** The reference
    computes it ONCE, globally, across every block of every stage
    (``calculate_drop_path_rates(drop_path_rate, layers, stagewise=True)``) and
    hands each stage its slice: stage 2 of a ``[2, 12, 24, 4]`` model starts where
    stage 1's rates ended, not at zero. ``drop_path_rates`` is therefore an
    EXPLICIT list of exactly ``depth`` floats, and the stage's only job is to hand
    element ``i`` to block ``i``. A reversed or off-by-one list produces an
    identically-shaped, subtly-wrong model, so the handoff carries its own pin
    (``test_per_block_drop_path_wiring``). Build the global schedule with
    :func:`dl_techniques.utils.drop_path.linear_drop_path_rates` and slice it.

    **The blocks live in a FLAT Python list** (``self.blocks``, named ``block_0``
    ... ``block_{depth-1}`` independently of the token mixer). A nested
    ``List[List[Layer]]`` attribute silently loses weights on a ``.keras`` round
    trip while the layer count, the variable paths AND the parameter total all
    still match (MEASURED, repo-wide); the round trip is pinned elementwise by
    ``test_roundtrip_preserves_block_weights_elementwise``.

    :param dim: Output channel count of the stage. Every block preserves it. Must
        be positive.
    :type dim: int
    :param depth: Number of token-mixer blocks. Must be positive.
    :type depth: int
    :param token_mixer: Which block type to repeat, ``'repmixer'`` or
        ``'attention'``. Defaults to ``'repmixer'``.
    :type token_mixer: str
    :param downsample: Whether the stage begins with a
        :class:`FastVitPatchEmbed`.
        ``False`` for the first stage of every MCi variant (the stem has already
        done the /4). Defaults to ``True``.
    :type downsample: bool
    :param se_downsample: Whether the downsample's large-kernel convolution uses a
        squeeze-and-excitation block. Defaults to ``False``.
    :type se_downsample: bool
    :param use_pos_emb: Whether to insert a
        :class:`~dl_techniques.layers.embedding.rep_conditional_pos_enc.RepConditionalPosEnc`
        after the downsample. Defaults to ``False``.
    :type use_pos_emb: bool
    :param pos_emb_spatial_shape: Depthwise kernel size of the positional encoding.
        Defaults to ``(7, 7)``, the reference value. Ignored when ``use_pos_emb``
        is ``False``.
    :type pos_emb_spatial_shape: Union[int, Tuple[int, int]]
    :param mlp_ratio: ConvMlp expansion ratio inside every block. Defaults to 4.0.
    :type mlp_ratio: float
    :param repmixer_kernel_size: Depthwise kernel size of the RepMixer token mixer.
        Used only when ``token_mixer == 'repmixer'``. Defaults to 3.
    :type repmixer_kernel_size: int
    :param head_dim: Per-head width of the attention token mixer; ``num_heads`` is
        ``dim // head_dim``. Used only when ``token_mixer == 'attention'``.
        Defaults to 32.
    :type head_dim: int
    :param normalization_type: Norm key for the attention block's pre-norm. Used
        only when ``token_mixer == 'attention'``. Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param down_patch_size: Kernel size of the downsample's large-kernel
        convolution. Defaults to 7.
    :type down_patch_size: int
    :param down_stride: Stride of the downsample. Defaults to 2.
    :type down_stride: int
    :param lkc_use_act: Whether the downsample's large-kernel convolution applies
        its activation. Defaults to ``True``.
    :type lkc_use_act: bool
    :param dropout_rate: Dropout rate inside every block's ConvMlp. Must be in
        ``[0, 1)``. Defaults to 0.0.
    :type dropout_rate: float
    :param drop_path_rates: Per-block stochastic-depth rates — a sequence of
        EXACTLY ``depth`` floats, block ``i`` receiving element ``i``. ``None``
        means all zeros. The stage never computes this schedule itself; see the
        class docstring. Defaults to ``None``.
    :type drop_path_rates: Optional[Sequence[float]]
    :param layer_scale_init_value: Constant initialization for every LayerScale
        gamma, or ``None`` to omit LayerScale. Defaults to ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param activation: Activation used inside the blocks and the downsample.
        Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for the convolution / projection
        kernels. Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer applied to every kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim``, ``depth``, ``down_patch_size``, ``down_stride``
        or ``repmixer_kernel_size`` are not positive, if ``token_mixer`` is not one
        of ``'repmixer'`` / ``'attention'``, if ``dropout_rate`` is outside
        ``[0, 1)``, or if ``drop_path_rates`` is given with a length other than
        ``depth``.

    Example:
        >>> import numpy as np
        >>> stage = FastVitStage(dim=64, depth=2, token_mixer='repmixer',
        ...                      downsample=True, drop_path_rates=[0.0, 0.1])
        >>> y = stage(np.zeros((2, 16, 16, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 8, 64)
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            token_mixer: str = 'repmixer',
            downsample: bool = True,
            se_downsample: bool = False,
            use_pos_emb: bool = False,
            pos_emb_spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
            mlp_ratio: float = 4.0,
            repmixer_kernel_size: int = 3,
            head_dim: int = 32,
            normalization_type: str = 'batch_norm',
            down_patch_size: int = 7,
            down_stride: int = 2,
            lkc_use_act: bool = True,
            dropout_rate: float = 0.0,
            drop_path_rates: Optional[Sequence[float]] = None,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if token_mixer not in _TOKEN_MIXERS:
            raise ValueError(
                f"token_mixer must be one of "
                f"{', '.join(repr(name) for name in _TOKEN_MIXERS)}, "
                f"got {token_mixer!r}"
            )
        if repmixer_kernel_size <= 0:
            raise ValueError(
                f"repmixer_kernel_size must be positive, "
                f"got {repmixer_kernel_size}"
            )
        if down_patch_size <= 0:
            raise ValueError(
                f"down_patch_size must be positive, got {down_patch_size}")
        if down_stride <= 0:
            raise ValueError(f"down_stride must be positive, got {down_stride}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        resolved_rates = self._resolve_drop_path_rates(drop_path_rates, depth)

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.depth = depth
        self.token_mixer = token_mixer
        self.downsample_enabled = bool(downsample)
        self.se_downsample = bool(se_downsample)
        self.use_pos_emb = bool(use_pos_emb)
        self.pos_emb_spatial_shape = pos_emb_spatial_shape
        self.mlp_ratio = float(mlp_ratio)
        self.repmixer_kernel_size = repmixer_kernel_size
        self.head_dim = head_dim
        self.normalization_type = normalization_type
        self.down_patch_size = down_patch_size
        self.down_stride = down_stride
        self.lkc_use_act = bool(lkc_use_act)
        self.dropout_rate = dropout_rate
        self.drop_path_rates = resolved_rates
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        # `downsample` and `pos_emb` are genuinely absent (None) when disabled
        # rather than being an Identity: the reference substitutes nn.Identity,
        # and a None attribute makes "is this stage a downsampling stage?"
        # answerable by inspection instead of by counting weights.
        self.downsample = (
            FastVitPatchEmbed(
                embed_dim=self.dim,
                patch_size=self.down_patch_size,
                stride=self.down_stride,
                use_se=self.se_downsample,
                lkc_use_act=self.lkc_use_act,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='downsample',
            )
            if self.downsample_enabled else None
        )
        self.pos_emb = (
            RepConditionalPosEnc(
                dim=self.dim,
                spatial_shape=self.pos_emb_spatial_shape,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='pos_emb',
            )
            if self.use_pos_emb else None
        )

        # FLAT list — a nested List[List[Layer]] silently drops weights on a
        # .keras round trip while counts, paths and parameter totals all match.
        # Block i receives drop_path_rates[i]; nothing else in this class is as
        # easy to get subtly and invisibly wrong.
        self.blocks: List[keras.layers.Layer] = [
            self._create_block(index) for index in range(self.depth)
        ]

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_drop_path_rates(
            drop_path_rates: Optional[Sequence[float]],
            depth: int
    ) -> List[float]:
        """Validate ``drop_path_rates`` and normalize it to a list of ``depth`` floats.

        :param drop_path_rates: ``None`` (meaning all zeros) or a sequence of
            exactly ``depth`` numbers.
        :type drop_path_rates: Optional[Sequence[float]]
        :param depth: Number of blocks in the stage.
        :type depth: int
        :return: A list of ``depth`` floats.
        :rtype: List[float]
        :raises ValueError: If the sequence length differs from ``depth``, or if
            any entry is not a real number.
        """
        if drop_path_rates is None:
            return [0.0] * depth
        try:
            rates = list(drop_path_rates)
        except TypeError:
            raise ValueError(
                f"drop_path_rates must be None or a sequence of {depth} floats, "
                f"got {drop_path_rates!r}"
            )
        if len(rates) != depth:
            raise ValueError(
                f"drop_path_rates must have exactly one entry per block: "
                f"depth={depth} but len(drop_path_rates)={len(rates)}. The stage "
                f"does NOT compute the schedule itself — the encoder slices the "
                f"global stagewise schedule and passes this stage its slice."
            )
        for rate in rates:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ValueError(
                    f"drop_path_rates entries must be real numbers, got {rate!r}"
                )
        return [float(rate) for rate in rates]

    def _create_block(self, index: int) -> keras.layers.Layer:
        """Create block ``index`` of the requested token-mixer type.

        :param index: Position of the block inside the stage; also the index into
            ``drop_path_rates``.
        :type index: int
        :return: An unbuilt block layer.
        :rtype: keras.layers.Layer
        """
        # Names are `block_{i}` and NOT derived from the token mixer, so the
        # variable paths are stable across a repmixer/attention configuration
        # change and a checkpoint's layout does not depend on it.
        name = f'block_{index}'
        drop_path_rate = self.drop_path_rates[index]

        if self.token_mixer == 'repmixer':
            return FastVitRepMixerBlock(
                dim=self.dim,
                kernel_size=self.repmixer_kernel_size,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=self.layer_scale_init_value,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=name,
            )
        return FastVitAttentionBlock(
            dim=self.dim,
            mlp_ratio=self.mlp_ratio,
            head_dim=self.head_dim,
            normalization_type=self.normalization_type,
            dropout_rate=self.dropout_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=self.layer_scale_init_value,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=name,
        )

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer in FORWARD order, then the layer itself.

        The downsample changes both spatial dimensions and the channel count, so
        the positional encoding and every block must be built on the DOWNSAMPLED
        shape — not on ``input_shape``.

        :param input_shape: Shape of the input tensor, ``(B, H, W, C_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4, or if the stage has no
            downsample and the input channel count is not ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitStage expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        input_shape = tuple(input_shape)

        if self.downsample is not None:
            self.downsample.build(input_shape)
            current_shape = self.downsample.compute_output_shape(input_shape)
        else:
            if input_shape[-1] is not None and input_shape[-1] != self.dim:
                raise ValueError(
                    f"A stage without a downsample cannot change the channel "
                    f"count: input channels must equal dim={self.dim}, got "
                    f"{input_shape[-1]}. Set downsample=True to project them."
                )
            current_shape = input_shape

        if self.pos_emb is not None:
            self.pos_emb.build(current_shape)
            current_shape = self.pos_emb.compute_output_shape(current_shape)

        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the stage.

        :param inputs: Input tensor of shape ``(B, H, W, C_in)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — every block contains BatchNormalization and
            a stochastic-depth branch that treats ``None`` as training.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H', W', dim)``.
        """
        x = inputs
        if self.downsample is not None:
            x = self.downsample(x, training=training)
        if self.pos_emb is not None:
            x = self.pos_emb(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        Only the downsample changes the shape; the positional encoding and every
        block preserve it.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, ceil(H/down_stride), ceil(W/down_stride), dim)`` when the
            stage downsamples, else ``(B, H, W, dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        if not self.downsample_enabled:
            return input_shape[:-1] + (self.dim,)
        stride = self.down_stride
        height = (
            None if input_shape[1] is None
            else (input_shape[1] + stride - 1) // stride
        )
        width = (
            None if input_shape[2] is None
            else (input_shape[2] + stride - 1) // stride
        )
        return (input_shape[0], height, width, self.dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'depth': self.depth,
            'token_mixer': self.token_mixer,
            'downsample': self.downsample_enabled,
            'se_downsample': self.se_downsample,
            'use_pos_emb': self.use_pos_emb,
            'pos_emb_spatial_shape': self.pos_emb_spatial_shape,
            'mlp_ratio': self.mlp_ratio,
            'repmixer_kernel_size': self.repmixer_kernel_size,
            'head_dim': self.head_dim,
            'normalization_type': self.normalization_type,
            'down_patch_size': self.down_patch_size,
            'down_stride': self.down_stride,
            'lkc_use_act': self.lkc_use_act,
            'dropout_rate': self.dropout_rate,
            'drop_path_rates': list(self.drop_path_rates),
            'layer_scale_init_value': self.layer_scale_init_value,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitStage":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitStage` instance.
        :rtype: FastVitStage
        """
        config = dict(config)
        spatial_shape = config.get('pos_emb_spatial_shape')
        if isinstance(spatial_shape, list):
            config['pos_emb_spatial_shape'] = tuple(spatial_shape)
        config['activation'] = keras.activations.deserialize(config['activation'])
        config['kernel_initializer'] = keras.initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)

# ---------------------------------------------------------------------
