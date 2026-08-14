"""
ConvUNext: Modern U-Net with ConvNeXt-Inspired Architecture

The single home for BOTH ConvUNext arms. ``create_convunext(..., use_bias=True)``
builds a bias-carrying network; ``use_bias=False`` builds the bias-free denoiser
that ``models/bias_free_denoisers/bfconvunext.py`` wraps. The model is a Keras
FUNCTIONAL graph — the subclassed ``ConvUNextModel`` that used to live here (with
its own ``ConvUNextStem``, its own variant dict, a bespoke inference-model helper
and pretrained-weight placeholders) has been deleted in favour of it.

ConvUNext combines U-Net and ConvNeXt:

- U-Net's encoder-decoder structure with skip connections
- ConvNeXt's modern architectural innovations via the existing
  ``ConvNextV1Block`` / ``ConvNextV2Block`` implementations
- Depthwise separable convolutions and an inverted bottleneck
- Global Response Normalization (GRN) for V2 blocks
- Larger kernel sizes (7x7) for better receptive fields
- Layer scaling for training stability and optional stochastic depth

Under ``use_bias=False`` the architecture aims at the bias-free principle: if the
input is scaled by alpha, the output is scaled by alpha, which is what makes a
denoiser generalize across noise levels and enables the Miyasawa/Tweedie
residual-as-score reading. ``create_convunext``'s docstring names the three
deliberate exceptions to that (the exempt activations, two hardcoded bias-free
sites, and GRN's un-threaded ``beta``).

Deep supervision provides better gradient flow to deeper layers, multi-scale
feature learning, more stable training of very deep networks, and curriculum
learning through weight scheduling. With it enabled the model outputs multiple
scales: output 0 is the final inference output (highest resolution) and outputs
1..N are intermediate supervision outputs at progressively lower resolutions.

Optional Laplacian-pyramid downsample/skip path (``use_laplacian_pyramid``, OFF by
default): every encoder down<->skip junction stops using ``MaxPooling2D`` + a raw
full-resolution skip and instead applies a single ``LaplacianPyramidLevel`` split::

    low, high = split(x)   # low = blur-then-subsample(x); high = x - upsample(low)

The coarse, anti-aliased ``low`` band descends the encoder; the high-frequency
residual ``high`` band becomes the skip. The two bands are exactly complementary
(``merge(low, high) == x``), so the split is lossless *taken together*.

The reason for it is NOT just lossless downsampling -- it is that **no single path
then carries all the information needed for reconstruction**. The skip holds only
the high band and the descending/bottleneck path holds only the low band, so
neither is a sufficient statistic; the decoder is forced to FUSE both to rebuild
the signal. This removes the classic U-Net shortcut where a full-resolution skip
carries the whole image, letting the network learn a near-identity copy and leaving
the encoder->bottleneck->decoder pathway lazy and underused. For a denoiser this
doubly matters: the trivial "copy the noisy input, do nothing" solution that hides
in a full-resolution skip is gone once that skip only holds the high-frequency
residual.

Secondary benefit -- an inductive bias matched to denoising: white Gaussian noise
is flat across frequency while natural-image signal concentrates in low
frequencies, so per band the SNR differs sharply. Splitting at every scale gives
the network the subband structure of classical optimal denoising
(wavelet-shrinkage / per-band Wiener). Crucially this costs nothing on the theory
side: ``LaplacianPyramidLevel`` is built only from linear ops (bias-free Gaussian
blur -> blur-pool -> bilinear upsample -> subtraction), so it is homogeneous of
degree 1 with zero additive offset.

Based on ConvNeXt innovations from "A ConvNet for the 2020s" (Liu et al., CVPR
2022) and "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
Autoencoders" (Woo et al., CVPR 2023). The Laplacian-pyramid split follows Burt &
Adelson, "The Laplacian Pyramid as a Compact Image Code" (1983).
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, List, Dict, Any, FrozenSet

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.match_channels import MatchChannels
from dl_techniques.layers.downsample_and_skip import DownsampleAndSkip
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.initializers import create_gabor_depthwise_conv2d


# ---------------------------------------------------------------------
# ConvUNext Stem Block
# ---------------------------------------------------------------------

# DECISION plan-2026-08-14T092357-0e3d792d/D-010: the `package=` string below says
# `dl_techniques.bias_free_denoisers`, NOT `dl_techniques.convunext`, even though this
# class now lives in `dl_techniques/models/convunext/model.py`. The mismatch is
# DELIBERATE and load-bearing. Keras keys a registered serializable on
# `package` + class name and NEVER on the defining module (measured on Keras 3.8.0,
# D-008), so keeping this string byte-unchanged keeps the registry key
# `dl_techniques.bias_free_denoisers>ConvUNextStem` stable across this relocation —
# which is what lets any `.keras` artifact written before the move still load.
# Do NOT "fix" this to `dl_techniques.convunext` for tidiness: that is a KEY CHANGE and
# it silently breaks every checkpoint containing this layer. See decisions.md D-010/D-005.
@keras.saving.register_keras_serializable(package="dl_techniques.bias_free_denoisers")
class ConvUNextStem(keras.layers.Layer):
    """ConvUNext stem block for initial feature extraction.

    Single home for BOTH ConvUNext arms: the bias-free denoiser stem (GRN +
    activation, ``use_bias=False``) and the standard ConvUNext stem (LayerNorm,
    ``use_bias=True``). The two used to be separate same-named classes in two
    modules; the normalization choice and the bias flag are now parameters.

    **Architecture**::

        Input(batch, height, width, channels)
               |
        Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)
               |
        <stem_normalization>            # via create_normalization_layer
               |
        Activation(activation)          # 'linear' reproduces a no-activation stem
               |
        Output(batch, height, width, filters)

    Spatial dimensions are preserved (``padding='same'``, stride 1).

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param kernel_size: Spatial size of the convolution kernel. Defaults to 7.
    :type kernel_size: int or tuple of 2 ints
    :param activation: Activation applied after the normalization. May be a string
        or a ``keras.layers.Layer`` instance (e.g. ``LeakyReLU(0.1)``). Defaults to
        ``'gelu'``. Pass ``'linear'`` for a stem with no activation.
    :type activation: str or keras.layers.Layer
    :param use_bias: Whether the stem convolution allocates a bias vector. Defaults
        to ``True``. Bias-free / Miyasawa denoisers must pass ``False`` — degree-1
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
        self.activation_name = activation
        self.use_bias = use_bias
        self.stem_normalization = stem_normalization
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Sublayers initialized in build()
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

        # Normalization through the norms factory so both arms are expressible:
        # 'global_response_norm' (ConvNeXt V2 / bias-free) or 'layer_norm' (standard
        # ConvNeXt). The factory's epsilon default (1e-6) equals both target classes'
        # own defaults used here previously, so neither arm's numerics move.
        self.norm = create_normalization_layer(
            self.stem_normalization,
            name='stem_norm'
        )

        # Explicitly build sublayers so weights materialize on .keras reload
        # (lazy auto-build drops their state during deserialization).
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(conv_output_shape)

        # Normalization is shape-preserving, so the activation input shape
        # == conv_output_shape.
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
        """Forward pass.

        :param inputs: Input tensor of shape ``(batch, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the call is in training mode.
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
        :return: Shape of the output tensor.
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
            # activation so LeakyReLU(alpha) round-trips through .keras; the string path
            # stays raw for backward-compat. Mirrors the block fix (D-001). Do NOT emit a
            # dict for a plain string activation — that would break existing 'gelu' configs.
            'activation': keras.layers.serialize(self.activation_name) if isinstance(
                self.activation_name, keras.layers.Layer) else self.activation_name,
            'use_bias': self.use_bias,
            'stem_normalization': self.stem_normalization,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ConvUNextStem':
        """Deserialize, reviving a layer-instance activation from its dict form.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: dict
        :return: Reconstructed layer instance.
        :rtype: ConvUNextStem
        """
        config = dict(config)
        if isinstance(config.get('activation'), dict):
            config['activation'] = keras.layers.deserialize(config['activation'])
        # kernel_initializer/kernel_regularizer dicts are passed straight to __init__,
        # where keras.*.get(...) accepts a serialized dict (Keras 3).
        return cls(**config)

# ---------------------------------------------------------------------
# Spatial wrapper around bias-free LinearAttention (4D <-> 3D)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()  # DECISION plan_2026-07-11_bb4b38b5/D-002
class SpatialLinearAttention(keras.layers.Layer):
    """Apply a bias-free LinearAttention over a 4D spatial feature map.

    ``LinearAttention`` (the repo's only Miyasawa-compliant, degree-1-homogeneous
    attention) accepts strictly 3D sequence input ``(B, N, dim)`` and raises on 4D.
    This thin wrapper flattens a bottleneck tensor ``(B, H, W, C)`` to
    ``(B, H*W, C)`` using DYNAMIC ``ops.shape`` (H/W are ``None`` at graph-build
    time whenever the model is built with ``input_shape=(None, None, C)``), attends,
    and reshapes back to ``(B, H, W, C)``. Output shape equals input shape.

    The attention sublayer is built through the attention factory with a hardcoded
    ``'linear'`` type and ``use_bias=False`` + the default ``feature_map='relu'`` so
    the bias-free / degree-1-homogeneity property is preserved (see D-001/D-002).
    That inner ``use_bias=False`` is HARDCODED and is deliberately NOT threaded from
    ``create_convunext``'s ``use_bias`` — see the builder docstring's "Deliberate
    asymmetries" section.

    Args:
        dim: Integer, channel count of the input feature map (``C``); also the
            attention embedding dim. Must be divisible by ``num_heads``.
        num_heads: Integer, number of attention heads. Defaults to 8.
        name: Optional string, layer name.
        **kwargs: Additional arguments for the Layer base class.
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

        # DECISION plan_2026-07-11_bb4b38b5/D-001: construct the bias-free attention via the
        # factory with a HARDCODED 'linear' type (the only degree-1-homogeneity-safe attention),
        # keeping use_bias=False + default feature_map='relu'. Do NOT import LinearAttention
        # directly (factory-first policy) and do NOT expose a type knob (any softmax type
        # silently breaks the Miyasawa property the denoiser depends on).
        self.attn = create_attention_layer(
            'linear', dim=self.dim, num_heads=self.num_heads,
            use_bias=False, name=f'{self.name}_linear'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build the nested attention on the flattened SEQUENCE shape.

        ``input_shape`` is the 4D spatial shape ``(B, H, W, C)``. In ``call`` the
        attention sublayer only ever sees the flattened 3D sequence
        ``(B, H*W, dim)``, so it must be built with a dynamic sequence length
        (``None``) and last dim ``self.dim``. Building the sublayer here (rather
        than letting it build lazily inside ``call``) materializes its 4 Dense
        projections BEFORE ``.keras`` load, so ``keras.models.load_model`` restores
        every weight instead of dropping the lazily-built objects (guide §3.2).
        """
        self.attn.build((input_shape[0], None, self.dim))
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Flatten spatial dims, attend, reshape back. Uses dynamic shapes."""
        shape = ops.shape(inputs)
        b, h, w = shape[0], shape[1], shape[2]
        seq = ops.reshape(inputs, [b, h * w, self.dim])
        attended = self.attn(seq, training=training)
        return ops.reshape(attended, [b, h, w, self.dim])

    def compute_output_shape(self, input_shape):
        """Shape-preserving."""
        return input_shape

    def get_config(self):
        """Get layer configuration (attn sublayer is rebuilt from these in __init__)."""
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
        })
        return config


# ---------------------------------------------------------------------
# ConvUNext Model Variant Configurations
# ---------------------------------------------------------------------

# The ONE variant dict for BOTH arms (plan invariant I-5). It used to be duplicated as
# `ConvUNextModel.MODEL_VARIANTS` (deleted) and `bfconvunext.CONVUNEXT_CONFIGS`; the two
# were mechanically identical apart from their `description` strings.
#
# It deliberately carries NO `block_normalization` key. Adding one here would flip the
# bias-ON variants too; only `bfconvunext.create_convunext_variant` selects 'batchnorm',
# and it does so at the wrapper (plan invariant I-3 / decisions.md D-003).
CONVUNEXT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'depth': 3,
        'initial_filters': 32,  # Start conservative to avoid OOM
        'blocks_per_level': 2,
        'convnext_version': 'v2',  # Use V2 by default for GRN
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
    """Apply a ConvNeXt block as a RESIDUAL branch with stochastic depth.

    ``dropout_rate`` is the standard (element-wise) MLP dropout applied INSIDE
    the block's inverted bottleneck (after the 4x-expansion activation in V1 /
    after GRN in V2, before the 1x1 reduce). It is NOT stochastic depth (that is
    ``drop_path_rate``, applied to the whole branch below). Default ``0.0`` adds
    no ``Dropout`` sublayer (passthrough ``Lambda``) and is byte-identical to the
    prior hardcoded behavior. ``spatial_dropout_rate`` stays hardcoded ``0.0``.

    ``use_bias`` is threaded from ``create_convunext``; it defaults to ``False``
    (the bias-free denoiser value this helper was written for), so an omitted
    argument reproduces the pre-merge graph exactly.

    ``ConvNextV1Block`` / ``ConvNextV2Block`` implement only the residual
    *branch* — they do NOT add the skip connection or apply drop-path (their
    ``dropout_rate`` is regular MLP dropout, not stochastic depth). The canonical
    ConvNeXt wiring (matching ``models/convnext/convnext_v1.py``) is::

        x = x + StochasticDepth(drop_path_rate)(block(x))

    The block input and output channel counts both equal ``filters`` (callers
    channel-adjust before the blocks), so the residual add is always valid and
    bias-free (identity + a homogeneous branch stays homogeneous).

    LayerScale ``gamma`` is initialized to 1e-4 (CaiT's moderate-depth default) so each
    residual branch starts small (a mild near-identity prior) while STILL receiving usable
    gradients from step 0: the gradient w.r.t. the branch weights is proportional to gamma,
    so an over-small init (the old 1e-6) throttles early learning until gamma slowly grows.
    A hard floor of 1e-6 (``ConvNext*Block.GAMMA_MIN_VALUE``, enforced by
    ``ValueRangeConstraint``) keeps gamma from collapsing to zero, which would permanently
    kill a branch (gamma==0 => zero branch gradient => stuck dead). Init stability does NOT
    depend on a tiny gamma: the main-path structural convs use orthogonal (norm-preserving)
    init, which is what actually prevents the variance explosion the old ``he_normal`` init
    caused (the full denoiser is init-stable across gamma in [1e-6, 1.0], verified by sweep).
    """
    residual = x
    # DECISION plan-2026-08-11T201945-91938f65/D-002 + D-004: ConvNextV1Block /
    # ConvNextV2Block are the residual BRANCH ONLY (they end at gamma(x), no add), so the
    # CALLER must supply the residual and the drop-path — which is what this helper is.
    # The drop-path schedule goes to StochasticDepth (drop-path on the residual BRANCH),
    # NEVER to the block's `dropout_rate=` kwarg: that kwarg is ordinary elementwise
    # dropout INSIDE the inverted-bottleneck MLP (convnext_v1_block.py:103-121) and has
    # nothing to do with stochastic depth. Do NOT collapse this to `x = block(x)` — that
    # silently drops the skip connection and makes stochastic depth meaningless. (These
    # anchors were carried here when the subclassed `ConvUNextModel`, whose inline encoder
    # loop originally held them, was deleted; this helper is now the ONE place the
    # invariant lives.) See that plan's decisions.md D-002, D-004.
    # DECISION plan_2026-06-21_eb7fd829/D-002: block activation is threaded via this single
    # choke-point (mirrors the kernel_regularizer / depthwise_* precedent) so one factory arg
    # reaches every encoder/bottleneck/decoder block at once. Factory default stays 'gelu' so
    # non-bfunet callers are byte-identical. (That claim was originally justified against two
    # named callers, `convnext` and `convnext_patch_vae`; the latter package has since been
    # deleted, so only the `convnext` half is still checkable.) NOTE (iter-2,
    # D-005/D-006 superseded the original iter-1 scope): the stem (ConvUNextStem, D-005) and the
    # deep-supervision head (_make_supervision_activation, D-006) are now ALSO configurable via
    # the factory's stem_activation / supervision_activation params (each default 'gelu'). See
    # decisions.md D-002/D-005/D-006.
    y = block_cls(
        kernel_size=kernel_size,
        filters=filters,
        activation=activation,
        use_bias=use_bias,         # False => bias-free / scaling-invariant
        dropout_rate=dropout_rate, # MLP dropout: 0.0 (default) keeps StochasticDepth-only regularization; >0 enables per-block dropout
        spatial_dropout_rate=0.0,  # not exposed (locked decision)
        gamma_initial_value=1e-4,  # LayerScale init (floored at GAMMA_MIN_VALUE=1e-6, can't die)
        kernel_regularizer=kernel_regularizer,
        depthwise_initializer=depthwise_initializer,
        depthwise_regularizer=depthwise_regularizer,
        normalization_type=normalization_type,  # 'layernorm' (default, degree-0) or 'batchnorm' (BiasFreeBatchNorm, degree-1 at inference)
        name=name,
    )(x)
    if drop_path_rate and drop_path_rate > 0.0:
        y = StochasticDepth(drop_path_rate, name=f'{name}_drop_path')(y)
    return keras.layers.Add(name=f'{name}_residual')([residual, y])


def _make_supervision_activation(activation, name):
    """Build a serialization-safe activation layer for the functional deep-supervision head.

    A bare ``keras.layers.Activation(<layer instance>)`` does NOT round-trip through
    ``.keras`` in a functional graph (the Functional from_config cannot deserialize a
    layer-instance activation). A string activation, and a bare cloned activation layer,
    both round-trip. So: clone a layer-instance activation (fresh, uniquely-named) and
    apply it directly; wrap a string in ``keras.layers.Activation``.
    """
    # DECISION plan_2026-06-21_eb7fd829/D-006: functional-graph activation must be a string
    # (-> Activation wrapper) or a CLONED bare layer; never Activation(<live layer instance>)
    # (does not round-trip, F9). See decisions.md D-006.
    if isinstance(activation, keras.layers.Layer):
        cfg = keras.layers.serialize(activation)
        cfg = {**cfg, "config": {**cfg["config"], "name": name}}
        return keras.layers.deserialize(cfg)
    return keras.layers.Activation(activation, name=name)


# ---------------------------------------------------------------------
# Bias-free (use_bias=False) guardrails
# ---------------------------------------------------------------------

# DECISION plan-2026-08-14T092357-0e3d792d/D-012: this is an ALLOWLIST of positively
# homogeneous activation NAMES, and it is deliberately NARROW and deliberately
# INCOMPLETE as a homogeneity certificate. Do NOT convert it to a denylist of
# {'gelu', 'tanh', 'sigmoid', ...}: a denylist silently admits every activation
# nobody thought of, which is exactly the failure a bias-free denoiser cannot
# observe (its outputs stay finite and its tests stay green while f(a*x) != a*f(x)).
# Do NOT widen this set to admit a value some caller happens to pass -- widening a
# rule to accommodate what the same change discovered is a self-serving edit; the
# correct response is to escalate. Note what is NOT here and why:
#   * 'gelu' -- the shipped default of `block_activation`, `stem_activation` and
#     `supervision_activation`. Those three are DELIBERATELY EXEMPT from the guard
#     (decisions.md D-006 / plan invariant I-6): guarding them would make the
#     model's own default configuration raise, which is a breakage, not a guard.
#   * `downsample_pool_type='max'` -- NOT guarded anywhere. Max pooling is
#     non-linear but IS positively homogeneous (max(a*x) == a*max(x) for a > 0);
#     conflating "non-linear" with "non-homogeneous" would wrongly ban it.
# Consequence, stated so no reader mistakes it: passing these guards is NOT a
# homogeneity certificate. See decisions.md D-006 and D-012.
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
    """Validate the arguments that break bias-freeness / degree-1 homogeneity.

    Called from :func:`create_convunext` **only** when ``use_bias is False``. Under
    ``use_bias=True`` none of these arguments is a defect, so the whole function is
    inert on that arm.

    Three hard guards (raise :class:`ValueError`) and one soft guard (warn):

    - ``final_activation`` must name a positively homogeneous activation, i.e. be a
      member of :data:`POSITIVELY_HOMOGENEOUS_ACTIVATIONS`.
    - ``gabor_activation`` likewise, but **only when** ``use_gabor_stem`` is True.
      With the Gabor stem off the argument is inert -- it reaches no layer -- so
      raising on it would fire on a configuration that is perfectly homogeneous.
    - ``supervision_norm_center=True`` puts a trainable additive offset (``beta``)
      on the deep-supervision head LayerNorm, which is a bias by another name.
    - ``block_normalization='layernorm'`` WARNS and builds. Per-input LayerNorm is
      scale-INVARIANT (degree 0), not degree-1, so it does break homogeneity -- but
      it is the shipped default of both arms (plan invariant I-3) and the byte-identity
      tripwire in ``test_bfconvunext_denoiser.py`` pins it, so raising would take down
      every existing bias-free caller. Raise-vs-warn here is a CONTRACT, not a comment.

    Two rulings a later reader will be tempted to "fix", recorded here on purpose:

    - **A callable (non-string) activation cannot be statically checked.** Its
      homogeneity is a property of code this function cannot inspect. It therefore
      WARNS and never raises. Do not turn that into a raise (it would ban a legitimate
      homogeneous lambda) and do not turn it into silence (the caller then has no
      signal at all).
    - **``supervision_norm_center=True`` raises even when
      ``enable_deep_supervision=False``**, i.e. even when no supervision head is
      built and the argument reaches nothing. This is deliberate: the guard's
      predicate is a pure function of its arguments, and the caller stated a
      contradictory intent. Do NOT gate this clause on ``enable_deep_supervision``.

    :param final_activation: The builder's ``final_activation`` argument.
    :type final_activation: str or callable
    :param gabor_activation: The builder's ``gabor_activation`` argument.
    :type gabor_activation: str or None
    :param use_gabor_stem: Whether the frozen Gabor stem is built at all; scopes
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
        # Callable / layer-instance activation: not statically checkable.
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
        # DECISION plan_2026-07-01_8054f023/D-001: 'batchnorm' selects the variance-only
        # BiasFreeBatchNorm inside every ConvNeXt block. A FIXED-STAT norm (dividing by a
        # frozen running_var constant at inference, no mean, no beta) restores degree-1
        # homogeneity f(ax)=a*f(x); the default per-input LayerNorm divides by a per-sample
        # std that itself scales with the input, so it is scale-INVARIANT (degree-0). Do NOT
        # substitute stock keras.layers.BatchNormalization(center=False) or any RMS-family
        # norm here — both were empirically non-homogeneous (moving_mean subtraction /
        # per-input RMS); only BiasFreeBatchNorm is degree-1 at inference. Default 'layernorm'
        # is byte-identical to the prior hardcoded LayerNorm, and stays 'layernorm' for BOTH
        # the bias-on and bias-off arms (plan invariant I-3): only
        # `bfconvunext.create_convunext_variant` selects 'batchnorm'. See decisions.md D-001.
        block_normalization: str = "layernorm",
        stem_activation: Union[str, keras.layers.Layer] = 'gelu',
        drop_path_rate: float = 0.1,
        final_activation: Union[str, callable] = 'linear',
        # Scale-preserving (norm-preserving) init for the main-path structural convs
        # (stem, channel-adjusts, final, supervision). With the residual trunk these
        # convs + concatenations must NOT amplify variance — 'he_normal' (scale=2)
        # compounds it and the deep U-Net explodes at init. 'orthogonal' preserves
        # the activation norm and stays bias-free (a linear, homogeneous map).
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
    """Build a ConvUNext model as a Keras FUNCTIONAL graph.

    Single home for both ConvUNext arms. ``use_bias=True`` (the default) builds a
    bias-carrying network; ``use_bias=False`` builds the bias-free denoiser that
    ``bfconvunext.create_convunext_denoiser`` wraps, whose output scales with its
    input (if the input is scaled by alpha, the output is scaled by alpha).

    ConvUNext leverages existing implementations:

    - U-Net's encoder-decoder structure with skip connections
    - ConvNeXt V1/V2 blocks (``ConvNextV1Block`` / ``ConvNextV2Block``)
    - Deep supervision for better training

    **Architecture**::

        Encoder:          ConvNeXt blocks + downsample/skip junction, per level
        Bottleneck:       optional bias-free linear attention, then ConvNeXt blocks
        Decoder:          upsample + skip merge + ConvNeXt blocks, per level
        Deep supervision: additional outputs at intermediate decoder levels

    During training with deep supervision enabled the model emits multiple scales:
    output 0 is the final full-resolution output, outputs 1..N are the supervision
    outputs from shallowest to deepest.

    **Deliberate asymmetries under** ``use_bias=False`` **(decisions.md D-004/D-006).**
    ``use_bias=False`` means "no bias on the threaded convolutions", NOT "provably
    degree-1 homogeneous". Three documented exceptions survive on purpose:

    1. **The exempt activations.** ``block_activation``, ``stem_activation`` and
       ``supervision_activation`` all default to ``'gelu'``, which is NOT positively
       homogeneous. They are deliberately left unguarded, because guarding them would
       make the shipped default configuration raise. Pass ``'relu'`` /
       ``'leaky_relu'`` / a ``LeakyReLU`` instance for a homogeneous network.
    2. **Two hardcoded** ``use_bias=False`` **sites are NOT threaded from this
       argument**: ``SpatialLinearAttention``'s internal
       ``create_attention_layer('linear', ..., use_bias=False)`` (a bias would break
       the Miyasawa property the denoiser depends on — see
       ``plan_2026-07-11_bb4b38b5/D-001``) and the frozen Gabor bank
       (``trainable=False``; a frozen biased filter bank is meaningless). They stay
       bias-free even when ``use_bias=True``.
    3. **GRN's** ``beta`` **is not threaded.** ``GlobalResponseNormalization`` has a
       ``use_beta`` parameter that neither the stem nor the ConvNeXt V2 blocks pass,
       so a trainable additive ``beta`` exists in every V2 block and in the
       ``'global_response_norm'`` stem even under ``use_bias=False``. This is a
       KNOWN non-strictness (D-GAP-1), not an oversight: threading it would change
       the bias-off arm's parameter count, which is the regression instrument this
       merge is validated against.

    **Guardrails under** ``use_bias=False`` **(**:func:`_validate_bias_free_arguments`
    **, decisions.md D-006/D-012).** Three arguments RAISE ``ValueError`` on the
    bias-off arm and are completely inert on the bias-on arm:
    ``final_activation`` and — only when ``use_gabor_stem=True`` —
    ``gabor_activation`` must be in :data:`POSITIVELY_HOMOGENEOUS_ACTIVATIONS`
    (``None``, ``'linear'``, ``'relu'``, ``'leaky_relu'``); and
    ``supervision_norm_center=True`` is rejected outright. Two rulings that look
    like bugs and are not: a **callable** activation cannot be checked statically,
    so it WARNS and builds; and ``supervision_norm_center=True`` raises **even when**
    ``enable_deep_supervision=False``, keeping the guard a pure function of its
    arguments. Separately, ``block_normalization='layernorm'`` (the default on both
    arms) only WARNS — raising would break every existing bias-free caller. Given
    exception 1 above, passing these guards is NOT a homogeneity certificate.

    :param input_shape: Shape of input images ``(height, width, channels)``.
    :type input_shape: tuple of 3 ints
    :param use_bias: Whether the threaded convolutions (stem, Gabor projection,
        channel adjusts, ConvNeXt blocks, supervision heads, final projection)
        allocate a bias vector. Defaults to ``True``. Pass ``False`` for the
        bias-free / Miyasawa denoiser arm; read the "Deliberate asymmetries"
        section above before treating ``False`` as a homogeneity guarantee.
    :type use_bias: bool
    :param depth: Depth of the U-Net (number of downsampling levels). Must be >= 2.
        Defaults to 4.
    :type depth: int
    :param initial_filters: Number of filters at the first level. Defaults to 64.
    :type initial_filters: int
    :param filter_multiplier: Per-encoder-level channel-growth multiplier (``>= 1``).
        Channels at level ``i`` are ``int(round(initial_filters * filter_multiplier
        ** i))``. Defaults to ``2.0`` (doubles per level, byte-identical to the
        historical int ``2``).
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
    :param use_gabor_stem: If True prepend a frozen (non-learnable) Gabor depthwise
        convolution stem (always bias-free) followed by a 1x1 projection to
        ``initial_filters``, instead of the standard ConvUNextStem. Defaults to
        False. The Gabor stem contributes zero trainable parameters.
    :type use_gabor_stem: bool
    :param gabor_filters: Depth multiplier for the Gabor depthwise stem; the stem
        emits ``input_channels * gabor_filters`` channels which the mandatory 1x1
        projection reduces to ``initial_filters``. Only used when
        ``use_gabor_stem=True``. Defaults to 32.
    :type gabor_filters: int
    :param gabor_kernel_size: Kernel size of the Gabor depthwise stem. Defaults to 11.
    :type gabor_kernel_size: int or tuple of 2 ints
    :param gabor_activation: Optional activation on the frozen Gabor stem. ``None``
        (default) = linear passthrough. Under ``use_bias=False`` it MUST be
        positively homogeneous (relu, leaky_relu, linear) — gelu/elu/tanh/sigmoid/
        mish break degree-1 homogeneity. Only used when ``use_gabor_stem=True``.
    :type gabor_activation: str or None
    :param gabor_stem_projection: If True (default) the Gabor stem is followed by the
        mandatory 1x1 projection that reduces ``input_channels * gabor_filters``
        channels down to ``initial_filters``. If False the projection is DROPPED and
        the Gabor bank feeds the encoder directly — valid ONLY when
        ``input_channels * gabor_filters == initial_filters`` exactly (raises
        ``ValueError`` otherwise). Removing the projection leaves all cross-channel
        mixing to the first ConvNeXt block (the depthwise Gabor bank does none).
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
        skip. **Ignored when** ``use_laplacian_pyramid=False`` (the high band only
        exists under the pyramid split). Defaults to 0, which adds zero layers. Must
        be non-negative.
    :type high_freq_blocks: int
    :param bottleneck_attention_blocks: Number of bias-free LinearAttention blocks
        inserted at the bottleneck right after the channel-adjust and BEFORE the
        bottleneck ConvNeXt stack. Each block is a residual
        ``x + StochasticDepth(rate)(SpatialLinearAttention(x))`` with a local
        drop-path ramp (first block = 0.0). The attention is
        degree-1-homogeneous / Miyasawa-safe (hardcoded ``'linear'`` type,
        ``use_bias=False``, ``feature_map='relu'``). Defaults to 0, which adds zero
        layers. Must be non-negative. When > 0, the bottleneck filter count must be
        divisible by ``bottleneck_attention_heads``.
    :type bottleneck_attention_blocks: int
    :param bottleneck_attention_heads: Number of attention heads per bottleneck
        attention block. Only used when ``bottleneck_attention_blocks > 0``.
        Defaults to 8. Must be >= 1 when attention blocks are enabled.
    :type bottleneck_attention_heads: int
    :param zero_pad_channels: If True replace every per-level channel-adjust 1x1
        convolution with a parameter-free channel match. Channel INCREASES (encoder
        levels and the bottleneck) are done by zero-padding the channel axis; channel
        DECREASES (the post-upsample decoder path) are done by slicing the upsampled
        branch to ``current_filters`` and ADDING the skip connection (the literal
        slice-the-concat is degenerate — it would discard the entire upsampled
        branch). The substitution is bias-free and homogeneous, removing all
        channel-adjust conv parameters. Defaults to False.
    :type zero_pad_channels: bool
    :param extra_zero_output_channels: If True, at decoder level 0 append
        ``output_channels`` zero-initialized feature channels before that level's
        ConvNeXt blocks (which are widened to ``initial_filters + output_channels``),
        and replace the final learned 1x1 output projection with a parameter-free
        slice that keeps the last ``output_channels`` channels. The residual blocks
        learn to write the output into the zero tail. Bias-free / homogeneous;
        default OFF.
    :type extra_zero_output_channels: bool
    :param final_projection_groups: Number of groups for the final 1x1
        ``final_output`` projection (``Conv2D(output_channels, 1, groups=...)``).
        Default 1 = a standard dense 1x1 conv. When > 1 the projection becomes a
        GROUPED conv: input feature channels and output channels are split into
        ``final_projection_groups`` groups and each output group is computed only
        from its own input group. Setting it to ``output_channels`` gives one group
        per output (e.g. color) channel. Requires the group count to divide BOTH the
        projection's input channel count and ``output_channels`` (raises
        ``ValueError`` otherwise), and is incompatible with
        ``extra_zero_output_channels`` (which has no learned ``final_output`` conv to
        group).
    :type final_projection_groups: int
    :param downsample_pool_type: ``'max'``, ``'average'`` or ``'strided_conv'``.
        Downsample op for the non-Laplacian encoder junction. ``'max'`` (default) =
        MaxPooling2D, NON-LINEAR but positively homogeneous. ``'average'`` =
        AveragePooling2D, a LINEAR operator that keeps the encoder path linear for the
        Miyasawa/Tweedie residual-as-score interpretation. ``'strided_conv'`` = a
        LEARNED, channel-preserving ``Conv2D(kernel_size=2, strides=2)`` that threads
        ``use_bias``; with ``use_bias=False`` it is linear and degree-1 homogeneous,
        so it is legal on the bias-free arm and is deliberately NOT guarded. Ignored
        when ``use_laplacian_pyramid=True``. The two pooling ops are weightless, so
        switching between them does not affect weight transfer; ``'strided_conv'``
        ADDS parameters at every encoder junction.
    :type downsample_pool_type: str
    :param expose_bottleneck: If True expose the deepest-encoder bottleneck latent as
        an additional, TRAILING model output: ``[denoised, ...(supervision)...,
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
        ReLU (the ``'leaky_relu'`` string resolves to slope 0.2). A layer instance
        round-trips through ``.keras`` (handled by ``ConvNext*Block.get_config``).
    :type block_activation: str or keras.layers.Layer
    :param block_normalization: The pre-activation normalization used INSIDE every
        ConvNeXt block. One of:

        - ``'layernorm'`` (default): per-input ``LayerNormalization``
          (epsilon=1e-6, center=use_bias, scale=True). NOTE: per-input LayerNorm is
          scale-INVARIANT (degree-0), NOT scale-homogeneous.
        - ``'batchnorm'``: the variance-only ``BiasFreeBatchNorm``. At inference
          (``training=False``) it divides by a FROZEN running_var constant (no mean,
          no beta), which restores degree-1 homogeneity ``f(a*x) = a*f(x)``. Pairs
          best with a homogeneous activation such as LeakyReLU. Homogeneity is an
          inference-time property: during training the layer uses per-batch variance
          and is degree-0.

        Threaded to every encoder/bottleneck/decoder block. The stem normalization
        and the deep-supervision-head LayerNorm are NOT covered by this parameter.
    :type block_normalization: str
    :param stem_activation: Activation for the ``ConvUNextStem``; default ``'gelu'``.
        Only used when the standard stem is built, i.e. ``use_gabor_stem=False``.
    :type stem_activation: str or keras.layers.Layer
    :param drop_path_rate: Stochastic-depth drop probability (the maximum of the
        per-stack linear ramps). Defaults to 0.1.
    :type drop_path_rate: float
    :param final_activation: Activation of the final output projection. Defaults to
        ``'linear'``.
    :type final_activation: str or callable
    :param kernel_initializer: Initializer for the main-path structural convolutions.
        Defaults to ``'orthogonal'`` (norm-preserving; ``'he_normal'`` compounds
        variance through the residual trunk and explodes a deep U-Net at init).
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for those convolutions.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param depthwise_initializer: Applied to the depthwise conv kernel of every
        ConvNeXt block. Defaults to None, which reproduces the block's own hardcoded
        ``TruncatedNormal(mean=0.0, stddev=0.02)``. For an orthonormal depthwise init
        pass keras ``Orthogonal(gain=1.0)`` (a ``(K,K,C,1)`` depthwise kernel
        flattens to a single column, so "orthonormal" here means unit-norm). The repo
        ``OrthonormalInitializer`` / ``HeOrthonormalInitializer`` (2D-only) and
        ``OrthogonalHypersphereInitializer`` (norm blow-up) are UNSUPPORTED here.
    :type depthwise_initializer: str or keras.initializers.Initializer or None
    :param depthwise_regularizer: Applied to the depthwise conv kernel of every
        ConvNeXt block. Defaults to None, which reproduces the block's own behavior
        (a deepcopy of ``kernel_regularizer``).
    :type depthwise_regularizer: str or keras.regularizers.Regularizer or None
    :param dropout_rate: Float in ``[0, 1)``. Standard (element-wise) MLP dropout
        applied inside each ConvNeXt block's inverted bottleneck. This is NOT
        stochastic depth (see ``drop_path_rate``). Default ``0.0`` = OFF: no
        ``Dropout`` sublayer is added. ``spatial_dropout_rate`` stays ``0.0`` (not
        exposed).
    :type dropout_rate: float
    :param enable_deep_supervision: Whether to add deep-supervision outputs. Defaults
        to False.
    :type enable_deep_supervision: bool
    :param supervision_norm_scale: Whether the deep-supervision head LayerNorm has a
        learnable scale (gamma). Defaults to True.
    :type supervision_norm_scale: bool
    :param supervision_norm_center: Whether the deep-supervision head LayerNorm has a
        learnable center (beta/bias). Defaults to False, which keeps the head free of
        an additive offset; set True only if you accept a bias-like offset there.
    :type supervision_norm_center: bool
    :param supervision_activation: Activation for the deep-supervision heads; default
        ``'gelu'``. Only used when ``enable_deep_supervision=True``.
    :type supervision_activation: str or keras.layers.Layer
    :param include_top: Whether to build the final ``output_channels`` projection.
        Defaults to True. With ``include_top=False`` the model's primary output is the
        full-resolution DECODER FEATURE MAP, exposed through a zero-parameter
        ``Activation('linear', name='decoder_features')`` tap, and the ``final_output``
        projection is NOT constructed at all.

        **Divergence from the deleted** ``ConvUNextModel`` **(decisions.md D-013).**
        That subclass CONSTRUCTED its final projection in ``__init__`` regardless and
        merely skipped applying it, so its ``include_top=False`` variant still carried
        the head's weights and a checkpoint could be moved between the two settings. A
        FUNCTIONAL graph cannot reproduce that: ``keras.Model(inputs, outputs)`` keeps
        only the layers on a path to an output, so a constructed-but-unapplied layer is
        pruned, owns no weights and is not reachable via ``get_layer``. The weight
        compatibility contract is therefore GONE, not preserved --
        ``include_top=False`` yields a strictly smaller weight list, and
        ``set_weights`` between the two configurations raises.
    :type include_top: bool
    :param output_channels: Number of channels of the final projection and of every
        deep-supervision output. Defaults to ``None``, which means the INPUT channel
        count (``input_shape[-1]``) -- the denoiser/autoencoder contract every existing
        caller relies on. Set it explicitly for a non-reconstruction head (e.g. ``1``
        for a single-channel mask). Also controls the width of the zero tail appended
        by ``extra_zero_output_channels``. Inert when ``include_top=False`` and
        ``enable_deep_supervision=False``.
    :type output_channels: int or None
    :param model_name: Name of the returned model. Defaults to ``'convunext'``.
    :type model_name: str

    :return: A functional ``keras.Model``.

        - ``enable_deep_supervision=False``: a single output tensor.
        - ``enable_deep_supervision=True``: ``[final_output, supervision...]``.
        - ``expose_bottleneck=True``: the outputs list gains a TRAILING
          ``bottleneck`` output.

    :rtype: keras.Model
    :raises ValueError: If ``depth < 2``, ``initial_filters`` is non-positive,
        ``filter_multiplier < 1``, ``blocks_per_level`` is non-positive,
        ``convnext_version`` is not ``'v1'``/``'v2'``, ``downsample_pool_type`` is
        not ``'max'``/``'average'``, or one of the documented option-combination
        constraints is violated.
    :raises TypeError: If ``input_shape`` is not a tuple of 3 integers.

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

    # Input validation
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
        # final_projection_groups ONLY parameterizes the `final_output` conv, which
        # include_top=False does not build. Raise rather than silently ignore it --
        # a silently inert argument is this repo's recorded defect class.
        raise ValueError(
            "final_projection_groups is only meaningful with include_top=True (it "
            "groups the final_output projection, which include_top=False does not "
            f"build); got final_projection_groups={final_projection_groups}"
        )

    # DECISION plan-2026-08-14T092357-0e3d792d/D-012: the homogeneity guardrails fire
    # ONLY on the bias-off arm. Do NOT hoist this call out of the `if` -- under
    # use_bias=True a 'sigmoid' final activation and a centered supervision norm are
    # ordinary, supported configurations (`ConvUNextModel`'s own deleted trainers used
    # exactly those), and an unconditional validator would break the bias-ON arm to
    # protect an invariant the bias-ON arm never claimed. See decisions.md D-012.
    if use_bias is False:
        _validate_bias_free_arguments(
            final_activation=final_activation,
            gabor_activation=gabor_activation,
            use_gabor_stem=use_gabor_stem,
            supervision_norm_center=supervision_norm_center,
            block_normalization=block_normalization,
        )

    # Select ConvNeXt block type
    ConvNextBlock = ConvNextV2Block if convnext_version == 'v2' else ConvNextV1Block

    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_images')

    # DECISION plan_2026-06-19_ed071c02/D-001: default-OFF additive frozen Gabor stem.
    # Non-learnable depthwise Gabor bank + mandatory bias-free 1x1 projection (output
    # channels of a depthwise conv = in_ch * gabor_filters). Reuse the existing builder,
    # do not rebuild. With use_gabor_stem=False this is a no-op rename (stem_input=inputs).
    if use_gabor_stem:
        gabor = create_gabor_depthwise_conv2d(
            filters=gabor_filters,
            kernel_size=gabor_kernel_size,
            activation=gabor_activation,
            strides=1,
            padding='same',
            # HARDCODED bias-free, NOT threaded from `use_bias` (decisions.md D-004):
            # the bank is frozen (trainable=False), so a bias on it is meaningless.
            use_bias=False,
            trainable=False,
            name='gabor_stem',
        )(inputs)
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
                f"Frozen Gabor stem enabled: filters={gabor_filters}, "
                f"kernel_size={gabor_kernel_size} -> 1x1 projection to {initial_filters}"
            )
        else:
            # No-projection Gabor stem: the depthwise bank emits exactly
            # input_channels * gabor_filters channels and feeds the encoder directly.
            # This is only well-defined when that count equals initial_filters (the
            # level-0 channel-adjust is then a no-op); otherwise there is no bias-free
            # parameter-free way to reach initial_filters here, so fail loudly rather
            # than silently pad/slice.
            gabor_out_ch = input_shape[-1] * gabor_filters
            if gabor_out_ch != initial_filters:
                raise ValueError(
                    "gabor_stem_projection=False requires the Gabor bank to emit exactly "
                    f"initial_filters channels, but input_channels({input_shape[-1]}) * "
                    f"gabor_filters({gabor_filters}) = {gabor_out_ch} != "
                    f"initial_filters({initial_filters}). Choose gabor_filters and "
                    "initial_filters so they match exactly, or keep gabor_stem_projection=True."
                )
            stem_input = gabor
            logger.info(
                f"Frozen Gabor stem enabled (NO projection): filters={gabor_filters}, "
                f"kernel_size={gabor_kernel_size} -> {gabor_out_ch} channels feed the "
                f"encoder directly (== initial_filters)"
            )
    else:
        stem_input = inputs

    # Calculate filter sizes for each level
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

    # Storage for skip connections and deep supervision outputs
    skip_connections: List[keras.layers.Layer] = []
    deep_supervision_outputs: List[keras.layers.Layer] = []

    # =========================================================================
    # ENCODER PATH (Contracting)
    # =========================================================================

    x = stem_input
    logger.info(f"Building ConvUNext encoder path with {depth} levels using ConvNeXt {convnext_version.upper()}")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"Encoder level {level}: {current_filters} filters")

        # First level: initial feature extraction + channel setup. The dedicated
        # ConvUNextStem is only needed when there is NO Gabor stem. When
        # use_gabor_stem=True the frozen Gabor bank + its mandatory 1x1 projection
        # already performed initial feature extraction AND set the channel count to
        # initial_filters (== current_filters at level 0), so the ConvUNextStem is
        # redundant. In that case fall through to the channel-adjust branch, which is a
        # no-op when channels already match (they do, by construction) and otherwise
        # keeps the residual ConvNeXt add valid at current_filters.
        if level == 0 and not use_gabor_stem:
            # Use stem block for initial feature extraction
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
            # Channel adjustment if needed. Covers level>0 and the gabor-stem level-0
            # case (ensures x has current_filters channels so the residual ConvNeXt
            # blocks below add correctly).
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

        # ConvNeXt blocks at current resolution (residual + drop-path)
        for block_idx in range(blocks_per_level):
            # Progressive (linearly-scaled) drop-path rate across depth.
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

        # Skip connection + downsample for this level. Under the Laplacian pyramid
        # path this is ONE channel-preserving split (high -> skip, low -> next level);
        # otherwise the original raw-skip + MaxPooling2D. The last encoder level's
        # downsample is the bottleneck downsample (preserved name). The junction Layer
        # WRAPS the pooling/pyramid op, so the caller-visible name now belongs to the
        # wrapper and the inner op is named '<name>_pool' / '<name>_pyramid' (accepted
        # graph change C-1). The returned order is (skip, downsampled) on both paths --
        # do NOT swap it; both outputs are rank-4 and a shape check cannot see the swap.
        junction_name = (
            f'encoder_downsample_{level}' if level < depth - 1 else 'bottleneck_downsample'
        )
        skip, x = DownsampleAndSkip(
            use_laplacian_pyramid=use_laplacian_pyramid,
            laplacian_kernel_size=laplacian_kernel_size,
            pool_type=downsample_pool_type,
            # Only the 'strided_conv' branch reads these three; the pooling and
            # pyramid branches are weightless and ignore them.
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=junction_name,
        )(x)

        # DECISION plan_2026-07-10_be906be8/D-002: optionally process the Laplacian
        # high-frequency band with N ConvNeXt blocks before it becomes the
        # decoder skip. Gated on use_laplacian_pyramid (the high band only exists then);
        # high_freq_blocks=0 (default) adds ZERO layers -> byte-identical OFF path, so
        # existing `.keras` checkpoints (whose layer names are load-bearing) still load.
        # Do NOT drop the use_laplacian_pyramid gate or the >0 gate: without the pyramid
        # there is no high band and this would rename/insert layers into the raw-skip path.
        # This SUPERSEDES plan_2026-07-06_b17c1f83/D-001, which pinned every high-freq block
        # to drop_path_rate=0.0 (no StochasticDepth at all). The high-freq stack now carries
        # a LOCAL linear drop-path ramp `drop_path_rate * hf_idx / high_freq_blocks` that
        # restarts at 0.0 per encoder level, mirroring the encoder/decoder "first block = 0.0"
        # convention. hf_idx=0 -> 0.0 => still NO StochasticDepth layer for the first block
        # (the round-trip-determinism concern is preserved for it); hf_idx>=1 gain a weightless
        # StochasticDepth sublayer. The `high_freq_blocks > 0` gate guarantees the denominator
        # is nonzero. StochasticDepth is inference-identity, so this only changes training-time
        # regularization; the OFF-by-default path (high_freq_blocks=0) is untouched.
        if high_freq_blocks > 0 and use_laplacian_pyramid:
            for hf_idx in range(high_freq_blocks):
                # Local linearly-scaled drop-path ramp (restarts at 0.0 per level's HF stack).
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

    # =========================================================================
    # BOTTLENECK
    # =========================================================================

    bottleneck_filters = filter_sizes[depth]
    logger.info(f"Building ConvUNext bottleneck with {bottleneck_filters} filters")

    # Channel adjustment for bottleneck
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

    # Optional bias-free attention blocks at the bottleneck (before the ConvNeXt stack).
    # DECISION plan_2026-07-11_bb4b38b5/D-002: gated on bottleneck_attention_blocks > 0 so the
    # default (0) adds ZERO layers -> byte-identical OFF path (existing .keras checkpoints
    # depend on this). Local drop-path ramp `drop_path_rate * attn_idx / bottleneck_attention_blocks`
    # restarts at 0.0 (first block gets no StochasticDepth), mirroring the ConvNeXt loop below.
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

    # Bottleneck ConvNeXt blocks (residual + drop-path)
    # DECISION plan_2026-07-10_be906be8/D-001: the bottleneck now uses a LOCAL linear
    # drop-path ramp that restarts at 0.0, mirroring the encoder ramp shape and the
    # decoder's "first block = 0.0" convention (see decoder loop below). This SUPERSEDES
    # plan_2026-06-20_0433c2f2/D-003, which pinned every bottleneck block to the flat
    # (unscaled) max drop_path_rate. The ramp `drop_path_rate * block_idx / blocks_per_level`
    # stays strictly in [0, drop_path_rate) for every block -> it can NEVER exceed
    # drop_path_rate (the exact concern D-003 raised about continuing the encoder's GLOBAL
    # index into the bottleneck). block_idx=0 -> 0.0 => _apply_residual_convnext_block adds
    # NO StochasticDepth layer for the first block. blocks_per_level >= 1 is guaranteed by
    # the validator above, so the denominator is never zero. StochasticDepth is
    # inference-identity, so existing trained checkpoints load and infer unchanged (block_0
    # only drops a weightless SD sublayer). Do NOT revert to a flat rate — in particular do
    # NOT restore the deleted ConvUNextModel's constant `drop_path_rate` bottleneck.
    for block_idx in range(blocks_per_level):
        # Local linearly-scaled drop-path ramp (restarts at 0.0 in the bottleneck stack).
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

    # Optional bottleneck tap: a zero-parameter linear marker on the deepest latent so it
    # can be exposed as an additional output and extracted post-hoc. Placed on the main
    # path (the decoder continues from it), so the named layer is retained even in a
    # single-output save. No-op when expose_bottleneck is False.
    if expose_bottleneck:
        x = keras.layers.Activation('linear', name='bottleneck')(x)
        bottleneck_output = x

    # =========================================================================
    # DECODER PATH (Expanding) with Deep Supervision
    # =========================================================================

    logger.info(f"Building ConvUNext decoder path with {depth} levels")

    for level in range(depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        logger.info(f"Decoder level {level}: {current_filters} filters")

        # Upsampling
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation='bilinear',
            name=f'decoder_upsample_{level}'
        )(x)

        # Get corresponding skip connection
        skip = skip_connections[level]

        # Ensure spatial dimensions match for concatenation
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            target_height, target_width = skip.shape[1], skip.shape[2]
            x = keras.layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation='bilinear',
                name=f'decoder_resize_{level}'
            )(x)

        # Merge skip connection.
        # DECISION plan_2026-06-26_90d8cbe6/D-003: under zero_pad_channels the decoder cannot
        # zero-pad (it must REDUCE channels). The literal "slice the [skip, up] concat to C" is
        # degenerate (concat order is [skip(C), up(2C)] so the first C channels are skip ONLY,
        # discarding the entire upsampled branch). Instead slice the UPSAMPLED tensor (2C) down
        # to C and ADD the C-channel skip — parameter-free, keeps both branches, bias-free,
        # homogeneous. OFF arm below is the verbatim original Concatenate + 1x1 Conv2D.
        if zero_pad_channels:
            x = keras.layers.Add(name=f'decoder_level_{level}_match_add')(
                [skip, MatchChannels(current_filters, name=f'decoder_level_{level}_match_channels')(x)]
            )
        else:
            x = keras.layers.Concatenate(
                axis=-1,
                name=f'decoder_concat_{level}'
            )([skip, x])

            # Channel adjustment after concatenation
            if x.shape[-1] != current_filters:
                x = keras.layers.Conv2D(
                    filters=current_filters,
                    kernel_size=1,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name=f'decoder_level_{level}_channel_adjust'
                )(x)

        # Optionally grow output channels at the finest decoder stage (level 0).
        # DECISION plan_2026-06-26_0ec1a304/D-001: append `output_channels` zero
        # channels here (before the level-0 blocks) and widen those blocks so their
        # residuals learn to write the image into the zero tail; the final projection
        # is then replaced by a tail-slice (see final-output block below). Level 0 only;
        # OFF path is byte-identical. Compose-safe with zero_pad_channels (pad happens
        # AFTER the skip-merge Add).
        block_filters = current_filters
        if extra_zero_output_channels and level == 0:
            block_filters = current_filters + output_channels
            x = MatchChannels(block_filters, name='extra_zero_output_pad')(x)

        # ConvNeXt blocks after merging (residual + drop-path)
        for block_idx in range(blocks_per_level):
            # The FIRST block at every decoder level carries NO stochastic depth
            # (drop_path == 0 => _apply_residual_convnext_block adds no StochasticDepth
            # layer); the remaining blocks keep the progressive (linearly-scaled) rate
            # across depth. Decoder-only — the encoder schedule is unchanged.
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

        # =====================================================================
        # DEEP SUPERVISION OUTPUT (if enabled and not the final level)
        # =====================================================================

        if enable_deep_supervision and level > 0:
            # Create supervision output at current scale
            supervision_branch = keras.layers.Conv2D(
                filters=current_filters // 2,
                kernel_size=1,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'supervision_intermediate_level_{level}'
            )(x)

            # Bias-free-by-default LayerNorm at the supervision head (replaces GRN, whose
            # trainable beta is a bias-like additive offset). scale/center read from args;
            # center=False keeps the head bias-free (no additive offset) but NOT
            # scale-homogeneous: per-input LayerNorm divides by a per-sample std that scales
            # with the input, so it is scale-INVARIANT (degree-0), NOT degree-1 f(ax)=a*f(x).
            # This deep-supervision-head LayerNorm is NOT covered by the block_normalization
            # param (out of scope; documented) — only the encoder/bottleneck/decoder block
            # norms are swappable to BiasFreeBatchNorm. `center` is read from
            # `supervision_norm_center`, NOT from `use_bias`: it is an explicit, separately
            # guarded knob (plan invariant I-6).
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

    # =========================================================================
    # FINAL OUTPUT LAYER (Primary inference output)
    # =========================================================================

    # Final projection to output channels.
    if extra_zero_output_channels and final_projection_groups != 1:
        raise ValueError(
            "final_projection_groups>1 is incompatible with extra_zero_output_channels: the "
            "latter drops the learned final_output Conv2D in favor of a parameter-free tail "
            "slice, so there is no projection to group. Use one or the other."
        )
    # DECISION plan-2026-08-14T092357-0e3d792d/D-013: include_top=False does NOT
    # construct the final projection. The deleted `ConvUNextModel` built its head in
    # __init__ and merely skipped APPLYING it, so its headless variant still carried
    # the head's weights (an explicit weight-compat contract). That contract is NOT
    # reproducible here and is deliberately not faked: `keras.Model(inputs, outputs)`
    # prunes every layer that is not on a path to an output, so a
    # constructed-but-unapplied Conv2D would own no weights, would not appear in
    # `model.layers`, and `get_layer('final_output')` would still raise -- MEASURED,
    # not assumed (decisions.md D-013 records the probe). Do NOT "fix" this by calling
    # the projection and dropping its tensor (identical pruning), nor by appending it
    # as a second output (that changes the output signature and silently re-adds the
    # head this argument exists to remove). The honest contract is: include_top=False
    # returns the decoder features and a STRICTLY SMALLER weight list; `set_weights`
    # between the two settings raises.
    if not include_top:
        final_output = keras.layers.Activation(
            'linear', name='decoder_features'
        )(x)
        logger.info(
            f"include_top=False: no final projection; primary output is the decoder "
            f"feature map with {final_output.shape[-1]} channels"
        )
    elif extra_zero_output_channels:
        # DECISION plan_2026-06-26_0ec1a304/D-001: keep ONLY the zero-grown tail
        # channels (last `output_channels`) as the output, dropping the learned 1x1
        # projection. Parameter-free, bias-free, homogeneous. final_activation is
        # applied so the contract (e.g. 'linear') matches the OFF path.
        final_output = MatchChannels(
            output_channels, slice_side='tail', name='final_output_tail_slice'
        )(x)
        if final_activation is not None and final_activation != 'linear':
            final_output = keras.layers.Activation(
                final_activation, name='final_output_activation'
            )(final_output)
    else:
        # Grouped final projection (default groups=1 == standard dense 1x1). groups>1 splits
        # input + output channels into disjoint groups; groups==output_channels gives one
        # group per output (color) channel.
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

    # =========================================================================
    # MODEL CREATION
    # =========================================================================

    if enable_deep_supervision and deep_supervision_outputs:
        # Return multiple outputs: [final_output, supervision_outputs...]
        # Order supervision outputs from shallowest to deepest (by resolution)
        ordered_supervision_outputs = list(reversed(deep_supervision_outputs))
        all_outputs = [final_output] + ordered_supervision_outputs
        if expose_bottleneck:
            all_outputs = all_outputs + [bottleneck_output]

        logger.info(f"Created ConvUNext deep supervision model with {len(all_outputs)} outputs:")
        logger.info(f"  - Final output (index 0): {final_output.shape}")
        for i, sup_output in enumerate(ordered_supervision_outputs):
            level = i + 1
            logger.info(f"  - Supervision output {i + 1} (index {i + 1}, level {level}): {sup_output.shape}")

        # Create model with multiple outputs
        model = keras.Model(
            inputs=inputs,
            outputs=all_outputs,
            name=model_name
        )

    else:
        # Single output model (standard U-Net or inference model)
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

    :param variant: One of ``'tiny'``, ``'small'``, ``'base'``, ``'large'``,
        ``'xlarge'``.
    :type variant: str
    :param input_shape: Shape of input images ``(height, width, channels)``.
    :type input_shape: tuple of 3 ints
    :param enable_deep_supervision: Whether to enable deep-supervision outputs.
        Defaults to False (matching ``create_convunext``).
    :type enable_deep_supervision: bool
    :param kwargs: Additional keyword arguments forwarded to ``create_convunext``,
        overriding the variant defaults (including ``use_bias``).
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

    # Override config with any provided kwargs
    config.update(kwargs)

    # Set model name if not provided
    if 'model_name' not in config:
        ds_suffix = '_ds' if enable_deep_supervision else ''
        convnext_version = config.get('convnext_version', 'v2')
        config['model_name'] = f'convunext_{variant}_{convnext_version}{ds_suffix}'

    # Set deep supervision
    config['enable_deep_supervision'] = enable_deep_supervision

    logger.info(f"Creating ConvUNext variant '{variant}': {description}")
    logger.info(f"ConvNeXt version: {config.get('convnext_version', 'v2').upper()}")
    logger.info(f"Deep supervision: {'enabled' if enable_deep_supervision else 'disabled'}")

    return create_convunext(
        input_shape=input_shape,
        **config
    )


# ---------------------------------------------------------------------
# Utility Functions for Deep Supervision
# ---------------------------------------------------------------------

# Re-exported so `models/convunext` has ONE inference-model helper and it is the
# canonical, functional-graph one. The deleted `ConvUNextModel` carried a bespoke
# copy that rebuilt a subclassed model from its config and round-tripped weights
# through a temp file; the shared util slices the functional graph instead
# (`training_model.output` / `.input`), which is what a functional ConvUNext needs.
from dl_techniques.utils.deep_supervision import (  # noqa: E402
    get_model_output_info,
    create_inference_model_from_training_model,
)
