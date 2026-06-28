"""
ConvUNext: Modern Bias-Free U-Net with ConvNeXt-Inspired Architecture

Implements a ConvUNext architecture with deep supervision leveraging existing
ConvNeXt V1/V2 blocks while maintaining bias-free properties for better
generalization across different noise levels and improved scaling invariance.

ConvUNext combines the best of U-Net and ConvNeXt architectures:
- U-Net's encoder-decoder structure with skip connections
- ConvNeXt's modern architectural innovations via existing implementations
- Bias-free design for scaling invariance (use_bias=False)

Key modern improvements over standard U-Net:
- Reuses existing ConvNeXt V1/V2 blocks with bias-free configuration
- Depthwise separable convolutions for efficiency
- Inverted bottleneck design (channel expansion then contraction)
- Global Response Normalization (GRN) for V2 blocks
- Configurable activation (default GELU at the factory level; the bfunet trainer defaults to LeakyReLU(0.1)) for stem, blocks, and deep-supervision
- Larger kernel sizes (7x7) for better receptive fields
- Layer scaling for training stability
- Optional stochastic depth for regularization

The architecture maintains the bias-free principle: if input is scaled by α,
output is also scaled by α, enabling better generalization across noise levels.

Deep supervision provides several benefits:
- Better gradient flow to deeper layers during training
- Multi-scale feature learning and supervision
- More stable training for very deep networks
- Curriculum learning capabilities through weight scheduling

The model outputs multiple scales during training:
- Output 0: Final inference output (highest resolution, primary output)
- Output 1-N: Intermediate supervision outputs at progressively lower resolutions

Optional Laplacian-pyramid downsample/skip path (``use_laplacian_pyramid``, OFF by default):
    When enabled, every encoder down<->skip junction stops using ``MaxPooling2D`` + a raw
    full-resolution skip and instead applies a single ``LaplacianPyramidLevel`` split:

        low, high = split(x)            # low = blur-then-subsample(x); high = x - upsample(low)

    The coarse, anti-aliased ``low`` band descends the encoder; the high-frequency residual
    ``high`` band becomes the skip. The two bands are exactly complementary
    (``merge(low, high) == x``), so the split is lossless *taken together*.

    The reason for it is NOT just lossless downsampling -- it is that **no single path then
    carries all the information needed for reconstruction**. The skip holds only the high
    band and the descending/bottleneck path holds only the low band, so neither is a
    sufficient statistic; the decoder is forced to FUSE both to rebuild the signal. This
    removes the classic U-Net shortcut where a full-resolution skip carries the whole image,
    letting the network learn a near-identity copy and leaving the encoder->bottleneck->decoder
    pathway lazy and underused. By partitioning the information into complementary bands, every
    path is made necessary and the full hierarchy has to participate in reconstruction. For a
    denoiser this doubly matters: the trivial "copy the noisy input, do nothing" solution that
    hides in a full-resolution skip is gone once that skip only holds the high-frequency residual.

    Secondary benefit -- an inductive bias matched to denoising: white Gaussian noise is flat
    across frequency while natural-image signal concentrates in low frequencies, so per band the
    SNR differs sharply (high bands are noise-dominated, the low band is signal-rich). Splitting
    at every scale gives the network the subband structure of classical optimal denoising
    (wavelet-shrinkage / per-band Wiener), with the high-band skips carrying exactly where
    shrinkage must act and the edge/detail the decoder re-injects to avoid over-smoothing.

    Crucially this costs nothing on the theory side: ``LaplacianPyramidLevel`` is built only from
    linear ops (bias-free Gaussian blur -> blur-pool -> bilinear upsample -> subtraction), so it
    is homogeneous of degree 1 with zero additive offset. The bias-free / scaling-invariance
    property (and the Miyasawa/Tweedie residual-as-score interpretation it enables) is preserved
    exactly; the net simply becomes a learned multiscale, band-wise score estimator.

Based on ConvNeXt innovations from "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
and "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
(Woo et al., CVPR 2023) applied to bias-free U-Net architecture. The Laplacian-pyramid
split follows Burt & Adelson, "The Laplacian Pyramid as a Compact Image Code" (1983).
"""

import keras
from typing import Optional, Union, Tuple, List, Dict, Any


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.initializers import create_gabor_depthwise_conv2d
from dl_techniques.layers.laplacian_filter import LaplacianPyramidLevel
from dl_techniques.layers.match_channels import MatchChannels

# ---------------------------------------------------------------------
# ConvUNext Bias-Free Building Blocks (Simple Stem)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextStem(keras.layers.Layer):
    """
    ConvUNext stem block for initial feature extraction using bias-free design.

    Simple stem that uses a single large kernel convolution followed by
    Global Response Normalization and a configurable activation (default GELU),
    while keeping channel count conservative to avoid OOM issues.

    Args:
        filters: Integer, number of output filters.
        kernel_size: Integer or tuple, size of convolution kernel. Defaults to 7.
        kernel_initializer: String or Initializer, weight initializer.
        kernel_regularizer: String or Regularizer, weight regularizer.
        **kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        activation: Union[str, keras.layers.Layer] = 'gelu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation_name = activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Sublayers initialized in build()
        self.conv = None
        self.grn = None
        self.activation_layer = None

    def build(self, input_shape):
        """Build the stem layers."""
        # Large kernel convolution (bias-free)
        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,  # Bias-free for scaling invariance
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stem_conv'
        )

        # Global Response Normalization (consistent with ConvNeXt V2)
        self.grn = GlobalResponseNormalization(name='stem_grn')

        # Explicitly build sublayers so weights materialize on .keras reload
        # (lazy auto-build drops their state during deserialization).
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.grn.build(conv_output_shape)

        # GRN is shape-preserving, so the activation input shape == conv_output_shape.
        self.activation_layer = keras.layers.Activation(
            self.activation_name, name='stem_activation'
        )
        self.activation_layer.build(conv_output_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.conv(inputs)
        x = self.grn(x)
        x = self.activation_layer(x)
        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        """Get layer configuration."""
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
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Deserialize, reviving a layer-instance activation from its dict form."""
        config = dict(config)
        if isinstance(config.get('activation'), dict):
            config['activation'] = keras.layers.deserialize(config['activation'])
        # kernel_initializer/kernel_regularizer dicts are passed straight to __init__,
        # where keras.*.get(...) accepts a serialized dict (Keras 3).
        return cls(**config)

# ---------------------------------------------------------------------
# ConvUNext Model Variant Configurations
# ---------------------------------------------------------------------

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
) -> keras.KerasTensor:
    """Apply a ConvNeXt block as a RESIDUAL branch with stochastic depth.

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
    # DECISION plan_2026-06-21_eb7fd829/D-002: block activation is threaded via this single
    # choke-point (mirrors the kernel_regularizer / depthwise_* precedent) so one factory arg
    # reaches every encoder/bottleneck/decoder block at once. Factory default stays 'gelu' so
    # non-bfunet callers (convnext, convnext_patch_vae) are byte-identical. NOTE (iter-2,
    # D-005/D-006 superseded the original iter-1 scope): the stem (ConvUNextStem, D-005) and the
    # deep-supervision head (_make_supervision_activation, D-006) are now ALSO configurable via
    # the factory's stem_activation / supervision_activation params (each default 'gelu'). See
    # decisions.md D-002/D-005/D-006.
    y = block_cls(
        kernel_size=kernel_size,
        filters=filters,
        activation=activation,
        use_bias=False,            # Bias-free for scaling invariance
        dropout_rate=0.0,          # regularization comes from StochasticDepth below
        spatial_dropout_rate=0.0,
        gamma_initial_value=1e-4,  # LayerScale init (floored at GAMMA_MIN_VALUE=1e-6, can't die)
        kernel_regularizer=kernel_regularizer,
        depthwise_initializer=depthwise_initializer,
        depthwise_regularizer=depthwise_regularizer,
        name=name,
    )(x)
    if drop_path_rate and drop_path_rate > 0.0:
        y = StochasticDepth(drop_path_rate, name=f'{name}_drop_path')(y)
    return keras.layers.Add(name=f'{name}_residual')([residual, y])


# DECISION plan_2026-06-19_c90809b5/D-001: ONE helper folds the two downsample sites
# (inter-level pools + the standalone bottleneck pool) into a uniform per-level call so
# the OFF/ON swap logic lives in one place. Do NOT special-case the bottleneck pool
# separately (duplicates the swap at two sites -> drift risk) and do NOT author a new
# pyramid layer (reuse-only: LaplacianPyramidLevel is already bias-free + registered).
# OFF path MUST reproduce the exact prior ops/names (MaxPooling2D named `downsample_name`,
# raw pre-downsample skip) — existing `.keras` checkpoints depend on byte-identical names.
# See decisions.md D-001.
def _downsample_and_skip(
        x: keras.KerasTensor,
        use_laplacian_pyramid: bool,
        laplacian_kernel_size: Tuple[int, int],
        downsample_name: str,
        pyramid_name: str,
        pool_type: str = "max",
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Produce ``(skip, downsampled)`` for one encoder junction.

    OFF path (default, byte-identical to the original architecture): the skip is
    the pre-downsample tensor and downsampling is ``MaxPooling2D(2, 2)`` named
    ``downsample_name``. With ``pool_type='average'`` the downsample uses
    ``AveragePooling2D(2, 2)`` instead -- a LINEAR (and bias-free / homogeneous)
    operator, so the encoder path stays linear for the Miyasawa/Tweedie
    residual-as-score interpretation (MaxPooling is non-linear). Pooling layers are
    weightless, so the swap does not affect checkpoint weight transfer.

    ON path: a channel-preserving, bias-free ``LaplacianPyramidLevel`` split. The
    full-resolution high-frequency band becomes the skip; the half-resolution low
    band continues down the encoder. Bias-free and homogeneous by construction
    (fixed blur + average pool + bilinear upsample, zero learnable bias). The
    pyramid already pools linearly, so ``pool_type`` does not apply here.
    """
    if use_laplacian_pyramid:
        low, high = LaplacianPyramidLevel(
            blur_kernel_size=laplacian_kernel_size,
            name=pyramid_name,
        )(x)
        return high, low
    skip = x
    pool_layer = (
        keras.layers.AveragePooling2D if pool_type == "average"
        else keras.layers.MaxPooling2D
    )
    downsampled = pool_layer(
        pool_size=(2, 2),
        name=downsample_name,
    )(x)
    return skip, downsampled


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
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_convunext_denoiser(
        input_shape: Tuple[int, int, int],
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: int = 2,
        blocks_per_level: int = 2,
        convnext_version: str = 'v2',
        stem_kernel_size: Union[int, Tuple[int, int]] = 7,
        use_gabor_stem: bool = False,
        gabor_filters: int = 32,
        gabor_kernel_size: Union[int, Tuple[int, int]] = 7,
        gabor_stem_projection: bool = True,
        use_laplacian_pyramid: bool = False,
        laplacian_kernel_size: Tuple[int, int] = (5, 5),
        zero_pad_channels: bool = False,
        extra_zero_output_channels: bool = False,
        final_projection_groups: int = 1,
        downsample_pool_type: str = "max",
        expose_bottleneck: bool = False,
        block_kernel_size: Union[int, Tuple[int, int]] = 7,
        block_activation: Union[str, keras.layers.Layer] = 'gelu',
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
        enable_deep_supervision: bool = False,
        supervision_norm_scale: bool = True,
        supervision_norm_center: bool = False,
        supervision_activation: Union[str, keras.layers.Layer] = 'gelu',
        model_name: str = 'convunext'
) -> keras.Model:
    """
    Create a ConvUNext model using existing ConvNeXt V1/V2 blocks with bias-free configuration.

    This function creates a complete ConvUNext architecture using existing ConvNeXt blocks
    with bias-free design (`use_bias=False`) and deep supervision capabilities. The model
    exhibits scaling-invariant properties: if the input is scaled by α, the output is also
    scaled by α.

    ConvUNext leverages existing implementations:
    - U-Net's encoder-decoder structure with skip connections
    - ConvNeXt V1/V2 blocks with bias-free configuration
    - Deep supervision for better training

    Key features:
    - Reuses existing ConvNeXt V1/V2 block implementations
    - Bias-free design via use_bias=False parameter
    - Depthwise separable convolutions for efficiency
    - Global Response Normalization (V2) or LayerNorm (V1)
    - Configurable block activation (default GELU; the bfunet trainer defaults to LeakyReLU(0.1))
    - Layer scaling for training stability
    - Optional stochastic depth for regularization
    - Larger kernels (7x7) for better receptive fields

    During training with deep supervision enabled, the model outputs multiple scales:
    - Output 0: Final inference output (full resolution)
    - Output 1: Second-to-last decoder level output
    - Output N: Deepest supervision level output

    Architecture:
    - Encoder: ConvNeXt blocks + downsampling at each level
    - Bottleneck: ConvNeXt blocks at the lowest resolution
    - Decoder: Upsampling + skip connections + ConvNeXt blocks
    - Deep Supervision: Additional outputs at intermediate decoder levels

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        depth: Integer, depth of the U-Net (number of downsampling levels). Defaults to 4.
        initial_filters: Integer, number of filters in the first level. Defaults to 64.
        filter_multiplier: Integer, multiplier for filters at each level. Defaults to 2.
        blocks_per_level: Integer, number of blocks per level. Defaults to 2.
        convnext_version: String, 'v1' or 'v2' to choose ConvNeXt version. Defaults to 'v2'.
        stem_kernel_size: Integer or tuple, size of stem convolution kernels. Defaults to 7.
        use_gabor_stem: Boolean, if True prepend a frozen (non-learnable) Gabor depthwise
            convolution stem (bias-free) followed by a bias-free 1x1 projection to
            `initial_filters`, before the standard ConvUNext stem. Defaults to False
            (byte-identical to the original architecture). The Gabor stem contributes
            zero trainable parameters.
        gabor_filters: Integer, depth multiplier for the Gabor depthwise stem; the stem
            emits `input_channels * gabor_filters` channels which the mandatory 1x1
            projection reduces to `initial_filters`. Only used when use_gabor_stem=True.
            Defaults to 32.
        gabor_kernel_size: Integer or tuple, kernel size of the Gabor depthwise stem.
            Only used when use_gabor_stem=True. Defaults to 7.
        gabor_stem_projection: Boolean, if True (default) the Gabor stem is followed by the
            mandatory bias-free 1x1 projection that reduces `input_channels * gabor_filters`
            channels down to `initial_filters`. If False the projection is DROPPED and the
            Gabor bank feeds the encoder directly — valid ONLY when
            `input_channels * gabor_filters == initial_filters` exactly (raises ValueError
            otherwise). Removing the projection keeps the stem bias-free/homogeneous but
            leaves all cross-channel mixing to the first ConvNeXt block (the depthwise Gabor
            bank does none). Only used when use_gabor_stem=True; default True is
            byte-identical to the original architecture.
        use_laplacian_pyramid: Boolean, if True replace each encoder downsample/skip
            junction with a bias-free `LaplacianPyramidLevel` split: the channel-preserving
            full-resolution high-frequency band becomes the skip connection and the
            half-resolution low-frequency band continues down the encoder. Defaults to False
            (byte-identical to the original MaxPooling2D architecture, including layer names).
            Contributes zero trainable parameters (the blur kernel is fixed).
        laplacian_kernel_size: Tuple of two ints, Gaussian blur kernel size for the
            Laplacian pyramid split. Only used when use_laplacian_pyramid=True. Defaults to (5, 5).
        zero_pad_channels: Boolean, if True replace every per-level channel-adjust 1x1
            convolution with a parameter-free channel match. Channel INCREASES (encoder
            levels and the bottleneck) are done by zero-padding the channel axis; channel
            DECREASES (the post-upsample decoder path) are done by slicing the upsampled
            branch to `current_filters` and ADDING the skip connection (the literal
            slice-the-concat is degenerate — it would discard the entire upsampled branch).
            The substitution is bias-free and homogeneous, removing all channel-adjust conv
            parameters. Defaults to False, which is byte-identical to the original
            learned-projection architecture (same layer names, same Conv2D ops, same outputs).
        extra_zero_output_channels: Boolean, if True, at decoder level 0 append
            `output_channels` zero-initialized feature channels before that level's ConvNeXt
            blocks (which are widened to `initial_filters + output_channels`), and replace the
            final learned 1x1 output projection with a parameter-free slice that keeps the last
            `output_channels` channels. The residual blocks learn to write the output into the
            zero tail. Bias-free / homogeneous; default OFF (byte-identical).
        final_projection_groups: Integer, number of groups for the final 1x1 `final_output`
            projection (`Conv2D(output_channels, 1, groups=...)`). Default 1 = a standard
            dense 1x1 conv (byte-identical to the original). When >1 the projection becomes a
            GROUPED conv: input feature channels and output channels are split into
            `final_projection_groups` groups and each output group is computed only from its
            own input group. Setting it to `output_channels` gives one group per output (e.g.
            color) channel — each output channel reads a DISJOINT `initial_filters /
            output_channels` slice of features. Requires both `initial_filters` and
            `output_channels` to be divisible by the group count (raises ValueError
            otherwise), and is incompatible with `extra_zero_output_channels` (which has no
            learned `final_output` conv to group). Stays bias-free / homogeneous (no bias, no
            centering) and round-trips through `.keras`.
        downsample_pool_type: 'max' or 'average'. Pooling op for the non-Laplacian encoder
            downsample. 'max' (default) = MaxPooling2D, byte-identical to the original
            architecture but NON-LINEAR. 'average' = AveragePooling2D, a LINEAR (bias-free,
            homogeneous) operator that keeps the encoder path linear for the Miyasawa/Tweedie
            residual-as-score interpretation. Ignored when use_laplacian_pyramid=True (the
            pyramid already pools linearly). Pooling layers are weightless, so this does not
            affect weight transfer. Defaults to 'max'.
        expose_bottleneck: Boolean, if True expose the deepest-encoder bottleneck latent
            as an additional, trailing model output. The model's call then returns
            `[denoised, ..., bottleneck]` (bottleneck LAST), where `bottleneck` has spatial
            `H/2**depth, W/2**depth` and `initial_filters * filter_multiplier**depth` channels.
            A zero-parameter linear `Activation('linear', name='bottleneck')` tap is inserted
            after the bottleneck blocks (bias-free, on the denoiser path). Defaults to False
            (byte-identical to the original single-output architecture). Useful for secondary /
            multi-task heads and debugging.
        block_kernel_size: Integer or tuple, size of block kernels. Defaults to 7.
        block_activation: String or keras Layer, activation applied inside every ConvNeXt
            block's inverted-bottleneck MLP. Defaults to 'gelu'. Pass a
            `keras.layers.LeakyReLU(negative_slope=0.1)` instance for slope-0.1 leaky ReLU
            (the 'leaky_relu' string resolves to slope 0.2). A layer instance round-trips
            through .keras serialization (handled by ConvNext*Block.get_config).
        stem_activation: String or keras Layer, activation for the ConvUNextStem; default
            'gelu'; only used when the standard stem is built, i.e. use_gabor_stem=False.
        drop_path_rate: Float, stochastic depth drop probability. Defaults to 0.1.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'orthogonal'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        depthwise_initializer: Optional String or Initializer, applied to the depthwise
            conv kernel of every ConvNeXt block. Defaults to None, which reproduces the
            current hardcoded behavior (TruncatedNormal(mean=0.0, stddev=0.02)). For an
            orthonormal depthwise init pass keras `Orthogonal(gain=1.0)` (unit-norm: a
            `(K,K,C,1)` depthwise kernel flattens to a single column, so "orthonormal"
            here means unit-norm). The repo `OrthonormalInitializer`/`HeOrthonormalInitializer`
            (2D-only) and `OrthogonalHypersphereInitializer` (norm blow-up) are UNSUPPORTED
            for the depthwise conv. Defaults to None.
        depthwise_regularizer: Optional String or Regularizer, applied to the depthwise
            conv kernel of every ConvNeXt block. Defaults to None, which reproduces the
            current behavior (a deepcopy of `kernel_regularizer`). Defaults to None.
        enable_deep_supervision: Boolean, whether to add deep supervision outputs. Defaults to False.
        supervision_norm_scale: Boolean, whether the deep-supervision head LayerNorm has a
            learnable scale (gamma). Defaults to True.
        supervision_norm_center: Boolean, whether the deep-supervision head LayerNorm has a
            learnable center (beta/bias). Defaults to False to keep the head bias-free
            (homogeneous), consistent with the rest of the model; set True only if you accept
            a bias-like additive offset at the supervision heads.
        supervision_activation: activation for the deep-supervision heads; default 'gelu';
            pass a keras.layers.LeakyReLU(0.1) instance for slope-0.1. Only used when
            enable_deep_supervision=True.
        model_name: String, name for the model. Defaults to 'convunext'.

    Returns:
        keras.Model: ConvUNext model ready for training.
                    - If deep_supervision=False: Single output tensor
                    - If deep_supervision=True: List of output tensors [final_output, intermediate_outputs...]
                    - If expose_bottleneck=True: the outputs list gains a trailing `bottleneck`
                      output (LAST), i.e. [final_output, ...(supervision if DS)..., bottleneck].

    Raises:
        ValueError: If depth is less than 3, initial_filters is non-positive,
                   filter_multiplier is less than 1, blocks_per_level is non-positive,
                   or convnext_version is not 'v1' or 'v2'.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create ConvUNext with ConvNeXt V2 blocks and deep supervision
        >>> model = create_convunext_denoiser(
        ...     input_shape=(256, 256, 3),
        ...     depth=4,
        ...     initial_filters=64,
        ...     convnext_version='v2',
        ...     enable_deep_supervision=True
        ... )
        >>>
        >>> # Create inference-only model with V1 blocks
        >>> inference_model = create_convunext_denoiser(
        ...     input_shape=(None, None, 3),  # Flexible spatial dimensions
        ...     depth=4,
        ...     initial_filters=64,
        ...     convnext_version='v1',
        ...     enable_deep_supervision=False
        ... )
    """

    # Input validation
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise TypeError("input_shape must be a tuple of 3 integers (height, width, channels)")

    if depth < 3:
        raise ValueError(f"depth must be at least 3, got {depth}")

    if initial_filters <= 0:
        raise ValueError(f"initial_filters must be positive, got {initial_filters}")

    if filter_multiplier < 1:
        raise ValueError(f"filter_multiplier must be at least 1, got {filter_multiplier}")

    if blocks_per_level <= 0:
        raise ValueError(f"blocks_per_level must be positive, got {blocks_per_level}")

    if convnext_version not in ['v1', 'v2']:
        raise ValueError(f"convnext_version must be 'v1' or 'v2', got {convnext_version}")

    if downsample_pool_type not in ['max', 'average']:
        raise ValueError(
            f"downsample_pool_type must be 'max' or 'average', got {downsample_pool_type}"
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
            strides=1,
            padding='same',
            use_bias=False,
            trainable=False,
            name='gabor_stem',
        )(inputs)
        if gabor_stem_projection:
            stem_input = keras.layers.Conv2D(
                filters=initial_filters,
                kernel_size=1,
                use_bias=False,  # Bias-free projection
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
    filter_sizes = [initial_filters * (filter_multiplier ** i) for i in range(depth + 1)]

    if use_laplacian_pyramid:
        logger.info(
            f"Laplacian pyramid downsample enabled: kernel_size={laplacian_kernel_size}, "
            f"split levels={depth} (high-band skips, low-band downsample; bias-free)"
        )
    else:
        logger.info(
            f"Encoder downsample pooling: {downsample_pool_type} "
            f"({'AveragePooling2D — linear, Miyasawa-clean' if downsample_pool_type == 'average' else 'MaxPooling2D — non-linear'})"
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
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'encoder_level_{level}_stem'
            )(x)
        else:
            # Channel adjustment if needed (bias-free). Covers level>0 and the
            # gabor-stem level-0 case (ensures x has current_filters channels so the
            # residual ConvNeXt blocks below add correctly).
            if x.shape[-1] != current_filters:
                if zero_pad_channels:
                    x = MatchChannels(current_filters, name=f'encoder_level_{level}_match_channels')(x)
                else:
                    x = keras.layers.Conv2D(
                        filters=current_filters,
                        kernel_size=1,
                        use_bias=False,  # Bias-free
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        name=f'encoder_level_{level}_channel_adjust'
                    )(x)

        # ConvNeXt blocks at current resolution (bias-free, residual + drop-path)
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
            )

        # Skip connection + downsample for this level. Under the Laplacian pyramid
        # path this is ONE channel-preserving split (high -> skip, low -> next level);
        # otherwise the original raw-skip + MaxPooling2D. The last encoder level's
        # downsample is the bottleneck downsample (preserved name for checkpoint compat).
        downsample_name = (
            f'encoder_downsample_{level}' if level < depth - 1 else 'bottleneck_downsample'
        )
        skip, x = _downsample_and_skip(
            x,
            use_laplacian_pyramid,
            laplacian_kernel_size,
            downsample_name=downsample_name,
            pyramid_name=f'encoder_pyramid_{level}',
            pool_type=downsample_pool_type,
        )
        skip_connections.append(skip)

    # =========================================================================
    # BOTTLENECK
    # =========================================================================

    bottleneck_filters = filter_sizes[depth]
    logger.info(f"Building ConvUNext bottleneck with {bottleneck_filters} filters")

    # Channel adjustment for bottleneck (bias-free)
    if x.shape[-1] != bottleneck_filters:
        if zero_pad_channels:
            x = MatchChannels(bottleneck_filters, name='bottleneck_match_channels')(x)
        else:
            x = keras.layers.Conv2D(
                filters=bottleneck_filters,
                kernel_size=1,
                use_bias=False,  # Bias-free
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='bottleneck_channel_adjust'
            )(x)

    # Bottleneck ConvNeXt blocks (bias-free, residual + drop-path)
    # DECISION plan_2026-06-20_0433c2f2/D-003: the bottleneck is the deepest point, so it
    # intentionally uses the maximum (unscaled) drop_path_rate, distinct from the
    # progressive linear ramp applied at the encoder/decoder blocks. Not a bug (a true
    # continuation of the ramp here would exceed drop_path_rate). No behaviour change.
    for block_idx in range(blocks_per_level):
        x = _apply_residual_convnext_block(
            x, ConvNextBlock, bottleneck_filters, block_kernel_size,
            drop_path_rate, kernel_regularizer,
            name=f'bottleneck_convnext_{convnext_version}_block_{block_idx}',
            activation=block_activation,
            depthwise_initializer=depthwise_initializer,
            depthwise_regularizer=depthwise_regularizer,
        )

    # Optional bottleneck tap: a zero-parameter linear (bias-free) marker on the deepest
    # latent so it can be exposed as an additional output and extracted post-hoc. Placed
    # on the denoiser path (the decoder continues from it), so the named layer is retained
    # even in a single-output save. No-op when expose_bottleneck is False.
    if expose_bottleneck:
        x = keras.layers.Activation('linear', name='bottleneck')(x)
        bottleneck_output = x

    # =========================================================================
    # DECODER PATH (Expanding) with Deep Supervision
    # =========================================================================

    logger.info(f"Building ConvUNext decoder path with {depth} levels")
    output_channels = input_shape[-1]

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

            # Channel adjustment after concatenation (bias-free)
            if x.shape[-1] != current_filters:
                x = keras.layers.Conv2D(
                    filters=current_filters,
                    kernel_size=1,
                    use_bias=False,  # Bias-free
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

        # ConvNeXt blocks after merging (bias-free, residual + drop-path)
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
            )

        # =====================================================================
        # DEEP SUPERVISION OUTPUT (if enabled and not the final level)
        # =====================================================================

        if enable_deep_supervision and level > 0:
            # Create supervision output at current scale (bias-free)
            supervision_branch = keras.layers.Conv2D(
                filters=current_filters // 2,
                kernel_size=1,
                use_bias=False,  # Bias-free
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'supervision_intermediate_level_{level}'
            )(x)

            # Bias-free-by-default LayerNorm at the supervision head (replaces GRN, whose
            # trainable beta is a bias-like additive offset). scale/center read from args;
            # center=False keeps the head bias-free (homogeneous), matching the model contract.
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
                use_bias=False,  # Bias-free
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

    # Final projection to output channels (bias-free).
    if extra_zero_output_channels and final_projection_groups != 1:
        raise ValueError(
            "final_projection_groups>1 is incompatible with extra_zero_output_channels: the "
            "latter drops the learned final_output Conv2D in favor of a parameter-free tail "
            "slice, so there is no projection to group. Use one or the other."
        )
    if extra_zero_output_channels:
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
        # group per output (color) channel. Bias-free (use_bias=False) regardless of groups.
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
            use_bias=False,  # Bias-free
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
        enable_deep_supervision: bool = True,
        **kwargs
) -> keras.Model:
    """
    Create a ConvUNext model with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        enable_deep_supervision: Boolean, whether to enable deep supervision outputs.
        **kwargs: Additional keyword arguments to override default parameters.

    Returns:
        keras.Model: ConvUNext model with the specified variant configuration.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        >>> # Standard usage with ConvNeXt V2 blocks and deep supervision
        >>> model = create_convunext_variant('base', (256, 256, 3), enable_deep_supervision=True)
        >>> model.summary()
        >>>
        >>> # Inference model with ConvNeXt V1 blocks
        >>> inference_model = create_convunext_variant('base', (None, None, 3),
        ...                                                   enable_deep_supervision=False,
        ...                                                   convnext_version='v1')
        >>>
        >>> # Custom parameters
        >>> model = create_convunext_variant('large', (224, 224, 1),
        ...                                         enable_deep_supervision=True,
        ...                                         convnext_version='v2',
        ...                                         drop_path_rate=0.3)
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

    return create_convunext_denoiser(
        input_shape=input_shape,
        **config
    )

# ---------------------------------------------------------------------
# Utility Functions for Deep Supervision
# ---------------------------------------------------------------------

from dl_techniques.utils.deep_supervision import (
    get_model_output_info,
    create_inference_model_from_training_model,
)

# ---------------------------------------------------------------------
