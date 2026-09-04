"""A U-Net denoiser built by `create_bfunet_denoiser` with every additive
constant removed, with optional deep supervision.

Removing every bias and normalization shift makes the network positively
homogeneous of degree one: scaling the input scales the output by the same
factor, `f(a*x) = a*f(x)`. A denoiser with that property generalizes across
noise levels instead of memorizing the range it trained on, and by
Miyasawa's relation its residual is proportional to the score of the noisy
image distribution. The property only holds if nothing in the graph adds a
constant, so `block_normalization` chooses between the variance-only
`BiasFreeBatchNorm`, which preserves it, and `layernorm`, which does not.
Structurally this is a standard U-Net, with an optional frozen Gabor stem
and an optional Laplacian-pyramid skip split, both off by default.

The builders are functional, returning `keras.Model(inputs, outputs)` with
no subclass, since converting them would invalidate existing checkpoints.
`pretrained=True` raises `NotImplementedError`; load a checkpoint written by
`src/train/bfunet/` with `pretrained="/path/to/file.keras"` or
`keras.models.load_model` instead.

References:
    - Mohan et al., 2020. Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks. ICLR 2020.
      (https://arxiv.org/abs/1906.05478)
    - Ronneberger et al., 2015. U-Net: Convolutional Networks for Biomedical
      Image Segmentation. (https://arxiv.org/abs/1505.04597)
    - Miyasawa, 1961. An empirical Bayes estimator of the mean of a normal
      population. Bull. Inst. Internat. Statist. 38.
    - Lee et al., 2015. Deeply-Supervised Nets. AISTATS 2015.
      (https://arxiv.org/abs/1409.5185)
    - Burt and Adelson, 1983. The Laplacian Pyramid as a Compact Image Code.
      IEEE Trans. Communications 31(4).
"""


import os
import keras
from typing import Optional, Union, Tuple, List, Dict, Any


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.conv_blocks.bias_free_conv2d import (
    BiasFreeConv2D,
    BiasFreeResidualBlock,
    resolve_denoiser_normalization,
)
from dl_techniques.initializers import create_gabor_depthwise_conv2d
from dl_techniques.layers.conv_blocks.match_channels import MatchChannels

from dl_techniques.layers.pooling.downsample_and_skip import DownsampleAndSkip

# ---------------------------------------------------------------------
# Model Variant Configurations
# ---------------------------------------------------------------------

BFUNET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'depth': 3,
        'initial_filters': 16,
        'blocks_per_level': 2,
        'description': 'Tiny BF-UNet (depth=3) for quick experiments.'
    },
    'small': {
        'depth': 3,
        'initial_filters': 24,
        'blocks_per_level': 2,
        'description': 'Small BF-UNet (depth=3) with minimal capacity.'
    },
    'base': {
        'depth': 4,
        'initial_filters': 32,
        'blocks_per_level': 3,
        'description': 'Base BF-UNet (depth=4) with standard configuration.'
    },
    'large': {
        'depth': 4,
        'initial_filters': 48,
        'blocks_per_level': 4,
        'description': 'Large BF-UNet (depth=4) with high capacity.'
    },
    'xlarge': {
        'depth': 5,
        'initial_filters': 64,
        'blocks_per_level': 5,
        'description': 'Extra-Large BF-UNet (depth=5) for maximum performance.'
    }
}

# ---------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------

# DECISION plan-2026-08-24T174647-07af0659/D-002: keep layer names, creation
# order, and the caller's initializer/regularizer objects exactly as in the
# original inline builder — a rename or re-resolved initializer silently
# breaks checkpoint loading or RNG draws with no error. See decisions.md.

def _validate_bfunet_args(
        input_shape: Tuple[int, int, int],
        depth: int,
        initial_filters: int,
        filter_multiplier: int,
        blocks_per_level: int,
        high_freq_blocks: int,
        block_normalization: str,
        downsample_pool_type: str,
        final_projection_groups: int,
        dropout_rate: float,
) -> str:
    """Validate the builder arguments and resolve the block normalization name.

    Arguments mirror the identically-named parameters of `create_bfunet_denoiser`.
    Checks run in the order written; when two arguments are invalid at once,
    the order determines which error message the caller sees.

    :return: The resolved normalization name every block and deep-supervision head must receive.
    :rtype: str
    :raises TypeError: If input_shape is not a tuple of 3 integers.
    :raises ValueError: If depth < 2, initial_filters <= 0, filter_multiplier < 1,
        blocks_per_level <= 0, high_freq_blocks < 0, block_normalization is not one
        of the three accepted names, downsample_pool_type is not 'max'/'average',
        final_projection_groups < 1, or dropout_rate is outside [0.0, 1.0).
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

    # DECISION plan_2026-07-04_58ac8e73/D-002: every ConvUNeXt-parity kwarg
    # defaults to a byte-identical no-op, so the off path reproduces the
    # original plain U-Net exactly. Do not change default behavior. See decisions.md.
    if block_normalization not in ('batchnorm', 'layernorm', 'bias_free_batchnorm'):
        raise ValueError(
            "block_normalization must be 'batchnorm', 'layernorm' or 'bias_free_batchnorm', "
            f"got {block_normalization}")
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-020: resolve 'batchnorm' to
    # BiasFreeBatchNorm rather than passing it straight through — stock
    # BatchNormalization's moving_mean breaks homogeneity. See decisions.md.
    block_norm = resolve_denoiser_normalization(block_normalization)
    if downsample_pool_type not in ('max', 'average'):
        raise ValueError(
            f"downsample_pool_type must be 'max' or 'average', got {downsample_pool_type}")
    if final_projection_groups < 1:
        raise ValueError(f"final_projection_groups must be >= 1, got {final_projection_groups}")
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

    return block_norm

def _build_gabor_stem(
        inputs: keras.KerasTensor,
        input_shape: Tuple[int, int, int],
        use_gabor_stem: bool,
        gabor_filters: int,
        gabor_kernel_size: Union[int, Tuple[int, int]],
        gabor_activation: Optional[str],
        gabor_stem_projection: bool,
        initial_filters: int,
        kernel_initializer: Union[str, keras.initializers.Initializer],
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
) -> keras.KerasTensor:
    """Build the optional frozen Gabor stem in front of the encoder.

    Arguments mirror the identically-named parameters of `create_bfunet_denoiser`.
    `kernel_initializer` / `kernel_regularizer` are forwarded as objects, never
    re-resolved.

    :return: The tensor the encoder path starts from — `inputs` unchanged when
        `use_gabor_stem=False`, a true no-op that adds zero layers.
    :rtype: keras.KerasTensor
    :raises ValueError: If `gabor_stem_projection=False` and
        `input_channels * gabor_filters != initial_filters`.
    """
    if use_gabor_stem:
        gabor = create_gabor_depthwise_conv2d(
            filters_per_channel=gabor_filters,
            kernel_size=gabor_kernel_size,
            activation=gabor_activation,
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
        else:
            gabor_out_ch = input_shape[-1] * gabor_filters
            if gabor_out_ch != initial_filters:
                raise ValueError(
                    "gabor_stem_projection=False requires input_channels * gabor_filters == "
                    f"initial_filters, but {input_shape[-1]} * {gabor_filters} = {gabor_out_ch} "
                    f"!= initial_filters({initial_filters}). Match them, or keep projection on."
                )
            stem_input = gabor
        logger.info(f"Frozen Gabor stem enabled: filters={gabor_filters}, "
                    f"kernel_size={gabor_kernel_size}, projection={gabor_stem_projection}")
    else:
        stem_input = inputs

    return stem_input

def _build_encoder_path(
        stem_input: keras.KerasTensor,
        filter_sizes: List[int],
        depth: int,
        blocks_per_level: int,
        kernel_size: Union[int, Tuple[int, int]],
        initial_kernel_size: Union[int, Tuple[int, int]],
        activation: Union[str, callable],
        kernel_initializer: Union[str, keras.initializers.Initializer],
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
        use_residual_blocks: bool,
        block_norm: str,
        dropout_rate: float,
        use_laplacian_pyramid: bool,
        laplacian_kernel_size: Tuple[int, int],
        downsample_pool_type: str,
        high_freq_blocks: int,
) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
    """Build the contracting path: `depth` levels of blocks, each followed by a junction.

    Arguments mirror the identically-named parameters of `create_bfunet_denoiser`,
    except `block_norm`, the resolved value from `_validate_bfunet_args`.

    :return: Tuple of the downsampled tensor entering the bottleneck and the
        list of skip connections, ordered shallowest first.
    :rtype: Tuple[keras.KerasTensor, List[keras.KerasTensor]]
    """
    skip_connections: List[keras.layers.Layer] = []

    x = stem_input
    logger.info(f"Building encoder path with {depth} levels")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"Encoder level {level}: {current_filters} filters")

        # Convolution blocks at current resolution
        for block_idx in range(blocks_per_level):
            if level == 0 and block_idx == 0:
                # first level
                x = BiasFreeConv2D(
                    filters=current_filters,
                    kernel_size=initial_kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    use_batch_norm=True,
                    normalization_type=block_norm,
                    dropout_rate=dropout_rate,
                    name=f'encoder_level_{level}_conv_{block_idx}'
                )(x)
            else:
                if use_residual_blocks:
                    x = BiasFreeResidualBlock(
                        filters=current_filters,
                        kernel_size=kernel_size,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        normalization_type=block_norm,
                        dropout_rate=dropout_rate,
                        name=f'encoder_level_{level}_residual_block_{block_idx}'
                    )(x)
                else:
                    x = BiasFreeConv2D(
                        filters=current_filters,
                        kernel_size=kernel_size,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        use_batch_norm=True,
                        normalization_type=block_norm,
                        dropout_rate=dropout_rate,
                        name=f'encoder_level_{level}_conv_{block_idx}'
                    )(x)

        # Skip connection + downsample. ALL levels route through the DownsampleAndSkip
        # Layer so the pool-type / Laplacian-pyramid swap lives in one place. The last
        # level's pool feeds the bottleneck and keeps the original name
        # 'bottleneck_downsample'. The junction Layer WRAPS the pooling/pyramid op, so
        # the caller-visible name now belongs to the wrapper and the inner op is named
        # '<name>_pool' / '<name>_pyramid' (accepted graph change C-2). The returned
        # order is (skip, downsampled) on both paths -- do NOT swap it.
        junction_name = (
            f'encoder_downsample_{level}' if level < depth - 1
            else 'bottleneck_downsample'
        )
        skip, x = DownsampleAndSkip(
            use_laplacian_pyramid=use_laplacian_pyramid,
            laplacian_kernel_size=laplacian_kernel_size,
            pool_type=downsample_pool_type,
            # bfunet is unconditionally bias-free, and `DownsampleAndSkip.use_bias`
            # (added for its learned 'strided_conv' branch) is READ BY THE TRAINERS'
            # compliance sweep -- `train/bfunet/train_unet_denoiser.py:198` walks
            # `_flatten_layers()` for any layer whose `use_bias` is truthy. Leaving the
            # constructor default (True) makes every junction report as a bias offender
            # even though the pooling branch has no weights at all. Keep this explicit.
            use_bias=False,
            name=junction_name,
        )(x)

        # DECISION plan_2026-07-06_b17c1f83/D-001: gate on both
        # use_laplacian_pyramid and high_freq_blocks > 0 — dropping either gate
        # inserts layers into the raw-skip path when no high band exists. See decisions.md.
        if high_freq_blocks > 0 and use_laplacian_pyramid:
            for hf_idx in range(high_freq_blocks):
                skip = BiasFreeResidualBlock(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    normalization_type=block_norm,
                    dropout_rate=dropout_rate,
                    name=f'skip_highfreq_block_{level}_{hf_idx}'
                )(skip)

        skip_connections.append(skip)

    return x, skip_connections

def _build_bottleneck(
        x: keras.KerasTensor,
        filter_sizes: List[int],
        depth: int,
        blocks_per_level: int,
        kernel_size: Union[int, Tuple[int, int]],
        activation: Union[str, callable],
        kernel_initializer: Union[str, keras.initializers.Initializer],
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
        use_residual_blocks: bool,
        block_norm: str,
        dropout_rate: float,
        expose_bottleneck: bool,
) -> Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]:
    """Build the bottleneck blocks at the lowest resolution.

    Arguments mirror the identically-named parameters of `create_bfunet_denoiser`,
    except `block_norm`, the resolved value from `_validate_bfunet_args`.

    :return: Tuple of the bottleneck tensor and the optional exposed
        bottleneck tap, `None` unless `expose_bottleneck=True`.
    :rtype: Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]
    """
    bottleneck_filters = filter_sizes[depth]
    logger.info(f"Building bottleneck with {bottleneck_filters} filters")

    # NOTE: the downsample INTO the bottleneck is produced by the last encoder-loop
    # iteration above (named 'bottleneck_downsample'), so there is no separate pool here.

    # Bottleneck convolution blocks
    for block_idx in range(blocks_per_level):
        if use_residual_blocks:
            x = BiasFreeResidualBlock(
                filters=bottleneck_filters,
                kernel_size=kernel_size,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                normalization_type=block_norm,
                dropout_rate=dropout_rate,
                name=f'bottleneck_residual_block_{block_idx}'
            )(x)
        else:
            x = BiasFreeConv2D(
                filters=bottleneck_filters,
                kernel_size=kernel_size,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=True,
                normalization_type=block_norm,
                dropout_rate=dropout_rate,
                name=f'bottleneck_conv_{block_idx}'
            )(x)

    # Optional bottleneck tap: a zero-parameter linear (bias-free) marker on the deepest
    # latent so it can be exposed as an additional output. No-op when expose_bottleneck=False.
    bottleneck_output = None
    if expose_bottleneck:
        x = keras.layers.Activation('linear', name='bottleneck')(x)
        bottleneck_output = x

    return x, bottleneck_output

def _build_decoder_path(
        x: keras.KerasTensor,
        skip_connections: List[keras.KerasTensor],
        filter_sizes: List[int],
        depth: int,
        output_channels: int,
        initial_filters: int,
        blocks_per_level: int,
        kernel_size: Union[int, Tuple[int, int]],
        activation: Union[str, callable],
        final_activation: Union[str, callable],
        kernel_initializer: Union[str, keras.initializers.Initializer],
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
        use_residual_blocks: bool,
        block_norm: str,
        dropout_rate: float,
        zero_pad_channels: bool,
        enable_deep_supervision: bool,
) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
    """Build the expanding path: upsample, merge the skip, run blocks, and
    optionally tap a deep-supervision head at every level above 0.

    Arguments mirror the identically-named parameters of `create_bfunet_denoiser`,
    except `block_norm`, the resolved value from `_validate_bfunet_args`.

    :return: Tuple of the full-resolution decoder tensor and the
        deep-supervision outputs in deep-to-shallow order; the caller reverses them.
    :rtype: Tuple[keras.KerasTensor, List[keras.KerasTensor]]
    """
    deep_supervision_outputs: List[keras.layers.Layer] = []

    logger.info(f"Building decoder path with {depth} levels")

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
        # Handle potential size mismatches due to pooling/upsampling
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            # Crop or pad to match dimensions
            target_height, target_width = skip.shape[1], skip.shape[2]
            x = keras.layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation='bilinear',
                name=f'decoder_resize_{level}'
            )(x)

        # Merge skip connection. OFF (default): Concatenate (byte-identical). ON
        # (zero_pad_channels): parameter-free channel match — slice the upsampled branch
        # to current_filters and ADD the (current_filters) skip. Bias-free either way.
        if zero_pad_channels:
            x = keras.layers.Add(name=f'decoder_add_{level}')(
                [skip, MatchChannels(current_filters, name=f'decoder_match_{level}')(x)]
            )
        else:
            x = keras.layers.Concatenate(
                axis=-1,
                name=f'decoder_concat_{level}'
            )([skip, x])

        # Convolution blocks after merging
        for block_idx in range(blocks_per_level):
            if use_residual_blocks:
                x = BiasFreeResidualBlock(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    normalization_type=block_norm,
                    dropout_rate=dropout_rate,
                    name=f'decoder_level_{level}_residual_block_{block_idx}'
                )(x)
            else:
                x = BiasFreeConv2D(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    use_batch_norm=True,
                    normalization_type=block_norm,
                    dropout_rate=dropout_rate,
                    name=f'decoder_level_{level}_conv_{block_idx}'
                )(x)

        # =====================================================================
        # DEEP SUPERVISION OUTPUT (if enabled and not the final level)
        # =====================================================================

        if enable_deep_supervision and level > 0:
            # Create supervision output at current scale from a branch
            # The supervision head is IN SCOPE for `block_normalization` (unlike
            # ConvUNext's, which documents its head LayerNorm as deliberately out of
            # scope). It feeds gradient straight into the decoder, so leaving it on stock
            # BN while every block used BiasFreeBatchNorm made that gradient
            # scale-dependent. `dropout_rate` is forwarded for the same reason.
            supervision_branch = BiasFreeConv2D(
                filters=initial_filters,
                kernel_size=3,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=True,
                normalization_type=block_norm,
                dropout_rate=dropout_rate,
                name=f'supervision_intermediate_level_{level}'
            )(x)

            supervision_output = BiasFreeConv2D(
                filters=output_channels,
                kernel_size=1,
                activation=final_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=False,
                name=f'supervision_output_level_{level}'
            )(supervision_branch)

            deep_supervision_outputs.append(supervision_output)

            logger.info(f"Added deep supervision output at level {level} "
                       f"with shape: {supervision_output.shape}")

    return x, deep_supervision_outputs

def _build_final_projection(
        x: keras.KerasTensor,
        output_channels: int,
        final_activation: Union[str, callable],
        kernel_initializer: Union[str, keras.initializers.Initializer],
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
        final_projection_groups: int,
) -> keras.KerasTensor:
    """Build the primary inference output: a bias-free 1x1 projection to `output_channels`.

    Arguments mirror the identically-named parameters of `create_bfunet_denoiser`.

    :return: The final output, named `'final_output'` on both branches.
    :rtype: keras.KerasTensor
    :raises ValueError: If `final_projection_groups > 1` does not divide both
        the incoming channel count and `output_channels`.
    """
    # Final convolution to output channels (no batch norm, custom activation).
    # OFF (final_projection_groups==1): the original bias-free 1x1 (byte-identical).
    # ON (>1): a grouped bias-free Conv2D so each output group reads a disjoint feature
    # group (groups==output_channels -> one group per color channel).
    if final_projection_groups == 1:
        final_output = BiasFreeConv2D(
            filters=output_channels,
            kernel_size=1,
            activation=final_activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_batch_norm=False,
            name='final_output'
        )(x)
    else:
        in_ch = x.shape[-1]
        if in_ch % final_projection_groups != 0 or output_channels % final_projection_groups != 0:
            raise ValueError(
                f"final_projection_groups={final_projection_groups} must divide BOTH the "
                f"final-projection input channels ({in_ch}) and output_channels "
                f"({output_channels}). Pick a group count dividing both, or use 1 (ungrouped)."
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

    return final_output

# ---------------------------------------------------------------------
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_bfunet_denoiser(
        input_shape: Tuple[int, int, int],
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: int = 2,
        blocks_per_level: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        initial_kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: Union[str, callable] = 'leaky_relu',
        final_activation: Union[str, callable] = 'linear',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_residual_blocks: bool = True,
        enable_deep_supervision: bool = False,
        # --- ConvUNeXt-parity features (all default to a byte-identical no-op) ---
        use_gabor_stem: bool = False,
        gabor_filters: int = 32,
        gabor_kernel_size: Union[int, Tuple[int, int]] = 11,
        gabor_activation: Optional[str] = None,
        gabor_stem_projection: bool = True,
        use_laplacian_pyramid: bool = False,
        high_freq_blocks: int = 0,
        laplacian_kernel_size: Tuple[int, int] = (5, 5),
        zero_pad_channels: bool = False,
        downsample_pool_type: str = "max",
        expose_bottleneck: bool = False,
        block_normalization: str = "batchnorm",
        final_projection_groups: int = 1,
        dropout_rate: float = 0.0,
        model_name: str = 'bias_free_unet'
) -> keras.Model:
    """Build a bias-free U-Net denoiser as a functional `keras.Model`.

    The model is homogeneous of degree 1: scaling the input by a scalar
    scales the output by the same scalar.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
           |
        Gabor stem              (optional)
           |
        encoder: conv blocks + downsample  x depth
           |
        bottleneck: conv blocks
           |
        decoder: upsample + skip concat + conv blocks  x depth
           |         |
           |     deep-supervision head    (per decoder level, optional)
           |
        final projection        -> [B, H, W, C]

        outputs: single tensor (enable_deep_supervision=False)
              or [final, level_1, ..., level_N] (enable_deep_supervision=True)

    :param input_shape: Shape of input images, `(height, width, channels)`.
    :type input_shape: Tuple[int, int, int]
    :param depth: Number of downsampling levels. Defaults to 4.
    :type depth: int
    :param initial_filters: Number of filters in the first level. Defaults to 64.
    :type initial_filters: int
    :param filter_multiplier: Filter multiplier per level. Defaults to 2.
    :type filter_multiplier: int
    :param blocks_per_level: Number of conv blocks per level. Defaults to 2.
    :type blocks_per_level: int
    :param kernel_size: Size of the convolutional kernels. Defaults to 3.
    :param initial_kernel_size: Size of the first convolutional kernel. Defaults to 5.
    :param activation: Activation function. Defaults to `'leaky_relu'`.
    :param final_activation: Final activation function. Defaults to `'linear'`.
    :param kernel_initializer: Weight initializer. Defaults to `'he_normal'`.
    :param kernel_regularizer: Optional weight regularizer.
    :param use_residual_blocks: Whether to use residual blocks. Defaults to True.
    :type use_residual_blocks: bool
    :param enable_deep_supervision: Whether to add deep-supervision outputs. Defaults to False.
    :type enable_deep_supervision: bool
    :param use_gabor_stem: Replace the learned first convolution with a frozen Gabor filter bank. Defaults to False.
    :param gabor_filters: Number of Gabor filters in the stem. Defaults to 32.
    :param gabor_kernel_size: Kernel size of the Gabor filters. Defaults to 11.
    :param gabor_activation: Activation after the Gabor stem, if any.
    :param gabor_stem_projection: Whether the Gabor stem includes a 1x1 projection. Defaults to True.
    :param use_laplacian_pyramid: Split each skip into a low-pass and a high-frequency band. Defaults to False.
    :param high_freq_blocks: Bias-free residual blocks applied to the Laplacian
        high-frequency skip band at each encoder level, ignored when
        `use_laplacian_pyramid=False`. Defaults to 0, a byte-identical no-op.
    :type high_freq_blocks: int
    :param laplacian_kernel_size: Kernel size of the Laplacian pyramid's Gaussian filter. Defaults to (5, 5).
    :param zero_pad_channels: Zero-pad instead of projecting channels at a junction. Defaults to False.
    :param downsample_pool_type: `'max'` or `'average'`. Defaults to `'max'`.
    :param expose_bottleneck: Also return the bottleneck feature map. Defaults to False.
    :param block_normalization: One of `'batchnorm'`, `'layernorm'`,
        `'bias_free_batchnorm'`. Defaults to `'batchnorm'`, which resolves to
        the variance-only `BiasFreeBatchNorm`, an exact synonym of
        `'bias_free_batchnorm'` here. Stock `BatchNormalization` is not
        reachable from this builder, since its `moving_mean` subtraction
        breaks homogeneity. `'layernorm'` is scale-invariant, not homogeneous.
        Applied to every encoder, bottleneck and decoder block, and to the
        deep-supervision heads.
    :type block_normalization: str
    :param final_projection_groups: Number of groups in the final projection convolution. Defaults to 1.
    :param dropout_rate: Dropout rate applied after the activation inside every
        block and every deep-supervision head. Defaults to 0.0, no Dropout sublayer.
    :type dropout_rate: float
    :param model_name: Model name. Defaults to `'bias_free_unet'`.
    :type model_name: str
    :return: A single output tensor if `enable_deep_supervision=False`, or a
        list `[final_output, ...intermediate_outputs]` if True.
    :rtype: keras.Model
    :raises ValueError: If depth is less than 2, initial_filters is non-positive,
        filter_multiplier is less than 1, or blocks_per_level is non-positive.
    :raises TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        ```python
        model = create_bfunet_denoiser(
            input_shape=(256, 256, 3),
            depth=4,
            initial_filters=64,
            enable_deep_supervision=True
        )
        inference_model = create_bfunet_denoiser(
            input_shape=(None, None, 3),
            depth=4,
            initial_filters=64,
            enable_deep_supervision=False
        )
        ```
    """
    block_norm = _validate_bfunet_args(
        input_shape=input_shape,
        depth=depth,
        initial_filters=initial_filters,
        filter_multiplier=filter_multiplier,
        blocks_per_level=blocks_per_level,
        high_freq_blocks=high_freq_blocks,
        block_normalization=block_normalization,
        downsample_pool_type=downsample_pool_type,
        final_projection_groups=final_projection_groups,
        dropout_rate=dropout_rate,
    )

    inputs = keras.Input(shape=input_shape, name='input_images')

    stem_input = _build_gabor_stem(
        inputs=inputs,
        input_shape=input_shape,
        use_gabor_stem=use_gabor_stem,
        gabor_filters=gabor_filters,
        gabor_kernel_size=gabor_kernel_size,
        gabor_activation=gabor_activation,
        gabor_stem_projection=gabor_stem_projection,
        initial_filters=initial_filters,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )

    # Calculate filter sizes for each level (int() keeps a float filter_multiplier safe;
    # for the default int multiplier this is a no-op).
    filter_sizes = [int(initial_filters * (filter_multiplier ** i)) for i in range(depth + 1)]

    output_channels = input_shape[-1]

    x, skip_connections = _build_encoder_path(
        stem_input=stem_input,
        filter_sizes=filter_sizes,
        depth=depth,
        blocks_per_level=blocks_per_level,
        kernel_size=kernel_size,
        initial_kernel_size=initial_kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_residual_blocks=use_residual_blocks,
        block_norm=block_norm,
        dropout_rate=dropout_rate,
        use_laplacian_pyramid=use_laplacian_pyramid,
        laplacian_kernel_size=laplacian_kernel_size,
        downsample_pool_type=downsample_pool_type,
        high_freq_blocks=high_freq_blocks,
    )

    x, bottleneck_output = _build_bottleneck(
        x=x,
        filter_sizes=filter_sizes,
        depth=depth,
        blocks_per_level=blocks_per_level,
        kernel_size=kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_residual_blocks=use_residual_blocks,
        block_norm=block_norm,
        dropout_rate=dropout_rate,
        expose_bottleneck=expose_bottleneck,
    )

    x, deep_supervision_outputs = _build_decoder_path(
        x=x,
        skip_connections=skip_connections,
        filter_sizes=filter_sizes,
        depth=depth,
        output_channels=output_channels,
        initial_filters=initial_filters,
        blocks_per_level=blocks_per_level,
        kernel_size=kernel_size,
        activation=activation,
        final_activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_residual_blocks=use_residual_blocks,
        block_norm=block_norm,
        dropout_rate=dropout_rate,
        zero_pad_channels=zero_pad_channels,
        enable_deep_supervision=enable_deep_supervision,
    )

    final_output = _build_final_projection(
        x=x,
        output_channels=output_channels,
        final_activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        final_projection_groups=final_projection_groups,
    )

    # =========================================================================
    # MODEL CREATION
    # =========================================================================

    if enable_deep_supervision and deep_supervision_outputs:
        # Return multiple outputs: [final_output, supervision_outputs...]
        # Order supervision outputs from shallowest to deepest (by resolution)
        # The final output (index 0) is the primary inference output
        # Supervision outputs (indices 1+) are ordered by decreasing resolution

        # Reverse the supervision outputs so they go from shallow to deep
        # deep_supervision_outputs was built as [level_3, level_2, level_1] (deep to shallow)
        # We want [level_1, level_2, level_3] (shallow to deep)
        ordered_supervision_outputs = list(reversed(deep_supervision_outputs))
        all_outputs = [final_output] + ordered_supervision_outputs
        if expose_bottleneck:
            all_outputs = all_outputs + [bottleneck_output]

        logger.info(f"Created deep supervision model with {len(all_outputs)} outputs:")
        logger.info(f"  - Final output (index 0): {final_output.shape}")
        for i, sup_output in enumerate(ordered_supervision_outputs):
            # Calculate the actual level based on reversed order
            level = i + 1  # levels 1, 2, 3 for indices 1, 2, 3
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

        logger.info("Created single-output model")

    logger.info(f"Created bias-free U-Net model '{model_name}' with depth {depth}")
    logger.info(f"Filter progression: {filter_sizes}")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")
    logger.info(f"Deep supervision enabled: {enable_deep_supervision}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model


# ---------------------------------------------------------------------
# Pretrained Weights Functions
# ---------------------------------------------------------------------

# Raises rather than falling back to random init, since no public BFUNet
# weights are distributed; load a local checkpoint from src/train/bfunet/ instead.
def _download_bfunet_weights(
        variant: str,
        dataset: str = "imagenet_denoising",
        cache_dir: Optional[str] = None
) -> str:
    """Resolve a download path for pretrained BFUNet weights; always raises.

    Not implemented: no public BFUNet weights ship with dl_techniques. Kept so
    `pretrained=True` fails loudly instead of silently returning a
    randomly-initialized denoiser.

    :param variant: Model variant name, unused.
    :param dataset: Dataset identifier, unused.
    :param cache_dir: Cache directory, unused.
    :raises NotImplementedError: Always.
    """
    raise NotImplementedError(
        f"No pretrained BFUNet weights are distributed with dl_techniques "
        f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
        f"checkpoint instead: create_bfunet_variant('{variant}', input_shape, "
        f"pretrained='/path/to/weights.keras')."
    )


def load_pretrained_weights_into_model(
        model: keras.Model,
        weights_path: str,
        skip_mismatch: bool = True
) -> None:
    """Load pretrained weights into a BFUNet model, tolerating shape mismatches.

    Weights are transferred layer by layer via
    :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
    the replacement for `model.load_weights(by_name=True)`, which raises on
    `.keras` files in Keras 3.8+.

    :param model: The BFUNet model to load weights into.
    :type model: keras.Model
    :param weights_path: Path to the weights file, `.keras` format.
    :type weights_path: str
    :param skip_mismatch: Whether to skip layers with mismatched shapes, useful
        when the pretrained and target models differ in input/output shape or
        deep-supervision settings. Maps to `strict=not skip_mismatch`.
    :type skip_mismatch: bool
    :raises FileNotFoundError: If weights_path doesn't exist.
    :raises ValueError: If weights cannot be loaded.

    Example:
        ```python
        model = create_bfunet_variant('base', (256, 256, 3))
        load_pretrained_weights_into_model(model, 'bfunet_base.keras')
        ```
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    try:
        # Weight transfer needs a built target.
        if not model.built:
            dummy_input = keras.random.normal((1,) + tuple(model.input.shape[1:]))
            model(dummy_input, training=False)

        logger.info(f"Loading pretrained weights from {weights_path}")

        report = load_weights_from_checkpoint(
            target=model,
            ckpt_path=weights_path,
            skip_prefixes=(),
            strict=not skip_mismatch,
        )

        logger.info(report.summary_string())
        if skip_mismatch:
            logger.info(
                "Weights loaded with skip_mismatch=True. "
                "Layers with shape mismatches were skipped."
            )
        else:
            logger.info("All weights loaded successfully.")

    except Exception as e:
        raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_bfunet_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        enable_deep_supervision: bool = False,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet_denoising",
        weights_input_shape: Optional[Tuple[int, int, int]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
) -> keras.Model:
    """Create a bias-free U-Net model from a named variant configuration.

    :param variant: One of `'tiny'`, `'small'`, `'base'`, `'large'`, `'xlarge'`.
    :type variant: str
    :param input_shape: Shape of input images, `(height, width, channels)`.
    :type input_shape: Tuple[int, int, int]
    :param enable_deep_supervision: Whether to enable deep-supervision outputs.
    :type enable_deep_supervision: bool
    :param pretrained: A path to a local `.keras` weights file, or `True` to
        raise `NotImplementedError` since no public BFUNet weights ship with
        dl_techniques, or `False`, the default, for random initialization.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for pretrained weights, one of
        `"imagenet_denoising"`, `"general_denoising"`. Used only if pretrained is a path.
    :type weights_dataset: str
    :param weights_input_shape: Input shape used during weight pretraining,
        needed only when loading weights with a different `input_shape`.
        Defaults to `(256, 256, 3)`.
    :param cache_dir: Optional directory to cache downloaded weights.
    :param kwargs: Additional keyword arguments overriding the variant defaults.
    :return: A bias-free U-Net model with the given variant's configuration.
    :rtype: keras.Model
    :raises ValueError: If variant is not recognized.
    :raises NotImplementedError: If pretrained is True.

    Example:
        ```python
        model = create_bfunet_variant('base', (256, 256, 3), enable_deep_supervision=True)
        model.summary()

        inference_model = create_bfunet_variant('base', (None, None, 3), enable_deep_supervision=False)

        model = create_bfunet_variant(
            'base',
            (256, 256, 3),
            pretrained='path/to/weights.keras'
        )

        model = create_bfunet_variant(
            'base',
            (256, 256, 1),
            pretrained='path/to/weights.keras',
            weights_input_shape=(256, 256, 3)
        )
        ```
    """
    if variant not in BFUNET_CONFIGS:
        available_variants = list(BFUNET_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = BFUNET_CONFIGS[variant].copy()
    description = config.pop('description')
    config.update(kwargs)

    if 'model_name' not in config:
        ds_suffix = '_ds' if enable_deep_supervision else ''
        config['model_name'] = f'bias_free_unet_{variant}{ds_suffix}'

    config['enable_deep_supervision'] = enable_deep_supervision

    logger.info(f"Creating bias-free U-Net variant '{variant}': {description}")
    logger.info(f"Deep supervision: {'enabled' if enable_deep_supervision else 'disabled'}")

    load_weights_path = None

    if pretrained:
        if isinstance(pretrained, str):
            load_weights_path = pretrained
            logger.info(f"Will load weights from local file: {load_weights_path}")
        else:
            load_weights_path = _download_bfunet_weights(
                variant=variant,
                dataset=weights_dataset,
                cache_dir=cache_dir
            )

        if weights_input_shape and input_shape != weights_input_shape:
            logger.info(
                f"Loading weights pretrained on {weights_input_shape} "
                f"for model with input shape {input_shape}. "
                f"Will skip layers with shape mismatches."
            )

    model = create_bfunet_denoiser(
        input_shape=input_shape,
        **config
    )

    if load_weights_path:
        # skip_mismatch is unconditionally True: a checkpoint may differ from
        # this model in input shape, output channels or deep-supervision head
        # count, and none of those are knowable before the transfer runs.
        load_pretrained_weights_into_model(
            model=model,
            weights_path=load_weights_path,
            skip_mismatch=True,
        )

    return model
