"""
Bias-Free U-Net Model with Deep Supervision and Variants

Implements a U-Net architecture with deep supervision where all additive constants
(bias terms) have been removed to enable better generalization across different
noise levels and improved scaling invariance properties. The deep supervision
outputs intermediate predictions at multiple scales during training, allowing
for better gradient flow and more stable training.

Deep supervision provides several benefits:
- Better gradient flow to deeper layers during training
- Multi-scale feature learning and supervision
- More stable training for very deep networks
- Curriculum learning capabilities through weight scheduling

The model outputs multiple scales during training:
- Output 0: Final inference output (highest resolution, primary output)
- Output 1-N: Intermediate supervision outputs at progressively lower resolutions

Based on the bias-free principles from "Robust and Interpretable Blind Image
Denoising via Bias-Free Convolutional Neural Networks" (Mohan et al., ICLR 2020)
applied to the U-Net architecture with deep supervision.
"""

import os
import keras
from typing import Optional, Union, Tuple, List, Dict, Any


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock
# ConvUNeXt-parity feature components (reused as-is; already bias-free + serializable).
from dl_techniques.initializers import create_gabor_depthwise_conv2d
from dl_techniques.layers.laplacian_filter import LaplacianPyramidLevel
from dl_techniques.layers.match_channels import MatchChannels


# DECISION plan_2026-07-04_58ac8e73/D-002: ONE helper folds the per-level downsample sites
# (inter-level pools + the last-level pool that feeds the bottleneck) into a uniform call so
# the OFF/ON swap logic lives in one place (mirrors bfconvunext _downsample_and_skip D-001).
# OFF path MUST reproduce the exact prior ops/names (MaxPooling2D named `downsample_name`,
# raw pre-downsample skip); a local copy is kept here rather than importing bfconvunext's
# private helper (cross-module `_`-import) or refactoring that checkpoint-sensitive file.
def _downsample_and_skip(
        x: keras.KerasTensor,
        use_laplacian_pyramid: bool,
        laplacian_kernel_size: Tuple[int, int],
        downsample_name: str,
        pyramid_name: str,
        pool_type: str = "max",
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Return ``(skip, downsampled)`` for one encoder junction.

    OFF path (default, byte-identical): skip = pre-downsample tensor; downsample =
    ``MaxPooling2D(2,2)`` named ``downsample_name`` (``AveragePooling2D`` when
    ``pool_type='average'`` — a linear, Miyasawa-clean operator). ON path: a bias-free
    ``LaplacianPyramidLevel`` split — high band becomes the skip, low band continues down.
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
    downsampled = pool_layer(pool_size=(2, 2), name=downsample_name)(x)
    return skip, downsampled

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
# Pretrained Weights Configuration
# ---------------------------------------------------------------------

BFUNET_PRETRAINED_WEIGHTS: Dict[str, Dict[str, str]] = {
    'tiny': {
        'imagenet_denoising': 'https://example.com/bfunet_tiny_imagenet_denoising.keras',
        'general_denoising': 'https://example.com/bfunet_tiny_general_denoising.keras',
    },
    'small': {
        'imagenet_denoising': 'https://example.com/bfunet_small_imagenet_denoising.keras',
        'general_denoising': 'https://example.com/bfunet_small_general_denoising.keras',
    },
    'base': {
        'imagenet_denoising': 'https://example.com/bfunet_base_imagenet_denoising.keras',
        'general_denoising': 'https://example.com/bfunet_base_general_denoising.keras',
    },
    'large': {
        'imagenet_denoising': 'https://example.com/bfunet_large_imagenet_denoising.keras',
        'general_denoising': 'https://example.com/bfunet_large_general_denoising.keras',
    },
    'xlarge': {
        'imagenet_denoising': 'https://example.com/bfunet_xlarge_imagenet_denoising.keras',
        'general_denoising': 'https://example.com/bfunet_xlarge_general_denoising.keras',
    }
}

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
        gabor_kernel_size: Union[int, Tuple[int, int]] = 7,
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
    """
    Create a bias-free U-Net model with optional deep supervision.

    This function creates a complete U-Net architecture using bias-free layers with
    deep supervision capabilities. The model exhibits scaling-invariant properties:
    if the input is scaled by α, the output is also scaled by α.

    During training with deep supervision enabled, the model outputs multiple scales:
    - Output 0: Final inference output (full resolution)
    - Output 1: Second-to-last decoder level output
    - Output N: Deepest supervision level output

    During inference, only the final output (index 0) is typically used.

    Architecture:
    - Encoder: Bias-free conv blocks + downsampling at each level
    - Bottleneck: Bias-free conv blocks at the lowest resolution
    - Decoder: Upsampling + skip connections + bias-free conv blocks
    - Deep Supervision: Additional outputs at intermediate decoder levels
    - Skip connections preserve high-resolution features

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        depth: Integer, depth of the U-Net (number of downsampling levels). Defaults to 4.
        initial_filters: Integer, number of filters in the first level. Defaults to 64.
        filter_multiplier: Integer, multiplier for filters at each level. Defaults to 2.
        blocks_per_level: Integer, number of conv blocks per level. Defaults to 2.
        kernel_size: Integer or tuple, size of convolutional kernels. Defaults to 3.
        initial_kernel_size: Integer or tuple, size of first convolutional kernels. Defaults to 5.
        activation: String or callable, activation function. Defaults to 'leaky_relu'.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'he_normal'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        use_residual_blocks: Boolean, whether to use residual blocks. Defaults to True.
        high_freq_blocks: Integer, number of bias-free residual blocks applied to the
            Laplacian high-frequency skip band at each encoder level before it becomes
            the decoder skip. **Ignored when use_laplacian_pyramid=False.** Defaults to 0
            (byte-identical no-op: adds ZERO layers, renames nothing).
        enable_deep_supervision: Boolean, whether to add deep supervision outputs. Defaults to True.
        model_name: String, name for the model. Defaults to 'bias_free_unet'.

    Returns:
        keras.Model: Bias-free U-Net model ready for training.
                    - If deep_supervision=False: Single output tensor
                    - If deep_supervision=True: List of output tensors [final_output, intermediate_outputs...]

    Raises:
        ValueError: If depth is less than 3, initial_filters is non-positive,
                   filter_multiplier is less than 1, or blocks_per_level is non-positive.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create standard bias-free U-Net with deep supervision
        >>> model = create_bfunet_denoiser(
        ...     input_shape=(256, 256, 3),
        ...     depth=4,
        ...     initial_filters=64,
        ...     enable_deep_supervision=True
        ... )
        >>> # Model outputs: [final_output, supervision_output_1, supervision_output_2, ...]
        >>>
        >>> # Create inference-only model (single output)
        >>> inference_model = create_bfunet_denoiser(
        ...     input_shape=(None, None, 3),  # Flexible spatial dimensions
        ...     depth=4,
        ...     initial_filters=64,
        ...     enable_deep_supervision=False  # Single output for inference
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

    if high_freq_blocks < 0:
        raise ValueError(f"high_freq_blocks must be non-negative, got {high_freq_blocks}")

    # DECISION plan_2026-07-04_58ac8e73/D-002: additive ConvUNeXt-parity features. Every
    # new kwarg defaults to a byte-identical no-op; the OFF path reproduces the original
    # plain U-Net exactly (same layer names/ops) so the ~30 existing tests + bfunet_conditional
    # stay green. Do NOT change default behavior.
    if block_normalization not in ('batchnorm', 'layernorm', 'bias_free_batchnorm'):
        raise ValueError(
            "block_normalization must be 'batchnorm', 'layernorm' or 'bias_free_batchnorm', "
            f"got {block_normalization}")
    if downsample_pool_type not in ('max', 'average'):
        raise ValueError(
            f"downsample_pool_type must be 'max' or 'average', got {downsample_pool_type}")
    if final_projection_groups < 1:
        raise ValueError(f"final_projection_groups must be >= 1, got {final_projection_groups}")
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_images')

    # Optional frozen Gabor stem (default OFF -> stem_input is inputs, a no-op). Reuses the
    # shared bias-free gabor bank + mandatory 1x1 projection (mirrors bfconvunext).
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

    # Calculate filter sizes for each level (int() keeps a float filter_multiplier safe;
    # for the default int multiplier this is a no-op).
    filter_sizes = [int(initial_filters * (filter_multiplier ** i)) for i in range(depth + 1)]

    # Storage for skip connections and deep supervision outputs
    skip_connections: List[keras.layers.Layer] = []
    deep_supervision_outputs: List[keras.layers.Layer] = []

    # =========================================================================
    # ENCODER PATH (Contracting)
    # =========================================================================

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
                    normalization_type=block_normalization,
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
                        normalization_type=block_normalization,
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
                        normalization_type=block_normalization,
                        dropout_rate=dropout_rate,
                        name=f'encoder_level_{level}_conv_{block_idx}'
                    )(x)

        # Skip connection + downsample. ALL levels route through the helper so the
        # pool-type / Laplacian-pyramid swap lives in one place. The last level's pool
        # feeds the bottleneck and keeps the original name 'bottleneck_downsample', so the
        # OFF path is byte-identical (skip = pre-pool x, MaxPooling2D at the same names).
        if level < depth - 1:
            skip, x = _downsample_and_skip(
                x, use_laplacian_pyramid, laplacian_kernel_size,
                f'encoder_downsample_{level}', f'encoder_pyramid_{level}',
                downsample_pool_type,
            )
        else:
            skip, x = _downsample_and_skip(
                x, use_laplacian_pyramid, laplacian_kernel_size,
                'bottleneck_downsample', 'bottleneck_pyramid',
                downsample_pool_type,
            )

        # DECISION plan_2026-07-06_b17c1f83/D-001: optionally process the Laplacian
        # high-frequency band with N bias-free residual blocks before it becomes the
        # decoder skip. Gated on use_laplacian_pyramid (the high band only exists then);
        # high_freq_blocks=0 (default) adds ZERO layers -> byte-identical OFF path, so
        # existing `.keras` checkpoints (whose layer names are load-bearing) still load.
        # Do NOT drop the use_laplacian_pyramid gate or the >0 gate: without the pyramid
        # there is no high band and this would rename/insert layers into the raw-skip path.
        # The high band is channel-preserving, so filters=current_filters is shape-safe.
        if high_freq_blocks > 0 and use_laplacian_pyramid:
            for hf_idx in range(high_freq_blocks):
                skip = BiasFreeResidualBlock(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    normalization_type=block_normalization,
                    dropout_rate=dropout_rate,
                    name=f'skip_highfreq_block_{level}_{hf_idx}'
                )(skip)

        skip_connections.append(skip)

    # =========================================================================
    # BOTTLENECK
    # =========================================================================

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
                normalization_type=block_normalization,
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
                normalization_type=block_normalization,
                dropout_rate=dropout_rate,
                name=f'bottleneck_conv_{block_idx}'
            )(x)

    # Optional bottleneck tap: a zero-parameter linear (bias-free) marker on the deepest
    # latent so it can be exposed as an additional output. No-op when expose_bottleneck=False.
    bottleneck_output = None
    if expose_bottleneck:
        x = keras.layers.Activation('linear', name='bottleneck')(x)
        bottleneck_output = x

    # =========================================================================
    # DECODER PATH (Expanding) with Deep Supervision
    # =========================================================================

    logger.info(f"Building decoder path with {depth} levels")
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
                    normalization_type=block_normalization,
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
                    normalization_type=block_normalization,
                    dropout_rate=dropout_rate,
                    name=f'decoder_level_{level}_conv_{block_idx}'
                )(x)

        # =====================================================================
        # DEEP SUPERVISION OUTPUT (if enabled and not the final level)
        # =====================================================================

        if enable_deep_supervision and level > 0:
            # Create supervision output at current scale from a branch
            supervision_branch = BiasFreeConv2D(
                filters=initial_filters,
                kernel_size=3,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=True,
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

    # =========================================================================
    # FINAL OUTPUT LAYER (Primary inference output)
    # =========================================================================

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

        logger.info(f"Created single-output model")

    logger.info(f"Created bias-free U-Net model '{model_name}' with depth {depth}")
    logger.info(f"Filter progression: {filter_sizes}")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")
    logger.info(f"Deep supervision enabled: {enable_deep_supervision}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model

# ---------------------------------------------------------------------
# Pretrained Weights Functions
# ---------------------------------------------------------------------

def _download_bfunet_weights(
        variant: str,
        dataset: str = "imagenet_denoising",
        cache_dir: Optional[str] = None
) -> str:
    """
    Download pretrained BFUNet weights from URL.

    Args:
        variant: String, model variant name (e.g., 'tiny', 'small', 'base').
        dataset: String, dataset the weights were trained on.
            Options: "imagenet_denoising", "general_denoising".
        cache_dir: Optional string, directory to cache downloaded weights.
            If None, uses default Keras cache directory.

    Returns:
        String, path to the downloaded weights file.

    Raises:
        ValueError: If variant or dataset is not available.

    Example:
        >>> weights_path = _download_bfunet_weights('base', 'imagenet_denoising')
    """
    if variant not in BFUNET_PRETRAINED_WEIGHTS:
        raise ValueError(
            f"No pretrained weights available for variant '{variant}'. "
            f"Available variants: {list(BFUNET_PRETRAINED_WEIGHTS.keys())}"
        )

    if dataset not in BFUNET_PRETRAINED_WEIGHTS[variant]:
        raise ValueError(
            f"No pretrained weights available for dataset '{dataset}'. "
            f"Available datasets for {variant}: "
            f"{list(BFUNET_PRETRAINED_WEIGHTS[variant].keys())}"
        )

    url = BFUNET_PRETRAINED_WEIGHTS[variant][dataset]

    logger.info(f"Downloading BFUNet-{variant} weights from {dataset}...")

    # Download weights using Keras utility
    weights_path = keras.utils.get_file(
        fname=f"bfunet_{variant}_{dataset}.keras",
        origin=url,
        cache_dir=cache_dir,
        cache_subdir="models/bfunet"
    )

    logger.info(f"Weights downloaded to: {weights_path}")
    return weights_path


def load_pretrained_weights_into_model(
        model: keras.Model,
        weights_path: str,
        skip_mismatch: bool = True,
        by_name: bool = True
) -> None:
    """
    Load pretrained weights into a BFUNet model.

    This function handles loading weights with smart mismatch handling,
    particularly useful when the input shape, output channels, or deep
    supervision settings differ between the pretrained and target models.

    Weights are transferred layer-by-layer via
    :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
    which is the canonical replacement for ``model.load_weights(by_name=True)``
    (the latter raises on ``.keras`` files in Keras 3.8+).

    Args:
        model: Keras Model, the BFUNet model to load weights into.
        weights_path: String, path to the weights file (.keras format).
        skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
            Useful when loading weights with different input/output shapes
            or deep supervision settings. Maps to ``strict=not skip_mismatch``.
        by_name: Boolean, retained for backward compatibility. Layer-by-layer
            transfer is always name-based; this argument is ignored.

    Raises:
        FileNotFoundError: If weights_path doesn't exist.
        ValueError: If weights cannot be loaded.

    Example:
        >>> model = create_bfunet_variant('base', (256, 256, 3))
        >>> load_pretrained_weights_into_model(model, 'bfunet_base.keras')
        >>>
        >>> # Load with different output channels (e.g., 1 channel grayscale)
        >>> model = create_bfunet_variant('base', (256, 256, 1))
        >>> load_pretrained_weights_into_model(
        ...     model, 'bfunet_base_rgb.keras', skip_mismatch=True
        ... )
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    del by_name  # name-based transfer is implicit; kept for signature stability

    try:
        # Build model if not already built (weight transfer needs a built target)
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
    """
    Create a bias-free U-Net model with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        enable_deep_supervision: Boolean, whether to enable deep supervision outputs.
        pretrained: Boolean or string. If True, loads pretrained weights from
            default URL. If string, treats it as a path to local weights file.
        weights_dataset: String, dataset for pretrained weights.
            Options: "imagenet_denoising", "general_denoising".
            Only used if pretrained=True.
        weights_input_shape: Tuple, input shape used during weight pretraining.
            Only needed if loading pretrained weights with different input_shape.
            Defaults to (256, 256, 3) for standard denoising weights.
        cache_dir: Optional string, directory to cache downloaded weights.
        **kwargs: Additional keyword arguments to override default parameters.

    Returns:
        keras.Model: Bias-free U-Net model with the specified variant configuration.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        >>> # Standard usage with deep supervision
        >>> model = create_bfunet_variant('base', (256, 256, 3), enable_deep_supervision=True)
        >>> model.summary()
        >>>
        >>> # Inference model (single output)
        >>> inference_model = create_bfunet_variant('base', (None, None, 3), enable_deep_supervision=False)
        >>>
        >>> # With custom parameters
        >>> model = create_bfunet_variant('large', (224, 224, 1),
        ...                                     enable_deep_supervision=True,
        ...                                     activation='gelu',
        ...                                     use_residual_blocks=False)
        >>>
        >>> # Load pretrained weights
        >>> model = create_bfunet_variant(
        ...     'base',
        ...     (256, 256, 3),
        ...     pretrained=True,
        ...     weights_dataset='imagenet_denoising'
        ... )
        >>>
        >>> # Fine-tune with different input channels (e.g., grayscale)
        >>> model = create_bfunet_variant(
        ...     'base',
        ...     (256, 256, 1),
        ...     pretrained=True,
        ...     weights_input_shape=(256, 256, 3)  # Pretrained on RGB
        ... )
        >>>
        >>> # Load from local weights file
        >>> model = create_bfunet_variant('large', (256, 256, 3), pretrained='path/to/weights.keras')
    """
    if variant not in BFUNET_CONFIGS:
        available_variants = list(BFUNET_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = BFUNET_CONFIGS[variant].copy()
    description = config.pop('description')

    # Override config with any provided kwargs
    config.update(kwargs)

    # Set model name if not provided
    if 'model_name' not in config:
        ds_suffix = '_ds' if enable_deep_supervision else ''
        config['model_name'] = f'bias_free_unet_{variant}{ds_suffix}'

    # Set deep supervision
    config['enable_deep_supervision'] = enable_deep_supervision

    logger.info(f"Creating bias-free U-Net variant '{variant}': {description}")
    logger.info(f"Deep supervision: {'enabled' if enable_deep_supervision else 'disabled'}")

    # Handle pretrained weights
    load_weights_path = None
    skip_mismatch = False

    if pretrained:
        if isinstance(pretrained, str):
            # Load from local file path
            load_weights_path = pretrained
            logger.info(f"Will load weights from local file: {load_weights_path}")
        else:
            # Download from URL
            try:
                load_weights_path = _download_bfunet_weights(
                    variant=variant,
                    dataset=weights_dataset,
                    cache_dir=cache_dir
                )
            except Exception as e:
                logger.warning(
                    f"Failed to download pretrained weights: {str(e)}. "
                    f"Continuing with random initialization."
                )
                load_weights_path = None

        # Determine if we need to skip mismatches
        if load_weights_path:
            # Check if input shape matches
            if weights_input_shape and input_shape != weights_input_shape:
                logger.info(
                    f"Loading weights pretrained on {weights_input_shape} "
                    f"for model with input shape {input_shape}. "
                    f"Will skip layers with shape mismatches."
                )
                skip_mismatch = True

            # Check if output channels match
            if input_shape[-1] != (weights_input_shape[-1] if weights_input_shape else 3):
                logger.info(
                    f"Output channels differ from pretrained weights. "
                    f"Will skip final output layers."
                )
                skip_mismatch = True

            # Check if deep supervision settings match
            # If we can't determine this from weights, we'll let skip_mismatch handle it
            skip_mismatch = True  # Always skip mismatches for safety with pretrained weights

    # Create model
    model = create_bfunet_denoiser(
        input_shape=input_shape,
        **config
    )

    # Load pretrained weights if available
    if load_weights_path:
        try:
            load_pretrained_weights_into_model(
                model=model,
                weights_path=load_weights_path,
                skip_mismatch=skip_mismatch,
                by_name=True
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {str(e)}")
            raise

    return model

# ---------------------------------------------------------------------
# Utility Functions for Deep Supervision
# ---------------------------------------------------------------------

from dl_techniques.utils.deep_supervision import (
    get_model_output_info,
    create_inference_model_from_training_model,
)

# ---------------------------------------------------------------------