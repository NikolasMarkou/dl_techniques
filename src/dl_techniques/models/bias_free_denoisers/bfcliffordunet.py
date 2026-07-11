"""
CliffordUNet: Bias-Free, Strictly-Homogeneous Clifford U-Net Denoiser.

A bias-free image-denoiser U-Net built from the Clifford geometric block
(:class:`CliffordNetBlock`) in a degree-1-homogeneous configuration. It mirrors
the encoder / bottleneck / decoder topology of ``create_convunext_denoiser``
(``bfconvunext.py``) but swaps every ConvNeXt block site for a Clifford block
configured for Miyasawa compliance:

- ``use_bias=False`` everywhere (no additive offsets),
- ``input_normalization_type="bias_free_batch_norm"`` -- the ``z_det`` (detail)
  stream norm: variance-only, fixed-statistic (divides by a frozen
  ``running_var`` at inference, no mean, no beta) so ``z_det`` stays DEGREE-1
  homogeneous at inference,
- ``normalization_type="zero_centered_rms_norm"`` -- the ``z_ctx`` (context)
  stream norm: PER-INPUT centered-RMS, purely multiplicative ``gamma``, no
  additive beta. Being scale-invariant (``norm(alpha*x) == norm(x)``) it makes
  ``z_ctx`` DEGREE-0,
- ``ctx_mode="abs"`` -- the context path does NOT subtract ``z_det`` (a
  ``ctx_mode="diff"`` ``z_ctx - z_det`` would mix the degree-0 context with the
  degree-1 detail and remix the degrees),
- ``activation="leaky_relu"``, ``dot_activation="leaky_relu"``,
  ``feature_activation="leaky_relu"`` (homogeneous nonlinearity, NOT SiLU/GELU),
- ``use_gate=False`` (drops the multiplicative sigmoid gate, which is degree-2
  in the input).

Homogeneity mechanism (HONEST)
------------------------------
The block's core is the bilinear Clifford geometric product ``z_det (X) z_ctx``.
With BOTH streams degree-1 the product would be DEGREE-2 in the input
(``f(alpha*x) ~ alpha^2 f(x)``) -- empirically confirmed (plan
``plan_2026-07-01_6dc255c1`` decision D-004). The fix here makes the CONTEXT
stream degree-0 (scale-free ``zero_centered_rms_norm`` + ``ctx_mode="abs"``) so:

    z_det (degree-1)  (X)  z_ctx (degree-0)  ->  degree-1 geometric product

and the whole block is DEGREE-1 homogeneous ``f(alpha * x) = alpha * f(x)``.

Epsilon-floor caveat
--------------------
Because the context norm is a per-input RMS with an epsilon regularizer,
homogeneity holds to ``rel_err < 1e-2`` only for input scale
``alpha in [0.5, 1000]`` -- the ``[-0.5, 0.5]`` denoising operating regime. At
extreme small scale (``alpha ~ 1e-3``) the RMS epsilon dominates and the
identity degrades (measured ``rel_err ~ 0.54``). This is outside the operating
range and is the same class of accepted deviation as the existing Miyasawa
clip-boundary caveat (strict ``residual = score`` also breaks at the clip
boundary). See decisions D-005 / D-006.

Design constraints honored here
-------------------------------
- ``CliffordNetBlock`` is ISOTROPIC (in-channels == out-channels == ``channels``).
  Every U-Net level channel-width change goes through an EXTERNAL bias-free
  ``Conv2D(filters, 1, use_bias=False)`` (mirroring ``bfconvunext``'s
  channel-adjust and ``autoencoder.py``'s ``enc_proj`` / ``dec_align``), never
  inside the block.
- Per-level ``shifts`` are sized so every shift satisfies ``s < channels`` at
  that level (the block silently drops ``s >= channels`` with a warning). A
  build-time assertion guarantees each level keeps >= 1 valid shift.
- The frozen Gabor stem (``create_gabor_depthwise_conv2d``, ``trainable=False``)
  + a mandatory bias-free 1x1 projection is reused exactly as ``bfconvunext``
  does, including the no-projection channel-match contract
  (``in_channels * gabor_filters == initial_filters``).
- ``LaplacianPyramidLevel`` (optional Laplacian downsample/skip) and
  ``MatchChannels`` (zero-pad channel matching) are reused from the SAME modules
  ``bfconvunext`` imports them from.
- ``final_activation="linear"`` (HARD invariant for bias-free homogeneity). The
  final output is a bias-free 1x1 ``Conv2D`` projection to ``output_channels``.
  Like ``bfconvunext``, the model emits the DIRECT prediction (not input + residual);
  the trainer owns the residual-learning convention.

Deep supervision is NOT wired (the trainer fail-fasts on it): passing
``enable_deep_supervision=True`` raises ``NotImplementedError``.
"""

import keras
from typing import Optional, Union, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import create_gabor_depthwise_conv2d
from dl_techniques.layers.laplacian_filter import LaplacianPyramidLevel
from dl_techniques.layers.match_channels import MatchChannels
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock


# ---------------------------------------------------------------------
# CliffordUNet Model Variant Configurations
# ---------------------------------------------------------------------
# Mirrors the CONVUNEXT_CONFIGS shape (depth / initial_filters /
# blocks_per_level / drop_path_rate / description) so the trainer clone can
# consume it identically, plus the Clifford-specific per-model base `shifts`.

CLIFFORDUNET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'depth': 3,
        'initial_filters': 24,
        'blocks_per_level': 1,
        'shifts': [1, 2],
        'drop_path_rate': 0.0,
        'description': 'Tiny CliffordUNet (depth=3) for quick experiments.',
    },
    'small': {
        'depth': 3,
        'initial_filters': 32,
        'blocks_per_level': 2,
        'shifts': [1, 2],
        'drop_path_rate': 0.1,
        'description': 'Small CliffordUNet (depth=3) with minimal capacity.',
    },
    'base': {
        'depth': 4,
        'initial_filters': 48,
        'blocks_per_level': 2,
        'shifts': [1, 2, 3],
        'drop_path_rate': 0.1,
        'description': 'Base CliffordUNet (depth=4) with standard configuration.',
    },
}


# ---------------------------------------------------------------------
# Homogeneous bias-free Clifford block config (single source of truth)
# ---------------------------------------------------------------------

def _homogeneous_block_kwargs() -> Dict[str, Any]:
    """Return the bias-free / degree-1-homogeneous CliffordNetBlock kwargs.

    Centralized so every block site (encoder / bottleneck / decoder) is
    guaranteed identical on the homogeneity axes. Do NOT inline these at each
    call site -- a single drifted flag silently breaks Miyasawa compliance at
    one level only, which is nearly invisible until the homogeneity probe.

    The degree bookkeeping (see D-005): the ``z_det`` (detail) stream uses the
    fixed-statistic ``bias_free_batch_norm`` -> DEGREE-1; the ``z_ctx``
    (context) stream uses the per-input, scale-invariant
    ``zero_centered_rms_norm`` -> DEGREE-0. Their geometric product is then
    degree-1 (degree-1 (X) degree-0), so the block is degree-1 homogeneous.

    Returns:
        Dict of CliffordNetBlock ctor kwargs pinning: no bias, BiasFreeBatchNorm
        for the input (detail) norm, ZeroCenteredRMSNorm for the context norm,
        LeakyReLU on all three activation axes, and the multiplicative gate
        removed.
    """
    # DECISION plan_2026-07-01_6dc255c1/D-005: the context (z_ctx) stream MUST be
    # degree-0. Do NOT set normalization_type to a degree-1 norm
    # (e.g. "bias_free_batch_norm") here, and do NOT use ctx_mode="diff" (which
    # subtracts the degree-1 z_det from z_ctx): either remixes the degrees and
    # makes the bilinear geometric product DEGREE-2 (f(alpha*x) ~ alpha^2 f(x)),
    # empirically confirmed in D-004. zero_centered_rms_norm is per-input and
    # scale-invariant -> degree-0; ctx_mode="abs" keeps z_ctx pure. See
    # decisions.md D-004/D-005/D-006.
    return dict(
        use_bias=False,
        input_normalization_type="bias_free_batch_norm",   # z_det -> degree-1
        normalization_type="zero_centered_rms_norm",        # z_ctx -> degree-0
        activation="leaky_relu",
        dot_activation="leaky_relu",
        feature_activation="leaky_relu",
        use_gate=False,
        use_global_context=False,
    )


def _size_shifts_for_level(base_shifts: List[int], channels: int) -> List[int]:
    """Clamp a base shift list so every kept shift satisfies ``s < channels``.

    ``SparseRollingGeometricProduct`` filters ``s >= channels`` with a warning
    (a full cyclic roll carries no new information). At coarse/narrow levels this
    could drop shifts; this helper sizes the list per level and falls back to
    ``[1]`` when nothing survives (requires ``channels >= 2``).

    Args:
        base_shifts: The model-level base shift offsets (ints >= 1).
        channels: The channel width at this U-Net level.

    Returns:
        A non-empty list of valid shift offsets, all ``< channels``.

    Raises:
        ValueError: If no valid shift can be produced (``channels < 2``).
    """
    valid = [int(s) for s in base_shifts if int(s) < channels]
    if not valid:
        if channels >= 2:
            valid = [1]
        else:
            raise ValueError(
                f"Cannot size shifts for a level with channels={channels}: "
                f"every base shift {list(base_shifts)} is >= channels and no "
                f"fallback shift < channels exists (need channels >= 2)."
            )
    return valid


def _downsample_and_skip(
    x: keras.KerasTensor,
    use_laplacian_pyramid: bool,
    laplacian_kernel_size: Tuple[int, int],
    downsample_name: str,
    pyramid_name: str,
    pool_type: str,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Produce ``(skip, downsampled)`` for one encoder junction.

    Mirrors ``bfconvunext._downsample_and_skip`` (reusing the SAME
    ``LaplacianPyramidLevel`` layer) so the Clifford U-Net's encoder junctions
    behave identically. OFF path: raw pre-downsample skip + linear
    (``average``) or non-linear (``max``) pooling. ON path: a bias-free
    ``LaplacianPyramidLevel`` split (high band -> skip, low band -> descend).
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
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_cliffordunet_denoiser(
        input_shape: Tuple[int, int, int],
        depth: int = 3,
        initial_filters: int = 32,
        filter_multiplier: float = 2.0,
        blocks_per_level: int = 2,
        shifts: Union[List[int], Tuple[int, ...]] = (1, 2),
        cli_mode: str = "full",
        ctx_mode: str = "abs",
        layer_scale_init: float = 1e-5,
        use_gabor_stem: bool = False,
        gabor_filters: int = 32,
        gabor_kernel_size: Union[int, Tuple[int, int]] = 7,
        gabor_stem_projection: bool = True,
        use_laplacian_pyramid: bool = False,
        high_freq_blocks: int = 0,
        laplacian_kernel_size: Tuple[int, int] = (5, 5),
        zero_pad_channels: bool = False,
        final_projection_groups: int = 1,
        downsample_pool_type: str = "max",
        expose_bottleneck: bool = False,
        enable_deep_supervision: bool = False,
        final_activation: Union[str, callable] = 'linear',
        drop_path_rate: float = 0.1,
        # Scale-preserving (norm-preserving) init for the external 1x1 projection
        # convs. With the residual Clifford trunk, these convs + concatenations
        # must NOT amplify variance; 'orthogonal' preserves the activation norm
        # and stays bias-free (a linear, homogeneous map). Mirrors bfconvunext.
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'orthogonal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        model_name: str = 'cliffordunet',
) -> keras.Model:
    """Create a bias-free, strictly-homogeneous Clifford U-Net image denoiser.

    Builds a full encoder / bottleneck / decoder U-Net (mirroring the
    ``create_convunext_denoiser`` topology) whose every block site is a
    :class:`CliffordNetBlock` in the bias-free degree-1-homogeneous
    configuration: a DEGREE-1 detail stream (``bias_free_batch_norm``) times a
    DEGREE-0 context stream (``zero_centered_rms_norm`` + ``ctx_mode="abs"``)
    through the bilinear geometric product yields a DEGREE-1 block (LeakyReLU +
    gate removed). The model is scaling-covariant in the operating regime: at
    inference, scaling the input by ``alpha`` scales the output by ``alpha``
    (``f(alpha * x) = alpha * f(x)``), enabling the Miyasawa/Tweedie
    residual-as-score interpretation. Homogeneity holds to ``rel_err < 1e-2``
    for ``alpha in [0.5, 1000]`` (the ``[-0.5, 0.5]`` regime); it degrades at
    extreme small ``alpha ~ 1e-3`` due to the per-input RMS epsilon floor --
    the same accepted deviation class as the Miyasawa clip-boundary caveat
    (D-005 / D-006).

    Args:
        input_shape: Tuple ``(height, width, channels)`` of the input images.
            ``height`` and ``width`` must be divisible by ``2 ** depth``.
        depth: Number of downsampling levels. Defaults to 3.
        initial_filters: Channel width at the finest (level-0) resolution.
            Defaults to 32.
        filter_multiplier: Float, per-encoder-level channel-growth multiplier
            (``>= 1``). Channels at level ``i`` are
            ``int(round(initial_filters * filter_multiplier ** i))``. Defaults to
            ``2.0`` (doubles per level, byte-identical to the historical int ``2``).
        blocks_per_level: Number of Clifford blocks per level. Defaults to 2.
        shifts: Base channel-shift offsets for the Clifford geometric product
            (ints >= 1). Sized per level so every kept shift satisfies
            ``s < channels`` at that level. Defaults to ``(1, 2)``.
        cli_mode: Algebraic components for the local interaction; one of
            ``"inner"``, ``"wedge"``, ``"full"``. Defaults to ``"full"``.
        ctx_mode: Context mode; one of ``"diff"``, ``"abs"``. Defaults to
            ``"abs"`` (HOMOGENEITY-CRITICAL): ``"abs"`` keeps the degree-0
            context stream pure, whereas ``"diff"`` subtracts the degree-1
            ``z_det`` from ``z_ctx`` and remixes the degrees, making the
            geometric product degree-2. See D-005.
        layer_scale_init: Initial LayerScale gamma for the Clifford GGR.
            Defaults to ``1e-5``.
        use_gabor_stem: If True, prepend a frozen (non-learnable) Gabor depthwise
            convolution stem (bias-free) followed by a mandatory bias-free 1x1
            projection to ``initial_filters``. Defaults to False. Contributes
            zero trainable parameters.
        gabor_filters: Depth multiplier for the Gabor depthwise stem; the stem
            emits ``input_channels * gabor_filters`` channels. Only used when
            ``use_gabor_stem=True``. Defaults to 32.
        gabor_kernel_size: Kernel size of the Gabor depthwise stem. Only used
            when ``use_gabor_stem=True``. Defaults to 7.
        gabor_stem_projection: If True (default) the Gabor stem is followed by
            the mandatory bias-free 1x1 projection to ``initial_filters``. If
            False the projection is DROPPED (valid only when
            ``input_channels * gabor_filters == initial_filters`` exactly).
        use_laplacian_pyramid: If True, replace each encoder downsample/skip
            junction with a bias-free ``LaplacianPyramidLevel`` split. Defaults
            to False.
        high_freq_blocks: Number of bias-free ``CliffordNetBlock`` blocks
            (externalized residual) applied to the Laplacian high-frequency skip
            band at each encoder level before it becomes the decoder skip.
            **Ignored when use_laplacian_pyramid=False.** Defaults to 0
            (byte-identical no-op: adds ZERO layers, renames nothing).
        laplacian_kernel_size: Gaussian blur kernel size for the Laplacian
            pyramid split. Only used when ``use_laplacian_pyramid=True``.
            Defaults to ``(5, 5)``.
        zero_pad_channels: If True, replace every per-level channel-adjust 1x1
            convolution with a parameter-free ``MatchChannels`` (encoder /
            bottleneck zero-pad; decoder slice-upsampled + add-skip). Bias-free /
            homogeneous. Defaults to False.
        final_projection_groups: Number of groups for the final 1x1 output
            projection. Defaults to 1 (standard dense 1x1). Must divide BOTH the
            final-projection input channels (``initial_filters``) and
            ``output_channels`` when > 1.
        downsample_pool_type: ``'max'`` (default, non-linear) or ``'average'``
            (linear, Miyasawa-clean) pooling for the non-Laplacian encoder
            downsample. Ignored when ``use_laplacian_pyramid=True``.
        expose_bottleneck: If True, expose the deepest-encoder bottleneck latent
            as a trailing model output. When True, a zero-parameter
            ``Activation('linear', name='bottleneck')`` tap is inserted after the
            bottleneck blocks (also enabling ConvUnextBottleneckMonitorCallback
            reuse, which looks up ``model.get_layer("bottleneck")``) and emitted
            as a second output. When False (default) no tap layer is added — the
            decoder consumes the bottleneck blocks' output directly. Defaults to
            False.
        enable_deep_supervision: Accepted for signature parity only. Deep
            supervision is NOT wired here; ``True`` raises ``NotImplementedError``.
        final_activation: Final activation. HARD invariant: keep ``'linear'`` for
            bias-free homogeneity. Defaults to ``'linear'``.
        drop_path_rate: Stochastic-depth drop probability (progressive across
            depth). Defaults to 0.1.
        kernel_initializer: Initializer for the external 1x1 projection convs.
            Defaults to ``'orthogonal'``.
        kernel_regularizer: Regularizer for the external 1x1 projection convs.
            Defaults to None.
        model_name: Name for the model. Defaults to ``'cliffordunet'``.

    Returns:
        keras.Model: A functional CliffordUNet denoiser. Single output
        (``final_output``) unless ``expose_bottleneck=True``, in which case the
        outputs are ``[final_output, bottleneck]``.

    Raises:
        TypeError: If ``input_shape`` is not a length-3 tuple.
        ValueError: On invalid ``depth`` / ``initial_filters`` /
            ``filter_multiplier`` / ``blocks_per_level`` / ``cli_mode`` /
            ``ctx_mode`` / ``downsample_pool_type`` / ``final_projection_groups``.
        NotImplementedError: If ``enable_deep_supervision=True``.
    """

    # --- Input validation ---
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

    if cli_mode not in ('inner', 'wedge', 'full'):
        raise ValueError(f"cli_mode must be 'inner', 'wedge', or 'full', got {cli_mode!r}")

    if ctx_mode not in ('diff', 'abs'):
        raise ValueError(f"ctx_mode must be 'diff' or 'abs', got {ctx_mode!r}")

    if downsample_pool_type not in ('max', 'average'):
        raise ValueError(
            f"downsample_pool_type must be 'max' or 'average', got {downsample_pool_type}"
        )

    if enable_deep_supervision:
        raise NotImplementedError(
            "enable_deep_supervision is not wired for the CliffordUNet denoiser "
            "(the trainer fail-fasts on it). Pass enable_deep_supervision=False."
        )

    base_shifts = [int(s) for s in shifts]
    if not base_shifts:
        raise ValueError("shifts must be a non-empty list of ints >= 1")

    block_kwargs = _homogeneous_block_kwargs()

    # --- Input layer ---
    inputs = keras.Input(shape=input_shape, name='input_images')

    # --- Optional frozen Gabor stem (bias-free) + mandatory 1x1 projection ---
    # Reuses the exact bfconvunext contract, including the no-projection
    # channel-match guard (in_channels * gabor_filters == initial_filters).
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
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='gabor_stem_projection',
            )(gabor)
            logger.info(
                f"Frozen Gabor stem enabled: filters={gabor_filters}, "
                f"kernel_size={gabor_kernel_size} -> 1x1 projection to {initial_filters}"
            )
        else:
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

    # --- Per-level filter sizes ---
    filter_sizes = [int(round(initial_filters * (filter_multiplier ** i))) for i in range(depth + 1)]

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

    # --- Storage ---
    skip_connections: List[keras.KerasTensor] = []
    # Record the per-level valid shift lists so the caller/test can assert them.
    level_shift_map: Dict[str, List[int]] = {}

    def _channel_adjust(t, filters, name):
        """External bias-free channel projection (isotropic-block boundary)."""
        if t.shape[-1] == filters:
            return t
        if zero_pad_channels:
            return MatchChannels(filters, name=f'{name}_match_channels')(t)
        return keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f'{name}_channel_adjust',
        )(t)

    # =========================================================================
    # ENCODER PATH (Contracting)
    # =========================================================================

    x = stem_input
    logger.info(f"Building CliffordUNet encoder path with {depth} levels")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"Encoder level {level}: {current_filters} filters")

        # Channel adjust to level width (also serves as the level-0 stem when
        # there is no Gabor stem). Bias-free / homogeneous.
        x = _channel_adjust(x, current_filters, f'encoder_level_{level}')

        level_shifts = _size_shifts_for_level(base_shifts, current_filters)
        level_shift_map[f'encoder_level_{level}'] = level_shifts

        for block_idx in range(blocks_per_level):
            current_drop_path = drop_path_rate * (
                level * blocks_per_level + block_idx
            ) / (depth * blocks_per_level)
            block_name = f'encoder_level_{level}_clifford_block_{block_idx}'
            # Externalized residual: the Clifford block is transform-only now; the
            # identity residual and stochastic-depth op are explicit model-level
            # ops so the graph is manually inspectable. Bias-free-safe (Add of the
            # pre-block tensor + StochasticDepth are homogeneous, no additive bias).
            h = CliffordNetBlock(
                channels=current_filters,
                shifts=level_shifts,
                cli_mode=cli_mode,
                ctx_mode=ctx_mode,
                layer_scale_init=layer_scale_init,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=block_name,
                **block_kwargs,
            )(x)
            if current_drop_path and current_drop_path > 0.0:
                h = StochasticDepth(current_drop_path, name=f'{block_name}_drop_path')(h)
            x = keras.layers.Add(name=f'{block_name}_residual')([x, h])

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

        # DECISION plan_2026-07-06_b17c1f83/D-001: optionally process the Laplacian
        # high-frequency band with N bias-free Clifford blocks before it becomes the
        # decoder skip. CliffordNetBlock is TRANSFORM-ONLY -> external residual Add is
        # mandatory (or the residual silently vanishes). Gated on use_laplacian_pyramid
        # (the high band only exists then); high_freq_blocks=0 (default) adds ZERO layers
        # -> byte-identical OFF path, so existing `.keras` checkpoints (whose layer names
        # are load-bearing) still load. Do NOT drop the use_laplacian_pyramid gate or the
        # >0 gate: without the pyramid there is no high band and this would rename/insert
        # layers into the raw-skip path. The Laplacian split is channel-preserving, so the
        # high band has current_filters channels and level_shifts stays valid (s < channels).
        # No StochasticDepth here (drop_path 0) -> deterministic round-trip.
        if high_freq_blocks > 0 and use_laplacian_pyramid:
            for hf_idx in range(high_freq_blocks):
                hf_block_name = f'skip_highfreq_block_{level}_{hf_idx}'
                hf = CliffordNetBlock(
                    channels=current_filters,
                    shifts=level_shifts,
                    cli_mode=cli_mode,
                    ctx_mode=ctx_mode,
                    layer_scale_init=layer_scale_init,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name=hf_block_name,
                    **block_kwargs,
                )(skip)
                skip = keras.layers.Add(name=f'{hf_block_name}_residual')([skip, hf])

        skip_connections.append(skip)

    # =========================================================================
    # BOTTLENECK
    # =========================================================================

    bottleneck_filters = filter_sizes[depth]
    logger.info(f"Building CliffordUNet bottleneck with {bottleneck_filters} filters")

    x = _channel_adjust(x, bottleneck_filters, 'bottleneck')

    bottleneck_shifts = _size_shifts_for_level(base_shifts, bottleneck_filters)
    level_shift_map['bottleneck'] = bottleneck_shifts

    # DECISION plan_2026-07-11_4426773d/D-001: the bottleneck now uses a LOCAL linear
    # drop-path ramp that restarts at 0.0 (block 0 -> 0.0, later blocks ramp up but stay
    # strictly < drop_path_rate), mirroring this file's encoder ramp shape and the ConvUNext
    # bottleneck change (plan_2026-07-10_be906be8/D-001). This replaces the prior FLAT
    # unscaled drop_path_rate applied to every bottleneck block. The ramp
    # `drop_path_rate * block_idx / blocks_per_level` can NEVER exceed drop_path_rate, and
    # blocks_per_level >= 1 is validated above so the denominator is never zero.
    # StochasticDepth is inference-identity, so existing trained checkpoints load and infer
    # unchanged (block_0 only drops a weightless SD sublayer). Do NOT revert to a flat rate.
    for block_idx in range(blocks_per_level):
        current_drop_path = drop_path_rate * block_idx / blocks_per_level
        block_name = f'bottleneck_clifford_block_{block_idx}'
        # Externalized residual (transform-only block); see encoder note.
        h = CliffordNetBlock(
            channels=bottleneck_filters,
            shifts=bottleneck_shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            layer_scale_init=layer_scale_init,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=block_name,
            **block_kwargs,
        )(x)
        if current_drop_path and current_drop_path > 0.0:
            h = StochasticDepth(current_drop_path, name=f'{block_name}_drop_path')(h)
        x = keras.layers.Add(name=f'{block_name}_residual')([x, h])

    # Bottleneck latent (deepest-encoder features). The named linear tap is only
    # needed when the bottleneck is exposed (second output + ConvUnextBottleneckMonitorCallback,
    # which looks up model.get_layer("bottleneck")). Inserting it unconditionally added a
    # redundant no-op Activation('linear') on the default expose_bottleneck=False path;
    # guard it to match bfconvunext (which also only taps under expose_bottleneck).
    bottleneck_output = x
    if expose_bottleneck:
        x = keras.layers.Activation('linear', name='bottleneck')(x)
        bottleneck_output = x

    # =========================================================================
    # DECODER PATH (Expanding)
    # =========================================================================

    logger.info(f"Building CliffordUNet decoder path with {depth} levels")
    output_channels = input_shape[-1]

    for level in range(depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        logger.info(f"Decoder level {level}: {current_filters} filters")

        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation='bilinear',
            name=f'decoder_upsample_{level}',
        )(x)

        skip = skip_connections[level]

        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            x = keras.layers.Resizing(
                height=skip.shape[1],
                width=skip.shape[2],
                interpolation='bilinear',
                name=f'decoder_resize_{level}',
            )(x)

        # Merge skip. Under zero_pad_channels: slice the upsampled tensor to
        # current_filters and ADD the skip (parameter-free, bias-free). Otherwise:
        # Concatenate + bias-free 1x1 channel-adjust (mirrors bfconvunext).
        if zero_pad_channels:
            x = keras.layers.Add(name=f'decoder_level_{level}_match_add')(
                [skip, MatchChannels(current_filters, name=f'decoder_level_{level}_match_channels')(x)]
            )
        else:
            x = keras.layers.Concatenate(
                axis=-1, name=f'decoder_concat_{level}'
            )([skip, x])
            x = _channel_adjust(x, current_filters, f'decoder_level_{level}')

        level_shifts = _size_shifts_for_level(base_shifts, current_filters)
        level_shift_map[f'decoder_level_{level}'] = level_shifts

        for block_idx in range(blocks_per_level):
            # First decoder block at each level carries no stochastic depth;
            # the rest keep the progressive rate (decoder-only, mirrors bfconvunext).
            if block_idx == 0:
                current_drop_path = 0.0
            else:
                current_drop_path = drop_path_rate * (
                    level * blocks_per_level + block_idx
                ) / (depth * blocks_per_level)
            block_name = f'decoder_level_{level}_clifford_block_{block_idx}'
            # Externalized residual (transform-only block); see encoder note.
            h = CliffordNetBlock(
                channels=current_filters,
                shifts=level_shifts,
                cli_mode=cli_mode,
                ctx_mode=ctx_mode,
                layer_scale_init=layer_scale_init,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=block_name,
                **block_kwargs,
            )(x)
            if current_drop_path and current_drop_path > 0.0:
                h = StochasticDepth(current_drop_path, name=f'{block_name}_drop_path')(h)
            x = keras.layers.Add(name=f'{block_name}_residual')([x, h])

    # =========================================================================
    # FINAL OUTPUT LAYER (bias-free 1x1 projection)
    # =========================================================================

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
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name='final_output',
    )(x)

    # =========================================================================
    # MODEL CREATION
    # =========================================================================

    if expose_bottleneck:
        model = keras.Model(
            inputs=inputs,
            outputs=[final_output, bottleneck_output],
            name=model_name,
        )
    else:
        model = keras.Model(inputs=inputs, outputs=final_output, name=model_name)

    # Expose the per-level valid shift map for external verification (SC3).
    model.cliffordunet_level_shifts = level_shift_map

    logger.info(f"Created CliffordUNet model '{model_name}' with depth {depth}")
    logger.info(f"Filter progression: {filter_sizes}")
    logger.info(f"Per-level shifts: {level_shift_map}")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")
    logger.info(f"Drop path rate: {drop_path_rate}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model


# ---------------------------------------------------------------------
# Variant Creation Function
# ---------------------------------------------------------------------

def create_cliffordunet_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        **kwargs: Any,
) -> keras.Model:
    """Create a CliffordUNet denoiser from a named variant.

    Args:
        variant: One of ``'tiny'``, ``'small'``, ``'base'``.
        input_shape: Tuple ``(height, width, channels)``.
        **kwargs: Overrides forwarded to :func:`create_cliffordunet_denoiser`
            (win over the variant defaults).

    Returns:
        keras.Model: The configured CliffordUNet denoiser.

    Raises:
        ValueError: If ``variant`` is not a key of ``CLIFFORDUNET_CONFIGS``.
    """
    if variant not in CLIFFORDUNET_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: "
            f"{list(CLIFFORDUNET_CONFIGS.keys())}"
        )

    config = CLIFFORDUNET_CONFIGS[variant].copy()
    description = config.pop('description')
    config.update(kwargs)

    if 'model_name' not in config:
        config['model_name'] = f'cliffordunet_{variant}'

    logger.info(f"Creating CliffordUNet variant '{variant}': {description}")

    return create_cliffordunet_denoiser(input_shape=input_shape, **config)
