"""
CliffordNet confidence-interval denoiser with Miyasawa's theorem compliance.

Extends the conditional CliffordNet denoiser to produce **confidence
intervals** instead of point estimates.  Two uncertainty estimation modes
are supported:

- **Gaussian heteroscedastic** (``uncertainty_mode="gaussian"``):
  Dual output heads predict mean and per-pixel log-variance.  Trained
  with Gaussian negative log-likelihood (NLL) so the network learns
  both the conditional expectation *and* the conditional variance:

      mu(y, c)     = E[x | y, c]          (Miyasawa-compliant, bias-free)
      sigma^2(y,c) = Var[x | y, c]        (second-order, may use bias)

  Confidence interval: mu +/- z_{alpha/2} * sigma.

- **Quantile regression** (``uncertainty_mode="quantile"``):
  Multiple output heads predict specified quantiles (default 5th, 50th,
  95th percentile).  Trained with pinball / quantile loss.  Does not
  assume Gaussian errors; captures asymmetric uncertainty.

  Confidence interval: [q_low, q_high].

All backbone layers remain strictly bias-free to satisfy Miyasawa's
theorem for the mean / median head.  The variance head is exempt from
the bias-free constraint because it estimates a second-order statistic,
not the score function.

Architecture::

    Inputs: noisy_target (B,H,W,C_target)
          + dense_condition (B,H,W,C_cond) [optional]
          + class_label (B,1) [optional]
        |
    Bias-free stem -> (B, H/2, W/2, ch[0])
        |
    Shared encoder/bottleneck/decoder backbone
    (identical to CliffordNetConditionalDenoiser)
        |
        +-- Mean head: LN(center=False) -> Conv2D 1x1 (bias-free)
        |       output = noisy_target + residual
        |
        +-- Variance head (gaussian mode):
        |       LN -> Conv2D 1x1 -> softplus clamp
        |       output = log_variance  (B,H,W,C_target)
        |
        +-- Quantile heads (quantile mode):
                LN -> Conv2D 1x1 per quantile
                output = list of quantile maps

Pre-defined variants
--------------------
- ``CliffordNetConfidenceDenoiser.tiny``  -- 3 levels, ~2.1M params
- ``CliffordNetConfidenceDenoiser.small`` -- 3 levels, ~5.3M params
- ``CliffordNetConfidenceDenoiser.base``  -- 4 levels, ~12.5M params
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    SparseRollingGeometricProduct,
)
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.utils.logger import logger

from .conditional_denoiser import (
    BiasFreeConditionedCliffordBlock,
    BiasFreeConditionedGGR,
    _linear_drop_path_rates,
    _DEFAULT_KERNEL_INIT,
)

# Allowed uncertainty modes
UncertaintyMode = str  # "gaussian" | "quantile"

# Default quantiles for quantile regression (90% CI)
_DEFAULT_QUANTILES = [0.05, 0.50, 0.95]

# Minimum variance floor to prevent numerical instability
_MIN_LOG_VARIANCE = -10.0
_MAX_LOG_VARIANCE = 10.0


# ===========================================================================
# Custom losses
# ===========================================================================


@keras.saving.register_keras_serializable()
class GaussianNLLLoss(keras.losses.Loss):
    """Gaussian negative log-likelihood loss for heteroscedastic regression.

    Given predicted mean ``mu`` and log-variance ``log_var``, computes::

        L = 0.5 * [log_var + (target - mu)^2 / exp(log_var)]

    The model output is expected to be ``(mu, log_var)`` concatenated
    along the last axis, i.e. shape ``(B, H, W, 2*C)``.

    :param in_channels: Number of target channels (to split mu / log_var).
    :param variance_regularization: Weight for variance regularization term
        that penalizes very large variance (prevents collapse to
        infinite uncertainty).
    """

    def __init__(
        self,
        in_channels: int = 1,
        variance_regularization: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.variance_regularization = variance_regularization

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute Gaussian NLL.

        :param y_true: Clean target ``(B, H, W, C)``.
        :param y_pred: Concatenated ``[mu, log_var]`` along last axis
            ``(B, H, W, 2*C)``.
        :return: Scalar loss.
        """
        mu = y_pred[..., :self.in_channels]
        log_var = y_pred[..., self.in_channels:]

        # Clamp log-variance for numerical stability
        log_var = keras.ops.clip(log_var, _MIN_LOG_VARIANCE, _MAX_LOG_VARIANCE)

        # Gaussian NLL: 0.5 * [log_var + (y - mu)^2 / exp(log_var)]
        precision = keras.ops.exp(-log_var)
        squared_error = keras.ops.square(y_true - mu)
        nll = 0.5 * (log_var + squared_error * precision)

        # Variance regularization: penalize very large variance
        var_reg = self.variance_regularization * keras.ops.mean(
            keras.ops.abs(log_var)
        )

        return keras.ops.mean(nll) + var_reg

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "variance_regularization": self.variance_regularization,
        })
        return config


@keras.saving.register_keras_serializable()
class PinballLoss(keras.losses.Loss):
    """Pinball (quantile) loss for quantile regression.

    For a target quantile tau::

        L_tau = max(tau * (y - q), (tau - 1) * (y - q))

    The model output is expected to be quantile predictions concatenated
    along the last axis: ``(B, H, W, num_quantiles * C)``.

    :param quantiles: List of quantile levels (e.g., [0.05, 0.50, 0.95]).
    :param in_channels: Number of target channels.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        in_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.quantiles = list(quantiles or _DEFAULT_QUANTILES)
        self.in_channels = in_channels

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute pinball loss across all quantiles.

        :param y_true: Clean target ``(B, H, W, C)``.
        :param y_pred: Concatenated quantile predictions
            ``(B, H, W, num_quantiles * C)``.
        :return: Scalar loss.
        """
        total_loss = keras.ops.zeros(())
        num_q = len(self.quantiles)

        for i, tau in enumerate(self.quantiles):
            q_pred = y_pred[..., i * self.in_channels:(i + 1) * self.in_channels]
            error = y_true - q_pred
            loss_i = keras.ops.maximum(tau * error, (tau - 1.0) * error)
            total_loss = total_loss + keras.ops.mean(loss_i)

        return total_loss / num_q

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "quantiles": self.quantiles,
            "in_channels": self.in_channels,
        })
        return config


# ===========================================================================
# Custom metrics for confidence intervals
# ===========================================================================


@keras.saving.register_keras_serializable()
class CoverageMetric(keras.metrics.Metric):
    """Empirical coverage: fraction of true values inside predicted CI.

    For Gaussian mode, expects concatenated ``[mu, log_var]``.
    Computes the fraction of pixels where::

        |y_true - mu| <= z_{alpha/2} * exp(0.5 * log_var)

    :param confidence_level: Target coverage (e.g., 0.90 for 90% CI).
    :param in_channels: Number of target channels.
    :param uncertainty_mode: ``"gaussian"`` or ``"quantile"``.
    :param quantiles: Quantile levels (only for quantile mode).
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        in_channels: int = 1,
        uncertainty_mode: str = "gaussian",
        quantiles: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.confidence_level = confidence_level
        self.in_channels = in_channels
        self.uncertainty_mode = uncertainty_mode
        self.quantiles = list(quantiles or _DEFAULT_QUANTILES)

        # z-score for the given confidence level (Gaussian)
        # Approximate: 1.645 for 90%, 1.96 for 95%, 2.576 for 99%
        import math
        alpha = 1.0 - confidence_level
        # Inverse normal CDF approximation (Abramowitz & Stegun)
        p = 1.0 - alpha / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        self.z_score = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
            1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
        )

        self.total_covered = self.add_weight(
            name="total_covered", initializer="zeros"
        )
        self.total_count = self.add_weight(
            name="total_count", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.uncertainty_mode == "gaussian":
            mu = y_pred[..., :self.in_channels]
            log_var = y_pred[..., self.in_channels:]
            log_var = keras.ops.clip(
                log_var, _MIN_LOG_VARIANCE, _MAX_LOG_VARIANCE
            )
            sigma = keras.ops.exp(0.5 * log_var)
            lower = mu - self.z_score * sigma
            upper = mu + self.z_score * sigma
        else:
            # Quantile mode: use first and last quantile as bounds
            lower = y_pred[..., :self.in_channels]
            upper = y_pred[..., -self.in_channels:]

        covered = keras.ops.logical_and(
            y_true >= lower, y_true <= upper
        )
        covered = keras.ops.cast(covered, "float32")

        count = keras.ops.cast(
            keras.ops.prod(keras.ops.convert_to_tensor(keras.ops.shape(y_true))),
            "float32",
        )
        self.total_covered.assign_add(keras.ops.sum(covered))
        self.total_count.assign_add(count)

    def result(self):
        return self.total_covered / keras.ops.maximum(self.total_count, 1.0)

    def reset_state(self):
        self.total_covered.assign(0.0)
        self.total_count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "confidence_level": self.confidence_level,
            "in_channels": self.in_channels,
            "uncertainty_mode": self.uncertainty_mode,
            "quantiles": self.quantiles,
        })
        return config


@keras.saving.register_keras_serializable()
class IntervalWidthMetric(keras.metrics.Metric):
    """Mean width of predicted confidence intervals.

    Smaller is better (sharper intervals), but must maintain coverage.

    :param in_channels: Number of target channels.
    :param uncertainty_mode: ``"gaussian"`` or ``"quantile"``.
    :param confidence_level: For Gaussian mode, the CI level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        uncertainty_mode: str = "gaussian",
        confidence_level: float = 0.90,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.uncertainty_mode = uncertainty_mode
        self.confidence_level = confidence_level

        import math
        alpha = 1.0 - confidence_level
        p = 1.0 - alpha / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        self.z_score = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
            1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
        )

        self.total_width = self.add_weight(
            name="total_width", initializer="zeros"
        )
        self.total_count = self.add_weight(
            name="total_count", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.uncertainty_mode == "gaussian":
            log_var = y_pred[..., self.in_channels:]
            log_var = keras.ops.clip(
                log_var, _MIN_LOG_VARIANCE, _MAX_LOG_VARIANCE
            )
            sigma = keras.ops.exp(0.5 * log_var)
            width = 2.0 * self.z_score * sigma
        else:
            lower = y_pred[..., :self.in_channels]
            upper = y_pred[..., -self.in_channels:]
            width = upper - lower

        count = keras.ops.cast(
            keras.ops.prod(keras.ops.convert_to_tensor(keras.ops.shape(width))),
            "float32",
        )
        self.total_width.assign_add(keras.ops.sum(width))
        self.total_count.assign_add(count)

    def result(self):
        return self.total_width / keras.ops.maximum(self.total_count, 1.0)

    def reset_state(self):
        self.total_width.assign(0.0)
        self.total_count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "uncertainty_mode": self.uncertainty_mode,
            "confidence_level": self.confidence_level,
        })
        return config


# ===========================================================================
# CliffordNetConfidenceDenoiser
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordNetConfidenceDenoiser(keras.Model):
    """Bias-free conditional CliffordNet denoiser with confidence intervals.

    Extends :class:`CliffordNetConditionalDenoiser` with a secondary
    output head that estimates per-pixel uncertainty, enabling
    construction of confidence intervals around the denoised prediction.

    Two uncertainty estimation strategies are supported:

    **Gaussian heteroscedastic** (``uncertainty_mode="gaussian"``):
        The model outputs ``(B, H, W, 2*in_channels)`` where the first
        ``in_channels`` are the denoised mean (bias-free, Miyasawa-
        compliant) and the remaining ``in_channels`` are the predicted
        log-variance.  Trained with :class:`GaussianNLLLoss`.

    **Quantile regression** (``uncertainty_mode="quantile"``):
        The model outputs ``(B, H, W, num_quantiles*in_channels)`` where
        each group of ``in_channels`` corresponds to one quantile level.
        Trained with :class:`PinballLoss`.  Default quantiles are
        ``[0.05, 0.50, 0.95]`` (90% prediction interval).

    In both modes the shared backbone (encoder, bottleneck, decoder) is
    bias-free.  The mean head is strictly bias-free (Miyasawa-compliant).
    The variance / quantile heads may use bias since they do not estimate
    the score function.

    :param in_channels: Number of input/output channels.
    :param uncertainty_mode: ``"gaussian"`` or ``"quantile"``.
    :param quantiles: Quantile levels for quantile mode.
    :param confidence_level: Target CI level (for metrics / inference).
    :param level_channels: Channel count at each encoder level.
    :param level_blocks: Number of CliffordNet blocks per level.
    :param level_shifts: Shift patterns per level.
    :param cli_mode: Geometric product mode.
    :param ctx_mode: Context mode.
    :param use_global_context: Add global branch.
    :param layer_scale_init: Initial LayerScale value.
    :param stochastic_depth_rate: Maximum drop-path rate.
    :param enable_dense_conditioning: Accept dense conditioning input.
    :param dense_cond_channels: Dense conditioning channel count.
    :param enable_discrete_conditioning: Accept discrete conditioning.
    :param num_classes: Number of classes for discrete conditioning.
    :param class_embedding_dim: Dimension of class embedding.
    :param variance_regularization: Weight for variance regularization
        in Gaussian NLL loss.
    :param kernel_initializer: Kernel initializer.
    :param kernel_regularizer: Kernel regularizer.
    """

    LAYERNORM_EPSILON: float = 1e-6

    def __init__(
        self,
        in_channels: int = 1,
        uncertainty_mode: UncertaintyMode = "gaussian",
        quantiles: Optional[List[float]] = None,
        confidence_level: float = 0.90,
        level_channels: Optional[List[int]] = None,
        level_blocks: Optional[List[int]] = None,
        level_shifts: Optional[List[List[int]]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        enable_dense_conditioning: bool = False,
        dense_cond_channels: Optional[int] = None,
        enable_discrete_conditioning: bool = False,
        num_classes: int = 0,
        class_embedding_dim: int = 128,
        variance_regularization: float = 0.01,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )
        if uncertainty_mode not in ("gaussian", "quantile"):
            raise ValueError(
                f"uncertainty_mode must be 'gaussian' or 'quantile', "
                f"got '{uncertainty_mode}'"
            )

        # Defaults
        if level_channels is None:
            level_channels = [64, 128, 256]
        if level_blocks is None:
            level_blocks = [2, 2, 4]
        if level_shifts is None:
            level_shifts = [[1, 2], [1, 2, 4], [1, 2, 4, 8]]
        if quantiles is None:
            quantiles = list(_DEFAULT_QUANTILES)

        if len(level_channels) != len(level_blocks):
            raise ValueError(
                f"level_channels ({len(level_channels)}) and "
                f"level_blocks ({len(level_blocks)}) must have same length"
            )
        if len(level_shifts) != len(level_channels):
            raise ValueError(
                f"level_shifts ({len(level_shifts)}) and "
                f"level_channels ({len(level_channels)}) must have same length"
            )

        # Store config
        self.in_channels = in_channels
        self.uncertainty_mode = uncertainty_mode
        self.quantiles = list(quantiles)
        self.confidence_level = confidence_level
        self.level_channels = list(level_channels)
        self.level_blocks = list(level_blocks)
        self.level_shifts = [list(s) for s in level_shifts]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.enable_dense_conditioning = enable_dense_conditioning
        self.dense_cond_channels = dense_cond_channels
        self.enable_discrete_conditioning = enable_discrete_conditioning
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.variance_regularization = variance_regularization
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.num_levels = len(level_channels)
        self.bottleneck_channels = level_channels[-1]

        self._build_conditioning()
        self._build_stem()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_heads()

        logger.info(
            f"Created CliffordNetConfidenceDenoiser "
            f"(mode={uncertainty_mode}, levels={self.num_levels}, "
            f"channels={level_channels}, blocks={level_blocks}, "
            f"dense_cond={enable_dense_conditioning}, "
            f"discrete_cond={enable_discrete_conditioning})"
        )

    # ------------------------------------------------------------------
    # Private builder helpers
    # ------------------------------------------------------------------

    def _common_conv_kwargs(self) -> Dict[str, Any]:
        return dict(
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

    def _build_conditioning(self) -> None:
        """Build conditioning pathways."""
        if self.enable_dense_conditioning:
            if self.dense_cond_channels is None:
                raise ValueError(
                    "dense_cond_channels required when "
                    "enable_dense_conditioning=True"
                )
            self.cond_projections = []
            for i in range(self.num_levels):
                proj = keras.layers.Conv2D(
                    filters=self.level_channels[i],
                    kernel_size=1,
                    padding="same",
                    name=f"cond_proj_level_{i}",
                    **self._common_conv_kwargs(),
                )
                self.cond_projections.append(proj)
            self.cond_proj_bottleneck = keras.layers.Conv2D(
                filters=self.bottleneck_channels,
                kernel_size=1,
                padding="same",
                name="cond_proj_bottleneck",
                **self._common_conv_kwargs(),
            )

        if self.enable_discrete_conditioning:
            if self.num_classes <= 0:
                raise ValueError(
                    "num_classes must be positive when "
                    "enable_discrete_conditioning=True"
                )
            self.class_embedding = keras.layers.Embedding(
                input_dim=self.num_classes,
                output_dim=self.class_embedding_dim,
                embeddings_initializer="he_normal",
                name="class_embedding",
            )

    def _build_stem(self) -> None:
        """Bias-free stem: Conv2D stride-2 + BN(center=False)."""
        self.stem_conv = keras.layers.Conv2D(
            filters=self.level_channels[0],
            kernel_size=3,
            strides=2,
            padding="same",
            name="stem_conv",
            **self._common_conv_kwargs(),
        )
        self.stem_norm = keras.layers.BatchNormalization(
            center=False, name="stem_norm"
        )

    def _build_encoder(self) -> None:
        """Build encoder levels with conditioned CliffordNet blocks."""
        total_blocks = sum(self.level_blocks)
        drop_rates = _linear_drop_path_rates(
            total_blocks, self.stochastic_depth_rate
        )

        self.encoder_levels = []
        block_idx = 0

        for level in range(self.num_levels):
            ch = self.level_channels[level]
            shifts = self.level_shifts[level]
            n_blocks = self.level_blocks[level]

            transition = None
            if level > 0:
                transition = keras.layers.Conv2D(
                    filters=ch,
                    kernel_size=1,
                    padding="same",
                    name=f"enc_transition_{level}",
                    **self._common_conv_kwargs(),
                )

            blocks = []
            for b in range(n_blocks):
                block = BiasFreeConditionedCliffordBlock(
                    channels=ch,
                    shifts=shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_global_context=self.use_global_context,
                    layer_scale_init=self.layer_scale_init,
                    drop_path_rate=drop_rates[block_idx],
                    enable_dense_conditioning=self.enable_dense_conditioning,
                    enable_discrete_conditioning=(
                        self.enable_discrete_conditioning
                    ),
                    class_embedding_dim=self.class_embedding_dim,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"enc_block_{level}_{b}",
                )
                blocks.append(block)
                block_idx += 1

            downsample = None
            if level < self.num_levels - 1:
                downsample = keras.layers.Conv2D(
                    filters=ch,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    name=f"enc_downsample_{level}",
                    **self._common_conv_kwargs(),
                )

            self.encoder_levels.append({
                "transition": transition,
                "blocks": blocks,
                "downsample": downsample,
            })

    def _build_bottleneck(self) -> None:
        """Build bottleneck blocks at lowest resolution."""
        ch = self.bottleneck_channels
        shifts = self.level_shifts[-1]
        n_blocks = self.level_blocks[-1]
        total_blocks = sum(self.level_blocks)
        max_drop = self.stochastic_depth_rate

        self.bottleneck_blocks = []
        for b in range(n_blocks):
            rate = max_drop * (total_blocks - 1 + b) / max(
                total_blocks + n_blocks - 2, 1
            )
            block = BiasFreeConditionedCliffordBlock(
                channels=ch,
                shifts=shifts,
                cli_mode=self.cli_mode,
                ctx_mode=self.ctx_mode,
                use_global_context=True,
                layer_scale_init=self.layer_scale_init,
                drop_path_rate=rate,
                enable_dense_conditioning=self.enable_dense_conditioning,
                enable_discrete_conditioning=(
                    self.enable_discrete_conditioning
                ),
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"bottleneck_block_{b}",
            )
            self.bottleneck_blocks.append(block)

    def _build_decoder(self) -> None:
        """Build decoder levels with skip connections."""
        self.decoder_levels = []

        for level in range(self.num_levels - 1, -1, -1):
            ch = self.level_channels[level]
            shifts = self.level_shifts[level]
            n_blocks = self.level_blocks[level]

            upsample = None
            if level < self.num_levels - 1:
                upsample = keras.layers.UpSampling2D(
                    size=2, interpolation="bilinear",
                    name=f"dec_upsample_{level}",
                )

            skip_proj = keras.layers.Conv2D(
                filters=ch,
                kernel_size=1,
                padding="same",
                name=f"dec_skip_proj_{level}",
                **self._common_conv_kwargs(),
            )

            blocks = []
            for b in range(n_blocks):
                block = BiasFreeConditionedCliffordBlock(
                    channels=ch,
                    shifts=shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_global_context=self.use_global_context,
                    layer_scale_init=self.layer_scale_init,
                    drop_path_rate=0.0,
                    enable_dense_conditioning=self.enable_dense_conditioning,
                    enable_discrete_conditioning=(
                        self.enable_discrete_conditioning
                    ),
                    class_embedding_dim=self.class_embedding_dim,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"dec_block_{level}_{b}",
                )
                blocks.append(block)

            self.decoder_levels.append({
                "upsample": upsample,
                "skip_proj": skip_proj,
                "blocks": blocks,
                "level": level,
            })

    def _build_heads(self) -> None:
        """Build output heads: mean + uncertainty.

        In Gaussian mode: mean head (bias-free) + log-variance head.
        In quantile mode: one head per quantile (each adds its own
        residual to the noisy input, so no separate mean head is needed).
        """
        if self.uncertainty_mode == "gaussian":
            # --- Mean head (bias-free, Miyasawa-compliant) ---
            self.mean_norm = keras.layers.LayerNormalization(
                epsilon=self.LAYERNORM_EPSILON,
                center=False,
                name="mean_norm",
            )
            self.mean_proj = keras.layers.Conv2D(
                filters=self.in_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                name="mean_proj",
                **self._common_conv_kwargs(),
            )
            self.mean_upsample = keras.layers.UpSampling2D(
                size=2, interpolation="bilinear", name="mean_upsample"
            )

        # --- Uncertainty head(s) ---
        if self.uncertainty_mode == "gaussian":
            # Log-variance head (may use bias for flexibility)
            self.var_norm = keras.layers.LayerNormalization(
                epsilon=self.LAYERNORM_EPSILON,
                name="var_norm",
            )
            self.var_proj = keras.layers.Conv2D(
                filters=self.in_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="var_proj",
            )
            self.var_upsample = keras.layers.UpSampling2D(
                size=2, interpolation="bilinear", name="var_upsample"
            )
        else:
            # Quantile heads: one projection per quantile
            self.quantile_norms = []
            self.quantile_projs = []
            self.quantile_upsamples = []
            for i, q in enumerate(self.quantiles):
                norm = keras.layers.LayerNormalization(
                    epsilon=self.LAYERNORM_EPSILON,
                    center=False,
                    name=f"quantile_norm_{i}",
                )
                proj = keras.layers.Conv2D(
                    filters=self.in_channels,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"quantile_proj_{i}",
                )
                up = keras.layers.UpSampling2D(
                    size=2, interpolation="bilinear",
                    name=f"quantile_upsample_{i}",
                )
                self.quantile_norms.append(norm)
                self.quantile_projs.append(proj)
                self.quantile_upsamples.append(up)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape) -> None:
        """Build all sublayers via symbolic forward pass.

        Stores ``input_shape`` for ``get_build_config`` so that
        ``.keras`` save/load round-trips can rebuild the model.
        """
        self._build_input_shape = input_shape

        if isinstance(input_shape, list):
            target_shape = input_shape[0]
        else:
            target_shape = input_shape

        if len(target_shape) == 3:
            target_shape = (None,) + tuple(target_shape)
        else:
            target_shape = tuple(target_shape)

        inputs = [keras.KerasTensor(target_shape)]

        if self.enable_dense_conditioning:
            cond_shape = (
                target_shape[0],
                target_shape[1],
                target_shape[2],
                self.dense_cond_channels,
            )
            inputs.append(keras.KerasTensor(cond_shape))

        if self.enable_discrete_conditioning:
            inputs.append(keras.KerasTensor((target_shape[0], 1)))

        _ = self.call(inputs)
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        if hasattr(self, "_build_input_shape"):
            shape = self._build_input_shape
            if isinstance(shape, list):
                return {"input_shape": [list(s) for s in shape]}
            return {"input_shape": list(shape)}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if "input_shape" in config:
            shape = config["input_shape"]
            if isinstance(shape, list) and shape and isinstance(shape[0], list):
                shape = [tuple(s) for s in shape]
            else:
                shape = tuple(shape)
            self.build(shape)

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------

    def _get_dense_cond_at_level(
        self,
        dense_cond: keras.KerasTensor,
        level: int,
        is_bottleneck: bool = False,
    ) -> keras.KerasTensor:
        """Downsample and project dense conditioning to match level."""
        cond = dense_cond
        for _ in range(level + 1):
            cond = keras.layers.AveragePooling2D(
                pool_size=2, padding="same"
            )(cond)

        if is_bottleneck:
            return self.cond_proj_bottleneck(cond)
        return self.cond_projections[level](cond)

    def _get_discrete_cond(
        self, class_label: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Embed class label to conditioning vector."""
        emb = self.class_embedding(class_label)
        return keras.ops.squeeze(emb, axis=1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass producing mean + uncertainty estimate.

        :param inputs: Either a single tensor or a list:
            - ``[noisy_target]``
            - ``[noisy_target, dense_cond]``
            - ``[noisy_target, class_label]`` (if only discrete)
            - ``[noisy_target, dense_cond, class_label]``
        :param training: Whether in training mode.
        :return: Concatenated output along last axis:
            - Gaussian: ``(B, H, W, 2*in_channels)`` = ``[mu, log_var]``
            - Quantile: ``(B, H, W, num_quantiles*in_channels)`` =
              ``[q_0, q_1, ..., q_N]``
        """
        # Parse inputs
        if isinstance(inputs, (list, tuple)):
            noisy_target = inputs[0]
            idx = 1
            dense_cond = None
            class_label = None
            if self.enable_dense_conditioning and idx < len(inputs):
                dense_cond = inputs[idx]
                idx += 1
            if self.enable_discrete_conditioning and idx < len(inputs):
                class_label = inputs[idx]
                idx += 1
        else:
            noisy_target = inputs
            dense_cond = None
            class_label = None

        # Prepare conditioning
        discrete_emb = None
        if self.enable_discrete_conditioning and class_label is not None:
            discrete_emb = self._get_discrete_cond(class_label)

        # Stem
        x = self.stem_conv(noisy_target)
        x = self.stem_norm(x, training=training)

        # Encoder
        skip_features = []
        for level_info in self.encoder_levels:
            level = self.encoder_levels.index(level_info)

            if level_info["transition"] is not None:
                x = level_info["transition"](x)

            level_dense_cond = None
            if self.enable_dense_conditioning and dense_cond is not None:
                level_dense_cond = self._get_dense_cond_at_level(
                    dense_cond, level
                )

            for block in level_info["blocks"]:
                x = block(
                    x,
                    dense_cond=level_dense_cond,
                    discrete_cond=discrete_emb,
                    training=training,
                )

            skip_features.append(x)

            if level_info["downsample"] is not None:
                x = level_info["downsample"](x)

        # Bottleneck
        bottle_dense_cond = None
        if self.enable_dense_conditioning and dense_cond is not None:
            bottle_dense_cond = self._get_dense_cond_at_level(
                dense_cond, self.num_levels - 1, is_bottleneck=True
            )
        for block in self.bottleneck_blocks:
            x = block(
                x,
                dense_cond=bottle_dense_cond,
                discrete_cond=discrete_emb,
                training=training,
            )

        # Decoder
        for dec_info in self.decoder_levels:
            level = dec_info["level"]

            if dec_info["upsample"] is not None:
                x = dec_info["upsample"](x)

            skip = skip_features[level]
            x = keras.ops.concatenate([x, skip], axis=-1)
            x = dec_info["skip_proj"](x)

            level_dense_cond = None
            if self.enable_dense_conditioning and dense_cond is not None:
                level_dense_cond = self._get_dense_cond_at_level(
                    dense_cond, level
                )

            for block in dec_info["blocks"]:
                x = block(
                    x,
                    dense_cond=level_dense_cond,
                    discrete_cond=discrete_emb,
                    training=training,
                )

        # --- Output heads ---
        if self.uncertainty_mode == "gaussian":
            # Mean head (bias-free, Miyasawa-compliant)
            mean_feat = self.mean_norm(x)
            mean_residual = self.mean_proj(mean_feat)
            mean_residual = self.mean_upsample(mean_residual)
            mu = noisy_target + mean_residual

            # Log-variance head
            var_feat = self.var_norm(x)
            log_var = self.var_proj(var_feat)
            log_var = self.var_upsample(log_var)
            log_var = keras.ops.clip(
                log_var, _MIN_LOG_VARIANCE, _MAX_LOG_VARIANCE
            )
            return keras.ops.concatenate([mu, log_var], axis=-1)
        else:
            # Quantile mode: each quantile head adds its own residual
            q_outputs = []
            for i in range(len(self.quantiles)):
                q_feat = self.quantile_norms[i](x)
                q_residual = self.quantile_projs[i](q_feat)
                q_residual = self.quantile_upsamples[i](q_residual)
                q_out = noisy_target + q_residual
                q_outputs.append(q_out)
            return keras.ops.concatenate(q_outputs, axis=-1)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict_with_intervals(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        confidence_level: Optional[float] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Run inference and return structured confidence intervals.

        :param inputs: Model inputs (same as ``call``).
        :param confidence_level: Override the default CI level.
        :return: Dictionary with keys:
            - ``"mean"`` or ``"median"``: Point estimate ``(B,H,W,C)``.
            - ``"lower"``: Lower bound ``(B,H,W,C)``.
            - ``"upper"``: Upper bound ``(B,H,W,C)``.
            - ``"uncertainty"``: Uncertainty map ``(B,H,W,C)``
              (sigma for Gaussian, interval width for quantile).
        """
        import math

        output = self(inputs, training=False)
        level = confidence_level or self.confidence_level

        if self.uncertainty_mode == "gaussian":
            mu = output[..., :self.in_channels]
            log_var = output[..., self.in_channels:]
            sigma = keras.ops.exp(0.5 * log_var)

            # Compute z-score for confidence level
            alpha = 1.0 - level
            p = 1.0 - alpha / 2.0
            t = math.sqrt(-2.0 * math.log(1.0 - p))
            z = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
                1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
            )

            return {
                "mean": mu,
                "lower": mu - z * sigma,
                "upper": mu + z * sigma,
                "uncertainty": sigma,
                "log_variance": log_var,
            }
        else:
            # Quantile mode: first quantile is lower, last is upper
            results = {}
            for i, q in enumerate(self.quantiles):
                q_val = output[
                    ..., i * self.in_channels:(i + 1) * self.in_channels
                ]
                results[f"q_{q:.2f}"] = q_val

            lower = output[..., :self.in_channels]
            upper = output[..., -self.in_channels:]

            # Median is the middle quantile if 0.50 is present
            median_idx = None
            for i, q in enumerate(self.quantiles):
                if abs(q - 0.50) < 1e-6:
                    median_idx = i
                    break

            point_est = output[
                ...,
                median_idx * self.in_channels:(median_idx + 1) * self.in_channels,
            ] if median_idx is not None else (lower + upper) / 2.0

            results.update({
                "median": point_est,
                "lower": lower,
                "upper": upper,
                "uncertainty": upper - lower,
            })
            return results

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]:
        if isinstance(input_shape, list):
            target_shape = input_shape[0]
        else:
            target_shape = input_shape

        if self.uncertainty_mode == "gaussian":
            out_channels = 2 * self.in_channels
        else:
            out_channels = len(self.quantiles) * self.in_channels

        return (*target_shape[:-1], out_channels)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "uncertainty_mode": self.uncertainty_mode,
            "quantiles": self.quantiles,
            "confidence_level": self.confidence_level,
            "level_channels": self.level_channels,
            "level_blocks": self.level_blocks,
            "level_shifts": self.level_shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "enable_dense_conditioning": self.enable_dense_conditioning,
            "dense_cond_channels": self.dense_cond_channels,
            "enable_discrete_conditioning": (
                self.enable_discrete_conditioning
            ),
            "num_classes": self.num_classes,
            "class_embedding_dim": self.class_embedding_dim,
            "variance_regularization": self.variance_regularization,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
        })
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any]
    ) -> "CliffordNetConfidenceDenoiser":
        for key in ("kernel_regularizer",):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, **kwargs: Any) -> None:
        if not self.built:
            logger.warning(
                "Model not built; call build() with input shape first."
            )
        super().summary(**kwargs)
        logger.info("CliffordNetConfidenceDenoiser configuration:")
        logger.info(f"  in_channels         : {self.in_channels}")
        logger.info(f"  uncertainty_mode    : {self.uncertainty_mode}")
        if self.uncertainty_mode == "quantile":
            logger.info(f"  quantiles           : {self.quantiles}")
        logger.info(f"  confidence_level    : {self.confidence_level}")
        logger.info(f"  level_channels      : {self.level_channels}")
        logger.info(f"  level_blocks        : {self.level_blocks}")
        logger.info(f"  level_shifts        : {self.level_shifts}")
        logger.info(f"  cli_mode            : {self.cli_mode}")
        logger.info(f"  ctx_mode            : {self.ctx_mode}")
        logger.info(f"  use_global_context  : {self.use_global_context}")
        logger.info(f"  dense_conditioning  : {self.enable_dense_conditioning}")
        logger.info(
            f"  discrete_conditioning: {self.enable_discrete_conditioning}"
        )
        logger.info(f"  bias_free_backbone  : True")

    # ------------------------------------------------------------------
    # Loss factory
    # ------------------------------------------------------------------

    def get_loss(self) -> keras.losses.Loss:
        """Return the appropriate loss for this model's uncertainty mode.

        :return: :class:`GaussianNLLLoss` or :class:`PinballLoss`.
        """
        if self.uncertainty_mode == "gaussian":
            return GaussianNLLLoss(
                in_channels=self.in_channels,
                variance_regularization=self.variance_regularization,
                name="gaussian_nll",
            )
        return PinballLoss(
            quantiles=self.quantiles,
            in_channels=self.in_channels,
            name="pinball",
        )

    def get_metrics(self) -> List[keras.metrics.Metric]:
        """Return uncertainty-aware metrics for this model.

        :return: List of metrics: coverage, interval width, MAE on mean.
        """
        return [
            CoverageMetric(
                confidence_level=self.confidence_level,
                in_channels=self.in_channels,
                uncertainty_mode=self.uncertainty_mode,
                quantiles=self.quantiles,
                name="coverage",
            ),
            IntervalWidthMetric(
                in_channels=self.in_channels,
                uncertainty_mode=self.uncertainty_mode,
                confidence_level=self.confidence_level,
                name="interval_width",
            ),
        ]

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": dict(
            level_channels=[64, 96, 128],
            level_blocks=[2, 2, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "small": dict(
            level_channels=[64, 128, 192],
            level_blocks=[2, 3, 3],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            level_channels=[96, 192, 256, 256],
            level_blocks=[2, 3, 4, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    @classmethod
    def from_variant(
        cls,
        variant: str,
        in_channels: int = 1,
        uncertainty_mode: UncertaintyMode = "gaussian",
        quantiles: Optional[List[float]] = None,
        confidence_level: float = 0.90,
        enable_dense_conditioning: bool = False,
        dense_cond_channels: Optional[int] = None,
        enable_discrete_conditioning: bool = False,
        num_classes: int = 0,
        class_embedding_dim: int = 128,
        **kwargs: Any,
    ) -> "CliffordNetConfidenceDenoiser":
        """Create from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``.
        :param in_channels: Number of input/output channels.
        :param uncertainty_mode: ``"gaussian"`` or ``"quantile"``.
        :param quantiles: Quantile levels for quantile mode.
        :param confidence_level: Target CI level.
        :param enable_dense_conditioning: Enable dense conditioning.
        :param dense_cond_channels: Dense conditioning channel count.
        :param enable_discrete_conditioning: Enable discrete conditioning.
        :param num_classes: Number of classes for discrete conditioning.
        :param class_embedding_dim: Class embedding dimension.
        :param kwargs: Override any default hyperparameter.
        :return: Configured model instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(
            f"Creating CliffordNetConfidenceDenoiser-{variant.upper()} "
            f"(mode={uncertainty_mode})"
        )
        return cls(
            in_channels=in_channels,
            uncertainty_mode=uncertainty_mode,
            quantiles=quantiles,
            confidence_level=confidence_level,
            enable_dense_conditioning=enable_dense_conditioning,
            dense_cond_channels=dense_cond_channels,
            enable_discrete_conditioning=enable_discrete_conditioning,
            num_classes=num_classes,
            class_embedding_dim=class_embedding_dim,
            **defaults,
        )

    @classmethod
    def tiny(
        cls, in_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetConfidenceDenoiser":
        """Tiny variant: 3 levels, ~2.1M params."""
        return cls.from_variant("tiny", in_channels=in_channels, **kwargs)

    @classmethod
    def small(
        cls, in_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetConfidenceDenoiser":
        """Small variant: 3 levels, ~5.3M params."""
        return cls.from_variant("small", in_channels=in_channels, **kwargs)

    @classmethod
    def base(
        cls, in_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetConfidenceDenoiser":
        """Base variant: 4 levels, ~12.5M params."""
        return cls.from_variant("base", in_channels=in_channels, **kwargs)
