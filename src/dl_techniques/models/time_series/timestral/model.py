"""The Timestral network and its single-block forecast primitive.

The model normalizes each row causally per patch, embeds patches from [values,
missing] channels, runs a time/variate transformer over [Enc; Dec] tokens (context
patches followed by learned forecast-query tokens), and reads quantiles for every
horizon time step from the decoder tokens.
"""

import keras
from keras import ops
import numpy as np
from keras.layers import Layer
from typing import Optional, Tuple, List, Dict, Any

from dl_techniques.models.time_series.timestral.config import TimestralConfig
from dl_techniques.models.time_series.timestral.layers import (
    ResidualBlock, Transformer, TwoStreamMask
)
from dl_techniques.models.time_series.timestral.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    first_observed_patch,
)
from dl_techniques.models.time_series.timestral.scaler import (
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    denormalize_quantiles,
    advancing_block_frames,
    instance_norm_stats,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# Role IDs for covariate rows
ROLE_TARGET = 0
ROLE_PAST_COV = 1
ROLE_KNOWN_COV = 2


class PatchRepresentation(Layer):
    """Patch embedding over per-step [values, missing] channels, plus forecast-query token."""

    def __init__(self, patch_size: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        self.patch_proj = ResidualBlock(
            2 * patch_size, 4 * d_model, d_model, use_bias=True, name="patch_proj",
        )
        self.forecast_query = None  # Created in build

    def build(self, input_shape):
        self.forecast_query = self.add_weight(
            name="forecast_query",
            shape=(self.d_model,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, scaled, observed):
        """(R, T) normalized values and observation mask to (R, T/patch_size, d_model)."""
        values = ops.reshape(scaled, (int(ops.shape(scaled)[0]), int(ops.shape(scaled)[1]) // self.patch_size, self.patch_size))
        missing = ops.logical_not(observed)
        missing = ops.reshape(missing, ops.shape(values))
        missing = ops.cast(missing, values.dtype)
        x = ops.concatenate([values, missing], axis=-1)
        return self.patch_proj(x)

    def forecast_tokens(self, rows: int, n_patches: int):
        """(rows, n_patches, d_model) learned forecast query tokens."""
        q = ops.reshape(self.forecast_query, (1, 1, self.d_model))
        return ops.broadcast_to(q, (rows, n_patches, self.d_model))

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "d_model": self.d_model})
        return config


def monotone_quantiles(raw, levels: Tuple[float, ...], width_factor: float):
    """Non-crossing quantiles from raw head outputs along last axis (monotone_cumsum).

    Median channel is free. Others fan out as:
    median +/- cumsum(width_factor * dp * softplus(raw)), dp = prob gap to next level inward.
    """
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]

    widths_t = ops.cast(ops.array(widths), raw.dtype)
    inc = width_factor * widths_t * ops.softplus(raw)

    median = raw[..., m : m + 1]
    up = ops.cumsum(inc[..., m + 1 :], axis=-1)
    down = ops.flip(ops.cumsum(ops.flip(inc[..., :m], axis=-1), axis=-1), axis=-1)

    work = str(np.result_type(getattr(raw.dtype, "name", str(raw.dtype)), "float32").name)
    big = np.finfo(work).max
    return ops.clip(
        ops.cast(
            ops.concatenate([median - down, median, median + up], axis=-1), work
        ),
        -big, big
    )


class QuantileHead(Layer):
    """Maps each decoder token to patch_size monotone quantile vectors in normalized space."""

    def __init__(self, config: TimestralConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.patch_size = config.patch_size
        self.num_quantiles = config.num_quantiles
        self.projection = ResidualBlock(
            config.d_model,
            4 * config.d_model,
            config.patch_size * config.num_quantiles,
            use_bias=False,
            name="projection",
        )

    def build(self, input_shape):
        self.projection.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        """(R, N, D) -> normalized quantiles (R, N * patch_size, Q) in at least fp32."""
        raw = self.projection(x)
        *leading, n_patches, _ = ops.shape(raw)
        preds = ops.reshape(raw, (*leading, n_patches, self.patch_size, self.num_quantiles))
        preds = ops.cast(preds, str(np.result_type(
            getattr(preds.dtype, "name", str(preds.dtype)), "float32").name))
        preds = monotone_quantiles(preds, self.config.quantile_knots, self.config.quantile_width_factor)
        return ops.reshape(preds, (*leading, n_patches * self.patch_size, self.num_quantiles))

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config


@register_dl_technique("dl_techniques.models.timestral.model")
class TimestralModel(keras.Model):
    """Timestral core model for probabilistic time series forecasting.

    A patch-based transformer with Decoupled Horizon Forecasting (DHF), trained
    to output 21 non-crossing quantiles per time step with support for past-only
    and known-future covariates.
    """

    def __init__(self, config: TimestralConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        c = config

        self.representation = PatchRepresentation(c.patch_size, c.d_model, name="representation")
        self.transformer = Transformer(
            block_schedule=c.block_schedule,
            embed_dim=c.d_model,
            num_heads=c.num_heads,
            mlp_hidden_dim=c.mlp_hidden_dim,
            rope_theta=c.rope_theta,
            rope_dim=c.rope_dim,
            norm_eps=c.norm_eps,
            attn_bias=c.attn_bias,
            mlp_bias=c.mlp_bias,
            time_attn_gate=c.time_attn_gate,
            variate_attn_gate=c.variate_attn_gate,
            name="transformer",
        )
        self.head = QuantileHead(c, name="quantile_head")

    def build(self, input_shape):
        """Build all sublayers with representative (static-oriented) shapes.

        Variables are shape-independent in the time/row dimensions, so building
        with one patch and one row creates the complete weight set.
        """
        c = self.config
        emb = c.d_model
        # Representation feeds [values, missing] channels: last dim 2*patch_size.
        self.representation.build((1, 2 * c.patch_size))
        self.transformer.build((1, 2, emb))
        self.head.build((1, 2, emb))
        super().build(input_shape)

    def scale(self, inputs, observed) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Causal per-patch normalization of packed (R, T) rows.

        Returns (scaled, loc, scale), each (R, T).
        """
        c = self.config
        loc, scale = causal_patch_loc_scale(
            inputs,
            observed,
            c.patch_size,
            c.minimum_scale,
            c.stabilize_with_global,
        )
        scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=c.use_asinh)
        return scaled, loc, scale

    def _known_cov_scaling(
        self, inputs, observed, scaled, loc, scale, known_rows
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Normalize known-future covariate rows with non-causal whole-row statistics.

        Indicator rows (values in {0, 1}) skip the asinh squash.
        """
        if not ops.any(known_rows):
            return scaled, loc, scale

        loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, self.config.minimum_scale)
        known_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, use_asinh=False)
        if self.config.use_asinh:
            work = str(np.result_type(
                getattr(known_scaled.dtype, "name", str(known_scaled.dtype)),
                "float32").name)
            squashed = ops.cast(ops.arcsinh(ops.cast(known_scaled, work)), known_scaled.dtype)
            known_scaled = ops.where(ops.logical_and(observed, ops.logical_not(is_binary)), squashed, known_scaled)

        sel = ops.expand_dims(known_rows, axis=-1)
        scaled = ops.where(sel, known_scaled, scaled)
        loc = ops.where(sel, loc_c, loc)
        scale = ops.where(sel, scale_c, scale)
        return scaled, loc, scale

    def _run_two_stream(
        self, x_enc, x_dec, time_mask, variate_mask, positions
    ) -> keras.KerasTensor:
        """Transformer over [Enc; Dec], then head on decoder tokens (normalized space)."""
        x = ops.concatenate([x_enc, x_dec], axis=1)
        mask = time_mask
        enc = int(ops.shape(x_enc)[1])
        if self.config.split_context_horizon_attn:
            mask = TwoStreamMask(mask=time_mask, enc_len=enc)
        h = self.transformer(x, time_mask=mask, variate_mask=variate_mask, positions=positions)
        return self.head(h[:, enc:, :])

    def predict_block(
        self,
        inputs,
        observed,
        group_ids,
        horizon_patches: int,
        role_ids: Optional[keras.KerasTensor] = None,
        known_future: Optional[keras.KerasTensor] = None,
        known_future_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Forecast one block of horizon_patches patches after packed context.

        Args:
            inputs: (R, T) context values, T multiple of patch_size.
            observed: (R, T) observation mask.
            group_ids: (R,) series id per row; rows of one series attend each other.
            horizon_patches: Number of patches to forecast.
            role_ids: (R,) row roles (ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV),
                or None when every row is a target.
            known_future: (R, horizon_patches * patch_size) known-future covariate
                values over this block; zeros on other rows.
            known_future_mask: Observation mask for known_future.

        Returns:
            Raw-space quantiles (R, horizon_patches * patch_size, Q), non-decreasing along Q.
        """
        if role_ids is None or not self.config.covariate_enabled or ops.all(role_ids == ROLE_TARGET):
            return self._predict_context_only(inputs, observed, group_ids, horizon_patches)

        r = ops.shape(inputs)[0]
        horizon_len = horizon_patches * self.config.patch_size

        if known_future is None:
            known_future = ops.zeros((r, horizon_len), dtype=inputs.dtype)
            known_future_mask = ops.zeros((r, horizon_len), dtype="bool")

        full_inputs = ops.concatenate([inputs, known_future], axis=-1)
        full_observed = ops.concatenate([observed, known_future_mask], axis=-1)
        return self._predict_window(full_inputs, full_observed, group_ids, horizon_patches, role_ids)

    def _predict_context_only(self, inputs, observed, group_ids, horizon_patches):
        """No covariates: encoder over context, h forecast-query tokens after it."""
        c = self.config
        ps = c.patch_size
        rows, t = ops.shape(inputs)
        assert t % ps == 0, f"Context length {t} must be multiple of patch_size {ps}"
        ll, h = t // ps, horizon_patches
        assert h >= 1, f"horizon_patches must be >= 1, got {h}"

        scaled, loc, scale = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, h)

        obs_per_patch = ops.any(
            ops.reshape(observed, (rows, ll, ps)), axis=-1
        )
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)

        first_obs = first_observed_patch(obs_per_patch, group_ids)
        positions = build_forecast_positions(ll, h, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        anchor_loc = ops.reshape(loc, (rows, ll, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = ops.reshape(scale, (rows, ll, ps))[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(preds, anchor_loc, anchor_scale, c.use_asinh)

    def _predict_window(self, full_inputs, full_observed, group_ids, horizon_patches, role_ids):
        """With covariates: encoder over context + horizon so known-future values are keys."""
        c = self.config
        ps = c.patch_size
        rows, w = ops.shape(full_inputs)
        assert w % ps == 0, f"Window length {w} must be multiple of patch_size {ps}"
        h = horizon_patches
        assert 1 <= h <= c.block_size, f"Decode block {h} must be in [1, {c.block_size}] with covariates"
        n = w // ps
        ll = n - h
        assert ll >= 1, f"Context must have at least one patch (window {n}, horizon {h})"

        target_rows = ops.equal(role_ids, ROLE_TARGET)
        known_rows = ops.equal(role_ids, ROLE_KNOWN_COV)

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)

        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = ops.any(
            ops.reshape(full_observed, (rows, n, ps)), axis=-1
        )
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)

        s_rows = ops.full((rows,), ll, dtype="int32")
        time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows, known_rows
        )

        first_obs = first_observed_patch(obs_per_patch, group_ids)
        positions = build_two_stream_positions(n, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        preds = ops.reshape(preds, (rows, n, ps, -1))
        horizon_preds = ops.reshape(preds[:, ll:], (rows, h * ps, -1))

        anchor_loc = ops.reshape(loc, (rows, n, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = ops.reshape(scale, (rows, n, ps))[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(horizon_preds, anchor_loc, anchor_scale, c.use_asinh)

    def _refine_block(self, quantiles, context, mask, target_rows):
        """Move decoded block from anchor frame to advancing refine frame.

        Model denormalizes whole block (R, H*patch_size, Q) with last context patch frame.
        First horizon patch keeps that frame; each later re-denormalized with advancing_block_frames.
        Covariate rows keep their quantiles.
        """
        ps = self.config.patch_size
        n_pred = int(ops.shape(quantiles)[1]) // ps
        if n_pred <= 1:
            return quantiles

        _, loc, scale = self.scale(context, mask)
        # Anchor frame: statistics of the last context patch.
        loc0 = loc[:, -1:]
        scale0 = scale[:, -1:]

        median = quantiles[..., self.config.median_index]
        median = ops.reshape(median, (-1, n_pred, ps))
        median = ops.cast(median, "float64")

        sinh_median = (median - ops.cast(loc0[..., None], "float64")) / ops.cast(scale0[..., None], "float64")
        loc_j, scale_j = advancing_block_frames(context, mask, sinh_median, self.config.minimum_scale)

        first = ops.concatenate([
            ops.ones_like(loc_j[:, :1], dtype="bool"),
            ops.zeros_like(loc_j[:, 1:], dtype="bool"),
        ], axis=1)
        loc_j = ops.where(first, ops.cast(loc0, "float32"), ops.cast(loc_j, "float32"))
        scale_j = ops.where(first, ops.cast(scale0, "float32"), ops.cast(scale_j, "float32"))

        loc_j = ops.repeat(loc_j, ps, axis=1)
        scale_j = ops.repeat(scale_j, ps, axis=1)

        sinh_norm = (quantiles - ops.expand_dims(loc0, -1)) / ops.expand_dims(scale0, -1)
        refined = denormalize_quantiles(
            sinh_norm,
            loc_j,
            scale_j,
            self.config.use_asinh,
        )

        if target_rows is not None:
            target_rows = ops.expand_dims(ops.expand_dims(target_rows, -1), -1)
            refined = ops.where(target_rows, refined, quantiles)
        return refined

    def call(self, inputs, training=None):
        """Standard Keras call for training: returns quantile predictions.

        This is a simplified path for training on fixed windows.
        For autoregressive forecasting, use TimestralForecaster.
        """
        # Inputs: (B, T, F) or (B, T)
        if inputs.ndim == 2:
            inputs = ops.expand_dims(inputs, -1)
        # For training, we expect a fixed window and predict the full horizon
        # This is a placeholder - the actual training uses predict_block directly
        raise NotImplementedError(
            "Use TimestralForecaster.forecast() for inference or "
            "TimestralTrainer for training with DHF block decoding."
        )

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["config"] = TimestralConfig.from_dict(config["config"])
        return cls(**config)


# Variant configurations matching Hugging Face models
TIMESTRAL_VARIANTS = {
    "5M": {
        "d_model": 256,
        "num_heads": 8,
        "mlp_hidden_dim": 1024,
        "block_schedule": (
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
        ),
    },
    "28M": {
        "d_model": 512,
        "num_heads": 8,
        "mlp_hidden_dim": 2048,
        "block_schedule": (
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
        ),
    },
    "123M": {
        "d_model": 768,
        "num_heads": 12,
        "mlp_hidden_dim": 3072,
        "block_schedule": (
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
        ),
    },
    "209M": {
        "d_model": 1024,
        "num_heads": 16,
        "mlp_hidden_dim": 4096,
        "block_schedule": (
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
            "time", "time_variate", "time", "time_variate",
        ),
    },
}

# Update the config module's variant dict
import dl_techniques.models.time_series.timestral.config as config_module
config_module.TIMESTRAL_VARIANTS = TIMESTRAL_VARIANTS


def create_timestral_model(
    variant: str = "28M",
    prediction_length: int = 96,
    patch_size: int = 32,
    quantile_levels: Optional[Tuple[float, ...]] = None,
    **kwargs
) -> TimestralModel:
    """Create a Timestral model from a predefined variant.

    Args:
        variant: One of "5M", "28M", "123M", "209M".
        prediction_length: Forecast horizon in time steps.
        patch_size: Patch size (default 32).
        quantile_levels: Custom quantile levels (default 21 levels).
        **kwargs: Additional args passed to TimestralConfig.

    Returns:
        TimestralModel instance.
    """
    if variant not in TIMESTRAL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Available: {list(TIMESTRAL_VARIANTS.keys())}")

    vcfg = TIMESTRAL_VARIANTS[variant].copy()
    vcfg.update(kwargs)

    kwargs = {}
    if quantile_levels is not None:
        kwargs["quantile_knots"] = tuple(quantile_levels)

    config = TimestralConfig(
        patch_size=patch_size,
        prediction_length=prediction_length,
        **kwargs,
        **vcfg,
    )
    return TimestralModel(config)


def create_timestral_by_variant(
    variant: str = "28M",
    prediction_length: int = 96,
    **kwargs
) -> TimestralModel:
    """Alias for create_timestral_model for API consistency."""
    return create_timestral_model(variant=variant, prediction_length=prediction_length, **kwargs)
