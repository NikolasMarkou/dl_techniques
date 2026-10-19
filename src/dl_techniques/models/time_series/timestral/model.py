"""The Timestral network and its single-block forecast primitive.

The model normalizes each row causally per patch, embeds patches from [values,
missing] channels, runs a time/variate transformer over [Enc; Dec] tokens (context
patches followed by learned forecast-query tokens), and reads quantiles for every
horizon time step from the decoder tokens.

``TimestralModel.from_variant`` builds one of the four released sizes from
``TimestralModel.MODEL_VARIANTS``; ``create_timestral_model`` delegates to it.
Released weights load through ``loading.load_model`` or
``TimestralForecaster.from_pretrained``, not through the factory.

References:
    - Mistral AI, 2026. Timestral: Mistral's time-series foundation models
      (announcement; the technical report is announced, not yet published).
      (https://mistral.ai/news/timestral)
    - Mistral AI. timestral, the reference PyTorch implementation this port
      follows. (https://github.com/mistralai/timestral)
    - Mistral AI. Released checkpoints Timestral-5M, -28M, -123M and -209M.
      (https://huggingface.co/mistralai/Timestral-209M)
"""

import types

import keras
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from dl_techniques.models.time_series.timestral import config as config_module
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

# Variant configurations of the four released checkpoints, read from each
# Hugging Face repo's config.json (mistralai/Timestral-{5M,28M,123M,209M});
# every other field is a TimestralConfig default there. config.py declares
# TIMESTRAL_VARIANTS empty; it is filled here, where the table lives. A
# read-only view, because TimestralModel.MODEL_VARIANTS is the same object.
_RELEASED_UNIT = ("time", "time", "time", "time_variate")
TIMESTRAL_VARIANTS = types.MappingProxyType({
    "5M": {
        "d_model": 192,
        "num_heads": 3,
        "mlp_hidden_dim": 512,
        "block_schedule": _RELEASED_UNIT * 2,
        "quantile_width_factor": 3.0,
    },
    "28M": {
        "d_model": 384,
        "num_heads": 6,
        "mlp_hidden_dim": 1024,
        "block_schedule": _RELEASED_UNIT * 3,
        "quantile_width_factor": 3.0,
    },
    "123M": {
        "d_model": 640,
        "num_heads": 10,
        "mlp_hidden_dim": 1712,
        "block_schedule": _RELEASED_UNIT * 5,
        "quantile_width_factor": 3.0,
    },
    "209M": {
        "d_model": 768,
        "num_heads": 12,
        "mlp_hidden_dim": 2048,
        "block_schedule": _RELEASED_UNIT * 6,
        "quantile_width_factor": 3.0,
    },
})
config_module.TIMESTRAL_VARIANTS = TIMESTRAL_VARIANTS


class PatchRepresentation(keras.layers.Layer):
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
        # call() feeds patch_proj the [values, missing] channels, whatever
        # shape this build was given.
        self.patch_proj.build((None, None, 2 * self.patch_size))
        super().build(input_shape)

    def call(self, scaled, observed):
        """(R, T) normalized values and observation mask to (R, T/patch_size, d_model)."""
        values = keras.ops.reshape(scaled, (int(keras.ops.shape(scaled)[0]), int(keras.ops.shape(scaled)[1]) // self.patch_size, self.patch_size))
        missing = keras.ops.logical_not(observed)
        missing = keras.ops.reshape(missing, keras.ops.shape(values))
        missing = keras.ops.cast(missing, values.dtype)
        x = keras.ops.concatenate([values, missing], axis=-1)
        return self.patch_proj(x)

    def forecast_tokens(self, rows: int, n_patches: int):
        """(rows, n_patches, d_model) learned forecast query tokens."""
        q = keras.ops.reshape(self.forecast_query, (1, 1, self.d_model))
        return keras.ops.broadcast_to(q, (rows, n_patches, self.d_model))

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

    widths_t = keras.ops.cast(keras.ops.array(widths), raw.dtype)
    inc = width_factor * widths_t * keras.ops.softplus(raw)

    median = raw[..., m : m + 1]
    up = keras.ops.cumsum(inc[..., m + 1 :], axis=-1)
    down = keras.ops.flip(keras.ops.cumsum(keras.ops.flip(inc[..., :m], axis=-1), axis=-1), axis=-1)

    work = str(np.result_type(getattr(raw.dtype, "name", str(raw.dtype)), "float32").name)
    big = np.finfo(work).max
    return keras.ops.clip(
        keras.ops.cast(
            keras.ops.concatenate([median - down, median, median + up], axis=-1), work
        ),
        -big, big
    )


class QuantileHead(keras.layers.Layer):
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
        *leading, n_patches, _ = keras.ops.shape(raw)
        preds = keras.ops.reshape(raw, (*leading, n_patches, self.patch_size, self.num_quantiles))
        preds = keras.ops.cast(preds, str(np.result_type(
            getattr(preds.dtype, "name", str(preds.dtype)), "float32").name))
        preds = monotone_quantiles(preds, self.config.quantile_knots, self.config.quantile_width_factor)
        return keras.ops.reshape(preds, (*leading, n_patches * self.patch_size, self.num_quantiles))

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

    MODEL_VARIANTS = TIMESTRAL_VARIANTS

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
        with one patch and one row creates the complete weight set (every
        sublayer's build() builds its own sublayers). A sublayer that a forward
        pass already built is left alone, so ``build`` also completes a model
        that ``predict_block`` built lazily, and a second ``build`` is a no-op.
        """
        c = self.config
        emb = c.d_model
        # DECISION plan-2026-10-14T062616-81938e81/D-162
        # WHAT NOT TO DO: do not reorder these three builds into call order
        # (representation, transformer, head). Each Dense draws its initializer
        # seed when it is constructed, and the head's Dense layers have always
        # been constructed first here; building the head first keeps an
        # explicitly built model's initial weights bit-identical to the port as
        # committed in 6d757c6a6. See decisions.md D-162.
        if not self.head.built:
            self.head.build((1, 2, emb))
        # Representation feeds [values, missing] channels: last dim 2*patch_size.
        if not self.representation.built:
            self.representation.build((1, 2 * c.patch_size))
        if not self.transformer.built:
            self.transformer.build((1, 2, emb))
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
        if not keras.ops.any(known_rows):
            return scaled, loc, scale

        loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, self.config.minimum_scale)
        known_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, use_asinh=False)
        if self.config.use_asinh:
            work = str(np.result_type(
                getattr(known_scaled.dtype, "name", str(known_scaled.dtype)),
                "float32").name)
            squashed = keras.ops.cast(keras.ops.arcsinh(keras.ops.cast(known_scaled, work)), known_scaled.dtype)
            known_scaled = keras.ops.where(keras.ops.logical_and(observed, keras.ops.logical_not(is_binary)), squashed, known_scaled)

        sel = keras.ops.expand_dims(known_rows, axis=-1)
        scaled = keras.ops.where(sel, known_scaled, scaled)
        loc = keras.ops.where(sel, loc_c, loc)
        scale = keras.ops.where(sel, scale_c, scale)
        return scaled, loc, scale

    def _run_two_stream(
        self, x_enc, x_dec, time_mask, variate_mask, positions
    ) -> keras.KerasTensor:
        """Transformer over [Enc; Dec], then head on decoder tokens (normalized space)."""
        x = keras.ops.concatenate([x_enc, x_dec], axis=1)
        mask = time_mask
        enc = int(keras.ops.shape(x_enc)[1])
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
        if role_ids is None or not self.config.covariate_enabled or keras.ops.all(role_ids == ROLE_TARGET):
            return self._predict_context_only(inputs, observed, group_ids, horizon_patches)

        r = keras.ops.shape(inputs)[0]
        horizon_len = horizon_patches * self.config.patch_size

        if known_future is None:
            known_future = keras.ops.zeros((r, horizon_len), dtype=inputs.dtype)
            known_future_mask = keras.ops.zeros((r, horizon_len), dtype="bool")

        full_inputs = keras.ops.concatenate([inputs, known_future], axis=-1)
        full_observed = keras.ops.concatenate([observed, known_future_mask], axis=-1)
        return self._predict_window(full_inputs, full_observed, group_ids, horizon_patches, role_ids)

    def _predict_context_only(self, inputs, observed, group_ids, horizon_patches):
        """No covariates: encoder over context, h forecast-query tokens after it."""
        c = self.config
        ps = c.patch_size
        rows, t = keras.ops.shape(inputs)
        assert t % ps == 0, f"Context length {t} must be multiple of patch_size {ps}"
        ll, h = t // ps, horizon_patches
        assert h >= 1, f"horizon_patches must be >= 1, got {h}"

        scaled, loc, scale = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, h)

        obs_per_patch = keras.ops.any(
            keras.ops.reshape(observed, (rows, ll, ps)), axis=-1
        )
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)

        first_obs = first_observed_patch(obs_per_patch, group_ids)
        positions = build_forecast_positions(ll, h, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        anchor_loc = keras.ops.reshape(loc, (rows, ll, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = keras.ops.reshape(scale, (rows, ll, ps))[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(preds, anchor_loc, anchor_scale, c.use_asinh)

    def _predict_window(self, full_inputs, full_observed, group_ids, horizon_patches, role_ids):
        """With covariates: encoder over context + horizon so known-future values are keys."""
        c = self.config
        ps = c.patch_size
        rows, w = keras.ops.shape(full_inputs)
        assert w % ps == 0, f"Window length {w} must be multiple of patch_size {ps}"
        h = horizon_patches
        assert 1 <= h <= c.block_size, f"Decode block {h} must be in [1, {c.block_size}] with covariates"
        n = w // ps
        ll = n - h
        assert ll >= 1, f"Context must have at least one patch (window {n}, horizon {h})"

        target_rows = keras.ops.equal(role_ids, ROLE_TARGET)
        known_rows = keras.ops.equal(role_ids, ROLE_KNOWN_COV)

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)

        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = keras.ops.any(
            keras.ops.reshape(full_observed, (rows, n, ps)), axis=-1
        )
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)

        s_rows = keras.ops.full((rows,), ll, dtype="int32")
        time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows, known_rows
        )

        first_obs = first_observed_patch(obs_per_patch, group_ids)
        positions = build_two_stream_positions(n, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        preds = keras.ops.reshape(preds, (rows, n, ps, -1))
        horizon_preds = keras.ops.reshape(preds[:, ll:], (rows, h * ps, -1))

        anchor_loc = keras.ops.reshape(loc, (rows, n, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = keras.ops.reshape(scale, (rows, n, ps))[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(horizon_preds, anchor_loc, anchor_scale, c.use_asinh)

    def _refine_block(self, quantiles, context, mask, target_rows):
        """Move decoded block from anchor frame to advancing refine frame.

        Model denormalizes whole block (R, H*patch_size, Q) with last context patch frame.
        First horizon patch keeps that frame; each later re-denormalized with advancing_block_frames.
        Covariate rows keep their quantiles.
        """
        ps = self.config.patch_size
        n_pred = int(keras.ops.shape(quantiles)[1]) // ps
        if n_pred <= 1:
            return quantiles

        _, loc, scale = self.scale(context, mask)
        # Anchor frame: statistics of the last context patch.
        loc0 = loc[:, -1:]
        scale0 = scale[:, -1:]

        median = quantiles[..., self.config.median_index]
        median = keras.ops.reshape(median, (-1, n_pred, ps))
        median = keras.ops.cast(median, "float64")

        sinh_median = (median - keras.ops.cast(loc0[..., None], "float64")) / keras.ops.cast(scale0[..., None], "float64")
        loc_j, scale_j = advancing_block_frames(context, mask, sinh_median, self.config.minimum_scale)

        first = keras.ops.concatenate([
            keras.ops.ones_like(loc_j[:, :1], dtype="bool"),
            keras.ops.zeros_like(loc_j[:, 1:], dtype="bool"),
        ], axis=1)
        loc_j = keras.ops.where(first, keras.ops.cast(loc0, "float32"), keras.ops.cast(loc_j, "float32"))
        scale_j = keras.ops.where(first, keras.ops.cast(scale0, "float32"), keras.ops.cast(scale_j, "float32"))

        loc_j = keras.ops.repeat(loc_j, ps, axis=1)
        scale_j = keras.ops.repeat(scale_j, ps, axis=1)

        # The block left predict_block as sinh(q) * scale0 + loc0, so this is
        # sinh(q) with the asinh already undone.
        sinh_norm = (quantiles - keras.ops.expand_dims(loc0, -1)) / keras.ops.expand_dims(scale0, -1)
        # DECISION plan-2026-10-14T062616-81938e81/D-167
        # WHAT NOT TO DO: do not pass self.config.use_asinh here. sinh_norm is
        # already in sinh space; the reference computes
        # sinh_norm * scale_j + loc_j, and a second sinh moved a three-block
        # forecast by 12% of its range against the reference. See decisions.md D-167.
        refined = denormalize_quantiles(sinh_norm, loc_j, scale_j, use_asinh=False)

        if target_rows is not None:
            target_rows = keras.ops.expand_dims(keras.ops.expand_dims(target_rows, -1), -1)
            refined = keras.ops.where(target_rows, refined, quantiles)
        return refined

    def _pack(self, values):
        """(B, T, V) to packed (B * V, T) rows: values with NaN set to 0, and the mask."""
        b, t, v = (int(d) if d is not None else None for d in values.shape)
        if b is None:
            raise ValueError(
                "TimestralModel needs a static batch size: batch the dataset with "
                "drop_remainder=True (masks and attention reshape by the row count)"
            )
        rows = keras.ops.reshape(keras.ops.transpose(values, (0, 2, 1)), (b * v, t))
        observed = keras.ops.isfinite(rows)
        return keras.ops.where(observed, rows, keras.ops.zeros_like(rows)), observed

    def _left_pad(self, rows, observed):
        """Left-pad (R, T) to a whole number of patches with unobserved zeros."""
        pad = -int(rows.shape[-1]) % self.config.patch_size
        if pad == 0:
            return rows, observed
        return (keras.ops.pad(rows, ((0, 0), (pad, 0))),
                keras.ops.pad(observed, ((0, 0), (pad, 0)), constant_values=False))

    def call(self, inputs, training=None):
        """One DHF decode block after each context: the forward that ``fit`` trains.

        Each batch item is one series and its variates attend each other. The context
        is packed one row per variate, left-padded to whole patches, and run through
        the same context-only path as ``predict_block`` for
        ``ceil(prediction_length / patch_size)`` forecast patches. The output is that
        path's raw-space quantiles: every horizon patch in the frame of the last
        context patch. ``TimestralForecaster`` moves later patches into their own
        frames (``_refine_block``) and decodes past one block, so honest multi-patch
        forecasts come from the forecaster.

        Args:
            inputs: (B, T) or (B, T, V) context, B static; NaN marks a missing value.
            training: Unused; the network has no dropout or other train-only op.

        Returns:
            (B, H, Q) quantiles for 2-D inputs, (B, H, V, Q) for 3-D, with
            H = ``config.prediction_length``, non-decreasing along Q.

        Raises:
            ValueError: If the batch size is not static.
        """
        c = self.config
        x = keras.ops.convert_to_tensor(inputs)
        univariate = len(x.shape) == 2
        if univariate:
            x = keras.ops.expand_dims(x, -1)
        rows, observed = self._left_pad(*self._pack(x))
        b, v = int(x.shape[0]), int(x.shape[2])
        group_ids = keras.ops.repeat(keras.ops.arange(b, dtype="int32"), v)
        horizon = c.prediction_length
        quantiles = self._predict_context_only(
            rows, observed, group_ids, -(-horizon // c.patch_size))[:, :horizon]
        quantiles = keras.ops.reshape(quantiles, (b, v, horizon, c.num_quantiles))
        quantiles = keras.ops.transpose(quantiles, (0, 2, 1, 3))
        return quantiles[:, :, 0] if univariate else quantiles

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        """Pinball loss over the quantile levels, each horizon patch in its own frame.

        The objective of the reference (Mistral AI's Timestral announcement: "a pinball
        loss over 21 quantiles, a discrete approximation of CRPS") with the frame its
        forecaster decodes in. Horizon patch ``j`` is scored in normalized space, with
        the causal statistics of the true series up to the end of patch ``j - 1``
        (context plus the first ``j`` target patches): the advancing refine frame with
        perfect medians, as ``scaler.advancing_block_frames`` describes training.

        - prediction: ``q = asinh((y_pred - loc_0) / scale_0)``, undoing the anchor
          frame ``call`` returned (``loc_0``, ``scale_0`` of the last context patch).
        - target: ``z = asinh((y - loc_j) / scale_j)``; without ``use_asinh`` both
          skip the asinh.
        - loss: mean over observed target steps and levels of
          ``max(tau * (z - q), (tau - 1) * (z - q))``, steps weighted by their series'
          ``sample_weight``; plus ``self.losses``.

        Args:
            x: The context ``call`` received, (B, T) or (B, T, V).
            y: The horizon, (B, H) or (B, H, V); NaN marks a step left out of the loss.
            y_pred: ``call(x)``; computed when None.
            sample_weight: Optional (B,) weight per series.
            training: Forwarded to ``call`` when ``y_pred`` is None.

        Returns:
            Scalar loss, float32 or wider.

        Raises:
            ValueError: If ``y`` is missing or its horizon is not
                ``config.prediction_length``.
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-169
        # WHAT NOT TO DO: do not move this into a keras.losses.Loss passed to
        # compile(), and do not score the raw-space quantiles. The frame of horizon
        # patch j needs the context (x) and the true earlier patches (y); a Loss sees
        # only (y, y_pred). Raw-space pinball weights series by their scale and
        # trains every patch in the anchor frame, which _refine_block would then
        # misread. No custom train_step either (repo rule); stock fit() calls this.
        # See decisions.md D-169.
        if y is None:
            raise ValueError("TimestralModel.compute_loss needs the horizon y")
        if y_pred is None:
            y_pred = self(x, training=training)
        c = self.config
        ps = c.patch_size
        horizon = c.prediction_length
        x = keras.ops.convert_to_tensor(x)
        y = keras.ops.convert_to_tensor(y)
        if len(x.shape) == 2:
            x, y = keras.ops.expand_dims(x, -1), keras.ops.expand_dims(y, -1)
            y_pred = keras.ops.expand_dims(y_pred, 2)
        if int(y.shape[1]) != horizon:
            raise ValueError(f"y covers {y.shape[1]} steps; prediction_length is {horizon}")
        b, v = int(x.shape[0]), int(x.shape[2])
        work = str(np.result_type(getattr(y_pred.dtype, "name", str(y_pred.dtype)), "float32").name)

        context, context_obs = self._left_pad(*self._pack(keras.ops.cast(x, work)))
        target, target_obs = self._pack(keras.ops.cast(y, work))
        h = -(-horizon // ps)
        tail = h * ps - horizon
        window = keras.ops.concatenate(
            [context, keras.ops.pad(target, ((0, 0), (0, tail)))], axis=-1)
        window_obs = keras.ops.concatenate(
            [context_obs, keras.ops.pad(target_obs, ((0, 0), (0, tail)), constant_values=False)],
            axis=-1)
        loc, scale = causal_patch_loc_scale(
            window, window_obs, ps, c.minimum_scale, c.stabilize_with_global)
        ll = int(context.shape[-1]) // ps
        loc = keras.ops.reshape(loc, (b * v, -1, ps))[:, ll - 1:ll - 1 + h, 0]
        scale = keras.ops.reshape(scale, (b * v, -1, ps))[:, ll - 1:ll - 1 + h, 0]
        anchor_loc, anchor_scale = loc[:, :1, None], scale[:, :1, None]
        loc = keras.ops.repeat(loc, ps, axis=1)[:, :horizon]
        scale = keras.ops.repeat(scale, ps, axis=1)[:, :horizon]

        pred = keras.ops.reshape(
            keras.ops.transpose(keras.ops.cast(y_pred, work), (0, 2, 1, 3)),
            (b * v, horizon, c.num_quantiles))
        q = (pred - anchor_loc) / anchor_scale
        z = (target - loc) / scale
        if c.use_asinh:
            q, z = keras.ops.arcsinh(q), keras.ops.arcsinh(z)
        tau = keras.ops.cast(keras.ops.array(c.quantile_knots), work)
        err = keras.ops.expand_dims(z, -1) - q
        pinball = keras.ops.mean(keras.ops.maximum(tau * err, (tau - 1.0) * err), axis=-1)

        weight = keras.ops.cast(target_obs, work)
        if sample_weight is not None:
            per_series = keras.ops.reshape(keras.ops.cast(sample_weight, work), (b,))
            weight = weight * keras.ops.expand_dims(keras.ops.repeat(per_series, v), -1)
        loss = keras.ops.sum(pinball * weight) / keras.ops.maximum(keras.ops.sum(weight), 1.0)
        for extra in self.losses:
            loss = loss + keras.ops.cast(extra, work)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["config"] = TimestralConfig.from_dict(config["config"])
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        prediction_length: int = 96,
        patch_size: int = 32,
        quantile_levels: Optional[Tuple[float, ...]] = None,
        **kwargs
    ) -> "TimestralModel":
        """Create a Timestral model from a predefined variant.

        Args:
            variant: One of "5M", "28M", "123M", "209M".
            prediction_length: Forecast horizon in time steps.
            patch_size: Patch size (default 32).
            quantile_levels: Custom quantile levels (default 21 levels).
            **kwargs: Additional args passed to TimestralConfig; they take
                precedence over the variant's values.

        Returns:
            TimestralModel instance (unbuilt; it carries random weights).

        Raises:
            ValueError: If ``variant`` is not a key of ``MODEL_VARIANTS``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        vcfg = cls.MODEL_VARIANTS[variant].copy()
        vcfg.update(kwargs)

        knots = {}
        if quantile_levels is not None:
            knots["quantile_knots"] = tuple(quantile_levels)

        config = TimestralConfig(
            patch_size=patch_size,
            prediction_length=prediction_length,
            **knots,
            **vcfg,
        )
        return cls(config)


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
    return TimestralModel.from_variant(
        variant,
        prediction_length=prediction_length,
        patch_size=patch_size,
        quantile_levels=quantile_levels,
        **kwargs
    )


def create_timestral_by_variant(
    variant: str = "28M",
    prediction_length: int = 96,
    **kwargs
) -> TimestralModel:
    """Alias for create_timestral_model for API consistency."""
    return TimestralModel.from_variant(variant, prediction_length=prediction_length, **kwargs)
