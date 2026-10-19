"""Probabilistic forecasting with Timestral, including the clampsym test-time augmentation.

Pure Keras 3 + NumPy. TimestralForecaster.forecast takes a context (B, V, T)
(series, variates, time; NaN marks a missing value) and returns quantile forecasts
(B, V, H, Q) as numpy arrays. Horizons longer than one decode block are forecast
block by block, feeding each block's median back as context.

Clampsym (both options on by default):

- Symmetric averaging forecasts the context and its negation, reflects the second
  forecast back (-Q(1 - p)), and averages the two. Every covariate row is negated
  with the context: numeric covariates, 0/1 indicators, target encodings and raw codes.
- Non-negative clipping clamps the forecast at 0 for every series whose observed
  context is entirely non-negative.

Categorical covariates are target-encoded against the series' forward-filled context
(see timestral.covariate_encoding) and then handled like numeric covariates.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

import keras
from keras import ops

from dl_techniques.models.time_series.timestral.covariate_encoding import target_encode
from dl_techniques.models.time_series.timestral.loading import load_model
from dl_techniques.models.time_series.timestral.model import (
    ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET, TimestralModel
)
from dl_techniques.models.time_series.timestral.scaler import causal_patch_loc_scale
from dl_techniques.utils.logger import logger

ArrayLike = Union[np.ndarray, "keras.KerasTensor"]


@dataclass(frozen=True)
class TimestralForecast:
    """Quantile forecast from TimestralForecaster.

    Attributes:
        quantiles: (B, V, H, Q) forecast values as a numpy array, sorted along
            the quantile axis.
        quantile_levels: The Q probability levels.
    """
    quantiles: np.ndarray
    quantile_levels: Tuple[float, ...]

    @property
    def median(self) -> np.ndarray:
        """(B, V, H) median forecast."""
        return self.quantiles[..., self.quantile_levels.index(0.5)]

    def quantile(self, level: float) -> np.ndarray:
        """(B, V, H) forecast at one of the model's quantile levels."""
        if level not in self.quantile_levels:
            raise ValueError(f"Level {level} not in {self.quantile_levels}")
        return self.quantiles[..., self.quantile_levels.index(level)]


def ffill_with_mask(inputs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Forward-fill unobserved positions; leading unobserved positions become 0."""
    if mask.all():
        return inputs
    idx = np.where(mask, np.arange(inputs.shape[-1]), 0)
    idx = np.maximum.accumulate(idx, axis=-1)
    filled = np.take_along_axis(inputs, idx, axis=-1)
    return np.where(mask, filled, 0.0)


def pad_to_patch(
    inputs: np.ndarray, mask: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Left-pad to a multiple of patch_size; padding is 0 and unobserved."""
    ctx_len = inputs.shape[-1]
    if ctx_len % patch_size != 0:
        pad_len = patch_size - (ctx_len % patch_size)
        inputs = np.pad(inputs, [(0, 0)] * (inputs.ndim - 1) + [(pad_len, 0)],
                        constant_values=0.0)
        mask = np.pad(mask, [(0, 0)] * (mask.ndim - 1) + [(pad_len, 0)],
                      constant_values=False)
    return inputs, mask


def replace_extreme_values(
    x: np.ndarray, threshold: float, replacement: float = 0.0
) -> np.ndarray:
    keep = np.isfinite(x) & (np.abs(x) < threshold)
    return np.where(keep, x, replacement)


def nonnegative_context(inputs: np.ndarray, observation_mask: np.ndarray) -> np.ndarray:
    """(B, V): whether every observed context value is non-negative (and one exists)."""
    observed = np.isfinite(inputs) & observation_mask
    return observed.any(axis=-1) & np.all((inputs >= 0) | ~observed, axis=-1)


class TimestralForecaster:
    """Block-autoregressive quantile forecaster around a TimestralModel.

    All arrays are numpy; the network forward pass runs through the Keras model.

    Args:
        model: The network.
        max_context: Context cap in time steps (most recent kept); defaults to
            the trained context length.
        per_shot_horizon_patches: Patches decoded per block, capped at the trained
            block size.
        quantile_real_cap_k: Clamp forecasts to [min - k * s, max + k * s] of the
            observed context, s being the scale of the last context patch. 0 disables.
        extreme_value_threshold: Fed-back medians at or above this magnitude become 0.
    """

    def __init__(
        self,
        model: TimestralModel,
        max_context: Optional[int] = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        extreme_value_threshold: float = 1e30,
    ) -> None:
        if per_shot_horizon_patches < 1:
            raise ValueError(
                f"per_shot_horizon_patches must be >= 1, got {per_shot_horizon_patches}"
            )
        self.model = model
        self.config = model.config
        self.max_context = (
            max_context if max_context is not None else model.config.context_length
        )
        self.per_shot_horizon_patches = min(
            per_shot_horizon_patches, model.config.block_size
        )
        self.quantile_real_cap_k = quantile_real_cap_k
        self.extreme_value_threshold = extreme_value_threshold
        self.quantile_levels = tuple(model.config.quantile_knots)
        self.median_index = model.config.median_index

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: Union[str, Path],
        max_context: Optional[int] = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[Union[str, bool]] = None,
        local_files_only: bool = False,
    ) -> "TimestralForecaster":
        """Load an exported checkpoint (config.json + model.safetensors).

        path_or_repo_id is a local directory or a Hugging Face Hub repo id.
        revision, cache_dir, token and local_files_only only apply to Hub repos.
        """
        model = load_model(
            path_or_repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        return cls(model, max_context, per_shot_horizon_patches, quantile_real_cap_k)

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def supports_covariates(self) -> bool:
        return self.config.covariate_enabled

    @property
    def supports_known_covariates(self) -> bool:
        return self.config.supports_known_covariates

    def forecast(
        self,
        context: ArrayLike,
        prediction_length: int,
        observation_mask: Optional[ArrayLike] = None,
        past_covariates: Optional[ArrayLike] = None,
        known_covariates: Optional[ArrayLike] = None,
        categorical_past_covariates: Optional[ArrayLike] = None,
        categorical_known_covariates: Optional[ArrayLike] = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ) -> TimestralForecast:
        """Forecast prediction_length steps after context.

        Args:
            context: (B, V, T), (B, T) (one variate) or (T,) (one series). NaN marks
                a missing value. The variates of a series are forecast jointly.
            prediction_length: Horizon in time steps.
            observation_mask: Optional boolean mask shaped like context; False marks
                additional missing values.
            past_covariates: (B, C_past, T) covariates known over the context only.
            known_covariates: (B, C_known, T + prediction_length) covariates known over
                the context and the horizon.
            categorical_past_covariates: (B, C, T) category codes (any numbers; NaN
                marks a missing value) known over the context only.
            categorical_known_covariates: (B, C, T + prediction_length) category codes
                known over the context and the horizon.
                Categorical covariates are target-encoded per series against the
                forward-filled target context. A series with several target variates
                has no single target to encode against, so its codes are used as is.
            symmetric_averaging: Average with the reflected forecast of -context, with
                every covariate row negated too.
            clip_nonnegative: Clamp at 0 the series whose observed context is
                non-negative.

        Returns:
            A TimestralForecast with quantiles (B, V, prediction_length, Q) as numpy.
        """
        if prediction_length < 1:
            raise ValueError(
                f"prediction_length must be >= 1, got {prediction_length}"
            )
        inputs, mask = self._as_context(context, observation_mask)
        b = inputs.shape[0]
        t = inputs.shape[-1]
        past = self._as_covariates(past_covariates, b, t, "past_covariates")
        known = self._as_covariates(
            known_covariates, b, t + prediction_length, "known_covariates"
        )
        cat_past = self._as_covariates(
            categorical_past_covariates, b, t, "categorical_past_covariates"
        )
        cat_known = self._as_covariates(
            categorical_known_covariates,
            b, t + prediction_length, "categorical_known_covariates",
        )
        covariates = {"past_covariates": past, "known_covariates": known}
        if cat_past is not None or cat_known is not None:
            target = self._encoding_target(inputs, mask)
            enc_past = self._encode_categorical(cat_past, target)
            enc_known = self._encode_categorical(cat_known, target)
            covariates = {
                "past_covariates": _cat_rows(past, enc_past),
                "known_covariates": _cat_rows(known, enc_known),
            }
        # Mirrored pass forecasts -context with every covariate row negated.
        # Target encodings are linear in the target, so negating them gives
        # the encodings against -context.
        mirrored_covariates = {name: _negate(r) for name, r in covariates.items()}
        if covariates["known_covariates"] is not None and not self.supports_known_covariates:
            raise ValueError("This checkpoint cannot condition on known-future covariates")

        quantiles = self._forecast_raw(inputs, mask, prediction_length, **covariates)
        if symmetric_averaging:
            levels = self.quantile_levels
            if not all(
                abs(lo + hi - 1.0) < 1e-9
                for lo, hi in zip(levels, reversed(levels))
            ):
                raise ValueError(
                    f"Symmetric averaging needs quantile levels symmetric about 0.5, "
                    f"got {levels}"
                )
            mirrored = self._forecast_raw(
                -inputs, mask, prediction_length, **mirrored_covariates
            )
            quantiles = 0.5 * (quantiles + (-mirrored)[..., ::-1])
        if clip_nonnegative:
            eligible = nonnegative_context(inputs, mask)
            quantiles = np.where(eligible[..., None, None],
                                 np.maximum(quantiles, 0.0), quantiles)
        return TimestralForecast(quantiles=quantiles, quantile_levels=self.quantile_levels)

    def _as_context(
        self,
        context: ArrayLike,
        observation_mask: Optional[ArrayLike],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(context, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, None]
        elif x.ndim == 2:
            x = x[:, None]
        if x.ndim != 3:
            raise ValueError(
                f"Context must be (T,), (B, T) or (B, V, T), got {x.shape}"
            )
        mask = np.isfinite(x)
        if observation_mask is not None:
            user_mask = np.broadcast_to(
                np.asarray(observation_mask, dtype=bool), x.shape
            )
            mask = mask & user_mask
        return np.where(mask, x, np.zeros_like(x)), mask

    def _encoding_target(
        self, inputs: np.ndarray, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """(B, T) target to encode against, or None when a series has several targets.

        The forward-filled target from its first observation on (so left padding or
        a late start does not enter the category means), over the last max_context
        steps, the ones the model reads; NaN elsewhere.
        """
        if inputs.shape[1] != 1:
            return None
        t = inputs.shape[-1]
        started = np.maximum.accumulate(mask[:, 0].astype(np.int32), axis=-1) > 0
        in_window = np.arange(t) >= t - self.max_context
        return np.where(
            started & in_window, ffill_with_mask(inputs, mask)[:, 0], np.nan
        )

    def _encode_categorical(
        self, codes: Optional[np.ndarray], target_context: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Target-encode (B, C, L) codes against (B, T) targets; raw codes when None."""
        if codes is None or target_context is None:
            return None if codes is None else np.asarray(codes, dtype=np.float32)
        t = target_context.shape[-1]
        rows = np.asarray(codes, dtype=np.float64)
        out = np.empty(rows.shape, dtype=np.float32)
        for i in range(rows.shape[0]):
            for j in range(rows.shape[1]):
                future = rows[i, j, t:] if rows.shape[-1] > t else None
                past_enc, future_enc = target_encode(
                    target_context[i], rows[i, j, :t], future
                )
                # A missing code before the target starts is padding, not the
                # missing category.
                out[i, j, :t] = np.where(
                    np.isnan(target_context[i]) & np.isnan(rows[i, j, :t]),
                    np.nan, past_enc,
                )
                if future_enc is not None:
                    out[i, j, t:] = future_enc
        return out

    def _as_covariates(
        self,
        values: Optional[ArrayLike],
        batch: int,
        length: int,
        name: str,
    ) -> Optional[np.ndarray]:
        """Covariates as float32 numpy (B, C, L); None stays None."""
        if values is None:
            return None
        if not self.supports_covariates:
            raise ValueError(
                f"This checkpoint was not trained with covariates; got {name}"
            )
        v = np.asarray(values, dtype=np.float32)
        if v.ndim == 2:
            v = v[:, None]
        if v.ndim != 3 or v.shape[0] != batch or v.shape[-1] != length:
            raise ValueError(
                f"{name} must be (B={batch}, C, {length}), got {v.shape}"
            )
        if v.shape[1] == 0:
            return None
        return v

    def _forecast_raw(
        self,
        inputs: np.ndarray,
        observation_mask: np.ndarray,
        prediction_length: int,
        past_covariates: Optional[np.ndarray] = None,
        known_covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One forecast pass: (B, V, T) inputs to (B, V, H, Q) raw-space quantiles."""
        rows, mask, role_ids, known_future, known_future_mask, n_tgt = (
            self._assemble_rows(
                inputs, observation_mask, prediction_length,
                past_covariates, known_covariates,
            )
        )
        quantiles = self._block_ar_decode(
            rows, mask, prediction_length, role_ids,
            known_future, known_future_mask,
        )
        quantiles = quantiles[:, :n_tgt]
        if self.quantile_real_cap_k > 0:
            quantiles = self._cap_quantiles(quantiles, rows[:, :n_tgt], mask[:, :n_tgt])
        if not np.isfinite(quantiles).all():
            raise ValueError(
                "Forecast produced non-finite values; the input is degenerate"
            )
        return quantiles

    def _cap_quantiles(
        self, quantiles: np.ndarray, inputs: np.ndarray, observation_mask: np.ndarray
    ) -> np.ndarray:
        b, v, h, q = quantiles.shape
        r, t = inputs.reshape(-1, inputs.shape[-1]).shape
        q_flat = quantiles.reshape(r, h, q)
        not_obs = ~observation_mask.reshape(r, t)
        x_flat = inputs.reshape(r, t)
        ctx_max = np.where(not_obs, -np.inf, x_flat).max(axis=-1, keepdims=True)
        ctx_min = np.where(not_obs, np.inf, x_flat).min(axis=-1, keepdims=True)
        ctx_max = np.where(np.isfinite(ctx_max), ctx_max, 0.0)
        ctx_min = np.where(np.isfinite(ctx_min), ctx_min, 0.0)
        c = self.config
        _, scale = causal_patch_loc_scale(
            x_flat, observation_mask.reshape(r, t), c.patch_size,
            c.minimum_scale, c.stabilize_with_global,
        )
        anchor_scale = np.asarray(scale)[..., -1:] * self.quantile_real_cap_k
        lower = (ctx_min - anchor_scale)[:, :, None]
        upper = (ctx_max + anchor_scale)[:, :, None]
        return np.clip(q_flat, lower, upper).reshape(b, v, h, q)

    def _assemble_rows(
        self,
        inputs: np.ndarray,
        observation_mask: np.ndarray,
        prediction_length: int,
        past_covariates: Optional[np.ndarray] = None,
        known_covariates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], int]:
        """Stack target and covariate rows.

        Returns:
            (rows, mask, role_ids, known_future, known_future_mask, n_tgt).
        """
        ps = self.patch_size
        b, n_tgt, ctx0 = inputs.shape
        horizon = math.ceil(prediction_length / ps) * ps

        ctx_rows = [ffill_with_mask(inputs, observation_mask)]
        ctx_masks = [observation_mask]
        roles = [ROLE_TARGET] * n_tgt

        n_past = 0
        if past_covariates is not None:
            if past_covariates.shape[-1] != ctx0:
                raise ValueError("Past covariates must cover exactly the context")
            n_past = past_covariates.shape[1]
            ctx_masks.append(~np.isnan(past_covariates))
            ctx_rows.append(np.nan_to_num(past_covariates, nan=0.0))
            roles += [ROLE_PAST_COV] * n_past

        n_known = 0
        known_future_vals: Optional[np.ndarray] = None
        known_future_obs: Optional[np.ndarray] = None
        if known_covariates is not None:
            n_known = known_covariates.shape[1]
            kc_ctx = known_covariates[..., :-prediction_length]
            kc_fut = known_covariates[..., -prediction_length:]
            if kc_ctx.shape[-1] != ctx0:
                raise ValueError(
                    "Known covariates must cover the context plus the horizon"
                )
            ctx_masks.append(~np.isnan(kc_ctx))
            ctx_rows.append(np.nan_to_num(kc_ctx, nan=0.0))
            roles += [ROLE_KNOWN_COV] * n_known
            known_future_obs = ~np.isnan(kc_fut)
            known_future_vals = np.nan_to_num(kc_fut, nan=0.0)

        rows = np.concatenate(ctx_rows, axis=1)
        mask = np.concatenate(ctx_masks, axis=1)
        if rows.shape[-1] > self.max_context:
            rows = rows[:, :, -self.max_context:]
            mask = mask[:, :, -self.max_context:]
        rows, mask = pad_to_patch(rows, mask, ps)
        v_all = rows.shape[1]
        if n_past == 0 and n_known == 0:
            return rows, mask, None, None, None, n_tgt

        role_ids = np.tile(np.asarray(roles, dtype=np.int32), b)
        known_future: Optional[np.ndarray] = None
        known_future_mask: Optional[np.ndarray] = None
        if n_known > 0:
            k0 = n_tgt + n_past
            plen = known_future_vals.shape[-1]
            kf = np.zeros((b, v_all, horizon), dtype=rows.dtype)
            kfm = np.zeros((b, v_all, horizon), dtype=bool)
            kf[:, k0:k0 + n_known, :plen] = known_future_vals
            kfm[:, k0:k0 + n_known, :plen] = known_future_obs
            known_future = kf.reshape(b * v_all, horizon)
            known_future_mask = kfm.reshape(b * v_all, horizon)
        return rows, mask, role_ids, known_future, known_future_mask, n_tgt

    def _block_ar_decode(
        self,
        inputs: np.ndarray,
        observation_mask: np.ndarray,
        prediction_length: int,
        role_ids: Optional[np.ndarray],
        known_future: Optional[np.ndarray],
        known_future_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Decode block by block in the packed (R, T) layout; returns (B, V, H, Q).

        After each block the context grows by the block's feedback: targets carry
        their median, known-future covariates their real future, past-only
        covariates a blank. Past max_context the oldest steps are dropped.
        """
        ps = self.patch_size
        b, v, ctx_len = inputs.shape
        context = inputs.reshape(b * v, ctx_len)
        mask = observation_mask.reshape(b * v, ctx_len)
        group_ids = np.repeat(np.arange(b), v)

        known_rows = past_rows = target_rows = None
        if role_ids is not None:
            known_rows = role_ids == ROLE_KNOWN_COV
            past_rows = (role_ids != ROLE_TARGET) & ~known_rows
            target_rows = role_ids == ROLE_TARGET

        blocks = []
        remaining = math.ceil(prediction_length / ps)
        offset = 0
        while remaining > 0:
            block = min(self.per_shot_horizon_patches, remaining)
            blen = block * ps
            kf_block = (
                known_future[:, offset:offset + blen]
                if known_future is not None else None
            )
            kfm_block = (
                known_future_mask[:, offset:offset + blen]
                if known_future_mask is not None else None
            )
            quantiles = self.model.predict_block(
                context, mask, group_ids, block, role_ids, kf_block, kfm_block
            )
            quantiles = np.asarray(quantiles, dtype=np.float32)
            quantiles = self._refine_block(quantiles, context, mask, target_rows)
            blocks.append(quantiles)
            remaining -= block
            offset += blen
            if remaining <= 0:
                break
            feedback = replace_extreme_values(
                quantiles[..., self.median_index], self.extreme_value_threshold
            )
            feedback_mask = np.ones_like(feedback, dtype=bool)
            if role_ids is not None:
                if kf_block is not None:
                    feedback = np.where(known_rows[:, None], kf_block, feedback)
                    feedback_mask = np.where(
                        known_rows[:, None], kfm_block, feedback_mask
                    )
                feedback = np.where(past_rows[:, None],
                                    np.zeros_like(feedback), feedback)
                feedback_mask = feedback_mask & ~past_rows[:, None]
            context = np.concatenate([context, feedback], axis=-1)
            mask = np.concatenate([mask, feedback_mask], axis=-1)
            if context.shape[-1] > self.max_context:
                keep = max(ps, (self.max_context // ps) * ps)
                context = context[..., -keep:]
                mask = mask[..., -keep:]

        packed = np.concatenate(blocks, axis=-2)[..., :prediction_length, :]
        return packed.reshape(b, v, prediction_length, -1)

    def _refine_block(
        self,
        quantiles: np.ndarray,
        context: np.ndarray,
        mask: np.ndarray,
        target_rows: Optional[np.ndarray],
    ) -> np.ndarray:
        """Move one decoded block from the anchor frame to the advancing refine frame.

        The model denormalizes the whole block (R, H * patch_size, Q) with the frame
        of the last context patch. The first horizon patch keeps that frame; each
        later one is re-denormalized with the model's advancing_frame logic.
        Covariate rows keep their quantiles.
        """
        ps = self.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        refined = np.asarray(self.model._refine_block(
            quantiles, context, mask, target_rows,
        ), dtype=np.float32)
        return refined


def _negate(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return -x if x is not None else None


def _cat_rows(
    a: Optional[np.ndarray], b: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Concatenate covariate rows along the channel axis, either side possibly None."""
    if a is None or b is None:
        return a if b is None else b
    return np.concatenate([a, b], axis=1)
