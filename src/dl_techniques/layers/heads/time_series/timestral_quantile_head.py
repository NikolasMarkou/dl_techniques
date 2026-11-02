"""Quantile head of the Timestral forecaster.

Maps each decoder token of shape ``(..., N, d_model)`` to ``patch_size``
non-crossing quantile vectors, so the output has ``N * patch_size`` time steps
and one channel per quantile level. The median channel is read directly from
the projection. Every other level fans out from it through a cumulative sum of
non-negative increments, which is what keeps the quantiles ordered.

The head takes plain integers and floats instead of a model config, so the
layers package does not depend on ``dl_techniques.models``.
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.time_series.residual_projection_block import ResidualProjectionBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _working_dtype(dtype: Any) -> str:
    """The name of ``dtype`` promoted to at least float32."""
    return str(np.result_type(getattr(dtype, "name", str(dtype)), "float32").name)


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

    work = _working_dtype(raw.dtype)
    big = np.finfo(work).max
    return keras.ops.clip(
        keras.ops.cast(
            keras.ops.concatenate([median - down, median, median + up], axis=-1), work
        ),
        -big, big
    )

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.heads.time_series.timestral_quantile_head")
class TimestralQuantileHead(keras.layers.Layer):
    """Maps each decoder token to patch_size monotone quantile vectors in normalized space.

    :param d_model: Width of the decoder tokens.
    :type d_model: int
    :param patch_size: Time steps predicted per token.
    :type patch_size: int
    :param quantile_levels: Quantile levels, strictly increasing and containing 0.5.
    :type quantile_levels: Sequence[float]
    :param width_factor: Increment scale of the monotone fan-out.
    :type width_factor: float
    :param kwargs: Additional arguments for the Layer base class.
    :ivar projection: Residual block producing ``patch_size * num_quantiles`` raw outputs.
    :vartype projection: ResidualProjectionBlock
    :raises ValueError: If ``quantile_levels`` does not contain 0.5.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        quantile_levels: Sequence[float],
        width_factor: float,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.patch_size = patch_size
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        if 0.5 not in self.quantile_levels:
            raise ValueError(
                f"TimestralQuantileHead: quantile_levels must contain 0.5, the "
                f"channel the others fan out from; got {self.quantile_levels}"
            )
        self.width_factor = width_factor
        self.num_quantiles = len(self.quantile_levels)
        self.projection = ResidualProjectionBlock(
            d_model,
            4 * d_model,
            patch_size * self.num_quantiles,
            use_bias=False,
            name="projection",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape) -> None:
        self.projection.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        """(R, N, D) -> normalized quantiles (R, N * patch_size, Q) in at least fp32."""
        raw = self.projection(x)
        *leading, n_patches, _ = keras.ops.shape(raw)
        preds = keras.ops.reshape(raw, (*leading, n_patches, self.patch_size, self.num_quantiles))
        preds = keras.ops.cast(preds, _working_dtype(preds.dtype))
        preds = monotone_quantiles(preds, self.quantile_levels, self.width_factor)
        return keras.ops.reshape(preds, (*leading, n_patches * self.patch_size, self.num_quantiles))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """(..., N, D) -> (..., N * patch_size, Q)."""
        *leading, n_patches, _ = input_shape
        steps = None if n_patches is None else n_patches * self.patch_size
        return (*leading, steps, self.num_quantiles)

    def compute_output_spec(self, x: Any) -> keras.KerasTensor:
        """Report the at-least-fp32 dtype ``call`` returns, not the compute dtype."""
        return keras.KerasTensor(
            self.compute_output_shape(x.shape), dtype=_working_dtype(self.compute_dtype))

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "patch_size": self.patch_size,
            "quantile_levels": list(self.quantile_levels),
            "width_factor": self.width_factor,
        })
        return config
