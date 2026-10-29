"""Mode pooling for sequence data.

This module provides :class:`ModePooling`, a pooling layer with no weights by
default (``learnable_bandwidth`` adds one scalar) that collapses a sequence of
token vectors ``(batch, seq_len, embed_dim)`` into the dominant mode of the
token set instead of its mean. The mode is found by mean-shift iterations on a
Gaussian (or von Mises-Fisher) kernel density of the tokens, started at the
masked mean and, optionally, at tokens spread over the sequence, so a minority
of far-away tokens does not drag the summary the way a mean does.

References:
    - Cheng, 1995. Mean shift, mode seeking, and clustering.
    - Comaniciu and Meer, 2002. Mean shift: a robust approach toward feature
      space analysis.
    - Carreira-Perpinan, 2007. Gaussian mean-shift is an EM algorithm.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

from dl_techniques.layers.mixtures.mean_shift_ops import (
    _check_kernel,
    kde_log_density,
    mean_shift_step,
    median_heuristic_bandwidth,
)
from dl_techniques.utils.dtype_policy import accumulation_dtype, stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tensors import l2_normalize
from dl_techniques.utils.validation import positive_float, positive_int

from .sequence_pooling import SequencePooling, _check_exclude_positions

_NORMALIZE_EPS = 1e-12

_BANDWIDTH_FLOOR = 1e-6


@register_dl_technique("dl_techniques.layers.sequence_pooling.mode_pooling")
class ModePooling(keras.layers.Layer):
    """Pool a token set to its dominant mode with mean-shift iterations.

    Every kept token is a key of a kernel density estimate. One or more start
    points move for ``num_steps`` mean-shift steps against the fixed tokens (the
    tokens never move, so the density does not blur), and the output is the
    start that ends at the highest density. Masked and excluded tokens have
    exactly zero weight. A fully masked row (no kept token, including the case
    where ``exclude_positions`` removes every position) returns the plain mean
    over all positions, an explicit rule of this layer and finite; the result is
    the same as ``AttentionPooling`` gives, which reaches it by degeneration.

    The median-distance bandwidth of ``bandwidth="auto"`` is large when most
    token pairs are far apart (for example many outliers scattered in different
    directions), and then the mode drifts toward the mean; lower
    ``bandwidth_scale`` or pass a float on such data. With ``num_starts > 1`` the start tokens are picked by position in
    the sequence, so the output depends on token order; with ``num_starts=1``
    it does not.

    ``bandwidth="auto"`` sorts the ``seq_len ** 2`` pair distances of every sample,
    about ``O(seq_len^2 log seq_len)`` per sample, and that sort dominates the cost.
    On CPU in eager mode at batch 8 and ``embed_dim`` 64 a call takes 72, 420 and
    1700 ms at ``seq_len`` 256, 1024 and 2048, against 37, 45 and 50 ms with a
    float bandwidth. Pass a float ``bandwidth`` for long sequences. Distances are
    formed on points centred by the mean of the kept tokens, so a constant offset of
    the tokens moves the mode by the offset and by float32 rounding only, and a
    masked token has zero weight at any bandwidth, however small.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, seq_len, embed_dim]   │
        │  keep = mask and not excluded    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  h = bandwidth (median / fixed)  │
        │  start 0 = masked mean           │
        └──────────────┬───────────────────┘
                       │
            ┌──────────┴───────────┐
            │ num_starts == 1      │ num_starts > 1
            ▼                      ▼
        ┌──────────────┐   ┌──────────────────────────┐
        │ z = mean     │   │ z = mean + tokens at     │
        │ [B, 1, D]    │   │ evenly spaced ranks      │
        │              │   │ [B, num_starts, D]       │
        └──────┬───────┘   └────────────┬─────────────┘
               │                        │
               └───────────┬────────────┘
                           ▼
        ┌──────────────────────────────────┐
        │  num_steps x mean-shift step     │
        │  z <- softmax(kernel logits) @ X │
        └──────────────┬───────────────────┘
                       │
            ┌──────────┴───────────┐
            ▼                      ▼
        ┌──────────────┐   ┌──────────────────────────┐
        │ z[:, 0]      │   │ argmax of KDE log density│
        │              │   │ (no gradient), gather z  │
        └──────┬───────┘   └────────────┬─────────────┘
               │                        │
               └───────────┬────────────┘
                           ▼
        ┌──────────────────────────────────┐
        │  Output [B, embed_dim]           │
        │  (+ log density [B] if asked)    │
        └──────────────────────────────────┘

    Args:
        num_steps: Number of mean-shift steps, an integer >= 0. With 0 the
            output is the masked mean (or the best start when ``num_starts > 1``).
        bandwidth: Kernel bandwidth ``h``. ``"auto"`` uses the median pairwise
            distance among the kept tokens of each sample (no gradient); a
            positive float is used as given.
        bandwidth_scale: Positive factor multiplying the bandwidth, estimated
            or given.
        learnable_bandwidth: If True, adds one scalar weight ``log_scale``
            (initialised to 0) and multiplies the bandwidth by ``exp(log_scale)``.
        num_starts: Number of start points, an integer >= 1. Start 0 is the
            masked mean; the others are kept tokens at evenly spaced ranks of
            the kept tokens in sequence order. Ties go to the lowest start.
        kernel: ``"gaussian"`` or ``"vmf"``. ``"vmf"`` normalises the tokens and
            every iterate to unit length, so the output is a unit vector.
        exclude_positions: Sequence positions removed from the key set, like
            the same argument of ``SequencePooling``.
        return_density: If True, ``call`` returns ``(pooled, log_density)`` with
            ``log_density`` of shape ``(batch,)``, the Gaussian KDE log density at
            the returned mode.
        **kwargs: Passed to :class:`keras.layers.Layer`.

    Raises:
        ValueError: If an argument is out of range or has the wrong type, or the
            input is not rank 3 with a known last axis.
    """

    def __init__(
        self,
        num_steps: int = 5,
        bandwidth: Union[str, float] = "auto",
        bandwidth_scale: float = 1.0,
        learnable_bandwidth: bool = False,
        num_starts: int = 1,
        kernel: str = "gaussian",
        exclude_positions: Optional[List[int]] = None,
        return_density: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialise the mode pooling layer."""
        super().__init__(**kwargs)

        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps < 0
        ):
            raise ValueError(
                f"ModePooling: num_steps must be an integer >= 0, got {num_steps!r}"
            )
        self.num_steps = num_steps
        if isinstance(bandwidth, str):
            if bandwidth != "auto":
                raise ValueError(
                    f"ModePooling: bandwidth must be 'auto' or a positive float, "
                    f"got {bandwidth!r}"
                )
            self.bandwidth = bandwidth
        else:
            self.bandwidth = positive_float(bandwidth, "bandwidth", "ModePooling")
        self.bandwidth_scale = positive_float(
            bandwidth_scale, "bandwidth_scale", "ModePooling"
        )
        for name, value in (
            ("learnable_bandwidth", learnable_bandwidth),
            ("return_density", return_density),
        ):
            if not isinstance(value, bool):
                raise ValueError(
                    f"ModePooling: {name} must be a bool, got {value!r}"
                )
        self.learnable_bandwidth = learnable_bandwidth
        self.num_starts = positive_int(num_starts, "num_starts", "ModePooling")
        _check_kernel(kernel)
        self.kernel = kernel
        self.exclude_positions = _check_exclude_positions(exclude_positions)
        self.return_density = return_density
        self.log_scale = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the optional bandwidth weight.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If the input is not rank 3 or its last axis is unknown.
        """
        if len(input_shape) != 3 or input_shape[-1] is None:
            raise ValueError(
                f"ModePooling: expected input (batch, seq_len, embed_dim) with a "
                f"known embed_dim, got shape {tuple(input_shape)}"
            )
        if self.learnable_bandwidth:
            self.log_scale = self.add_weight(
                name="log_scale",
                shape=(),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def _keep_mask(
        self,
        keys: keras.KerasTensor,
        mask: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Fold the caller's mask and ``exclude_positions`` into a boolean keep-mask.

        Args:
            keys: Tokens ``(batch, seq_len, embed_dim)``.
            mask: Optional keep-mask ``(batch, seq_len)``, 1/True = keep.

        Returns:
            Boolean keep-mask ``(batch, seq_len)``.
        """
        keep = keras.ops.ones_like(keys[:, :, 0], dtype="bool")
        if mask is not None:
            keep = keras.ops.logical_and(keep, keras.ops.cast(mask, "bool"))
        if self.exclude_positions:
            positions = keras.ops.arange(keras.ops.shape(keys)[1])
            for pos in self.exclude_positions:
                keep = keras.ops.logical_and(
                    keep, keras.ops.expand_dims(positions != pos, 0)
                )
        return keep

    def _unit(self, x: keras.KerasTensor, acc: str) -> keras.KerasTensor:
        """L2-normalise the last axis; a zero row stays zero.

        Args:
            x: Tensor ``(..., embed_dim)`` in the accumulation dtype.
            acc: Accumulation dtype name.

        Returns:
            Tensor of the same shape with unit-norm rows.
        """
        return l2_normalize(x, axis=-1, eps=stability_floor(acc, _NORMALIZE_EPS))

    def _resolve_bandwidth(
        self,
        keys: keras.KerasTensor,
        keep: keras.KerasTensor,
        acc: str,
    ) -> keras.KerasTensor:
        """Return the bandwidth ``(batch or 1, 1, 1)`` in the accumulation dtype.

        Args:
            keys: Tokens ``(batch, seq_len, embed_dim)`` in the accumulation dtype.
            keep: Boolean keep-mask ``(batch, seq_len)``.
            acc: Accumulation dtype name.

        Returns:
            Positive bandwidth tensor, floored away from zero.
        """
        if self.bandwidth == "auto":
            h = median_heuristic_bandwidth(keys, keep)
        else:
            h = keras.ops.reshape(
                keras.ops.convert_to_tensor(self.bandwidth, dtype=acc), (1, 1, 1)
            )
        h = h * self.bandwidth_scale
        if self.learnable_bandwidth:
            h = h * keras.ops.exp(keras.ops.cast(self.log_scale, acc))
        return keras.ops.maximum(h, stability_floor(acc, _BANDWIDTH_FLOOR))

    def _token_starts(
        self,
        keys: keras.KerasTensor,
        keep: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Pick ``num_starts - 1`` kept tokens at evenly spaced ranks.

        The rank of start ``s`` among the ``n`` kept tokens is
        ``floor((n - 1) * s / (num_starts - 1))``, so the last start is the last
        kept token. A row with no kept token gets position 0 clipped into range.

        Args:
            keys: Tokens ``(batch, seq_len, embed_dim)``.
            keep: Boolean keep-mask ``(batch, seq_len)``.

        Returns:
            Start tokens ``(batch, num_starts - 1, embed_dim)``.
        """
        kept_so_far = keras.ops.cumsum(keras.ops.cast(keep, "int32"), axis=1)
        last_rank = keras.ops.maximum(kept_so_far[:, -1:] - 1, 0)
        steps = keras.ops.arange(1, self.num_starts, dtype="int32")
        ranks = keras.ops.floor_divide(
            last_rank * keras.ops.expand_dims(steps, 0), self.num_starts - 1
        )
        # The first position whose cumulative kept count exceeds the rank is
        # the position of the rank-th kept token.
        before = keras.ops.cast(
            kept_so_far[:, None, :] <= ranks[:, :, None], "int32"
        )
        positions = keras.ops.minimum(
            keras.ops.sum(before, axis=-1), keras.ops.shape(keys)[1] - 1
        )
        return SequencePooling._take_positions(keys, positions)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Pool the tokens to their dominant mode.

        Args:
            inputs: Tokens ``(batch, seq_len, embed_dim)``.
            mask: Optional keep-mask ``(batch, seq_len)``, 1/True = keep. A
                masked or excluded token has exactly zero weight. A row with no
                kept token returns the plain mean over all positions.

        Returns:
            Pooled tensor ``(batch, embed_dim)``, or ``(pooled, log_density)`` with
            ``log_density`` of shape ``(batch,)`` when ``return_density`` is True.
        """
        acc = accumulation_dtype(self.compute_dtype)
        keys = keras.ops.cast(inputs, acc)
        if self.kernel == "vmf":
            keys = self._unit(keys, acc)
        keep = self._keep_mask(keys, mask)
        any_kept = keras.ops.any(keep, axis=1, keepdims=True)

        # A row with no kept token averages every position (explicit rule).
        averaged = keras.ops.cast(
            keras.ops.logical_or(keep, keras.ops.logical_not(any_kept)), acc
        )
        start = keras.ops.sum(keys * averaged[:, :, None], axis=1) / keras.ops.sum(
            averaged, axis=1, keepdims=True
        )
        if self.kernel == "vmf":
            start = self._unit(start, acc)

        h = self._resolve_bandwidth(keys, keep, acc)

        # DECISION plan-2026-10-29T104646-27f045c5/D-009: the starts move against the fixed tokens and the best one is picked by a hard argmax without gradient.
        # Do not blur the keys with mean_shift_iterate, soften the pick, or gather with take_along_axis. See decisions.md.
        z = keras.ops.expand_dims(start, 1)
        if self.num_starts > 1:
            z = keras.ops.concatenate([z, self._token_starts(keys, keep)], axis=1)
        for _ in range(self.num_steps):
            z = mean_shift_step(z, keys, keys, h, keep, self.kernel)[0]
            if self.kernel == "vmf":
                z = self._unit(z, acc)

        log_density = None
        if self.num_starts > 1 or self.return_density:
            log_density = kde_log_density(z, keys, h, keep)
        if self.num_starts > 1:
            best = keras.ops.cast(
                keras.ops.argmax(keras.ops.stop_gradient(log_density), axis=1),
                "int32",
            )
            pooled = SequencePooling._take_positions(z, best)
            log_density = SequencePooling._take_positions(log_density, best)
        else:
            pooled = z[:, 0, :]
            if log_density is not None:
                log_density = log_density[:, 0]

        pooled = keras.ops.where(any_kept, pooled, start)
        pooled = keras.ops.cast(pooled, self.compute_dtype)
        if self.return_density:
            return pooled, keras.ops.cast(log_density, self.compute_dtype)
        return pooled

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...],
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """Compute the output shape.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            ``(batch, embed_dim)``, or that and ``(batch,)`` when
            ``return_density`` is True.
        """
        pooled = (input_shape[0], input_shape[-1])
        if self.return_density:
            return pooled, (input_shape[0],)
        return pooled

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            "num_steps": self.num_steps,
            "bandwidth": self.bandwidth,
            "bandwidth_scale": self.bandwidth_scale,
            "learnable_bandwidth": self.learnable_bandwidth,
            "num_starts": self.num_starts,
            "kernel": self.kernel,
            "exclude_positions": list(self.exclude_positions),
            "return_density": self.return_density,
        })
        return config
