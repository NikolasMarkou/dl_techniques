"""
Dechunk layer for H-Net dynamic chunking.

``DeChunkLayer`` takes the chunk-resolution hidden states ``(B, M, D)`` from the
deeper stage, together with the routing probabilities and boundary mask at full
resolution, and returns ``(B, L, D)``. It smooths the inner sequence with an
exponential moving average gated by each chunk's boundary probability,

    h_t = p_t * x_t + (1 - p_t) * h_{t-1},   h_{-1} = 0

then scatters the result so every non-boundary position repeats the most recent
chunk value.

The recurrence runs as a ``keras.ops.while_loop`` and fills its accumulator with
a one-hot masked add, where the reference calls a fused CUDA scan kernel. ``p``
is clamped before the scan and gathered through the same stable partition
``ChunkLayer`` uses, so inner column ``j`` carries the probability of the
position that produced it. The inner width ``M`` is the fixed ``max_chunks`` cap
and may be smaller or larger than ``L``; the scatter index is clipped to
``[0, M - 1]``. The layer holds no weights.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
"""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

from .indexing import batched_gather, dim, pad_permutation_to_width

# ---------------------------------------------------------------------

#: Default lower clamp on the boundary probability (dc.py:256).
DEFAULT_CLAMP_MIN = 1e-4

#: Default upper clamp on the boundary probability (dc.py:256).
DEFAULT_CLAMP_MAX = 1.0 - 1e-4

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.dynamic_chunking.dechunk_layer")
class DeChunkLayer(keras.layers.Layer):
    """Smooth an inner chunk sequence with an EMA and scatter it to full resolution.

    The layer combines the optional padding mask into ``boundary_mask`` with a
    logical and, rebuilds ``ChunkLayer``'s stable partition on that same
    predicate, gathers the clamped boundary probabilities into inner order, runs
    the EMA along the inner axis, and scatters the smoothed values back with
    ``cumsum(keep) - 1``. It has no weights, and gradients flow through the scan
    into the inner hidden states.

    Architecture:

    .. code-block:: text

                        boundary_mask [B, L]     mask [B, L]
                                  │                   │
                                  └─────── and ───────┘
                                        │
                                keep [B, L] bool
                                        │
                                  ┌─────┴─────────────┐
                                  ▼                   │
                    ┌───────────────────┐             │
                    │ stable partition, │             │
                    │ pad to width M    │             │
                    └─────────┬─────────┘             │
                           [B, M]                     │
                              ▼                       │
                    ┌───────────────────┐             │
   p_full [B, L] ──►│ batched_gather    │             │
                    └─────────┬─────────┘             │
                          p [B, M]                    │
                              ▼                       │
                    ┌───────────────────┐             │
 inner [B, M, D] ──►│ EMA scan          │             │
                    └─────────┬─────────┘             │
                        out [B, M, D]                 │
                              │                       ▼
                              │         ┌───────────────────┐
                              │         │ cumsum(keep) - 1, │
                              │         │ clip to [0, M-1]  │
                              │         └─────────┬─────────┘
                              │                [B, L]
                              └─────────┬─────────┘
                                        ▼
                              ┌───────────────────┐
                              │ batched_gather    │
                              └─────────┬─────────┘
                                        ▼
                                output [B, L, D]

    ``p_full`` is ``clip(boundary_prob[..., -1], clamp_min, clamp_max)``.

    EMA scan, step t:

    .. code-block:: text

                carry              p[:, t]            inner[:, t]
                  │                   │                   │
                  └───────────────────┴───────────────────┘
                                      ▼
                  ┌───────────────────────────────────────┐
                  │ carry = p_t * x_t + (1 - p_t) * carry │
                  └───────────────────┬───────────────────┘
                                      ▼
                  ┌───────────────────────────────────────┐
                  │ accumulator += carry masked to row t  │
                  └───────────────────┬───────────────────┘
                                      ▼
                          accumulator [B, M, D]

    Inner width M against outer length L:

    .. code-block:: text

        M > L   permutation right-padded with index 0 to M columns
        M = L   permutation used unchanged
        M < L   permutation truncated to the first M columns

    The padding and truncation come from
    :func:`~dl_techniques.layers.dynamic_chunking.indexing.pad_permutation_to_width`,
    which :class:`~dl_techniques.layers.dynamic_chunking.chunk_layer.ChunkLayer`
    also calls, so both layers treat the three cases the same way. The scatter
    index is clipped because ``M`` is the fixed ``max_chunks`` cap rather than
    the batch's own boundary count: a row with more than ``M`` boundaries would
    otherwise read past the inner sequence, and a caller who did not force a
    boundary at position 0 would read index ``-1``.

    :param clamp_min: Lower bound applied to ``p`` before the scan. Defaults to
        ``1e-4`` (``dc.py:256``).
    :type clamp_min: float
    :param clamp_max: Upper bound applied to ``p`` before the scan. Defaults to
        ``1 - 1e-4`` (``dc.py:256``).
    :type clamp_max: float
    :param kwargs: Additional :class:`keras.layers.Layer` arguments.

    :raises ValueError: If the clamp bounds are not finite floats with
        ``0 < clamp_min < clamp_max <= 1``.

    Example:
        >>> import keras, numpy as np
        >>> layer = DeChunkLayer()
        >>> inner = keras.ops.convert_to_tensor(np.random.randn(2, 3, 8).astype("float32"))
        >>> boundary = np.zeros((2, 12), dtype=bool)
        >>> boundary[:, 0] = True
        >>> prob = keras.ops.convert_to_tensor(boundary.astype("float32"))
        >>> out = layer(
        ...     inner,
        ...     boundary_prob=prob,
        ...     boundary_mask=keras.ops.convert_to_tensor(boundary),
        ... )
        >>> out.shape
        (2, 12, 8)
    """

    def __init__(
        self,
        clamp_min: float = DEFAULT_CLAMP_MIN,
        clamp_max: float = DEFAULT_CLAMP_MAX,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (("clamp_min", clamp_min), ("clamp_max", clamp_max)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a float, got {type(value).__name__}"
                )
        if not 0.0 < float(clamp_min) < float(clamp_max) <= 1.0:
            raise ValueError(
                "clamp bounds must satisfy 0 < clamp_min < clamp_max <= 1, got "
                f"clamp_min={clamp_min}, clamp_max={clamp_max}"
            )

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Validate the inner-sequence rank.

        This layer creates no weights. Keras passes the shape of the first call
        argument, so ``input_shape`` is the inner hidden states' shape and never
        ``boundary_prob``'s.

        :param input_shape: Shape of ``inner_hidden_states``, ``(B, M, D)``.
        :type input_shape: tuple

        :raises ValueError: If the input is not rank 3.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"DeChunkLayer expects rank-3 (batch, max_chunks, d_model) inner "
                f"hidden states, got shape {input_shape}"
            )
        super().build(input_shape)

    def _ema_scan(
        self, inner_hidden_states: keras.KerasTensor, p: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Run ``h_t = p_t * x_t + (1 - p_t) * h_{t-1}`` along the inner axis.

        :param inner_hidden_states: ``(B, M, D)``.
        :type inner_hidden_states: keras.KerasTensor
        :param p: ``(B, M)`` clamped boundary probability, already in the inner
            (boundary) order.
        :type p: keras.KerasTensor
        :return: ``(B, M, D)`` smoothed sequence.
        :rtype: keras.KerasTensor
        """
        inner_len = keras.ops.shape(inner_hidden_states)[1]

        # h_{-1} = 0 (dc.py:232-237).
        carry = keras.ops.zeros_like(inner_hidden_states[:, 0, :])
        accumulator = keras.ops.zeros_like(inner_hidden_states)
        inner_positions = keras.ops.arange(inner_len, dtype="int32")

        def cond(step, carry_state, accumulated):
            del carry_state, accumulated
            return step < inner_len

        def body(step, carry_state, accumulated):
            # DECISION D-014: keep the loop and the masked add; the closed form
            # nans at M >= 64, slice_update has no gradient. See decisions.md.
            p_step = keras.ops.expand_dims(p[:, step], axis=-1)
            carry_state = (
                p_step * inner_hidden_states[:, step, :]
                + (1.0 - p_step) * carry_state
            )
            # One-hot over the inner axis, selecting the current step.
            write_mask = keras.ops.cast(
                keras.ops.equal(inner_positions, step), carry_state.dtype
            )
            accumulated = accumulated + keras.ops.expand_dims(
                carry_state, axis=1
            ) * write_mask[None, :, None]
            return step + 1, carry_state, accumulated

        # maximum_iterations is the exact trip count; it lets the XLA backward
        # pass size its tensor list.
        _, _, accumulated = keras.ops.while_loop(
            cond,
            body,
            (0, carry, accumulator),
            maximum_iterations=inner_len,
        )
        return accumulated

    def call(
        self,
        inner_hidden_states: keras.KerasTensor,
        boundary_prob: Optional[keras.KerasTensor] = None,
        boundary_mask: Optional[keras.KerasTensor] = None,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Smooth the inner sequence and scatter it back to full resolution.

        :param inner_hidden_states: ``(B, M, D)`` chunk-resolution hidden states,
            as returned by the deeper stage.
        :type inner_hidden_states: keras.KerasTensor
        :param boundary_prob: ``(B, L, 2)`` routing distribution, or ``(B, L)``
            holding ``p`` directly. ``[..., -1]`` is taken either way. Required;
            it defaults to ``None`` only so Keras can introspect the signature.
        :type boundary_prob: keras.KerasTensor or None
        :param boundary_mask: ``(B, L)`` boolean or integer mask, where ``True``
            means the position starts a chunk. Required.
        :type boundary_mask: keras.KerasTensor or None
        :param mask: Optional ``(B, L)`` validity mask, combined into
            ``boundary_mask`` with a logical and, as
            :class:`~dl_techniques.layers.dynamic_chunking.chunk_layer.ChunkLayer`
            does, so both layers partition on the same predicate.
        :type mask: keras.KerasTensor or None
        :param training: Unused. Accepted so the layer composes with parents
            that pass it through.
        :type training: bool or None
        :return: ``(B, L, D)`` full-resolution hidden states.
        :rtype: keras.KerasTensor

        :raises ValueError: If ``boundary_prob`` or ``boundary_mask`` is missing.
        """
        del training

        if boundary_prob is None or boundary_mask is None:
            raise ValueError(
                "DeChunkLayer requires boundary_prob and boundary_mask; call it "
                "as layer(inner_hidden_states, boundary_prob=..., "
                "boundary_mask=...)"
            )

        compute_dtype = inner_hidden_states.dtype

        # Clamp the full-resolution probabilities before the gather (dc.py:256).
        # [..., -1] is the boundary channel, and the identity for a (B, L) input.
        if len(keras.ops.shape(boundary_prob)) == 3:
            p_full = boundary_prob[..., -1]
        else:
            p_full = boundary_prob
        p_full = keras.ops.clip(
            keras.ops.cast(p_full, compute_dtype), self.clamp_min, self.clamp_max
        )

        keep = keras.ops.cast(boundary_mask, "bool")
        if mask is not None:
            keep = keras.ops.logical_and(keep, keras.ops.cast(mask, "bool"))

        # ChunkLayer's stable partition, rebuilt on the same predicate so inner
        # column j carries its own position's probability (dc.py:265-269).
        seq_len = keras.ops.shape(keep)[1]
        # M, resolved statically when known.
        inner_len = dim(inner_hidden_states, 1)
        # DECISION D-010(b): masks come from arange broadcasts; keras.ops.tril
        # and triu raise TypeError under tf.function. See decisions.md.
        positions = keras.ops.expand_dims(
            keras.ops.arange(seq_len, dtype="int32"), axis=0
        )
        token_idx = positions + keras.ops.cast(
            keras.ops.logical_not(keep), "int32"
        ) * keras.ops.cast(seq_len, "int32")
        seq_sorted_indices = keras.ops.argsort(token_idx, axis=1)

        # DECISION D-029: take the permutation to width M with the same helper
        # ChunkLayer uses; slicing gives min(L, M) and breaks L < M. decisions.md.
        gather_idx = pad_permutation_to_width(
            seq_sorted_indices, inner_len
        )
        p = batched_gather(p_full, gather_idx)

        # The kernel-free recurrence (dc.py:333); see _ema_scan's D-014 anchor.
        out = self._ema_scan(inner_hidden_states, p)

        # Every non-boundary position repeats the most recent chunk value
        # (dc.py:302-308). M is fixed here, so the index needs clipping.
        plug_back_idx = (
            keras.ops.cumsum(keras.ops.cast(keep, "int32"), axis=1) - 1
        )
        plug_back_idx = keras.ops.minimum(
            keras.ops.maximum(plug_back_idx, 0),
            keras.ops.cast(inner_len - 1, "int32"),
        )

        # Per-row gather through the same helper ChunkLayer uses; take_along_axis
        # and numpy fancy indexing both fail here (D-029 in indexing).
        return batched_gather(out, plug_back_idx)

    def compute_output_shape(
        self,
        inner_hidden_states_shape: Tuple[Optional[int], ...],
        boundary_prob_shape: Optional[Tuple[Optional[int], ...]] = None,
        boundary_mask_shape: Optional[Tuple[Optional[int], ...]] = None,
        mask_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape ``(B, L, D)``.

        ``L`` is the full-resolution length and it comes from ``boundary_mask``
        or ``boundary_prob``, not from the inner sequence, so it is ``None``
        when neither shape is supplied.

        :param inner_hidden_states_shape: ``(B, M, D)``.
        :type inner_hidden_states_shape: tuple
        :param boundary_prob_shape: ``(B, L[, 2])`` or ``None``.
        :type boundary_prob_shape: tuple or None
        :param boundary_mask_shape: ``(B, L)`` or ``None``.
        :type boundary_mask_shape: tuple or None
        :param mask_shape: Unused; accepted so Keras may forward every call
            argument's shape.
        :type mask_shape: tuple or None
        :return: ``(B, L, D)``.
        :rtype: tuple
        """
        del mask_shape

        seq_len = None
        if boundary_mask_shape is not None:
            seq_len = boundary_mask_shape[1]
        elif boundary_prob_shape is not None:
            seq_len = boundary_prob_shape[1]

        return (inner_hidden_states_shape[0], seq_len, inner_hidden_states_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Serializable configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update(
            {"clamp_min": self.clamp_min, "clamp_max": self.clamp_max}
        )
        return config

# ---------------------------------------------------------------------
