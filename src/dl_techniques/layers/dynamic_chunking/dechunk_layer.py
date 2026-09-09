"""H-Net's dechunk layer: the EMA smoother that returns to full resolution.

``DeChunkLayer`` is the third and last of H-Net's three dynamic-chunking layers.
:class:`~dl_techniques.layers.dynamic_chunking.routing_module.RoutingModule`
decides where chunks begin,
:class:`~dl_techniques.layers.dynamic_chunking.chunk_layer.ChunkLayer` gathers the
boundary tokens down to a short inner sequence, the deeper stage runs on that
short sequence, and this layer puts the result back::

    inner (B, M, D) + boundary_prob (B, L[, 2]) + boundary_mask (B, L)  ->  (B, L, D)

Two operations, in this order (``dc.py:239-308``):

1. **An exponential moving average along the INNER axis**, gated by the boundary
   probability of the chunk that column came from::

       h_t = p_t * x_t + (1 - p_t) * h_{t-1},    h_{-1} = 0

   A confidently-detected chunk (``p -> 1``) replaces the running state; an
   unconfident one blends with its predecessor. ``p`` is clamped to
   ``[1e-4, 1 - 1e-4]`` first (``dc.py:256``), so neither gate ever closes fully.

2. **A scatter back to full resolution** with
   ``plug_back_idx = cumsum(boundary_mask) - 1`` (``dc.py:302-308``): every
   non-boundary position repeats the value of the most recent chunk.

Why the recurrence is written as a loop, and why it must stay one
-----------------------------------------------------------------

The reference computes step 1 by re-expressing the EMA as a diagonal SSM and
handing it to ``mamba_chunk_scan_combined`` (``dc.py:275-294``: ``dt =
log(1/(1-p))``, ``x = h/dt``, ``A = -1``, ``B = p``, ``C = 1``, whose discretized
update is exactly ``h_t = (1-p_t) h_{t-1} + p_t x_t``). That is a *throughput*
dependency on a CUDA kernel, not a definitional one; the unambiguous ground truth
is the reference's own kernel-free ``step()`` at ``dc.py:333``::

    result = p * current_hidden_states + (1 - p) * inference_params.last_value

This port implements that recurrence with ``keras.ops.while_loop``. The obvious
"optimization" -- reassociating into the closed form ``A_t = prod(1 - p_s)``,
``h_t = A_t * cumsum(p_s x_s / A_s)`` -- is REJECTED on measurement, not on
taste, and so is the obvious way of filling the accumulator
(``keras.ops.slice_update``, which is not differentiable on this backend); see
the ``# DECISION`` anchor in :meth:`DeChunkLayer._ema_scan` for both.

Divergences from the reference, both forced by the fixed-width port (D-007)
--------------------------------------------------------------------------

``plug_back_idx`` is clipped to ``[0, M - 1]``. Upstream it needs no clipping
because ``M`` is the batch's own maximum boundary count and position 0 is always
a boundary (``dc.py:95-96``), so the index is always in range. Here ``M`` is the
fixed ``max_chunks`` cap, so a row with more boundaries than ``M`` would index
past the inner sequence, and a caller who did not force a boundary at position 0
would index ``-1``. Both are clipped to a defined, deterministic value rather
than left to the backend's out-of-range behaviour, which differs between NumPy
(wraps) and TensorFlow (undefined). Both clips are guarded.

The inner width ``M`` may also EXCEED the outer length ``L``, which is the normal
case for a short input: ``default_max_chunks(2, max_seq_len=2048)`` is ``(1024,)``,
so any prompt below 1024 bytes has ``L < M``. The permutation is then right-padded
with index ``0`` to width ``M``, by the SAME
:func:`~dl_techniques.layers.dynamic_chunking.indexing.pad_permutation_to_width`
``ChunkLayer`` calls -- the two layers' handling of ``M > L``, ``M == L`` and
``M < L`` is symmetric by construction rather than by convention, because the
asymmetry (``ChunkLayer`` padded, this layer did not) was a real shipped defect
that made every H-Net raise on any sequence shorter than its own ``max_chunks[0]``.

The layer holds **no weights**: it is a gather, a clamp, a scan and a scatter.
The test suite asserts that absence deliberately rather than skipping the
gradient-flow check.

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

#: ``torch.clamp(boundary_prob[..., -1].float(), min=1e-4, ...)`` -- ``dc.py:256``.
DEFAULT_CLAMP_MIN = 1e-4

#: ``torch.clamp(..., max=1 - (1e-4))`` -- ``dc.py:256``.
DEFAULT_CLAMP_MAX = 1.0 - 1e-4


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.dynamic_chunking.dechunk_layer")
class DeChunkLayer(keras.layers.Layer):
    """Smooth an inner chunk sequence with an EMA and scatter it to full resolution.

    Architecture:

    .. code-block:: text

        boundary_prob (B, L, 2)            boundary_mask (B, L)     mask (B, L)
              |                                   |                      |
        p = clamp(prob[..., -1], 1e-4, 1-1e-4)    +---- logical_and -----+
              |             (dc.py:256)           |
              |                              keep (B, L) bool
              |                                   |
              |     token_idx = arange(L) + (~keep) * L      (dc.py:265-269)
              |     seq_sorted_indices = argsort(token_idx)
              +--> batched_gather(p, pad_to_width(idx, M))  (dc.py:271-273)
                                     |
                                p (B, M)          inner (B, M, D)
                                     |                   |
                                     +---- EMA scan -----+   (dc.py:333)
                                     h_t = p_t x_t + (1-p_t) h_{t-1}
                                              |
                                        out (B, M, D)
                                              |
                      plug_back_idx = clip(cumsum(keep) - 1, 0, M-1)
                                              |            (dc.py:302-308)
                                        batched_gather
                                              |
                                        (B, L, D)

    ``p`` is gathered through ChunkLayer's *same* stable partition, so inner
    column ``j`` is paired with the boundary probability of the very position
    ChunkLayer put there. The two layers therefore agree by construction; a
    divergence in either one's partition is a defect the parity guards see.

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
        """Validate the inner-sequence rank. This layer creates no weights.

        Keras forwards the shape of the FIRST call argument here because
        ``build`` takes a single parameter, so ``input_shape`` is the inner
        hidden states' shape, never ``boundary_prob``'s.

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
        """``h_t = p_t * x_t + (1 - p_t) * h_{t-1}`` along the inner axis.

        :param inner_hidden_states: ``(B, M, D)``.
        :type inner_hidden_states: keras.KerasTensor
        :param p: ``(B, M)`` clamped boundary probability, already in the inner
            (boundary) order.
        :type p: keras.KerasTensor
        :return: ``(B, M, D)`` smoothed sequence.
        :rtype: keras.KerasTensor
        """
        inner_len = keras.ops.shape(inner_hidden_states)[1]

        # h_{-1} = 0 -- `dc.py:232-237` allocates `DeChunkState.last_value` as
        # `torch.zeros`.
        carry = keras.ops.zeros_like(inner_hidden_states[:, 0, :])  # (B, D)
        accumulator = keras.ops.zeros_like(inner_hidden_states)  # (B, M, D)
        inner_positions = keras.ops.arange(inner_len, dtype="int32")  # (M,)

        def cond(step, carry_state, accumulated):
            del carry_state, accumulated
            return step < inner_len

        def body(step, carry_state, accumulated):
            # DECISION plan-2026-09-09T042752-6d66ac56/D-014: this sequential
            # `while_loop` recurrence is the SHIPPED formulation and it was
            # chosen by measurement. Do NOT "vectorize" it into the closed-form
            # cumulative-product reassociation
            #     A_t = cumprod(1 - p) ; h_t = A_t * cumsum(p_s x_s / A_s)
            # which looks like a free O(1)-depth speedup and is instead
            # STRUCTURALLY unusable: `A_t = (1 - p)^t` at the clamp bound
            # `p = 1 - 1e-4` is `1e-4t`, which underflows float32 at `t ~ 10`
            # and float64 at `t ~ 77`; `1 / A_s` then overflows and `0 * inf`
            # gives `nan`. MEASURED (D-010(c)): the closed form returns `nan` at
            # `M >= 64` in the `p = 1 - 1e-4` regime -- the regime `dc.py:256`'s
            # own clamp deliberately produces -- in float64 as well as float32,
            # while this loop stays within 4.9e-07 of a float64 reference across
            # all 12 regime x length cells tested. Do NOT reach for
            # `mamba_chunk_scan_combined` either: that is the reference's own
            # SPEED dependency on a CUDA kernel, algebraically the same
            # recurrence (G2), and it does not exist here.
            #
            # And do NOT write the per-step result back with
            # `keras.ops.slice_update`, which is the obvious way to fill the
            # accumulator and is what this loop was first written with. On the
            # TensorFlow backend it lowers to `XlaDynamicUpdateSlice`, which has
            # **no registered gradient**: the forward pass is correct and
            # `tape.gradient(...)` then raises `LookupError: gradient registry
            # has no entry for: XlaDynamicUpdateSlice`, so the whole H-Net stack
            # below this layer would be untrainable while every value test still
            # passed (MEASURED here, 2026-09-09). `keras.ops.scan` is not the
            # replacement either: it differentiates eagerly and traces forward
            # under XLA, but its backward pass raises `XLA compilation requires a
            # fixed tensor list size` because the TensorArray it stacks into has
            # no maximum. The one-hot masked ADD below is plain arithmetic, so it
            # differentiates everywhere, and `maximum_iterations` (set to the
            # exact trip count, so it changes nothing semantically) is what lets
            # the XLA backward pass size its tensor list.
            # Guards: test_dechunk_layer.py::TestGuardOneOracleParity,
            # ::TestGuardThreeUnderflowAtTheClamp and ::TestGradientFlow.
            # Rationale: decisions.md D-014.
            p_step = keras.ops.expand_dims(p[:, step], axis=-1)  # (B, 1)
            carry_state = (
                p_step * inner_hidden_states[:, step, :]
                + (1.0 - p_step) * carry_state
            )
            write_mask = keras.ops.cast(
                keras.ops.equal(inner_positions, step), carry_state.dtype
            )  # (M,) one-hot on the current step
            accumulated = accumulated + keras.ops.expand_dims(
                carry_state, axis=1
            ) * write_mask[None, :, None]
            return step + 1, carry_state, accumulated

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
        :param boundary_prob: ``(B, L, 2)`` routing distribution or ``(B, L)``
            holding ``p`` directly; ``[..., -1]`` is taken either way, matching
            ``dc.py:256``. Required; it defaults to ``None`` only so Keras can
            introspect the signature.
        :type boundary_prob: keras.KerasTensor or None
        :param boundary_mask: ``(B, L)`` boolean/int mask, ``True`` = this
            position starts a chunk. Required.
        :type boundary_mask: keras.KerasTensor or None
        :param mask: Optional ``(B, L)`` validity mask, ANDed into
            ``boundary_mask`` exactly as
            :class:`~dl_techniques.layers.dynamic_chunking.chunk_layer.ChunkLayer`
            does, so both layers partition on the same predicate.
        :type mask: keras.KerasTensor or None
        :param training: Unused -- nothing here is mode-dependent. Accepted so
            the layer composes with parents that pass it through.
        :type training: bool or None
        :return: ``(B, L, D)`` full-resolution hidden states.
        :rtype: keras.KerasTensor

        :raises ValueError: If ``boundary_prob`` or ``boundary_mask`` is missing.
        """
        del training  # nothing is mode-dependent

        if boundary_prob is None or boundary_mask is None:
            raise ValueError(
                "DeChunkLayer requires boundary_prob and boundary_mask; call it "
                "as layer(inner_hidden_states, boundary_prob=..., "
                "boundary_mask=...)"
            )

        compute_dtype = inner_hidden_states.dtype

        # dc.py:256 -- the clamp is on the FULL-resolution probabilities, before
        # the gather. `[..., -1]` reads the boundary channel of a (B, L, 2)
        # distribution and is the identity for a (B, L) input.
        if len(keras.ops.shape(boundary_prob)) == 3:
            p_full = boundary_prob[..., -1]
        else:
            p_full = boundary_prob
        p_full = keras.ops.clip(
            keras.ops.cast(p_full, compute_dtype), self.clamp_min, self.clamp_max
        )  # (B, L)

        keep = keras.ops.cast(boundary_mask, "bool")
        if mask is not None:
            keep = keras.ops.logical_and(keep, keras.ops.cast(mask, "bool"))

        # dc.py:265-269 -- ChunkLayer's stable partition, recomputed here on the
        # same predicate so inner column j is paired with the probability of the
        # position ChunkLayer placed there. Built from an `arange` broadcast and
        # never from `keras.ops.tril`/`triu`, which raise `TypeError: ('pred must
        # not be a Python bool', True)` under plain `tf.function` as well as
        # under XLA (D-010(b)).
        seq_len = keras.ops.shape(keep)[1]
        inner_len = dim(inner_hidden_states, 1)  # M, statically when known
        positions = keras.ops.expand_dims(
            keras.ops.arange(seq_len, dtype="int32"), axis=0
        )  # (1, L)
        token_idx = positions + keras.ops.cast(
            keras.ops.logical_not(keep), "int32"
        ) * keras.ops.cast(seq_len, "int32")
        seq_sorted_indices = keras.ops.argsort(token_idx, axis=1)  # (B, L)

        # dc.py:271-273 -- take the permutation to the INNER width M, through
        # ChunkLayer's own `pad_permutation_to_width`, so the two layers treat
        # `M > L`, `M == L` and `M < L` IDENTICALLY.
        #
        # DECISION plan-2026-09-09T042752-6d66ac56/D-029: do NOT go back to
        # `seq_sorted_indices[:, :inner_len]`. That slice yields `min(L, M)`
        # columns, not `M`, while `_ema_scan` below iterates `M` times and reads
        # `p[:, step]` -- so every `L < M` call died with
        # `InvalidArgumentError: slice index <L> of dimension 1 out of bounds`.
        # ChunkLayer already right-padded its own gather for exactly this case;
        # this side did not, and the ONE-SIDEDNESS was the defect. It was not an
        # edge case: `default_max_chunks(2, max_seq_len=2048)` is `(1024,)`, so
        # `create_hnet("hnet_1stage_L")` rejected every input shorter than 1024
        # bytes at its own constructor defaults, and a 19-byte generation prompt
        # is what found it. Guards:
        # test_dechunk_layer.py::TestGuardFiveWidthSymmetry and
        # test_model.py::TestShortSequencesAtTheConstructorDefaults.
        # Rationale: decisions.md D-029.
        gather_idx = pad_permutation_to_width(
            seq_sorted_indices, inner_len
        )  # (B, M)
        p = batched_gather(p_full, gather_idx)  # (B, M)

        # dc.py:333 -- the kernel-free recurrence; see `_ema_scan`'s D-014 anchor.
        out = self._ema_scan(inner_hidden_states, p)  # (B, M, D)

        # dc.py:302-308 -- every non-boundary position repeats the most recent
        # chunk value. The clip is this port's divergence: upstream `M` is the
        # batch's own maximum boundary count so the index is always in range,
        # while here `M` is the FIXED `max_chunks` cap (D-007) and a row with
        # more boundaries than `M` would run past the inner sequence. `-1` is
        # clipped away too: `dc.py:95-96` forces position 0 to be a boundary, so
        # a `-1` index means the caller bypassed the router, and a defined value
        # beats NumPy's wrap-around or TensorFlow's undefined out-of-range read.
        plug_back_idx = (
            keras.ops.cumsum(keras.ops.cast(keep, "int32"), axis=1) - 1
        )  # (B, L)
        plug_back_idx = keras.ops.minimum(
            keras.ops.maximum(plug_back_idx, 0),
            keras.ops.cast(inner_len - 1, "int32"),
        )

        # A per-row gather, through the SAME helper ChunkLayer uses -- see the
        # D-029 anchor in indexing.batched_gather for why `take_along_axis` is
        # unusable here under XLA. NEVER numpy fancy indexing either --
        # `t[b_idx, i_idx]` raises eagerly on a TF tensor, so a layer using it is
        # dead on every forward pass.
        return batched_gather(out, plug_back_idx)  # (B, L, D)

    def compute_output_shape(
        self,
        inner_hidden_states_shape: Tuple[Optional[int], ...],
        boundary_prob_shape: Optional[Tuple[Optional[int], ...]] = None,
        boundary_mask_shape: Optional[Tuple[Optional[int], ...]] = None,
        mask_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Optional[int], ...]:
        """Output shape ``(B, L, D)``.

        ``L`` is the FULL-resolution length and it is carried by
        ``boundary_mask`` / ``boundary_prob``, not by the inner sequence, so it
        is ``None`` when neither shape is supplied.

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
