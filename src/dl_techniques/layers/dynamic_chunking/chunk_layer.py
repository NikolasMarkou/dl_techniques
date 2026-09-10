"""H-Net's chunk layer: the ragged-selection operator, at a FIXED width.

``ChunkLayer`` is the second of H-Net's three dynamic-chunking layers. It reads a
full-resolution sequence ``(B, L, D)`` together with the boolean boundary mask
:class:`~dl_techniques.layers.dynamic_chunking.routing_module.RoutingModule`
produced for it, and hands the deeper stage a SHORTER sequence containing only
the boundary tokens::

    next_hidden_states[b, j] = hidden_states[b, position of the j-th boundary in row b]

The selection is a *stable partition*: boundary positions move to the front in
their original order, non-boundary positions follow in their original order. It
is expressed exactly as the reference expresses it (``dc.py:186-199``)::

    token_idx = arange(L)[None, :] + (~boundary_mask) * L
    seq_sorted_indices = argsort(token_idx, axis=1)

The keys are distinct by construction -- kept positions occupy ``[0, L)`` and
dropped positions ``[L, 2L)``, and within each group they are the distinct
original indices -- so ``argsort`` **stability is never load-bearing** here. That
is stated rather than assumed: plan step 2(a) measured zero duplicate keys in the
worst case over 200 random draws, and the gather equalled a hand-built
position-order partition on 1000 of 1000 random masks (decision D-010(a)).

**The one deliberate divergence from the reference is the output width**, and it
is decision D-007, anchored at its site in :meth:`ChunkLayer.call`. The reference
computes ``next_max_seqlen = max(boundary_mask.sum(-1))`` over the batch; this
port takes a fixed ``max_chunks`` constructor argument.

Only the padded/masked branch of ``ChunkLayer.forward`` is ported. The packed
(``cu_seqlens``) branch is a CUDA kernel-API artifact and is deliberately absent
(D-007); the reference file is vendored for the test suite under
``tests/test_layers/test_dynamic_chunking/_reference/``.

The layer holds **no weights**. It is a pure gather plus a mask construction, so
gradient flows through it into ``hidden_states`` and there is nothing here for an
optimizer to update. The test suite asserts that absence deliberately rather than
skipping the per-weight gradient check.

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

from .indexing import batched_gather, pad_permutation_to_width

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.dynamic_chunking.chunk_layer")
class ChunkLayer(keras.layers.Layer):
    """Select the boundary tokens of a sequence into a fixed-width inner sequence.

    Architecture:

    .. code-block:: text

        hidden_states (B, L, D)   boundary_mask (B, L)      mask (B, L)
               |                        |                        |
               |                        +----- logical_and ------+
               |                        |
               |             keep (B, L) bool
               |                        |
               |        token_idx = arange(L) + (~keep) * L      (dc.py:188-190)
               |                        |
               |             argsort(token_idx, axis=1)          (dc.py:191)
               |                        |
               |        [:, :max_chunks]  <-- D-007 FIXED width
               |                        |
               +---- batched_gather ----+                        (dc.py:193-199)
                              |
                next_hidden_states (B, max_chunks, D)

        num_tokens = sum(keep, axis=-1)                          (dc.py:183)
        inner_mask = arange(max_chunks)[None, :] < num_tokens[:, None]
                                                                 (dc.py:201-204)

    Two consequences of taking ``dc.py:201-204`` verbatim under a fixed width,
    both intended and both guarded:

    - Columns beyond a row's own boundary count hold whatever the stable
      partition placed there -- NON-boundary hidden states, in position order.
      They are garbage marked invalid by ``inner_mask``, exactly as upstream.
    - A row with MORE boundaries than ``max_chunks`` gets an all-``True``
      ``inner_mask``, which is correct: every column is a real chunk, and the
      row's tail boundaries are simply lost. That loss is D-007's named new
      failure mode.

    ``max_chunks`` may exceed ``L``. The permutation is right-padded with index
    ``0`` before slicing, so the surplus columns hold row-0 hidden states and are
    marked invalid (``num_tokens <= L < max_chunks``). The padding is built by
    ``repeat`` on a static count rather than by a data-dependent ``pad``, so the
    whole path stays traceable. Both the padding and the truncation live in
    :func:`~dl_techniques.layers.dynamic_chunking.indexing.pad_permutation_to_width`,
    which :class:`~dl_techniques.layers.dynamic_chunking.dechunk_layer.DeChunkLayer`
    calls with the same width -- the two layers used to hold this rule in two
    hand-copied blocks and drifted apart, which is exactly the defect the shared
    helper exists to prevent.

    :param max_chunks: Fixed output width ``C``. Constructor argument, never
        derived from the data -- see the D-007 anchor in :meth:`call`.
    :type max_chunks: int
    :param kwargs: Additional :class:`keras.layers.Layer` arguments.

    :raises ValueError: If ``max_chunks`` is not a positive integer.

    Example:
        >>> import keras, numpy as np
        >>> layer = ChunkLayer(max_chunks=4)
        >>> h = keras.ops.convert_to_tensor(np.random.randn(2, 16, 8).astype("float32"))
        >>> boundary = keras.ops.convert_to_tensor(
        ...     np.array([[True] + [False] * 15] * 2)
        ... )
        >>> inner, inner_mask = layer(h, boundary_mask=boundary)
        >>> inner.shape, inner_mask.shape
        ((2, 4, 8), (2, 4))
    """

    def __init__(self, max_chunks: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(max_chunks, int) or isinstance(max_chunks, bool):
            raise ValueError(
                f"max_chunks must be an int, got {type(max_chunks).__name__}"
            )
        if max_chunks <= 0:
            raise ValueError(f"max_chunks must be positive, got {max_chunks}")

        self.max_chunks = max_chunks

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Validate the hidden-state rank. This layer creates no weights.

        Keras forwards the shape of the FIRST call argument here because
        ``build`` takes a single parameter, so ``input_shape`` is
        ``hidden_states``' shape, never ``boundary_mask``'s.

        :param input_shape: Shape of ``hidden_states``, ``(B, L, D)``.
        :type input_shape: tuple

        :raises ValueError: If the input is not rank 3.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"ChunkLayer expects rank-3 (batch, seq_len, d_model) hidden "
                f"states, got shape {input_shape}"
            )
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        boundary_mask: Optional[keras.KerasTensor] = None,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Gather the boundary tokens into a fixed-width inner sequence.

        :param hidden_states: ``(B, L, D)`` full-resolution hidden states.
        :type hidden_states: keras.KerasTensor
        :param boundary_mask: ``(B, L)`` boolean/int mask, ``True`` = this
            position starts a chunk. Required; it defaults to ``None`` only so
            Keras can introspect the signature.
        :type boundary_mask: keras.KerasTensor or None
        :param mask: Optional ``(B, L)`` validity mask. ANDed into
            ``boundary_mask``, so a padded position can never be selected even if
            an upstream router left it set. Idempotent when the router already
            applied it, which is the normal case.
        :type mask: keras.KerasTensor or None
        :param training: Unused -- nothing here is mode-dependent. Accepted so
            the layer composes with parents that pass it through.
        :type training: bool or None
        :return: ``(next_hidden_states (B, max_chunks, D), inner_mask
            (B, max_chunks) bool)``.
        :rtype: tuple

        :raises ValueError: If ``boundary_mask`` is not supplied.
        """
        del training  # nothing is mode-dependent

        if boundary_mask is None:
            raise ValueError(
                "ChunkLayer requires boundary_mask; call it as "
                "layer(hidden_states, boundary_mask=...)"
            )

        # DECISION plan-2026-09-09T042752-6d66ac56/D-013: the padding mask is AND-ed
        # into `boundary_mask` here. This is a deliberate DIVERGENCE and must not be
        # "corrected" to match `dc.py`, whose padded branch reads `boundary_mask`
        # alone because its router has already masked it. The AND is defensive: a
        # padded position can then never be selected as a chunk boundary even if an
        # upstream router left the bit set, and it is idempotent whenever the router
        # did its job, so it costs nothing in the normal case. Guarded by
        # `test_chunk_layer.py::test_the_padding_mask_removes_a_boundary_the_router_left_set`.
        # See decisions.md D-013.
        keep = keras.ops.cast(boundary_mask, "bool")
        if mask is not None:
            keep = keras.ops.logical_and(keep, keras.ops.cast(mask, "bool"))

        # dc.py:183 -- how many real chunks this row actually has.
        num_tokens = keras.ops.sum(keras.ops.cast(keep, "int32"), axis=-1)  # (B,)

        # dc.py:188-191 -- the stable partition. Kept positions take keys
        # 0..L-1 and dropped positions L..2L-1, so every key is DISTINCT by
        # construction and argsort stability is not load-bearing (measured:
        # zero duplicate keys worst case over 200 draws, and the gather equalled
        # a hand-built position-order partition 1000/1000 -- D-010(a)). Masks are
        # built from `arange` broadcasts and never from `keras.ops.tril`/`triu`,
        # which raise `TypeError: ('pred must not be a Python bool', True)` under
        # plain `tf.function` as well as under XLA (D-010(b)).
        seq_len = keras.ops.shape(hidden_states)[1]
        positions = keras.ops.expand_dims(
            keras.ops.arange(seq_len, dtype="int32"), axis=0
        )  # (1, L)
        token_idx = positions + keras.ops.cast(
            keras.ops.logical_not(keep), "int32"
        ) * keras.ops.cast(seq_len, "int32")
        seq_sorted_indices = keras.ops.argsort(token_idx, axis=1)  # (B, L)

        # DECISION plan-2026-09-09T042752-6d66ac56/D-007: the output width is the
        # CONSTRUCTOR argument `self.max_chunks`. Do NOT "restore fidelity" to
        # `dc.py:184`'s `next_max_seqlen = int(boundary_mask.sum(-1).max())`, and
        # do NOT make the width any other function of the batch:
        #   * a data-dependent width cannot be built symbolically and does not
        #     trace under `tf.function`/`jit_compile=True`;
        #   * `max(...)` over the batch axis makes row i's padding a function of
        #     row j's bytes -- a batch-independence violation no shape test can
        #     see (Problem Statement invariant 2).
        # And the slice below keeps the FIRST `max_chunks` boundaries in POSITION
        # order. Do NOT replace it with a top-k over boundary probability: a late
        # high-scoring boundary displacing an earlier one makes an earlier token's
        # chunk assignment depend on a future token. That defect is MEASURED in
        # this repo (`layers/blt/blt_blocks.py:410-420`, 4.85e-01 movement in
        # pre-perturbation logits) and guarded there by
        # `test_future_byte_does_not_change_the_past`.
        # Guards: test_chunk_layer.py::TestGuardTwoPositionOrderTruncation and
        # ::TestGuardThreeBatchIndependence. Rationale: decisions.md D-007.
        # Truncation and right-padding both live in `pad_permutation_to_width`,
        # which DeChunkLayer calls with the SAME width, so the two sides cannot
        # disagree about what inner column j means. See indexing.py's docstring:
        # the asymmetry (this layer padded, DeChunkLayer did not) WAS the bug.
        gather_idx = pad_permutation_to_width(
            seq_sorted_indices, self.max_chunks
        )  # (B, C)

        # dc.py:193-199 -- a per-row gather. NOT `keras.ops.take_along_axis`;
        # see the D-029 anchor in indexing.batched_gather for why that op is
        # unusable here under XLA.
        next_hidden_states = batched_gather(hidden_states, gather_idx)  # (B, C, D)

        # dc.py:201-204 -- at a fixed width, a row with more boundaries than
        # `max_chunks` is all-valid and loses its tail chunks (D-007).
        inner_positions = keras.ops.expand_dims(
            keras.ops.arange(self.max_chunks, dtype="int32"), axis=0
        )  # (1, C)
        inner_mask = keras.ops.less(
            inner_positions, keras.ops.expand_dims(num_tokens, axis=-1)
        )  # (B, C) bool

        return next_hidden_states, inner_mask

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Shapes of the two outputs, from stored config alone.

        The output width is ``max_chunks`` and nothing about it depends on the
        data -- that is the whole point of D-007, and this method is where it
        becomes visible to the functional API.

        :param input_shape: ``(B, L, D)``.
        :type input_shape: tuple
        :return: ``((B, max_chunks, D), (B, max_chunks))``.
        :rtype: tuple
        """
        batch_size, d_model = input_shape[0], input_shape[-1]
        return (
            (batch_size, self.max_chunks, d_model),
            (batch_size, self.max_chunks),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Serializable configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({"max_chunks": self.max_chunks})
        return config
