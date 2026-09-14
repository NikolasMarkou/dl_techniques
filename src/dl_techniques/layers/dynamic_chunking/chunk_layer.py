"""Chunk layer for H-Net dynamic chunking.

``ChunkLayer`` reads full-resolution hidden states ``(B, L, D)`` and a boolean
boundary mask, and returns the boundary tokens as a shorter sequence plus a
validity mask. The selection is a stable partition: boundary positions move to
the front in their original order and the rest follow in their original order.
It is built from an argsort over ``arange(L) + (~keep) * L`` rather than a
ragged gather, so the shapes stay static and the path traces under
``tf.function`` and XLA.

The output width is the constructor argument ``max_chunks``, not the per-batch
maximum boundary count that the reference uses, so a row with more boundaries
than ``max_chunks`` loses its tail boundaries. Only the padded branch of the
reference implementation is ported; the packed ``cu_seqlens`` branch is absent.
The layer holds no weights.

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

    The layer combines the optional padding mask into ``boundary_mask`` with a
    logical and, builds a permutation that moves the kept positions to the front
    in position order, pads or truncates that permutation to ``max_chunks``
    columns, and gathers the hidden states through it. It also returns a
    validity mask built from each row's boundary count. The layer has no
    weights, and gradients flow through the gather into ``hidden_states``.

    Architecture:

    .. code-block:: text

         boundary_mask [B, L]        mask [B, L] (optional)
                      │                           │
                      └─────────── and ───────────┘
                                  │
                          keep [B, L] bool
                                  │
                      ┌───────────┴───────────┐
                      ▼                       ▼
             ┌──────────────────┐    ┌──────────────────┐
             │ argsort(arange   │    │ sum(keep, -1)    │
             │  + ~keep * L)    │    │                  │
             └────────┬─────────┘    └────────┬─────────┘
                   [B, L]                    [B]
                      ▼                       ▼
             ┌──────────────────┐    ┌──────────────────┐
             │ pad/truncate     │    │ arange(C) <      │
             │  to width C      │    │  num_tokens      │
             └────────┬─────────┘    └────────┬─────────┘
                   [B, C]                     ▼
                      ▼               inner_mask [B, C]
             ┌──────────────────┐
             │ batched_gather   │ ◄── hidden_states [B, L, D]
             └────────┬─────────┘
                      ▼
             next_hidden_states [B, C, D]

    C is ``max_chunks`` and is fixed at construction.

    Fixed-width behaviour:

    .. code-block:: text

        row has      inner columns 0..C-1          inner_mask
        ───────────  ────────────────────────────  ──────────
        < C          boundaries, then non-         True up to
                     boundary tokens in            the boundary
                     position order                count
        = C          boundaries only               all True
        > C          first C boundaries in         all True
                     position order, tail lost

    ``max_chunks`` may exceed ``L``. The permutation is right-padded with index
    ``0`` before slicing, so the surplus columns gather each row's first token
    and are marked invalid, since ``num_tokens <= L < max_chunks``. The padding
    uses ``repeat`` on a static count rather than a data-dependent ``pad``, so
    the path stays traceable. Padding and truncation both live in
    :func:`~dl_techniques.layers.dynamic_chunking.indexing.pad_permutation_to_width`,
    which :class:`~dl_techniques.layers.dynamic_chunking.dechunk_layer.DeChunkLayer`
    calls with the same width, so both sides agree on what inner column j means.

    :param max_chunks: Fixed output width ``C``. Taken from the constructor and
        never derived from the data.
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
        """Validate the hidden-state rank.

        This layer creates no weights. Keras passes the shape of the first call
        argument, so ``input_shape`` is ``hidden_states``' shape and never
        ``boundary_mask``'s.

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
        :param boundary_mask: ``(B, L)`` boolean or integer mask, where ``True``
            means the position starts a chunk. Required; it defaults to ``None``
            only so Keras can introspect the signature.
        :type boundary_mask: keras.KerasTensor or None
        :param mask: Optional ``(B, L)`` validity mask, combined into
            ``boundary_mask`` with a logical and so a padded position can never
            be selected. Has no effect when the router already applied it.
        :type mask: keras.KerasTensor or None
        :param training: Unused. Accepted so the layer composes with parents
            that pass it through.
        :type training: bool or None
        :return: ``(next_hidden_states (B, max_chunks, D), inner_mask
            (B, max_chunks) bool)``.
        :rtype: tuple

        :raises ValueError: If ``boundary_mask`` is not supplied.
        """
        del training

        if boundary_mask is None:
            raise ValueError(
                "ChunkLayer requires boundary_mask; call it as "
                "layer(hidden_states, boundary_mask=...)"
            )

        # DECISION D-013: mask is combined into boundary_mask here, where dc.py
        # reads boundary_mask alone. Keep the and. See decisions.md.
        keep = keras.ops.cast(boundary_mask, "bool")
        if mask is not None:
            keep = keras.ops.logical_and(keep, keras.ops.cast(mask, "bool"))

        # Number of real chunks in each row, shape (B,) (dc.py:183).
        num_tokens = keras.ops.sum(keras.ops.cast(keep, "int32"), axis=-1)

        # Kept keys occupy [0, L) and dropped keys [L, 2L), so every key is
        # distinct and argsort stability is not required (dc.py:188-191).
        seq_len = keras.ops.shape(hidden_states)[1]
        positions = keras.ops.expand_dims(
            keras.ops.arange(seq_len, dtype="int32"), axis=0
        )
        token_idx = positions + keras.ops.cast(
            keras.ops.logical_not(keep), "int32"
        ) * keras.ops.cast(seq_len, "int32")
        # DECISION D-010(b): masks come from arange broadcasts; keras.ops.tril
        # and triu raise TypeError under tf.function. See decisions.md.
        seq_sorted_indices = keras.ops.argsort(token_idx, axis=1)

        # DECISION D-007: width is max_chunks, not a per-batch max; the slice
        # keeps the first C boundaries in position order, not top-k
        # (decisions.md).
        gather_idx = pad_permutation_to_width(
            seq_sorted_indices, self.max_chunks
        )

        # Per-row gather, not take_along_axis, which is unusable here under XLA
        # (dc.py:193-199; D-029 in indexing.batched_gather).
        next_hidden_states = batched_gather(hidden_states, gather_idx)

        # A row with more than max_chunks boundaries gets an all-True mask and
        # loses its tail chunks (dc.py:201-204).
        inner_positions = keras.ops.expand_dims(
            keras.ops.arange(self.max_chunks, dtype="int32"), axis=0
        )
        inner_mask = keras.ops.less(
            inner_positions, keras.ops.expand_dims(num_tokens, axis=-1)
        )

        return next_hidden_states, inner_mask

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Return the shapes of the two outputs.

        Both widths come from ``max_chunks``, so neither depends on the data.

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

# ---------------------------------------------------------------------

