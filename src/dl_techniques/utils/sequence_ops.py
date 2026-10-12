"""Shared sequence operations for dynamic chunking and patching.

Provides the canonical implementation of permutation padding and batched gathering
used by H-Net's chunk/dechunk layers and BLT's dynamic patcher. Having ONE
definition prevents the two sides from drifting apart (D-029).

Also defines the :class:`DynamicSegmenter` protocol capturing the shared
pattern: score → boundary_mask → gather → process → scatter.
Implemented by:
  - BLT: DynamicPatcher (entropy) + LocalEncoder/Decoder
  - H-Net: RoutingModule (cosine) + ChunkLayer/DeChunkLayer
"""

from typing import Union, Protocol, runtime_checkable

import keras

# ---------------------------------------------------------------------


@runtime_checkable
class DynamicSegmenter(Protocol):
    """Dynamic sequence segmenter protocol.

    Captures the shared pattern: score → boundary_mask → gather → process → scatter.
    Implemented by:
      - BLT: DynamicPatcher (entropy) + LocalEncoder/Decoder
      - H-Net: RoutingModule (cosine) + ChunkLayer/DeChunkLayer

    This is a structural protocol (no explicit declaration needed).
    """

    def segment(self, scores: keras.KerasTensor) -> keras.KerasTensor:
        """Convert scores to a boundary mask or segment ids.

        :param scores: Per-position scores, shape ``(B, L)`` (entropy in nats,
            boundary probability, cosine similarity, etc.).
        :return: Boundary mask ``(B, L)`` bool or segment ids ``(B, L)`` int.
        """
        ...

    def gather(self, hidden: keras.KerasTensor, segment_ids: keras.KerasTensor) -> keras.KerasTensor:
        """Gather hidden states at segment boundaries into a shorter sequence.

        :param hidden: Full-resolution hidden states, shape ``(B, L, D)``.
        :param segment_ids: Segment ids or boundary mask, shape ``(B, L)``.
        :return: Compressed sequence, shape ``(B, max_segments, D)``.
        """
        ...

    def scatter(self, inner: keras.KerasTensor, segment_ids: keras.KerasTensor) -> keras.KerasTensor:
        """Scatter inner sequence back to full resolution.

        :param inner: Compressed sequence, shape ``(B, M, D)``.
        :param segment_ids: Segment ids or boundary mask, shape ``(B, L)``.
        :return: Full-resolution sequence, shape ``(B, L, D)``.
        """
        ...


def cap_boundaries_by_position(
    boundary_mask: keras.KerasTensor, max_segments: int
) -> keras.KerasTensor:
    """Keep first ``max_segments`` boundaries in position order; truncate tail.

    This is the POSITION-ORDER CAPPING rule (D-007/D-012): when a sequence has
    more boundaries than ``max_segments``, we keep the FIRST boundaries in
    position order and drop the tail. We do NOT re-rank by score magnitude.

    Both BLT's DynamicPatcher and H-Net's ChunkLayer implement this identical rule.
    This helper centralises it so neither side can drift.

    :param boundary_mask: Boolean mask ``(B, L)`` where ``True`` = boundary.
    :param max_segments: Maximum number of segments/boundaries to keep.
    :return: Capped boundary mask ``(B, L)`` bool with at most ``max_segments``
        True values per row, preserving position order.
    """
    # Count boundaries per row (cumulative sum gives position-order index)
    # Each True gets a 1-based index; we keep those <= max_segments
    boundary_idx = keras.ops.cumsum(keras.ops.cast(boundary_mask, "int32"), axis=1)
    # Keep only boundaries whose index <= max_segments
    capped = boundary_idx <= max_segments
    return keras.ops.logical_and(boundary_mask, capped)


def dim(x: keras.KerasTensor, axis: int) -> Union[int, keras.KerasTensor]:
    """Return an axis length as a Python ``int`` when it is statically known.

    On the TensorFlow backend this is exactly ``keras.ops.shape(x)[axis]`` --
    MEASURED, see the D-032 anchor above: ``keras.ops.shape`` already yields a
    Python ``int`` per statically known axis, eagerly and under tracing alike.
    The explicit static branch is therefore a **statement of intent, not an
    optimisation and not an XLA requirement**; it exists so the two chunking
    layers ask for a length in one named way.

    What the callers rely on, and what is guarded, is the RESULT: a statically
    known axis is a Python ``int`` (so ``pad_permutation_to_width``'s ``width``
    can be a compile-time constant and the traced graph keeps a readable inner
    length), and an unknown one is whatever the backend hands back for it.

    :param x: Any tensor.
    :type x: keras.KerasTensor
    :param axis: Axis index; negative indices are allowed.
    :type axis: int
    :return: ``int`` when the axis is static; otherwise whatever the backend
        reports for that axis -- a scalar tensor under tracing, and ``None``
        for a purely symbolic ``KerasTensor``.
    :rtype: int or keras.KerasTensor
    """
    static = x.shape[axis]
    if static is not None:
        return int(static)
    return keras.ops.shape(x)[axis]


def pad_permutation_to_width(
    indices: keras.KerasTensor, width: Union[int, keras.KerasTensor]
) -> keras.KerasTensor:
    """Right-pad a ``(B, L)`` permutation with index ``0`` and cut it to ``width``.

    This is the ONE definition of what happens when the inner width ``M`` and the
    outer length ``L`` disagree, and it is deliberately one-sided in neither
    direction:

    * ``L > width`` -- the trailing columns are dropped, keeping the FIRST
      ``width`` boundaries in position order (D-007's truncation rule).
    * ``L < width`` -- the missing columns are filled with index ``0``, so the
      gather stays in range and the padded columns carry position 0's value.
      Those columns are never valid: ``ChunkLayer``'s ``inner_mask`` is
      ``arange(width) < num_tokens`` and ``num_tokens <= L``, and
      ``DeChunkLayer``'s ``plug_back_idx`` is likewise bounded by
      ``num_tokens - 1``, so nothing ever reads them back.
    * ``L == width`` -- an exact copy.

    :param indices: ``(B, L)`` integer permutation.
    :type indices: keras.KerasTensor
    :param width: Target width ``M``. A Python ``int`` whenever the caller knows
        it statically, which every shipped caller does.
    :type width: int or keras.KerasTensor
    :return: ``(B, width)`` integer indices.
    :rtype: keras.KerasTensor
    """
    filler = keras.ops.repeat(
        keras.ops.zeros_like(indices[:, :1]), width, axis=1
    )  # (B, width)
    return keras.ops.concatenate([indices, filler], axis=1)[:, :width]


def batched_gather(
    params: keras.KerasTensor, indices: keras.KerasTensor
) -> keras.KerasTensor:
    """Gather per row along axis 1, without ``take_along_axis``'s dynamic broadcast.

    ``params[b, indices[b, w], ...]`` for a rank-2 ``(B, L)`` or rank-3
    ``(B, L, D)`` ``params``.

    .. code-block:: text

        # DECISION plan-2026-09-09T042752-6d66ac56/D-029
        # Do NOT "simplify" this back to
        #     keras.ops.take_along_axis(params, indices[..., None], axis=1)
        # which is what both layers shipped with and which is one line shorter.
        # Keras' TensorFlow backend implements that op with
        # `tf.broadcast_dynamic_shape(tf.shape(x), tf.shape(indices))`
        # (`keras/src/backend/tensorflow/numpy.py`), i.e. a `BroadcastArgs` node
        # whose operand is a runtime shape tensor. Under XLA that is rejected:
        # `INVALID_ARGUMENT: Input 0 to node .../BroadcastArgs with op
        # BroadcastArgs must be a compile-time constant.` MEASURED on GPU 0 with
        # the default `jit_compile="auto"`: `predict`, `evaluate` AND `fit` all
        # raise it on a plain NumPy array, so an ordinary
        # `model.predict(x)` was dead on arrival.
        # Do NOT "fix" it by pinning `jit_compile=False` on the model or in the
        # tests either -- that hides the defect from every caller who does not,
        # and a compile-option opt-out in this repo must override BOTH
        # `compile()` and `compile_from_config` or it silently reverts on
        # reload. Reshaping the gather so no broadcast is needed is the real fix.
        # Guards: test_chunk_layer.py::TestGuardFourXlaGather,
        # test_dechunk_layer.py::TestGuardFourXlaGather and
        # test_model.py::TestDefaultCompileEntryPoints.
        # Rationale: decisions.md D-029.

    The row offsets are built by a ``cumsum`` over a column of ones rather than
    by ``arange(batch_size)``, because ``batch_size`` is exactly the dimension
    that is not statically known. The result is value-identical to
    ``take_along_axis``: this is a pure gather and performs no arithmetic on
    ``params``.

    :param params: ``(B, L)`` or ``(B, L, D)``.
    :type params: keras.KerasTensor
    :param indices: ``(B, W)`` integer indices into axis 1 of ``params``. Must
        already be in range; nothing here clips.
    :type indices: keras.KerasTensor
    :return: ``(B, W)`` or ``(B, W, D)``, matching ``params``' rank.
    :rtype: keras.KerasTensor

    :raises ValueError: If ``params`` is not rank 2 or rank 3.
    """
    rank = len(params.shape)
    if rank not in (2, 3):
        raise ValueError(
            f"batched_gather expects a rank-2 or rank-3 params tensor, got rank "
            f"{rank} with shape {tuple(params.shape)}"
        )

    seq_len = dim(params, 1)
    indices = keras.ops.cast(indices, "int32")

    # (B, 1) holding 0, 1, ..., B-1 -- derived from the tensor's CONTENT, so no
    # `arange` over the unknown batch dimension is needed.
    row_ids = (
        keras.ops.cumsum(keras.ops.ones_like(indices[:, :1]), axis=0) - 1
    )  # (B, 1) int32
    flat_indices = indices + row_ids * keras.ops.cast(seq_len, "int32")

    if rank == 2:
        flat = keras.ops.reshape(params, [-1])
        return keras.ops.take(flat, flat_indices, axis=0)  # (B, W)

    flat = keras.ops.reshape(params, [-1, dim(params, -1)])  # (B*L, D)
    return keras.ops.take(flat, flat_indices, axis=0)  # (B, W, D)

# ---------------------------------------------------------------------