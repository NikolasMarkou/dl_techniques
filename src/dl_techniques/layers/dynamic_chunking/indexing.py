"""Shared index arithmetic for H-Net's chunk/dechunk pair.

:class:`~dl_techniques.layers.dynamic_chunking.chunk_layer.ChunkLayer` and
:class:`~dl_techniques.layers.dynamic_chunking.dechunk_layer.DeChunkLayer` both
gather through the *same* stable partition of the same boundary predicate -- one
to build the inner sequence, the other to pair each inner column with the
probability of the position that produced it. Anything either of them does to
that permutation, the other must do identically, or inner column ``j`` stops
meaning the same thing on the two sides.

That symmetry used to be a convention held by two hand-copied code blocks, and
the convention broke: ``ChunkLayer`` right-padded the permutation when the inner
width ``M`` exceeded the outer length ``L`` and ``DeChunkLayer`` did not, so every
H-Net raised on any sequence shorter than its own ``max_chunks[0]`` -- including
at the constructor default, where ``default_max_chunks(2, max_seq_len=2048)``
yields ``(1024,)`` and any prompt below 1024 bytes was rejected. The two
functions here are the repair: the padding rule and the gather live in ONE place
and both layers call them, so the two sides cannot drift apart again.

Both functions are pure, stateless, weight-free and written in ``keras.ops``
only.
"""

from typing import Union

import keras

# ---------------------------------------------------------------------

__all__ = [
    "batched_gather",
    "dim",
    "pad_permutation_to_width",
]

# ---------------------------------------------------------------------


def dim(x: keras.KerasTensor, axis: int) -> Union[int, keras.KerasTensor]:
    """Return an axis length as a Python ``int`` when it is statically known.

    Preferring the static value matters under XLA: a Python ``int`` becomes a
    compile-time constant, while ``keras.ops.shape(x)[axis]`` becomes a tensor
    that XLA may refuse as an operand to a shape-consuming op.

    :param x: Any tensor.
    :type x: keras.KerasTensor
    :param axis: Axis index; negative indices are allowed.
    :type axis: int
    :return: ``int`` when the axis is static, otherwise a scalar tensor.
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
