"""Independent NumPy oracle for the ``cluster_axis`` value-layout contract.

Shared by ``test_gmm.py`` and ``test_kmeans.py`` (two call sites, hence a module
rather than two copies -- this is the instrument that validates correctness, so a
bug fixed in one copy silently surviving in the other would be the worst possible
place for duplication).

**Why this is independent of the implementation.** ``base.py::_reshape_output``
restores the axis order with a computed *permutation* (a count of preceding
non-feature axes for ``assignments``; an ``argsort`` of the forward perm for
``mixture``). This oracle never computes a permutation. It iterates the
non-feature multi-index one slice at a time, pulls each feature vector out of the
input by direct indexing, and writes each result back by direct indexing. An
off-by-one or an inverted permutation in the implementation therefore cannot be
mirrored here, because there is no permutation here to get wrong.

**Interface contract.**

``build_cluster_axis_oracle(x, cluster_axis, output_mode, n_prototypes,
flat_forward) -> np.ndarray``

:param x: Input array, any rank >= 2.
:param cluster_axis: The axes clustered over; negatives allowed, order-insensitive.
:param output_mode: ``'assignments'`` or ``'mixture'``.
:param n_prototypes: K -- the prototype count (ignored for ``'mixture'``).
:param flat_forward: Callable mapping a ``(N, feature_dims)`` float32 array to an
    ``(N, W)`` array. Supply a *flat twin* layer: same class, same weights,
    default ``cluster_axis=-1``, so it runs the last-axis fast path -- the one
    layout path independently established as correct.
:returns: float64 array in the DECLARED output layout.
:raises ValueError: if ``output_mode`` is not one of the two legal values.
"""

from typing import Callable, Sequence

import numpy as np

__all__ = ["build_cluster_axis_oracle", "flat_twin_forward"]


def build_cluster_axis_oracle(
    x: np.ndarray,
    cluster_axis: Sequence[int],
    output_mode: str,
    n_prototypes: int,
    flat_forward: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Build the expected output layout by direct per-slice indexing."""
    if output_mode not in ("assignments", "mixture"):
        raise ValueError(
            f"output_mode must be 'assignments' or 'mixture', got {output_mode!r}"
        )

    rank = x.ndim
    axes = sorted(a if a >= 0 else rank + a for a in cluster_axis)
    non_feature = [i for i in range(rank) if i not in axes]

    # One feature vector per non-feature multi-index. Slicing with integers at the
    # non-feature axes leaves the cluster axes in their original ascending order,
    # so a row-major flatten reproduces exactly what the forward path collapses.
    nf_shape = tuple(x.shape[i] for i in non_feature)
    keys = list(np.ndindex(*nf_shape))
    rows = []
    for key in keys:
        selector = [slice(None)] * rank
        for axis, value in zip(non_feature, key):
            selector[axis] = value
        rows.append(np.asarray(x[tuple(selector)]).reshape(-1))

    flat_in = np.stack(rows, axis=0).astype("float32")
    flat_out = np.asarray(flat_forward(flat_in), dtype=np.float64)

    if output_mode == "mixture":
        # Target shape is the input shape: each reconstructed feature vector goes
        # straight back into its own cluster-axis block.
        out = np.empty(x.shape, dtype=np.float64)
        block_shape = tuple(x.shape[c] for c in axes)
        for key, row in zip(keys, flat_out):
            selector = [slice(None)] * rank
            for axis, value in zip(non_feature, key):
                selector[axis] = value
            out[tuple(selector)] = row.reshape(block_shape)
        return out

    # assignments: the whole cluster-axis group collapses to a single K axis that
    # keeps the ORIGINAL position of the lowest cluster axis. Expressed as a
    # surviving-label list, so no "how many axes precede it" arithmetic is needed.
    surviving = sorted(non_feature + [axes[0]])
    out = np.empty(
        tuple(n_prototypes if a == axes[0] else x.shape[a] for a in surviving),
        dtype=np.float64,
    )
    k_pos = surviving.index(axes[0])
    for key, row in zip(keys, flat_out):
        selector = [slice(None)] * len(surviving)
        for axis, value in zip(non_feature, key):
            selector[surviving.index(axis)] = value
        selector[k_pos] = slice(None)
        out[tuple(selector)] = row
    return out


def flat_twin_forward(
    layer_cls: type,
    config: dict,
    param_names: Sequence[str],
    source_layer: object,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a ``flat_forward`` callable backed by a weight-cloned 2-D twin.

    The twin is constructed with the default ``cluster_axis=-1`` so it takes the
    last-axis fast path, then has every parameter in ``param_names`` assigned from
    ``source_layer``. Both layers therefore carry byte-identical weights and differ
    only in how they lay their input and output out.

    :raises AttributeError: if ``source_layer`` lacks any name in ``param_names``
        (i.e. it was never built, or the name list is stale).
    """
    from keras import ops

    def _forward(flat_in: np.ndarray) -> np.ndarray:
        twin = layer_cls(**config)
        twin(flat_in)  # build
        for name in param_names:
            getattr(twin, name).assign(ops.convert_to_numpy(getattr(source_layer, name)))
        return np.asarray(ops.convert_to_numpy(twin(flat_in)))

    return _forward
