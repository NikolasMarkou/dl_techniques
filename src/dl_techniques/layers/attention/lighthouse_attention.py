"""
Sparse causal attention over a coarse-to-fine pyramid of pooled representations.

A Keras 3 / TF 2.18 implementation of Lighthouse Attention. Every position
reads its whole causal prefix without paying the quadratic cost of reading all
of it at full resolution. Plain self-attention spends the same precision on
distant and on local context. Lighthouse spends precision where the content
warrants it: salient history at token granularity, the rest through
progressively coarser summaries.

Architecture:
    The single ``N x N`` score matrix is replaced by a pyramid of pooled
    representations plus a sparse, content-selected read over that pyramid.
    Five stages:

    -   **Pyramid construction.** Q, K and V are projected once and mean-pooled
        into ``num_levels`` levels with branching factor ``pooling_factor``.
        Level 0 is the base sequence at full resolution. Level ``l`` holds
        ``N / p^l`` entries, each summarising a window of ``p^l`` consecutive
        tokens. Concatenating the levels gives ``S_pyr = sum_l N / p^l``
        candidate entries per sequence.
    -   **Scoring.** Every candidate gets a per-head L2-norm score. The norms
        ``||Q||`` and ``||K||`` are taken on the RAW level-0 projections and
        max-pooled up the pyramid. They are not recomputed on the pooled
        tensors, so a window inherits the salience of its strongest token. The
        two terms are combined per entry by a max, giving the joint QK / KQ
        score, then reduced over heads into a single ``(B, S_pyr)`` map.
    -   **Selection.** Two index sets are retained. A MANDATORY set, fixed at
        build time, holds every coarsest-level entry plus the leading level-0
        entries. Together those guarantee at least one contributor at every base
        position. A DISCRETIONARY set of ``top_k`` entries is then chosen by
        score from the remaining candidates, with the budget PARTITIONED BY
        CAUSAL BLOCK so entries never compete across time (D-023). The union is
        sorted by causal position, which restores temporal order.
    -   **Sub-attention.** Q, K and V are gathered at the selected indices, and
        one causal scaled dot-product attention runs over the resulting
        sub-sequence. The indices are sorted by causal position, so an ordinary
        lower-triangular mask over the gathered scores stops any query from
        READING an entry that summarises its future.

        **The exact causality guarantee, which is not all of causality.** The
        mask governs reading. It says nothing about which entries are in the
        gathered sequence at all. Selection is content-dependent, so a token
        that changes its own score can change the set. Before D-023 a single
        global ``top_k`` let a token evict an entry belonging to an arbitrarily
        earlier position. Measured: perturbing token 31 evicted the entry base
        position 15 reads as itself, moving output 15 by 2.585.

        The per-block budget bounds that to one block. A perturbation at token
        ``T`` can move outputs at positions in ``T``'s own causal block of span
        ``p^(L-1)``, and never in any earlier block. Measured over all 28
        perturbation positions of the guard config: 0 cross-block leaks.
        Positions earlier in ``T``'s own block can still move. That residual is
        real, reproduced, and pinned by ``test_causality_is_per_position`` as a
        strict xfail. Closing it needs per-query selection and a block-wise
        SDPA, which is a different layer shape. Don't describe this layer as
        per-position causal.
    -   **Scatter-back.** Each entry's output is written to the base positions
        its window covers, offset by a causal shift of ``p^l - 1``. That shift
        is what makes coarse entries safe: a level-``l`` summary contributes
        only at or after the last token it pooled, so no future information
        leaks backwards. Accumulation uses ``keras.ops.segment_sum``, which is
        deterministic and needs no floating-point atomics.

    .. warning::
       Causality is **BLOCK-GRANULAR, not per-position.** A perturbation at
       token ``T`` can move outputs inside ``T``'s own causal block, so a query
       at position ``i`` may respond to position ``i + 1`` when both fall in the
       same block. Measured at ``N=32, L=2, p=2, top_k=16``: 8 violating
       ``(i, j)`` cells in the 32x32 support matrix, every one of them
       ``j == i + 1``. The residual is pinned by the strict-xfail
       ``test_causality_is_per_position``, so it can be neither forgotten nor
       quietly fixed. The per-block guarantee is pinned by
       ``test_causality_no_cross_block_leakage``.

       Don't use this layer where strict per-position causality is required.
       Autoregressive decoding is the obvious case.

       This block replaced a "currently BROKEN" warning. That warning said
       perturbing the last token "changes outputs at positions ``< N // 2``"
       with the mechanism "NOT DIAGNOSED", citing D-009 (2026-07-27). D-023
       (2026-07-29) fixed exactly that by partitioning ``top_k`` per causal
       block, two days after the warning was written, and the warning then sat
       stale for a month. Re-measured 2026-08-27: the described symptom is
       exactly 0.0, and all four tests D-009 named as red now pass
       (``12 passed, 1 xfailed``).

    A ``full_attention`` flag bypasses the pyramid entirely and runs plain
    causal attention over the full sequence. Set it at construction, or toggle
    it at runtime with ``set_full_attention``. It exists for two-stage training,
    where a model pretrained with sparse reads is resumed on dense attention.

Foundational Mathematics:
    The gathered sub-sequence has the static length::

        S = N / p^(L-1)  +  (p^(L-1) - 1)  +  top_k
            \\_________/     \\___________/     \\____/
            coarsest level   prefix cover    discretionary

    So attention costs ``O(S^2 * d)``, plus ``O(N)`` for pooling, scoring and
    scatter, against ``O(N^2 * d)`` for dense attention. ``S`` does not grow
    with ``N`` except through the ``N / p^(L-1)`` term, so the pyramid turns the
    quadratic term into one that shrinks geometrically with depth.

    The trade-off, stated exactly: any interaction not covered by a selected
    entry is served by a coarse MEAN of its window rather than by the exact
    token, if it is served at all. Raising ``top_k`` buys back exactness
    linearly. Raising ``num_levels`` buys back cost geometrically. The mandatory
    set exists so the approximation is never a HOLE: every base position always
    has at least one contributor, however coarse.

# DECISION plan_2026-07-26_c41d09b2/D-004
A revision of D-001, after a line-by-line audit against arXiv:2605.06554v1.
Three defects were corrected. Each is named because each failed silently: the
layer trained and converged in every case. Eight places in this file cite these
items by letter, so the letters (a), (b), (c) and their order are part of the
contract. Every quantity below describes the PRE-FIX code and is not a property
of the layer today. The originating plan directory is gone, so this comment is
the record.
  (a) The scorer read POST-QK-norm projections. RMSNorm maps every position to
      a near-constant L2 norm, and that norm is exactly the signal the scorer
      ranks on, so selection was close to arbitrary. The scorer now reads the
      raw ``W_Q x`` / ``W_K x`` projections, per paper Eq. 4, and
      ``qk_norm_type`` defaults to ``None``.
  (b) The coarsest level CONSUMED the ``top_k`` budget through a ``+1e9`` score
      boost, instead of being additive to it (paper Eq. 8). At ``L=3, p=4,
      top_k=1536`` and ``N >= 24576`` the whole budget went to coarsest
      entries, so no finer level was ever selected. Above ``N = 98304`` not
      even all coarsest entries fitted, and about 75% of base positions emitted
      exact zeros. The boost is gone.
  (c) The causal sort key was the window START, ``i*p^l``. An entry's causal
      timestamp is the LAST token it pooled, ``i*p^l + p^l - 1``, which is also
      where its scatter range begins. Sorting by the start let a coarse entry
      precede a fine entry lying inside its window, and the triangular mask
      then let that fine entry read its own future. The sort and the mask now
      use ``_causal_pos``. Ties are benign: two entries with equal timestamp
      both end there, so neither contains the other's future.

References:
    - Long Context Pre-Training with Lighthouse Attention (arXiv:2605.06554v1).
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

# ---------------------------------------------------------------------

import keras
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
# `import keras` only, per the repo convention. This file used
# `from keras import ops, layers, initializers, regularizers` until 2026-08-27.
# It was the last such import in layers/attention/, outside the five legacy
# files that are left alone on purpose (decisions.md D-026).

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Module-level static pyramid helpers (pure Python / numpy).
# These are called once in build() — they depend only on (N, L, p).
# ---------------------------------------------------------------------

# Additive mask sentinel. -1e9 overflows to -inf in float16, which turns a fully
# masked row into NaN under some probability activations; use a finite value that
# still underflows to zero after exponentiation in each dtype.
_MASK_SENTINEL: Dict[str, float] = {
    "float16": -6.0e4,
    "bfloat16": -1.0e9,
    "float32": -1.0e9,
    "float64": -1.0e9,
}


def _mask_value(dtype: Any) -> float:
    """Return a dtype-safe large-negative additive mask value.

    :param dtype: a backend tensor's ``.dtype`` -- i.e. a ``tf.DType`` or the
        plain dtype-name string. **Not** a ``keras.DTypePolicy`` and not
        ``None``; see the anchor below for what that would silently do.
    :type dtype: Any
    :return: the :data:`_MASK_SENTINEL` entry for that dtype; ``-1e9`` for any
        name not in the table.
    :rtype: float
    """
    # DECISION plan-2026-09-03T033750-9bdf25f4/D-007
    # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`
    # (a Keras-2 residue banned across `src/`; `str` alone mis-renders a
    # `tf.DType` as "<dtype: 'float16'>"). WHAT NOT TO DO: do not widen the
    # `Any` in the signature into a promise. The old symbol RAISED TypeError on a
    # `keras.DTypePolicy` and normalized `None` to floatx; this idiom does
    # neither, and the miss lands on `.get(..., -1.0e9)` -- so
    # `_mask_value(layer.dtype_policy)` under `mixed_float16` returns -1e9, which
    # `keras.ops.cast(..., "float16")` turns into **-inf** (measured), and one
    # fully-masked row is then NaN. That is silent, not loud: unlike
    # `common.py:apply_attention_mask`, where any unrecognised name still lands on
    # float32 via `mask_dtype`, there is no second line here to catch it.
    # Pass a TENSOR's `.dtype`, as both call sites do -- pinned by
    # `tests/test_layers/test_attention/test_the_lighthouse_mask_value_contract.py`.
    # See decisions.md D-007 and plans/DECISIONS.md D-018.
    return _MASK_SENTINEL.get(getattr(dtype, "name", None) or str(dtype), -1.0e9)


def _compute_level_sizes(n: int, num_levels: int, pooling_factor: int) -> np.ndarray:
    """Return the per-level entry count ``N / p^l`` for ``l = 0..L-1``."""
    return np.array(
        [n // (pooling_factor ** l) for l in range(num_levels)],
        dtype=np.int64,
    )


def _compute_base_starts(n: int, num_levels: int, pooling_factor: int) -> np.ndarray:
    """Return base-window-start positions for every pyramid entry, flat order.

    For each level ``l`` with ``N/p^l`` entries, the m-th entry covers the
    window ``[m * p^l, (m+1) * p^l)``. The returned int array is shape
    ``(S_pyr,)`` with ``S_pyr = sum_l N/p^l``.
    """
    parts: List[np.ndarray] = []
    for l in range(num_levels):
        fanout = pooling_factor ** l
        parts.append(np.arange(n // fanout, dtype=np.int64) * fanout)
    return np.concatenate(parts, axis=0)


def _compute_level_ids(n: int, num_levels: int, pooling_factor: int) -> np.ndarray:
    """Return per-entry level-id for every pyramid entry, flat order."""
    parts: List[np.ndarray] = []
    for l in range(num_levels):
        n_l = n // (pooling_factor ** l)
        parts.append(np.full(n_l, l, dtype=np.int64))
    return np.concatenate(parts, axis=0)


def _compute_causal_positions(
    n: int, num_levels: int, pooling_factor: int
) -> np.ndarray:
    """Return the causal timestamp of every pyramid entry, flat order.

    An entry's timestamp is the index of the *last* base token it pooled,
    ``i * p^l + p^l - 1``. This is the earliest position at which the entry's
    content is legitimately in the past, and it coincides with the start of the
    entry's scatter range (Eq. 10). It is therefore the only sound key for both
    the causal argsort and the triangular mask over the gathered sub-sequence.
    Ordering by the window *start* instead lets an entry read a coarse summary
    that spans its own future. See DECISION D-004(c).
    """
    starts = _compute_base_starts(n, num_levels, pooling_factor)
    levels = _compute_level_ids(n, num_levels, pooling_factor)
    return starts + (pooling_factor ** levels) - 1


def _compute_scatter_targets(
    n: int, num_levels: int, pooling_factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scatter target positions + validity mask, causally shifted.

    For each pyramid entry ``m`` at level ``l`` with base start ``b``, its
    output is scattered to base positions ``b + (p^l - 1) + k`` for
    ``k = 0..p^l - 1`` clipped to ``[0, N)`` — that is, Eq. 10's range
    ``[i p^l + p^l - 1, i p^l + 2 p^l - 2]``. The ``p^l - 1`` shift is the
    *causal* shift: it ensures that the entry's contribution lands strictly
    at-or-after the last input position it summarises (no future leakage).

    Returns ``(targets, valid_mask)`` shapes ``(S_pyr, MAX_FANOUT)``.
    Invalid positions (``k >= p^l`` or target ``>= N``) carry target ``N``
    (sentinel — segment_sum drops the trailing row) and ``valid_mask=False``.
    """
    max_fanout = pooling_factor ** (num_levels - 1)
    s_pyr = int(_compute_level_sizes(n, num_levels, pooling_factor).sum())
    # Target N is the sentinel meaning "drop this row".
    targets = np.full((s_pyr, max_fanout), n, dtype=np.int64)
    valid = np.zeros((s_pyr, max_fanout), dtype=bool)

    # Vectorised per level. The original triple loop ran
    # sum_l (N/p^l)*p^l = L*N Python iterations, which is minutes of build time
    # at N = 1M.
    offset = 0
    for l in range(num_levels):
        fanout = pooling_factor ** l
        n_l = n // fanout
        starts = np.arange(n_l, dtype=np.int64) * fanout + (fanout - 1)
        cand = starts[:, None] + np.arange(fanout, dtype=np.int64)[None, :]
        ok = cand < n
        targets[offset:offset + n_l, :fanout] = np.where(ok, cand, n)
        valid[offset:offset + n_l, :fanout] = ok
        offset += n_l
    return targets, valid


def _compute_mandatory_indices(
    n: int, num_levels: int, pooling_factor: int
) -> np.ndarray:
    """Return the flat-pyramid indices that are always retained.

    Two groups, both fixed at build time and both *additive* to the ``top_k``
    budget rather than competing with it (DECISION D-004(b)):

    - **All coarsest-level entries.** Paper §3.4: the coarsest level is cheap and
      guarantees a contributor at most base positions; Eq. 8 counts its
      ``N/p^(L-1)`` entries on top of the budget.
    - **Level-0 entries ``0 .. p^(L-1) - 2``.** The coarsest windows scatter to
      ``[i F + F - 1, i F + 2F - 2]`` with ``F = p^(L-1)``, whose union is
      ``[F - 1, N - 1]``: positions ``0 .. F - 2`` are unreachable from the
      coarsest level. A level-0 entry has fanout 1 and shift 0, so entry ``i``
      writes to base position ``i`` exactly, making this the cheapest set that
      closes the prefix hole.
    """
    sizes = _compute_level_sizes(n, num_levels, pooling_factor)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    coarsest_l = num_levels - 1
    coarsest = np.arange(
        int(offsets[coarsest_l]), int(offsets[coarsest_l + 1]), dtype=np.int64
    )
    fanout_max = pooling_factor ** coarsest_l
    # The prefix indices are level-0 entries, which sit first in flat order.
    prefix = np.arange(min(fanout_max - 1, n), dtype=np.int64)
    return np.union1d(coarsest, prefix).astype(np.int64)


# ---------------------------------------------------------------------

# DECISION plan-2026-07-27T130643-38c5646a/D-009 — HISTORICAL: a cross-block
# causality defect, FIXED by D-023 (per-causal-block `top_k`). Re-measured
# 2026-08-27 (N=32, L=2, p=2, top_k=16): perturbing the last token moves positions
# `< N // 2` by EXACTLY 0.0. Kept point: don't fix a causality defect in a docs
# pass, and don't relax a red test to look green. See decisions.md D-009, D-003.


@register_dl_technique("dl_techniques.layers.attention.lighthouse_attention")
class LighthouseAttention(keras.layers.Layer):
    """Lighthouse Attention — coarse-to-fine pyramid + top-K causal SDPA.

    A Keras 3 port of the Lighthouse Attention mechanism. It builds a symmetric
    Q/K/V pyramid of ``num_levels`` levels with mean-pool branching factor
    ``pooling_factor``. It scores each pyramid entry with a per-head L2-norm
    scorer over the raw projections, taking the joint QK / KQ max. It retains a
    mandatory index set plus the ``top_k`` highest-scoring remaining entries,
    and runs one causal SDPA on the gathered sub-sequence. Outputs are scattered
    back to base positions with a causal ``p^l - 1`` shift, via
    ``keras.ops.segment_sum``.

    Set ``full_attention=True``, or call ``set_full_attention(True)`` at
    runtime, to bypass the pyramid path entirely and run plain causal MHA. That
    is the Stage-2 SDPA-resume path.

    **Architecture Overview:**

    .. code-block:: text

        Input  x  [B, N, dim]
                  ▼
        Wq / Wk / Wv  ►  Q_raw, K_raw, V   [B, N, H, D]
                  │
            ┌─────┴──────────────────┐
            │ raw Q, K       Q, K, V │
            ▼                        ▼
        ┌────────────────────┐ ┌────────────────────────┐
        │ SELECTOR           │ │ TRUNK                  │
        │ ||Q_raw||          │ │ q_norm / k_norm        │
        │ and ||K_raw||,     │ │ (optional; the SELECTOR│
        │ at LEVEL 0 ONLY    │ │ never sees them)       │
        │        ▼           │ │          ▼             │
        │ MAX-pool them up   │ │ MEAN-pool up the       │
        │ the pyramid, Eq. 5.│ │ pyramid, l = 0..L-1,   │
        │ NEVER recomputed   │ │ window p^l. Level 0 is │
        │ on pooled Q/K      │ │ an identity copy.      │
        │        ▼           │ │          ▼             │
        │ max(s_QK, s_KQ),   │ │ Q_pyr  K_pyr  V_pyr    │
        │ then mean or max   │ │ [B, S_pyr, H, D]       │
        │ over H             │ └───────────┬────────────┘
        │ s  [B, S_pyr]      │             │
        │        ▼           │             │
        │ top_k over the     │             │
        │ NON-mandatory pool │             │
        │ only, per causal   │             │
        │ block; UNION the   │             │
        │ mandatory set      │             │
        │ (every coarsest    │             │
        │ entry + level-0    │             │
        │ prefix 0..F-2)     │             │
        │        ▼           │             │
        │ argsort by causal  │             │
        │ pos t = i·p^l      │             │
        │           + p^l - 1│             │
        └──────────┬─────────┘             │
                   │ I  [B, S]. ONE index  │
                   │ set, shared by every  │
                   │ head.                 │
                   └───────────┬───────────┘
                               ▼
        gather pyramid entries at I   ►  [B, S, H, D]
                               ▼
        causal SDPA over S: triangular mask, attn_prob,
        dropout. Exact over S, because I is causally sorted.
                               ▼
        scatter back via segment_sum to base positions
        i·p^l + p^l - 1 + k. Fan-in <= L; contributions are
        SUMMED and are NOT normalised by fan-in.
                               ▼
        Wo  ►  Output  [B, N, dim]

        Causality here is BLOCK-granular, not per-position. See the
        warning below before using this in a decoder.

    Shapes use ``B`` = batch, ``N`` = seq_len, ``H`` = num_heads,
    ``D`` = head_dim, ``L`` = num_levels, ``p`` = pooling_factor,
    ``F`` = p^(L-1), ``S_pyr`` = sum_l N/p^l, and
    ``S`` = N/F + (F - 1) + top_k.

    ``full_attention=True`` bypasses everything between the projections and
    ``Wo``, running a single causal SDPA over all ``N`` positions.

    :param dim: Model dimension (hidden size). Must be positive.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive and
        divide ``dim`` unless ``head_dim`` is explicitly set.
    :type num_heads: int
    :param head_dim: Dimension per head. If ``None``, ``head_dim = dim //
        num_heads``. Defaults to ``None``.
    :type head_dim: Optional[int]
    :param num_levels: Number of pyramid levels (``L``). Defaults to 3.
    :type num_levels: int
    :param pooling_factor: Branching factor per level (``p``). Defaults to 4.
    :type pooling_factor: int
    :param top_k: Discretionary pyramid entries selected per batch element, on
        top of the mandatory set. Defaults to 1536. Clipped at build time to the
        number of non-mandatory candidates.
    :type top_k: int
    :param scorer: Scorer type. Only ``"norm"`` supported (port compromise).
    :type scorer: str
    :param score_head_reduction: How the per-head scores collapse to the single
        shared index set, ``"mean"`` or ``"max"``. Defaults to ``"mean"``;
        ``"max"`` lets one outlier head dictate selection for all heads.
    :type score_head_reduction: str
    :param full_attention: If ``True``, bypass pyramid path → plain causal
        SDPA over the full sequence. Defaults to ``False``.
    :type full_attention: bool
    :param qk_norm_type: Norm layer type applied to Q, K *after* scoring, or
        ``None`` to disable. Defaults to ``None``: the paper's scorer ranks
        ``||Q||`` and ``||K||``, and RMSNorm makes both near-constant across
        positions, so normalising before the scorer erases the selection signal
        (DECISION D-004(a)).
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional kwargs forwarded to the norm
        factory. Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param probability_type: Score-normalization strategy applied to the
        attention logits via :class:`ProbabilityOutput`. Defaults to
        ``"softmax"``. Routing / hierarchical types are not supported.
    :type probability_type: str
    :param probability_config: Optional kwargs forwarded to
        :class:`ProbabilityOutput`. Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param use_bias: Use bias in Dense projections. Defaults to ``False``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for Dense kernels.
        Defaults to ``"glorot_uniform"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for biases. Defaults to ``"zeros"``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional kernel regularizer.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param dropout_rate: Dropout applied to the normalized attention weights in
        both the pyramid and full-attention paths. Defaults to 0.0.
    :type dropout_rate: float
    :param kwargs: Additional arguments for the ``Layer`` base class.

    :raises ValueError: If any argument is invalid.

    .. note::
        **Call-signature contract.** ``call()`` accepts only ``(inputs,
        training=None)``. There is NO ``attention_mask`` parameter: masking is
        internal and is never caller-supplied.

        The layer also requires a statically-known sequence length. The pyramid
        index buffers are built in ``build()`` from the concrete ``N`` of
        ``input_shape``. Built with a dynamic or ``None`` sequence dimension,
        ``call()`` raises ``RuntimeError`` ("requires a statically known
        sequence length"). Build with a concrete ``N``, which is the common
        training case. A static BATCH dimension is not required, but it does let
        ``segment_sum`` take a concrete ``num_segments``, which ``jax.jit``
        needs.

    .. warning::
        **Causality is block-granular, not per-position.** Cross-block causality
        holds: a perturbation cannot reach an EARLIER causal block. Within a
        block it can, so a query at position ``i`` may respond to ``i + 1`` when
        both fall in the same block. Measured at ``N=32, L=2, p=2, top_k=16``:
        8 violating cells, all ``j == i + 1``.

        Don't use this layer where strict per-position causality decides
        correctness. Autoregressive decoding and next-token training are the
        cases that break. The block-level guarantee is pinned by
        ``test_causality_no_cross_block_leakage``, and the residual by the
        strict-xfail ``test_causality_is_per_position``.

        This replaced a stale "not currently causal / MECHANISM IS NOT
        DIAGNOSED" warning. See the D-009 anchor above the class.

    .. warning::
        **This layer accepts NO attention mask.** ``call(inputs, training=None)``
        has no ``attention_mask`` parameter, which is why
        ``TransformerLayer`` lists it in ``_MASKLESS_ATTENTION_TYPES`` and
        SILENTLY DISCARDS a caller's mask for this type.

        The cost is severity-asymmetric, measured 2026-08-27:

        * **right-padding: exactly 0.0.** The causal design already isolates
          trailing padding, so the missing mask costs nothing.
        * **left-padding: 21.61 absolute.** This is the convention causal-LM
          decoding actually uses, and padded positions contaminate real ones.

        If your batch is left-padded, do not use this layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        num_levels: int = 3,
        pooling_factor: int = 4,
        top_k: int = 1536,
        scorer: str = "norm",
        score_head_reduction: str = "mean",
        full_attention: bool = False,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

        The four Dense projections, the optional QK norms, the probability
        layer and the optional dropout are all created here. The pyramid index
        buffers are NOT: they are pure functions of ``(N, num_levels,
        pooling_factor)`` and ``N`` is only known in :meth:`build`. See the
        class docstring for the parameter reference.

        :raises ValueError: For any invalid argument; see the class docstring.
        """
        super().__init__(**kwargs)

        # ---- validation ----
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by num_heads ({num_heads}) "
                    f"when head_dim is not specified."
                )
            head_dim = dim // num_heads
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if pooling_factor < 2:
            raise ValueError(f"pooling_factor must be >= 2, got {pooling_factor}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if scorer != "norm":
            raise ValueError(
                f"scorer must be 'norm' (only port-supported scorer), got {scorer!r}"
            )
        if score_head_reduction not in ("mean", "max"):
            raise ValueError(
                f"score_head_reduction must be 'mean' or 'max', got "
                f"{score_head_reduction!r}"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )
        if probability_type in (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type={probability_type!r} is not supported for "
                f"LighthouseAttention score normalization."
            )

        # ---- store config ----
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        self.top_k = top_k
        self.scorer = scorer
        self.score_head_reduction = score_head_reduction
        self.full_attention = bool(full_attention)
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.dropout_rate = dropout_rate

        # Derived static ints.
        self._p_pow_max: int = pooling_factor ** (num_levels - 1)
        self._scale: float = 1.0 / float(head_dim) ** 0.5

        # ---- sub-layers (built in build()) ----
        proj_units = num_heads * head_dim

        def _dense(units: int, name: str) -> keras.layers.Dense:
            """Build one projection Dense with this layer's shared settings.

            :param units: Output width of the projection.
            :type units: int
            :param name: Sub-layer name.
            :type name: str
            :return: An unbuilt ``Dense``.
            :rtype: keras.layers.Dense
            """
            return keras.layers.Dense(
                units,
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=name,
            )

        self.wq = _dense(proj_units, "wq")
        self.wk = _dense(proj_units, "wk")
        self.wv = _dense(proj_units, "wv")
        self.wo = _dense(dim, "wo")

        # QK-norm is opt-in and is applied only to the tensors that enter the
        # pyramid, never to the tensors the scorer reads (DECISION D-004(a)).
        if qk_norm_type is not None:
            norm_kwargs = dict(qk_norm_kwargs or {})
            self.q_norm = create_normalization_layer(
                qk_norm_type, name="q_norm", **norm_kwargs
            )
            self.k_norm = create_normalization_layer(
                qk_norm_type, name="k_norm", **norm_kwargs
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )
        self.attn_dropout = (
            keras.layers.Dropout(dropout_rate, name="attn_dropout")
            if dropout_rate > 0.0
            else None
        )

        # Numpy buffers populated in build().
        self._level_sizes: Optional[np.ndarray] = None
        self._level_offsets: Optional[np.ndarray] = None
        self._base_starts: Optional[np.ndarray] = None
        self._causal_pos: Optional[np.ndarray] = None
        self._scatter_targets: Optional[np.ndarray] = None
        self._scatter_valid_mask: Optional[np.ndarray] = None
        self._mandatory_indices: Optional[np.ndarray] = None
        self._candidate_indices: Optional[np.ndarray] = None
        self._S_pyr: Optional[int] = None
        self._S_sel: Optional[int] = None
        self._effective_k: Optional[int] = None
        self._N_static: Optional[int] = None
        self._B_static: Optional[int] = None
        self._max_fanout: int = self._p_pow_max

        # Device-side copies of the numpy buffers, materialised lazily on first
        # call and reused thereafter. Converting them per call costs a
        # host-to-device copy of O(S_pyr * p^(L-1)) ints every step — hundreds of
        # MB per layer at N = 1M.
        self._const_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_full_attention(self, full: bool) -> None:
        """Toggle the Stage-2 SDPA-resume mode at runtime.

        :param full: If ``True``, ``call`` bypasses the pyramid path and
            runs plain causal SDPA over the full sequence.
        """
        self.full_attention = bool(full)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary holding every ``__init__`` argument. ``N`` is not
            stored: the pyramid buffers are rebuilt from ``input_shape``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "num_levels": self.num_levels,
                "pooling_factor": self.pooling_factor,
                "top_k": self.top_k,
                "scorer": self.scorer,
                "score_head_reduction": self.score_head_reduction,
                "full_attention": self.full_attention,
                "qk_norm_type": self.qk_norm_type,
                "qk_norm_kwargs": self.qk_norm_kwargs,
                "probability_type": self.probability_type,
                "probability_config": self.probability_config,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape. Only the last axis changes, to ``dim``.

        :param input_shape: ``(B, N, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, N, dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], input_shape[1], self.dim)

    # ------------------------------------------------------------------
    # build()
    # ------------------------------------------------------------------
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers and precompute static pyramid buffers.

        Numpy buffers (level sizes, base-window-starts, causal positions,
        scatter targets, mandatory/candidate index sets) are stored on ``self``
        as plain numpy arrays — they are pure functions of ``(N, num_levels,
        pooling_factor)`` and are re-derived in a fresh ``build()`` after
        ``from_config()`` restoration. See LESSONS: frozen tensor state must NOT
        live in plain ``keras.ops.*`` tensors created in ``build()``.

        :param input_shape: ``(B, N, dim)``.
        :raises ValueError: Non-3D input, or static ``N`` not divisible by
            ``pooling_factor ** (num_levels - 1)``.
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape (batch, seq_len, dim), got {input_shape}"
            )
        self._B_static = input_shape[0]
        n_static = input_shape[1]

        if n_static is not None:
            if n_static % self._p_pow_max != 0:
                raise ValueError(
                    f"seq_len N={n_static} must be divisible by "
                    f"pooling_factor ** (num_levels - 1) = {self._p_pow_max}. "
                    f"Either pad inputs or adjust (num_levels, pooling_factor)."
                )
            self._populate_pyramid_buffers(int(n_static))
        # else: call() raises; dynamic-N pyramid construction is build-time-only.

        # Build sub-layers explicitly (mandatory for .keras save/load).
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        head_shape = (input_shape[0], input_shape[1], self.num_heads, self.head_dim)
        if self.q_norm is not None:
            self.q_norm.build(head_shape)
            self.k_norm.build(head_shape)
        # Output projection consumes (B, N, H*D).
        self.wo.build((input_shape[0], input_shape[1], self.num_heads * self.head_dim))

        # Build ProbabilityOutput with the shape of the path that actually runs.
        # Softmax/sparsemax/threshmax operate along the last axis and are
        # shape-agnostic in practice, so the square choice below is immaterial to
        # them; it matters only for a variant carrying key-axis weights.
        s_q = self._S_sel if (self._S_sel and not self.full_attention) else input_shape[1]
        self.attn_prob.build((input_shape[0], self.num_heads, s_q, s_q))
        if self.attn_dropout is not None:
            self.attn_dropout.build((input_shape[0], self.num_heads, s_q, s_q))

        super().build(input_shape)

    def _populate_pyramid_buffers(self, n: int) -> None:
        """Compute and store static numpy buffers for a given N.

        Idempotent — safe to recompute if ``N`` changes between calls (though
        Keras layers are typically built once with a fixed N). Invalidates the
        device-side cache.
        """
        L, p = self.num_levels, self.pooling_factor
        sizes = _compute_level_sizes(n, L, p)
        self._level_sizes = sizes
        self._level_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self._base_starts = _compute_base_starts(n, L, p)
        self._causal_pos = _compute_causal_positions(n, L, p)
        targets, valid = _compute_scatter_targets(n, L, p)
        self._scatter_targets = targets
        self._scatter_valid_mask = valid
        self._S_pyr = int(sizes.sum())
        self._N_static = n
        self._const_cache = {}

        # Mandatory (always-retained) set and its complement, the pool the
        # discretionary top_k draws from. Keeping these disjoint is what makes
        # the budget additive rather than self-competing (DECISION D-004(b)).
        mandatory = _compute_mandatory_indices(n, L, p)
        self._mandatory_indices = mandatory
        mask = np.ones((self._S_pyr,), dtype=bool)
        mask[mandatory] = False
        self._candidate_indices = np.nonzero(mask)[0].astype(np.int64)

        n_cand = int(self._candidate_indices.size)
        self._effective_k = int(min(self.top_k, n_cand))

        # DECISION plan-2026-07-29T110112-09832856/D-023 — PARTITION THE
        # DISCRETIONARY BUDGET BY CAUSAL TIME. Don't replace this with one global
        # `keras.ops.top_k`: measured pre-fix (N=32, L=2, p=2, top_k=16), perturbing
        # token 31 EVICTED pyramid entry 15, moving output 15 by 2.585; a triangular
        # mask cannot stop that. Buys BLOCK causality only. See decisions.md D-023.
        cand_causal = self._causal_pos[self._candidate_indices]
        block_span = int(self._max_fanout)
        self._sel_block_span = block_span
        nb = max(1, int(np.ceil(n / float(block_span))))
        block_of_cand = np.minimum(cand_causal // block_span, nb - 1)

        # Every block must be able to fund its share, so the per-block budget
        # is capped by the thinnest block. Blocks with no candidates at all are
        # dropped from the partition rather than funded with padding.
        occupied = [
            np.nonzero(block_of_cand == b)[0] for b in range(nb)
        ]
        occupied = [idx for idx in occupied if idx.size > 0]
        nb_eff = len(occupied)
        thinnest = min(int(idx.size) for idx in occupied) if nb_eff else 0

        per_block = self._effective_k // nb_eff if nb_eff else 0
        per_block = int(min(per_block, thinnest))
        if per_block < 1 and nb_eff:
            # Budget smaller than the block count: fund the earliest blocks
            # one entry each. Earliest, not highest-scoring, because "which
            # blocks get funded" must not itself depend on future content.
            per_block = 1
            occupied = occupied[: self._effective_k]
            nb_eff = len(occupied)

        self._sel_num_blocks = nb_eff
        self._sel_per_block = per_block
        width = max((int(idx.size) for idx in occupied), default=0)
        block_cand = np.full((nb_eff, width), -1, dtype=np.int64)
        for b, idx in enumerate(occupied):
            block_cand[b, : idx.size] = self._candidate_indices[idx]
        self._sel_block_cand = block_cand
        self._sel_block_valid = block_cand >= 0

        budget = per_block * nb_eff
        if budget != self._effective_k:
            logger.info(
                "LighthouseAttention: causal-block partition trimmed the "
                "discretionary budget %d -> %d (%d blocks x %d, span %d). The "
                "budget must divide evenly across causal blocks and cannot "
                "exceed the thinnest block's %d candidates; see D-023.",
                self._effective_k, budget, nb_eff, per_block, block_span,
                thinnest,
            )
        self._effective_k = int(budget)
        self._S_sel = int(mandatory.size) + self._effective_k

        if self._effective_k < self.top_k:
            logger.warning(
                "LighthouseAttention: top_k=%d exceeds the %d non-mandatory "
                "pyramid entries at N=%d; clipping to %d. The gathered "
                "sub-sequence covers the whole pyramid, so this configuration is "
                "dense attention with extra bookkeeping.",
                self.top_k, n_cand, n, self._effective_k,
            )

        # De-anchored 2026-08-27: this read `# DECISION D-001 item 2`, a BARE
        # anchor with no plan-id prefix, which `validate-plan.mjs` reports as
        # `anchor-unqualified`. Its owner is `plan_2026-07-26_c41d09b2`: the
        # D-004 block above calls itself a revision of D-001. That plan
        # directory has been sliding-window trimmed and no D-001 entry survives
        # anywhere in `plans/`, so qualifying it would just produce an
        # unresolvable orphan. The reasoning is kept inline instead, which is
        # what an anchor is for: segment_sum has no broadcast form, so the
        # scatter materialises one value row per (entry, target) pair.
        tiled = self._S_sel * self._max_fanout
        if tiled > 4 * n:
            logger.warning(
                "LighthouseAttention: scatter-back will materialise "
                "S*p^(L-1) = %d rows against N = %d (%.1fx). The deterministic "
                "segment_sum scatter is the dominant memory term in this "
                "configuration; lower num_levels or top_k, or accept the cost.",
                tiled, n, tiled / float(n),
            )

        logger.info(
            "LighthouseAttention: N=%d L=%d p=%d -> S_pyr=%d, mandatory=%d "
            "(coarsest %d + prefix %d), top_k=%d, gathered S=%d (%.3f of N)",
            n, self.num_levels, self.pooling_factor, self._S_pyr,
            int(mandatory.size), n // self._max_fanout,
            max(self._max_fanout - 1, 0), self._effective_k,
            self._S_sel, self._S_sel / float(n),
        )

    # ------------------------------------------------------------------
    # Cached device-side constants
    # ------------------------------------------------------------------
    def _const(self, name: str, arr: np.ndarray, dtype: str) -> Any:
        """Return a cached device tensor for a static numpy buffer."""
        cached = self._const_cache.get(name)
        if cached is None:
            cached = keras.ops.convert_to_tensor(arr.astype(dtype))
            self._const_cache[name] = cached
        return cached

    # ------------------------------------------------------------------
    # Pyramid pool + norm scorer
    # ------------------------------------------------------------------
    def _pyramid_pool(self, x_heads: keras.KerasTensor) -> keras.KerasTensor:
        """Mean-pool a (B, N, H, D) tensor into a (B, S_pyr, H, D) pyramid.

        Level 0 is an identity copy (N entries). Each successive level ``l``
        reshapes the base sequence into ``(B, N/p^l, p^l, H, D)`` and reduces
        over the window axis with ``keras.ops.mean`` — equivalent to pooling from
        level ``l-1`` because the windows are equal-sized and nested. Levels are
        concatenated along the sequence axis from level 0 to L-1 (Eq. 3).
        """
        n = self._N_static
        h, d = self.num_heads, self.head_dim
        parts: List[keras.KerasTensor] = []
        for l in range(self.num_levels):
            fanout = self.pooling_factor ** l
            if l == 0:
                parts.append(x_heads)
            else:
                # (B, N, H, D) -> (B, N/p^l, p^l, H, D) -> mean over axis=2.
                # Batch stays dynamic via -1; every other dim is static.
                reshaped = keras.ops.reshape(x_heads, (-1, n // fanout, fanout, h, d))
                parts.append(keras.ops.mean(reshaped, axis=2))
        # Result: (B, S_pyr, H, D).
        return keras.ops.concatenate(parts, axis=1)

    def _norm_scorer(
        self,
        q_raw: keras.KerasTensor,
        k_raw: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Per-head L2-norm scorer with max-pool over coarser levels.

        Paper Eq. 4-5: norms are computed once on the level-0 projections and
        *max-pooled* up the pyramid — NOT recomputed on mean-pooled projections,
        so a coarse span inherits the salience of its strongest token. Returns
        ``(s_qk_pyr, s_kq_pyr)`` each shape ``(B, S_pyr, H)``, where
        ``s_qk = ||Q||`` and ``s_kq = ||K||`` are the two terms whose per-entry
        max gives the joint scorer.

        Both arguments must be the **raw** projections. Passing QK-normalised
        tensors flattens ``||Q||`` and ``||K||`` to a near-constant and destroys
        the ranking (DECISION D-004(a)).
        """
        n = self._N_static
        h = self.num_heads

        # Both are (B, N, H).
        s_q0 = keras.ops.norm(q_raw, axis=-1)
        s_k0 = keras.ops.norm(k_raw, axis=-1)

        q_parts: List[keras.KerasTensor] = []
        k_parts: List[keras.KerasTensor] = []
        for l in range(self.num_levels):
            fanout = self.pooling_factor ** l
            if l == 0:
                q_parts.append(s_q0)
                k_parts.append(s_k0)
            else:
                q_resh = keras.ops.reshape(s_q0, (-1, n // fanout, fanout, h))
                k_resh = keras.ops.reshape(s_k0, (-1, n // fanout, fanout, h))
                q_parts.append(keras.ops.max(q_resh, axis=2))
                k_parts.append(keras.ops.max(k_resh, axis=2))
        # Both concatenations are (B, S_pyr, H).
        return (
            keras.ops.concatenate(q_parts, axis=1),
            keras.ops.concatenate(k_parts, axis=1),
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    # Top-K is shared across heads: ONE (B, S) index set, not one per head.
    # That is a port simplification, from the same retired D-001 as the
    # de-anchoring note in `_populate_pyramid_buffers`. The mandatory set is
    # additive to top_k rather than competing with it, and the union is sorted
    # by causal timestamp, so a plain triangular mask over the gathered
    # sub-sequence is exact.
    def _select(
        self,
        s_qk_pyr: keras.KerasTensor,
        s_kq_pyr: keras.KerasTensor,
        batch_size: Any,
    ) -> keras.KerasTensor:
        """Return the (B, S) gathered index set, sorted by causal position."""
        # Per-entry max across the QK / KQ streams, then collapse heads.
        # joint is (B, S_pyr, H); s_shared is (B, S_pyr).
        joint = keras.ops.maximum(s_qk_pyr, s_kq_pyr)
        if self.score_head_reduction == "mean":
            s_shared = keras.ops.mean(joint, axis=-1)
        else:
            s_shared = keras.ops.max(joint, axis=-1)

        mand_1d = self._const("mandatory", self._mandatory_indices, "int32")
        selected = keras.ops.broadcast_to(
            mand_1d[None, :], (batch_size, int(self._mandatory_indices.size))
        )

        if self._effective_k > 0:
            # PER-CAUSAL-BLOCK top_k, NOT a global one. See the D-023 anchor in
            # `_populate_pyramid_buffers`: a single global budget lets a future
            # token evict a past token's entry, which is the causality defect
            # this layer shipped with. Confining each competition to one causal
            # block bounds the blast radius of a perturbation to that block.
            nb, per_b = self._sel_num_blocks, self._sel_per_block
            # block_cand is (nb, W), padded with -1; valid is (nb, W).
            block_cand = self._const(
                "block_cand", self._sel_block_cand, "int32"
            )
            valid = self._const(
                "block_valid", self._sel_block_valid.astype(np.int64), "bool"
            )

            # (B, nb, W) scores. Padded slots must never win, so they are
            # pushed below any real score with the dtype-safe sentinel rather
            # than -inf (which NaNs a fully padded row under fp16).
            flat = keras.ops.reshape(block_cand, (-1,))
            s_blocks = keras.ops.take(s_shared, flat, axis=1)
            s_blocks = keras.ops.reshape(
                s_blocks, (-1, nb, int(self._sel_block_cand.shape[1]))
            )
            s_blocks = keras.ops.where(
                valid[None, :, :],
                s_blocks,
                keras.ops.cast(_mask_value(s_blocks.dtype), s_blocks.dtype),
            )

            # local_idx is (B, nb, per_b).
            _, local_idx = keras.ops.top_k(s_blocks, k=per_b)
            # picked holds the chosen pyramid indices, also (B, nb, per_b).
            picked = keras.ops.take_along_axis(
                keras.ops.broadcast_to(
                    block_cand[None, :, :],
                    keras.ops.shape(s_blocks),
                ),
                local_idx,
                axis=-1,
            )
            fine = keras.ops.reshape(picked, (-1, nb * per_b))
            selected = keras.ops.concatenate([selected, fine], axis=1)

        # Sort by causal timestamp (last summarised base position). Ordering by
        # the window *start* instead would let a fine entry read a coarse summary
        # spanning its own future — see DECISION D-004(c).
        causal_t = self._const("causal_pos", self._causal_pos, "int32")
        # t_of_selected is (B, S).
        t_of_selected = keras.ops.take(causal_t, selected, axis=0)
        order = keras.ops.argsort(t_of_selected, axis=-1)
        return keras.ops.take_along_axis(selected, order, axis=-1)

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------
    def _causal_sdpa(
        self,
        q_heads: keras.KerasTensor,
        k_heads: keras.KerasTensor,
        v_heads: keras.KerasTensor,
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Causal scaled dot-product attention over (B, S, H, D) tensors.

        Written out rather than delegated to ``keras.ops.dot_product_attention`` so
        that ``self.attn_prob`` controls score normalization. Shared by the
        pyramid path (where ``S`` is the gathered sub-sequence, and the mask is
        exact because the gather is sorted by causal timestamp) and the Stage-2
        full-attention path (where ``S = N``).
        """
        # (B, S, H, D) -> (B, H, S, D)
        q_t = keras.ops.transpose(q_heads, (0, 2, 1, 3))
        k_t = keras.ops.transpose(k_heads, (0, 2, 1, 3))
        v_t = keras.ops.transpose(v_heads, (0, 2, 1, 3))

        # DECISION plan_2026-06-14_33b77a7a/D-002
        # Reuse the precomputed self._scale. Don't recompute keras.ops.sqrt per
        # call. In float32, keras.ops.cast(self._scale, dt) equals
        # 1/keras.ops.sqrt(cast(head_dim, dt)); measured 0 mismatches over
        # head_dim in {8, 16, 32, 64, 128}.
        # The originating plan directory is gone, so this comment is the record.
        scale = keras.ops.cast(self._scale, q_t.dtype)
        scores = keras.ops.matmul(q_t, keras.ops.transpose(k_t, (0, 1, 3, 2))) * scale

        # Lower-triangular keep-mask, inclusive of the diagonal.
        i = keras.ops.arange(keras.ops.shape(scores)[-2])
        j = keras.ops.arange(keras.ops.shape(scores)[-1])
        keep = keras.ops.expand_dims(j, 0) <= keras.ops.expand_dims(i, -1)
        scores = keras.ops.where(
            keep, scores, keras.ops.cast(_mask_value(scores.dtype), scores.dtype)
        )

        attn = self.attn_prob(scores)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)
        # out_t is (B, H, S, D); the transpose returns (B, S, H, D).
        out_t = keras.ops.matmul(attn, v_t)
        return keras.ops.transpose(out_t, (0, 2, 1, 3))

    def _gather_and_attend(
        self,
        q_pyr: keras.KerasTensor,
        k_pyr: keras.KerasTensor,
        v_pyr: keras.KerasTensor,
        sel_idx: keras.KerasTensor,
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Gather pyramid entries at ``sel_idx`` and run causal SDPA."""
        # sel_idx: (B, S). Expand to (B, S, 1, 1) for gather along axis=1.
        idx_exp = sel_idx[:, :, None, None]
        # Each gather returns (B, S, H, D).
        q_g = keras.ops.take_along_axis(q_pyr, idx_exp, axis=1)
        k_g = keras.ops.take_along_axis(k_pyr, idx_exp, axis=1)
        v_g = keras.ops.take_along_axis(v_pyr, idx_exp, axis=1)
        return self._causal_sdpa(q_g, k_g, v_g, training)

    # ------------------------------------------------------------------
    # Scatter-back
    # ------------------------------------------------------------------
    # Flat segment_sum: encode (batch, target) as a single segment id
    # (`b * (N+1) + target`) so one 1-D segment_sum covers all batches
    # deterministically. Target N is the sentinel for "drop" (invalid /
    # out-of-range positions); the trailing slice removes it. Contributions from
    # different levels sum at shared positions, with fan-in bounded by L
    # (Eq. 11), and are not normalized by fan-in. That is the intended
    # behaviour, not an oversight.
    def _scatter_back(
        self,
        out_g: keras.KerasTensor,
        sel_idx: keras.KerasTensor,
        batch_size: Any,
        n: int,
    ) -> keras.KerasTensor:
        """Scatter (B, S, H, D) sub-attention output back to (B, N, H, D)."""
        h, d = self.num_heads, self.head_dim
        s_sel, fanout = self._S_sel, self._max_fanout

        targets_t = self._const("targets", self._scatter_targets, "int32")
        valid_t = self._const("valid", self._scatter_valid_mask, "float32")
        # targets_g is (B, S, F).
        targets_g = keras.ops.take(targets_t, sel_idx, axis=0)
        valid_g = keras.ops.cast(keras.ops.take(valid_t, sel_idx, axis=0), out_g.dtype)

        # Tile output along fanout: (B, S, F, H, D). segment_sum has no broadcast
        # form, so this F-fold expansion is inherent to the deterministic path;
        # build() warns when it dominates.
        out_tiled = keras.ops.repeat(out_g[:, :, None, :, :], fanout, axis=2)
        out_tiled = out_tiled * valid_g[..., None, None]

        # out_flat is (B*S*F, H, D); targets_flat is (B, S*F).
        out_flat = keras.ops.reshape(out_tiled, (-1, h, d))
        targets_flat = keras.ops.reshape(targets_g, (-1, s_sel * fanout))

        # Encode (batch, target) into one segment id; N+1 slots per batch with N
        # as the "drop" sentinel.
        batch_offset = keras.ops.arange(batch_size, dtype="int32") * (n + 1)
        flat_segments = keras.ops.reshape(
            keras.ops.cast(targets_flat, "int32") + batch_offset[:, None], (-1,)
        )

        # A static batch lets num_segments be a Python int, which jax.jit needs.
        num_segments = (
            self._B_static * (n + 1)
            if self._B_static is not None
            else batch_size * (n + 1)
        )
        # The segment_sum result is (B*(N+1), H, D).
        scattered = keras.ops.segment_sum(
            out_flat, flat_segments, num_segments=num_segments
        )
        scattered = keras.ops.reshape(scattered, (-1, n + 1, h, d))
        # Drop the trailing sentinel row.
        return scattered[:, :n, :, :]

    # ------------------------------------------------------------------
    # call()
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: ``(B, N, dim)``.
        :param training: Propagated to the attention-weight dropout.
        :return: ``(B, N, dim)``.
        :raises RuntimeError: If the layer was built without a static ``N``.
        """
        batch_size = keras.ops.shape(inputs)[0]
        if self._N_static is None:
            raise RuntimeError(
                "LighthouseAttention requires a statically known sequence "
                "length. Build the layer with a concrete N."
            )
        n = self._N_static
        h, d = self.num_heads, self.head_dim

        # Project Q, K, V -> (B, N, H, D). These are the RAW projections.
        q_raw = keras.ops.reshape(self.wq(inputs), (-1, n, h, d))
        k_raw = keras.ops.reshape(self.wk(inputs), (-1, n, h, d))
        v = keras.ops.reshape(self.wv(inputs), (-1, n, h, d))

        # Optional QK-norm, applied only to what enters attention — the scorer
        # below reads q_raw / k_raw (DECISION D-004(a)).
        q = self.q_norm(q_raw) if self.q_norm is not None else q_raw
        k = self.k_norm(k_raw) if self.k_norm is not None else k_raw

        if self.full_attention:
            # Stage-2 SDPA-resume path: plain causal MHA over the full
            # sequence. out is (B, N, H, D).
            out = self._causal_sdpa(q, k, v, training)
            return self.wo(keras.ops.reshape(out, (-1, n, h * d)))

        # Lighthouse pyramid path. Shapes, in order: q_pyr / k_pyr / v_pyr are
        # (B, S_pyr, H, D), sel_idx is (B, S), out_g is (B, S, H, D), and
        # out_base is (B, N, H, D).
        q_pyr = self._pyramid_pool(q)
        k_pyr = self._pyramid_pool(k)
        v_pyr = self._pyramid_pool(v)
        s_qk_pyr, s_kq_pyr = self._norm_scorer(q_raw, k_raw)
        sel_idx = self._select(s_qk_pyr, s_kq_pyr, batch_size)
        out_g = self._gather_and_attend(
            q_pyr, k_pyr, v_pyr, sel_idx, training
        )
        out_base = self._scatter_back(out_g, sel_idx, batch_size, n)
        return self.wo(keras.ops.reshape(out_base, (-1, n, h * d)))

# ---------------------------------------------------------------------