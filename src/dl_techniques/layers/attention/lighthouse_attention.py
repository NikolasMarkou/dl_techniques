"""
Sparse causal attention over a coarse-to-fine pyramid of pooled representations.

This module provides a Keras 3 / TF 2.18 implementation of Lighthouse Attention.
Its fundamental purpose is to let every position read its entire causal prefix
without paying the quadratic cost of reading all of it at full resolution.
Standard self-attention spends identical precision on distant and on local
context; Lighthouse instead spends precision where the content warrants it,
resolving salient history at token granularity and the remainder through
progressively coarser summaries.

Architecturally, the single ``N x N`` score matrix is replaced by a *pyramid* of
pooled representations plus a sparse, content-selected read over that pyramid.
The mechanism proceeds in five stages:

-   **Pyramid construction.** Q, K and V are projected once and mean-pooled into
    ``num_levels`` levels with branching factor ``pooling_factor``. Level 0 is
    the base sequence at full resolution; level ``l`` holds ``N / p^l`` entries,
    each summarising a window of ``p^l`` consecutive tokens. Concatenating the
    levels yields ``S_pyr = sum_l N / p^l`` candidate entries per sequence.
-   **Scoring.** Every candidate receives a per-head L2-norm score. The norms
    ``||Q||`` and ``||K||`` are taken on the *raw* level-0 projections and
    *max-pooled* up the pyramid rather than recomputed on the pooled tensors, so
    a window inherits the salience of its most prominent token. The two terms are
    combined per entry by a max (the joint QK / KQ score) and reduced over heads
    into a single ``(B, S_pyr)`` map.
-   **Selection.** Two index sets are retained. A *mandatory* set, fixed at build
    time, holds every coarsest-level entry plus the leading level-0 entries, and
    together these guarantee at least one contributor at every base position. A
    *discretionary* set of ``top_k`` entries is then chosen by score from the
    remaining candidates. The union is sorted by causal position, restoring
    temporal order.
-   **Sub-attention.** Q, K and V are gathered at the selected indices and a
    single causal scaled dot-product attention runs over the resulting
    sub-sequence. Because the indices are sorted by causal position, an ordinary
    lower-triangular mask over the gathered scores reproduces causality with
    respect to the original sequence.
-   **Scatter-back.** Each entry's output is written to the base positions its
    window covers, offset by a causal shift of ``p^l - 1``. That shift is what
    makes coarse entries safe: a level-``l`` summary contributes only at or after
    the last token it pooled, so no future information leaks backwards.
    Accumulation uses ``keras.ops.segment_sum``, which is deterministic and needs
    no floating-point atomics.

The gathered sub-sequence has the static length
``S = N/p^(L-1) + (p^(L-1) - 1) + top_k``, so attention costs
``O(S^2 * d)`` with an additional ``O(N)`` for pooling, scoring and scatter. The
trade-off is that any interaction not covered by a selected entry is served, if
at all, by a coarse mean rather than an exact token.

A ``full_attention`` flag (set at construction, or toggled at runtime via
``set_full_attention``) bypasses the pyramid entirely and runs plain causal
attention over the full sequence. This exists for two-stage training, in which a
model pretrained with sparse reads is resumed on dense attention.

# DECISION plan_2026-05-17_8babb636/D-001
PORT COMPROMISES (vs. CUDA/Triton reference kernels):
  1. Top-K: ``keras.ops.top_k`` over flat pyramid scores (NOT chunked-bitonic
     stratified). Stratification is replaced by an explicit *mandatory* index
     set, fixed at build time and additive to the ``top_k`` budget: all
     coarsest-level entries (paper Eq. 8) plus level-0 entries
     ``0 .. p^(L-1) - 2``, which are the only entries able to cover the sequence
     prefix. Strictly weaker than stratified selection but hole-free.
  2. Scatter-back: ``keras.ops.segment_sum`` (deterministic, slower) replaces
     fp-atomic-add scatter. No non-determinism trade-off. Costs
     ``O(S * p^(L-1) * d)`` intermediate memory, since segment_sum has no
     broadcast form; ``build()`` warns when that exceeds ``4N``.
  3. Single-device only: NO context parallelism (CP), NO ring attention,
     NO ``enable_load_balance``.
  4. Scorer: ``"norm"`` only — NO ``dilated`` / NO ``gla`` scorers.
  5. Top-K shared across heads (single ``(B, K)`` index set) — NOT per-head. The
     per-head scores are reduced by ``score_head_reduction`` (default ``"mean"``;
     ``"max"`` lets a single outlier head dictate the gather for all heads).
  6. No ``topk % 128`` / ``p`` power-of-2 asserts (CUDA-layout-tied).
  7. Training-only by default; ``set_full_attention(True)`` enables the
     Stage-2 SDPA-resume mode (plain causal MHA over the full sequence).
  8. Selection combines the QK and KQ streams by per-entry ``max`` and takes
     ``top_k`` *distinct entries*. The paper's Eq. 6 takes ``k`` scores from the
     ``2 * S_pyr`` union of both streams, which yields between ``k/2`` and ``k``
     distinct entries. ``top_k`` here is therefore a denser budget than the
     paper's ``k``; throughput figures are not directly comparable.

# DECISION plan_2026-07-26_c41d09b2/D-004
REVISION of D-001 following a line-by-line audit against arXiv:2605.06554v1.
Three defects were corrected and are called out because each failed silently —
the layer trained and converged in every case:
  (a) The scorer read *post*-QK-norm projections. RMSNorm maps every position to
      a near-constant L2 norm, which is precisely the signal the norm scorer
      ranks on, so selection was close to arbitrary. The scorer now reads the raw
      ``W_Q x`` / ``W_K x`` projections, per paper Eq. 4, and ``qk_norm_type``
      defaults to ``None``.
  (b) The coarsest level *consumed* the ``top_k`` budget via a ``+1e9`` score
      boost instead of being additive (paper Eq. 8). At ``L=3, p=4,
      top_k=1536`` and ``N >= 24576`` the entire budget went to coarsest
      entries, so no finer level was ever selected; above ``N = 98304`` not even
      all coarsest entries fitted and ~75% of base positions emitted exact
      zeros. The boost is gone; see D-001 item 1.
  (c) The causal sort key was the window *start* ``i*p^l``. An entry's causal
      timestamp is the *last* token it pooled, ``i*p^l + p^l - 1``, which is
      also where its scatter range begins. Sorting by the start let a coarse
      entry precede a fine entry lying inside its window, and the triangular
      mask then let that fine entry read its own future. The sort and mask now
      use ``_causal_pos``. Ties are benign: two entries with equal timestamp both
      end there, so neither contains the other's future.

References:
    - Long Context Pre-Training with Lighthouse Attention (arXiv:2605.06554v1).
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.utils.logger import logger

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
    """Return a dtype-safe large-negative additive mask value."""
    return _MASK_SENTINEL.get(keras.backend.standardize_dtype(dtype), -1.0e9)


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
    the causal argsort and the triangular mask over the gathered sub-sequence:
    ordering by the window *start* instead lets an entry read a coarse summary
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
    targets = np.full((s_pyr, max_fanout), n, dtype=np.int64)  # sentinel = N
    valid = np.zeros((s_pyr, max_fanout), dtype=bool)

    # Vectorised per level: the original triple loop ran sum_l (N/p^l)*p^l = L*N
    # Python iterations, i.e. minutes of build time at N = 1M.
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
    prefix = np.arange(min(fanout_max - 1, n), dtype=np.int64)  # level-0 block
    return np.union1d(coarsest, prefix).astype(np.int64)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LighthouseAttention(keras.layers.Layer):
    """Lighthouse Attention — coarse-to-fine pyramid + top-K causal SDPA.

    A Keras 3 port of the Lighthouse Attention mechanism. Builds a symmetric
    Q/K/V pyramid of ``num_levels`` levels with mean-pool branching factor
    ``pooling_factor``, scores each pyramid entry with a per-head L2-norm
    scorer over the raw projections (joint QK / KQ max), retains a mandatory
    index set plus the ``top_k`` highest-scoring remaining entries, and runs a
    single causal SDPA on the gathered sub-sequence. Outputs are scattered back
    to base positions with a causal ``p^l - 1`` shift via
    ``keras.ops.segment_sum``.

    Set ``full_attention=True`` (or call ``set_full_attention(True)``
    at runtime) to bypass the pyramid path entirely and run plain causal
    MHA, used for Stage-2 SDPA-resume training.

    **Architecture Overview:**

    Shapes use ``B`` = batch, ``N`` = seq_len, ``H`` = num_heads,
    ``D`` = head_dim, ``L`` = num_levels, ``p`` = pooling_factor,
    ``S_pyr`` = sum_l N/p^l, and ``S`` = N/p^(L-1) + (p^(L-1) - 1) + top_k.

    .. code-block:: text

                           Input: x  (B, N, dim)
                                     │
                                     ▼
            ┌────────────────────────────────────────────────────┐
            │  Wq / Wk / Wv                                      │
            │  Q_raw, K_raw, V              (B, N, H, D)         │
            └──────────┬───────────────────────────┬─────────────┘
                       │  raw Q, K                 │  Q, K, V
                       ▼                           ▼
            ┌──────────────────────┐  ┌──────────────────────────┐
            │  SELECTOR            │  │  TRUNK                   │
            │                      │  │                          │
            │  ||Q_raw||_2  and    │  │  q_norm / k_norm         │
            │  ||K_raw||_2  at     │  │  (optional, and never    │
            │  level 0 only        │  │   before the scorer)     │
            │                      │  │                          │
            │  max-pool up the     │  │  Pyramid pool:           │
            │  pyramid   (Eq. 5)   │  │  mean over p^l,          │
            │                      │  │  l = 0 .. L-1            │
            │  max(s_QK, s_KQ),    │  │                          │
            │  reduce over H       │  │  Q_pyr K_pyr V_pyr       │
            │  ->  s  (B, S_pyr)   │  │  (B, S_pyr, H, D)        │
            │                      │  └────────────┬─────────────┘
            │                      │               │
            │  top_k over the      │               │
            │  NON-mandatory       │               │
            │  candidates only     │               │
            │                      │               │
            │  U  mandatory set    │               │
            │  (all coarsest +     │               │
            │   level-0 prefix)    │               │
            │                      │               │
            │  sort by causal pos  │               │
            │  i p^l + p^l - 1     │               │
            └──────────┬───────────┘               │
                       │  I  (B, S)                │
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │  Gather at I              │
                       │  (B, S, H, D)             │
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │  Causal SDPA over S       │
                       │  triangular mask is exact │
                       │  because I is sorted      │
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │  Scatter back             │
                       │  targets i p^l+p^l-1+k    │
                       │  segment_sum, fan-in <= L │
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │            Wo             │
                       └─────────────┬─────────────┘
                                     ▼
                            Output: (B, N, dim)

    ``full_attention=True`` bypasses everything between the projections and
    ``Wo``, running a single causal SDPA over all ``N`` positions.

    :param dim: Model dimension (hidden size). Must be positive.
    :param num_heads: Number of attention heads. Must be positive and
        divide ``dim`` unless ``head_dim`` is explicitly set.
    :param head_dim: Dimension per head. If ``None``, ``head_dim = dim //
        num_heads``. Defaults to ``None``.
    :param num_levels: Number of pyramid levels (``L``). Defaults to 3.
    :param pooling_factor: Branching factor per level (``p``). Defaults to 4.
    :param top_k: Discretionary pyramid entries selected per batch element, on
        top of the mandatory set. Defaults to 1536. Clipped at build time to the
        number of non-mandatory candidates.
    :param scorer: Scorer type. Only ``"norm"`` supported (port compromise).
    :param score_head_reduction: How the per-head scores collapse to the single
        shared index set, ``"mean"`` or ``"max"``. Defaults to ``"mean"``;
        ``"max"`` lets one outlier head dictate selection for all heads.
    :param full_attention: If ``True``, bypass pyramid path → plain causal
        SDPA over the full sequence. Defaults to ``False``.
    :param qk_norm_type: Norm layer type applied to Q, K *after* scoring, or
        ``None`` to disable. Defaults to ``None``: the paper's scorer ranks
        ``||Q||`` and ``||K||``, and RMSNorm makes both near-constant across
        positions, so normalising before the scorer erases the selection signal
        (DECISION D-004(a)).
    :param qk_norm_kwargs: Optional kwargs forwarded to the norm
        factory. Defaults to ``None``.
    :param probability_type: Score-normalization strategy applied to the
        attention logits via :class:`ProbabilityOutput`. Defaults to
        ``"softmax"``. Routing / hierarchical types are not supported.
    :param probability_config: Optional kwargs forwarded to
        :class:`ProbabilityOutput`. Defaults to ``None``.
    :param use_bias: Use bias in Dense projections. Defaults to ``False``.
    :param kernel_initializer: Initializer for Dense kernels.
        Defaults to ``"glorot_uniform"``.
    :param bias_initializer: Initializer for biases. Defaults to ``"zeros"``.
    :param kernel_regularizer: Optional kernel regularizer.
    :param dropout_rate: Dropout applied to the normalized attention weights in
        both the pyramid and full-attention paths. Defaults to 0.0.

    :raises ValueError: If any argument is invalid.

    .. note::
        **Call-signature contract.** ``call()`` accepts only ``(inputs,
        training=None)`` — there is **no** ``attention_mask`` parameter
        (causality is enforced internally via the causal sort key and the
        scatter-back shift). Additionally, the layer requires a
        **statically-known sequence length**: the pyramid index buffers are
        constructed in ``build()`` from the concrete ``N`` of ``input_shape``.
        If the layer is built with a dynamic / ``None`` sequence dimension,
        ``call()`` raises ``RuntimeError`` ("requires a statically known
        sequence length"). Build with a concrete ``N`` (the common training
        case). A static *batch* dimension is not required but does let
        ``segment_sum`` receive a concrete ``num_segments``, which ``jax.jit``
        needs.
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
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
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
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.dropout_rate = dropout_rate

        # Derived static ints.
        self._p_pow_max: int = pooling_factor ** (num_levels - 1)
        self._scale: float = 1.0 / float(head_dim) ** 0.5

        # ---- sub-layers (built in build()) ----
        proj_units = num_heads * head_dim

        def _dense(units: int, name: str) -> layers.Dense:
            return layers.Dense(
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
            layers.Dropout(dropout_rate, name="attn_dropout")
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
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
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
        live in plain ``ops.*`` tensors created in ``build()``.

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
        self._S_sel = int(mandatory.size) + self._effective_k

        if self._effective_k < self.top_k:
            logger.warning(
                "LighthouseAttention: top_k=%d exceeds the %d non-mandatory "
                "pyramid entries at N=%d; clipping to %d. The gathered "
                "sub-sequence covers the whole pyramid, so this configuration is "
                "dense attention with extra bookkeeping.",
                self.top_k, n_cand, n, self._effective_k,
            )

        # DECISION D-001 item 2: segment_sum has no broadcast form, so the
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
            cached = ops.convert_to_tensor(arr.astype(dtype))
            self._const_cache[name] = cached
        return cached

    # ------------------------------------------------------------------
    # Pyramid pool + norm scorer
    # ------------------------------------------------------------------
    def _pyramid_pool(self, x_heads: keras.KerasTensor) -> keras.KerasTensor:
        """Mean-pool a (B, N, H, D) tensor into a (B, S_pyr, H, D) pyramid.

        Level 0 is an identity copy (N entries). Each successive level ``l``
        reshapes the base sequence into ``(B, N/p^l, p^l, H, D)`` and reduces
        over the window axis with ``ops.mean`` — equivalent to pooling from
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
                reshaped = ops.reshape(x_heads, (-1, n // fanout, fanout, h, d))
                parts.append(ops.mean(reshaped, axis=2))
        return ops.concatenate(parts, axis=1)  # (B, S_pyr, H, D)

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

        s_q0 = ops.norm(q_raw, axis=-1)  # (B, N, H)
        s_k0 = ops.norm(k_raw, axis=-1)  # (B, N, H)

        q_parts: List[keras.KerasTensor] = []
        k_parts: List[keras.KerasTensor] = []
        for l in range(self.num_levels):
            fanout = self.pooling_factor ** l
            if l == 0:
                q_parts.append(s_q0)
                k_parts.append(s_k0)
            else:
                q_resh = ops.reshape(s_q0, (-1, n // fanout, fanout, h))
                k_resh = ops.reshape(s_k0, (-1, n // fanout, fanout, h))
                q_parts.append(ops.max(q_resh, axis=2))
                k_parts.append(ops.max(k_resh, axis=2))
        return (
            ops.concatenate(q_parts, axis=1),  # (B, S_pyr, H)
            ops.concatenate(k_parts, axis=1),  # (B, S_pyr, H)
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    # Top-K is shared across heads (single (B, S) index set, not per-head) —
    # port simplification, DECISION D-001 item 5. The mandatory set is additive
    # to top_k rather than competing with it, and the union is sorted by causal
    # timestamp so that a plain triangular mask over the gathered sub-sequence is
    # exact.
    def _select(
        self,
        s_qk_pyr: keras.KerasTensor,
        s_kq_pyr: keras.KerasTensor,
        batch_size: Any,
    ) -> keras.KerasTensor:
        """Return the (B, S) gathered index set, sorted by causal position."""
        # Per-entry max across the QK / KQ streams, then collapse heads.
        joint = ops.maximum(s_qk_pyr, s_kq_pyr)  # (B, S_pyr, H)
        if self.score_head_reduction == "mean":
            s_shared = ops.mean(joint, axis=-1)  # (B, S_pyr)
        else:
            s_shared = ops.max(joint, axis=-1)

        mand_1d = self._const("mandatory", self._mandatory_indices, "int32")
        selected = ops.broadcast_to(
            mand_1d[None, :], (batch_size, int(self._mandatory_indices.size))
        )

        if self._effective_k > 0:
            cand_1d = self._const("candidates", self._candidate_indices, "int32")
            # Score only the non-mandatory pool, then map local -> pyramid index.
            s_cand = ops.take(s_shared, cand_1d, axis=1)  # (B, S_cand)
            _, local_idx = ops.top_k(s_cand, k=self._effective_k)
            fine = ops.take(cand_1d, local_idx, axis=0)  # (B, k)
            selected = ops.concatenate([selected, fine], axis=1)

        # Sort by causal timestamp (last summarised base position). Ordering by
        # the window *start* instead would let a fine entry read a coarse summary
        # spanning its own future — see DECISION D-004(c).
        causal_t = self._const("causal_pos", self._causal_pos, "int32")
        t_of_selected = ops.take(causal_t, selected, axis=0)  # (B, S)
        order = ops.argsort(t_of_selected, axis=-1)
        return ops.take_along_axis(selected, order, axis=-1)

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

        Written out rather than delegated to ``ops.dot_product_attention`` so
        that ``self.attn_prob`` controls score normalization. Shared by the
        pyramid path (where ``S`` is the gathered sub-sequence, and the mask is
        exact because the gather is sorted by causal timestamp) and the Stage-2
        full-attention path (where ``S = N``).
        """
        # (B, S, H, D) -> (B, H, S, D)
        q_t = ops.transpose(q_heads, (0, 2, 1, 3))
        k_t = ops.transpose(k_heads, (0, 2, 1, 3))
        v_t = ops.transpose(v_heads, (0, 2, 1, 3))

        # DECISION plan_2026-06-14_33b77a7a/D-002: reuse precomputed self._scale (D-002 pattern); ops.cast(self._scale,dt) == 1/ops.sqrt(cast(head_dim,dt)) in float32. Do NOT recompute ops.sqrt per call.
        scale = ops.cast(self._scale, q_t.dtype)
        scores = ops.matmul(q_t, ops.transpose(k_t, (0, 1, 3, 2))) * scale

        # Lower-triangular keep-mask, inclusive of the diagonal.
        i = ops.arange(ops.shape(scores)[-2])
        j = ops.arange(ops.shape(scores)[-1])
        keep = ops.expand_dims(j, 0) <= ops.expand_dims(i, -1)
        scores = ops.where(
            keep, scores, ops.cast(_mask_value(scores.dtype), scores.dtype)
        )

        attn = self.attn_prob(scores)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)
        out_t = ops.matmul(attn, v_t)  # (B, H, S, D)
        return ops.transpose(out_t, (0, 2, 1, 3))  # (B, S, H, D)

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
        q_g = ops.take_along_axis(q_pyr, idx_exp, axis=1)  # (B, S, H, D)
        k_g = ops.take_along_axis(k_pyr, idx_exp, axis=1)
        v_g = ops.take_along_axis(v_pyr, idx_exp, axis=1)
        return self._causal_sdpa(q_g, k_g, v_g, training)

    # ------------------------------------------------------------------
    # Scatter-back
    # ------------------------------------------------------------------
    # Flat segment_sum: encode (batch, target) as a single segment id
    # (`b * (N+1) + target`) so one 1-D segment_sum covers all batches
    # deterministically. Target N is the sentinel for "drop" (invalid /
    # out-of-range positions); the trailing slice removes it. Contributions from
    # different levels sum at shared positions, with fan-in bounded by L
    # (Eq. 11), and are deliberately not normalized by fan-in.
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
        targets_g = ops.take(targets_t, sel_idx, axis=0)  # (B, S, F)
        valid_g = ops.cast(ops.take(valid_t, sel_idx, axis=0), out_g.dtype)

        # Tile output along fanout: (B, S, F, H, D). segment_sum has no broadcast
        # form, so this F-fold expansion is inherent to the deterministic path;
        # build() warns when it dominates.
        out_tiled = ops.repeat(out_g[:, :, None, :, :], fanout, axis=2)
        out_tiled = out_tiled * valid_g[..., None, None]

        out_flat = ops.reshape(out_tiled, (-1, h, d))              # (B*S*F, H, D)
        targets_flat = ops.reshape(targets_g, (-1, s_sel * fanout))  # (B, S*F)

        # Encode (batch, target) into one segment id; N+1 slots per batch with N
        # as the "drop" sentinel.
        batch_offset = ops.arange(batch_size, dtype="int32") * (n + 1)
        flat_segments = ops.reshape(
            ops.cast(targets_flat, "int32") + batch_offset[:, None], (-1,)
        )

        # A static batch lets num_segments be a Python int, which jax.jit needs.
        num_segments = (
            self._B_static * (n + 1)
            if self._B_static is not None
            else batch_size * (n + 1)
        )
        scattered = ops.segment_sum(
            out_flat, flat_segments, num_segments=num_segments
        )  # (B*(N+1), H, D)
        scattered = ops.reshape(scattered, (-1, n + 1, h, d))
        return scattered[:, :n, :, :]  # drop the trailing sentinel row

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
        batch_size = ops.shape(inputs)[0]
        if self._N_static is None:
            raise RuntimeError(
                "LighthouseAttention requires a statically known sequence "
                "length. Build the layer with a concrete N."
            )
        n = self._N_static
        h, d = self.num_heads, self.head_dim

        # Project Q, K, V -> (B, N, H, D). These are the RAW projections.
        q_raw = ops.reshape(self.wq(inputs), (-1, n, h, d))
        k_raw = ops.reshape(self.wk(inputs), (-1, n, h, d))
        v = ops.reshape(self.wv(inputs), (-1, n, h, d))

        # Optional QK-norm, applied only to what enters attention — the scorer
        # below reads q_raw / k_raw (DECISION D-004(a)).
        q = self.q_norm(q_raw) if self.q_norm is not None else q_raw
        k = self.k_norm(k_raw) if self.k_norm is not None else k_raw

        if self.full_attention:
            # Stage-2 SDPA-resume path: plain causal MHA over the full sequence.
            out = self._causal_sdpa(q, k, v, training)  # (B, N, H, D)
            return self.wo(ops.reshape(out, (-1, n, h * d)))

        # Lighthouse pyramid path.
        q_pyr = self._pyramid_pool(q)  # (B, S_pyr, H, D)
        k_pyr = self._pyramid_pool(k)
        v_pyr = self._pyramid_pool(v)
        s_qk_pyr, s_kq_pyr = self._norm_scorer(q_raw, k_raw)
        sel_idx = self._select(s_qk_pyr, s_kq_pyr, batch_size)          # (B, S)
        out_g = self._gather_and_attend(
            q_pyr, k_pyr, v_pyr, sel_idx, training
        )                                                              # (B, S, H, D)
        out_base = self._scatter_back(out_g, sel_idx, batch_size, n)    # (B, N, H, D)
        return self.wo(ops.reshape(out_base, (-1, n, h * d)))

# ---------------------------------------------------------------------