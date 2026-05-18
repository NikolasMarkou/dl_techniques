"""
Lighthouse Attention Layer — Keras 3 port.

This module provides a Keras 3 / TF 2.18 implementation of the Lighthouse
Attention mechanism (arXiv:2605.06554v1). Lighthouse builds a coarse-to-fine
*pyramid* of mean-pooled Q/K/V representations across ``num_levels`` levels
with branching factor ``pooling_factor``, scores every pyramid entry via a
per-head L2-norm scorer (QK ⊕ KQ, joint max), selects the top-K entries with
``keras.ops.top_k`` (always retaining the coarsest level via a +1e9 score
boost), runs a single causal scaled-dot-product attention on the gathered
sub-sequence, and scatters the result back to base positions with a causal
``p^l - 1`` shift via ``keras.ops.segment_sum`` (deterministic, atomic-free).
A ``set_full_attention(bool)`` mutator bypasses the pyramid path and runs
plain causal SDPA for two-stage SDPA-resume training.

# DECISION plan_2026-05-17_8babb636/D-001
PORT COMPROMISES (vs. CUDA/Triton reference kernels):
  1. Top-K: ``keras.ops.top_k`` over flat pyramid scores (NOT chunked-bitonic
     stratified). Stratified guarantees are replaced by "always-keep coarsest
     level" via a +1e9 score boost — finer-level holes fall back to the
     coarsest contribution only. Strictly weaker but correct.
  2. Scatter-back: ``keras.ops.segment_sum`` (deterministic, slower) replaces
     fp-atomic-add scatter. No non-determinism trade-off.
  3. Single-device only: NO context parallelism (CP), NO ring attention,
     NO ``enable_load_balance``.
  4. Scorer: ``"norm"`` only — NO ``dilated`` / NO ``gla`` scorers.
  5. Top-K shared across heads (single ``(B, K)`` index set) — NOT per-head.
  6. No ``topk % 128`` / ``p`` power-of-2 asserts (CUDA-layout-tied).
  7. Training-only by default; ``set_full_attention(True)`` enables the
     Stage-2 SDPA-resume mode (plain causal MHA over the full sequence).

References:
    - Lighthouse Attention (arXiv:2605.06554v1).
"""

import keras
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Module-level static pyramid helpers (pure Python / numpy).
# These are called once in build() — they depend only on (N, L, p).
# ---------------------------------------------------------------------


def _compute_level_sizes(n: int, num_levels: int, pooling_factor: int) -> np.ndarray:
    """Return the per-level entry count ``N / p^l`` for ``l = 0..L-1``."""
    sizes = np.array(
        [n // (pooling_factor ** l) for l in range(num_levels)],
        dtype=np.int64,
    )
    return sizes


def _compute_base_starts(n: int, num_levels: int, pooling_factor: int) -> np.ndarray:
    """Return base-window-start positions for every pyramid entry, flat order.

    For each level ``l`` with ``N/p^l`` entries, the m-th entry covers the
    window ``[m * p^l, (m+1) * p^l)``. The returned int array is shape
    ``(S_pyr,)`` with ``S_pyr = sum_l N/p^l``.
    """
    parts: List[np.ndarray] = []
    for l in range(num_levels):
        fanout = pooling_factor ** l
        n_l = n // fanout
        parts.append(np.arange(n_l, dtype=np.int64) * fanout)
    return np.concatenate(parts, axis=0)


def _compute_level_ids(n: int, num_levels: int, pooling_factor: int) -> np.ndarray:
    """Return per-entry level-id for every pyramid entry, flat order."""
    parts: List[np.ndarray] = []
    for l in range(num_levels):
        n_l = n // (pooling_factor ** l)
        parts.append(np.full(n_l, l, dtype=np.int64))
    return np.concatenate(parts, axis=0)


# DECISION plan_2026-05-17_8babb636/D-005
def _compute_scatter_targets(
    n: int, num_levels: int, pooling_factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scatter target positions + validity mask, causally shifted.

    For each pyramid entry ``m`` at level ``l`` with base start ``b``, its
    output is scattered to base positions ``b + (p^l - 1) + k`` for
    ``k = 0..p^l - 1`` clipped to ``[0, N)``. The ``p^l - 1`` shift is the
    *causal* shift — it ensures that the entry's contribution lands strictly
    at-or-after the last input position it summarises (no future leakage).

    Returns ``(targets, valid_mask)`` shapes ``(S_pyr, MAX_FANOUT)``.
    Invalid positions (``k >= p^l`` or target ``>= N``) carry target ``N``
    (sentinel — segment_sum drops the trailing row) and ``valid_mask=False``.
    """
    max_fanout = pooling_factor ** (num_levels - 1)
    s_pyr = int(_compute_level_sizes(n, num_levels, pooling_factor).sum())
    targets = np.full((s_pyr, max_fanout), n, dtype=np.int64)  # sentinel = N
    valid = np.zeros((s_pyr, max_fanout), dtype=bool)

    offset = 0
    for l in range(num_levels):
        fanout = pooling_factor ** l
        n_l = n // fanout
        shift = fanout - 1  # p^l - 1
        for m in range(n_l):
            base_start = m * fanout
            for k in range(fanout):
                tgt = base_start + shift + k
                if 0 <= tgt < n:
                    targets[offset + m, k] = tgt
                    valid[offset + m, k] = True
        offset += n_l
    return targets, valid


def _compute_coarsest_indices(
    n: int, num_levels: int, pooling_factor: int
) -> np.ndarray:
    """Return flat-pyramid indices of all entries at the coarsest level."""
    sizes = _compute_level_sizes(n, num_levels, pooling_factor)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    coarsest_l = num_levels - 1
    start = int(offsets[coarsest_l])
    stop = int(offsets[coarsest_l + 1])
    return np.arange(start, stop, dtype=np.int64)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LighthouseAttention(keras.layers.Layer):
    """Lighthouse Attention — coarse-to-fine pyramid + top-K causal SDPA.

    A Keras 3 port of the Lighthouse Attention mechanism. Builds a symmetric
    Q/K/V pyramid of ``num_levels`` levels with mean-pool branching factor
    ``pooling_factor``, scores each pyramid entry with a per-head L2-norm
    scorer (joint QK ⊕ KQ max), selects the top-``top_k`` entries (always
    keeping the coarsest level), and runs a single causal SDPA on the
    gathered sub-sequence. Outputs are scattered back to base positions with
    a causal ``p^l - 1`` shift via ``keras.ops.segment_sum``.

    Set ``full_attention=True`` (or call ``set_full_attention(True)``
    at runtime) to bypass the pyramid path entirely and run plain causal
    MHA, used for Stage-2 SDPA-resume training.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────────────────────────┐
        │                  LighthouseAttention                          │
        │                                                               │
        │   Input [B, N, dim]                                           │
        │          │                                                    │
        │          ▼                                                    │
        │   ┌─────────────────────────────────────────────────────┐     │
        │   │  Wq / Wk / Wv ──► (B, N, H, D)                      │     │
        │   │       │                                             │     │
        │   │       ▼                                             │     │
        │   │  q_norm(Q),  k_norm(K)         [QK-norm]            │     │
        │   └─────────────────────────────────────────────────────┘     │
        │          │                                                    │
        │          ├──────────── full_attention=True ──────────┐        │
        │          │                                           │        │
        │          ▼                                           ▼        │
        │   ┌──────────────────────────────────┐    ┌───────────────┐   │
        │   │     Pyramid path (default)       │    │ Stage-2 SDPA  │   │
        │   │                                  │    │ resume:       │   │
        │   │  (i) Mean-pool Q/K/V across L    │    │ causal SDPA   │   │
        │   │      levels with branch p:       │    │ over full N   │   │
        │   │       (B, S_pyr, H, D)           │    │               │   │
        │   │                                  │    └───────┬───────┘   │
        │   │  (ii) Norm scorer on level-0     │            │           │
        │   │       ||Q||, ||K||  max-pooled   │            │           │
        │   │       up the pyramid             │            │           │
        │   │       s = max(||Q||, ||K||)      │            │           │
        │   │                                  │            │           │
        │   │  (iii) Reduce-over-heads → (B,S) │            │           │
        │   │       + coarsest-level boost     │            │           │
        │   │         (always-keep, +1e9)      │            │           │
        │   │                                  │            │           │
        │   │  (iv) top_k → (B, K)             │            │           │
        │   │       argsort by base position   │            │           │
        │   │       (causal ordering)          │            │           │
        │   │                                  │            │           │
        │   │  (v) Gather (Q,K,V)_pyr at K     │            │           │
        │   │      → (B, K, H, D)              │            │           │
        │   │      causal SDPA on sub-seq      │            │           │
        │   │                                  │            │           │
        │   │  (vi) Scatter-back via           │            │           │
        │   │       segment_sum to (B,N,H,D)   │            │           │
        │   │       causal shift  p^l - 1      │            │           │
        │   └──────────────────┬───────────────┘            │           │
        │                      │                            │           │
        │                      └────────────┬───────────────┘           │
        │                                   ▼                           │
        │                          Wo  ──► Output                       │
        │                                                               │
        │   Output [B, N, dim]                                          │
        └───────────────────────────────────────────────────────────────┘

    :param dim: Model dimension (hidden size). Must be positive.
    :param num_heads: Number of attention heads. Must be positive and
        divide ``dim`` unless ``head_dim`` is explicitly set.
    :param head_dim: Dimension per head. If ``None``, ``head_dim = dim //
        num_heads``. Defaults to ``None``.
    :param num_levels: Number of pyramid levels (``L``). Defaults to 3.
    :param pooling_factor: Branching factor per level (``p``). Defaults to 4.
    :param top_k: Maximum pyramid entries selected per batch element.
        Defaults to 1536. Clipped at call time to ``min(top_k, S_pyr)``.
    :param scorer: Scorer type. Only ``"norm"`` supported (port compromise).
    :param full_attention: If ``True``, bypass pyramid path → plain causal
        SDPA over the full sequence. Defaults to ``False``.
    :param normalization_type: Norm layer type for Q, K projections (QK-norm
        convention). Defaults to ``"rms_norm"``.
    :param normalization_kwargs: Optional kwargs forwarded to the norm
        factory. Defaults to ``None``.
    :param use_bias: Use bias in Dense projections. Defaults to ``False``.
    :param kernel_initializer: Initializer for Dense kernels.
        Defaults to ``"glorot_uniform"``.
    :param bias_initializer: Initializer for biases. Defaults to ``"zeros"``.
    :param kernel_regularizer: Optional kernel regularizer.
    :param dropout_rate: Dropout applied to attention scores in the SDPA
        path. Currently informational — ``keras.ops.dot_product_attention``
        does not accept dropout in 3.8, so this is reserved for forward
        compatibility. Defaults to 0.0.

    :raises ValueError: If any argument is invalid.
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
        full_attention: bool = False,
        normalization_type: str = "rms_norm",
        normalization_kwargs: Optional[Dict[str, Any]] = None,
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
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

        # ---- store config ----
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        self.top_k = top_k
        self.scorer = scorer
        self.full_attention = bool(full_attention)
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.dropout_rate = dropout_rate

        # Derived static int (used for N divisibility check).
        self._p_pow_max: int = pooling_factor ** (num_levels - 1)
        self._scale: float = 1.0 / float(head_dim) ** 0.5

        # ---- sub-layers (built in build()) ----
        proj_units = num_heads * head_dim
        norm_kwargs = dict(normalization_kwargs or {})

        self.wq = layers.Dense(
            proj_units,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="wq",
        )
        self.wk = layers.Dense(
            proj_units,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="wk",
        )
        self.wv = layers.Dense(
            proj_units,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="wv",
        )
        self.wo = layers.Dense(
            dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="wo",
        )
        self.q_norm = create_normalization_layer(
            normalization_type, name="q_norm", **norm_kwargs
        )
        self.k_norm = create_normalization_layer(
            normalization_type, name="k_norm", **norm_kwargs
        )

        # Numpy buffers populated in build().
        self._level_sizes: Optional[np.ndarray] = None
        self._level_offsets: Optional[np.ndarray] = None
        self._base_starts: Optional[np.ndarray] = None
        self._scatter_targets: Optional[np.ndarray] = None
        self._scatter_valid_mask: Optional[np.ndarray] = None
        self._coarsest_indices: Optional[np.ndarray] = None
        self._S_pyr: Optional[int] = None
        self._N_static: Optional[int] = None
        self._max_fanout: int = self._p_pow_max

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
    # Serialization (Step 1 — full round-trip; build() populated in Step 2)
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
                "full_attention": self.full_attention,
                "normalization_type": self.normalization_type,
                "normalization_kwargs": self.normalization_kwargs,
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

        Numpy buffers (level sizes, base-window-starts, scatter targets,
        coarsest indices) are stored on ``self`` as plain numpy arrays —
        they are pure functions of ``(N, num_levels, pooling_factor)`` and
        are re-derived in a fresh ``build()`` after ``from_config()``
        restoration. See LESSONS: frozen tensor state must NOT live in
        plain ``ops.*`` tensors created in ``build()``.

        :param input_shape: ``(B, N, dim)``.
        :raises ValueError: 2D input or static ``N`` not divisible by
            ``pooling_factor ** (num_levels - 1)``.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape (batch, seq_len, dim), got {input_shape}"
            )
        n_static = input_shape[1]

        if n_static is not None:
            if n_static % self._p_pow_max != 0:
                raise ValueError(
                    f"seq_len N={n_static} must be divisible by "
                    f"pooling_factor ** (num_levels - 1) = {self._p_pow_max}. "
                    f"Either pad inputs or adjust (num_levels, pooling_factor)."
                )
            self._populate_pyramid_buffers(int(n_static))
        # else: deferred to call-time once N is concrete.

        # Build sub-layers explicitly (mandatory for .keras save/load).
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        # q_norm / k_norm apply per-head (last axis = head_dim).
        head_shape = (input_shape[0], input_shape[1], self.num_heads, self.head_dim)
        self.q_norm.build(head_shape)
        self.k_norm.build(head_shape)
        # Output projection consumes (B, N, H*D).
        self.wo.build((input_shape[0], input_shape[1], self.num_heads * self.head_dim))

        super().build(input_shape)

    def _populate_pyramid_buffers(self, n: int) -> None:
        """Compute and store static numpy buffers for a given N.

        Idempotent — safe to recompute if ``N`` changes between calls (though
        Keras layers are typically built once with a fixed N).
        """
        L, p = self.num_levels, self.pooling_factor
        sizes = _compute_level_sizes(n, L, p)
        self._level_sizes = sizes
        self._level_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self._base_starts = _compute_base_starts(n, L, p)
        targets, valid = _compute_scatter_targets(n, L, p)
        self._scatter_targets = targets
        self._scatter_valid_mask = valid
        self._coarsest_indices = _compute_coarsest_indices(n, L, p)
        self._S_pyr = int(sizes.sum())
        self._N_static = n

    # ------------------------------------------------------------------
    # Pyramid pool + norm scorer (private helpers — Step 2)
    # ------------------------------------------------------------------
    def _pyramid_pool(self, x_heads: keras.KerasTensor) -> keras.KerasTensor:
        """Mean-pool a (B, N, H, D) tensor into a (B, S_pyr, H, D) pyramid.

        Level 0 = identity copy (N entries). Each successive level l reshapes
        the base sequence into ``(B, N/p^l, p^l, H, D)`` and reduces over
        the window axis with ``ops.mean``. All levels are concatenated along
        the sequence axis in coarse-order from level 0 -> L-1.
        """
        B = ops.shape(x_heads)[0]
        N = ops.shape(x_heads)[1]
        H = self.num_heads
        D = self.head_dim
        parts: List[keras.KerasTensor] = []
        for l in range(self.num_levels):
            fanout = self.pooling_factor ** l
            if l == 0:
                parts.append(x_heads)
            else:
                # (B, N, H, D) -> (B, N/p^l, p^l, H, D) -> mean over axis=2
                reshaped = ops.reshape(x_heads, (B, N // fanout, fanout, H, D))
                pooled = ops.mean(reshaped, axis=2)
                parts.append(pooled)
        return ops.concatenate(parts, axis=1)  # (B, S_pyr, H, D)

    def _norm_scorer(
        self,
        q_heads: keras.KerasTensor,
        k_heads: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Per-head L2-norm scorer with max-pool over coarser levels.

        Paper §3 stage (ii): norms are computed once on level-0 projections
        and *max-pooled* up the pyramid — NOT recomputed on mean-pooled
        projections. Returns ``(s_qk_pyr, s_kq_pyr)`` each shape
        ``(B, S_pyr, H)``.

        For Lighthouse, ``s_qk = ||Q||`` and ``s_kq = ||K||`` are the two
        terms whose per-entry max gives the joint scorer.
        """
        # Level-0 norms over the head_dim axis.
        s_q0 = ops.norm(q_heads, axis=-1)  # (B, N, H)
        s_k0 = ops.norm(k_heads, axis=-1)  # (B, N, H)

        B = ops.shape(q_heads)[0]
        N = ops.shape(q_heads)[1]
        H = self.num_heads

        q_parts: List[keras.KerasTensor] = []
        k_parts: List[keras.KerasTensor] = []
        for l in range(self.num_levels):
            fanout = self.pooling_factor ** l
            if l == 0:
                q_parts.append(s_q0)
                k_parts.append(s_k0)
            else:
                q_resh = ops.reshape(s_q0, (B, N // fanout, fanout, H))
                k_resh = ops.reshape(s_k0, (B, N // fanout, fanout, H))
                q_parts.append(ops.max(q_resh, axis=2))
                k_parts.append(ops.max(k_resh, axis=2))
        s_qk_pyr = ops.concatenate(q_parts, axis=1)  # (B, S_pyr, H)
        s_kq_pyr = ops.concatenate(k_parts, axis=1)  # (B, S_pyr, H)
        return s_qk_pyr, s_kq_pyr

    # ------------------------------------------------------------------
    # Step 3 — selection + sub-attention + scatter-back
    # ------------------------------------------------------------------
    # DECISION plan_2026-05-17_8babb636/D-002
    # Top-K is shared across heads (single (B, K) index set, not per-head)
    # — port simplification. Coarsest-level always retained via a +1e9 score
    # boost (DECISION D-003). Indices are sorted by base position before
    # SDPA so that `is_causal=True` on the gathered sub-sequence preserves
    # the original temporal ordering.
    def _select_topk(
        self,
        s_qk_pyr: keras.KerasTensor,
        s_kq_pyr: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Pick top-K pyramid entries per batch element, causally sorted."""
        # Per-entry max across QK / KQ, then reduce over heads -> (B, S_pyr).
        joint = ops.maximum(s_qk_pyr, s_kq_pyr)  # (B, S_pyr, H)
        s_shared = ops.max(joint, axis=-1)       # (B, S_pyr)

        # DECISION plan_2026-05-17_8babb636/D-003
        # Always-keep coarsest-level entries via +1e9 score boost — replaces
        # the paper's chunked-bitonic stratified guarantee. Finer-level holes
        # fall back to the (always-present) coarsest contribution only.
        coarsest_boost_np = np.zeros((self._S_pyr,), dtype=np.float32)
        coarsest_boost_np[self._coarsest_indices] = 1.0e9
        coarsest_boost = ops.convert_to_tensor(coarsest_boost_np)
        coarsest_boost = ops.cast(coarsest_boost, s_shared.dtype)
        s_shared = s_shared + coarsest_boost[None, :]  # broadcast over batch

        effective_k = min(self.top_k, self._S_pyr)
        _, top_idx = ops.top_k(s_shared, k=effective_k)  # (B, K) int32

        # Sort selected indices by base position for causal SDPA ordering.
        base_starts_t = ops.convert_to_tensor(self._base_starts.astype(np.int32))
        base_pos_of_selected = ops.take(base_starts_t, top_idx, axis=0)  # (B, K)
        order = ops.argsort(base_pos_of_selected, axis=-1)
        top_idx = ops.take_along_axis(top_idx, order, axis=-1)
        return top_idx

    def _gather_and_attend(
        self,
        q_pyr: keras.KerasTensor,
        k_pyr: keras.KerasTensor,
        v_pyr: keras.KerasTensor,
        top_idx: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Gather pyramid entries at ``top_idx`` and run causal SDPA."""
        # top_idx: (B, K). Expand to (B, K, 1, 1) for gather along axis=1.
        idx_exp = top_idx[:, :, None, None]
        q_g = ops.take_along_axis(q_pyr, idx_exp, axis=1)  # (B, K, H, D)
        k_g = ops.take_along_axis(k_pyr, idx_exp, axis=1)
        v_g = ops.take_along_axis(v_pyr, idx_exp, axis=1)
        # Causal SDPA on the (sorted) sub-sequence.
        out_g = ops.dot_product_attention(q_g, k_g, v_g, is_causal=True)
        return out_g  # (B, K, H, D)

    # DECISION plan_2026-05-17_8babb636/D-004
    # Scatter-back via flat segment_sum: encode (batch, target) as a single
    # segment id (`b * (N+1) + target`) so a single 1-D segment_sum covers
    # all batches deterministically. Target N is the sentinel for "drop"
    # (invalid / out-of-range positions); the trailing slice removes it.
    def _scatter_back(
        self,
        out_g: keras.KerasTensor,
        top_idx: keras.KerasTensor,
        batch_size: keras.KerasTensor,
        n: int,
    ) -> keras.KerasTensor:
        """Scatter (B, K, H, D) sub-attention output back to (B, N, H, D)."""
        K_eff = ops.shape(top_idx)[1]
        H, D = self.num_heads, self.head_dim
        max_fanout = self._max_fanout

        # Gather scatter targets + validity for the selected entries.
        targets_t = ops.convert_to_tensor(self._scatter_targets.astype(np.int32))
        valid_t = ops.convert_to_tensor(self._scatter_valid_mask.astype("float32"))
        targets_g = ops.take(targets_t, top_idx, axis=0)   # (B, K, F)
        valid_g = ops.take(valid_t, top_idx, axis=0)       # (B, K, F)

        # Tile output along fanout: (B, K, F, H, D)
        out_tiled = ops.repeat(out_g[:, :, None, :, :], max_fanout, axis=2)
        out_tiled = out_tiled * valid_g[..., None, None]

        # Flatten (K, F) -> K*F per batch.
        out_flat = ops.reshape(out_tiled, (batch_size, K_eff * max_fanout, H, D))
        targets_flat = ops.reshape(targets_g, (batch_size, K_eff * max_fanout))

        # Encode (batch, target) into a single int segment id using sentinel
        # row index (N+1 slots per batch); N is the sentinel for "drop".
        batch_offset = ops.arange(batch_size, dtype="int32") * (n + 1)
        flat_segments = (
            ops.cast(targets_flat, "int32") + batch_offset[:, None]
        )  # (B, K*F)
        flat_segments_1d = ops.reshape(flat_segments, (-1,))                # (B*K*F,)
        flat_updates = ops.reshape(out_flat, (-1, H, D))                     # (B*K*F, H, D)

        num_segments = batch_size * (n + 1)
        scattered = ops.segment_sum(
            flat_updates, flat_segments_1d, num_segments=num_segments
        )  # (B*(N+1), H, D)
        scattered = ops.reshape(scattered, (batch_size, n + 1, H, D))
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
        :param training: Unused at the moment — reserved for forward-compat
            dropout support inside ``ops.dot_product_attention``.
        :return: ``(B, N, dim)``.
        """
        del training  # not used in 3.8 SDPA
        batch_size = ops.shape(inputs)[0]
        # Resolve N: prefer static, fall back to dynamic.
        if self._N_static is not None:
            n = self._N_static
        else:
            # Dynamic-N path: deferred buffer population.
            # Practically all training configs are static-N; raise for now
            # since dynamic-N pyramid construction is build-time-only.
            raise RuntimeError(
                "LighthouseAttention requires a statically known sequence "
                "length. Build the layer with a concrete N."
            )

        H, D = self.num_heads, self.head_dim

        # Project Q, K, V -> (B, N, H, D)
        q = ops.reshape(self.wq(inputs), (batch_size, n, H, D))
        k = ops.reshape(self.wk(inputs), (batch_size, n, H, D))
        v = ops.reshape(self.wv(inputs), (batch_size, n, H, D))

        # QK-norm: apply pre-SDPA norm to Q, K heads.
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.full_attention:
            # Stage-2 SDPA-resume path: plain causal MHA on the full seq.
            out = ops.dot_product_attention(q, k, v, is_causal=True)
            out = ops.reshape(out, (batch_size, n, H * D))
            return self.wo(out)

        # Lighthouse pyramid path.
        q_pyr = self._pyramid_pool(q)  # (B, S_pyr, H, D)
        k_pyr = self._pyramid_pool(k)
        v_pyr = self._pyramid_pool(v)
        s_qk_pyr, s_kq_pyr = self._norm_scorer(q, k)
        top_idx = self._select_topk(s_qk_pyr, s_kq_pyr)               # (B, K)
        out_g = self._gather_and_attend(q_pyr, k_pyr, v_pyr, top_idx)  # (B, K, H, D)
        out_base = self._scatter_back(out_g, top_idx, batch_size, n)   # (B, N, H, D)
        out_flat = ops.reshape(out_base, (batch_size, n, H * D))
        return self.wo(out_flat)

# ---------------------------------------------------------------------
