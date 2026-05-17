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
    MHA — used for Stage-2 SDPA-resume training.

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

# ---------------------------------------------------------------------
