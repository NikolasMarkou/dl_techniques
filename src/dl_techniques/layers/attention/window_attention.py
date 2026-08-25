"""
Unified windowed multi-head self-attention for sequence processing.

This module provides a highly configurable windowed multi-head self-attention
layer. It unifies three distinct partitioning strategies—standard grid-based
windowing, frequency-proximate zigzag windowing and a one-dimensional
symmetric band over the raw token sequence—into a single interface,
controlled by the `partition_mode` parameter.

The layer takes a 1D sequence and computes self-attention within a local
neighbourhood, offering different locality biases. ``'grid'`` and ``'zigzag'``
reshape the sequence into a 2-D grid and partition it into tiles; ``'band'``
does neither.

**Read the cost model before reaching for this layer as an "efficient
attention", and read it PER MODE — the three modes do not share one.** With
``M = window_size ** 2``:

* ``'band'`` — a dense ``N x N`` banded mask over standard attention,
  ``O(N^2)``, the same order as plain global attention. It never pads and has
  no ``window_size ** 4`` floor. It is also NOT the ``O(N * window_size)`` that
  "sliding window" is usually taken to mean; a fused banded kernel is not
  reachable from ``keras.ops``.
* ``'grid'`` — ``O(N * M)`` for ``N > M``, and ``O(N^2)`` for ``N <= M``, where
  there is exactly one window and the layer attends over the ``N`` REAL tokens
  with the relative-position bias gathered at their grid coordinates.
* ``'zigzag'`` — ``O(max(N, M) * M)``, INCLUDING a constant ``O(M^2) =
  O(window_size ** 4)`` floor for ``N <= M``, where the padded grid is a single
  window and the layer computes dense attention over ``M`` padded positions:
  ``(M / N) ** 2`` times *more* than global attention over the ``N`` real
  tokens. At ``window_size = 128`` (``M = 16384``) that threshold sits above
  any sequence most callers will ever pass.

**``'grid'`` and ``'zigzag'`` had the SAME degenerate cost until 2026-08-25**
(``plan-2026-08-25T053412-0f1fa04f``, D-001/D-002). Both partition modes had
their pad LEAK fixed (D-007/D-009/D-011: pad slots were entering the softmax, so
the ragged regime was returning a WRONG answer, not merely an expensive one),
and both now carry the ``N < M`` short-circuit -- ``'grid'`` first, ``'zigzag'``
in step 7.1 (D-014), which is the change that removed the last inversion.
MEASURED on ``(1, 128, 64)`` at ``window_size=128``, CPU, peak RSS: ``'grid'``
0.680 GB, ``'band'`` 0.679 GB, ``'multi_head'`` 0.674 GB, ``'zigzag'``
0.678 GB. Before the two short-circuits those read 21.695 GB and 17.503 GB
respectively. NO partition mode of this layer costs more than plain global
attention at any ``N``. See "Foundational Mathematics" below.

Architecture:
    One pipeline, one branch. Both partition modes share the same five-stage
    shape, and differ only in stage 3 — which tokens end up in a window
    together:

    1.  **Grid formation.** The 1D sequence ``(B, N, dim)`` is padded up to
        ``H*W`` and reshaped to a 2D grid ``(B, H, W, dim)``. A 1D sequence is
        used as the public interface (rather than a 2D map) so the layer drops
        into sequence models; the 2D grid is an internal device for defining
        locality.
    2.  **Window padding.** The grid is padded so both spatial extents are
        divisible by ``window_size``. All padding introduced in stages 1-2 is
        stripped again in stage 5, so the layer is shape-preserving end to end.
    3.  **Partitioning — the only real difference between the modes.**
        ``'grid'`` takes contiguous ``window_size x window_size`` tiles (the
        Swin convention: spatially adjacent tokens attend together).
        ``'zigzag'`` instead groups tokens that are proximate along a zigzag
        traversal of the grid, giving a frequency-proximate locality bias — the
        useful choice when the sequence's neighbours in *index* space matter
        more than its neighbours in the synthetic 2D layout.
    4.  **Per-window attention.** Each window is handed to
        :class:`SingleWindowAttention`, which owns the QKV projection, optional
        QK-normalization, relative position bias and probability output. This
        layer contributes no attention math of its own — it is a partitioning
        wrapper, which is why the two modes can share it.
    5.  **Reverse and unpad.** Windows are merged back to the grid, the grid is
        flattened back to a sequence, and every token added by stages 1-2 is
        dropped.

    The two ASCII flows below trace stages 1-5 concretely for each mode.

Foundational Mathematics:
    Full self-attention over ``N`` tokens costs ``O(N^2)``. Restricting
    attention to non-overlapping windows of ``M = window_size ** 2`` tokens
    leaves ``ceil(N / M)`` independent ``O(M^2)`` problems. The ``ceil`` is the
    whole story and is *not* a rounding detail:

    ``_call_grid`` pads ``N`` up to ``H * W`` with ``H = W = ceil(sqrt(N))``,
    then pads ``H`` and ``W`` up to a multiple of ``window_size``. The window
    count is therefore ``ceil(H / window_size) ** 2``, which has a **floor of
    1** — never zero windows, and never a window holding fewer than ``M``
    positions, because the shortfall is made up with padding. Total cost:

        ``cost(N, M) = ceil(H / window_size)^2 * M^2  =  O(max(N, M) * M)``

    Two regimes follow, and only one of them is the advertised one:

    * ``N > M`` — the intended regime. ``O(N * M)``, linear in ``N`` for fixed
      ``M``, cheaper than ``O(N^2)`` global attention by a factor ``N / M``.
    * ``N <= M`` — the degenerate regime. Geometrically the situation is the
      same in both grid modes: ``H <= window_size``, the padded grid is exactly
      one ``window_size x window_size`` tile, and ``_window_partition`` yields a
      single window holding ``M`` positions of which ``M - N`` are padding.
      Computing dense attention over all ``M`` of them costs the constant
      ``M^2`` regardless of ``N``, i.e. ``(M / N)^2`` times *more* than global
      attention over the ``N`` real tokens — at ``window_size = 128`` that is
      every ``N <= 16384``, with a ``16384 x 16384`` score matrix per head per
      sample whether ``N`` is 128 or 8192.

      **Both grid modes SHORT-CIRCUIT that regime** — ``'grid'`` since
      2026-08-25, ``'zigzag'`` since later the same day (step 7.1). One window
      means the mathematically correct answer is dense attention over the ``N``
      REAL tokens, with the relative-position bias gathered at the slots the
      layout gives them (their grid coordinates under ``'grid'``, their position
      in the scan under ``'zigzag'``), which is what both now compute:
      ``O(N^2)``, never worse than plain global attention. The result is bitwise
      identical to the old code wherever the old code was correct, and the ragged
      cases where it was not are now correct too -- the pad slots used to enter
      the softmax, so an all-ones attention mask (a mathematical no-op) moved the
      output by up to 0.980964.

      MEASURED on ``(1, 128, 64)`` at ``window_size=128``, CPU peak RSS via the
      registry keys: ``'window'`` 0.680 GB, ``'window_band'`` 0.679,
      ``'multi_head'`` 0.674 and ``'window_zigzag'`` 0.678 -- all four at parity.
      Before the two short-circuits the same two window keys read 21.695 GB and
      17.503 GB.

    Choosing ``window_size`` is therefore choosing a *minimum* cost, never a
    maximum one, in neither mode any more: a "generous" window used to make the
    layer strictly more expensive than the global attention it replaces at every
    sequence length that fits inside one window, and now merely stops buying you
    anything.

    The other price is that information cannot cross a window boundary within
    one layer; both modes address this the same way a Swin stack does, by
    relying on the *caller* to alternate partitions (or shift them) between
    layers rather than by widening any single window.

================================================
Partition Mode 1: 'grid' (Swin Transformer-style)
================================================

Complete Architecture Flow (`partition_mode='grid'`)::

    ┌─────────────────────────────────────────────────────────────┐
    │ WindowAttention — partition_mode='grid' (Swin-style tiles)  │
    │                                                             │
    │  INPUT: 1-D sequence  [B, N, dim]                           │
    │                      ▼                                      │
    │  1. grid formation — pad N up to N_grid = H*W and reshape   │
    │     to a 2-D grid.  H = W = ceil(sqrt(N)).                  │
    │     grid [B, H, W, dim]                                     │
    │                      ▼                                      │
    │  2. window padding — pad H and W up to a multiple of        │
    │     window_size.   padded grid [B, H_pad, W_pad, dim]       │
    │                      ▼                                      │
    │  3. window partition — contiguous ws x ws tiles, so         │
    │     spatially adjacent tokens attend together.              │
    │     windows [B*num_win, ws^2, dim]                          │
    │                      ▼                                      │
    │  4. SingleWindowAttention on every window.  ALL of the      │
    │     attention math (QKV, QK-norm, relative position         │
    │     bias, ProbabilityOutput) lives there — this layer       │
    │     is a partitioning wrapper and owns none of it.          │
    │                      ▼                                      │
    │  5. window reverse — merge the tiles back to the grid       │
    │                      ▼                                      │
    │  6. unpad — slice [:H, :W], reshape to [B, N_grid, dim],    │
    │     slice to N                                              │
    │                      ▼                                      │
    │  OUTPUT: 1-D sequence  [B, N, dim]                          │
    │                                                             │
    │  an optional RANK-2 (B, N) key mask takes the identical      │
    │  pad ► reshape ► pad ► partition path and rides into each    │
    │  window with its tokens.  A RANK-3 (B, ws^2, ws^2) PAIRWISE  │
    │  mask is already in window coordinates and is forwarded      │
    │  verbatim to stage 4 — only valid when N == ws^2 (one        │
    │  window), which is how SwinTransformerBlock calls this       │
    │  layer; any other N raises.                                  │
    └─────────────────────────────────────────────────────────────┘

================================================
Partition Mode 2: 'zigzag' (Frequency Locality)
================================================

Complete Architecture Flow (`partition_mode='zigzag'`)::

    ┌─────────────────────────────────────────────────────────────┐
    │ WindowAttention — partition_mode='zigzag' (freq. locality)  │
    │                                                             │
    │  INPUT: 1-D sequence  [B, N, dim]                           │
    │                      ▼                                      │
    │  1. grid formation — pad N up to N_grid = H*W.  This        │
    │     path stays 1-D: there is no reshape to a 2-D map.       │
    │                      ▼                                      │
    │  2. zigzag reorder — take(x, zigzag_indices, axis=1),       │
    │     grouping tokens that are proximate along a zigzag       │
    │     traversal of the notional grid.                         │
    │                      ▼                                      │
    │  3. window partition — pad to a multiple of the window      │
    │     length ws^2, then reshape.                              │
    │     windows [B*num_win, ws^2, dim]                          │
    │                      ▼                                      │
    │  4. SingleWindowAttention on every window (relative         │
    │     position bias is usually disabled on this path).        │
    │                      ▼                                      │
    │  5. merge windows, slice back to N_grid                     │
    │                      ▼                                      │
    │  6. inverse zigzag — take(x, inverse_zigzag_indices),       │
    │     then slice to N                                         │
    │                      ▼                                      │
    │  OUTPUT: 1-D sequence  [B, N, dim]                          │
    │                                                             │
    │  an optional mask is padded, reordered and partitioned      │
    │  the same way and rides into each window with its           │
    │  tokens.                                                    │
    └─────────────────────────────────────────────────────────────┘

Complexity Analysis
-------------------

``M = window_size²`` is the number of token slots in one window; ``N`` is the
sequence length. Windows are padded to ``M`` positions, so ``M`` — not ``N`` —
sets the floor.

============== ================== ==============================================
Operation      Complexity         Notes
============== ================== ==============================================
Partitioning   O(max(N, M))       Reshape/reorder over the PADDED grid
Attention      O(max(N, M) × M)   ceil(N/M) windows, each an M×M score matrix
Total          O(max(N, M) × M)   Linear in N only for N > M
--             --                 --
N > M          O(N × M)           The intended regime; beats O(N²) by N/M
N < M          O(N²)              ONE window, attended over the N REAL tokens
                                  (short-circuit, 2026-08-25 -- 'grid' first,
                                  'zigzag' in step 7.1). Never worse than
                                  plain global attention, in EITHER mode.
N == M         O(M²) = O(N²)      ONE window with nothing to pad; the ordinary
                                  partition path, and how Swin calls this
                                  layer. Identical cost, kept bitwise.
============== ================== ==============================================

References:
    - Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision
      Transformer using Shifted Windows". ICCV.
      (https://arxiv.org/abs/2103.14030)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

# ---------------------------------------------------------------------

import math
import numpy as np
import keras
from typing import Any, Dict, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .single_window_attention import SingleWindowAttention
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

# ---------------------------------------------------------------------
# Probability types not supported in window attention (score-level routing
# is structurally incompatible with windowed score tensors).
# ---------------------------------------------------------------------

_VALID_PARTITION_MODES = ("grid", "zigzag", "band")

_DISALLOWED_PROB_TYPES = (
    "routing",
    "deterministic_routing",
    "hierarchical",
    "hierarchical_routing",
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WindowAttention(keras.layers.Layer):
    """
    Unified window-based multi-head self-attention layer.

    Partitions a 1-D token sequence into local windows and computes
    multi-head self-attention within each window via
    ``SingleWindowAttention``. Two partitioning strategies are supported:
    ``'grid'`` (Swin Transformer-style 2D spatial windowing) and
    ``'zigzag'`` (2D zigzag scan grouping frequency-proximate tokens), and
    ``'band'`` (a 1-D SYMMETRIC band over the token sequence — ``window_size``
    is a HALF-WIDTH IN TOKENS, query ``i`` attends key ``j`` iff
    ``abs(i - j) <= window_size``, with no grid folding and no square padding).
    All padding, reshaping, partitioning, and merging are handled
    internally.

    **Cost, per mode — they do not share one.** With ``M = window_size ** 2``:

    * ``'band'`` never pads: a dense ``N x N`` banded mask over standard
      attention, ``O(N^2)``, the same order as plain global attention. NOT the
      ``O(N * window_size)`` a "sliding window" is usually taken to mean.
    * ``'grid'``: ``O(N * M)`` for ``N > M``; ``O(N^2)`` for ``N <= M``, where a
      short-circuit (2026-08-25) attends over the ``N`` REAL tokens instead of
      ``M`` padded slots.
    * ``'zigzag'``: ``O(max(N, M) * M)``, with NO ``O(M^2)`` floor since step
      7.1 — for ``N < M`` the zigzag layout is also exactly one window (it folds
      the sequence into a ``ceil(sqrt(N))``-square grid, so ``N_grid <= M``), and
      that case is short-circuited to ``O(N^2)`` over the ``N`` real tokens with
      the relative-position bias gathered at each token's position in the SCAN.

    MEASURED 2026-08-25 on ``(1, 128, 64)`` at ``window_size=128``, CPU, peak
    RSS: ``'grid'`` 0.680 GB, ``'band'`` 0.679 GB, ``'multi_head'`` 0.674 GB,
    ``'zigzag'`` 0.678 GB — four-way parity. Before the two short-circuits the
    two window modes read 21.695 GB and 17.503 GB. See the module docstring's "Foundational
    Mathematics" section. The guard that used to pin the degeneracy at
    ``ModernBERT``'s shipped ``window_size = 128``
    (``tests/test_models/test_modern_bert/test_shipped_window_size.py``) was
    DELETED on 2026-08-25, per its own docstring's instruction, because
    ModernBERT no longer uses this layout at all -- its local layers are
    ``'band'``.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │                     WindowAttention                     │
        │                                                         │
        │  Input: 1-D sequence  (B, N, dim)                       │
        │                     ▼                                   │
        │  pad N ► reshape to a 2-D grid  /  zigzag reorder       │
        │                     ▼                                   │
        │  partition into non-overlapping windows                 │
        │  (B*num_win, ws^2, dim)                                 │
        │                     ▼                                   │
        │  SingleWindowAttention per window                       │
        │                     ▼                                   │
        │  merge windows ► unpad / inverse zigzag                 │
        │                     ▼                                   │
        │  Output: 1-D sequence  (B, N, dim)                      │
        └─────────────────────────────────────────────────────────┘

    :param dim: Dimension of the input tokens (channels).
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param partition_mode: The partitioning strategy. One of `'grid'`,
        `'zigzag'` or `'band'`. Default: 'grid'.

        * ``'grid'`` — Swin-style 2-D tiles. ``window_size`` is an EDGE
          LENGTH: the sequence is folded into a ``ceil(sqrt(N))``-square grid
          and cut into ``window_size x window_size`` tiles.
        * ``'zigzag'`` — the same tiles over a zigzag reordering of the grid.
        * ``'band'`` — a 1-D SYMMETRIC band over the token sequence, with NO
          grid folding and NO square padding. ``window_size`` is a HALF-WIDTH
          IN TOKENS: query ``i`` attends key ``j`` iff
          ``abs(i - j) <= window_size``. Non-causal (both directions), which
          is what a text ENCODER such as ModernBERT specifies — upstream's
          ``local_attention`` is a full span, and its half-width is
          ``local_attention // 2``. Requires
          ``use_relative_position_bias=False``; see that parameter.
    :type partition_mode: Literal["grid", "zigzag", "band"]
    :param attention_mode: The type of attention projection in each window. One
        of `'linear'` or `'kan_key'`. Default: 'linear'.
    :type attention_mode: Literal["linear", "kan_key"]
    :param probability_type: Identifier for the attention probability
        distribution produced from raw scores. Forwarded to
        :class:`ProbabilityOutput`. Common values include ``'softmax'``,
        ``'adaptive'`` (a.k.a. ``'adaptive_softmax'``), ``'sparsemax'``, and
        ``'threshmax'``. Score-level routing variants
        (``'routing'`` / ``'deterministic_routing'`` /
        ``'hierarchical'`` / ``'hierarchical_routing'``) are not supported by
        windowed attention and will raise ``ValueError``. Default: 'softmax'.
    :type probability_type: str
    :param probability_config: Optional config dict forwarded to
        :class:`ProbabilityOutput` (e.g. adaptive softmax temperature
        parameters). Default: None.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to Q and K prior
        to the attention dot product. If ``None``, no Q/K normalization is
        applied. See ``dl_techniques.layers.norms.factory`` for valid types.
        Default: None.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional kwargs forwarded to
        :func:`create_normalization_layer` for the Q/K normalization layers.
        Default: None.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param use_relative_position_bias: If True, add a learnable relative
        position bias to the attention scores. Recommended for `'grid'` mode.
        For `'zigzag'` mode, this is often set to `False` as the spatial
        relationship is already altered. For `'band'` mode it is REFUSED, not
        silently disabled: the bias is indexed by 2-D coordinates inside a
        ``window_size x window_size`` tile and a 1-D band has no tile, so
        ``partition_mode='band'`` with ``use_relative_position_bias=True``
        raises ``ValueError``. Use ``create_band_window_attention`` (or the
        ``'window_band'`` factory key), which defaults it to ``False``.
        Default: True.
    :type use_relative_position_bias: bool
    :param qkv_bias: If True, add a learnable bias to the QKV projection.
        Only used when `attention_mode` is `'linear'`. Default: True.
    :type qkv_bias: bool
    :param qk_scale: Override for query-key scaling factor. Defaults to
        `head_dim ** -0.5`.
    :type qk_scale: Optional[float]
    :param dropout_rate: Dropout rate for attention scores. Default: 0.0.
    :type dropout_rate: float
    :param proj_bias: If True, add a learnable bias to the output projection.
        Default: True.
    :type proj_bias: bool
    :param kan_grid_size: Grid size for the KAN layer. Only used when
        `attention_mode` is `'kan_key'`. Default: 5.
    :type kan_grid_size: int
    :param kan_spline_order: Spline order for the KAN layer. Only used when
        `attention_mode` is `'kan_key'`. Default: 3.
    :type kan_spline_order: int
    :param kan_activation: Activation for the KAN layer. Only used when
        `attention_mode` is `'kan_key'`. Default: 'swish'.
    :type kan_activation: str
    :param kernel_initializer: Initializer for kernel weights.
        Default: 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights. Default: 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for kernel weights. Default: None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for bias weights. Default: None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Other keyword arguments for base Layer.
    :type kwargs: Any
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        partition_mode: Literal["grid", "zigzag", "band"] = "grid",
        attention_mode: Literal["linear", "kan_key"] = "linear",
        use_relative_position_bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout_rate: float = 0.0,
        proj_bias: bool = True,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_activation: str = "swish",
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[
            str, keras.initializers.Initializer
        ] = "zeros",
        kernel_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        bias_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if partition_mode not in _VALID_PARTITION_MODES:
            raise ValueError(
                f"partition_mode must be one of "
                f"{list(_VALID_PARTITION_MODES)}; got {partition_mode!r}."
            )

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-010
        # The relative-position bias is REFUSED under `partition_mode='band'`,
        # not quietly turned off.
        #
        # WHY: the bias is a 2-D GRID concept. `SingleWindowAttention` gathers
        # it through an index that maps a tile slot to
        # `(slot // window_size, slot % window_size)`. A 1-D band has no tile
        # and no such coordinate, so every row gathered would be arbitrary --
        # and an arbitrary learnable bias passes every shape, dtype and
        # finiteness check there is.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT mirror `create_zigzag_window_attention` by
        #     `setdefault`-ing it to False on the CLASS. A wrapper default is
        #     overridable and `WindowAttention(..., partition_mode='band',
        #     use_relative_position_bias=True)` would then return a layer whose
        #     `get_config()` says `True` while the maths says otherwise -- the
        #     "silently gets a bias that means nothing" outcome this check
        #     exists to prevent. The DEFAULT-off lives on the wrapper
        #     (`create_band_window_attention`), exactly like zigzag; the REFUSAL
        #     lives here.
        #   * Do NOT relax this to a warning. This repo escalates warnings to
        #     errors in pytest but not at runtime, so a warning would be silent
        #     in production and loud only in the one place it is not needed.
        # See decisions.md D-009 (plan-2026-08-25T053412-0f1fa04f).
        if partition_mode == "band":
            if use_relative_position_bias:
                raise ValueError(
                    "WindowAttention(partition_mode='band') requires "
                    "use_relative_position_bias=False. The relative position "
                    "bias is indexed by 2-D coordinates inside a "
                    "window_size x window_size tile; a 1-D band has no tile, "
                    "so the gathered rows would be arbitrary. Pass "
                    "use_relative_position_bias=False, or use "
                    "create_band_window_attention / the 'window_band' factory "
                    "key, which default it to False."
                )
            if window_size < 0:
                raise ValueError(
                    f"WindowAttention(partition_mode='band') requires "
                    f"window_size >= 0 (it is a HALF-WIDTH in tokens, not an "
                    f"edge length); got {window_size}."
                )

        # Validate probability_type: score-level routing variants are not
        # supported because window partitioning fragments the score tensor.
        if probability_type in _DISALLOWED_PROB_TYPES:
            raise ValueError(
                f"probability_type='{probability_type}' is not supported for "
                "WindowAttention: score-level routing is incompatible with "
                "windowed score tensors. Use 'softmax', 'adaptive', "
                "'sparsemax', or 'threshmax'."
            )

        # Store all parameters for get_config()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.partition_mode = partition_mode
        self.attention_mode = attention_mode
        self.use_relative_position_bias = use_relative_position_bias
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout_rate = dropout_rate
        self.proj_bias = proj_bias
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_activation = deserialize_activation(kan_activation)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
        # Resolve to objects so the stored attributes round-trip cleanly via
        # get_config (the child SingleWindowAttention also resolves these).
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        # placeholder
        self._call_internal = None

        # Create the core attention layer that operates on a single window.
        # ``WindowAttention`` is a partitioning wrapper: the actual Q/K
        # projection, score computation, probability distribution
        # (``attn_prob``) and Q/K normalization all live in
        # ``SingleWindowAttention``. We forward the canonical Group-C
        # parameters straight through.
        self.attention = SingleWindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attention_mode=attention_mode,
            use_relative_position_bias=use_relative_position_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout_rate=dropout_rate,
            proj_bias=proj_bias,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            kan_activation=kan_activation,
            probability_type=probability_type,
            probability_config=probability_config,
            qk_norm_type=qk_norm_type,
            qk_norm_kwargs=qk_norm_kwargs,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            # DECISION plan-2026-08-19T163559-499b6f0e/D-081: the name is
            # EXPLICIT, and it is spelled as the auto-name Keras would have
            # produced for the first instance in a process. Without it the
            # global `auto_name` counter numbers this sub-layer per process, so
            # two instances of the SAME builder disagree by weight path
            # (`single_window_attention` vs `single_window_attention_12`) and
            # R-072 build parity cannot be asserted for any consumer.
            #
            # MEASURED consequence, and it is not free: a model holding MORE
            # than one window-attention layer had them numbered
            # (`single_window_attention_1`, `_12`, `_19`); they now all read
            # `single_window_attention` and are distinguished by their parent,
            # so the weight PATHS move for `swin_transformer`, `scunet` and
            # `modern_bert`. A `.keras` archive is positional and round-trips
            # unaffected -- CPU-eager forward delta measured EXACTLY 0.0 for all
            # three -- but a by-NAME load against a checkpoint written before
            # this line will not find these tensors.
            name="single_window_attention",
        )

        # Attributes for zigzag mode, computed in build()
        if self.partition_mode == "zigzag":
            self.H = None
            self.W = None
            self.N_grid = None
            self.pad_len_seq = None
            self.zigzag_indices = None
            self.inverse_zigzag_indices = None

    # DECISION plan-2026-08-25T053412-0f1fa04f/D-014
    # This is NUMPY, not ``keras.ops``, and the layout it describes therefore
    # has exactly ONE representation that both the tensor pipeline and the
    # relative-position bias can read.
    #
    # WHAT NOT TO DO, and why:
    #   * Do NOT put it back on ``keras.ops``. ``build()`` can run inside a
    #     ``tf.function`` trace, so the returned tensor would be owned by THAT
    #     trace and unusable from any other one -- the same trap D-006 records
    #     for the relative-position index. Worse for step 7.1: the degenerate
    #     short-circuit needs the INVERSE permutation as a Python-visible
    #     numpy array (``SingleWindowAttention.set_window_slots`` takes numpy by
    #     construction, because the map selects bias-table ROWS at trace time --
    #     and since 2026-08-25 it is deliberately NOT a call keyword either; see
    #     D-015), and ``convert_to_numpy`` on a graph tensor raises.
    #   * Do NOT compute a SECOND, hand-written zigzag permutation next to the
    #     short-circuit instead of reusing this one. Two copies of a layout is
    #     the "kept in lockstep" shape this repo treats as a defect, and a
    #     wrong permutation is INVISIBLE whenever the vector it permutes is all
    #     ones (measured, step 4.1 injection (d)).
    #   * Do NOT worry about ``argsort`` tie-breaking differing from
    #     ``keras.ops.argsort``: ``combined_key = s * H + secondary`` is
    #     INJECTIVE (for a fixed anti-diagonal ``s`` the secondary key is a
    #     bijection of the row index, and the per-``s`` ranges
    #     ``[s*H, s*H + H - 1]`` are disjoint), so the sort has no ties to
    #     break and both spellings return the identical permutation.
    # See decisions.md D-014 (plan-2026-08-25T053412-0f1fa04f).

    @staticmethod
    def _generate_zigzag_indices(H: int, W: int) -> np.ndarray:
        """Generate zigzag scan indices for an ``H x W`` grid.

        ``result[p]`` is the row-major position, in the ``H x W`` grid, of the
        ``p``-th token of the zigzag scan -- i.e. exactly the gather index
        :meth:`_call_zigzag` feeds to ``keras.ops.take``.

        :param H: Grid height.
        :type H: int
        :param W: Grid width.
        :type W: int
        :return: 1-D int32 index array of length ``H * W``.
        :rtype: np.ndarray
        """
        r_grid, c_grid = np.meshgrid(
            np.arange(H, dtype=np.int32),
            np.arange(W, dtype=np.int32),
            indexing="ij",
        )
        r_flat = r_grid.reshape(-1)
        c_flat = c_grid.reshape(-1)

        s = r_flat + c_flat
        secondary_key = np.where(s % 2 == 1, r_flat, H - 1 - r_flat)
        combined_key = s * H + secondary_key

        return np.argsort(combined_key).astype(np.int32)

    def _attend(
        self,
        inputs: keras.KerasTensor,
        *,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        window_slots: Optional[np.ndarray] = None,
        pad_to_window: bool = True,
    ) -> keras.KerasTensor:
        """Invoke the inner :class:`SingleWindowAttention` with a slot map.

        Interface contract -- this is the ONLY place in this class that calls
        ``self.attention``, and every partition path goes through it:

        * ``window_slots`` is a concrete numpy array (the static layout) or
          ``None``. It is handed over by ``set_window_slots()``, NOT as a call
          keyword, and is cleared in a ``finally`` so it can never survive into
          the next call. See the D-015 anchor in
          ``SingleWindowAttention.__init__`` for why the keyword channel is
          unusable.
        * ``attention_mask`` / ``training`` / ``pad_to_window`` are forwarded
          verbatim; ``attention_mask`` is a tensor and belongs on the traced
          channel, ``pad_to_window`` is a Python bool that Keras leaves alone.
        * Failure mode: any exception from the inner layer propagates unchanged,
          with the slot map already cleared.

        :param inputs: Tokens to attend, ``(B', N', dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Mask forwarded to the inner layer, or ``None``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training-mode flag.
        :type training: Optional[bool]
        :param window_slots: Static ``(N',)`` slot map, or ``None`` for the
            pad-to-``window_size ** 2`` behaviour.
        :type window_slots: Optional[np.ndarray]
        :param pad_to_window: ``False`` only for ``partition_mode='band'``.
        :type pad_to_window: bool
        :return: Attended tokens, ``(B', N', dim)``.
        :rtype: keras.KerasTensor
        """
        self.attention.set_window_slots(window_slots)
        try:
            return self.attention(
                inputs,
                attention_mask=attention_mask,
                training=training,
                pad_to_window=pad_to_window,
            )
        finally:
            self.attention.set_window_slots(None)

    @staticmethod
    def _single_window_slots(seq_len: int, window_size: int) -> np.ndarray:
        """Map each token of a degenerate one-window sequence to its tile slot.

        The grid path lays a length-``seq_len`` sequence out row-major into a
        ``H x W`` grid with ``H = W = ceil(sqrt(seq_len))``, pads that grid up to
        ``window_size x window_size`` and flattens it row-major, so token ``i``
        lands at tile slot ``(i // W) * window_size + (i % W)``. That mapping --
        NOT the identity, whenever ``W < window_size`` -- is what the
        relative-position bias must be gathered at.

        :param seq_len: Number of real tokens, ``< window_size ** 2``.
        :type seq_len: int
        :param window_size: Side length of the square window, in tokens.
        :type window_size: int
        :return: int32 array of shape ``(seq_len,)`` of row-major tile slots.
        :rtype: np.ndarray
        """
        grid_side = int(math.ceil(math.sqrt(seq_len)))
        tokens = np.arange(seq_len, dtype=np.int32)
        return (
                (tokens // grid_side) * window_size + (tokens % grid_side)
        ).astype(np.int32)

    @staticmethod
    def _pads_exist(seq_len: Optional[int], window_size: int) -> bool:
        """Does the grid path create pad slots for this sequence length?

        The grid path lays ``seq_len`` tokens row-major into an ``S x S`` grid with
        ``S = ceil(sqrt(seq_len))`` and then pads that grid up to a whole number of
        ``window_size x window_size`` tiles. Two independent pads can appear:

        * a SEQUENCE pad of ``S ** 2 - seq_len`` slots, nonzero unless ``seq_len``
          is a perfect square;
        * a TILE pad of ``(window_size - S % window_size) % window_size`` rows and
          the same number of columns, nonzero unless ``S`` is a multiple of
          ``window_size``.

        :param seq_len: Statically-known sequence length, or ``None``.
        :type seq_len: Optional[int]
        :param window_size: Tile side length, in tokens.
        :type window_size: int
        :return: ``True`` if either pad is nonzero, or if ``seq_len`` is unknown --
            an unknown length is treated as padded, because masking pads that do not
            exist is a no-op while failing to mask pads that do exist is the D-011
            leak.
        :rtype: bool
        """
        if seq_len is None:
            return True
        seq_len = int(seq_len)
        grid_side = int(math.ceil(math.sqrt(seq_len)))
        return (
            grid_side * grid_side != seq_len
            or grid_side % window_size != 0
        )

    def _window_partition(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Partition a 4-D grid tensor into non-overlapping windows.

        :param x: Tensor of shape ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :return: Windows of shape ``(B*num_windows, ws, ws, C)``.
        :rtype: keras.KerasTensor
        """
        B, H, W, C = keras.ops.shape(x)
        ws = self.window_size
        x = keras.ops.reshape(x, (B, H // ws, ws, W // ws, ws, C))
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        windows = keras.ops.reshape(x, (-1, ws, ws, C))
        return windows

    def _window_reverse(
        self, windows: keras.KerasTensor, H: int, W: int
    ) -> keras.KerasTensor:
        """Merge windows back into a 4-D grid tensor.

        :param windows: Window tensor of shape ``(B*num_windows, ws, ws, C)``.
        :type windows: keras.KerasTensor
        :param H: Padded grid height.
        :type H: int
        :param W: Padded grid width.
        :type W: int
        :return: Reconstructed grid of shape ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        ws = self.window_size
        num_windows_h = H // ws
        num_windows_w = W // ws
        num_windows_total = keras.ops.shape(windows)[0]
        B = num_windows_total // (num_windows_h * num_windows_w)
        x = keras.ops.reshape(
            windows, (B, num_windows_h, num_windows_w, ws, ws, -1)
        )
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = keras.ops.reshape(x, (B, H, W, -1))
        return x

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and precompute zigzag indices if needed.

        :param input_shape: Shape tuple ``(batch, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        if self.partition_mode == "zigzag":
            N_actual = input_shape[1]
            if N_actual is None:
                raise ValueError(
                    "WindowAttention with partition_mode='zigzag' requires a "
                    "fixed sequence length for XLA compatibility."
                )

            self.H = int(math.ceil(math.sqrt(N_actual)))
            self.W = self.H
            self.N_grid = self.H * self.W
            self.pad_len_seq = self.N_grid - N_actual

            self.zigzag_indices = self._generate_zigzag_indices(self.H, self.W)
            # `inverse_zigzag_indices[i]` is the ZIGZAG POSITION of grid slot `i`.
            # It is both the scatter-back gather of `_call_zigzag` and, restricted
            # to the first `N_actual` entries, the `window_slots` vector the
            # degenerate short-circuit hands to `SingleWindowAttention` -- one
            # array, two readers, no second copy of the layout.
            self.inverse_zigzag_indices = np.argsort(
                self.zigzag_indices
            ).astype(np.int32)

        self.attention.build(
            (None, self.window_size * self.window_size, self.dim)
        )

        if self.partition_mode == "grid":
            self._call_internal = self._call_grid
        elif self.partition_mode == "zigzag":
            self._call_internal = self._call_zigzag
        elif self.partition_mode == "band":
            self._call_internal = self._call_band
        else:
            # Should not be reachable due to __init__ validation
            raise RuntimeError(f"Invalid partition mode: {self.partition_mode}")
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        return self._call_internal(inputs, attention_mask, training)

    def _call_grid(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass using grid-based spatial partitioning.

        :param inputs: Sequence tensor ``(B, N, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional keep predicate (``1 = attend``), either
            rank-2 ``(B, N)`` (a key mask over the UNPARTITIONED sequence, which
            is re-partitioned here alongside the data) or rank-3
            ``(B, ws**2, ws**2)`` (a PAIRWISE mask already expressed in
            partitioned window coordinates, forwarded verbatim — see the
            ``ValueError`` below).
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :raises ValueError: If ``attention_mask`` is rank-3 while ``N`` is not
            statically equal to ``window_size ** 2``. A rank-3 mask is in
            already-partitioned coordinates, so it is only meaningful for a
            degenerate one-window grid.
        :return: Output tensor ``(B, N, dim)``.
        :rtype: keras.KerasTensor
        """
        ws = self.window_size

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-007
        # When `N < window_size ** 2` the grid below degenerates to EXACTLY ONE
        # window, and the pad-then-partition path attends `window_size ** 2`
        # slots to answer a question about `N` tokens. MEASURED at
        # `N=128, window_size=128`: `H_pad * W_pad / N == 128.0` -- 128 real
        # tokens inflated to 16,384 slots, a `(1, 4, 16384, 16384)` float32 score
        # matrix (4.0 GiB) with 3-4 live copies through clip / mask / softmax, for
        # a peak of 17.69 GB against plain `multi_head`'s 0.678 GB on the same
        # `(1, 128, 64)` input. So this branch attends the N REAL tokens instead,
        # telling `SingleWindowAttention` where each of them sits in the tile so
        # the relative-position bias is gathered at the SAME table rows.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT widen this to `N <= window_size ** 2`. At `N == window_size
        #     ** 2` there is nothing to skip (the slots are `arange(N)` and the
        #     sub-index IS the full index), so it buys nothing -- and that is the
        #     exact `N` at which `SwinTransformerBlock` calls this layer with a
        #     rank-3 pairwise mask in already-partitioned coordinates. Leaving
        #     that case on the original path keeps the D-001 verbatim-mask
        #     contract untouched and keeps Swin bit-for-bit unchanged.
        #   * Do NOT take this branch when the sequence length is not statically
        #     known: the slot vector is a numpy constant derived from `N`.
        #   * Do NOT drop the `N > 1` guard. At `N == 1` the short-circuit is
        #     mathematically right -- one token attends to itself, softmax over a
        #     length-1 axis is exactly 1.0 -- but `keras.ops.softmax` emits a
        #     `UserWarning` for a size-1 reduction axis, and this repo's pytest
        #     config escalates warnings to errors, so
        #     `test_arbitrary_shapes_and_windows[{3,4,8}-1-grid]` fail on the
        #     WARNING, not on any value. `N == 1` therefore stays on the padding
        #     path, where the softmax axis is `window_size ** 2`. It is the one
        #     length at which this layer still inflates, and it is cheap for the
        #     window sizes that reach it.
        #   * Do NOT "simplify" by calling plain dense attention here. That drops
        #     the relative-position bias, which is indexed by GRID COORDINATE --
        #     token `i` sits at grid `(i // W, i % W)`, i.e. tile slot
        #     `(i // W) * ws + (i % W)`, which is NOT `i` whenever `W < ws`.
        #
        # ACCEPTED COST -- and it is a VALUE change, not just a cost change.
        # On the old path the `window_size ** 2 - N` zero-filled pad slots were
        # never masked (the internal padding mask sees `N_actual == N_target ==
        # window_size ** 2` and is all ones), so they contributed keys and values
        # to every real token's softmax. MEASURED on the pre-fix code, no-mask
        # versus an explicit all-ones `(B, N)` mask (which DOES mask the pads):
        # max |delta| 0.350777 at `ws=4, N=9`, 0.0903779 at `ws=8, N=50`, and
        # exactly 0.0 at `ws=8, N=64` (`N == ws**2`, nothing to pad). Removing
        # those slots is the fix, and it necessarily moves the ragged-`N` output.
        # See decisions.md D-007 (plan-2026-08-25T053412-0f1fa04f).
        static_n = inputs.shape[1]
        degenerate = (
            static_n is not None
            and 1 < int(static_n) < ws * ws
            and (attention_mask is None or len(attention_mask.shape) == 2)
        )
        if degenerate:
            return self._attend(
                inputs,
                attention_mask=attention_mask,
                training=training,
                window_slots=self._single_window_slots(int(static_n), ws),
            )

        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]

        H = W = keras.ops.cast(
            keras.ops.ceil(
                keras.ops.sqrt(keras.ops.cast(N_actual, "float32"))
            ),
            "int32",
        )
        N_grid = H * W
        pad_amount_seq = keras.ops.maximum(0, N_grid - N_actual)

        x = keras.ops.pad(inputs, [[0, 0], [0, pad_amount_seq], [0, 0]])
        x = keras.ops.reshape(x, (B, H, W, C))

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = keras.ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        H_pad, W_pad = H + pad_h, W + pad_w

        windows = self._window_partition(x)
        windows = keras.ops.reshape(windows, (-1, ws * ws, C))

        window_mask = None
        key_mask = None
        if attention_mask is not None and len(attention_mask.shape) == 3:
            # DECISION plan-2026-07-31T042809-ddc92265/D-001
            # A rank-3 mask is already expressed in PARTITIONED WINDOW
            # coordinates — `(num_windows, ws**2, ws**2)`, the layout
            # `SwinTransformerBlock` builds — so it is forwarded VERBATIM and
            # must NOT go through the re-partition path below.
            #
            # WHAT NOT TO DO, and why:
            #   * Do NOT run it through the rank-2 pipeline (pad -> reshape to
            #     (B,H,W) -> window_partition). That pipeline reads its input as
            #     an UNPARTITIONED (B, N) image mask; feeding it a pairwise mask
            #     re-partitions the KEY axis as if it were spatial, which is a
            #     silent wrong-geometry mask, not an error.
            #   * Do NOT skip the guard below and "let it work when it works".
            #     The verbatim forward is only meaningful when this layer's own
            #     internal grid is DEGENERATE (exactly one window per batch
            #     element), because only then do the caller's window index and
            #     this layer's batch index coincide. On any other `N` the mask
            #     would land on the wrong windows silently.
            # See decisions.md D-001 (plan-2026-07-31T042809-ddc92265).
            static_n = inputs.shape[1]
            if static_n is None or int(static_n) != ws * ws:
                raise ValueError(
                    f"WindowAttention(partition_mode='grid') received a rank-3 "
                    f"pairwise attention_mask of shape "
                    f"{tuple(attention_mask.shape)}, but the sequence length is "
                    f"{static_n} rather than window_size**2 = {ws * ws}. A rank-3 "
                    f"mask is expressed in already-partitioned window coordinates, "
                    f"so it is only meaningful when this layer's internal grid is "
                    f"degenerate (one window per batch element) — which is how "
                    f"`SwinTransformerBlock` calls it, after partitioning "
                    f"externally. Partition your input into windows first, or pass "
                    f"a rank-2 (B, N) key mask instead."
                )
            window_mask = attention_mask
        else:
            # DECISION plan-2026-08-25T053412-0f1fa04f/D-011
            # `attention_mask=None` is NOT "no mask" here -- it is "every token is
            # real", and the pad slots this method just created are not tokens. When
            # the caller passes nothing, an all-ones key mask is SYNTHESIZED so the
            # sequence pad (`ceil(sqrt(N))**2 - N` slots) and the tile pad
            # (`pad_h`/`pad_w` rows and columns) travel down the SAME zero-padding
            # pipeline the caller's own mask would, and are excluded from the
            # softmax. Before this, `None` skipped the pipeline entirely and the
            # zero-filled pads entered every real token's softmax as ordinary keys
            # and values.
            #
            # MEASURED on 8435dcc2f, max |delta| between `attention_mask=None` and
            # an explicit all-ones `(B, N)` mask -- a mask that masks no REAL token
            # and is therefore a mathematical no-op:
            #     ws=8, N=100 -> 1.0258900   (grid side 10 padded to 16: the pads
            #                                 outnumber the real tokens in the last
            #                                 tile, so the softmax is dominated by
            #                                 zeros)
            #     ws=4, N=20  -> 0.7214095
            #     ws=2, N=15  -> 0.2340506
            # and exactly 0.0 wherever there is nothing to pad. See D-011.
            #
            # WHAT NOT TO DO, and why:
            #   * Do NOT synthesize the mask UNCONDITIONALLY. When the geometry
            #     tiles exactly there are no pads, the synthesized mask is all ones
            #     through to `apply_attention_mask`, and it is a no-op -- but a
            #     no-op that allocates a `(B, ws**2)` int32 tensor per window on the
            #     path Swin, FastVLM and TiRex take, for nothing. The `_pads_exist`
            #     guard keeps `N == ws**2` and every exactly-tiling `N` on the
            #     byte-identical original path, which is what keeps the 12 strict
            #     bitwise cells of
            #     `test_window_attention_restructure_is_inert.py` bitwise.
            #   * Do NOT "fix" this inside `SingleWindowAttention` instead. By the
            #     time the windows reach it, `N_actual == N_target == ws**2` and its
            #     own padding mask is all ones: the pads are indistinguishable from
            #     real tokens down there. The geometry is only known HERE.
            #   * Do NOT build the ones with `keras.ops.ones((B, N_actual), ...)`.
            #     `B` and `N_actual` come from `keras.ops.shape`, so they are TENSORS
            #     under a `tf.function` trace; `ones_like` on a rank-2 slice of the
            #     input carries the dynamic shape without materializing it as a
            #     Python value.
            key_mask = attention_mask
            if key_mask is None and self._pads_exist(inputs.shape[1], ws):
                key_mask = keras.ops.ones_like(inputs[:, :, 0], dtype="int32")
        if window_mask is None and key_mask is not None:
            mask = keras.ops.pad(key_mask, [[0, 0], [0, pad_amount_seq]])
            mask = keras.ops.reshape(mask, (B, H, W))
            mask = keras.ops.pad(mask, [[0, 0], [0, pad_h], [0, pad_w]])
            mask = keras.ops.expand_dims(mask, axis=-1)
            mask_windows = self._window_partition(mask)
            window_mask = keras.ops.reshape(mask_windows, (-1, ws * ws))

        attn_windows = self._attend(
            windows, attention_mask=window_mask, training=training
        )
        attn_windows = keras.ops.reshape(attn_windows, (-1, ws, ws, C))
        x = self._window_reverse(attn_windows, H_pad, W_pad)

        x = x[:, :H, :W, :]
        x = keras.ops.reshape(x, (B, N_grid, C))
        x = x[:, :N_actual, :]
        return x

    def _call_band(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Forward pass using a 1-D symmetric band over the token sequence.

        Query ``i`` attends key ``j`` iff ``abs(i - j) <= window_size``. There
        is no grid, no tile and no square padding: the layer runs standard
        attention over the ``N`` real tokens and supplies the band as a
        pairwise ``(1, N, N)`` keep predicate.

        :param inputs: Sequence tensor ``(B, N, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional keep predicate (``1 = attend``), either
            rank-2 ``(B, N)`` (a key mask) or rank-3 ``(B, N, N)`` (pairwise).
            It is COMPOSED with the band, never substituted for it.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(B, N, dim)``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-25T053412-0f1fa04f/D-010
        # The band is built as a KEEP predicate (`1 = attend`) and handed to
        # `SingleWindowAttention` on its rank-3 PAIRWISE branch, which routes it
        # through `common.apply_attention_mask`.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT copy `gemma3_transformer.py:_create_attention_mask`
        #     verbatim. That one is the idiom, and it is CAUSAL and in SUPPRESS
        #     semantics (`j > i` OR-ed with `(i - j) >= sliding_window_size`,
        #     inverted once by its caller). This band is SYMMETRIC and
        #     non-causal -- ModernBERT is an encoder -- so it is
        #     `abs(i - j) <= window_size`, and it is already in the keep
        #     polarity `apply_attention_mask` wants. Taking gemma's expression
        #     unchanged would make every token blind to its own future
        #     neighbours, which no shape or finiteness check can see.
        #   * Do NOT hand-roll an additive `-1e9` sentinel here. `-1e9` is
        #     `-inf` in float16 and `0 * -inf = NaN`; this repo has a recorded
        #     10-site fp16 mask-NaN family, and `apply_attention_mask` is the
        #     single fixed instance of that pattern.
        #   * Do NOT REPLACE the caller's `attention_mask` with the band. The
        #     two are composed multiplicatively so a padded key stays masked
        #     inside the band; substituting either way un-masks real padding or
        #     un-masks the far context, both silently.
        #   * Do NOT reuse `window_slots` to suppress the internal padding.
        #     Its values are validated into `[0, window_size ** 2)` because they
        #     index a TILE -- at ModernBERT's `window_size = 128 // 2 = 64` that
        #     caps the sequence at 4096 tokens for a layout that has no such
        #     bound. `pad_to_window=False` is the layout-free spelling; see the
        #     D-009 anchor in `single_window_attention.py`.
        #
        # ACCEPTED COST, stated because D-027 records ten previously-INVERTED
        # cost claims about this exact layer: a dense `N x N` banded mask is
        # `O(N^2)`, the SAME asymptotics as full attention. It is not the
        # `O(N * W)` the name "sliding window" suggests -- that needs a fused
        # kernel this repo has no path to. What it buys over `'grid'` is that
        # `N` real tokens are never inflated to `window_size ** 2` slots.
        # Measure it, do not trust this sentence:
        #   .venv/bin/python -c "import resource, numpy as np, keras; from
        #   dl_techniques.layers.attention.window_attention import
        #   WindowAttention; x = np.zeros((1, 512, 64), 'float32');
        #   WindowAttention(64, 64, 4, partition_mode='band',
        #   use_relative_position_bias=False)(x);
        #   print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)"
        # See decisions.md D-009 (plan-2026-08-25T053412-0f1fa04f).
        n_tokens = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(n_tokens, dtype="int32")
        distance = keras.ops.absolute(
            keras.ops.expand_dims(positions, axis=-1)
            - keras.ops.expand_dims(positions, axis=0)
        )
        # (1, N, N) keep predicate: 1 where the key is inside the band.
        band_keep = keras.ops.expand_dims(
            keras.ops.cast(distance <= self.window_size, "int32"), axis=0
        )

        if attention_mask is None:
            keep = band_keep
        elif len(attention_mask.shape) == 3:
            # (B, N, N) pairwise caller mask, AND-ed with the band.
            keep = keras.ops.cast(attention_mask, "int32") * band_keep
        elif len(attention_mask.shape) == 2:
            # (B, N) key mask -> (B, 1, N), AND-ed with the band.
            keep = (
                keras.ops.cast(
                    keras.ops.expand_dims(attention_mask, axis=1), "int32"
                )
                * band_keep
            )
        else:
            raise ValueError(
                f"WindowAttention(partition_mode='band') accepts a rank-2 "
                f"(B, N) key mask or a rank-3 (B, N, N) pairwise mask; got "
                f"rank {len(attention_mask.shape)} with shape "
                f"{tuple(attention_mask.shape)}."
            )

        return self._attend(
            inputs,
            attention_mask=keep,
            training=training,
            pad_to_window=False,
        )

    def _call_zigzag(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass using zigzag frequency-locality partitioning.

        :param inputs: Sequence tensor ``(B, N, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask ``(B, N)``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(B, N, dim)``.
        :rtype: keras.KerasTensor
        """
        ws = self.window_size

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-014
        # The SAME degenerate-single-window short-circuit `_call_grid` has carried
        # since step 3 (D-007), on the path that never got it. When
        # `N < window_size ** 2` the zigzag layout below is ALSO exactly one window:
        # it squares the sequence into a `ceil(sqrt(N)) x ceil(sqrt(N))` grid, so
        # `N_grid = ceil(sqrt(N)) ** 2 <= window_size ** 2` and `num_windows == 1`.
        # Every token attends every other token either way -- the zigzag order is a
        # PERMUTATION of the single window's slots, not a different neighbourhood --
        # so the only things the padding path adds are the `win_len - N` zero slots
        # (which D-011 then has to mask back out) and their cost.
        #
        # MEASURED, `(1, 128, 64)` at `window_size=128`, CPU peak RSS, this repo's
        # four comparable modes: `window` (grid) 0.649 GB, `window_band` 0.648,
        # `multi_head` 0.643, and `window_zigzag` 21.695 -- 128 real tokens inflated
        # to 16,384 slots and attended densely, a 33x penalty for asking a LOCAL
        # attention layer a question plain global attention answers in 0.643 GB.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT reuse `_single_window_slots`. That is the GRID layout's slot
        #     map (`(i // grid_side) * ws + (i % grid_side)`), and zigzag does not
        #     lay tokens out row-major. The right slot for token `i` is its ZIGZAG
        #     POSITION, `inverse_zigzag_indices[i]`, which is what the padding path
        #     below puts it at -- so the relative-position bias is gathered at the
        #     identical table rows and the bias is bit-for-bit unchanged.
        #   * Do NOT widen this to `N <= ws ** 2`. At `N == ws ** 2` both pads are
        #     zero, so the padding path attends exactly the N real tokens already
        #     and there is nothing to save; taking the short-circuit there would
        #     only move 8 harness cells off the byte-identical path for no gain.
        #   * Do NOT drop the `N > 1` guard, for the same reason `_call_grid` keeps
        #     it: `keras.ops.softmax` warns on a size-1 reduction axis and this
        #     repo's pytest config escalates warnings to errors.
        #   * Do NOT take this branch on a rank-3 mask. `_call_zigzag`'s mask
        #     pipeline is rank-2-only (it pads and permutes a `(B, N)` key mask); a
        #     rank-3 pairwise mask has no meaning here and must reach the existing
        #     code, which is where it fails.
        #
        # ACCEPTED COST -- a VALUE change in this regime, exactly as D-007 ruled for
        # grid. The short-circuit sums `N` products where the padding path summed
        # `win_len` products of which `win_len - N` are exactly zero, AND it sums
        # them in token order rather than zigzag order, so the two differ at float32
        # REDUCTION ORDER. MEASURED against the pad-masked pre-restructure reference,
        # worst case over the six affected harness cells: 2.086e-07, one to two
        # float32 ulps. See decisions.md D-007, D-009 and D-014.
        static_n = inputs.shape[1]
        degenerate = (
            static_n is not None
            and 1 < int(static_n) < ws * ws
            and (attention_mask is None or len(attention_mask.shape) == 2)
        )
        if degenerate:
            return self._attend(
                inputs,
                attention_mask=attention_mask,
                training=training,
                window_slots=self.inverse_zigzag_indices[: int(static_n)],
            )

        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        win_len = ws * ws
        pad_len_win = (win_len - (self.N_grid % win_len)) % win_len

        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, self.pad_len_seq], [0, 0]]
        )

        zigzag_sequence = keras.ops.take(
            padded_inputs, self.zigzag_indices, axis=1
        )

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-011
        # The SAME synthesized-mask fix as `_call_grid`, for the SAME reason, on a
        # path that never had the D-007 short-circuit at all. This method pads
        # twice -- `pad_len_seq` slots to square the sequence into the zigzag grid,
        # then `pad_len_win` slots to fill the last window -- and with
        # `attention_mask=None` neither pad was ever masked, so the zero-filled
        # slots entered the softmax as ordinary keys and values at EVERY ragged
        # `N`, not merely below `window_size ** 2`.
        #
        # MEASURED on 8435dcc2f, max |delta| between `attention_mask=None` and an
        # explicit all-ones `(B, N)` mask (a mathematical no-op):
        #     ws=4, N=9  -> 0.3826489   (pad_len_seq=0, pad_len_win=7 of 16 slots)
        #     ws=7, N=25 -> 0.2675675   (pad_len_seq=0, pad_len_win=24 of 49)
        #     ws=8, N=50 -> 0.0807583   (pad_len_seq=14, pad_len_win=0)
        # and exactly 0.0 at `N in {4, 16, 64, 196, 256}` for their window sizes,
        # where both pads are zero. See D-011.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT gate on `self.pad_len_seq` alone. `ws=4, N=9` has
        #     `pad_len_seq == 0` -- 9 is a perfect square, so the zigzag grid is
        #     exactly 3x3 -- and leaks 0.38 anyway, entirely through `pad_len_win`.
        #     Both pads have to be in the condition.
        #   * Do NOT permute the synthesized mask by hand. It is created in
        #     UNPERMUTED token coordinates and then passed through the same
        #     `pad -> take(zigzag_indices) -> pad` pipeline as the data, so the
        #     zigzag permutation is applied to it exactly once, by the same code.
        #     Building it after the permutation would make the pad positions a
        #     second, hand-maintained copy of the layout.
        key_mask = attention_mask
        if key_mask is None and (self.pad_len_seq > 0 or pad_len_win > 0):
            key_mask = keras.ops.ones_like(inputs[:, :, 0], dtype="int32")

        zigzag_mask = None
        if key_mask is not None:
            padded_mask = keras.ops.pad(
                key_mask, [[0, 0], [0, self.pad_len_seq]]
            )
            zigzag_mask = keras.ops.take(
                padded_mask, self.zigzag_indices, axis=1
            )

        padded_zigzag_seq = keras.ops.pad(
            zigzag_sequence, [[0, 0], [0, pad_len_win], [0, 0]]
        )

        num_windows = (self.N_grid + pad_len_win) // win_len
        windows = keras.ops.reshape(
            padded_zigzag_seq, (B * num_windows, win_len, C)
        )

        attn_mask_for_windows = None
        if zigzag_mask is not None:
            padded_zigzag_mask = keras.ops.pad(
                zigzag_mask, [[0, 0], [0, pad_len_win]], constant_values=0
            )
            attn_mask_for_windows = keras.ops.reshape(
                padded_zigzag_mask, (B * num_windows, win_len)
            )

        attn_windows = self._attend(
            windows, attention_mask=attn_mask_for_windows, training=training
        )

        merged_zigzag_seq = keras.ops.reshape(
            attn_windows, (B, num_windows * win_len, C)
        )
        unpadded_zigzag_seq = merged_zigzag_seq[:, : self.N_grid, :]

        sequence_unpadded = keras.ops.take(
            unpadded_zigzag_seq, self.inverse_zigzag_indices, axis=1
        )

        output = sequence_unpadded[:, :N_actual, :]
        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape (identical to the input shape).

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "partition_mode": self.partition_mode,
                "attention_mode": self.attention_mode,
                "use_relative_position_bias": self.use_relative_position_bias,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "dropout_rate": self.dropout_rate,
                "proj_bias": self.proj_bias,
                "kan_grid_size": self.kan_grid_size,
                "kan_spline_order": self.kan_spline_order,
                "kan_activation": serialize_activation(self.kan_activation),
                "probability_type": self.probability_type,
                "probability_config": self.probability_config,
                "qk_norm_type": self.qk_norm_type,
                "qk_norm_kwargs": self.qk_norm_kwargs,
                "kernel_initializer": keras.initializers.serialize(
                    keras.initializers.get(self.kernel_initializer)
                ),
                "bias_initializer": keras.initializers.serialize(
                    keras.initializers.get(self.bias_initializer)
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    keras.regularizers.get(self.kernel_regularizer)
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    keras.regularizers.get(self.bias_regularizer)
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowAttention":
        """Create a layer from its configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: New ``WindowAttention`` instance.
        :rtype: WindowAttention
        """
        return cls(**config)

# ---------------------------------------------------------------------

"""
Utility functions for creating common variations of the WindowAttention layer.

This module provides factory functions that simplify the creation of specialized
WindowAttention layers by pre-configuring them for common use cases. This
promotes code readability and reduces boilerplate when experimenting with
different attention architectures.

Each function wraps the core `WindowAttention` layer, setting specific
parameters like `partition_mode`, `attention_mode`, or `normalization`
to sensible defaults for that particular variant. All other parameters can
still be overridden via keyword arguments.

Available Factories:
--------------------
- `create_grid_window_attention`:
    Standard Swin Transformer-style spatial windowing.

- `create_zigzag_window_attention`:
    Windowing on a zigzag-reordered sequence for frequency locality.

- `create_band_window_attention`:
    A 1-D symmetric band over the token sequence (`window_size` is a HALF-WIDTH
    in tokens), which is what text encoders such as ModernBERT specify. No grid
    folding, no square padding. Defaults `use_relative_position_bias=False`.

- `create_kan_key_window_attention`:
    Window attention using a non-linear KAN layer for the Key projection.

- `create_adaptive_softmax_window_attention`:
    Window attention with adaptive temperature softmax for better calibration.

Public vs. internal surface:
----------------------------
Only the first THREE are public. `create_grid_window_attention`,
`create_zigzag_window_attention` and `create_band_window_attention` back the `'window'`,
`'window_zigzag'` and `'window_band'` keys in `attention/factory.py` — the factory
dispatches through these wrappers, NOT through the `WindowAttention` class directly,
because each key carries a different `use_relative_position_bias` default that the class
itself does not encode. One key per partition mode over ONE shared implementation is the
convention here; do not add a second spelling (a `partition_mode` entry in `'window'`'s
`optional_params`, or a `'sliding_window'` key) for a mode that already has one.

`create_kan_key_window_attention` and `create_adaptive_softmax_window_attention` are
INTENTIONALLY NOT public: they are neither exported from `attention/__init__.py` nor
registered in `factory.py`, and their only callers in the repository are
`tests/test_layers/test_attention/test_window_attention.py`. They exist as convenience
constructors for the `attention_mode='kan_key'` and `probability_type='adaptive'`
configurations. Do NOT register them in `factory.py` to "fix the inconsistency" — the
registry surface is frozen, and adding keys changes the public API. Either configuration
is reachable today by passing those kwargs to `WindowAttention` directly.
"""

# ---------------------------------------------------------------------


def create_grid_window_attention(
    dim: int, window_size: int, num_heads: int, **kwargs: Any
) -> WindowAttention:
    """
    Creates a standard spatial window attention layer (Swin-style).

    This factory configures `WindowAttention` for grid-based partitioning,
    which is ideal for tasks where 2D spatial locality is important.
    It defaults to using relative position bias, as is standard for this
    architecture.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param kwargs: Additional keyword arguments to pass to the `WindowAttention`
        constructor (e.g., `dropout_rate`, `qkv_bias`).
    :type kwargs: Any
    :return: A `WindowAttention` layer configured for grid partitioning.
    :rtype: WindowAttention
    """
    # Default to using relative position bias, but allow override.
    kwargs.setdefault("use_relative_position_bias", True)

    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode="grid",
        **kwargs,
    )

# ---------------------------------------------------------------------


def create_zigzag_window_attention(
    dim: int, window_size: int, num_heads: int, **kwargs: Any
) -> WindowAttention:
    """
    Creates a window attention layer with zigzag partitioning.

    This factory configures `WindowAttention` to reorder the sequence
    along a 2D zigzag path before windowing. This groups frequency-proximate
    tokens, inducing a locality bias sensitive to frequency bands. It defaults
    to *disabling* relative position bias, as the original spatial grid is
    intentionally broken by the zigzag scan.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param kwargs: Additional keyword arguments to pass to the `WindowAttention`
        constructor (e.g., `dropout_rate`, `proj_bias`).
    :type kwargs: Any
    :return: A `WindowAttention` layer configured for zigzag partitioning.
    :rtype: WindowAttention
    """
    # Default to disabling relative position bias, but allow override.
    kwargs.setdefault("use_relative_position_bias", False)

    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode="zigzag",
        **kwargs,
    )

# ---------------------------------------------------------------------


def create_band_window_attention(
    dim: int, window_size: int, num_heads: int, **kwargs: Any
) -> WindowAttention:
    """
    Creates a 1-D symmetric sliding-band attention layer.

    This factory configures `WindowAttention` for `partition_mode='band'`:
    ``window_size`` is a HALF-WIDTH IN TOKENS, and query ``i`` attends key ``j``
    iff ``abs(i - j) <= window_size``. The band is symmetric and non-causal,
    which is what a text ENCODER specifies — upstream ModernBERT's
    ``local_attention`` is a FULL span, so the value to pass here is
    ``local_attention // 2``. There is no grid folding and no square padding.

    It defaults to *disabling* the relative position bias, which the ``'band'``
    layout in fact REFUSES (it is indexed by 2-D tile coordinates a 1-D band does
    not have); passing ``use_relative_position_bias=True`` raises.

    **Cost, measured rather than asserted.** A dense ``N x N`` banded mask is
    ``O(N^2)`` — the SAME asymptotics as full attention. It is not the
    ``O(N * W)`` the phrase "sliding window" suggests; that needs a fused kernel
    this repo has no path to. What the band buys over ``'grid'`` is that ``N``
    real tokens are never inflated to ``window_size ** 2`` slots. To measure::

        .venv/bin/python -c "import resource, numpy as np; \
        from dl_techniques.layers.attention.window_attention import \
        create_band_window_attention as f; x = np.zeros((1, 512, 64), 'float32'); \
        f(dim=64, window_size=64, num_heads=4)(x); \
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)"

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The band HALF-WIDTH, in tokens. ``window_size=64``
        means each query sees 64 tokens either side, a 129-token span
        including itself.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param kwargs: Additional keyword arguments to pass to the `WindowAttention`
        constructor (e.g., `dropout_rate`, `proj_bias`).
    :type kwargs: Any
    :return: A `WindowAttention` layer configured for a 1-D band.
    :rtype: WindowAttention
    """
    # Default to disabling relative position bias; the band layout has no 2-D
    # tile for it to be indexed by, and the class REFUSES an explicit True.
    kwargs.setdefault("use_relative_position_bias", False)

    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode="band",
        **kwargs,
    )

# ---------------------------------------------------------------------


def create_kan_key_window_attention(
    dim: int,
    window_size: int,
    num_heads: int,
    partition_mode: Literal["grid", "zigzag", "band"] = "grid",
    **kwargs: Any,
) -> WindowAttention:
    """
    Creates a window attention layer with a non-linear KAN Key projection.

    **Intentionally non-public**: not exported from ``attention/__init__.py`` and not
    registered in ``attention/factory.py``. Its only callers are in
    ``tests/test_layers/test_attention/test_window_attention.py``. See the module-level
    "Public vs. internal surface" note above.

    This factory configures `WindowAttention` to use a `KANLinear` layer for
    projecting the Key tensor. This allows for more expressive similarity
    matching compared to a standard linear projection.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param partition_mode: The partitioning strategy (`'grid'`, `'zigzag'` or
        `'band'`). Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag", "band"]
    :param kwargs: Additional keyword arguments to pass to `WindowAttention`,
        especially KAN-specific ones like `kan_grid_size`,
        `kan_spline_order`.
    :type kwargs: Any
    :return: A `WindowAttention` layer with a KAN-based Key projection.
    :rtype: WindowAttention
    """
    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode=partition_mode,
        attention_mode="kan_key",
        **kwargs,
    )

# ---------------------------------------------------------------------


def create_adaptive_softmax_window_attention(
    dim: int,
    window_size: int,
    num_heads: int,
    partition_mode: Literal["grid", "zigzag", "band"] = "grid",
    **kwargs: Any,
) -> WindowAttention:
    """
    Creates a window attention layer with adaptive temperature softmax.

    **Intentionally non-public**: not exported from ``attention/__init__.py`` and not
    registered in ``attention/factory.py``. Its only callers are in
    ``tests/test_layers/test_attention/test_window_attention.py``. See the module-level
    "Public vs. internal surface" note above.

    This factory configures `WindowAttention` to use `AdaptiveTemperatureSoftmax`
    for normalization. This can improve model calibration and performance by
    dynamically adjusting the sharpness of the attention distribution based on
    model confidence.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param partition_mode: The partitioning strategy (`'grid'`, `'zigzag'` or
        `'band'`). Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag", "band"]
    :param kwargs: Additional keyword arguments to pass to `WindowAttention`,
        especially `probability_config` for adaptive-softmax temperature
        parameters.
    :type kwargs: Any
    :return: A `WindowAttention` layer with adaptive softmax normalization.
    :rtype: WindowAttention
    """
    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode=partition_mode,
        probability_type="adaptive",
        **kwargs,
    )

# ---------------------------------------------------------------------
