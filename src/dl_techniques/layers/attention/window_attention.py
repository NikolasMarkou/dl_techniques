"""
Unified windowed multi-head self-attention for sequence processing.

This module provides a highly configurable windowed multi-head self-attention
layer. It unifies two distinct partitioning strategies—standard grid-based
windowing and frequency-proximate zigzag windowing—into a single interface,
controlled by the `partition_mode` parameter.

The layer takes a 1D sequence, internally reshapes it, partitions it
according to the chosen mode, and computes self-attention within each local
window, offering different locality biases.

**Read the cost model before reaching for this layer as an "efficient
attention".** Every window is padded up to exactly ``M = window_size ** 2``
positions, so the cost is ``O(max(N, M) * M)`` — asymptotically linear in
``N`` only *above* ``N = M``, and pinned at a constant floor of ``M ** 2 =
window_size ** 4`` below it. Whenever ``N <= M`` the padded grid is a single
window and the layer computes **dense attention over an M-token padded
sequence**, which costs ``(M / N) ** 2`` times *more* than plain global
attention on the ``N`` real tokens, not less. At ``window_size = 128``
(``M = 16384``) that threshold sits above any sequence most callers will
ever pass. See "Foundational Mathematics" below for the derivation.

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
    * ``N <= M`` — **degenerate**. ``H <= window_size``, so ``pad_h =
      window_size - H``, the padded grid is exactly one ``window_size x
      window_size`` tile, ``_window_partition`` yields a single window, and the
      layer computes dense attention over ``M`` positions of which ``M - N``
      are padding. The cost is the constant ``M^2`` regardless of ``N``, i.e.
      ``(M / N)^2`` times *more* than global attention over the ``N`` real
      tokens. At ``window_size = 128`` this covers every ``N <= 16384``: the
      per-layer score matrix is ``16384 x 16384 ~ 2.7e8`` entries per head per
      sample whether ``N`` is 128 or 8192.

    Choosing ``window_size`` is therefore choosing a *minimum* cost, not a
    maximum one: a window size picked to be "generous" makes the layer
    strictly more expensive than the global attention it replaces at every
    sequence length that fits inside one window.

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
N <= M         O(M²)              ONE padded window: dense attention, and
                                  (M/N)² times MORE work than global O(N²)
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
import keras
from typing import Any, Dict, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .single_window_attention import SingleWindowAttention

# ---------------------------------------------------------------------
# Probability types not supported in window attention (score-level routing
# is structurally incompatible with windowed score tensors).
# ---------------------------------------------------------------------

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
    ``'zigzag'`` (2D zigzag scan grouping frequency-proximate tokens).
    All padding, reshaping, partitioning, and merging are handled
    internally.

    **Cost.** With ``M = window_size ** 2`` token slots per window, the cost is
    ``O(max(N, M) * M)``: linear in ``N`` only while ``N > M``, and a constant
    ``O(M^2)`` floor below that. For ``N <= M`` the internal grid pads up to a
    single ``window_size x window_size`` tile, so this layer performs **dense
    attention over M padded positions** — ``(M / N) ** 2`` times more work than
    global attention over the ``N`` real tokens. This is a property of the
    padding, not an approximation: see the module docstring's "Foundational
    Mathematics" section, and
    ``tests/test_models/test_modern_bert/test_shipped_window_size.py``, which
    pins the degeneracy at the ``window_size = 128`` that ``ModernBERT``
    actually ships.

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
    :param partition_mode: The partitioning strategy. One of `'grid'` or
        `'zigzag'`. Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag"]
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
        relationship is already altered. Default: True.
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
        partition_mode: Literal["grid", "zigzag"] = "grid",
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
        self.kan_activation = kan_activation
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

    @staticmethod
    def _generate_zigzag_indices(
        H: keras.KerasTensor, W: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Generate zigzag scan indices for an ``H x W`` grid.

        :param H: Grid height.
        :type H: keras.KerasTensor
        :param W: Grid width.
        :type W: keras.KerasTensor
        :return: 1-D int32 index tensor of length ``H * W``.
        :rtype: keras.KerasTensor
        """
        r_coords = keras.ops.arange(0, H, dtype="int32")
        c_coords = keras.ops.arange(0, W, dtype="int32")
        r_grid, c_grid = keras.ops.meshgrid(r_coords, c_coords, indexing="ij")

        r_flat = keras.ops.reshape(r_grid, (-1,))
        c_flat = keras.ops.reshape(c_grid, (-1,))

        s = r_flat + c_flat
        secondary_key = keras.ops.where(s % 2 == 1, r_flat, H - 1 - r_flat)
        combined_key = s * H + secondary_key

        return keras.ops.argsort(combined_key)

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

            H_tensor = keras.ops.convert_to_tensor(self.H, dtype="int32")
            W_tensor = keras.ops.convert_to_tensor(self.W, dtype="int32")
            self.zigzag_indices = self._generate_zigzag_indices(
                H_tensor, W_tensor
            )
            self.inverse_zigzag_indices = keras.ops.argsort(
                self.zigzag_indices
            )

        self.attention.build(
            (None, self.window_size * self.window_size, self.dim)
        )

        if self.partition_mode == "grid":
            self._call_internal = self._call_grid
        elif self.partition_mode == "zigzag":
            self._call_internal = self._call_zigzag
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
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        ws = self.window_size

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
        elif attention_mask is not None:
            mask = keras.ops.pad(attention_mask, [[0, 0], [0, pad_amount_seq]])
            mask = keras.ops.reshape(mask, (B, H, W))
            mask = keras.ops.pad(mask, [[0, 0], [0, pad_h], [0, pad_w]])
            mask = keras.ops.expand_dims(mask, axis=-1)
            mask_windows = self._window_partition(mask)
            window_mask = keras.ops.reshape(mask_windows, (-1, ws * ws))

        attn_windows = self.attention(
            windows, attention_mask=window_mask, training=training
        )
        attn_windows = keras.ops.reshape(attn_windows, (-1, ws, ws, C))
        x = self._window_reverse(attn_windows, H_pad, W_pad)

        x = x[:, :H, :W, :]
        x = keras.ops.reshape(x, (B, N_grid, C))
        x = x[:, :N_actual, :]
        return x

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
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        win_len = self.window_size * self.window_size

        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, self.pad_len_seq], [0, 0]]
        )

        zigzag_sequence = keras.ops.take(
            padded_inputs, self.zigzag_indices, axis=1
        )

        zigzag_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, self.pad_len_seq]]
            )
            zigzag_mask = keras.ops.take(
                padded_mask, self.zigzag_indices, axis=1
            )

        pad_len_win = (win_len - (self.N_grid % win_len)) % win_len
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

        attn_windows = self.attention(
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
                "kan_activation": self.kan_activation,
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

- `create_kan_key_window_attention`:
    Window attention using a non-linear KAN layer for the Key projection.

- `create_adaptive_softmax_window_attention`:
    Window attention with adaptive temperature softmax for better calibration.

Public vs. internal surface:
----------------------------
Only the first TWO are public. `create_grid_window_attention` and
`create_zigzag_window_attention` back the `'window'` and `'window_zigzag'` keys in
`attention/factory.py` — the factory dispatches through these wrappers, NOT through the
`WindowAttention` class directly, because each key carries a different
`use_relative_position_bias` default that the class itself does not encode.

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


def create_kan_key_window_attention(
    dim: int,
    window_size: int,
    num_heads: int,
    partition_mode: Literal["grid", "zigzag"] = "grid",
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
    :param partition_mode: The partitioning strategy (`'grid'` or `'zigzag'`).
        Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag"]
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
    partition_mode: Literal["grid", "zigzag"] = "grid",
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
    :param partition_mode: The partitioning strategy (`'grid'` or `'zigzag'`).
        Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag"]
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
