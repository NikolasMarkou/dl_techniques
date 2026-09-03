"""
Unified windowed multi-head self-attention, built by the ``WindowAttention``
class, with three ``partition_mode`` choices: Swin-style 2-D grid tiles, a
zigzag reordering of the same grid, and a 1-D symmetric band over the raw
token sequence.

Input and output are both 1-D sequences of the same shape; padding,
reshaping, partitioning and merging all happen internally. The three modes
do not share one cost model. Writing M = window_size ** 2: 'grid' and
'zigzag' cost O(N*M) for N > M and fall back to O(N**2) below that, never
worse than global attention; 'band' is always O(N**2), the same order as
global attention, and never pads.

A caller rarely builds ``WindowAttention`` directly -- see
:func:`create_grid_window_attention`, :func:`create_zigzag_window_attention`
and :func:`create_band_window_attention`, which set each mode's own relative
position bias default. Full behavior and diagrams are on the class and on
each factory function.

References:
    - Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision
      Transformer using Shifted Windows. (https://arxiv.org/abs/2103.14030)
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

from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from .single_window_attention import SingleWindowAttention

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


@register_dl_technique("dl_techniques.layers.attention.window_attention")
class WindowAttention(keras.layers.Layer):
    """
    Unified window-based multi-head self-attention layer.

    Partitions a 1-D token sequence into local neighbourhoods and computes
    multi-head self-attention inside each one, via
    :class:`SingleWindowAttention`. Three partitioning strategies share the
    class:

    * ``'grid'`` -- Swin Transformer-style 2-D spatial tiles. ``window_size``
      is an edge length.
    * ``'zigzag'`` -- the same tiles over a zigzag scan of the grid, grouping
      frequency-proximate tokens.
    * ``'band'`` -- a 1-D symmetric band over the token sequence.
      ``window_size`` is a half-width in tokens: query ``i`` attends key
      ``j`` iff ``abs(i - j) <= window_size``. No grid folding, no square
      padding.

    All padding, reshaping, partitioning and merging happen internally, and
    the output has the input's shape.

    Cost, per mode -- they do not share one. With ``M = window_size ** 2``:

    * ``'band'`` never pads. A dense ``N x N`` banded mask over standard
      attention is ``O(N^2)``, the same order as plain global attention, not
      the ``O(N * window_size)`` a "sliding window" usually means.
    * ``'grid'`` and ``'zigzag'`` are ``O(N * M)`` for ``N > M``, and
      ``O(N^2)`` for ``1 < N <= M``, where a short-circuit attends the ``N``
      real tokens instead of ``M`` padded slots, so neither mode has an
      ``O(window_size ** 4)`` floor.
    * ``N == 1`` still pads to ``M`` in both grid modes, because
      ``keras.ops.softmax`` warns on a size-1 reduction axis and this repo's
      pytest config turns that warning into an error.

    Peak RSS on ``(1, 128, 64)`` at ``window_size=128``, CPU, one fresh
    process per mode, measured 2026-08-28: ``'grid'`` 0.681 GB, ``'band'``
    0.679 GB, ``'multi_head'`` 0.675 GB, ``'zigzag'`` 0.678 GB -- a bare
    ``import keras`` in the same environment already costs 0.655 GB, so no
    mode stands out; before the two short-circuits above, the two grid modes
    read 21.695 GB and 17.503 GB on that same input.

    Architecture:

    .. code-block:: text

              Input: 1-D sequence  (B, N, dim)
                            │
        0. SHORT-CIRCUIT, 'grid' and 'zigzag' ONLY: if N is
           static, 1 < N < ws^2, and the mask is rank 2 or absent,
           attend the N REAL tokens as ONE window and skip
           stages 1-3 and 5.  D-007 (grid), D-014 (zigzag).
                            │
             ┌──────────────┼───────────────┐
          'grid'        'zigzag'          'band'
             ▼              ▼               ▼
        1. pad N to    1. pad N to     1-3. NO grid fold
           H*W, with      N_grid = H*W       and NO padding
           H = W =        and STOP —         of any kind.
           ceil(          the sequence       Build the band
           sqrt(N))       STAYS 1-D          predicate
             ▼              ▼                instead:
        2. pad H and   2. take(x, self.      keep[i,j] =
           W each up      zigzag_indices,    |i-j| <= ws,
           to a           axis=1) — the      a (1, N, N)
           multiple       reorder, still     int32 mask
           of ws          1-D                AND-ed with
             ▼              ▼                any caller
        3. cut into    3. pad that FLAT      mask
           contiguous     sequence up to
           ws x ws        a multiple of
           tiles          ws^2, reshape
             │              │               │
             └──────┬───────┘               │
                    ▼                       │
        both grid modes now hold            │
        (B*num_win, ws^2, dim)              │
                    └───────────┬───────────┘
                                ▼
        4. SingleWindowAttention  (QKV, optional QK-norm,
           relative position bias, ProbabilityOutput —
           all the attention math lives there)
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
        5. grid: merge, unpad,        5. nothing to undo
           slice to N. zigzag:           (no pad added)
           merge, slice to N_grid,
           take(inverse_zigzag_
           indices), slice to N
                └───────────┬────────────┘
                            ▼
              Output: 1-D sequence  (B, N, dim)

    ``'band'`` never enters stages 1-3, so it has no tile and refuses the
    relative position bias. The two grid modes do not pad the same way:
    ``'grid'`` pads the sequence and then pads H and W, while ``'zigzag'``
    pads the sequence, reorders it, and then pads the flat result up to a
    multiple of ``window_size ** 2``. Each builder's own diagram below
    repeats its own path.

    Partition modes, and the builders that reach them:

    .. code-block:: text

        mode      builder / factory key     window_size    RPB default
        --------  ------------------------  -------------  -----------
        'grid'    create_grid_window_...    tile EDGE      True
                  'window'                  LENGTH
        'zigzag'  create_zigzag_window_...  tile EDGE      False
                  'window_zigzag'           LENGTH         (overridable)
        'band'    create_band_window_...    half-width     False
                  'window_band'             IN TOKENS      (True raises)

        grid folding + square padding:  'grid' yes, 'zigzag' yes,
                                        'band' no
        RPB = use_relative_position_bias.  The class default is True.
        Each builder sets its own with `setdefault`, so a caller can
        still override it — except under 'band', which raises on True.

    :param dim: Dimension of the input tokens (channels).
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param partition_mode: The partitioning strategy. One of `'grid'`,
        `'zigzag'` or `'band'`. Default: 'grid'.

        * ``'grid'`` — Swin-style 2-D tiles. ``window_size`` is an edge
          length: the sequence is folded into a ``ceil(sqrt(N))``-square grid
          and cut into ``window_size x window_size`` tiles.
        * ``'zigzag'`` — the same tiles over a zigzag reordering of the grid.
        * ``'band'`` — a 1-D symmetric band over the token sequence, with no
          grid folding and no square padding. ``window_size`` is a
          half-width in tokens: query ``i`` attends key ``j`` iff
          ``abs(i - j) <= window_size``. Non-causal (both directions), which
          is what a text encoder such as ModernBERT specifies — upstream's
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
        relationship is already altered. For `'band'` mode it is refused, not
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
        """Validate the configuration and store the resolved constants.

        The mode-dependent refusals live here: ``partition_mode`` must name a
        known partitioner, and ``'band'`` refuses a relative-position bias
        outright rather than turning it off, because a 1-D band has no tile to
        index one from. The softmax temperature is resolved once, taking
        ``qk_scale`` when given and ``head_dim ** -0.5`` otherwise. The
        projections and the bias table are created in :meth:`build`, where the
        input width is known. See the class docstring for the parameter
        reference.

        :raises ValueError: For any invalid argument; see the class docstring's
            ``:raises:`` list.
        """
        super().__init__(**kwargs)

        if partition_mode not in _VALID_PARTITION_MODES:
            raise ValueError(
                f"partition_mode must be one of "
                f"{list(_VALID_PARTITION_MODES)}; got {partition_mode!r}."
            )

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-010: refuse RPB under 'band'
        # (raise), never quietly turn it off -- a 1-D band has no tile to index, so the gathered rows would be arbitrary. See decisions.md.
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
            # DECISION plan-2026-08-19T163559-499b6f0e/D-081: name spelled explicitly
            # (matches Keras' per-process auto-name) -- dropping it moved weight paths for swin_transformer, scunet, modern_bert. See decisions.md.
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

    # DECISION plan-2026-08-25T053412-0f1fa04f/D-014: zigzag layout stays numpy,
    # not keras.ops -- build() can trace inside tf.function and needs a Python-visible inverse (D-015). See decisions.md.

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
          ``None``. It is handed over by ``set_window_slots()``, not as a call
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
        not the identity, whenever ``W < window_size`` -- is what the
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
        """Dispatch to the partition mode chosen at construction time.

        ``build()`` binds ``self._call_internal`` to :meth:`_call_grid`,
        :meth:`_call_zigzag` or :meth:`_call_band`, so the branch is resolved
        once rather than per call. Each of those three documents the mask ranks
        it accepts; they are not the same.

        :param inputs: Sequence tensor ``(B, N, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional keep predicate (``1 = attend``).
            ``'grid'`` takes rank-2 ``(B, N)`` or rank-3 ``(B, ws**2, ws**2)``,
            ``'band'`` takes rank-2 ``(B, N)`` or rank-3 ``(B, N, N)``, and
            ``'zigzag'`` takes rank-2 only.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training-mode flag, forwarded unchanged.
        :type training: Optional[bool]
        :return: Output tensor ``(B, N, dim)``, the input's shape.
        :rtype: keras.KerasTensor
        """
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
            ``(B, ws**2, ws**2)`` (a pairwise mask already expressed in
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

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-007: when N < ws**2, attend the
        # N real tokens as one window instead of padding to ws**2 -- the old path inflated N=128,ws=128 to a 4.0 GiB score matrix. Value-changing fix; see decisions.md.
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
            # DECISION plan-2026-07-31T042809-ddc92265/D-001: forward a rank-3 mask
            # verbatim (already in partitioned window coordinates from SwinTransformerBlock) -- never re-partition it, and only when the grid is degenerate. See decisions.md.
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
            # DECISION plan-2026-08-25T053412-0f1fa04f/D-011: synthesize an all-ones
            # key mask when attention_mask is None, gated by _pads_exist -- unmasked pads leaked into softmax (max delta 1.0259 at ws=8,N=100). See decisions.md.
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
            It is composed with the band, never substituted for it.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(B, N, dim)``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-25T053412-0f1fa04f/D-010: band is a keep predicate,
        # composed (never substituted) with the caller's mask; not gemma3's causal/suppress expression, and no hand-rolled -1e9 (NaN under float16). See decisions.md.
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

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-014: the same single-window
        # short-circuit as _call_grid's D-007, for zigzag -- cut CPU peak RSS from 21.695 GB to 0.678 GB at N=128,ws=128. See decisions.md.
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

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-011: same synthesized-mask fix
        # as _call_grid, gated on EITHER pad (pad_len_seq alone missed a 0.38 leak from pad_len_win at ws=4,N=9). See decisions.md.
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
Factory functions that pre-configure ``WindowAttention`` for common cases.
Each wraps the class, setting ``partition_mode`` and its mode's own
``use_relative_position_bias`` default; other parameters can still be
overridden.

``create_grid_window_attention``, ``create_zigzag_window_attention`` and
``create_band_window_attention`` are the public surface, backing the
``'window'``, ``'window_zigzag'`` and ``'window_band'`` keys in
``attention/factory.py`` -- the factory dispatches through these wrappers,
not through ``WindowAttention`` directly, because each key's
``use_relative_position_bias`` default is not encoded on the class itself.
Do not add a second spelling for a mode that already has one.

``create_kan_key_window_attention`` and ``create_adaptive_softmax_window_attention``
are intentionally not public: not exported from ``attention/__init__.py``,
not registered in ``factory.py``, called only from
``tests/test_layers/test_attention/test_window_attention.py``. Either
configuration is reachable today by passing those kwargs to
``WindowAttention`` directly; do not register them to "fix the
inconsistency" -- the registry surface is frozen.
"""

# ---------------------------------------------------------------------


def create_grid_window_attention(
    dim: int, window_size: int, num_heads: int, **kwargs: Any
) -> WindowAttention:
    """
    Create a Swin-style grid window attention layer.

    Configures :class:`WindowAttention` for ``partition_mode='grid'``: the
    sequence is folded into a ``ceil(sqrt(N))``-square grid and cut into
    contiguous ``window_size x window_size`` tiles, so spatially adjacent
    tokens attend together. ``window_size`` is an edge length here. This is
    the ``'window'`` factory key.

    Relative position bias defaults to ``True``, which is standard for this
    architecture, and a caller can still pass ``False``.

    Architecture:

    .. code-block:: text

        (B, N, dim)
             ▼
        pad N to H*W, H = W = ceil(sqrt(N)) ► reshape to (B, H, W, dim)
             ▼
        pad H, W up to a multiple of window_size
             ▼
        contiguous ws x ws tiles ► (B*num_win, ws^2, dim)
             ▼
        SingleWindowAttention per window   (+ relative position bias)
             ▼
        merge tiles ► unpad ► slice to N ► (B, N, dim)

    See :class:`WindowAttention` for the full three-mode diagram and for the
    table comparing this key with ``'window_zigzag'`` and ``'window_band'``.

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
    Create a window attention layer with zigzag partitioning.

    Configures :class:`WindowAttention` for ``partition_mode='zigzag'``: the
    sequence is reordered along a 2-D zigzag path before windowing, which
    groups frequency-proximate tokens and gives a locality bias sensitive to
    frequency bands. ``window_size`` is an edge length here. This is the
    ``'window_zigzag'`` factory key.

    Relative position bias defaults to ``False``, because the zigzag scan
    breaks the spatial grid the bias is indexed by. A caller can still pass
    ``True``; the class accepts it on this path.

    Architecture:

    .. code-block:: text

        (B, N, dim)
             ▼
        pad N to N_grid = H*W, H = W = ceil(sqrt(N))   (stays 1-D)
             ▼
        take(x, zigzag_indices, axis=1)
             ▼
        pad to a multiple of ws^2 ► reshape (B*num_win, ws^2, dim)
             ▼
        SingleWindowAttention per window
             ▼
        merge ► slice to N_grid ► take(inverse_zigzag_indices)
             ▼
        slice to N ► (B, N, dim)

    See :class:`WindowAttention` for the full three-mode diagram and for the
    table comparing this key with ``'window'`` and ``'window_band'``.

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
    r"""
    Create a 1-D symmetric sliding-band attention layer.

    Configures :class:`WindowAttention` for ``partition_mode='band'``. Here
    ``window_size`` is a half-width in tokens: query ``i`` attends key ``j``
    iff ``abs(i - j) <= window_size``. The band is symmetric and non-causal,
    which is what a text ENCODER specifies. Upstream ModernBERT's
    ``local_attention`` is a FULL span, so the value to pass here is
    ``local_attention // 2``. There is no grid folding and no square padding.
    This is the ``'window_band'`` factory key.

    Relative position bias defaults to ``False``, and this layout REFUSES it
    outright: the bias is indexed by 2-D tile coordinates that a 1-D band does
    not have, so ``use_relative_position_bias=True`` raises.

    Architecture:

    .. code-block:: text

        (B, N, dim)
             ▼
        no grid fold, no square pad — stages 1-2 are skipped
             ▼
        keep[i, j] = abs(i - j) <= window_size   ► (1, N, N) int32
        AND-ed with the caller's mask, never substituted for it
             ▼
        SingleWindowAttention(pad_to_window=False) over all N tokens
        (no relative position bias on this path)
             ▼
        (B, N, dim)   — nothing to unpad

    See :class:`WindowAttention` for the full three-mode diagram and for the
    table comparing this key with ``'window'`` and ``'window_zigzag'``.

    Cost: a dense ``N x N`` banded mask is
    ``O(N^2)``, the same asymptotics as full attention. It is not the
    ``O(N * W)`` that "sliding window" suggests; that needs a fused kernel this
    repo has no path to. What the band buys over ``'grid'`` is that ``N`` real
    tokens are never inflated to ``window_size ** 2`` slots. To measure::

        .venv/bin/python -c "import resource, numpy as np; \
        from dl_techniques.layers.attention.window_attention import \
        create_band_window_attention as f; x = np.zeros((1, 512, 64), 'float32'); \
        f(dim=64, window_size=64, num_heads=4)(x); \
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)"

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The band half-width, in tokens. ``window_size=64``
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
    Create a window attention layer with a non-linear KAN Key projection.

    **Intentionally non-public**: not exported from ``attention/__init__.py``
    and not registered in ``attention/factory.py``. Its only callers are in
    ``tests/test_layers/test_attention/test_window_attention.py``. See the
    module-level "Public vs. internal surface" note above.

    Configures :class:`WindowAttention` to project the Key tensor with a
    ``KANLinear`` layer instead of a dense one, which allows a more expressive
    similarity match. Only the Key projection changes; the partitioning is
    whatever ``partition_mode`` selects.

    No diagram here: the data flow is :class:`WindowAttention`'s Architecture
    Overview unchanged.

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
    Create a window attention layer with adaptive temperature softmax.

    **Intentionally non-public**: not exported from ``attention/__init__.py``
    and not registered in ``attention/factory.py``. Its only callers are in
    ``tests/test_layers/test_attention/test_window_attention.py``. See the
    module-level "Public vs. internal surface" note above.

    Configures :class:`WindowAttention` with ``probability_type='adaptive'``,
    so scores become weights through an adaptive-temperature softmax. That can
    improve calibration by sharpening or flattening the distribution with the
    model's confidence. Only the probability step changes.

    No diagram here: the data flow is :class:`WindowAttention`'s Architecture
    Overview unchanged.

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
