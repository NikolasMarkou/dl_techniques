"""
Unified single-window multi-head self-attention.

Multi-head self-attention over ONE window, with two things made configurable
instead of forked into separate classes: how ``K`` is projected, and how raw
scores become probabilities. :class:`WindowAttention` is the partitioning
wrapper that feeds this layer; this layer owns all the attention math.

Sizing depends on how the caller drives it, and there are THREE regimes, not
one. By default the layer pads its input up to ``window_size ** 2`` tokens and
strips the padding off the output, so a caller never has to pre-pad and a
partial trailing window needs no separate code path. A caller that already
knows the layout can instead call ``set_window_slots()``, and the ``N_actual``
real tokens then ARE the window -- no internal padding, and the
relative-position bias is gathered at the slots given. A caller with no tile
at all passes ``pad_to_window=False``, which attends the ``N_actual`` tokens
with no padding and no bias; that is the mode
``WindowAttention(partition_mode='band')`` uses.

Architecture:
    1.  **Projection.** Two mutually exclusive modes:

        -   ``'linear'``: one fused dense layer produces ``Q``, ``K``, ``V``.
        -   ``'kan_key'``: separate dense layers produce ``Q`` and ``V``,
            while ``K`` comes from a KAN linear layer, which injects a
            non-linear key projection.

    2.  **Sizing.** One of the three regimes above. Whichever runs, an
        internal padding mask is ALWAYS built and ALWAYS applied, so there is
        no "no mask" path through this layer.

    3.  **Scoring.** ``qk_norm_type`` optionally normalizes ``Q`` and ``K``
        before the score matmul, which stabilises the logits. An optional
        learnable relative position bias, indexed by intra-window 2-D
        coordinates in the Swin convention, is added to the scores. The
        scores are then clipped to ``[-30, 30]`` BEFORE the mask bias is
        applied; the D-010 anchor in ``call()`` says why that order matters.

    4.  **Probability output.** Scores become weights through
        :class:`ProbabilityOutput`, selected by ``probability_type`` /
        ``probability_config``. Score-level routing strategies (``routing``,
        ``deterministic_routing``, ``hierarchical``, ``hierarchical_routing``)
        are rejected at construction time. They are not normalizations of raw
        attention scores.

Foundational Mathematics:
    The layer computes scaled dot-product attention::

        Attention(Q, K, V) = prob( Q K^T / sqrt(d_k) + bias ) V

    where ``prob`` is the configurable :class:`ProbabilityOutput` strategy and
    ``bias`` is the optional relative position bias. Restricting the sum to
    one window of ``M = window_size ** 2`` tokens is what turns global
    attention's ``O(N^2)`` into ``O(N * M)`` for a sequence of ``N`` tokens.

    The bias table is indexed by RELATIVE coordinate, not absolute position,
    so it holds ``(2 * window_size - 1) ** 2`` rows -- one per possible 2-D
    displacement inside the tile -- times ``num_heads``, instead of one entry
    per absolute position pair. That is ~1.04 MB even at ``window_size=128``.
    The INDEX that reads those rows is the expensive part, at
    ``O(window_size ** 4)``, which is why it is built lazily and cached rather
    than allocated in ``__init__``. See the D-006 anchor there.

References:
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer
      using Shifted Windows. (https://arxiv.org/abs/2103.14030)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)

"""

# ---------------------------------------------------------------------

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import apply_attention_mask
from ..ffn.kan_linear import KANLinear
from ..activations import ProbabilityOutput
from ..norms.factory import create_normalization_layer
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.attention.single_window_attention")
class SingleWindowAttention(keras.layers.Layer):
    """
    Unified multi-head self-attention for a single window.

    Merges several attention variants into one configurable layer: a standard
    linear QKV projection or a non-linear KAN-based Key projection, combined
    with a pluggable probability output applied to the attention scores (via
    :class:`ProbabilityOutput`).

    It computes ``Attention(Q, K, V) = prob(Q K^T / sqrt(d_k) + bias) V``,
    where ``prob`` is the configured :class:`ProbabilityOutput` and ``bias`` is
    an optional learnable relative position bias, indexed by intra-window 2-D
    coordinates in the Swin convention.

    **Three sizing regimes, not one.** By default the layer pads the input up
    to ``window_size ** 2`` tokens and strips the padding off the output. A
    caller that supplies a slot map through :meth:`set_window_slots` instead
    attends the ``N_actual`` real tokens and gathers the bias at those slots.
    A caller passing ``pad_to_window=False`` attends the ``N_actual`` real
    tokens with no tile at all, and must have the bias switched off. The two
    non-default spellings are mutually exclusive and are checked as such.

    Whichever regime runs, an internal padding mask is ALWAYS built and ALWAYS
    applied. There is no unmasked path through this layer, which is why the
    mask bias line here had no safe fallback under ``mixed_float16``; the
    D-007 anchor in :meth:`call` records what that cost.

    **Architecture Overview:**

    .. code-block:: text

        Input  (B, N_actual, dim)
              │
              ▼
        ┌────────────────────────────────────────────────────┐
        │ SIZE THE WINDOW — one of three, then build the     │
        │ internal padding mask (always)                     │
        │   default          pad to N_target = ws^2          │
        │   window_slots     N_target = len(slots), no pad   │
        │   pad_to_window=F  N_target = N_actual,   no pad   │
        └────────────────────────────────────────────────────┘
              ▼
        ┌────────────────────────────────────────────────────┐
        │ QKV PROJECTION                                     │
        │   'linear'  : fused Dense(3*dim)                   │
        │   'kan_key' : Dense(Q) + KANLinear(K) + Dense(V)   │
        │   reshape ► (B, heads, N_target, head_dim) each    │
        └────────────────────────────────────────────────────┘
              ▼
        [q_norm / k_norm]  (optional, qk_norm_type)
              ▼
        scores = (Q * scale) @ K^T        (B, heads, N, N)
              ▼
        [+ relative position bias]  (optional, gathered at the
                                     full index or at the slots)
              ▼
        clip(scores, -30, 30)       ON THE RAW SCORES — see D-010
              ▼
        ┌────────────────────────────────────────────────────┐
        │ MASK — internal padding mask, times the caller's   │
        │ mask if any.  rank-2 (B, N) ► (B, 1, 1, N)         │
        │                rank-3 (B, N, N) ► (B, 1, N, N)     │
        │ Fully-masked slices are rescued, not left as -inf. │
        └────────────────────────────────────────────────────┘
              ▼
        ProbabilityOutput ► [dropout] ► weights @ V
              ▼
        transpose ► reshape ► output Dense projection
              ▼
        slice [:, :N_actual, :]      (a no-op when nothing was padded)
              ▼
        Output (B, N_actual, dim)
    :param dim: Total model dimension (split across heads). Must be positive
        and divisible by ``num_heads``.
    :type dim: int
    :param window_size: Height/width of the square attention window. The
        layer pads inputs up to ``window_size ** 2`` tokens.
    :type window_size: int
    :param num_heads: Number of attention heads. Must divide ``dim``.
    :type num_heads: int
    :param attention_mode: Projection mode. ``'linear'`` for standard
        dense QKV or ``'kan_key'`` for a KAN-based Key projection.
        Defaults to ``'linear'``.
    :type attention_mode: str
    :param probability_type: Probability strategy identifier forwarded to
        :class:`ProbabilityOutput` for converting attention scores into
        attention weights. Defaults to ``'softmax'``. Score-level routing
        strategies (``'routing'``, ``'deterministic_routing'``,
        ``'hierarchical'``, ``'hierarchical_routing'``) are not allowed.
    :type probability_type: str
    :param probability_config: Optional configuration dictionary forwarded
        to :class:`ProbabilityOutput` as its ``type_config`` argument.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied independently
        to ``Q`` and ``K`` before computing attention scores. When provided,
        normalization layers are constructed via
        :func:`create_normalization_layer`. Defaults to ``None`` (no QK-norm).
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when ``qk_norm_type`` is set.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param use_relative_position_bias: Whether to add a learnable relative
        position bias to attention scores. Defaults to ``True``.
    :type use_relative_position_bias: bool
    :param qkv_bias: Whether the fused QKV dense uses bias (linear mode).
        Defaults to ``True``.
    :type qkv_bias: bool
    :param qk_scale: Override for the QK scaling factor. If ``None``,
        defaults to ``head_dim ** -0.5``.
    :type qk_scale: Optional[float]
    :param dropout_rate: Dropout rate applied to attention weights. Must be
        between 0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param proj_bias: Whether the output projection uses bias.
        Defaults to ``True``.
    :type proj_bias: bool
    :param kan_grid_size: Grid size for the KAN layer (``kan_key`` mode).
        Defaults to 5.
    :type kan_grid_size: int
    :param kan_spline_order: Spline order for the KAN layer.
        Defaults to 3.
    :type kan_spline_order: int
    :param kan_activation: Activation for the KAN layer.
        Defaults to ``'swish'``.
    :type kan_activation: str
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
        Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments forwarded to the base Layer.

    :raises ValueError: If ``attention_mode`` is not one of
        ``{'linear', 'kan_key'}`` or if ``probability_type`` is a score-level
        routing strategy (``'routing'``, ``'deterministic_routing'``,
        ``'hierarchical'``, ``'hierarchical_routing'``).
    """

    def __init__(
            self,
            dim: int,
            window_size: int,
            num_heads: int,
            attention_mode: str = "linear",
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

        Only the mode string and the probability type are checked here; the
        shape-dependent arguments are validated in :meth:`build`, where the
        input width is known. The softmax temperature is resolved once, taking
        ``qk_scale`` when given and ``head_dim ** -0.5`` otherwise. The
        projections and the relative-position bias table are created in
        :meth:`build`. See the class docstring for the parameter reference.

        :raises ValueError: If ``attention_mode`` is not one of ``"linear"`` or
            ``"kan_key"``, or if ``probability_type`` names a score-level
            routing or hierarchical variant.
        """
        super().__init__(**kwargs)

        # Validate inputs
        valid_modes = {"linear", "kan_key"}
        if attention_mode not in valid_modes:
            raise ValueError(
                f"Invalid attention_mode. Expected one of {valid_modes}, "
                f"got '{attention_mode}'"
            )
        invalid_prob_types = {
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        }
        if probability_type in invalid_prob_types:
            raise ValueError(
                f"Invalid probability_type '{probability_type}'. Score-level "
                f"routing strategies {invalid_prob_types} are not allowed for "
                f"SingleWindowAttention."
            )

        # Store ALL configuration parameters
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (
            qk_scale if qk_scale is not None else self.head_dim ** -0.5
        )
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
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE sub-layers based on configuration
        if self.attention_mode == "linear":
            self.qkv = keras.layers.Dense(
                self.dim * 3,
                use_bias=self.qkv_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="qkv",
            )
        elif self.attention_mode == "kan_key":
            self.query = keras.layers.Dense(
                self.dim, use_bias=False, name="query"
            )
            self.key = KANLinear(
                features=self.dim,
                grid_size=self.kan_grid_size,
                spline_order=self.kan_spline_order,
                activation=self.kan_activation,
                name="key_kan",
            )
            self.value = keras.layers.Dense(
                self.dim, use_bias=False, name="value"
            )

        self.proj = keras.layers.Dense(
            self.dim,
            use_bias=self.proj_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj",
        )
        self.attn_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="attn_dropout")
            if self.dropout_rate > 0.0
            else None
        )

        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type,
                name="q_norm",
                **(self.qk_norm_kwargs or {}),
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type,
                name="k_norm",
                **(self.qk_norm_kwargs or {}),
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-006 — the
        # `(window_size**2, window_size**2)` relative-position INDEX is built on
        # first use by `_relative_position_index()`, per slot vector, and cached.
        # It is O(window_size**4): MEASURED, building it in `__init__` took
        # construction peak RSS at `window_size=128` from 0.62 GB to 5.64 GB.
        #   * Do NOT move it back into `__init__` — constructing a layer must not
        #     allocate O(W^4).
        #   * Do NOT make it an `add_weight(zeros)` + `.assign()` in `build()`:
        #     under `StatelessScope` the assign is DISCARDED and the tensor stays
        #     ALL ZEROS. It is a CONSTANT, not a weight.
        #   * Do NOT build it with `keras.ops` — a trace-owned graph tensor is
        #     unusable from any other trace; numpy has no graph affinity.
        #   * Do NOT build the FULL `(W**2, W**2)` index in `build()` either; a
        #     `window_slots` caller needs only the `(n, n)` sub-index.
        # See decisions.md D-006 and D-007 (plan-2026-08-25T053412-0f1fa04f).
        self.relative_position_index = None
        self._relative_position_index_cache: Dict[bytes, np.ndarray] = {}

        # DECISION plan-2026-08-25T053412-0f1fa04f/D-015 — the window-slot map lives
        # on the INSTANCE, set by `set_window_slots()` just before a call, never as a
        # `call()` kwarg: `Layer.__call__` maps `convert_input` over every argument,
        # so an ndarray kwarg becomes a symbolic tensor under trace and `np.asarray`
        # raises `NotImplementedError: Cannot convert a symbolic tf.Tensor`.
        #   * Do NOT put `window_slots` back in the `call()` signature. MEASURED on
        #     b380c6f79 for `1 < N < window_size ** 2` in BOTH 'grid' and 'zigzag':
        #     eager `layer(x)`/`model(x)` PASS while `predict`, `fit` and any
        #     `@tf.function` raise — every functional consumer broken.
        #   * Do NOT "fix" that by catching `NotImplementedError` or by an
        #     `isinstance(x, np.ndarray)` test inside `call()`; that silently SKIPS
        #     the range and length validation below.
        #   * Do NOT make the callee tensor-agnostic instead — the `(n, n)`
        #     sub-index is CACHED by `slots.tobytes()`, and a symbolic vector has
        #     no bytes.
        #   * Do NOT skip the `finally`-reset in `_attend()`; a stale slot map is
        #     invisible — right dtype, plausible length.
        # See decisions.md D-015 (plan-2026-08-25T053412-0f1fa04f).
        self._window_slots: Optional[np.ndarray] = None

        # DECISION plan-2026-08-27T040114-580f8b63/D-012 — one entry per sequence
        # length: the slot map this instance committed to at that length.
        # `set_window_slots()` reads it to refuse a second, different layout at the
        # same length, which a reused trace would silently ignore. Not serialized.
        # See decisions.md D-012 (plan-2026-08-27T040114-580f8b63).
        self._slot_layout_by_length: Dict[int, bytes] = {}

    def set_window_slots(
            self, window_slots: Optional[np.ndarray]
    ) -> None:
        """Set (or clear) the window-slot map used by the NEXT call.

        Contract: the caller sets this immediately before invoking the layer and
        clears it (``None``) immediately after, in a ``finally``; see
        :meth:`WindowAttention._attend`, which is the only supported caller.
        Off the ``call()`` argument channel on purpose -- see the D-015 anchor in
        ``__init__``.

        .. warning::

           The slot map is read as PYTHON state inside ``call()``, so it is baked
           into a ``tf.function`` trace at trace time and a graph is NOT retraced
           when it changes. Setting a DIFFERENT slot map at the SAME sequence
           length therefore returns the FIRST trace's answer, silently. Measured
           at ``window_size=4``, ``N=6``, two slot maps A and B::

               eager   |A - B|            = 1.624966e-02
               traced  |A - B|            = 0.0            <- stale
               traced B vs eager A        = 0.0            <- returns A's answer
               traced B vs eager B        = 1.624966e-02

           This is why the raise below exists rather than a docstring note.

        :param window_slots: ``(N_actual,)`` int32-coercible array naming, for
            each input token, its row-major slot inside the ``window_size x
            window_size`` tile, or ``None`` for the pad-to-``window_size ** 2``
            behaviour. Validated on use, not here.
        :type window_slots: Optional[np.ndarray]
        :raises ValueError: If a slot map is set that differs from one already
            used at the SAME length on this instance. See the note above.
        :return: ``None``.
        :rtype: None
        """
        if window_slots is None:
            self._window_slots = None
            return

        slots = np.asarray(window_slots, dtype=np.int32)

        # DECISION plan-2026-08-27T040114-580f8b63/D-012 — one instance at one `N`
        # has ONE legal slot map (D-015); two maps at one length are misuse, so this
        # raises rather than return a stale trace's answer. Both alternatives are
        # closed: `window_slots` back on `call()` is D-015's measured NO, and an
        # in-`call()` comparison cannot work. See decisions.md D-012.
        previous = self._slot_layout_by_length.get(int(slots.shape[0]))
        if previous is not None and previous != slots.tobytes():
            raise ValueError(
                f"set_window_slots() was given a different slot map for length "
                f"{int(slots.shape[0])} than the one this layer has already used. "
                "The slot map is read as Python state inside call(), so it is "
                "baked into a tf.function trace; a graph traced with the earlier "
                "map is reused unchanged and would silently return that map's "
                "answer. The slot map is fully determined by (partition_mode, "
                "window_size, N), so one instance at one length has exactly one "
                "legal map -- use a separate layer instance per layout."
            )
        self._slot_layout_by_length[int(slots.shape[0])] = slots.tobytes()
        self._window_slots = slots

    def _relative_position_index(
            self, window_slots: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Build (and cache) the relative-position index for a set of window slots.

        Each entry ``index[i, j]`` is the row into the
        ``(2 * window_size - 1) ** 2`` relative-position bias table that encodes
        the 2-D displacement from slot ``window_slots[j]`` to slot
        ``window_slots[i]``, where a slot is a row-major position in a
        ``window_size x window_size`` tile.

        This is the same pairwise-difference computation the Swin reference
        implementation performs (``coords_flatten[:, :, None] -
        coords_flatten[:, None, :]``, then row-major encode) -- it is NOT a closed
        form, because upstream has none. Two things differ, both for memory and
        neither for value. (a) The two coordinate axes are differenced SEPARATELY
        and accumulated in place instead of being stacked into a
        ``(2, n, n)`` tensor that is then transposed, sliced twice and
        ``astype``-copied; MEASURED at ``window_size=128`` this is 2.0 GB of
        transients instead of 6.0 GB. (b) The slot set is a PARAMETER, so a caller
        attending ``n < window_size ** 2`` real tokens pays ``O(n^2)`` rather than
        ``O(window_size ** 4)``.

        :param window_slots: Row-major slot ids inside the window, shape ``(n,)``,
            or ``None`` for the full window (``arange(window_size ** 2)``).
        :type window_slots: Optional[np.ndarray]
        :return: Index matrix of shape ``(n, n)``, dtype ``int32``, every entry in
            ``[0, (2 * window_size - 1) ** 2)``.
        :rtype: np.ndarray
        :raises RuntimeError: If the constructed index has the wrong shape, or is
            all zeros at ``window_size > 1``.
        """
        ws = self.window_size
        cache_key = (
            b"full" if window_slots is None else window_slots.tobytes()
        )
        cached = self._relative_position_index_cache.get(cache_key)
        if cached is not None:
            return cached

        slots = (
            np.arange(ws * ws, dtype=np.int32)
            if window_slots is None
            else np.asarray(window_slots, dtype=np.int32)
        )
        # Row-major slot -> (row, column) inside the window.
        coords_h = slots // ws
        coords_w = slots % ws
        # (n, n): displacement along each axis, shifted to start at 0.
        index = coords_h[:, None] - coords_h[None, :]
        index += ws - 1
        index *= 2 * ws - 1
        relative_w = coords_w[:, None] - coords_w[None, :]
        relative_w += ws - 1
        index += relative_w

        # The constant is checked rather than assumed: a silently-zero or
        # wrongly-shaped index gathers row 0 of the bias table for every pair and
        # turns the relative-position bias into a per-head scalar -- a change no
        # shape assertion and no finiteness check can see.
        expected_shape = (slots.shape[0], slots.shape[0])
        if index.shape != expected_shape:
            raise RuntimeError(
                f"relative_position_index has shape {index.shape}, expected "
                f"{expected_shape}."
            )
        if ws > 1 and slots.shape[0] > 1 and not index.any():
            raise RuntimeError(
                f"relative_position_index is all zeros at window_size={ws} over "
                f"{slots.shape[0]} slots; only a single slot, or window_size=1, "
                "may produce an all-zero index."
            )

        self._relative_position_index_cache[cache_key] = index
        if window_slots is None:
            self.relative_position_index = index
        return index

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer weights and sub-layers.

        Allocates the learnable relative position bias table and explicitly
        builds the QKV / KAN / projection sub-layers against the *padded*
        per-window shape ``(B, window_size ** 2, dim)``. Normalization
        sub-layers are built with the full attention-score shape so they
        capture the correct last-axis dimensionality for serialization.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, dim)``. ``seq_len`` may be less than
            ``window_size ** 2`` since the layer pads internally.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        if self.use_relative_position_bias:
            # The INDEX is not built here -- see the D-006/D-007 anchor in
            # `__init__`. Only the (tiny) learnable table is a build-time
            # allocation: `(2W-1)**2 x heads`, ~1.04 MB even at W=128.
            num_relative_positions = (2 * self.window_size - 1) ** 2
            self.relative_position_bias_table = self.add_weight(
                name="relative_position_bias_table",
                shape=(num_relative_positions, self.num_heads),
                initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
                dtype=self.dtype,
            )

        # Sub-layers see the padded per-window shape.
        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size

        if self.attention_mode == "linear":
            self.qkv.build(padded_shape)
        else:
            self.query.build(padded_shape)
            self.key.build(padded_shape)
            self.value.build(padded_shape)
        self.proj.build(padded_shape)

        if self.attn_dropout is not None:
            self.attn_dropout.build(None)

        # Normalization layers act on attention scores: build with the
        # correct (B, heads, N, N) shape rather than None.
        num_tokens_in_window = self.window_size * self.window_size
        attention_scores_shape = (
            input_shape[0],
            self.num_heads,
            num_tokens_in_window,
            num_tokens_in_window,
        )

        self.attn_prob.build(attention_scores_shape)

        if self.q_norm is not None:
            qk_shape = (
                input_shape[0],
                self.num_heads,
                num_tokens_in_window,
                self.head_dim,
            )
            self.q_norm.build(qk_shape)
            self.k_norm.build(qk_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            pad_to_window: bool = True,
    ) -> keras.KerasTensor:
        """
        Forward pass for the unified single-window attention.

        Sizes the window, runs multi-head self-attention with optional
        relative position bias and configurable normalization, then slices any
        padding off the output. The internal padding mask is combined
        multiplicatively with any caller-supplied ``attention_mask`` before it
        becomes an additive ``MASK_BIAS_VALUE`` bias on the scores.

        The window-slot map is NOT an argument here. It is set on the instance
        by :meth:`set_window_slots` immediately before the call; the D-015
        anchor in ``__init__`` says why it must stay off the traced argument
        channel. When it is set it asserts *these are all the tokens there
        are*: the layer attends the ``N_actual`` real tokens with no internal
        padding and gathers the relative-position bias at those slots'
        coordinates. When it is ``None``, ``pad_to_window`` decides between
        the default pad-to-``window_size ** 2`` behaviour and no padding
        at all.

        :param inputs: Token embeddings of shape ``(B, N_actual, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional **keep predicate** (``1 = attend``),
            in one of two ranks:

            * rank 2, ``(B, N_actual)`` — a KEY-only mask (1 for valid tokens,
              0 for padding), broadcast over every query row. This is the
              original contract and is unchanged.
            * rank 3, ``(B, N_actual, N_actual)`` — a PAIRWISE
              ``(query, key)`` mask: ``mask[b, q, k] == 1`` means query ``q``
              may attend to key ``k``. Required for shifted-window attention
              (SW-MSA), whose permitted keys genuinely depend on the query.

            Either rank is combined multiplicatively with the internal padding
            mask (over the KEY axis in both cases).
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating whether in training mode.
        :type training: Optional[bool]
        :param pad_to_window: If ``True`` (the default, and the only behaviour
            any pre-2026-08-25 caller had) the input is padded up to
            ``window_size ** 2`` slots, which is what a *tile* partition
            requires. Pass ``False`` to attend the ``N_actual`` real tokens with
            NO internal padding and no tile at all -- the mode
            ``WindowAttention(partition_mode='band')`` uses, where
            ``window_size`` is a 1-D half-width in tokens and
            ``window_size ** 2`` has no meaning. ``pad_to_window=False``
            requires ``use_relative_position_bias=False`` (the bias is indexed
            by 2-D tile coordinates that do not exist without a tile) and is
            mutually exclusive with ``window_slots``, which is the *tile-aware*
            spelling of the same "do not pad" instruction.
        :type pad_to_window: bool
        :return: Attended output of shape ``(B, N_actual, dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If a window-slot map is set and its length does not
            match a statically-known sequence length, or any slot is outside
            ``[0, window_size ** 2)``; or if ``pad_to_window=False`` is combined
            with a slot map or with ``use_relative_position_bias=True``.
        """
        # Read once: the slot map is instance state set by `set_window_slots()`
        # right before this call, NOT a traced argument. See D-015 in __init__.
        window_slots = self._window_slots
        input_shape = keras.ops.shape(inputs)
        B_actual, N_actual = input_shape[0], input_shape[1]
        if window_slots is None:
            # DECISION plan-2026-08-25T053412-0f1fa04f/D-010 — `pad_to_window=False`
            # is the ONLY way to run this layer without a `window_size x window_size`
            # tile. It exists for `WindowAttention(partition_mode='band')`, where
            # `window_size` is a 1-D half-width in TOKENS.
            #   * Do NOT express band mode as `window_slots=np.arange(N)`: slots are
            #     validated into `[0, window_size ** 2)` because they index a TILE,
            #     so at `window_size=64` any sequence over 4096 tokens raises.
            #   * Do NOT let `pad_to_window=False` coexist with a relative-position
            #     bias — without a grid the gathered rows are arbitrary, and an
            #     arbitrary learnable bias passes every shape/dtype/finiteness check.
            #     It is REFUSED here rather than forced off.
            #   * Do NOT add a second boolean meaning the same thing; this flag and
            #     `window_slots` are mutually exclusive by explicit check.
            # See decisions.md D-010 (plan-2026-08-25T053412-0f1fa04f).
            if pad_to_window:
                N_target = self.window_size * self.window_size
            else:
                if self.use_relative_position_bias:
                    raise ValueError(
                        "SingleWindowAttention(pad_to_window=False) requires "
                        "use_relative_position_bias=False: the relative-position "
                        "bias is indexed by 2-D coordinates inside a "
                        "window_size x window_size tile, and pad_to_window=False "
                        "means there is no tile. Construct the layer with "
                        "use_relative_position_bias=False."
                    )
                N_target = N_actual
        else:
            if not pad_to_window:
                raise ValueError(
                    "SingleWindowAttention received both a window-slot map "
                    "(set_window_slots) and pad_to_window=False. Both mean 'do "
                    "not pad internally'; the slot map is the tile-aware "
                    "spelling (it also selects the relative-position rows), "
                    "pad_to_window=False is the layout-free one. Use exactly one."
                )
            static_n = inputs.shape[1]
            if static_n is not None and int(static_n) != int(
                    window_slots.shape[0]
            ):
                raise ValueError(
                    f"window_slots has length {int(window_slots.shape[0])} but "
                    f"the input sequence length is {int(static_n)}. It must name "
                    "one window slot per input token."
                )
            if window_slots.min() < 0 or window_slots.max() >= (
                    self.window_size * self.window_size
            ):
                raise ValueError(
                    f"window_slots values must lie in "
                    f"[0, {self.window_size ** 2}) for window_size="
                    f"{self.window_size}; got range "
                    f"[{int(window_slots.min())}, {int(window_slots.max())}]."
                )
            # No internal padding: the real tokens ARE the window.
            N_target = int(window_slots.shape[0])

        padding_amount = N_target - N_actual
        # Shape: (B, N_actual, dim) -> (B, N_target, dim), N_target = window_size**2
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, padding_amount], [0, 0]]
        )

        # Shape: (B, N_actual) + (B, padding_amount) -> (B, N_target)
        internal_padding_mask = keras.ops.concatenate(
            [
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32"),
            ],
            axis=1,
        )

        # DECISION plan-2026-07-31T042809-ddc92265/D-001 — a rank-3
        # `(B, N_actual, N_actual)` mask is a PAIRWISE (query, key) keep predicate
        # and gets its OWN branch. It is not a variant spelling of the rank-2
        # `(B, N_actual)` key-only mask below. Shifted-window self-attention
        # (SW-MSA) cannot be expressed any other way: after Swin's cyclic roll one
        # physical window holds tokens from up to four pre-roll regions, and which
        # keys a query may see depends on the QUERY's region. The rank-2 contract
        # broadcasts one key predicate over every query row and is structurally
        # incapable of saying that.
        #
        #   * Do NOT collapse this back into the rank-2 `reshape(..., (B,1,1,N))`
        #     path, e.g. by reducing the pairwise mask with `any(mask, axis=-2)`.
        #     That reduction is exactly the information loss the branch avoids: for
        #     an SW-MSA mask every key is kept by SOME query, so it yields all-ones
        #     and masks NOTHING, with no shape error and a finite output.
        #   * Do NOT invert the polarity "to match" some other site. This is a
        #     `1 = attend` predicate handed to the shared mask helper, which
        #     performs no polarity inference (its own docstring says so). An
        #     inversion raises nothing, changes no shape and stays finite, and it
        #     makes the layer attend to exactly the pairs it was told to forbid.
        #     `TestSingleWindowAttentionPairwiseMask` is the only guard that sees
        #     it, and it measures the inverted control in the same test.
        #   * Do NOT mask the QUERY axis with the internal padding mask here. It is
        #     combined as `(B, 1, N_target)` — key-only — exactly as the rank-2
        #     branch does, so the two branches agree BIT-for-BIT on a mask that is
        #     constant over queries (pinned by
        #     `test_a_query_broadcast_rank_3_mask_equals_the_rank_2_mask`).
        #
        # NOTE what that does and does NOT avoid, because an earlier version of this
        # comment claimed more than is true. It keeps the INTERNAL padding mask off
        # the query axis. It does NOT keep the query axis unmasked, because the
        # `ops.pad` below zero-pads the USER's rank-3 mask on BOTH axes. So whenever
        # `N_actual < N_target` the padded query rows ARE fully masked and the
        # helper's fully-masked-row rescue DOES fire on them. MEASURED at
        # `(window_size=4, N_actual=10, B=2)`: 12 of 32 query slices keep nothing on
        # this branch, versus 0 on the rank-2 branch and 0 with no mask at all. It
        # is harmless — those rows' outputs are sliced off unread below, and the
        # rescue keeps them finite rather than NaN — and it is unreachable from the
        # Swin caller, which always passes `N == window_size ** 2` exactly
        # (measured: 0 of 32 there).
        #
        # ACCEPTED COST: the predicate handed to the mask helper is `(B, 1, N, N)`
        # rather than `(B, 1, 1, N)`, i.e. O(N^2) mask memory on this path only. The
        # rank-2 and `None` paths are untouched.
        # See decisions.md D-001 (plan-2026-07-31T042809-ddc92265).
        user_mask_is_pairwise = (
                attention_mask is not None and len(attention_mask.shape) == 3
        )

        final_attention_mask = internal_padding_mask
        if user_mask_is_pairwise:
            # Shape: (B, N_actual, N_actual) -> (B, N_target, N_target)
            padded_user_mask = keras.ops.pad(
                attention_mask,
                [[0, 0], [0, padding_amount], [0, padding_amount]],
            )
            # Shape: (B, N_target, N_target) * (B, 1, N_target) -> (B, N_t, N_t)
            final_attention_mask = (
                    keras.ops.cast(padded_user_mask, "int32")
                    * keras.ops.expand_dims(internal_padding_mask, axis=1)
            )
        elif attention_mask is not None:
            # Shape: (B, N_actual) -> (B, N_target)
            padded_user_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, padding_amount]]
            )
            final_attention_mask = (
                    keras.ops.cast(padded_user_mask, "int32")
                    * internal_padding_mask
            )

        B, N, C = keras.ops.shape(padded_inputs)
        if self.attention_mode == "linear":
            # Shape: (B, N, dim) -> (B, N, 3*dim)
            qkv = self.qkv(padded_inputs, training=training)
            # Shape: (B, N, 3*dim) -> (B, N, 3, H, head_dim)
            qkv = keras.ops.reshape(
                qkv, (B, N, 3, self.num_heads, self.head_dim)
            )
            # Shape: (B, N, 3, H, head_dim) -> (3, B, H, N, head_dim)
            qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
            # Shape: (3, B, H, N, head_dim) -> 3x (B, H, N, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q_proj = self.query(padded_inputs, training=training)
            k_proj = self.key(padded_inputs, training=training)
            v_proj = self.value(padded_inputs, training=training)
            # Shape: (B, N, dim) -> (B, N, H, head_dim) -> (B, H, N, head_dim), each
            q = keras.ops.transpose(
                keras.ops.reshape(q_proj, (B, N, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )
            k = keras.ops.transpose(
                keras.ops.reshape(k_proj, (B, N, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )
            v = keras.ops.transpose(
                keras.ops.reshape(v_proj, (B, N, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )

        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        q = q * self.scale
        # Shape: (B, H, N, head_dim) @ (B, H, head_dim, N) -> (B, H, N, N)
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

        if self.use_relative_position_bias:
            # Shape: table (2*Ws-1)^2 x H, gathered by (N_target*N_target,) index
            #        -> (N_target*N_target, H). The index is the FULL
            #        (Ws**2, Ws**2) one when the layer padded internally, and the
            #        (N_target, N_target) sub-index at the caller's slots when it
            #        did not -- the SAME table rows either way, so the bias a real
            #        token pair receives does not depend on which path ran.
            relative_position_index = self._relative_position_index(window_slots)
            relative_position_bias = keras.ops.take(
                self.relative_position_bias_table,
                keras.ops.reshape(relative_position_index, (-1,)),
                axis=0,
            )
            # Shape: (N_target*N_target, H) -> (N_target, N_target, H)
            relative_position_bias = keras.ops.reshape(
                relative_position_bias, (N_target, N_target, -1)
            )
            # Shape: (N_target, N_target, H) -> (H, N_target, N_target)
            relative_position_bias = keras.ops.transpose(
                relative_position_bias, (2, 0, 1)
            )
            # Shape: (B, H, N, N) + (1, H, N, N) -> (B, H, N, N)
            attn = attn + keras.ops.expand_dims(relative_position_bias, 0)

        # DECISION plan-2026-07-27T183600-b4ef45f0/D-010 — `clip(attn, -30, 30)` runs
        # HERE, on the RAW scores. Do NOT move it back below the mask bias (clipping
        # BIASED logits floors a masked position at `-30`, so the mask stops masking:
        # MEASURED leak 0.0 at bias 0.0, 2.73e-04 at -20, 0.439 at -50, against a
        # kept-token signal of 32.6), and do NOT delete it. See decisions.md D-010.
        attn = keras.ops.clip(attn, -30.0, 30.0)

        # THIS SITE'S MASK POLARITY, passed through verbatim: `broadcast_mask` is a
        # `1 = keep` predicate, so it IS the keep predicate the shared mask helper
        # wants. Do NOT "normalize" it into a `> 0` comparison and do NOT invert it.
        # The helper performs no polarity inference by design, so an inversion
        # raises nothing, changes no shape and stays finite — the layer would just
        # attend to the padding. `TestSingleWindowAttentionMaskPolarity` is the only
        # guard that can see it.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-007 — `out_dtype` is pinned to
        # the SCORES' own dtype. The ARITHMETIC form this replaces,
        # `attn + (1.0 - keep) * -1e9`, is `0 * -inf = NaN` at every UNMASKED
        # position in float16, and this site had NO safe path because the mask is
        # unconditional: MEASURED at (B=2, N=64, dim=64, window_size=8, num_heads=4)
        # under `mixed_float16`, 8192/8192 NaN with no mask, an all-ones mask, right
        # padding and left padding; float32 gave 0/8192. Do NOT use
        # `out_dtype=None`; `attn_prob` autocasts float32 back to float16.
        # See decisions.md D-007 (plan-2026-07-27T183600-b4ef45f0).
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-009 — the fully-masked-slice
        # rescue arrives via `apply_attention_mask`'s DEFAULT `rescue_axis=-1`, so
        # an all-`-inf` row is never FORMED and no NaN gradient is created. The mask
        # has no QUERY axis here, so "keeps nothing" means "this batch element keeps
        # no key at all". Do NOT pass `rescue_axis=None` to "get the loud NaN back":
        # that also restores the NaN GRADIENT.
        # See decisions.md D-009 and D-008 (plan-2026-07-27T183600-b4ef45f0).
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-017 — the softmax axis is
        # DERIVED from this layer's own `probability_config`, not the helper's `-1`
        # default; a caller can move the reduction axis. MEASURED at the sibling
        # `gated_attention` under `mixed_float16` with `{"axis": -2}` and a dead KEY
        # COLUMN: 8192/8192 non-finite. Because this site's mask is ALWAYS reshaped
        # to `(B, 1, 1, N)`, `axis=-2` now gets a named `ValueError` rather than a
        # mask that cannot mask. Do NOT restore a bare `-1`, and do NOT read this as
        # the rank/shape INFERENCE `common.py` forbids.
        # See decisions.md D-017 (plan-2026-07-27T183600-b4ef45f0).
        if user_mask_is_pairwise:
            broadcast_mask = keras.ops.reshape(
                final_attention_mask, (B, 1, N, N)
            )
        else:
            broadcast_mask = keras.ops.reshape(
                final_attention_mask, (B, 1, 1, N)
            )
        attn = apply_attention_mask(
            attn,
            broadcast_mask,
            # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`:
            # a Keras-2 residue banned across `src/`, and `str` alone mis-renders a
            # `tf.DType`. Full note and the measured equivalence at `common.py`; D-007.
            out_dtype=(getattr(attn.dtype, "name", None) or str(attn.dtype)),
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

        attn = self.attn_prob(attn, training=training)

        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)
        # Shape: (B, H, N, N) @ (B, H, N, head_dim) -> (B, H, N, head_dim)
        x = keras.ops.matmul(attn, v)
        # Shape: (B, H, N, head_dim) -> (B, N, H, head_dim)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        # Shape: (B, N, H, head_dim) -> (B, N, dim)
        x = keras.ops.reshape(x, (B, N, C))
        # Shape: (B, N, dim) -> (B, N, dim)
        x = self.proj(x, training=training)

        # Shape: (B, N_target, dim) -> (B, N_actual, dim)  [strip stage-1 padding]
        output = x[:, :N_actual, :]
        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, identical to the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
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
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config

# ---------------------------------------------------------------------
