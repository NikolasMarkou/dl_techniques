"""Progressive Focused Attention (PFA), built by
:class:`ProgressiveFocusedAttention`.

Each layer is a Swin-style windowed attention block, but before the softmax
the raw scores are multiplied elementwise by the previous layer's
post-softmax attention weights::

    scores = (Q K^T / sqrt(d_head) + SW-MSA mask) * prev_attn_map

This runs in the logit domain, not the probability domain: a connection an
earlier layer suppressed is pulled toward the softmax's uniform point rather
than masked to ``-inf``, and can recover if the current layer's raw score is
large. ``call()`` returns a tuple ``(output, attention_weights)``; the second
element is what the next block passes in as ``prev_attn_map``.

The sparse attention path (``sparsity_mode='top_k'`` or ``'threshold'``) is a
documented stub: construction raises ``NotImplementedError`` for any mode
other than ``'none'``. ``shift_size > 0`` (SW-MSA) requires a statically
known height and width; ``build()`` raises otherwise.

References:
    - Long et al., 2025. Progressive Focused Transformer for Single Image
      Super-Resolution. (CVPR)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer
      using Shifted Windows. (https://arxiv.org/abs/2103.14030)
"""

import keras
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any, Literal

from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.keras_registration import register_dl_technique

SparsityMode = Literal['none', 'top_k', 'threshold']


@register_dl_technique("dl_techniques.layers.attention.progressive_focused_attention")
class ProgressiveFocusedAttention(keras.layers.Layer):
    """Windowed self-attention with progressive focusing from the previous layer's map.

    Partitions the input into non-overlapping ``window_size x window_size``
    windows and runs scaled dot-product attention inside each, with optional
    locally-enhanced positional encoding (LePE) via a depthwise convolution
    on the values. Before the softmax, the raw scores are multiplied
    elementwise by ``prev_attn_map`` (identity when ``None``, the first
    layer). Shifted windows (SW-MSA) add a cyclic roll and an additive mask
    so information crosses window boundaries in alternating layers.

    Architecture:

    .. code-block:: text

        x [B,H,W,C]
             │
             ▼
        cyclic roll  ('shift_size' only)
             │
             ▼
        window partition + flatten  [B*nW, ws², C]
             │
             ▼
        ┌──────────────────────────┐
        │ qkv Dense(3C) -> Q,K,V   │
        │ optional qk-norm         │
        │ optional LePE on V       │
        │ scores = QKᵀ*scale       │
        │  + SW-MSA mask           │
        │  * prev_attn_map         │  ◄── prev_attn_map (optional)
        │ weights = probability()  │
        │ out = weights @ V        │
        │  -> proj                 │
        └─────────────┬────────────┘
                       ▼
        window reverse, roll back
                       │
                       ▼
        output [B,H,W,C]   attn_weights [B*nW,heads,ws²,ws²]
                                    │
                          (-> next layer's prev_attn_map)

    ``call()`` returns a tuple ``(output, attention_weights)`` and
    ``compute_output_shape()`` returns a matching tuple of two shapes; the
    second element is what the next block consumes as ``prev_attn_map``.

    Constructor arguments are stored under leading-underscore private names
    (``self._dim``, ``self._num_heads``, and so on) except the four
    later-added arguments (``probability_type``, ``probability_config``,
    ``qk_norm_type``, ``qk_norm_kwargs``), which use the public spelling.
    The private names are read by external callers and tests, so this pass
    leaves them as-is; ``get_config()`` already emits the public spelling
    for every key, so serialization is unaffected.

    :meth:`_window_partition` and :meth:`_window_reverse` are private copies
    of the free functions in
    :mod:`~dl_techniques.layers.attention.window_attention`, kept separate
    because they close over ``self._window_size`` and because
    :meth:`_compute_attention_mask` depends on their exact reshape/transpose
    order — a merge would have to prove the two orders stay identical.

    :param dim: Embedding dimension (number of channels).
    :type dim: int
    :param num_heads: Number of attention heads. Must divide ``dim`` evenly.
    :type num_heads: int
    :param window_size: Size of the attention window.
    :type window_size: int
    :param shift_size: Shift size for SW-MSA. Use 0 for W-MSA,
        ``window_size // 2`` for shifted windows.
    :type shift_size: int
    :param top_k: Number of top-k tokens to attend to when
        ``sparsity_mode='top_k'``. ``None`` attends to all tokens.

        .. warning::
           The sparse attention path is not implemented. The ``'top_k'``
           and ``'threshold'`` sparsity modes are no-op stubs (the layer
           always performs dense attention). Constructing the layer with a
           non-default ``sparsity_mode`` raises ``NotImplementedError``.
           Only ``sparsity_mode='none'`` (the default, dense attention) is
           supported.
    :type top_k: Optional[int]
    :param sparsity_threshold: Threshold for sparsity-based attention
        masking when ``sparsity_mode='threshold'``. Not implemented (see
        ``top_k``).
    :type sparsity_threshold: float
    :param sparsity_mode: Sparse attention mode: ``'none'``, ``'top_k'``, or
        ``'threshold'``. Only ``'none'`` is implemented; the other two raise
        ``NotImplementedError`` (dense fallback only).
    :type sparsity_mode: SparsityMode
    :param qkv_bias: Whether to include bias terms in QKV projections.
    :type qkv_bias: bool
    :param attention_dropout_rate: Dropout rate for attention weights.
    :type attention_dropout_rate: float
    :param projection_dropout_rate: Dropout rate for output projection.
    :type projection_dropout_rate: float
    :param use_lepe: Whether to use Locally-Enhanced Positional Encoding via
        depthwise convolution on value vectors.

        LePE bypasses the SW-MSA mask: it is a depthwise convolution applied
        to V inside the window, so after the cyclic shift it mixes
        neighbouring shifted-space positions and carries information across
        a region boundary that the additive mask blocks for the attention
        path. Measured 2026-08-27 at ``window_size=4, shift_size=2,
        H=W=8``: perturbing a masked-out key moves its guarded query from
        ``-0.803`` to ``-5.616`` with ``use_lepe=True``, and leaves it
        bit-unchanged with ``use_lepe=False``. This is faithful to CSWin —
        LePE is a positional encoding on V, not an attention term — but the
        mask's isolation guarantee covers the attention path only.
    :type use_lepe: bool
    :param lepe_kernel_size: Kernel size for LePE depthwise convolution.
    :type lepe_kernel_size: int
    :param kernel_initializer: Initializer for projection weight matrices.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param probability_type: Score-normalization strategy applied to the
        attention logits via
        :class:`~dl_techniques.layers.activations.ProbabilityOutput`.
        Defaults to ``'softmax'``. Routing / hierarchical variants are
        rejected because they alter the output shape.
    :type probability_type: str
    :param probability_config: Optional keyword arguments forwarded to
        :class:`~dl_techniques.layers.activations.ProbabilityOutput`.
        Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied per-head to Q
        and K before scoring, forwarded to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.
        ``None`` disables QK-norm. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`
        for both the Q and K norms. Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``dim`` is not divisible by ``num_heads``.
    :raises ValueError: If ``shift_size`` is negative or not less than
        ``window_size``.
    :raises ValueError: If ``sparsity_mode`` is not one of ``'none'``,
        ``'top_k'``, ``'threshold'``.
    :raises ValueError: If ``sparsity_mode='top_k'`` and ``top_k`` is
        ``None``, or if ``top_k`` is set but not positive.
    :raises ValueError: If ``attention_dropout_rate`` or
        ``projection_dropout_rate`` is outside ``[0.0, 1.0]``.
    :raises ValueError: If ``probability_type`` is a routing / hierarchical
        variant.
    :raises ValueError: From ``build()``, if ``shift_size > 0`` and the
        input height or width is not statically known.
    :raises NotImplementedError: If ``sparsity_mode`` is anything other than
        ``'none'`` — the sparse path is a stub (see ``top_k`` above).
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 8,
            shift_size: int = 0,
            top_k: Optional[int] = None,
            sparsity_threshold: float = 0.0,
            sparsity_mode: SparsityMode = 'none',
            qkv_bias: bool = True,
            attention_dropout_rate: float = 0.0,
            projection_dropout_rate: float = 0.0,
            use_lepe: bool = True,
            lepe_kernel_size: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Initialize ProgressiveFocusedAttention layer."""
        super().__init__(**kwargs)

        self._dim = dim
        self._num_heads = num_heads
        self._window_size = window_size
        self._shift_size = shift_size
        self._top_k = top_k
        self._sparsity_threshold = sparsity_threshold
        self._sparsity_mode = sparsity_mode
        self._qkv_bias = qkv_bias
        self._attention_dropout_rate = attention_dropout_rate
        self._projection_dropout_rate = projection_dropout_rate
        self._use_lepe = use_lepe
        self._lepe_kernel_size = lepe_kernel_size
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        self._validate_config()

        # Routing/hierarchical probability variants return tuples or a
        # different shape, which this layer's tuple contract cannot absorb.
        if self.probability_type in (
                "routing",
                "deterministic_routing",
                "hierarchical",
                "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type '{self.probability_type}' is not supported "
                f"by ProgressiveFocusedAttention (routing/hierarchical variants "
                f"alter output shape)."
            )

        self._head_dim = dim // num_heads

        # `head_dim ** -0.5` and `1.0/math.sqrt(head_dim)` differ in the last
        # ULP for 16 of 27 realistic head dims (measured 2026-08-28); keep
        # this form rather than the shared helper to avoid a numerics change
        # on trained weights.
        self._scale = self._head_dim ** -0.5
        self._window_area = window_size * window_size

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

        # Sub-layers are created here (Keras-3 authoring guide F7); build()
        # only resolves input shapes and builds them explicitly.
        self._qkv = keras.layers.Dense(
            self._dim * 3,
            use_bias=self._qkv_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="qkv_projection"
        )

        self._proj = keras.layers.Dense(
            self._dim,
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="output_projection"
        )

        if self._attention_dropout_rate > 0.0:
            self._attn_drop = keras.layers.Dropout(
                self._attention_dropout_rate,
                name="attention_dropout"
            )
        else:
            self._attn_drop = None

        if self._projection_dropout_rate > 0.0:
            self._proj_drop = keras.layers.Dropout(
                self._projection_dropout_rate,
                name="projection_dropout"
            )
        else:
            self._proj_drop = None

        if self._use_lepe:
            self._lepe = keras.layers.DepthwiseConv2D(
                kernel_size=self._lepe_kernel_size,
                strides=1,
                padding='same',
                depth_multiplier=1,
                depthwise_initializer=self._kernel_initializer,
                use_bias=True,
                bias_initializer=self._bias_initializer,
                name="lepe_conv"
            )
        else:
            self._lepe = None

    def _validate_config(self) -> None:
        """Validate layer configuration parameters.

        :raises ValueError: If any configuration parameter is invalid or incompatible.
        """
        # The shared `common.validate_head_divisibility` message omits the
        # fractional head_dim, which is the most useful number for picking a
        # valid (dim, num_heads) pair, so this site keeps its own check.
        if self._dim % self._num_heads != 0:
            raise ValueError(
                f"dim ({self._dim}) must be divisible by "
                f"num_heads ({self._num_heads}). "
                f"Got head_dim = {self._dim / self._num_heads}"
            )

        if self._shift_size < 0:
            raise ValueError(
                f"shift_size ({self._shift_size}) must be non-negative"
            )

        if self._shift_size >= self._window_size:
            raise ValueError(
                f"shift_size ({self._shift_size}) must be less than "
                f"window_size ({self._window_size}). "
                f"Typically use shift_size = window_size // 2 for SW-MSA."
            )

        if self._sparsity_mode not in ('none', 'top_k', 'threshold'):
            raise ValueError(
                f"sparsity_mode must be one of 'none', 'top_k', 'threshold', "
                f"got '{self._sparsity_mode}'"
            )

        if self._sparsity_mode == 'top_k' and self._top_k is None:
            raise ValueError(
                "top_k must be specified when sparsity_mode='top_k'"
            )

        if self._top_k is not None and self._top_k <= 0:
            raise ValueError(
                f"top_k ({self._top_k}) must be positive"
            )

        # DECISION plan_2026-06-14_0c5d4a21/D-004: reject non-'none'
        # sparsity_mode here rather than accept it silently -- _apply_sparsity
        # is a no-op stub, so accepting it would compute dense attention
        # while claiming sparse focusing. See decisions.md.
        if self._sparsity_mode != 'none':
            raise NotImplementedError(
                f"sparsity_mode='{self._sparsity_mode}' is not implemented in "
                f"ProgressiveFocusedAttention. The sparse attention path is a no-op "
                f"stub that falls back to dense attention; advertising sparse focusing "
                f"while computing dense attention would be misleading. Only "
                f"sparsity_mode='none' (dense attention) is supported. "
                f"(top_k={self._top_k!r} is therefore unused.)"
            )

        if self._attention_dropout_rate < 0.0 or self._attention_dropout_rate > 1.0:
            raise ValueError(
                f"attention_dropout_rate ({self._attention_dropout_rate}) must be "
                f"between 0.0 and 1.0"
            )

        if self._projection_dropout_rate < 0.0 or self._projection_dropout_rate > 1.0:
            raise ValueError(
                f"projection_dropout_rate ({self._projection_dropout_rate}) must be "
                f"between 0.0 and 1.0"
            )

    def build(self, input_shape: Union[tuple, list]) -> None:
        """Build layer weights and sub-layers.

        :param input_shape: Shape tuple or list of shape tuples for input tensor.
            Expected shape: ``(batch_size, height, width, dim)``.
        :type input_shape: Union[tuple, list]
        """
        # build() may re-enter via from_config/functional reuse; a second
        # .build() on an already-built child raises, so guard here.
        if self.built:
            return

        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        qkv_input_shape = (None, self._window_area, self._dim)
        self._qkv.build(qkv_input_shape)
        self._proj.build(qkv_input_shape)

        if self._use_lepe:
            lepe_input_shape = (None, self._window_size, self._window_size, self._dim)
            self._lepe.build(lepe_input_shape)

        if self._shift_size > 0:
            height = x_shape[1]
            width = x_shape[2]
            if height is None or width is None:
                raise ValueError(
                    "ProgressiveFocusedAttention with shift_size > 0 (SW-MSA) "
                    "requires statically-known height and width; got input "
                    f"shape {x_shape!r}. The shifted-window attention mask "
                    "geometry cannot be built from dynamic (None) spatial "
                    "dimensions. Provide a fixed-size input (e.g. via a static "
                    "input shape) or use shift_size=0 (W-MSA)."
                )
            self._attn_mask = self._compute_attention_mask(height, width)
        else:
            self._attn_mask = None

        # Attention scores have shape (B*nW, num_heads, window_area, window_area).
        score_shape = (
            None,
            self._num_heads,
            self._window_area,
            self._window_area,
        )
        self.attn_prob.build(score_shape)

        if self.q_norm is not None:
            qk_shape = (
                None,
                self._num_heads,
                self._window_area,
                self._head_dim,
            )
            self.q_norm.build(qk_shape)
            self.k_norm.build(qk_shape)

        super().build(input_shape)

    def _compute_attention_mask(self, height: int, width: int) -> np.ndarray:
        """Compute attention mask for shifted window attention (SW-MSA).

        Builds the SW-MSA mask for the actual static feature-map size
        ``(height, width)``. The mask image is partitioned into windows
        using the same ordering as :meth:`_window_partition` (B-major /
        window-minor), so each mask entry aligns with its corresponding
        window slot.

        SW-MSA requires statically-known ``height`` and ``width``: the mask
        geometry cannot be constructed from a dynamic ``None`` spatial
        dimension. ``build()`` therefore raises ``ValueError`` when either is
        ``None``, rather than silently emitting a wrong-geometry mask.

        :param height: Static feature-map height. Must be divisible by
            ``window_size`` and ``>= 2 * window_size``.
        :type height: int
        :param width: Static feature-map width. Same divisibility/size rule.
        :type width: int
        :return: Attention mask of shape ``(num_windows, window_area, window_area)``
            with 0.0 for valid pairs and -100.0 for masked pairs, where
            ``num_windows = (height // ws) * (width // ws)``.
        :rtype: numpy.ndarray
        """
        # Three regions per axis after the shift.
        h_slices = (
            slice(0, -self._window_size),
            slice(-self._window_size, -self._shift_size),
            slice(-self._shift_size, None)
        )
        w_slices = (
            slice(0, -self._window_size),
            slice(-self._window_size, -self._shift_size),
            slice(-self._shift_size, None)
        )

        img_mask = np.zeros((1, height, width, 1), dtype=np.float32)

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Same reshape/transpose order as _window_partition, so mask window
        # order matches: (1,H,W,1) -> (1,nH,ws,nW,ws,1) -> transpose(0,1,3,2,4,5).
        num_windows_h = height // self._window_size
        num_windows_w = width // self._window_size
        img_mask = img_mask.reshape(
            1, num_windows_h, self._window_size,
            num_windows_w, self._window_size, 1
        )
        img_mask = img_mask.transpose(0, 1, 3, 2, 4, 5)
        mask_windows = img_mask.reshape(
            num_windows_h * num_windows_w, self._window_size * self._window_size
        )

        attn_mask = mask_windows[:, :, np.newaxis] - mask_windows[:, np.newaxis, :]
        attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)

        # DECISION plan-2026-08-27T040114-580f8b63/D-011: plain numpy array
        # only -- a bare Variable/add_weight/convert_to_tensor here each break
        # differently (build-stack charge, StatelessScope discard, out-of-scope
        # FuncGraph). See decisions.md.
        return attn_mask

    def _window_partition(
            self,
            x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Partition input feature map into non-overlapping windows.

        Mirrors ``window_partition`` in
        :mod:`~dl_techniques.layers.attention.window_attention`, kept
        separate for the reasons in the class docstring. The reshape/
        transpose order here (B-major, window-minor) must match
        :meth:`_compute_attention_mask`, which builds the SW-MSA mask with
        the same ordering.

        :param x: Input tensor of shape ``(batch_size, height, width, channels)``.
            Height and width must be divisible by window_size.
        :type x: keras.KerasTensor

        :return: Partitioned windows of shape
            ``(batch_size * num_windows, window_size, window_size, channels)``.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(x)[0]
        height = keras.ops.shape(x)[1]
        width = keras.ops.shape(x)[2]
        channels = keras.ops.shape(x)[3]

        num_windows_h = height // self._window_size
        num_windows_w = width // self._window_size

        # (B, H, W, C) -> (B, nH, ws, nW, ws, C) -> (B, nH, nW, ws, ws, C)
        x = keras.ops.reshape(
            x,
            (batch_size, num_windows_h, self._window_size,
             num_windows_w, self._window_size, channels)
        )
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))

        # Flatten batch and window dims: (B, nH, nW, ws, ws, C) -> (B*nH*nW, ws, ws, C)
        windows = keras.ops.reshape(
            x,
            (-1, self._window_size, self._window_size, channels)
        )

        return windows

    def _window_reverse(
            self,
            windows: keras.KerasTensor,
            height: int,
            width: int
    ) -> keras.KerasTensor:
        """Reverse window partition to reconstruct the spatial feature map.

        Mirrors ``window_reverse`` in
        :mod:`~dl_techniques.layers.attention.window_attention`, kept
        separate for the same reasons as :meth:`_window_partition`. Unlike
        the sibling, this version derives the batch size as
        ``ops.shape(windows)[0] // num_windows`` instead of taking it as an
        argument, so the two signatures are not interchangeable.

        :param windows: Windows tensor of shape
            ``(batch_size * num_windows, window_size, window_size, channels)``.
        :type windows: keras.KerasTensor
        :param height: Target height of the reconstructed feature map.
        :type height: int
        :param width: Target width of the reconstructed feature map.
        :type width: int

        :return: Reconstructed spatial tensor of shape
            ``(batch_size, height, width, channels)``.
        :rtype: keras.KerasTensor
        """
        channels = keras.ops.shape(windows)[-1]
        num_windows_h = height // self._window_size
        num_windows_w = width // self._window_size
        num_windows = num_windows_h * num_windows_w
        batch_size = keras.ops.shape(windows)[0] // num_windows

        x = keras.ops.reshape(
            windows,
            (batch_size, num_windows_h, num_windows_w,
             self._window_size, self._window_size, channels)
        )
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = keras.ops.reshape(x, (batch_size, height, width, channels))

        return x

    def _apply_progressive_focusing(
            self,
            attn_scores: keras.KerasTensor,
            prev_attn_map: Optional[keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Apply progressive focusing via Hadamard product with previous attention map.

        :param attn_scores: Current attention scores of shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
        :type attn_scores: keras.KerasTensor
        :param prev_attn_map: Previous layer's attention weights of the same shape.
            If ``None``, returns scores unchanged (first layer).
        :type prev_attn_map: Optional[keras.KerasTensor]

        :return: Focused attention scores incorporating previous layer guidance.
        :rtype: keras.KerasTensor
        """
        if prev_attn_map is None:
            return attn_scores

        focused_scores = attn_scores * prev_attn_map

        return focused_scores

    def _apply_sparsity(
            self,
            attn_scores: keras.KerasTensor,
            prev_attn_map: Optional[keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Apply sparsity masking to attention scores based on previous layer guidance.

        :param attn_scores: Current attention scores of shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
        :type attn_scores: keras.KerasTensor
        :param prev_attn_map: Previous layer's attention map for guidance.
            If ``None``, returns scores unchanged.
        :type prev_attn_map: Optional[keras.KerasTensor]

        :return: The scores unchanged on every path that runs today. Only
            the unreachable ``'threshold'`` branch sets masked positions to
            ``-1e9``; see the warning below.
        :rtype: keras.KerasTensor

        .. warning::
            This method is unreachable below its first branch.
            ``_validate_config()`` raises ``NotImplementedError`` for every
            ``sparsity_mode`` other than ``'none'``, so in practice only the
            early ``return attn_scores`` executes. The ``'threshold'`` and
            ``'top_k'`` branches are retained as the starting point for the
            eventual real implementation, not as working code: the
            ``'top_k'`` branch in particular computes ``top_indices`` and an
            all-zeros ``mask`` and then uses neither, returning the
            unmodified dense scores. Two latent issues to fix when the path
            is revived, reported here rather than patched: (1) the
            ``'threshold'`` branch's literal ``-1e9`` becomes ``-inf`` under
            ``mixed_float16`` — it should route through
            :data:`~dl_techniques.layers.attention.common.MASK_BIAS_VALUE` and
            :func:`~dl_techniques.layers.attention.common.mask_dtype`; (2)
            ``k = min(self._top_k, seq_len)`` mixes a Python ``int`` with the
            traced tensor returned by ``ops.shape(...)[-1]``, which fails
            under ``@tf.function``/jit.
        """
        if self._sparsity_mode == 'none' or prev_attn_map is None:
            return attn_scores

        if self._sparsity_mode == 'threshold':
            mask = keras.ops.cast(
                prev_attn_map >= self._sparsity_threshold,
                dtype=attn_scores.dtype
            )
            attn_scores = keras.ops.where(
                mask > 0.5,
                attn_scores,
                keras.ops.full_like(attn_scores, -1e9)
            )

        elif self._sparsity_mode == 'top_k':
            seq_len = keras.ops.shape(attn_scores)[-1]
            k = min(self._top_k, seq_len)

            prev_mean = keras.ops.mean(prev_attn_map, axis=1, keepdims=True)
            _, top_indices = keras.ops.top_k(prev_mean, k=k)

            # A full implementation would scatter using top_indices; this
            # stub computes it and discards it. See the warning above.
            mask = keras.ops.zeros_like(attn_scores)

        return attn_scores

    def call(
            self,
            x: keras.KerasTensor,
            prev_attn_map: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass of Progressive Focused Attention.

        :param x: Input tensor of shape ``(batch_size, height, width, dim)``.
            Height and width must be divisible by window_size.
        :type x: keras.KerasTensor
        :param prev_attn_map: Previous layer's attention map for progressive focusing.
            Shape ``(batch*num_windows, num_heads, window_area, window_area)``.
            If ``None``, standard windowed attention is computed.
        :type prev_attn_map: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Affects dropout behavior.
        :type training: Optional[bool]

        :return: Tuple of ``(output, attention_weights)`` where output has shape
            ``(batch_size, height, width, dim)`` and attention_weights has shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]

        :raises ValueError: If input height or width is not divisible by window_size.
        """
        input_shape = keras.ops.shape(x)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        if self._shift_size > 0:
            # Negative shift now; shifted back after attention.
            shifted_x = keras.ops.roll(
                x,
                shift=(-self._shift_size, -self._shift_size),
                axis=(1, 2)
            )
        else:
            shifted_x = x

        # (B, H, W, C) -> (B*nW, ws, ws, C)
        x_windows = self._window_partition(shifted_x)
        num_windows = keras.ops.shape(x_windows)[0]

        # (B*nW, ws, ws, C) -> (B*nW, ws*ws, C)
        x_flat = keras.ops.reshape(
            x_windows,
            (num_windows, self._window_area, self._dim)
        )

        # (B*nW, ws*ws, C) -> (B*nW, ws*ws, 3*C)
        qkv = self._qkv(x_flat)

        qkv = keras.ops.reshape(
            qkv,
            (num_windows, self._window_area, 3, self._num_heads, self._head_dim)
        )
        # -> (3, B*nW, num_heads, ws*ws, head_dim)
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        if self._lepe is not None:
            # (B*nW, num_heads, ws*ws, head_dim) -> (B*nW, ws, ws, C)
            v_spatial = keras.ops.transpose(v, (0, 2, 1, 3))
            v_spatial = keras.ops.reshape(
                v_spatial,
                (num_windows, self._window_size, self._window_size, self._dim)
            )
            lepe = self._lepe(v_spatial)
            lepe = keras.ops.reshape(
                lepe,
                (num_windows, self._window_area, self._num_heads, self._head_dim)
            )
            lepe = keras.ops.transpose(lepe, (0, 2, 1, 3))
            v = v + lepe

        attn_scores = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 3, 2))
        ) * self._scale

        if self._attn_mask is not None and self._shift_size > 0:
            # _attn_mask: (nW, wa, wa) in the same window order as
            # _window_partition; broadcast over batch and heads.
            num_windows_per_image = (height // self._window_size) * (width // self._window_size)
            mask = keras.ops.reshape(
                keras.ops.cast(
                    keras.ops.convert_to_tensor(self._attn_mask), attn_scores.dtype
                ),
                (num_windows_per_image, 1, self._window_area, self._window_area)
            )
            # Tile B times to match _window_partition's B-major flattening.
            mask = keras.ops.tile(mask, (batch_size, 1, 1, 1))
            attn_scores = attn_scores + mask

        attn_scores = self._apply_progressive_focusing(attn_scores, prev_attn_map)
        attn_scores = self._apply_sparsity(attn_scores, prev_attn_map)

        attn_weights = self.attn_prob(attn_scores)

        if self._attn_drop is not None:
            attn_weights = self._attn_drop(attn_weights, training=training)

        attn_output = keras.ops.matmul(attn_weights, v)

        # (B*nW, num_heads, ws*ws, head_dim) -> (B*nW, ws*ws, num_heads, head_dim)
        attn_output = keras.ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = keras.ops.reshape(
            attn_output,
            (num_windows, self._window_area, self._dim)
        )

        output = self._proj(attn_output)

        if self._proj_drop is not None:
            output = self._proj_drop(output, training=training)

        output = keras.ops.reshape(
            output,
            (num_windows, self._window_size, self._window_size, self._dim)
        )
        output = self._window_reverse(output, height, width)

        if self._shift_size > 0:
            output = keras.ops.roll(
                output,
                shift=(self._shift_size, self._shift_size),
                axis=(1, 2)
            )

        return output, attn_weights

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Configuration dictionary containing all parameters needed
            to reconstruct this layer.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self._dim,
            "num_heads": self._num_heads,
            "window_size": self._window_size,
            "shift_size": self._shift_size,
            "top_k": self._top_k,
            "sparsity_threshold": self._sparsity_threshold,
            "sparsity_mode": self._sparsity_mode,
            "qkv_bias": self._qkv_bias,
            "attention_dropout_rate": self._attention_dropout_rate,
            "projection_dropout_rate": self._projection_dropout_rate,
            "use_lepe": self._use_lepe,
            "lepe_kernel_size": self._lepe_kernel_size,
            "kernel_initializer": keras.initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self._bias_initializer
            ),
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "qk_norm_type": self.qk_norm_type,
            "qk_norm_kwargs": self.qk_norm_kwargs,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ProgressiveFocusedAttention":
        """Create layer from configuration dictionary.

        :param config: Configuration dictionary from ``get_config``.
        :type config: Dict[str, Any]

        :return: New instance created from configuration.
        :rtype: ProgressiveFocusedAttention
        """
        config = config.copy()
        config["kernel_initializer"] = keras.initializers.deserialize(
            config.get("kernel_initializer", "glorot_uniform")
        )
        config["bias_initializer"] = keras.initializers.deserialize(
            config.get("bias_initializer", "zeros")
        )
        return cls(**config)

    def compute_output_shape(
            self,
            input_shape: Union[tuple, list]
    ) -> Tuple[tuple, tuple]:
        """Compute output shape of the layer.

        :param input_shape: Input shape(s). Either a single tuple for x, or a list
            of tuples for ``(x, prev_attn_map)``.
        :type input_shape: Union[tuple, list]

        :return: Tuple of ``(output_shape, attn_map_shape)`` where output_shape
            matches the input x shape and attn_map_shape is the attention weights shape.
        :rtype: Tuple[tuple, tuple]

        .. note::
            Returning a tuple of shapes is correct here and must not be
            collapsed to a single shape: :meth:`call` returns
            ``(output, attention_weights)``, so Keras needs both.
            ``attn_batch`` becomes ``None`` whenever the batch or spatial
            dims are dynamic, since ``B * nW`` is not computable from a
            symbolic shape.
        """
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        output_shape = x_shape

        batch = x_shape[0]
        h, w = x_shape[1], x_shape[2]

        if h is not None and w is not None:
            num_windows = (h // self._window_size) * (w // self._window_size)
            if batch is not None:
                attn_batch = batch * num_windows
            else:
                attn_batch = None
        else:
            attn_batch = None

        attn_map_shape = (
            attn_batch,
            self._num_heads,
            self._window_area,
            self._window_area
        )

        return output_shape, attn_map_shape
