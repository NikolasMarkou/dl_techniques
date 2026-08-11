"""
BEiT self-attention: T5-style 2D relative position bias with an asymmetric QKV bias.

This module implements the self-attention block of BEiT (*BERT Pre-Training of Image
Transformers*, Bao, Dong & Wei, arXiv:2106.08254). It exists as a standalone layer
because BEiT differs from a generic ViT attention in exactly two structural ways that
no existing layer in this package can express:

1.  **Asymmetric QKV bias.** The query and value projections carry a learnable bias;
    the key projection has **no bias parameter at all** — structurally absent, not
    zero-initialized and not frozen. Every other attention class in this package
    (``MultiHeadAttention``, ``MultiHeadCrossAttention``, ``WindowAttention``,
    ``SingleWindowAttention``) exposes a single ``use_bias`` / ``qkv_bias`` flag that
    governs Q, K and V together, and the self-attention path of
    ``MultiHeadCrossAttention`` fuses them into one ``Dense(dim * 3)``, so the
    asymmetry is unreachable by subclassing.

2.  **A cls-augmented relative position bias table.** BEiT adds a learnable, per-head,
    displacement-indexed bias to the attention **logits, before the softmax**. Its
    table has ``(2*Wh - 1) * (2*Ww - 1) + 3`` rows over a ``(Wh, Ww)`` patch grid: the
    coordinate-derived rows for every distinct patch-to-patch displacement, plus three
    dedicated rows for the cls-to-token, token-to-cls and cls-to-cls relations, which
    have no well-defined 2D displacement. ``SingleWindowAttention``'s table is Swin's
    ``(2*W - 1) ** 2`` square-window form with no cls slots, so its index arithmetic is
    a different function of the window size and cannot be shared.

Architecture:
    ::

        Input  (B, N + 1, dim)          N = Wh * Ww patch tokens, +1 cls token
          |
          +-- q = Dense(dim, use_bias=qv_bias)
          +-- k = Dense(dim, use_bias=False)      <- ALWAYS bias-free (BEiT)
          +-- v = Dense(dim, use_bias=qv_bias)
          |
          v   reshape each to (B, num_heads, N + 1, head_dim)
        scores = q @ k^T * scale                  (B, num_heads, N+1, N+1)
          |
          +-- + relative_position_bias_table[rel_pos_index]   <- pre-softmax, ADDITIVE
          +-- + additive attention-mask bias (only when a mask is supplied)
          |
        attn = softmax(scores, axis=-1)  ->  dropout  ->  @ v
          |
        merge heads -> Dense(dim, use_bias=use_proj_bias) -> dropout
          |
        Output (B, N + 1, dim)

Foundational Mathematics:
    With queries ``Q``, keys ``K``, values ``V`` and per-head dimension ``d_h``, the
    attention of head ``h`` over a sequence of ``N + 1`` tokens is::

        A_h = softmax( (Q_h K_h^T) / sqrt(d_h) + B_h )
        O_h = A_h V_h

    where ``B_h`` is the relative position bias, a **real-valued** matrix read out of a
    learnable table ``T`` of shape ``(M, num_heads)``, ``M = (2Wh-1)(2Ww-1)+3``, via a
    static integer index matrix ``R`` of shape ``(N+1, N+1)``::

        B_h[i, j] = T[R[i, j], h]

    For two patch tokens at grid positions ``(y_i, x_i)`` and ``(y_j, x_j)``::

        R[i, j] = ( (y_i - y_j) + Wh - 1 ) * (2*Ww - 1) + ( (x_i - x_j) + Ww - 1 )

    so the bias depends only on the *displacement* between the two patches, and the
    ``(2Wh-1)(2Ww-1)`` distinct displacements exhaust the coordinate-derived rows. The
    remaining three rows are assigned to the cls relations::

        R[0, j] = M - 3   (cls attends to a patch)
        R[i, 0] = M - 2   (a patch attends to cls)
        R[0, 0] = M - 1   (cls attends to cls)

    Because ``B`` enters *inside* the softmax as an additive term, it re-weights the
    attention distribution rather than the output values, and it is shared across the
    batch.

References:
    - Bao, H., Dong, L., Piao, S., & Wei, F. (2022). "BEiT: BERT Pre-Training of Image
      Transformers". ICLR. arXiv:2106.08254. (The q/v-only bias and the cls-augmented
      relative position bias; ``microsoft/unilm/beit/modeling_finetune.py``.)
    - Raffel et al. (2020). "Exploring the Limits of Transfer Learning with a Unified
      Text-to-Text Transformer". JMLR. (The relative-position-bias-as-logit-offset idea
      BEiT credits.)
    - Liu et al. (2021). "Swin Transformer". ICCV. (The square-window
      ``(2W-1)**2`` table this one deliberately does *not* reuse; see
      ``single_window_attention.py``.)
"""

# ---------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple, Union

import keras
import numpy as np
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import (
    apply_attention_mask,
    compute_attention_scale,
    mask_dtype,
    validate_head_divisibility,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BeitAttention(keras.layers.Layer):
    """
    BEiT multi-head self-attention with a cls-augmented relative position bias.

    Operates on a sequence of ``Wh * Ww`` patch tokens preceded by a single cls token,
    i.e. an input of shape ``(batch, Wh * Ww + 1, dim)``, and returns a tensor of the
    same shape. The relative position bias is added to the attention logits directly,
    before the softmax and before any attention mask — it is a real-valued learned
    offset and must never be routed through an attention-mask helper, which treats its
    argument as a binary keep predicate.

    :param dim: Model / embedding dimension of the tokens. Must be positive and
        divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Default: 12.
    :type num_heads: int
    :param window_size: The **patch grid** ``(Wh, Ww)`` the sequence describes, e.g.
        ``(14, 14)`` for a 224px image at patch size 16. An ``int`` means a square grid.
        This is the grid of *patch* tokens only — the cls token is accounted for
        separately by the ``+ 3`` rows of the bias table, so the expected sequence
        length is ``Wh * Ww + 1``.
    :type window_size: Union[int, Tuple[int, int]]
    :param use_relative_position_bias: If ``True``, allocate the learnable bias table
        and add its gathered values to the attention logits. If ``False``, the layer is
        a plain (relative-position-free) attention and no table weight is created.
        Default: True.
    :type use_relative_position_bias: bool
    :param qv_bias: If ``True``, the **query and value** projections carry a learnable
        bias and the **key** projection does not. If ``False``, none of the three do.
        This is BEiT's ``qkv_bias`` configuration flag, renamed here to say what it
        actually does: the reference concatenates ``q_bias``, a non-trainable
        ``zeros_like(v_bias)`` and ``v_bias`` before a fused projection, so no ``k_bias``
        parameter is ever created. Default: True.
    :type qv_bias: bool
    :param use_proj_bias: If ``True``, the output projection carries a learnable bias.
        Default: True.
    :type use_proj_bias: bool
    :param attn_dropout_rate: Dropout rate applied to the attention probabilities.
        Must lie in ``[0, 1]``. Default: 0.0.
    :type attn_dropout_rate: float
    :param proj_dropout_rate: Dropout rate applied after the output projection. Must
        lie in ``[0, 1]``. Default: 0.0.
    :type proj_dropout_rate: float
    :param scale: Override for the softmax temperature multiplying ``q @ k^T``.
        ``None`` (the default) uses ``1 / sqrt(head_dim)``.
    :type scale: Optional[float]
    :param kernel_initializer: Initializer for the projection kernels.
        Default: ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the projection biases. Default: ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the projection kernels. Default: None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the projection biases. Default: None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    :type kwargs: Any

    :raises ValueError: If ``dim`` is not positive, if ``num_heads`` is not positive,
        if ``dim`` is not divisible by ``num_heads``, if any component of
        ``window_size`` is not positive, or if either dropout rate lies outside
        ``[0, 1]``.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 12,
            window_size: Union[int, Tuple[int, int]] = None,
            use_relative_position_bias: bool = True,
            qv_bias: bool = True,
            use_proj_bias: bool = True,
            attn_dropout_rate: float = 0.0,
            proj_dropout_rate: float = 0.0,
            scale: Optional[float] = None,
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

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        validate_head_divisibility(dim, num_heads)

        if window_size is None:
            raise ValueError(
                "window_size is required: pass the patch grid (Wh, Ww) this "
                "attention operates over, e.g. window_size=(14, 14) for a 224px "
                "image at patch size 16. An int means a square grid."
            )
        window_size = self._normalize_window_size(window_size)

        for rate_name, rate in (
                ("attn_dropout_rate", attn_dropout_rate),
                ("proj_dropout_rate", proj_dropout_rate),
        ):
            if not 0.0 <= float(rate) <= 1.0:
                raise ValueError(
                    f"{rate_name} must be in [0, 1], got {rate}"
                )

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_relative_position_bias = use_relative_position_bias
        self.qv_bias = qv_bias
        self.use_proj_bias = use_proj_bias
        self.attn_dropout_rate = float(attn_dropout_rate)
        self.proj_dropout_rate = float(proj_dropout_rate)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.head_dim = dim // num_heads
        # `scale` is stored as given (possibly None) so `get_config()` round-trips the
        # caller's intent rather than a resolved number. The resolved value reuses
        # `common.compute_attention_scale`, the package's single definition of the
        # softmax temperature; it equals `head_dim ** -0.5` up to at most one ULP (see
        # that helper's docstring), and it is a Python float so it folds into the graph.
        self.scale = scale
        self._scale_value = (
            float(scale) if scale is not None
            else compute_attention_scale(self.head_dim)
        )

        # Sequence length this layer expects: every patch token plus the cls token.
        self.num_patches = self.window_size[0] * self.window_size[1]
        self.num_tokens = self.num_patches + 1
        # Coordinate-derived displacement rows, plus the three cls-relation rows.
        self.num_relative_distance = (
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) + 3
        )

        # Sub-layers are created unconditionally in __init__ and built in build().
        dense_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )
        self.q_dense = keras.layers.Dense(
            self.dim, use_bias=self.qv_bias, name="q", **dense_kwargs
        )
        # DECISION plan-2026-08-11T012340-f63796dc/D-001
        # `use_bias=False` here is BEiT's ARCHITECTURE, not an oversight and not a
        # forgotten `self.qv_bias`. The reference builds a fused QKV projection whose
        # bias vector is `cat(q_bias, zeros_like(v_bias, requires_grad=False), v_bias)`
        # — there is no `k_bias` parameter in the checkpoint at all — and HF's
        # independent-projection port spells the same invariant as
        # `nn.Linear(..., bias=False)` for K.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT "unify" this with `use_bias=self.qv_bias` so all three projections
        #     agree. That silently adds `dim` parameters that BEiT does not have, makes
        #     the layer structurally incompatible with any BEiT checkpoint, and produces
        #     no error, no shape mismatch and a perfectly plausible loss curve.
        #   * Do NOT "fix" it as `use_bias=True` with a zero initializer or a frozen
        #     bias. A zero-initialized bias TRAINS; a frozen one still occupies a
        #     variable slot and shifts every weight index after it. The parameter must
        #     be structurally ABSENT, which is what `use_bias=False` gives.
        #   * Do NOT expose a `k_bias` constructor flag "for symmetry". The whole reason
        #     this layer is standalone rather than a subclass of
        #     `MultiHeadCrossAttention` is that a single fused `use_bias` flag cannot
        #     express this asymmetry (D-001 / H-4); re-introducing the flag would
        #     re-introduce the defect the layer exists to avoid.
        # Pinned by `TestBeitAttentionNoKBias` (exact bias-parameter count, not `> 0`).
        # See decisions.md D-001 (plan-2026-08-11T012340-f63796dc).
        self.k_dense = keras.layers.Dense(
            self.dim, use_bias=False, name="k", **dense_kwargs
        )
        self.v_dense = keras.layers.Dense(
            self.dim, use_bias=self.qv_bias, name="v", **dense_kwargs
        )
        self.proj = keras.layers.Dense(
            self.dim, use_bias=self.use_proj_bias, name="proj", **dense_kwargs
        )
        self.attn_dropout = keras.layers.Dropout(
            self.attn_dropout_rate, name="attn_dropout"
        )
        self.proj_dropout = keras.layers.Dropout(
            self.proj_dropout_rate, name="proj_dropout"
        )

        # Created in build(): the learnable table and its static integer index.
        self.relative_position_bias_table = None
        self._rel_pos_index = None

    @staticmethod
    def _normalize_window_size(
            window_size: Union[int, Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Coerce ``window_size`` to a validated ``(Wh, Ww)`` integer pair.

        :param window_size: An ``int`` (square grid) or a 2-element sequence
            ``(Wh, Ww)``.
        :type window_size: Union[int, Tuple[int, int]]

        :raises ValueError: If the value is neither an int nor a 2-element sequence,
            or if any component is not a positive integer.

        :return: The grid as a ``(Wh, Ww)`` tuple of Python ints.
        :rtype: Tuple[int, int]
        """
        if isinstance(window_size, (int, np.integer)) and not isinstance(
                window_size, bool
        ):
            window_size = (int(window_size), int(window_size))
        elif isinstance(window_size, (tuple, list)) and len(window_size) == 2:
            window_size = (int(window_size[0]), int(window_size[1]))
        else:
            raise ValueError(
                "window_size must be an int (square patch grid) or a 2-element "
                f"(Wh, Ww) tuple, got {window_size!r}"
            )
        if window_size[0] <= 0 or window_size[1] <= 0:
            raise ValueError(
                "window_size components must be positive, got "
                f"{window_size!r}"
            )
        return window_size

    def _build_relative_position_index(self) -> np.ndarray:
        """Build the static ``(N + 1, N + 1)`` relative-position index matrix.

        Transcribes the reference construction (``microsoft/unilm``
        ``modeling_finetune.py``): patch-to-patch entries are the flattened 2D
        displacement, shifted to start at zero and row-major-encoded with a stride of
        ``2 * Ww - 1``; the cls row, cls column and cls-to-cls cell take the last three
        table rows, **assigned in that order** so that the cls-to-cls cell wins.

        :return: Integer index matrix of shape ``(Wh * Ww + 1, Wh * Ww + 1)``, every
            entry in ``[0, num_relative_distance)``.
        :rtype: np.ndarray
        """
        wh, ww = self.window_size
        num_patches = wh * ww

        coords_h = np.arange(wh)
        coords_w = np.arange(ww)
        # (2, Wh, Ww) -> (2, N)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        # (2, N, N) -> (N, N, 2)
        relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = relative_coords.transpose(1, 2, 0).astype(np.int64)
        # Shift both displacement axes to start at 0, then row-major encode.
        relative_coords[:, :, 0] += wh - 1
        relative_coords[:, :, 1] += ww - 1
        relative_coords[:, :, 0] *= 2 * ww - 1

        index = np.zeros(
            (num_patches + 1, num_patches + 1), dtype=np.int64
        )
        index[1:, 1:] = relative_coords.sum(-1)
        # Order matters: the cls ROW is written first, then the cls COLUMN (which
        # overwrites [0, 0]), then [0, 0] itself. Writing them in any other order
        # leaves the wrong value in the cls-to-cls cell.
        index[0, 0:] = self.num_relative_distance - 3
        index[0:, 0] = self.num_relative_distance - 2
        index[0, 0] = self.num_relative_distance - 1
        return index

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Allocate the bias table, precompute its index, and build every sub-layer.

        :param input_shape: Shape tuple ``(batch, Wh * Ww + 1, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the sequence length is statically known and differs
            from ``Wh * Ww + 1``, or if the feature dimension is statically known and
            differs from ``dim``.
        """
        if self.built:
            return

        input_shape = tuple(input_shape)
        if len(input_shape) != 3:
            raise ValueError(
                "BeitAttention expects a rank-3 input (batch, seq_len, dim), got "
                f"shape {input_shape}"
            )
        seq_len = input_shape[-2]
        if seq_len is not None and int(seq_len) != self.num_tokens:
            raise ValueError(
                f"BeitAttention(window_size={self.window_size}) expects a sequence "
                f"length of Wh*Ww + 1 = {self.num_tokens} (one cls token followed by "
                f"{self.num_patches} patch tokens), but received an input with "
                f"sequence length {int(seq_len)}. The relative-position index is "
                "built from window_size, so a mismatch would gather the wrong bias "
                "for every pair of tokens."
            )
        feature_dim = input_shape[-1]
        if feature_dim is not None and int(feature_dim) != self.dim:
            raise ValueError(
                f"BeitAttention(dim={self.dim}) received an input whose last "
                f"dimension is {int(feature_dim)}"
            )

        if self.use_relative_position_bias:
            self.relative_position_bias_table = self.add_weight(
                name="relative_position_bias_table",
                shape=(self.num_relative_distance, self.num_heads),
                initializer="zeros",
                trainable=True,
                dtype=self.dtype,
            )
            # A derived, non-trainable constant: it is a pure function of
            # `window_size`, so it is deliberately NOT a weight. It survives
            # serialization because `build()` recomputes it from the restored config.
            self._rel_pos_index = ops.convert_to_tensor(
                self._build_relative_position_index().reshape(-1),
                dtype="int32",
            )

        projection_shape = (input_shape[0], self.num_tokens, self.dim)
        self.q_dense.build(projection_shape)
        self.k_dense.build(projection_shape)
        self.v_dense.build(projection_shape)
        self.proj.build(projection_shape)
        self.attn_dropout.build(
            (input_shape[0], self.num_heads, self.num_tokens, self.num_tokens)
        )
        self.proj_dropout.build(projection_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Compute BEiT self-attention over ``(batch, Wh*Ww + 1, dim)``.

        :param inputs: Token sequence of shape ``(batch, Wh*Ww + 1, dim)``, cls first.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional **keep predicate** (nonzero / ``True`` means
            "attend to this position"), broadcastable against the attention logits.
            Accepted as rank 2 ``(B, N+1)`` (a key mask), rank 3 ``(B, N+1, N+1)``
            (pairwise), or rank 4 ``(B, heads, N+1, N+1)``. BEiT's own patch grid is
            unpadded, so this is normally ``None``; the parameter exists because
            ``TransformerLayer`` passes it positionally by keyword to every attention
            type outside ``_MASKLESS_ATTENTION_TYPES``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Keras training-mode flag, forwarded to the dropout layers.
        :type training: Optional[bool]

        :raises ValueError: If ``attention_mask`` has a rank other than 2, 3 or 4.

        :return: Tensor of shape ``(batch, Wh*Ww + 1, dim)``.
        :rtype: keras.KerasTensor
        """
        # The batch axis is left dynamic as `-1` in every reshape (graph-safe; never
        # read as a Python int), while the sequence length is the STATIC `num_tokens`
        # that `build()` already validated the input against.
        seq_len = self.num_tokens

        q = self._split_heads(self.q_dense(inputs), seq_len)
        k = self._split_heads(self.k_dense(inputs), seq_len)
        v = self._split_heads(self.v_dense(inputs), seq_len)

        # (B, H, N+1, head_dim) @ (B, H, head_dim, N+1) -> (B, H, N+1, N+1)
        scores = ops.matmul(
            q * self._scale_value, ops.transpose(k, (0, 1, 3, 2))
        )

        if self.use_relative_position_bias:
            # The bias is REAL-VALUED and is added to the logits HERE, directly. It
            # must never be routed through `apply_attention_mask` / any mask argument:
            # that helper binarizes its predicate (`> 0`), which would collapse the
            # learned table to a keep/drop decision, silently and without a shape
            # error. See `common.apply_attention_mask`'s binary-`keep` precondition.
            # (M, heads) gathered by (N+1)^2 indices -> ((N+1)^2, heads)
            bias = ops.take(
                self.relative_position_bias_table, self._rel_pos_index, axis=0
            )
            bias = ops.reshape(bias, (seq_len, seq_len, self.num_heads))
            # (N+1, N+1, heads) -> (heads, N+1, N+1) -> (1, heads, N+1, N+1)
            bias = ops.transpose(bias, (2, 0, 1))
            scores = scores + ops.expand_dims(
                ops.cast(bias, scores.dtype), axis=0
            )

        # The softmax runs in float32 (or float64 under a float64 policy) regardless of
        # the compute dtype, which is what `common.mask_dtype` returns; under
        # `mixed_float16` an fp16 softmax over a biased logit row is where this package
        # has repeatedly measured NaNs. The result is cast back to the compute dtype.
        softmax_dtype = mask_dtype(
            keras.backend.standardize_dtype(scores.dtype)
        )
        if attention_mask is not None:
            keep = self._broadcast_mask(attention_mask, seq_len)
            # `attention_mask` is passed through VERBATIM as the keep predicate; this
            # site performs no polarity inference (see `common.apply_attention_mask`).
            scores = apply_attention_mask(
                scores, keep, out_dtype=softmax_dtype, rescue_axis=-1
            )
        else:
            scores = ops.cast(scores, softmax_dtype)

        attn = ops.softmax(scores, axis=-1)
        attn = ops.cast(attn, self.compute_dtype)
        attn = self.attn_dropout(attn, training=training)

        # (B, H, N+1, N+1) @ (B, H, N+1, head_dim) -> (B, H, N+1, head_dim)
        out = ops.matmul(attn, v)
        # (B, H, N+1, head_dim) -> (B, N+1, H, head_dim) -> (B, N+1, dim)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, seq_len, self.dim))
        out = self.proj(out)
        out = self.proj_dropout(out, training=training)
        return out

    def _split_heads(
            self,
            projected: keras.KerasTensor,
            seq_len: int,
    ) -> keras.KerasTensor:
        """Reshape ``(B, N+1, dim)`` into ``(B, num_heads, N+1, head_dim)``.

        :param projected: Output of one of the q/k/v projections.
        :type projected: keras.KerasTensor
        :param seq_len: Static sequence length ``Wh * Ww + 1``.
        :type seq_len: int
        :return: Head-split tensor.
        :rtype: keras.KerasTensor
        """
        reshaped = ops.reshape(
            projected, (-1, seq_len, self.num_heads, self.head_dim)
        )
        return ops.transpose(reshaped, (0, 2, 1, 3))

    def _broadcast_mask(
            self,
            attention_mask: keras.KerasTensor,
            seq_len: int,
    ) -> keras.KerasTensor:
        """Reshape a rank-2/3/4 keep predicate to broadcast against the logits.

        :param attention_mask: Keep predicate supplied by the caller.
        :type attention_mask: keras.KerasTensor
        :param seq_len: Static sequence length ``Wh * Ww + 1``.
        :type seq_len: int
        :raises ValueError: If the mask rank is not 2, 3 or 4.
        :return: A mask of rank 4, broadcastable against ``(B, H, N+1, N+1)``.
        :rtype: keras.KerasTensor
        """
        rank = len(attention_mask.shape)
        if rank == 2:
            return ops.reshape(attention_mask, (-1, 1, 1, seq_len))
        if rank == 3:
            return ops.reshape(attention_mask, (-1, 1, seq_len, seq_len))
        if rank == 4:
            return attention_mask
        raise ValueError(
            "BeitAttention: attention_mask must have rank 2 (B, N+1), rank 3 "
            f"(B, N+1, N+1) or rank 4 (B, heads, N+1, N+1), got rank {rank} with "
            f"shape {tuple(attention_mask.shape)}"
        )

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape, which is identical to the input shape.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Serialize every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": tuple(self.window_size),
                "use_relative_position_bias": self.use_relative_position_bias,
                "qv_bias": self.qv_bias,
                "use_proj_bias": self.use_proj_bias,
                "attn_dropout_rate": self.attn_dropout_rate,
                "proj_dropout_rate": self.proj_dropout_rate,
                "scale": self.scale,
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BeitAttention":
        """Reconstruct a layer from its serialized configuration.

        :param config: Dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new, unbuilt ``BeitAttention``.
        :rtype: BeitAttention
        """
        config = dict(config)
        for key in ("kernel_initializer", "bias_initializer"):
            if isinstance(config.get(key), dict):
                config[key] = keras.initializers.deserialize(config[key])
        for key in ("kernel_regularizer", "bias_regularizer"):
            if isinstance(config.get(key), dict):
                config[key] = keras.regularizers.deserialize(config[key])
        window_size = config.get("window_size")
        if isinstance(window_size, list):
            config["window_size"] = tuple(window_size)
        return cls(**config)

# ---------------------------------------------------------------------
