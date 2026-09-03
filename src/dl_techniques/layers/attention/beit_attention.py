"""
BeitAttention, BEiT's self-attention layer with T5-style 2D relative position
bias and an asymmetric QKV bias.

The query and value projections carry a learnable bias; the key projection has
no bias parameter at all, structurally absent rather than zero-initialized.
A learnable, per-head, displacement-indexed bias is added to the attention
logits before the softmax: ``A_h = softmax(Q_h K_h^T / sqrt(d_h) + B_h)``,
where ``B_h[i, j] = T[R[i, j], h]`` reads a table ``T`` of shape
``(M, num_heads)``, ``M = (2Wh-1)(2Ww-1) + 3``, through a static integer index
``R`` derived from patch-to-patch displacement plus three rows for the
cls-to-token, token-to-cls and cls-to-cls relations.

Operates on ``(batch, Wh*Ww + 1, dim)`` with the cls token first; the
sequence length is fixed by ``window_size`` and checked in ``build()``.

References:
    - Bao et al., 2022. BEiT: BERT Pre-Training of Image Transformers.
      (https://arxiv.org/abs/2106.08254)
    - Raffel et al., 2020. Exploring the Limits of Transfer Learning with a
      Unified Text-to-Text Transformer. (relative-position-bias-as-logit-offset)
    - Liu et al., 2021. Swin Transformer. (the square-window table this one
      does not reuse; see single_window_attention.py)
"""

# ---------------------------------------------------------------------

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer
from .common import (
    mask_dtype,
    apply_attention_mask,
    compute_attention_scale,
    validate_head_divisibility,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.attention.beit_attention")
class BeitAttention(keras.layers.Layer):
    """
    BEiT multi-head self-attention with a cls-augmented relative position bias.

    Operates on a sequence of ``Wh * Ww`` patch tokens preceded by a single cls token,
    i.e. an input of shape ``(batch, Wh * Ww + 1, dim)``, and returns a tensor of the
    same shape. The relative position bias is added to the attention logits directly,
    before the softmax and before any attention mask — it is a real-valued learned
    offset and must never be routed through an attention-mask helper, which treats its
    argument as a binary keep predicate.

    Architecture:

    .. code-block:: text

        Input (B, N+1, D)   cls token first, then N = Wh*Ww patches
                │
                ├───────────────┬───────────────┐
                ▼               ▼               ▼
          ┌──────────┐    ┌──────────┐    ┌──────────┐
          │ q Dense  │    │ k Dense  │    │ v Dense  │
          │ (D, D)   │    │ (D, D)   │    │ (D, D)   │
          │ bias if  │    │ no bias  │    │ bias if  │
          │ qv_bias  │    │ variable │    │ qv_bias  │
          └──────────┘    └──────────┘    └──────────┘
                │               │               │
                ▼               ▼               ▼
              split heads -> (B, H, N+1, head_dim)
                │               │               │
                └───────┬───────┘               │
                        ▼                       │
             scores = (q * scale) @ kT          │
                  (B, H, N+1, N+1)              │
                        │                       │
                        ▼                       │
          ┌──────────────────────────────┐      │
          │ + relative position bias     │      │ (optional:
          │   B[h,i,j] = table[R[i,j],h] │      │  use_relative
          │   table (M, H) is a weight   │      │  _position
          │   R is a static int index    │      │  _bias)
          │   M = (2Wh-1)(2Ww-1) + 3     │      │
          └──────────────────────────────┘      │
                        │                       │
                        ▼                       │
             keep mask, if one was given        │
             softmax(axis=-1) in >= float32     │
             cast back, attn_dropout            │
                        │                       │
                        └───────┬───────────────┘
                                ▼
                     out = attn @ v  (B, H, N+1, head_dim)
                                ▼
                     merge heads -> (B, N+1, D)
                                ▼
                        ┌──────────────────┐
                        │ proj Dense (D,D) │
                        │ bias if          │
                        │ use_proj_bias    │
                        └──────────────────┘
                                ▼
                     proj_dropout -> Output (B, N+1, D)

    The k projection owns no bias variable. At dim=32 the built layer has
    8 weights and exactly 3 biases: q, v and proj.

    :param dim: Model / embedding dimension of the tokens. Must be positive and
        divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Default: 12.
    :type num_heads: int
    :param window_size: The patch grid ``(Wh, Ww)`` the sequence describes, e.g.
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
    :param qv_bias: If ``True``, the query and value projections carry a learnable
        bias and the key projection does not. If ``False``, none of the three do.
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
        """Validate the configuration and create the four projections.

        The three attention sub-layers whose width depends only on ``dim`` are
        created here; the relative-position bias table and its index map are
        allocated in :meth:`build`, because their shapes follow from
        ``window_size`` and the sequence length the input carries. The softmax
        temperature is resolved once here and stored alongside the caller's
        original ``scale`` so that :meth:`get_config` round-trips intent rather
        than a resolved number. See the class docstring for the parameter
        reference.

        :raises ValueError: For any invalid argument; see the class docstring's
            ``:raises:`` list.
        """
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
        # `scale` is stored as given (possibly None) so get_config() round-trips the
        # caller's intent; the resolved value reuses common.compute_attention_scale.
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

        # DECISION plan-2026-08-23T091307-9a110062/D-560: a callable, not a shared
        # dict, so each of the four projections clones its own initializer. Collapsing to `dict(...)` measured max|delta|=0.0 across 24 pairs. See decisions.md.
        def dense_kwargs() -> Dict[str, Any]:
            """Build a fresh keyword set for one projection Dense layer.

            Called once per projection. Every call clones the initializers, so
            the four Dense layers never share an initializer instance.

            :return: Keyword arguments for :class:`keras.layers.Dense`, holding
                freshly cloned kernel and bias initializers and the shared
                kernel and bias regularizers.
            :rtype: Dict[str, Any]
            """
            return dict(
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
            )
        self.q_dense = keras.layers.Dense(
            self.dim, use_bias=self.qv_bias, name="q", **dense_kwargs()
        )
        # DECISION plan-2026-08-11T012340-f63796dc/D-001: use_bias=False here, never
        # a zero or frozen bias — the reference fuses QKV behind one bias vector with no k_bias term, so unifying to use_bias=self.qv_bias silently breaks every checkpoint. See decisions.md.
        self.k_dense = keras.layers.Dense(
            self.dim, use_bias=False, name="k", **dense_kwargs()
        )
        self.v_dense = keras.layers.Dense(
            self.dim, use_bias=self.qv_bias, name="v", **dense_kwargs()
        )
        self.proj = keras.layers.Dense(
            self.dim, use_bias=self.use_proj_bias, name="proj", **dense_kwargs()
        )
        self.attn_dropout = keras.layers.Dropout(
            self.attn_dropout_rate, name="attn_dropout"
        )
        self.proj_dropout = keras.layers.Dropout(
            self.proj_dropout_rate, name="proj_dropout"
        )

        # Created in build(): the learnable table and its static integer index.
        # `_rel_pos_index` is a numpy array, not a tensor; see the D-011 note in build().
        self.relative_position_bias_table = None
        self._rel_pos_index: Optional[np.ndarray] = None

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
        table rows, assigned in that order so the cls-to-cls cell wins.

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
        # Written in this order (row, then column, then [0,0]) so the cls-to-cls
        # cell ends up with the right value.
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
            # DECISION plan-2026-08-11T012340-f63796dc/D-011: keep this index a numpy
            # array, not a converted tensor — build() can run lazily inside a traced train step, and a stored tensor there makes model.fit() raise InaccessibleTensorError. See decisions.md.
            self._rel_pos_index = np.ascontiguousarray(
                self._build_relative_position_index().reshape(-1),
                dtype=np.int32,
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
        :param attention_mask: Optional keep predicate (nonzero / ``True`` means
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
        # Batch axis stays dynamic (-1); seq_len is the static num_tokens build() validated.
        seq_len = self.num_tokens

        q = self._split_heads(self.q_dense(inputs), seq_len)
        k = self._split_heads(self.k_dense(inputs), seq_len)
        v = self._split_heads(self.v_dense(inputs), seq_len)

        # (B, H, N+1, head_dim) @ (B, H, head_dim, N+1) -> (B, H, N+1, N+1)
        scores = keras.ops.matmul(
            q * self._scale_value, keras.ops.transpose(k, (0, 1, 3, 2))
        )

        if self.use_relative_position_bias:
            # The bias is real-valued and added to the logits directly, never routed
            # through apply_attention_mask, which binarizes its predicate and would collapse the learned table to a keep/drop decision.
            # `_rel_pos_index` (a numpy array) is converted to a tensor here so the
            # constant materializes in the currently-tracing graph, not the one that built it.
            bias = keras.ops.take(
                self.relative_position_bias_table,
                keras.ops.convert_to_tensor(self._rel_pos_index, dtype="int32"),
                axis=0,
            )
            bias = keras.ops.reshape(bias, (seq_len, seq_len, self.num_heads))
            # (N+1, N+1, heads) -> (heads, N+1, N+1) -> (1, heads, N+1, N+1)
            bias = keras.ops.transpose(bias, (2, 0, 1))
            scores = scores + keras.ops.expand_dims(
                keras.ops.cast(bias, scores.dtype), axis=0
            )

        # Softmax runs in float32+ regardless of compute dtype (common.mask_dtype);
        # an fp16 softmax over a biased logit row has repeatedly measured NaNs. See common.py D-007.
        softmax_dtype = mask_dtype(
            getattr(scores.dtype, "name", None) or str(scores.dtype)
        )
        if attention_mask is not None:
            keep = self._broadcast_mask(attention_mask, seq_len)
            # attention_mask passes through verbatim as the keep predicate; no polarity inference.
            scores = apply_attention_mask(
                scores, keep, out_dtype=softmax_dtype, rescue_axis=-1
            )
        else:
            scores = keras.ops.cast(scores, softmax_dtype)

        attn = keras.ops.softmax(scores, axis=-1)
        attn = keras.ops.cast(attn, self.compute_dtype)
        attn = self.attn_dropout(attn, training=training)

        # (B, H, N+1, N+1) @ (B, H, N+1, head_dim) -> (B, H, N+1, head_dim)
        out = keras.ops.matmul(attn, v)
        # (B, H, N+1, head_dim) -> (B, N+1, H, head_dim) -> (B, N+1, dim)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (-1, seq_len, self.dim))
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
        reshaped = keras.ops.reshape(
            projected, (-1, seq_len, self.num_heads, self.head_dim)
        )
        return keras.ops.transpose(reshaped, (0, 2, 1, 3))

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
            return keras.ops.reshape(attention_mask, (-1, 1, 1, seq_len))
        if rank == 3:
            return keras.ops.reshape(attention_mask, (-1, 1, seq_len, seq_len))
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
