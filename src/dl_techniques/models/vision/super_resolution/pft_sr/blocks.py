"""
The official PFT-SR transformer layer and the geometry helpers of its body.

A Keras port of ``PFTransformerLayer`` and ``WindowAttention`` from
CVL-UESTC/PFT-SR ``basicsr/archs/pft_arch.py`` (master, file commit
``0f4f38039d3e``, read 2026-10-20; line numbers in comments are that file's).
One layer is:

.. code-block:: text

    x' = x + W-MSA/SW-MSA(LN(x))           (with progressive focusing, below)
    y  = x' + ConvFFN(LN(x'))
    ConvFFN(t) = fc2(u + GELU(dwconv_k(u))),  u = GELU(fc1(t))

The attention takes ``q, k, v`` from one ``Linear(C, 3C)`` over the whole map
and a LePE term ``GELU(dwconv_k(v))``, also over the whole map, that is added to
``attn @ v`` (not to ``v``). Per window, ``attn = softmax(q k^T / sqrt(d) + B)``
with ``B`` a learned relative position bias (one column per head when
``dim > 100``, one shared column otherwise, as the official code has it) and,
in a shifted layer, the -100 cross-region mask.

Progressive focusing (PFA). The official network threads two attention chains
through all of its layers, one for the plain-window layers and one for the
shifted ones (``pfa_values[shift]``), so a layer inherits from the previous
layer OF ITS CHAIN. A layer that inherits a map multiplies its post-softmax
attention by it and renormalises, ``a <- (a * a_prev + eps) / (sum(a * a_prev) + eps)``
with ``eps = 1e-20``. A layer whose ``num_topk`` entry is below its chain's
current width keeps only its top ``k`` keys per query, from then on the chain
attends only inside those keys, and each later top-k picks inside the previous
one. The official code stores the kept keys as an index tensor and multiplies
with custom CUDA kernels; this port keeps the dense ``(windows, heads, n, n)``
map and a boolean mask of the kept keys, which computes the same numbers: the
softmax over the kept keys is a softmax with the others at ``-inf``, and the
dropped keys hold exactly zero.

References:
    - Long et al., 2025. Progressive Focused Transformer for Single Image
      Super-Resolution. CVPR. (https://arxiv.org/abs/2503.20337)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. ICCV. (https://arxiv.org/abs/2103.14030)
"""

import functools
import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.dtype_policy import accumulation_dtype
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: torch ``nn.LayerNorm``'s default, the official ``norm_layer`` (the norm
#: factory's own default is 1e-6).
LAYER_NORM_EPSILON = 1e-5

#: ``WindowAttention.eps`` (:225), in the PFA renormalisation.
PFA_EPSILON = 1e-20

#: ``PFT.calculate_mask`` (:1098): the additive bias between shifted regions.
SHIFT_MASK_VALUE = -100.0

#: ``PFT._init_weights`` (:1030): ``trunc_normal_(std=.02)`` on every Linear and
#: on the bias table. basicsr's ``trunc_normal_`` cuts at the ABSOLUTE bounds
#: (-2, 2), i.e. at 100 standard deviations: a plain normal, not Keras'
#: ``TruncatedNormal`` (which cuts at 2 standard deviations).
LINEAR_INIT_STDDEV = 0.02

# ---------------------------------------------------------------------


def relative_position_index(window_size: int) -> np.ndarray:
    """
    The official ``calculate_rpi_sa`` (:1067): row of the bias table each
    (query, key) pair of a window reads.

    :param window_size: Window side.
    :type window_size: int
    :return: Integer array ``(window_size**2, window_size**2)`` with values in
        ``[0, (2 * window_size - 1)**2)``.
    :rtype: numpy.ndarray
    """
    coords = np.stack(np.meshgrid(np.arange(window_size), np.arange(window_size),
                                  indexing="ij"))
    flat = coords.reshape(2, -1)
    rel = (flat[:, :, None] - flat[:, None, :]).transpose(1, 2, 0)
    rel = rel + (window_size - 1)
    return (rel[:, :, 0] * (2 * window_size - 1) + rel[:, :, 1]).astype("int32")


@functools.lru_cache(maxsize=32)
def shifted_window_mask(height: int, width: int, window_size: int,
                        shift_size: int) -> np.ndarray:
    """
    The official ``calculate_mask`` (:1081): the additive mask a shifted layer
    puts between tokens that the cyclic shift brought together from different
    regions of the map.

    Cached and read-only: every shifted layer of a model reads the same array.

    :param height: Map height, a multiple of ``window_size``.
    :type height: int
    :param width: Map width, a multiple of ``window_size``.
    :type width: int
    :param window_size: Window side.
    :type window_size: int
    :param shift_size: The cyclic shift (the official network uses
        ``window_size // 2``).
    :type shift_size: int
    :return: Float32 array ``(num_windows, n, n)``, ``n = window_size**2``,
        0 within a region and :data:`SHIFT_MASK_VALUE` across regions, windows
        in :func:`window_partition` order.
    :rtype: numpy.ndarray
    """
    img_mask = np.zeros((height, width), dtype=np.float32)
    slices = (slice(0, -window_size), slice(-window_size, -shift_size),
              slice(-shift_size, None))
    region = 0
    for rows in slices:
        for cols in slices:
            img_mask[rows, cols] = region
            region += 1
    windows = img_mask.reshape(height // window_size, window_size,
                               width // window_size, window_size)
    windows = windows.transpose(0, 2, 1, 3).reshape(-1, window_size * window_size)
    diff = windows[:, None, :] - windows[:, :, None]
    mask = np.where(diff != 0, SHIFT_MASK_VALUE, 0.0).astype(np.float32)
    mask.setflags(write=False)
    return mask


def mirror_pad(x: keras.KerasTensor, window_size: int) -> keras.KerasTensor:
    """
    Pad H and W up to a multiple of ``window_size`` exactly as ``PFT.forward``
    (:1103-1109): ``cat([x, flip(x)])[:size]``, i.e. symmetric padding with
    the edge row repeated.

    :param x: Image batch ``(B, H, W, C)`` with static ``H`` and ``W``.
    :type x: keras.KerasTensor
    :param window_size: The multiple to pad to.
    :type window_size: int
    :return: ``(B, H', W', C)``, ``H'`` and ``W'`` the next multiples.
    :rtype: keras.KerasTensor
    :raises ValueError: If ``H`` or ``W`` is not static, or if a pad is longer
        than the image along its axis: the official concatenation is then
        shorter than the size it slices, and its code fails.
    """
    height, width = x.shape[1], x.shape[2]
    if height is None or width is None:
        raise ValueError(
            f"PFT-SR needs a static height and width to pad to a multiple of "
            f"window_size and to build its shifted-window mask, got {tuple(x.shape)}."
        )
    for axis, size in ((1, height), (2, width)):
        target = -(-size // window_size) * window_size
        if target == size:
            continue
        if target - size > size:
            raise ValueError(
                f"The official PFT-SR mirror pad cannot extend a side of {size} to "
                f"{target} (window_size {window_size}): it pads with the flipped "
                f"image, so a side must be at least half the padded size. Use an "
                f"input with every side >= {-(-target // 2)}."
            )
        x = keras.ops.concatenate([x, keras.ops.flip(x, axis=axis)], axis=axis)
        x = x[:, :target] if axis == 1 else x[:, :, :target]
    return x


def window_partition(x: keras.KerasTensor, window_size: int) -> keras.KerasTensor:
    """
    ``(B, H, W, C) -> (B * nW, window_size**2, C)``, windows row-major per
    image (the official ``window_partition``, :172, flattened).

    :param x: Map with static ``H``, ``W``, ``C``, multiples of ``window_size``.
    :type x: keras.KerasTensor
    :param window_size: Window side.
    :type window_size: int
    :return: Window tokens.
    :rtype: keras.KerasTensor
    """
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x = keras.ops.reshape(x, (-1, height // window_size, window_size,
                              width // window_size, window_size, channels))
    x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return keras.ops.reshape(x, (-1, window_size * window_size, channels))


def window_reverse(windows: keras.KerasTensor, window_size: int, height: int,
                   width: int) -> keras.KerasTensor:
    """
    Inverse of :func:`window_partition` (the official ``window_reverse``, :186).

    :param windows: ``(B * nW, window_size**2, C)``.
    :type windows: keras.KerasTensor
    :param window_size: Window side.
    :type window_size: int
    :param height: Map height.
    :type height: int
    :param width: Map width.
    :type width: int
    :return: ``(B, height, width, C)``.
    :rtype: keras.KerasTensor
    """
    channels = windows.shape[-1]
    x = keras.ops.reshape(windows, (-1, height // window_size, width // window_size,
                                    window_size, window_size, channels))
    x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return keras.ops.reshape(x, (-1, height, width, channels))


def topk_mask(scores: keras.KerasTensor, k: int) -> keras.KerasTensor:
    """
    Boolean mask of the ``k`` largest entries along the last axis.

    Exactly ``k`` per row: built by scattering the ``top_k`` indices, not by
    comparing against the k-th value, which keeps every tie and would keep more
    than ``k`` keys on a flat map (a constant image gives one). Ties go to the
    lower index (TensorFlow's ``top_k``); the official ``torch.topk(...,
    sorted=False)`` leaves the tie order unspecified.

    :param scores: ``(..., n)``.
    :type scores: keras.KerasTensor
    :param k: Entries kept per row, ``1 <= k <= n``.
    :type k: int
    :return: Boolean tensor of the shape of ``scores``.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-214: scatter the top_k
    # indices into the mask. Do NOT threshold at the k-th value (it keeps every
    # tie, so more than k keys on a flat map) and do NOT one_hot the indices
    # (an n * n * k tensor). See decisions.md.
    # SUPERSEDED in part by plan-2026-10-14T062616-81938e81/D-276
    # (2026-10-22): the mask stays, because the gather form that carries the
    # top_k indices instead was measured and not shipped (the D-212 note in
    # PFTLayer._attend). "Exactly k, never a threshold" holds in both forms.
    # See decisions.md.
    n = scores.shape[-1]
    rows_2d = keras.ops.reshape(scores, (-1, n))
    _, indices = keras.ops.top_k(rows_2d, k)
    num_rows = keras.ops.shape(rows_2d)[0]
    row_ids = keras.ops.repeat(keras.ops.arange(num_rows)[:, None], k, axis=1)
    coords = keras.ops.stack(
        [keras.ops.reshape(row_ids, (-1,)),
         keras.ops.reshape(keras.ops.cast(indices, row_ids.dtype), (-1,))], axis=1)
    hits = keras.ops.scatter(coords, keras.ops.ones((num_rows * k,), dtype="float32"),
                             (num_rows, n))
    return keras.ops.reshape(hits > 0.5, keras.ops.shape(scores))


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.pft_sr.blocks")
class PFTLayer(keras.layers.Layer):
    """
    One official PFT transformer layer: ``PFTransformerLayer`` with its
    ``WindowAttention`` (CVL-UESTC/PFT-SR, :202-481).

    .. code-block:: text

        x ─► LN ─► wqkv ─► q, k, v ─────────────┐
                               └─► dwconv+GELU ─┤ lepe
            [roll -shift] ─► windows ─► attn = softmax(q k^T * s + B [+ mask])
                                         attn = PFA(attn, prev) ; top-k
                                         out  = proj(attn @ v + lepe)
            windows ─► map [roll +shift] ─► x' = x + out
        x' ─► LN ─► fc1 ─► GELU ─► u + GELU(dwconv(u)) ─► fc2 ─► y = x' + ...

    Called as ``y, values, mask = layer(x, prev_values=..., prev_mask=...)``:
    ``prev_values``/``prev_mask`` are what the previous layer of the same
    chain returned (``None`` for the first layer of a chain; ``mask`` stays
    ``None`` while the chain is dense). ``values`` is this layer's attention map
    ``(B * nW, heads, n, n)``, zero outside ``mask``.

    :param dim: Channels ``C``.
    :type dim: int
    :param num_heads: Attention heads; must divide ``dim``.
    :type num_heads: int
    :param window_size: Window side; ``n = window_size**2`` keys per window.
    :type window_size: int
    :param shift_size: Cyclic shift, 0 (plain windows) or below
        ``window_size`` (the official network uses ``window_size // 2``).
    :type shift_size: int
    :param convffn_kernel_size: Kernel of the LePE and ConvFFN depthwise convs
        (the official ``convffn_kernel_size``; both released configs use 7).
        Odd, so ``'same'`` padding is the official ``(k - 1) // 2``.
    :type convffn_kernel_size: int
    :param mlp_ratio: ConvFFN hidden width ``int(dim * mlp_ratio)``.
    :type mlp_ratio: float
    :param qkv_bias: Bias on ``wqkv``.
    :type qkv_bias: bool
    :param keep_topk: Keys kept per query by this layer's top-k, or ``None``
        to keep the chain's width (no top-k). The model sets it from
        ``num_topk`` by the official rule (a top-k only below the width).
    :type keep_topk: Optional[int]
    :param drop_path_rate: Stochastic depth on both residual branches, the
        SwinIR placement. Not in the official PFT (which has none); 0 builds
        nothing.
    :type drop_path_rate: float
    :param kwargs: Base ``Layer`` arguments.
    :raises ValueError: On an invalid argument.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int,
            shift_size: int = 0,
            convffn_kernel_size: int = 5,
            mlp_ratio: float = 2.0,
            qkv_bias: bool = True,
            keep_topk: Optional[int] = None,
            drop_path_rate: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not 0 <= shift_size < window_size:
            raise ValueError(
                f"shift_size must be in [0, window_size), got {shift_size} "
                f"for window_size {window_size}")
        if convffn_kernel_size < 1 or convffn_kernel_size % 2 == 0:
            raise ValueError(
                f"convffn_kernel_size must be a positive odd integer, got "
                f"{convffn_kernel_size}: the official padding (k - 1) // 2 is "
                f"'same' only for an odd kernel")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if keep_topk is not None and not 1 <= keep_topk <= window_size ** 2:
            raise ValueError(
                f"keep_topk must be None or in [1, window_size**2 = "
                f"{window_size ** 2}], got {keep_topk}")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(f"drop_path_rate must be in [0, 1), got {drop_path_rate}")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.convffn_kernel_size = convffn_kernel_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.keep_topk = keep_topk
        self.drop_path_rate = drop_path_rate

        self.head_dim = dim // num_heads
        self.hidden_dim = int(dim * mlp_ratio)
        self.attention_scale = self.head_dim ** -0.5
        self.num_tokens = window_size * window_size
        # :228-233, the official rule: per-head columns only above 100 channels.
        self.bias_columns = num_heads if dim > 100 else 1
        self._rpi = relative_position_index(window_size).reshape(-1)

        self.norm1 = create_normalization_layer(
            "layer_norm", epsilon=LAYER_NORM_EPSILON, name="norm1")
        self.wqkv = self._linear(3 * dim, "wqkv", use_bias=qkv_bias)
        self.v_lepe = keras.layers.DepthwiseConv2D(
            convffn_kernel_size, padding="same", use_bias=True, name="v_lepe")
        self.proj = self._linear(dim, "proj")
        self.norm2 = create_normalization_layer(
            "layer_norm", epsilon=LAYER_NORM_EPSILON, name="norm2")
        self.fc1 = self._linear(self.hidden_dim, "fc1")
        self.ffn_dwconv = keras.layers.DepthwiseConv2D(
            convffn_kernel_size, padding="same", use_bias=True, name="ffn_dwconv")
        self.fc2 = self._linear(dim, "fc2")
        self.drop_path = (StochasticDepth(drop_path_rate=drop_path_rate, name="drop_path")
                          if drop_path_rate > 0.0 else None)

    @staticmethod
    def _linear(units: int, name: str, use_bias: bool = True) -> keras.layers.Dense:
        """An official ``nn.Linear`` with ``_init_weights``: N(0, 0.02) kernel,
        zero bias (a fresh initializer per layer, never a shared seedless one)."""
        return keras.layers.Dense(
            units, use_bias=use_bias,
            kernel_initializer=keras.initializers.RandomNormal(stddev=LINEAR_INIT_STDDEV),
            bias_initializer="zeros", name=name)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build every sub-layer and the bias table from the channel count alone.

        :param input_shape: ``(B, H, W, dim)``; ``H`` and ``W`` may be unknown.
        :type input_shape: tuple
        """
        if self.built:
            return
        spatial = (None, None, None, self.dim)
        hidden = (None, None, None, self.hidden_dim)
        self.norm1.build(spatial)
        self.wqkv.build(spatial)
        self.v_lepe.build(spatial)
        self.proj.build((None, self.num_tokens, self.dim))
        self.norm2.build(spatial)
        self.fc1.build(spatial)
        self.ffn_dwconv.build(hidden)
        self.fc2.build(hidden)
        if self.drop_path is not None:
            self.drop_path.build(spatial)
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=((2 * self.window_size - 1) ** 2, self.bias_columns),
            initializer=keras.initializers.RandomNormal(stddev=LINEAR_INIT_STDDEV),
            trainable=True,
        )
        super().build(input_shape)

    def _attend(
            self,
            windows: keras.KerasTensor,
            height: int,
            width: int,
            prev_values: Optional[keras.KerasTensor],
            prev_mask: Optional[keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Optional[keras.KerasTensor]]:
        """``WindowAttention.forward`` (:240) on ``(B * nW, n, 4C)`` window
        tokens holding ``q | k | v | lepe``."""
        n, heads, d = self.num_tokens, self.num_heads, self.head_dim
        t = keras.ops.reshape(windows, (-1, n, 4, heads, d))
        t = keras.ops.transpose(t, (2, 0, 3, 1, 4))
        q, k, v, lepe = t[0], t[1], t[2], t[3]

        logits = keras.ops.matmul(q * self.attention_scale, keras.ops.transpose(k, (0, 1, 3, 2)))
        bias = keras.ops.take(self.relative_position_bias_table, self._rpi, axis=0)
        bias = keras.ops.transpose(keras.ops.reshape(bias, (n, n, self.bias_columns)),
                                   (2, 0, 1))
        # The softmax, the PFA product and the top-k run in the compute dtype,
        # raised to float32 under a half-precision policy (PFA_EPSILON is 0 in
        # float16, and a top-k over underflowed zeros would pick dropped keys).
        work = accumulation_dtype(logits.dtype)
        logits = keras.ops.cast(logits + keras.ops.cast(bias[None], logits.dtype), work)

        # DECISION plan-2026-10-14T062616-81938e81/D-212: the official sparse
        # top-k runs here as a DENSE (B*nW, heads, n, n) map under a boolean
        # mask of the kept keys. Do NOT gather the kept keys per query: that is
        # an (n, k, d) tensor per window and head, larger than the n x n map
        # wherever k * d > n (the first sparse layers of both released configs),
        # and keras has no sparse matmul. See decisions.md.
        # SUPERSEDED in part by plan-2026-10-14T062616-81938e81/D-276
        # (2026-10-22); the ruling stands. The gather form (flat-row take,
        # indices ascending) was built and measured on the RTX 4090: under XLA
        # its backward keeps the gathered keys and values, (k_prev + k) * d
        # floats per query, so the whole-model training-step peak fell only to
        # 0.709 of this form (light, 64 px, batch 2) and 0.449 (base, 64 px,
        # batch 1), short of the 0.70 bar. Unfused (jit_compile=False) it was
        # 1.03x (light) and out of memory on 24 GB where this form needs
        # 13.1 GiB (base). See decisions.md.
        if prev_mask is None:
            if self.shift_size > 0:
                region = shifted_window_mask(height, width, self.window_size, self.shift_size)
                logits = keras.ops.reshape(logits, (-1, region.shape[0], heads, n, n))
                logits = logits + keras.ops.cast(region, work)[None, :, None]
                logits = keras.ops.reshape(logits, (-1, heads, n, n))
            attn = keras.ops.softmax(logits, axis=-1)
        else:
            # DECISION plan-2026-10-14T062616-81938e81/D-213: once a chain is
            # sparse, attend over the kept keys only and add NO shift mask: the
            # official sparse branch (:271-284) adds the bias table only. Do not
            # "restore" the -100 here; kept cross-region keys carry an inherited
            # weight near zero instead. See decisions.md.
            attn = keras.ops.softmax(
                keras.ops.where(prev_mask, logits, float("-inf")), axis=-1)

        if prev_values is not None:
            attn = attn * keras.ops.cast(prev_values, work)
            attn = (attn + PFA_EPSILON) / (
                keras.ops.sum(attn, axis=-1, keepdims=True) + PFA_EPSILON)
            if prev_mask is not None:
                attn = keras.ops.where(prev_mask, attn, keras.ops.zeros_like(attn))

        mask = prev_mask
        if self.keep_topk is not None:
            # Rank the kept keys only: a kept key whose weight underflowed to 0
            # would otherwise tie with the dropped keys' zeros.
            scores = attn if prev_mask is None else keras.ops.where(
                prev_mask, attn, keras.ops.full_like(attn, -1.0))
            mask = topk_mask(scores, self.keep_topk)
            attn = keras.ops.where(mask, attn, keras.ops.zeros_like(attn))

        out = keras.ops.matmul(keras.ops.cast(attn, v.dtype), v) + lepe
        out = keras.ops.reshape(keras.ops.transpose(out, (0, 2, 1, 3)), (-1, n, self.dim))
        return self.proj(out), attn, mask

    def call(
            self,
            inputs: keras.KerasTensor,
            prev_values: Optional[keras.KerasTensor] = None,
            prev_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Optional[keras.KerasTensor]]:
        """
        One layer forward (``PFTransformerLayer.forward``, :438).

        :param inputs: ``(B, H, W, dim)``, ``H`` and ``W`` static multiples of
            ``window_size``.
        :type inputs: keras.KerasTensor
        :param prev_values: The chain's previous attention map, or ``None``.
        :type prev_values: Optional[keras.KerasTensor]
        :param prev_mask: The chain's kept keys, or ``None`` while it is dense.
        :type prev_mask: Optional[keras.KerasTensor]
        :param training: Training flag (stochastic depth only).
        :type training: Optional[bool]
        :return: ``(output, values, mask)``.
        :rtype: tuple
        :raises ValueError: If ``H`` or ``W`` is unknown or not a multiple of
            ``window_size``.
        """
        height, width = inputs.shape[1], inputs.shape[2]
        if height is None or width is None or height % self.window_size \
                or width % self.window_size:
            raise ValueError(
                f"PFTLayer needs a static height and width that are multiples of "
                f"window_size {self.window_size}, got {tuple(inputs.shape)}; the "
                f"model mirror-pads its input to one.")
        shortcut = inputs
        qkv = self.wqkv(self.norm1(inputs))
        lepe = keras.ops.gelu(self.v_lepe(qkv[..., 2 * self.dim:]), approximate=False)
        tokens = keras.ops.concatenate([qkv, lepe], axis=-1)
        if self.shift_size > 0:
            tokens = keras.ops.roll(tokens, shift=(-self.shift_size, -self.shift_size),
                                    axis=(1, 2))
        out, values, mask = self._attend(
            window_partition(tokens, self.window_size), height, width,
            prev_values, prev_mask)
        out = window_reverse(out, self.window_size, height, width)
        if self.shift_size > 0:
            out = keras.ops.roll(out, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        if self.drop_path is not None:
            out = self.drop_path(out, training=training)
        x = shortcut + out

        u = keras.ops.gelu(self.fc1(self.norm2(x)), approximate=False)
        u = u + keras.ops.gelu(self.ffn_dwconv(u), approximate=False)
        y = self.fc2(u)
        if self.drop_path is not None:
            y = self.drop_path(y, training=training)
        return x + y, values, mask

    def get_config(self) -> Dict[str, Any]:
        """
        :return: Every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "convffn_kernel_size": self.convffn_kernel_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "keep_topk": self.keep_topk,
            "drop_path_rate": self.drop_path_rate,
        })
        return config

# ---------------------------------------------------------------------
