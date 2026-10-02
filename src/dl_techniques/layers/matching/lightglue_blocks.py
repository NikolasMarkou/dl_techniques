"""
LightGlue transformer blocks: rotary self-attention and bidirectional cross-attention.

One LightGlue layer is a self-attention block on each image followed by a single
cross-attention block that updates both images at once. Both blocks end in the same
message-passing FFN: the node state ``x`` is concatenated with the attention message,
pushed through ``Dense(2d) -> LayerNorm -> GELU -> Dense(d)`` and added back to ``x``.

Architecture:
    ::

        LightGlueSelfBlock                      LightGlueCrossBlock

        x (B,L,d)   freqs (2,B,1,L,Dh)          x0 (B,M,d)          x1 (B,N,d)
          |                                        |                   |
          | Wqkv: Dense(3d)                        | to_qk, to_v       | to_qk, to_v   (SHARED)
          v                                        v                   v
        (B,L,H,Dh,3) -> q, k, v                 qk0 * Dh^-1/4       qk1 * Dh^-1/4
          |                                        \\                  /
          | rotary(q), rotary(k)                    sim = qk0 qk1^T   (computed ONCE)
          v                                        /                  \\
        softmax(q k^T * Dh^-1/2, mask) v     softmax_row(sim) v1   softmax_col(sim)^T v0
          |                                        |                   |
          | out_proj                               | to_out            | to_out        (SHARED)
          v                                        v                   v
        x + ffn([x, message])                   x0 + ffn([x0, m0])  x1 + ffn([x1, m1])  (SHARED ffn)

Weight mapping from the official PyTorch state dict (``transformers.i.self_attn.*`` and
``transformers.i.cross_attn.*``). Every sublayer is a Keras attribute named after its
torch counterpart, so the mapping is mechanical: a ``Linear`` weight of shape
``(out, in)`` is assigned to the Dense ``kernel`` by TRANSPOSE (``kernel = W.T``), the
bias is copied, a LayerNorm ``weight``/``bias`` maps to ``gamma``/``beta``.

    ==========================================  ====================================
    torch key                                   Keras variable
    ==========================================  ====================================
    ``self_attn.Wqkv.{weight,bias}``            ``block.Wqkv.{kernel,bias}``
    ``self_attn.out_proj.{weight,bias}``        ``block.out_proj.{kernel,bias}``
    ``cross_attn.to_qk.{weight,bias}``          ``block.to_qk.{kernel,bias}``
    ``cross_attn.to_v.{weight,bias}``           ``block.to_v.{kernel,bias}``
    ``cross_attn.to_out.{weight,bias}``         ``block.to_out.{kernel,bias}``
    ``*.ffn.0.{weight,bias}``                   ``block.ffn_0.{kernel,bias}``
    ``*.ffn.1.{weight,bias}``                   ``block.ffn_1.{gamma,beta}``
    ``*.ffn.3.{weight,bias}``                   ``block.ffn_3.{kernel,bias}``
    ==========================================  ====================================

The fused ``Wqkv`` output columns are laid out ``(num_heads, head_dim, 3)``: column
``h * Dh * 3 + c * 3 + s`` is head ``h``, channel ``c``, slot ``s`` in ``{q, k, v}``
(torch ``qkv.unflatten(-1, (H, -1, 3))``). A ``(3, H, Dh)`` layout has identical shapes
and silently permutes every pretrained weight, so it is guarded by a value test.

References:
    - Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). "LightGlue: Local
      Feature Matching at Light Speed". arXiv:2306.13643.
"""

import math

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.common import (
    apply_attention_mask,
    compute_attention_scale,
    mask_dtype,
    validate_head_divisibility,
)
from dl_techniques.layers.matching.learned_fourier_rotary import apply_rotary_interleaved
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _validate_dims(dim: int, num_heads: int, layer_norm_epsilon: float) -> None:
    """Shared constructor validation for both blocks.

    :raises ValueError: If ``dim`` or ``num_heads`` is not positive, ``dim`` is not
        divisible by ``num_heads``, the head dimension is odd (the interleaved rotary
        needs channel pairs), or the epsilon is not positive.
    """
    if dim <= 0 or num_heads <= 0:
        raise ValueError(f"dim and num_heads must be positive, got {dim}, {num_heads}")
    validate_head_divisibility(dim, num_heads)
    if (dim // num_heads) % 2 != 0:
        raise ValueError(
            f"head_dim (dim // num_heads = {dim // num_heads}) must be even: the "
            "interleaved rotary encoding rotates adjacent channel pairs"
        )
    if layer_norm_epsilon <= 0:
        raise ValueError(f"layer_norm_epsilon must be positive, got {layer_norm_epsilon}")


def _make_ffn_sublayers(block: keras.layers.Layer, dim: int, epsilon: float) -> None:
    """Attach the FFN sublayers ``ffn_0``, ``ffn_1``, ``ffn_3`` to ``block``.

    Names follow the torch ``nn.Sequential`` indices (``ffn.0`` Linear, ``ffn.1``
    LayerNorm, ``ffn.2`` GELU which has no weights, ``ffn.3`` Linear).

    :param block: Owning layer; receives the three attributes.
    :param dim: Node width ``d``.
    :param epsilon: LayerNorm epsilon (reference 1e-5).
    """
    policy = block.dtype_policy
    block.ffn_0 = keras.layers.Dense(2 * dim, dtype=policy, name="ffn_0")
    block.ffn_1 = keras.layers.LayerNormalization(epsilon=epsilon, dtype=policy, name="ffn_1")
    block.ffn_3 = keras.layers.Dense(dim, dtype=policy, name="ffn_3")


def _build_ffn_sublayers(block: keras.layers.Layer, dim: int) -> None:
    """Explicitly build the three FFN sublayers for a ``(..., 2 * dim)`` input."""
    block.ffn_0.build((None, None, 2 * dim))
    block.ffn_1.build((None, None, 2 * dim))
    block.ffn_3.build((None, None, 2 * dim))


def _ffn_activation(x: Any) -> Any:
    """Exact (erf) GELU, torch ``nn.GELU()``. The tanh approximation is a different function.

    Spelled ``0.5 x (1 + erf(x / sqrt 2))`` with ``keras.ops.erf`` rather than
    ``keras.activations.gelu(approximate=False)``: measured under a float64 policy
    the TensorFlow backend's ``gelu`` is wrong by about 4.5e-9 (its ``sqrt(2)``
    constant is single precision), which the float64 parity test catches.
    """
    return 0.5 * x * (1.0 + keras.ops.erf(x * (1.0 / math.sqrt(2.0))))


def _ffn_residual(block: keras.layers.Layer, x: Any, message: Any) -> Any:
    """``x + ffn(concat([x, message]))`` with the block's ``ffn_0``/``ffn_1``/``ffn_3``.

    Interface contract: ``block`` must carry the three sublayers made by
    :func:`_make_ffn_sublayers` and built by :func:`_build_ffn_sublayers`; ``x`` and
    ``message`` are ``(B, N, d)`` of one dtype. Returns ``(B, N, d)``. Not a layer on
    purpose: the weights stay attributes of the block, matching the torch names.
    """
    hidden = keras.ops.concatenate([x, message], axis=-1)
    hidden = block.ffn_0(hidden)
    hidden = block.ffn_1(hidden)
    hidden = _ffn_activation(hidden)
    return x + block.ffn_3(hidden)


# DECISION plan-2026-10-02T084508-dd2c07ac/D-009
# The fused Wqkv columns are (H, Dh, 3), exactly torch's unflatten(-1, (H, -1, 3)). Do NOT
# switch to the more usual (3, H, Dh): the shapes are identical, so only values show it, and
# every converted official checkpoint would be silently permuted. Guards:
# tests/test_layers/test_matching/test_lightglue_blocks.py::TestMutationGuards::
# test_wrong_qkv_layout_is_caught and tests/test_models/test_lightglue/test_torch_reference.py.
# See decisions.md D-009.
def _split_qkv(qkv: Any, num_heads: int, head_dim: int) -> Tuple[Any, Any, Any]:
    """Split a fused ``(B, L, 3 * H * Dh)`` projection into q, k, v of ``(B, H, L, Dh)``.

    The column layout is ``(H, Dh, 3)`` (torch ``unflatten(-1, (H, -1, 3))``), NOT
    ``(3, H, Dh)``. Both give the same shapes.
    """
    shape = keras.ops.shape(qkv)
    grouped = keras.ops.reshape(qkv, (shape[0], shape[1], num_heads, head_dim, 3))
    grouped = keras.ops.transpose(grouped, (0, 2, 1, 3, 4))      # (B, H, L, Dh, 3)
    return grouped[..., 0], grouped[..., 1], grouped[..., 2]


def _split_heads(x: Any, num_heads: int, head_dim: int) -> Any:
    """``(B, N, H * Dh)`` -> ``(B, H, N, Dh)``."""
    shape = keras.ops.shape(x)
    x = keras.ops.reshape(x, (shape[0], shape[1], num_heads, head_dim))
    return keras.ops.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: Any, dim: int) -> Any:
    """``(B, H, N, Dh)`` -> ``(B, N, H * Dh)``."""
    x = keras.ops.transpose(x, (0, 2, 1, 3))
    shape = keras.ops.shape(x)
    return keras.ops.reshape(x, (shape[0], shape[1], dim))


def _check_build_shape(name: str, shape: Tuple[Any, ...], dim: int) -> None:
    if len(shape) != 3:
        raise ValueError(f"{name} must be (batch, num_points, dim), got {shape}")
    if shape[-1] is not None and shape[-1] != dim:
        raise ValueError(f"Last dimension of {name} ({shape[-1]}) must equal dim ({dim})")


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.matching.lightglue_blocks")
class LightGlueSelfBlock(keras.layers.Layer):
    """LightGlue self-attention block with learned-Fourier rotary and the concat-FFN.

    Computes ``x + ffn(concat([x, out_proj(attention(x))]))`` where the attention uses
    a fused ``Wqkv`` projection, rotary-encoded queries and keys and scale
    ``head_dim ** -0.5``.

    The softmax chain runs in at least float32 (``mask_dtype``) whatever the policy and
    the context is cast back to the compute dtype. A query row that keeps no key (a
    padded row, or an image with no real keypoints) is rescued by
    :func:`apply_attention_mask` into a finite spread over all keys, so every output is
    finite. Only rows of REAL keypoints are meaningful; the reference zeroes the other
    rows, this layer does not, and downstream code must ignore them.

    :param dim: Node width ``d``. Must be divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. ``dim // num_heads`` must be even.
    :type num_heads: int
    :param layer_norm_epsilon: Epsilon of the FFN LayerNorm. The reference uses 1e-5,
        which differs from the Keras default of 1e-3 and changes outputs.
    :type layer_norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar Wqkv: Fused projection ``Dense(3 * dim)``.
    :ivar out_proj: Message projection ``Dense(dim)``.
    :ivar ffn_0: ``Dense(2 * dim)``.
    :ivar ffn_1: ``LayerNormalization``.
    :ivar ffn_3: ``Dense(dim)``.

    Call arguments:
        x: ``(B, L, dim)`` node states.
        freqs: ``(2, B, 1, L, head_dim)`` rotary table from
            :class:`LearnedFourierRotaryEncoding`.
        mask: Optional ``(B, L)`` keep mask, bool or 0/1, 1 = real keypoint.

    Output shape:
        ``(B, L, dim)``, the dtype of ``x`` under the layer's compute dtype.

    :raises ValueError: From ``__init__`` for bad ``dim`` / ``num_heads`` / epsilon or an
        odd head dimension; from ``build()`` for a wrong rank or last dimension.

    Example:

    .. code-block:: python

        enc = LearnedFourierRotaryEncoding(head_dim=16)
        block = LightGlueSelfBlock(dim=64, num_heads=4)
        x = keras.random.normal((2, 50, 64))
        out = block(x, enc(keras.random.uniform((2, 50, 2))))   # (2, 50, 64)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 4,
            layer_norm_epsilon: float = 1e-5,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        _validate_dims(dim, num_heads, layer_norm_epsilon)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scale = compute_attention_scale(self.head_dim)

        policy = self.dtype_policy
        self.Wqkv = keras.layers.Dense(3 * dim, dtype=policy, name="Wqkv")
        self.out_proj = keras.layers.Dense(dim, dtype=policy, name="out_proj")
        _make_ffn_sublayers(self, dim, layer_norm_epsilon)

    def build(
            self,
            x_shape: Tuple[Any, ...],
            freqs_shape: Optional[Tuple[Any, ...]] = None,
            mask_shape: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        """Build every sublayer explicitly.

        :param x_shape: ``(batch, num_points, dim)``.
        :param freqs_shape: Rotary table shape; accepted for the call signature, unused.
        :param mask_shape: Mask shape; accepted for the call signature, unused.
        :raises ValueError: If ``x_shape`` is not rank 3 or its last dim is not ``dim``.
        """
        if self.built:
            return
        _check_build_shape("x", x_shape, self.dim)
        self.Wqkv.build((None, None, self.dim))
        self.out_proj.build((None, None, self.dim))
        _build_ffn_sublayers(self, self.dim)
        super().build(x_shape)

    def call(self, x: Any, freqs: Any, mask: Optional[Any] = None) -> Any:
        """Rotary self-attention then the concat-FFN residual.

        :param x: ``(B, L, dim)``.
        :param freqs: ``(2, B, 1, L, head_dim)``.
        :param mask: Optional ``(B, L)`` keep mask, 1 = real.
        :return: ``(B, L, dim)``.
        """
        q, k, v = _split_qkv(self.Wqkv(x), self.num_heads, self.head_dim)
        q = apply_rotary_interleaved(freqs, q)
        k = apply_rotary_interleaved(freqs, k)

        md = mask_dtype(self.compute_dtype)
        logits = keras.ops.matmul(
            keras.ops.cast(q, md), keras.ops.transpose(keras.ops.cast(k, md), (0, 1, 3, 2))
        ) * self.scale
        if mask is not None:
            real = keras.ops.cast(mask, md) > 0.0                          # (B, L)
            keep = keras.ops.logical_and(real[:, None, :, None], real[:, None, None, :])
            logits = apply_attention_mask(logits, keep)
        weights = keras.ops.softmax(logits, axis=-1)
        context = keras.ops.matmul(keras.ops.cast(weights, v.dtype), v)    # (B, H, L, Dh)

        message = self.out_proj(_merge_heads(context, self.dim))
        return _ffn_residual(self, x, message)

    def compute_output_shape(
            self,
            x_shape: Tuple[Any, ...],
            freqs_shape: Optional[Tuple[Any, ...]] = None,
            mask_shape: Optional[Tuple[Any, ...]] = None,
    ) -> Tuple[Any, ...]:
        """Output has the shape of ``x``."""
        return tuple(x_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.matching.lightglue_blocks")
class LightGlueCrossBlock(keras.layers.Layer):
    """LightGlue bidirectional cross-attention block.

    One shared ``to_qk`` and one shared ``to_v`` project both images. The similarity
    ``sim = (qk0 * s)(qk1 * s)^T`` with ``s = head_dim ** -0.25`` is computed ONCE
    and softmaxed over the other image's axis for each direction (rows for 0 -> 1,
    columns for 1 -> 0). Both directions then share ``to_out`` and the FFN, exactly as
    in the reference, so image 0 and image 1 use the same weights.

    Masking and numerics follow :class:`LightGlueSelfBlock`: float32-or-wider softmax,
    rescue of keep-nothing rows (finite, only real rows meaningful). When one image has
    no real keypoint, the rows of the OTHER image are rescued into a spread over padded
    keys; the reference would output zeros there.

    :param dim: Node width ``d``. Must be divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. ``dim // num_heads`` must be even
        (kept equal to the self block so one head count configures both).
    :type num_heads: int
    :param layer_norm_epsilon: Epsilon of the FFN LayerNorm (reference 1e-5).
    :type layer_norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar to_qk: ``Dense(dim)`` shared by both images.
    :ivar to_v: ``Dense(dim)`` shared by both images.
    :ivar to_out: ``Dense(dim)`` shared by both directions.
    :ivar ffn_0: ``Dense(2 * dim)``.
    :ivar ffn_1: ``LayerNormalization``.
    :ivar ffn_3: ``Dense(dim)``.

    Call arguments:
        x0: ``(B, M, dim)``.
        x1: ``(B, N, dim)``.
        mask0: Optional ``(B, M)`` keep mask, 1 = real.
        mask1: Optional ``(B, N)`` keep mask, 1 = real.

    Output shape:
        Tuple ``((B, M, dim), (B, N, dim))``.

    :raises ValueError: As :class:`LightGlueSelfBlock`.

    Example:

    .. code-block:: python

        block = LightGlueCrossBlock(dim=64, num_heads=4)
        y0, y1 = block(keras.random.normal((2, 50, 64)), keras.random.normal((2, 40, 64)))
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 4,
            layer_norm_epsilon: float = 1e-5,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        _validate_dims(dim, num_heads, layer_norm_epsilon)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        # sim scale 1/sqrt(Dh) split as Dh**-0.25 on each side
        self.side_scale = compute_attention_scale(self.head_dim) ** 0.5

        policy = self.dtype_policy
        self.to_qk = keras.layers.Dense(dim, dtype=policy, name="to_qk")
        self.to_v = keras.layers.Dense(dim, dtype=policy, name="to_v")
        self.to_out = keras.layers.Dense(dim, dtype=policy, name="to_out")
        _make_ffn_sublayers(self, dim, layer_norm_epsilon)

    def build(
            self,
            x0_shape: Tuple[Any, ...],
            x1_shape: Optional[Tuple[Any, ...]] = None,
            mask0_shape: Optional[Tuple[Any, ...]] = None,
            mask1_shape: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        """Build every sublayer explicitly.

        :param x0_shape: ``(batch, M, dim)``.
        :param x1_shape: ``(batch, N, dim)``.
        :param mask0_shape: Accepted for the call signature, unused.
        :param mask1_shape: Accepted for the call signature, unused.
        :raises ValueError: If a state shape is not rank 3 or its last dim is not ``dim``.
        """
        if self.built:
            return
        _check_build_shape("x0", x0_shape, self.dim)
        if x1_shape is not None:
            _check_build_shape("x1", x1_shape, self.dim)
        for layer in (self.to_qk, self.to_v, self.to_out):
            layer.build((None, None, self.dim))
        _build_ffn_sublayers(self, self.dim)
        super().build(x0_shape)

    def call(
            self,
            x0: Any,
            x1: Any,
            mask0: Optional[Any] = None,
            mask1: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Cross-attend in both directions from one similarity, then the shared FFN.

        :return: ``(y0, y1)`` of shapes ``(B, M, dim)`` and ``(B, N, dim)``.
        """
        h, dh = self.num_heads, self.head_dim
        qk0 = _split_heads(self.to_qk(x0), h, dh) * self.side_scale
        qk1 = _split_heads(self.to_qk(x1), h, dh) * self.side_scale
        v0 = _split_heads(self.to_v(x0), h, dh)
        v1 = _split_heads(self.to_v(x1), h, dh)

        md = mask_dtype(self.compute_dtype)
        sim = keras.ops.matmul(
            keras.ops.cast(qk0, md), keras.ops.transpose(keras.ops.cast(qk1, md), (0, 1, 3, 2))
        )                                                                   # (B, H, M, N)
        sim_t = keras.ops.transpose(sim, (0, 1, 3, 2))                      # (B, H, N, M)
        if mask0 is not None and mask1 is not None:
            real0 = keras.ops.cast(mask0, md) > 0.0
            real1 = keras.ops.cast(mask1, md) > 0.0
            keep = keras.ops.logical_and(real0[:, None, :, None], real1[:, None, None, :])
            sim = apply_attention_mask(sim, keep)
            sim_t = apply_attention_mask(sim_t, keras.ops.transpose(keep, (0, 1, 3, 2)))
        elif mask0 is not None or mask1 is not None:
            # One-sided mask: padded keys of that image are masked for the other image.
            # keep is (B, 1, 1, N) against sim (B, H, M, N), and (B, 1, 1, M) for the transpose.
            if mask1 is not None:
                real1 = keras.ops.cast(mask1, md) > 0.0
                sim = apply_attention_mask(sim, real1[:, None, None, :])
            else:
                real0 = keras.ops.cast(mask0, md) > 0.0
                sim_t = apply_attention_mask(sim_t, real0[:, None, None, :])

        attn0 = keras.ops.cast(keras.ops.softmax(sim, axis=-1), v1.dtype)    # 0 -> 1
        attn1 = keras.ops.cast(keras.ops.softmax(sim_t, axis=-1), v0.dtype)  # 1 -> 0
        ctx0 = keras.ops.matmul(attn0, v1)                                  # (B, H, M, Dh)
        ctx1 = keras.ops.matmul(attn1, v0)                                  # (B, H, N, Dh)

        m0 = self.to_out(_merge_heads(ctx0, self.dim))
        m1 = self.to_out(_merge_heads(ctx1, self.dim))
        return _ffn_residual(self, x0, m0), _ffn_residual(self, x1, m1)

    def compute_output_shape(
            self,
            x0_shape: Tuple[Any, ...],
            x1_shape: Optional[Tuple[Any, ...]] = None,
            mask0_shape: Optional[Tuple[Any, ...]] = None,
            mask1_shape: Optional[Tuple[Any, ...]] = None,
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Output shapes are those of ``x0`` and ``x1``."""
        return tuple(x0_shape), tuple(x1_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config

# ---------------------------------------------------------------------
