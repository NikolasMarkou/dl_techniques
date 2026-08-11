"""
SAM 2 Hiera trunk: the hierarchical, window-attention image backbone.
=====================================================================

Four public classes -- :class:`HieraPatchEmbed`,
:class:`HieraMultiScaleAttention`, :class:`HieraBlock` and :class:`Hiera` --
plus :func:`hiera_block_specs`, a pure configuration function that derives the
per-block geometry from a stage description without constructing anything.

Based on:
---------
- Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos."
- Ryali, C. et al. (2023). "Hiera: A Hierarchical Vision Transformer without
  the Bells-and-Whistles."

Key Features:
------------
- A learned background positional embedding plus a tiled window positional
  embedding, added ONCE, at the stem.
- One flat block list partitioned into four stages; at each boundary the
  channel width and the head count both double while the grid is halved by a
  max-pool applied to the attention QUERIES.
- Four feature levels out, one per stage, in ASCENDING stage order:
  ``outputs[0]`` is finest and narrowest, ``outputs[-1]`` coarsest and widest.
  That is the REVERSE of the channel list the FPN neck is configured with, and
  the reversal is deliberate on both sides.

Architecture Overview:
---------------------
1. **HieraPatchEmbed** -- one overlapping strided convolution, 4x reduction,
   spatial grid kept as ``(batch, height, width, channels)``.
2. **HieraMultiScaleAttention** -- windowed attention with query pooling.
3. **HieraBlock** -- attention and MLP, carrying the stage-boundary projection.
4. **Hiera** -- the block list, returning one feature map per stage.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM2.hiera import Hiera, hiera_block_specs
trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2))
levels = trunk(images)               # four maps, finest first
```

Measured caveats:
----------------
Two details are correctness bugs with NO shape error if ported wrong, so both
are called out at their code sites and guarded behaviourally by
``tests/test_models/test_sam2/test_hiera.py``:

- **The window size lags one block behind the stage transition.** The first
  block of a new stage uses the PREVIOUS stage's window size. See
  :func:`hiera_block_specs`.
- **Query pooling is asymmetric.** Inside attention only ``q`` is pooled; ``k``
  and ``v`` keep the full window resolution, and the residual shortcut is
  pooled by the same factor on a separate path. See
  :class:`HieraMultiScaleAttention` and :class:`HieraBlock`.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.stochastic_depth import StochasticDepth

# ---------------------------------------------------------------------


def hiera_block_specs(
        stages: Sequence[int],
        window_spec: Sequence[int],
        global_att_blocks: Optional[Sequence[int]] = None,
        q_pool: int = 3,
        embed_dim: int = 96,
        num_heads: int = 1,
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
) -> List[Dict[str, Any]]:
    """Derive the per-block geometry of a Hiera trunk. Pure function.

    This is the single home of the trunk's block schedule. It touches no
    tensors and holds no state, so a test can assert the whole schedule against
    a hand-derived table without constructing a model.

    .. warning::

        **The window size lags one block behind the stage transition.** For
        block ``i`` the window size is read as ``window_spec[cur_stage - 1]``
        BEFORE ``cur_stage`` is advanced for that same block. The consequence
        is that the first block of stage *n* uses stage *n-1*'s window size,
        while the remaining blocks of stage *n* use stage *n*'s. Choosing the
        window size after the increment — the "obvious" ordering — produces a
        model that builds, runs, trains and returns plausible numbers, with no
        shape error anywhere. Do not "tidy" this ordering.

    Blocks listed in ``global_att_blocks`` override the window size to ``0``,
    which means global (un-windowed) attention over the whole grid.

    :param stages: Number of blocks per stage, e.g. ``(2, 6, 36, 4)``.
    :type stages: Sequence[int]
    :param window_spec: Per-stage window size, same length as ``stages``.
    :type window_spec: Sequence[int]
    :param global_att_blocks: Absolute block indices forced to global attention.
    :type global_att_blocks: Optional[Sequence[int]]
    :param q_pool: How many of the ``len(stages) - 1`` stage transitions pool
        the queries. SAM 2 ships ``3``, i.e. all of them.
    :type q_pool: int
    :param embed_dim: Channel width of stage 1.
    :type embed_dim: int
    :param num_heads: Attention heads in stage 1.
    :type num_heads: int
    :param dim_mul: Channel-width multiplier applied at each stage boundary.
    :type dim_mul: float
    :param head_mul: Head-count multiplier applied at each stage boundary.
    :type head_mul: float
    :return: One dictionary per block with keys ``dim``, ``dim_out``,
        ``num_heads``, ``window_size``, ``q_pool``, ``stage`` (1-based) and
        ``is_stage_end``.
    :rtype: List[Dict[str, Any]]
    :raises ValueError: If ``stages`` and ``window_spec`` differ in length, if
        any entry is non-positive, or if ``q_pool`` is out of range.
    """
    if len(stages) != len(window_spec):
        raise ValueError(
            f"stages ({len(stages)} entries) and window_spec "
            f"({len(window_spec)} entries) must have the same length"
        )
    if len(stages) < 1:
        raise ValueError("stages must contain at least one entry")
    if any(int(s) <= 0 for s in stages):
        raise ValueError(f"every stage must hold at least one block, got {tuple(stages)}")
    if any(int(w) <= 0 for w in window_spec):
        raise ValueError(f"every window_spec entry must be positive, got {tuple(window_spec)}")
    if not 0 <= q_pool <= len(stages) - 1:
        raise ValueError(
            f"q_pool must be in [0, {len(stages) - 1}] for {len(stages)} "
            f"stages, got {q_pool}"
        )

    depth = int(sum(int(s) for s in stages))
    stage_ends = [
        sum(int(s) for s in stages[: k + 1]) - 1 for k in range(len(stages))
    ]
    q_pool_blocks = [end + 1 for end in stage_ends[:-1]][:q_pool]
    global_blocks = set(int(b) for b in (global_att_blocks or ()))

    specs: List[Dict[str, Any]] = []
    dim = int(embed_dim)
    heads = int(num_heads)
    cur_stage = 1

    for i in range(depth):
        dim_out = dim

        # DECISION plan-2026-08-04T044628-4c240b4c/D-010
        # The window size is read with the OLD `cur_stage`, i.e. BEFORE the
        # stage-transition block below can advance it. This one-block lag is
        # upstream's documented behaviour, not an accident. Moving these three
        # lines after the transition block is a silent correctness bug: shapes
        # are unaffected because the window size only changes how the grid is
        # partitioned, never the tensor geometry that leaves the block.
        window_size = int(window_spec[cur_stage - 1])
        if i in global_blocks:
            window_size = 0

        if i - 1 in stage_ends:
            dim_out = int(dim * dim_mul)
            heads = int(heads * head_mul)
            cur_stage += 1

        specs.append({
            "dim": dim,
            "dim_out": dim_out,
            "num_heads": heads,
            "window_size": window_size,
            "q_pool": i in q_pool_blocks,
            "stage": cur_stage,
            "is_stage_end": i in stage_ends,
        })
        dim = dim_out

    return specs


# ---------------------------------------------------------------------
# window plumbing (module-private, static shapes only)
# ---------------------------------------------------------------------


def _window_partition(
        x: Any,
        window_size: int,
        height: int,
        width: int,
        channels: int,
) -> Tuple[Any, Tuple[int, int]]:
    """Split ``(B, H, W, C)`` into ``(B * num_windows, ws, ws, C)``.

    When the grid is not divisible by the window size the bottom and right
    edges are ZERO-padded and the padded tokens are **not** masked out of the
    attention softmax — they participate fully, contributing a bias toward the
    projection of a zero feature vector. This is upstream SAM 2's accepted
    behaviour, reproduced here deliberately. Do NOT "fix" it by adding an
    attention mask: that would change every windowed block's output on any
    non-divisible grid and silently diverge from the reference model (and from
    any converted upstream checkpoint) with no shape error to notice it.

    :param x: Input tensor ``(batch, height, width, channels)``.
    :type x: Any
    :param window_size: Square window edge length.
    :type window_size: int
    :param height: Static grid height.
    :type height: int
    :param width: Static grid width.
    :type width: int
    :param channels: Static channel count.
    :type channels: int
    :return: The windowed tensor and the padded grid ``(padded_h, padded_w)``.
    :rtype: Tuple[Any, Tuple[int, int]]
    """
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    padded_h, padded_w = height + pad_h, width + pad_w

    x = ops.reshape(
        x,
        (-1, padded_h // window_size, window_size,
         padded_w // window_size, window_size, channels),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, window_size, window_size, channels))
    return x, (padded_h, padded_w)


def _window_unpartition(
        x: Any,
        window_size: int,
        pad_hw: Tuple[int, int],
        hw: Tuple[int, int],
        channels: int,
) -> Any:
    """Invert :func:`_window_partition` and crop the padding away.

    :param x: Windowed tensor ``(batch * num_windows, ws, ws, channels)``.
    :type x: Any
    :param window_size: Square window edge length.
    :type window_size: int
    :param pad_hw: Padded grid ``(padded_h, padded_w)``.
    :type pad_hw: Tuple[int, int]
    :param hw: Unpadded target grid ``(height, width)``.
    :type hw: Tuple[int, int]
    :param channels: Static channel count.
    :type channels: int
    :return: Tensor ``(batch, height, width, channels)``.
    :rtype: Any
    """
    padded_h, padded_w = pad_hw
    height, width = hw
    x = ops.reshape(
        x,
        (-1, padded_h // window_size, padded_w // window_size,
         window_size, window_size, channels),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, padded_h, padded_w, channels))
    if padded_h > height or padded_w > width:
        x = x[:, :height, :width, :]
    return x


def _do_pool(x: Any, q_stride: Tuple[int, int]) -> Any:
    """Max-pool ``(B, H, W, C)`` by ``q_stride``, floor semantics, no padding.

    :param x: Input tensor.
    :type x: Any
    :param q_stride: Pooling window and stride ``(sh, sw)``.
    :type q_stride: Tuple[int, int]
    :return: Pooled tensor.
    :rtype: Any
    """
    return ops.max_pool(x, pool_size=q_stride, strides=q_stride, padding="valid")


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HieraPatchEmbed(keras.layers.Layer):
    """Hiera's stem: an OVERLAPPING strided convolution that keeps the grid.

    Unlike a ViT patch embedding, the kernel is larger than the stride
    (``7`` vs ``4``) and the input is explicitly zero-padded by ``3`` on every
    side, so receptive fields of neighbouring output cells overlap. The output
    keeps its spatial layout as ``(B, H // stride, W // stride, embed_dim)``; it
    is **not** flattened to a token sequence, because every downstream block
    partitions it spatially.

    :param embed_dim: Output channel width.
    :type embed_dim: int
    :param kernel_size: Convolution kernel edge length. Defaults to ``7``.
    :type kernel_size: int
    :param stride: Convolution stride. Defaults to ``4``.
    :type stride: int
    :param padding: Symmetric zero-padding applied before the convolution.
        Defaults to ``3``.
    :type padding: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If any of ``embed_dim``, ``kernel_size`` or ``stride``
        is non-positive, or if ``padding`` is negative.

    Example:
        >>> import numpy as np
        >>> stem = HieraPatchEmbed(embed_dim=16)
        >>> stem(np.zeros((1, 64, 64, 3), dtype="float32")).shape
        (1, 16, 16, 16)
    """

    def __init__(
            self,
            embed_dim: int = 96,
            kernel_size: int = 7,
            stride: int = 4,
            padding: int = 3,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}")

        self.embed_dim = int(embed_dim)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.pad = keras.layers.ZeroPadding2D(
            padding=self.padding, name="pad")
        self.proj = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",
            name="proj",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the convolution.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank-4.
        """
        if self.built:
            return
        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"HieraPatchEmbed expects a rank-4 channels-last input, got "
                f"shape {input_shape}"
            )
        self.pad.build(input_shape)
        self.proj.build(self.pad.compute_output_shape(input_shape))
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Embed an image into a spatial grid of features.

        :param inputs: ``(batch, height, width, channels)``.
        :type inputs: Any
        :return: ``(batch, out_h, out_w, embed_dim)``.
        :rtype: Any
        """
        return self.proj(self.pad(inputs))

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, out_h, out_w, embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch, height, width, _ = tuple(input_shape)
        out_h = out_w = None
        if height is not None:
            out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        if width is not None:
            out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return (batch, out_h, out_w, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HieraMultiScaleAttention(keras.layers.Layer):
    """Hiera's mask-unit attention, with ASYMMETRIC query pooling.

    Operates on a spatial grid ``(B, H, W, dim)`` — when the enclosing
    :class:`HieraBlock` is windowed, ``B`` is the batch times the window count
    and ``H = W = window_size``.

    .. warning::

        When ``q_stride`` is set, **only the queries are pooled**, and the
        pooling happens AFTER the qkv projection and the head split. Keys and
        values keep the full ``H * W`` token count, so the attention matrix is
        rectangular: ``(B, heads, H' * W', H * W)``. Pooling ``k`` and ``v`` as
        well is a plausible-looking "symmetry fix" that runs, trains, and
        silently changes what every stage-transition block computes.

    **Data flow:**

    .. code-block:: text

        x (B, H, W, dim)
          └─ qkv ─► (B, H*W, 3, heads, head_dim)
                     ├─ q ─► max_pool(q_stride) ─► (B, H'*W', heads, head_dim)
                     ├─ k ────────────────────────► (B, H*W,   heads, head_dim)
                     └─ v ────────────────────────► (B, H*W,   heads, head_dim)
                        softmax(q k^T / sqrt(head_dim)) · v
                        └─► (B, H', W', dim_out) ─► proj

    :param dim: Input channel width.
    :type dim: int
    :param dim_out: Output channel width. Equals ``dim`` inside a stage and
        ``dim * dim_mul`` at a stage transition.
    :type dim_out: int
    :param num_heads: Attention heads. Must divide ``dim_out``.
    :type num_heads: int
    :param q_stride: Query pooling window/stride, or ``None`` for no pooling.
    :type q_stride: Optional[Sequence[int]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim`` or ``dim_out`` is non-positive, or if
        ``dim_out`` is not divisible by ``num_heads``.

    Example:
        >>> import numpy as np
        >>> attn = HieraMultiScaleAttention(dim=16, dim_out=32, num_heads=2,
        ...                                 q_stride=(2, 2))
        >>> attn(np.zeros((1, 4, 4, 16), dtype="float32")).shape
        (1, 2, 2, 32)
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int = 1,
            q_stride: Optional[Sequence[int]] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim_out <= 0:
            raise ValueError(f"dim_out must be positive, got {dim_out}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim_out % num_heads != 0:
            raise ValueError(
                f"dim_out ({dim_out}) must be divisible by num_heads "
                f"({num_heads})"
            )

        self.dim = int(dim)
        self.dim_out = int(dim_out)
        self.num_heads = int(num_heads)
        self.q_stride = (
            (int(q_stride[0]), int(q_stride[1])) if q_stride is not None else None
        )

        # Derived, non-config.
        self.head_dim = self.dim_out // self.num_heads
        self._scale = float(self.head_dim) ** -0.5

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.qkv = keras.layers.Dense(self.dim_out * 3, name="qkv")
        self.proj = keras.layers.Dense(self.dim_out, name="proj")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the qkv and output projections.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank-4.
        """
        if self.built:
            return
        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"HieraMultiScaleAttention expects a rank-4 channels-last "
                f"input, got shape {input_shape}"
            )
        self.qkv.build(input_shape)
        self.proj.build(self.compute_output_shape(input_shape)[:-1] + (self.dim_out,))
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Run mask-unit attention with query-only pooling.

        :param inputs: ``(batch, height, width, dim)`` with STATIC spatial dims.
        :type inputs: Any
        :return: ``(batch, out_h, out_w, dim_out)``.
        :rtype: Any
        :raises ValueError: If the spatial dimensions are not statically known.
        """
        height, width = self._static_grid(inputs)
        num_tokens = height * width

        qkv = self.qkv(inputs)
        qkv = ops.reshape(
            qkv, (-1, num_tokens, 3, self.num_heads, self.head_dim))
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        out_h, out_w = height, width
        if self.q_stride is not None:
            # Queries only. `k` and `v` deliberately keep `num_tokens` rows.
            q = ops.reshape(q, (-1, height, width, self.dim_out))
            q = _do_pool(q, self.q_stride)
            out_h = height // self.q_stride[0]
            out_w = width // self.q_stride[1]
            q = ops.reshape(
                q, (-1, out_h * out_w, self.num_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = ops.matmul(q * self._scale, ops.transpose(k, (0, 1, 3, 2)))
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)

        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, out_h, out_w, self.dim_out))
        return self.proj(out)

    @staticmethod
    def _static_grid(x: Any) -> Tuple[int, int]:
        """Read the statically-known spatial dimensions of a rank-4 tensor.

        :param x: Tensor ``(batch, height, width, channels)``.
        :type x: Any
        :return: ``(height, width)``.
        :rtype: Tuple[int, int]
        :raises ValueError: If either spatial dimension is dynamic.
        """
        height, width = x.shape[1], x.shape[2]
        if height is None or width is None:
            raise ValueError(
                f"Hiera requires STATIC spatial dimensions; got shape "
                f"{x.shape}. Trace with a static input signature."
            )
        return int(height), int(width)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, out_h, out_w, dim_out)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch, height, width, _ = tuple(input_shape)
        if self.q_stride is not None:
            if height is not None:
                height = height // self.q_stride[0]
            if width is not None:
                width = width // self.q_stride[1]
        return (batch, height, width, self.dim_out)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "dim_out": self.dim_out,
            "num_heads": self.num_heads,
            "q_stride": self.q_stride,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HieraBlock(keras.layers.Layer):
    """One Hiera block: windowed mask-unit attention plus an MLP.

    Both sub-blocks are pre-norm with a residual add. At a stage transition
    (``dim != dim_out``) the residual shortcut is projected to the new width and
    pooled by ``q_stride``, mirroring the query pooling that happens inside the
    attention — one logical pooling operation applied on two separate paths.

    **Architecture:**

    .. code-block:: text

        x ──► norm1 ──┬─► proj ─► pool ─────────────────► shortcut
                      └─► window_partition ─► attn ─► window_unpartition ─┐
                                                    shortcut + drop_path ◄┘
              └─► norm2 ─► Dense ─► act ─► Dense ─► drop_path ─► + ──► out

    The window size used to REASSEMBLE the grid after attention is the block's
    window size divided by ``q_stride[0]``, and the padded extent is recomputed
    from the ALREADY-POOLED shortcut — never reused from the pre-pool
    partitioning.

    :param dim: Input channel width.
    :type dim: int
    :param dim_out: Output channel width.
    :type dim_out: int
    :param num_heads: Attention heads.
    :type num_heads: int
    :param mlp_ratio: MLP hidden width as a multiple of ``dim_out``. Defaults to
        ``4.0``.
    :type mlp_ratio: float
    :param drop_path: Stochastic-depth rate on both residual branches. Defaults
        to ``0.0``.
    :type drop_path: float
    :param q_stride: Query/shortcut pooling window and stride, or ``None``.
    :type q_stride: Optional[Sequence[int]]
    :param window_size: Attention window edge length; ``0`` means global
        attention over the whole grid. Defaults to ``0``.
    :type window_size: int
    :param activation: MLP hidden activation. Defaults to ``'gelu'``.
    :type activation: str
    :param layer_norm_epsilon: Epsilon of both layer normalizations. Defaults to
        ``1e-6``.
    :type layer_norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``window_size`` is negative, if ``mlp_ratio`` is
        non-positive, or if the attention geometry is invalid (propagated).

    Example:
        >>> import numpy as np
        >>> block = HieraBlock(dim=16, dim_out=32, num_heads=2,
        ...                    q_stride=(2, 2), window_size=4)
        >>> block(np.zeros((1, 8, 8, 16), dtype="float32")).shape
        (1, 4, 4, 32)
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int = 1,
            mlp_ratio: float = 4.0,
            drop_path: float = 0.0,
            q_stride: Optional[Sequence[int]] = None,
            window_size: int = 0,
            activation: str = "gelu",
            layer_norm_epsilon: float = 1e-6,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if window_size < 0:
            raise ValueError(f"window_size must be non-negative, got {window_size}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if q_stride is not None and 0 < window_size < int(q_stride[0]):
            # The post-attention window is `window_size // q_stride[0]`; below
            # 1 the grid cannot be reassembled at all.
            raise ValueError(
                f"a query-pooling block needs window_size >= q_stride[0]; got "
                f"window_size={window_size}, q_stride={tuple(q_stride)}"
            )

        self.dim = int(dim)
        self.dim_out = int(dim_out)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.drop_path = float(drop_path)
        self.q_stride = (
            (int(q_stride[0]), int(q_stride[1])) if q_stride is not None else None
        )
        self.window_size = int(window_size)
        self.activation = activation
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        # Derived, non-config.
        self.hidden_dim = int(self.dim_out * self.mlp_ratio)
        self.activation_fn = keras.activations.get(self.activation)

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="norm1")
        self.attn = HieraMultiScaleAttention(
            dim=self.dim,
            dim_out=self.dim_out,
            num_heads=self.num_heads,
            q_stride=self.q_stride,
            name="attn",
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="norm2")
        self.mlp_fc1 = keras.layers.Dense(self.hidden_dim, name="mlp_fc1")
        self.mlp_fc2 = keras.layers.Dense(self.dim_out, name="mlp_fc2")
        # DECISION plan-2026-08-04T044628-4c240b4c/D-012
        # `proj` exists only on stage-transition blocks upstream. It is created
        # here unconditionally (the repo's authoring rule forbids conditional
        # sub-layer creation) but is BUILT only when it is used, so an unused
        # instance contributes zero weights and the parameter count still
        # matches the reference model.
        # Do NOT "simplify" either half: creating it conditionally breaks the
        # authoring rule, and building it unconditionally silently inflates
        # every non-transition block by `dim * dim_out + dim_out` parameters,
        # which step 8's `hiera_l` parameter audit would then have to absorb as
        # a fudge factor. See decisions.md D-012.
        self.proj = keras.layers.Dense(self.dim_out, name="proj")
        self.drop_path_layer = StochasticDepth(
            drop_path_rate=self.drop_path, name="drop_path")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer explicitly.

        :param input_shape: ``(batch, height, width, dim)`` with static spatial
            dimensions.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank-4, if its width does
            not equal ``dim``, or if the spatial dimensions are dynamic.
        """
        if self.built:
            return

        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"HieraBlock expects a rank-4 channels-last input, got shape "
                f"{input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"input width ({input_shape[-1]}) must equal dim ({self.dim})"
            )
        height, width = input_shape[1], input_shape[2]
        if height is None or width is None:
            raise ValueError(
                f"HieraBlock requires STATIC spatial dimensions, got shape "
                f"{input_shape}"
            )

        self.norm1.build(input_shape)
        if self.dim != self.dim_out:
            self.proj.build(input_shape)

        # Attention sees one window at a time when the block is windowed.
        if self.window_size > 0:
            attn_shape = (None, self.window_size, self.window_size, self.dim)
        else:
            attn_shape = (None, height, width, self.dim)
        self.attn.build(attn_shape)

        out_shape = self.compute_output_shape(input_shape)
        self.norm2.build(out_shape)
        self.mlp_fc1.build(out_shape)
        self.mlp_fc2.build((*out_shape[:-1], self.hidden_dim))
        self.drop_path_layer.build(out_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Run windowed attention and the MLP.

        :param inputs: ``(batch, height, width, dim)``.
        :type inputs: Any
        :param training: Keras training flag; gates stochastic depth.
        :type training: Optional[bool]
        :return: ``(batch, out_h, out_w, dim_out)``.
        :rtype: Any
        """
        height, width = HieraMultiScaleAttention._static_grid(inputs)

        x = self.norm1(inputs)

        # Stage transition: project AND pool the shortcut. The pooling factor is
        # the same `q_stride` the attention applies to its queries, but this is
        # a second, independent application on the residual path.
        if self.dim != self.dim_out:
            shortcut = self.proj(x)
            if self.q_stride is not None:
                shortcut = _do_pool(shortcut, self.q_stride)
        else:
            shortcut = inputs

        window_size = self.window_size
        pad_hw = (height, width)
        if window_size > 0:
            x, pad_hw = _window_partition(
                x, window_size, height, width, self.dim)

        x = self.attn(x)

        out_h, out_w = height, width
        if self.q_stride is not None:
            out_h = height // self.q_stride[0]
            out_w = width // self.q_stride[1]
            # Recomputed from the POOLED grid, never reused from the pre-pool
            # partition above.
            if window_size > 0:
                window_size = window_size // self.q_stride[0]
                pad_h = (window_size - out_h % window_size) % window_size
                pad_w = (window_size - out_w % window_size) % window_size
                pad_hw = (out_h + pad_h, out_w + pad_w)

        if self.window_size > 0:
            x = _window_unpartition(
                x, window_size, pad_hw, (out_h, out_w), self.dim_out)

        x = shortcut + self.drop_path_layer(x, training=training)
        hidden = self.activation_fn(self.mlp_fc1(self.norm2(x)))
        x = x + self.drop_path_layer(self.mlp_fc2(hidden), training=training)
        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, out_h, out_w, dim_out)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch, height, width, _ = tuple(input_shape)
        if self.q_stride is not None:
            if height is not None:
                height = height // self.q_stride[0]
            if width is not None:
                width = width // self.q_stride[1]
        return (batch, height, width, self.dim_out)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "dim_out": self.dim_out,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "drop_path": self.drop_path,
            "q_stride": self.q_stride,
            "window_size": self.window_size,
            "activation": self.activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Hiera(keras.layers.Layer):
    """The SAM 2 Hiera vision trunk.

    Returns one feature level per stage, in **ascending stage order**: level 0
    is the finest and narrowest, level ``-1`` the coarsest and widest.

    ``MODEL_VARIANTS`` is the SINGLE home of every shipped trunk geometry,
    including the small ``tiny`` geometry used by the test suite. Do not restate
    a variant's numbers anywhere else — read them from here, or construct
    through :meth:`from_variant`.

    The ``tiny`` geometry is not an arbitrary shrink; it is chosen so that the
    small model exercises every mechanism the large one has:

    * four stages, so the stage-transition schedule is real;
    * ``global_att_blocks=(2,)``, so at least one block is un-windowed;
    * ``window_spec[2] = 3`` against that stage's ``4 x 4`` grid, so the
      non-divisible zero-pad path is exercised;
    * ``q_pool=3``, so all three query-pooling transitions are live;
    * two blocks in the final stage, so ``window_spec[3]`` is actually read
      (with one block it never would be, because of the one-block lag);
    * total stride 32 and ``dim // num_heads == 16`` at every stage, so the head
      width stays a multiple of 4.

    :param embed_dim: Channel width of stage 1.
    :type embed_dim: int
    :param num_heads: Attention heads in stage 1.
    :type num_heads: int
    :param stages: Blocks per stage.
    :type stages: Sequence[int]
    :param global_att_blocks: Absolute block indices forced to global attention.
    :type global_att_blocks: Optional[Sequence[int]]
    :param window_spec: Per-stage window size.
    :type window_spec: Sequence[int]
    :param window_pos_embed_bkg_spatial_size: Spatial size of the learned
        background positional embedding, bicubic-resized to the stem grid.
    :type window_pos_embed_bkg_spatial_size: Sequence[int]
    :param q_pool: Number of stage transitions that pool the queries.
    :type q_pool: int
    :param q_stride: Query/shortcut pooling window and stride.
    :type q_stride: Sequence[int]
    :param dim_mul: Channel multiplier per stage transition.
    :type dim_mul: float
    :param head_mul: Head multiplier per stage transition.
    :type head_mul: float
    :param mlp_ratio: Per-block MLP expansion.
    :type mlp_ratio: float
    :param drop_path_rate: Maximum stochastic-depth rate; ramps linearly from
        ``0`` across the block list.
    :type drop_path_rate: float
    :param image_size: Input resolution this geometry is designed for. Used to
        derive the stem grid when the input shape is not fully known, and
        validated against the input shape when it is.
    :type image_size: int
    :param patch_kernel_size: Stem convolution kernel.
    :type patch_kernel_size: int
    :param patch_stride: Stem convolution stride.
    :type patch_stride: int
    :param patch_padding: Stem zero-padding.
    :type patch_padding: int
    :param activation: Per-block MLP activation.
    :type activation: str
    :param layer_norm_epsilon: Epsilon of every layer normalization.
    :type layer_norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If the block schedule is invalid (propagated from
        :func:`hiera_block_specs`), or if the stem grid is not divisible by
        ``window_spec[0]`` (which the tiled window positional embedding
        requires).

    Example:
        >>> import numpy as np
        >>> trunk = Hiera.from_variant("tiny")
        >>> levels = trunk(np.zeros((1, 64, 64, 3), dtype="float32"))
        >>> [tuple(level.shape) for level in levels]
        [(1, 16, 16, 16), (1, 8, 8, 32), (1, 4, 4, 64), (1, 2, 2, 128)]
    """

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # The small geometry used by the test suite. Structurally faithful:
        # see the class docstring for what each number buys.
        "tiny": {
            "embed_dim": 16,
            "num_heads": 1,
            "stages": (1, 2, 1, 2),
            "global_att_blocks": (2,),
            "window_spec": (4, 2, 3, 2),
            "window_pos_embed_bkg_spatial_size": (7, 7),
            "q_pool": 3,
            "image_size": 64,
        },
        # SAM 2.1-L. `num_heads=2` gives a head width of 72 at every stage,
        # which is a multiple of 4 as the axial RoPE downstream requires.
        "hiera_l": {
            "embed_dim": 144,
            "num_heads": 2,
            "stages": (2, 6, 36, 4),
            "global_att_blocks": (23, 33, 43),
            "window_spec": (8, 4, 16, 8),
            "window_pos_embed_bkg_spatial_size": (7, 7),
            "q_pool": 3,
            "image_size": 1024,
        },
    }

    def __init__(
            self,
            embed_dim: int = 96,
            num_heads: int = 1,
            stages: Sequence[int] = (2, 3, 16, 3),
            global_att_blocks: Optional[Sequence[int]] = (12, 16, 20),
            window_spec: Sequence[int] = (8, 4, 14, 7),
            window_pos_embed_bkg_spatial_size: Sequence[int] = (7, 7),
            q_pool: int = 3,
            q_stride: Sequence[int] = (2, 2),
            dim_mul: float = 2.0,
            head_mul: float = 2.0,
            mlp_ratio: float = 4.0,
            drop_path_rate: float = 0.0,
            image_size: int = 1024,
            patch_kernel_size: int = 7,
            patch_stride: int = 4,
            patch_padding: int = 3,
            activation: str = "gelu",
            layer_norm_epsilon: float = 1e-6,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}")

        # Store ALL configuration parameters.
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.stages = tuple(int(s) for s in stages)
        self.global_att_blocks = (
            tuple(int(b) for b in global_att_blocks)
            if global_att_blocks is not None else None
        )
        self.window_spec = tuple(int(w) for w in window_spec)
        self.window_pos_embed_bkg_spatial_size = (
            int(window_pos_embed_bkg_spatial_size[0]),
            int(window_pos_embed_bkg_spatial_size[1]),
        )
        self.q_pool = int(q_pool)
        self.q_stride = (int(q_stride[0]), int(q_stride[1]))
        self.dim_mul = float(dim_mul)
        self.head_mul = float(head_mul)
        self.mlp_ratio = float(mlp_ratio)
        self.drop_path_rate = float(drop_path_rate)
        self.image_size = int(image_size)
        self.patch_kernel_size = int(patch_kernel_size)
        self.patch_stride = int(patch_stride)
        self.patch_padding = int(patch_padding)
        self.activation = activation
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        # Derived, non-config.
        self.block_specs = hiera_block_specs(
            stages=self.stages,
            window_spec=self.window_spec,
            global_att_blocks=self.global_att_blocks,
            q_pool=self.q_pool,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dim_mul=self.dim_mul,
            head_mul=self.head_mul,
        )
        self.depth = len(self.block_specs)
        self.stage_ends = [
            sum(self.stages[: k + 1]) - 1 for k in range(len(self.stages))
        ]
        #: Output channel widths in DESCENDING stage order — the order the FPN
        #: neck's `backbone_channel_list` uses. The forward pass returns levels
        #: in the opposite (ascending) order.
        self.channel_list = [
            self.block_specs[end]["dim_out"] for end in self.stage_ends[::-1]
        ]

        # Linear stochastic-depth decay across depth. The helper already covers
        # both branches this used to spell out by hand: at `depth <= 1` it
        # returns `[0.0] * depth`, and at `drop_path_rate == 0.0` its step is
        # 0.0 so every element rounds to exactly 0.0 (measured, not assumed).
        drop_path_schedule = linear_drop_path_rates(
            self.depth, self.drop_path_rate
        )

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.patch_embed = HieraPatchEmbed(
            embed_dim=self.embed_dim,
            kernel_size=self.patch_kernel_size,
            stride=self.patch_stride,
            padding=self.patch_padding,
            name="patch_embed",
        )
        self.blocks = [
            HieraBlock(
                dim=spec["dim"],
                dim_out=spec["dim_out"],
                num_heads=spec["num_heads"],
                mlp_ratio=self.mlp_ratio,
                drop_path=drop_path_schedule[index],
                q_stride=self.q_stride if spec["q_pool"] else None,
                window_size=spec["window_size"],
                activation=self.activation,
                layer_norm_epsilon=self.layer_norm_epsilon,
                name=f"block_{index}",
            )
            for index, spec in enumerate(self.block_specs)
        ]

        self.pos_embed = None
        self.pos_embed_window = None
        self._stem_grid: Tuple[int, int] = (0, 0)

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "Hiera":
        """Construct a trunk from :attr:`MODEL_VARIANTS`.

        :param variant: Variant key, e.g. ``'tiny'`` or ``'hiera_l'``.
        :type variant: str
        :param kwargs: Explicit overrides; any value given here wins over the
            variant table.
        :type kwargs: Any
        :return: The configured trunk.
        :rtype: Hiera
        :raises ValueError: If ``variant`` is not a known key.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown Hiera variant '{variant}'. Available: "
                f"{sorted(cls.MODEL_VARIANTS)}"
            )
        config = dict(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)
        logger.info("Creating Hiera trunk variant '%s'", variant)
        return cls(**config)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the stem, the positional embeddings and every block.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank-4, or if the stem
            grid is not divisible by ``window_spec[0]``.
        """
        if self.built:
            return

        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"Hiera expects a rank-4 channels-last input, got shape "
                f"{input_shape}"
            )
        height = input_shape[1] if input_shape[1] is not None else self.image_size
        width = input_shape[2] if input_shape[2] is not None else self.image_size
        stem_shape = (None, int(height), int(width), input_shape[3])

        self.patch_embed.build(stem_shape)
        stem_out = self.patch_embed.compute_output_shape(stem_shape)
        grid_h, grid_w = int(stem_out[1]), int(stem_out[2])
        self._stem_grid = (grid_h, grid_w)

        window_edge = self.window_spec[0]
        if grid_h % window_edge != 0 or grid_w % window_edge != 0:
            raise ValueError(
                f"the stem grid ({grid_h}x{grid_w}) must be divisible by "
                f"window_spec[0] ({window_edge}); the window positional "
                f"embedding is TILED, not interpolated, so a non-integer tile "
                f"count cannot be expressed"
            )

        bkg_h, bkg_w = self.window_pos_embed_bkg_spatial_size
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, bkg_h, bkg_w, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.pos_embed_window = self.add_weight(
            name="pos_embed_window",
            shape=(1, window_edge, window_edge, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )

        shape = (None, grid_h, grid_w, self.embed_dim)
        for block in self.blocks:
            block.build(shape)
            shape = block.compute_output_shape(shape)

        logger.debug(
            "Hiera built: stem grid %dx%d, %d blocks, channel_list %s",
            grid_h, grid_w, self.depth, self.channel_list,
        )
        super().build(input_shape)

    def _get_pos_embed(self) -> Any:
        """Build the stem positional embedding for the static stem grid.

        The learned background embedding is bicubic-resized from its stored
        ``(bkg_h, bkg_w)`` size to the stem grid; the learned window embedding
        is TILED (repeated, never interpolated) to the same extent; the two are
        summed.

        :return: ``(1, grid_h, grid_w, embed_dim)``.
        :rtype: Any
        """
        grid_h, grid_w = self._stem_grid
        # The target size is a plain Python tuple of ints derived from config,
        # so this resize is graph-traceable.
        pos = ops.image.resize(
            self.pos_embed, (grid_h, grid_w), interpolation="bicubic")
        window_edge = self.window_spec[0]
        window = ops.tile(
            self.pos_embed_window,
            (1, grid_h // window_edge, grid_w // window_edge, 1),
        )
        return pos + window

    def call(self, inputs: Any, training: Optional[bool] = None) -> List[Any]:
        """Run the trunk.

        :param inputs: ``(batch, height, width, channels)``.
        :type inputs: Any
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: One feature map per stage, in ASCENDING stage order (finest
            first).
        :rtype: List[Any]
        """
        x = self.patch_embed(inputs)
        x = x + ops.cast(self._get_pos_embed(), x.dtype)

        outputs: List[Any] = []
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
            if index in self.stage_ends:
                outputs.append(x)
        return outputs

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> List[Tuple[Optional[int], ...]]:
        """Return one output shape per stage, derived from stored config.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: One shape per stage, ascending stage order.
        :rtype: List[Tuple[Optional[int], ...]]
        """
        shape = self.patch_embed.compute_output_shape(tuple(input_shape))
        shapes: List[Tuple[Optional[int], ...]] = []
        for index, block in enumerate(self.blocks):
            shape = block.compute_output_shape(shape)
            if index in self.stage_ends:
                shapes.append(shape)
        return shapes

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "stages": self.stages,
            "global_att_blocks": self.global_att_blocks,
            "window_spec": self.window_spec,
            "window_pos_embed_bkg_spatial_size":
                self.window_pos_embed_bkg_spatial_size,
            "q_pool": self.q_pool,
            "q_stride": self.q_stride,
            "dim_mul": self.dim_mul,
            "head_mul": self.head_mul,
            "mlp_ratio": self.mlp_ratio,
            "drop_path_rate": self.drop_path_rate,
            "image_size": self.image_size,
            "patch_kernel_size": self.patch_kernel_size,
            "patch_stride": self.patch_stride,
            "patch_padding": self.patch_padding,
            "activation": self.activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config

# ---------------------------------------------------------------------
