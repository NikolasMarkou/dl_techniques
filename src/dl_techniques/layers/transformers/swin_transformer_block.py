"""
Swin Transformer Block Implementation

This module implements the core SwinTransformerBlock layer, which forms the fundamental
building block of the Swin Transformer architecture - a hierarchical vision_heads transformer
that has revolutionized computer vision_heads by achieving state-of-the-art performance across
multiple tasks including image classification, object detection, and semantic segmentation.

Architecture Overview
--------------------

The Swin Transformer introduces a paradigm shift from traditional Vision Transformers (ViTs)
by replacing global self-attention with a more computationally efficient windowed attention
mechanism. The name "Swin" stands for "Shifted Windows", which refers to the key innovation
that enables the model to capture both local and global dependencies effectively.

Key Innovations:

1. **Windowed Multi-Head Self-Attention (W-MSA)**: Divides input feature maps into
   non-overlapping windows and computes self-attention within each window, reducing
   computational complexity from O((HW)²) to O(M²HW), where M is the fixed window size.

2. **Shifted Window Multi-Head Self-Attention (SW-MSA)**: Shifts windows by ⌊M/2⌋ pixels
   in both horizontal and vertical directions, enabling cross-window connections and
   information exchange between neighboring windows.

3. **Hierarchical Architecture**: Uses a pyramid-like structure with patch merging layers
   that progressively reduce spatial resolution while increasing feature dimensions,
   similar to CNNs but maintaining transformer capabilities.

4. **Linear Computational Complexity**: Achieves O(HW) complexity with respect to input
   size, making it suitable for high-resolution images and dense prediction tasks.

Mathematical Formulation
-----------------------

Each SwinTransformerBlock performs the following operations:

```
ẑˡ = W-MSA(LN(zˡ⁻¹)) + zˡ⁻¹                    (for regular blocks)
ẑˡ = SW-MSA(LN(zˡ⁻¹)) + zˡ⁻¹                   (for shifted blocks)
zˡ = MLP(LN(ẑˡ)) + ẑˡ
```

Where:
- zˡ represents the output features of block l
- LN denotes LayerNormalization
- W-MSA/SW-MSA are (shifted) windowed multi-head self-attention
- MLP is a two-layer feed-forward network with GELU activation

Window Partitioning and Merging
-------------------------------

The attention computation operates on windows of size M×M:
1. **Partition**: Input (H, W, C) → (⌈H/M⌉×⌈W/M⌉, M, M, C)
2. **Attention**: Applied within each M×M window independently
3. **Merge**: Reconstruct to original spatial dimensions (H, W, C)

For shifted windows, cyclic shifts are applied before partitioning and reversed after
attention computation, enabling cross-window communication.

Performance Characteristics
--------------------------

Computational Complexity:
- Traditional ViT: O(4hwC² + 2(hw)²C) per block
- Swin Transformer: O(4hwC² + 2M²hwC) per block
- Memory efficient for large images due to fixed window size M

Typical configurations:
- Swin-T: embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24]
- Swin-S: embed_dim=96, depths=[2,2,18,2], num_heads=[3,6,12,24]
- Swin-B: embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]
- Swin-L: embed_dim=192, depths=[2,2,18,2], num_heads=[6,12,24,48]

Usage Example
------------

```python
# Basic Swin Transformer block
block = SwinTransformerBlock(
    dim=96,
    num_heads=3,
    window_size=7,
    shift_size=0,  # Regular window attention
    mlp_ratio=4.0
)

# Shifted window variant (typical for odd-numbered layers)
shifted_block = SwinTransformerBlock(
    dim=96,
    num_heads=3,
    window_size=7,
    shift_size=3,  # Shifted window attention
    drop_path=0.1  # Stochastic depth
)

# Process 4D image tensor
inputs = keras.Input(shape=(224, 224, 96))
outputs = block(inputs)
```

References
----------

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and Guo, B., 2021.
  "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
  arXiv preprint arXiv:2103.14030.
  https://arxiv.org/abs/2103.14030

- ICCV 2021 Best Paper Award Winner
"""

import keras
from keras import ops, initializers, regularizers
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import window_reverse, window_partition

from ..ffn import SwinMLP
from ..stochastic_depth import StochasticDepth
from ..attention.window_attention import WindowAttention


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinTransformerBlock(keras.layers.Layer):
    """
    Swin Transformer Block with windowed multi-head self-attention.

    Implements the core Swin Transformer block: pre-normalization windowed
    multi-head self-attention with optional cyclic shift, followed by a
    pre-normalization SwinMLP, both wrapped with residual connections and
    optional stochastic depth regularization. Computational complexity is
    ``O(M^2 * H * W)`` where ``M`` is the window size, providing linear
    scaling with respect to input spatial resolution.

    ``x' = x + DropPath(W-MSA(LN(x)))``
    ``y  = x' + DropPath(MLP(LN(x')))``

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input (B, H, W, C)                  │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  LayerNorm1                          │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  [Cyclic Shift + SW-MSA keep mask]   │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Window Partition ─► Window Attention│
        │  (masked when shifted) ─► Merge      │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  [Reverse Cyclic Shift]              │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  StochasticDepth ─► + Residual       │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  LayerNorm2 ─► SwinMLP               │
        │  ─► StochasticDepth ─► + Residual    │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Output (B, H, W, C)                 │
        └──────────────────────────────────────┘

    :param dim: Number of input channels.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param window_size: Side length of attention windows. Default: 8.
    :type window_size: int
    :param shift_size: Cyclic shift amount for SW-MSA. Use
        ``window_size // 2`` for standard shifted windows, 0 for regular.
        Any nonzero value additionally activates the SW-MSA pairwise keep
        mask, built per call from the runtime spatial extent, which forbids
        attention between tokens the roll brought together from non-adjacent
        regions. Following the reference Swin implementation, the shift is
        treated as ``0`` (no roll, no mask, full attention over the single
        window) whenever a statically-known ``H``/``W`` *equals*
        ``window_size``; a statically-known dim strictly below
        ``window_size`` raises. Dynamic (``None``) dims are never checked and
        never downgraded.
    :type shift_size: int
    :param mlp_ratio: MLP hidden dim / embedding dim ratio. Default: 4.0.
    :type mlp_ratio: float
    :param qkv_bias: Whether QKV projections use bias. Default: True.
    :type qkv_bias: bool
    :param dropout_rate: Dropout rate for MLP and projections. Default: 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate for attention weights.
    :type attention_dropout_rate: float
    :param stochastic_depth_rate: Drop-path rate. Default: 0.0.
    :type stochastic_depth_rate: float
    :param activation: Activation function for MLP. Default: ``'gelu'``.
    :type activation: Union[str, Callable]
    :param use_bias: Whether normalization and projections use bias.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Bias weight initializer.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Union[str, regularizers.Regularizer]]
    :param bias_regularizer: Bias weight regularizer.
    :type bias_regularizer: Optional[Union[str, regularizers.Regularizer]]
    :param activity_regularizer: Activity regularizer.
    :type activity_regularizer: Optional[Union[str, regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dimension, head, or rate parameters are invalid.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            activity_regularizer=activity_regularizer,
            **kwargs
        )

        # Comprehensive input validation
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Got head_dim={dim // num_heads} with remainder {dim % num_heads}"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if shift_size < 0:
            raise ValueError(f"shift_size must be non-negative, got {shift_size}")
        if shift_size >= window_size:
            raise ValueError(
                f"shift_size ({shift_size}) must be less than window_size ({window_size})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not (0 <= attention_dropout_rate < 1):
            raise ValueError(f"attn_dropout_rate must be in [0, 1), got {attention_dropout_rate}")
        if not (0 <= stochastic_depth_rate < 1):
            raise ValueError(f"drop_path must be in [0, 1), got {stochastic_depth_rate}")

        # Store ALL configuration parameters for serialization
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

        # Store and serialize initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Following Pattern 2: Composite Layer from the guide

        # Layer normalization layers
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=self.use_bias,
            scale=True,  # Always use scale parameter
            beta_initializer=self.bias_initializer if self.use_bias else "zeros",
            gamma_initializer="ones",
            beta_regularizer=self.bias_regularizer if self.use_bias else None,
            gamma_regularizer=None,  # Typically don't regularize scale parameters
            name="norm1"
        )

        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=self.use_bias,
            scale=True,  # Always use scale parameter
            beta_initializer=self.bias_initializer if self.use_bias else "zeros",
            gamma_initializer="ones",
            beta_regularizer=self.bias_regularizer if self.use_bias else None,
            gamma_regularizer=None,  # Typically don't regularize scale parameters
            name="norm2"
        )

        # Window attention layer
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            dropout_rate=self.attention_dropout_rate,
            proj_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            partition_mode="grid",
            probability_type="softmax",
            attention_mode="linear",
            name="attn"
        )

        # Stochastic depth layer (optional)
        if self.stochastic_depth_rate > 0.0:
            self.drop_path_layer = StochasticDepth(
                drop_path_rate=self.stochastic_depth_rate,
                name="drop_path"
            )
        else:
            self.drop_path_layer = None

        # MLP layer
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = SwinMLP(
            hidden_dim=mlp_hidden_dim,
            output_dim=self.dim,  # Explicit output dimension
            use_bias=self.use_bias,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp"
        )

        logger.debug(
            f"Initialized SwinTransformerBlock: dim={dim}, num_heads={num_heads}, "
            f"window_size={window_size}, shift_size={shift_size}, "
            f"mlp_ratio={mlp_ratio}, drop_path={stochastic_depth_rate}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for serialization safety.

        :param input_shape: Shape tuple ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If shape is not 4-D or channels != dim.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"SwinTransformerBlock expects 4D input (batch, height, width, channels), "
                f"got shape {input_shape}"
            )

        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channels ({input_shape[-1]}) must match dim ({self.dim})"
            )

        # Build sub-layers in computational order following the forward pass

        # 1. Build normalization layers (operate on original input shape)
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # 2. Build attention layer with windowed input shape
        # After window partitioning: (batch * num_windows, window_size^2, channels)
        windowed_shape = (None, self.window_size * self.window_size, self.dim)
        self.attn.build(windowed_shape)

        # 3. Build stochastic depth layer if it exists (operates on original shape)
        if self.drop_path_layer is not None:
            self.drop_path_layer.build(input_shape)

        # 4. Build MLP layer (operates on original input shape)
        self.mlp.build(input_shape)

        logger.debug(
            f"Built SwinTransformerBlock: input_shape={input_shape}, "
            f"windowed_shape={windowed_shape}"
        )

        # Always call parent build at the end
        super().build(input_shape)

    def _build_swmsa_keep_mask(
        self,
        batch_size: Union[int, keras.KerasTensor],
        height: Union[int, keras.KerasTensor],
        width: Union[int, keras.KerasTensor],
        shift: Union[int, keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Build the SW-MSA pairwise keep mask for the current feature map.

        After the cyclic roll by ``-shift_size``, one physical window can hold
        tokens that came from up to four spatially distant regions of the
        pre-roll feature map. Those tokens must not attend to one another. The
        standard Swin construction labels every pre-roll position with one of
        nine region ids (a 3x3 product of row/column bands) and permits a
        ``(query, key)`` pair inside a window iff both carry the same label.

        The region image is partitioned with the very same
        :func:`~dl_techniques.utils.tensors.window_partition` that partitions
        the data in :meth:`call`, so the mask's window order matches the data's
        window order **by construction** rather than by a re-derived index
        formula.

        :param batch_size: Batch size ``B`` (may be a dynamic scalar).
        :type batch_size: Union[int, keras.KerasTensor]
        :param height: Feature-map height ``H`` (may be a dynamic scalar).
        :type height: Union[int, keras.KerasTensor]
        :param width: Feature-map width ``W`` (may be a dynamic scalar).
        :type width: Union[int, keras.KerasTensor]
        :param shift: The effective shift for this call, as returned by
            :meth:`_resolve_shift_size`. May be a **runtime scalar** that is
            ``0`` on a single-window feature map, in which case the returned
            predicate is all-ones (see the D-012 note below).
        :type shift: Union[int, keras.KerasTensor]
        :return: Int32 keep predicate (``1 = attend``) of shape
            ``(B * num_windows, window_size**2, window_size**2)``, laid out
            B-major / window-minor to match ``window_partition``.
        :rtype: keras.KerasTensor
        """
        ws = self.window_size

        # DECISION plan-2026-07-31T042809-ddc92265/D-012
        # `shift` is a PARAMETER, never `self.shift_size`, because the effective
        # shift can be decided at RUNTIME (see `_resolve_shift_size`). Reading
        # `self.shift_size` here would rebuild the region bands for a shift the
        # roll did not apply -- a mask/roll desynchronization, which is silent.
        #
        # A runtime `shift == 0` yields an ALL-ONES predicate, structurally:
        # `rows >= height - 0` is false for every row in `0..H-1`, so the second
        # band term vanishes and `row_id = cast(rows >= H - ws)`. Because
        # `window_partition` requires `H % ws == 0`, the `H - ws` boundary always
        # lands on a window edge, so every window is uniform in `row_id` (and
        # likewise in `col_id`) and `equal(m_i, m_j)` is 1 everywhere. An
        # all-ones keep predicate is bit-identical to `attention_mask=None`
        # (`apply_attention_mask` builds its bias with `ops.where`, so an
        # all-kept row receives no bias at all) -- measured, not assumed.
        # See decisions.md D-012 (plan-2026-07-31T042809-ddc92265).

        # DECISION plan-2026-07-31T042809-ddc92265/D-002
        # The mask is built HERE, per call, from the DYNAMIC `(B, H, W)` of the
        # incoming tensor — not in `build()` from static dims.
        #
        # WHY: `models/thera/tails.py` builds every one of its Swin blocks with
        # `(B, None, None, embed_dim)` on purpose, and its `call()` reflect-pads
        # H and W up to a window-size multiple using SYMBOLIC `ops.mod` amounts.
        # The spatial extent is therefore unknown at build time AND varies per
        # call by design (THERA's hypernetwork feeds arbitrary crop sizes).
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT move this to `build()` and read `input_shape[1:3]`. They are
        #     `None` for the only production consumer with a shifted block, so
        #     the mask would be built from `None` — a `TypeError` at best and a
        #     wrong-geometry mask at worst.
        #   * Do NOT copy `progressive_focused_attention.py`'s build-time
        #     `ValueError` on dynamic `H`/`W`. That layer has no dynamic-shape
        #     consumer; this one does, and the raise would make `thera/pro` and
        #     `thera/plus` dead on forward — trading a silent correctness bug for
        #     a loud outage. `_resolve_shift_size` deliberately skips a `None`
        #     dim -- it neither raises nor downgrades -- for exactly this reason.
        #   * Do NOT cache the mask on `self` keyed by nothing. The geometry
        #     changes per call; a stale cache is a silent wrong-geometry mask.
        #   * Do NOT re-derive the window order with an index formula instead of
        #     calling `window_partition` below. A mismatch between the mask's
        #     window order and the data's is SILENT: no shape error, finite
        #     output, wrong attention.
        # ACCEPTED COST: a handful of cheap elementwise ops per forward pass,
        # negligible next to the attention matmul.
        # See decisions.md D-002 (plan-2026-07-31T042809-ddc92265).
        rows = ops.arange(height, dtype="int32")
        cols = ops.arange(width, dtype="int32")

        # Region bands, equivalent to the canonical slice-counter construction
        # (0, -ws), (-ws, -shift), (-shift, None) applied to rows and columns.
        row_id = (
            ops.cast(rows >= height - ws, "int32")
            + ops.cast(rows >= height - shift, "int32")
        )
        col_id = (
            ops.cast(cols >= width - ws, "int32")
            + ops.cast(cols >= width - shift, "int32")
        )
        # Shape: (H, 1) * 3 + (1, W) -> (H, W), values in 0..8
        region = (
            ops.expand_dims(row_id, axis=-1) * 3
            + ops.expand_dims(col_id, axis=0)
        )

        # Shape: (H, W) -> (1, H, W, 1) -> (num_windows, ws, ws, 1)
        region_windows = window_partition(
            ops.reshape(ops.cast(region, "float32"), (1, height, width, 1)),
            ws,
        )
        # Shape: (num_windows, ws, ws, 1) -> (num_windows, ws*ws)
        region_windows = ops.reshape(region_windows, (-1, ws * ws))

        # Shape: (nw, ws*ws, 1) == (nw, 1, ws*ws) -> (nw, ws*ws, ws*ws)
        keep = ops.cast(
            ops.equal(
                ops.expand_dims(region_windows, axis=-1),
                ops.expand_dims(region_windows, axis=-2),
            ),
            "int32",
        )

        # Tile over the batch in the SAME order `window_partition` folds it:
        # B-major / window-minor (assumption A3, `utils/tensors.py:431-435`).
        # Shape: (B, 1, 1, 1) * (1, nw, ws*ws, ws*ws) -> (B, nw, ws*ws, ws*ws)
        keep = ops.ones((batch_size, 1, 1, 1), dtype="int32") * ops.expand_dims(
            keep, axis=0
        )
        return ops.reshape(keep, (-1, ws * ws, ws * ws))

    def _resolve_shift_size(
        self, x: keras.KerasTensor
    ) -> Union[int, keras.KerasTensor]:
        """Resolve the shift actually applied to this input's geometry.

        Returns ``self.shift_size`` in the normal case and ``0`` when the
        feature map is exactly one window across — the reference Swin rule
        ``if min(input_resolution) <= window_size: shift_size = 0``. When the
        spatial dims are statically known the answer is a Python ``int``; when
        either is dynamic the answer is a **runtime int32 scalar** carrying the
        same rule, so a dynamically-shaped block and a statically-shaped block
        agree on the same tensor (D-012).

        :param x: The block's ``(B, H, W, C)`` input.
        :type x: keras.KerasTensor
        :return: The effective shift for this call: ``0``, ``self.shift_size``,
            or a runtime scalar equal to one of the two.
        :rtype: Union[int, keras.KerasTensor]
        :raises ValueError: If ``shift_size > 0`` and a statically-known
            ``H``/``W`` is strictly smaller than ``window_size``.
        """
        if self.shift_size == 0:
            return 0

        ws = self.window_size

        # DECISION plan-2026-07-31T042809-ddc92265/D-006
        # At a statically-known `H == window_size` (or `W == window_size`) the
        # feature map is a SINGLE window, and this block follows the reference
        # Swin rule: `if min(input_resolution) <= window_size: shift_size = 0`.
        # The shift is dropped entirely -- no roll, no mask, plain full W-MSA.
        #
        # The single return value is deliberate: `call()` derives BOTH the roll
        # and the mask from it, so they cannot desynchronize. A roll without a
        # mask is exactly the F-01 bug this plan fixes; a mask without a roll is
        # equally wrong.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT restore a `H < 2 * window_size` guard. Its premise -- that
        #     the 3x3 region bands "degenerate" below that size -- was FALSIFIED
        #     by measurement: `_build_swmsa_keep_mask` is array-equal to two
        #     independent oracles at `(8,8,8,4)` and `(7,7,7,3)`. The guard
        #     rejects provably correct geometry, and it crashed 9 passing tests
        #     across `models/scunet` (8x8 bottleneck, ws=8) and
        #     `models/swin_transformer` (7x7 stage 4, ws=7 -- canonical Swin-T
        #     at 224x224). Because `window_partition` requires `H % ws == 0`, a
        #     static `H < 2*ws` can only ever mean `H == ws`.
        #   * Do NOT region-mask the single full window instead. It is a legal
        #     mask, but it is not what upstream Swin does, and this block is
        #     meant to be checkpoint-compatible with that lineage.
        #   * Do NOT raise on a dynamic (`None`) dim. `models/thera/tails.py`
        #     builds every Swin block with `(B, None, None, C)` on purpose; a
        #     raise there trades a silent correctness bug for a dead model
        #     (D-002, assumption A2).
        #   * Do NOT push this into `models/scunet` / `models/swin_transformer`
        #     by passing `shift_size=0` there. That widens the blast radius and
        #     leaves every future caller exposed to the same trap.
        # ACCEPTED COST: not numerically neutral. `use_relative_position_bias`
        # defaults to True, so dropping the roll changes which token sits at
        # which within-window position and therefore changes the RPB pairing.
        # `scunet` and `swin_transformer` numerics move at their `H == ws`
        # stages. Accepted as an extension of invariant I5 (F-01 is
        # deliberately not output-neutral).
        # See decisions.md D-006 (plan-2026-07-31T042809-ddc92265).
        single_window = False
        has_dynamic_dim = False
        for axis_name, axis_index in (("height", 1), ("width", 2)):
            static_dim = x.shape[axis_index]
            if static_dim is None:
                has_dynamic_dim = True
                continue
            if static_dim < ws:
                raise ValueError(
                    f"SwinTransformerBlock(shift_size={self.shift_size}, "
                    f"window_size={ws}) received {axis_name}={static_dim}, "
                    f"which is smaller than window_size={ws}. A feature map "
                    f"smaller than one window cannot be partitioned; enlarge "
                    f"the feature map or reduce window_size."
                )
            if static_dim == ws:
                single_window = True

        if single_window:
            return 0
        if not has_dynamic_dim:
            return self.shift_size

        # DECISION plan-2026-07-31T042809-ddc92265/D-012
        # A dynamic (`None`) spatial dim gets the SAME single-window rule, but
        # evaluated at RUNTIME rather than at trace time.
        #
        # WHY: without this the block is SHAPE-DEPENDENT. The identical layer,
        # with identical weights, on the identical tensor, returned two
        # different answers depending only on whether `H` was statically known:
        # traced against `TensorSpec([None, None, None, C])` at `H == W == ws`
        # the loop above could not see the geometry, the static fallback could
        # not fire, and the block kept the roll + region mask while an eager
        # call on the same tensor dropped both. Measured at
        # `(dim=32, heads=4, ws=4, shift=2)`, `(1,4,4,32)`, seeded weights:
        # 512/512 elements differed, max |diff| 251.4 (97% relative).
        # `models/thera/tails.py` is the one consumer that builds every block
        # with `(B, None, None, C)`, so it took the opposite branch from
        # `models/swin_transformer` at identical geometry.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT "fix" this by making the dynamic path region-mask a single
        #     window instead. That is a legal mask but it is NOT the reference
        #     rule, so it would leave the two paths disagreeing anyway (the RPB
        #     pairing differs the moment the roll is applied).
        #   * Do NOT branch in Python on `ops.shape(x)[1] > ws`. Inside a trace
        #     that is a symbolic bool; `if` on it either raises or silently
        #     picks whichever branch the tracer folded, which is precisely the
        #     shape-dependence being removed here.
        #   * Do NOT hoist this to `build()`. The dim is `None` there by
        #     construction for the only dynamic consumer (D-002).
        #   * Do NOT return a float or a bool. The value is multiplied into
        #     `ops.arange` band comparisons and fed to `ops.roll`; int32 is what
        #     both accept.
        # ACCEPTED COST: `ops.roll` now receives a tensor shift on the dynamic
        # path (measured to work eagerly, under `tf.function` and under
        # `jit_compile=True`), and the mask is built unconditionally on that
        # path — an all-ones predicate when the runtime shift is 0, which is
        # bit-identical to `attention_mask=None` (measured).
        # See decisions.md D-012 (plan-2026-07-31T042809-ddc92265).
        runtime_shape = ops.shape(x)
        multi_window = ops.logical_and(
            ops.greater(runtime_shape[1], ws),
            ops.greater(runtime_shape[2], ws),
        )
        return ops.cast(multi_window, "int32") * self.shift_size

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the SwinTransformerBlock.

        When the effective shift is nonzero the block builds the SW-MSA
        pairwise keep mask for the current (possibly dynamic) spatial extent
        and passes it to the window attention, so tokens the cyclic roll
        brought together from non-adjacent regions cannot attend to one
        another. The effective shift comes from :meth:`_resolve_shift_size`,
        which drops it to ``0`` on a single-window feature map (D-006) — the
        roll and the mask are therefore always in agreement. That rule is
        applied at trace time when ``H``/``W`` are static and at runtime when
        they are not, so the block's output does **not** depend on whether its
        spatial extent happened to be statically known (D-012).

        :param x: Input tensor ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the input tensor is not 4-D, or if
            ``shift_size > 0`` and a statically-known ``H``/``W`` is strictly
            smaller than ``window_size``.
        """
        input_shape = ops.shape(x)
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got shape {input_shape}"
            )

        # Single source of truth for this call's shift; both the roll and the
        # mask below are derived from it (D-006). On a statically-shaped input
        # it is a Python int and the shift branch is decided at trace time; on
        # a dynamically-shaped one it is a runtime int32 scalar that is 0 on a
        # single-window feature map, so the branch is taken unconditionally and
        # the shift itself carries the decision (D-012).
        shift = self._resolve_shift_size(x)
        shift_is_static = isinstance(shift, int)
        apply_shift = shift > 0 if shift_is_static else True

        B, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        shortcut = x

        # =============================================
        # Multi-head Self-Attention Block
        # =============================================

        # Layer Norm 1 (pre-attention normalization)
        x = self.norm1(x, training=training)

        # Apply cyclic shift for shifted window attention
        if apply_shift:
            # Shift windows by (-shift, -shift)
            shifted_x = ops.roll(
                x,
                shift=(-shift, -shift),
                axis=(1, 2)
            )
            # SW-MSA pairwise keep mask, (B*num_windows, ws**2, ws**2).
            attention_mask = self._build_swmsa_keep_mask(B, H, W, shift)
        else:
            shifted_x = x
            attention_mask = None

        # Partition into windows: (B, H, W, C) -> (B*num_windows, window_size, window_size, C)
        x_windows = window_partition(shifted_x, self.window_size)

        # Reshape for attention: (B*num_windows, window_size*window_size, C)
        x_windows = ops.reshape(
            x_windows,
            (-1, self.window_size * self.window_size, C)
        )

        # Apply window-based multi-head self-attention. `attention_mask` is the
        # rank-3 pairwise keep predicate (D-001 in the attention primitives);
        # it is None for regular (unshifted) W-MSA, which keeps that path
        # bit-identical to pre-mask behaviour (invariant I1).
        attn_windows = self.attn(
            x_windows, attention_mask=attention_mask, training=training
        )

        # Reshape back to window format: (B*num_windows, window_size, window_size, C)
        attn_windows = ops.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, C)
        )

        # Merge windows back: (B*num_windows, window_size, window_size, C) -> (B, H, W, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift if it was applied
        if apply_shift:
            x = ops.roll(
                x,
                shift=(shift, shift),
                axis=(1, 2)
            )

        # Apply stochastic depth and residual connection
        if self.drop_path_layer is not None:
            x = shortcut + self.drop_path_layer(x, training=training)
        else:
            x = shortcut + x

        # =============================================
        # MLP Block
        # =============================================

        shortcut = x

        # Layer Norm 2 (pre-MLP normalization)
        x = self.norm2(x, training=training)

        # Apply MLP transformation
        x = self.mlp(x, training=training)

        # Apply stochastic depth and residual connection
        if self.drop_path_layer is not None:
            x = shortcut + self.drop_path_layer(x, training=training)
        else:
            x = shortcut + x

        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape (same as input).

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
