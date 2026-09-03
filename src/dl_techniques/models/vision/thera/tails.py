"""THERA feature-refiner "tails" as Keras layers, plus a size-keyed builder.

`build_thera_tail` returns one of three optional feature refiners for THERA's
sampled field output, selected by a model-size key: `TheraTailAir` (identity),
`TheraTailPlus` (a depthwise ConvNeXt stack with a 1x1 projection inserted at
each channel-width change), or `TheraTailPro` (a SwinIR body: `conv_first`,
a long-residual stack of RSTBs, `conv_after_body`, `conv_before_upsample`).
Swin and ConvNeXt internals are reused from their existing layers rather than
reimplemented here.

The `pro` tail reflect-pads `H, W` up to the next multiple of `window_size`
before the Swin stack and crops back afterwards, so any spatial size is
accepted and the output spatial shape always matches the input's.

References:
    - Becker et al. Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
      Neural Heat Fields.
    - Liu et al., 2021. Swin Transformer. ICCV 2021.
    - Liu et al., 2022. A ConvNet for the 2020s. CVPR 2022.
    - Liang et al. SwinIR.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.transformers.swin_transformer_block import (
    SwinTransformerBlock,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# THERA "plus" ConvNeXt block defs: (n_dims, kernel). group_features=True
# (depthwise) is intrinsic to ConvNextV1Block, so it is not stored per-block.
THERA_PLUS_BLOCKS: List[Tuple[int, int]] = (
    [(64, 3)] * 6 + [(96, 3)] * 7 + [(128, 3)] * 3
)

# THERA "pro" SwinIR config.
THERA_PRO_EMBED_DIM: int = 180
THERA_PRO_DEPTHS: Tuple[int, ...] = (7, 6)
THERA_PRO_NUM_HEADS: Tuple[int, ...] = (6, 6)
THERA_PRO_WINDOW_SIZE: int = 8
THERA_PRO_MLP_RATIO: float = 2.0
THERA_PRO_NUM_FEAT: int = 64

LEAKY_RELU_SLOPE: float = 0.2

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.thera.tails")
class TheraTailAir(keras.layers.Layer):
    """The `air` tail: an identity feature refiner, no-op passthrough.

    A registered layer rather than a bare lambda, so `build_thera_tail`
    returns the same serializable interface regardless of size.

    Architecture:

    .. code-block:: text

        x [B, H, W, C]
        │
        ▼ (identity)
        x [B, H, W, C]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.thera.tails")
class _Projection(keras.layers.Layer):
    """Channel-count adapter: `LayerNorm` then a 1x1 `Conv2D`.

    Inserted by the `plus` tail before a ConvNeXt block whenever the channel
    count changes, since the reused depthwise `ConvNextV1Block` runs at a
    fixed channel width and cannot itself change the channel count.

    Architecture:

    .. code-block:: text

        x [B, H, W, C_in]
        │
        ┌─────▼─────┐
        │ layernorm  │
        └─────┬─────┘
              ▼
        ┌─────────────┐
        │ conv 1x1      │  n_dims channels
        └─────┬───────┘
              ▼
        x [B, H, W, n_dims]
    """

    def __init__(self, n_dims: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if n_dims <= 0:
            raise ValueError(f"n_dims must be positive, got {n_dims}")
        self.n_dims = n_dims

        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="norm"
        )
        self.proj = keras.layers.Conv2D(
            filters=self.n_dims,
            kernel_size=1,
            padding="same",
            name="proj",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.norm.build(input_shape)
        self.proj.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x = self.norm(x, training=training)
        x = self.proj(x, training=training)
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape[:-1]) + (self.n_dims,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"n_dims": self.n_dims})
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.thera.tails")
class TheraTailPlus(keras.layers.Layer):
    """The `plus` tail: a depthwise-ConvNeXt feature refiner.

    For each `(dims, k)` in `block_defs`, inserts a `_Projection(dims)`
    whenever `dims` differs from the running channel count, then a
    `ConvNextV1Block(kernel_size=k, filters=dims)`. Sub-layers are created in
    `__init__` from the static `block_defs`, not in `build`, so the `.keras`
    weight structure is fixed at construction time.

    Architecture:

    .. code-block:: text

        x [B, H, W, C_in]
        │
        ▼
        for (dims, k) in block_defs:
            ┌──────────────┐
            │ _Projection    │  (optional) only if dims changed
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ ConvNextV1Block│  depthwise + 4x MLP + residual
            └──────┬───────┘
        │
        ▼
        x [B, H, W, block_defs[-1][0]]

    :param block_defs: Sequence of ``(n_dims, kernel_size)`` tuples.
    :type block_defs: Optional[Sequence[Tuple[int, int]]]
    :param in_channels: Input channel count, used only to decide whether the
        first block needs a leading `_Projection`. If ``None``, assumes the
        input matches ``block_defs[0][0]``. When constructing this tail for a
        backbone whose output channels differ from ``block_defs[0][0]``, pass
        ``in_channels`` explicitly or :meth:`build` raises ``ValueError``.
        :func:`build_thera_tail` does not need this if the caller passes it through.
    :type in_channels: Optional[int]
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        block_defs: Optional[Sequence[Tuple[int, int]]] = None,
        in_channels: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if block_defs is None:
            block_defs = THERA_PLUS_BLOCKS
        # Normalize to a list of (int, int) tuples (JSON round-trips tuples to lists).
        self.block_defs: List[Tuple[int, int]] = [
            (int(d), int(k)) for (d, k) in block_defs
        ]
        if not self.block_defs:
            raise ValueError("block_defs must be non-empty")

        self.in_channels: Optional[int] = (
            None if in_channels is None else int(in_channels)
        )

        # Sub-layers are created here, not in build(), so the creation order and
        # per-sub-layer name (proj_{i} / convnext_{i}) stay stable for .keras reload.
        self._sublayers: List[keras.layers.Layer] = []
        self._sublayer_out_dims: List[int] = []

        current = (
            self.in_channels
            if self.in_channels is not None
            else self.block_defs[0][0]
        )
        for i, (dims, k) in enumerate(self.block_defs):
            if current != dims:
                proj = _Projection(n_dims=dims, name=f"proj_{i}")
                self._sublayers.append(proj)
                self._sublayer_out_dims.append(dims)
                current = dims
            block = ConvNextV1Block(
                kernel_size=k, filters=dims, name=f"convnext_{i}"
            )
            self._sublayers.append(block)
            self._sublayer_out_dims.append(dims)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Propagate shapes to the sub-layers created in `__init__`.

        Fails loudly if the actual input width disagrees with the
        `in_channels`/`block_defs[0][0]` assumption baked into the pre-created
        sub-layer stack, rather than silently feeding wrong channels forward.

        :param input_shape: Input shape, channels-last.
        :raises ValueError: If the actual input channel count does not match
            the constructed `in_channels` (or `block_defs[0][0]`).
        """
        in_ch = input_shape[-1]
        expected_in = (
            self.in_channels if self.in_channels is not None
            else self.block_defs[0][0]
        )
        if in_ch is not None and in_ch != expected_in:
            raise ValueError(
                f"TheraTailPlus was constructed for input channels {expected_in} "
                f"(in_channels or block_defs[0][0]) but received input with "
                f"{in_ch} channels. Pass in_channels={in_ch} at construction so "
                f"the leading projection is created."
            )

        shape = tuple(input_shape)
        for layer, out_dims in zip(self._sublayers, self._sublayer_out_dims):
            layer.build(shape)
            shape = shape[:-1] + (out_dims,)

        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        for layer in self._sublayers:
            x = layer(x, training=training)
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape[:-1]) + (self.block_defs[-1][0],)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # Serialize as a list of lists (tuples are not JSON-native).
        config.update(
            {
                "block_defs": [[d, k] for (d, k) in self.block_defs],
                "in_channels": self.in_channels,
            }
        )
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.thera.tails")
class TheraTailPro(keras.layers.Layer):
    """The `pro` tail: a SwinIR (RSTB) feature refiner.

    A `conv_first` lifts the input to `embed_dim`, a long-residual stack of
    RSTBs (each a run of Swin blocks with alternating shift, plus a conv
    bypass and residual) refines it, then `conv_after_body` and
    `conv_before_upsample` produce the `num_feat`-channel output. `H, W` are
    reflect-padded up to a multiple of `window_size` before the Swin stack and
    cropped back afterwards, so any spatial size is accepted and the output
    shape matches the input's.

    Architecture:

    .. code-block:: text

        x [B, H, W, C]
        │
        ┌─────▼─────┐
        │ conv 3x3    │  = res
        └─────┬─────┘
              ▼
        reflect-pad H, W to a window_size multiple
              ├─────────────────────────────┐ (long residual)
              ▼                              │
        ┌─────────────┐                      │
        │ RSTB x len(depths)│                  │  each: Swin blocks, conv, residual
        └─────┬───────┘                      │
              ▼                              │
        ┌─────────────┐                      │
        │ conv 3x3      │                      │
        └─────┬───────┘                      │
              ▼                              │
             add  ◄──────────────────────────┘
              ▼
        crop back to original H, W
              ▼
        ┌─────────────┐
        │ conv 3x3      │  -> num_feat
        └─────┬───────┘
              ▼
        leaky_relu
              ▼
        x [B, H, W, num_feat]

    :param embed_dim: Swin working width.
    :type embed_dim: int
    :param depths: Swin blocks per RSTB.
    :type depths: Sequence[int]
    :param num_heads: Attention heads per RSTB.
    :type num_heads: Sequence[int]
    :param window_size: Attention window side length.
    :type window_size: int
    :param mlp_ratio: Swin MLP expansion ratio.
    :type mlp_ratio: float
    :param num_feat: Output channel count after `conv_before_upsample`.
    :type num_feat: int
    """

    def __init__(
        self,
        embed_dim: int = THERA_PRO_EMBED_DIM,
        depths: Sequence[int] = THERA_PRO_DEPTHS,
        num_heads: Sequence[int] = THERA_PRO_NUM_HEADS,
        window_size: int = THERA_PRO_WINDOW_SIZE,
        mlp_ratio: float = THERA_PRO_MLP_RATIO,
        num_feat: int = THERA_PRO_NUM_FEAT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = int(embed_dim)
        self.depths = tuple(int(d) for d in depths)
        self.num_heads = tuple(int(h) for h in num_heads)
        self.window_size = int(window_size)
        self.mlp_ratio = float(mlp_ratio)
        self.num_feat = int(num_feat)

        if len(self.depths) != len(self.num_heads):
            raise ValueError(
                f"depths ({self.depths}) and num_heads ({self.num_heads}) "
                f"must have equal length"
            )
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_feat <= 0:
            raise ValueError(f"num_feat must be positive, got {self.num_feat}")
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")

        # conv_first: lift input channels to embed_dim.
        self.conv_first = keras.layers.Conv2D(
            filters=self.embed_dim, kernel_size=3, padding="same", name="conv_first"
        )

        # Swin blocks are stored in a flat list, not a nested list-of-lists: a
        # nested List[List[Layer]] fails to round-trip weights through .keras.
        # Stage boundaries are recovered from self.depths.
        self.swin_blocks: List[SwinTransformerBlock] = []
        self.rstb_convs: List[keras.layers.Conv2D] = []
        for stage, (depth, heads) in enumerate(zip(self.depths, self.num_heads)):
            for i in range(depth):
                shift = 0 if (i % 2 == 0) else self.window_size // 2
                self.swin_blocks.append(
                    SwinTransformerBlock(
                        dim=self.embed_dim,
                        num_heads=heads,
                        window_size=self.window_size,
                        shift_size=shift,
                        mlp_ratio=self.mlp_ratio,
                        name=f"rstb{stage}_swin{i}",
                    )
                )
            self.rstb_convs.append(
                keras.layers.Conv2D(
                    filters=self.embed_dim,
                    kernel_size=3,
                    padding="same",
                    name=f"rstb{stage}_conv",
                )
            )

        self.conv_after_body = keras.layers.Conv2D(
            filters=self.embed_dim, kernel_size=3, padding="same", name="conv_after_body"
        )
        self.conv_before_upsample = keras.layers.Conv2D(
            filters=self.num_feat, kernel_size=3, padding="same", name="conv_before_upsample"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sublayer explicitly, for reload-safe weights.

        Spatial dims may be padded at call time, so sublayers build with
        dynamic (`None`) spatial dims; only channels matter for their weights.
        """
        embed_shape = (input_shape[0], None, None, self.embed_dim)

        self.conv_first.build(input_shape)

        for blk in self.swin_blocks:
            blk.build(embed_shape)
        for stage_conv in self.rstb_convs:
            stage_conv.build(embed_shape)

        self.conv_after_body.build(embed_shape)
        self.conv_before_upsample.build(embed_shape)

        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the Swin stack over a window-size-aligned reflect pad.

        :param x: Input tensor of shape `(B, H, W, C)`.
        :param training: Whether to run the sub-layers in training mode.
        :return: Output tensor of shape `(B, H, W, num_feat)`.
        :rtype: keras.KerasTensor
        :raises ValueError: If a statically known spatial extent is too small
            for its reflect pad (the smallest accepted extent is
            `window_size // 2 + 1`). Skipped for a dynamic (`None`) extent.
        """
        # Original spatial dims for the post-stack crop.
        shape = ops.shape(x)
        h, w = shape[1], shape[2]

        x = self.conv_first(x, training=training)

        # DECISION plan_2026-06-11_f662207d/D-007: reflect-pad H,W to a window
        # multiple, run the Swin stack, crop back to the original H,W -- do not
        # require window-aligned inputs, since callers feed arbitrary crop sizes. See decisions.md.
        ws = self.window_size

        # DECISION plan-2026-07-31T210633-b63a35aa/D-004: TF's MirrorPad requires
        # every pad amount strictly less than the dimension it pads, so a small
        # enough H or W raises; this static-only check catches it before the op does. See decisions.md.
        for axis_name, extent in (("height", x.shape[1]), ("width", x.shape[2])):
            if extent is None:
                continue
            pad_amount = (-extent) % ws
            if pad_amount >= extent:
                raise ValueError(
                    f"TheraTailPro reflect-pads {axis_name} up to the next "
                    f"multiple of window_size={ws}, but a reflect pad must be "
                    f"strictly smaller than the extent it pads: "
                    f"{axis_name}={extent} needs a pad of {pad_amount}, which "
                    f"is not less than {extent}. The smallest {axis_name} this "
                    f"tail accepts at window_size={ws} is {ws // 2 + 1}. Pass a "
                    f"larger input, or a smaller window_size."
                )

        # Pad up to the next window-size multiple (0 if already a multiple).
        # Use keras.ops.mod (NOT Python % on the symbolic ops.shape scalars h,w).
        pad_h = ops.mod(ws - ops.mod(h, ws), ws)
        pad_w = ops.mod(ws - ops.mod(w, ws), ws)
        x = ops.pad(
            x,
            [(0, 0), (0, pad_h), (0, pad_w), (0, 0)],
            mode="reflect",
        )

        res = x
        offset = 0
        for stage, depth in enumerate(self.depths):
            res2 = x
            for blk in self.swin_blocks[offset:offset + depth]:
                x = blk(x, training=training)
            offset += depth
            x = self.rstb_convs[stage](x, training=training)
            x = x + res2

        x = self.conv_after_body(x, training=training) + res

        # Crop back to the original (pre-pad) H, W (E1).
        x = x[:, :h, :w, :]

        x = self.conv_before_upsample(x, training=training)
        x = keras.activations.leaky_relu(x, negative_slope=LEAKY_RELU_SLOPE)
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape[:-1]) + (self.num_feat,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depths": list(self.depths),
                "num_heads": list(self.num_heads),
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "num_feat": self.num_feat,
            }
        )
        return config


# ---------------------------------------------------------------------


def build_thera_tail(
    size: str, in_channels: Optional[int] = None
) -> keras.layers.Layer:
    """Build the THERA feature-refiner tail for a model-size key.

    :param size: One of `'air'` (identity), `'plus'` (ConvNeXt), or `'pro'`
        (SwinIR/RSTB).
    :type size: str
    :param in_channels: Input channel count from the backbone; forwarded to
        the `plus` tail so it creates the leading 1x1 projection when the
        backbone does not emit 64 channels. Ignored by `air`/`pro`.
    :type in_channels: Optional[int]
    :return: The corresponding tail layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If `size` is not one of the three known keys.

    .. note::
        For the `plus` tail with a backbone that does not emit 64 channels,
        pass `in_channels` explicitly or :meth:`TheraTailPlus.build` raises
        `ValueError`.
    """
    if size == "air":
        return TheraTailAir(name="thera_tail_air")
    elif size == "plus":
        return TheraTailPlus(in_channels=in_channels, name="thera_tail_plus")
    elif size == "pro":
        return TheraTailPro(name="thera_tail_pro")
    raise ValueError(
        f"Unknown THERA tail size '{size}'; expected one of 'air', 'plus', 'pro'"
    )

# ---------------------------------------------------------------------
