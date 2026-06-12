"""THERA feature-refiner "tails" as Keras layers + a size-keyed builder.

THERA (Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields)
refines the sampled field output with one of three optional *tails* selected by
a model-size key. The reference JAX/Flax dispatcher (THERA ``model/tail.py``)::

    def build_tail(size):
        if size == 'air':
            return lambda x, _: x                                  # identity
        elif size == 'plus':
            blocks = [(64, 3, True)] * 6 + [(96, 3, True)] * 7 + [(128, 3, True)] * 3
            return ConvNeXt(blocks)                                # n_dims/k/depthwise
        elif size == 'pro':
            return SwinIR(depths=[7, 6], num_heads=[6, 6])         # embed_dim=180, ...

- ``air``  -> identity passthrough (no refinement).
- ``plus`` -> a ``ConvNeXt`` stack. The reference inserts a ``Projection``
  (``LayerNorm`` -> ``Conv1x1(n_dims)``) whenever a block changes the channel
  count, then a depthwise ConvNeXt block. Output channels = last block dim (128).
- ``pro``  -> a ``SwinIR`` body: ``conv_first(3x3)`` -> a long-residual stack of
  RSTBs (each = ``depth`` Swin blocks + a ``3x3`` conv bypass + residual) ->
  ``conv_after_body(3x3)`` (+ long residual) -> ``conv_before_upsample(3x3)`` +
  leaky-relu, returning ``num_feat`` channels. Spatial size is preserved.

REUSE (no Swin/ConvNeXt internals are reimplemented here):
- ``ConvNextV1Block`` (depthwise conv + LN + 4x inverted-bottleneck MLP)
  is THERA's ``group_features=True`` ConvNeXt block. Its depthwise conv operates
  at a fixed channel width (output channels == ``filters``), so the ConvNeXt tail
  stack runs at one working width per block group; a ``_Projection`` 1x1 conv
  adapts the channel count at each group transition.
- ``SwinTransformerBlock`` consumes/returns NHWC ``(B, H, W, C)``, infers ``H, W``
  dynamically, and builds its own (shifted) window attention mask. No patch
  embed/unembed is needed.

Edge case E1 (non-window-divisible / non-square input into ``pro``): the Swin
window attention requires ``H`` and ``W`` to be multiples of ``window_size``. The
``pro`` tail reflect-pads ``H, W`` up to the next multiple before the Swin stack
and crops back to the exact original ``H, W`` afterwards, so any spatial size is
handled and the output spatial shape always equals the input's.

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields"; Liu et al., "Swin Transformer" (ICCV 2021); Liu et al.,
    "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022); Liang et al., "SwinIR".
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.transformers.swin_transformer_block import (
    SwinTransformerBlock,
)

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


@keras.saving.register_keras_serializable()
class TheraTailAir(keras.layers.Layer):
    """The ``air`` tail: an identity feature refiner (passthrough).

    **Intent**: Give THERA's ``air`` size (no refinement, ``lambda x, _: x``) a
    uniform serializable Keras interface matching the ``plus``/``pro`` tails, so
    the builder returns the same layer type regardless of size.

    **Architecture**::

        Input (B, H, W, C) --> [identity] --> Output (B, H, W, C)

    THERA's ``air`` size applies no refinement (``lambda x, _: x``). This is a
    tiny registered layer rather than a bare lambda so the builder returns a
    uniform serializable interface across the three sizes.

    The tail accepts a second ignored positional argument (THERA passes the
    query scale / a placeholder to every tail; the hypernetwork calls
    ``tail(x, training)``).
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


@keras.saving.register_keras_serializable()
class _Projection(keras.layers.Layer):
    """THERA ``Projection``: ``LayerNorm`` -> ``Conv2D(n_dims, 1x1)``.

    **Intent**: Provide the channel-count adapter the ``plus`` tail inserts
    before a ConvNeXt block on a channel change. The ConvNeXt tail block stack
    operates at a fixed channel width: the reused depthwise ``ConvNextV1Block``
    keeps the channel count fixed at ``filters`` throughout the block, so a 1x1
    projection is needed to adapt the running channel count (e.g. the backbone's
    64-ch feature map) to the next block group's working width.

    **Architecture**::

        Input (B, H, W, C_in)
              |
        LayerNorm(epsilon=1e-6)
              |
        Conv2D(n_dims, kernel=1, padding=same)   # 1x1 channel projection
              |
        Output (B, H, W, n_dims)                 # spatial size preserved

    Inserted by the ``plus`` tail before a ConvNeXt block whenever the channel
    count changes (a depthwise ConvNeXt block runs at a fixed channel width and
    cannot itself change the channel count). Spatial size is preserved.
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


@keras.saving.register_keras_serializable()
class TheraTailPlus(keras.layers.Layer):
    """The ``plus`` tail: a depthwise-ConvNeXt feature refiner.

    **Intent**: Provide THERA's ``plus``-size feature refiner as a single
    serializable Keras layer that grows the sampled field's channels through a
    ConvNeXt stack. Sub-layers are created in ``__init__`` from the static
    ``block_defs`` (guide Pitfall 1: never create sub-layers in ``build``);
    ``build`` only propagates shapes so the ``.keras`` weight structure is fixed
    at construction time and round-trips byte-identically.

    **Architecture**::

        Input (B, H, W, C_in)  [C_in == block_defs[0][0], i.e. 64 for THERA]
              |
              v
        for (dims, k) in block_defs:           # THERA default: 16 blocks
            if dims != current_dim:
                _Projection(dims)              # LayerNorm -> Conv1x1(dims)
            ConvNextV1Block(kernel_size=k, filters=dims)   # depthwise + 4x MLP + residual
              |
              v
        Output (B, H, W, block_defs[-1][0])    [128 for THERA default]

    Assembles the THERA ``ConvNeXt(block_defs)`` stack: for each ``(dims, k)``
    block def, if ``dims`` differs from the running channel count a
    ``_Projection(dims)`` (LayerNorm + 1x1 conv) is inserted first, then a
    ``ConvNextV1Block(kernel_size=k, filters=dims)`` is applied. Output channels
    equal the last block's ``dims`` (128 for the THERA default).

    :param block_defs: Sequence of ``(n_dims, kernel_size)`` tuples. Defaults to
        THERA's ``[(64, 3)] * 6 + [(96, 3)] * 7 + [(128, 3)] * 3``.
    :param in_channels: Input channel count, used only to decide whether the very
        first block needs a leading ``_Projection``. Defaults to ``None``, which
        assumes the input matches ``block_defs[0][0]`` (THERA backbones always
        emit 64 = the first ``plus`` block dim, so no leading projection).
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

        # Sub-layers are created HERE (guide Pitfall 1: never in build()). The
        # projection-insertion logic depends only on consecutive block dims
        # (static); the first block's projection depends on in_channels, which
        # defaults to block_defs[0][0] (THERA backbones emit 64 = first block dim,
        # so no leading projection). build() only propagates shapes.
        #
        # INV (.keras round-trip): the creation ORDER and per-sub-layer name
        # (proj_{i} / convnext_{i}, i = block index) are byte-identical to the
        # former build()-time loop so existing saved weights reload unchanged.
        self._sublayers: List[keras.layers.Layer] = []
        # Parallel list of out_dims so build() can propagate shapes.
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
        # Sub-layers already exist (created in __init__); build() ONLY propagates
        # shapes (Keras-3 four-strike build ordering: every sub-layer must be
        # built with the right propagated channel shape for .keras weight reload).
        #
        # The leading-projection decision was made in __init__ from `in_channels`
        # (default: block_defs[0][0]). If the actual input width disagrees, the
        # pre-created sub-layer stack would silently feed wrong channels into the
        # first ConvNeXt block. Fail loud and direct the caller to pass
        # `in_channels` at construction (so the leading _Projection is created).
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


@keras.saving.register_keras_serializable()
class TheraTailPro(keras.layers.Layer):
    """The ``pro`` tail: a SwinIR (RSTB) feature refiner.

    **Intent**: Provide THERA's ``pro``-size feature refiner as a single
    serializable Keras layer: a SwinIR body (long-residual RSTB stack) that
    refines the sampled field while preserving spatial size and handling
    arbitrary (non-window-divisible, non-square) inputs via reflect-pad/crop.

    **Architecture**::

        Input (B, H, W, C)
              |
        conv_first(3x3) -> embed_dim ; res = x
              |
        reflect-pad H,W up to next multiple of window_size (E1)
              |
        for stage in depths:                 # one RSTB per depth
            res2 = x
            depth x SwinTransformerBlock(shift = 0 / window//2 alternating)
            rstb_conv(3x3) ; x = x + res2
              |
        x = conv_after_body(3x3) + res       # long residual
        crop back to original H,W (E1)
              |
        x = leaky_relu(conv_before_upsample(3x3) -> num_feat)
              |
        Output (B, H, W, num_feat)

    Forward (THERA ``SwinIR`` body, spatial-shape-preserving)::

        x = conv_first(x)                       # 3x3 -> embed_dim
        res = x
        for d in depths:                        # one RSTB per depth
            x = rstb_d(x)                        # depth Swin blocks + 3x3 bypass + residual
        x = conv_after_body(x) + res            # 3x3, long residual
        x = leaky_relu(conv_before_upsample(x)) # 3x3 -> num_feat
        return x                                 # (B, H, W, num_feat)

    Each RSTB = ``res2 = x; for i in range(depth): SwinBlock(shift = 0 if i even
    else window // 2)(x); x = conv(x); return x + res2``.

    Edge case E1: ``H, W`` are reflect-padded up to a multiple of
    ``window_size`` before the Swin stack and cropped back to the original
    ``H, W`` afterwards, so non-window-divisible / non-square inputs are handled
    and the output spatial shape exactly matches the input's.

    :param embed_dim: Swin working width. Default 180.
    :param depths: Swin blocks per RSTB. Default ``(7, 6)``.
    :param num_heads: Attention heads per RSTB. Default ``(6, 6)``.
    :param window_size: Attention window side length. Default 8.
    :param mlp_ratio: Swin MLP expansion ratio. Default 2.0.
    :param num_feat: Output channel count after ``conv_before_upsample``. Default 64.
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

        # RSTB stack. The Swin blocks are stored in a FLAT list (not a nested
        # list-of-lists): Keras reliably tracks AND restores layers held in a
        # flat attribute list, but a nested List[List[Layer]] silently fails to
        # round-trip each inner block's weights through .keras (the inner blocks
        # are tracked for trainable_weights yet their weights are NOT restored
        # on reload -> verified 100% output mismatch). Stage boundaries are
        # recovered from self.depths.
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
        # Explicit sublayer builds (Keras-3 four-strike ordering): every sublayer
        # must be built with the right propagated NHWC shape so .keras reload
        # restores its weights. Spatial dims may be padded at call time, so build
        # with dynamic (None) spatial dims; channels are what matter for weights.
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
        # Original spatial dims for the post-stack crop (E1).
        shape = ops.shape(x)
        h, w = shape[1], shape[2]

        x = self.conv_first(x, training=training)

        # DECISION plan_2026-06-11_f662207d/D-007: the pro tail reflect-pads H,W
        # up to a window-size multiple, runs the Swin stack, then crops back to
        # the exact original H,W (E1). Do NOT assert divisibility / require
        # window-aligned inputs instead -- THERA's hypernetwork feeds arbitrary
        # crop sizes here, and the reference SwinIR pads-then-crops. Do NOT
        # reimplement Swin/ConvNeXt internals: SwinTransformerBlock /
        # ConvNextV1Block are reused verbatim (Plus inserts a LayerNorm+1x1
        # _Projection only on channel change). See decisions.md D-007.
        # Reflect-pad H, W up to a multiple of window_size (E1). The Swin window
        # attention requires divisibility; conv-first output keeps NHWC layout.
        ws = self.window_size
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

    Mirrors THERA's ``build_tail(size)`` dispatcher.

    :param size: One of ``'air'`` (identity), ``'plus'`` (ConvNeXt), or
        ``'pro'`` (SwinIR/RSTB).
    :param in_channels: Input channel count from the backbone; forwarded to the
        ``plus`` tail so it creates the leading 1x1 projection when the backbone
        does not emit 64 channels. Ignored by ``air``/``pro``.
    :return: The corresponding tail layer.
    :raises ValueError: If ``size`` is not one of the three known keys.
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
