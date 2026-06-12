"""Flux2 KL-VAE for the Ideogram4 Keras port (NHWC, channels-last).

A faithful Keras 3 reimplementation of the Flux2 KL autoencoder used by
Ideogram4 (PyTorch ``autoencoder.py``). The PyTorch reference is channels-FIRST
(NCHW); this port is channels-LAST (NHWC) throughout. The building blocks are:

- :class:`ResnetBlock` -- ``GroupNorm32 -> swish -> Conv3x3 -> GroupNorm32 ->
  swish -> Conv3x3`` with a 1x1 conv skip when ``in_ch != out_ch``.
- :class:`AttnBlock` -- ``GroupNorm32`` + 1x1 q/k/v convs + scaled-dot-product
  self-attention over the flattened ``H*W`` spatial tokens + 1x1 proj_out, added
  residually. Used ONLY at the bottleneck (lowest resolution) -- never at full
  resolution (O((HW)^2) memory; see plan Failure Modes).
- :class:`Downsample` -- asymmetric pad ``[[0,0],[0,1],[0,1],[0,0]]`` (bottom +
  right by 1) then ``Conv2D(stride=2, padding="valid")``.
- :class:`Upsample` -- nearest x2 upsample + ``Conv2D(3x3, same)``.
- :class:`Encoder` / :class:`Decoder` -- the standard Flux2 down/up stacks with a
  mid block (ResnetBlock + AttnBlock + ResnetBlock).
- :class:`AutoEncoder` -- holds Encoder + Decoder, exposes ``encode`` (returns
  ``(z_mean, z_log_var)``), a KL :class:`Sampling` reparameterization, and
  ``decode``. ``call`` runs encode -> sample -> decode and returns the
  reconstruction. At pipeline inference only the DECODER is used.

GroupNorm uses the built-in ``keras.layers.GroupNormalization(groups=32,
epsilon=1e-6)`` directly (no repo-norms wrapper). Every channel count fed to a
GroupNorm must be divisible by 32; the config invariant (config.py
``_validate_vae_groupnorm``) guarantees this for the provided presets, and the
ctor also asserts it defensively.

swish(x) = x * sigmoid(x) is ``keras.activations.silu``.
"""

from __future__ import annotations

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.sampling import Sampling
from dl_techniques.models.ideogram4.config import (
    AutoEncoderParams,
    get_ideogram4_config,
)

# ---------------------------------------------------------------------

_GN_GROUPS: int = 32
_GN_EPS: float = 1e-6


def _group_norm(name: str) -> keras.layers.GroupNormalization:
    """Build a Flux2 ``GroupNorm32`` (groups=32, eps=1e-6) over the channel axis."""
    return keras.layers.GroupNormalization(
        groups=_GN_GROUPS, axis=-1, epsilon=_GN_EPS, name=name
    )


def _check_div32(value: int, what: str) -> None:
    """Assert ``value`` is divisible by 32 (GroupNormalization groups=32)."""
    if value % _GN_GROUPS != 0:
        raise ValueError(
            f"{what} ({value}) must be divisible by {_GN_GROUPS} "
            f"(GroupNormalization groups={_GN_GROUPS})."
        )


# ---------------------------------------------------------------------
# ResnetBlock
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class ResnetBlock(keras.layers.Layer):
    """Flux2 residual block: (GroupNorm32 + swish + Conv3x3) x2 + skip.

    .. code-block:: text

        x ---> GN32 -> swish -> Conv3x3 -> GN32 -> swish -> Conv3x3 ---> + ---> h
         \\______________________ skip (1x1 conv if in!=out) ___________/

    The skip is identity when ``in_channels == out_channels``, otherwise a
    learned ``Conv2D(out_channels, 1)``.

    :param in_channels: Input channel count (must be divisible by 32 -- it feeds
        the first GroupNorm).
    :type in_channels: int
    :param out_channels: Output channel count (must be divisible by 32 -- it feeds
        the second GroupNorm).
    :type out_channels: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``in_channels`` or ``out_channels`` is not divisible
        by 32.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        _check_div32(in_channels, "ResnetBlock in_channels (feeds GroupNorm)")
        _check_div32(out_channels, "ResnetBlock out_channels (feeds GroupNorm)")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = _group_norm(name="norm1")
        self.conv1 = keras.layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same", name="conv1"
        )
        self.norm2 = _group_norm(name="norm2")
        self.conv2 = keras.layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same", name="conv2"
        )
        # 1x1 projection skip only when the channel count changes.
        # DECISION plan_2026-06-12_59a18a10/D-007: nin_shortcut kept conditional
        # (created only when in_channels != out_channels). This is STRUCTURAL
        # (matches the PyTorch reference), NOT a runtime config toggle. Do NOT
        # "always create" an identity 1x1 conv when in==out per the guide's
        # weight-compat rule: that rule targets runtime toggles, and an identity
        # 1x1 conv here would add dead parameters and diverge from PyTorch. See
        # decisions.md D-007.
        self.nin_shortcut: Optional[keras.layers.Conv2D] = None
        if in_channels != out_channels:
            self.nin_shortcut = keras.layers.Conv2D(
                out_channels, kernel_size=1, strides=1, padding="valid",
                name="nin_shortcut",
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Explicit sub-layer build (guide §3.2): build each owned sub-layer in
        # forward order with propagated shapes; the residual/skip path builds on
        # the ORIGINAL input_shape. No new sub-layers created here.
        self.norm1.build(input_shape)
        norm1_shape = self.norm1.compute_output_shape(input_shape)
        self.conv1.build(norm1_shape)
        conv1_shape = self.conv1.compute_output_shape(norm1_shape)
        self.norm2.build(conv1_shape)
        norm2_shape = self.norm2.compute_output_shape(conv1_shape)
        self.conv2.build(norm2_shape)
        if self.nin_shortcut is not None:
            self.nin_shortcut.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        h = inputs
        h = self.norm1(h, training=training)
        h = keras.activations.silu(h)
        h = self.conv1(h, training=training)
        h = self.norm2(h, training=training)
        h = keras.activations.silu(h)
        h = self.conv2(h, training=training)

        skip = inputs
        if self.nin_shortcut is not None:
            skip = self.nin_shortcut(inputs, training=training)
        return skip + h

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape[:-1]) + (self.out_channels,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
            }
        )
        return config


# ---------------------------------------------------------------------
# AttnBlock (spatial self-attention over H*W tokens)
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class AttnBlock(keras.layers.Layer):
    """Spatial self-attention block (Flux2 mid-block only).

    ``GroupNorm32`` -> 1x1 q/k/v convs -> flatten ``(B, H, W, C)`` to
    ``(B, H*W, C)`` -> scaled-dot-product self-attention (scale ``1/sqrt(C)``)
    over the ``H*W`` tokens -> reshape back -> 1x1 ``proj_out`` -> added
    residually to the input.

    .. warning::
        This block is O((H*W)^2) in memory and is intended ONLY for the VAE
        bottleneck (lowest resolution). Placing it at full resolution OOMs.

    :param channels: Channel count (must be divisible by 32 -- feeds GroupNorm).
    :type channels: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``channels`` is not divisible by 32.
    """

    def __init__(
        self,
        channels: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        _check_div32(channels, "AttnBlock channels (feeds GroupNorm)")
        self.channels = channels

        self.norm = _group_norm(name="norm")
        self.q = keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="valid", name="q"
        )
        self.k = keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="valid", name="k"
        )
        self.v = keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="valid", name="v"
        )
        self.proj_out = keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="valid", name="proj_out"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Explicit sub-layer build (guide §3.2). norm + q/k/v all consume the
        # input shape; proj_out consumes the (channel-preserving) attention
        # output, which has the same shape as the input. No new sub-layers.
        self.norm.build(input_shape)
        norm_shape = self.norm.compute_output_shape(input_shape)
        self.q.build(norm_shape)
        self.k.build(norm_shape)
        self.v.build(norm_shape)
        # Attention preserves (B, H, W, C); proj_out builds on that shape.
        self.proj_out.build(self.q.compute_output_shape(norm_shape))
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        h = self.norm(inputs, training=training)
        q = self.q(h, training=training)
        k = self.k(h, training=training)
        v = self.v(h, training=training)

        # Flatten spatial dims to tokens: (B, H, W, C) -> (B, H*W, C).
        shp = keras.ops.shape(h)
        b, hh, ww, c = shp[0], shp[1], shp[2], shp[3]
        n = hh * ww
        # Build reshape targets with ops.stack (not python tuples of symbolic
        # scalars) for graph safety (guide §4).
        bnc = keras.ops.stack([b, n, c])
        q = keras.ops.reshape(q, bnc)
        k = keras.ops.reshape(k, bnc)
        v = keras.ops.reshape(v, bnc)

        # Scaled dot-product attention over the H*W tokens (scale 1/sqrt(C)).
        scale = 1.0 / keras.ops.sqrt(keras.ops.cast(c, h.dtype))
        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 2, 1))) * scale
        attn = keras.ops.softmax(scores, axis=-1)
        out = keras.ops.matmul(attn, v)  # (B, N, C)

        # Restore spatial layout, project, and add residually.
        out = keras.ops.reshape(out, keras.ops.stack([b, hh, ww, c]))
        out = self.proj_out(out, training=training)
        return inputs + out

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


# ---------------------------------------------------------------------
# Downsample (asymmetric pad + stride-2 valid conv)
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class Downsample(keras.layers.Layer):
    """Stride-2 spatial downsample with asymmetric padding.

    NHWC asymmetric pad ``[[0,0],[0,1],[0,1],[0,0]]`` (pad bottom + right by 1)
    then ``Conv2D(channels, 3, stride=2, padding="valid")``. This reproduces the
    PyTorch ``F.pad(x, (0,1,0,1))`` + stride-2 conv exactly (PyTorch pads the
    LAST two NCHW spatial dims on the high side, which in NHWC are H and W).

    :param channels: Output channel count.
    :type channels: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
        self,
        channels: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.conv = keras.layers.Conv2D(
            channels, kernel_size=3, strides=2, padding="valid", name="conv"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Explicit sub-layer build (guide §3.2). The conv consumes the
        # asymmetrically PADDED input (bottom+right +1), so build it on that
        # padded shape to match call().
        b, h, w, ch = input_shape
        padded_shape = (
            b,
            None if h is None else h + 1,
            None if w is None else w + 1,
            ch,
        )
        self.conv.build(padded_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # Pad bottom + right by 1 (NHWC), then stride-2 valid conv.
        padded = keras.ops.pad(
            inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="constant"
        )
        return self.conv(padded, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        b, h, w, _ = input_shape
        # (dim + 1 pad - 3 kernel) // 2 + 1 = (dim - 2) // 2 + 1 = ceil(dim / 2).
        out_h = None if h is None else (h + 1 - 3) // 2 + 1
        out_w = None if w is None else (w + 1 - 3) // 2 + 1
        return (b, out_h, out_w, self.channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


# ---------------------------------------------------------------------
# Upsample (nearest x2 + 3x3 conv)
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class Upsample(keras.layers.Layer):
    """Nearest-neighbour x2 upsample + ``Conv2D(3x3, same)`` (Flux2 Upsample).

    # DECISION plan_2026-06-12_59a18a10/D-005: this is a thin sub-layer wrapper
    # of ``UpSampling2D(2, "nearest") + Conv2D(channels, 3, same)``, NOT a call to
    # ``layers.upsample.upsample(..., "nearest_conv2d_3x3", ...)``. Do NOT replace
    # it with that function: ``upsample()`` is a FUNCTIONAL-graph builder that
    # takes a live tensor and returns a tensor (it calls ``UpSampling2D``/
    # ``conv2d_wrapper`` on its input inline), so it cannot be owned as a
    # constructed sub-layer inside a subclassed ``keras.Model`` Decoder without
    # building a Functional sub-model. The op sequence here is byte-identical to
    # the ``nearest_conv2d_3x3`` branch (UpSampling2D size=2 nearest -> Conv2D
    # kernel 3, stride 1, padding same). See decisions.md D-005.

    :param channels: Output channel count.
    :type channels: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
        self,
        channels: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.up = keras.layers.UpSampling2D(
            size=(2, 2), interpolation="nearest", name="up"
        )
        self.conv = keras.layers.Conv2D(
            channels, kernel_size=3, strides=1, padding="same", name="conv"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Explicit sub-layer build (guide §3.2). UpSampling2D doubles spatial
        # dims (channels unchanged); conv consumes that upsampled shape.
        self.up.build(input_shape)
        up_shape = self.up.compute_output_shape(input_shape)
        self.conv.build(up_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x = self.up(inputs)
        return self.conv(x, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        b, h, w, _ = input_shape
        out_h = None if h is None else h * 2
        out_w = None if w is None else w * 2
        return (b, out_h, out_w, self.channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


# ---------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class Encoder(keras.layers.Layer):
    """Flux2 KL-VAE encoder: image -> ``2 * z_channels`` latent params.

    ``conv_in`` (Conv3x3) -> per-level [num_res_blocks ResnetBlocks + Downsample
    between levels] -> mid (ResnetBlock + AttnBlock + ResnetBlock) -> norm_out
    (GroupNorm32 + swish) -> conv_out (Conv3x3 to ``2 * z_channels``) ->
    ``quant_conv`` (Conv1x1 to ``2 * z_channels``). The output channels carry the
    concatenated ``(mu || logvar)`` (each ``z_channels`` wide).

    :param resolution: Square input edge (informational; not used to build).
    :param in_channels: Input image channels (RGB = 3).
    :param ch: Base channel width.
    :param ch_mult: Per-stage channel multipliers over ``ch``.
    :param num_res_blocks: ResnetBlocks per resolution stage.
    :param z_channels: Latent channels; output is ``2 * z_channels``.
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        z_channels: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.resolution = resolution
        self.in_channels = in_channels
        self.ch = ch
        self.ch_mult = tuple(int(m) for m in ch_mult)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        num_resolutions = len(self.ch_mult)

        # conv_in: in_channels -> ch.
        self.conv_in = keras.layers.Conv2D(
            ch, kernel_size=3, strides=1, padding="same", name="conv_in"
        )

        # Down stack: flat lists of sub-layers (NO List[List] containers).
        self.down_blocks: List[ResnetBlock] = []
        self.down_samplers: List[Downsample] = []
        block_in = ch
        for level in range(num_resolutions):
            block_out = ch * self.ch_mult[level]
            for blk in range(num_res_blocks):
                self.down_blocks.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        name=f"down_{level}_block_{blk}",
                    )
                )
                block_in = block_out
            # Downsample between levels (not after the last level).
            if level != num_resolutions - 1:
                self.down_samplers.append(
                    Downsample(block_in, name=f"down_{level}_downsample")
                )
            else:
                self.down_samplers.append(None)

        # Mid block: ResnetBlock + AttnBlock + ResnetBlock (channels = block_in).
        self.mid_block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, name="mid_block_1"
        )
        self.mid_attn = AttnBlock(block_in, name="mid_attn")
        self.mid_block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, name="mid_block_2"
        )

        # Output: GroupNorm32 + swish + Conv3x3 to 2*z, then 1x1 quant_conv.
        self.norm_out = _group_norm(name="norm_out")
        self.conv_out = keras.layers.Conv2D(
            2 * z_channels, kernel_size=3, strides=1, padding="same",
            name="conv_out",
        )
        self.quant_conv = keras.layers.Conv2D(
            2 * z_channels, kernel_size=1, strides=1, padding="valid",
            name="quant_conv",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Explicit sub-layer build (guide §3.2): thread the shape through the
        # whole stack in the SAME order call() consumes it, using each child's
        # compute_output_shape to get the next shape. No new sub-layers.
        self.conv_in.build(input_shape)
        shape = self.conv_in.compute_output_shape(input_shape)

        num_resolutions = len(self.ch_mult)
        block_idx = 0
        for level in range(num_resolutions):
            for _ in range(self.num_res_blocks):
                blk = self.down_blocks[block_idx]
                blk.build(shape)
                shape = blk.compute_output_shape(shape)
                block_idx += 1
            sampler = self.down_samplers[level]
            if sampler is not None:
                sampler.build(shape)
                shape = sampler.compute_output_shape(shape)

        # Mid block on the bottleneck shape.
        self.mid_block_1.build(shape)
        shape = self.mid_block_1.compute_output_shape(shape)
        self.mid_attn.build(shape)
        shape = self.mid_attn.compute_output_shape(shape)
        self.mid_block_2.build(shape)
        shape = self.mid_block_2.compute_output_shape(shape)

        self.norm_out.build(shape)
        shape = self.norm_out.compute_output_shape(shape)
        self.conv_out.build(shape)
        shape = self.conv_out.compute_output_shape(shape)
        self.quant_conv.build(shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        h = self.conv_in(inputs, training=training)

        num_resolutions = len(self.ch_mult)
        block_idx = 0
        for level in range(num_resolutions):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, training=training)
                block_idx += 1
            sampler = self.down_samplers[level]
            if sampler is not None:
                h = sampler(h, training=training)

        h = self.mid_block_1(h, training=training)
        h = self.mid_attn(h, training=training)
        h = self.mid_block_2(h, training=training)

        h = self.norm_out(h, training=training)
        h = keras.activations.silu(h)
        h = self.conv_out(h, training=training)
        h = self.quant_conv(h, training=training)
        return h

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        b, h, w, _ = input_shape
        num_down = len(self.ch_mult) - 1
        for _ in range(num_down):
            h = None if h is None else (h + 1) // 2
            w = None if w is None else (w + 1) // 2
        return (b, h, w, 2 * self.z_channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "resolution": self.resolution,
                "in_channels": self.in_channels,
                "ch": self.ch,
                "ch_mult": list(self.ch_mult),
                "num_res_blocks": self.num_res_blocks,
                "z_channels": self.z_channels,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Encoder":
        config = dict(config)
        if "ch_mult" in config:
            config["ch_mult"] = tuple(config["ch_mult"])
        return cls(**config)


# ---------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class Decoder(keras.layers.Layer):
    """Flux2 KL-VAE decoder: ``z_channels`` latent -> reconstructed image.

    ``post_quant_conv`` (Conv1x1) -> conv_in (Conv3x3 to ``ch * ch_mult[-1]``) ->
    mid (ResnetBlock + AttnBlock + ResnetBlock) -> per-level reversed
    [(num_res_blocks + 1) ResnetBlocks + Upsample between levels] -> norm_out
    (GroupNorm32 + swish) -> conv_out (Conv3x3 to ``out_channels``).

    :param resolution: Square output edge (informational).
    :param ch: Base channel width.
    :param out_channels: Output (reconstructed) channels.
    :param ch_mult: Per-stage channel multipliers over ``ch``.
    :param num_res_blocks: Base ResnetBlocks per stage (decoder uses +1).
    :param z_channels: Latent channels at the decoder input.
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
        self,
        resolution: int,
        ch: int,
        out_channels: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        z_channels: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.resolution = resolution
        self.ch = ch
        self.out_channels = out_channels
        self.ch_mult = tuple(int(m) for m in ch_mult)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        num_resolutions = len(self.ch_mult)

        block_in = ch * self.ch_mult[-1]

        # post_quant_conv: z -> z (1x1), then conv_in: z -> block_in (3x3).
        self.post_quant_conv = keras.layers.Conv2D(
            z_channels, kernel_size=1, strides=1, padding="valid",
            name="post_quant_conv",
        )
        self.conv_in = keras.layers.Conv2D(
            block_in, kernel_size=3, strides=1, padding="same", name="conv_in"
        )

        # Mid block (channels = block_in).
        self.mid_block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, name="mid_block_1"
        )
        self.mid_attn = AttnBlock(block_in, name="mid_attn")
        self.mid_block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, name="mid_block_2"
        )

        # Up stack (reversed levels): (num_res_blocks + 1) ResnetBlocks per level,
        # Upsample between levels (not after level 0). Flat lists, NO List[List].
        self.up_blocks: List[ResnetBlock] = []
        self.up_samplers: List[Optional[Upsample]] = []
        for level in reversed(range(num_resolutions)):
            block_out = ch * self.ch_mult[level]
            for blk in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        name=f"up_{level}_block_{blk}",
                    )
                )
                block_in = block_out
            # Upsample between levels (not after the final, level 0).
            if level != 0:
                self.up_samplers.append(
                    Upsample(block_in, name=f"up_{level}_upsample")
                )
            else:
                self.up_samplers.append(None)

        # Output: GroupNorm32 + swish + Conv3x3 to out_channels.
        self.norm_out = _group_norm(name="norm_out")
        self.conv_out = keras.layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same",
            name="conv_out",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Explicit sub-layer build (guide §3.2): thread the shape through the
        # whole stack in the SAME order call() consumes it, using each child's
        # compute_output_shape to get the next shape. No new sub-layers.
        self.post_quant_conv.build(input_shape)
        shape = self.post_quant_conv.compute_output_shape(input_shape)
        self.conv_in.build(shape)
        shape = self.conv_in.compute_output_shape(shape)

        # Mid block on the bottleneck shape.
        self.mid_block_1.build(shape)
        shape = self.mid_block_1.compute_output_shape(shape)
        self.mid_attn.build(shape)
        shape = self.mid_attn.compute_output_shape(shape)
        self.mid_block_2.build(shape)
        shape = self.mid_block_2.compute_output_shape(shape)

        num_resolutions = len(self.ch_mult)
        block_idx = 0
        for sampler_pos in range(num_resolutions):
            for _ in range(self.num_res_blocks + 1):
                blk = self.up_blocks[block_idx]
                blk.build(shape)
                shape = blk.compute_output_shape(shape)
                block_idx += 1
            sampler = self.up_samplers[sampler_pos]
            if sampler is not None:
                sampler.build(shape)
                shape = sampler.compute_output_shape(shape)

        self.norm_out.build(shape)
        shape = self.norm_out.compute_output_shape(shape)
        self.conv_out.build(shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        h = self.post_quant_conv(inputs, training=training)
        h = self.conv_in(h, training=training)

        h = self.mid_block_1(h, training=training)
        h = self.mid_attn(h, training=training)
        h = self.mid_block_2(h, training=training)

        num_resolutions = len(self.ch_mult)
        block_idx = 0
        # up_samplers were appended in reversed-level order; iterate in lockstep.
        for sampler_pos in range(num_resolutions):
            for _ in range(self.num_res_blocks + 1):
                h = self.up_blocks[block_idx](h, training=training)
                block_idx += 1
            sampler = self.up_samplers[sampler_pos]
            if sampler is not None:
                h = sampler(h, training=training)

        h = self.norm_out(h, training=training)
        h = keras.activations.silu(h)
        h = self.conv_out(h, training=training)
        return h

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        b, h, w, _ = input_shape
        num_up = len(self.ch_mult) - 1
        for _ in range(num_up):
            h = None if h is None else h * 2
            w = None if w is None else w * 2
        return (b, h, w, self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "resolution": self.resolution,
                "ch": self.ch,
                "out_channels": self.out_channels,
                "ch_mult": list(self.ch_mult),
                "num_res_blocks": self.num_res_blocks,
                "z_channels": self.z_channels,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Decoder":
        config = dict(config)
        if "ch_mult" in config:
            config["ch_mult"] = tuple(config["ch_mult"])
        return cls(**config)


# ---------------------------------------------------------------------
# AutoEncoder
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class AutoEncoder(keras.Model):
    """Flux2 KL-VAE: Encoder + KL ``Sampling`` reparameterization + Decoder.

    Public surface:

    - ``encode(x) -> (z_mean, z_log_var)`` -- runs the encoder and splits its
      ``2 * z_channels`` output along the channel axis into ``(mu, logvar)``,
      each ``z_channels`` wide. This is DETERMINISTIC.
    - ``sample(z_mean, z_log_var) -> z`` -- KL reparameterization via the shared
      :class:`Sampling` layer (rank-agnostic; accepts ``(B, H, W, z)``).
    - ``decode(z) -> image`` -- runs the decoder on a latent of shape
      ``(B, H, W, z_channels)``.
    - ``call(x, training=None) -> reconstruction`` -- ``encode -> sample ->
      decode``, returning the reconstruction tensor.

    At pipeline inference only :meth:`decode` is used (the diffusion sampler
    produces the latent directly).

    The PyTorch ``AutoEncoder`` carries a ``BatchNorm2d`` on patchified latents
    for the training/latent-norm path; it is NOT used by decode-at-inference. It
    is OMITTED here to avoid dead code -- the pipeline applies the explicit
    shift/scale latent normalization (``latent_norm.py``) instead.

    :param params: The :class:`AutoEncoderParams` describing the VAE.
    :param sampling_seed: Optional seed for the KL :class:`Sampling` layer.
    :param kwargs: Additional ``keras.Model`` arguments.

    :raises TypeError: If ``params`` is not an :class:`AutoEncoderParams`.
    """

    def __init__(
        self,
        params: AutoEncoderParams,
        sampling_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(params, AutoEncoderParams):
            raise TypeError(
                f"params must be an AutoEncoderParams, got {type(params)}"
            )

        self.params = params
        self.sampling_seed = sampling_seed
        self.z_channels = params.z_channels

        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            name="encoder",
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            ch=params.ch,
            out_channels=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            name="decoder",
        )
        self.sampling = Sampling(seed=sampling_seed, name="sampling")

        logger.debug(
            f"Initialized AutoEncoder(z_channels={params.z_channels}, "
            f"ch={params.ch}, ch_mult={params.ch_mult}, "
            f"num_res_blocks={params.num_res_blocks})"
        )

    def encode(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode an image to ``(z_mean, z_log_var)`` (deterministic).

        :param x: Input image ``(B, H, W, in_channels)``.
        :param training: Forwarded to the encoder.
        :return: ``(z_mean, z_log_var)``, each ``(B, H', W', z_channels)``.
        """
        moments = self.encoder(x, training=training)  # (B, H', W', 2*z)
        z_mean = moments[..., : self.z_channels]
        z_log_var = moments[..., self.z_channels:]
        return z_mean, z_log_var

    def sample(
        self,
        z_mean: keras.KerasTensor,
        z_log_var: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Reparameterized KL sample ``z = mu + exp(0.5*logvar) * eps``."""
        return self.sampling([z_mean, z_log_var])

    def decode(
        self,
        z: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Decode a latent ``(B, H', W', z_channels)`` to an image."""
        return self.decoder(z, training=training)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Full VAE forward: encode -> sample -> decode -> reconstruction.

        :param inputs: Input image ``(B, H, W, in_channels)``.
        :param training: Forwarded to encoder/decoder.
        :return: Reconstruction ``(B, H, W, out_channels)``.
        """
        z_mean, z_log_var = self.encode(inputs, training=training)
        z = self.sample(z_mean, z_log_var)
        return self.decode(z, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        b, h, w, _ = input_shape
        return (b, h, w, self.params.out_ch)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "params": self.params.to_dict(),
                "sampling_seed": self.sampling_seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AutoEncoder":
        config = dict(config)
        config["params"] = AutoEncoderParams.from_dict(config["params"])
        return cls(**config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_ideogram4_autoencoder(
    variant: str = "tiny",
    sampling_seed: Optional[int] = None,
    **overrides: Any,
) -> AutoEncoder:
    """Build a Flux2 :class:`AutoEncoder` from a named preset.

    Retrieves the ``(config, ae)`` pair for ``variant`` via
    :func:`get_ideogram4_config` (which runs all VAE invariants including the
    GroupNorm-divisibility check), applies any ``AutoEncoderParams`` field
    ``overrides`` (re-validated), and returns the constructed model.

    :param variant: One of the config presets (``"tiny"`` or ``"full"``).
    :param sampling_seed: Optional seed for the KL :class:`Sampling` layer.
    :param overrides: Field overrides applied to the preset
        :class:`AutoEncoderParams` (e.g. ``ch=64``).
    :return: The constructed (un-built) autoencoder model.
    """
    _, ae = get_ideogram4_config(variant)
    if overrides:
        merged = {**ae.to_dict(), **overrides}
        ae = AutoEncoderParams.from_dict(merged)

    logger.info(
        "Creating Ideogram4 AutoEncoder variant='%s' (ch=%d, ch_mult=%s, "
        "z_channels=%d)",
        variant,
        ae.ch,
        ae.ch_mult,
        ae.z_channels,
    )
    return AutoEncoder(params=ae, sampling_seed=sampling_seed)

# ---------------------------------------------------------------------
