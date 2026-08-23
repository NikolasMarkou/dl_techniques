"""
Single-image super-resolution with a chain of windowed transformer blocks whose
attention maps are inherited layer to layer, then pixel-shuffle upsampling.

Super-resolution is an inverse problem in which the missing high-frequency
content has to be inferred from context, and the useful context is often far from
the pixel being reconstructed: a repeated texture, an edge continuing across the
image, a self-similar patch elsewhere in the scene. That argues for attention.
But attention over a full-resolution feature map is quadratic in pixel count, and
super-resolution runs at full resolution throughout — there is no downsampling
pyramid to hide behind — so the standard remedy is Swin's window partition, which
restricts attention to non-overlapping `ws x ws` tiles and recovers cross-window
reach by cyclically shifting the tiling in alternate layers. That reduces cost
from `O((HW)^2 C)` to `O(HW * ws^2 * C)` and is the substrate this model is
built on.

What "PFT" adds on top of that substrate is the observation that a deep stack of
such blocks recomputes its attention from scratch at every layer, throwing away
what the layer below already established about which token pairs matter. The
progressive focusing step reuses it: the previous block's attention map is
multiplied elementwise into the current block's raw scores before normalization.

`S_foc = (Q K^T / sqrt(d_h) + M_swmsa) (*) A_prev`, then `A = softmax(S_foc)`

The domain of that product is the detail worth being precise about, because it is
easy to assume the wrong one. The Hadamard product lands on the **logits**, not on
the probabilities. Where `A_prev[i, j]` is near zero the focused logit is pulled
toward *zero*, which is the uniform point of the softmax, not toward `-inf`; a
pair suppressed by an earlier layer is attenuated but recoverable if this layer's
own score is large. It also means that because `A_prev >= 0` while `S` is signed,
the product reverses the ordering among negative logits. Both are properties of
the mechanism as specified and the layers are trained jointly under them, so
neither is a defect to be "corrected".

One half of the published method is deliberately absent. The paper pairs
progressive focusing with sparse matrix multiplication that *skips* computing
similarities for pairs the inherited map has already ruled out — that is where its
efficiency claim comes from. In this implementation the sparse path is a stub, and
`ProgressiveFocusedAttention` raises `NotImplementedError` at construction for any
`sparsity_mode` other than `'none'` rather than silently running dense attention
while advertising sparse. Everything here therefore computes dense windowed
attention: the focusing behaviour reproduces, the speedup does not.

The block itself is pre-norm with two residual branches, `x' = x + DropPath(PFA(Norm1(x), A_prev))`
followed by `y = x' + DropPath(FFN(Norm2(x')))`, and it returns a *pair* —
`(features, attn_map)` — because the attention map is a second output that the
next block consumes. That is why `call` here threads two values through the loop
rather than one, and why the first block is invoked with a bare tensor while every
subsequent block receives a two-element **list**: a tuple would be misread by
Keras' shape machinery as a single shape, and a structured input containing `None`
raises before `call` is ever reached.

`num_blocks` looks like it describes a multi-resolution hierarchy and does not.
Every stage runs at the same spatial resolution and the same `embed_dim`; there is
no patch merging, no downsampling and no channel progression anywhere in the deep
feature extractor. The stage boundaries affect exactly two things: layer naming,
and the shift schedule, which restarts at `shift_size = 0` on each stage's first
block and alternates from there. The attention chain does *not* restart — a single
`prev_attn_map` is threaded across all stages unbroken from the first block to the
last. Uniform resolution is precisely what makes that possible, since the map's
shape `(B*nW, heads, ws^2, ws^2)` is constant throughout.

Around the block chain sits the standard SR skeleton: one 3x3 convolution produces
shallow features, the block chain refines them, `conv_after_body` projects the
result and a long skip adds the shallow features back, and only then does
resolution change. Deferring all upsampling to the very end is the design decision
that makes the model affordable — every attention operation runs at low-resolution
input size, and the `scale**2` factor is paid once, by rearranging channels into
space rather than by convolving at the output size. `'pixelshuffle'` composes 4x
as two 2x stages, which is cheaper and better-conditioned than a single 16x
channel expansion; `'pixelshuffledirect'` does the single-step version for
lightweight models. Note that `'nearest+conv'` derives its stage count as
`int(log2(scale))` and so silently under-upsamples for non-power-of-two scales,
and that `'pixelshuffle'` accepts only scales 2, 3 and 4.

Two constraints follow from the windowing and bind the caller. Input height and
width must be divisible by `window_size`, since the partition does not pad. More
restrictively, shifted-window blocks need **statically known** spatial dimensions:
the SW-MSA mask encodes which regions became non-adjacent under the cyclic shift
and is materialized at build time, so it cannot be constructed from dynamic
`None` dims. This model is not resolution-agnostic at call time the way a purely
convolutional SR network is.

Two implementation choices are deliberate and load-bearing. The stochastic-depth
schedule comes from `linear_drop_path_rates`, which returns plain Python floats;
an earlier `keras.ops.linspace` plus `.item()` formulation raised `AttributeError`
on TF eager tensors and left the entire `drop_path_rate > 0` branch dead, so no
backend-tensor linspace belongs here. And `build` ends with a concrete dummy
forward pass through `self.call`. The sub-layers are created in `build` but would
otherwise construct their weights lazily on first call; `build_from_config` during
a `.keras` reload recreates them unbuilt, and weight restoration then fails on
layers that "were never built". Forcing materialization here is what makes the
round-trip work — and it matters more than usual because the blocks are held in a
nested `List[List[Layer]]`, a structure that can otherwise restore fresh weights
while every layer count and path still matches.

References:
    - Long et al., 2025. Progressive Focused Transformer for Single Image
      Super-Resolution. CVPR. (https://arxiv.org/abs/2503.20337)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. ICCV. (https://arxiv.org/abs/2103.14030)
      The window partition, cyclic shift and SW-MSA mask reused wholesale.
    - Liang et al., 2021. SwinIR: Image Restoration Using Swin Transformer.
      (https://arxiv.org/abs/2108.10257)
      The shallow-feature / deep-feature / long-skip / upsample skeleton.
    - Shi et al., 2016. Real-Time Single Image and Video Super-Resolution Using an
      Efficient Sub-Pixel Convolutional Neural Network.
      (https://arxiv.org/abs/1609.05158)
      The pixel-shuffle upsampler.
    - Dong et al., 2022. CSWin Transformer: A General Vision Transformer Backbone
      with Cross-Shaped Windows. (https://arxiv.org/abs/2107.00652)
      Origin of LePE, the depthwise positional encoding applied to values.
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from typing import Any, Dict, Optional, List, Literal
from dl_techniques.layers.transformers.progressive_focused_transformer import PFTBlock
from dl_techniques.layers.pixel_unshuffle import PixelShuffle2D
from dl_techniques.utils.drop_path import linear_drop_path_rates


@keras.saving.register_keras_serializable()
class PFTSR(keras.Model):
    """
    Progressive Focused Transformer for Single Image Super-Resolution.

    A state-of-the-art transformer-based super-resolution model that achieves
    excellent performance through progressive focused attention mechanism.

    The architecture consists of:
    1. Shallow feature extraction (single conv layer)
    2. Deep feature extraction (multiple stages of PFT blocks)
    3. Reconstruction module (conv + pixel shuffle upsampling)

    Key innovations:
    - Progressive Focused Attention (PFA) that inherits attention maps across layers
    - Windowed attention with shifted windows for efficient computation
    - LePE (Locally-Enhanced Positional Encoding) for better spatial modeling

    Args:
        scale: Integer, upsampling scale factor (2, 3, or 4). Default: 4.
        in_channels: Integer, number of input image channels. Default: 3.
        embed_dim: Integer, embedding dimension. Default: 60.
        num_blocks: List of integers, number of PFT blocks in each stage.
            Default: [4, 4, 4, 6, 6, 6].
        num_heads: Integer, number of attention heads. Default: 6.
        window_size: Integer, size of the attention window. Default: 8. Note the
            two paper-sourced variants override this to 32 (see MODEL_VARIANTS);
            8 is only this constructor's own fallback.
        mlp_ratio: Float, expansion ratio for MLP. Default: 2.0.
        qkv_bias: Boolean, whether to use bias in QKV projections. Default: True.
        attention_dropout_rate: Float, dropout rate for attention. Default: 0.0.
        projection_dropout_rate: Float, dropout rate for projections. Default: 0.0.
        drop_path_rate: Float, stochastic depth rate. Default: 0.0.
        norm_type: String, normalization type ('layer_norm' or 'rms_norm'). Default: 'layer_norm'.
        use_lepe: Boolean, whether to use LePE. Default: True.
        upsampler: String, upsampling method ('pixelshuffle', 'pixelshuffledirect',
            or 'nearest+conv'). Default: 'pixelshuffle'.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, in_channels)`.
        Height and width should be divisible by window_size.

    Output shape:
        4D tensor with shape: `(batch_size, height * scale, width * scale, in_channels)`.

    Example:
        >>> import keras
        >>> # Create PFT-SR model for 4x super-resolution
        >>> model = PFTSR(scale=4, embed_dim=60, num_blocks=[4, 4, 4, 6, 6, 6])
        >>>
        >>> # Low-resolution input (48x48)
        >>> lr_image = keras.random.normal((1, 48, 48, 3))
        >>>
        >>> # Super-resolve to high-resolution (192x192)
        >>> sr_image = model(lr_image)
        >>> print(sr_image.shape)
        (1, 192, 192, 3)
        >>>
        >>> # The paper's PFT_light config, spelled out explicitly
        >>> model_light = PFTSR(scale=4, embed_dim=52, num_blocks=[2, 4, 6, 6, 6],
        ...                     num_heads=4, mlp_ratio=1.0, window_size=32)
        >>> sr_image_light = model_light(lr_image)
    """

    #: Public-name registry of the three named PFT-SR sizes (models/CLAUDE.md
    #: Axis 2). Hoisted verbatim out of ``create_pft_sr``'s body on 2026-08-19,
    #: where it was a local ``configs`` dict that nothing outside the function
    #: could enumerate. The remaining ``PFTSR(...)`` arguments the factory passes
    #: (``window_size``, ``upsampler``, the dropout rates, ...) are IDENTICAL
    #: across all three variants and stay in the factory rather than being
    #: restated three times here.
    #: DECISION plan-2026-08-23T091307-9a110062/D-463
    #: ``light`` and ``base`` are the paper's own two released configs, quoted field
    #: for field from the official training YAMLs so this package's "based on the
    #: CVPR 2025 paper" claim is true of its numbers and not just its mechanism:
    #:   base  <- options/train/001_PFT_SRx2_scratch.yml       (network_g)
    #:   light <- options/train/101_PFT_light_SRx2_scratch.yml (network_g)
    #:   https://github.com/CVL-UESTC/PFT-SR
    #: ``window_size`` is listed HERE, not left to the constructor default of 8,
    #: because 32 is part of what those two rows quote -- the published models are
    #: 32x32-window models and an 8x8 window is a different architecture wearing
    #: their name. Note the consequence before "fixing" it back: at window_size 32
    #: every shifted block materializes a (1, 1024, 1024) non-trainable
    #: ``attention_mask`` buffer, so ``count_params()`` reports ~13.2M for ``light``
    #: and ~34.4M for ``base`` while the TRAINABLE counts are 636,691 and 18,656,163
    #: (MEASURED). The inflated total is a mask-buffer artifact, not model size.
    #: ``repo_medium`` is this repo's own tier and has NO upstream counterpart --
    #: the official repo publishes exactly two configs, PFT and PFT_light, with no
    #: third size. It was called ``large`` while ``base`` was (wrongly) 60-wide; at
    #: the published 240 that name became false in this table's own terms, since 80
    #: now sits BETWEEN the two published sizes rather than above them.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # PFT_light, 101_PFT_light_SRx2_scratch.yml network_g
        'light': {
            'embed_dim': 52,
            'num_blocks': [2, 4, 6, 6, 6],
            'num_heads': 4,
            'mlp_ratio': 1.0,
            'window_size': 32,
        },
        # PFT, 001_PFT_SRx2_scratch.yml network_g
        'base': {
            'embed_dim': 240,
            'num_blocks': [4, 4, 4, 6, 6, 6],
            'num_heads': 6,
            'mlp_ratio': 2.0,
            'window_size': 32,
        },
        # Repo-original. Not in the paper, not in the official repo, not a rung
        # above ``base``. Kept because it is a cheap mid-size model, not because
        # anything published looks like it.
        'repo_medium': {
            'embed_dim': 80,
            'num_blocks': [6, 6, 6, 8, 8, 8],
            'num_heads': 8,
            'mlp_ratio': 2.0,
            'window_size': 8,
        }
    }

    def __init__(
            self,
            scale: int = 4,
            in_channels: int = 3,
            embed_dim: int = 60,
            num_blocks: List[int] = None,
            num_heads: int = 6,
            window_size: int = 8,
            mlp_ratio: float = 2.0,
            qkv_bias: bool = True,
            attention_dropout_rate: float = 0.0,
            projection_dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
            use_lepe: bool = True,
            upsampler: Literal['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'] = 'pixelshuffle',
            **kwargs
    ):
        super().__init__(**kwargs)

        if num_blocks is None:
            num_blocks = [4, 4, 4, 6, 6, 6]

        self.scale = scale
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.norm_type = norm_type
        self.use_lepe = use_lepe
        self.upsampler = upsampler

        # Calculate total number of blocks for stochastic depth
        self.total_blocks = sum(num_blocks)

        # Stochastic depth decay rule.
        # History: this used to be a `keras.ops.linspace` + `x.item()` list
        # comprehension. TF's EagerTensor has no `.item()`, so it raised
        # AttributeError and the whole `drop_path_rate > 0` branch was dead --
        # the SECOND blocker on that path, after the `StochasticDepth(drop_rate=)`
        # kwarg. Do NOT reintroduce a backend-tensor linspace here:
        # `linear_drop_path_rates` returns plain Python floats, so there is no
        # tensor-to-scalar conversion to get wrong, and it also covers the
        # `drop_path_rate == 0.0` and `total_blocks <= 1` cases (all zeros),
        # which is why the old explicit else-branch is gone.
        self.dpr = linear_drop_path_rates(self.total_blocks, drop_path_rate)

    def build(self, input_shape):
        """
        Build model layers.

        Args:
            input_shape: Shape tuple of the input.
        """
        # 1. Shallow feature extraction
        self.conv_first = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_first"
        )

        # 2. Deep feature extraction with PFT blocks
        self.stages = []
        block_idx = 0

        for stage_idx, num_blocks_in_stage in enumerate(self.num_blocks):
            stage_blocks = []

            for block_idx_in_stage in range(num_blocks_in_stage):
                # Alternate between regular and shifted window attention
                shift_size = 0 if (block_idx_in_stage % 2 == 0) else self.window_size // 2

                block = PFTBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    attention_dropout_rate=self.attention_dropout_rate,
                    projection_dropout_rate=self.projection_dropout_rate,
                    drop_path_rate=self.dpr[block_idx],
                    norm_type=self.norm_type,
                    use_lepe=self.use_lepe,
                    name=f"stage{stage_idx}_block{block_idx_in_stage}"
                )
                stage_blocks.append(block)
                block_idx += 1

            self.stages.append(stage_blocks)

        # 3. Reconstruction
        self.conv_after_body = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_after_body"
        )

        # 4. Upsampling
        if self.upsampler == 'pixelshuffle':
            # Traditional upsampling with pixel shuffle
            self.upsample = self._build_pixelshuffle_upsampler()
        elif self.upsampler == 'pixelshuffledirect':
            # Direct pixel shuffle
            self.upsample = self._build_pixelshuffledirect_upsampler()
        else:  # nearest+conv
            self.upsample = self._build_nearest_upsampler()

        # 5. Final reconstruction
        self.conv_last = keras.layers.Conv2D(
            filters=self.in_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_last"
        )

        # Explicitly materialize every sublayer via a concrete dummy forward so
        # their weights exist on .keras reload. The sublayers are created here but
        # build lazily on first call; without this, build_from_config re-creates
        # them unbuilt and weight loading fails ("... was never built"). Calling
        # self.call directly (not self(...)) avoids re-entering build().
        if all(d is not None for d in input_shape[1:]):
            dummy = keras.ops.zeros((1,) + tuple(input_shape[1:]))
            self.call(dummy, training=False)

        super().build(input_shape)

    def _build_pixelshuffle_upsampler(self) -> keras.Sequential:
        """
        Build pixel shuffle upsampler.

        Returns:
            Sequential model for upsampling.
        """
        layers = []

        if self.scale == 2 or self.scale == 3:
            layers.append(
                keras.layers.Conv2D(
                    self.embed_dim * (self.scale ** 2),
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name=f"upsample_conv"
                )
            )
            # DECISION plan_2026-06-15_39a31d4a/D-003: keras.ops.nn.depth_to_space
            # absent in Keras 3.8 -> use PixelShuffle2D (serializable, graph-safe);
            # PFTBlock kwarg is drop_path_rate, not drop_path. Do NOT restore the
            # Lambda(keras.ops.nn.depth_to_space) form (symbol does not exist /
            # breaks .keras round-trip). block_size matches the depth_to_space
            # factor at each site (self.scale / 2 / 2 / self.scale). See D-003.
            layers.append(
                PixelShuffle2D(block_size=self.scale, name="pixel_shuffle")
            )
        elif self.scale == 4:
            # 4x = 2x + 2x
            layers.append(
                keras.layers.Conv2D(
                    self.embed_dim * 4,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name="upsample_conv1"
                )
            )
            layers.append(
                PixelShuffle2D(block_size=2, name="pixel_shuffle1")
            )
            layers.append(
                keras.layers.Conv2D(
                    self.embed_dim * 4,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name="upsample_conv2"
                )
            )
            layers.append(
                PixelShuffle2D(block_size=2, name="pixel_shuffle2")
            )
        else:
            raise ValueError(f"Unsupported scale: {self.scale}")

        return keras.Sequential(layers, name="upsampler")

    def _build_pixelshuffledirect_upsampler(self) -> keras.Sequential:
        """
        Build direct pixel shuffle upsampler.

        Returns:
            Sequential model for upsampling.
        """
        layers = [
            keras.layers.Conv2D(
                self.embed_dim * (self.scale ** 2),
                kernel_size=3,
                strides=1,
                padding='same',
                name="upsample_conv"
            ),
            PixelShuffle2D(block_size=self.scale, name="pixel_shuffle")
        ]

        return keras.Sequential(layers, name="upsampler")

    def _build_nearest_upsampler(self) -> keras.Sequential:
        """
        Build nearest neighbor + conv upsampler.

        Only power-of-two scales are reachable here: the stage count is
        ``int(log2(scale))`` doubling stages, so any other scale would silently emit
        the wrong resolution (``scale=3`` gives one stage, i.e. a 2x output for a 3x
        request).

        Returns:
            Sequential model for upsampling.

        Raises:
            ValueError: If ``scale`` is not a power of two.
        """
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-078
        # This upsampler can only realize powers of two, and it used to accept anything.
        # `_build_pixelshuffle_upsampler` has always raised for unsupported scales; this
        # branch did not, so `PFTSR(scale=3, upsampler='nearest+conv')` built happily and
        # returned a 2x image for a 3x request. The module docstring named the defect and
        # shipped it anyway. `create_pft_sr` hardcodes `upsampler='pixelshuffle'`, so only
        # direct `PFTSR(...)` construction reaches here — which is exactly why it went
        # unnoticed, not a reason to leave it. Do NOT "fix" this by rounding the stage
        # count up: three doubling stages give 8x, not 3x. See decisions.md D-078.
        if self.scale < 1 or (self.scale & (self.scale - 1)) != 0:
            raise ValueError(
                f"Unsupported scale for upsampler='nearest+conv': {self.scale}. "
                f"This upsampler stacks int(log2(scale)) doubling stages, so it can "
                f"only realize powers of two (1, 2, 4, 8, ...); scale={self.scale} "
                f"would silently emit a "
                f"{2 ** max(self.scale.bit_length() - 1, 0) if self.scale > 0 else 0}x "
                f"image. "
                f"Use upsampler='pixelshuffle' (supports 2, 3 and 4) instead."
            )

        layers = []

        for i in range(int(keras.ops.log2(self.scale))):
            layers.extend([
                keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
                keras.layers.Conv2D(
                    self.embed_dim,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name=f"upsample_conv{i}"
                )
            ])

        return keras.Sequential(layers, name="upsampler")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of PFT-SR.

        Args:
            inputs: Input low-resolution images of shape (batch_size, height, width, channels).
            training: Boolean or boolean tensor, whether in training mode.

        Returns:
            Super-resolved high-resolution images.
        """
        # 1. Shallow feature extraction
        x = self.conv_first(inputs)
        residual = x

        # 2. Deep feature extraction with progressive focused attention
        prev_attn_map = None

        for stage_blocks in self.stages:
            for block in stage_blocks:
                # DECISION plan_2026-06-15_39a31d4a/D-003: do NOT pass
                # (x, None) as a structured input to PFTBlock — Keras' __call__
                # shape machinery (optree get_shapes_dict) raises on the None
                # element. The first block has no prior attention map; call it
                # with bare x (PFTBlock.call handles the non-tuple case and sets
                # prev_attn_map=None internally). See decisions.md D-003.
                # Pass a LIST (not a tuple) for the structured input so
                # PFTBlock.build's `isinstance(input_shape, list)` branch fires
                # and extracts x_shape = input_shape[0]; a tuple is mis-read as a
                # single shape -> `Invalid dtype: tuple` at norm build.
                if prev_attn_map is None:
                    x, prev_attn_map = block(x, training=training)
                else:
                    x, prev_attn_map = block(
                        [x, prev_attn_map], training=training
                    )

        # 3. Reconstruction
        x = self.conv_after_body(x)
        x = x + residual  # Global residual connection

        # 4. Upsampling
        x = self.upsample(x)

        # 5. Final reconstruction
        output = self.conv_last(x)

        return output

    def get_config(self):
        """Return model configuration."""
        config = super().get_config()
        config.update({
            "scale": self.scale,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "attention_dropout_rate": self.attention_dropout_rate,
            "projection_dropout_rate": self.projection_dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "norm_type": self.norm_type,
            "use_lepe": self.use_lepe,
            "upsampler": self.upsampler,
        })
        return config


def create_pft_sr(
        scale: int = 4,
        variant: Literal['base', 'light', 'repo_medium'] = 'base',
        **kwargs: Any,
) -> PFTSR:
    """
    Factory function to create PFT-SR models with predefined configurations.

    Args:
        scale: Integer, upsampling scale factor (2, 3, or 4).
        variant: String, model variant:
            - 'light': the paper's PFT_light (52 channels, [2, 4, 6, 6, 6] blocks,
              4 heads, mlp_ratio 1.0, window 32)
            - 'base': the paper's PFT (240 channels, [4, 4, 4, 6, 6, 6] blocks,
              6 heads, mlp_ratio 2.0, window 32)
            - 'repo_medium': repo-original mid-size tier (80 channels,
              [6, 6, 6, 8, 8, 8] blocks, window 8). No published counterpart --
              the official repo ships only the two configs above.
        **kwargs: Any :class:`PFTSR` constructor argument, overriding the variant
            table. This is the only route to ``window_size``, ``drop_path_rate``,
            ``upsampler``, ``norm_type``, ``use_lepe`` and the dropout knobs.

    Returns:
        PFTSR model instance.

    Example:
        >>> # Create base model for 4x SR
        >>> model = create_pft_sr(scale=4, variant='base')
        >>>
        >>> # Create lightweight model for 2x SR
        >>> model_light = create_pft_sr(scale=2, variant='light')
        >>>
        >>> # Create the repo-original mid-size model for 4x SR
        >>> model_medium = create_pft_sr(scale=4, variant='repo_medium')
    """
    if variant not in PFTSR.MODEL_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Available variants: {list(PFTSR.MODEL_VARIANTS.keys())}"
        )

    # DECISION plan-2026-08-19T163559-499b6f0e/D-118: `config.update(kwargs)` is the
    # house `from_variant` shape. The `.copy()` + `list(...)` beneath it protect the
    # LOCAL dict that `config.update(kwargs)` then mutates -- without them a caller's
    # override would be written straight into `PFTSR.MODEL_VARIANTS[variant]`. They do
    # NOT repair a model-side alias, and it is worth saying why the obvious probe
    # misleads here: `m.num_blocks is PFTSR.MODEL_VARIANTS[variant]['num_blocks']` reads
    # False even on the pre-fix code, because Keras 3 auto-tracking rewraps any list
    # assigned to a Layer/Model attribute as a NEW `TrackedList`. MEASURED against the
    # unfixed source: `m.num_blocks.append(99)` left the class table at [4,4,4,6,6,6].
    # The nine arguments this call used to spell out (`in_channels=3`,
    # `window_size=8`, `qkv_bias=True`, the three dropouts, `norm_type`, `use_lepe`,
    # `upsampler`) were each byte-identical to `PFTSR.__init__`'s own default, and
    # hard-coding them is exactly what made every one of them unreachable through
    # this factory. Do NOT re-inline them.
    config = PFTSR.MODEL_VARIANTS[variant].copy()
    config['num_blocks'] = list(config['num_blocks'])
    config.update(kwargs)

    return PFTSR(scale=scale, **config)