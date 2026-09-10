"""DocRes: one Restormer backbone that serves every document-restoration task.

DocRes takes a 6-channel input -- the 3-channel RGB document image with a
3-channel classical-CV "DTSPrompt" stacked on top of it -- and emits a
3-channel restored image. There is no task embedding, no task head and no
conditioning module anywhere in this file: the *only* thing that distinguishes
dewarping from deshadowing from binarization is which three prompt channels the
caller stacked, which output channels the loss supervises, and what the
inference shim does afterwards. That is the paper's central claim, and it is
why a single network class is enough.

The principle
-------------
The backbone is an unmodified Restormer (Zamir et al., 2022). Restormer's
mechanism is a transformer whose attention runs over the CHANNEL axis rather
than the pixel axis: for a feature map with ``c`` channels and ``n = h*w``
pixels the attention matrix is ``c x c``, so cost is linear in ``n`` instead of
quadratic. That is what makes a full-resolution document page -- routinely
1024 px on a side or larger -- affordable at all, and it is why this
architecture, rather than a ViT, is the one a document-restoration generalist
can be built on.

The architecture
----------------
A four-level U-shaped encoder/decoder over
:class:`~dl_techniques.models.vision.image_restoration.doc_res.components.RestormerTransformerBlock`,
with ``dim`` doubling and resolution halving at every step down::

    inputs (B, H, W, 6)
      │
      ▼  patch_embed 3x3, bias-free
    enc1_in (H, W, 48) ──────────────────────────────────────┐
      ▼  encoder_level1: 2 blocks, 1 head                    │
    out_enc1 (H, W, 48) ─────────────────────────────┐       │
      ▼  down1_2                                     │       │
      ▼  encoder_level2: 3 blocks, 2 heads           │       │
    out_enc2 (H/2, W/2, 96) ─────────────────┐       │       │
      ▼  down2_3                             │       │       │
      ▼  encoder_level3: 3 blocks, 4 heads   │       │       │
    out_enc3 (H/4, W/4, 192) ────────┐       │       │       │
      ▼  down3_4                     │       │       │       │
      ▼  latent: 4 blocks, 8 heads   │       │       │       │
      ▼  up4_3          (H/4, 192)   │       │       │       │
    concat ◄─────────────────────────┘       │       │       │
      ▼  reduce_chan_level3 1x1 (384 -> 192) │       │       │
      ▼  decoder_level3: 3 blocks, 4 heads   │       │       │
      ▼  up3_2          (H/2, 96)            │       │       │
    concat ◄─────────────────────────────────┘       │       │
      ▼  reduce_chan_level2 1x1 (192 -> 96)          │       │
      ▼  decoder_level2: 3 blocks, 2 heads           │       │
      ▼  up2_1          (H, 48)                      │       │
    concat ◄─────────────────────────────────────────┘       │
      ▼  *** NO reduce_chan_level1 *** -> stays at 96        │
      ▼  decoder_level1: 2 blocks, 1 head, at 96 channels    │
      ▼  refinement: 4 blocks, 1 head, at 96 channels        │
      +  skip_conv 1x1 (48 -> 96) ◄──────────────────────────┘
      ▼  output 3x3, bias-free
    outputs (B, H, W, 3)

Six things about that diagram are asymmetries a reader would otherwise "fix",
and every one of them is faithful to the reference implementation rather than
an oversight. They are enumerated, with their upstream line numbers, at
``# DECISION plan-2026-09-08T111844-de235227/D-015`` in this file, and each is
commented again at the site where it is wired.

Deliberate behavioural choices
------------------------------
* **Spatial dimensions must be divisible by 8** and this model raises a
  ``ValueError`` naming the offending size rather than padding internally. The
  reference pads OUTSIDE the network, in its inference script; padding is a
  property of the inference contract (the padding has to be *removed* from the
  prediction afterwards, which the network cannot do for a caller who never
  learns it happened), so it lives in the inference shim here too. See
  ``# DECISION plan-2026-09-08T111844-de235227/D-016``.
* **``pretrained=True`` raises ``NotImplementedError``.** No DocRes weights are
  distributed with this library, and -- separately and more fundamentally --
  the upstream PyTorch checkpoint cannot be transferred into this port at all,
  because this repo's pixel-shuffle layers use a different channel-block
  ordering from ``torch.nn.PixelShuffle`` (measured; see
  ``components.py``'s D-007 anchors). Returning random weights, or warning and
  continuing, would be worse than raising.
* **``MODEL_VARIANTS`` has exactly two rows**, both cited to a real source: the
  DocRes configuration and the Restormer paper default. There is no
  name-to-scale indirection and therefore no second hyperparameter table -- a
  variant here *is* a depth configuration, so splitting the single table would
  create an indirection with nothing on either side of it.

References:
    - Zhang et al., 2024. DocRes: A Generalist Model Toward Unifying Document
      Image Restoration Tasks. CVPR 2024. (https://arxiv.org/abs/2405.04408)
    - Zamir et al., 2022. Restormer: Efficient Transformer for High-Resolution
      Image Restoration. CVPR 2022. (https://arxiv.org/abs/2111.09881)
"""

import keras
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import materialize_sublayers

from .components import (
    RestormerDownsample,
    RestormerTransformerBlock,
    RestormerUpsample,
)

# ---------------------------------------------------------------------
# The encoder/decoder has three downsampling stages, each of which halves the
# spatial resolution, so the input must be divisible by 2**3.
# ---------------------------------------------------------------------

SPATIAL_DIVISOR: int = 8

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_res.model")
class DocRes(keras.Model):
    """The DocRes generalist document-restoration network.

    A four-level Restormer U-Net. See the module docstring for the wiring
    diagram and for the list of deliberate asymmetries; this class documents
    the knobs.

    :param dim: Channel width of encoder level 1. Every deeper level is
        ``dim * 2**k``. Defaults to 48, the value at every DocRes call site.
    :type dim: int
    :param num_blocks: Transformer blocks at encoder levels 1-4. Level 4 is the
        latent bottleneck. Decoder level ``k`` reuses ``num_blocks[k - 1]``.
        Defaults to ``[2, 3, 3, 4]`` (the DocRes configuration) when ``None``.
    :type num_blocks: Optional[Sequence[int]]
    :param num_refinement_blocks: Blocks in the full-resolution refinement
        stage, which runs at ``2 * dim`` channels. Independent of
        ``num_blocks``. Defaults to 4.
    :type num_refinement_blocks: int
    :param heads: MDTA head counts at levels 1-4. Defaults to ``[1, 2, 4, 8]``
        when ``None``, which holds the per-head channel width at ``dim`` on
        every level.
    :type heads: Optional[Sequence[int]]
    :param ffn_expansion_factor: GDFN hidden-width multiplier; the hidden width
        is ``int(dim_level * factor)`` with Python truncation. Defaults to 2.66.
    :type ffn_expansion_factor: float
    :param use_bias: Whether the convolutions inside the transformer blocks
        carry a bias. Defaults to ``False``. Note that the resampler and
        projection convolutions are bias-free unconditionally -- upstream
        hard-codes ``bias=False`` on those regardless of this flag.
    :type use_bias: bool
    :param input_channels: Channels of the input tensor. Defaults to 6: three
        RGB channels plus the three DTSPrompt channels the caller stacks on
        them. This default is the DocRes contract, not a suggestion -- there is
        no 3-channel DocRes.
    :type input_channels: int
    :param output_channels: Channels of the restored output. Defaults to 3 for
        every task, even those whose loss supervises fewer than three of them.
    :type output_channels: int
    :param kwargs: Additional keyword arguments for ``keras.Model``.
    :type kwargs: Any

    :raises ValueError: If any argument is non-positive, if ``num_blocks`` or
        ``heads`` does not have exactly four entries, or if a level's channel
        width is not divisible by that level's head count.

    Example:
        .. code-block:: python

            model = DocRes.from_variant("docres")
            restored = model(rgb_plus_prompt)   # (B, H, W, 6) -> (B, H, W, 3)
    """

    # A variant is a depth configuration, and both rows below are real: each is
    # the value some published artifact actually uses. Do not add a row that is
    # merely a plausible interpolation between them -- there is no evidence for
    # such a model and naming one would imply there is.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "docres": {
            "dim": 48,
            "num_blocks": [2, 3, 3, 4],
            "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8],
            "ffn_expansion_factor": 2.66,
            "use_bias": False,
            "input_channels": 6,
            "output_channels": 3,
            "description": (
                "The shipped DocRes configuration, identical at all three "
                "upstream call sites (train.py:86-97, inference.py:275-286, "
                "eval.py:272-283) -- arXiv:2405.04408. ~15.2M parameters."
            ),
        },
        "restormer_base": {
            "dim": 48,
            "num_blocks": [4, 6, 6, 8],
            "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8],
            "ffn_expansion_factor": 2.66,
            "use_bias": False,
            "input_channels": 6,
            "output_channels": 3,
            "description": (
                "The Restormer paper default and the upstream class default "
                "(restormer_arch.py:198), which every DocRes call site "
                "overrides -- arXiv:2111.09881. ~26.1M parameters. Provided "
                "for depth ablations; DocRes itself never uses it."
            ),
        },
    }

    def __init__(
            self,
            dim: int = 48,
            num_blocks: Optional[Sequence[int]] = None,
            num_refinement_blocks: int = 4,
            heads: Optional[Sequence[int]] = None,
            ffn_expansion_factor: float = 2.66,
            use_bias: bool = False,
            input_channels: int = 6,
            output_channels: int = 3,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer."""
        super().__init__(**kwargs)

        # DECISION plan-2026-09-08T111844-de235227/D-015 (asymmetry 1): the
        # default depth is [2, 3, 3, 4], NOT the Restormer paper's [4, 6, 6, 8]
        # that the upstream class carries as ITS default
        # (`restormer_arch.py:198`). Every DocRes call site overrides it to the
        # lighter schedule, and so does upstream's own self-test block
        # (`:284-300`), so [2, 3, 3, 4] is the real DocRes model and the class
        # default here is the shipped configuration rather than the inherited
        # one. The paper default is still reachable as the "restormer_base"
        # variant; do not restore it as the default to "match Restormer".
        num_blocks = [2, 3, 3, 4] if num_blocks is None else list(num_blocks)
        heads = [1, 2, 4, 8] if heads is None else list(heads)

        self._validate(
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            input_channels=input_channels,
            output_channels=output_channels,
        )

        self.dim = dim
        self.num_blocks = num_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.use_bias = use_bias
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Level widths: 48, 96, 192, 384 at dim=48.
        level_dims = [int(dim * 2 ** k) for k in range(4)]
        self._level_dims = level_dims

        self.patch_embed = keras.layers.Conv2D(
            filters=level_dims[0],
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="patch_embed",
        )

        self.encoder_level1 = self._make_stage(
            count=num_blocks[0], width=level_dims[0], num_heads=heads[0],
            prefix="encoder_level1",
        )
        self.down1_2 = RestormerDownsample(level_dims[0], name="down1_2")
        self.encoder_level2 = self._make_stage(
            count=num_blocks[1], width=level_dims[1], num_heads=heads[1],
            prefix="encoder_level2",
        )
        self.down2_3 = RestormerDownsample(level_dims[1], name="down2_3")
        self.encoder_level3 = self._make_stage(
            count=num_blocks[2], width=level_dims[2], num_heads=heads[2],
            prefix="encoder_level3",
        )
        self.down3_4 = RestormerDownsample(level_dims[2], name="down3_4")
        self.latent = self._make_stage(
            count=num_blocks[3], width=level_dims[3], num_heads=heads[3],
            prefix="latent",
        )

        self.up4_3 = RestormerUpsample(level_dims[3], name="up4_3")
        self.reduce_chan_level3 = keras.layers.Conv2D(
            filters=level_dims[2],
            kernel_size=1,
            use_bias=use_bias,
            name="reduce_chan_level3",
        )
        # DECISION plan-2026-09-08T111844-de235227/D-015 (asymmetry 6): the
        # decoder reuses its MATCHING ENCODER level's head count, not a mirrored
        # or monotonically decreasing schedule -- upstream
        # `restormer_arch.py:224,229,233` reads `heads[2]`, `heads[1]`,
        # `heads[0]` for decoder levels 3, 2, 1. Do not "correct" this into a
        # reversed sequence.
        self.decoder_level3 = self._make_stage(
            count=num_blocks[2], width=level_dims[2], num_heads=heads[2],
            prefix="decoder_level3",
        )

        self.up3_2 = RestormerUpsample(level_dims[2], name="up3_2")
        self.reduce_chan_level2 = keras.layers.Conv2D(
            filters=level_dims[1],
            kernel_size=1,
            use_bias=use_bias,
            name="reduce_chan_level2",
        )
        self.decoder_level2 = self._make_stage(
            count=num_blocks[1], width=level_dims[1], num_heads=heads[1],
            prefix="decoder_level2",
        )

        self.up2_1 = RestormerUpsample(level_dims[1], name="up2_1")

        # DECISION plan-2026-09-08T111844-de235227/D-015 (asymmetries 2 and 3):
        # there is deliberately NO `reduce_chan_level1`. Levels 3 and 2 follow
        # their concat with a 1x1 that halves the width back; level 1 does not
        # (upstream comment, `restormer_arch.py:231`: "NO 1x1 conv to reduce
        # channels"), so `decoder_level1` -- and the `refinement` stage after
        # it, and the `output` conv after that -- all run at `2 * dim` = 96
        # channels, DOUBLE the level-1 embedding width. Adding the missing
        # 1x1 for symmetry would change the width of the last third of the
        # network and make every parameter count wrong.
        self._decoder_level1_dim = level_dims[1]
        self.decoder_level1 = self._make_stage(
            count=num_blocks[0], width=level_dims[1], num_heads=heads[0],
            prefix="decoder_level1",
        )
        self.refinement = self._make_stage(
            count=num_refinement_blocks, width=level_dims[1],
            num_heads=heads[0], prefix="refinement",
        )

        # DECISION plan-2026-09-08T111844-de235227/D-015 (asymmetry 4):
        # `skip_conv` is built and applied UNCONDITIONALLY. Upstream guards its
        # construction behind `dual_pixel_task=True`
        # (`restormer_arch.py:238-241`) but uses it with no guard at all in
        # `forward` (`:277`), so a `dual_pixel_task=False` Restormer raises
        # `AttributeError` on its first forward pass -- the flag has exactly
        # one working setting. Every DocRes call site passes `True`. This port
        # therefore exposes no `dual_pixel_task` argument: a knob with one
        # legal value is not a knob, and reproducing the latent crash would be
        # reproducing a defect rather than a behaviour.
        self.skip_conv = keras.layers.Conv2D(
            filters=level_dims[1],
            kernel_size=1,
            use_bias=use_bias,
            name="skip_conv",
        )

        self.output_conv = keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="output_conv",
        )

        logger.info(
            f"Created DocRes: dim={dim}, num_blocks={num_blocks}, "
            f"refinement={num_refinement_blocks}, heads={heads}, "
            f"{input_channels}->{output_channels} channels"
        )

    @staticmethod
    def _validate(
            dim: int,
            num_blocks: List[int],
            num_refinement_blocks: int,
            heads: List[int],
            ffn_expansion_factor: float,
            input_channels: int,
            output_channels: int,
    ) -> None:
        """Reject an unbuildable configuration before any sub-layer exists.

        :raises ValueError: Naming the offending value in every case.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % 2 != 0:
            raise ValueError(
                f"dim must be even: the resamplers halve the channel count "
                f"at every level, got {dim}"
            )
        if len(num_blocks) != 4:
            raise ValueError(
                f"num_blocks must have exactly 4 entries, one per encoder "
                f"level, got {len(num_blocks)}: {num_blocks}"
            )
        if len(heads) != 4:
            raise ValueError(
                f"heads must have exactly 4 entries, one per level, got "
                f"{len(heads)}: {heads}"
            )
        if any(n <= 0 for n in num_blocks):
            raise ValueError(
                f"every entry of num_blocks must be positive, got {num_blocks}"
            )
        if any(h <= 0 for h in heads):
            raise ValueError(
                f"every entry of heads must be positive, got {heads}"
            )
        if num_refinement_blocks <= 0:
            raise ValueError(
                f"num_refinement_blocks must be positive, got "
                f"{num_refinement_blocks}"
            )
        if ffn_expansion_factor <= 0:
            raise ValueError(
                f"ffn_expansion_factor must be positive, got "
                f"{ffn_expansion_factor}"
            )
        if input_channels <= 0:
            raise ValueError(
                f"input_channels must be positive, got {input_channels}"
            )
        if output_channels <= 0:
            raise ValueError(
                f"output_channels must be positive, got {output_channels}"
            )

        level_dims = [int(dim * 2 ** k) for k in range(4)]
        for level, (width, num_heads) in enumerate(zip(level_dims, heads), 1):
            if width % num_heads != 0:
                raise ValueError(
                    f"level {level} width ({width}) is not divisible by its "
                    f"head count ({num_heads})"
                )
        # `decoder_level1` and `refinement` run at level-2 width with the
        # LEVEL-1 head count, a pairing no loop above covers.
        if level_dims[1] % heads[0] != 0:
            raise ValueError(
                f"the decoder level 1 / refinement width ({level_dims[1]}) is "
                f"not divisible by heads[0] ({heads[0]})"
            )

    def _make_stage(
            self,
            count: int,
            width: int,
            num_heads: int,
            prefix: str,
    ) -> List[RestormerTransformerBlock]:
        """Return ``count`` transformer blocks, each explicitly named.

        Explicit names are what makes the explicit-``build`` and lazy-``build``
        weight paths comparable: Keras auto-increments generated names per
        instance, so two separately-constructed models would otherwise differ
        at every unnamed level.

        :param count: Number of blocks in the stage.
        :type count: int
        :param width: Channel width every block in the stage operates at.
        :type width: int
        :param num_heads: MDTA head count for every block in the stage.
        :type num_heads: int
        :param prefix: Name prefix; block ``i`` is named ``f"{prefix}_{i}"``.
        :type prefix: str
        :return: The freshly created blocks, in application order.
        :rtype: List[RestormerTransformerBlock]
        """
        return [
            RestormerTransformerBlock(
                dim=width,
                num_heads=num_heads,
                ffn_expansion_factor=self.ffn_expansion_factor,
                use_bias=self.use_bias,
                name=f"{prefix}_{index}",
            )
            for index in range(count)
        ]

    def _check_spatial_divisibility(
            self,
            height: Optional[int],
            width: Optional[int],
    ) -> None:
        """Raise unless both known spatial extents are divisible by 8.

        # DECISION plan-2026-09-08T111844-de235227/D-016: this model REFUSES a
        # non-divisible input rather than padding internally. The three
        # downsampling stages each halve the resolution, so an odd extent
        # anywhere in the chain silently loses a row or column of the encoder
        # feature it is later concatenated with -- and the symptom surfaces
        # deep inside a pixel-unshuffle as a shape mismatch that names none of
        # this. Padding is the caller's contract because the padding has to be
        # REMOVED from the prediction afterwards, which this model cannot do
        # for a caller who was never told it happened; upstream pads in its
        # inference script for exactly that reason, and so does step 13's
        # inference shim here. Do not add auto-padding to this class.

        ``None`` extents are skipped: a symbolic build at
        ``(None, None, None, 6)`` is legitimate and is how the model is
        normally built.

        :param height: Static height, or ``None`` if unknown.
        :type height: Optional[int]
        :param width: Static width, or ``None`` if unknown.
        :type width: Optional[int]
        :raises ValueError: If a known extent is not a multiple of 8.
        """
        offenders = [
            (axis, extent)
            for axis, extent in (("height", height), ("width", width))
            if extent is not None and extent % SPATIAL_DIVISOR != 0
        ]
        if offenders:
            described = ", ".join(f"{axis}={extent}" for axis, extent in offenders)
            raise ValueError(
                f"DocRes requires spatial dimensions divisible by "
                f"{SPATIAL_DIVISOR} (three downsampling stages), got "
                f"{described}. Pad the input to a multiple of "
                f"{SPATIAL_DIVISOR} before calling and crop the prediction "
                f"back afterwards; this model does not pad internally."
            )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer by tracing ``call`` symbolically.

        Without this, a subclassed model inherits ``Layer.build``, which marks
        the model built while every sub-layer is still unbuilt -- so a
        ``.keras`` reload would restore nothing and the first forward pass
        would create fresh random weights with nothing raising.

        :param input_shape: Shape of the input to ``call``, normally
            ``(None, None, None, input_channels)``.
        :type input_shape: Any
        :raises ValueError: If the trailing dimension is a known value other
            than ``input_channels``, or a known spatial extent is not divisible
            by 8.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        if (
                input_shape[-1] is not None
                and input_shape[-1] != self.input_channels
        ):
            raise ValueError(
                f"Expected {self.input_channels} input channels (RGB plus the "
                f"DTSPrompt), got {input_shape[-1]}"
            )
        self._check_spatial_divisibility(input_shape[1], input_shape[2])

        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Restore a document image from its RGB-plus-prompt stack.

        :param inputs: Tensor of shape
            ``(batch, height, width, input_channels)``. Height and width must
            be divisible by 8.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag.
        :type training: Optional[bool]
        :return: Tensor of shape
            ``(batch, height, width, output_channels)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If a statically-known spatial extent is not
            divisible by 8. Eager calls always know their extents, so this
            fires there on every call, not only on the first.
        """
        static_shape = inputs.shape
        self._check_spatial_divisibility(static_shape[1], static_shape[2])

        enc1_in = self.patch_embed(inputs)

        x = enc1_in
        for block in self.encoder_level1:
            x = block(x, training=training)
        out_enc1 = x

        x = self.down1_2(out_enc1, training=training)
        for block in self.encoder_level2:
            x = block(x, training=training)
        out_enc2 = x

        x = self.down2_3(out_enc2, training=training)
        for block in self.encoder_level3:
            x = block(x, training=training)
        out_enc3 = x

        x = self.down3_4(out_enc3, training=training)
        for block in self.latent:
            x = block(x, training=training)

        x = self.up4_3(x, training=training)
        x = keras.ops.concatenate([x, out_enc3], axis=-1)
        x = self.reduce_chan_level3(x)
        for block in self.decoder_level3:
            x = block(x, training=training)

        x = self.up3_2(x, training=training)
        x = keras.ops.concatenate([x, out_enc2], axis=-1)
        x = self.reduce_chan_level2(x)
        for block in self.decoder_level2:
            x = block(x, training=training)

        x = self.up2_1(x, training=training)
        # No reduce_chan_level1 here, by construction -- see the D-015
        # asymmetry-2 comment in `__init__`. The concatenated 2*dim width is
        # carried all the way to `output_conv`.
        x = keras.ops.concatenate([x, out_enc1], axis=-1)
        for block in self.decoder_level1:
            x = block(x, training=training)
        for block in self.refinement:
            x = block(x, training=training)

        x = x + self.skip_conv(enc1_in)

        # DECISION plan-2026-09-08T111844-de235227/D-015 (asymmetry 5): the
        # result of `output_conv` is returned DIRECTLY. There is no
        # `+ inputs[..., :3]` residual, unlike the usual image-restoration
        # convention -- upstream `forward` ends at
        # `return self.output(out_dec_level1)` (`restormer_arch.py:278-280`)
        # with no input add anywhere. Several of DocRes's tasks (binarization
        # emits logits, dewarping emits a coordinate field) are not
        # perturbations of their input at all, so an input residual would be
        # actively wrong for them, not merely unfaithful.
        return self.output_conv(x)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the input shape with the channel axis replaced.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: ``(batch, height, width, output_channels)``.
        :rtype: tuple
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        return (batch, height, width, self.output_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument needed to recreate this model.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_blocks": self.num_blocks,
            "num_refinement_blocks": self.num_refinement_blocks,
            "heads": self.heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "use_bias": self.use_bias,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
        })
        return config

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: bool = False,
            **kwargs: Any
    ) -> "DocRes":
        """Create a DocRes model from one of the two cited configurations.

        :param variant: A key of :attr:`MODEL_VARIANTS`: ``"docres"`` or
            ``"restormer_base"``.
        :type variant: str
        :param pretrained: Must be ``False``. No DocRes weights are
            distributed with this library.
        :type pretrained: bool
        :param kwargs: Constructor overrides applied on top of the variant's
            configuration.
        :type kwargs: Any
        :return: The constructed, unbuilt model.
        :rtype: DocRes
        :raises ValueError: If ``variant`` is not a known key; the message
            lists the available ones.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown DocRes variant '{variant}'. Available variants: "
                f"{sorted(cls.MODEL_VARIANTS.keys())}"
            )

        if pretrained:
            raise NotImplementedError(
                f"No pretrained weights are distributed for DocRes variant "
                f"'{variant}'. Train from scratch with "
                f"`src/train/doc_res/`, or restore your own checkpoint with "
                f"`model.load_weights(path)`. The upstream PyTorch checkpoint "
                f"cannot be converted: this port's pixel-shuffle layers use a "
                f"different channel-block ordering (see the D-007 anchors in "
                f"components.py)."
            )

        config = dict(cls.MODEL_VARIANTS[variant])
        config.pop("description", None)
        config.update(kwargs)
        return cls(**config)


# ---------------------------------------------------------------------


def create_doc_res(
        variant: str = "docres",
        pretrained: bool = False,
        **kwargs: Any
) -> DocRes:
    """Create a DocRes model. The module-level entry point.

    :param variant: A key of :attr:`DocRes.MODEL_VARIANTS`. Defaults to
        ``"docres"``.
    :type variant: str
    :param pretrained: Must be ``False``; see :meth:`DocRes.from_variant`.
    :type pretrained: bool
    :param kwargs: Constructor overrides forwarded unchanged.
    :type kwargs: Any
    :return: The constructed, unbuilt model.
    :rtype: DocRes
    """
    return DocRes.from_variant(variant, pretrained=pretrained, **kwargs)
