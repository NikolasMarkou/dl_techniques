"""The U2NET-P building blocks of DocScanner's segmentation stage.

DocScanner's first stage is a *pruned* U2-Net (``U2NETP``, ``seg.py:451-543``)
that predicts a document/background confidence map at 288x288. Its only
structural ingredient is the **ReSidual U-block** (RSU): a small U-Net that
lives inside one stage of a larger U-Net, which is what "nested U-structure"
means in the U2-Net paper. This module carries those blocks and nothing else --
the stage ladder that wires eleven of them together is :mod:`.model`'s job.

What is here
------------
* :class:`REBNCONV` -- the atom: ``Conv2D 3x3 (dilated) -> BatchNorm -> ReLU``.
* :class:`RSU7`, :class:`RSU6`, :class:`RSU5`, :class:`RSU4` -- pooling ladders
  with 6, 5, 4 and 3 encoder levels respectively. All four share the private
  base :class:`_ResidualUBlock`; they differ ONLY in the level count.
* :class:`RSU4F` -- the "flat" variant: no pooling at all, constant spatial
  resolution, receptive field grown by dilation ``1/2/4/8`` instead.

Three torch-vs-Keras defaults that are load-bearing here
--------------------------------------------------------
Each of the three was MEASURED against a hand-written reference rather than
assumed, because all three are silent: the wrong choice produces the right
shapes, the right dtype and a model that trains.

1. **``MaxPool2d(2, stride=2, ceil_mode=True)``** has no Keras equivalent knob.
   ``padding="same"`` reproduces it exactly -- see the anchor at
   :data:`_POOL_PADDING`.
2. **``BatchNorm2d``'s epsilon and momentum** differ from Keras' defaults by
   100x and by a reversed convention -- see :data:`BATCH_NORM_EPSILON`.
3. **A dilated 3x3 at stride 1**: torch's ``padding=dirate, dilation=dirate``
   IS Keras' ``padding="same", dilation_rate=dirate``. Measured at
   ``dirate in {1, 2, 4, 8}`` on ``11x9`` and ``37x53`` inputs against a
   hand-written ``F.conv2d`` reference: max absolute difference ``5.5e-7``,
   i.e. float32 rounding. (This is NOT contradicted by D-012, which found
   Keras' ``"same"`` to disagree with torch at stride **2**; that asymmetry is
   a stride phenomenon, not a padding-mode one, and every convolution in this
   module is stride 1.)

The resampling convention here is the OPPOSITE of ``warp.py``'s
---------------------------------------------------------------
``_upsample_like`` uses half-pixel-centre (``align_corners=False``) bilinear
resizing. :mod:`.warp`'s ``sample_at_pixel_coords`` uses ``align_corners=True``.
That is not an inconsistency to tidy up -- see the anchor above
:func:`_upsample_like`.

Registration convention
-----------------------
Classes here register under ``dl_techniques.models.doc_scanner.u2net_blocks``
-- the key strips BOTH the ``vision`` family and the ``image_restoration``
subfamily, per the repo-wide convention. It is NOT the full import path.

A deliberate non-port: weight initialization
--------------------------------------------
``seg.py`` never re-initializes its convolutions, so upstream they carry
torch's ``Conv2d`` default (``kaiming_uniform_(a=sqrt(5))``) and Keras' carry
``glorot_uniform``. That divergence is left in place: this port ships no
transferred checkpoint (``pretrained=True`` raises), the two initializers have
the same order of magnitude, and inventing a torch-default initializer here
would add a knob nobody upstream chose. Contrast ``components.py``'s
``_kaiming_fan_out``, which exists because ``extractor.py:106-108``
EXPLICITLY re-initializes -- i.e. there the initializer is a stated choice, and
here it is not.

References:
    - Qin et al., 2020. U2-Net: Going Deeper with Nested U-Structure for
      Salient Object Detection. Pattern Recognition.
      (https://arxiv.org/abs/2005.09007)
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2.
    - Upstream release: https://github.com/fh2019ustc/DocScanner -- ``seg.py``.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants.
#
# As everywhere in this port, the citation beside a value is the line that USES
# it, never a constructor default no call site reaches (D-006).
# ---------------------------------------------------------------------

# DECISION plan-2026-09-10T065432-05fcb6dd/D-021: `nn.BatchNorm2d(out_ch)`
# (`seg.py:40`) is constructed with torch's defaults, which are `eps=1e-5` and
# `momentum=0.1`. Keras' `BatchNormalization` defaults are `epsilon=1e-3` (100x
# larger) and `momentum=0.99`. The two `momentum` parameters also mean OPPOSITE
# things: torch's is the weight of the NEW batch statistic, Keras' is the weight
# of the RETAINED running statistic, so torch's 0.1 is Keras' 0.9 -- copying the
# number across would be a 10x error in the other direction on top of the
# default mismatch. Both are passed explicitly at every construction site below.
# Neither has any shape, dtype or finiteness symptom; the epsilon shows up only
# as a systematic activation-scale bias and the momentum only as a different
# inference-time running mean after training. See decisions.md D-021.
BATCH_NORM_EPSILON: float = 1e-5

#: Keras' retained-statistic weight equal to torch's ``momentum=0.1``
#: new-statistic weight. See :data:`BATCH_NORM_EPSILON`'s anchor.
BATCH_NORM_MOMENTUM: float = 0.9

# DECISION plan-2026-09-10T065432-05fcb6dd/D-019: this is the `ceil_mode=True`
# remedy. `seg.py` pools with `nn.MaxPool2d(2, stride=2, ceil_mode=True)`
# (`:65`, `:141`, ... eleven sites) and Keras' `MaxPooling2D` has NO `ceil_mode`
# knob. Do NOT use `padding="valid"`: at an ODD input size torch's ceil_mode
# rounds the output UP (9 -> 5) and "valid" rounds it DOWN (9 -> 4), so the
# decoder's `_upsample_like` would silently resample a half-size-off tensor
# and the block would still return the right OUTER shape, because
# `_upsample_like` targets the skip and hides the discrepancy.
#
# MEASURED, not assumed: `padding="same"` matches torch ceil_mode EXACTLY --
# same output size and the same SELECTED ELEMENTS -- at (H, W) in
# {(9, 7), (8, 6), (5, 11), (37, 53)} against a hand-written reference that
# implements torch's clamp-the-ragged-window rule. The probe fed an
# all-NEGATIVE input on purpose: TF pads max-pooling with -inf, so a
# hypothetical ZERO-padded implementation would have won every ragged window
# with a padded 0 and been caught. It was not caught, i.e. no padded value
# participates. See decisions.md D-019 and the `TestTheCeilModeRemedy` guards.
_POOL_PADDING: str = "same"

#: The pooling window and stride, ``nn.MaxPool2d(2, stride=2, ...)``.
_POOL_SIZE: int = 2

#: ``REBNCONV``'s kernel: ``nn.Conv2d(in_ch, out_ch, 3, ...)``, ``seg.py:38``.
_CONV_KERNEL_SIZE: int = 3

#: The dilation of the innermost ("bottom") convolution of every pooling
#: ladder: ``rebnconv7``/``rebnconv6``/``rebnconv5``/``rebnconv4`` are built
#: with ``dirate=2`` (``seg.py:81``, ``:154``, ``:219``, ``:272``) where every
#: other convolution in the ladder uses ``dirate=1``. It is what replaces a
#: further pooling level at the bottom of the U.
_LADDER_BOTTOM_DILATION: int = 2

#: ``RSU4F``'s encoder dilations, ``seg.py:317-322``: ``rebnconv1..4`` at
#: ``dirate=1, 2, 4, 8``. There is no pooling, so this doubling IS the
#: block's entire receptive-field growth.
_FLAT_ENCODER_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8)

#: ``RSU4F``'s decoder dilations, ``seg.py:324-326``: ``rebnconv3d`` at 4,
#: ``rebnconv2d`` at 2, ``rebnconv1d`` at 1. Written innermost-first so it
#: indexes the same way as the encoder tuple above.
_FLAT_DECODER_DILATIONS: Tuple[int, ...] = (1, 2, 4)

#: The width multiplier on every decoder convolution's INPUT: each decoder
#: consumes ``concat(upsampled_deeper_decoder, matching_encoder_skip)`` and
#: both halves are ``mid_channels`` wide -- ``REBNCONV(mid_ch * 2, mid_ch)``,
#: ``seg.py:83-88``.
_DECODER_CONCAT_FACTOR: int = 2


# DECISION plan-2026-09-10T065432-05fcb6dd/D-020: this resize is HALF-PIXEL
# (`align_corners=False`), which is the OPPOSITE convention from the one
# `warp.py::sample_at_pixel_coords` implements (`align_corners=True`). That is
# deliberate and it is what the two upstream call sites ask for:
# `seg.py:51` is `F.interpolate(..., mode='bilinear', align_corners=False)`,
# while `model.py`'s rectifier stage samples through `F.grid_sample(...,
# align_corners=True)`. Do NOT "unify" them, and in particular do NOT reuse
# `sample_at_pixel_coords` here to save a function: the two stages are trained
# INDEPENDENTLY (paper §4.3) and each one's geometry is fixed by its own
# reference. A package containing both conventions is exactly the kind of thing
# a later reader tidies into a single helper, so it is stated here rather than
# left to be inferred.
#
# MEASURED: `keras.ops.image.resize(..., interpolation="bilinear")` reproduces
# torch's `F.interpolate(mode='bilinear', align_corners=False)` to float32
# rounding (max abs diff 6.6e-6 over 3x4->5x7, 5x3->10x6, 2x2->5x5,
# 19x27->37x53) against a hand-written half-pixel reference. See decisions.md
# D-020.
def _upsample_like(
        source: keras.KerasTensor,
        target: keras.KerasTensor,
) -> keras.KerasTensor:
    """Bilinearly resize ``source`` to ``target``'s spatial size.

    The port of ``seg.py:50-53``'s ``_upsample_like``. It exists because the
    ``ceil_mode`` pooling above can produce a ragged ladder -- at an input
    height of 37 the levels run ``37, 19, 10, 5, 3, 2`` -- so a decoder tensor
    is NOT always exactly twice the size of the skip it must be concatenated
    with. Resizing to the skip's measured size, rather than upsampling by a
    factor of 2, is what makes odd input sizes work at all.

    :param source: The tensor to resize, ``(batch, h, w, channels)``.
    :type source: keras.KerasTensor
    :param target: The tensor whose spatial size is wanted. Only its shape is
        read; its values are untouched.
    :type target: keras.KerasTensor
    :return: ``source`` resampled to ``(batch, target_h, target_w, channels)``.
    :rtype: keras.KerasTensor
    """
    static_height, static_width = target.shape[1], target.shape[2]
    if static_height is None or static_width is None:
        # A fully dynamic spatial axis. `keras.ops.shape` returns scalar
        # tensors on the TF backend, which `resize` accepts; the static branch
        # is preferred whenever it is available because it keeps the output
        # spec statically known for `compute_output_shape`.
        dynamic_shape = keras.ops.shape(target)
        size = (dynamic_shape[1], dynamic_shape[2])
    else:
        size = (static_height, static_width)

    return keras.ops.image.resize(
        source,
        size,
        interpolation="bilinear",
        data_format="channels_last",
    )


# ---------------------------------------------------------------------
# The atom.
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_scanner.u2net_blocks")
class REBNCONV(keras.layers.Layer):
    """``Conv2D 3x3 (dilated) -> BatchNormalization -> ReLU`` (``seg.py:34-46``).

    The name is upstream's and is kept verbatim (RE-lu, B-atch N-orm, CONV) so
    that a reader diffing this file against ``seg.py`` sees the same identifier
    on both sides. It is the ONLY parameterized operation in the whole
    segmentation stage: every RSU block is a wiring of these.

    The convolution is stride 1 with ``padding="same"``, which is exactly
    torch's ``padding=dirate, dilation=dirate`` -- see this module's docstring
    for the measurement. Spatial size is therefore preserved for any dilation.

    :param filters: Output channel count (upstream's ``out_ch``).
    :type filters: int
    :param dilation_rate: Upstream's ``dirate``. ``1`` everywhere except the
        bottom of a pooling ladder (``2``) and inside :class:`RSU4F`
        (``1/2/4/8``).
    :type dilation_rate: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If ``filters`` or ``dilation_rate`` is not positive.
    """

    def __init__(
            self,
            filters: int,
            dilation_rate: int = 1,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if dilation_rate <= 0:
            raise ValueError(
                f"dilation_rate must be positive, got {dilation_rate}"
            )

        self.filters = filters
        self.dilation_rate = dilation_rate

        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=_CONV_KERNEL_SIZE,
            strides=1,
            padding="same",
            dilation_rate=dilation_rate,
            use_bias=True,
            name="conv",
        )
        self.norm = keras.layers.BatchNormalization(
            axis=-1,
            epsilon=BATCH_NORM_EPSILON,
            momentum=BATCH_NORM_MOMENTUM,
            name="norm",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build both sub-layers explicitly, propagating the shape by hand.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :raises ValueError: If the input is not rank 4.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"REBNCONV expects a 4D input shape "
                f"(batch, height, width, channels), got {len(input_shape)}D: "
                f"{input_shape}"
            )

        self.conv.build(tuple(input_shape))
        self.norm.build(self.conv.compute_output_shape(tuple(input_shape)))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Convolve, normalize, rectify.

        :param inputs: ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Passed to the batch normalization layer, which is the
            only training-mode-dependent operation in this module.
        :type training: Optional[bool]
        :return: ``(batch, height, width, filters)``.
        :rtype: keras.KerasTensor
        """
        outputs = self.conv(inputs)
        outputs = self.norm(outputs, training=training)
        return keras.ops.relu(outputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Spatial size is preserved; only the channel axis changes.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :return: ``(batch, height, width, filters)``.
        :rtype: tuple
        """
        return (*tuple(input_shape)[:-1], self.filters)

    def get_config(self) -> Dict[str, Any]:
        """Every constructor argument, for a lossless round trip.

        :return: The serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config


# ---------------------------------------------------------------------
# The pooling ladders: RSU7 / RSU6 / RSU5 / RSU4.
# ---------------------------------------------------------------------


class _ResidualUBlock(keras.layers.Layer):
    """Shared implementation of the four POOLING ReSidual U-blocks.

    ``seg.py:57-307`` spells RSU7, RSU6, RSU5 and RSU4 out four times. They are
    the same block: the only difference is the number of encoder levels (6, 5,
    4, 3), and therefore the number of pools (one fewer) and of decoder
    convolutions (the same as the level count). Rather than transcribe four
    near-identical 60-line ``forward`` methods -- four places for a
    concatenation order or an off-by-one to diverge -- the wiring lives here
    once and each public class supplies its level count.

    Interface contract for the four subclasses
    ------------------------------------------
    A subclass sets exactly one thing, the class attribute
    ``_ENCODER_LEVELS: int`` (``>= 2``), and adds the registration decorator
    and a docstring. It inherits ``__init__``, ``build``, ``call``,
    ``compute_output_shape`` and ``get_config`` unchanged. ``get_config``
    serializes ``mid_channels`` and ``out_channels`` only; the level count is
    part of the CLASS, not of the config, exactly as upstream makes it part of
    the class name -- so a reloaded ``RSU6`` can never come back as an ``RSU5``.

    The topology, for ``N = _ENCODER_LEVELS``
    -----------------------------------------
    .. code-block:: text

        hxin = rebnconvin(x)                  in_ch -> out_ch, dirate 1
        hx1  = enc[0](hxin)                   out_ch -> mid,   dirate 1
        hx2  = enc[1](pool(hx1))              mid -> mid,      dirate 1
        ...                                   (N-1 pools in total)
        hxN  = enc[N-1](pool(hx{N-1}))
        bottom = bottom_conv(hxN)             mid -> mid,      dirate 2
        d = dec[N-1](concat([bottom, hxN]))   2*mid -> mid   (NO upsample here:
                                                 the bottom conv does not pool)
        d = dec[i](concat([up(d, hx{i+1}), hx{i+1}]))   for i = N-2 .. 1
        d = dec[0](concat([up(d, hx1), hx1])) 2*mid -> out_ch
        return d + hxin                       the RESIDUAL that makes it "RSU"

    Two things here are shape-preserving and therefore invisible to a shape
    test, which is why both carry their own guard:

    * **The concatenation order is ``(deeper, skip)``**, i.e. the upsampled
      decoder tensor FIRST -- ``torch.cat((hx6dup, hx5), 1)``, ``seg.py:117``.
      Both halves are ``mid_channels`` wide, so swapping them changes nothing
      about any shape.
    * **The final residual add is present.** ``return hx1d + hxin``,
      ``seg.py:129``. Dropping it leaves a block that still maps
      ``(B, H, W, in)`` to ``(B, H, W, out)``.

    :param mid_channels: Upstream's ``mid_ch`` -- the width of the ladder's
        interior. Read from ``_VARIANT_SPEC["seg_mid_channels"]``, never as a
        literal.
    :type mid_channels: int
    :param out_channels: Upstream's ``out_ch`` -- the width of the outer
        projection and therefore of the block's output.
    :type out_channels: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If either width is not positive.
    """

    #: Overridden by every concrete subclass. The base is never instantiated.
    _ENCODER_LEVELS: int = 0

    def __init__(
            self,
            mid_channels: int,
            out_channels: int,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if self._ENCODER_LEVELS < 2:
            raise ValueError(
                f"{type(self).__name__}._ENCODER_LEVELS must be at least 2, "
                f"got {self._ENCODER_LEVELS}. _ResidualUBlock is an abstract "
                f"base and is not instantiated directly; use RSU7/6/5/4."
            )
        if mid_channels <= 0:
            raise ValueError(
                f"mid_channels must be positive, got {mid_channels}"
            )
        if out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive, got {out_channels}"
            )

        self.mid_channels = mid_channels
        self.out_channels = out_channels

        # The outer projection, `in_ch -> out_ch`. Its output is BOTH the input
        # of the ladder and the residual added back at the very end, which is
        # why the block's output width is `out_channels` and not `mid_channels`.
        self.rebnconvin = REBNCONV(out_channels, dilation_rate=1, name="rebnconvin")

        self._create_ladder_sublayers()

    def _create_ladder_sublayers(self) -> None:
        """Construct everything BETWEEN the stem and the residual add.

        Overridden by :class:`RSU4F`, which has a genuinely different interior.
        It is a hook rather than a create-then-replace in ``RSU4F.__init__``
        because reassigning a tracked sub-layer attribute is not a documented
        way to untrack the first one: the discarded layers would still allocate
        weights, still appear in ``model.weights``, and the gradient-flow
        oracle would then correctly report weights no gradient reaches.
        """
        mid_channels = self.mid_channels
        out_channels = self.out_channels
        levels = self._ENCODER_LEVELS

        # Encoder: `enc[0]` maps out_ch -> mid, the rest mid -> mid.
        self.encoder_convs: List[REBNCONV] = [
            REBNCONV(mid_channels, dilation_rate=1, name=f"rebnconv{i + 1}")
            for i in range(levels)
        ]
        # One fewer pool than levels: `seg.py` pools AFTER every encoder
        # convolution except the last (`rebnconv6` in RSU7 has no `pool6`).
        self.pools: List[keras.layers.MaxPooling2D] = [
            keras.layers.MaxPooling2D(
                pool_size=_POOL_SIZE,
                strides=_POOL_SIZE,
                padding=_POOL_PADDING,
                name=f"pool{i + 1}",
            )
            for i in range(levels - 1)
        ]

        # The bottom of the U: a dilation-2 convolution INSTEAD of one more
        # pooling level, applied to the last encoder output at its own
        # resolution.
        self.bottom_conv = REBNCONV(
            mid_channels,
            dilation_rate=_LADDER_BOTTOM_DILATION,
            name=f"rebnconv{levels + 1}",
        )

        # Decoder: `dec[i]` is upstream's `rebnconv{i+1}d`. All consume
        # `2 * mid_channels`; only `dec[0]` emits `out_channels`.
        self.decoder_convs: List[REBNCONV] = [
            REBNCONV(
                out_channels if i == 0 else mid_channels,
                dilation_rate=1,
                name=f"rebnconv{i + 1}d",
            )
            for i in range(levels)
        ]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer, propagating shapes down and back up the U.

        Shapes are propagated by ASKING each sub-layer
        (``compute_output_shape``) rather than by recomputing ``ceil(n / 2)``
        here: the pooling rule is the one thing in this module most likely to
        be got wrong, and stating it twice would mean two places to get it
        wrong.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :raises ValueError: If the input is not rank 4.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"{type(self).__name__} expects a 4D input shape "
                f"(batch, height, width, channels), got {len(input_shape)}D: "
                f"{input_shape}"
            )

        shape = tuple(input_shape)
        self.rebnconvin.build(shape)
        self._build_ladder_sublayers(self.rebnconvin.compute_output_shape(shape))

        super().build(input_shape)

    def _build_ladder_sublayers(
            self,
            stem_shape: Tuple[Optional[int], ...],
    ) -> None:
        """Build everything between the stem and the residual add.

        Overridden by :class:`RSU4F`. Shapes are propagated by ASKING each
        sub-layer (``compute_output_shape``) rather than by recomputing
        ``ceil(n / 2)`` here: the pooling rule is the one thing in this module
        most likely to be got wrong, and stating it twice would mean two places
        to get it wrong.

        :param stem_shape: The shape of ``rebnconvin``'s output.
        :type stem_shape: tuple
        """
        # Down the ladder, recording the shape of each skip.
        skip_shapes: List[Tuple[Optional[int], ...]] = []
        level_shape = stem_shape
        for index, conv in enumerate(self.encoder_convs):
            conv.build(level_shape)
            level_shape = conv.compute_output_shape(level_shape)
            skip_shapes.append(level_shape)
            if index < len(self.pools):
                self.pools[index].build(level_shape)
                level_shape = self.pools[index].compute_output_shape(level_shape)

        self.bottom_conv.build(skip_shapes[-1])
        deeper_shape = self.bottom_conv.compute_output_shape(skip_shapes[-1])

        # Back up. The concatenated width is the deeper tensor's plus the
        # skip's; the spatial size is always the SKIP's, because the deeper
        # tensor is resized onto it.
        for index in range(len(self.decoder_convs) - 1, -1, -1):
            skip_shape = skip_shapes[index]
            concat_shape = (
                *skip_shape[:-1],
                deeper_shape[-1] + skip_shape[-1],
            )
            self.decoder_convs[index].build(concat_shape)
            deeper_shape = self.decoder_convs[index].compute_output_shape(
                concat_shape
            )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the nested U and add the outer residual.

        :param inputs: ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to every :class:`REBNCONV`.
        :type training: Optional[bool]
        :return: ``(batch, height, width, out_channels)`` -- the SAME spatial
            size as the input, whatever the pooling ladder did in between.
        :rtype: keras.KerasTensor
        """
        stem = self.rebnconvin(inputs, training=training)

        # The RESIDUAL. `seg.py:129`: `return hx1d + hxin`.
        return self._call_ladder(stem, training=training) + stem

    def _call_ladder(
            self,
            stem: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run everything between the stem and the residual add.

        Overridden by :class:`RSU4F`.

        :param stem: ``rebnconvin``'s output, ``(batch, h, w, out_channels)``.
        :type stem: keras.KerasTensor
        :param training: Forwarded to every :class:`REBNCONV`.
        :type training: Optional[bool]
        :return: The last decoder's output, ``(batch, h, w, out_channels)``.
        :rtype: keras.KerasTensor
        """
        skips: List[keras.KerasTensor] = []
        level = stem
        for index, conv in enumerate(self.encoder_convs):
            level = conv(level, training=training)
            skips.append(level)
            if index < len(self.pools):
                level = self.pools[index](level)

        deeper = self.bottom_conv(skips[-1], training=training)

        for index in range(len(self.decoder_convs) - 1, -1, -1):
            skip = skips[index]
            if index != len(self.decoder_convs) - 1:
                # The innermost decoder is skipped by this branch on purpose:
                # `bottom_conv` does not pool, so `deeper` is already at
                # `skips[-1]`'s resolution and upstream concatenates it with no
                # `_upsample_like` at all (`seg.py:115`).
                deeper = _upsample_like(deeper, skip)
            # ORDER: deeper FIRST, skip second -- `torch.cat((hx6dup, hx5), 1)`.
            merged = keras.ops.concatenate([deeper, skip], axis=-1)
            deeper = self.decoder_convs[index](merged, training=training)

        return deeper

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Spatial size in, spatial size out; the channel axis becomes
        ``out_channels``.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :return: ``(batch, height, width, out_channels)``.
        :rtype: tuple
        """
        return (*tuple(input_shape)[:-1], self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        """The two widths. The level count is part of the CLASS.

        :return: The serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "mid_channels": self.mid_channels,
                "out_channels": self.out_channels,
            }
        )
        return config


@register_dl_technique("dl_techniques.models.doc_scanner.u2net_blocks")
class RSU7(_ResidualUBlock):
    """A 6-level ReSidual U-block with 5 pools (``seg.py:57-129``).

    The outermost stage of the U2NET-P encoder and of its decoder
    (``stage1``/``stage1d``), where the spatial resolution is highest and the
    ladder therefore has the most room to descend.

    See :class:`_ResidualUBlock` for the topology, the concatenation order and
    the residual add. Constructor arguments are identical.
    """

    _ENCODER_LEVELS: int = 6


@register_dl_technique("dl_techniques.models.doc_scanner.u2net_blocks")
class RSU6(_ResidualUBlock):
    """A 5-level ReSidual U-block with 4 pools (``seg.py:133-198``).

    ``stage2``/``stage2d`` of the U2NET-P.

    See :class:`_ResidualUBlock` for the topology, the concatenation order and
    the residual add. Constructor arguments are identical.
    """

    _ENCODER_LEVELS: int = 5


@register_dl_technique("dl_techniques.models.doc_scanner.u2net_blocks")
class RSU5(_ResidualUBlock):
    """A 4-level ReSidual U-block with 3 pools (``seg.py:202-254``).

    ``stage3``/``stage3d`` of the U2NET-P.

    See :class:`_ResidualUBlock` for the topology, the concatenation order and
    the residual add. Constructor arguments are identical.
    """

    _ENCODER_LEVELS: int = 4


@register_dl_technique("dl_techniques.models.doc_scanner.u2net_blocks")
class RSU4(_ResidualUBlock):
    """A 3-level ReSidual U-block with 2 pools (``seg.py:258-307``).

    ``stage4``/``stage4d`` of the U2NET-P -- the shallowest ladder before the
    architecture switches to the pooling-free :class:`RSU4F`.

    See :class:`_ResidualUBlock` for the topology, the concatenation order and
    the residual add. Constructor arguments are identical.
    """

    _ENCODER_LEVELS: int = 3


# ---------------------------------------------------------------------
# The flat ladder: RSU4F.
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_scanner.u2net_blocks")
class RSU4F(_ResidualUBlock):
    """The pooling-FREE ReSidual U-block (``seg.py:308-343``).

    ``stage5``, ``stage6`` and ``stage5d`` of the U2NET-P. By the time the
    U2NET-P's outer encoder reaches these stages the feature map is already
    1/16 and 1/32 of the input, and pooling further would leave a handful of
    pixels; so RSU4F replaces the whole pooling ladder with DILATION. The
    encoder runs at ``dirate 1, 2, 4, 8`` and the decoder back down at
    ``4, 2, 1``, the spatial resolution never changes, and there is no
    ``_upsample_like`` anywhere in the block.

    It shares :class:`_ResidualUBlock`'s constructor, ``compute_output_shape``
    and ``get_config`` -- the widths and the outer residual are identical --
    but overrides the wiring, which genuinely differs. Note the subclass sets
    ``_ENCODER_LEVELS`` even though it owns no pools: the base's constructor
    validates it, and it is the honest level count (4 convolutions, the last of
    which is the "bottom" in the ladder's sense).

    Guarding the "no pooling" claim on structure alone (no ``MaxPooling2D``
    sub-layer exists) is necessary but not sufficient -- a wrong dilation
    schedule would also leave no pooling layer. The behavioural half of that
    guard measures the influence spread of an impulse and compares it against
    the reach a ``1/2/4/8`` dilation cascade actually has.

    :param mid_channels: Upstream's ``mid_ch``.
    :type mid_channels: int
    :param out_channels: Upstream's ``out_ch``.
    :type out_channels: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    """

    _ENCODER_LEVELS: int = 4

    def _create_ladder_sublayers(self) -> None:
        """Build the flat cascade instead of the base's pooling ladder.

        Note what is NOT created: no ``MaxPooling2D`` and no separate "bottom"
        convolution. ``rebnconv4``, the dilation-8 encoder convolution, IS the
        bottom (``seg.py:321``), and it is the last encoder output that the
        innermost decoder concatenates with its own skip. Creating the base's
        ``bottom_conv`` and then leaving it unused would still allocate
        weights, still list them in ``layer.weights``, and the gradient-flow
        oracle would then correctly report weights no gradient reaches.
        """
        mid_channels = self.mid_channels
        out_channels = self.out_channels

        # Stated, not merely absent: the two attributes the base would fill are
        # bound to their empty values so every block in this module answers the
        # same questions, and so "RSU4F has no pooling" is a readable fact at
        # the construction site rather than an inference from a missing name.
        self.pools: List[keras.layers.MaxPooling2D] = []
        self.bottom_conv: Optional[REBNCONV] = None

        self.encoder_convs: List[REBNCONV] = [
            REBNCONV(mid_channels, dilation_rate=dilation, name=f"rebnconv{i + 1}")
            for i, dilation in enumerate(_FLAT_ENCODER_DILATIONS)
        ]
        self.decoder_convs: List[REBNCONV] = [
            REBNCONV(
                out_channels if i == 0 else mid_channels,
                dilation_rate=dilation,
                name=f"rebnconv{i + 1}d",
            )
            for i, dilation in enumerate(_FLAT_DECODER_DILATIONS)
        ]

    def _build_ladder_sublayers(
            self,
            stem_shape: Tuple[Optional[int], ...],
    ) -> None:
        """Build the flat cascade. No pooling, so no shape ever changes.

        :param stem_shape: The shape of ``rebnconvin``'s output.
        :type stem_shape: tuple
        """
        level_shape = stem_shape
        skip_shapes: List[Tuple[Optional[int], ...]] = []
        for conv in self.encoder_convs:
            conv.build(level_shape)
            level_shape = conv.compute_output_shape(level_shape)
            skip_shapes.append(level_shape)

        deeper_shape = skip_shapes[-1]
        for index in range(len(self.decoder_convs) - 1, -1, -1):
            skip_shape = skip_shapes[index]
            concat_shape = (
                *skip_shape[:-1],
                deeper_shape[-1] + skip_shape[-1],
            )
            self.decoder_convs[index].build(concat_shape)
            deeper_shape = self.decoder_convs[index].compute_output_shape(
                concat_shape
            )

    def _call_ladder(
            self,
            stem: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the dilation cascade at constant spatial resolution.

        :param stem: ``rebnconvin``'s output, ``(batch, h, w, out_channels)``.
        :type stem: keras.KerasTensor
        :param training: Forwarded to every :class:`REBNCONV`.
        :type training: Optional[bool]
        :return: The last decoder's output, ``(batch, h, w, out_channels)``.
        :rtype: keras.KerasTensor
        """
        skips: List[keras.KerasTensor] = []
        level = stem
        for conv in self.encoder_convs:
            level = conv(level, training=training)
            skips.append(level)

        # `hx4` is both the last encoder output and the innermost decoder's
        # partner: `hx3d = rebnconv3d(cat((hx4, hx3), 1))`, `seg.py:339`.
        # There is no `_upsample_like` anywhere in this block.
        deeper = skips[-1]
        for index in range(len(self.decoder_convs) - 1, -1, -1):
            # ORDER: deeper FIRST, skip second, as in the pooling ladders.
            merged = keras.ops.concatenate([deeper, skips[index]], axis=-1)
            deeper = self.decoder_convs[index](merged, training=training)

        return deeper

# ---------------------------------------------------------------------

__all__ = [
    "BATCH_NORM_EPSILON",
    "BATCH_NORM_MOMENTUM",
    "REBNCONV",
    "RSU4",
    "RSU4F",
    "RSU5",
    "RSU6",
    "RSU7",
]
