"""Behavioural guards for the U2NET-P building blocks.

Every claim this file makes is one that the obvious tests cannot see. An RSU
block maps ``(B, H, W, in)`` to ``(B, H, W, out)`` -- and it goes on doing that
with the decoder concatenation reversed, with the outer residual deleted, with
the pooling rounding the wrong way and with a batch-norm epsilon 100x too
large. A shape test, a finiteness test and a serialization round trip are all
structurally blind to the entire contents of these blocks.

So the instruments here are:

1. **An independent reference forward.** ``_reference_ladder_forward`` is
   written straight from ``seg.py``'s ``RSU7.forward``, calling the block's own
   already-built sub-layers in the order the reference states. Asserting the
   block equals it pins the pool placement, the ``_upsample_like`` targets, the
   ``(deeper, skip)`` concatenation order, which tensor the bottom convolution
   consumes and the residual add -- all at once, by VALUE. It is deliberately
   NOT written by calling ``_call_ladder``; a reference derived from the code
   under test measures nothing.
2. **A hand-written torch reference for ``ceil_mode``.** Compared elementwise,
   on an ALL-NEGATIVE input, so it judges WHICH ELEMENTS were pooled and not
   merely the output size. A matching size with a different element selection
   is this plan's recurring silent-defect class.
3. **A linearised impulse probe.** With every convolution set to a positive
   constant and every batch norm to the identity, the whole block is monotone
   and ReLU is an identity, so an output pixel is non-zero exactly when a path
   connects it to the impulse. That turns "RSU4F has no pooling" from a claim
   about sub-layer types into two measurements: a reach of exactly 23 over a
   centred box (against 28 and not-a-box for the pooled ``RSU4``), and EXACT
   shift-equivariance (error ``0.0``) where the pooled sibling aliases onto its
   2-grid (error ``0.082`` at a response scale of ``2.74``).

Fixtures are NON-SQUARE (``32x24``) wherever an axis could be confused, and one
whole class works at an ODD, non-power-of-two size (``37x53``) because that is
the only regime in which the ``ceil_mode`` remedy is not inert.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.image_restoration.doc_scanner import components
from dl_techniques.models.vision.image_restoration.doc_scanner import u2net_blocks
from dl_techniques.models.vision.image_restoration.doc_scanner.u2net_blocks import (
    BATCH_NORM_EPSILON,
    BATCH_NORM_MOMENTUM,
    REBNCONV,
    RSU4,
    RSU4F,
    RSU5,
    RSU6,
    RSU7,
    _ResidualUBlock,
    _upsample_like,
)

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from ..roundtrip_instrument_oracle import (
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_roundtrip,
)
from ..smoke_contract_oracle import assert_finite

# ---------------------------------------------------------------------
# The subject.
#
# STAND-IN widths, small and pairwise distinct: `mid` and `out` must differ, or
# every `mid*2 -> mid` decoder and the `mid*2 -> out` final one have the same
# signature and a mis-wired last decoder is invisible. The SHIPPED widths (16
# and 64) are exercised separately, once, by the parameter-count arm.
# ---------------------------------------------------------------------

MID = 4
OUT = 6

#: Non-square, and both even, so the pooling ladder is exact at every level.
HEIGHT, WIDTH = 32, 24

#: ODD on both axes and not a power of two: 37 pools to 19, 10, 5, 3, 2 and 53
#: to 27, 14, 7, 4, 2. Every one of those five levels is ragged.
ODD_HEIGHT, ODD_WIDTH = 37, 53

BATCH = 2

#: The repo convention for a value assertion: an explicit absolute tolerance
#: and no relative component.
ATOL = 1e-6

#: The four POOLING ladders and their level counts, which is the only thing
#: that distinguishes them.
POOLING_LADDERS = [(RSU7, 6), (RSU6, 5), (RSU5, 4), (RSU4, 3)]

ALL_BLOCKS = [RSU7, RSU6, RSU5, RSU4, RSU4F]


def _inputs(height: int = HEIGHT, width: int = WIDTH, channels: int = 3) -> np.ndarray:
    """Deterministic, sign-varying, non-constant input."""
    count = BATCH * height * width * channels
    return np.linspace(-1.0, 1.0, count, dtype="float32").reshape(
        (BATCH, height, width, channels))


def _block(cls, mid: int = MID, out: int = OUT):
    return cls(mid_channels=mid, out_channels=out)


# ---------------------------------------------------------------------
# Reference implementations, written from the upstream source.
# ---------------------------------------------------------------------


def _torch_maxpool2d_ceil(array: np.ndarray, kernel: int = 2, stride: int = 2):
    """``nn.MaxPool2d(kernel, stride, ceil_mode=True)``, by hand, channels-last.

    torch's ``ceil_mode`` rounds the output size UP and CLAMPS the ragged
    window to the input -- no padded value ever participates in the maximum,
    which is the half of the contract an output-size assertion cannot see.

    Returns both the pooled values and, for each output element, the ``(row,
    column)`` of the input element that won it.
    """
    batch, height, width, channels = array.shape

    def out_size(extent: int) -> int:
        size = -(-(extent - kernel) // stride) + 1
        while (size - 1) * stride >= extent:
            # torch drops a window that would START past the end of the input.
            size -= 1
        return size

    out_h, out_w = out_size(height), out_size(width)
    pooled = np.empty((batch, out_h, out_w, channels), array.dtype)
    winners = np.empty((batch, out_h, out_w, channels, 2), int)
    for i in range(out_h):
        row_lo, row_hi = i * stride, min(i * stride + kernel, height)
        for j in range(out_w):
            col_lo, col_hi = j * stride, min(j * stride + kernel, width)
            window = array[:, row_lo:row_hi, col_lo:col_hi, :]
            pooled[:, i, j, :] = window.max(axis=(1, 2))
            flat = window.reshape(batch, -1, channels).argmax(axis=1)
            winners[:, i, j, :, 0] = row_lo + flat // (col_hi - col_lo)
            winners[:, i, j, :, 1] = col_lo + flat % (col_hi - col_lo)
    return pooled, winners


def _torch_interpolate_bilinear(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """``F.interpolate(mode='bilinear', align_corners=False)``, by hand.

    Half-pixel centres: source ``= (o + 0.5) * n_in / n_out - 0.5``, clamped to
    the input extent. This is the OPPOSITE convention from ``warp.py``'s
    ``align_corners=True`` sampler, which is exactly what D-020 records.
    """
    batch, height, width, channels = array.shape
    out = np.zeros((batch, out_h, out_w, channels), np.float64)
    for i in range(out_h):
        yy = np.clip((i + 0.5) * height / out_h - 0.5, 0.0, height - 1)
        y0 = int(np.floor(yy))
        y1 = min(y0 + 1, height - 1)
        wy = yy - y0
        for j in range(out_w):
            xx = np.clip((j + 0.5) * width / out_w - 0.5, 0.0, width - 1)
            x0 = int(np.floor(xx))
            x1 = min(x0 + 1, width - 1)
            wx = xx - x0
            out[:, i, j, :] = (
                (1 - wy) * ((1 - wx) * array[:, y0, x0, :] + wx * array[:, y0, x1, :])
                + wy * ((1 - wx) * array[:, y1, x0, :] + wx * array[:, y1, x1, :])
            )
    return out


def _torch_conv2d_dilated(
        array: np.ndarray,
        kernel: np.ndarray,
        bias: np.ndarray,
        dilation: int,
) -> np.ndarray:
    """``nn.Conv2d(3, padding=dilation, dilation=dilation, stride=1)``, by hand."""
    batch, height, width, _ = array.shape
    padded = np.pad(
        array, ((0, 0), (dilation, dilation), (dilation, dilation), (0, 0)))
    out = np.zeros((batch, height, width, kernel.shape[-1]), np.float64)
    for i in range(height):
        for j in range(width):
            acc = np.zeros((batch, kernel.shape[-1]))
            for ki in range(3):
                for kj in range(3):
                    acc += padded[:, i + ki * dilation, j + kj * dilation, :] @ kernel[ki, kj]
            out[:, i, j, :] = acc
    return out + bias


def _reference_ladder_forward(block, inputs, training: bool = False):
    """``RSU7.forward`` (``seg.py:91-129``), transcribed, over ``block``'s layers.

    Written from the UPSTREAM source, not from ``_ResidualUBlock._call_ladder``.
    It uses the block's own already-built sub-layers, so it shares the weights
    and any disagreement is a WIRING disagreement -- the concatenation order,
    which tensor the bottom convolution eats, where each ``_upsample_like``
    points, and whether the residual is added.
    """
    hxin = block.rebnconvin(inputs, training=training)

    encoder_outputs = []
    hx = hxin
    for index, conv in enumerate(block.encoder_convs):
        hx = conv(hx, training=training)
        encoder_outputs.append(hx)
        if index < len(block.pools):
            hx = block.pools[index](hx)

    # `hx7 = self.rebnconv7(hx6)` -- the LAST ENCODER OUTPUT, not the pooled
    # tensor, and not `hxin`.
    deepest = block.bottom_conv(encoder_outputs[-1], training=training)

    # `hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))` -- no upsample here.
    last = len(block.decoder_convs) - 1
    decoded = block.decoder_convs[last](
        keras.ops.concatenate([deepest, encoder_outputs[last]], axis=-1),
        training=training,
    )
    for index in range(last - 1, -1, -1):
        skip = encoder_outputs[index]
        upsampled = _upsample_like(decoded, skip)
        decoded = block.decoder_convs[index](
            keras.ops.concatenate([upsampled, skip], axis=-1),
            training=training,
        )

    # `return hx1d + hxin`
    return decoded + hxin


def _reference_flat_forward(block, inputs, training: bool = False):
    """``RSU4F.forward`` (``seg.py:331-343``), transcribed, over ``block``'s layers."""
    hxin = block.rebnconvin(inputs, training=training)

    encoder_outputs = []
    hx = hxin
    for conv in block.encoder_convs:
        hx = conv(hx, training=training)
        encoder_outputs.append(hx)

    decoded = encoder_outputs[-1]
    for index in range(len(block.decoder_convs) - 1, -1, -1):
        decoded = block.decoder_convs[index](
            keras.ops.concatenate([decoded, encoder_outputs[index]], axis=-1),
            training=training,
        )
    return decoded + hxin


# ---------------------------------------------------------------------
# The linearising harness for the receptive-field probe.
# ---------------------------------------------------------------------

#: The positive constant every convolution kernel is set to. Small enough that
#: eight stacked ``3x3 x C`` sums neither overflow nor decay to zero at
#: float32.
_POSITIVE_KERNEL = 0.05


def _linearise(block) -> None:
    """Make a BUILT block monotone and ReLU-transparent.

    Every convolution kernel becomes a positive constant with a zero bias, and
    every batch norm becomes the identity (``gamma=1, beta=0, mean=0, var=1``,
    read at inference). A non-negative input then stays non-negative
    everywhere, so ``relu`` is the identity, every path is strictly positive,
    and an output pixel is EXACTLY zero if and only if no path connects it to a
    non-zero input pixel. Without this, ReLU could zero a live path and the
    measured reach would silently undercount.
    """
    for layer in block._flatten_layers(include_self=False):
        if isinstance(layer, keras.layers.Conv2D):
            layer.kernel.assign(keras.ops.full(layer.kernel.shape, _POSITIVE_KERNEL))
            if layer.bias is not None:
                layer.bias.assign(keras.ops.zeros(layer.bias.shape))
        elif isinstance(layer, keras.layers.BatchNormalization):
            layer.gamma.assign(keras.ops.ones(layer.gamma.shape))
            layer.beta.assign(keras.ops.zeros(layer.beta.shape))
            layer.moving_mean.assign(keras.ops.zeros(layer.moving_mean.shape))
            layer.moving_variance.assign(
                keras.ops.ones(layer.moving_variance.shape))


def _impulse_response(block, extent: int = 61, offset: int = 0) -> np.ndarray:
    """The linearised block's response to one positive impulse.

    The block is built, linearised and then fed a single positive impulse at
    ``(centre, centre + offset)``. Every value in the result is non-negative,
    and is strictly positive exactly where a path connects it to the impulse.
    """
    block(np.zeros((1, extent, extent, 3), "float32"), training=False)
    _linearise(block)

    canvas = np.zeros((1, extent, extent, 3), "float32")
    canvas[0, extent // 2, extent // 2 + offset, :] = 1.0
    return np.asarray(block(canvas, training=False))[0]


def _influence_radius_and_shape(block, extent: int = 61):
    """``(Chebyshev radius, is the support a CENTRED square box)``.

    Both halves matter. A dilated stack of odd square kernels has a support
    that is exactly a centred box; a ``ceil_mode`` pooling ladder does not,
    because ``padding="same"`` pads only the bottom/right and the ragged edge
    is therefore one-sided. Reporting the radius alone would throw that away.
    """
    touched = _impulse_response(block, extent).any(axis=-1) > 0.0
    rows, cols = np.where(touched)

    centre = extent // 2
    radius = int(max(
        centre - rows.min(), rows.max() - centre,
        centre - cols.min(), cols.max() - centre))
    assert radius < centre, (
        f"the {extent}x{extent} canvas is too small for a reach of {radius}; "
        f"the support is clipped and the number is a floor, not the reach")

    box = np.zeros((extent, extent), bool)
    box[centre - radius:centre + radius + 1,
        centre - radius:centre + radius + 1] = True
    return radius, bool(np.array_equal(touched, box))


def _shift_equivariance_error(block, extent: int = 61, margin: int = 5):
    """``(max |response(x) shifted by 1 - response(x+1)|, response scale)``.

    A stack of stride-1 dilated convolutions is EXACTLY shift-equivariant away
    from the padding boundary: move the impulse one column and the whole
    response moves one column. A stride-2 pooling ladder is not -- it aliases
    onto a 2-grid, so a one-pixel shift lands the impulse in a different pooling
    phase and changes the SHAPE of the response, not merely its position. That
    is the sharpest behavioural difference between the flat and pooled blocks.
    """
    at_zero = _impulse_response(block, extent, offset=0)
    at_one = _impulse_response(block, extent, offset=1)
    lo, hi = margin, extent - margin
    error = float(np.abs(at_zero[lo:hi, lo - 1:hi - 1]
                         - at_one[lo:hi, lo:hi]).max())
    return error, float(np.abs(at_zero).max())


# =====================================================================
# 1. Shapes -- necessary, and explicitly not sufficient.
# =====================================================================


class TestEveryBlockPreservesSpatialSizeAndSetsTheOutputWidth:
    """``(B, H, W, in) -> (B, H, W, out)`` for all five blocks.

    This class is the WEAKEST instrument in the file and is here only because
    its failure would be catastrophic. Everything it could miss is covered
    below.
    """

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    @pytest.mark.parametrize("channels", [3, 64, 128])
    def test_the_output_shape_is_the_input_size_at_out_channels(self, cls, channels):
        block = _block(cls)
        outputs = block(_inputs(channels=channels), training=False)
        assert tuple(outputs.shape) == (BATCH, HEIGHT, WIDTH, OUT)
        assert_finite(outputs)

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_compute_output_shape_agrees_with_the_real_call(self, cls):
        block = _block(cls)
        declared = block.compute_output_shape((None, HEIGHT, WIDTH, 3))
        actual = block(_inputs(), training=False).shape
        assert declared == (None, HEIGHT, WIDTH, OUT)
        assert tuple(actual)[1:] == declared[1:]

    @pytest.mark.parametrize("cls,levels", POOLING_LADDERS)
    def test_the_pool_count_is_one_fewer_than_the_level_count(self, cls, levels):
        """``seg.py`` pools after every encoder convolution EXCEPT the last."""
        block = _block(cls)
        assert cls._ENCODER_LEVELS == levels
        assert len(block.encoder_convs) == levels
        assert len(block.pools) == levels - 1
        assert len(block.decoder_convs) == levels

    def test_only_the_last_decoder_widens_back_to_out_channels(self):
        """``rebnconv1d`` emits ``out_ch``; every other decoder emits ``mid_ch``."""
        block = _block(RSU7)
        assert block.decoder_convs[0].filters == OUT
        for conv in block.decoder_convs[1:]:
            assert conv.filters == MID
        for conv in block.encoder_convs:
            assert conv.filters == MID
        assert block.rebnconvin.filters == OUT


# =====================================================================
# 2. The ceil_mode remedy -- D-019.
# =====================================================================


class TestTheCeilModeRemedy:
    """``padding="same"`` IS torch's ``ceil_mode=True``, elements included."""

    @pytest.mark.parametrize("cls,_levels", POOLING_LADDERS)
    def test_every_pool_is_configured_as_the_remedy_says(self, cls, _levels):
        block = _block(cls)
        for pool in block.pools:
            assert pool.pool_size == (2, 2)
            assert pool.strides == (2, 2)
            assert pool.padding == "same", (
                "padding='valid' rounds an odd extent DOWN where torch's "
                "ceil_mode rounds it UP; see D-019")

    @pytest.mark.parametrize(
        "height,width", [(9, 7), (8, 6), (5, 11), (37, 53), (32, 24)])
    def test_the_pooled_values_match_a_hand_written_torch_reference(
            self, height, width, golden_reference_device):
        """Values, on an ALL-NEGATIVE input, so a zero-pad would be caught.

        TF pads max-pooling with ``-inf``. A hypothetical zero-padded
        implementation would win every ragged window with a padded ``0.0``,
        which on an all-negative input is larger than every real element -- so
        this fixture is what makes the comparison capable of failing.
        """
        rng = np.random.default_rng(0)
        array = (-1.0 - rng.random((2, height, width, 3))).astype("float32")
        expected, _ = _torch_maxpool2d_ceil(array)

        pool = _block(RSU7).pools[0]
        actual = np.asarray(pool(keras.ops.convert_to_tensor(array)))

        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("height,width", [(9, 7), (5, 11), (37, 53)])
    def test_the_pool_selects_the_same_ELEMENTS_not_merely_the_same_count(
            self, height, width, golden_reference_device):
        """A matching size with a different selection is the silent defect.

        Every input element is made UNIQUE, so the pooled value identifies its
        source element exactly and the comparison is about WHICH element won,
        not about how many there were.
        """
        array = np.arange(
            2 * height * width * 3, dtype="float32"
        ).reshape((2, height, width, 3))
        # Shuffle so the maximum is not trivially the bottom-right corner of
        # every window, which would make an off-by-one invisible.
        rng = np.random.default_rng(7)
        array = rng.permutation(array.reshape(-1)).reshape(array.shape).copy()

        expected, winners = _torch_maxpool2d_ceil(array)
        pool = _block(RSU7).pools[0]
        actual = np.asarray(pool(keras.ops.convert_to_tensor(array)))

        np.testing.assert_array_equal(actual, expected)
        # And prove the selection is non-trivial: at least one ragged window
        # must have had fewer than 4 candidates, or this fixture is inert.
        if height % 2 or width % 2:
            assert (winners[..., 0].max() == height - 1
                    or winners[..., 1].max() == width - 1)

    def test_valid_padding_would_round_the_other_way(self, golden_reference_device):
        """The NEGATIVE control: the rejected configuration really differs.

        Without this, the assertion above could be passing because ``valid``
        and ``same`` agree at these sizes, which they do at even ones.
        """
        array = np.linspace(-2.0, -1.0, 2 * 9 * 7 * 3, dtype="float32").reshape(
            (2, 9, 7, 3))
        expected, _ = _torch_maxpool2d_ceil(array)
        valid = keras.layers.MaxPooling2D(2, strides=2, padding="valid")
        actual = np.asarray(valid(keras.ops.convert_to_tensor(array)))
        assert actual.shape != expected.shape


class TestAnOddInputSurvivesTheWholeLadder:
    """37x53 through every block: 5 ragged levels, and the size comes back."""

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_the_odd_input_size_is_returned_unchanged(self, cls):
        block = _block(cls)
        outputs = block(_inputs(ODD_HEIGHT, ODD_WIDTH), training=False)
        assert tuple(outputs.shape) == (BATCH, ODD_HEIGHT, ODD_WIDTH, OUT)
        assert_finite(outputs)

    def test_the_ladders_intermediate_sizes_are_the_ceil_mode_ones(self):
        """37 -> 19 -> 10 -> 5 -> 3 -> 2, not 18 -> 9 -> 4 -> 2 -> 1."""
        block = _block(RSU7)
        block.build((None, ODD_HEIGHT, ODD_WIDTH, 3))

        shape = (None, ODD_HEIGHT, ODD_WIDTH, MID)
        heights = [ODD_HEIGHT]
        widths = [ODD_WIDTH]
        for pool in block.pools:
            shape = pool.compute_output_shape(shape)
            heights.append(shape[1])
            widths.append(shape[2])
        assert heights == [37, 19, 10, 5, 3, 2]
        assert widths == [53, 27, 14, 7, 4, 2]

    def test_at_an_even_power_of_two_size_the_remedy_is_inert(self):
        """288 is divisible by 32, which is why this is invisible in production.

        Recorded as an explicit measurement rather than left as a claim in a
        docstring: it is the reason the odd-size arms above exist at all.
        """
        block = _block(RSU7)
        block.build((None, 288, 288, 3))
        shape = (None, 288, 288, MID)
        sizes = [288]
        for pool in block.pools:
            shape = pool.compute_output_shape(shape)
            sizes.append(shape[1])
        assert sizes == [288, 144, 72, 36, 18, 9]


# =====================================================================
# 3. The wiring -- against an independent reference forward.
# =====================================================================


class TestTheWiringMatchesTheUpstreamForwardExactly:
    """The block equals a transcription of ``seg.py``'s own ``forward``.

    Goes RED under: a swapped ``(skip, deeper)`` concatenation (both halves are
    ``mid``-wide, so it is shape-preserving), a bottom convolution fed the
    POOLED tensor, an ``_upsample_like`` pointed at the wrong level, and a
    missing residual add. None of those has any shape symptom.
    """

    @pytest.mark.parametrize("cls,_levels", POOLING_LADDERS)
    @pytest.mark.parametrize("size", [(HEIGHT, WIDTH), (ODD_HEIGHT, ODD_WIDTH)])
    def test_a_pooling_ladder_equals_its_upstream_transcription(
            self, cls, _levels, size, golden_reference_device):
        block = _block(cls)
        inputs = _inputs(*size)
        actual = np.asarray(block(inputs, training=False))
        expected = np.asarray(_reference_ladder_forward(block, inputs))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)

    @pytest.mark.parametrize("size", [(HEIGHT, WIDTH), (ODD_HEIGHT, ODD_WIDTH)])
    def test_the_flat_block_equals_its_upstream_transcription(
            self, size, golden_reference_device):
        block = _block(RSU4F)
        inputs = _inputs(*size)
        actual = np.asarray(block(inputs, training=False))
        expected = np.asarray(_reference_flat_forward(block, inputs))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)

    def test_the_reference_is_not_vacuous_under_a_swapped_concatenation(
            self, golden_reference_device):
        """The negative control the arms above need.

        If a reversed ``concat`` produced the same numbers -- because the two
        halves happened to be equal, say -- then the comparison above would be
        satisfied by a defect. Here the reversed reference is computed and
        asserted to DIFFER, which is what makes the equality meaningful.
        """
        block = _block(RSU4F)
        inputs = _inputs()
        block(inputs, training=False)

        hxin = block.rebnconvin(inputs, training=False)
        encoder_outputs = []
        hx = hxin
        for conv in block.encoder_convs:
            hx = conv(hx, training=False)
            encoder_outputs.append(hx)

        swapped = encoder_outputs[-1]
        for index in range(len(block.decoder_convs) - 1, -1, -1):
            swapped = block.decoder_convs[index](
                # REVERSED on purpose: skip first, deeper second.
                keras.ops.concatenate(
                    [encoder_outputs[index], swapped], axis=-1),
                training=False,
            )
        swapped = np.asarray(swapped + hxin)

        correct = np.asarray(block(inputs, training=False))
        assert np.abs(correct - swapped).max() > 1e-3, (
            "a reversed concatenation produced the same output, so the "
            "equality assertions above cannot see the order at all")


class TestTheOuterResidualIsPresent:
    """``return hx1d + hxin`` (``seg.py:129``), not ``return hx1d``.

    Deleting the add leaves ``(B, H, W, out)`` finite float32, saves, reloads
    and trains. The only thing that sees it is a value comparison against the
    ladder's own output.
    """

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_the_output_is_the_ladder_plus_the_stem(
            self, cls, golden_reference_device):
        block = _block(cls)
        inputs = _inputs()
        actual = np.asarray(block(inputs, training=False))

        stem = block.rebnconvin(inputs, training=False)
        ladder = np.asarray(block._call_ladder(stem, training=False))
        np.testing.assert_allclose(
            actual, ladder + np.asarray(stem), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_dropping_the_add_would_change_the_output(
            self, cls, golden_reference_device):
        """Non-vacuity: the stem is not (near-)zero, so the add BITES.

        ``rebnconvin`` ends in a ReLU, so a pathological configuration could
        make the stem identically zero and the assertion above would hold for
        ``return hx1d`` too.
        """
        block = _block(cls)
        inputs = _inputs()
        actual = np.asarray(block(inputs, training=False))
        stem = np.asarray(block.rebnconvin(inputs, training=False))
        ladder = actual - stem
        assert np.abs(actual - ladder).max() > 1e-3


# =====================================================================
# 4. RSU4F really has no pooling -- structurally AND behaviourally.
# =====================================================================


class TestRSU4FIsFlat:
    """No pooling layer exists, AND the reach is the dilation cascade's."""

    def test_no_pooling_sublayer_exists_anywhere_in_it(self):
        block = _block(RSU4F)
        block.build((None, HEIGHT, WIDTH, 3))
        pools = [layer for layer in block._flatten_layers(include_self=False)
                 if isinstance(layer, keras.layers.MaxPooling2D)]
        assert pools == []
        assert block.pools == []
        assert block.bottom_conv is None, (
            "an unused bottom convolution would still allocate weights and "
            "the gradient-flow oracle would then report them as dead")

    def test_its_dilation_schedule_is_1_2_4_8_then_4_2_1(self):
        block = _block(RSU4F)
        assert [c.dilation_rate for c in block.encoder_convs] == [1, 2, 4, 8]
        # `decoder_convs` is indexed innermost-LAST: index 0 is `rebnconv1d`.
        assert [c.dilation_rate for c in block.decoder_convs] == [1, 2, 4]
        assert block.rebnconvin.dilation_rate == 1

    def test_the_measured_reach_is_exactly_the_dilation_sum(
            self, golden_reference_device):
        """23, derived from the schedule, measured from an impulse.

        Eight 3x3 convolutions on the only path -- ``rebnconvin`` (1), the
        encoder at 1, 2, 4, 8 and the decoder at 4, 2, 1 -- each reach
        ``dilation`` pixels per side, so the Chebyshev reach is
        ``1 + 1 + 2 + 4 + 8 + 4 + 2 + 1 = 23``, and the support is a CENTRED
        BOX because nothing here is stride-2 or one-sidedly padded.
        """
        expected = 1 + sum([1, 2, 4, 8]) + sum([4, 2, 1])
        assert expected == 23
        radius, is_box = _influence_radius_and_shape(_block(RSU4F))
        assert radius == expected
        assert is_box

    def test_the_pooled_siblings_reach_differs_and_is_NOT_a_centred_box(
            self, golden_reference_device):
        """The negative control: the probe DISCRIMINATES pooled from flat.

        ``RSU4`` has the same NUMBER of convolutions and the same widths as
        ``RSU4F``; the only difference is that it pools. Measured: reach 28
        against 23, and its support is not a centred box, because
        ``padding="same"`` pads only the bottom and the right.
        """
        radius, is_box = _influence_radius_and_shape(_block(RSU4))
        assert radius == 28
        assert not is_box

    def test_it_is_EXACTLY_shift_equivariant_where_a_pooled_block_is_not(
            self, golden_reference_device):
        """The strongest "no pooling" evidence, and it is behavioural.

        A stride-1 dilated cascade commutes with translation exactly. A
        stride-2 pooling ladder does not: a one-pixel shift changes the pooling
        PHASE, so the response changes shape. Measured here at ``0.0`` for
        ``RSU4F`` against ``0.082`` on a response whose scale is ``2.74`` for
        the pooled ``RSU4``.

        This is the arm that a "count the MaxPooling2D sub-layers" assertion
        cannot make: a block that kept its pools but never called them, or that
        strided a convolution instead, would pass the structural check and fail
        this one.
        """
        flat_error, flat_scale = _shift_equivariance_error(_block(RSU4F))
        assert flat_scale > 0.1, "the linearised response is degenerate"
        assert flat_error == 0.0

        pooled_error, pooled_scale = _shift_equivariance_error(_block(RSU4))
        assert pooled_error > 0.01 * pooled_scale


# =====================================================================
# 5. REBNCONV -- the atom, and the two torch defaults it must not inherit.
# =====================================================================


class TestREBNCONV:
    """The dilated conv / batch norm / ReLU atom."""

    @pytest.mark.parametrize("dilation", [1, 2, 4, 8])
    @pytest.mark.parametrize("size", [(11, 9), (ODD_HEIGHT, ODD_WIDTH)])
    def test_same_padding_at_stride_one_IS_torchs_padding_equals_dilation(
            self, dilation, size, golden_reference_device):
        """The claim D-012 does NOT contradict: at stride 1 they agree exactly.

        D-012 measured Keras' ``"same"`` to disagree with torch at stride 2
        (pad 0/1 versus 1/1). Every convolution in this module is stride 1, so
        the divergence does not apply -- but "does not apply" is a claim, and
        this is its measurement.
        """
        height, width = size
        array = np.linspace(
            -1.0, 1.0, 2 * height * width * 3, dtype="float32"
        ).reshape((2, height, width, 3))

        conv = keras.layers.Conv2D(
            4, 3, strides=1, padding="same", dilation_rate=dilation)
        actual = np.asarray(conv(keras.ops.convert_to_tensor(array)), np.float64)
        expected = _torch_conv2d_dilated(
            array.astype(np.float64),
            np.asarray(conv.kernel, np.float64),
            np.asarray(conv.bias, np.float64),
            dilation,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)

    def test_the_batch_norm_carries_torchs_epsilon_and_not_kerass(self):
        """``1e-5``, not Keras' ``1e-3``. A 100x with no shape symptom."""
        assert BATCH_NORM_EPSILON == 1e-5
        assert keras.layers.BatchNormalization().epsilon == 1e-3
        block = REBNCONV(filters=5, dilation_rate=1)
        assert block.norm.epsilon == 1e-5

    def test_the_batch_norm_momentum_is_the_CONVERTED_torch_value(self):
        """torch's ``momentum=0.1`` weights the NEW batch; Keras' the RETAINED.

        So the port's value is ``1 - 0.1 = 0.9``, NOT ``0.1`` -- transcribing
        the number across would be a 10x error on top of Keras' own ``0.99``
        default.
        """
        assert BATCH_NORM_MOMENTUM == 0.9
        assert keras.layers.BatchNormalization().momentum == 0.99
        assert REBNCONV(filters=5).norm.momentum == 0.9

    def test_the_spatial_size_survives_every_dilation(self):
        for dilation in (1, 2, 4, 8):
            conv = REBNCONV(filters=5, dilation_rate=dilation)
            outputs = conv(_inputs(), training=False)
            assert tuple(outputs.shape) == (BATCH, HEIGHT, WIDTH, 5)

    def test_the_output_is_non_negative_because_relu_is_last(self):
        """Order is conv -> norm -> ReLU, so nothing negative escapes."""
        outputs = np.asarray(
            REBNCONV(filters=5)(_inputs() * 10.0, training=False))
        assert outputs.min() >= 0.0
        assert outputs.max() > 0.0

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_width_or_dilation_is_refused(self, bad):
        with pytest.raises(ValueError, match="positive"):
            REBNCONV(filters=bad)
        with pytest.raises(ValueError, match="positive"):
            REBNCONV(filters=4, dilation_rate=bad)

    def test_a_non_4d_input_is_refused_by_name(self):
        with pytest.raises(ValueError, match="4D"):
            REBNCONV(filters=4).build((None, 10, 3))


class TestTheBottomConvolutionReplacesAPoolingLevel:
    """``rebnconv{N+1}`` runs at ``dirate=2`` where the ladder runs at 1."""

    @pytest.mark.parametrize("cls,levels", POOLING_LADDERS)
    def test_the_bottom_convolution_is_dilated_and_the_rest_are_not(
            self, cls, levels):
        block = _block(cls)
        assert block.bottom_conv.dilation_rate == 2
        assert block.bottom_conv.name == f"rebnconv{levels + 1}"
        assert all(c.dilation_rate == 1 for c in block.encoder_convs)
        assert all(c.dilation_rate == 1 for c in block.decoder_convs)
        assert block.rebnconvin.dilation_rate == 1


# =====================================================================
# 6. _upsample_like -- D-020, the OPPOSITE convention from warp.py.
# =====================================================================


class TestUpsampleLikeIsHalfPixelAndNotAlignCorners:
    """``align_corners=False``, deliberately unlike ``sample_at_pixel_coords``."""

    @pytest.mark.parametrize(
        "src,dst", [((3, 4), (5, 7)), ((5, 3), (10, 6)), ((2, 2), (5, 5)),
                    ((19, 27), (37, 53))])
    def test_it_matches_a_hand_written_half_pixel_reference(
            self, src, dst, golden_reference_device):
        rng = np.random.default_rng(3)
        source = rng.standard_normal((2, *src, 3)).astype("float32")
        target = np.zeros((2, *dst, 3), "float32")

        actual = np.asarray(
            _upsample_like(keras.ops.convert_to_tensor(source),
                           keras.ops.convert_to_tensor(target)), np.float64)
        expected = _torch_interpolate_bilinear(
            source.astype(np.float64), *dst)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)

    def test_an_align_corners_reference_would_give_DIFFERENT_numbers(
            self, golden_reference_device):
        """The two conventions are not interchangeable -- the point of D-020.

        ``align_corners=True`` maps the source's first and last CENTRES onto
        the destination's first and last centres; half-pixel maps the source's
        first and last EDGES onto the destination's. At a 2->4 upsample they
        disagree everywhere except by symmetry.
        """
        source = np.array(
            [[[0.0], [1.0]], [[2.0], [3.0]]], "float32")[None, ...]
        target = np.zeros((1, 4, 4, 1), "float32")

        half_pixel = np.asarray(
            _upsample_like(keras.ops.convert_to_tensor(source),
                           keras.ops.convert_to_tensor(target)))[0, :, 0, 0]
        # align_corners=True at 2 -> 4 samples the source rows at
        # 0, 1/3, 2/3, 1, i.e. values 0, 2/3, 4/3, 2 down column 0.
        align_corners = np.array([0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0], "float32")
        assert np.abs(half_pixel - align_corners).max() > 0.1

    def test_it_survives_a_fully_dynamic_spatial_axis(self):
        """The ``keras.ops.shape`` fallback branch is REACHED and works.

        A test that only ever builds at a static size never executes it, and a
        broken fallback then surfaces only in whatever downstream consumer
        first passes ``(None, None, 3)``.
        """
        inputs = keras.Input(shape=(None, None, 3))
        outputs = _block(RSU7)(inputs)
        assert outputs.shape == (None, None, None, OUT)

        model = keras.Model(inputs, outputs)
        result = model(_inputs(ODD_HEIGHT, ODD_WIDTH), training=False)
        assert tuple(result.shape) == (BATCH, ODD_HEIGHT, ODD_WIDTH, OUT)


# =====================================================================
# 7. The symbolic path (process rule 6 / D-018).
# =====================================================================


class TestEveryBlockTracesSymbolically:
    """A ``KerasTensor`` in, a ``KerasTensor`` out -- no eager shortcut.

    Steps 1-6 of this plan were layer-only and never traced anything
    symbolically; a real Keras 3.8 ``extract_patches`` bug hid behind that
    until step 7 (D-018). Every block added after that is traced here before it
    reaches a ``keras.Model``.
    """

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_a_kerastensor_input_produces_a_kerastensor_of_the_right_shape(
            self, cls):
        inputs = keras.Input(shape=(ODD_HEIGHT, ODD_WIDTH, 3))
        outputs = _block(cls)(inputs)
        assert isinstance(outputs, keras.KerasTensor)
        assert outputs.shape == (None, ODD_HEIGHT, ODD_WIDTH, OUT)

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_compute_output_spec_does_not_raise(self, cls):
        block = _block(cls)
        spec = block.compute_output_spec(
            keras.KerasTensor((None, HEIGHT, WIDTH, 3)))
        assert spec.shape == (None, HEIGHT, WIDTH, OUT)

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_it_runs_inside_a_traced_tf_function_with_an_unknown_batch(self, cls):
        block = _block(cls)
        block.build((None, HEIGHT, WIDTH, 3))
        traced = tf.function(
            lambda batch: block(batch, training=False),
            input_signature=[tf.TensorSpec((None, HEIGHT, WIDTH, 3), tf.float32)],
        )
        outputs = traced(tf.convert_to_tensor(_inputs()))
        assert tuple(outputs.shape) == (BATCH, HEIGHT, WIDTH, OUT)


# =====================================================================
# 8. The shared oracles, adopted rather than reimplemented.
# =====================================================================


def _functional(cls):
    """The block wrapped so the shared model oracles can judge it."""
    def build() -> keras.Model:
        inputs = keras.Input(shape=(HEIGHT, WIDTH, 3))
        return keras.Model(inputs, _block(cls)(inputs), name=f"rsu_{cls.__name__}")
    return build


class TestTheBlocksTrainAndRoundTrip:

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_every_trainable_weight_receives_a_live_gradient(self, cls):
        """Run at ``training=False``, for a MEASURED reason -- see below.

        ``training=True`` puts every batch normalization on batch statistics,
        which removes the preceding convolution's bias exactly: the bias shifts
        every element of a channel by the same amount and the normalization
        then subtracts the channel mean. Its gradient is therefore
        mathematically zero, and what the tape reports is float32 cancellation
        noise -- ``2.2e-6`` at best, and an exact ``0.0`` for whichever bias
        happens to cancel cleanly, which made this arm FLAKE at
        ``training=True``. At ``training=False`` the moving statistics are
        constants, the bias is live, and the smallest bias gradient measured is
        ``3.2e-4``.
        """
        model = _functional(cls)()
        assert_gradients_reach_every_trainable_weight(
            model, _inputs(), training=False)

    def test_the_conv_biases_are_MATHEMATICALLY_DEAD_under_training_batchnorm(
            self, golden_reference_device):
        """The measurement behind the arm above, recorded rather than implied.

        Upstream has the same redundancy -- ``nn.Conv2d(..., bias=True)``
        immediately followed by ``nn.BatchNorm2d`` (``seg.py:38-40``), torch's
        ``bias`` default -- and this port keeps it, because dropping
        ``use_bias`` would change the parameter count away from the reference
        for a saving of 14 scalars per block.

        Two-sided on purpose: dead under training statistics, LIVE under the
        moving ones. A one-sided version would be satisfied by a block whose
        biases are dead everywhere, which is a different (and real) defect.
        """
        import tensorflow as tf

        block = _block(RSU7)
        inputs = _inputs()
        block(inputs, training=True)

        def bias_gradients(training: bool):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    tf.square(block(inputs, training=training)))
            grads = tape.gradient(loss, block.trainable_variables)
            return {
                variable.path: float(tf.reduce_max(tf.abs(grad)))
                for variable, grad in zip(block.trainable_variables, grads)
                if variable.path.endswith("conv/bias")
            }

        under_batch_statistics = bias_gradients(training=True)
        under_moving_statistics = bias_gradients(training=False)

        assert len(under_batch_statistics) == 14
        assert max(under_batch_statistics.values()) < 1e-4, (
            "a convolution bias followed by a batch norm on BATCH statistics "
            "cannot have a real gradient; a large one here means the two are "
            "no longer adjacent")
        assert min(under_moving_statistics.values()) > 1e-6, (
            "under MOVING statistics the bias must be live; if it is not, the "
            "biases are dead everywhere and this is a real defect")

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_the_saved_and_reloaded_block_reproduces_its_output_exactly(self, cls):
        report = measure_roundtrip(_functional(cls), _inputs, training=False)
        assert report["self_max_delta"] == 0.0, (
            "the block became non-deterministic; the exact bound below would "
            "then be measuring that instead of the round trip")
        assert_roundtrip_output_values(report, atol=0.0)

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_the_weights_are_restored_before_the_reloaded_model_is_called(self, cls):
        report = measure_roundtrip(_functional(cls), _inputs, training=False)
        assert report["call_count_before_weight_read"] == 0
        assert_weights_restored_before_first_call(report, atol=0.0)

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_the_config_round_trips_every_constructor_argument(self, cls):
        block = _block(cls)
        clone = cls.from_config(block.get_config())
        assert type(clone) is cls
        assert clone.mid_channels == MID
        assert clone.out_channels == OUT

    def test_the_rebnconv_config_round_trips_every_constructor_argument(self):
        conv = REBNCONV(filters=7, dilation_rate=4)
        clone = REBNCONV.from_config(conv.get_config())
        assert clone.filters == 7
        assert clone.dilation_rate == 4

    @pytest.mark.parametrize("cls,levels", POOLING_LADDERS)
    def test_the_level_count_is_NOT_serialized_so_a_reload_cannot_change_class(
            self, cls, levels):
        """A reloaded ``RSU6`` can never come back as an ``RSU5``.

        The level count belongs to the CLASS, exactly as upstream puts it in
        the class NAME. If it were a config field, a hand-edited archive could
        produce a block whose class and topology disagree.
        """
        config = _block(cls).get_config()
        assert "levels" not in config
        assert "_ENCODER_LEVELS" not in config
        assert cls.from_config(config)._ENCODER_LEVELS == levels


# =====================================================================
# 9. The width table and the registration keys.
# =====================================================================


class TestTheSegmenterWidthsComeFromTheVariantTable:
    """``mid=16``, ``out=64``, cited to ``seg.py:456-478``'s CALL SITES."""

    def test_the_pruned_widths_are_pinned_against_their_literals(self):
        row = components._VARIANT_SPEC["docscanner-l"]
        assert row["seg_mid_channels"] == 16
        assert row["seg_out_channels"] == 64

    def test_they_are_NOT_the_dead_constructor_defaults(self):
        """``RSU7.__init__``'s upstream signature default is ``mid_ch=12``.

        Nothing constructs an RSU that way (D-006's trap). A 12-wide interior
        passes every shape assertion in this file, because the outer projection
        fixes the block's output width.
        """
        assert components._VARIANT_SPEC["docscanner-l"]["seg_mid_channels"] != 12
        assert components._VARIANT_SPEC["docscanner-l"]["seg_out_channels"] != 3

    def test_they_are_independent_of_the_rectifier_rows(self):
        """No derivation links the two stages; the port must not invent one."""
        row = components._VARIANT_SPEC["docscanner-l"]
        assert row["seg_out_channels"] != row["encoder_stem_channels"]

    def test_no_width_literal_is_hard_coded_in_the_block_module(self):
        """The blocks take their widths as arguments; the table is the source."""
        import inspect
        source = inspect.getsource(u2net_blocks)
        for forbidden in ("mid_channels = 16", "out_channels = 64",
                          "mid_channels=16", "out_channels=64"):
            assert forbidden not in source

    def test_the_shipped_widths_give_the_upstream_parameter_counts(self):
        """A hand-derived count from ``seg.py``'s own module list.

        Every ``REBNCONV`` is ``3*3*Cin*Cout + Cout`` convolution parameters
        plus ``4*Cout`` batch-norm ones (torch: 2 parameters + 2 buffers; Keras:
        2 trainable + 2 non-trainable, which is why ``count_params`` is
        comparable across the two).
        """
        mid = components._VARIANT_SPEC["docscanner-l"]["seg_mid_channels"]
        out = components._VARIANT_SPEC["docscanner-l"]["seg_out_channels"]

        def rebnconv(c_in: int, c_out: int) -> int:
            return 3 * 3 * c_in * c_out + c_out + 4 * c_out

        for cls, levels in POOLING_LADDERS:
            expected = rebnconv(3, out)                    # rebnconvin
            expected += rebnconv(out, mid)                 # rebnconv1
            expected += (levels - 1) * rebnconv(mid, mid)  # rebnconv2..N
            expected += rebnconv(mid, mid)                 # the bottom conv
            expected += (levels - 1) * rebnconv(2 * mid, mid)
            expected += rebnconv(2 * mid, out)             # rebnconv1d
            block = cls(mid_channels=mid, out_channels=out)
            block.build((None, HEIGHT, WIDTH, 3))
            assert block.count_params() == expected, cls.__name__

        flat = rebnconv(3, out) + rebnconv(out, mid) + 3 * rebnconv(mid, mid)
        flat += 2 * rebnconv(2 * mid, mid) + rebnconv(2 * mid, out)
        block = RSU4F(mid_channels=mid, out_channels=out)
        block.build((None, HEIGHT, WIDTH, 3))
        assert block.count_params() == flat


class TestRegistrationKeysStripFamilyAndSubfamily:
    """H-3. ``dl_techniques.models.doc_scanner.u2net_blocks``, literal.

    Asserted with a literal ``==``, never through a save/load round trip: the
    process shares ONE registry, so a round trip resolves a typo'd key to the
    typo'd class perfectly happily.
    """

    _PACKAGE = "dl_techniques.models.doc_scanner.u2net_blocks"

    @pytest.mark.parametrize(
        "cls_name",
        ["REBNCONV", "RSU7", "RSU6", "RSU5", "RSU4", "RSU4F"],
    )
    def test_the_registered_name_is_exactly_this_string(self, cls_name):
        cls = getattr(u2net_blocks, cls_name)
        assert keras.saving.get_registered_name(cls) == (
            f"dl_techniques.models.doc_scanner.u2net_blocks>{cls_name}")

    @pytest.mark.parametrize(
        "cls_name",
        ["REBNCONV", "RSU7", "RSU6", "RSU5", "RSU4", "RSU4F"],
    )
    def test_the_shared_registration_contract_holds(
            self, cls_name, registration_contract):
        registration_contract(
            getattr(u2net_blocks, cls_name), expected_package=self._PACKAGE)

    def test_the_private_base_is_NOT_registered(self):
        """``_ResidualUBlock`` is never instantiated and never serialized.

        Registering an abstract base would put a class that raises on
        construction into the global deserialization registry.
        """
        assert keras.saving.get_registered_name(_ResidualUBlock) == (
            "_ResidualUBlock")

    def test_the_base_refuses_to_be_instantiated_directly(self):
        with pytest.raises(ValueError, match="_ENCODER_LEVELS"):
            _ResidualUBlock(mid_channels=MID, out_channels=OUT)


class TestConstructionIsValidated:

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    @pytest.mark.parametrize("bad", [0, -3])
    def test_a_non_positive_width_is_refused(self, cls, bad):
        with pytest.raises(ValueError, match="positive"):
            cls(mid_channels=bad, out_channels=OUT)
        with pytest.raises(ValueError, match="positive"):
            cls(mid_channels=MID, out_channels=bad)

    @pytest.mark.parametrize("cls", ALL_BLOCKS)
    def test_a_non_4d_input_is_refused_by_name(self, cls):
        with pytest.raises(ValueError, match="4D"):
            _block(cls).build((None, 10, 3))
