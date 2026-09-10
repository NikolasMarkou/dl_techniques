"""Behavioural guards for :class:`DocScannerSegmenter`, the U2NET-P stage.

The shared oracles under ``tests/test_models/*_oracle.py`` carry everything
generic -- the ``.keras`` round trip on VALUES, weights-restored-before-first-
call, build parity, gradient flow, the falsifiable smoke contract. None of them
is re-implemented here.

What IS written here is the set of claims specific to this assembly, every one
of which is **shape-preserving**: a segmenter carrying any of these defects
returns seven finite ``(B, H, W, 1)`` maps in ``[0, 1]``, trains, saves and
reloads.

1. **``side6`` reads the ENCODER's ``hx6``, not a decoder output**
   (``seg.py:546``). Six side heads, five decoder stages. Feeding ``side6`` the
   deepest decoder output ``hx5d`` instead is the natural way to "fix" that
   asymmetry, and both tensors are ``out_channels`` wide and both get resized
   onto ``d1``, so the wiring error is invisible to every shape assertion.
2. **The decoder concatenates ``(upsampled_deeper, skip)`` in that order**
   (``seg.py:517``). Both halves are ``out_channels`` wide.
3. **All seven outputs are DISTINCT tensors.** A wiring bug that returns ``d0``
   seven times is shape-perfect.
4. **``d0`` is the learned 1x1 FUSION of the six side maps** (``seg.py:548``),
   not a copy of ``d1``.
5. **The ragged (odd-size) path works end to end.** This is what step 8's
   ``ceil_mode`` remedy and resize-onto-the-skip decoder were built for, and
   the ladder only actually goes ragged at a size 288 is not.

Non-square fixtures are used throughout, for the reason ``test_components.py``
states at length: a square fixture is structurally blind to an h/w swap.

The parameter count is asserted against an INDEPENDENT arithmetic transcription
of ``seg.py``'s own construction (see :func:`_reference_parameter_counts`), not
against a number copied out of this port.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.image_restoration.doc_scanner import model as model_module
from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
    SEG_OUTPUT_CHANNELS,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.model import (
    DocScannerSegmenter,
    create_doc_scanner_segmenter,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.u2net_blocks import (
    RSU4,
    RSU4F,
    RSU5,
    RSU6,
    RSU7,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    stop_all_gradients,
)
from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing
from ..roundtrip_instrument_oracle import (
    assert_build_parity,
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_build_parity,
    measure_roundtrip,
)
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

# ---------------------------------------------------------------------
# The subject.
#
# The widths are STAND-INS, deliberately small and pairwise distinct so that a
# confusion between two of them cannot hide behind two equal numbers -- the
# same rule `test_model.py` and `test_u2net_blocks.py` apply. The shipped
# 16/64 widths are exercised separately, by the `docscanner-l` arm.
#
# BATCH is 2, never 1: the smoke oracle's `slice_leading_axis` breaker slices
# each output leaf to `leaf[:1]`, which is a no-op at batch 1, and the
# meta-test would then report "the contract accepts a broken forward" about a
# breaker that broke nothing.
#
# HEIGHT/WIDTH are NON-SQUARE and, deliberately, NOT multiples of 32: at 24x40
# the encoder ladder runs 24, 12, 6, 3, 2, 1 on the height axis and
# 40, 20, 10, 5, 3, 2 on the width, so the ceil_mode path is live in the
# default fixture rather than only in the one test that asks for it.
#
# The widths are 7 and 13 rather than something smaller, and that is MEASURED
# rather than aesthetic. See `TestTheFixtureIsNotDegenerate`: at mid/out = 3/5
# the deepest blocks' ReLU is DEAD at `training=False` on this fixture and the
# gradient-flow arm reads 20 of 462 weights as having an identically-zero
# gradient -- a false RED about the port, caused by the stand-in. Measured
# across seeds 0-7 at (24, 40) and (37, 53): 3/5 dies, 6/11 dies, 7/13 does
# not. Do not shrink them back.
# ---------------------------------------------------------------------

BATCH = 2
HEIGHT, WIDTH = 24, 40
MID_CHANNELS = 7
OUT_CHANNELS = 13

_SMALL = dict(
    mid_channels=MID_CHANNELS,
    out_channels=OUT_CHANNELS,
    output_channels=SEG_OUTPUT_CHANNELS,
)

INPUT_SHAPE = (None, HEIGHT, WIDTH, 3)

#: The number of maps ``call`` returns: ``d0`` plus six side outputs.
N_OUTPUTS = 7

#: The repo convention for a value assertion: explicit absolute tolerance, no
#: relative component.
ATOL = 1e-6


def _build(**overrides) -> DocScannerSegmenter:
    """The subject, UNBUILT. Oracles that need a built model call it."""
    return DocScannerSegmenter(**{**_SMALL, **overrides})


def _inputs(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """A DETERMINISTIC input. The oracles call this twice and compare."""
    rng = np.random.default_rng(11)
    return rng.standard_normal((BATCH, height, width, 3)).astype("float32")


def _built(**overrides) -> DocScannerSegmenter:
    model = _build(**overrides)
    model(_inputs())
    return model


def _as_numpy(tensor) -> np.ndarray:
    return keras.ops.convert_to_numpy(tensor)


@pytest.fixture
def built_model() -> DocScannerSegmenter:
    keras.utils.set_random_seed(0)
    return _built()


# ---------------------------------------------------------------------
# 1. The seven-output contract
# ---------------------------------------------------------------------


class TestTheSevenOutputContract:
    """Seven sigmoid maps, all at the INPUT resolution."""

    def test_it_returns_exactly_seven_maps(self, built_model):
        out = built_model(_inputs(), training=False)
        assert isinstance(out, list), type(out)
        assert len(out) == N_OUTPUTS

    def test_every_map_is_at_the_input_resolution(self, built_model):
        out = built_model(_inputs(), training=False)
        for index, tensor in enumerate(out):
            assert tuple(tensor.shape) == (
                BATCH, HEIGHT, WIDTH, SEG_OUTPUT_CHANNELS), (
                f"output {index} has shape {tuple(tensor.shape)}")

    def test_every_map_is_a_probability(self, built_model):
        """The sigmoid is asserted by its RANGE, and non-vacuously: a constant
        0.5 map would also be in ``[0, 1]``, so the spread is checked too."""
        out = [_as_numpy(t) for t in built_model(_inputs(), training=False)]
        for index, array in enumerate(out):
            assert_finite(array)
            assert array.min() >= 0.0, (index, array.min())
            assert array.max() <= 1.0, (index, array.max())
            assert array.std() > 0.0, (
                f"output {index} is CONSTANT; a sigmoid range assertion is "
                f"satisfied by a degenerate map")

    def test_compute_output_shape_reports_all_seven(self, built_model):
        shapes = built_model.compute_output_shape(INPUT_SHAPE)
        assert shapes == [
            (None, HEIGHT, WIDTH, SEG_OUTPUT_CHANNELS)] * N_OUTPUTS

    def test_the_output_channel_count_is_not_hard_coded_to_one(self):
        """``output_channels`` is a knob, so a literal 1 anywhere would show
        up here and nowhere else."""
        model = _built(output_channels=2)
        out = model(_inputs(), training=False)
        for tensor in out:
            assert tuple(tensor.shape) == (BATCH, HEIGHT, WIDTH, 2)

    @pytest.mark.parametrize("height,width", [(37, 53), (17, 9), (33, 31)])
    def test_a_ragged_odd_size_survives_the_whole_ladder(self, height, width):
        """The ceil_mode + resize-onto-the-skip path, end to end.

        288 is divisible by 32, so NOTHING in the shipped resolution exercises
        the remedy step 8 built. These sizes do: at height 37 the six encoder
        levels are 37, 19, 10, 5, 3, 2 and no concatenation would meet without
        both halves of it.
        """
        keras.utils.set_random_seed(0)
        model = _build()
        out = model(_inputs(height, width), training=False)
        assert len(out) == N_OUTPUTS
        for tensor in out:
            assert tuple(tensor.shape) == (
                BATCH, height, width, SEG_OUTPUT_CHANNELS)
            assert_finite(_as_numpy(tensor))

    def test_a_non_4d_build_shape_raises(self):
        with pytest.raises(ValueError, match="4D input shape"):
            _build().build((None, HEIGHT, WIDTH))

    @pytest.mark.parametrize(
        "field", ["mid_channels", "out_channels", "output_channels"])
    def test_a_non_positive_width_raises_naming_the_field(self, field):
        with pytest.raises(ValueError, match=field):
            _build(**{field: 0})


class TestAllSevenOutputsAreDistinct:
    """A forward that returned ``d0`` seven times is shape-perfect.

    So is one that returned the same side map under seven names. The claim is
    checked pairwise on a real input, not by identity of Python objects: two
    distinct objects carrying identical values would be the same defect.
    """

    def test_no_two_outputs_are_equal(self, built_model):
        out = [_as_numpy(t) for t in built_model(_inputs(), training=False)]
        for i in range(N_OUTPUTS):
            for j in range(i + 1, N_OUTPUTS):
                delta = float(np.max(np.abs(out[i] - out[j])))
                assert delta > 1e-5, (
                    f"outputs {i} and {j} are the same map (max abs delta "
                    f"{delta:.3e}). Seven outputs of the right shape is not "
                    f"seven outputs.")


# ---------------------------------------------------------------------
# 2. The spies.
#
# Two of the four claims below are about WHICH TENSOR reached a sub-layer, and
# the wrong tensor has the right shape in both cases. A spy that records the
# actual arrays is the only instrument that can see that; a shape assertion
# cannot, and an output-difference assertion cannot say WHY the output moved.
# ---------------------------------------------------------------------


def _record_inputs(monkeypatch, layer) -> list:
    """Record the first positional argument of every call to ``layer``.

    Interface contract -- three callers below:

    * Parameters: pytest's ``monkeypatch``, and a BUILT sub-layer instance.
    * Returns: a list that fills with ``np.ndarray`` copies, in call order.
    * Failure mode: none of its own; it delegates to the original ``call``.

    The instance's ``call`` is replaced rather than its ``__call__``: Keras 3
    computes the call signature and the training-argument flags once, in
    ``Layer.__init__``, so an instance-level ``call`` override is transparent
    to ``__call__``'s dispatch. The model must already be BUILT before the spy
    is installed -- ``build()`` traces ``call`` on symbolic ``KerasTensor``s,
    which have no values to convert.
    """
    seen: list = []
    original = layer.call

    def spy(*args, **kwargs):
        seen.append(_as_numpy(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(layer, "call", spy)
    return seen


class TestTheSixthSideHeadReadsTheEncoder:
    """``seg.py:546``: ``d6 = self.side6(hx6)`` -- the ENCODER's deepest output.

    There are SIX side heads and FIVE decoder stages. The wrong wiring, feeding
    ``side6`` the deepest decoder output ``hx5d``, is entirely shape-preserving
    downstream: both tensors are ``out_channels`` wide and both are resized onto
    ``d1`` before the fusion.

    Two independent instruments, because each answers a different objection:
    a spy says WHICH array arrived, and a causal perturbation says the
    dependence structure is right even if the spy were mis-installed.
    """

    def test_side6_receives_the_deepest_encoder_stages_output(
            self, monkeypatch, built_model):
        seen_side6 = _record_inputs(monkeypatch, built_model.side_convs[-1])
        seen_stage6 = _record_inputs(
            monkeypatch, built_model.encoder_stages[-1])

        built_model(_inputs(), training=False)

        assert len(seen_side6) == 1
        expected = _as_numpy(
            built_model.encoder_stages[-1](seen_stage6[0], training=False))
        np.testing.assert_allclose(
            seen_side6[0], expected, rtol=0, atol=ATOL,
            err_msg=(
                "side6 did not receive stage6's output. It is wired to a "
                "decoder output -- see the D-022 anchor in model.py."))

    def test_it_is_NOT_the_deepest_decoder_output(
            self, monkeypatch, built_model):
        """Non-vacuity: the arm above only bites if ``hx5d`` differs from
        ``hx6`` on this fixture, which is asserted rather than assumed."""
        seen_side6 = _record_inputs(monkeypatch, built_model.side_convs[-1])
        seen_stage5d = _record_inputs(
            monkeypatch, built_model.decoder_stages[0])

        built_model(_inputs(), training=False)

        hx5d = _as_numpy(
            built_model.decoder_stages[0](seen_stage5d[0], training=False))
        got = seen_side6[0]
        assert got.shape[-1] == hx5d.shape[-1], (
            "the two candidate sources differ in WIDTH on this fixture, so "
            "the swap this test targets would have been caught by a shape "
            "assertion and the test is not measuring what it claims")
        assert got.shape[1:3] != hx5d.shape[1:3] or float(
            np.max(np.abs(got - hx5d))) > 1e-5

    def test_d6_does_not_depend_on_the_deepest_decoder_stage(self, built_model):
        """The causal arm. ``hx6`` is UPSTREAM of ``stage5d``.

        Perturbing ``stage5d``'s weights must leave ``d6`` bit-identical and
        must move ``d5`` -- the second half is the positive control that makes
        the first half non-vacuous.
        """
        before = [_as_numpy(t)
                  for t in built_model(_inputs(), training=False)]

        rng = np.random.default_rng(3)
        stage5d = built_model.decoder_stages[0]
        for weight in stage5d.trainable_weights:
            weight.assign(
                _as_numpy(weight) + rng.standard_normal(
                    weight.shape).astype("float32"))

        after = [_as_numpy(t) for t in built_model(_inputs(), training=False)]

        # d6 is outputs[6]; d5 is outputs[5].
        np.testing.assert_allclose(
            after[6], before[6], rtol=0, atol=0.0,
            err_msg=(
                "d6 MOVED when the deepest decoder stage was perturbed, so "
                "side6 is reading a decoder output rather than hx6."))
        assert float(np.max(np.abs(after[5] - before[5]))) > 1e-5, (
            "the positive control failed: perturbing stage5d did not move d5 "
            "either, so the d6 arm above proves nothing")


class TestTheDecoderConcatenationOrder:
    """``seg.py:517``: ``stage5d(cat((hx6up, hx5), 1))`` -- deeper FIRST.

    Both halves are ``out_channels`` wide, so the swap is shape-preserving. The
    spy splits the tensor the decoder actually received and checks each half
    against the tensor it must be.
    """

    def test_the_first_half_is_the_upsampled_deeper_tensor(
            self, monkeypatch, built_model):
        seen_stage5d = _record_inputs(monkeypatch, built_model.decoder_stages[0])
        seen_stage6 = _record_inputs(monkeypatch, built_model.encoder_stages[-1])
        seen_stage5 = _record_inputs(monkeypatch, built_model.encoder_stages[-2])

        built_model(_inputs(), training=False)

        merged = seen_stage5d[0]
        hx6 = _as_numpy(
            built_model.encoder_stages[-1](seen_stage6[0], training=False))
        hx5 = _as_numpy(
            built_model.encoder_stages[-2](seen_stage5[0], training=False))
        expected_deeper = _as_numpy(
            model_module._upsample_like(
                keras.ops.convert_to_tensor(hx6),
                keras.ops.convert_to_tensor(hx5)))

        assert merged.shape[-1] == 2 * OUT_CHANNELS
        np.testing.assert_allclose(
            merged[..., :OUT_CHANNELS], expected_deeper, rtol=0, atol=ATOL,
            err_msg=("the FIRST half of the decoder's input is not the "
                     "upsampled deeper tensor: the concatenation order is "
                     "swapped (seg.py:517)."))

    def test_the_second_half_is_the_encoder_skip(
            self, monkeypatch, built_model):
        seen_stage5d = _record_inputs(monkeypatch, built_model.decoder_stages[0])
        seen_stage5 = _record_inputs(monkeypatch, built_model.encoder_stages[-2])

        built_model(_inputs(), training=False)

        hx5 = _as_numpy(
            built_model.encoder_stages[-2](seen_stage5[0], training=False))
        np.testing.assert_allclose(
            seen_stage5d[0][..., OUT_CHANNELS:], hx5, rtol=0, atol=ATOL,
            err_msg=("the SECOND half of the decoder's input is not the "
                     "encoder skip."))

    def test_the_two_halves_are_distinguishable_on_this_fixture(
            self, monkeypatch, built_model):
        """Non-vacuity. If the upsampled ``hx6`` happened to equal ``hx5``,
        both arms above would pass under a swapped order."""
        seen_stage5d = _record_inputs(monkeypatch, built_model.decoder_stages[0])
        built_model(_inputs(), training=False)
        merged = seen_stage5d[0]
        delta = float(np.max(np.abs(
            merged[..., :OUT_CHANNELS] - merged[..., OUT_CHANNELS:])))
        assert delta > 1e-5, (
            f"the two concatenated halves are numerically identical "
            f"(max abs delta {delta:.3e}); an order guard cannot bite")


class TestTheFusionIsALearnedCombinationOfAllSix:
    """``seg.py:548``: ``d0 = outconv(cat((d1, ..., d6), 1))``.

    A ``d0`` that is a copy of ``d1`` -- the single most likely fusion bug --
    is shape-perfect and even plausible-looking, because ``d1`` is the map at
    the reference resolution.
    """

    def test_d0_is_not_a_copy_of_any_side_map(self, built_model):
        out = [_as_numpy(t) for t in built_model(_inputs(), training=False)]
        for index in range(1, N_OUTPUTS):
            assert float(np.max(np.abs(out[0] - out[index]))) > 1e-5, (
                f"d0 equals d{index}: the fusion is a pass-through")

    def test_the_fusion_reads_all_six_channels(self, monkeypatch, built_model):
        """Structural: the tensor ``outconv`` receives is ``6 * output_channels``
        wide, and its kernel is that wide too."""
        seen = _record_inputs(monkeypatch, built_model.outconv)
        built_model(_inputs(), training=False)
        assert seen[0].shape[-1] == 6 * SEG_OUTPUT_CHANNELS
        kernel = _as_numpy(built_model.outconv.kernel)
        assert kernel.shape == (1, 1, 6 * SEG_OUTPUT_CHANNELS,
                                SEG_OUTPUT_CHANNELS)

    @pytest.mark.parametrize("selected", [0, 3, 5])
    def test_d0_MOVES_when_any_one_of_the_six_moves(
            self, built_model, selected):
        """The causal arm, one probe per named side head.

        Perturbing side head ``selected`` must move ``d0``. A fusion that read
        only ``d1`` would pass for ``selected == 0`` and fail for the rest,
        which is exactly the shape-perfect defect this class exists for.
        """
        before = _as_numpy(built_model(_inputs(), training=False)[0])

        rng = np.random.default_rng(7)
        conv = built_model.side_convs[selected]
        for weight in conv.trainable_weights:
            weight.assign(
                _as_numpy(weight) + rng.standard_normal(
                    weight.shape).astype("float32"))

        after = _as_numpy(built_model(_inputs(), training=False)[0])
        assert float(np.max(np.abs(after - before))) > 1e-5, (
            f"d0 did not move when side{selected + 1} was perturbed: the "
            f"fusion is not reading all six side maps")


class TestTheDecoderConsumesTwiceTheSkipWidth:
    """``seg.py:473-477``: every decoder RSU is declared ``RSU*(128, 16, 64)``.

    128 is ``2 * seg_out_channels``, and this port DERIVES it rather than
    passing it: the RSU blocks infer their input width from the tensor. So the
    claim is asserted on the BUILT weight, which is where a wrong derivation
    would actually show up.
    """

    def test_every_decoder_stems_kernel_is_twice_the_skip_width(
            self, built_model):
        for index, stage in enumerate(built_model.decoder_stages):
            kernel = _as_numpy(stage.rebnconvin.conv.kernel)
            assert kernel.shape[2] == 2 * OUT_CHANNELS, (
                f"decoder stage {index} ({stage.name}) reads "
                f"{kernel.shape[2]} channels, expected {2 * OUT_CHANNELS}")

    def test_every_encoder_stem_after_the_first_reads_one_skip_width(
            self, built_model):
        """The contrast that makes the number above mean something."""
        for index, stage in enumerate(built_model.encoder_stages[1:], start=1):
            kernel = _as_numpy(stage.rebnconvin.conv.kernel)
            assert kernel.shape[2] == OUT_CHANNELS, (
                f"encoder stage {index + 1} reads {kernel.shape[2]} channels")


class TestTheLadderIsTheOneSegPyDeclares:
    """The stage classes and their order, ``seg.py:456-477``."""

    def test_the_encoder_classes_are_RSU7_6_5_4_4F_4F(self, built_model):
        assert [type(s) for s in built_model.encoder_stages] == [
            RSU7, RSU6, RSU5, RSU4, RSU4F, RSU4F]

    def test_the_decoder_classes_mirror_it_deepest_first(self, built_model):
        assert [type(s) for s in built_model.decoder_stages] == [
            RSU4F, RSU4, RSU5, RSU6, RSU7]

    def test_there_are_five_pools_for_six_encoder_stages(self, built_model):
        assert len(built_model.pools) == 5
        assert len(built_model.encoder_stages) == 6

    def test_there_are_six_side_heads_and_five_decoder_stages(
            self, built_model):
        """The asymmetry, stated as a fact rather than left to be noticed."""
        assert len(built_model.side_convs) == 6
        assert len(built_model.decoder_stages) == 5

    def test_the_side_heads_are_3x3_and_the_fusion_is_1x1(self, built_model):
        for conv in built_model.side_convs:
            assert conv.kernel_size == (3, 3), conv.name
            assert _as_numpy(conv.kernel).shape[2] == OUT_CHANNELS
        assert built_model.outconv.kernel_size == (1, 1)

    def test_the_pools_carry_the_ceil_mode_remedy(self, built_model):
        """D-019's ``padding="same"``, read off the constructed layers.

        ``"valid"`` here would change the ladder at an odd size and nothing
        about the output shape.
        """
        for pool in built_model.pools:
            assert pool.padding == "same", pool.name
            assert pool.pool_size == (2, 2)
            assert pool.strides == (2, 2)


# ---------------------------------------------------------------------
# 3. Serialization, gradients, build
# ---------------------------------------------------------------------


def _contract(output) -> None:
    """The forward contract. Shared with the meta-test so it is falsifiable."""
    assert isinstance(output, list), (
        f"DocScannerSegmenter returns a list of seven maps, got "
        f"{type(output)}")
    assert len(output) == N_OUTPUTS, len(output)
    for tensor in output:
        assert tuple(tensor.shape) == (
            BATCH, HEIGHT, WIDTH, SEG_OUTPUT_CHANNELS), tuple(tensor.shape)
        array = keras.ops.convert_to_numpy(tensor)
        assert array.min() >= 0.0 and array.max() <= 1.0


class TestTheSharedOracleAdoptions:
    """One arm per oracle. Nothing here is re-implemented locally."""

    def test_the_round_trip_reproduces_the_output_values_exactly(self):
        report = measure_roundtrip(_build, _inputs, training=False)
        assert report["self_max_delta"] == 0.0, (
            f"DocScannerSegmenter is not deterministic (self spread "
            f"{report['self_max_delta']:.6e})")
        assert_roundtrip_output_values(report, atol=0.0)

    def test_the_weights_are_restored_before_the_loaded_model_is_called(self):
        report = measure_roundtrip(_build, _inputs, training=False)
        assert report["call_count_before_weight_read"] == 0
        assert_weights_restored_before_first_call(report, atol=0.0)

    def test_the_lazy_build_costs_nothing(self):
        report = assert_lazy_build_costs_nothing(
            _build, _inputs, input_shape=INPUT_SHAPE, atol=0.0)
        assert report["n_weights"] == report["n_weights_reloaded"]
        assert report["perturb_liveness"] > 0.0

    def test_the_explicit_build_matches_the_lazy_build_by_weight_path(self):
        """The H-6 claim made measurable: ``build()`` calls
        ``materialize_sublayers``, and if it did not, the explicit path would
        materialize nothing and the two weight-path sets would differ."""
        report = measure_build_parity(_build, _inputs, input_shape=INPUT_SHAPE)
        assert_build_parity(report, autoname_stems=(),
                            expect_path_collisions=0)

    def test_gradients_reach_every_trainable_weight(self):
        """Run at ``training=False``, deliberately, and on a WIDE ENOUGH stand-in.

        Two independent constraints meet on this one arm, and both were
        measured rather than chosen:

        * **The mode.** Step 8 measured (D-021) that ``REBNCONV``'s conv bias is
          mathematically DEAD under training-mode batch normalization -- the
          norm subtracts the very channel shift the bias applied, so the
          analytic gradient is exactly zero and what a tape reports is float32
          cancellation noise (one weight read an exact 0.0 and the arm FLAKED).
          Under moving statistics the same biases are live. At
          ``training=True`` the assertion would be false of the ARCHITECTURE,
          not of the port.
        * **The width.** At ``training=False`` and at init the batch norm is an
          identity (moving mean 0, moving variance 1), so a narrow REBNCONV
          deep in the ladder can have EVERY channel negative and its ReLU dies.
          That is what forced 7/13; see ``TestTheFixtureIsNotDegenerate``,
          which asserts the precondition directly rather than leaving this arm
          to fail as if the port were at fault.
        """
        keras.utils.set_random_seed(0)
        model = _built()
        report = assert_gradients_reach_every_trainable_weight(
            model, _inputs(), training=False)
        assert len(report) == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        """RED proof for the arm above, using the oracle's OWN injection."""
        keras.utils.set_random_seed(0)
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _inputs(), training=False)

    def test_the_forward_satisfies_its_contract(self, built_model):
        _contract(built_model(_inputs(), training=False))

    def test_the_smoke_contract_rejects_a_broken_forward(self, built_model):
        """The oracle requires ``AssertionError`` SPECIFICALLY."""
        messages = assert_contract_rejects_a_broken_forward(
            built_model, _inputs(), _contract)
        assert messages


class TestTheFixtureIsNotDegenerate:
    """The gradient arm's PRECONDITION, asserted rather than assumed.

    At ``training=False`` and at initialization a Keras ``BatchNormalization``
    is an identity: its moving mean is 0 and its moving variance is 1, so
    ``REBNCONV`` reduces to ``conv -> relu``. Deep in the ladder the spatial
    map is 1x2 and -- in ``RSU4F``'s dilation-8 convolution -- every off-centre
    tap of the 3x3 kernel lands in the zero padding, so the pre-ReLU tensor has
    very few independent values. With a narrow stand-in they can all be
    negative, the ReLU output is identically zero, and every weight behind it
    reads an identically-zero gradient.

    That is a property of running a /32 ladder on a 24-pixel fixture, NOT a
    defect in this port: the shipped 16/64 widths do not show it, and neither
    does ``training=True``. It is recorded here so the width choice above is a
    measurement with a stated mechanism rather than a number that happened to
    make a test green.
    """

    @pytest.mark.parametrize("stage_name", ["stage6", "stage5d"])
    def test_the_deepest_convolutions_relu_is_alive(
            self, monkeypatch, built_model, stage_name):
        """Directly on the ACTIVATIONS, so it cannot be satisfied by luck in
        the tape.

        The probe targets ``RSU4F``'s dilation-8 encoder convolution -- the
        narrowest, deepest, most padding-dominated tensor in the model, and the
        one that died first at 3/5. The block's own OUTPUT is NOT probed: the
        RSU adds an outer residual, so a completely dead ladder still emits a
        non-zero block output.
        """
        stage = {
            "stage6": built_model.encoder_stages[-1],
            "stage5d": built_model.decoder_stages[0],
        }[stage_name]
        deepest = stage.encoder_convs[-1]

        seen = _record_inputs(monkeypatch, deepest)
        built_model(_inputs(), training=False)
        assert len(seen) == 1

        activation = _as_numpy(deepest(seen[0], training=False))
        assert float(np.max(activation)) > 0.0, (
            f"{stage_name}'s deepest REBNCONV emits an identically-zero ReLU "
            f"on this fixture. The stand-in widths are too narrow -- see this "
            f"class's docstring. Widen them; do not waive the gradient arm.")

    def test_the_gradient_arms_fixture_has_no_dead_weight(self):
        """The same claim the gradient arm makes, stated as the fixture's
        property: if this fails, widen the stand-in -- do not waive."""
        keras.utils.set_random_seed(0)
        model = _built()
        report = assert_gradients_reach_every_trainable_weight(
            model, _inputs(), training=False)
        zero = [path for path, value in report.items() if value == 0.0]
        assert not zero, (
            f"{len(zero)} weight(s) read an identically-zero gradient on the "
            f"stand-in fixture. This is the dying-ReLU degeneracy documented "
            f"on this class, not a port defect: widen MID_CHANNELS / "
            f"OUT_CHANNELS and re-measure. First offenders: {zero[:4]}")


class TestTheSymbolicBuildTrace:
    """Rule 6 of this plan: anything reachable from ``keras.Model.build()``
    must be exercised on the SYMBOLIC path, not merely eagerly.

    Step 7's D-018 is exactly this failure: ``extract_patches`` was correct
    eagerly and raised ``UnboundLocalError`` under a symbolic trace, and 158
    tests were green because none of them had built a ``keras.Model``.
    """

    def test_build_traces_call_on_kerastensors_and_materializes_everything(
            self):
        model = _build()
        model.build(INPUT_SHAPE)
        assert model.built
        for stage in model.encoder_stages + model.decoder_stages:
            assert stage.built, stage.name
        for conv in model.side_convs:
            assert conv.built, conv.name
        assert model.outconv.built

    def test_call_runs_on_a_purely_symbolic_input(self):
        """The trace itself, invoked directly rather than trusted."""
        model = _build()
        outputs = model.call(keras.KerasTensor(INPUT_SHAPE))
        assert len(outputs) == N_OUTPUTS
        for tensor in outputs:
            assert isinstance(tensor, keras.KerasTensor)
            assert tuple(tensor.shape) == (
                None, HEIGHT, WIDTH, SEG_OUTPUT_CHANNELS)

    def test_a_functional_graph_can_be_built_over_it(self):
        """The strongest form of the symbolic claim: a real functional model."""
        model = _build()
        inputs = keras.Input(shape=(HEIGHT, WIDTH, 3))
        wrapper = keras.Model(inputs, model(inputs))
        out = wrapper(_inputs())
        assert len(out) == N_OUTPUTS


class TestARealSaveAndLoad:
    """``model.save()`` / ``load_model()`` through ``tmp_path``.

    NEVER repo-root ``results/`` -- that tree is gitignored, untracked and
    therefore unrecoverable, and an autouse fixture asserts no test writes
    into it.
    """

    def test_a_real_save_and_load_model_reproduces_the_values(self, tmp_path):
        keras.utils.set_random_seed(0)
        model = _built()
        x = _inputs()
        before = [_as_numpy(t) for t in model(x, training=False)]

        path = tmp_path / "segmenter.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)

        after = [_as_numpy(t) for t in reloaded(x, training=False)]
        assert len(after) == N_OUTPUTS
        for index, (lhs, rhs) in enumerate(zip(before, after)):
            np.testing.assert_allclose(
                rhs, lhs, rtol=0, atol=ATOL,
                err_msg=f"output {index} changed across a real save/load")

    def test_get_config_round_trips_every_constructor_argument(self):
        model = _build()
        config = model.get_config()
        for key, value in _SMALL.items():
            assert config[key] == value, key
        clone = DocScannerSegmenter.from_config(config)
        assert clone.get_config() == config


# ---------------------------------------------------------------------
# 4. The variant table
# ---------------------------------------------------------------------


class TestTheSegmenterVariantTable:
    """The projection rule of D-023, checked from BOTH sides."""

    def test_there_is_exactly_one_row_and_it_is_docscanner_l(self):
        assert list(DocScannerSegmenter.MODEL_VARIANTS) == ["docscanner-l"]

    def test_every_seg_prefixed_spec_key_reaches_the_row(self):
        from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
            _VARIANT_SPEC,
        )
        spec = _VARIANT_SPEC["docscanner-l"]
        row = DocScannerSegmenter.MODEL_VARIANTS["docscanner-l"]
        # The prefix is a LITERAL here, never read from the module's own
        # `_SEG_SPEC_KEY_PREFIX`: reading it would move both sides together and
        # this would stop measuring anything.
        seg_keys = [key for key in spec if key.startswith("seg_")]
        assert seg_keys, "the spec carries no seg_* keys at all"
        for key in seg_keys:
            assert row[key[len("seg_"):]] == spec[key], key

    def test_no_rectifier_key_leaks_into_the_row(self):
        """The inclusion rule's whole point. Step 8 broke three of the
        rectifier's tests by adding two keys to the shared spec; an inclusion
        rule cannot fail that way, and this is what says so."""
        from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
            _VARIANT_SPEC,
        )
        spec = _VARIANT_SPEC["docscanner-l"]
        row = DocScannerSegmenter.MODEL_VARIANTS["docscanner-l"]
        for key in spec:
            if not key.startswith("seg_"):
                assert key not in row, (
                    f"the rectifier width '{key}' leaked into the segmenter's "
                    f"variant row")

    def test_the_row_carries_the_widths_seg_py_declares(self):
        row = DocScannerSegmenter.MODEL_VARIANTS["docscanner-l"]
        assert row["mid_channels"] == 16
        assert row["out_channels"] == 64
        assert row["output_channels"] == 1

    def test_the_row_is_a_valid_constructor_call(self):
        row = dict(DocScannerSegmenter.MODEL_VARIANTS["docscanner-l"])
        row.pop("description")
        model = DocScannerSegmenter(**row)
        assert model.mid_channels == 16

    def test_the_shipped_variant_builds_and_runs(self):
        """The one arm that exercises the REAL 16/64 widths, at a small
        resolution so it costs seconds rather than minutes."""
        model = create_doc_scanner_segmenter()
        out = model(np.zeros((1, 32, 24, 3), dtype="float32"), training=False)
        assert len(out) == N_OUTPUTS
        assert tuple(out[0].shape) == (1, 32, 24, 1)

    def test_pretrained_true_raises_naming_the_missing_checkpoint(self):
        with pytest.raises(NotImplementedError) as excinfo:
            DocScannerSegmenter.from_variant("docscanner-l", pretrained=True)
        message = str(excinfo.value)
        assert "seg.pth" in message
        assert "k[6:]" in message

    def test_the_factory_function_raises_too(self):
        with pytest.raises(NotImplementedError):
            create_doc_scanner_segmenter(pretrained=True)

    def test_an_unknown_variant_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="docscanner-l"):
            DocScannerSegmenter.from_variant("u2net-full")

    def test_from_variant_overrides_reach_the_constructor(self):
        model = DocScannerSegmenter.from_variant(
            "docscanner-l", mid_channels=4)
        assert model.mid_channels == 4
        assert model.out_channels == 64


class TestRegistration:
    """H-3: the key strips the family AND the subfamily."""

    def test_the_key_strips_both_the_family_and_the_subfamily(self):
        assert keras.saving.get_registered_name(DocScannerSegmenter) == (
            "dl_techniques.models.doc_scanner.model>DocScannerSegmenter")

    def test_the_full_import_path_is_NOT_a_registration_key(self):
        assert keras.saving.get_registered_object(
            "dl_techniques.models.vision.image_restoration.doc_scanner.model"
            ">DocScannerSegmenter") is None


# ---------------------------------------------------------------------
# 5. The parameter count, against an INDEPENDENT arithmetic reference
# ---------------------------------------------------------------------


def _rebnconv_params(in_ch: int, out_ch: int) -> tuple:
    """``(trainable, non_trainable)`` of one ``seg.py:34-46`` REBNCONV.

    Interface contract -- called by :func:`_rsu_params` only:

    * Parameters: the block's input and output widths.
    * Returns: a 2-tuple of ints.
    * Failure mode: none.

    ``nn.Conv2d(in, out, 3, ...)`` with ``bias=True`` is ``9 * in * out + out``;
    ``nn.BatchNorm2d(out)`` is ``2 * out`` learnable (weight, bias) plus
    ``2 * out`` buffers (running mean, running var). Keras counts the buffers
    as non-trainable weights, torch as buffers, which is why the two are
    reported separately rather than summed.
    """
    return (9 * in_ch * out_ch + out_ch + 2 * out_ch, 2 * out_ch)


def _rsu_params(in_ch: int, mid_ch: int, out_ch: int, levels: int,
                flat: bool) -> tuple:
    """``(trainable, non_trainable)`` of one RSU block, ``seg.py:57-343``.

    Interface contract -- called by :func:`_reference_parameter_counts` only:

    * Parameters: the three widths, the encoder level count, and whether the
      block is the pooling-free ``RSU4F`` (which has no separate bottom
      convolution -- its dilation-8 encoder convolution IS the bottom).
    * Returns: a 2-tuple of ints.
    * Failure mode: none.

    Transcribed from ``seg.py``'s ``__init__`` bodies, NOT from this port. A
    count derived from the port would agree with the port by construction.
    """
    blocks = [(in_ch, out_ch)]                    # rebnconvin
    blocks.append((out_ch, mid_ch))               # rebnconv1
    blocks += [(mid_ch, mid_ch)] * (levels - 1)   # rebnconv2..N
    if not flat:
        blocks.append((mid_ch, mid_ch))           # the dilation-2 bottom
        decoders = levels
    else:
        decoders = levels - 1
    blocks += [(2 * mid_ch, mid_ch)] * (decoders - 1)
    blocks.append((2 * mid_ch, out_ch))           # rebnconv1d

    trainable = sum(_rebnconv_params(a, b)[0] for a, b in blocks)
    non_trainable = sum(_rebnconv_params(a, b)[1] for a, b in blocks)
    return (trainable, non_trainable)


def _reference_parameter_counts(in_ch: int, mid_ch: int, out_ch: int,
                                output_ch: int) -> tuple:
    """``(trainable, non_trainable)`` of the whole ``U2NETP``, ``seg.py:451-486``."""
    stages = [
        (in_ch, 6, False),           # stage1  RSU7
        (out_ch, 5, False),          # stage2  RSU6
        (out_ch, 4, False),          # stage3  RSU5
        (out_ch, 3, False),          # stage4  RSU4
        (out_ch, 4, True),           # stage5  RSU4F
        (out_ch, 4, True),           # stage6  RSU4F
        (2 * out_ch, 4, True),       # stage5d RSU4F
        (2 * out_ch, 3, False),      # stage4d RSU4
        (2 * out_ch, 4, False),      # stage3d RSU5
        (2 * out_ch, 5, False),      # stage2d RSU6
        (2 * out_ch, 6, False),      # stage1d RSU7
    ]
    trainable = 0
    non_trainable = 0
    for stage_in, levels, flat in stages:
        stage_trainable, stage_non_trainable = _rsu_params(
            stage_in, mid_ch, out_ch, levels, flat)
        trainable += stage_trainable
        non_trainable += stage_non_trainable

    # side1..side6 = Conv2d(out_ch, output_ch, 3, padding=1)
    trainable += 6 * (9 * out_ch * output_ch + output_ch)
    # outconv = Conv2d(6 * output_ch, output_ch, 1)
    trainable += 6 * output_ch * output_ch + output_ch
    return (trainable, non_trainable)


class TestTheParameterCount:
    """Counted against an arithmetic transcription of ``seg.py``, not a
    number copied out of this port.

    A parameter count is one of the few instruments that sees a wrong WIDTH
    anywhere in an eleven-block ladder: every shape test in this file passes at
    any uniform mid/out pair.
    """

    def test_the_small_fixture_matches_the_reference_arithmetic(self):
        model = _built()
        expected_trainable, expected_non_trainable = _reference_parameter_counts(
            3, MID_CHANNELS, OUT_CHANNELS, SEG_OUTPUT_CHANNELS)
        trainable = int(sum(
            np.prod(w.shape) for w in model.trainable_weights))
        non_trainable = int(sum(
            np.prod(w.shape) for w in model.non_trainable_weights))
        assert trainable == expected_trainable
        assert non_trainable == expected_non_trainable

    def test_the_shipped_widths_match_the_reference_arithmetic(self):
        model = create_doc_scanner_segmenter()
        model.build((None, 32, 32, 3))
        expected_trainable, expected_non_trainable = _reference_parameter_counts(
            3, 16, 64, 1)
        trainable = int(sum(
            np.prod(w.shape) for w in model.trainable_weights))
        non_trainable = int(sum(
            np.prod(w.shape) for w in model.non_trainable_weights))
        assert trainable == expected_trainable
        assert non_trainable == expected_non_trainable
        # The published U2NET-P size, recorded as an OBSERVATION. The paper's
        # 8.5M for DocScanner-L is a two-stage total and is discussed in the
        # package README rather than asserted anywhere: 8.5M is quoted to two
        # significant figures, so it cannot be a pass/fail gate.
        assert model.count_params() == trainable + non_trainable
