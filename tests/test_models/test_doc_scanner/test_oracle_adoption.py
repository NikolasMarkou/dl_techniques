"""The three shared oracles the rest of this test directory does not yet carry.

Five of the eight shared instruments were adopted as each class was written:
``roundtrip_instrument_oracle``, ``gradient_flow_oracle``,
``lazy_build_contract_oracle`` and ``smoke_contract_oracle`` in all three of
``test_model.py`` / ``test_segmenter.py`` / ``test_composite.py``, and
``knob_sensitivity_oracle`` in ``test_model.py`` alone. This module adds what
was still owed and nothing else:

* ``delta_impulse_orientation_oracle`` -- the END-TO-END orientation of the
  coordinate pipeline, which no unit test in ``test_components.py`` can make.
  The primitives (``coords_grid``'s ``(x, y)`` order, the half-pixel
  convention, ``convex_upsample``'s ``(kh, kw, c)`` patch layout) are each
  pinned where they live; what is pinned HERE is that the assembly of them
  still moves the response the same way the input moved. This repo's own
  history is the argument for the instrument: a sign error in an ``ops.roll``
  survived 249 tests because every one of them looked at values or shapes and
  none looked at direction.
* ``precision_arm_oracle`` -- ``mixed_float16`` (four parts), ``float64`` and
  eager-vs-XLA, on all three classes.
* ``knob_sensitivity_oracle`` -- on the two classes that did not have it, with
  the structural/value split the oracle's own docstring requires.

Why a module of its own rather than three edits to three files
--------------------------------------------------------------
All three instruments are cross-cutting (they judge every class the same way),
and the impulse probe needs a probe grid four times larger than the 32x24
fixture the other modules share -- see ``IMPULSE_HEIGHT`` below for why it
cannot simply reuse it. Keeping them together also keeps the adoption matrix
readable in one place instead of spread across three 800-line modules.

The stand-in widths are IMPORTED from ``test_composite.py`` rather than
restated. A fourth hand-maintained copy of the same dict is a thing to keep in
lockstep by hand; ``test_composite.py`` already holds both stages' widths and
already documents why those particular numbers (D-024's measured
non-degenerate 7 / 13 pair) were chosen.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.image_restoration.doc_scanner.model import (
    DocScanner,
    DocScannerRectifier,
    DocScannerSegmenter,
)

from ..delta_impulse_orientation_oracle import (
    DEFAULT_MIN_RATIO,
    assert_orientation_is_diagonal,
    centroid_response_matrix,
    impulse_energy_map,
    transposed_stride_injection,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..precision_arm_oracle import (
    assert_float64_arm,
    assert_precision_arm,
    assert_xla_equivalence,
    run_backward,
)
from .test_composite import _REC, _SEG

# ---------------------------------------------------------------------
# Subjects. `_REC` and `_SEG` are `test_composite.py`'s stand-in widths,
# imported (see the module docstring). `iters` is overridden per probe: the
# impulse probe pays for `iters` full forwards per impulse and three impulses
# per response matrix, so it runs the loop the smallest number of times that
# still reaches every stage of the coordinate pipeline -- see IMPULSE_ITERS.
# ---------------------------------------------------------------------

BATCH = 2
HEIGHT, WIDTH = 32, 24          # NON-SQUARE, both multiples of 8


def _rectifier(**overrides: Any) -> DocScannerRectifier:
    return DocScannerRectifier(**{**_REC, **overrides})


def _segmenter(**overrides: Any) -> DocScannerSegmenter:
    return DocScannerSegmenter(**{**_SEG, **overrides})


def _composite(
        segmenter_overrides: Dict[str, Any] = None,
        rectifier_overrides: Dict[str, Any] = None,
) -> DocScanner:
    return DocScanner(
        segmenter_config={**_SEG, **(segmenter_overrides or {})},
        rectifier_config={**_REC, **(rectifier_overrides or {})},
    )


def _inputs() -> np.ndarray:
    """A deterministic input in ``[0, 1]`` -- the documented domain.

    Interface contract, because every arm below calls it:

    * Parameters: none.
    * Returns: ``(BATCH, HEIGHT, WIDTH, 3)`` float32, the SAME values on every
      call (the precision and knob oracles compare across calls, so a fresh
      draw would make them compare two different inputs).
    * Failure mode: none.
    """
    return np.random.RandomState(0).rand(
        BATCH, HEIGHT, WIDTH, 3).astype("float32")


# =====================================================================
# 1. delta_impulse_orientation_oracle -- the end-to-end warp direction
# =====================================================================
#
# The probe grid is 128 x 96, not the 32 x 24 the rest of this directory uses,
# and the reason is arithmetic rather than taste. The instrument shifts the
# impulse by one full stride step and measures how far the response centroid
# moves; at 32 x 24 the 1/8-resolution field the update block operates on is
# 4 x 3 cells, so a single 5-tap separable GRU pass already spans the whole of
# it and there is nothing left for a shift to be measured against. 128 x 96
# gives a 16 x 12 field and a step of 48 input pixels (6 cells).
#
# `iters` is 2, not 1, and that is load-bearing rather than a cost compromise.
# `sample_at_pixel_coords` is called at the END of each loop step, so at
# `iters=1` its result is computed and never read: a probe run at one iteration
# leaves the whole feature-resampling half of the coordinate pipeline outside
# what it can see. Two iterations is the smallest count that puts it inside.
#
# MEASURED at this grid and this iteration count, and this is the calibration
# for MIN_RATIO below. Over seeds 0..11 the diagonal-to-off-diagonal ratio of
# the centroid response matrix is:
#
#     real       2.264  2.756  9.792  10.448  11.900  23.932
#                24.690 43.577 49.741 59.456  105.157 136.588
#     injected   0.0027 0.0032 0.0037 0.0055  0.0075  0.0081
#                0.0113 0.0193 0.0242 0.0292  0.0341  0.0375
#
# with the DIAGONAL POSITIVE in all twelve real runs and the smallest diagonal
# entry 6.87. The two populations are separated by 60x (2.264 against 0.0375)
# and MIN_RATIO is placed at their geometric middle, 0.3: 7.5x below the
# weakest real reading and 8x above the strongest injected one.
SEEDS = (0, 1, 11)              # 0 by convention; 1 and 11 are the two
                                # WEAKEST measured real readings, so the arm is
                                # pinned to its worst case and not to a lucky
                                # seed.
IMPULSE_HEIGHT, IMPULSE_WIDTH = 128, 96
IMPULSE_BASE = (32, 24)
IMPULSE_STEP = 48
IMPULSE_ITERS = 2

# DECISION plan-2026-09-10T065432-05fcb6dd/D-028: 0.3, not the oracle's
# DEFAULT_MIN_RATIO of 20. Do NOT "restore" the default: this subject is not a
# stride path (the nine paths that calibrated 20 all were), and at 20 the arm
# is RED on a CORRECT model at 4 of 12 seeds. Do NOT lower it further either --
# 0.3 is the geometric middle of the two measured populations above, and below
# ~0.04 the oracle's own transposed-stride injection stops being RED, which
# would leave a probe that cannot fail for the reason it exists. Any change
# here has to be re-measured, not re-argued. See decisions.md D-028, including
# its table of the three real coordinate-pipeline defects this arm is MEASURED
# blind to.
MIN_RATIO = 0.3


def _impulse_subject(seed: int) -> DocScannerRectifier:
    keras.utils.set_random_seed(seed)
    model = _rectifier(iters=IMPULSE_ITERS)
    model.build((None, IMPULSE_HEIGHT, IMPULSE_WIDTH, 3))
    return model


def _impulse_forward(model: Any):
    return lambda x: model(x, training=False)


class TestTheEndToEndCoordinatePipelineIsOriented:
    """The assembly, not the primitives.

    ``DocScannerRectifier``'s forward IS the whole coordinate pipeline:
    ``coords_grid`` at two resolutions, the accumulation
    ``coords1 = coords1 + delta_flow``, ``convex_upsample`` of
    ``coords1 - coords0`` and ``sample_at_pixel_coords`` of the encoder's
    features. A flipped sign or a transposed axis anywhere along it leaves
    every shape, every parameter count, every round trip and every finiteness
    assertion in this directory green.

    WHAT THIS ARM DOES NOT CATCH, measured rather than assumed. Four real
    mutations were applied in place to ``warp.py`` and the tracked file
    restored sha256-identical afterwards; the counts are the arms of THIS class
    against the arms of ``test_components.py``:

    ===================================================  =========  ==========
    mutation                                             this arm   components
    ===================================================  =========  ==========
    ``coords_grid`` emits ``(y, x)`` (D-009 #1)          8 passed   RED
    the sampler drops the x/y swap (D-009 #1)            8 passed   RED
    convex interleave ``i <-> j`` (D-010 #3)             8 passed   1 RED
    the 576 split reversed to ``(8, 8, 9)`` (D-010 #2)   7 RED      17 RED
    ===================================================  =========  ==========

    So this arm is NOT a replacement for the unit guards and must not be read
    as one: three of the four defects it is natural to expect it to catch are
    invisible to it, because a centroid statistic sees WHERE the response
    landed and those three change WHICH VALUE landed there (a coordinate-channel
    swap re-samples the features from the wrong place, and an 8x8 block
    transpose moves mass by at most 7 sub-pixels, both of which are far below
    the diffuse response's centroid resolution). What it adds is the claim no
    unit test can make -- that the whole assembly's response still moves WITH
    the input, at full scale -- and it is RED against a whole-map axis swap and
    against the 576-split reversal. Do NOT delete a ``test_components.py``
    ordering guard on the strength of this class.

    Why the default ``min_ratio`` of 20 is not used here is argued at
    :data:`MIN_RATIO`: this subject is not a pure stride path. Its
    ``_instance_norm`` (D-011) normalizes over the WHOLE spatial map, so every
    output position depends on every input position and the impulse support is
    the entire map -- measured ``frac_nonzero = 1.000``. That is precisely the
    case the oracle's docstring routes to the centroid statistic rather than to
    a support box, but the diffuse floor it creates drags the centroid toward
    the image centre and shrinks the measured slope.
    """

    def test_the_default_min_ratio_is_still_the_one_this_arm_deviates_from(self):
        """A rename or a re-tune of the oracle's default must reach this file.

        ``MIN_RATIO`` is a DEVIATION, justified against a measured population.
        If the oracle's own default moves, that justification has to be re-read
        rather than silently kept.
        """
        assert DEFAULT_MIN_RATIO == 20.0
        assert MIN_RATIO < DEFAULT_MIN_RATIO

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_row_shift_moves_the_response_down_and_a_column_shift_right(
            self, seed):
        model = _impulse_subject(seed)
        matrix = assert_orientation_is_diagonal(
            _impulse_forward(model),
            (IMPULSE_HEIGHT, IMPULSE_WIDTH, 3),
            IMPULSE_BASE,
            IMPULSE_STEP,
            label=f"DocScannerRectifier end-to-end (seed {seed})",
            min_ratio=MIN_RATIO,
        )
        # The oracle asserts the sign and the ratio; this pins the SCALE too.
        # A response that moved by a thousandth of a pixel would satisfy
        # "positive and diagonal" and would mean the shift never propagated.
        assert min(matrix[0, 0], matrix[1, 1]) > 3.0, matrix

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_probe_is_red_under_a_transposed_stride(self, seed):
        """The oracle's own dead-component injection, at every pinned seed.

        Without this the arm above is a test that has never been seen fail.
        """
        model = _impulse_subject(seed)
        with pytest.raises(AssertionError):
            assert_orientation_is_diagonal(
                transposed_stride_injection(_impulse_forward(model)),
                (IMPULSE_HEIGHT, IMPULSE_WIDTH, 3),
                IMPULSE_BASE,
                IMPULSE_STEP,
                label=f"DocScannerRectifier transposed (seed {seed})",
                min_ratio=MIN_RATIO,
            )

    def test_a_square_probe_grid_is_refused(self):
        """R-140's discriminating condition, asserted at this call site.

        The grid above is non-square deliberately. This arm fails if anybody
        "simplifies" it to a square one, which would make the two arms above
        pass while measuring nothing.
        """
        model = _impulse_subject(0)
        with pytest.raises(AssertionError, match="SQUARE"):
            centroid_response_matrix(
                _impulse_forward(model),
                (IMPULSE_HEIGHT, IMPULSE_HEIGHT, 3),
                IMPULSE_BASE,
                IMPULSE_STEP,
                label="square control",
            )


class TestWhyTheCompositeDoesNotCarryTheImpulseProbe:
    """The instrument does not fit :class:`DocScanner`, and the reason is D-025.

    :func:`impulse_energy_map` is exact because it subtracts ``f(0)``: outside
    the impulse's receptive field the two inputs are IDENTICAL and the two
    outputs cancel bit for bit. That all-zero baseline is what the composite's
    hard threshold destroys. At zero input every side head sees a zero
    activation and a zero-initialized bias, so the confidence map is EXACTLY
    ``0.5`` everywhere (measured, all three seeds below); ``conf > 0.5`` is
    then false everywhere, the mask is all zeros, and the rectifier receives an
    all-zero tensor whether or not the impulse was there. Whether any response
    survives at all depends on whether the single impulse pixel happens to push
    its own confidence above the threshold -- MEASURED at the 128 x 96 grid,
    total response energy 128.9 at seed 0, 171.6 at seed 2, and EXACTLY 0.0 at
    seed 1, where the oracle raises "the impulse never reached the output".

    A probe whose subject is alive at two seeds and dead at a third is not an
    instrument, and the fix is not to loosen it: the composite adds NO spatial
    operation over the rectifier. Its four lines are a threshold, an
    elementwise product and an affine calibration, none of which can transpose
    an axis or flip a coordinate sign. The orientation claim pinned on the
    rectifier above therefore covers the assembly, and this class pins the two
    facts that argument rests on rather than leaving them as prose.
    """

    def test_the_confidence_map_is_exactly_the_threshold_on_a_zero_input(self):
        """Why the zero baseline gates the probe. Three seeds, not one."""
        for seed in (0, 1, 2):
            keras.utils.set_random_seed(seed)
            model = _composite()
            model.build((None, HEIGHT, WIDTH, 3))
            zeros = np.zeros((1, HEIGHT, WIDTH, 3), dtype="float32")
            confidence = np.asarray(keras.ops.convert_to_numpy(
                model.segmenter(zeros, training=False)[0]))
            assert confidence.min() == confidence.max() == pytest.approx(0.5), (
                f"seed {seed}: confidence at zero input is "
                f"[{confidence.min()}, {confidence.max()}], not the flat 0.5 "
                "the argument above rests on")

    def test_the_impulse_response_through_the_mask_is_seed_dependent(self):
        """The measurement that disqualifies the composite as a probe subject.

        Not "the composite is dead" -- it is alive at two of these three seeds.
        The finding is that its liveness under an all-zero baseline is decided
        by a hard threshold, so the instrument cannot be relied on here.
        """
        energies = {}
        for seed in (0, 1, 2):
            keras.utils.set_random_seed(seed)
            model = _composite(rectifier_overrides={"iters": IMPULSE_ITERS})
            model.build((None, IMPULSE_HEIGHT, IMPULSE_WIDTH, 3))
            energies[seed] = float(impulse_energy_map(
                lambda x: model(x, training=False),
                (IMPULSE_HEIGHT, IMPULSE_WIDTH, 3),
                IMPULSE_HEIGHT // 2,
                IMPULSE_WIDTH // 2,
                label=f"DocScanner (seed {seed})",
            ).sum())
        assert min(energies.values()) == 0.0, energies
        assert max(energies.values()) > 0.0, energies

    @staticmethod
    def _mask_and_delta(seed: int) -> Tuple[float, float]:
        """Return ``(mask at the perturbed pixel, max|output change|)``.

        Interface contract -- two callers, the two arms below:

        * Parameter: ``seed``, seeding the whole build.
        * Returns: the composite's binary mask value at the pixel that gets
          perturbed (batch element 0), and how far the output moved.
        * Failure mode: none of its own.
        """
        keras.utils.set_random_seed(seed)
        model = _composite()
        model.build((None, HEIGHT, WIDTH, 3))
        base = _inputs()
        perturbed = base.copy()
        perturbed[:, HEIGHT // 2, WIDTH // 2, :] += 1.0
        confidence = np.asarray(keras.ops.convert_to_numpy(
            model.segmenter(base, training=False)[0]))
        mask = float(confidence[0, HEIGHT // 2, WIDTH // 2, 0] > 0.5)
        delta = float(np.max(np.abs(
            np.asarray(keras.ops.convert_to_numpy(model(base, training=False)))
            - np.asarray(keras.ops.convert_to_numpy(
                model(perturbed, training=False))))))
        return mask, delta

    def test_with_the_mask_OPEN_the_composite_responds_to_its_input(self):
        """The control for the two arms above: the model is not input-blind.

        MEASURED at seed 0: the confidence spans [0.485, 0.531] and 90.8% of
        pixels clear the threshold, the perturbed pixel among them.
        """
        mask, delta = self._mask_and_delta(0)
        assert mask == 1.0, "seed 0's mask no longer covers the probed pixel"
        assert delta > 0.0, (
            "the composite's output does not move when a pixel INSIDE its own "
            "mask is perturbed, so it is input-blind and the zero-energy "
            "reading above says nothing about the threshold")

    def test_with_the_mask_CLOSED_the_same_perturbation_moves_nothing(self):
        """And this is the mechanism, not a coincidence of the impulse grid.

        MEASURED at seed 1: the confidence spans [0.289, 0.508] and only 0.13%
        of pixels clear the threshold; the probed pixel is not one of them, and
        the output does not move by a single ULP. An untrained segmenter's
        confidence sits within a few percent of 0.5 everywhere, so which side
        of the threshold it lands on is decided by the initializer -- which is
        exactly why the impulse probe cannot be pointed at this class.
        """
        mask, delta = self._mask_and_delta(1)
        assert mask == 0.0, "seed 1's mask now covers the probed pixel"
        assert delta == 0.0, (
            f"the output moved by {delta} although the perturbed pixel is "
            "masked out; the mask is no longer strictly multiplicative")


# =====================================================================
# 2. precision_arm_oracle -- mixed_float16, float64, eager vs XLA
# =====================================================================
#
# MEASURED, and worth recording because it is the one number a reader of this
# model will want: the rectifier emits ABSOLUTE pixel coordinates, so its
# output magnitude is the image extent. At the shipped 288 x 288 resolution the
# float16 arm reads absmax 289.5 against float32's 289.596, and float16's
# spacing in [256, 512) is exactly 0.25 -- the measured unique values near the
# maximum are 288.5, 288.75, 289.0, 289.25, 289.5. So fp16 resolves ADJACENT
# INTEGER pixels at this resolution with a 4x margin (the D-046 concern in
# `layers/spatial_layer.py` bites at 2048, not at 288), but it quantizes the
# sub-pixel part of every coordinate to a quarter of a pixel. That is an
# accuracy statement about fp16 INFERENCE for this model, not a defect, and it
# is not asserted below because the arms here run at the small stand-in grid.


class TestThePrecisionArms:
    """All three classes, all three arms. No arm is skipped and none is xfail.

    The composite's ``mixed_float16`` arm carries one waiver,
    ``allowed_none_grads``, and it is the same waiver ``test_composite.py``
    already makes on the gradient oracle: D-025's hard threshold means no
    gradient ever reaches the segmenter. It is DERIVED from the model rather
    than written as a literal, and the float32 control below makes it
    two-sided.
    """

    def test_the_rectifier_runs_under_mixed_float16(self):
        assert_precision_arm(lambda: _rectifier(), _inputs)

    def test_the_segmenter_runs_under_mixed_float16(self):
        assert_precision_arm(lambda: _segmenter(), _inputs)

    def test_the_composite_runs_under_mixed_float16(self):
        keras.utils.set_random_seed(0)
        counted = _composite()
        counted.build((None, HEIGHT, WIDTH, 3))
        dead = len(counted.segmenter.trainable_variables)
        assert 0 < dead < len(counted.trainable_variables), (
            "the derived waiver is degenerate: the segmenter holds "
            f"{dead} of {len(counted.trainable_variables)} trainable variables")

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-030: `allowed_none_grads`
        # is DERIVED from the model, never written as the literal 462 the run
        # happens to report -- a literal goes stale the moment a stand-in width
        # moves, and it goes stale silently upward, waiving more than the
        # segmenter. Do NOT reach for `check_backward=False` instead: that
        # deletes part 4 of the arm, which the oracle's own docstring records
        # as the part that caught four of five real fp16 defects. See
        # decisions.md D-030.
        reports = assert_precision_arm(
            lambda: _composite(), _inputs, allowed_none_grads=dead)

        # Two-sided, and required by the oracle's own contract for
        # `allowed_none_grads`: the same gradients must be None under float32.
        # If they were not, the reading would be an fp16 finding wearing an
        # architecture's name.
        fp16_none = reports["backward_mixed_float16"]["n_none"]
        f32_none = run_backward(lambda: _composite(), _inputs, "float32")["n_none"]
        assert fp16_none == f32_none == dead, (
            f"fp16 left {fp16_none} gradients None, float32 left {f32_none}, "
            f"and the segmenter holds {dead} trainable variables; the waiver "
            "is only honest when all three agree")

    def test_the_rectifier_runs_in_float64(self):
        assert_float64_arm(lambda: _rectifier(), _inputs)

    def test_the_segmenter_runs_in_float64(self):
        assert_float64_arm(lambda: _segmenter(), _inputs)

    def test_the_composite_runs_in_float64(self):
        assert_float64_arm(lambda: _composite(), _inputs)

    def test_the_rectifier_compiles_under_xla(self):
        """D-018 inlined ``extract_patches``' body to work around a Keras 3.8
        symbolic-path bug; this is the arm that would see the inlined form
        diverge from eager under a traced graph."""
        assert_xla_equivalence(lambda: _rectifier(), _inputs)

    def test_the_segmenter_compiles_under_xla(self):
        assert_xla_equivalence(lambda: _segmenter(), _inputs)

    def test_the_composite_compiles_under_xla(self):
        assert_xla_equivalence(lambda: _composite(), _inputs)


# =====================================================================
# 3. knob_sensitivity_oracle -- the two classes that lacked it
# =====================================================================


class TestTheSegmentersKnobsAreStructural:
    """The segmenter has no value knob at all -- every argument it takes is a
    width, and a width changes the weight-SHAPE signature.

    The oracle's split is used as its docstring states: a structural knob is
    pinned on the signature, NEVER on an output difference, because two builds
    with different shapes consume different RNG draws and an
    output-difference assertion is satisfied by that alone.
    """

    def _built(self, **overrides: Any) -> DocScannerSegmenter:
        model = _segmenter(**overrides)
        model.build((None, HEIGHT, WIDTH, 3))
        return model

    def test_mid_channels_is_structural(self):
        signatures = assert_structural_knob_changes_weights(
            {
                7: lambda: self._built(),
                9: lambda: self._built(mid_channels=9),
                11: lambda: self._built(mid_channels=11),
            },
            knob="mid_channels",
        )
        # Stronger than "the signature changed": the RSU blocks' interior gets
        # WIDER, so the total parameter count must grow monotonically.
        totals = [
            sum(int(np.prod(shape)) for shape in signatures[key])
            for key in (7, 9, 11)
        ]
        assert totals[0] < totals[1] < totals[2], totals

    def test_out_channels_is_structural(self):
        assert_structural_knob_changes_weights(
            {
                13: lambda: self._built(),
                17: lambda: self._built(out_channels=17),
            },
            knob="out_channels",
        )

    def test_output_channels_is_structural(self):
        """It is the side heads' filter count, so it moves their kernels."""
        assert_structural_knob_changes_weights(
            {
                1: lambda: self._built(),
                3: lambda: self._built(output_channels=3),
            },
            knob="output_channels",
        )


class TestTheCompositesKnobsReachBothStages:
    """A composite that quietly ignored half of one config dict would build,
    run, serialize and round-trip. These arms are what convict it."""

    def _built(self, **kwargs: Any) -> DocScanner:
        model = _composite(**kwargs)
        model.build((None, HEIGHT, WIDTH, 3))
        return model

    def test_a_segmenter_width_is_a_structural_knob_of_the_composite(self):
        assert_structural_knob_changes_weights(
            {
                7: lambda: self._built(),
                9: lambda: self._built(segmenter_overrides={"mid_channels": 9}),
            },
            knob="segmenter_config['mid_channels']",
        )

    def test_a_rectifier_width_is_a_structural_knob_of_the_composite(self):
        """``hidden_dim`` cannot move alone -- the constructor asserts two
        relationships around it -- so the whole consistent group moves, exactly
        as ``test_model.py``'s arm does."""
        assert_structural_knob_changes_weights(
            {
                8: lambda: self._built(),
                12: lambda: self._built(rectifier_overrides={
                    "hidden_dim": 12, "context_dim": 12,
                    "gru_input_dim": 24, "fnet_output_dim": 24,
                    "motion_output_dim": 12}),
            },
            knob="rectifier_config['hidden_dim']",
        )

    def test_iters_is_a_VALUE_knob_of_the_composite(self):
        """It reuses ONE update block, so the weight signature is unchanged and
        the oracle's value form applies -- the same classification
        ``test_model.py`` makes for the rectifier alone, carried through the
        assembly to prove the composite does not pin its own iteration count.
        """
        deltas = assert_value_knob_changes_output(
            {
                1: lambda: self._built(rectifier_overrides={"iters": 1}),
                4: lambda: self._built(rectifier_overrides={"iters": 4}),
            },
            _inputs(),
            knob="rectifier_config['iters']",
            atol=1e-5,
        )
        assert all(value > 1e-5 for value in deltas.values()), deltas
