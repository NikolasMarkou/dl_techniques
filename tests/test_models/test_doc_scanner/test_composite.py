"""Behavioural guards for :class:`DocScanner`, the assembled two-stage model.

Everything this class adds over the two stages it owns is four lines of
``call`` -- and every one of those four lines has a wrong variant that is
entirely SHAPE-PRESERVING, which is why each gets a guard of its own here
rather than being covered by "the composite runs".

1. **The mask is MULTIPLICATIVE, not concatenated** (``inference.py:26``,
   ``x = msk * x``). Concatenating is the other common way to hand a mask to a
   downstream network; it would change the rectifier's input width from 3 to 4,
   which the feature encoder infers from the tensor and therefore accepts
   silently.
2. **The confidence map is THRESHOLDED, not used raw** (``inference.py:25``,
   ``msk = (msk > 0.5).float()``). A soft mask trains, is finite, keeps every
   shape -- and is a different pipeline. The comparison is strict ``>``, so a
   confidence of exactly ``0.5`` is masked OUT; that is asserted, not assumed.
3. **No gradient reaches the segmenter.** The threshold has zero gradient
   almost everywhere, so the composite CANNOT be trained end to end. That is
   the architecture (the paper trains the two modules independently, §4.3), not
   a defect, and it is pinned here so that nobody "fixes" it with a
   straight-through estimator later. The claim is made two-sided by the shared
   gradient oracle's own ``expect_zero`` mechanism: every segmenter weight must
   be dead AND every rectifier weight must be live.
4. **The calibration is applied EXACTLY ONCE** (``inference.py:29``,
   ``bm = (2 * (bm / 286.8) - 1) * 0.99``). Applying it twice, or not at all,
   leaves a finite map of the right shape.

The fixtures are NON-SQUARE (32 x 24) for the reason ``test_model.py`` and
``test_components.py`` both state: a square fixture is structurally blind to an
h/w swap.

``assert_gradients_reach_every_trainable_weight`` is adopted with a NAMED,
two-sided waiver rather than skipped. It is the only oracle whose plain form
does not apply to this class, and the reason is the subject of guard 3 above.
"""

import hashlib
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
    BM_CALIBRATION_DIVISOR,
    BM_CALIBRATION_SCALE,
    FLOW_CHANNELS,
    SEG_MASK_THRESHOLD,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.model import (
    DocScanner,
    DocScannerRectifier,
    DocScannerSegmenter,
    create_doc_scanner,
    create_doc_scanner_rectifier,
    create_doc_scanner_segmenter,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.warp import (
    coords_grid,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
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
)

# ---------------------------------------------------------------------
# The subject.
#
# Both stages are STAND-INS at small, pairwise-distinct widths. The segmenter's
# 7 / 13 are D-024's measured non-degenerate pair, kept here so this file's
# fixture cannot differ from `test_segmenter.py`'s in a way nobody notices.
#
# BATCH is 2, never 1: the smoke oracle's `slice_leading_axis` breaker slices
# each output leaf to `leaf[:1]`, a no-op at batch 1.
# ---------------------------------------------------------------------

BATCH = 2
HEIGHT, WIDTH = 32, 24          # NON-SQUARE, both multiples of 8
ITERS = 2

_SEG = dict(mid_channels=7, out_channels=13, output_channels=1)

_REC = dict(
    hidden_dim=8,
    context_dim=8,
    gru_input_dim=16,
    fnet_output_dim=16,
    encoder_stem_channels=6,
    encoder_stage_channels=(6, 10, 12),
    motion_output_dim=8,
    motion_corr_hidden=7,
    motion_corr_out=5,
    motion_flow_hidden=9,
    motion_flow_out=4,
    flow_head_hidden=11,
    mask_head_hidden=13,
    iters=ITERS,
)

INPUT_SHAPE = (None, HEIGHT, WIDTH, 3)

#: The repo convention for a value assertion: an explicit absolute tolerance
#: and no relative component.
ATOL = 1e-6

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src/dl_techniques/models/vision/image_restoration/doc_scanner/model.py"
)


def _build(**overrides) -> DocScanner:
    """The subject, UNBUILT."""
    config = dict(segmenter_config=dict(_SEG), rectifier_config=dict(_REC))
    config.update(overrides)
    return DocScanner(**config)


def _inputs() -> np.ndarray:
    """A deterministic input in [0, 1] -- the documented domain."""
    return np.random.RandomState(0).rand(
        BATCH, HEIGHT, WIDTH, 3).astype("float32")


def _built(**overrides) -> DocScanner:
    model = _build(**overrides)
    model.build(INPUT_SHAPE)
    return model


@pytest.fixture(scope="module")
def built_model() -> DocScanner:
    keras.utils.set_random_seed(0)
    return _built()


def _as_numpy(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _calibrate(array: np.ndarray) -> np.ndarray:
    """``inference.py:29``, written out INDEPENDENTLY of the source under test.

    Interface contract -- three callers here:

    * Parameter: ``array``, absolute pixel coordinates.
    * Returns: the calibrated map.
    * Failure mode: none.

    Deliberately NOT importing a helper from the package: an oracle that calls
    the code it is checking cannot catch a change to that code.
    """
    return (2.0 * (array / BM_CALIBRATION_DIVISOR) - 1.0) * BM_CALIBRATION_SCALE


# ---------------------------------------------------------------------
# Spies. Both patch the CLASS, as `test_model.py`'s seam guards do.
# ---------------------------------------------------------------------


def _spy_on_the_rectifiers_input(monkeypatch):
    """Record what the rectifier is called with, and with which flag.

    Interface contract -- several callers, all of the mask/threshold arms:

    * Parameter: pytest's ``monkeypatch``.
    * Returns: a list that fills with ``(inputs, training)`` per call.
    * Failure mode: none of its own.
    """
    seen = []
    original = DocScannerRectifier.call

    def spy(self, inputs, training=None):
        seen.append((inputs, training))
        return original(self, inputs, training=training)

    monkeypatch.setattr(DocScannerRectifier, "call", spy)
    return seen


def _force_the_confidence_map(monkeypatch, confidence_fn):
    """Replace the segmenter's forward with a controlled seven-map return.

    Interface contract -- several callers:

    * Parameters: ``monkeypatch``; ``confidence_fn(inputs) -> (B, H, W, 1)``,
      the ``d0`` the composite must consume.
    * Returns: nothing.
    * Failure mode: none of its own.

    The other six maps are returned as ZEROS. If the composite ever read one of
    them instead of ``d0``, every arm built on this helper would go RED, which
    is the intent.
    """
    def fake(self, inputs, training=None):
        d0 = confidence_fn(inputs)
        zeros = keras.ops.zeros_like(d0)
        return [d0] + [zeros] * 6

    monkeypatch.setattr(DocScannerSegmenter, "call", fake)


def _straddling_confidence(inputs):
    """A ``d0`` that takes values strictly below, exactly at, and above 0.5.

    The exactly-0.5 plane is the arm that pins the comparison as strict ``>``.
    """
    batch, height, width, _ = inputs.shape
    ramp = np.tile(
        np.linspace(0.0, 1.0, width, dtype="float32"), (batch, height, 1))
    ramp[:, ::2, :] = SEG_MASK_THRESHOLD      # a whole row of exact ties
    return keras.ops.convert_to_tensor(ramp[..., None])


# ---------------------------------------------------------------------
# 1. The output contract
# ---------------------------------------------------------------------


class TestTheOutputContract:
    """One calibrated backward map, at the input resolution."""

    def test_inference_returns_one_calibrated_backward_map(self, built_model):
        out = built_model(_inputs(), training=False)
        assert tuple(out.shape) == (BATCH, HEIGHT, WIDTH, FLOW_CHANNELS)
        assert_finite(out)

    def test_it_is_not_a_list_the_way_the_segmenter_is(self, built_model):
        out = built_model(_inputs(), training=False)
        assert not isinstance(out, (list, tuple, dict)), type(out)

    def test_the_compute_output_shape_agrees_with_the_forward(
            self, built_model):
        declared = built_model.compute_output_shape(
            (BATCH, HEIGHT, WIDTH, 3))
        actual = tuple(built_model(_inputs(), training=False).shape)
        assert tuple(declared) == actual

    def test_at_the_papers_own_288_the_range_is_plausible_for_a_grid(self):
        """``(B, 288, 288, 3)`` in, a normalized map out.

        The bound is deliberately loose -- the point is that the output is a
        NORMALIZED grid and not the rectifier's raw pixel coordinates, which at
        288 run to 287. A calibrated identity grid spans
        ``[-0.99, +0.9914]``; a refinement residual moves it, so anything
        inside +/- 2 is plausible and 287 is not.
        """
        keras.utils.set_random_seed(0)
        model = _build()
        model.build((None, 288, 288, 3))
        page = np.random.RandomState(1).rand(1, 288, 288, 3).astype("float32")
        out = _as_numpy(model(page, training=False))
        assert out.shape == (1, 288, 288, FLOW_CHANNELS)
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 2.0, (
            f"max |bm| = {np.abs(out).max()}; the composite is emitting "
            f"something that is not a calibrated grid (raw pixel coordinates "
            f"at 288 reach 287)")

    def test_a_non_multiple_of_eight_raises_naming_the_size(self):
        """The rectifier's constraint, surfaced by the composite."""
        with pytest.raises(ValueError, match="divisible by 8"):
            _build().build((None, 30, 24, 3))

    def test_a_non_rank_four_shape_raises_naming_this_model(self):
        with pytest.raises(ValueError, match="DocScanner expects a 4D"):
            _build().build((None, 32, 3))


# ---------------------------------------------------------------------
# 2. The mask is MULTIPLICATIVE, not concatenated
# ---------------------------------------------------------------------


class TestTheMaskIsAppliedByMultiplicationNotConcatenation:
    """``inference.py:26``: ``x = msk * x``.

    Two arms, because the two failure modes are different:

    * a STRUCTURAL arm -- the rectifier's input must still carry 3 channels.
      Concatenation would make it 4, which the feature encoder infers from the
      tensor and accepts without a word;
    * a VALUE arm -- with the mask everywhere zero the rectifier must see an
      identically zero image, and with it everywhere one it must see the input
      unchanged. Concatenation passes neither, but so would a mask applied to
      the wrong operand, which the structural arm alone cannot see.
    """

    def test_the_rectifier_still_receives_three_channels(
            self, monkeypatch, built_model):
        seen = _spy_on_the_rectifiers_input(monkeypatch)
        built_model(_inputs(), training=False)
        assert len(seen) == 1
        assert tuple(seen[0][0].shape) == (BATCH, HEIGHT, WIDTH, 3), (
            f"the rectifier was handed {tuple(seen[0][0].shape)}; a 4th "
            f"channel means the mask was CONCATENATED rather than multiplied "
            f"in (inference.py:26). See the D-025 anchor in model.py.")

    def test_a_zero_mask_hands_the_rectifier_a_zeroed_image(
            self, monkeypatch, built_model):
        _force_the_confidence_map(
            monkeypatch, lambda x: keras.ops.zeros_like(x[..., :1]))
        seen = _spy_on_the_rectifiers_input(monkeypatch)
        built_model(_inputs(), training=False)
        np.testing.assert_allclose(
            _as_numpy(seen[0][0]), 0.0, rtol=0, atol=0.0)

    def test_an_all_ones_mask_hands_the_rectifier_the_input_unchanged(
            self, monkeypatch, built_model):
        """The other side of the arm above: multiplication by 1 is identity.

        Without this, a composite that passed a CONSTANT zero downstream would
        satisfy the zero-mask arm perfectly.
        """
        _force_the_confidence_map(
            monkeypatch, lambda x: keras.ops.ones_like(x[..., :1]))
        seen = _spy_on_the_rectifiers_input(monkeypatch)
        page = _inputs()
        built_model(page, training=False)
        np.testing.assert_allclose(
            _as_numpy(seen[0][0]), page, rtol=0, atol=0.0)

    def test_the_fixture_can_tell_the_two_apart(self):
        """Non-vacuity: the input is not itself zero, so the two arms above
        are distinguishable on this fixture."""
        assert np.abs(_inputs()).max() > 0.1


# ---------------------------------------------------------------------
# 3. The confidence map is THRESHOLDED, not used raw
# ---------------------------------------------------------------------


class TestTheConfidenceMapIsThresholdedNotUsedRaw:
    """``inference.py:25``: ``msk = (msk > 0.5).float()``."""

    @staticmethod
    def _seen_input(monkeypatch, model):
        _force_the_confidence_map(monkeypatch, _straddling_confidence)
        seen = _spy_on_the_rectifiers_input(monkeypatch)
        page = _inputs()
        model(page, training=False)
        return page, _as_numpy(seen[0][0])

    def test_the_masked_image_is_binary_gated(self, monkeypatch, built_model):
        page, masked = self._seen_input(monkeypatch, built_model)
        confidence = _as_numpy(
            _straddling_confidence(keras.ops.convert_to_tensor(page)))
        gate = (confidence > SEG_MASK_THRESHOLD).astype("float32")
        np.testing.assert_allclose(masked, gate * page, rtol=0, atol=0.0)

    def test_it_is_NOT_soft_gated(self, monkeypatch, built_model):
        """The mutation this file exists to catch, asserted directly."""
        page, masked = self._seen_input(monkeypatch, built_model)
        confidence = _as_numpy(
            _straddling_confidence(keras.ops.convert_to_tensor(page)))
        soft = confidence * page
        assert not np.allclose(masked, soft, rtol=0, atol=1e-3), (
            "the rectifier's input equals `confidence * image`, i.e. the raw "
            "sigmoid was used as a soft mask. inference.py:25 binarizes first.")

    def test_a_confidence_of_exactly_one_half_is_masked_OUT(
            self, monkeypatch, built_model):
        """Strict ``>``, not ``>=``. One pixel value decides it, and the
        straddling fixture puts a whole row of exact ties in the way."""
        page, masked = self._seen_input(monkeypatch, built_model)
        confidence = _as_numpy(
            _straddling_confidence(keras.ops.convert_to_tensor(page)))
        ties = np.isclose(confidence[..., 0], SEG_MASK_THRESHOLD)
        assert ties.any(), "the fixture carries no exact ties; arm is vacuous"
        np.testing.assert_allclose(masked[ties], 0.0, rtol=0, atol=0.0)

    def test_the_fixture_straddles_the_threshold_on_both_sides(self):
        """Non-vacuity for all three arms above."""
        confidence = _as_numpy(
            _straddling_confidence(keras.ops.convert_to_tensor(_inputs())))
        assert (confidence < SEG_MASK_THRESHOLD).any()
        assert (confidence > SEG_MASK_THRESHOLD).any()

    def test_only_d0_is_consumed(self, monkeypatch, built_model):
        """The other six maps are deep supervision. ``_force_the_confidence_map``
        returns them as zeros, so a composite reading any of them would gate
        the image to zero -- and the binary arm above would go RED."""
        page, masked = self._seen_input(monkeypatch, built_model)
        assert np.abs(masked).max() > 0.0, (
            "the masked image is identically zero, which is what reading one "
            "of the six deep-supervision maps instead of d0 would produce")


# ---------------------------------------------------------------------
# 4. No gradient reaches the segmenter
# ---------------------------------------------------------------------


class TestNoGradientReachesTheSegmenter:
    """The threshold is non-differentiable, and that is the ARCHITECTURE.

    ``(msk > 0.5)`` has zero gradient almost everywhere, so the composite
    cannot be trained end to end. The paper trains the two modules
    INDEPENDENTLY (§4.3), so nothing is meant to flow here. Pinned rather than
    assumed, so that a later reader does not "repair" it with a straight-
    through estimator or a soft threshold.

    The instrument is CONNECTIVITY (``tape.gradient(...) is None``), not
    magnitude -- the same choice, for the same reason, as
    ``TestTheCoordinateFieldIsDetachedEachIteration`` in ``test_model.py``.
    """

    @staticmethod
    def _report(training: bool = True):
        keras.utils.set_random_seed(0)
        model = _built()
        return model, gradient_report(model, _inputs(), training=training)

    def test_every_segmenter_weight_is_disconnected(self):
        model, report = self._report()
        segmenter = {w.path for w in model.segmenter.trainable_weights}
        assert segmenter, "no segmenter weights; the probe is vacuous"
        offenders = {
            path for path in segmenter if report.get(path) is not None
        }
        assert not offenders, (
            f"{len(offenders)} segmenter weight(s) are on the composite's "
            f"backward graph. The 0.5 threshold (inference.py:25) has zero "
            f"gradient a.e., so this can only mean it was softened or "
            f"bypassed. See the D-025 anchor in model.py. First few: "
            f"{sorted(offenders)[:3]}")

    def test_every_rectifier_weight_IS_connected(self):
        """Non-vacuity, half one: ``None`` above must mean SEVERED, not
        "nothing is on the tape"."""
        model, report = self._report()
        rectifier = {w.path for w in model.rectifier.trainable_weights}
        assert rectifier
        dead = {path for path in rectifier if report.get(path) is None}
        assert not dead, (
            f"{len(dead)} rectifier weight(s) received no gradient either; the "
            f"probe cannot distinguish a severed path from a dead model")

    def test_the_instrument_sees_the_segmenter_when_it_is_trained_alone(self):
        """Non-vacuity, half two: the SAME instrument on the segmenter by
        itself must report live gradients. If it could not, the arm above would
        pass for an instrument reason rather than an architectural one."""
        keras.utils.set_random_seed(0)
        stage = DocScannerSegmenter(**_SEG)
        stage.build(INPUT_SHAPE)
        report = gradient_report(stage, _inputs(), training=True)
        assert all(value is not None for value in report.values()), (
            "the segmenter, trained on its own, still shows disconnected "
            "weights; the probe is broken")

    def test_the_named_two_sided_waiver_holds(self):
        """The shared oracle, adopted with ``expect_zero`` rather than skipped.

        Two-sided by construction: a segmenter weight that DID receive a
        gradient makes the waiver "obsolete" and raises, and a pattern that
        matched nothing raises as a stale waiver.
        """
        keras.utils.set_random_seed(0)
        model = _built()
        report = assert_gradients_reach_every_trainable_weight(
            model, _inputs(), training=True, expect_zero=("segmenter/",))
        assert len(report) == len(model.trainable_weights)

    def test_the_waiver_is_not_a_blanket(self):
        """``expect_zero=("segmenter/",)`` must not accidentally cover the
        rectifier -- otherwise the arm above waives the whole model."""
        keras.utils.set_random_seed(0)
        model = _built()
        rectifier = [w.path for w in model.rectifier.trainable_weights]
        assert not any("segmenter/" in path for path in rectifier)


# ---------------------------------------------------------------------
# 5. The calibration is applied EXACTLY ONCE
# ---------------------------------------------------------------------


class TestTheCalibrationIsAppliedExactlyOnce:
    """``inference.py:29``: ``bm = (2 * (bm / 286.8) - 1) * 0.99``.

    Same discipline as the mask head's 0.25 (D-016): zero the flow head so the
    rectifier emits EXACTLY the identity coordinate grid, then assert the
    composite's output is that grid calibrated once, against an independently
    written formula. Not-at-all and twice-over are both checked to differ from
    it on this fixture, so the equality is not a coincidence of scale.
    """

    @staticmethod
    def _model_with_a_dead_flow_head() -> DocScanner:
        keras.utils.set_random_seed(0)
        model = _built()
        head = model.rectifier.update_block.flow_head.conv2
        head.kernel.assign(keras.ops.zeros_like(head.kernel))
        head.bias.assign(keras.ops.zeros_like(head.bias))
        return model

    @staticmethod
    def _identity_grid() -> np.ndarray:
        return _as_numpy(coords_grid(BATCH, HEIGHT, WIDTH))

    def test_the_output_is_the_identity_grid_calibrated_once(
            self, monkeypatch):
        _force_the_confidence_map(
            monkeypatch, lambda x: keras.ops.ones_like(x[..., :1]))
        out = _as_numpy(
            self._model_with_a_dead_flow_head()(_inputs(), training=False))
        np.testing.assert_allclose(
            out, _calibrate(self._identity_grid()), rtol=0, atol=ATOL)

    def test_it_is_not_the_uncalibrated_grid(self, monkeypatch):
        _force_the_confidence_map(
            monkeypatch, lambda x: keras.ops.ones_like(x[..., :1]))
        out = _as_numpy(
            self._model_with_a_dead_flow_head()(_inputs(), training=False))
        assert not np.allclose(
            out, self._identity_grid(), rtol=0, atol=1e-3), (
            "the composite emitted the rectifier's raw pixel coordinates; the "
            "inference.py:29 calibration was dropped")

    def test_it_is_not_the_grid_calibrated_twice(self, monkeypatch):
        _force_the_confidence_map(
            monkeypatch, lambda x: keras.ops.ones_like(x[..., :1]))
        out = _as_numpy(
            self._model_with_a_dead_flow_head()(_inputs(), training=False))
        assert not np.allclose(
            out, _calibrate(_calibrate(self._identity_grid())),
            rtol=0, atol=1e-3), (
            "the calibration was applied twice -- once in the rectifier and "
            "once here, or twice here")

    def test_the_three_forms_are_pairwise_distinct_on_this_fixture(self):
        """Non-vacuity for the two arms above."""
        grid = self._identity_grid()
        once, twice = _calibrate(grid), _calibrate(_calibrate(grid))
        assert not np.allclose(grid, once, atol=1e-3)
        assert not np.allclose(once, twice, atol=1e-3)

    def test_the_constants_are_the_transcribed_ones(self):
        """286.8 and 0.99, not 288 and 1.0. See the D-027 anchor."""
        assert BM_CALIBRATION_DIVISOR == 286.8
        assert BM_CALIBRATION_SCALE == 0.99


# ---------------------------------------------------------------------
# 6. The rectifier is called with training=False, always (D-026)
# ---------------------------------------------------------------------


class TestTheOutputFormDoesNotDependOnTheTrainingFlag:
    """The rectifier's ``training`` selects its output RANK, not a norm mode.

    Forwarding this class's flag would make the composite emit
    ``(B, 12, H, W, 2)`` inside ``fit()`` -- on a model that cannot be fit at
    all (see guard 3). So it is pinned to ``False`` at the call site.
    """

    def test_the_rectifier_is_called_with_training_false_even_under_true(
            self, monkeypatch, built_model):
        seen = _spy_on_the_rectifiers_input(monkeypatch)
        built_model(_inputs(), training=True)
        assert seen[0][1] is False, (
            f"the rectifier received training={seen[0][1]!r}; D-026 pins it to "
            f"False because that flag selects the OUTPUT RANK")

    def test_both_flags_give_the_same_output_shape(self, built_model):
        page = _inputs()
        assert (
            tuple(built_model(page, training=True).shape)
            == tuple(built_model(page, training=False).shape)
            == (BATCH, HEIGHT, WIDTH, FLOW_CHANNELS)
        )

    def test_the_default_flag_also_gives_the_inference_form(self, built_model):
        out = built_model(_inputs())
        assert tuple(out.shape) == (BATCH, HEIGHT, WIDTH, FLOW_CHANNELS)

    def test_the_flag_still_reaches_the_segmenter(
            self, monkeypatch, built_model):
        """Two-sided: pinning the rectifier's flag must not pin the
        segmenter's, whose RSU blocks carry BatchNormalization."""
        seen = []

        original = DocScannerSegmenter.call

        def spy(self, inputs, training=None):
            seen.append(training)
            return original(self, inputs, training=training)

        monkeypatch.setattr(DocScannerSegmenter, "call", spy)
        built_model(_inputs(), training=True)
        built_model(_inputs(), training=False)
        assert seen == [True, False], seen


# ---------------------------------------------------------------------
# 7. The variant table, the factory and `pretrained`
# ---------------------------------------------------------------------


class TestTheVariantTable:
    """One row, PAIRED from the two stages' own rows rather than restated."""

    def test_there_is_exactly_one_row_and_it_is_docscanner_l(self):
        assert list(DocScanner.MODEL_VARIANTS) == ["docscanner-l"]

    def test_the_segmenter_half_is_the_segmenters_own_row(self):
        expected = dict(DocScannerSegmenter.MODEL_VARIANTS["docscanner-l"])
        expected.pop("description")
        assert (
            DocScanner.MODEL_VARIANTS["docscanner-l"]["segmenter_config"]
            == expected
        )

    def test_the_rectifier_half_is_the_rectifiers_own_row(self):
        expected = dict(DocScannerRectifier.MODEL_VARIANTS["docscanner-l"])
        expected.pop("description")
        assert (
            DocScanner.MODEL_VARIANTS["docscanner-l"]["rectifier_config"]
            == expected
        )

    def test_neither_half_carries_a_description(self):
        """A stray ``description`` would be passed to a stage constructor."""
        row = DocScanner.MODEL_VARIANTS["docscanner-l"]
        assert "description" not in row["segmenter_config"]
        assert "description" not in row["rectifier_config"]

    def test_the_row_has_a_description_that_names_the_pipeline(self):
        description = DocScanner.MODEL_VARIANTS["docscanner-l"]["description"]
        assert "286.8" in description and "0.5" in description

    def test_an_unknown_variant_raises_listing_the_known_ones(self):
        with pytest.raises(ValueError, match="docscanner-l"):
            create_doc_scanner("docscanner-xl")

    def test_the_shipped_variant_builds_at_288_with_the_measured_count(self):
        """8,465,629 parameters -- the two stages' measured counts, summed.

        Asserted as an INTERNAL consistency claim (the composite adds no
        weights of its own), never against the paper's 8.5M: that figure is
        quoted to two significant figures and any total in [8.45M, 8.55M]
        rounds to it, so it corroborates the width table to about +/-1% and is
        not a number to gate on.
        """
        keras.utils.set_random_seed(0)
        model = create_doc_scanner("docscanner-l")
        model.build((None, 288, 288, 3))
        assert model.count_params() == 8_465_629
        assert model.count_params() == (
            model.segmenter.count_params() + model.rectifier.count_params())


class TestPretrainedRaisesOnAllThree:
    """H-4, for every public entry point this package has."""

    @pytest.mark.parametrize(
        "factory",
        [
            create_doc_scanner,
            create_doc_scanner_rectifier,
            create_doc_scanner_segmenter,
        ],
        ids=["composite", "rectifier", "segmenter"],
    )
    def test_the_factory_raises(self, factory):
        with pytest.raises(NotImplementedError):
            factory(pretrained=True)

    @pytest.mark.parametrize(
        "cls", [DocScanner, DocScannerRectifier, DocScannerSegmenter],
        ids=["composite", "rectifier", "segmenter"],
    )
    def test_from_variant_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls.from_variant("docscanner-l", pretrained=True)

    def test_the_composites_message_names_BOTH_stages_reasons(self):
        with pytest.raises(NotImplementedError) as excinfo:
            create_doc_scanner(pretrained=True)
        message = str(excinfo.value)
        assert "seg.pth" in message, message
        assert "D-011" in message and "D-012" in message, message

    def test_an_empty_stage_config_is_refused(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            DocScanner(segmenter_config={}, rectifier_config=dict(_REC))


# ---------------------------------------------------------------------
# 8. Serialization and the shared oracles
# ---------------------------------------------------------------------


def _contract(output) -> None:
    """The forward contract. Shared with the meta-test so it is falsifiable."""
    assert not isinstance(output, (dict, list, tuple)), (
        f"DocScanner returns a single calibrated backward map, got "
        f"{type(output)}")
    assert tuple(output.shape) == (BATCH, HEIGHT, WIDTH, FLOW_CHANNELS), (
        tuple(output.shape))


class TestSerialization:
    """A REAL save/load, asserted on VALUES."""

    def test_a_real_save_and_load_model_reproduces_the_values(self, tmp_path):
        keras.utils.set_random_seed(0)
        model = _built()
        page = _inputs()
        before = _as_numpy(model(page, training=False))

        path = tmp_path / "doc_scanner.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)

        assert isinstance(reloaded, DocScanner)
        after = _as_numpy(reloaded(page, training=False))
        np.testing.assert_allclose(before, after, rtol=0, atol=ATOL)

    def test_get_config_round_trips_both_stage_configurations(self):
        config = _build().get_config()
        assert config["segmenter_config"] == _SEG
        assert config["rectifier_config"] == {
            **_REC, "encoder_stage_channels": _REC["encoder_stage_channels"]}

    def test_the_config_reconstructs_an_identical_configuration(self):
        original = _build()
        clone = DocScanner.from_config(original.get_config())
        assert clone.get_config() == original.get_config()

    def test_the_registration_key_strips_family_and_subfamily(self):
        registered = keras.saving.get_registered_name(DocScanner)
        assert registered == "dl_techniques.models.doc_scanner.model>DocScanner"


class TestTheSharedOracleAdoptions:
    """One arm per oracle. Nothing generic is re-implemented here.

    ``assert_gradients_reach_every_trainable_weight`` is adopted in
    ``TestNoGradientReachesTheSegmenter`` instead, with its named waiver.
    """

    def test_the_round_trip_reproduces_the_output_values_exactly(self):
        report = measure_roundtrip(_build, _inputs, training=False)
        assert report["self_max_delta"] == 0.0, (
            f"DocScanner is not deterministic at inference (self spread "
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
        report = measure_build_parity(_build, _inputs, input_shape=INPUT_SHAPE)
        assert_build_parity(report, autoname_stems=(), expect_path_collisions=0)

    def test_the_forward_satisfies_its_contract(self, built_model):
        _contract(built_model(_inputs(), training=False))

    def test_the_smoke_contract_rejects_a_broken_forward(self, built_model):
        messages = assert_contract_rejects_a_broken_forward(
            built_model, _inputs(), _contract)
        assert messages


# ---------------------------------------------------------------------
# 9. The source under test is the TRACKED file
# ---------------------------------------------------------------------


class TestTheGuardsRunAgainstTheShippedSource:
    """The RED proofs for this file mutate ``model.py`` IN PLACE.

    ``pyproject.toml``'s ``pythonpath = ["src"]`` overrides ``PYTHONPATH``, so a
    scratchpad copy of the package is never loaded and a GREEN from that method
    would mean UNTESTED rather than survived (a recorded incident). This arm
    records where the file is and that the imported module comes from it, so a
    future mutation pass has an anchor for the restore check.
    """

    def test_the_imported_module_is_the_tracked_file(self):
        from dl_techniques.models.vision.image_restoration.doc_scanner import (
            model as model_module,
        )
        assert Path(model_module.__file__).resolve() == _SOURCE

    def test_the_source_hashes(self):
        """Not an assertion about the CONTENT -- just proof the path reads."""
        digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
        assert len(digest) == 64
