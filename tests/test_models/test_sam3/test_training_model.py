"""Tests for ``models/SAM/SAM3/training_model.py`` -- the packed-tensor wrapper.

What this file guards, and why each guard is shaped the way it is:

- **The pack/unpack round trip by VALUE, never by shape.** A round-trip test
  that compares only shapes passes on a pack that drops every channel and on a
  model that restored zero weights. Every assertion below is on VALUES, and the
  synthetic fixtures put each channel in a DISJOINT NUMERIC BAND (score ~100,
  box channels ~200/300/400/500, masks ~600, presence ~900) so a swapped,
  dropped or broadcast channel is visible rather than merely improbable. The
  band separation is itself asserted, so the liveness arm cannot rot.
- **The oracle is hand-written NumPy**, placing fields at LITERAL channels, and
  a separate test pins the module's imported constants against those literals.
  Neither the oracle nor its expected values are ever obtained by calling the
  functions under test.
- **The three-way width contract**, which cannot slice-fail loudly: two of its
  three legs are proven to RAISE and the third is pinned by equality across the
  model, the loss and the packer.
- **A real stock ``fit()`` step**, asserting a FINITE loss AND that weights
  MOVED. A ``fit()`` that runs but moves nothing is the failure mode here, and
  a loss number alone cannot see it.
- **The ``.keras`` round trip at ``training=False``**. At ``training=None`` a
  correct round trip measures deltas of 0.2-2.2 that look exactly like
  reinitialized weights (D-123), so the flag is part of the guard.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.losses.sam3_detection_loss import (
    META_IS_EXHAUSTIVE,
    META_KEEP_LOSS,
    META_NUM_BOXES,
    PACKED_BOX_START,
    PACKED_MASK_START,
    PACKED_SCORE_CHANNEL,
    Sam3DetectionLoss,
    packed_channel_count,
    unpack_predictions,
    unpack_targets,
)
from dl_techniques.models.SAM.SAM3 import Sam3Image
from dl_techniques.models.SAM.SAM3.training_model import (
    Sam3TrainingModel,
    compile_sam3_trainer,
    pack_predictions,
    pack_targets,
    select_prediction_blocks,
)

# ---------------------------------------------------------------------
# Disjoint value bands. A channel that is dropped, duplicated, swapped or
# broadcast lands OUTSIDE its band, which is what makes the round-trip
# assertions discriminating rather than merely satisfiable.
# ---------------------------------------------------------------------
BAND_SCORE = 100.0
BAND_BOX = (200.0, 300.0, 400.0, 500.0)
BAND_MASK = 600.0
BAND_PRESENCE = 900.0
BAND_HALF_WIDTH = 1.0

#: `tiny`'s geometry, read from the variant table rather than restated.
TINY_QUERIES = Sam3Image.MODEL_VARIANTS["tiny"]["num_queries"]
TINY_IMG = Sam3Image.MODEL_VARIANTS["tiny"]["img_size"]
TINY_CTX = Sam3Image.MODEL_VARIANTS["tiny"]["context_length"]
TINY_VOCAB = Sam3Image.MODEL_VARIANTS["tiny"]["vocab_size"]


# ---------------------------------------------------------------------
# float64 NumPy oracles -- written from the LAYOUT SPEC using literal channel
# indices, never by calling the implementation. `test_layout_constants_*`
# below pins the module's constants against exactly these literals, so the
# oracle stays independent AND a constant drift still fires.
# ---------------------------------------------------------------------


def np_pack_predictions(logits, boxes, presence, masks=None):
    """Hand-written packer for ``(B, Q + 1, C)`` predictions."""
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    presence = np.asarray(presence, dtype=np.float64)
    batch, queries = logits.shape[0], logits.shape[1]
    mask_size = 0
    if masks is not None:
        masks = np.asarray(masks, dtype=np.float64)
        masks = masks.reshape(batch, queries, -1)
        mask_size = masks.shape[-1]
    channels = 5 + mask_size

    packed = np.zeros((batch, queries + 1, channels), dtype=np.float64)
    packed[:, :queries, 0] = logits[..., 0]
    packed[:, :queries, 1:5] = boxes
    if mask_size:
        packed[:, :queries, 5:] = masks
    packed[:, queries, 0] = presence[..., 0]
    return packed


def np_pack_targets(boxes, valid, masks=None, num_boxes=None,
                    is_exhaustive=None):
    """Hand-written packer for ``(B, N_max + 1, C)`` targets."""
    boxes = np.asarray(boxes, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.float64)
    batch, slots = valid.shape
    mask_size = 0
    if masks is not None:
        masks = np.asarray(masks, dtype=np.float64).reshape(batch, slots, -1)
        mask_size = masks.shape[-1]
    channels = 5 + mask_size

    packed = np.zeros((batch, slots + 1, channels), dtype=np.float64)
    packed[:, :slots, 0] = valid
    packed[:, :slots, 1:5] = boxes
    if mask_size:
        packed[:, :slots, 5:] = masks

    # `derive_keep_loss`'s reference expression: an instance counts only if it
    # is a real row AND its box has positive width AND positive height.
    visible = (valid > 0.0) & (boxes[..., 2] > 0.0) & (boxes[..., 3] > 0.0)
    keep = (visible.sum(axis=-1) != 0).astype(np.float64)
    counts = ((valid > 0.0).sum(axis=-1).astype(np.float64)
              if num_boxes is None else np.asarray(num_boxes, np.float64))
    exhaustive = (np.ones(batch, np.float64) if is_exhaustive is None
                  else np.asarray(is_exhaustive, np.float64))

    packed[:, slots, 0] = keep
    packed[:, slots, 1] = counts
    packed[:, slots, 2] = exhaustive
    return packed


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


def as_dense(grad):
    """Densify a gradient before comparing it numerically.

    MEASURED: this model's gradient list contains ``tf.IndexedSlices`` (the
    token embedding's sparse lookup), and ``np.array`` on one of those yields a
    dtype-``object`` array on which ``np.isfinite`` raises ``TypeError`` -- so a
    NaN check written without this helper does not check anything, it errors.
    """
    if isinstance(grad, tf.IndexedSlices):
        grad = tf.convert_to_tensor(grad)
    return np.array(grad)


def banded(rng, shape, centre):
    """Seeded values inside ``centre +- BAND_HALF_WIDTH``, all distinct."""
    return (centre + rng.uniform(-BAND_HALF_WIDTH, BAND_HALF_WIDTH, shape)
            ).astype(np.float32)


@pytest.fixture
def rng():
    return np.random.default_rng(20260805)


@pytest.fixture
def synthetic_outputs(rng):
    """A banded stand-in for ``Sam3Image``'s output dict, ``B=3, Q=4``."""
    batch, queries, grid = 3, 4, 3
    boxes = np.stack(
        [banded(rng, (batch, queries), centre) for centre in BAND_BOX],
        axis=-1)
    return {
        "pred_logits": banded(rng, (batch, queries, 1), BAND_SCORE),
        "pred_boxes": boxes,
        "pred_masks": banded(rng, (batch, queries, grid, grid), BAND_MASK),
        "presence_logit": banded(rng, (batch, 1), BAND_PRESENCE),
        "semantic_seg": banded(rng, (batch, grid, grid, 1), -7.0),
    }


@pytest.fixture
def synthetic_targets(rng):
    """A banded padded-GT fixture, ``B=3, N_max=6``.

    Image 0 is a ZERO-GT image (an ordinary member of the layout), image 1 is
    fully populated (6 instances, which exceeds `tiny`'s Q=5), image 2 is
    partially populated.
    """
    batch, slots, grid = 3, 6, 3
    boxes = np.stack(
        [banded(rng, (batch, slots), centre) for centre in BAND_BOX], axis=-1)
    valid = np.zeros((batch, slots), np.float32)
    valid[1, :] = 1.0
    valid[2, :2] = 1.0
    masks = banded(rng, (batch, slots, grid, grid), BAND_MASK)
    return boxes, valid, masks


@pytest.fixture(scope="module")
def tiny_model():
    """One built `tiny` wrapper, shared -- construction dominates these tests."""
    keras.utils.set_random_seed(1234)
    model = Sam3TrainingModel(Sam3Image.from_variant("tiny"))
    model.build(None)
    return model


def tiny_inputs(rng, batch=2):
    return {
        "image": rng.normal(size=(batch, TINY_IMG, TINY_IMG, 3)
                            ).astype("float32"),
        "token_ids": rng.integers(
            0, TINY_VOCAB, size=(batch, TINY_CTX)).astype("int32"),
    }


# ---------------------------------------------------------------------
# the oracle's own premises
# ---------------------------------------------------------------------


class TestOraclePremises:
    """Guards on the things the oracles above assume."""

    def test_layout_constants_are_the_literals_the_oracle_hardcodes(self):
        """The NumPy oracle writes channels 0/1:5/5: and meta 0/1/2 as
        LITERALS, on purpose, so it is independent of the implementation. This
        test is what keeps that independence honest: if the layout ever moves,
        this fires instead of the oracle silently agreeing with a moved port.
        """
        assert PACKED_SCORE_CHANNEL == 0
        assert PACKED_BOX_START == 1
        assert PACKED_MASK_START == 5
        assert META_KEEP_LOSS == 0
        assert META_NUM_BOXES == 1
        # D-010: channel 2 is `is_exhaustive`, NOT a reserved zero.
        assert META_IS_EXHAUSTIVE == 2

    def test_the_value_bands_are_disjoint(self):
        """The liveness arm of every round-trip assertion below.

        Without disjoint bands a swapped or broadcast channel could still land
        on a plausible value; with them, any cross-channel contamination is a
        difference of at least 98.
        """
        centres = [BAND_SCORE, *BAND_BOX, BAND_MASK, BAND_PRESENCE]
        for i, left in enumerate(centres):
            for right in centres[i + 1:]:
                assert abs(left - right) > 2.0 * BAND_HALF_WIDTH

    def test_a_dropped_channel_would_actually_be_caught(self, rng):
        """Proves the round-trip comparison discriminates, by breaking it.

        Packs with the box block zeroed and asserts the SAME comparison the
        real tests use goes RED. Without this, a value-exact assertion that
        happens to compare a tensor with itself would look identical.
        """
        boxes = np.stack(
            [banded(rng, (2, 3), c) for c in BAND_BOX], axis=-1)
        good = np_pack_targets(boxes, np.ones((2, 3), np.float32))
        broken = np_pack_targets(np.zeros_like(boxes),
                                 np.ones((2, 3), np.float32))
        assert not np.allclose(good, broken)
        assert np.abs(good - broken).max() > 100.0


# ---------------------------------------------------------------------
# SC-7 -- the packing round trip, value-exact
# ---------------------------------------------------------------------


class TestPackedRoundTrip:
    """SC-7. pack -> unpack -> identity, by VALUE, at ``atol=0``."""

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_pack_predictions_matches_the_numpy_oracle(
            self, synthetic_outputs, include_masks):
        packed = np.array(pack_predictions(synthetic_outputs, include_masks))
        oracle = np_pack_predictions(
            synthetic_outputs["pred_logits"],
            synthetic_outputs["pred_boxes"],
            synthetic_outputs["presence_logit"],
            synthetic_outputs["pred_masks"] if include_masks else None)
        assert packed.shape == oracle.shape
        np.testing.assert_allclose(packed, oracle, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_pack_unpack_predictions_is_exact_identity(
            self, synthetic_outputs, include_masks):
        packed = pack_predictions(synthetic_outputs, include_masks)
        back = unpack_predictions(packed, include_masks)

        np.testing.assert_array_equal(
            np.array(back["pred_logits"]),
            np.array(synthetic_outputs["pred_logits"])[..., 0])
        np.testing.assert_array_equal(
            np.array(back["pred_boxes"]),
            np.array(synthetic_outputs["pred_boxes"]))
        np.testing.assert_array_equal(
            np.array(back["presence_logit"]),
            np.array(synthetic_outputs["presence_logit"]))
        if include_masks:
            raw = np.array(synthetic_outputs["pred_masks"])
            np.testing.assert_array_equal(
                np.array(back["pred_masks"]),
                raw.reshape(raw.shape[0], raw.shape[1], -1))
        else:
            assert back["pred_masks"] is None

    def test_presence_row_non_score_channels_are_exactly_zero(
            self, synthetic_outputs):
        """The layout says every non-score channel of the presence row is
        zero-filled and unread. A pack that leaked the last query's box there
        would still round-trip the queries perfectly."""
        packed = np.array(pack_predictions(synthetic_outputs, True))
        presence_row = packed[:, -1, :]
        assert np.all(presence_row[:, PACKED_BOX_START:] == 0.0)
        # ...and the score channel is NOT zero, so the row is not simply blank.
        assert np.all(np.abs(presence_row[:, PACKED_SCORE_CHANNEL]) > 1.0)

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_pack_targets_matches_the_numpy_oracle(
            self, synthetic_targets, include_masks):
        boxes, valid, masks = synthetic_targets
        packed = np.array(pack_targets(
            boxes, valid,
            target_masks=masks if include_masks else None,
            include_masks=include_masks))
        oracle = np_pack_targets(
            boxes, valid, masks if include_masks else None)
        assert packed.shape == oracle.shape
        np.testing.assert_allclose(packed, oracle, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_pack_unpack_targets_is_exact_identity(
            self, synthetic_targets, include_masks):
        boxes, valid, masks = synthetic_targets
        packed = pack_targets(
            boxes, valid,
            target_masks=masks if include_masks else None,
            include_masks=include_masks)
        back = unpack_targets(packed, include_masks)

        np.testing.assert_array_equal(
            np.array(back["target_valid"]), valid)
        np.testing.assert_array_equal(
            np.array(back["target_boxes"]), boxes)
        if include_masks:
            np.testing.assert_array_equal(
                np.array(back["target_masks"]),
                masks.reshape(masks.shape[0], masks.shape[1], -1))
        else:
            assert back["target_masks"] is None

    def test_the_zero_gt_image_round_trips_as_an_ordinary_row(
            self, synthetic_targets):
        """Image 0 has no valid instance. Its meta row must read
        ``keep_loss=0``, ``num_boxes=0`` -- and its BOX channels must still
        round-trip, because the layout does not special-case it."""
        boxes, valid, _ = synthetic_targets
        packed = pack_targets(boxes, valid, include_masks=False)
        back = unpack_targets(packed, False)

        assert float(np.array(back["keep_loss"])[0, 0]) == 0.0
        assert float(np.array(back["num_boxes"])[0]) == 0.0
        assert float(np.array(back["keep_loss"])[1, 0]) == 1.0
        assert float(np.array(back["num_boxes"])[1]) == float(valid.shape[1])
        np.testing.assert_array_equal(
            np.array(back["target_valid"])[0], np.zeros(valid.shape[1]))
        np.testing.assert_array_equal(
            np.array(back["target_boxes"])[0], boxes[0])

    def test_the_meta_row_carries_is_exhaustive_not_a_reserved_zero(
            self, synthetic_targets):
        """D-010. Channel 2 is READ by the ``weak_loss=True`` divisor, so a
        producer that leaves it zero declares every image non-exhaustive."""
        boxes, valid, _ = synthetic_targets
        flags = np.array([1.0, 0.0, 1.0], np.float32)
        packed = pack_targets(
            boxes, valid, is_exhaustive=flags, include_masks=False)
        np.testing.assert_array_equal(
            np.array(unpack_targets(packed, False)["is_exhaustive"]), flags)
        # Default is all-ones, and that default is not the same tensor.
        default = pack_targets(boxes, valid, include_masks=False)
        np.testing.assert_array_equal(
            np.array(unpack_targets(default, False)["is_exhaustive"]),
            np.ones(3, np.float32))

    def test_more_gt_than_queries_round_trips_and_matches_min_q_n(
            self, tiny_model, synthetic_targets, rng):
        """``N_max = 6 > Q = 5``. The packing is unaffected (the two axes are
        independent) and the matcher assigns exactly ``min(Q, N)`` pairs, with
        the surplus GT contributing nothing."""
        boxes, valid, _ = synthetic_targets
        assert valid.shape[1] > tiny_model.num_queries

        # Boxes must be plausible cxcywh for the loss; the ROUND TRIP itself is
        # asserted on the banded fixture above.
        real_boxes = rng.uniform(0.15, 0.6, boxes.shape).astype("float32")
        packed_true = pack_targets(real_boxes, valid, include_masks=False)
        np.testing.assert_array_equal(
            np.array(unpack_targets(packed_true, False)["target_boxes"]),
            real_boxes)

        packed_pred = tiny_model(tiny_inputs(rng, batch=3), training=False)
        terms = Sam3DetectionLoss().compute_terms(packed_true, packed_pred)
        # image 0 zero-GT -> 0 pairs; image 1 has 6 GT vs Q=5 -> 5; image 2 -> 2
        assert float(np.array(terms["num_matched"])) == 7.0
        assert np.isfinite(float(np.array(terms["loss_bbox"])))

    def test_a_mixed_batch_round_trips_every_member(self, synthetic_targets):
        """One batch containing a zero-GT image, an ``N > Q`` image and a
        partially populated one -- packed together, unpacked, compared
        element-wise."""
        boxes, valid, masks = synthetic_targets
        packed = pack_targets(
            boxes, valid, target_masks=masks, include_masks=True)
        back = unpack_targets(packed, True)
        for index in range(valid.shape[0]):
            np.testing.assert_array_equal(
                np.array(back["target_boxes"])[index], boxes[index])
            np.testing.assert_array_equal(
                np.array(back["target_valid"])[index], valid[index])
            np.testing.assert_array_equal(
                np.array(back["target_masks"])[index],
                masks[index].reshape(masks.shape[1], -1))
        np.testing.assert_array_equal(
            np.array(back["keep_loss"]).ravel(), np.array([0.0, 1.0, 1.0]))

    def test_the_real_model_output_round_trips_against_sam3_itself(
            self, tiny_model, rng):
        """The wrapper's own ``call`` is the pack, so this compares it against
        the wrapped model's raw dict -- the one place the two can disagree."""
        inputs = tiny_inputs(rng, batch=2)
        packed = tiny_model(inputs, training=False)
        raw = tiny_model.sam3(inputs, training=False)
        back = unpack_predictions(packed, False)

        np.testing.assert_array_equal(
            np.array(back["pred_logits"]),
            np.array(raw["pred_logits"])[..., 0])
        np.testing.assert_array_equal(
            np.array(back["pred_boxes"]), np.array(raw["pred_boxes"]))
        np.testing.assert_array_equal(
            np.array(back["presence_logit"]),
            np.array(raw["presence_logit"]))


# ---------------------------------------------------------------------
# SC-7 -- the three-way width contract
# ---------------------------------------------------------------------


class TestThreeWayWidthAgreement:
    """The training model, the loss and the packer must agree on ``C``."""

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_model_loss_and_packer_land_on_one_width(
            self, include_masks, rng):
        keras.utils.set_random_seed(7)
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=include_masks)
        grid = model.mask_grid[0] * model.mask_grid[1]
        expected = packed_channel_count(grid if include_masks else 0)

        y_pred = model(tiny_inputs(rng, batch=2), training=False)
        boxes = rng.uniform(0.15, 0.6, (2, 3, 4)).astype("float32")
        valid = np.ones((2, 3), "float32")
        masks = rng.uniform(0.0, 1.0, (2, 3) + model.mask_grid).astype(
            "float32")
        y_true = pack_targets(
            boxes, valid, target_masks=masks if include_masks else None,
            include_masks=include_masks)

        assert model.packed_channels == expected
        assert int(y_pred.shape[-1]) == expected
        assert int(y_true.shape[-1]) == expected
        assert model.packed_target_spec(3) == (4, expected)

        # ...and the loss, the third party, actually consumes both.
        total = Sam3DetectionLoss(include_masks=include_masks)(y_true, y_pred)
        assert np.isfinite(float(np.array(total)))

    def test_compile_raises_when_the_loss_disagrees(self, tiny_model):
        with pytest.raises(ValueError, match="include_masks disagrees"):
            compile_sam3_trainer(
                tiny_model, loss=Sam3DetectionLoss(include_masks=True))

    def test_compile_raises_in_the_other_direction_too(self):
        keras.utils.set_random_seed(11)
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=True)
        with pytest.raises(ValueError, match="include_masks disagrees"):
            compile_sam3_trainer(
                model, loss=Sam3DetectionLoss(include_masks=False))

    def test_pack_targets_raises_on_a_mask_flag_disagreement(self, rng):
        boxes = rng.uniform(0.15, 0.6, (2, 3, 4)).astype("float32")
        valid = np.ones((2, 3), "float32")
        masks = rng.uniform(0.0, 1.0, (2, 3, 4, 4)).astype("float32")
        with pytest.raises(ValueError, match="requires target_masks"):
            pack_targets(boxes, valid, include_masks=True)
        with pytest.raises(ValueError, match="include_masks=False"):
            pack_targets(boxes, valid, target_masks=masks,
                         include_masks=False)

    def test_the_default_loss_always_agrees_with_the_model(self):
        """The failure-proof path: ``loss=None`` cannot disagree."""
        keras.utils.set_random_seed(13)
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=True)
        compile_sam3_trainer(model, optimizer="sgd")
        assert model.loss.include_masks is True


# ---------------------------------------------------------------------
# SC-8 -- a real stock fit() step
# ---------------------------------------------------------------------


class TestStockFit:
    """SC-8. No custom ``train_step``, ``jit_compile=False``, weights MOVE."""

    def test_one_fit_step_is_finite_and_moves_weights(self, rng):
        keras.utils.set_random_seed(2026)
        model = Sam3TrainingModel(Sam3Image.from_variant("tiny"))
        compile_sam3_trainer(model, optimizer=keras.optimizers.Adam(1e-2))
        assert model.jit_compile is False

        inputs = tiny_inputs(rng, batch=4)
        boxes = rng.uniform(0.2, 0.6, (4, 3, 4)).astype("float32")
        valid = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]],
                         "float32")
        y_true = np.array(pack_targets(boxes, valid, include_masks=False))

        # Build FIRST. `trainable_variables` is EMPTY on an unbuilt subclassed
        # model, so a before/after `zip` over it would compare nothing and
        # report "0 moved" whatever `fit()` did -- measured, this exact test
        # failed that way before the build was added.
        model(inputs, training=False)
        before = [np.array(v) for v in model.trainable_variables]
        assert before, "no trainable variables to compare"
        history = model.fit(inputs, y_true, epochs=1, batch_size=2, verbose=0)
        after = [np.array(v) for v in model.trainable_variables]

        loss = float(history.history["loss"][0])
        assert np.isfinite(loss), f"fit() loss is not finite: {loss}"

        moved = sum(1 for a, b in zip(before, after) if not np.array_equal(a, b))
        assert moved > 0, "fit() completed but no weight moved"
        # Not a token handful: a joint loss reaches most of the decoder.
        assert moved >= 100, f"only {moved} of {len(before)} variables moved"

    def test_an_all_negative_batch_is_finite_with_no_nan_gradient(self, rng):
        """Every image zero-GT. ``num_boxes`` clamps to 1, the classification
        term is fully gated to zero, and ``presence_loss`` supervises the
        negative -- the step must not divide by zero or NaN."""
        keras.utils.set_random_seed(99)
        model = Sam3TrainingModel(Sam3Image.from_variant("tiny"))
        loss_fn = Sam3DetectionLoss()

        inputs = tiny_inputs(rng, batch=3)
        boxes = np.zeros((3, 4, 4), "float32")
        valid = np.zeros((3, 4), "float32")
        y_true = pack_targets(boxes, valid, include_masks=False)
        assert float(np.max(np.array(
            unpack_targets(y_true, False)["keep_loss"]))) == 0.0

        with tf.GradientTape() as tape:
            y_pred = model(inputs, training=True)
            total = loss_fn(y_true, y_pred)
        grads = tape.gradient(total, model.trainable_variables)

        assert np.isfinite(float(np.array(total)))
        live = [g for g in grads if g is not None]
        assert live, "no gradient reached any variable"
        dense = [as_dense(g) for g in live]
        assert all(bool(np.all(np.isfinite(d))) for d in dense)
        assert any(float(np.max(np.abs(d))) > 0.0 for d in dense), \
            "an all-negative batch produced an entirely zero gradient"


# ---------------------------------------------------------------------
# training= forwarding, and serialization
# ---------------------------------------------------------------------


class TestTrainingFlagIsForwarded:
    """H-9 / D-123. ``training=None`` is NOT inference at a non-zero rate."""

    def test_training_true_and_false_differ_at_a_nonzero_drop_path_rate(
            self, rng):
        """The liveness arm for the explicit ``training=`` forwarding. At
        `tiny`'s shipped ``drop_path_rate=0.0`` this would be INERT, so the
        variant is built with the rate turned up on purpose."""
        keras.utils.set_random_seed(5)
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny", drop_path_rate=0.5))
        inputs = tiny_inputs(rng, batch=2)

        eval_a = np.array(model(inputs, training=False))
        eval_b = np.array(model(inputs, training=False))
        np.testing.assert_allclose(eval_a, eval_b, rtol=0.0, atol=0.0)

        train_outputs = [np.array(model(inputs, training=True))
                         for _ in range(6)]
        assert any(not np.allclose(t, eval_a) for t in train_outputs), \
            "training=True is indistinguishable from training=False"


class TestSerialization:
    """A `.keras` round trip compared by VALUE, at ``training=False``."""

    def test_get_config_round_trip_preserves_the_mask_switch(self):
        keras.utils.set_random_seed(3)
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=True)
        clone = Sam3TrainingModel.from_config(model.get_config())
        assert clone.include_masks is True
        assert clone.packed_channels == model.packed_channels
        assert clone.num_queries == model.num_queries

    def test_keras_round_trip_is_value_identical_at_training_false(self, rng):
        keras.utils.set_random_seed(21)
        model = Sam3TrainingModel(Sam3Image.from_variant("tiny"))
        inputs = tiny_inputs(rng, batch=2)
        original = np.array(model(inputs, training=False))

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sam3_trainer.keras")
            model.save(path)
            restored = keras.models.load_model(path)

        assert restored.include_masks == model.include_masks
        assert restored.packed_channels == model.packed_channels
        reloaded = np.array(restored(inputs, training=False))
        delta = float(np.abs(original - reloaded).max())
        # MEASURED 0.0 EXACTLY, in BOTH the TF32-on and TF32-off regimes on
        # this GPU (M5). The bound is loose only so a future variant's op
        # ordering cannot make a correct restore look broken.
        assert delta < 1e-5, (
            f"round-trip max |delta| = {delta}; a nested-sublayer weight loss "
            "restores FRESH kernels while every count and path still matches")

    def test_the_restored_model_still_trains(self, rng):
        """A round trip that restores an incomplete weight set can still
        produce matching outputs by luck; this checks the restored object is a
        working trainable model, not a frozen echo."""
        keras.utils.set_random_seed(31)
        model = Sam3TrainingModel(Sam3Image.from_variant("tiny"))
        model(tiny_inputs(rng, batch=2), training=False)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sam3_trainer.keras")
            model.save(path)
            restored = keras.models.load_model(path)

        assert len(restored.trainable_variables) == len(
            model.trainable_variables)
        compile_sam3_trainer(restored, optimizer=keras.optimizers.Adam(1e-2))
        boxes = rng.uniform(0.2, 0.6, (2, 2, 4)).astype("float32")
        y_true = np.array(pack_targets(
            boxes, np.ones((2, 2), "float32"), include_masks=False))
        before = [np.array(v) for v in restored.trainable_variables]
        restored.fit(tiny_inputs(rng, batch=2), y_true, epochs=1,
                     batch_size=2, verbose=0)
        after = [np.array(v) for v in restored.trainable_variables]
        assert any(not np.array_equal(a, b) for a, b in zip(before, after))


class TestOutputContract:
    """Shape / config surface the data pipeline and the trainer read."""

    def test_compute_output_shape_agrees_with_the_executed_forward(
            self, tiny_model, rng):
        declared = tiny_model.compute_output_shape(None)
        measured = tuple(
            tiny_model(tiny_inputs(rng, batch=2), training=False).shape)
        assert declared[1:] == measured[1:]
        assert declared[1] == TINY_QUERIES + 1

    def test_the_wrapper_adds_no_parameters_of_its_own(self, tiny_model):
        assert (tiny_model.count_params()
                == tiny_model.sam3.count_params())

    def test_a_non_sam3_argument_raises(self):
        with pytest.raises(ValueError, match="must be a Sam3Image"):
            Sam3TrainingModel(keras.layers.Dense(3))


# ---------------------------------------------------------------------
# Deep supervision -- guards G1 and G2, plus the model-vs-loss agreement leg.
#
# The layout: `[Q query rows | 1 presence row]` repeated `1 + n` times, the MAIN
# block FIRST, `y_true` untouched. G1 pins that the `deep_supervision=False`
# tensor did not move and that it is exactly the deep-supervised tensor's first
# block; G2 pins that every auxiliary block round-trips value-exactly. Each is
# RED-proven by its own mutation (a block reorder for G1, an off-by-one stride
# for G2).
# ---------------------------------------------------------------------


def shifted_outputs(outputs, shift):
    """A DISTINCT stand-in layer: every field moved by ``shift``.

    Distinct on purpose -- three copies of one block would make a stride error
    invisible, because every stride would read the same numbers.
    """
    return {key: (np.asarray(value) + shift).astype("float32")
            for key, value in outputs.items()}


class TestDeepSupervisionPacking:
    """G1/G2 -- the packed block layout, by VALUE."""

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_the_main_block_is_bit_identical_to_the_no_aux_packing(
            self, synthetic_outputs, include_masks):
        """G1. The first ``Q + 1`` rows of the deep-supervised tensor are
        byte-for-byte the tensor this packer has always produced -- which is
        itself pinned against the hand-written NumPy oracle here, so the
        comparison is not the implementation agreeing with itself.

        A block REORDER (auxiliary blocks first) fails this and nothing else.
        """
        aux = [shifted_outputs(synthetic_outputs, 1000.0),
               shifted_outputs(synthetic_outputs, 2000.0)]
        plain = np.array(pack_predictions(synthetic_outputs, include_masks))
        deep = np.array(pack_predictions(synthetic_outputs, include_masks,
                                         aux_outputs=aux))

        oracle = np_pack_predictions(
            synthetic_outputs["pred_logits"], synthetic_outputs["pred_boxes"],
            synthetic_outputs["presence_logit"],
            synthetic_outputs["pred_masks"] if include_masks else None)
        np.testing.assert_allclose(plain, oracle, rtol=0.0, atol=0.0)

        rows = plain.shape[1]
        assert deep.shape == (plain.shape[0], rows * 3, plain.shape[2])
        np.testing.assert_array_equal(deep[:, :rows, :], plain)
        # Non-vacuity: the auxiliary blocks are NOT the main block, so the
        # equality above is a statement about ORDER and not a tautology.
        assert not np.array_equal(deep[:, rows:2 * rows, :], plain)

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_unpack_round_trips_every_auxiliary_block_value_exactly(
            self, synthetic_outputs, include_masks):
        """G2. Every block, main and auxiliary, comes back with the values it
        went in with. An off-by-one block STRIDE fails this and nothing else."""
        aux = [shifted_outputs(synthetic_outputs, 1000.0),
               shifted_outputs(synthetic_outputs, 2000.0)]
        packed = pack_predictions(synthetic_outputs, include_masks,
                                  aux_outputs=aux)
        back = unpack_predictions(packed, include_masks, num_aux_layers=2)

        assert len(back["aux"]) == 2
        for block, expected in zip([back] + back["aux"],
                                   [synthetic_outputs] + aux):
            np.testing.assert_array_equal(
                np.array(block["pred_logits"]),
                np.array(expected["pred_logits"])[..., 0])
            np.testing.assert_array_equal(
                np.array(block["pred_boxes"]),
                np.array(expected["pred_boxes"]))
            np.testing.assert_array_equal(
                np.array(block["presence_logit"]),
                np.array(expected["presence_logit"]))

    def test_an_auxiliary_blocks_mask_channels_are_zero_filled(
            self, synthetic_outputs):
        """D-005: the auxiliary blocks carry NO masks, because the loss
        computes no mask term for them. The main block's are non-zero in the
        same tensor, so this is not measuring an all-zero mask block."""
        aux = [shifted_outputs(synthetic_outputs, 1000.0)]
        packed = np.array(pack_predictions(synthetic_outputs, True,
                                           aux_outputs=aux))
        rows = packed.shape[1] // 2
        queries = rows - 1
        assert np.all(packed[:, rows:rows + queries, PACKED_MASK_START:] == 0.0)
        assert np.abs(packed[:, :queries, PACKED_MASK_START:]).min() > 0.0

    def test_the_presence_row_of_every_block_is_zero_outside_its_score(
            self, synthetic_outputs):
        aux = [shifted_outputs(synthetic_outputs, 1000.0)]
        packed = np.array(pack_predictions(synthetic_outputs, True,
                                           aux_outputs=aux))
        rows = packed.shape[1] // 2
        for block in range(2):
            presence_row = packed[:, (block + 1) * rows - 1, :]
            assert np.all(presence_row[:, PACKED_BOX_START:] == 0.0)
            assert np.all(np.abs(presence_row[:, PACKED_SCORE_CHANNEL]) > 1.0)


class TestDeepSupervisionOnTheRealModel:
    """The wrapper end to end: same weights, one flag, two layouts."""

    @pytest.fixture(scope="class")
    def deep_model(self, tiny_model):
        """The SAME `Sam3Image`, wrapped with deep supervision on.

        Sharing the wrapped model is what makes the bit-equality below a
        statement about the LAYOUT rather than about two random inits.
        """
        model = Sam3TrainingModel(tiny_model.sam3, deep_supervision=True)
        model.build(None)
        return model

    def test_the_aux_count_is_derived_from_the_decoder_depth(
            self, tiny_model, deep_model):
        assert tiny_model.num_aux_layers == 0
        assert deep_model.num_aux_layers == (
            tiny_model.sam3.transformer.num_layers - 1)
        assert deep_model.num_aux_layers >= 1

    def test_compute_output_shape_counts_every_block(self, deep_model):
        expected_rows = ((deep_model.num_queries + 1)
                         * (1 + deep_model.num_aux_layers))
        assert deep_model.compute_output_shape() == (
            None, expected_rows, deep_model.packed_channels)

    def test_the_main_block_is_bit_identical_to_the_flag_off_output(
            self, tiny_model, deep_model, rng):
        """G1 on the real wrapper: turning the flag on APPENDS, it does not
        perturb. `training=False` is explicit -- at `None` this stack drops
        paths and a bit-equality claim would be comparing two draws (D-123)."""
        inputs = tiny_inputs(rng, batch=2)
        plain = np.array(tiny_model(inputs, training=False))
        deep = np.array(deep_model(inputs, training=False))
        rows = plain.shape[1]
        assert deep.shape[1] == rows * (1 + deep_model.num_aux_layers)
        np.testing.assert_array_equal(deep[:, :rows, :], plain)
        assert not np.array_equal(deep[:, rows:2 * rows, :], plain)

    def test_a_deep_supervised_step_runs_and_moves_weights(
            self, deep_model, synthetic_targets, rng):
        """A finite loss alone cannot see a step that moves nothing."""
        boxes, valid, _ = synthetic_targets
        boxes, valid = boxes[:2], valid[:2]
        inputs = tiny_inputs(rng, batch=2)
        y_true = np.array(pack_targets(boxes, valid))

        compile_sam3_trainer(
            deep_model, optimizer=keras.optimizers.SGD(learning_rate=1.0))
        before = [np.array(w) for w in deep_model.sam3.transformer.weights]
        history = deep_model.fit(inputs, y_true, epochs=1, batch_size=2,
                                 verbose=0)
        after = [np.array(w) for w in deep_model.sam3.transformer.weights]

        assert np.isfinite(history.history["loss"][0])
        assert any(not np.array_equal(a, b) for a, b in zip(before, after))

    def test_masks_on_AND_deep_supervision_on_zero_fills_only_the_aux_masks(
            self, tiny_model, rng):
        """D-005's PACKER half at the one combination nothing else builds.

        `test_an_auxiliary_blocks_mask_channels_are_zero_filled` above pins the
        same rule on `pack_predictions` fed SYNTHETIC dicts, and the loss side
        is pinned at `include_masks=True` with a synthetic aux stack -- but
        before this test no test constructed
        `Sam3TrainingModel(include_masks=True, deep_supervision=True)`, so the
        packer's `masks=None` on the auxiliary block was reviewed once and
        never executed by the suite (review-iter-1 NOTE 10).

        Three assertions, and the last two are what stop the first from being
        vacuous: a packer that zero-filled EVERY mask channel, or emitted
        constant auxiliary blocks, would satisfy assertion 1 alone.
        """
        model = Sam3TrainingModel(tiny_model.sam3, include_masks=True,
                                  deep_supervision=True)
        model.build(None)
        assert model.num_aux_layers >= 1

        packed = np.array(model(tiny_inputs(rng, batch=2), training=False))
        rows = model.num_queries + 1
        assert packed.shape[1] == rows * (1 + model.num_aux_layers)
        assert packed.shape[2] > PACKED_MASK_START

        main = packed[:, :rows, :]
        aux = packed[:, rows:, :]
        # 1. every auxiliary block's mask channels are EXACTLY zero.
        np.testing.assert_array_equal(
            aux[:, :, PACKED_MASK_START:],
            np.zeros_like(aux[:, :, PACKED_MASK_START:]))
        # 2. the MAIN block's mask channels are not (so "all zero" is a
        #    property of the auxiliary blocks, not of the mask block).
        assert np.max(np.abs(main[:, :, PACKED_MASK_START:])) > 0.0
        # 3. the auxiliary blocks' BOX channels are not zero either (so the
        #    auxiliary blocks carry real per-layer predictions).
        assert np.max(np.abs(
            aux[:, :, PACKED_BOX_START:PACKED_MASK_START])) > 0.0

    def test_deep_supervision_round_trips_through_get_config(self, deep_model):
        config = deep_model.get_config()
        assert config["deep_supervision"] is True
        restored = Sam3TrainingModel.from_config(config)
        assert restored.deep_supervision is True
        assert restored.num_aux_layers == deep_model.num_aux_layers


class TestDeepSupervisionSerialization:
    """The `.keras` round trip at ``deep_supervision=True``, compared by VALUE.

    `TestSerialization` above covers the `deep_supervision=False` layout only,
    so before this class NOTHING save/load-tested the packed AUXILIARY blocks --
    i.e. exactly the `call_per_layer` -> aux-block path. That is the package's
    own recorded trap: a nested sub-layer store can restore FRESH kernels while
    the sub-layer count, the weight PATHS and the total parameter count all
    still match, and only a VALUE diff sees it.

    `training=False` is explicit on BOTH sides: the shared `StochasticDepth`
    short-circuits on `training is False` only, so at `None` a CORRECT restore
    reads deltas that look exactly like reinitialized weights (D-123).
    """

    @pytest.fixture(scope="class")
    def round_trip(self):
        """One save/load pair, shared -- `tiny` construction dominates here.

        Returns ``(original_model, restored_model, inputs, original_output)``.
        """
        keras.utils.set_random_seed(77)
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), deep_supervision=True)
        inputs = tiny_inputs(np.random.default_rng(20260806), batch=2)
        original = np.array(model(inputs, training=False))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sam3_deep_trainer.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        return model, restored, inputs, original

    def test_the_auxiliary_block_being_compared_is_not_vacuous(
            self, round_trip):
        """Anti-vacuity, asserted BEFORE the equality claim below is trusted.

        An exact-equality assertion over an all-zero block, or over a block
        that merely repeats the main block, would pass on a packer that never
        wrote the auxiliary layer's outputs at all.
        """
        model, _, _, original = round_trip
        rows = model.num_queries + 1
        assert model.num_aux_layers >= 1, (
            "`tiny` has 2 decoder layers, so this arm must exercise at least "
            "one auxiliary block")
        assert original.shape[1] == rows * (1 + model.num_aux_layers)
        aux_block = original[:, rows:2 * rows, :]
        assert np.abs(aux_block).max() > 0.0
        assert not np.array_equal(aux_block, original[:, :rows, :])

    def test_the_restored_deep_model_declares_the_same_layout(
            self, round_trip):
        model, restored, _, _ = round_trip
        assert restored.deep_supervision is True
        assert restored.num_aux_layers == model.num_aux_layers
        assert restored.packed_channels == model.packed_channels
        assert len(restored.trainable_variables) == len(
            model.trainable_variables)

    def test_keras_round_trip_is_value_identical_with_deep_supervision(
            self, round_trip):
        """MEASURED exactly 0.0 over every block, main and auxiliary.

        RED-proven by perturbing a single restored kernel: this assertion --
        and only this one -- fires. The companion test below keeps that proof
        live in the suite rather than leaving it in a one-off scratch run.
        """
        _, restored, inputs, original = round_trip
        reloaded = np.array(restored(inputs, training=False))
        assert reloaded.shape == original.shape
        delta = float(np.abs(original - reloaded).max())
        assert delta == 0.0, (
            f"deep-supervised round-trip max |delta| = {delta}; a nested "
            "sub-layer weight loss restores FRESH kernels while every count "
            "and path still matches")

    def test_the_delta_reader_sees_a_single_perturbed_restored_weight(
            self, round_trip):
        """Liveness for the assertion above: it compares VALUES, not shapes.

        A count/shape/path-only assertion passes on a model that restored zero
        weights. Perturbing ONE restored kernel by 0.1 must move the very
        quantity the guard reads; the perturbation is undone afterwards so the
        shared fixture is left as found.
        """
        _, restored, inputs, original = round_trip
        victim = restored.sam3.transformer.trainable_variables[0]
        before = np.array(victim)
        try:
            victim.assign(before + 0.1)
            perturbed = np.array(restored(inputs, training=False))
            assert float(np.abs(original - perturbed).max()) > 0.0
        finally:
            victim.assign(before)
        assert float(np.abs(
            original - np.array(restored(inputs, training=False))).max()) == 0.0


class TestNumAuxLayersAgreement:
    """The model-vs-loss agreement leg, on the ROW axis this time."""

    @pytest.fixture(scope="class")
    def deep_model(self, tiny_model):
        model = Sam3TrainingModel(tiny_model.sam3, deep_supervision=True)
        model.build(None)
        return model

    def test_compile_raises_when_the_loss_disagrees_on_num_aux_layers(
            self, deep_model):
        with pytest.raises(ValueError, match="num_aux_layers"):
            compile_sam3_trainer(deep_model, loss=Sam3DetectionLoss())

    def test_compile_raises_when_the_loss_expects_aux_and_the_model_does_not(
            self, tiny_model):
        with pytest.raises(ValueError, match="num_aux_layers"):
            compile_sam3_trainer(
                tiny_model, loss=Sam3DetectionLoss(num_aux_layers=2))

    def test_a_default_constructed_loss_inherits_the_models_aux_count(
            self, deep_model):
        compile_sam3_trainer(deep_model)
        assert deep_model.loss.num_aux_layers == deep_model.num_aux_layers
        assert deep_model.jit_compile is False


# ---------------------------------------------------------------------
# Encoder query selection -- the packed-block ARITHMETIC (invariant I-5) and
# the block ORDER.
#
# `num_aux_layers = (L - 1 if deep_supervision else 0) + (1 if query_selection
# else 0)`. The dangerous combination is `{deep_supervision=False,
# query_selection=True}`: `call_per_layer(include_proposals=True)` returns all
# `L` decoder blocks with the encoder block after them, so a packer that spells
# the auxiliary blocks `per_layer[1:]` yields `L` blocks where `num_aux_layers`
# says `1` -- and `unpack_predictions` validates NOTHING, so the loss then
# slices garbage and reports six finite, plausible, fabricated per-term losses
# instead of raising. Every assertion below is on a MEASURED row count or a
# MEASURED value, never on the flag.
# ---------------------------------------------------------------------

#: The four flag combinations and their `num_aux_layers` at an L-layer decoder,
#: written as the ARITHMETIC rather than as numbers, so the table cannot quietly
#: be re-fitted to whatever the implementation happens to return. The literal
#: `L=3` numbers `0 / 2 / 1 / 3` are asserted separately below.
COMPOSITION_TABLE = (
    (False, False, lambda layers: 0),
    (True, False, lambda layers: max(layers - 1, 0)),
    (False, True, lambda layers: 1),
    (True, True, lambda layers: max(layers - 1, 0) + 1),
)


def query_selection_sam3(decoder_layers=3):
    """A `tiny` model WITH the proposal head, at a chosen decoder depth.

    `tiny` ships 2 decoder layers; the plan's composition table is stated at
    `L = 3`, and `L = 1` is a named edge case, so the depth is a parameter.
    """
    keras.utils.set_random_seed(4242)
    return Sam3Image.from_variant(
        "tiny", decoder_layers=decoder_layers, query_selection=True)


def built_wrapper(sam3, deep_supervision, query_selection):
    model = Sam3TrainingModel(sam3, deep_supervision=deep_supervision,
                              query_selection=query_selection)
    model.build(None)
    return model


class TestNumAuxLayersComposition:
    """I-5, asserted at all four flag combinations plus the `L = 1` edge."""

    @pytest.fixture(scope="class")
    def sam3_three(self):
        return query_selection_sam3(decoder_layers=3)

    def test_the_fixture_really_has_three_decoder_layers(self, sam3_three):
        """The premise of every literal in the table below."""
        assert int(sam3_three.transformer.num_layers) == 3
        assert sam3_three.query_selection is True
        assert sam3_three.query_selection_head is not None

    def test_the_four_combination_table_at_three_decoder_layers(
            self, sam3_three):
        """`{F,F} -> 0`, `{T,F} -> 2`, `{F,T} -> 1`, `{T,T} -> 3`.

        RED-proven by dropping the `+ (1 if query_selection else 0)` term from
        the composition: the `{F,T}` and `{T,T}` rows fire.
        """
        expected = {(False, False): 0, (True, False): 2,
                    (False, True): 1, (True, True): 3}
        measured = {}
        for deep, query in expected:
            model = built_wrapper(sam3_three, deep, query)
            measured[(deep, query)] = model.num_aux_layers
        assert measured == expected

    def test_the_table_is_the_arithmetic_and_not_three_hardcoded_numbers(
            self, sam3_three):
        layers = int(sam3_three.transformer.num_layers)
        for deep, query, rule in COMPOSITION_TABLE:
            model = built_wrapper(sam3_three, deep, query)
            assert model.num_aux_layers == rule(layers), (
                f"deep_supervision={deep}, query_selection={query}")

    def test_a_single_layer_decoder_still_gets_the_encoder_block(self):
        """`L = 1`: deep supervision contributes 0, query selection still 1.

        The named edge case. A composition spelled `L - 1 + qs` with the deep
        supervision term NOT gated by its own flag reads the same here, which is
        why the `{F,T}` row is asserted beside the `{T,T}` one.
        """
        sam3 = query_selection_sam3(decoder_layers=1)
        assert int(sam3.transformer.num_layers) == 1
        assert built_wrapper(sam3, True, False).num_aux_layers == 0
        assert built_wrapper(sam3, False, True).num_aux_layers == 1
        assert built_wrapper(sam3, True, True).num_aux_layers == 1

    def test_query_selection_without_a_proposal_head_raises(self, tiny_model):
        """The wrapper flag cannot out-run the wrapped model.

        Without this raise, `call_per_layer(include_proposals=True)` would
        append NO block (there are no proposals) while `num_aux_layers` counted
        one, and the loss would slice a stride that does not exist.
        """
        assert tiny_model.sam3.query_selection is False
        with pytest.raises(ValueError, match="query_selection"):
            Sam3TrainingModel(tiny_model.sam3, query_selection=True)


class TestComputeOutputShapeFollowsTheComposition:
    """`compute_output_shape` is VERIFIED against the real forward pass."""

    @pytest.fixture(scope="class")
    def sam3_three(self):
        return query_selection_sam3(decoder_layers=3)

    @pytest.mark.parametrize("deep,query", [(False, False), (True, False),
                                            (False, True), (True, True)])
    def test_the_declared_shape_equals_the_measured_one(
            self, sam3_three, rng, deep, query):
        """Declared and MEASURED, at every combination.

        A declared shape that merely restates `num_aux_layers` proves nothing
        about what `call` packs; the forward pass is what closes that gap, and
        it is the quantity `unpack_predictions` divides by.
        """
        model = built_wrapper(sam3_three, deep, query)
        rows = (model.num_queries + 1) * (1 + model.num_aux_layers)
        assert model.compute_output_shape() == (
            None, rows, model.packed_channels)

        packed = np.array(model(tiny_inputs(rng, batch=2), training=False))
        assert packed.shape == (2, rows, model.packed_channels)


class TestEncoderBlockIsPackedLast:
    """The ORDER, pinned by VALUE -- the loss weights every block equally."""

    @pytest.fixture(scope="class")
    def sam3_three(self):
        return query_selection_sam3(decoder_layers=3)

    def test_the_last_packed_block_is_the_encoder_one_at_both_flags_on(
            self, sam3_three):
        """`{T,T}`: blocks are `[main, dec_0, dec_1, ENCODER]`, in that order.

        The loss applies the SAME per-block terms at the SAME weight to every
        block (D-004 of `plan-2026-08-06T055747-1e650383`), so a wrong ORDER is
        INVISIBLE in the loss value and wrong in every per-block diagnostic.
        Only a value comparison per block position can see it.
        """
        model = built_wrapper(sam3_three, True, True)
        inputs = tiny_inputs(np.random.default_rng(31337), batch=2)
        blocks = model.sam3.call_per_layer(
            inputs, training=False, include_proposals=True)
        assert len(blocks) == 4, "3 decoder blocks + 1 encoder block"

        packed = np.array(model(inputs, training=False))
        rows = model.num_queries + 1
        assert packed.shape[1] == rows * 4

        boxes = slice(PACKED_BOX_START, PACKED_MASK_START)
        for index, block in enumerate(blocks):
            np.testing.assert_array_equal(
                packed[:, index * rows:index * rows + model.num_queries,
                       boxes],
                np.array(block["pred_boxes"]),
                err_msg=f"block {index} is not the one packed at that offset")

        # Non-vacuity: the four blocks are DISTINCT, so the per-position
        # equalities above are statements about order and not tautologies.
        packed_boxes = [packed[:, i * rows:i * rows + model.num_queries, boxes]
                        for i in range(4)]
        for index in range(3):
            assert not np.array_equal(packed_boxes[index], packed_boxes[3]), (
                f"decoder block {index} equals the encoder block; the order "
                "assertion above cannot discriminate")

    def test_at_query_selection_only_the_single_aux_block_is_the_encoder_one(
            self, sam3_three):
        """`{F,T}` -- the combination no prior run has produced.

        This is the mis-slice trap in the flesh: `call_per_layer` still returns
        all 3 decoder blocks, and the ONE auxiliary block packed must be the
        ENCODER's, not decoder layer 0's.
        """
        model = built_wrapper(sam3_three, False, True)
        assert model.num_aux_layers == 1
        inputs = tiny_inputs(np.random.default_rng(31338), batch=2)
        blocks = model.sam3.call_per_layer(
            inputs, training=False, include_proposals=True)

        packed = np.array(model(inputs, training=False))
        rows = model.num_queries + 1
        assert packed.shape[1] == rows * 2
        boxes = slice(PACKED_BOX_START, PACKED_MASK_START)
        aux = packed[:, rows:rows + model.num_queries, boxes]

        np.testing.assert_array_equal(aux, np.array(blocks[-1]["pred_boxes"]))
        for index in (1, 2):
            assert not np.array_equal(
                aux, np.array(blocks[index]["pred_boxes"])), (
                f"the single auxiliary block is decoder block {index}, not the "
                "encoder's")

    def test_the_encoder_block_carries_the_heads_own_selected_boxes(
            self, sam3_three):
        """One more link back: the last block IS the proposal head's output.

        `call_per_layer` is the intermediary; this asserts the packed rows
        against the HEAD's `selected_boxes` directly, so a block that merely
        looked encoder-shaped cannot pass.
        """
        model = built_wrapper(sam3_three, True, True)
        inputs = tiny_inputs(np.random.default_rng(31339), batch=2)
        proposals = model.sam3._forward_all(inputs, training=False)[4]
        packed = np.array(model(inputs, training=False))
        rows = model.num_queries + 1
        np.testing.assert_array_equal(
            packed[:, -rows:-1, PACKED_BOX_START:PACKED_MASK_START],
            np.array(proposals["selected_boxes"]))
        np.testing.assert_array_equal(
            packed[:, -rows:-1, PACKED_SCORE_CHANNEL],
            np.array(proposals["selected_objectness"])[..., 0])


class TestSelectPredictionBlocks:
    """The selection helper itself, on hand-built stand-ins.

    One home for "which blocks get packed" -- `Sam3TrainingModel.call` and
    `train.sam3.train_sam3.evaluate_sam3` both call it, and they drifted once
    (D-006). Testing it directly is what makes both call sites cheap to trust.
    """

    @staticmethod
    def _blocks(count):
        return [{"tag": index} for index in range(count)]

    def test_it_reproduces_the_composition_table(self):
        # 3 decoder blocks + 1 encoder block.
        with_encoder = self._blocks(4)
        without = self._blocks(3)
        assert select_prediction_blocks(without, False, False)[1] == []
        assert select_prediction_blocks(without, True, False)[1] == without[1:]
        assert select_prediction_blocks(
            with_encoder, False, True)[1] == [with_encoder[3]]
        assert select_prediction_blocks(
            with_encoder, True, True)[1] == with_encoder[1:]

    def test_the_main_block_is_always_element_zero(self):
        blocks = self._blocks(4)
        for deep, query in ((False, False), (True, False), (False, True),
                            (True, True)):
            source = blocks if query else blocks[:3]
            assert select_prediction_blocks(
                source, deep, query)[0] is source[0]

    def test_the_encoder_block_is_never_taken_as_a_decoder_auxiliary(self):
        """`{F,T}` must select exactly one block and it must be the LAST."""
        blocks = self._blocks(4)
        main, aux = select_prediction_blocks(blocks, False, True)
        assert main == {"tag": 0}
        assert aux == [{"tag": 3}]

    def test_a_disagreeing_expected_count_raises(self):
        with pytest.raises(ValueError, match="mis-slice"):
            select_prediction_blocks(self._blocks(4), False, True,
                                     expected_aux=3)
        # ...and the agreeing count does not.
        select_prediction_blocks(self._blocks(4), True, True, expected_aux=3)


class TestTheLossReadsTheEncoderBlock:
    """The MANDATED two-mutation guard (the G3-A / G3-B lesson).

    "The loss reads the encoder block" is ONE guard with TWO assertions, each
    with its own mutation, because the prior plan MEASURED that the gradient
    assertion alone stays GREEN under the second mutation's condition:

    * A (`test_the_proposal_head_receives_gradient_only_through_its_own_block`)
      is RED when `pack_predictions` receives a `stop_gradient`-ed encoder
      block -- the head's weights then have no route to the loss at all.
    * B (`test_the_compiled_loss_slices_the_encoder_block_as_the_last_aux`) is
      RED when the LOSS's `num_aux_layers` excludes the encoder block while the
      model still packs it: the stride moves, and the block the loss reads last
      is no longer the encoder's rows.
    """

    @pytest.fixture(scope="class")
    def model(self):
        return built_wrapper(query_selection_sam3(decoder_layers=3),
                             True, True)

    def test_the_proposal_head_receives_gradient_only_through_its_own_block(
            self, model, synthetic_targets):
        """A: the packed encoder block is the head's ONLY gradient route.

        The proposals enter the decoder `stop_gradient`-ed (D-006), so if the
        encoder block were dropped from the packing -- or packed detached --
        every one of these weights would be dead while the loss stayed finite
        and every shape stayed right.

        MEASURED, and NOT asserted as "all twelve weights move": at
        INITIALIZATION exactly 8 of the head's 12 weights get a non-zero
        gradient. `box_head_2`'s kernel is ZERO-initialized (D-112's
        precedent, so every proposal starts exactly at its grid anchor), and a
        zero last kernel back-propagates exactly zero into the stack behind it
        -- `box_head_0` and `box_head_1` are dead until `box_head_2` itself
        moves. That is a property of the initialization, not of the packing,
        and the companion `fit()` test below shows those four waking up on the
        second step. Asserting "all twelve" here would have been a wrong
        expectation dressed as a guard.
        """
        boxes, valid, _ = synthetic_targets
        inputs = tiny_inputs(np.random.default_rng(555), batch=2)
        y_true = ops.convert_to_tensor(
            np.array(pack_targets(boxes[:2], valid[:2])))
        loss = Sam3DetectionLoss(num_aux_layers=model.num_aux_layers)
        head = model.sam3.query_selection_head
        assert head.trainable_variables, "the head owns no weights"

        with tf.GradientTape() as tape:
            value = loss(y_true, model(inputs, training=True))
        grads = tape.gradient(value, head.trainable_variables)

        assert np.isfinite(float(ops.convert_to_numpy(value)))
        norms = {}
        for grad, weight in zip(grads, head.trainable_variables):
            assert grad is not None, f"{weight.path} got a None gradient"
            norms[weight.path.split("/")[-2] + "/"
                  + weight.path.split("/")[-1]] = float(
                      np.abs(as_dense(grad)).max())

        # The objectness stack is live end to end -- it is what SELECTS, and a
        # dead objectness head is this mechanism's named vacuity mode.
        for index in range(3):
            for field in ("kernel", "bias"):
                key = f"objectness_head_{index}/{field}"
                assert norms[key] > 0.0, f"{key} got an all-zero gradient"
        # ...and the box stack is live at its last projection, which is the
        # only place it CAN be live at initialization.
        assert norms["box_head_2/kernel"] > 0.0
        assert norms["box_head_2/bias"] > 0.0
        # The zero-init consequence, pinned as the measurement it is rather
        # than left as an unexplained gap in the assertion above.
        assert norms["box_head_0/kernel"] == 0.0
        assert norms["box_head_1/kernel"] == 0.0

    def test_the_compiled_loss_slices_the_encoder_block_as_the_last_aux(
            self, model):
        """B: the rows the loss reads LAST are the head's own selected boxes.

        `compile_sam3_trainer` constructs the loss, so this measures the number
        the trainer actually ships -- not one the test chose. It then unpacks
        the model's real packed tensor AT THAT STRIDE and compares the last
        auxiliary block against the proposal head's output value-exactly.
        """
        compile_sam3_trainer(model)
        inputs = tiny_inputs(np.random.default_rng(556), batch=2)
        proposals = model.sam3._forward_all(inputs, training=False)[4]
        packed = model(inputs, training=False)
        # Unpacked at the LOSS's OWN stride, not the model's: that is the
        # number the compiled trainer actually slices with, and the two
        # disagreeing is the whole failure mode.
        back = unpack_predictions(packed, model.include_masks,
                                  model.loss.num_aux_layers)

        np.testing.assert_array_equal(
            np.array(back["aux"][-1]["pred_boxes"]),
            np.array(proposals["selected_boxes"]))
        # Non-vacuity: the last auxiliary block is not the main block, so the
        # equality above is a statement about the STRIDE.
        assert not np.array_equal(
            np.array(back["aux"][-1]["pred_boxes"]),
            np.array(back["pred_boxes"]))


class TestNumAuxLayersAgreementAtEveryCombination:
    """`compile_sam3_trainer`'s raise must keep passing at all four."""

    @pytest.fixture(scope="class")
    def sam3_three(self):
        return query_selection_sam3(decoder_layers=3)

    @pytest.mark.parametrize("deep,query", [(False, False), (True, False),
                                            (False, True), (True, True)])
    def test_the_default_loss_inherits_the_composed_count(
            self, sam3_three, deep, query):
        model = built_wrapper(sam3_three, deep, query)
        compile_sam3_trainer(model)
        assert model.loss.num_aux_layers == model.num_aux_layers
        assert model.jit_compile is False

    @pytest.mark.parametrize("deep,query", [(False, False), (True, False),
                                            (False, True), (True, True)])
    def test_a_loss_that_disagrees_by_one_still_raises(
            self, sam3_three, deep, query):
        """Off by one in EITHER direction, at every combination."""
        model = built_wrapper(sam3_three, deep, query)
        with pytest.raises(ValueError, match="num_aux_layers"):
            compile_sam3_trainer(
                model,
                loss=Sam3DetectionLoss(
                    num_aux_layers=model.num_aux_layers + 1))
        if model.num_aux_layers:
            with pytest.raises(ValueError, match="num_aux_layers"):
                compile_sam3_trainer(
                    model,
                    loss=Sam3DetectionLoss(
                        num_aux_layers=model.num_aux_layers - 1))


class TestQuerySelectionSerialization:
    """The `.keras` round trip with the encoder block in the layout.

    `training=False` on BOTH sides: at `None` a CORRECT restore reads deltas
    that look exactly like reinitialized weights (D-123). Quoted from a CPU run
    -- GPU 1 is not bit-reproducible run to run (~5e-6 on a `pred_boxes` sum,
    measured on a pristine tree in step 5), so an exact-delta claim taken there
    would be a coin flip.
    """

    @pytest.fixture(scope="class", params=[(False, True), (True, True)],
                    ids=["query_only", "both_flags"])
    def round_trip(self, request):
        deep, query = request.param
        keras.utils.set_random_seed(88)
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny", decoder_layers=3,
                                   query_selection=True),
            deep_supervision=deep, query_selection=query)
        model.build(None)
        inputs = tiny_inputs(np.random.default_rng(20260807), batch=2)
        original = np.array(model(inputs, training=False))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sam3_qsel_trainer.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        return model, restored, inputs, original

    def test_the_encoder_block_being_compared_is_not_vacuous(self, round_trip):
        model, _, _, original = round_trip
        rows = model.num_queries + 1
        assert original.shape[1] == rows * (1 + model.num_aux_layers)
        encoder_block = original[:, -rows:, :]
        assert np.abs(encoder_block).max() > 0.0
        assert not np.array_equal(encoder_block, original[:, :rows, :])

    def test_the_restored_model_declares_the_same_layout(self, round_trip):
        model, restored, _, _ = round_trip
        assert restored.query_selection is True
        assert restored.deep_supervision is model.deep_supervision
        assert restored.num_aux_layers == model.num_aux_layers
        assert restored.sam3.query_selection is True
        assert len(restored.trainable_variables) == len(
            model.trainable_variables)

    def test_the_round_trip_is_value_identical(self, round_trip):
        _, restored, inputs, original = round_trip
        reloaded = np.array(restored(inputs, training=False))
        assert reloaded.shape == original.shape
        delta = float(np.abs(original - reloaded).max())
        assert delta == 0.0, (
            f"query-selection round-trip max |delta| = {delta}; a nested "
            "sub-layer weight loss restores FRESH kernels while every count "
            "and path still matches")

    def test_the_delta_reader_sees_a_perturbed_proposal_head_weight(
            self, round_trip):
        """Liveness, on the PROPOSAL HEAD specifically.

        The generic round-trip liveness arm perturbs a decoder weight, which
        would still pass on a restore that dropped the proposal head entirely.
        """
        _, restored, inputs, original = round_trip
        victim = restored.sam3.query_selection_head.trainable_variables[0]
        before = np.array(victim)
        try:
            victim.assign(before + 0.5)
            perturbed = np.array(restored(inputs, training=False))
            assert float(np.abs(original - perturbed).max()) > 0.0
        finally:
            victim.assign(before)
        assert float(np.abs(
            original - np.array(restored(inputs, training=False))).max()) == 0.0

    def test_query_selection_round_trips_through_get_config(self, round_trip):
        model, _, _, _ = round_trip
        config = model.get_config()
        assert config["query_selection"] is True
        restored = Sam3TrainingModel.from_config(config)
        assert restored.query_selection is True
        assert restored.num_aux_layers == model.num_aux_layers


class TestAStepRunsWithTheEncoderBlockPacked:
    """A real `fit()` step at `{T,T}`, asserting the HEAD's weights moved."""

    def test_a_few_steps_move_every_proposal_head_weight(self,
                                                         synthetic_targets):
        """A finite loss alone cannot see a step that moves nothing.

        THREE epochs, deliberately: `box_head_2`'s kernel is zero-initialized,
        so on the FIRST step it back-propagates exactly zero into `box_head_0`
        and `box_head_1` (measured in the gradient guard above). They move only
        once `box_head_2` is no longer zero. A one-epoch version of this test
        would have to exempt four weights and would then be blind to a genuinely
        dead box stack.

        `learning_rate=0.05`, not the `1.0` the deep-supervision step test uses:
        at `1.0` this model diverges within two steps and the matcher then
        raises `matrix contains invalid numeric entries` on a NaN cost -- a real
        failure of the test's own setup, not of the code under test.
        """
        model = built_wrapper(query_selection_sam3(decoder_layers=3),
                              True, True)
        boxes, valid, _ = synthetic_targets
        inputs = tiny_inputs(np.random.default_rng(999), batch=2)
        y_true = np.array(pack_targets(boxes[:2], valid[:2]))

        compile_sam3_trainer(
            model, optimizer=keras.optimizers.SGD(learning_rate=0.05))
        head = model.sam3.query_selection_head
        before = [np.array(w) for w in head.trainable_variables]
        history = model.fit(inputs, y_true, epochs=3, batch_size=2, verbose=0)
        after = [np.array(w) for w in head.trainable_variables]

        assert np.isfinite(history.history["loss"][0])
        for weight, first, second in zip(head.trainable_variables, before,
                                         after):
            assert not np.array_equal(first, second), (
                f"{weight.path} did not move in three real fit() steps")
