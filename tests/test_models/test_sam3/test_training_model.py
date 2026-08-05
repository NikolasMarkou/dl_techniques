"""Tests for ``models/sam3/training_model.py`` -- the packed-tensor wrapper.

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
from dl_techniques.models.sam3 import Sam3Image
from dl_techniques.models.sam3.training_model import (
    Sam3TrainingModel,
    compile_sam3_trainer,
    pack_predictions,
    pack_targets,
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
