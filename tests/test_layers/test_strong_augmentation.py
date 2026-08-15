"""Tests for the StrongAugmentation layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.strong_augmentation import StrongAugmentation

B, H, W, C = 4, 8, 8, 3


@pytest.fixture
def sample():
    return np.random.default_rng(0).uniform(0, 1, size=(B, H, W, C)).astype("float32")


class TestStrongAugmentation:

    def test_construction(self):
        layer = StrongAugmentation(cutmix_prob=0.5)
        assert layer.cutmix_prob == 0.5

    @pytest.mark.parametrize("bad", [
        {"cutmix_prob": 1.5},
        {"color_jitter_strength": -0.1},
        {"cutmix_ratio_range": (0.5, 0.1)},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            StrongAugmentation(**bad)

    def test_inference_is_identity(self, sample):
        out = StrongAugmentation()(sample, training=False)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(out), sample, atol=1e-6)

    def test_training_preserves_shape(self, sample):
        out = StrongAugmentation(cutmix_prob=1.0)(sample, training=True)
        assert tuple(out.shape) == (B, H, W, C)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        assert StrongAugmentation().compute_output_shape((B, H, W, C)) == (B, H, W, C)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = StrongAugmentation(cutmix_prob=0.5, name="aug")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "aug.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"StrongAugmentation": StrongAugmentation}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1), atol=1e-6
        )

    def test_get_config(self):
        layer = StrongAugmentation(cutmix_prob=0.3, color_jitter_strength=0.4)
        rebuilt = StrongAugmentation.from_config(layer.get_config())
        assert rebuilt.cutmix_prob == 0.3 and rebuilt.color_jitter_strength == 0.4

    def test_get_config_round_trips_input_value_range(self):
        layer = StrongAugmentation(input_value_range=None)
        assert StrongAugmentation.from_config(layer.get_config()).input_value_range is None
        layer = StrongAugmentation(input_value_range=(-1.0, 1.0))
        assert StrongAugmentation.from_config(
            layer.get_config()
        ).input_value_range == (-1.0, 1.0)

    @pytest.mark.parametrize("bad", [(1.0, 0.0), (0.0, 0.0), (0.0,)])
    def test_invalid_input_value_range_raises(self, bad):
        with pytest.raises(ValueError, match="input_value_range"):
            StrongAugmentation(input_value_range=bad)


# ---------------------------------------------------------------------
# C-2 / C-19 — target-aware CutMix and the declared input range.
#
# The oracle for the CutMix box is read off the *image*, never off the mask the
# implementation returned: the two synthetic scenes are constant-valued (row 0
# all `_SCENE_A`, row 1 all `_SCENE_B`) and colour jitter is switched off
# (strength 0 makes brightness/contrast factors exactly 1.0), so after the mix
# the pasted rectangle in `x_aug[i]` is exactly the set of pixels carrying the
# donor's constant. The depth targets carry a different pair of constants, so
# "the target was mixed by the same box" is a pixel-set equality between two
# independently derived masks.
# ---------------------------------------------------------------------

_SCENE_A, _SCENE_B = 0.0, 1.0
_DEPTH_A, _DEPTH_B = 3.0, 11.0


@pytest.fixture
def pinned_reverse_permutation(monkeypatch):
    """Force CutMix's donor permutation to the reversal of the batch.

    Without this the layer draws `keras.random.shuffle`, which for a 2-row batch
    returns the identity roughly half the time — and an identity permutation
    makes every donor its own row, so the probe would silently measure nothing.
    """
    monkeypatch.setattr(
        keras.random, "shuffle", lambda x, **kwargs: keras.ops.flip(x, axis=0)
    )


def _two_scene_batch():
    x = np.stack([
        np.full((H, W, C), _SCENE_A, dtype="float32"),
        np.full((H, W, C), _SCENE_B, dtype="float32"),
    ])
    y = np.stack([
        np.full((H, W, 1), _DEPTH_A, dtype="float32"),
        np.full((H, W, 1), _DEPTH_B, dtype="float32"),
    ])
    return x, y


class TestTargetAwareCutMix:
    """C-2(a): the CutMix box must be applied to the depth target as well."""

    def test_cut_region_of_target_carries_the_donor_depth(
        self, pinned_reverse_permutation
    ):
        """RED against a CutMix that returns only the mixed image.

        Row 0 receives a rectangle of row 1's scene. Wherever the image shows
        the donor's constant, the target must show the *donor's* depth.
        """
        layer = StrongAugmentation(
            cutmix_prob=1.0, cutmix_ratio_range=(0.5, 0.5), color_jitter_strength=0.0
        )
        x, y = _two_scene_batch()
        x_aug, mix = layer.augment_with_mix(x, training=True)
        y_mix = np.asarray(layer.apply_mix_to_target(y, mix))
        x_aug = np.asarray(x_aug)

        cut = np.isclose(x_aug[0, :, :, 0], _SCENE_B)
        assert cut.sum() > 0, "liveness: no pixel of row 0 was replaced at all"
        assert np.all(np.isclose(y_mix[0, :, :, 0][cut], _DEPTH_B)), (
            "target's cut region does not carry the donor's depth: got "
            f"{sorted(set(np.round(y_mix[0, :, :, 0][cut], 4).tolist()))}, "
            f"expected [{_DEPTH_B}]"
        )

    def test_region_outside_the_cut_keeps_the_original_depth(
        self, pinned_reverse_permutation
    ):
        """Anti-vacuity twin of the assertion above.

        A target replaced *wholesale* by the donor's would satisfy the cut-region
        assertion. This one fails for that mutation and passes for the correct
        box mix, so the pair pins the box, not merely the donor.
        """
        layer = StrongAugmentation(
            cutmix_prob=1.0, cutmix_ratio_range=(0.5, 0.5), color_jitter_strength=0.0
        )
        x, y = _two_scene_batch()
        x_aug, mix = layer.augment_with_mix(x, training=True)
        y_mix = np.asarray(layer.apply_mix_to_target(y, mix))
        x_aug = np.asarray(x_aug)

        kept = np.isclose(x_aug[0, :, :, 0], _SCENE_A)
        assert kept.sum() > 0, "liveness: the whole row was replaced, no kept region"
        assert np.all(np.isclose(y_mix[0, :, :, 0][kept], _DEPTH_A)), (
            "target outside the cut region no longer carries the original depth: got "
            f"{sorted(set(np.round(y_mix[0, :, :, 0][kept], 4).tolist()))}, "
            f"expected [{_DEPTH_A}]"
        )

    def test_target_mix_accepts_a_different_channel_count(
        self, pinned_reverse_permutation
    ):
        """The trainer's target is depth+validity `(B, H, W, 2)`, not `(B, H, W, 1)`."""
        layer = StrongAugmentation(
            cutmix_prob=1.0, cutmix_ratio_range=(0.5, 0.5), color_jitter_strength=0.0
        )
        x, y = _two_scene_batch()
        y2 = np.concatenate([y, np.ones_like(y)], axis=-1)
        _, mix = layer.augment_with_mix(x, training=True)
        assert tuple(layer.apply_mix_to_target(y2, mix).shape) == (2, H, W, 2)

    def test_no_mix_descriptor_outside_training(self):
        x, _ = _two_scene_batch()
        out, mix = StrongAugmentation().augment_with_mix(x, training=False)
        assert mix is None
        np.testing.assert_allclose(keras.ops.convert_to_numpy(out), x, atol=1e-6)


class TestDeclaredInputValueRange:
    """C-2(d): the colour-jitter clip must follow the declared input range."""

    def test_negative_pixel_survives_when_range_is_unbounded(self):
        """RED against the unconditional `ops.clip(x, 0.0, 1.0)`.

        `color_jitter_strength=0.0` makes brightness and contrast exactly 1.0, so
        the clip is the only operation that can change the value — the probe
        isolates it with nothing else moving.
        """
        layer = StrongAugmentation(
            cutmix_prob=0.0, color_jitter_strength=0.0, input_value_range=None
        )
        x = np.full((2, H, W, C), -0.7, dtype="float32")
        out = np.asarray(layer(x, training=True))
        assert float(out.min()) < -0.6, (
            "a standardized/[-1,1] input was clamped on the training path: "
            f"min={float(out.min())}, expected about -0.7"
        )

    def test_declared_unit_range_still_clips(self):
        """Anti-vacuity control: the knob must be live in both directions."""
        layer = StrongAugmentation(
            cutmix_prob=0.0, color_jitter_strength=0.0, input_value_range=(0.0, 1.0)
        )
        x = np.full((2, H, W, C), -0.7, dtype="float32")
        out = np.asarray(layer(x, training=True))
        assert float(out.min()) == pytest.approx(0.0, abs=1e-6)
