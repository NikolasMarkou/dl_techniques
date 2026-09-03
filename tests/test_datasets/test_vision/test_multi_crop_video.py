"""Tests for ``make_multi_crop_video_map_fn`` (shape correctness, same-crop-
across-frames property, construction-time validation)."""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.multi_crop_video import make_multi_crop_video_map_fn


def _make_clip(num_frames=4, size=32, channels=3, seed=0, repeat_frame=False):
    rng = np.random.default_rng(seed)
    if repeat_frame:
        frame = rng.random((size, size, channels), dtype=np.float32)
        clip = np.stack([frame] * num_frames, axis=0)
    else:
        clip = rng.random((num_frames, size, size, channels), dtype=np.float32)
    return tf.constant(clip, dtype=tf.float32)


class TestMultiCropVideoShape:
    def test_output_shapes(self):
        fn = make_multi_crop_video_map_fn(
            crop_size=16, num_frames=4, local_crops_number=3,
        )
        clip = _make_clip(num_frames=4, size=32)
        outputs, label = fn({"pixels": clip}, tf.constant(0.0))

        assert outputs["global_frame"].shape == (4, 16, 16, 3)
        assert outputs["local_frames"].shape == (3, 4, 16, 16, 3)
        assert label.numpy() == 0.0

    def test_zero_local_crops_shape(self):
        fn = make_multi_crop_video_map_fn(
            crop_size=16, num_frames=4, local_crops_number=0,
        )
        clip = _make_clip(num_frames=4, size=32)
        outputs, _ = fn({"pixels": clip}, tf.constant(0.0))
        assert outputs["local_frames"].shape == (0, 4, 16, 16, 3)

    def test_outputs_are_finite(self):
        fn = make_multi_crop_video_map_fn(
            crop_size=16, num_frames=4, local_crops_number=2,
        )
        clip = _make_clip(num_frames=4, size=32)
        outputs, _ = fn({"pixels": clip}, tf.constant(0.0))
        assert np.all(np.isfinite(outputs["global_frame"].numpy()))
        assert np.all(np.isfinite(outputs["local_frames"].numpy()))


class TestMultiCropVideoSameCropAcrossFrames:
    """The crop box (and every other per-view augmentation decision) must be
    IDENTICAL across all T frames of one view -- proven by feeding a clip
    whose T frames are pixel-identical and checking the augmented output's T
    frames are then ALSO pixel-identical. A differing crop box (or a
    differing flip/jitter/blur decision) on identical source content would
    produce differing output frames.
    """

    def test_global_view_identical_across_frames(self):
        fn = make_multi_crop_video_map_fn(
            crop_size=16, num_frames=5, local_crops_number=1,
        )
        clip = _make_clip(num_frames=5, size=32, repeat_frame=True)
        outputs, _ = fn({"pixels": clip}, tf.constant(0.0))
        global_frame = outputs["global_frame"].numpy()  # (T, S, S, C)

        for t in range(1, global_frame.shape[0]):
            np.testing.assert_allclose(
                global_frame[0], global_frame[t], atol=1e-5, rtol=0
            )

    def test_local_views_identical_across_frames(self):
        fn = make_multi_crop_video_map_fn(
            crop_size=16, num_frames=5, local_crops_number=2,
        )
        clip = _make_clip(num_frames=5, size=32, repeat_frame=True)
        outputs, _ = fn({"pixels": clip}, tf.constant(0.0))
        local_frames = outputs["local_frames"].numpy()  # (V, T, S, S, C)

        for v in range(local_frames.shape[0]):
            for t in range(1, local_frames.shape[1]):
                np.testing.assert_allclose(
                    local_frames[v, 0], local_frames[v, t], atol=1e-5, rtol=0
                )

    def test_different_views_can_differ_from_each_other(self):
        """A sanity anti-vacuity check: the SAME-across-frames property above
        must not be trivially true because the transform ignores randomness
        entirely -- different views (global vs. local, or two local views)
        should generally differ from one another on non-constant content."""
        fn = make_multi_crop_video_map_fn(
            crop_size=16, num_frames=3, local_crops_number=1,
            local_scale=(0.05, 0.2), global_scale=(0.6, 1.0),
        )
        clip = _make_clip(num_frames=3, size=64, repeat_frame=False, seed=3)
        outputs, _ = fn({"pixels": clip}, tf.constant(0.0))
        global_frame = outputs["global_frame"].numpy()
        local_frame = outputs["local_frames"].numpy()[0]

        assert not np.allclose(global_frame, local_frame, atol=1e-4)


class TestMultiCropVideoValidation:
    def test_non_positive_crop_size_raises(self):
        with pytest.raises(ValueError, match="crop_size"):
            make_multi_crop_video_map_fn(crop_size=0, num_frames=4)

    def test_negative_local_crops_number_raises(self):
        with pytest.raises(ValueError, match="local_crops_number"):
            make_multi_crop_video_map_fn(
                crop_size=16, num_frames=4, local_crops_number=-1
            )

    def test_local_scale_exceeding_global_scale_raises(self):
        with pytest.raises(ValueError, match="local_scale"):
            make_multi_crop_video_map_fn(
                crop_size=16, num_frames=4,
                global_scale=(0.1, 0.3), local_scale=(0.05, 0.5),
            )
