"""Tests for the DINO multi-crop tf.data transform.

Written for plan-2026-08-01T105809-dc0c402e step 9. Four things here are load
bearing and must not be "simplified":

1. **The vacuity boundary.** A map fn that returns `n` identical copies of the
   input passes every shape, dtype, static-spec and batching assertion in this
   file. `TestCropsAreGenuinelyDifferent` is the only thing standing between
   that map fn and a green suite, which is why it asserts pairwise separation
   over EVERY view pair rather than "the output is not the input".
2. **The batching test builds a REAL `tf.data.Dataset` and ITERATES it.** A
   shape assertion on a single element does not prove `tf.data` can batch it;
   a ragged or dynamically-shaped element only fails at the first batch.
3. **The global-vs-local area test measures a statistical property over many
   samples**, because one crop pair proves nothing about two overlapping scale
   RANGES. Its statistic is total variation after the resize: a small AREA
   upsampled to the same pixel size is measurably SMOOTHER. It runs with every
   photometric augmentation disabled, so the only thing it can be measuring is
   the crop.
4. **The `local_crop_size` raise is asserted on the MESSAGE**, not on the
   exception type (`plans/LESSONS.md`: a predicted exception type is wrong more
   often than the failure class).

Every guard was RED-proven by injecting the corresponding dead component; the
injected component and the assertion that fired are recorded in the plan's
`verification.md`.
"""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.multi_crop import (
    N_GLOBAL_VIEWS,
    make_multi_crop_map_fn,
)

# The source is only modestly larger than the crop ON PURPOSE. TF's bilinear
# resize does NOT antialias, so a DOWNsample of noise subsamples it and leaves
# the total variation at the source's level whatever the factor -- which makes
# the area statistic in `TestGlobalsCoverMoreAreaThanLocals` blind. At
# `SOURCE_SIZE == CROP_SIZE` every crop is UPsampled, the smoothing is
# monotone in the zoom factor, and the measured global/local TV ratio is 0.57
# (vs 0.85 at CROP_SIZE=32, which did not clear the threshold).
SOURCE_SIZE = 64
CROP_SIZE = 64
CHANNELS = 3
N_LOCAL = 4
N_VIEWS = N_GLOBAL_VIEWS + N_LOCAL


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _source_images(count=8, seed=0):
    """High-frequency noise images.

    Noise is deliberate: the area test's statistic is high-frequency energy
    after a resize, and a smooth source would leave both crop scales with the
    same (near-zero) statistic, i.e. a vacuous test.
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(
        -1.0, 1.0, size=(count, SOURCE_SIZE, SOURCE_SIZE, CHANNELS)
    ).astype("float32")


def _crop_only_map_fn(seed, **overrides):
    """A map fn with every PHOTOMETRIC augmentation disabled."""
    kwargs = dict(
        n_local_crops=N_LOCAL,
        flip_prob=0.0,
        color_jitter_prob=0.0,
        grayscale_prob=0.0,
        global_blur_probs=(0.0, 0.0),
        local_blur_prob=0.0,
        seed=seed,
    )
    kwargs.update(overrides)
    return make_multi_crop_map_fn(CROP_SIZE, **kwargs)


def _run(map_fn, images):
    """Map `map_fn` over `images` through a real tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices(
        (images, np.arange(len(images), dtype="int32"))
    ).map(map_fn)
    return np.stack([v.numpy() for v, _ in ds])


def _total_variation(views):
    """Mean |horizontal + vertical neighbour difference| per view.

    A crop covering a SMALL area is upsampled to `CROP_SIZE` and is therefore
    interpolated (smooth, low TV); a crop covering a LARGE area is downsampled
    and keeps the source's high-frequency content (high TV). So this is a
    monotone proxy for "how much source area did this view cover".
    """
    horizontal = np.abs(views[..., 1:, :] - views[..., :-1, :]).mean(
        axis=(-3, -2, -1))
    vertical = np.abs(views[..., 1:, :, :] - views[..., :-1, :, :]).mean(
        axis=(-3, -2, -1))
    return 0.5 * (horizontal + vertical)


# ---------------------------------------------------------------------


class TestElementContract:
    """The element must be exactly what `DINOTrainingModel.call` documents."""

    def test_n_global_views_agrees_with_the_training_model(self):
        """The two constants are declared in two modules; pin them together."""
        from dl_techniques.models.dino.dino_training import (
            N_GLOBAL_VIEWS as MODEL_N_GLOBAL_VIEWS,
        )

        assert N_GLOBAL_VIEWS == MODEL_N_GLOBAL_VIEWS, (
            f"multi_crop.N_GLOBAL_VIEWS ({N_GLOBAL_VIEWS}) has drifted from "
            f"dino_training.N_GLOBAL_VIEWS ({MODEL_N_GLOBAL_VIEWS}). The "
            f"dataset element and the model's input contract must agree on "
            f"how many leading views are global."
        )

    def test_element_is_a_fixed_shape_stack_of_2_plus_n_local_views(self):
        map_fn = _crop_only_map_fn(seed=1)
        views, label = map_fn(_source_images(1)[0], tf.constant(7, tf.int32))

        assert views.shape == (
            N_GLOBAL_VIEWS + N_LOCAL, CROP_SIZE, CROP_SIZE, CHANNELS)
        assert views.dtype == tf.float32
        assert int(label.numpy()) == 7

    def test_every_view_is_at_the_global_resolution(self):
        """D-002: locals are the same PIXEL size as globals, smaller AREA."""
        views = _run(_crop_only_map_fn(seed=2), _source_images(2))
        assert views.shape[2:4] == (CROP_SIZE, CROP_SIZE)

    def test_accepts_a_bare_image_without_a_label(self):
        map_fn = _crop_only_map_fn(seed=3)
        views, label = map_fn(_source_images(1)[0])
        assert views.shape[0] == N_VIEWS
        assert int(label.numpy()) == 0

    @pytest.mark.parametrize("n_local", [0, 1, 6])
    def test_view_count_tracks_n_local_crops(self, n_local):
        map_fn = _crop_only_map_fn(seed=4, n_local_crops=n_local)
        views, _ = map_fn(_source_images(1)[0])
        assert views.shape[0] == N_GLOBAL_VIEWS + n_local


class TestBatchable:
    """`tf.data` must be able to BATCH the element, not just produce it."""

    def test_a_real_dataset_batches_and_iterates(self):
        images = _source_images(8)
        ds = (
            tf.data.Dataset.from_tensor_slices(
                (images, np.arange(8, dtype="int32")))
            .map(_crop_only_map_fn(seed=5))
            .batch(4)
        )

        spec = ds.element_spec[0]
        assert spec.shape.as_list() == [
            None, N_VIEWS, CROP_SIZE, CROP_SIZE, CHANNELS], (
            f"the batched spec is not fully static: {spec.shape}. A dynamic "
            f"per-view dimension would only fail once a model is attached."
        )

        batches = [v.numpy() for v, _ in ds]
        assert len(batches) == 2
        for batch in batches:
            assert batch.shape == (
                4, N_VIEWS, CROP_SIZE, CROP_SIZE, CHANNELS)
            assert np.isfinite(batch).all()

    def test_parallel_mapping_also_batches(self):
        """The real pipeline maps with AUTOTUNE parallelism."""
        images = _source_images(8)
        ds = (
            tf.data.Dataset.from_tensor_slices(images)
            .map(_crop_only_map_fn(seed=6),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(4)
        )
        assert sum(int(v.shape[0]) for v, _ in ds) == 8


class TestCropsAreGenuinelyDifferent:
    """THE vacuity boundary: `n` copies of the input pass everything else."""

    def test_every_pair_of_views_differs(self):
        views = _run(_crop_only_map_fn(seed=7), _source_images(4))

        identical = []
        for sample in range(views.shape[0]):
            for i in range(N_VIEWS):
                for j in range(i + 1, N_VIEWS):
                    delta = float(
                        np.abs(views[sample, i] - views[sample, j]).max())
                    if delta < 1e-4:
                        identical.append((sample, i, j, delta))
        assert not identical, (
            f"{len(identical)} view pair(s) are identical tensors, e.g. "
            f"{identical[0]}. A map fn returning n copies of the same crop "
            f"passes every shape, dtype and batching test in this file."
        )

    def test_the_two_global_views_differ_from_each_other(self):
        """Views 0 and 1 are the pair the teacher sees; they must not match."""
        views = _run(_crop_only_map_fn(seed=8), _source_images(6))
        deltas = np.abs(views[:, 0] - views[:, 1]).max(axis=(1, 2, 3))
        assert deltas.min() > 1e-3, (
            f"the two GLOBAL views coincide on at least one sample "
            f"(min max-abs delta {deltas.min():.3e}); the teacher would then "
            f"see the same crop twice."
        )

    def test_views_are_not_the_plain_resized_source(self):
        images = _source_images(4)
        views = _run(_crop_only_map_fn(seed=9), images)
        resized = tf.image.resize(
            images, (CROP_SIZE, CROP_SIZE), method="bilinear").numpy()

        for sample in range(views.shape[0]):
            for view in range(N_VIEWS):
                delta = float(
                    np.abs(views[sample, view] - resized[sample]).max())
                assert delta > 1e-4, (
                    f"sample {sample} view {view} is the untouched resized "
                    f"source (max-abs delta {delta:.3e}) -- the crop is an "
                    f"identity no-op."
                )

    def test_different_samples_get_different_crops(self):
        """One shared crop box reused across the batch is also a defect."""
        images = np.repeat(_source_images(1, seed=11), 4, axis=0)
        views = _run(_crop_only_map_fn(seed=10), images)
        deltas = [
            float(np.abs(views[0, 0] - views[k, 0]).max()) for k in (1, 2, 3)
        ]
        assert min(deltas) > 1e-3, (
            f"the SAME source image produced the same view-0 crop across "
            f"samples (deltas {deltas}); the randomness is not per-sample."
        )


class TestGlobalsCoverMoreAreaThanLocals:
    """The scale ranges are the whole point of multi-crop."""

    def test_globals_keep_more_high_frequency_content_than_locals(self):
        views = _run(_crop_only_map_fn(seed=1234), _source_images(48, seed=3))
        tv = _total_variation(views)  # (samples, n_views)

        global_tv = float(tv[:, :N_GLOBAL_VIEWS].mean())
        local_tv = float(tv[:, N_GLOBAL_VIEWS:].mean())

        # Non-vacuity: the statistic must be able to see anything at all.
        assert global_tv > 0.05 and local_tv > 0.0, (
            f"degenerate statistic (global {global_tv:.4f}, local "
            f"{local_tv:.4f}); this test cannot distinguish anything."
        )
        assert local_tv < 0.8 * global_tv, (
            f"local views are not covering a smaller AREA than global views: "
            f"mean total variation is {local_tv:.4f} (local) vs "
            f"{global_tv:.4f} (global), ratio {local_tv / global_tv:.3f}. A "
            f"small crop resized UP to the same pixel size is interpolated and "
            f"must be measurably smoother; equal scale ranges make this "
            f"ratio ~1.0 and multi-crop decorative."
        )

    def test_the_effect_scales_with_the_local_range(self):
        """A TIGHTER local range must smooth the local views FURTHER."""
        images = _source_images(32, seed=4)
        wide = _run(
            _crop_only_map_fn(seed=99, local_scale=(0.30, 0.40)), images)
        tight = _run(
            _crop_only_map_fn(seed=99, local_scale=(0.02, 0.05)), images)

        wide_tv = float(_total_variation(wide)[:, N_GLOBAL_VIEWS:].mean())
        tight_tv = float(_total_variation(tight)[:, N_GLOBAL_VIEWS:].mean())
        assert tight_tv < wide_tv, (
            f"local_scale is not reaching the crop: a (0.02, 0.05) range gave "
            f"total variation {tight_tv:.4f}, no smoother than a (0.30, 0.40) "
            f"range's {wide_tv:.4f}."
        )


class TestLocalCropSizeIsRefused:
    """D-002's named limitation must be loud, not silent."""

    def test_a_different_local_crop_size_raises_naming_interpolation(self):
        # Asserted on the MESSAGE, not the type: the actionable content is the
        # word "interpolation" plus the reason.
        with pytest.raises(
            Exception,
            match=r"positional-embedding interpolation",
        ):
            make_multi_crop_map_fn(CROP_SIZE, local_crop_size=CROP_SIZE // 2)

    def test_the_message_names_the_backlog_document(self):
        with pytest.raises(
            Exception,
            match=r"src/dl_techniques/models/dino/README\.md",
        ):
            make_multi_crop_map_fn(CROP_SIZE, local_crop_size=CROP_SIZE * 2)

    @pytest.mark.parametrize("local_crop_size", [None, CROP_SIZE])
    def test_an_equal_or_omitted_local_crop_size_is_accepted(
            self, local_crop_size):
        map_fn = make_multi_crop_map_fn(
            CROP_SIZE, local_crop_size=local_crop_size,
            n_local_crops=N_LOCAL, seed=12,
        )
        views, _ = map_fn(_source_images(1)[0])
        assert views.shape == (N_VIEWS, CROP_SIZE, CROP_SIZE, CHANNELS)


class TestRandomness:
    """A map fn that ignores its seed is a real failure mode."""

    def test_a_fixed_seed_is_reproducible(self):
        images = _source_images(4, seed=5)
        first = _run(_crop_only_map_fn(seed=4242), images)
        second = _run(_crop_only_map_fn(seed=4242), images)
        np.testing.assert_array_equal(first, second)

    def test_a_fixed_seed_is_reproducible_with_every_augmentation_on(self):
        images = _source_images(4, seed=5)
        first = _run(
            make_multi_crop_map_fn(
                CROP_SIZE, n_local_crops=N_LOCAL, seed=777), images)
        second = _run(
            make_multi_crop_map_fn(
                CROP_SIZE, n_local_crops=N_LOCAL, seed=777), images)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_differ(self):
        images = _source_images(4, seed=5)
        first = _run(_crop_only_map_fn(seed=1), images)
        second = _run(_crop_only_map_fn(seed=2), images)
        assert float(np.abs(first - second).max()) > 1e-3

    def test_no_seed_is_non_deterministic(self):
        images = _source_images(4, seed=5)
        first = _run(_crop_only_map_fn(seed=None), images)
        second = _run(_crop_only_map_fn(seed=None), images)
        assert float(np.abs(first - second).max()) > 1e-3, (
            "two unseeded pipelines produced bit-identical crops; the map fn "
            "is not actually random."
        )


class TestPhotometricAugmentationsAreLive:
    """Each optional augmentation must change the pixels when enabled."""

    def test_blur_smooths_the_view(self):
        """Compared on view 0, whose crop is drawn BEFORE any blur draw."""
        images = _source_images(16, seed=6)
        sharp = _run(_crop_only_map_fn(seed=31), images)
        blurred = _run(
            _crop_only_map_fn(
                seed=31, global_blur_probs=(1.0, 0.0),
                blur_sigma_range=(1.5, 2.0)),
            images,
        )
        sharp_tv = float(_total_variation(sharp)[:, 0].mean())
        blurred_tv = float(_total_variation(blurred)[:, 0].mean())
        assert blurred_tv < 0.5 * sharp_tv, (
            f"Gaussian blur did not smooth view 0: total variation "
            f"{blurred_tv:.4f} vs {sharp_tv:.4f} unblurred."
        )

    def test_grayscale_collapses_the_channels(self):
        images = _source_images(4, seed=7)
        views = _run(_crop_only_map_fn(seed=32, grayscale_prob=1.0), images)
        spread = np.abs(views - views.mean(axis=-1, keepdims=True)).max()
        assert float(spread) < 1e-5, (
            f"grayscale_prob=1.0 left a per-channel spread of {spread:.3e}"
        )

    def test_horizontal_flip_is_applied(self):
        images = _source_images(4, seed=8)
        unflipped = _run(_crop_only_map_fn(seed=33), images)
        flipped = _run(_crop_only_map_fn(seed=33, flip_prob=1.0), images)
        np.testing.assert_allclose(
            flipped[:, 0], unflipped[:, 0, :, ::-1, :], atol=1e-6)

    def test_colour_jitter_changes_the_values(self):
        images = _source_images(4, seed=9)
        plain = _run(_crop_only_map_fn(seed=34), images)
        jittered = _run(
            _crop_only_map_fn(
                seed=34, color_jitter_prob=1.0, brightness=0.5, contrast=0.5),
            images,
        )
        assert float(np.abs(plain[:, 0] - jittered[:, 0]).max()) > 1e-3


class TestConstructionErrors:
    """Configuration is validated EAGERLY, when the pipeline is built."""

    def test_non_positive_crop_size_raises(self):
        with pytest.raises(ValueError, match="global_crop_size must be"):
            make_multi_crop_map_fn(0)

    def test_negative_n_local_crops_raises(self):
        with pytest.raises(ValueError, match="n_local_crops must be"):
            make_multi_crop_map_fn(CROP_SIZE, n_local_crops=-1)

    @pytest.mark.parametrize(
        "scale", [(0.0, 1.0), (0.5, 0.2), (0.5, 1.5)])
    def test_bad_scale_range_raises(self, scale):
        with pytest.raises(ValueError, match="global_scale must be"):
            make_multi_crop_map_fn(CROP_SIZE, global_scale=scale)

    def test_locals_reaching_a_larger_area_than_globals_raises(self):
        with pytest.raises(ValueError, match="multi-crop requires the local"):
            make_multi_crop_map_fn(
                CROP_SIZE, global_scale=(0.4, 0.6), local_scale=(0.05, 0.9))

    def test_out_of_range_probability_raises(self):
        with pytest.raises(ValueError, match=r"flip_prob must be in \[0, 1\]"):
            make_multi_crop_map_fn(CROP_SIZE, flip_prob=1.5)

    def test_bad_blur_sigma_range_raises(self):
        with pytest.raises(ValueError, match="blur_sigma_range must be"):
            make_multi_crop_map_fn(CROP_SIZE, blur_sigma_range=(2.0, 0.5))

    def test_bad_aspect_ratio_range_raises(self):
        with pytest.raises(ValueError, match="aspect_ratio_range must be"):
            make_multi_crop_map_fn(CROP_SIZE, aspect_ratio_range=(1.5, 0.5))


class TestEndToEndWithTheTrainingModel:
    """The seam most likely to be off by one axis, checked here not at step 12."""

    def test_a_batched_element_forward_passes_through_DINOTrainingModel(self):
        from dl_techniques.models.dino.dino_training import (
            create_dino_training_model,
        )

        # NOTE: the backbone is the stock "tiny" variant, not a shrunken one.
        # `create_dino_teacher_student_pair` forwards **kwargs into
        # `from_variant`, so passing `embed_dim=32` raises `TypeError: got
        # multiple values for keyword argument 'embed_dim'` -- a pre-existing
        # residual recorded in this plan's progress.md, not something this test
        # should work around silently.
        out_dim = 16
        model = create_dino_training_model(
            "tiny",
            image_size=CROP_SIZE,
            patch_size=16,
            n_local_views=N_LOCAL,
            dino_out_dim=out_dim,
        )

        ds = (
            tf.data.Dataset.from_tensor_slices(_source_images(4, seed=12))
            .map(make_multi_crop_map_fn(
                CROP_SIZE, n_local_crops=N_LOCAL, seed=13))
            .batch(2)
        )
        views, _ = next(iter(ds))

        output = model(views)
        assert tuple(output.shape) == (
            2 * model.n_pairs, 2 * out_dim), (
            f"the multi-crop element does not line up with the model's packed "
            f"output contract: got {tuple(output.shape)}, expected "
            f"{(2 * model.n_pairs, 2 * out_dim)}."
        )
        assert np.isfinite(output.numpy()).all()

# ---------------------------------------------------------------------
