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
    make_stateless_multi_crop_map_fn,
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


def _run(map_fn, images, num_parallel_calls=None):
    """Map `map_fn` over `images` through a real tf.data pipeline.

    `num_parallel_calls=None` is a SERIAL `.map()`. That is the ONLY
    configuration in which the `seed` argument reproduces anything -- see
    `TestRandomness`. Pass `tf.data.AUTOTUNE` to exercise the configuration the
    shipped trainer actually uses (`build_raw_image_dataset` passes AUTOTUNE).
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (images, np.arange(len(images), dtype="int32"))
    ).map(map_fn, num_parallel_calls=num_parallel_calls)
    return np.stack([v.numpy() for v, _ in ds])


def _stateless_crop_only_map_fn(seed, **overrides):
    """`_crop_only_map_fn`'s stateless twin: same config, different RNG."""
    kwargs = dict(
        n_local_crops=N_LOCAL,
        flip_prob=0.0,
        color_jitter_prob=0.0,
        grayscale_prob=0.0,
        global_blur_probs=(0.0, 0.0),
        local_blur_prob=0.0,
    )
    kwargs.update(overrides)
    return make_stateless_multi_crop_map_fn(CROP_SIZE, seed=seed, **kwargs)


def _run_indexed(map_fn, images, indices=None, num_parallel_calls=None):
    """Map an INDEXED map fn over `images`, keyed by `indices`.

    The index is supplied explicitly rather than by `.enumerate()` so that the
    cross-epoch test can hand the SAME image two DIFFERENT indices, which is
    what a second epoch does (`build_raw_image_dataset` enumerates AFTER
    `repeat()`).
    """
    if indices is None:
        indices = np.arange(len(images), dtype="int64")
    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(indices, dtype="int64"), images)
    ).map(map_fn, num_parallel_calls=num_parallel_calls)
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

    @pytest.mark.parametrize("factory", [
        lambda **kw: make_multi_crop_map_fn(CROP_SIZE, **kw),
        lambda **kw: make_stateless_multi_crop_map_fn(CROP_SIZE, seed=1, **kw),
    ], ids=["stateful", "stateless"])
    def test_a_different_local_crop_size_raises_naming_interpolation(
            self, factory):
        # Asserted on the MESSAGE, not the type: the actionable content is the
        # word "interpolation" plus the reason.
        #
        # BOTH factories are exercised (plan I-7): the stateless path shares
        # this validation only because it delegates to the same private
        # implementation, and "shares it by construction" is exactly the kind of
        # claim that stops being true after one refactor.
        with pytest.raises(
            Exception,
            match=r"positional-embedding interpolation",
        ):
            factory(local_crop_size=CROP_SIZE // 2)

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
    """A map fn that ignores its seed is a real failure mode.

    **Read the scope, not just the names.** `seed` seeds ONE shared
    `tf.random.Generator`, i.e. a STREAM, not each element. Every guarantee
    below therefore holds for a SERIAL `.map()` only. The shipped trainer maps
    with `num_parallel_calls=tf.data.AUTOTUNE`, where several elements read that
    one generator concurrently and the element-to-draw assignment varies run to
    run -- MEASURED at HEAD, seed 777, 8 images: serial identical=True, parallel
    identical=False maxdiff 1.5312, serial-vs-parallel maxdiff 1.5908. Until
    this iteration those three facts were invisible here, because every
    determinism test used the serial map and none of them said so.
    """

    def test_a_fixed_seed_is_reproducible_under_a_serial_map(self):
        images = _source_images(4, seed=5)
        first = _run(_crop_only_map_fn(seed=4242), images)
        second = _run(_crop_only_map_fn(seed=4242), images)
        np.testing.assert_array_equal(first, second)

    def test_a_fixed_seed_is_reproducible_with_every_augmentation_on(self):
        """Same scope as above: SERIAL map."""
        images = _source_images(4, seed=5)
        first = _run(
            make_multi_crop_map_fn(
                CROP_SIZE, n_local_crops=N_LOCAL, seed=777), images)
        second = _run(
            make_multi_crop_map_fn(
                CROP_SIZE, n_local_crops=N_LOCAL, seed=777), images)
        np.testing.assert_array_equal(first, second)

    def test_the_seed_seeds_a_STREAM_not_each_element(self):
        """The MECHANISM that scopes the guarantee, asserted deterministically.

        Two successive calls of the SAME map fn on the SAME image must differ:
        the generator advances per call, so the draw an element receives depends
        on how many draws happened before it. That is exactly why a parallel map
        -- which reorders those draws across elements -- is not reproducible.

        This is asserted here rather than by comparing two AUTOTUNE runs,
        because "two parallel runs came out different" depends on the thread
        scheduler and could pass or fail for reasons unrelated to the code. The
        stream property does not: it is a property of `tf.random.Generator`.
        """
        map_fn = _crop_only_map_fn(seed=4242)
        image = _source_images(1, seed=5)[0]
        first, _ = map_fn(image)
        second, _ = map_fn(image)
        assert float(np.abs(first.numpy() - second.numpy()).max()) > 1e-3, (
            "two successive calls of one seeded map fn on the SAME image "
            "returned identical views -- the seed would then be per-element "
            "and the parallel-map caveat below would be unnecessary"
        )
        # ... and a FRESH map fn at the same seed reproduces the FIRST call,
        # which is what makes the serial guarantee above real.
        again, _ = _crop_only_map_fn(seed=4242)(image)
        np.testing.assert_array_equal(first.numpy(), again.numpy())

    def test_the_pipeline_is_well_formed_at_the_trainer_s_real_parallelism(self):
        """Exercise `num_parallel_calls=AUTOTUNE` -- the SHIPPED configuration.

        A guard that only holds in a configuration nothing runs is worse than
        none, so the shipped setting is executed here. What it can assert is the
        element contract (shape, dtype, finiteness), NOT reproducibility: see
        the class docstring for the measured non-determinism.
        """
        images = _source_images(8, seed=5)
        views = _run(
            make_multi_crop_map_fn(
                CROP_SIZE, n_local_crops=N_LOCAL, seed=777),
            images,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        assert views.shape == (8, N_VIEWS, CROP_SIZE, CROP_SIZE, CHANNELS)
        assert views.dtype == np.float32
        assert np.isfinite(views).all()

    def test_the_seed_docstring_scopes_its_own_guarantee(self):
        """The caveat is load-bearing prose, so it is pinned like code.

        `train_dino.py` presents `--seed` beside `set_seeds(config.seed)` as run
        reproducibility. It is not, for this transform. If the qualification
        ever gets "cleaned up" out of the docstring, this fails.
        """
        doc = make_multi_crop_map_fn.__doc__ or ""
        module_doc = (
            __import__(
                "dl_techniques.datasets.vision.multi_crop",
                fromlist=["multi_crop"],
            ).__doc__ or ""
        )
        assert "SERIAL" in doc, (
            "make_multi_crop_map_fn's docstring no longer scopes `seed` to a "
            "serial map, but the guarantee is still serial-only"
        )
        assert "num_parallel_calls" in module_doc
        assert "AUTOTUNE" in module_doc

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


class TestStatelessRandomness:
    """`make_stateless_multi_crop_map_fn` — the guarantee `seed` never gave.

    Three properties, and all three are needed. Reproducibility ALONE is
    satisfied by a map fn that returns the same augmentation for an image
    forever, which is the frozen-per-image failure `D-035` already RED-proved
    wrong; cross-index variation ALONE is satisfied by the stateful stream this
    replaces; and both together are satisfied by a key derivation that drives
    the crop, the flip and the blur off ONE correlated draw. Hence the third.
    """

    def test_two_same_seed_pipelines_agree_under_AUTOTUNE(self):
        """The SHIPPED parallelism, where the stateful path is NOT reproducible.

        `TestRandomness` pins the measurement this closes: at seed 777 two
        AUTOTUNE runs of the stateful map fn differ with maxdiff 1.5312. A
        serial `.map()` version of this test would pass at HEAD and prove
        nothing.
        """
        images = _source_images(16, seed=21)
        first = _run_indexed(
            _stateless_crop_only_map_fn(seed=4242), images,
            num_parallel_calls=tf.data.AUTOTUNE)
        second = _run_indexed(
            _stateless_crop_only_map_fn(seed=4242), images,
            num_parallel_calls=tf.data.AUTOTUNE)
        np.testing.assert_array_equal(first, second)

        # Non-vacuity: a map fn that ignored its seed entirely would also pass
        # the equality above.
        other = _run_indexed(
            _stateless_crop_only_map_fn(seed=4243), images,
            num_parallel_calls=tf.data.AUTOTUNE)
        assert float(np.abs(first - other).max()) > 1e-3, (
            "two DIFFERENT seeds produced bit-identical crops; the seed does "
            "not reach the stateless key"
        )

    def test_the_same_image_is_augmented_differently_at_a_different_index(self):
        """Cross-epoch variation: epoch 2's copy of image k has a NEW index.

        `build_raw_image_dataset` enumerates AFTER `repeat()` precisely so that
        this holds. If the key ignored the index, every epoch would replay one
        frozen augmentation per image and the determinism test above would
        still be green.
        """
        image = _source_images(1, seed=22)
        map_fn = _stateless_crop_only_map_fn(seed=99)

        # Same image, three indices -- i.e. the same source record as seen on
        # three successive epochs of an enumerated, repeated pipeline.
        views = _run_indexed(
            map_fn, np.repeat(image, 3, axis=0), indices=[0, 7, 4242])

        deltas = [
            float(np.abs(views[0] - views[1]).max()),
            float(np.abs(views[0] - views[2]).max()),
            float(np.abs(views[1] - views[2]).max()),
        ]
        assert min(deltas) > 1e-3, (
            f"the SAME source image got the same augmentation at three "
            f"different element indices (max-abs deltas {deltas}); the "
            f"augmentation is FROZEN per image, which is worse for SSL than a "
            f"non-reproducible stream (D-035)."
        )

        # ... and the SAME index reproduces, which is what makes the variation
        # above attributable to the index rather than to residual state.
        again = _run_indexed(map_fn, image, indices=[0])
        np.testing.assert_array_equal(views[0], again[0])

    def test_every_draw_in_one_element_uses_a_DIFFERENT_key(self, monkeypatch):
        """Guard (d): the crop, the flip and the blur must not share a key.

        Asserted on the keys themselves rather than on a statistical
        correlation of the pixels: one shared key per element makes every
        augmentation decision the SAME uniform sample (the crop would sit on
        the image diagonal, and the flip would fire exactly when the crop area
        was small), and no reproducibility or variation test can see that.
        """
        seen_keys = []
        real_stateless_uniform = tf.random.stateless_uniform

        def spy(shape, seed, **kwargs):
            seen_keys.append(tuple(int(v) for v in np.asarray(seed)))
            return real_stateless_uniform(shape, seed=seed, **kwargs)

        monkeypatch.setattr(tf.random, "stateless_uniform", spy)

        # Every probability is 0.0 or resolved at trace time, so the draw count
        # is exactly the crop's: 4 draws (area, aspect, offset_x, offset_y) per
        # view, over N_GLOBAL_VIEWS + 1 views.
        map_fn = _stateless_crop_only_map_fn(seed=7, n_local_crops=1)
        map_fn(tf.constant(3, tf.int64), _source_images(1, seed=23)[0])

        expected = 4 * (N_GLOBAL_VIEWS + 1)
        assert len(seen_keys) == expected, (
            f"expected {expected} crop draws in one element, saw "
            f"{len(seen_keys)}; the draw accounting below is not measuring "
            f"what it claims"
        )
        assert len(set(seen_keys)) == len(seen_keys), (
            f"only {len(set(seen_keys))} distinct keys across "
            f"{len(seen_keys)} draws in ONE element: {seen_keys}. Draws "
            f"sharing a key return the SAME sample, so the crop offset, the "
            f"aspect ratio and every probability draw would be perfectly "
            f"correlated."
        )
        # The element index is the SECOND key component and is the only part
        # that may repeat -- pin that it is really the index, so a key that
        # varied only in its counter (and so ignored the element) is RED.
        assert {key[1] for key in seen_keys} == {3}


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
