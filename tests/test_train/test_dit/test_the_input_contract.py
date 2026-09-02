"""Guard: the DiT latent input contract, its ``.npz`` branch, and the packed target.

The defect classes this exists to catch
---------------------------------------
1. **A CONTRACT STATED ONLY IN PROSE.** ``synthetic_data``'s docstring makes
   claims -- channels-last, ``0.18215`` already applied, labels in range -- that
   no shape check sees on its own. :func:`validate_records` is where the
   checkable half becomes an exception, and this file constructs every one of
   its rejection paths. A documented-but-unasserted contract is prose.

2. **THE PACKED ``y_true`` LAYOUT (D-002).** ``[0:C]=noise``, ``[C:2C]=x_start``,
   ``[2C:2C+1]=t``. Swapping the first two halves yields a target of the
   IDENTICAL shape and dtype that trains a plausible, wrong model. The layout is
   therefore sliced apart and compared against the tensors the pipeline actually
   drew, not against a re-derivation.

3. **THE JOIN BETWEEN THE PIPELINE AND THE LOSS.** ``DDPMHybridLoss`` does not
   receive ``x_t``; it RE-DERIVES it from the packed ``(noise, x_start, t)``.
   If the pipeline noised with one ``t`` and packed another -- or built its
   tables from a different schedule, or respaced them -- the model would be
   trained on a state that does not correspond to its target, with no shape,
   dtype or finiteness symptom anywhere.
   ``TestTheLossRederivesTheSameXT`` is the single most important arm here.

4. **``seed=0`` SILENTLY BEHAVING AS UNSEEDED.** A truthiness test on a seed
   (``if seed:`` / ``if not seed:``) is a measured defect class in this repo.
   ``TestSeedZeroIsHonoured`` asks for the most obvious reproducible seed there
   is and checks it reproduces.

Traps designed out
------------------
* **The x_t comparison must go through the LOSS's own decoding.** It reads
  ``noise``/``x_start``/``t`` back out of the packed target with the loss's own
  ``_unpack`` and gathers its coefficients with the loss's own ``_gather`` off
  the loss's own ``self.schedule``. Only the two-term sum is retyped; every part
  that can be mis-wired belongs to the loss.
* **The class-correlation threshold is DERIVED from the generator's own
  parameters, not pasted.** ``class_signal``/``noise_std``/``n`` give an expected
  separation ratio in closed form, and the arm asserts the measurement clears a
  fraction of it. An anti-vacuity sibling re-runs the same statistic at
  ``class_signal=0.0`` and asserts it collapses to about 1, proving the statistic
  discriminates rather than always firing.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import keras

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit.config import DiffusionConfig
from train.dit.synthetic_data import (
    CONTRACT_KEYS,
    LATENT_SCALE_FACTOR,
    build_dit_dataset,
    build_training_diffusion,
    latents_nchw_to_nhwc,
    load_records_npz,
    pack_target,
    prepare_training_batch,
    save_records_npz,
    synthetic_records,
    validate_records,
)

SEED = 17
COUNT = 24
BATCH = 6


@pytest.fixture()
def config() -> DiffusionConfig:
    """A tiny but legal diffusion config: 8x8x4 latents, 5 classes, T=50."""
    return DiffusionConfig(
        input_size=8,
        in_channels=4,
        num_classes=5,
        num_timesteps=50,
        schedule_name="linear",
    )


@pytest.fixture()
def records(config):
    return synthetic_records(COUNT, config, seed=SEED)


# ---------------------------------------------------------------------
# the contract itself
# ---------------------------------------------------------------------


class TestTheGeneratorSatisfiesItsOwnContract:
    def test_keys_shapes_and_dtypes(self, records, config):
        assert set(records) == set(CONTRACT_KEYS)
        assert validate_records(records, config) == COUNT
        assert records["latent"].shape == (COUNT, 8, 8, 4)
        assert records["latent"].dtype == np.float32
        assert records["label"].shape == (COUNT,)
        assert records["label"].dtype == np.int32

    def test_labels_are_in_range(self, records, config):
        assert records["label"].min() >= 0
        assert records["label"].max() < config.num_classes

    def test_the_scale_factor_is_the_published_one(self):
        # The producer applies it; this package only names it. A drift here
        # silently changes what "a latent" means for every downstream run.
        assert LATENT_SCALE_FACTOR == 0.18215


class TestTheNchwTranspose:
    def test_it_moves_the_channel_axis_to_the_end(self):
        nchw = np.arange(2 * 4 * 3 * 5, dtype="float32").reshape(2, 4, 3, 5)
        nhwc = latents_nchw_to_nhwc(nchw)
        assert nhwc.shape == (2, 3, 5, 4)
        # Independently computed destination index, not a re-invocation of the
        # module's own permutation.
        for n, c, h, w in [(0, 0, 0, 0), (1, 3, 2, 4), (0, 2, 1, 3)]:
            assert nhwc[n, h, w, c] == nchw[n, c, h, w]

    def test_it_rejects_a_non_rank_4_array(self):
        with pytest.raises(ValueError, match="rank-4"):
            latents_nchw_to_nhwc(np.zeros((3, 4, 5), dtype="float32"))


class TestTheValidatorNamesTheOffendingKey:
    def test_missing_key(self, records, config):
        broken = {"latent": records["latent"]}
        with pytest.raises(KeyError, match="label"):
            validate_records(broken, config)

    def test_wrong_rank(self, records, config):
        broken = dict(records)
        broken["latent"] = records["latent"][:, :, :, 0]
        with pytest.raises(ValueError, match="'latent' must be rank 4"):
            validate_records(broken, config)

    def test_nchw_shape_is_named_as_such(self, config):
        broken = {
            "latent": np.zeros((3, 4, 8, 8), dtype="float32"),
            "label": np.zeros((3,), dtype="int32"),
        }
        with pytest.raises(ValueError, match="channels-LAST"):
            validate_records(broken, config)
        with pytest.raises(ValueError, match="latents_nchw_to_nhwc"):
            validate_records(broken, config)

    def test_wrong_latent_dtype(self, records, config):
        broken = dict(records)
        broken["latent"] = records["latent"].astype("int32")
        with pytest.raises(ValueError, match="'latent' must be a floating dtype"):
            validate_records(broken, config)

    def test_wrong_label_dtype(self, records, config):
        broken = dict(records)
        broken["label"] = records["label"].astype("float32")
        with pytest.raises(ValueError, match="'label' must be an integer dtype"):
            validate_records(broken, config)

    def test_label_out_of_range(self, records, config):
        broken = dict(records)
        labels = records["label"].copy()
        labels[0] = config.num_classes
        broken["label"] = labels
        with pytest.raises(ValueError, match="'label' must lie in"):
            validate_records(broken, config)

    def test_negative_label(self, records, config):
        broken = dict(records)
        labels = records["label"].copy()
        labels[3] = -1
        broken["label"] = labels
        with pytest.raises(ValueError, match="'label' must lie in"):
            validate_records(broken, config)

    def test_mismatched_counts(self, records, config):
        broken = {"latent": records["latent"], "label": records["label"][:-1]}
        with pytest.raises(ValueError, match="ragged record batch"):
            validate_records(broken, config)

    def test_empty_batch(self, config):
        broken = {
            "latent": np.zeros((0, 8, 8, 4), dtype="float32"),
            "label": np.zeros((0,), dtype="int32"),
        }
        with pytest.raises(ValueError, match="empty"):
            validate_records(broken, config)


class TestTheNpzRoundTrip:
    def test_it_reads_back_what_it_wrote(self, records, config, tmp_path):
        path = save_records_npz(records, tmp_path / "shard.npz")
        assert path.exists()
        loaded = load_records_npz(path)
        assert validate_records(loaded, config) == COUNT
        np.testing.assert_array_equal(loaded["latent"], records["latent"])
        np.testing.assert_array_equal(loaded["label"], records["label"])

    def test_a_missing_member_raises_naming_it(self, records, tmp_path):
        path = tmp_path / "partial.npz"
        np.savez(path, latent=records["latent"])
        with pytest.raises(KeyError, match="label"):
            load_records_npz(path)

    def test_the_loader_does_not_silently_repair_a_bad_shard(
        self, records, config, tmp_path
    ):
        # A float64 NCHW shard must be REJECTED at the validator, not coerced.
        path = tmp_path / "bad.npz"
        np.savez(
            path,
            latent=np.transpose(records["latent"], (0, 3, 1, 2)).astype("float64"),
            label=records["label"],
        )
        loaded = load_records_npz(path)
        with pytest.raises(ValueError, match="channels-LAST"):
            validate_records(loaded, config)


# ---------------------------------------------------------------------
# the training element
# ---------------------------------------------------------------------


class TestTheElementShapesAndDtypes:
    def test_prepare_training_batch(self, records, config):
        (x_t, t, y), y_true = prepare_training_batch(records, config, seed=3)
        assert x_t.shape == (COUNT, 8, 8, 4)
        assert x_t.dtype == np.float32
        assert t.shape == (COUNT,)
        assert t.dtype == np.int32
        assert y.shape == (COUNT,)
        assert y.dtype == np.int32
        assert y_true.shape == (COUNT, 8, 8, 2 * 4 + 1)
        assert y_true.dtype == np.float32
        assert np.isfinite(x_t).all() and np.isfinite(y_true).all()

    def test_t_is_inside_the_chain(self, records, config):
        _, _ = prepare_training_batch(records, config, seed=1)
        for batch_seed in range(6):
            (_, t, _), _ = prepare_training_batch(records, config, seed=batch_seed)
            assert t.min() >= 0
            assert t.max() < config.num_timesteps

    def test_the_dataset_emits_the_same_shapes(self, records, config):
        dataset = build_dit_dataset(records, config, BATCH, seed=SEED, steps=3)
        elements = list(dataset.as_numpy_iterator())
        assert len(elements) == 3
        for (x_t, t, y), y_true in elements:
            assert x_t.shape == (BATCH, 8, 8, 4)
            assert x_t.dtype == np.float32
            assert t.shape == (BATCH,) and t.dtype == np.int32
            assert y.shape == (BATCH,) and y.dtype == np.int32
            assert y_true.shape == (BATCH, 8, 8, 9)
            assert y_true.dtype == np.float32

    def test_the_npz_path_reaches_the_pipeline(self, records, config, tmp_path):
        # The advertised --train-npz branch, CONSTRUCTED rather than trusted.
        path = save_records_npz(records, tmp_path / "shard.npz")
        loaded = load_records_npz(path)
        dataset = build_dit_dataset(loaded, config, BATCH, seed=SEED, steps=2)
        (x_t, t, y), y_true = next(iter(dataset.as_numpy_iterator()))
        assert x_t.shape == (BATCH, 8, 8, 4)
        assert y_true.shape == (BATCH, 8, 8, 9)
        assert set(np.unique(y)).issubset(set(range(config.num_classes)))

    def test_the_dataset_is_infinite_without_steps(self, records, config):
        dataset = build_dit_dataset(records, config, BATCH, seed=SEED)
        assert tf.data.experimental.cardinality(dataset).numpy() in (
            tf.data.UNKNOWN_CARDINALITY,
            tf.data.INFINITE_CARDINALITY,
        )
        iterator = iter(dataset)
        for _ in range(COUNT // BATCH + 2):
            next(iterator)

    def test_the_dataset_rejects_a_bad_batch_size(self, records, config):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            build_dit_dataset(records, config, 0)
        with pytest.raises(ValueError, match="exceeds the record count"):
            build_dit_dataset(records, config, COUNT + 1)
        with pytest.raises(ValueError, match="steps must be positive"):
            build_dit_dataset(records, config, BATCH, steps=0)


class TestThePackedLayout:
    """D-002: slice the target apart and check each third against what was drawn."""

    def test_the_three_thirds_are_noise_x_start_and_t(self, records, config):
        channels = config.in_channels
        (x_t, t, _), y_true = prepare_training_batch(records, config, seed=5)

        noise_third = y_true[..., 0:channels]
        x_start_third = y_true[..., channels: 2 * channels]
        t_plane = y_true[..., 2 * channels]

        # x_start is what the RECORDS carried -- exactly.
        np.testing.assert_array_equal(x_start_third, records["latent"])

        # t is the drawn timestep, broadcast over every spatial position.
        np.testing.assert_array_equal(
            t_plane, np.broadcast_to(t.astype("float32")[:, None, None], t_plane.shape)
        )

        # The noise third is the residual that, with x_start and t, reproduces
        # x_t through the forward process -- so it cannot be the x_start half.
        process = build_training_diffusion(config)
        rebuilt = keras.ops.convert_to_numpy(
            process.q_sample(records["latent"], t, noise=noise_third)
        )
        np.testing.assert_allclose(rebuilt, x_t, atol=1e-6, rtol=0)

    def test_the_two_halves_are_not_interchangeable(self, records, config):
        # Anti-vacuity: the halves genuinely differ, so an equality check
        # between them is a real discriminator rather than trivially true.
        channels = config.in_channels
        _, y_true = prepare_training_batch(records, config, seed=5)
        assert not np.allclose(
            y_true[..., 0:channels], y_true[..., channels: 2 * channels]
        )

    def test_pack_target_rejects_mismatched_arguments(self):
        noise = np.zeros((3, 4, 4, 2), dtype="float32")
        with pytest.raises(ValueError, match="same shape"):
            pack_target(noise, np.zeros((3, 4, 4, 3), dtype="float32"), np.zeros(3))
        with pytest.raises(ValueError, match="rank 4"):
            pack_target(noise[0], np.zeros((4, 4, 2), dtype="float32"), np.zeros(1))
        with pytest.raises(ValueError, match=r"t must be"):
            pack_target(noise, noise, np.zeros(4))

    def test_the_channel_count_is_what_the_loss_demands(self, records, config):
        _, y_true = prepare_training_batch(records, config, seed=5)
        loss = DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        )
        y_pred = np.zeros(
            (COUNT, 8, 8, 2 * config.in_channels), dtype="float32"
        )
        # No raise: the loss's own static-shape validator accepts this target.
        loss._validate_static_shapes(
            keras.ops.convert_to_tensor(y_true), keras.ops.convert_to_tensor(y_pred)
        )


class TestTheLossRederivesTheSameXT:
    """The join between step 3 (the loss) and step 11 (the pipeline).

    The loss never receives ``x_t``. It decodes ``(noise, x_start, t)`` out of
    the packed target with its OWN ``_unpack`` and gathers its coefficients from
    its OWN ``self.schedule`` with its OWN ``_gather``; only the two-term sum is
    retyped here. Any disagreement -- a swapped pack, a second ``t``, a
    respaced or differently-named schedule -- shows up as a numeric gap.
    """

    @staticmethod
    def _loss_side_x_t(loss: DDPMHybridLoss, y_true, y_pred):
        noise, x_start, t, _, _ = loss._unpack(
            keras.ops.convert_to_tensor(y_true),
            keras.ops.convert_to_tensor(y_pred),
            "float32",
        )
        n_bcast = len(keras.ops.shape(x_start)) - 1
        sched = loss.schedule
        x_t = loss._gather(
            sched.sqrt_alphas_cumprod, t, n_bcast, "float32"
        ) * x_start + loss._gather(
            sched.sqrt_one_minus_alphas_cumprod, t, n_bcast, "float32"
        ) * noise
        return keras.ops.convert_to_numpy(x_t), keras.ops.convert_to_numpy(t)

    def test_it_matches_the_pipelines_x_t(self, records, config):
        (x_t, t, _), y_true = prepare_training_batch(records, config, seed=9)
        loss = DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        )
        y_pred = np.zeros((COUNT, 8, 8, 2 * config.in_channels), dtype="float32")
        loss_x_t, loss_t = self._loss_side_x_t(loss, y_true, y_pred)

        # The timestep the loss reads back off the plane is the one that noised.
        np.testing.assert_array_equal(loss_t.astype("int32"), t)
        np.testing.assert_allclose(loss_x_t, x_t, atol=1e-6, rtol=0)

    def test_it_matches_through_the_dataset_too(self, records, config):
        dataset = build_dit_dataset(records, config, BATCH, seed=SEED, steps=2)
        loss = DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        )
        y_pred = np.zeros((BATCH, 8, 8, 2 * config.in_channels), dtype="float32")
        for (x_t, t, _), y_true in dataset.as_numpy_iterator():
            loss_x_t, loss_t = self._loss_side_x_t(loss, y_true, y_pred)
            np.testing.assert_array_equal(loss_t.astype("int32"), t)
            np.testing.assert_allclose(loss_x_t, x_t, atol=1e-6, rtol=0)

    def test_the_comparison_is_not_vacuous(self, records, config):
        # Control: the same statistic against a DIFFERENT t must NOT agree, so a
        # green reading above means the two sides agree rather than that the
        # instrument always agrees.
        (x_t, t, _), y_true = prepare_training_batch(records, config, seed=9)
        shifted = (t + 1) % config.num_timesteps
        channels = config.in_channels
        wrong = pack_target(
            y_true[..., 0:channels], y_true[..., channels: 2 * channels], shifted
        )
        loss = DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        )
        y_pred = np.zeros((COUNT, 8, 8, 2 * channels), dtype="float32")
        loss_x_t, _ = self._loss_side_x_t(loss, wrong, y_pred)
        assert not np.allclose(loss_x_t, x_t, atol=1e-6, rtol=0)


class TestARealLossEvaluation:
    def test_it_returns_a_finite_per_sample_vector(self, records, config):
        _, y_true = prepare_training_batch(records, config, seed=11)
        loss = DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        )
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal(
            (COUNT, 8, 8, 2 * config.in_channels)
        ).astype("float32")
        value = keras.ops.convert_to_numpy(
            loss.call(
                keras.ops.convert_to_tensor(y_true),
                keras.ops.convert_to_tensor(y_pred),
            )
        )
        assert value.shape == (COUNT,)
        assert np.isfinite(value).all()

    def test_a_dataset_batch_evaluates_too(self, records, config):
        dataset = build_dit_dataset(records, config, BATCH, seed=SEED, steps=1)
        (_, _, _), y_true = next(iter(dataset.as_numpy_iterator()))
        loss = DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        )
        rng = np.random.default_rng(1)
        y_pred = rng.standard_normal(
            (BATCH, 8, 8, 2 * config.in_channels)
        ).astype("float32")
        value = keras.ops.convert_to_numpy(
            loss.call(
                keras.ops.convert_to_tensor(y_true),
                keras.ops.convert_to_tensor(y_pred),
            )
        )
        assert value.shape == (BATCH,)
        assert np.isfinite(value).all()


# ---------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------


class TestSeedZeroIsHonoured:
    """RED against ``if not seed:`` -- a measured defect class in this repo."""

    def test_two_generator_draws_at_seed_zero_are_identical(self, config):
        first = synthetic_records(COUNT, config, seed=0)
        second = synthetic_records(COUNT, config, seed=0)
        np.testing.assert_array_equal(first["latent"], second["latent"])
        np.testing.assert_array_equal(first["label"], second["label"])

    def test_a_different_seed_differs(self, config):
        first = synthetic_records(COUNT, config, seed=0)
        other = synthetic_records(COUNT, config, seed=1)
        assert not np.array_equal(first["latent"], other["latent"])

    def test_two_batches_at_seed_zero_are_identical(self, records, config):
        (a_x, a_t, a_y), a_true = prepare_training_batch(records, config, seed=0)
        (b_x, b_t, b_y), b_true = prepare_training_batch(records, config, seed=0)
        np.testing.assert_array_equal(a_x, b_x)
        np.testing.assert_array_equal(a_t, b_t)
        np.testing.assert_array_equal(a_y, b_y)
        np.testing.assert_array_equal(a_true, b_true)

    def test_a_different_batch_seed_differs(self, records, config):
        (_, a_t, _), _ = prepare_training_batch(records, config, seed=0)
        (_, b_t, _), _ = prepare_training_batch(records, config, seed=1)
        assert not np.array_equal(a_t, b_t)

    def test_two_datasets_at_seed_zero_are_identical(self, records, config):
        def draw():
            dataset = build_dit_dataset(records, config, BATCH, seed=0, steps=2)
            return list(dataset.as_numpy_iterator())

        first, second = draw(), draw()
        for (a_in, a_true), (b_in, b_true) in zip(first, second):
            for a, b in zip(a_in, b_in):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a_true, b_true)

    def test_a_different_dataset_seed_differs(self, records, config):
        a = list(build_dit_dataset(records, config, BATCH, seed=0, steps=2)
                 .as_numpy_iterator())
        b = list(build_dit_dataset(records, config, BATCH, seed=2, steps=2)
                 .as_numpy_iterator())
        assert not np.array_equal(a[0][0][0], b[0][0][0])

    def test_seed_none_is_the_unseeded_branch(self, config):
        first = synthetic_records(COUNT, config, seed=None)
        second = synthetic_records(COUNT, config, seed=None)
        assert not np.array_equal(first["latent"], second["latent"])


# ---------------------------------------------------------------------
# the class correlation
# ---------------------------------------------------------------------


def _separation_ratio(records: dict, config: DiffusionConfig) -> float:
    """Between-class RMS distance of the class means, in units of the null.

    The null is the RMS distance two class means would show if the classes were
    identical: each mean of ``per_class`` samples carries noise of standard
    deviation ``noise_std / sqrt(per_class)`` per element, so a DIFFERENCE of two
    such means has ``sqrt(2) * noise_std / sqrt(per_class)``. Every number below
    comes from the generator's parameters; nothing is pasted.
    """
    latent, label = records["latent"], records["label"]
    means = np.stack(
        [latent[label == c].mean(axis=0) for c in range(config.num_classes)]
    )
    pairs = [
        means[a] - means[b]
        for a in range(config.num_classes)
        for b in range(a + 1, config.num_classes)
    ]
    return float(np.sqrt(np.mean(np.square(np.stack(pairs)))))


class TestTheClassCorrelationIsReal:
    """Without it, ``--smoke``'s "val_loss falls" criterion is unfalsifiable."""

    PER_CLASS = 40
    NOISE_STD = 1.0
    SIGNAL = 1.0

    def _draw(self, config, class_signal):
        """Draw enough samples that every class is populated many times over."""
        total = self.PER_CLASS * config.num_classes
        return synthetic_records(
            total,
            config,
            seed=7,
            class_signal=class_signal,
            noise_std=self.NOISE_STD,
        )

    def test_class_means_separate_beyond_the_sampling_null(self, config):
        records = self._draw(config, self.SIGNAL)
        counts = np.bincount(records["label"], minlength=config.num_classes)
        smallest = int(counts.min())
        assert smallest > 0

        measured = _separation_ratio(records, config)
        # Null: two independent class means differ by sqrt(2)*sigma/sqrt(n).
        null = np.sqrt(2.0) * self.NOISE_STD / np.sqrt(smallest)
        # Signal: two independent mean fields differ by sqrt(2)*class_signal.
        expected = np.sqrt(2.0 * self.SIGNAL**2 + null**2)
        assert measured > 3.0 * null, (
            f"class means separated by {measured:.4f}, null {null:.4f}"
        )
        assert measured == pytest.approx(expected, rel=0.25)

    def test_the_statistic_collapses_without_the_correlation(self, config):
        # ANTI-VACUITY. At class_signal=0 the generator is pure noise and the
        # same statistic must land on the null, proving it discriminates.
        records = self._draw(config, 0.0)
        counts = np.bincount(records["label"], minlength=config.num_classes)
        smallest = int(counts.min())
        measured = _separation_ratio(records, config)
        null = np.sqrt(2.0) * self.NOISE_STD / np.sqrt(counts.max())
        assert measured < 3.0 * np.sqrt(2.0) * self.NOISE_STD / np.sqrt(smallest)
        assert measured > 0.2 * null

    def test_the_generator_rejects_a_negative_knob(self, config):
        with pytest.raises(ValueError, match="class_signal"):
            synthetic_records(4, config, seed=0, class_signal=-1.0)
        with pytest.raises(ValueError, match="noise_std"):
            synthetic_records(4, config, seed=0, noise_std=-0.5)

    def test_num_samples_must_be_positive(self, config):
        with pytest.raises(ValueError, match="num_samples must be positive"):
            synthetic_records(0, config, seed=0)
