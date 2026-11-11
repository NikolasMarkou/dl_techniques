"""The denoising data of ``train.api``: stateless noise, a finite patch epoch, a fixed validation set.

Covers ``train.common.denoising_data`` (listing, selection, the pre-fit scan, noise, pipelines, fingerprint) and the
``DenoisingData`` group of ``train.api.denoising_args`` (its refusals, its file form, the glue that reads it).

The bounds below are measured, not guessed. Each statistical test states its sample size, the standard error that
size gives and the bound it uses (a multiple of that error); the seeds are fixed, so a run is deterministic.

* Variance of ``y | x``: a patch of 256 x 256 x 3 = 196,608 values has a relative standard error of
  ``sqrt(2 / N) = 0.32 %`` on a variance; the bound is 3 % (about 9 standard errors).
* Input PSNR of ``sigma ~ U(0, 0.25)``: the per-image PSNR ``-20 log10(sigma)`` has a standard deviation of 8.69 dB, so
  6,000 images give a standard error of 0.11 dB; the bound is 0.4 dB (3.6 standard errors) around the 20.73 dB of
  ``-20 / ln(10) * (ln(0.25) - 1)``.
* Two-sample Kolmogorov-Smirnov distance of the per-patch sigma estimates (legacy against this module), 1,500 patches
  of 16 x 16 x 3 each: the 99.9 % critical distance is ``1.95 * sqrt(2 / 1500) = 0.071``; the bound is 0.08. A noise
  with a different sigma range sits above 0.3 (the positive control).
"""
import dataclasses
import os
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from scipy import stats

from train.api import denoising_args as args
from train.api.config import AUTO
from train.api.errors import ConfigError, ConfigValidationError
from train.common import denoising_data as dd
from train.common.classification_data import FINGERPRINT_MODES, DataSourceError

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def write_images(root: Path, count: int, *, size: Tuple[int, int] = (48, 64), seed: int = 0, start: int = 0,
                 suffix: str = "png") -> List[str]:
    """Write ``count`` random RGB images ``size`` = (height, width) under ``root``; return their paths."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for index in range(start, start + count):
        path = root / f"img_{index:03d}.{suffix}"
        Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def spec(**overrides: Any) -> dd.NoiseSpec:
    """A ``NoiseSpec`` (every field is required there) with the config's defaults, overridden by name."""
    values = dict(kind="additive", sigma_min=0.0, sigma_max=0.25, sigma_max_start=0.025, composite_ratio=0.5, clip=False)
    values.update(overrides)
    return dd.NoiseSpec(**values)


def load(train: Sequence[Path], val: Sequence[Path], **overrides: Any) -> dd.DenoisingBatches:
    """``load_denoising_data`` of small folders with small defaults, overridden by name."""
    values: Dict[str, Any] = dict(
        patch_size=32, channels=3, patches_per_image=2, batch_size=4, seed=1, noise=spec(), val_seed=7,
        max_train_files=None, max_val_files=None, max_samples=None)
    values.update(overrides)
    return dd.load_denoising_data([str(p) for p in train], [str(p) for p in val], **values)


@pytest.fixture
def folders(tmp_path: Path) -> Tuple[Path, Path]:
    """A train folder of 8 images and a validation folder of 4."""
    write_images(tmp_path / "train", 8)
    write_images(tmp_path / "val", 4, seed=1)
    return tmp_path / "train", tmp_path / "val"


def first_batches(dataset: "tf.data.Dataset", count: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [(x.numpy(), y.numpy()) for x, y in dataset.take(count)]


# ---------------------------------------------------------------------
# One home for the choices
# ---------------------------------------------------------------------


def test_the_config_and_the_loader_name_the_same_noise_kinds_and_schedules():
    from dl_techniques.callbacks.noise_sigma_curriculum import NoiseSigmaCurriculumCallback

    assert args.NOISE_KINDS == dd.NOISE_KINDS
    assert set(args.CURRICULUM_SCHEDULES) == set(NoiseSigmaCurriculumCallback.SCHEDULES)


def test_the_composite_ratio_help_states_the_ratio_the_loader_uses():
    help_text = {f.name: f.metadata["help"] for f in dataclasses.fields(args.DenoisingData)}["noise_composite_ratio"]
    assert str(dd.DEFAULT_COMPOSITE_RATIO) in help_text


def test_the_defaults_are_the_decided_ones():
    data = args.load_denoising_data_config({"train_dirs": ["a"], "val_dirs": ["b"]})
    assert data.noise_clip is False, "D-009: the noisy input is not clipped unless asked"
    assert data.noise_kind == "additive" and data.noise_composite_ratio is None
    assert data.curriculum_epochs == AUTO and data.curriculum_schedule == "linear"
    assert (data.noise_sigma_min, data.noise_sigma_max_start, data.noise_sigma_max) == (0.0, 0.025, 0.25)
    assert (data.max_train_files, data.max_val_files, data.max_samples) == (10000, 500, None)
    assert data.format == "folder"


# ---------------------------------------------------------------------
# Noise: a pure function of (clean, sigma, seed)
# ---------------------------------------------------------------------

SEED = tf.constant([11, 22], tf.int64)


def noisy(clean, sigma, seed=SEED, kind="additive", ratio=0.5, clip=False):
    return dd.add_noise(tf.constant(clean, tf.float32), sigma, seed, kind=kind, composite_ratio=ratio, clip=clip).numpy()


@pytest.mark.parametrize("kind", dd.NOISE_KINDS)
def test_noise_is_a_pure_function_of_clean_sigma_and_seed(kind):
    clean = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    first = noisy(clean, 0.1, kind=kind)
    assert first.tobytes() == noisy(clean, 0.1, kind=kind).tobytes()
    assert first.tobytes() != noisy(clean, 0.1, seed=tf.constant([11, 23], tf.int64), kind=kind).tobytes()
    assert first.tobytes() != noisy(clean, 0.2, kind=kind).tobytes()


def expected_variance(kind: str, x: float, sigma: float, ratio: float) -> float:
    if kind == "additive":
        return sigma ** 2
    if kind == "multiplicative":
        return sigma ** 2 * x ** 2
    return sigma ** 2 * x ** 2 + (ratio * sigma) ** 2


@pytest.mark.parametrize("kind", dd.NOISE_KINDS)
@pytest.mark.parametrize("sigma", [0.05, 0.2])
def test_the_variance_of_y_given_x_matches_its_formula_over_a_grid(kind, sigma):
    """Var(y | x) is s^2, s^2 x^2 and s^2 x^2 + (r s)^2; N = 196,608 values per x gives a 0.32 % standard error, bound 3 %."""
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        out = noisy(np.full((256, 256, 3), x, np.float32), sigma, kind=kind, ratio=0.5)
        want = expected_variance(kind, x, sigma, 0.5)
        got = float(np.var(out - x))
        assert abs(got / want - 1) < 0.03, (kind, sigma, x, got, want)
        # the mean of y is x: 5 standard errors of the mean
        assert abs(float(np.mean(out)) - x) < 5 * np.sqrt(want / out.size), (kind, sigma, x)


@pytest.mark.parametrize("kind", dd.NOISE_KINDS)
def test_sigma_zero_returns_the_clean_image(kind):
    clean = np.random.default_rng(1).random((16, 16, 3), dtype=np.float32)
    assert np.array_equal(noisy(clean, 0.0, kind=kind), clean)


def test_the_clip_changes_the_noisy_image_only_when_asked():
    clean = np.zeros((32, 32, 3), np.float32)
    unclipped, clipped = noisy(clean, 0.2), noisy(clean, 0.2, clip=True)
    assert unclipped.min() < 0 < unclipped.max() and clipped.min() >= 0 and clipped.max() <= 1
    assert not np.array_equal(unclipped, clipped)


def test_a_noise_kind_that_is_none_of_the_three_is_refused():
    with pytest.raises(ValueError, match="noise kind"):
        noisy(np.zeros((4, 4, 3), np.float32), 0.1, kind="poisson")
    with pytest.raises(ValueError, match="noise kind"):
        spec(kind="poisson")


@pytest.mark.parametrize("bad", [dict(sigma_min=0.3), dict(sigma_min=-0.1), dict(sigma_max_start=-0.5, sigma_min=0.1),
                                 dict(composite_ratio=-1.0)])
def test_a_noise_spec_out_of_range_is_refused(bad):
    with pytest.raises(ValueError):
        spec(**bad)


def batch_noise(kind: str, sigma_max: float, n: int = 1500, size: int = 16) -> np.ndarray:
    """The per-patch sigma estimate of ``n`` patches of constant 0.5 noised with this module's draw."""
    seeds = tf.stack([tf.range(n, dtype=tf.int64), tf.fill([n], tf.constant(9, tf.int64))], axis=1)
    sigmas = tf.vectorized_map(lambda s: dd.draw_sigma(s, 0.0, sigma_max), seeds)
    clean = tf.fill([n, size, size, 3], 0.5)
    out = dd.add_noise(clean, sigmas[:, None, None, None], tf.constant([5, 6], tf.int64), kind=kind,
                       composite_ratio=0.5, clip=False)
    return per_patch_sigma(out.numpy() - 0.5, kind)


def per_patch_sigma(residual: np.ndarray, kind: str) -> np.ndarray:
    """``std(y - x) / scale`` per patch for a constant x = 0.5; ``scale`` is what a unit sigma gives."""
    scale = {"additive": 1.0, "multiplicative": 0.5, "composite": float(np.sqrt(0.25 + 0.25))}[kind]
    return residual.reshape(len(residual), -1).std(axis=1) / scale


def legacy_sigmas(kind: str, n: int = 1500, size: int = 16) -> np.ndarray:
    from train.bfunet.common import make_curriculum_noise_fn

    config = types.SimpleNamespace(noise_sigma_min=0.0, noise_type=kind, composite_additive_ratio=0.5)
    fn = make_curriculum_noise_fn(config, tf.Variable(0.25, dtype=tf.float32), clip_noise=False)
    tf.random.set_seed(1234)
    clean = tf.fill([size, size, 3], 0.5)
    residuals = np.stack([(fn(clean)[0] - clean).numpy() for _ in range(n)])
    return per_patch_sigma(residuals, kind)


@pytest.mark.parametrize("kind", dd.NOISE_KINDS)
def test_the_distribution_agrees_with_the_legacy_curriculum_noise(kind):
    """KS distance of the per-patch sigma estimates, legacy against this module: bound 0.08 (critical 0.071 at 99.9 %)."""
    legacy, mine = legacy_sigmas(kind), batch_noise(kind, 0.25)
    assert stats.ks_2samp(legacy, mine).statistic < 0.08
    # the positive control: the same test sees a noise whose sigma range is half as wide
    assert stats.ks_2samp(legacy, batch_noise(kind, 0.125)).statistic > 0.3


def per_image_psnr(noisy_images: np.ndarray, clean: np.ndarray) -> float:
    mse = ((noisy_images - clean) ** 2).reshape(len(clean), -1).mean(axis=1)
    return float(np.mean(-10 * np.log10(mse)))


def test_the_unclipped_input_psnr_of_a_zero_to_a_quarter_sigma_is_20_73_db_and_the_clip_changes_it():
    """6,000 images: standard error 0.11 dB, bound 0.4 dB around 20.73 dB; a clip adds noise-free pixels, so it raises it."""
    n = 6000
    seeds = tf.stack([tf.range(n, dtype=tf.int64), tf.fill([n], tf.constant(3, tf.int64))], axis=1)
    sigmas = tf.vectorized_map(lambda s: dd.draw_sigma(s, 0.0, 0.25), seeds)[:, None, None, None]
    clean = tf.random.stateless_uniform([n, 16, 16, 1], seed=[4, 4], dtype=tf.float32)
    closed_form = -20 / np.log(10) * (np.log(0.25) - 1)
    assert abs(closed_form - 20.73) < 0.005
    unclipped = per_image_psnr(dd.add_noise(clean, sigmas, SEED, kind="additive", composite_ratio=0.5, clip=False).numpy(),
                               clean.numpy())
    clipped = per_image_psnr(dd.add_noise(clean, sigmas, SEED, kind="additive", composite_ratio=0.5, clip=True).numpy(),
                             clean.numpy())
    assert abs(unclipped - 20.73) < 0.4, unclipped
    assert clipped - unclipped > 0.3, (clipped, unclipped)


# ---------------------------------------------------------------------
# Listing and selection
# ---------------------------------------------------------------------


def test_a_directory_that_is_missing_or_holds_no_image_is_refused_naming_the_key(tmp_path):
    with pytest.raises(DataSourceError) as missing:
        dd.list_images(str(tmp_path / "nope"), "train_dirs")
    assert missing.value.key == "train_dirs" and "nope" in str(missing.value)
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("x")
    with pytest.raises(DataSourceError, match="holds no images"):
        dd.list_images(str(tmp_path / "empty"), "val_dirs")


def test_the_listing_is_recursive_sorted_and_skips_hidden_entries(tmp_path):
    write_images(tmp_path / "root", 2)
    write_images(tmp_path / "root" / "sub", 2, start=10)
    write_images(tmp_path / "root" / ".hidden", 1, start=20)
    (tmp_path / "root" / ".dot.png").write_bytes(b"x")
    (tmp_path / "root" / "z.JPG").write_bytes(b"x")
    names = [os.path.relpath(p, tmp_path / "root") for p in dd.list_images(str(tmp_path / "root"), "train_dirs")]
    assert names == ["img_000.png", "img_001.png", "z.JPG", os.path.join("sub", "img_010.png"), os.path.join("sub", "img_011.png")]


def test_a_dangling_symbolic_link_is_named_not_skipped(tmp_path):
    write_images(tmp_path / "root", 1)
    os.symlink(tmp_path / "gone.png", tmp_path / "root" / "broken.png")
    with pytest.raises(DataSourceError, match="broken.png.*leads nowhere"):
        dd.list_images(str(tmp_path / "root"), "train_dirs")


def test_a_file_in_both_the_train_and_the_validation_list_is_refused(tmp_path):
    write_images(tmp_path / "train", 3)
    os.symlink(tmp_path / "train", tmp_path / "val_link")
    with pytest.raises(DataSourceError, match="in both") as error:
        dd.list_sources([str(tmp_path / "train")], [str(tmp_path / "val_link")])
    assert error.value.key == "val_dirs"


def test_with_a_cap_every_train_directory_weighs_the_same_and_a_small_one_wraps():
    pools = [[f"a{i}" for i in range(5)], ["b0", "b1"]]
    chosen = dd.select_train_files(pools, 8, seed=3)
    by_dir = {0: [p for i, p in chosen if i == 0], 1: [p for i, p in chosen if i == 1]}
    assert len(by_dir[0]) == len(by_dir[1]) == 4
    assert len(set(by_dir[0])) == 4 and sorted(by_dir[1]) == ["b0", "b0", "b1", "b1"]
    assert chosen == dd.select_train_files(pools, 8, seed=3) and chosen != dd.select_train_files(pools, 8, seed=4)
    odd = dd.select_train_files(pools, 9, seed=3)
    assert [sum(1 for i, _ in odd if i == k) for k in (0, 1)] == [5, 4], "the first directories get the remainder"


def test_without_a_cap_every_file_is_used_once_and_a_cap_below_the_directory_count_is_refused():
    pools = [["a0", "a1", "a2"], ["b0"]]
    assert dd.select_train_files(pools, None, seed=0) == [(0, "a0"), (0, "a1"), (0, "a2"), (1, "b0")]
    with pytest.raises(ValueError, match="below the 2 train directories"):
        dd.select_train_files(pools, 1, seed=0)


def test_the_validation_subset_is_a_function_of_val_seed_alone_and_keeps_listing_order():
    pools = [[f"v{i:02d}" for i in range(20)]]
    first = dd.select_val_files(pools, 5, val_seed=2)
    assert first == dd.select_val_files(pools, 5, val_seed=2) and first != dd.select_val_files(pools, 5, val_seed=3)
    assert [p for _, p in first] == sorted(p for _, p in first)
    assert dd.select_val_files(pools, None, val_seed=2) == [(0, p) for p in pools[0]]


# ---------------------------------------------------------------------
# Images: grey, RGBA, small, flat black, undecodable
# ---------------------------------------------------------------------


def test_grey_and_rgba_images_decode_to_the_requested_channels(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    rng = np.random.default_rng(5)
    Image.fromarray(rng.integers(1, 256, (40, 40), dtype=np.uint8), "L").save(root / "grey.png")
    Image.fromarray(rng.integers(1, 256, (40, 40, 4), dtype=np.uint8), "RGBA").save(root / "rgba.png")
    Image.fromarray(rng.integers(1, 256, (40, 40, 3), dtype=np.uint8)).save(root / "rgb.png")
    write_images(tmp_path / "val", 2)
    for channels in (3, 1):
        data = load([root], [tmp_path / "val"], channels=channels, patches_per_image=2, batch_size=2)
        (x, y), = first_batches(data.train_ds, 1)
        assert x.shape == y.shape == (2, 32, 32, channels)
        assert np.isfinite(x).all() and 0 <= y.min() and y.max() <= 1
        assert data.input_shape == (32, 32, channels)
        assert data.n_train == 6 and all(first_batches(data.select_ds, 1)[0][0].shape[-1] == channels for _ in (0,))


def test_an_image_smaller_than_the_patch_is_upscaled_and_gives_patches(tmp_path):
    write_images(tmp_path / "train", 4, size=(16, 24))
    write_images(tmp_path / "val", 2, size=(20, 20), seed=1)
    data = load([tmp_path / "train"], [tmp_path / "val"], patch_size=32, batch_size=4, patches_per_image=2)
    (x, y), = first_batches(data.train_ds, 1)[:1]
    assert y.shape == (4, 32, 32, 3) and y.std() > 0.05, "the upscaled image is an image, not a flat fill"
    assert len(first_batches(data.select_ds, 10)) == 1


def test_a_flat_black_image_leaves_the_list_and_the_epoch_is_one_image_shorter(tmp_path):
    train = tmp_path / "train"
    write_images(train, 5)
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(train / "black.png")
    write_images(tmp_path / "val", 2)
    data = load([train], [tmp_path / "val"], patches_per_image=2, batch_size=2)
    assert len(data.train_files) == 5 and "black.png" not in [os.path.basename(f) for f in data.train_files]
    assert data.n_train == 10 and int(data.train_ds.cardinality()) == 5


def test_a_directory_of_only_flat_black_images_is_refused(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(train / "black.png")
    write_images(tmp_path / "val", 2)
    with pytest.raises(DataSourceError, match="no image with a value above 0") as error:
        load([train], [tmp_path / "val"])
    assert error.value.key == "train_dirs"


@pytest.mark.parametrize("where, key", [("train", "train_dirs"), ("val", "val_dirs")])
def test_an_undecodable_file_is_named_before_fit(tmp_path, where, key):
    write_images(tmp_path / "train", 4)
    write_images(tmp_path / "val", 3)
    (tmp_path / where / "zz_broken.png").write_bytes(b"this is not a png")
    with pytest.raises(DataSourceError) as error:
        load([tmp_path / "train"], [tmp_path / "val"])
    assert error.value.key == key and "zz_broken.png" in str(error.value) and "cannot be decoded" in str(error.value)


def test_a_batch_larger_than_an_epochs_patches_is_refused(tmp_path):
    write_images(tmp_path / "train", 2)
    write_images(tmp_path / "val", 2)
    with pytest.raises(DataSourceError, match="exceeds the 4 training patches"):
        load([tmp_path / "train"], [tmp_path / "val"], batch_size=8, patches_per_image=2)


def test_max_samples_caps_the_train_and_the_validation_files(folders):
    train, val = folders
    data = load([train], [val], max_samples=3, batch_size=2)
    assert len(data.train_files) == 3 and len(data.val_files) == 3
    capped = load([train], [val], max_train_files=5, max_val_files=2, max_samples=6, batch_size=2)
    assert len(capped.train_files) == 5 and len(capped.val_files) == 2


# ---------------------------------------------------------------------
# The epoch: its length, as fit sees it
# ---------------------------------------------------------------------


class CountBatches(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.per_epoch: List[int] = []

    def on_epoch_begin(self, epoch, logs=None):
        self.per_epoch.append(0)

    def on_train_batch_end(self, batch, logs=None):
        self.per_epoch[-1] += 1


def tiny_model(channels: int = 3) -> keras.Model:
    inputs = keras.Input((None, None, channels))
    model = keras.Model(inputs, keras.layers.Conv2D(channels, 1)(inputs))
    model.compile("adam", "mse", jit_compile=False)
    return model


def test_the_epoch_states_its_length_and_fit_takes_exactly_that_many_steps(folders):
    train, val = folders
    data = load([train], [val], patches_per_image=3, batch_size=5)  # 8 files x 3 = 24 patches: 4 steps, 4 dropped
    assert data.n_train == 24 and data.steps_per_epoch == 24 // 5
    assert int(data.train_ds.cardinality()) == data.steps_per_epoch, "the length is stated, not unknown"
    assert int(data.select_ds.cardinality()) == -(-data.n_select // 5) == 3
    counter = CountBatches()
    history = tiny_model().fit(data.train_ds, validation_data=data.select_ds, epochs=3, callbacks=[counter], verbose=0)
    assert counter.per_epoch == [data.steps_per_epoch] * 3
    assert sum(1 for _ in data.train_ds) == data.steps_per_epoch
    assert sum(len(x) for x, _ in data.select_ds) == data.n_select == 12
    assert np.isfinite(history.history["loss"]).all() and len(history.history["val_loss"]) == 3


def test_a_cap_with_wrap_makes_the_epoch_files_times_patches(tmp_path):
    write_images(tmp_path / "small", 3)
    write_images(tmp_path / "big", 12, seed=1)
    write_images(tmp_path / "val", 2, seed=2)
    data = load([tmp_path / "small", tmp_path / "big"], [tmp_path / "val"], max_train_files=10, batch_size=4)
    assert len(data.train_files) == 10 and data.n_train == 20
    small = [f for f in data.train_files if f.startswith(str(tmp_path / "small"))]
    assert len(small) == 5 and len(set(small)) == 3, "5 files from a directory of 3: a file repeats"
    assert sum(1 for _ in data.train_ds) == 5


# ---------------------------------------------------------------------
# Seeds: training noise differs, validation does not
# ---------------------------------------------------------------------


def test_two_seeds_give_different_training_noise_and_the_same_validation_set(folders):
    train, val = folders
    one, two, again = load([train], [val], seed=1), load([train], [val], seed=2), load([train], [val], seed=1)
    x1, x2, x1b = (first_batches(d.train_ds, 2) for d in (one, two, again))
    assert all(a[0].tobytes() != b[0].tobytes() for a, b in zip(x1, x2)), "another seed: another noise"
    assert all(a[1].tobytes() != b[1].tobytes() for a, b in zip(x1, x2)), "and other crops and order"
    assert [a[0].tobytes() for a in x1] == [a[0].tobytes() for a in x1b], "one seed: one stream"
    for a, b in zip(first_batches(one.select_ds, 3), first_batches(two.select_ds, 3)):
        assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


def test_two_validation_passes_are_byte_equal_and_val_seed_changes_them(folders):
    train, val = folders
    data = load([train], [val])
    first, second = first_batches(data.select_ds, 10), first_batches(data.select_ds, 10)
    assert len(first) == 2 and [x.tobytes() for x, _ in first] == [x.tobytes() for x, _ in second]
    assert [y.tobytes() for _, y in first] == [y.tobytes() for _, y in second]
    other = first_batches(load([train], [val], val_seed=8).select_ds, 10)
    assert first[0][0].tobytes() != other[0][0].tobytes() and first[0][1].tobytes() != other[0][1].tobytes()


def test_every_epoch_has_its_own_training_noise(folders):
    train, val = folders
    data = load([train], [val])
    epoch_one, epoch_two = first_batches(data.train_ds, 1)[0], first_batches(data.train_ds, 1)[0]
    assert epoch_one[0].tobytes() != epoch_two[0].tobytes()


def test_the_validation_sigma_is_drawn_from_the_configured_range_and_is_fixed_per_patch(tmp_path):
    write_images(tmp_path / "train", 4)
    write_images(tmp_path / "val", 20, size=(40, 40))
    flat = tmp_path / "flat"
    flat.mkdir()
    for i in range(20):
        Image.fromarray(np.full((40, 40, 3), 128, np.uint8)).save(flat / f"{i}.png")
    data = load([tmp_path / "train"], [flat], noise=spec(sigma_min=0.04, sigma_max=0.12, sigma_max_start=0.05), patches_per_image=4)
    residual = np.concatenate([x - y for x, y in data.select_ds.take(40)])
    sigma_hat = residual.reshape(len(residual), -1).std(axis=1)
    assert 0.04 * 0.9 < sigma_hat.min() and sigma_hat.max() < 0.12 * 1.1 and sigma_hat.std() > 0.015


def test_a_default_config_leaves_the_noisy_input_unclipped_and_noise_clip_true_clips_it(tmp_path):
    """D-009 through the whole path (config, glue, pipeline): dark images plus sigma up to 0.25 leave [0, 1] unless clipped."""
    for name, count in (("train", 6), ("val", 2)):
        (tmp_path / name).mkdir()
        for i in range(count):
            Image.fromarray(np.full((40, 40, 3), 5, np.uint8)).save(tmp_path / name / f"{i}.png")
    base = {"train_dirs": [str(tmp_path / "train")], "val_dirs": [str(tmp_path / "val")], "noise_sigma_max_start": 0.25}
    for extra, stays_in_range in (({}, False), ({"noise_clip": True}, True)):
        data = args.load_denoising(args.load_denoising_data_config({**base, **extra}), patch_size=32, channels=3,
                                   patches_per_image=2, batch_size=4, seed=1)
        data.sigma_max_var.assign(0.25)
        noisy_input = np.concatenate([x for x, _ in data.train_ds.take(3)])
        validation = np.concatenate([x for x, _ in data.select_ds])
        assert bool(0 <= noisy_input.min() and noisy_input.max() <= 1) == stays_in_range, extra
        assert bool(0 <= validation.min() and validation.max() <= 1) == stays_in_range, extra


def test_an_augment_hook_gets_a_seed_and_changes_only_the_clean_patch(folders):
    train, val = folders

    def flip(seed, patch):
        return tf.image.stateless_random_flip_left_right(patch, seed=seed)

    plain = first_batches(load([train], [val]).train_ds, 2)
    hooked = first_batches(load([train], [val], augment=flip).train_ds, 2)
    assert hooked[0][1].tobytes() == first_batches(load([train], [val], augment=flip).train_ds, 1)[0][1].tobytes()
    flipped = 0
    for (_, clean_plain), (_, clean_hooked) in zip(plain, hooked):
        for a, b in zip(clean_plain, clean_hooked):
            assert np.array_equal(a, b) or np.array_equal(a, b[:, ::-1])
            flipped += not np.array_equal(a, b)
    assert flipped >= 1, "a hook that is never applied would pass the loop above"


# ---------------------------------------------------------------------
# The curriculum variable is read at the epoch boundary
# ---------------------------------------------------------------------

STD = {"epoch": None, "batches": []}


def _record(x):
    STD["batches"].append((STD["epoch"], float(np.std(x.numpy(), axis=(1, 2, 3)).max())))


class Recorder(keras.layers.Layer):
    """Stores, for every training batch, the largest std of a patch: the sigma the pipeline drew for it.

    The clean patches of these tests are flat grey, so a noisy patch's std over its pixels IS its sigma (a patch of
    32 x 32 x 3 = 3,072 values: 1.3 % standard error). A ``py_function`` in the compiled train step reads the batch.
    """

    def call(self, x, training=None):
        if training:
            tf.py_function(_record, [x], [])
        return x


class Alternate(keras.callbacks.Callback):
    """Sets the upper bound to 0.01 on even epochs and 0.25 on odd ones, and tells the recorder the epoch."""

    def __init__(self, variable):
        super().__init__()
        self.variable = variable

    def on_epoch_begin(self, epoch, logs=None):
        STD["epoch"] = epoch
        self.variable.assign(0.01 if epoch % 2 == 0 else 0.25)


class AnnounceEpoch(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        STD["epoch"] = epoch


def grey_folders(tmp_path: Path, files: int) -> Tuple[Path, Path]:
    for name, count in (("train", files), ("val", 4)):
        (tmp_path / name).mkdir()
        for i in range(count):
            Image.fromarray(np.full((48, 48, 3), 128, np.uint8)).save(tmp_path / name / f"{i}.png")
    return tmp_path / "train", tmp_path / "val"


def fit_with_recorder(data: dd.DenoisingBatches, callbacks: List[keras.callbacks.Callback], epochs: int):
    """Fit a tiny model on ``data`` (with its validation set) and return ``[(epoch, largest std in the batch), ...]``."""
    STD["batches"].clear()
    STD["epoch"] = None
    inputs = keras.Input((32, 32, 3))
    model = keras.Model(inputs, keras.layers.Conv2D(3, 1)(Recorder()(inputs)))
    model.compile("adam", "mse", jit_compile=False)
    model.fit(data.train_ds, validation_data=data.select_ds, epochs=epochs, verbose=0, callbacks=callbacks)
    return list(STD["batches"])


@pytest.mark.parametrize("files, ppi", [(24, 2), (8, 1), (8, 2)])
def test_a_batch_reads_the_sigma_max_of_its_own_epoch_not_the_previous_one(tmp_path, files, ppi):
    """MEASURED (step 5): Keras creates the iterator of an epoch after ``on_epoch_begin``, so no batch of 460 read the
    previous epoch's value. A batch of 8 patches whose largest std is below 0.0125 was drawn under 0.01 (the chance that
    8 patches of the 0.25 range all fall below it is 0.05^8), so a batch that disagrees with its epoch read another value.
    """
    train, val = grey_folders(tmp_path, files)
    cfg = args.load_denoising_data_config(
        {"train_dirs": [str(train)], "val_dirs": [str(val)], "max_train_files": files, "noise_sigma_max_start": 0.01,
         "curriculum_epochs": 2})
    data = args.load_denoising(cfg, patch_size=32, channels=3, patches_per_image=ppi, batch_size=8, seed=3)
    batches = fit_with_recorder(data, [Alternate(data.sigma_max_var)], epochs=10)
    assert len(batches) == 10 * data.steps_per_epoch
    stale = [(epoch, round(top, 4)) for epoch, top in batches if (top < 0.0125) != (epoch % 2 == 0)]
    assert not stale, f"batches that read another epoch's sigma_max: {stale}"


def test_the_sigma_of_a_batch_follows_the_real_curriculum_and_never_exceeds_it(tmp_path):
    """With the real callback (0.05 to 0.25, linear over 4 epochs): a batch never exceeds 1.1 x the schedule at its
    epoch (1.3 % standard error per std estimate), and an epoch of 96 patches reaches 0.85 x it (a stale epoch,
    0.73 x at worst, would not; P(96 draws all below 0.85 of the range) = 1.7e-7)."""
    train, val = grey_folders(tmp_path, 24)
    cfg = args.load_denoising_data_config(
        {"train_dirs": [str(train)], "val_dirs": [str(val)], "max_train_files": 24, "noise_sigma_max_start": 0.05,
         "curriculum_epochs": 4})
    data = args.load_denoising(cfg, patch_size=32, channels=3, patches_per_image=4, batch_size=8, seed=3)
    batches = fit_with_recorder(data, [AnnounceEpoch(), args.curriculum_callback(cfg, data.sigma_max_var)], epochs=4)
    schedule = [0.05 + (0.25 - 0.05) * e / 3 for e in range(4)]
    for epoch in range(4):
        tops = [top for e, top in batches if e == epoch]
        assert len(tops) == data.steps_per_epoch
        assert max(tops) <= 1.1 * schedule[epoch], (epoch, tops, schedule[epoch])
        assert max(tops) >= 0.85 * schedule[epoch], (epoch, tops, schedule[epoch])


# ---------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------


def fingerprint(train, val, mode="listing", **overrides):
    return load(train, val, **overrides, fingerprint=mode).fingerprint


def test_the_fingerprint_follows_the_files_and_not_where_the_tree_is(tmp_path, folders):
    train, val = folders
    base = fingerprint([train], [val])
    assert base == fingerprint([train], [val]) and len(base) == 64
    moved = tmp_path / "moved"
    os.makedirs(moved)
    os.replace(train, moved / "t")
    os.replace(val, moved / "v")
    assert fingerprint([moved / "t"], [moved / "v"]) == base, "paths are relative to their directory"
    write_images(moved / "t", 1, start=50)
    assert fingerprint([moved / "t"], [moved / "v"]) != base


def test_the_listing_fingerprint_misses_a_same_size_edit_and_the_content_one_sees_it(tmp_path):
    """A bmp of a given shape has a fixed size, so another image of that shape is an edit that keeps every name and size."""
    train, val = tmp_path / "train", tmp_path / "val"
    write_images(train, 4, suffix="bmp")
    write_images(val, 2, seed=1, suffix="bmp")
    listing, content = fingerprint([train], [val]), fingerprint([train], [val], "content")
    assert listing != content and set(FINGERPRINT_MODES) == {"listing", "content"}
    target = sorted(train.glob("*.bmp"))[0]
    size = target.stat().st_size
    write_images(tmp_path / "other", 1, seed=99, suffix="bmp")
    target.write_bytes((tmp_path / "other" / "img_000.bmp").read_bytes())
    assert target.stat().st_size == size
    assert fingerprint([train], [val]) == listing
    assert fingerprint([train], [val], "content") != content


def test_an_unknown_fingerprint_mode_is_refused(folders):
    train, val = folders
    with pytest.raises(ValueError, match="fingerprint mode"):
        fingerprint([train], [val], "sha")


# ---------------------------------------------------------------------
# The DenoisingData group: refusals
# ---------------------------------------------------------------------

BASE = {"train_dirs": ["/data/a", "/data/b"], "val_dirs": ["/data/v"]}


def problems_of(mapping: Dict[str, Any]) -> List[str]:
    """The lines ``DenoisingData.problems`` gives for ``BASE`` with ``mapping`` laid over it, keyed as a task config would."""
    return args.DenoisingData(**{**BASE, **mapping}).problems("data")


def refusal(mapping: Dict[str, Any]) -> List[str]:
    """The lines ``validate`` refuses a mapping with (the group at the top, so keys have no ``data.`` prefix)."""
    with pytest.raises(ConfigValidationError) as error:
        args.load_denoising_data_config({**BASE, **mapping})
    return error.value.errors


@pytest.mark.parametrize(
    "mapping, key, phrase",
    [
        ({"noise_composite_ratio": 0.3}, "data.noise_composite_ratio", "COMPOSITE"),
        ({"noise_kind": "multiplicative", "noise_composite_ratio": 0.3}, "data.noise_composite_ratio", "'multiplicative'"),
        ({"noise_sigma_max": 0.0}, "data.noise_sigma_max", "above noise_sigma_min"),
        ({"noise_sigma_min": 0.25}, "data.noise_sigma_max", "above noise_sigma_min"),
        ({"noise_sigma_min": 0.05}, "data.noise_sigma_min", "not above noise_sigma_max_start"),
        ({"curriculum_schedule": "exp", "noise_sigma_max_start": 0.0}, "data.noise_sigma_max_start", "cannot leave 0"),
        ({"val_dirs": ["/data/a"]}, "data.val_dirs", "same directory as"),
        ({"val_dirs": ["/data/a/inner"]}, "data.val_dirs", "lies inside or around"),
        ({"val_dirs": ["/data"]}, "data.val_dirs", "lies inside or around"),
        ({"train_dirs": ["/data/a", "/data/a/"]}, "data.train_dirs", "distinct directories"),
        ({"train_dirs": []}, "data.train_dirs", "at least one"),
        ({"val_dirs": [" "]}, "data.val_dirs", "non-empty"),
        ({"max_train_files": 1}, "data.max_train_files", "at least the 2 train_dirs"),
        ({"max_samples": 1}, "data.max_samples", "at least the 2 train_dirs"),
        ({"max_val_files": 0}, "data.max_val_files", "positive int"),
        ({"noise_sigma_min": -0.1}, "data.noise_sigma_min", "at least 0"),
        ({"noise_kind": "composite", "noise_composite_ratio": -1.0}, "data.noise_composite_ratio", "at least 0"),
        ({"curriculum_epochs": 0}, "data.curriculum_epochs", "auto or an int"),
        ({"val_seed": -1}, "data.val_seed", "at least 0"),
    ],
    ids=lambda value: str(value)[:48],
)
def test_each_refusal_names_its_key_and_says_why(mapping, key, phrase):
    lines = problems_of(mapping)
    assert any(f"'{key}'" in line and phrase in line for line in lines), lines
    # and the same line reaches the user through validate, at the top level of a group (no task prefix)
    assert any(f"'{key.removeprefix('data.')}'" in line and phrase in line for line in refusal(mapping)), refusal(mapping)


@pytest.mark.parametrize(
    "mapping, key",
    [({"noise_kind": "poisson"}, "noise_kind"), ({"curriculum_schedule": "step"}, "curriculum_schedule"),
     ({"noise_clip": "yes"}, "noise_clip"), ({"trian_dirs": ["x"]}, "trian_dirs")],
)
def test_a_value_of_the_wrong_kind_or_an_unknown_key_is_refused_by_name(mapping, key):
    assert any(f"'{key}'" in line for line in refusal(mapping))


def test_a_config_that_is_fine_has_no_problem():
    assert problems_of({}) == []


def test_every_problem_is_reported_at_once():
    lines = refusal({"noise_composite_ratio": 0.3, "noise_sigma_max": 0.0, "val_seed": -1})
    assert len(lines) >= 3


def test_a_valid_composite_config_and_an_exp_schedule_from_a_positive_start_pass():
    data = args.load_denoising_data_config(
        {**BASE, "noise_kind": "composite", "noise_composite_ratio": 0.0, "curriculum_schedule": "exp",
         "curriculum_epochs": 3})
    assert data.noise_composite_ratio == 0.0 and data.curriculum_epochs == 3
    assert args.noise_spec(data).composite_ratio == 0.0
    assert args.noise_spec(args.load_denoising_data_config(BASE)).composite_ratio == dd.DEFAULT_COMPOSITE_RATIO


def test_the_refusals_of_a_config_leave_nothing_on_disk(tmp_path, folders):
    """Every refusal made before the run directory is claimed creates no file: nothing is written for ``out``."""
    out = tmp_path / "out"
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    train, val = folders
    for mapping in ({"noise_composite_ratio": 0.3}, {"noise_sigma_max": 0.0}, {"val_dirs": [str(train)]},
                    {"train_dirs": [str(tmp_path / "missing")]}):
        mapping = {"train_dirs": [str(train)], "val_dirs": [str(val)], **mapping}
        with pytest.raises((ConfigValidationError, ConfigError)):
            args.denoising_facts(args.load_denoising_data_config(mapping))
    assert not out.exists() and sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_the_facts_list_the_folders_and_name_the_key_of_a_bad_one(folders, tmp_path):
    train, val = folders
    base = {"train_dirs": [str(train)], "val_dirs": [str(val)]}
    facts = args.denoising_facts(args.load_denoising_data_config(base))
    assert (facts.dataset, facts.num_classes, facts.image_size) == ("folder", None, None)
    shared = tmp_path / "val_with_a_train_file"
    shared.mkdir()
    os.symlink(sorted(train.glob("*.png"))[0], shared / "same.png")
    for key, mapping in (("data.val_dirs", {"val_dirs": [str(tmp_path / "missing")]}),
                         ("data.train_dirs", {"train_dirs": [str(tmp_path / "nowhere")]}),
                         ("data.train_dirs", {"train_dirs": [str(tmp_path / "train" / "empty_of_images")]}),
                         ("data.val_dirs", {"val_dirs": [str(shared)]})):
        with pytest.raises((ConfigError, ConfigValidationError)) as error:
            args.denoising_facts(args.load_denoising_data_config({**base, **mapping}))
        assert key in str(error.value) or key.removeprefix("data.") in str(error.value), (key, str(error.value))
    with pytest.raises(ConfigError, match=r"data\.val_dirs.*in both"):
        args.denoising_facts(args.load_denoising_data_config({**base, "val_dirs": [str(shared)]}))


def test_a_yaml_file_anchors_its_relative_directories_and_names_itself_in_an_error(tmp_path, folders):
    pytest.importorskip("yaml")
    (tmp_path / "conf").mkdir()
    path = tmp_path / "conf" / "data.yaml"
    path.write_text("train_dirs: [../train]\nval_dirs: [../val]\nnoise_kind: composite\n")
    data = args.load_denoising_data_config(path)
    assert data.train_dirs == [str(tmp_path / "train")] and data.val_dirs == [str(tmp_path / "val")]
    path.write_text("train_dirs: [../train]\nval_dirs: [../val]\nnoise_sigma_max: 0\n")
    with pytest.raises(ConfigValidationError) as error:
        args.load_denoising_data_config(path)
    assert all(str(path) in line for line in error.value.errors)


def test_the_curriculum_callback_needs_a_resolved_epoch_count_and_carries_the_schedule():
    data = args.load_denoising_data_config({**BASE, "curriculum_schedule": "cosine", "noise_sigma_max_start": 0.05})
    with pytest.raises(ConfigError, match="still 'auto'"):
        args.curriculum_callback(data, tf.Variable(0.05))
    variable = tf.Variable(0.05)
    callback = args.curriculum_callback(dataclasses.replace(data, curriculum_epochs=5), variable)
    assert (callback.schedule, callback.total_epochs, callback.sigma_max_start, callback.sigma_max_end) == (
        "cosine", 5, 0.05, 0.25)
    callback.on_epoch_begin(4)
    assert float(variable) == pytest.approx(0.25)
    callback.on_epoch_begin(0)
    assert float(variable) == pytest.approx(0.05)
