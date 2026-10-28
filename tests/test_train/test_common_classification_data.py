"""``train.common.classification_data``: the one ingestion path of the labelled-image trainers.

``_load_raw`` (the only ``keras.datasets`` call under ``src/train``) is replaced by a fake whose
pixel ``[0, 0, 0]`` is the sample's identity, so a split can be audited image by image; the
ImageNet path runs on the 3-class fake tree of ``tests/imagenet_fake_tree.py``.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

import dl_techniques.datasets.vision.classification as classification  # noqa: E402
import train.common.classification_data as cd  # noqa: E402
from tests.classification_fakes import use_fake_imagenette  # noqa: E402
from tests.imagenet_fake_tree import (  # noqa: E402
    PER_CLASS,
    VAL_CLASSES,
    build_fake_imagenet,
    use_the_fake_tree_class_count,
)

N_TRAIN, N_TEST = 200, 50  # identities (0..249) fit a uint8


def _fake_raw(name):
    """Samples carry their identity in pixel [0, 0, 0]; labels cycle through the classes."""
    spec = classification.get_spec(name)
    shape = spec.input_shape[:2] if name == "mnist" else spec.input_shape

    def split(start, n):
        x = np.zeros((n, *shape), np.uint8)
        ids = np.arange(start, start + n, dtype=np.uint8)
        x[(slice(None), *([0] * len(shape)))] = ids
        return x, (np.arange(start, start + n) % spec.num_classes).astype(np.int64)

    return split(0, N_TRAIN), split(N_TRAIN, N_TEST)


@pytest.fixture(autouse=True)
def fake_raw(monkeypatch):
    calls = []

    def load(name):
        calls.append(name)
        return _fake_raw(name)

    monkeypatch.setattr(cd, "_load_raw", load)
    return calls


def _ids(x: np.ndarray, std=1.0, mean=0.0) -> np.ndarray:
    """Identities back from pixel [0, 0, 0] of a (N, H, W, C) array in [0, 1] or normalized."""
    return np.rint((x[:, 0, 0, 0] * std + mean) * 255).astype(int)


def _collect(dataset):
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in dataset])
    return np.concatenate(xs), np.concatenate(ys)


def _load(name="cifar10", **kwargs):
    settings = dict(batch_size=8, seed=3, normalize="unit")
    settings.update(kwargs)
    return cd.load_classification_data(name, **settings)


# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------


def test_train_select_and_test_are_disjoint_and_cover_the_dataset() -> None:
    data = _load()
    a = data.arrays
    train, select, test = _ids(a.x_train), _ids(a.x_select), _ids(a.x_test)
    assert (data.n_train, data.n_select, data.n_test) == (len(train), len(select), len(test))
    assert len(select) == int(N_TRAIN * 0.1) and len(train) == N_TRAIN - len(select)
    assert not set(train) & set(select)
    assert set(train) | set(select) == set(range(N_TRAIN))
    assert sorted(test) == list(range(N_TRAIN, N_TRAIN + N_TEST))


def test_the_test_split_is_never_the_selection_split() -> None:
    data = _load()
    assert not set(_ids(data.arrays.x_select)) & set(_ids(data.arrays.x_test))
    _, y_sel = _collect(data.select_ds)
    assert len(y_sel) == data.n_select


def test_the_split_is_deterministic_in_the_seed() -> None:
    a, b, c = _load(seed=1), _load(seed=1), _load(seed=2)
    np.testing.assert_array_equal(a.arrays.x_select, b.arrays.x_select)
    assert not np.array_equal(_ids(a.arrays.x_select), _ids(c.arrays.x_select))


def test_max_samples_caps_the_train_pool_and_the_test_split() -> None:
    data = _load(max_samples=40)
    assert data.n_test == 40 < N_TEST
    assert data.n_select + data.n_train == 40 and data.n_select == 4
    assert _load(max_samples=500).n_test == N_TEST  # a cap above the size changes nothing


def test_a_validation_split_is_honoured() -> None:
    assert _load(validation_split=0.25).n_select == int(N_TRAIN * 0.25)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_a_validation_split_outside_the_open_unit_interval_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        _load(validation_split=bad)


def test_a_split_that_holds_out_zero_samples_is_refused() -> None:
    with pytest.raises(ValueError, match="holds out 0 of"):
        _load(validation_split=0.001, max_samples=50)


def test_a_batch_larger_than_the_train_split_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="batch_size 500 exceeds the 180 train samples"):
        _load(batch_size=500)


def test_the_train_pipeline_yields_steps_per_epoch_full_batches() -> None:
    data = _load(batch_size=16)
    batches = [x.shape[0] for x, _ in data.train_ds]
    assert data.steps_per_epoch == len(batches) == data.n_train // 16
    assert set(batches) == {16}


def test_the_only_raw_read_is_the_seam(fake_raw) -> None:
    _load("cifar100")
    assert fake_raw == ["cifar100"]


# ---------------------------------------------------------------------
# Normalization, channels, labels, augmentation order
# ---------------------------------------------------------------------


def test_standardize_uses_the_spec_constants() -> None:
    data = _load(normalize="standardize")
    spec = classification.get_spec("cifar10")
    np.testing.assert_allclose(data.mean, spec.mean, rtol=1e-6)
    np.testing.assert_allclose(data.std, spec.std, rtol=1e-6)
    x, _ = _collect(data.test_ds)
    want = (data.arrays.x_test - data.mean) / data.std
    np.testing.assert_allclose(x, want, atol=1e-5)
    np.testing.assert_allclose(data.x_fig, want, atol=1e-5)


def test_unit_leaves_the_pixels_in_zero_one_and_reports_identity_statistics() -> None:
    data = _load(normalize="unit")
    x, _ = _collect(data.test_ds)
    assert x.min() >= 0.0 and x.max() <= 1.0
    np.testing.assert_array_equal(data.mean, np.zeros(3))
    np.testing.assert_array_equal(data.std, np.ones(3))


def test_mnist_is_native_one_channel_and_can_repeat_to_three() -> None:
    one, three = _load("mnist"), _load("mnist", channels=3, normalize="standardize")
    assert one.input_shape == (28, 28, 1) and one.arrays.x_train.shape[-1] == 1
    assert three.input_shape == (28, 28, 3) and three.arrays.x_train.shape[-1] == 3
    spec = classification.get_spec("mnist")
    np.testing.assert_allclose(three.mean, [spec.mean[0]] * 3, rtol=1e-6)
    x, _ = _collect(three.test_ds)
    np.testing.assert_allclose(x[..., 0], x[..., 2])


@pytest.mark.parametrize("name, channels", [("mnist", 2), ("cifar10", 1), ("cifar100", 4)])
def test_a_channel_count_the_dataset_cannot_give_is_refused(name, channels) -> None:
    with pytest.raises(ValueError, match="is not available"):
        _load(name, channels=channels)


def test_one_hot_labels_reach_the_arrays_and_the_pipelines() -> None:
    data = _load(labels="one_hot")
    assert data.arrays.y_train.shape == (data.n_train, 10)
    assert (data.arrays.y_train.sum(axis=1) == 1).all()
    _, y = _collect(data.train_ds)
    assert y.shape[1] == 10 and (y.sum(axis=1) == 1).all()
    assert data.y_fig.shape == (data.n_test, 10)


def test_sparse_labels_are_int32_vectors() -> None:
    data = _load()
    assert data.arrays.y_train.dtype == np.int32 and data.arrays.y_train.ndim == 1
    assert data.class_names == list(classification.CIFAR10_CLASS_NAMES)


def test_the_augment_hook_sees_unit_pixels_before_normalization() -> None:
    """The hook turns every pixel into 0.5; normalization must then give (0.5 - mean) / std."""
    data = _load(normalize="standardize", augment=lambda image, label: (image * 0.0 + 0.5, label))
    x, _ = next(iter(data.train_ds))
    want = (0.5 - data.mean) / data.std
    np.testing.assert_allclose(x.numpy()[:, 3, 3, :], np.tile(want, (x.shape[0], 1)), atol=1e-5)


def test_flip_pad_crop_keeps_the_shape_and_only_moves_pixels() -> None:
    image = tf.reshape(tf.range(1, 4 * 4 * 3 + 1, dtype=tf.float32), (4, 4, 3))
    out, label = cd.flip_pad_crop(pad=2)(image, tf.constant(7))
    assert out.shape == image.shape and int(label) == 7
    assert set(np.unique(out.numpy())) <= set(np.unique(image.numpy())) | {0.0}


@pytest.mark.parametrize("kwargs, message", [
    (dict(normalize="whiten"), "normalize must be one of"),
    (dict(labels="soft"), "labels must be one of"),
    (dict(holdout_per_class=5), "holdout_per_class=5 does not apply"),
])
def test_an_option_that_does_not_apply_is_refused_naming_it(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _load(**kwargs)


def test_an_unknown_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="got 'svhn'"):
        _load("svhn")


# ---------------------------------------------------------------------
# ImageNet, on the fake tree (3 classes, 12 train images, 6 val)
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")


@pytest.fixture
def three_class_imagenet(monkeypatch):
    use_the_fake_tree_class_count(monkeypatch)


def _imagenet(tree, **kwargs):
    settings = dict(batch_size=4, seed=3, holdout_per_class=1, image_size=32,
                    imagenet_root=tree, eval_samples=4)
    settings.update(kwargs)
    return cd.load_classification_data("imagenet", **settings)


def test_imagenet_selects_on_a_holdout_and_tests_on_the_official_val(tree, three_class_imagenet):
    data = _imagenet(tree)
    assert (data.n_train, data.n_select, data.n_test) == (3 * PER_CLASS - 3, 3, len(VAL_CLASSES))
    assert data.arrays is None and data.input_shape == (32, 32, 3) and data.num_classes == 3
    assert data.selection_split == "1 images per class held out of train (3 images)"
    assert data.test_split == "the official val split"
    assert data.steps_per_epoch == data.n_train // 4
    assert data.x_fig.shape == (4, 32, 32, 3) and data.y_fig.tolist() == VAL_CLASSES[:4]


def test_imagenet_labels_and_unit_pixels(tree, three_class_imagenet):
    data = _imagenet(tree, labels="one_hot", normalize="unit")
    x, y = _collect(data.test_ds)
    assert y.shape == (len(VAL_CLASSES), 3) and (y.sum(axis=1) == 1).all()
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert data.y_fig.shape == (4, 3)
    np.testing.assert_array_equal(data.mean, np.zeros(3))


def test_imagenet_standardizes_with_the_spec_statistics(tree, three_class_imagenet):
    data = _imagenet(tree)
    spec = classification.get_spec("imagenet")
    np.testing.assert_allclose(data.mean, spec.mean, rtol=1e-6)
    np.testing.assert_allclose(data.std, spec.std, rtol=1e-6)


def test_imagenet_refuses_a_validation_split_and_an_in_memory_augment(tree, three_class_imagenet):
    with pytest.raises(ValueError, match="validation_split=0.1 does not apply to ImageNet"):
        _imagenet(tree, validation_split=0.1)
    with pytest.raises(ValueError, match="augment applies to in-memory datasets"):
        _imagenet(tree, augment=cd.flip_pad_crop())
    with pytest.raises(ValueError, match="channels=1 is not available for 'imagenet'"):
        _imagenet(tree, channels=1)


def test_a_missing_imagenet_root_is_refused_before_anything_loads(tmp_path, three_class_imagenet):
    with pytest.raises(ValueError, match="ImageNet root does not exist"):
        _imagenet(str(tmp_path / "nope"))


def test_the_class_count_of_the_spec_is_checked_against_the_tree(tree):
    with pytest.raises(ValueError, match="num_classes=1000 but"):
        _imagenet(tree)


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------


def test_the_selection_flags_default_to_none_so_misuse_is_refused() -> None:
    parser = argparse.ArgumentParser()
    cd.add_selection_arguments(parser)
    args = parser.parse_args([])
    assert args.validation_split is None and args.holdout_per_class is None
    assert args.max_samples is None and args.eval_samples == cd.DEFAULT_EVAL_SAMPLES
    args = parser.parse_args(["--validation-split", "0.2", "--holdout-per-class", "4",
                              "--max-samples", "9", "--imagenet-root", "/x"])
    assert (args.validation_split, args.holdout_per_class, args.max_samples,
            args.imagenet_root) == (0.2, 4, 9, "/x")


def test_the_loader_seam_refuses_a_dataset_it_cannot_read(monkeypatch) -> None:
    monkeypatch.undo()
    with pytest.raises(ValueError, match="no in-memory loader for dataset 'imagenet'"):
        cd._load_raw("imagenet")


# ---------------------------------------------------------------------
# load_split_arrays: the same split with no pipeline
# ---------------------------------------------------------------------


def test_the_split_arrays_are_the_loaders_split() -> None:
    arrays = cd.load_split_arrays("cifar10", seed=3)
    data = _load(seed=3)
    for name in ("x_train", "x_select", "x_test", "index_train", "index_select", "index_test"):
        np.testing.assert_array_equal(getattr(arrays, name), getattr(data.arrays, name))


def test_the_split_arrays_need_no_batch_size_and_carry_the_metadata() -> None:
    arrays = cd.load_split_arrays("cifar100", seed=0, labels="one_hot")
    assert arrays.input_shape == (32, 32, 3) and arrays.num_classes == 100
    assert arrays.class_names[0] == "apple" and arrays.y_train.shape[1] == 100


def test_the_row_indices_align_a_second_label_set_with_the_split() -> None:
    """The indices are rows of the Keras arrays: pixel [0, 0, 0] is the row's identity."""
    arrays = cd.load_split_arrays("cifar10", seed=5)
    np.testing.assert_array_equal(_ids(arrays.x_train), arrays.index_train)
    np.testing.assert_array_equal(_ids(arrays.x_select), arrays.index_select)
    np.testing.assert_array_equal(_ids(arrays.x_test), arrays.index_test + N_TRAIN)


def test_split_arrays_refuse_imagenet_and_a_bad_split() -> None:
    with pytest.raises(ValueError, match="not an in-memory dataset"):
        cd.load_split_arrays("imagenet", seed=0)
    with pytest.raises(ValueError, match="strictly inside"):
        cd.load_split_arrays("mnist", seed=0, validation_split=0.0)
    with pytest.raises(ValueError, match="labels must be one of"):
        cd.load_split_arrays("mnist", seed=0, labels="soft")


# ---------------------------------------------------------------------
# resize, eval_samples=0, colour augmentation
# ---------------------------------------------------------------------


def test_resize_upsamples_every_pipeline_and_reports_the_resized_shape() -> None:
    data = _load(normalize="unit", resize=48)
    assert data.input_shape == (48, 48, 3)
    for dataset in (data.train_ds, data.select_ds, data.test_ds):
        assert next(iter(dataset))[0].shape[1:] == (48, 48, 3)
    # the arrays and the figure subset keep the native resolution
    assert data.arrays.x_train.shape[1:] == (32, 32, 3) and data.x_fig.shape[1:] == (32, 32, 3)


def test_resize_is_refused_for_imagenet(tree, three_class_imagenet) -> None:
    with pytest.raises(ValueError, match="resize applies to in-memory datasets"):
        _imagenet(tree, resize=64)


def test_resize_happens_before_the_hook_and_commutes_with_standardizing() -> None:
    seen = {}

    def hook(image, label):
        seen["shape"] = tuple(image.shape)
        return image, label

    _load(resize=40, augment=hook)
    assert seen["shape"] == (40, 40, 3)


def test_eval_samples_zero_keeps_no_figure_images(tree, three_class_imagenet) -> None:
    data = _imagenet(tree, eval_samples=0)
    assert data.x_fig.shape == (0, 32, 32, 3) and data.y_fig.shape == (0,)
    assert data.n_test == len(VAL_CLASSES)


def test_imagenet_colour_augmentation_reaches_the_loader(tree, three_class_imagenet, monkeypatch) -> None:
    seen = {}
    real = cd.load_imagenet_selection_splits

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cd, "load_imagenet_selection_splits", spy)
    _imagenet(tree, imagenet_augment_color=True)
    assert seen["augment_color"] is True and seen["augment"] is True
    _imagenet(tree, imagenet_augment=False)
    assert seen["augment"] is False and seen["augment_color"] is False


def test_the_selection_flags_can_be_trimmed_to_what_a_trainer_uses() -> None:
    def parse(**kwargs):
        parser = argparse.ArgumentParser()
        cd.add_selection_arguments(parser, **kwargs)
        return {a.dest for a in parser._actions} - {"help"}

    full = parse()
    assert {"validation_split", "max_samples", "holdout_per_class", "imagenet_root",
            "devkit_dir", "eval_samples"} <= full
    assert parse(imagenet=False) == {"validation_split", "max_samples"}
    assert "validation_split" not in parse(in_memory=False)
    assert "eval_samples" not in parse(figures=False)


# ---------------------------------------------------------------------
# Imagenette, on the fake record seam (60 train jpegs, 20 validation jpegs)
# ---------------------------------------------------------------------


@pytest.fixture
def imagenette_records(monkeypatch):
    return use_fake_imagenette(monkeypatch)


def _imagenette(**kwargs):
    settings = dict(batch_size=6, seed=3, image_size=24, normalize="unit", eval_samples=4)
    settings.update(kwargs)
    return cd.load_classification_data("imagenette", **settings)


def test_imagenette_selects_on_a_seeded_slice_of_train_and_tests_on_the_validation_records(
        imagenette_records) -> None:
    data = _imagenette()
    assert (data.n_train, data.n_select, data.n_test) == (54, 6, 20)
    assert data.steps_per_epoch == 9 and data.arrays is None
    assert data.input_shape == (24, 24, 3) and data.num_classes == 10
    assert data.selection_split == "validation_split=0.1 of the train records (seeded permutation)"
    assert data.test_split == "the labelled validation split"
    assert data.class_names[:2] == ["tench", "English springer"]
    assert data.x_fig.shape == (4, 24, 24, 3) and data.y_fig.shape == (4,)


def test_imagenette_honours_the_split_the_cap_and_the_seed(imagenette_records) -> None:
    assert _imagenette(validation_split=0.25, batch_size=3).n_select == 15
    capped = _imagenette(max_samples=15, batch_size=3)
    assert (capped.n_train, capped.n_select, capped.n_test) == (14, 1, 15)
    first = next(iter(_imagenette(seed=1).select_ds))[0].numpy()
    again = next(iter(_imagenette(seed=1).select_ds))[0].numpy()
    other = next(iter(_imagenette(seed=2).select_ds))[0].numpy()
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_imagenette_labels_normalization_and_figures(imagenette_records) -> None:
    data = _imagenette(labels="one_hot", normalize="standardize", eval_samples=0)
    x, y = next(iter(data.train_ds))
    assert y.shape == (6, 10) and (y.numpy().sum(axis=1) == 1).all()
    assert x.numpy().min() < 0.0 or x.numpy().max() > 1.0, "standardized, not unit"
    assert data.x_fig.shape == (0, 24, 24, 3) and data.y_fig.shape == (0, 10)
    np.testing.assert_allclose(data.mean, classification.get_spec("imagenette").mean, rtol=1e-6)


def test_imagenette_passes_the_crop_policy_and_the_augmentation_switch(
        imagenette_records, monkeypatch) -> None:
    seen = []
    real = cd.load_imagenette

    def spy(split, config):
        seen.append((split, config.crop_policy, config.augment))
        return real(split, config)

    monkeypatch.setattr(cd, "load_imagenette", spy)
    _imagenette(crop_policy="resize", imagenet_augment=False)
    assert {(policy, augment) for _, policy, augment in seen} == {("resize", False)}
    seen.clear()
    _imagenette()
    assert {(policy, augment) for _, policy, augment in seen} == {("random_resized_crop", True)}


@pytest.mark.parametrize("kwargs, message", [
    (dict(holdout_per_class=5), "holdout_per_class=5 does not apply to 'imagenette'"),
    (dict(imagenet_augment_color=True), "imagenet_augment_color does not apply to 'imagenette'"),
    (dict(augment=cd.flip_pad_crop()), "augment applies to in-memory datasets"),
    (dict(resize=64), "resize applies to in-memory datasets"),
    (dict(channels=1), "channels=1 is not available for 'imagenette'"),
    (dict(validation_split=0.0), "strictly inside"),
    (dict(crop_policy="centre"), "Unknown crop_policy"),
])
def test_imagenette_refuses_an_option_that_does_not_apply_naming_it(
        imagenette_records, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _imagenette(**kwargs)


def test_imagenette_reads_only_the_two_record_splits(imagenette_records) -> None:
    _imagenette()
    assert set(imagenette_records) == {"train", "validation"}


def test_a_trainer_can_use_the_spec_to_size_its_head() -> None:
    spec = classification.get_spec("imagenette")
    assert (spec.num_classes, spec.train_size, spec.source, spec.in_memory) == (
        10, 9469, "records", False)


# ---------------------------------------------------------------------
# select_steps, capped_steps and the shared unbatched example stage
# ---------------------------------------------------------------------


def test_select_steps_is_the_real_batch_count_of_select_ds(monkeypatch) -> None:
    """An in-memory selection split is batched at EVAL_BATCH_SIZE, not at the trainer's batch
    size, so a trainer must not derive its validation steps from ``batch_size``."""
    from tests.classification_fakes import use_tiny_raw

    use_tiny_raw(monkeypatch, n_train=3000, n_test=10)
    data = cd.load_classification_data(
        "cifar10", batch_size=8, seed=0, eval_samples=0)
    assert data.n_select == 300
    batches = sum(1 for _ in data.select_ds)
    assert data.select_steps == batches == -(-300 // cd.EVAL_BATCH_SIZE)
    assert data.select_steps != -(-data.n_select // 8), "derived from the trainer's batch size"


def test_select_steps_of_a_file_based_dataset_follows_the_trainer_batch(monkeypatch) -> None:
    use_fake_imagenette(monkeypatch)
    data = cd.load_classification_data(
        "imagenette", batch_size=4, seed=0, image_size=24, normalize="unit", eval_samples=0)
    assert data.select_steps == sum(1 for _ in data.select_ds) == -(-data.n_select // 4)


@pytest.mark.parametrize("steps, select, cap, expected", [
    (13, 2, None, (13, 2)),
    (13, 2, 1, (1, 1)),
    (13, 2, 5, (5, 2)),
    (13, 7, 5, (5, 5)),
])
def test_capped_steps_clamps_each_count_by_max_steps(steps, select, cap, expected) -> None:
    assert cd.capped_steps(steps, select, cap) == expected


def test_array_examples_trains_shuffled_resized_augmented_normalized_and_evaluates_in_order() -> None:
    x = np.stack([np.full((8, 8, 3), i / 255.0, np.float32) for i in range(10)])
    y = np.arange(10, dtype=np.int32)
    calls = []

    def augment(image, label):
        calls.append("augment")
        return image + 0.0, label

    kwargs = dict(seed=3, normalize=lambda image: image * 2.0, resize=cd.resizer(4))
    evaluation = list(cd.array_examples(x, y, training=False, augment=augment, **kwargs))
    assert [int(label) for _, label in evaluation] == list(range(10)), "evaluation is in order"
    assert not calls, "evaluation must not augment"
    assert evaluation[0][0].shape == (4, 4, 3)
    np.testing.assert_allclose(evaluation[7][0].numpy(), 2.0 * 7 / 255.0, atol=1e-6)

    training = [int(label) for _, label in
                cd.array_examples(x, y, training=True, augment=augment, **kwargs)]
    assert sorted(training) == list(range(10)) and training != list(range(10))
    assert calls, "training augments"


def test_flip_pad_crop_with_a_seed_is_reproducible_and_without_one_keeps_its_signature() -> None:
    image = tf.reshape(tf.range(8 * 8 * 3, dtype=tf.float32), (8, 8, 3))

    def run(seed):
        # Op-level seeds are deterministic per graph, so build the pipeline twice.
        ds = tf.data.Dataset.from_tensors((image, 0)).repeat(4).map(
            cd.flip_pad_crop(mode="REFLECT", seed=seed))
        return np.stack([img.numpy() for img, _ in ds])

    np.testing.assert_array_equal(run(5), run(5))
    out, label = cd.flip_pad_crop()(image, 7)
    assert out.shape == image.shape and int(label) == 7
