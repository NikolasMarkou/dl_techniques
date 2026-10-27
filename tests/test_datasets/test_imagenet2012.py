"""Tests for ``dl_techniques.datasets.vision.imagenet2012``.

A tiny fake ImageNet tree is written into ``tmp_path``; nothing here touches
the real dataset except the final, skip-gated census.
"""

import os

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision import imagenet2012 as m
from dl_techniques.datasets.vision.imagenet2012 import (
    IMAGENET2012_DEFAULT_ROOT,
    ImageNet2012Config,
    load_imagenet2012,
)
from tests.imagenet_fake_tree import (
    PER_CLASS,
    VAL_CLASSES,
    WNIDS,
    build_fake_imagenet,
    jpeg as _jpeg,
    make_devkit as _make_devkit,
)


@pytest.fixture
def root(tmp_path):
    return build_fake_imagenet(tmp_path / "imagenet2012")


def _cfg(root, **kw):
    kw.setdefault("image_size", 32)
    kw.setdefault("batch_size", 4)
    return ImageNet2012Config(root=root, **kw)


def _all(data):
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in data.dataset])
    return np.concatenate(xs), np.concatenate(ys)


class TestTrain:
    def test_classes_are_sorted_wnids_and_labels_follow(self, root):
        data = load_imagenet2012("train", _cfg(root, shuffle=False, augment=False))
        assert data.class_names == WNIDS
        assert data.num_classes == 3
        assert data.num_samples == 3 * PER_CLASS
        _, y = _all(data)
        assert y.tolist() == [c for c in range(3) for _ in range(PER_CLASS)]

    def test_batch_shape_dtype_and_steps(self, root):
        data = load_imagenet2012("train", _cfg(root, batch_size=5))
        x, y = next(iter(data.dataset))
        assert x.shape == (5, 32, 32, 3) and x.dtype == tf.float32
        assert y.dtype == tf.int32
        assert data.steps_per_epoch == 3 and sum(1 for _ in data.dataset) == 3

    def test_cardinality_is_known_and_equals_steps_per_epoch(self, root):
        # An unknown cardinality makes Keras log a phantom zero batch per epoch.
        for kw in (dict(), dict(augment_color=True), dict(drop_remainder=True)):
            for split in ("train", "val"):
                data = load_imagenet2012(split, _cfg(root, batch_size=5, **kw))
                n = int(tf.data.experimental.cardinality(data.dataset).numpy())
                assert n == data.steps_per_epoch == sum(1 for _ in data.dataset)

    def test_drop_remainder(self, root):
        data = load_imagenet2012("train", _cfg(root, batch_size=5, drop_remainder=True))
        assert data.steps_per_epoch == 2 and sum(1 for _ in data.dataset) == 2

    def test_max_images_is_a_seeded_spread_not_a_prefix(self, root):
        a = load_imagenet2012("train", _cfg(root, max_images=6, shuffle=False, augment=False))
        b = load_imagenet2012("train", _cfg(root, max_images=6, shuffle=False, augment=False))
        _, ya = _all(a)
        _, yb = _all(b)
        assert a.num_samples == 6 and ya.tolist() == yb.tolist()
        assert len(set(ya.tolist())) > 1  # a prefix of 6 would stop inside class 1

    def test_same_seed_reproduces_augmentation_and_epochs_differ(self, root):
        cfg = _cfg(root, shuffle=False, augment=True, seed=7)
        x1, _ = _all(load_imagenet2012("train", cfg))
        x2, _ = _all(load_imagenet2012("train", cfg))
        np.testing.assert_array_equal(x1, x2)
        data = load_imagenet2012("train", cfg)
        e1, _ = _all(data)
        e2, _ = _all(data)
        np.testing.assert_array_equal(e1, x1)
        assert not np.array_equal(e1, e2), "every epoch reused one augmentation"

    def test_train_works_without_devkit(self, root):
        cfg = _cfg(root, devkit_dir=os.path.join(root, "nope"))
        assert load_imagenet2012("train", cfg).num_samples == 3 * PER_CLASS


class TestColorAugmentation:
    def test_color_changes_pixels_but_not_geometry_and_is_reproducible(self, root):
        base = dict(shuffle=False, augment=True, normalize="unit", seed=5)
        plain, _ = _all(load_imagenet2012("train", _cfg(root, **base)))
        col1, _ = _all(load_imagenet2012("train", _cfg(root, augment_color=True, **base)))
        col2, _ = _all(load_imagenet2012("train", _cfg(root, augment_color=True, **base)))
        assert plain.shape == col1.shape
        assert not np.array_equal(plain, col1), "augment_color changed nothing"
        np.testing.assert_array_equal(col1, col2)
        assert col1.min() >= 0.0 and col1.max() <= 1.0

    def test_color_is_ignored_for_val_and_without_augment(self, root):
        val_a, _ = _all(load_imagenet2012("val", _cfg(root)))
        val_b, _ = _all(load_imagenet2012("val", _cfg(root, augment_color=True)))
        np.testing.assert_array_equal(val_a, val_b)
        off = dict(shuffle=False, augment=False)
        a, _ = _all(load_imagenet2012("train", _cfg(root, **off)))
        b, _ = _all(load_imagenet2012("train", _cfg(root, augment_color=True, **off)))
        np.testing.assert_array_equal(a, b)


class TestNormalize:
    def test_modes(self, root):
        base = dict(shuffle=False, augment=False, max_images=1)
        raw, _ = _all(load_imagenet2012("train", _cfg(root, normalize="none", **base)))
        unit, _ = _all(load_imagenet2012("train", _cfg(root, normalize="unit", **base)))
        imn, _ = _all(load_imagenet2012("train", _cfg(root, normalize="imagenet", **base)))
        np.testing.assert_allclose(unit, raw / 255.0, atol=1e-5, rtol=0)
        mean = np.array(m.IMAGENET_MEAN, np.float32)
        std = np.array(m.IMAGENET_STD, np.float32)
        np.testing.assert_allclose(imn, (unit - mean) / std, atol=1e-5, rtol=0)

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError, match="normalize"):
            ImageNet2012Config(normalize="zscore")


class TestVal:
    def test_labels_go_through_ilsvrc_id_then_wnid(self, root):
        data = load_imagenet2012("val", _cfg(root, batch_size=6))
        assert data.num_samples == len(VAL_CLASSES)
        _, y = _all(data)
        assert y.tolist() == VAL_CLASSES

    def test_devkit_data_subfolder_is_found(self, root, tmp_path):
        sub = tmp_path / "kit" / "data"
        _make_devkit(str(sub))
        data = load_imagenet2012("val", _cfg(root, devkit_dir=str(tmp_path / "kit")))
        assert _all(data)[1].tolist() == VAL_CLASSES

    def test_missing_devkit_names_what_is_needed(self, root):
        cfg = _cfg(root, devkit_dir=os.path.join(root, "nope"))
        with pytest.raises(FileNotFoundError, match="ILSVRC2012_devkit_t12"):
            load_imagenet2012("val", cfg)

    def test_ground_truth_shorter_than_val_is_refused(self, root):
        gt = os.path.join(root, "devkit", "ILSVRC2012_validation_ground_truth.txt")
        with open(gt, "w") as f:
            f.write("1\n2\n")
        with pytest.raises(ValueError, match="ground truth"):
            load_imagenet2012("val", _cfg(root))

    def test_eval_is_deterministic_and_unshuffled(self, root):
        cfg = _cfg(root, shuffle=True, augment=True)  # both ignored for val
        a, _ = _all(load_imagenet2012("val", cfg))
        b, _ = _all(load_imagenet2012("val", cfg))
        np.testing.assert_array_equal(a, b)


class TestDecode:
    # Regression coverage, not a guard: tf.io.decode_jpeg also accepts PNG
    # bytes in TF 2.18, so reverting to it does not fail this test.
    def test_png_bytes_under_a_jpeg_name_decode(self, root):
        victim = os.path.join(root, "train", WNIDS[0], f"{WNIDS[0]}_0.JPEG")
        png = tf.io.encode_png(tf.zeros([30, 30, 3], tf.uint8))
        tf.io.write_file(victim, png)
        data = load_imagenet2012("train", _cfg(root, shuffle=False, augment=False))
        x, _ = _all(data)
        assert x.shape[0] == 3 * PER_CLASS and np.isfinite(x).all()


class TestIndexCache:
    def test_cache_written_and_reused(self, root):
        classes = m._list_classes(os.path.join(root, "train"))
        first = m._load_or_build_index(root, "train", classes)
        cache = os.path.join(root, ".index", "train.txt")
        assert os.path.isfile(cache)
        monkey_calls = []
        orig = m._scan
        m._scan = lambda *a, **k: monkey_calls.append(1) or orig(*a, **k)
        try:
            second = m._load_or_build_index(root, "train", classes)
        finally:
            m._scan = orig
        assert second == first and monkey_calls == []

    def test_cache_invalidated_when_a_file_is_added(self, root):
        classes = m._list_classes(os.path.join(root, "train"))
        before = m._load_or_build_index(root, "train", classes)
        _jpeg(os.path.join(root, "train", WNIDS[1], "extra.JPEG"), 20, 20, 9)
        after = m._load_or_build_index(root, "train", classes)
        assert len(after) == len(before) + 1

    def test_unwritable_cache_is_not_fatal(self, root):
        os.makedirs(os.path.join(root, ".index"))
        os.chmod(os.path.join(root, ".index"), 0o500)
        try:
            data = load_imagenet2012("train", _cfg(root))
        finally:
            os.chmod(os.path.join(root, ".index"), 0o700)
        assert data.num_samples == 3 * PER_CLASS


class TestErrors:
    def test_bad_split(self, root):
        with pytest.raises(ValueError, match="split"):
            load_imagenet2012("test", _cfg(root))

    def test_missing_root(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train"):
            load_imagenet2012("train", _cfg(str(tmp_path / "absent")))


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(IMAGENET2012_DEFAULT_ROOT, "train")),
    reason="real ImageNet not on this machine",
)
def test_real_train_census():
    data = load_imagenet2012("train", ImageNet2012Config(batch_size=8, max_images=16))
    assert data.num_classes == 1000
    assert data.class_names[0] == "n01440764"
    full = m._load_or_build_index(
        IMAGENET2012_DEFAULT_ROOT, "train", data.class_names
    )
    assert len(full) == 1_281_167


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(IMAGENET2012_DEFAULT_ROOT, "devkit", "meta.mat")),
    reason="real ImageNet devkit not on this machine",
)
def test_real_val_labels_are_fifty_per_class():
    classes = m._list_classes(os.path.join(IMAGENET2012_DEFAULT_ROOT, "train"))
    files = m._load_or_build_index(IMAGENET2012_DEFAULT_ROOT, "val", classes)
    labels = m._read_val_labels(
        os.path.join(IMAGENET2012_DEFAULT_ROOT, "devkit"), classes, files
    )
    assert len(files) == 50_000
    assert np.bincount(labels, minlength=1000).tolist() == [50] * 1000
    assert classes[labels[0]] == "n01751748"  # val image #1 is a sea snake
