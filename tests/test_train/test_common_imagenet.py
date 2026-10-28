"""Tests for ``train.common.imagenet``, the ImageNet wiring the trainers share."""

import argparse

import numpy as np
import pytest

import train.common as common
from dl_techniques.datasets.vision.imagenet2012 import (
    IMAGENET2012_DEFAULT_ROOT,
    ImageNet2012Config,
    load_imagenet2012,
)
from tests.imagenet_fake_tree import PER_CLASS, VAL_CLASSES, WNIDS, build_fake_imagenet
from train.common.imagenet import (
    DEFAULT_HOLDOUT_PER_CLASS,
    add_holdout_argument,
    add_imagenet_arguments,
    eval_head,
    load_imagenet_selection_splits,
    require_imagenet_root,
)


@pytest.fixture
def root(tmp_path):
    return build_fake_imagenet(tmp_path / "imagenet2012")


class TestArguments:
    def test_defaults(self):
        parser = argparse.ArgumentParser()
        add_imagenet_arguments(parser)
        args = parser.parse_args([])
        assert args.imagenet_root == IMAGENET2012_DEFAULT_ROOT
        assert args.devkit_dir is None

    def test_both_flags_are_read(self):
        parser = argparse.ArgumentParser()
        add_imagenet_arguments(parser)
        args = parser.parse_args(["--imagenet-root", "/a", "--devkit-dir", "/b"])
        assert (args.imagenet_root, args.devkit_dir) == ("/a", "/b")

    def test_the_note_prefixes_both_help_strings(self):
        parser = argparse.ArgumentParser()
        add_imagenet_arguments(parser, note="Used only when X: ")
        helps = [a.help for a in parser._actions if a.dest in ("imagenet_root", "devkit_dir")]
        assert len(helps) == 2 and all(h.startswith("Used only when X: ") for h in helps)

    def test_the_names_are_exported_from_the_hub(self):
        for name in ("add_imagenet_arguments", "require_imagenet_root", "add_holdout_argument",
                     "load_imagenet_selection_splits", "eval_head",
                     "DEFAULT_HOLDOUT_PER_CLASS"):
            assert name in common.__all__ and hasattr(common, name)


class TestRoot:
    def test_a_missing_root_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="ImageNet root does not exist"):
            require_imagenet_root(str(tmp_path / "nope"))

    def test_an_existing_root_passes(self, tmp_path):
        require_imagenet_root(str(tmp_path))


class TestEvalHead:
    def test_size_dtype_and_label_order(self, root):
        _, _, val = _selection(root)
        x, y = eval_head(val, 4)
        assert x.shape == (4, 32, 32, 3) and x.dtype == np.float32
        assert y.tolist() == VAL_CLASSES[:4]

    def test_it_is_deterministic(self, root):
        _, _, val = _selection(root, augment=True)
        x1, _ = eval_head(val, 6)
        x2, _ = eval_head(val, 6)
        assert np.array_equal(x1, x2)

    def test_n_beyond_val_returns_all_of_it(self, root):
        _, _, val = _selection(root)
        x, y = eval_head(val, 1000)
        assert len(x) == len(y) == len(VAL_CLASSES)

    def test_a_non_positive_n_is_refused(self, root):
        _, _, val = _selection(root)
        with pytest.raises(ValueError, match="positive"):
            eval_head(val, 0)


def _selection(root, **kw):
    kw.setdefault("devkit_dir", None)
    kw.setdefault("image_size", 32)
    kw.setdefault("batch_size", 4)
    kw.setdefault("seed", 0)
    kw.setdefault("num_parallel_calls", 1)
    kw.setdefault("augment", False)
    kw.setdefault("holdout_per_class", 1)
    return load_imagenet_selection_splits(root, **kw)


def _pixels(data):
    return np.concatenate([x.numpy() for x, _ in data.dataset])


class TestHoldoutArgument:
    def test_default_and_override(self):
        parser = argparse.ArgumentParser()
        add_holdout_argument(parser)
        assert parser.parse_args([]).holdout_per_class == DEFAULT_HOLDOUT_PER_CLASS == 10
        assert parser.parse_args(["--holdout-per-class", "3"]).holdout_per_class == 3


class TestSelectionSplits:
    def test_the_three_sizes(self, root):
        train, select, test = _selection(root)
        assert train.num_samples == 3 * (PER_CLASS - 1)
        assert select.num_samples == 3
        assert test.num_samples == len(VAL_CLASSES)

    def test_test_is_the_official_val_split(self, root):
        _, _, test = _selection(root)
        val = load_imagenet2012("val", ImageNet2012Config(
            root=root, image_size=32, batch_size=4, num_parallel_calls=1))
        assert np.array_equal(_pixels(test), _pixels(val))

    def test_selection_images_are_not_in_train(self, root):
        train, select, _ = _selection(root, normalize="none")
        rows = lambda d: {r.tobytes() for r in _pixels(d)}
        assert not rows(train) & rows(select)

    def test_train_drops_the_remainder_and_the_others_keep_every_image(self, root):
        train, select, test = _selection(root, batch_size=5)
        assert train.steps_per_epoch == (3 * (PER_CLASS - 1)) // 5
        assert sum(len(y) for _, y in select.dataset) == 3
        assert sum(len(y) for _, y in test.dataset) == len(VAL_CLASSES)

    def test_selection_is_never_augmented(self, root):
        _, select, _ = _selection(root, augment=True, seed=1)
        _, select2, _ = _selection(root, augment=True, seed=2)
        assert np.array_equal(_pixels(select), _pixels(select2))

    def test_normalize_reaches_every_split(self, root):
        for data in _selection(root, normalize="unit"):
            x = _pixels(data)
            assert x.min() >= 0.0 and x.max() <= 1.0
        for data in _selection(root, normalize="imagenet"):
            assert _pixels(data).min() < 0.0

    def test_max_images_caps_train_and_test_but_not_selection(self, root):
        train, select, test = _selection(root, max_images=2, batch_size=1)
        assert (train.num_samples, select.num_samples, test.num_samples) == (2, 3, 2)

    def test_a_holdout_of_zero_is_refused(self, root):
        with pytest.raises(ValueError, match="holdout_per_class must be positive.*got 0"):
            _selection(root, holdout_per_class=0)

    def test_class_count_is_checked(self, root):
        _selection(root, num_classes=3)
        with pytest.raises(ValueError, match="num_classes=1000 but .* has 3 classes"):
            _selection(root, num_classes=1000)

    def test_color_augmentation_reaches_train_only(self, root):
        plain, select, _ = _selection(root, augment=True, augment_color=False, batch_size=9)
        colour, select_c, _ = _selection(root, augment=True, augment_color=True, batch_size=9)
        a = next(iter(plain.dataset))[0].numpy()
        b = next(iter(colour.dataset))[0].numpy()
        assert not np.array_equal(a, b)
        assert np.array_equal(_pixels(select), _pixels(select_c))
