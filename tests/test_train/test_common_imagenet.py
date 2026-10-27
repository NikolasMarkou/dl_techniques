"""Tests for ``train.common.imagenet``, the ImageNet wiring the trainers share."""

import argparse

import numpy as np
import pytest

import train.common as common
from dl_techniques.datasets.vision.imagenet2012 import IMAGENET2012_DEFAULT_ROOT
from tests.imagenet_fake_tree import PER_CLASS, VAL_CLASSES, WNIDS, build_fake_imagenet
from train.common.imagenet import (
    add_imagenet_arguments,
    eval_head,
    load_imagenet_splits,
    require_imagenet_root,
)


@pytest.fixture
def root(tmp_path):
    return build_fake_imagenet(tmp_path / "imagenet2012")


def _splits(root, **kw):
    kw.setdefault("devkit_dir", None)
    kw.setdefault("image_size", 32)
    kw.setdefault("batch_size", 4)
    kw.setdefault("seed", 0)
    kw.setdefault("num_parallel_calls", 1)
    kw.setdefault("augment", False)
    return load_imagenet_splits(root, **kw)


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
        for name in ("add_imagenet_arguments", "require_imagenet_root",
                     "load_imagenet_splits", "eval_head"):
            assert name in common.__all__ and hasattr(common, name)


class TestRoot:
    def test_a_missing_root_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="ImageNet root does not exist"):
            require_imagenet_root(str(tmp_path / "nope"))

    def test_an_existing_root_passes(self, tmp_path):
        require_imagenet_root(str(tmp_path))


class TestSplits:
    def test_train_drops_the_remainder_and_val_keeps_every_image(self, root):
        # 12 train images at batch 4 divide evenly, so use 5 to leave a remainder.
        train, val = _splits(root, batch_size=5)
        assert train.num_samples == 3 * PER_CLASS
        assert train.steps_per_epoch == (3 * PER_CLASS) // 5
        assert sum(len(y) for _, y in val.dataset) == len(VAL_CLASSES)
        assert val.steps_per_epoch == -(-len(VAL_CLASSES) // 5)

    def test_class_count_is_checked(self, root):
        _splits(root, num_classes=3)
        with pytest.raises(ValueError, match="num_classes=1000 but .* has 3 classes"):
            _splits(root, num_classes=1000)

    def test_without_num_classes_nothing_is_checked(self, root):
        assert _splits(root)[0].num_classes == 3

    def test_color_augmentation_reaches_train_only(self, root):
        plain, _ = _splits(root, augment=True, augment_color=False, batch_size=12)
        colour, val = _splits(root, augment=True, augment_color=True, batch_size=12)
        a = next(iter(plain.dataset))[0].numpy()
        b = next(iter(colour.dataset))[0].numpy()
        assert not np.array_equal(a, b)
        # val ignores both switches: it equals a build with them off
        _, val_off = _splits(root, augment=False, augment_color=False, batch_size=12)
        v_on = next(iter(val.dataset))[0].numpy()
        v_off = next(iter(val_off.dataset))[0].numpy()
        assert np.array_equal(v_on, v_off)


class TestEvalHead:
    def test_size_dtype_and_label_order(self, root):
        _, val = _splits(root)
        x, y = eval_head(val, 4)
        assert x.shape == (4, 32, 32, 3) and x.dtype == np.float32
        assert y.tolist() == VAL_CLASSES[:4]

    def test_it_is_deterministic(self, root):
        _, val = _splits(root, augment=True)
        x1, _ = eval_head(val, 6)
        x2, _ = eval_head(val, 6)
        assert np.array_equal(x1, x2)

    def test_n_beyond_val_returns_all_of_it(self, root):
        _, val = _splits(root)
        x, y = eval_head(val, 1000)
        assert len(x) == len(y) == len(VAL_CLASSES)

    def test_a_non_positive_n_is_refused(self, root):
        _, val = _splits(root)
        with pytest.raises(ValueError, match="positive"):
            eval_head(val, 0)
