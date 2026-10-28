"""CliffordCLIP selects on a held-out slice of TRAIN images and scores the official split once.

Before, ``fit(validation_data=...)``, the retrieval probes and the final reported recall all
read COCO val2017 (CC3M validation), so the number a run reported was on the data runs were
compared by. ``_prepare_datasets`` now cuts the selection split out of the train images, BY
IMAGE (COCO lists about five captions per image and retrieval scores one per image), and
returns the official split untouched as the test split.

The loaders are replaced by fakes whose image paths carry an identity, so the audit is on the
real ``_prepare_datasets`` output, not on a re-implementation of the cut.
"""

import argparse

import numpy as np
import pytest

import train.cliffordnet.train_clip as trainer

N_TRAIN_IMAGES = 30
CAPTIONS = 5
N_VAL_IMAGES = 12


class _Encoder:
    n_vocab = 100
    eot_token = 99


def _fake_coco(captured):
    def load(split, coco_root, max_samples, encoder, context_length):
        captured.append((split, max_samples))
        if split == "train":
            paths = [f"/coco/train2017/{i:06d}.jpg"
                     for i in range(N_TRAIN_IMAGES) for _ in range(CAPTIONS)]
        else:
            paths = [f"/coco/val2017/{i:06d}.jpg" for i in range(N_VAL_IMAGES)]
        if max_samples is not None:
            paths = paths[:max_samples]
        tokens = np.arange(len(paths), dtype=np.int32)[:, None] * np.ones((1, 4), np.int32)
        return paths, tokens

    return load


def _args(**kwargs):
    base = dict(synthetic=False, dataset="coco2017", coco_root="/unused", cc3m_root="/unused",
                max_train_samples=None, max_test_samples=None, selection_images=6, seed=11,
                image_size=32, context_length=4)
    base.update(kwargs)
    return argparse.Namespace(**base)


@pytest.fixture
def splits(monkeypatch):
    captured = []
    monkeypatch.setattr(trainer, "load_coco2017_local_split", _fake_coco(captured))
    return trainer._prepare_datasets(_args(), _Encoder()), captured


def test_the_three_splits_share_no_image(splits) -> None:
    result, _ = splits
    train, select, test = (set(part[0]) for part in (result.train, result.select, result.test))
    assert not train & select, "a selection image is also a fit image"
    assert not select & test and not train & test


def test_the_test_split_is_the_official_validation_split_untouched(splits) -> None:
    result, captured = splits
    assert result.test_name == "COCO val2017"
    assert len(result.test[0]) == N_VAL_IMAGES
    assert all("/val2017/" in path for path in result.test[0])
    assert ("val", None) in captured


def test_the_selection_comes_from_train_one_caption_per_image(splits) -> None:
    result, _ = splits
    select_paths = result.select[0]
    assert len(select_paths) == len(set(select_paths)) == 6
    assert all("/train2017/" in path for path in select_paths)


def test_a_held_out_image_keeps_no_caption_in_the_fit_set(splits) -> None:
    result, _ = splits
    assert not set(result.select[0]) & set(result.train[0])
    assert len(result.train[0]) == (N_TRAIN_IMAGES - 6) * CAPTIONS


def test_selection_images_and_the_seed_reach_the_cut(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "load_coco2017_local_split", _fake_coco([]))
    four = trainer._prepare_datasets(_args(selection_images=4), _Encoder())
    nine = trainer._prepare_datasets(_args(selection_images=9), _Encoder())
    other_seed = trainer._prepare_datasets(_args(selection_images=9, seed=12), _Encoder())
    assert (len(four.select[0]), len(nine.select[0])) == (4, 9)
    assert set(nine.select[0]) != set(other_seed.select[0])


def test_the_default_selection_size_is_the_size_of_val2017(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "load_coco2017_local_split", _fake_coco([]))
    assert trainer._DEFAULT_SELECTION_IMAGES == 5000
    with pytest.raises(ValueError, match="Lower --selection-images"):
        trainer._prepare_datasets(_args(selection_images=None), _Encoder())


def test_max_test_samples_caps_the_test_split_and_max_train_samples_the_pool(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "load_coco2017_local_split", _fake_coco([]))
    result = trainer._prepare_datasets(
        _args(max_train_samples=50, max_test_samples=7, selection_images=3), _Encoder())
    assert len(result.test[0]) == 7
    assert len(result.train[0]) + len(result.select[0]) <= 50


def test_the_synthetic_splits_are_three_independent_datasets() -> None:
    args = _args(synthetic=True, max_train_samples=16, max_test_samples=10,
                 selection_images=12, context_length=16)
    result = trainer._prepare_datasets(args, _Encoder())
    assert (len(result.train[0]), len(result.select[0]), len(result.test[0])) == (16, 12, 10)
    assert not np.array_equal(result.select[0][:10], result.test[0])


def test_the_retired_flag_is_gone() -> None:
    with pytest.raises(SystemExit):
        trainer.parse_arguments(["--max-val-samples", "8"])
    args = trainer.parse_arguments(["--selection-images", "9", "--max-test-samples", "7"])
    assert (args.selection_images, args.max_test_samples) == (9, 7)
