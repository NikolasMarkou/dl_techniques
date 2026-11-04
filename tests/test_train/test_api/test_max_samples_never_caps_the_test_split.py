"""``data.max_samples`` bounds the TRAIN pool (fit and selection together) and never the test split (plan 98b866fb 7.2, D-022).

COCO's held-out test had become 4,000 of 5,000 pairs because the cap that was chosen to bound training
cost also cut the split every reported test number, figure and analyzer pass is computed on. Comparable
tools cap each split with its own key (Hugging Face ``max_train_samples`` and ``max_eval_samples``,
Lightning ``limit_train_batches`` and ``limit_test_batches``), and a training cap never shrinks the test.

One function holds the cut (``dl_techniques.datasets.vision.splits.seeded_split_indices``), so this file
pins it directly, then each loader the API reaches: classification folder, csv, in-memory, segmentation
folder, csv. (Pet, Imagenette and ImageNet are pinned beside their loaders.)
"""

import numpy as np
import pytest

from dl_techniques.datasets.vision.splits import seeded_split_indices, seeded_test_rows
from train.common import segmentation_data as sd

from ._segmentation_fixtures import load, make_folder, write_csv


@pytest.mark.parametrize("cap", [None, 1000, 40, 10])
def test_the_test_rows_are_the_whole_split_in_the_seeded_order_whatever_the_cap(cap):
    fit, select, test = seeded_split_indices(100, 37, 5, 0.1, cap)
    assert test.tolist() == seeded_test_rows(37, 5).tolist()
    assert sorted(test.tolist()) == list(range(37))
    pool = 100 if cap is None else min(cap, 100)
    assert len(fit) + len(select) == pool


def test_the_train_cut_is_the_same_cut_a_cap_always_made():
    # The cap's effect on the train pool is unchanged: the first `cap` rows of the seeded permutation.
    fit, select, _ = seeded_split_indices(100, 30, 7, 0.2, 50)
    order = np.random.default_rng(7).permutation(100)[:50]
    assert select.tolist() == order[:10].tolist() and fit.tolist() == order[10:].tolist()


def test_a_test_listing_is_scored_in_full_by_the_segmentation_loaders(tmp_path):
    images, masks = make_folder(tmp_path, count=12)
    t_images, t_masks = make_folder(tmp_path, count=9, prefix="t_", size=(40, 40))
    folder = sd.scan_segmentation_folder(str(images), str(masks), str(t_images), str(t_masks))
    rows = [(str(i), str(m)) for i, m in zip(sorted(images.glob("*.png")), sorted(masks.glob("*.png")))]
    t_rows = [(str(i), str(m)) for i, m in zip(sorted(t_images.glob("*.png")), sorted(t_masks.glob("*.png")))]
    csv = sd.read_segmentation_csv(
        str(write_csv(tmp_path / "train.csv", rows)), str(write_csv(tmp_path / "test.csv", t_rows)))
    for listing in (folder, csv):  # the folder loader and the csv loader share one cut
        data = load(listing, max_samples=4, validation_split=0.5)
        assert data.n_train + data.n_select == 4
        assert data.n_test == 9
        assert int(data.test_ds.cardinality()) * 2 >= 9
