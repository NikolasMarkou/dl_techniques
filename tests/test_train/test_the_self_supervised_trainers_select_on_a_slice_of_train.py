"""DINO, BEiT MIM, the BEiT tokenizer and ET masked completion never select on the test set.

Each builds its pipelines with ``train.common.image_streams.load_image_streams``: the
selection split (what ``validation_data``, early stopping, the best checkpoint and DINO's
per-epoch k-NN probe read) is a seeded slice of train, the test split is separate, and the
fit, selection and test rows are pairwise disjoint.

The audit is behavioural, image by image: the fake Imagenette records give row ``i`` the mean
colour ``4 * i`` (test rows ``4 * i + 2``), so the identity of every image is read back from
the pixels of the REAL ``build_datasets`` / ``build_knn_datasets`` output, with the trainers'
own element maps in place. A census can name a variable; only this can see which rows it holds.
"""

import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.splits import seeded_split_indices, seeded_test_rows
from tests.classification_fakes import (
    IMAGENETTE_FAKE_TEST,
    IMAGENETTE_FAKE_TRAIN,
    identities_of,
    use_fake_imagenette,
)
from train.beit import train_mim as beit_mim
from train.beit import train_tokenizer as beit_tokenizer
from train.dino import train_dino as dino
from train.energy_transformer import train_masked_completion as et_mim

SEED = 5
N_TRAIN, N_TEST = IMAGENETTE_FAKE_TRAIN, IMAGENETTE_FAKE_TEST
GRID = 4  # patches per side at image 64 / patch 16


def _fake_tokenizer(images):
    """A BATCHED tokenizer stand-in: code ids on a ``GRID x GRID`` grid."""
    return tf.zeros((tf.shape(images)[0], GRID, GRID), tf.int32)


def _ids(ds, offset, n_batches=None):
    return identities_of("imagenette", ds, offset, n_batches)


def _beit_mim(tmp_path, **kwargs):
    checkpoint = tmp_path / "tokenizer.keras"
    checkpoint.write_bytes(b"")  # __post_init__ only needs a .keras file that exists
    config = beit_mim.TrainingConfig(
        dataset="imagenette", image_size=64, patch_size=16, batch_size=6, seed=SEED,
        augment_data=False, num_mask_patches=6, min_mask_patches_per_block=2,
        tokenizer_checkpoint=str(checkpoint), output_dir=str(tmp_path / "results"), **kwargs)
    return config, lambda: beit_mim.build_datasets(config, _fake_tokenizer)


def _beit_tokenizer(tmp_path, **kwargs):
    config = beit_tokenizer.TrainingConfig(
        dataset="imagenette", image_size=64, batch_size=6, seed=SEED,
        augment_data=False, output_dir=str(tmp_path / "results"), **kwargs)
    return config, lambda: beit_tokenizer.build_datasets(config)


def _et_mim(tmp_path, **kwargs):
    config = et_mim.TrainingConfig(
        dataset="imagenette", image_size=64, patch_size=16, batch_size=6, seed=SEED,
        augment_data=False, output_dir=str(tmp_path / "results"), **kwargs)
    return config, lambda: et_mim.build_datasets(config)


BUILDERS = {"beit_mim": _beit_mim, "beit_tokenizer": _beit_tokenizer, "et_masked_completion": _et_mim}


@pytest.fixture(autouse=True)
def records(monkeypatch):
    return use_fake_imagenette(monkeypatch)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_selection_is_a_slice_of_train_and_the_test_split_is_separate(name, tmp_path) -> None:
    config, build = BUILDERS[name](tmp_path)
    train_ds, val_ds, test_ds, steps, val_steps = build()
    _, select_rows, _ = seeded_split_indices(N_TRAIN, N_TEST, SEED, 0.1)

    fit = _ids(train_ds, 0, steps)
    select = _ids(val_ds, 0)
    test = _ids(test_ds, 2)

    assert list(select) == select_rows.tolist(), "selection is not the classifiers' seeded slice"
    assert not set(fit) & set(select), "a selection image is also a fit image"
    assert list(test) == seeded_test_rows(N_TEST, SEED).tolist(), "test is not the validation records"
    assert (steps, val_steps) == (54 // 6, 1)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_the_validation_data_a_trainer_would_pass_is_not_the_test_split(name, tmp_path) -> None:
    """`val_ds`, what a trainer passes as validation_data, is not the test split."""
    _, build = BUILDERS[name](tmp_path)
    _, val_ds, test_ds, _, _ = build()
    assert len(_ids(val_ds, 0)) == 6
    assert len(_ids(test_ds, 2)) == N_TEST


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_the_selection_split_and_the_cap_reach_the_streams(name, tmp_path) -> None:
    _, build = BUILDERS[name](tmp_path, validation_split=0.25, max_samples=40)
    train_ds, val_ds, test_ds, steps, val_steps = build()
    assert len(_ids(val_ds, 0)) == 10
    assert len(_ids(test_ds, 2)) == 20
    assert (steps, val_steps) == (30 // 6, 2)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_max_steps_caps_both_step_counts(name, tmp_path) -> None:
    _, build = BUILDERS[name](tmp_path, max_steps=2)
    *_, steps, val_steps = build()
    assert (steps, val_steps) == (2, 1)


def _dino_config(**kwargs):
    return dino.TrainingConfig(
        dataset="imagenette", variant="tiny", global_crop_size=32, patch_size=16,
        dino_out_dim=16, n_local_crops=1, batch_size=6, seed=SEED, knn_bank_batches=4,
        knn_query_batches=1, **kwargs)


def test_dino_banks_the_fit_rows_and_queries_the_selection_slice_never_the_test_split() -> None:
    config = _dino_config()
    bank, query = dino.build_knn_datasets(config)
    test_ds, n_batches = dino.build_knn_test_dataset(config)
    _, select_rows, _ = seeded_split_indices(N_TRAIN, N_TEST, SEED, 0.1)

    bank_ids = set(_ids(bank, 0))
    query_ids = _ids(query, 0)
    assert query_ids.tolist() == select_rows[:6].tolist(), "queries are not the selection slice"
    assert not bank_ids & set(select_rows.tolist()), "the bank holds a selection row"
    assert list(_ids(test_ds, 2)) == seeded_test_rows(N_TEST, SEED).tolist()
    assert n_batches == 4  # ceil(20 / 6)


def test_dino_trains_on_the_fit_rows_only() -> None:
    """Views are multi-crops, so identity is not readable; the stream length is."""
    config = _dino_config(max_steps=3)
    train_ds, steps = dino.build_dataset(config)
    assert steps == 3
    views, _ = next(iter(train_ds))
    assert views.shape[0] == 6 and views.shape[1] == config.n_views
