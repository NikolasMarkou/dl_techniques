"""AlexNet, BEiT and Energy-Transformer classification read their data through the loader.

``build_datasets`` (BEiT, ET) and ``create_imagenette_data`` (AlexNet) return a selection
split cut from train and a separate test split. The record reader and the CIFAR arrays are
replaced by fakes, so nothing touches the TFDS records or the Keras cache.
"""

import importlib
import inspect

import numpy as np
import pytest

import train.common.classification_data as classification_data
from tests.classification_fakes import tiny_raw, use_fake_imagenette

MODULES = ["train.beit.train_classification", "train.energy_transformer.train_classification"]


def _config(module, **kwargs):
    settings = dict(dataset="imagenette", image_size=32, patch_size=8, batch_size=6,
                    augment_data=False, seed=3)
    settings.update(kwargs)
    return module.TrainingConfig(**settings)


@pytest.mark.parametrize("name", MODULES)
def test_imagenette_gives_a_selection_slice_and_a_test_split(name, monkeypatch) -> None:
    module = importlib.import_module(name)
    use_fake_imagenette(monkeypatch)
    train_ds, val_ds, test_ds, steps, val_steps, num_classes = module.build_datasets(_config(module))
    assert num_classes == 10 and steps == 54 // 6 and val_steps == 1
    assert sum(len(y) for _, y in train_ds.take(steps)) == 54
    assert sum(len(y) for _, y in val_ds) == 6
    assert sum(len(y) for _, y in test_ds) == 20


@pytest.mark.parametrize("name", MODULES)
def test_cifar10_is_cut_the_same_way(name, monkeypatch) -> None:
    module = importlib.import_module(name)
    monkeypatch.setattr(classification_data, "_load_raw", tiny_raw(n_train=100, n_test=30))
    train_ds, val_ds, test_ds, steps, val_steps, num_classes = module.build_datasets(
        _config(module, dataset="cifar10", image_size=32, batch_size=10, augment_data=True))
    assert num_classes == 10 and steps == 9 and val_steps == 1
    x, _ = next(iter(train_ds))
    assert x.shape == (10, 32, 32, 3)
    assert sum(len(y) for _, y in test_ds) == 30


@pytest.mark.parametrize("name", MODULES)
def test_cifar10_validation_steps_are_the_real_batch_count_of_the_selection_split(
        name, monkeypatch) -> None:
    """The in-memory selection split is batched at EVAL_BATCH_SIZE, not at the trainer's
    batch size; asking Keras for ceil(n_select / batch_size) steps overran it."""
    module = importlib.import_module(name)
    monkeypatch.setattr(classification_data, "_load_raw", tiny_raw(n_train=3000, n_test=30))
    _, val_ds, _, _, val_steps, _ = module.build_datasets(
        _config(module, dataset="cifar10", image_size=32, batch_size=10))
    assert val_steps == sum(1 for _ in val_ds) == -(-300 // classification_data.EVAL_BATCH_SIZE)


@pytest.mark.parametrize("name", MODULES)
def test_max_steps_caps_both_step_counts(name, monkeypatch) -> None:
    module = importlib.import_module(name)
    use_fake_imagenette(monkeypatch)
    *_, steps, val_steps, _ = module.build_datasets(_config(module, max_steps=2, batch_size=3))
    assert (steps, val_steps) == (2, 2)


@pytest.mark.parametrize("name", MODULES + ["train.alexnet.train_alexnet"])
def test_the_classifier_has_no_second_imagenette_reader(name) -> None:
    source = inspect.getsource(importlib.import_module(name))
    assert "build_raw_image_dataset" not in source


def test_alexnet_reads_imagenette_through_the_loader(monkeypatch) -> None:
    from train.alexnet import train_alexnet as module

    use_fake_imagenette(monkeypatch)
    config = module.TrainingConfig(dataset="imagenette", image_size=227, batch_size=6, seed=3)
    # 227 px keeps the model's own input; use a cap so the fake 60-image train set suffices
    data = module.create_imagenette_data(config)
    assert (data.n_train, data.n_select, data.n_test) == (54, 6, 20)
    x, _ = next(iter(data.train_ds))
    assert x.shape == (6, 227, 227, 3) and np.isfinite(x.numpy()).all()
