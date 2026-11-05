"""What the recipe decides must reach the run, and ``config.json`` must say so.

Review iter 2, concerns 3 and 7.

* C3: the default cliffordnet recipe erases to the DATASET MEAN, as
  ``train_cliffordnet.py`` does. Left at ``build_op``'s default the fill is zero:
  black on the [0, 1] image, ``-1.99`` after standardize. The arm reads erased
  pixels of the real training pipeline.
* C7: ``config.json`` holds the resolved config AND, under ``used``, the model
  keywords the family's resolver produced (a regularizer as its serialized
  config) and the recipe's loader mapping, so a run says what it was built with
  even after a recipe is edited. ``config_requested.json`` stays the request:
  it has no ``used``.

The C7 arms stop at the data loader (``batch_size=10000`` is refused there),
which is after ``config.json`` is written and before any model is built, so
they cost no fit. The tests write only under ``tmp_path``.
"""

import json

import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api.config import validate
from train.api.errors import ConfigError
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
    train_image_classification,
)

from tests.classification_fakes import use_tiny_raw


@pytest.fixture(scope="module", autouse=True)
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


@pytest.fixture(autouse=True)
def one_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


# ---------------------------------------------------------------------
# C3: erased pixels are the dataset mean
# ---------------------------------------------------------------------


def erased_batch(dataset: str, tmp_path):
    """Return ``(pixels, spec)``: one training batch with erasing probability 1.0."""
    from dl_techniques.datasets.vision.classification import get_spec

    config = validate(
        ImageClassificationConfig,
        {
            "out": str(tmp_path / "run"),
            "model": "vit/vit_pico",
            "data": {"dataset": dataset},
            "batch_size": 8,
            "transform_args": {"augment": "cifar_autoaugment", "random_erasing_prob": 1.0},
        },
    )
    trainer = ImageClassificationTrainer(config)
    family, variant = registry.parse_model_name(config.model)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    assert resolved.transform_args.normalize == "standardize"
    data = trainer.build_data(resolved, context)
    images, _ = next(iter(data.train_ds))
    return np.asarray(images), get_spec(dataset)


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_an_erased_pixel_is_zero_after_standardize_not_black(dataset, tmp_path):
    images, spec = erased_batch(dataset, tmp_path)
    erased = np.all(np.abs(images) < 1e-5, axis=-1)
    black = (0.0 - np.asarray(spec.mean)) / np.asarray(spec.std)
    # Probability 1.0: every image holds an erased rectangle of at least 2% of its
    # area (20 of 1024 pixels, less rounding), and its pixels are the mean, which is
    # 0 once standardized. A natural pixel is never within 1e-5 of the mean in all
    # three channels, and a black fill (the padding of the crop is black, so black
    # pixels exist anyway) would leave no such pixel at all.
    counts = erased.reshape(len(images), -1).sum(axis=1)
    assert (counts >= 16).all(), f"erased pixels per image: {counts.tolist()}"
    assert black.min() < -1.5, "the control: black standardizes far from the mean"


# ---------------------------------------------------------------------
# C7: config.json records what the run used
# ---------------------------------------------------------------------


def failing_run(out, **changes):
    """Run until the loader refuses ``batch_size`` and return ``config.json`` as read."""
    given = {
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "batch_size": 10000,
        "analysis_args": {"model_analysis": False},
        **changes,
    }
    with pytest.raises(ConfigError, match="'out' is now used"):
        train_image_classification(out=str(out), **given)
    return json.loads((out / "config.json").read_text())


def test_the_model_arguments_the_recipe_supplied_are_recorded(tmp_path):
    record = failing_run(tmp_path / "run")
    # The config's own field is what the caller typed; the resolver's output is `used`.
    assert record["model_args"] == {}
    assert record["used"]["model_args"] == {
        "dropout_rate": 0.1,
        "attention_dropout_rate": 0.1,
        "patch_size": 4,
        "kernel_regularizer": None,
    }
    assert record["used"]["loader"] == {}


def test_a_caller_key_replaces_the_recipes_in_the_record(tmp_path):
    record = failing_run(tmp_path / "run", model_args={"attention_dropout_rate": 0.3})
    assert record["model_args"] == {"attention_dropout_rate": 0.3}
    assert record["used"]["model_args"]["attention_dropout_rate"] == 0.3
    assert record["used"]["model_args"]["dropout_rate"] == 0.1


def test_the_sgd_l2_is_recorded_as_a_serialized_regularizer(tmp_path):
    record = failing_run(
        tmp_path / "run",
        optim_args={"optimizer": "sgd", "lr": 0.05, "weight_decay": 0.01},
    )
    regularizer = record["used"]["model_args"]["kernel_regularizer"]
    assert regularizer["class_name"] == "L2"
    assert regularizer["config"] == {"l2": 0.005}  # wd / 2: the penalty whose gradient is the SGD weight decay 0.01 (D-042)
    # The adamw run above records None: the optimizer decays, the model does not.
    assert failing_run(tmp_path / "other")["used"]["model_args"]["kernel_regularizer"] is None


def test_the_recipes_loader_mapping_is_recorded(tmp_path):
    record = failing_run(
        tmp_path / "run", model="convnext_v1/cifar10", data={"dataset": "mnist"}
    )
    assert record["used"]["loader"] == {"channels": 3}


def test_the_request_file_is_still_the_request(tmp_path):
    out = tmp_path / "run"
    failing_run(out)
    requested = json.loads((out / "config_requested.json").read_text())
    assert "used" not in requested
    assert requested["model_args"] == {}


def test_a_changed_recipe_value_changes_only_the_used_key(tmp_path):
    first = failing_run(tmp_path / "a", model_args={"patch_size": 4})
    second = failing_run(tmp_path / "b", model_args={"patch_size": 8})
    changed = {key for key in first if first[key] != second[key]} - {"out"}
    assert changed == {"model_args", "used"}
    assert first["used"]["model_args"]["patch_size"] == 4
    assert second["used"]["model_args"]["patch_size"] == 8
