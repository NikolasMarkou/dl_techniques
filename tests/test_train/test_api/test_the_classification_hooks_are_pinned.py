"""The hooks of the classification task, one by one, on stand-in data (review-iter-1 concern 10).

Seven of the review's mutants survived the whole directory because each is a line of one hook that
only a full fit could reach, and the fit tests assert something else. A hook is a function of a
config and a few facts, so it is called here directly:

* ``input_check`` refuses a class count AND an image side that ``describe_data`` did not promise;
* ``epoch_line_metrics`` names the top-k pair at the configured ``k``;
* ``recipe_notices`` says what it says for a file dataset, a dataset with no overlay, and one with;
* ``task_notes`` writes each line of the summary's ``notes`` for the switch that causes it,
  the conformal one included;
* ``summary_fields`` sit between ``n_test`` and ``epochs_requested`` in the summary head, which is
  the key order of every ``results_summary.json`` (the untracked golden saw it, no test did);
* the conformal record says the best weights were NOT selected on the calibration split.

Nothing is trained; the file writes nothing.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api.config import validate
from train.api.data_args import DataFacts
from train.api.tasks.image_classification import ImageClassificationConfig, ImageClassificationTrainer


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def task(tmp_path, **groups):
    mapping = {
        "out": str(tmp_path / "run"),
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "batch_size": 4,
        **groups,
    }
    return ImageClassificationTrainer(validate(ImageClassificationConfig, mapping))


def data(classes=10, side=32, with_test=True):
    return SimpleNamespace(
        num_classes=classes, input_shape=(side, side, 3), test_ds=object() if with_test else None,
        x_fig=np.zeros((7, side, side, 3)), mean=[0.5], std=[0.25],
    )


def test_input_check_passes_the_promised_data_and_refuses_each_way_it_differs(world, tmp_path):
    trainer = task(tmp_path)
    facts = DataFacts(dataset="cifar10", num_classes=10, image_size=32)
    trainer.input_check(facts, data())  # control
    with pytest.raises(RuntimeError, match=r"said 10 classes at 32 px, but build_data loaded 9 classes"):
        trainer.input_check(facts, data(classes=9))
    # The side alone: the class count agrees, so only the second half of the condition can see it.
    with pytest.raises(RuntimeError, match=r"said 10 classes at 32 px, but build_data loaded 10 classes with input shape \(28, 28, 3\)"):
        trainer.input_check(facts, data(side=28))


@pytest.mark.parametrize("k", [3, 5])
def test_the_epoch_line_names_the_top_k_pair_at_the_configured_k(world, tmp_path, k):
    trainer = task(tmp_path, metric_args={"top_k": k})
    assert trainer.epoch_line_metrics(trainer.config) == (
        "loss", "accuracy", f"top{k}_accuracy", "val_loss", "val_accuracy", f"val_top{k}_accuracy",
    )


def test_the_recipe_notices_for_a_file_dataset_an_unlisted_dataset_and_a_listed_one(world, tmp_path):
    trainer = task(tmp_path)
    folder = trainer.recipe_notices(DataFacts(dataset="folder", num_classes=3, image_size=32), "vit")
    assert [level for level, _ in folder] == ["info"]
    assert "folder data: no legacy trainer reads files" in folder[0][1] and "vit recipe" in folder[0][1]
    csv = trainer.recipe_notices(DataFacts(dataset="csv", num_classes=3, image_size=32), "resnet")
    assert [level for level, _ in csv] == ["info"] and "csv data" in csv[0][1]
    listed = next(iter(registry.MODELS["vit"].defaults["per_dataset"]))
    assert trainer.recipe_notices(DataFacts(dataset=listed, num_classes=3, image_size=32), "vit") == []
    # A dataset the family has no overlay for gets the warning, not the file notice.
    unlisted = next(name for name in ("mnist", "cifar10", "cifar100", "imagenette", "imagenet")
                    if name not in registry.MODELS["resnet"].defaults.get("per_dataset", {}))
    warned = trainer.recipe_notices(DataFacts(dataset=unlisted, num_classes=3, image_size=32), "resnet")
    assert [level for level, _ in warned] == ["warning"]
    assert f"No recipe overlay for resnet on {unlisted}" in warned[0][1]


def lines_of(trainer, test_split=True):
    return trainer.task_notes(trainer.config, {"loss": 2.3}, 1.0, data(with_test=test_split))


def test_each_switch_writes_its_own_note_and_only_that_one(world, tmp_path):
    base = lines_of(task(tmp_path, analysis_args={"conformal_alpha": 0.2}))
    assert len(base) == 4  # the four fixed lines: figures, model analysis and conformal are all on
    conformal_off = lines_of(task(tmp_path))
    assert [line for line in conformal_off if line not in base] == [
        "analysis_args.conformal_alpha is null: `conformal` is null"
    ]
    assert any("the data has no test split" in line for line in lines_of(task(tmp_path), test_split=False))
    assert not any("the data has no test split" in line for line in conformal_off)
    many = lines_of(task(
        tmp_path,
        checkpoint_args={"save_best": False, "save_last": False},
        analysis_args={"figures": False, "model_analysis": False},
    ))
    joined = "\n".join(many)
    for key in ("save_best is false", "save_last is false", "figures is false", "model_analysis is false"):
        assert key in joined, key


def test_the_summary_fields_sit_between_n_test_and_epochs_requested(world, tmp_path):
    trainer = task(tmp_path)
    resolved_data = SimpleNamespace(
        test_ds=object(), input_shape=(32, 32, 3), num_classes=10, steps_per_epoch=5, selection_split="val",
        test_split="test", n_train=40, n_select=8, n_test=8, x_fig=np.zeros((7, 32, 32, 3)),
        mean=[0.5], std=[0.25],
    )
    head = trainer._summary_head(
        trainer.config, "vit", "vit_pico", tmp_path / "run", resolved_data, 123, {"loss": 2.3}, 1.0, False,
        {"gpu_name": None, "tf_visible_devices": 0, "cuda_visible_devices": ""},
    )
    keys = list(head)
    assert keys[keys.index("n_test") + 1:keys.index("epochs_requested")] == ["figure_samples", "input_normalization"]
    assert head["figure_samples"] == 7 and head["input_normalization"] == {"mean": [0.5], "std": [0.25]}
