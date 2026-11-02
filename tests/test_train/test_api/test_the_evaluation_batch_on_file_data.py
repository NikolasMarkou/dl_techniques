"""What ``analysis_args.eval_batch_size`` does on folder data, and what its help says about it (review-iter-2 concern 11).

The help said file data "ignore it" and that it drove "every evaluation and prediction". Both were
wrong in opposite directions, found by reading the sites and running folder data at 224 px:

* the figures' ``predict`` and the final-model load check DO read it for every dataset, folder
  and csv data included; only the batching of the selection and test SPLITS (what ``fit`` validates
  on and ``evaluate`` scores) is the training ``batch_size`` for ImageNet, Imagenette, folder and csv;
* the analyzer's information-flow pass sizes its own batch from the analyzer's memory budget and does
  not read it (47 images against an evaluation batch of 8 in the gate log);
* the recipes' large-image values (8 and 16 for the cliffordnet families) are keyed to the datasets
  ``imagenet`` and ``imagenette``, so folder and csv data at 224 px resolve to the shared 256, and
  the base recipe's batch of 128 for training. Measured on the GPU (review-iter-2): nano logged
  "Allocator ran out of memory trying to allocate 12.27GiB" and a garbage collection at that
  setting, then finished; ``cn_64_5`` finished clean at batch 2. No death was reproduced, so the
  answer is a WARNING (naming the values a user should set), not a changed default: the measured
  values belong to one card and one dataset family, and a recipe keyed by image size is a design
  change, not a fix.
"""

import pytest

from train.api import registry
from train.api.args import AnalysisArgs
from train.common.classification_data import EVAL_BATCH_SIZE

from .test_folder_and_csv_data import folder_config, resolve, tree, world  # noqa: F401  (fixtures)
from .test_the_evaluation_batch import SpyPredict


def help_text():
    import dataclasses

    return {f.name: f.metadata["help"] for f in dataclasses.fields(AnalysisArgs)}["eval_batch_size"]


def test_the_help_states_where_the_key_is_read_and_where_it_is_not():
    text = help_text()
    # Read: the figures (every dataset), the analyzer's cap, the load check, the in-memory splits.
    assert "figures' predictions" in text and "every dataset" in text
    assert "load check" in text and "CIFAR, MNIST" in text
    # Not read: the splits of file and ImageNet data, and the information-flow pass.
    assert "validation and test splits of ImageNet, Imagenette, folder and csv" in text
    assert "information-flow" in text and "does not read it" in text
    # The overlay is keyed to a dataset, not to a size: folder data is told to set it.
    assert "folder and csv data take 256" in text
    # The old overstatements are gone.
    assert "every evaluation and prediction" not in text
    assert "folder and csv splits are evaluated at the training batch_size and ignore it" not in text


def test_the_figures_predict_at_the_key_on_folder_data_and_the_splits_stay_at_the_training_batch(
    world, tree, tmp_path, monkeypatch
):
    from train.api.tasks.image_classification import ImageClassificationTrainer

    spy = SpyPredict(monkeypatch)
    config = folder_config(
        tmp_path / "run", tree, epochs=1, batch_size=4,
        metric_args={"success_threshold": 0.0}, analysis_args={"model_analysis": False, "eval_batch_size": 3},
    )
    ImageClassificationTrainer(config).run()
    # The figures' call (every test image, at the key), then the load check's two at the cap (3 < 32).
    assert [b for _, b in spy.calls] == [3, 3, 3], spy.calls
    # ... while the splits ``fit`` validates on are batched at the training batch (4), not at the key (3).
    trainer, resolved, context = resolve(config)
    data = trainer.build_data(resolved, context)
    assert resolved.analysis_args.eval_batch_size == 3 and resolved.batch_size == 4
    widths = {int(images.shape[0]) for images, _ in data.select_ds} | {
        int(images.shape[0]) for images, _ in data.test_ds}
    assert max(widths) == 4 and 3 not in widths, widths


def test_folder_data_at_224_px_with_a_cliffordnet_resolves_to_the_shared_defaults_and_is_warned(
    world, tree, tmp_path
):
    trainer, resolved, _ = resolve(folder_config(
        tmp_path / "a", tree, model="cliffordnet/nano", transform_args={"image_size": 224}))
    imagenet = registry.MODELS["cliffordnet"].defaults["per_dataset"]["imagenet"]
    assert resolved.analysis_args.eval_batch_size == EVAL_BATCH_SIZE
    assert resolved.batch_size != imagenet["batch_size"]  # the base recipe's batch, not the measured one
    warnings = [text for level, text in trainer._notices if level == "warning"]
    (text,) = [t for t in warnings if "cliffordnet" in t and "224" in t]
    assert str(imagenet["batch_size"]) in text and "analysis_args.eval_batch_size" in text
    assert "batch_size" in text and "were measured on imagenet and imagenette" in text


@pytest.mark.parametrize("model, size", [
    ("vit/vit_pico", 224),          # no large-image batch in its recipe
    ("cliffordnet/nano", 128),      # below the measured size
], ids=["other_family", "small_images"])
def test_the_warning_is_not_given_where_nothing_was_measured_to_fail(world, tree, tmp_path, model, size):
    trainer, _, _ = resolve(folder_config(tmp_path / "a", tree, model=model, transform_args={"image_size": size}))
    assert not [t for level, t in trainer._notices if level == "warning"]
