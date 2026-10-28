"""``train.convnext.common`` trains on the extracted ImageNet through the shared loader.

The tree is a 3-class fake (``tests/imagenet_fake_tree.py``): 12 train images, 6 val.
``holdout_per_class=1`` takes 3 of the train images out for selection, so the run has
9 train, 3 selection and 6 test images. The model is the real ConvNeXt ``cifar10``
variant at 32 px; what is under test is the data path: which datasets reach ``fit``,
which images reach the figures and the analyzer, what ``evaluate`` sees, the protocol
the summary records, and the flags that apply to one dataset only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.convnext.common as common  # noqa: E402
from dl_techniques.datasets.vision.imagenet2012 import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageNet2012Config,
    load_imagenet2012,
)
from tests.imagenet_fake_tree import PER_CLASS, VAL_CLASSES, WNIDS, build_fake_imagenet  # noqa: E402
from train.common import run_summary  # noqa: E402

N_TRAIN_FILES = 3 * PER_CLASS
N_HOLDOUT = 3
N_TRAIN = N_TRAIN_FILES - N_HOLDOUT
EVAL_SAMPLES = 4
FIGURES = (
    "training_dashboard.png", "confusion_matrix.png", "per_class_metrics.png",
    "confidence_calibration.png", "misclassifications.png", "classification_report.json",
)


def _config(root: str, out: Path, name: str, **overrides) -> "common.TrainingConfig":
    settings = dict(
        dataset="imagenet", imagenet_root=root, variant="cifar10", strides=2, image_size=32,
        holdout_per_class=1, eval_samples=EVAL_SAMPLES, epochs=2, batch_size=4,
        learning_rate=1e-3, seed=3, output_dir=str(out), experiment_name=name,
    )
    settings.update(overrides)
    return common.TrainingConfig(**settings)


def _count(dataset) -> int:
    return sum(len(y) for _, y in dataset)


def _pixels(dataset) -> np.ndarray:
    return np.concatenate([x.numpy() for x, _ in dataset])


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")


@pytest.fixture(scope="module")
def run(tree, tmp_path_factory):
    """One real run with spies on everything that decides which images are used."""
    out = tmp_path_factory.mktemp("convnext_imagenet")
    seen = SimpleNamespace(evaluated=[], fit=None, figures=None, analysis=None)

    real_evaluate = common._evaluate
    real_fit = keras.Model.fit
    real_figures = run_summary.write_classification_figures

    def evaluate(model, dataset):
        seen.evaluated.append(_count(dataset))
        return real_evaluate(model, dataset)

    def fit(self, x=None, *args, **kwargs):
        seen.fit = SimpleNamespace(
            train=_count(x), validation_pixels=_pixels(kwargs["validation_data"]))
        return real_fit(self, x, *args, **kwargs)

    def figures(vis_dir, x_test, y_test, probs, class_names, **kwargs):
        seen.figures = SimpleNamespace(
            x=x_test, y=y_test, probs=probs, class_names=class_names, kwargs=kwargs)
        return real_figures(vis_dir, x_test, y_test, probs, class_names, **kwargs)

    def analysis(model, test_data, *args, **kwargs):
        seen.analysis = test_data

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(common, "_evaluate", evaluate)
        mp.setattr(keras.Model, "fit", fit)
        mp.setattr(run_summary, "write_classification_figures", figures)
        mp.setattr(common, "run_model_analysis", analysis)
        summary = common.train(_config(tree, out, "imagenet_run"))
    return SimpleNamespace(out=out, run_dir=out / "imagenet_run", summary=summary, seen=seen,
                           tree=tree)


def test_the_run_finishes_and_writes_its_artifacts(run) -> None:
    assert run.summary["status"] == "ok"
    for name in ("config.json", "training_log.csv", "best_model.keras", "final_model.keras",
                 "results_summary.json", "run.log"):
        assert (run.run_dir / name).is_file(), name
    for name in FIGURES:
        assert (run.run_dir / "visualizations" / name).is_file(), name


def test_the_summary_records_the_protocol(run) -> None:
    s = json.loads((run.run_dir / "results_summary.json").read_text())
    assert s["dataset"] == "imagenet" and s["variant"] == "cifar10"
    assert (s["n_train"], s["n_val"], s["n_test"]) == (N_TRAIN, N_HOLDOUT, len(VAL_CLASSES))
    assert s["figure_samples"] == EVAL_SAMPLES
    assert s["holdout_per_class"] == 1 and s["validation_split"] is None
    assert s["selection_split"] == "1 images per class held out of train (3 images)"
    assert s["test_split"] == "the official val split"
    assert s["num_classes"] == 3 and s["input_shape"] == [32, 32, 3]
    assert s["steps_per_epoch"] == N_TRAIN // 4
    assert s["initial_loss_sanity_eval"]["n_samples"] == N_HOLDOUT
    assert any("first 4 of the 6 test images" in n for n in s["notes"])


def test_fit_trains_on_train_minus_holdout_and_selects_on_the_holdout(run, tree) -> None:
    holdout = load_imagenet2012("holdout", ImageNet2012Config(
        root=tree, image_size=32, batch_size=4, holdout_per_class=1))
    val = load_imagenet2012("val", ImageNet2012Config(root=tree, image_size=32, batch_size=4))
    assert run.seen.fit.train == (N_TRAIN // 4) * 4
    np.testing.assert_array_equal(run.seen.fit.validation_pixels, _pixels(holdout.dataset))
    assert run.seen.fit.validation_pixels.shape[0] == N_HOLDOUT
    assert not np.array_equal(run.seen.fit.validation_pixels, _pixels(val.dataset)[:N_HOLDOUT])


def test_evaluate_sees_every_selection_and_test_image(run) -> None:
    """The epoch-0 baseline on the 3 selection images, then best (reloaded and in-memory)
    and final on all 6 test images; never the 4-image figure subset."""
    assert run.seen.evaluated == [N_HOLDOUT, 6, 6, 6]


def test_figures_and_the_analyzer_use_the_first_eval_samples_test_images(run, tree) -> None:
    figures = run.seen.figures
    val = load_imagenet2012("val", ImageNet2012Config(root=tree, image_size=32, batch_size=4))
    assert figures.x.shape == (EVAL_SAMPLES, 32, 32, 3)
    np.testing.assert_array_equal(figures.x, _pixels(val.dataset)[:EVAL_SAMPLES])
    assert figures.y.tolist() == VAL_CLASSES[:EVAL_SAMPLES]
    assert figures.probs.shape == (EVAL_SAMPLES, 3)
    np.testing.assert_allclose(figures.probs.sum(axis=-1), 1.0, atol=1e-5)
    assert figures.class_names == WNIDS
    np.testing.assert_allclose(figures.kwargs["mean"], IMAGENET_MEAN, rtol=1e-6)
    np.testing.assert_allclose(figures.kwargs["std"], IMAGENET_STD, rtol=1e-6)
    x_analysis, y_analysis = run.seen.analysis
    np.testing.assert_array_equal(x_analysis, figures.x)
    np.testing.assert_array_equal(y_analysis, figures.y)


def test_a_missing_imagenet_root_leaves_no_run_directory(tmp_path) -> None:
    config = _config(str(tmp_path / "nope"), tmp_path, "x")
    with pytest.raises(ValueError, match="ImageNet root does not exist"):
        common.train(config)
    assert not (tmp_path / "x").exists()


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


def test_imagenet_defaults_are_resolved_per_dataset(tmp_path) -> None:
    config = common.TrainingConfig(dataset="imagenet", output_dir=str(tmp_path))
    assert (config.variant, config.strides) == ("tiny", 4)
    assert (config.drop_path_rate, config.dropout_rate) == (0.1, 0.0)
    assert config.validation_split is None
    assert config.holdout_per_class == 10 and config.image_size == 224
    for dataset in ("mnist", "cifar10", "cifar100"):
        other = common.TrainingConfig(dataset=dataset, output_dir=str(tmp_path))
        assert (other.variant, other.strides, other.validation_split) == ("cifar10", 2, 0.1)


def test_an_explicit_variant_and_strides_win_over_the_dataset_default(tmp_path) -> None:
    config = common.TrainingConfig(
        dataset="imagenet", variant="small", strides=2, output_dir=str(tmp_path))
    assert (config.variant, config.strides) == ("small", 2)


def test_a_validation_split_is_refused_with_imagenet(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation_split=0.1 does not apply to ImageNet"):
        common.TrainingConfig(dataset="imagenet", validation_split=0.1, output_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ["holdout_per_class", "image_size", "eval_samples"])
def test_a_non_positive_imagenet_count_is_refused_naming_it(tmp_path, field) -> None:
    with pytest.raises(ValueError, match=f"{field} must be >= 1, got 0"):
        common.TrainingConfig(dataset="imagenet", output_dir=str(tmp_path), **{field: 0})


def test_a_batch_larger_than_the_imagenet_train_pool_is_refused_before_a_run_directory(
        tmp_path) -> None:
    with pytest.raises(ValueError, match="batch_size 64 exceeds the 10 train samples"):
        common.TrainingConfig(dataset="imagenet", max_samples=10, batch_size=64,
                              output_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
