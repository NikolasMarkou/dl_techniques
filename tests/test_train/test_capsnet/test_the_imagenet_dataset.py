"""``train_capsnet --dataset imagenet`` reads the extracted ImageNet through the shared loader.

The tree is a 3-class fake (``tests/imagenet_fake_tree.py``): 12 train images, 6 val.
``--holdout-per-class 1`` takes 3 train images out for selection, so the run has 9 train,
3 selection and 6 test images. The ImageNet dataset spec is patched to 3 classes for the run; the
1000-class model is built once, separately, to pin the size the module docstring claims.
What is under test is the data path (which datasets reach ``fit`` and ``evaluate``, what
the callbacks and figures get, the label and pixel format CapsNet needs), the flags, and
the monitored metric.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from dl_techniques.datasets.vision.imagenet2012 import (  # noqa: E402
    ImageNet2012Config,
    load_imagenet2012,
)
from dl_techniques.models.vision.capsnet import create_capsnet  # noqa: E402
from tests.imagenet_fake_tree import (  # noqa: E402
    PER_CLASS,
    VAL_CLASSES,
    build_fake_imagenet,
    use_the_fake_tree_class_count,
)
from train.capsnet import train_capsnet as module  # noqa: E402

N_CLASSES = 3
N_HOLDOUT = 3
N_TRAIN = 3 * PER_CLASS - N_HOLDOUT
BATCH = 4
EVAL_SAMPLES = 4


def _count(dataset) -> int:
    return sum(len(x) for x, _ in dataset)


def _pixels(dataset) -> np.ndarray:
    return np.concatenate([x.numpy() for x, _ in dataset])


def _argv(tree, *extra):
    return ["prog", "--dataset", "imagenet", "--imagenet-root", tree, "--image-size", "32",
            "--epochs", "2", "--batch-size", str(BATCH), "--holdout-per-class", "1",
            "--eval-samples", str(EVAL_SAMPLES), "--patience", "5", *extra]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")


@pytest.fixture(scope="module")
def run(tree, tmp_path_factory):
    """One real run; the trainer writes to ``results/`` under the working directory."""
    work = tmp_path_factory.mktemp("capsnet_run")
    seen = SimpleNamespace(fit=None, evaluated=[], final=None, callbacks=None)
    real_fit, real_evaluate = keras.Model.fit, keras.Model.evaluate
    real_final = module.generate_final_visualizations

    def fit(self, x=None, y=None, *args, **kwargs):
        batches = list(x)
        seen.fit = SimpleNamespace(
            y_is_none=y is None,
            images=np.concatenate([b[0].numpy() for b in batches]),
            labels=np.concatenate([b[1].numpy() for b in batches]),
            batch_sizes=[len(b[0]) for b in batches],
            validation_pixels=_pixels(kwargs["validation_data"]),
            callbacks=kwargs["callbacks"],
        )
        return real_fit(self, x, y, *args, **kwargs)

    def evaluate(self, x=None, *args, **kwargs):
        seen.evaluated.append(_count(x))
        return real_evaluate(self, x, *args, **kwargs)

    def final(**kwargs):
        seen.final = kwargs["test_data"]
        return real_final(**kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work)
        use_the_fake_tree_class_count(mp)
        mp.setattr(module, "setup_gpu", lambda *a, **k: None)
        mp.setattr(keras.Model, "fit", fit)
        mp.setattr(keras.Model, "evaluate", evaluate)
        mp.setattr(module, "generate_final_visualizations", final)
        mp.setattr(sys, "argv", _argv(tree))
        module.main()
    (results_dir,) = (work / "results").iterdir()
    return SimpleNamespace(work=work, results_dir=results_dir, seen=seen, tree=tree)


def test_fit_gets_batched_datasets_with_one_hot_labels_and_unit_pixels(run) -> None:
    fit = run.seen.fit
    assert fit.y_is_none
    assert fit.images.shape == ((N_TRAIN // BATCH) * BATCH, 32, 32, 3)
    assert fit.images.min() >= 0.0 and fit.images.max() <= 1.0
    assert fit.labels.shape == (len(fit.images), N_CLASSES)
    assert set(np.unique(fit.labels)) <= {0.0, 1.0} and (fit.labels.sum(axis=1) == 1).all()
    assert set(fit.batch_sizes) == {BATCH}


def test_selection_is_the_holdout_and_never_the_official_val_split(run, tree) -> None:
    holdout = load_imagenet2012("holdout", ImageNet2012Config(
        root=tree, image_size=32, batch_size=BATCH, holdout_per_class=1, normalize="unit"))
    val = load_imagenet2012("val", ImageNet2012Config(
        root=tree, image_size=32, batch_size=BATCH, normalize="unit"))
    validation = run.seen.fit.validation_pixels
    np.testing.assert_array_equal(validation, _pixels(holdout.dataset))
    assert len(validation) == N_HOLDOUT
    assert not np.array_equal(validation, _pixels(val.dataset)[:N_HOLDOUT])


def test_the_test_metrics_use_every_official_val_image(run) -> None:
    # fit evaluates the selection split once per epoch (3 images); the test split comes last, whole.
    assert run.seen.evaluated == [N_HOLDOUT, N_HOLDOUT, len(VAL_CLASSES)]


def test_the_final_figures_get_the_first_eval_samples_test_images_as_one_hot(run, tree) -> None:
    x, y = run.seen.final
    val = load_imagenet2012("val", ImageNet2012Config(
        root=tree, image_size=32, batch_size=BATCH, normalize="unit"))
    assert x.shape == (EVAL_SAMPLES, 32, 32, 3) and y.shape == (EVAL_SAMPLES, N_CLASSES)
    np.testing.assert_array_equal(x, _pixels(val.dataset)[:EVAL_SAMPLES])
    assert y.argmax(axis=1).tolist() == VAL_CLASSES[:EVAL_SAMPLES]


def test_the_run_writes_its_artifacts(run) -> None:
    for name in ("capsnet_imagenet_final.keras", "training_summary.txt", "training_log.csv",
                 "best_model.keras"):
        assert (run.results_dir / name).is_file(), name
    assert (run.results_dir / "final_visualizations").is_dir()
    assert any((run.results_dir / "reconstructions_per_epoch").iterdir())


def test_the_monitored_metric_exists_so_selection_can_happen(run) -> None:
    """``val_accuracy`` is not a CapsNet metric and ``val_capsule_accuracy`` is never logged
    (``test_step`` nests the accuracy under ``compile_metrics``): monitoring either made early
    stopping and the best-model checkpoint inert. The monitored key must be in the epoch logs."""
    monitors = {cb.monitor for cb in run.seen.fit.callbacks
                if isinstance(cb, (keras.callbacks.EarlyStopping, keras.callbacks.ModelCheckpoint))}
    assert monitors == {"val_loss"}
    header = (run.results_dir / "training_log.csv").read_text().splitlines()[0].split(",")
    assert "val_loss" in header


def test_a_missing_imagenet_root_is_refused_before_a_run_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", _argv(str(tmp_path / "nope")))
    with pytest.raises(ValueError, match="ImageNet root does not exist"):
        module.main()
    assert not (tmp_path / "results").exists()


# ---------------------------------------------------------------------
# Model size, config and flags
# ---------------------------------------------------------------------


def test_the_imagenet_config_is_the_cifar_capsules_on_1000_classes_at_32px() -> None:
    config = module.create_model_config("imagenet")
    cifar = module.create_model_config("cifar10")
    assert (config["num_classes"], config["input_shape"]) == (1000, (32, 32, 3))
    assert {k: v for k, v in config.items() if k not in ("num_classes", "input_shape")} == {
        k: v for k, v in cifar.items() if k not in ("num_classes", "input_shape")}
    assert module.create_model_config("imagenet", 48)["input_shape"] == (48, 48, 3)


def test_the_module_docstring_routing_size_is_true() -> None:
    """65M routing weights: 512 primary capsules x 1000 classes x (16 x 8)."""
    model = create_capsnet(**module.create_model_config("imagenet"))
    model(tf.zeros((1, 32, 32, 3)), training=False)
    assert tuple(model.digit_caps.W.shape) == (1, 512, 1000, 16, 8)
    assert int(np.prod(model.digit_caps.W.shape)) == 65_536_000
    assert "about 65M routing" in (module.__doc__ or "").replace("\n", " ")


@pytest.mark.parametrize("dataset,expected", [("mnist", 64), ("cifar10", 64), ("imagenet", 32)])
def test_the_batch_size_default_is_per_dataset(monkeypatch, dataset, expected) -> None:
    seen = {}
    monkeypatch.setattr(module, "train_model", lambda args: seen.update(batch=args.batch_size))
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset", dataset])
    module.main()
    assert seen["batch"] == expected


def test_an_explicit_batch_size_wins(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(module, "train_model", lambda args: seen.update(batch=args.batch_size))
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset", "imagenet", "--batch-size", "7"])
    module.main()
    assert seen["batch"] == 7


def test_the_imagenet_flags_reach_the_namespace(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(module, "train_model", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--dataset", "imagenet", "--imagenet-root", "/a", "--devkit-dir", "/b",
        "--holdout-per-class", "13", "--eval-samples", "777", "--max-samples", "321",
        "--image-size", "48"])
    module.main()
    assert (seen["imagenet_root"], seen["devkit_dir"], seen["holdout_per_class"],
            seen["eval_samples"], seen["max_samples"], seen["image_size"]) == (
        "/a", "/b", 13, 777, 321, 48)
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset", "imagenet"])
    module.main()
    assert (seen["holdout_per_class"], seen["eval_samples"], seen["image_size"]) == (None, 2000, 32)


def test_the_callbacks_monitor_a_metric_the_model_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    callbacks, _ = module.create_capsnet_callbacks(
        "t", (np.zeros((2, 28, 28, 1), "float32"), np.zeros((2, 10), "float32")), "mnist")
    stops = [c for c in callbacks if isinstance(c, keras.callbacks.EarlyStopping)]
    assert [c.monitor for c in stops] == ["val_loss"]
    assert stops[0].mode == "min"


def test_the_imagenet_run_has_no_per_epoch_weight_analysis_and_the_small_datasets_keep_it(
        run, tmp_path, monkeypatch) -> None:
    """The analyzer on the 65M-weight routing tensor dominated a smoke run's time and memory."""
    names = {type(cb).__name__ for cb in run.seen.fit.callbacks}
    assert "EpochAnalyzerCallback" not in names
    assert not (run.results_dir / "epoch_analysis").exists()
    monkeypatch.chdir(tmp_path)
    callbacks, _ = module.create_capsnet_callbacks(
        "t", (np.zeros((2, 28, 28, 1), "float32"), np.zeros((2, 10), "float32")), "mnist")
    assert "EpochAnalyzerCallback" in {type(cb).__name__ for cb in callbacks}
