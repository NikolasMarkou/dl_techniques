"""``train_capsnet`` fits conformal prediction sets on the whole TEST split.

CapsNet emits capsule lengths in [0, 1], not logits, so the exported model wraps it with
``ProbabilityLogits`` before the set layer's softmax. One fake-ImageNet run (3 classes, 6
test images) is checked for the files, the calibration set, the flag in the record and the
exported model; ``fit_conformal_sets`` is called directly on a small model for the array
datasets (MNIST/CIFAR path) and for ``--validation-split 0``, where the "validation" data is
the test split and the record must say the model was chosen on the calibration data.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dl_techniques.models.vision.capsnet import create_capsnet  # noqa: E402
from tests.imagenet_fake_tree import VAL_CLASSES, build_fake_imagenet  # noqa: E402
from train.capsnet import train_capsnet as module  # noqa: E402
from train.common.conformal import (  # noqa: E402
    CAVEAT_NOT_SELECTED_ON_CALIBRATION,
    CAVEAT_SELECTED_ON_CALIBRATION,
    CONFORMAL_JSON_NAME,
    CONFORMAL_MODEL_NAME,
    load_conformal_model,
)

N_CLASSES = 3
BATCH = 4


def _argv(tree, *extra):
    return ["prog", "--dataset", "imagenet", "--imagenet-root", tree, "--image-size", "32",
            "--epochs", "1", "--batch-size", str(BATCH), "--holdout-per-class", "1",
            "--eval-samples", "4", "--patience", "5", "--conformal-alpha", "0.3", *extra]


def _run(tree, work, *extra):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work)
        mp.setattr(module, "IMAGENET_NUM_CLASSES", N_CLASSES)
        mp.setattr(module, "setup_gpu", lambda *a, **k: None)
        mp.setattr(sys, "argv", _argv(tree, *extra))
        module.main()
    (results_dir,) = (work / "results").iterdir()
    return results_dir


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tree = build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")
    seen = SimpleNamespace(datasets=[], kwargs=[], classifier=None)
    real = module.run_conformal

    def spy(model, dataset, *args, **kwargs):
        seen.datasets.append(sum(len(x) for x, _ in dataset))
        seen.labels = np.concatenate([y.numpy() for _, y in dataset])
        seen.kwargs.append(kwargs)
        seen.classifier = model
        return real(model, dataset, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, "run_conformal", spy)
        on = _run(tree, tmp_path_factory.mktemp("on"))
    off = _run(tree, tmp_path_factory.mktemp("off"), "--no-conformal")
    return SimpleNamespace(on=on, off=off, seen=seen)


def test_an_imagenet_run_writes_the_record_and_the_exported_model(runs) -> None:
    record = json.loads((runs.on / CONFORMAL_JSON_NAME).read_text())
    assert record["alpha"] == 0.3 and record["model"] == "capsnet_imagenet"
    assert (runs.on / CONFORMAL_MODEL_NAME).is_file()


def test_the_calibration_set_is_the_whole_test_split_with_integer_labels(runs) -> None:
    assert runs.seen.datasets == [len(VAL_CLASSES)]
    assert runs.seen.labels.tolist() == VAL_CLASSES
    assert runs.seen.labels.ndim == 1
    record = json.loads((runs.on / CONFORMAL_JSON_NAME).read_text())
    assert record["n_calibration"] == len(VAL_CLASSES)


def test_the_record_says_the_model_was_not_chosen_on_the_calibration_data(runs) -> None:
    assert runs.seen.kwargs[0]["selection_overlaps_calibration"] is False
    record = json.loads((runs.on / CONFORMAL_JSON_NAME).read_text())
    assert record["model_selected_on_calibration_set"] is False
    assert record["caveat"] == CAVEAT_NOT_SELECTED_ON_CALIBRATION
    assert "held-out slice of train" in record["weights"]


def test_the_exported_model_reloads_and_returns_logits_probs_and_a_mask(runs) -> None:
    model = load_conformal_model(runs.on / CONFORMAL_MODEL_NAME)
    x = np.random.default_rng(0).random((5, 32, 32, 3), dtype=np.float32)
    logits, probs, in_set = model(x)
    assert tuple(logits.shape) == tuple(probs.shape) == tuple(in_set.shape) == (5, N_CLASSES)
    assert set(np.unique(keras.ops.convert_to_numpy(in_set))) <= {0.0, 1.0}


def test_the_exported_scores_are_the_log_odds_of_the_capsule_lengths(runs) -> None:
    """The wrapper keeps the model's own class order and puts the lengths on the real line."""
    classifier = runs.seen.classifier
    assert any(isinstance(layer, module.ProbabilityLogits) for layer in classifier.layers)
    x = np.random.default_rng(1).random((6, 32, 32, 3), dtype=np.float32)
    logits = keras.ops.convert_to_numpy(classifier(x))
    # recompute the lengths from the wrapped CapsNet
    caps = next(layer for layer in classifier.layers if isinstance(layer, keras.Model))
    lengths = keras.ops.convert_to_numpy(caps(x)["length"]).astype("float64")
    p = np.clip(lengths, 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(logits, np.log(p) - np.log1p(-p), rtol=0, atol=1e-4)
    assert np.array_equal(np.argsort(logits, axis=-1), np.argsort(lengths, axis=-1))


def test_no_conformal_leaves_no_files(runs) -> None:
    assert not (runs.off / CONFORMAL_JSON_NAME).exists()
    assert not (runs.off / CONFORMAL_MODEL_NAME).exists()


def test_the_conformal_flags_are_declared(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(module, "train_model", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["prog", "--conformal-alpha", "0.05"])
    module.main()
    assert seen["conformal_alpha"] == 0.05 and seen["no_conformal"] is False
    monkeypatch.setattr(sys, "argv", ["prog", "--no-conformal"])
    module.main()
    assert seen["no_conformal"] is True and seen["conformal_alpha"] == 0.1


# ---------------------------------------------------------------------
# The array datasets and the validation-split 0 case, on a small model
# ---------------------------------------------------------------------


def _small_model():
    keras.utils.set_random_seed(0)
    model = create_capsnet(
        num_classes=4, input_shape=(32, 32, 3), conv_filters=[8], primary_capsules=4,
        primary_capsule_dim=4, digit_capsule_dim=4, reconstruction=True,
        decoder_architecture=[16], optimizer="adam", learning_rate=1e-3,
    )
    model(np.zeros((1, 32, 32, 3), dtype="float32"), training=False)
    return model


def _arrays(n=40):
    rng = np.random.default_rng(2)
    x = rng.random((n, 32, 32, 3), dtype=np.float32)
    y = keras.utils.to_categorical(rng.integers(0, 4, size=n), 4)
    return x, y


def _args(tmp_path, **overrides):
    values = dict(dataset="cifar10", conformal_alpha=0.1, validation_split=0.1, seed=1,
                  batch_size=8)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_an_array_dataset_calibrates_on_all_of_its_one_hot_test_labels(tmp_path) -> None:
    x, y = _arrays()
    record = module.fit_conformal_sets(
        _small_model(), _args(tmp_path), x, y, (32, 32, 3), str(tmp_path))
    assert record["n_calibration"] == 40
    assert record["model_selected_on_calibration_set"] is False
    assert record["caveat"] == CAVEAT_NOT_SELECTED_ON_CALIBRATION
    assert "validation_split=0.1" in record["weights"]
    assert (tmp_path / CONFORMAL_MODEL_NAME).is_file()


def test_a_validation_split_of_zero_is_recorded_as_selected_on_the_calibration_data(tmp_path) -> None:
    x, y = _arrays()
    record = module.fit_conformal_sets(
        _small_model(), _args(tmp_path, validation_split=0.0), x, y, (32, 32, 3), str(tmp_path))
    assert record["model_selected_on_calibration_set"] is True
    assert record["caveat"] == CAVEAT_SELECTED_ON_CALIBRATION
    assert "validation_split 0" in record["weights"]
