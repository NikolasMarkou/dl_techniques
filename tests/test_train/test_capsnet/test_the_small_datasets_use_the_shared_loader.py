"""``train_capsnet --dataset mnist`` reads its data through the shared loader.

The raw arrays are a fake (``classification_data._load_raw``), so nothing touches the Keras
cache. What is under test: the selection split is a held-out slice of TRAIN and never the
test set (the old trainer sliced the first 10% of train unshuffled and, with
``--validation-split 0``, validated on the test set), the labels reaching ``fit`` are
one-hot, the pixels stay in [0, 1], and the test metrics use the whole test split.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.common.classification_data as classification_data  # noqa: E402
from train.capsnet import train_capsnet as module  # noqa: E402

N_TRAIN, N_TEST, MAX_SAMPLES, BATCH = 120, 60, 80, 8


def _fake_raw(name):
    assert name == "mnist"
    rng = np.random.default_rng(0)

    def split(n):
        return (rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8),
                rng.integers(0, 10, size=n).astype(np.int64))

    return split(N_TRAIN), split(N_TEST)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    work = tmp_path_factory.mktemp("capsnet_mnist")
    seen = SimpleNamespace(fit=None, evaluated=[])
    real_fit, real_evaluate = keras.Model.fit, keras.Model.evaluate

    def count(dataset):
        return sum(len(x) for x, _ in dataset)

    def fit(self, x=None, y=None, *args, **kwargs):
        batches = list(x)
        seen.fit = SimpleNamespace(
            y_is_none=y is None, images=np.concatenate([b[0].numpy() for b in batches]),
            labels=np.concatenate([b[1].numpy() for b in batches]),
            validation=count(kwargs["validation_data"]))
        return real_fit(self, x, y, *args, **kwargs)

    def evaluate(self, x=None, *args, **kwargs):
        seen.evaluated.append(count(x))
        return real_evaluate(self, x, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work)
        mp.setattr(classification_data, "_load_raw", _fake_raw)
        mp.setattr(module, "setup_gpu", lambda *a, **k: None)
        mp.setattr(keras.Model, "fit", fit)
        mp.setattr(keras.Model, "evaluate", evaluate)
        mp.setattr(sys, "argv", [
            "prog", "--dataset", "mnist", "--epochs", "1", "--batch-size", str(BATCH),
            "--max-samples", str(MAX_SAMPLES), "--eval-samples", "8", "--patience", "2"])
        module.main()
    (results_dir,) = (work / "results").iterdir()
    return SimpleNamespace(results_dir=results_dir, seen=seen)


def test_fit_gets_one_hot_labels_unit_pixels_and_a_train_slice_without_the_selection_rows(run):
    fit = run.seen.fit
    n_select = int(MAX_SAMPLES * 0.1)
    n_train = MAX_SAMPLES - n_select
    assert fit.y_is_none
    assert fit.images.shape == ((n_train // BATCH) * BATCH, 28, 28, 1)
    assert fit.images.min() >= 0.0 and fit.images.max() <= 1.0
    assert fit.labels.shape == (len(fit.images), 10) and (fit.labels.sum(axis=1) == 1).all()
    assert fit.validation == n_select


def test_the_test_metrics_use_the_whole_test_split_and_selection_never_does(run):
    # fit evaluates the selection slice once per epoch (8 images), then the test split whole
    # (60 images: the 80-image cap is above its size).
    assert run.seen.evaluated == [8, N_TEST]


def test_the_run_writes_its_artifacts(run):
    for name in ("capsnet_mnist_final.keras", "training_summary.txt", "training_log.csv",
                 "best_model.keras"):
        assert (run.results_dir / name).is_file(), name
