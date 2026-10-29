"""``train_cliffordnet --dataset imagenette`` reads the prepared TFDS records through the shared loader.

The records are the in-memory fakes of ``tests/classification_fakes.py`` (60 train, 20
validation, 10 classes). The model is a tiny stand-in sized from the class count, so what is
exercised is the trainer's data path: the input size, the datasets that reach ``fit``, the
head that reaches the figures and the analysis, the flags that apply to one dataset only,
and the run files.
"""

import json
import sys

import keras
import numpy as np
import pytest

from tests.classification_fakes import use_fake_imagenette
from train.cliffordnet import train_cliffordnet as module


class _Seen:
    input_shape = None
    analysis_x = None
    figure_args = None
    figure_kwargs = None
    fit_data = None


def _tiny(**kwargs):
    inputs = keras.Input((24, 24, 3))
    x = keras.layers.Conv2D(4, 3, strides=4)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, keras.layers.Dense(kwargs["num_classes"])(x))


def _patch(monkeypatch):
    use_fake_imagenette(monkeypatch)
    real = module.run_summary.write_classification_figures

    def spy(*args, **kwargs):
        _Seen.figure_args, _Seen.figure_kwargs = args, kwargs
        return real(*args, **kwargs)

    def analysis(*_a, **kwargs):
        _Seen.analysis_x = kwargs["test_data"]

    real_fit = keras.Model.fit

    def fit(self, x=None, *args, **kwargs):
        _Seen.fit_data = x
        return real_fit(self, x, *args, **kwargs)

    monkeypatch.setattr(module.run_summary, "write_classification_figures", spy)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, "build_model", _tiny)
    monkeypatch.setattr(module, "run_model_analysis", analysis)
    monkeypatch.setattr(keras.Model, "fit", fit)


def _run(monkeypatch, out, *extra):
    _patch(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--dataset", "imagenette", "--image-size", "24", "--batch-size", "6",
        "--epochs", "1", "--eval-samples", "5", "--output-dir", str(out), *extra])
    module.main()
    (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
    return run_dir


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield _run(mp, tmp_path_factory.mktemp("clifford_imagenette"))
    finally:
        mp.undo()


def test_the_figures_and_the_analysis_get_the_head_of_the_test_split(run):
    x, y = _Seen.analysis_x
    assert x.shape == (5, 24, 24, 3) and y.shape == (5,)
    _vis, fx, fy, probs, names = _Seen.figure_args
    assert fx.shape == (5, 24, 24, 3) and probs.shape == (5, 10) and len(names) == 10
    assert np.allclose(_Seen.figure_kwargs["mean"], [0.485, 0.456, 0.406])


def test_fit_sees_the_loaders_batches_at_the_asked_size(run):
    x, _ = next(iter(_Seen.fit_data))
    assert x.shape == (6, 24, 24, 3)


def test_the_config_records_the_imagenette_run(run):
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["dataset"] == "imagenette" and cfg["image_size"] == 24
    assert cfg["validation_split"] is None or 0.0 < cfg["validation_split"] < 1.0
    assert cfg["num_classes"] == 10 and cfg["input_shape"] == [24, 24, 3]
    assert cfg["augment"] == "none" and cfg["random_erasing_prob"] == 0.0
    assert "imagenet_root" not in cfg and "conformal_alpha" not in cfg


def test_the_summary_names_the_size_and_the_head(run):
    text = (run / "training_summary.txt").read_text()
    assert "Image size: 24" in text and "first 5 val images" in text
    assert "Augmentation: none" in text


def test_a_preset_trains_on_imagenette_on_top_of_the_loaders_crop(monkeypatch, tmp_path):
    run_dir = _run(monkeypatch, tmp_path / "out", "--augment", "photometric")
    assert json.loads((run_dir / "config.json").read_text())["augment"] == "photometric"


def test_eval_samples_must_be_positive_and_imagenette_takes_no_conformal_step(monkeypatch, capsys):
    monkeypatch.setattr(module, "train_model", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset", "imagenette", "--eval-samples", "0"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2 and "--eval-samples" in capsys.readouterr().err
