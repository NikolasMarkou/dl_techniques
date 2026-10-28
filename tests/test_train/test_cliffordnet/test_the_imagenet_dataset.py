"""``train_cliffordnet --dataset imagenet`` reads the extracted ImageNet through the shared loader.

The tree is a 3-class fake (``tests/imagenet_fake_tree.py``): 12 train images, 6 val.
The model is a tiny stand-in sized from the dataset's class count, so what is
exercised is the trainer's data path: the datasets that reach ``fit``, the val head
that reaches the figures and the analysis, the flags that apply to one dataset only,
and the run files.
"""

import json
import sys

import keras
import numpy as np
import pytest

from tests.imagenet_fake_tree import build_fake_imagenet
from train.cliffordnet import train_cliffordnet as module

VIS_FILES = [
    "training_dashboard.png",
    "confusion_matrix.png",
    "per_class_metrics.png",
    "confidence_calibration.png",
    "misclassifications.png",
    "classification_report.json",
]


class _Seen:
    num_classes = None
    analysis_x = None
    figure_kwargs = None
    figure_args = None
    order = None
    conformal_record = None


def _tiny(**kwargs):
    _Seen.num_classes = kwargs["num_classes"]
    inputs = keras.Input((32, 32, 3))
    x = keras.layers.Conv2D(4, 3, strides=4)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, keras.layers.Dense(kwargs["num_classes"])(x))


def _analysis(*_a, **kwargs):
    _Seen.analysis_x = kwargs["test_data"]
    _Seen.order.append("analysis")


def _patch(monkeypatch):
    _Seen.order = []
    real_conformal = module.run_conformal

    def conformal_spy(*args, **kwargs):
        _Seen.order.append("conformal")
        _Seen.conformal_record = real_conformal(*args, **kwargs)
        return _Seen.conformal_record

    monkeypatch.setattr(module, "run_conformal", conformal_spy)
    real = module.run_summary.write_classification_figures

    def spy(*args, **kwargs):
        _Seen.figure_args, _Seen.figure_kwargs = args, kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(module.run_summary, "write_classification_figures", spy)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, "build_model", _tiny)
    monkeypatch.setattr(module, "run_model_analysis", _analysis)


def _argv(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")
    out = tmp_path_factory.mktemp("clifford_imagenet")
    mp = pytest.MonkeyPatch()
    _patch(mp)
    _argv(mp, "--dataset", "imagenet", "--imagenet-root", root, "--image-size", "32",
          "--batch-size", "4", "--epochs", "1", "--eval-samples", "5",
          "--conformal-alpha", "0.3", "--output-dir", str(out))
    try:
        module.main()
    finally:
        mp.undo()
    (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
    return run_dir, root


def test_the_model_is_sized_from_the_class_folders(run):
    assert _Seen.num_classes == 3


@pytest.mark.parametrize("name", VIS_FILES)
def test_the_figures_are_written(run, name):
    assert (run[0] / "visualizations" / name).is_file()


def test_the_analysis_gets_the_val_head_not_all_of_val(run):
    x, y = _Seen.analysis_x
    assert x.shape == (5, 32, 32, 3) and y.shape == (5,)


def test_the_figures_get_the_val_head_and_the_wnids_and_can_unstandardize(run):
    _vis_dir, x, y, probs, names = _Seen.figure_args
    assert x.shape == (5, 32, 32, 3) and probs.shape == (5, 3)
    assert names == ["n0000000a", "n0000000b", "n0000000c"]
    assert np.allclose(_Seen.figure_kwargs["mean"], [0.485, 0.456, 0.406])
    assert np.allclose(_Seen.figure_kwargs["std"], [0.229, 0.224, 0.225])


def test_the_config_records_the_imagenet_run(run):
    cfg = json.loads((run[0] / "config.json").read_text())
    assert cfg["dataset"] == "imagenet"
    assert cfg["imagenet_root"] == run[1]
    assert cfg["eval_samples"] == 5
    assert cfg["conformal_alpha"] == 0.3
    assert cfg["input_shape"] == [32, 32, 3]
    assert cfg["num_classes"] == 3
    assert cfg["random_erasing_prob"] == 0.0


def test_the_summary_names_the_root_and_the_head(run):
    text = (run[0] / "training_summary.txt").read_text()
    assert f"ImageNet root: {run[1]}" in text
    assert "first 5 val images" in text


def test_the_conformal_step_sees_all_of_val_and_runs_before_the_analysis(run):
    assert _Seen.conformal_record["n_calibration"] == 6  # not the 5-image head
    assert _Seen.order == ["conformal", "analysis"]
    assert _Seen.conformal_record["weights"].startswith("best checkpoint")


def test_the_conformal_files_are_in_the_run_directory(run):
    record = json.loads((run[0] / "conformal.json").read_text())
    assert record["alpha"] == 0.3 and record["model"] == "cliffordnet_nano_imagenet"
    assert (run[0] / "conformal_model.keras").is_file()


def test_the_summary_reports_the_conformal_coverage(run):
    text = (run[0] / "training_summary.txt").read_text()
    assert "Conformal sets (alpha 0.3" in text and "on random half splits" in text


def test_no_conformal_skips_the_step(tmp_path, monkeypatch):
    root = build_fake_imagenet(tmp_path / "tree" / "imagenet2012")
    _patch(monkeypatch)
    _argv(monkeypatch, "--dataset", "imagenet", "--imagenet-root", root, "--image-size", "32",
          "--batch-size", "4", "--epochs", "1", "--eval-samples", "5", "--no-conformal",
          "--output-dir", str(tmp_path / "out"))
    module.main()
    (run_dir,) = [p for p in (tmp_path / "out").iterdir() if p.is_dir()]
    assert not (run_dir / "conformal.json").exists()
    assert _Seen.order == ["analysis"]
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["conformal_alpha"] is None


def test_an_alpha_outside_the_open_unit_interval_is_refused(monkeypatch, capsys):
    _argv(monkeypatch, "--dataset", "imagenet", "--conformal-alpha", "1.0")
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


# ---------------------------------------------------------------------
# Flags that apply to one dataset only
# ---------------------------------------------------------------------


def _parsed(monkeypatch, *argv):
    seen = {}
    monkeypatch.setattr(module, "train_model", lambda args: seen.update(vars(args)))
    _argv(monkeypatch, *argv)
    module.main()
    return seen


def test_imagenet_resolves_its_defaults(monkeypatch):
    got = _parsed(monkeypatch, "--dataset", "imagenet")
    assert got["eval_samples"] == 2000
    assert got["random_erasing_prob"] == 0.0
    assert got["imagenet_root"].endswith("imagenet2012") and got["devkit_dir"] is None


def test_cifar_keeps_random_erasing_0_25(monkeypatch):
    got = _parsed(monkeypatch, "--dataset", "cifar10")
    assert got["random_erasing_prob"] == 0.25 and got["eval_samples"] is None


def _no_training(monkeypatch):
    """A refusal that fails to fire must stop here, not train on the real dataset."""
    def reached(args):
        raise AssertionError("train_model was reached")
    monkeypatch.setattr(module, "train_model", reached)


def test_random_erasing_is_refused_with_imagenet(monkeypatch, capsys):
    _no_training(monkeypatch)
    _argv(monkeypatch, "--dataset", "imagenet", "--random-erasing-prob", "0.1")
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "--random-erasing-prob" in capsys.readouterr().err


def test_eval_samples_is_refused_with_cifar(monkeypatch, capsys):
    _no_training(monkeypatch)
    _argv(monkeypatch, "--dataset", "cifar10", "--eval-samples", "10")
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "--eval-samples" in capsys.readouterr().err


def test_a_non_positive_eval_samples_is_refused(monkeypatch, capsys):
    _no_training(monkeypatch)
    _argv(monkeypatch, "--dataset", "imagenet", "--eval-samples", "0")
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_a_missing_root_is_refused_before_any_work(monkeypatch, tmp_path):
    _patch(monkeypatch)
    _argv(monkeypatch, "--dataset", "imagenet", "--imagenet-root",
          str(tmp_path / "nope"), "--output-dir", str(tmp_path))
    with pytest.raises(ValueError, match="ImageNet root does not exist"):
        module.main()
    assert not any(tmp_path.iterdir())


def test_the_help_lists_the_imagenet_flags(monkeypatch, capsys):
    _argv(monkeypatch, "--help")
    with pytest.raises(SystemExit):
        module.main()
    out = capsys.readouterr().out
    for flag in ("--imagenet-root", "--devkit-dir", "--eval-samples",
                 "--conformal-alpha", "--no-conformal"):
        assert flag in out
