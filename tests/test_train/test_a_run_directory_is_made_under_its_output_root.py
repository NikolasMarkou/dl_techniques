"""Two trainers that made their run directory under a cwd-relative ``results/`` whatever they were told (step 13.9.2 of plan
98b866fb, review-iter-2-pass2 follow-up): ``som_nd_soft/train_cifar.py`` never handed ``config.output_dir`` to
``create_callbacks``, and ``yolo12/train_multitask.py`` hardcoded ``results/`` with no flag to change it. Each run here starts
in an empty working directory and is given a separate output root; the run directory must appear under the root and nothing
under the working directory. The default root of both stays ``results`` (relative to the cwd, as in every trainer here).
"""

import argparse
from types import SimpleNamespace

import keras
import numpy as np
import pytest

import train.som_nd_soft.train_cifar as som_cifar
import train.yolo12.train_multitask as yolo12_multitask


class Stopped(Exception):
    """Raised by a spy once the run directory is known: nothing needs to train."""


@pytest.fixture
def roots(monkeypatch, tmp_path):
    """``(cwd, out)``: an empty working directory and a separate output root."""
    cwd, out = tmp_path / "cwd", tmp_path / "out"
    cwd.mkdir()
    out.mkdir()
    monkeypatch.chdir(cwd)
    return cwd, out


def test_som_cifar_makes_its_run_directory_under_config_output_dir(monkeypatch, roots):
    cwd, out = roots
    images = np.zeros((8, 32, 32, 3), "float32")
    monkeypatch.setattr(som_cifar, "load_cifar_data", lambda config: SimpleNamespace(
        x_train=images, x_select=images[:4], x_test=images[:4], y_test=np.zeros((4,), "int64")))
    monkeypatch.setattr(som_cifar, "create_cifar_som_autoencoder", lambda config: keras.Sequential(
        [keras.Input((32, 32, 3)), keras.layers.Conv2D(3, 1)]))
    made = {}
    real = som_cifar.create_callbacks

    def spy(**kwargs):
        made["callbacks"], made["dir"] = real(**kwargs)
        raise Stopped

    monkeypatch.setattr(som_cifar, "create_callbacks", spy)
    with pytest.raises(Stopped):
        som_cifar.train_cifar_som_autoencoder(som_cifar.CIFARSOMConfig(
            output_dir=str(out), experiment_name="som", epochs=1, batch_size=4))
    assert sorted(p.name for p in cwd.iterdir()) == [], "the run wrote under the working directory"
    assert [p.name for p in out.iterdir()] == [made["dir"].rsplit("/", 1)[-1]]
    assert made["dir"].startswith(str(out) + "/") and made["dir"].rsplit("/", 1)[-1].startswith("cifar_softsom_som_")


def test_som_cifar_default_root_is_results():
    assert som_cifar.CIFARSOMConfig().output_dir == "results"


def test_yolo12_multitask_makes_its_run_directory_under_output_dir(monkeypatch, roots):
    cwd, out = roots
    monkeypatch.setattr(yolo12_multitask, "setup_gpu", lambda *a, **k: None)

    def stop(*args, **kwargs):
        raise Stopped

    monkeypatch.setattr(yolo12_multitask, "create_dataset_splits", stop)
    with pytest.raises(Stopped):
        yolo12_multitask.train_model(argparse.Namespace(
            optimizer="adamw", weight_decay=1e-4, tasks=["detection"], scale="n", data_dir=str(cwd / "data"),
            patch_size=64, seed=0, output_dir=str(out)))
    assert sorted(p.name for p in cwd.iterdir()) == [], "the run wrote under the working directory"
    made = [p.name for p in out.iterdir()]
    assert len(made) == 1 and made[0].startswith("yolov12_multitask_n_detection_"), made


def test_yolo12_multitask_output_dir_flag_defaults_to_results(monkeypatch):
    """The flag exists and keeps the old location by default (``main`` is driven up to ``train_model``)."""
    seen = {}

    def capture(args):
        seen.update(vars(args))

    monkeypatch.setattr(yolo12_multitask, "train_model", capture)
    monkeypatch.setattr(yolo12_multitask, "set_seeds", lambda seed: None)
    monkeypatch.setattr("sys.argv", ["train_multitask", "--data-dir", "."])
    yolo12_multitask.main()
    assert seen["output_dir"] == "results"
    monkeypatch.setattr("sys.argv", ["train_multitask", "--data-dir", ".", "--output-dir", "elsewhere"])
    yolo12_multitask.main()
    assert seen["output_dir"] == "elsewhere"
