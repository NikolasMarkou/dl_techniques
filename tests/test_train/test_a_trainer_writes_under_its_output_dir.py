"""A trainer's run directory is the one its output flag names.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-094). At 561e02165:

- ``train/mobilenet/train_mobilenet.py``, ``train/cliffordnet/train_cliffordnet.py``
  and ``train/darkir/train_darkir.py`` built their parser on
  ``create_base_argument_parser()`` (no output flag) and called
  ``create_callbacks()`` with its default ``output_root="results"``, and
  ``train/cliffordnet/train_downsampling_techniques.py`` hardcoded
  ``os.path.join("results", ...)``: a run could only write into ``results/``
  under the working directory.
- ``train/resnet/train_resnet.py`` and ``train/depth_anything/train_depth_anything.py``
  took ``--output-dir`` but built their callbacks without it, so the checkpoint,
  the CSV log and the analyzer output went to a second, timestamped directory
  under ``results/`` (measured: a ResNet run with ``--output-dir`` in the
  scratchpad added ``results/resnet_imagenette64_20261016_115218``), and
  Depth Anything then overwrote ``config.output_dir`` with that directory.

The first four tests run the real ``main()`` with ``--output-dir`` up to the
point where the run directory is chosen, with the data, the model and the
callbacks stubbed, and read the directory the trainer asked for. The last two
call each trainer's own ``create_callbacks`` with the shared helper stubbed.
"""

import sys
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

from tests.classification_fakes import use_raw_from_unit, use_tiny_raw


class _Stop(Exception):
    """Raised by the stubbed ``create_callbacks`` once it has its arguments."""


def _tiny_classifier(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(input_shape)
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    return keras.Model(inputs, keras.layers.Dense(num_classes)(x))


def _tiny_cifar(*_args, **_kwargs):
    rng = np.random.default_rng(0)
    x = rng.random((8, 32, 32, 3), dtype=np.float32)
    y = rng.integers(0, 10, size=(8,))
    return (x, y), (x, y), (32, 32, 3), 10


def _capture_callbacks(monkeypatch, module):
    seen = {}

    def fake_create_callbacks(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(module, "create_callbacks", fake_create_callbacks)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    return seen


def _run_main(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(_Stop):
        module.main()


def test_mobilenet_writes_under_its_output_dir(monkeypatch, tmp_path):
    from train.mobilenet import train_mobilenet as module

    seen = _capture_callbacks(monkeypatch, module)
    use_tiny_raw(monkeypatch)
    monkeypatch.setattr(
        module, "create_model", lambda *a, **k: _tiny_classifier()
    )
    monkeypatch.setattr(keras.Model, "summary", lambda *a, **k: None)
    _run_main(
        monkeypatch, module,
        ["--epochs", "1", "--batch-size", "4", "--output-dir", str(tmp_path)],
    )
    assert seen["output_root"] == str(tmp_path)


def test_cliffordnet_writes_under_its_output_dir(monkeypatch, tmp_path):
    from train.cliffordnet import train_cliffordnet as module

    seen = _capture_callbacks(monkeypatch, module)
    use_raw_from_unit(monkeypatch, _tiny_cifar)
    monkeypatch.setattr(
        module, "build_model", lambda **k: _tiny_classifier(num_classes=10)
    )
    _run_main(
        monkeypatch, module,
        ["--dataset", "cifar10", "--epochs", "1", "--batch-size", "4",
         "--output-dir", str(tmp_path)],
    )
    assert seen["output_root"] == str(tmp_path)


def test_darkir_writes_under_its_output_dir(monkeypatch, tmp_path):
    from train.darkir import train_darkir as module

    seen = _capture_callbacks(monkeypatch, module)

    def tiny_pairs(*_args, **_kwargs):
        x = np.zeros((2, 32, 32, 3), dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, x)).batch(2)
        return ds, ds, 2

    def tiny_restorer(**_kwargs):
        inputs = keras.Input((None, None, 3))
        return keras.Model(inputs, keras.layers.Conv2D(3, 1)(inputs))

    monkeypatch.setattr(module, "load_restoration_dataset", tiny_pairs)
    monkeypatch.setattr(module, "create_darkir_model", tiny_restorer)
    _run_main(
        monkeypatch, module,
        ["--img-size", "32", "--epochs", "1", "--perceptual-weight", "0",
         "--output-dir", str(tmp_path)],
    )
    assert seen["output_root"] == str(tmp_path)


def test_the_downsampling_sweep_writes_under_its_output_dir(monkeypatch, tmp_path):
    from train.cliffordnet import train_downsampling_techniques as module

    # A run that ignored the flag would write ``results/`` here, not into
    # the repository.
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    roots = []

    def fake_train_one_variant(variant_name, args, run_root, *rest):
        roots.append(run_root)
        return {
            "variant": variant_name, "parameters": 1,
            "best_val_accuracy": 0.0, "final_val_accuracy": 0.0,
            "best_val_top5_accuracy": 0.0, "wall_seconds": 0.0,
            "epochs_trained": 1, "results_dir": run_root,
        }

    use_raw_from_unit(monkeypatch, _tiny_cifar)
    monkeypatch.setattr(module, "train_one_variant", fake_train_one_variant)
    args = module._parse_args(
        ["--variant", "V1_3stage_strided_conv", "--output-dir", str(out)]
    )
    module.train_all_variants(args)
    assert len(roots) == 1
    assert roots[0].startswith(str(out))
    assert not (tmp_path / "results").exists()


def _capture_common_callbacks(monkeypatch, module):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return [], kwargs.get("run_dir", "unset")

    monkeypatch.setattr(module, "create_common_callbacks", fake)
    return seen


def test_resnet_callbacks_write_into_its_run_dir(monkeypatch, tmp_path):
    from train.resnet import train_resnet as module

    seen = _capture_common_callbacks(monkeypatch, module)
    config = module.TrainingConfig(
        imagenet_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        experiment_name="run",
    )
    _, results_dir = module.create_callbacks(config, 1)
    assert seen["run_dir"] == str(tmp_path / "out" / "run")
    assert results_dir == str(tmp_path / "out" / "run")


def test_depth_anything_callbacks_write_into_its_run_dir(monkeypatch, tmp_path):
    from train.depth_anything import train_depth_anything as module

    seen = _capture_common_callbacks(monkeypatch, module)
    monkeypatch.setattr(module, "_load_validation_samples", lambda *a, **k: None)
    config = module.DepthAnythingTrainingConfig(
        output_dir=str(tmp_path / "out"), experiment_name="run",
    )
    _, results_dir = module.create_callbacks(config, [], [])
    assert seen["run_dir"] == str(Path(tmp_path) / "out" / "run")
    assert results_dir == seen["run_dir"]
    # The config keeps the directory it was given.
    assert config.output_dir == str(tmp_path / "out")
    assert config.experiment_name == "run"
