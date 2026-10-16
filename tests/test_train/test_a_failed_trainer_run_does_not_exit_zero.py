"""A trainer whose run failed must not exit 0.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-095). Measured at
561e02165:

- ``train/darkir/train_darkir.py::main`` caught every ``Exception`` from
  ``train_model``, logged it, and returned, so a crashed run exited 0.
- ``train/cliffordnet/train_downsampling_techniques.py::train_all_variants``
  logged a variant whose build or training failed (``train_one_variant``
  returns ``None``, or raises) and went on to the next, and returned normally
  even when every variant had failed.

The sweep still trains every variant it can and writes the comparison of the
ones that trained; it raises at the end, naming the failed variants.

Step 13.5 (D-102) extends the guard to the end of a run: a failed final save,
history write or summary write raises instead of being logged (seven trainers,
below).
"""

import sys

import numpy as np
import pytest


def _tiny_cifar(*_args, **_kwargs):
    x = np.zeros((4, 32, 32, 3), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int64)
    return (x, y), (x, y), (32, 32, 3), 100


def test_darkir_main_propagates_a_training_error(monkeypatch):
    from train.darkir import train_darkir as module

    def crash(_args):
        raise RuntimeError("the run crashed")

    monkeypatch.setattr(module, "train_model", crash)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(RuntimeError, match="the run crashed"):
        module.main()


def _sweep(monkeypatch, tmp_path, outcome):
    from train.cliffordnet import train_downsampling_techniques as module

    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_train_one_variant(variant_name, args, run_root, *rest):
        calls.append(variant_name)
        return outcome(variant_name, run_root)

    monkeypatch.setattr(module, "load_dataset", _tiny_cifar)
    monkeypatch.setattr(module, "train_one_variant", fake_train_one_variant)
    args = module._parse_args(["--output-dir", str(tmp_path / "out")])
    return module, args, calls


def _row(variant_name, run_root):
    return {
        "variant": variant_name, "parameters": 1,
        "best_val_accuracy": 0.5, "final_val_accuracy": 0.5,
        "best_val_top5_accuracy": 0.5, "wall_seconds": 0.0,
        "epochs_trained": 1, "results_dir": run_root,
    }


def test_a_sweep_where_a_variant_returns_none_raises(monkeypatch, tmp_path):
    def outcome(name, run_root):
        return None if name == "V2_3stage_avgpool" else _row(name, run_root)

    module, args, calls = _sweep(monkeypatch, tmp_path, outcome)
    with pytest.raises(RuntimeError, match="V2_3stage_avgpool"):
        module.train_all_variants(args)
    # Every variant was still attempted.
    assert calls == list(module.VARIANTS)


def test_a_sweep_where_a_variant_raises_raises(monkeypatch, tmp_path):
    def outcome(name, run_root):
        if name == "V3_3stage_patch_merging":
            raise ValueError("boom")
        return _row(name, run_root)

    module, args, calls = _sweep(monkeypatch, tmp_path, outcome)
    with pytest.raises(RuntimeError, match="V3_3stage_patch_merging"):
        module.train_all_variants(args)
    assert calls == list(module.VARIANTS)


def test_a_sweep_where_every_variant_trains_returns(monkeypatch, tmp_path):
    module, args, calls = _sweep(monkeypatch, tmp_path, _row)
    module.train_all_variants(args)
    assert calls == list(module.VARIANTS)


# ---------------------------------------------------------------------------
# A failed final save, history or summary write fails the run (step 13.5, D-102)
# ---------------------------------------------------------------------------
#
# Measured at ffa4c2870: ``bit_diffusion``, ``sam``, ``sam2``, ``sam3`` and
# ``dit`` caught every ``Exception`` from the final ``model.save`` and logged
# it; ``bit_diffusion`` and ``sam3`` did the same for their history JSON;
# ``depth_anything`` and ``omnipoint`` logged a failed save (and Depth
# Anything a failed summary) as a warning. Each run then exited 0 with the
# artefact missing. Each case below runs the trainer's REAL ``train_*`` body
# with data, model and fit stubbed, so the only real code is the trainer's own
# control flow after ``fit``.

import builtins
import collections
import types

import keras


class _DiskFull(OSError):
    """The error every failing write in this section raises."""


class _StubModel:
    """Model stand-in: ``fit`` returns a two-epoch history, ``save`` may fail."""

    def __init__(self, fail_save):
        self.fail_save = fail_save
        self.saved = []

    def compile(self, *args, **kwargs):
        return None

    def summary(self, *args, **kwargs):
        return None

    def count_params(self):
        return 1

    def fit(self, *args, **kwargs):
        history = collections.defaultdict(lambda: [1.0, 0.5])
        history.update({"loss": [1.0, 0.5], "val_loss": [1.0, 0.5]})
        return types.SimpleNamespace(history=history)

    def save(self, path, *args, **kwargs):
        if self.fail_save:
            raise _DiskFull(f"disk full writing {path}")
        self.saved.append(str(path))


def _failing_open(name):
    """``open`` that raises for one file name and is the builtin otherwise."""

    def fake_open(file, *args, **kwargs):
        if str(file).endswith(name):
            raise _DiskFull(f"disk full writing {file}")
        return builtins.open(file, *args, **kwargs)

    return fake_open


class _Sized:
    def __init__(self, n=2):
        self.n = n

    def __len__(self):
        return self.n


def _bit_diffusion(monkeypatch, tmp_path, model):
    from train.bit_diffusion import train_bit_diffusion as module

    monkeypatch.setattr(module, "build_sde", lambda config: None)
    monkeypatch.setattr(module, "create_model", lambda config: model)
    monkeypatch.setattr(module, "create_datasets", lambda config, sde: (None, None))
    monkeypatch.setattr(module, "build_optimizer", lambda *a, **k: None)
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_bit_diffusion(config)


def _sam(monkeypatch, tmp_path, model):
    from train.sam import train_sam as module

    monkeypatch.setattr(module, "create_dataset", lambda config: (None, None))
    monkeypatch.setattr(module, "create_training_model", lambda config: model)
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_sam(config)


def _sam2(monkeypatch, tmp_path, model):
    from train.sam2 import train_sam2 as module

    monkeypatch.setattr(module, "create_dataset", lambda config: (None, None))
    monkeypatch.setattr(module, "create_sam2_training_model", lambda config: model)
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_sam2(config)


def _sam3(monkeypatch, tmp_path, model):
    from train.sam3 import train_sam3 as module

    monkeypatch.setattr(module, "create_training_model", lambda config: model)
    monkeypatch.setattr(module, "create_datasets",
                        lambda config, m, **kwargs: (None, None))
    monkeypatch.setattr(module, "build_callbacks", lambda *a: [])
    monkeypatch.setattr(module, "evaluate_sam3",
                        lambda m, ds: collections.defaultdict(lambda: 1.0))
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_sam3(config)


def _dit(monkeypatch, tmp_path, model):
    from train.dit import train_dit as module

    class _StubEMA(keras.callbacks.Callback):
        updates, decay = 0, 0.9999

        def __init__(self, decay):
            super().__init__()

        def shadow_values(self):
            return []

    monkeypatch.setattr(module, "create_model", lambda config: model)
    monkeypatch.setattr(module, "create_datasets", lambda config: (None, None))
    monkeypatch.setattr(module, "build_dit_optimizer", lambda *a, **k: None)
    monkeypatch.setattr(module, "DDPMHybridLoss", lambda **kwargs: None)
    monkeypatch.setattr(module, "WeightEMACallback", _StubEMA)
    config = module.config_from_argv(["--output-dir", str(tmp_path)])
    return module, lambda: module.train_dit(config)


def _depth_anything(monkeypatch, tmp_path, model):
    from train.depth_anything import train_depth_anything as module

    paths = [str(tmp_path / f"p{i}") for i in range(20)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(module, "discover_megadepth_pairs",
                        lambda *a, **k: (paths, paths))
    monkeypatch.setattr(module, "MegaDepthDataset", lambda *a, **k: _Sized())
    monkeypatch.setattr(module, "create_model", lambda config: model)
    monkeypatch.setattr(module, "create_callbacks",
                        lambda *a, **k: ([], str(run_dir)))
    config = module.DepthAnythingTrainingConfig(
        output_dir=str(tmp_path), experiment_name="run", epochs=2,
    )
    return module, lambda: module.train_depth_anything(config)


def _omnipoint(monkeypatch, tmp_path, model):
    from train.omnipoint import train_omnipoint as module

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(module, "build_datasets",
                        lambda config: (_Sized(), _Sized()))
    monkeypatch.setattr(module, "create_model", lambda config: (model, model))
    monkeypatch.setattr(module, "create_callbacks",
                        lambda *a, **k: ([], str(run_dir)))
    config = module._config_from_args(module.parse_arguments(
        ["--smoke", "--output-dir", str(tmp_path)]))
    config.show_plots = False
    return module, lambda: module.train_omnipoint(config)


_TRAINERS = {
    "bit_diffusion": _bit_diffusion,
    "sam": _sam,
    "sam2": _sam2,
    "sam3": _sam3,
    "dit": _dit,
    "depth_anything": _depth_anything,
    "omnipoint": _omnipoint,
}


@pytest.mark.parametrize("trainer", sorted(_TRAINERS))
def test_a_run_whose_final_save_fails_raises(monkeypatch, tmp_path, trainer):
    model = _StubModel(fail_save=True)
    _module, run = _TRAINERS[trainer](monkeypatch, tmp_path, model)
    with pytest.raises(_DiskFull, match="disk full"):
        run()


@pytest.mark.parametrize("trainer", sorted(_TRAINERS))
def test_a_run_whose_writes_succeed_returns_and_saves(
        monkeypatch, tmp_path, trainer):
    # The stubs are not what makes the failing case raise.
    model = _StubModel(fail_save=False)
    _module, run = _TRAINERS[trainer](monkeypatch, tmp_path, model)
    run()
    assert len(model.saved) == 1, model.saved


@pytest.mark.parametrize("trainer, artefact", [
    ("bit_diffusion", "training_history.json"),
    ("sam3", "training_history.json"),
    ("depth_anything", "training_summary.txt"),
])
def test_a_run_whose_history_or_summary_write_fails_raises(
        monkeypatch, tmp_path, trainer, artefact):
    model = _StubModel(fail_save=False)
    module, run = _TRAINERS[trainer](monkeypatch, tmp_path, model)
    monkeypatch.setattr(module, "open", _failing_open(artefact), raising=False)
    with pytest.raises(_DiskFull, match=artefact):
        run()
    # The model is saved before the reporting write, so a failed report never
    # costs the weights.
    assert len(model.saved) == 1, model.saved
