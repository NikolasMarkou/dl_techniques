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
