"""The CliffordNet trainer's defaults are the training recipe of paper v2.

Guard for plan-2026-10-23T100412-e7eaaa02 (D-007, D-008). Sources: arXiv:2601.06793v2
section 4 ("200 epochs, AdamW, cosine annealing, AutoAugment, Random Erasing",
DropPath, and "without using warm-up training epochs") and the authors' `train.py`
and `networks.py` (read through a fetch summary on 2026-10-23, not byte-checked):
lr 1e-3, weight decay 0.1, batch 128, random erasing 0.25, drop path 0.3 and 0.4
for cliffordnet_64_5. Two known differences remain, on purpose: our cosine ends at
1e-5 (alpha 0.01) where theirs ends at 0, and ours decays per step, theirs per epoch.
"""

import sys

import keras
import numpy as np
import pytest

from train.cliffordnet import train_cliffordnet as module


class _Stop(Exception):
    """Raised by a stub once it has captured what the test reads."""


def _parsed_args(monkeypatch, *argv):
    seen = {}
    monkeypatch.setattr(module, "train_model", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()
    return seen


def test_the_defaults_are_the_paper_recipe(monkeypatch):
    got = _parsed_args(monkeypatch)
    assert got["dataset"] == "cifar100"
    assert got["variant"] == "nano"
    assert got["epochs"] == 200
    assert got["batch_size"] == 128
    assert got["learning_rate"] == 1e-3
    assert got["weight_decay"] == 0.1
    assert got["warmup_epochs"] == 0
    assert got["stochastic_depth_rate"] == 0.3
    assert got["random_erasing_prob"] == 0.25
    assert got["cli_mode"] == "full"
    assert got["ctx_mode"] == "diff"
    assert got["layer_scale_init"] == 1e-5
    assert got["use_global_context"] is False


@pytest.mark.parametrize(
    "variant,rate",
    [("nano", 0.3), ("lite", 0.3), ("cn_32_3", 0.3), ("cn_64_5", 0.4)],
)
def test_the_drop_path_default_follows_the_variant(monkeypatch, variant, rate):
    got = _parsed_args(monkeypatch, "--variant", variant)
    assert got["stochastic_depth_rate"] == rate


def test_an_explicit_drop_path_rate_wins(monkeypatch):
    got = _parsed_args(
        monkeypatch, "--variant", "cn_64_5", "--stochastic-depth-rate", "0.1"
    )
    assert got["stochastic_depth_rate"] == 0.1


def test_the_variant_choices_are_the_model_presets(monkeypatch, capsys):
    from dl_techniques.models.vision.cliffordnet import CliffordNet

    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit):
        module.main()
    help_text = capsys.readouterr().out
    for name in [*CliffordNet.MODEL_VARIANTS, "custom"]:
        assert name in help_text


def test_the_schedule_starts_at_the_peak_rate_without_warm_up(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.random((16, 32, 32, 3), dtype=np.float32)
    y = rng.integers(0, 10, size=(16,))
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: (
        (x, y), (x, y), (32, 32, 3), 10))

    def tiny(**_kwargs):
        inputs = keras.Input((32, 32, 3))
        h = keras.layers.GlobalAveragePooling2D()(inputs)
        return keras.Model(inputs, keras.layers.Dense(10)(h))

    monkeypatch.setattr(module, "build_model", tiny)
    seen = {}

    def capture(config, lr_schedule):
        seen["lr0"] = float(lr_schedule(0))
        raise _Stop

    monkeypatch.setattr(module, "optimizer_builder", capture)
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--dataset", "cifar10", "--epochs", "2", "--batch-size", "8",
         "--output-dir", str(tmp_path)],
    )
    with pytest.raises(_Stop):
        module.main()
    assert seen["lr0"] == pytest.approx(1e-3, rel=1e-4)


# ---------------------------------------------------------------------
# A flag the user did not pass takes the preset's value (D-008). Before, the
# parser defaults cli_mode='full' / use_global_context=False were always passed
# as overrides, so `--variant lite_g` trained WITHOUT its global branch.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,cli_mode,glob",
    [
        ("nano", "full", False),
        ("lite_g", "full", True),
        ("nano_g", "full", True),
        ("cn_32_3", "full", False),
        ("cn_32_5", "inner", False),
        ("cn_64_5", "inner", False),
        ("custom", "full", False),
    ],
)
def test_an_unset_flag_takes_the_presets_value(monkeypatch, variant, cli_mode, glob):
    got = _parsed_args(monkeypatch, "--variant", variant)
    assert got["cli_mode"] == cli_mode
    assert got["use_global_context"] is glob
    assert got["ctx_mode"] == "diff"


def test_an_explicit_flag_overrides_the_preset(monkeypatch):
    got = _parsed_args(
        monkeypatch, "--variant", "lite", "--use-global-context",
        "--cli-mode", "wedge", "--ctx-mode", "abs",
    )
    assert got["use_global_context"] is True
    assert got["cli_mode"] == "wedge"
    assert got["ctx_mode"] == "abs"


@pytest.mark.parametrize(
    "variant,cli_mode,glob",
    [("lite_g", "full", True), ("nano_g", "full", True), ("cn_64_5", "inner", False)],
)
def test_the_built_model_has_the_presets_branches(monkeypatch, variant, cli_mode, glob):
    got = _parsed_args(monkeypatch, "--variant", variant)
    model = module.build_model(
        variant=got["variant"], num_classes=100, channels=got["channels"],
        depth=got["depth"], shifts=[1, 2], cli_mode=got["cli_mode"],
        ctx_mode=got["ctx_mode"], use_global_context=got["use_global_context"],
        layer_scale_init=got["layer_scale_init"],
        stochastic_depth_rate=got["stochastic_depth_rate"],
        dropout_rate=got["dropout_rate"], weight_decay=got["weight_decay"],
    )
    assert model.use_global_context is glob
    assert model.cli_mode == cli_mode
    assert model.depth == {"lite_g": 12, "nano_g": 12, "cn_64_5": 64}[variant]
