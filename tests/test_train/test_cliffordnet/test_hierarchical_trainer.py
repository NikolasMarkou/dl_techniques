"""The CliffordNet trainer trains the hierarchical model, and refuses flags that do not apply.

Guard for plan-2026-10-23T132722-77331e7a (D-001, D-002). The new variants are
``hier_345`` (a preset of ``HierarchicalCliffordNet``) and ``custom_hier``
(``--stage-depths`` / ``--stage-dims``). A flag that does not apply to the chosen
architecture must fail loudly: ``--variant lite_g`` once trained WITHOUT its
global branch because a parser default silently overrode the preset
(plan-2026-10-23T100412-e7eaaa02, D-009), and the same bug class is closed here
for the new variants. ``best_model.keras`` is reloaded only at the END of a run,
so the end-to-end tests run the real ``main()`` to that point.
"""

import json
import sys
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.cliffordnet import CliffordNet
from dl_techniques.models.vision.cliffordnet.hierarchical import HierarchicalCliffordNet
from train.cliffordnet import train_cliffordnet as module

VIS_FILES = [
    "training_dashboard.png",
    "confusion_matrix.png",
    "per_class_metrics.png",
    "confidence_calibration.png",
    "misclassifications.png",
    "classification_report.json",
]


def _parsed_args(monkeypatch, *argv):
    seen = {}
    monkeypatch.setattr(module, "train_model", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()
    return seen


# ---------------------------------------------------------------------
# Choices, resolution, defaults
# ---------------------------------------------------------------------


def test_the_variant_choices_include_the_hierarchical_ones_and_every_isotropic_one(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit):
        module.main()
    help_text = capsys.readouterr().out
    names = [
        *CliffordNet.MODEL_VARIANTS,
        *HierarchicalCliffordNet.MODEL_VARIANTS,
        "custom",
        "custom_hier",
    ]
    for name in names:
        assert name in help_text, name


@pytest.mark.parametrize("variant", ["hier_345", "custom_hier"])
def test_a_hierarchical_variant_resolves_from_its_preset(monkeypatch, variant):
    got = _parsed_args(monkeypatch, "--variant", variant)
    assert got["cli_mode"] == "full"
    assert got["ctx_mode"] == "diff"
    assert got["use_global_context"] is False
    assert got["stochastic_depth_rate"] == 0.1  # the authors' hierarchical default


def test_an_explicit_drop_path_rate_still_wins_for_a_hierarchical_variant(monkeypatch):
    got = _parsed_args(
        monkeypatch, "--variant", "hier_345", "--stochastic-depth-rate", "0.25"
    )
    assert got["stochastic_depth_rate"] == 0.25


def test_the_isotropic_variants_keep_their_resolution(monkeypatch):
    got = _parsed_args(monkeypatch, "--variant", "nano")
    assert (got["stochastic_depth_rate"], got["warmup_epochs"]) == (0.3, 0)
    got = _parsed_args(monkeypatch, "--variant", "custom")
    assert (got["channels"], got["depth"]) == (128, 12)
    assert got["stage_depths"] is None and got["stage_dims"] is None


# ---------------------------------------------------------------------
# A flag that does not apply fails loudly
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,flag",
    [
        (["--variant", "hier_345", "--use-global-context"], "--use-global-context"),
        (["--variant", "hier_345", "--channels", "64"], "--channels"),
        (["--variant", "hier_345", "--depth", "4"], "--depth"),
        (["--variant", "hier_345", "--stage-depths", "1,1"], "--stage-depths"),
        (["--variant", "hier_345", "--stage-dims", "8,16"], "--stage-dims"),
        (["--variant", "custom_hier", "--channels", "64"], "--channels"),
        (["--variant", "custom_hier", "--depth", "4"], "--depth"),
        (["--variant", "custom_hier", "--use-global-context"], "--use-global-context"),
        (["--variant", "nano", "--stage-depths", "1,1"], "--stage-depths"),
        (["--variant", "lite", "--stage-dims", "8,16"], "--stage-dims"),
        (["--variant", "custom", "--stage-depths", "1,1"], "--stage-depths"),
        (["--variant", "custom", "--stage-dims", "8,16"], "--stage-dims"),
    ],
)
def test_a_flag_that_does_not_apply_is_refused_by_name(monkeypatch, capsys, argv, flag):
    monkeypatch.setattr(module, "train_model", lambda args: pytest.fail("trained"))
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


# ---------------------------------------------------------------------
# build_model dispatch
# ---------------------------------------------------------------------


def _build(variant, **overrides):
    kwargs = dict(
        variant=variant, num_classes=10, channels=128, depth=12, shifts=[1, 2],
        cli_mode="full", ctx_mode="diff", use_global_context=False,
        layer_scale_init=1e-5, stochastic_depth_rate=0.1, dropout_rate=0.0,
        weight_decay=0.1,
    )
    kwargs.update(overrides)
    return module.build_model(**kwargs)


def test_build_model_returns_the_hierarchical_preset():
    model = _build("hier_345")
    assert isinstance(model, HierarchicalCliffordNet)
    assert model.stage_depths == [3, 4, 5]
    assert model.stochastic_depth_rate == 0.1


def test_build_model_builds_custom_hier_from_the_stage_lists():
    model = _build("custom_hier", stage_depths=[2, 1], stage_dims=[8, 16], shifts=[1])
    assert isinstance(model, HierarchicalCliffordNet)
    assert (model.stage_depths, model.stage_dims, model.shifts) == ([2, 1], [8, 16], [1])


def test_build_model_still_builds_the_isotropic_variants():
    assert isinstance(_build("nano"), CliffordNet)
    assert _build("lite_g", use_global_context=True).use_global_context is True
    assert _build("custom", channels=16, depth=2).depth == 2


# ---------------------------------------------------------------------
# End to end: the real main(), including the best-checkpoint reload at the end
# ---------------------------------------------------------------------


def _tiny_cifar(*_args, **_kwargs):
    rng = np.random.default_rng(0)
    x_train = rng.random((32, 32, 32, 3), dtype=np.float32)
    y_train = rng.integers(0, 10, size=(32,))
    x_test = rng.random((16, 32, 32, 3), dtype=np.float32)
    y_test = rng.integers(0, 10, size=(16,))
    return (x_train, y_train), (x_test, y_test), (32, 32, 3), 10


def _run(tmp_path_factory, *extra):
    out = tmp_path_factory.mktemp("hier_run")
    mp = pytest.MonkeyPatch()
    mp.setattr(module, "setup_gpu", lambda *a, **k: None)
    mp.setattr(module, "load_dataset", _tiny_cifar)
    mp.setattr(module, "run_model_analysis", lambda *a, **k: None)
    mp.setattr(
        sys, "argv",
        ["prog", "--dataset", "cifar10", "--epochs", "2", "--batch-size", "8",
         "--output-dir", str(out), *extra],
    )
    try:
        module.main()
    finally:
        mp.undo()
    (run,) = [p for p in out.iterdir() if p.is_dir()]
    return Path(run)


@pytest.fixture(scope="module")
def preset_run(tmp_path_factory):
    return _run(tmp_path_factory, "--variant", "hier_345")


@pytest.fixture(scope="module")
def custom_run(tmp_path_factory):
    return _run(
        tmp_path_factory, "--variant", "custom_hier",
        "--stage-depths", "1,1", "--stage-dims", "8,16", "--shifts", "1",
    )


@pytest.mark.parametrize("fixture_name", ["preset_run", "custom_run"])
def test_a_run_writes_the_standard_artifacts(request, fixture_name):
    run = request.getfixturevalue(fixture_name)
    assert (run / "best_model.keras").is_file()
    assert list(run.glob("*_final.keras"))
    for name in VIS_FILES:
        assert (run / "visualizations" / name).stat().st_size > 0, name
    assert "lr" in (run / "training_log.csv").read_text().splitlines()[0].split(",")


@pytest.mark.parametrize("fixture_name", ["preset_run", "custom_run"])
def test_the_best_checkpoint_reloads_as_the_hierarchical_model(request, fixture_name):
    run = request.getfixturevalue(fixture_name)
    model = keras.models.load_model(run / "best_model.keras")
    assert isinstance(model, HierarchicalCliffordNet)


def test_config_json_records_the_architecture_that_trained(custom_run, preset_run):
    custom = json.loads((custom_run / "config.json").read_text())
    assert custom["variant"] == "custom_hier"
    assert custom["stage_depths"] == [1, 1] and custom["stage_dims"] == [8, 16]
    preset = json.loads((preset_run / "config.json").read_text())
    assert preset["variant"] == "hier_345"
    assert preset["stochastic_depth_rate"] == 0.1
