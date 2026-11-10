"""The ``image_denoising`` task end to end, per family (plan d483ffab step 8, D-012, D-013).

Real fits on the CPU of the five families (``bfcnn``, ``bfunet``, ``bfconvunext``, ``convunext_biased``, ``scunet``, all
``tiny``) on synthetic folders of 64 x 64 images, 32 px patches, six train files and three validation files, two epochs,
so every number the summary reports can be recomputed here by hand instead of read back from the code under test:

* the PSNR of the noisy validation set (``input_psnr``) is recomputed from the same fixed patches with numpy, and the
  identity loss from the same pairs; the epoch-0 ratio is their quotient;
* the verdict is the best ``val_psnr`` of the selection window MINUS ``input_psnr``, against ``success_gain_db``;
* the function and ``python -m train image_denoising`` write the same ``config.json`` bytes, for every family (the run
  stops at ``build_data``, after both files are written, as the classification file does);
* a scripted monitor whose best ``val_psnr`` falls inside the noise ramp: the window does not select it, the control
  with a window of the whole run does, and the file saved is the weights of the epoch the summary names;
* the short-run table: the curriculum, the selection start and the warmup of runs of 1, 2, 3, 5, 10, 30 and 100 epochs;
* ``compile_options``: float32 passes no ``jit_compile`` for any family, ``mixed_float16`` passes ``False`` for the
  three families XLA rejects (measured in step 2) and nothing for the two it accepts; a spy on ``compile`` shows it.

The tests write only under ``tmp_path``. Every real run sets ``max_train_files`` and ``max_val_files`` to the number of
files: the default of 10,000 repeats a small folder to fill an epoch (the DIV2K rule), which is 5,000 steps here.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest
from PIL import Image

import train.api.recipes as recipes
from train.api import cli, registry
from train.api import denoising_args as dargs
from train.api.config import validate
from train.api.denoising_models import XLA_REJECTS_MIXED_FLOAT16, compile_options_for
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_denoising import (
    ImageDenoisingConfig,
    ImageDenoisingTrainer,
    train_image_denoising,
)
from train.common import classification_run
from train.common import denoising_data as dd
from train.common import denoising_run as dr

REPO_ROOT = Path(__file__).resolve().parents[3]

FAMILIES = ("bfcnn", "bfunet", "bfconvunext", "convunext_biased", "scunet")
BIAS_FREE = ("bfcnn", "bfunet", "bfconvunext")

SIZE = 32
PATCHES = 2
BATCH = 4
TRAIN_FILES, VAL_FILES = 6, 3
EPOCHS = 2

#: What each family's run asks beyond the common request, so that one run per family also exercises a noise kind, an
#: augmentation, a channel count and a loss: ``(data, transform_args, loss_args)`` overrides.
CASES = {
    "bfcnn": ({}, {}, {}),
    "bfunet": ({"noise_kind": "multiplicative", "noise_clip": True}, {}, {}),
    "bfconvunext": ({}, {"augment": "dihedral"}, {}),
    "convunext_biased": ({}, {"channels": 1}, {"name": "charbonnier"}),
    "scunet": ({"noise_kind": "composite", "noise_composite_ratio": 0.3}, {}, {}),
}


@pytest.fixture(scope="module")
def world():
    """An empty registry holding the built-in families and the real task rows."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def write_images(root: Path, count: int, seed: int) -> Path:
    """Write ``count`` smooth 64 x 64 RGB images (a tinted wave), so a denoiser has structure to keep."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:64, 0:64] / 64.0
    for index in range(count):
        tint = rng.uniform(0.25, 0.75, (1, 1, 3))
        wave = 0.2 * np.sin(6 * columns + index)[..., None] * np.cos(5 * rows)[..., None]
        Image.fromarray((np.clip(tint + wave, 0, 1) * 255).astype(np.uint8)).save(root / f"i{index:02d}.png")
    return root


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("denoise_task")
    return write_images(root / "train", TRAIN_FILES, 0), write_images(root / "val", VAL_FILES, 1)


def mapping(out, folders, family="bfcnn", epochs=EPOCHS, **changes):
    """The request of a run of ``family`` as a nested mapping; ``changes`` replace top-level keys."""
    data_extra, transform_extra, loss_extra = CASES.get(family, ({}, {}, {}))
    train, val = folders
    request = {
        "out": str(out),
        "model": f"{family}/tiny",
        "data": {
            "train_dirs": [str(train)], "val_dirs": [str(val)],
            "max_train_files": TRAIN_FILES, "max_val_files": VAL_FILES, **data_extra,
        },
        "epochs": epochs,
        "batch_size": BATCH,
        "transform_args": {"patch_size": SIZE, "patches_per_image": PATCHES, **transform_extra},
        "loss_args": dict(loss_extra),
        "logger_args": {"tensorboard": None},
        "analysis_args": {"figures": False},
    }
    request.update(changes)
    return request


def config_of(out, folders, family="bfcnn", **changes):
    return validate(ImageDenoisingConfig, mapping(out, folders, family, **changes))


def read_json(path):
    return json.loads(Path(path).read_text())


def saved_weights(path):
    return [np.array(w) for w in keras.models.load_model(path, compile=False).get_weights()]


def same_weights(first, second):
    return len(first) == len(second) and all(np.array_equal(a, b) for a, b in zip(first, second))


def flatten(prefix, value, tokens):
    """The ``key=value`` tokens of a nested mapping, as the command line takes them."""
    if isinstance(value, dict) and prefix not in ("model_args",):
        for key, inner in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, inner, tokens)
    elif isinstance(value, bool):
        tokens.append(f"{prefix}={'true' if value else 'false'}")
    elif value is None:
        tokens.append(f"{prefix}=null")
    elif isinstance(value, (list, dict)):
        tokens.append(f"{prefix}={json.dumps(value, separators=(',', ':'))}")
    else:
        tokens.append(f"{prefix}={value}")
    return tokens


def hand_scores(folders, request):
    """The identity's PSNR and loss and the patch count, recomputed with numpy from the run's own fixed patches."""
    data_cfg = validate(dargs.DenoisingData, request["data"])
    transform = request["transform_args"]
    batches = dd.load_denoising_data(
        data_cfg.train_dirs, data_cfg.val_dirs, patch_size=transform["patch_size"],
        channels=transform.get("channels", 3), patches_per_image=PATCHES, batch_size=BATCH, seed=42,
        noise=dargs.noise_spec(data_cfg), val_seed=data_cfg.val_seed, max_train_files=TRAIN_FILES,
        max_val_files=VAL_FILES, max_samples=None,
    )
    psnrs, losses = [], []
    loss = request.get("loss_args", {}).get("name", "mse")
    for noisy, clean in batches.select_ds.as_numpy_iterator():
        difference = (noisy - clean).astype(np.float64)
        per_patch = (difference ** 2).reshape(len(difference), -1).mean(axis=1)
        psnrs.extend(10.0 * np.log10(1.0 / np.maximum(per_patch, keras.config.epsilon())))
        elementwise = difference ** 2 if loss == "mse" else np.sqrt(difference ** 2 + dr.CHARBONNIER_EPSILON ** 2)
        losses.extend(elementwise.reshape(len(difference), -1).mean(axis=1))
    return float(np.mean(psnrs)), float(np.mean(losses)), len(psnrs)


# ---------------------------------------------------------------------
# One home for the choices
# ---------------------------------------------------------------------


def test_the_config_and_the_run_modules_name_the_same_losses_and_augmentations():
    assert dargs.LOSS_NAMES == dr.LOSS_NAMES
    assert dargs.PATCH_AUGMENTS == dd.PATCH_AUGMENTS
    assert dr.DENOISING_METRIC_NAMES == ("mae", "psnr", "ssim")


def test_the_task_lists_the_five_families_through_the_real_command_line(tmp_path):
    """``python -m train list_models task=image_denoising`` in a child process: 23 variants of five families."""
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    finished = subprocess.run(
        [sys.executable, "-m", "train", "list_models", "task=image_denoising"],
        capture_output=True, text=True, cwd=tmp_path, env=environment, timeout=300,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]
    names = [line for line in finished.stdout.splitlines() if "/" in line and " " not in line.strip()]
    assert sorted({name.split("/")[0] for name in names}) == sorted(FAMILIES)
    assert len(names) == 23 and len(set(names)) == 23
    helped = subprocess.run(
        [sys.executable, "-m", "train", "help"], capture_output=True, text=True, cwd=tmp_path, env=environment,
        timeout=300,
    )
    assert helped.returncode == 0 and "image_denoising" in helped.stdout


# ---------------------------------------------------------------------
# A real run per family, against a hand count
# ---------------------------------------------------------------------


@pytest.fixture(scope="module", params=FAMILIES)
def finished(world, folders, request, tmp_path_factory):
    family = request.param
    out = tmp_path_factory.mktemp("run_" + family) / "run"
    asked = mapping(out, folders, family)
    result = train_image_denoising(**asked)
    summary = read_json(result.summary_path)
    return family, asked, result, summary, read_json(Path(result.run_dir) / "training_history.json")


def test_input_psnr_and_the_epoch_0_ratio_are_the_hand_counted_identity(finished, folders):
    family, asked, _, summary, _ = finished
    psnr, identity_loss, patches = hand_scores(folders, asked)
    assert patches == VAL_FILES * PATCHES == summary["n_val"]
    assert summary["input_psnr"] == pytest.approx(psnr, abs=1e-3)
    epoch_zero = summary["initial_loss_sanity_eval"]["loss"]
    assert summary["initial_loss_ratio"] == pytest.approx(epoch_zero / identity_loss, rel=1e-4)
    # No verdict on the initialization, and the note says why (the ratio is not a scale test).
    assert summary["init_scale_warning"] is None
    assert any("init_scale_warning" in note and "IDENTITY" in note for note in summary["notes"])


def test_the_verdict_is_the_window_best_val_psnr_minus_input_psnr_against_the_gain(finished):
    family, _, result, summary, history = finished
    assert summary["success_metric"] == "val_psnr" and summary["monitor"] == "val_psnr"
    assert summary["monitor_mode"] == "max"
    window = history["val_psnr"][summary["curriculum"]["selection_start_epoch"]:]
    assert summary["success_value"] == pytest.approx(max(window) - summary["input_psnr"], abs=1e-4)
    assert summary["success_threshold"] == 1.0
    assert summary["success"] is bool(summary["success_value"] >= summary["success_threshold"])
    assert result.success is summary["success"]
    # The control for "the raw val_psnr": the two numbers are different ones here.
    assert summary["success_value"] != max(history["val_psnr"])


def test_the_summary_says_what_it_trained(finished):
    family, _, _, summary, history = finished
    data_extra, transform_extra, loss_extra = CASES[family]
    channels = transform_extra.get("channels", 3)
    assert summary["task"] == "image_denoising" and summary["model_family"] == family and summary["variant"] == "tiny"
    assert "num_classes" not in summary, "a denoiser has no classes (D-005)"
    assert summary["input_shape"] == [SIZE, SIZE, channels]
    assert summary["noise"]["kind"] == data_extra.get("noise_kind", "additive")
    assert summary["noise"]["clip"] is data_extra.get("noise_clip", False)
    assert summary["noise"]["composite_ratio"] == data_extra.get("noise_composite_ratio")
    assert summary["noise"]["sigma_max"] == 0.25 and summary["noise"]["val_seed"] == 0
    # Epochs 2: the curriculum is round(0.8 * 2) = 2 long and the window opens at its last epoch.
    assert summary["curriculum"] == {
        "schedule": "linear", "epochs": 2, "sigma_max_start": 0.025, "sigma_max_end": 0.25, "selection_start_epoch": 1,
    }
    only_additive = summary["noise"]["kind"] == "additive" and not summary["noise"]["clip"]
    assert summary["score_identity"]["holds"] is only_additive
    assert summary["selection"] == {
        "start_epoch": 1, "window_epochs": 1, "curriculum_epochs": 2, "best_epoch": 2, "final_is_best": True,
    }
    assert summary["best_epoch"] == 2 and summary["epochs_run"] == EPOCHS
    for key in ("loss", "mae", "psnr", "ssim", "val_loss", "val_mae", "val_psnr", "val_ssim"):
        assert len(history[key]) == EPOCHS and all(math.isfinite(v) for v in history[key]), key
    assert summary["held_out"]["status"] == "skipped" and summary["held_out"]["sigmas_255"] == [15.0, 25.0, 50.0]
    assert summary["test_metrics_best"] is None and summary["test_metrics_final"] is None


def test_the_run_directory_holds_the_contract_files(finished):
    _, _, result, summary, _ = finished
    run_dir = Path(result.run_dir)
    for name in ("config_requested.json", "config.json", "results_summary.json", "training_log.csv",
                 "training_history.json", "best_model.keras", "final_model.keras", "run.log"):
        assert (run_dir / name).is_file(), name
    assert summary["model_loading_validated"] is True
    assert summary["analyzer"]["status"] == "skipped"


def test_the_resolved_config_holds_the_derived_values_and_the_request_keeps_auto(finished):
    family, asked, result, _, _ = finished
    resolved = read_json(Path(result.run_dir) / "config.json")
    requested = read_json(Path(result.run_dir) / "config_requested.json")
    assert resolved["data"]["curriculum_epochs"] == 2 and requested["data"]["curriculum_epochs"] == "auto"
    assert resolved["optim_args"]["warmup_epochs"] == 1 and requested["optim_args"]["warmup_epochs"] == "auto"
    assert resolved["metric_args"]["watch_metric"] == "val_psnr"
    assert resolved["metric_args"]["success_gain_db"] == 1.0 and requested["metric_args"]["success_gain_db"] == "auto"
    assert resolved["transform_args"]["augment"] == CASES[family][1].get("augment", "flip")
    assert "normalize" not in resolved["transform_args"], "inputs lie in [0, 1] and are never standardized"


# ---------------------------------------------------------------------
# The function and the command line are one entry point
# ---------------------------------------------------------------------


class StopAtData(Exception):
    """Raised in place of ``build_data``: ``config.json`` is already on disk by then."""


@pytest.mark.parametrize("family", FAMILIES)
def test_the_function_and_the_cli_write_the_same_config_bytes(family, world, folders, tmp_path):
    reached = []

    def stop(self, config, context):
        reached.append(context.family)
        raise StopAtData

    asked = mapping(tmp_path / "run", folders, family, seed=7)
    asked["optim_args"] = {"lr": 1e-3}
    asked["metric_args"] = {"success_gain_db": 2.0, "test_sigmas_255": [15, 30]}
    tokens = flatten("", asked, [])
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setattr(ImageDenoisingTrainer, "build_data", stop)
        with pytest.raises(StopAtData):
            train_image_denoising(**asked)
        (tmp_path / "run").rename(tmp_path / "function")
        try:
            cli.main(["image_denoising", *tokens])
        except StopAtData:
            pass
        (tmp_path / "run").rename(tmp_path / "cli")
    assert reached == [family, family]
    for name in ("config.json", "config_requested.json"):
        assert (tmp_path / "function" / name).read_bytes() == (tmp_path / "cli" / name).read_bytes(), name
    resolved = read_json(tmp_path / "cli" / "config.json")
    assert resolved["model"] == f"{family}/tiny" and resolved["optim_args"]["lr"] == 1e-3 and resolved["seed"] == 7
    assert resolved["metric_args"]["success_gain_db"] == 2.0
    assert resolved["metric_args"]["test_sigmas_255"] == [15.0, 30.0]


# ---------------------------------------------------------------------
# Compile options: nothing at float32, False where XLA rejects the gradient
# ---------------------------------------------------------------------


class StopAtCompile(Exception):
    """Raised by the spy in place of the real ``compile`` of the run's model."""


@pytest.fixture
def compile_spy(monkeypatch):
    """Record the keywords of the run's ``compile`` and stop there; the identity model's own compile passes through."""
    calls = []
    original = keras.Model.compile

    def spy(self, *arguments, **keywords):
        if self.name == "identity":
            return original(self, *arguments, **keywords)
        calls.append((arguments, keywords))
        raise StopAtCompile

    monkeypatch.setattr(keras.Model, "compile", spy)
    return calls


def test_the_rejected_cells_are_the_three_the_step_2_measurement_names():
    assert sorted(XLA_REJECTS_MIXED_FLOAT16) == ["bfconvunext", "bfunet", "convunext_biased"]
    for family in FAMILIES:
        for precision in ("float32", "mixed_bfloat16"):
            assert compile_options_for(family, precision) == {}, (family, precision)
        expected = {"jit_compile": False} if family in XLA_REJECTS_MIXED_FLOAT16 else {}
        assert compile_options_for(family, "mixed_float16") == expected, family


@pytest.mark.parametrize("precision", ["float32", "mixed_float16"])
@pytest.mark.parametrize("family", FAMILIES)
def test_the_one_compile_call_gets_jit_compile_false_only_on_a_rejected_cell(
    family, precision, world, folders, tmp_path, compile_spy
):
    with pytest.raises(StopAtCompile):
        train_image_denoising(**mapping(tmp_path / "run", folders, family, precision=precision))
    assert len(compile_spy) == 1
    _, keywords = compile_spy[0]
    rejected = precision == "mixed_float16" and family in XLA_REJECTS_MIXED_FLOAT16
    assert set(keywords) == {"optimizer", "loss", "metrics"} | ({"jit_compile"} if rejected else set()), keywords
    if rejected:
        assert keywords["jit_compile"] is False
    assert [metric.name for metric in keywords["metrics"]] == ["mae", "psnr", "ssim"]
    assert keras.mixed_precision.global_policy().name == "float32", "the precision policy is restored"


# ---------------------------------------------------------------------
# The selection window: a ramp epoch that peaks is not selected
# ---------------------------------------------------------------------

#: ``val_psnr`` by epoch: 50 and 45 inside the noise ramp (the curriculum is 3 epochs), 40 and 39 inside the window.
#: All are above the input PSNR of the fixture (about 26 dB), so a positive gain can be set either side of them.
RAMP_PEAK = (50.0, 45.0, 40.0, 39.0)


class Recorder(keras.callbacks.Callback):
    """Scripts ``val_psnr`` and records the weights at each epoch end (it runs before the checkpoint callbacks)."""

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        logs["val_psnr"] = self.script[epoch]
        self.weights.append([np.array(w) for w in self.model.get_weights()])


class Windowed(ImageDenoisingTrainer):
    """The real task with a scripted ``val_psnr``."""

    script = RAMP_PEAK
    recorder = None

    def _callbacks(self, *arguments):
        epoch_line, callbacks = super()._callbacks(*arguments)
        type(self).recorder = Recorder(self.script)
        # Index 1: after the learning-rate logger, before EarlyStopping and the checkpoint, which read `logs` in order.
        callbacks.insert(1, type(self).recorder)
        return epoch_line, callbacks


class FromTheStart(Windowed):
    """The control: the same run with a window of the whole run."""

    def selection_start_epoch(self, config):
        return 0


def run_scripted(trainer_class, folders, tmp_path_factory, name, **changes):
    out = tmp_path_factory.mktemp(name) / "run"
    asked = mapping(out, folders, "bfcnn", epochs=4, **changes)
    asked["data"]["curriculum_epochs"] = 3
    config = validate(ImageDenoisingConfig, asked)
    result = trainer_class(config).run()
    return result, out, read_json(result.summary_path), trainer_class.recorder


@pytest.fixture(scope="module")
def windowed(world, folders, tmp_path_factory):
    return run_scripted(Windowed, folders, tmp_path_factory, "windowed")


@pytest.fixture(scope="module")
def from_the_start(world, folders, tmp_path_factory):
    return run_scripted(FromTheStart, folders, tmp_path_factory, "from_the_start")


def test_the_ramp_peak_is_not_selected_and_the_window_best_is(windowed):
    result, out, summary, recorder = windowed
    assert summary["curriculum"]["epochs"] == 3 and summary["curriculum"]["selection_start_epoch"] == 2
    # The raw maximum (50) is epoch 1, inside the ramp; the saved and reported epoch is the best of the window.
    assert max(read_json(out / "training_history.json")["val_psnr"]) == 50.0
    assert summary["best_epoch"] == 3 and result.best_metric == 40.0
    assert summary["selection"] == {
        "start_epoch": 2, "window_epochs": 2, "curriculum_epochs": 3, "best_epoch": 3, "final_is_best": False,
    }
    assert same_weights(saved_weights(out / "best_model.keras"), recorder.weights[2])
    assert not same_weights(saved_weights(out / "best_model.keras"), recorder.weights[0])
    assert same_weights(saved_weights(out / "final_model.keras"), recorder.weights[3])
    # Liveness: the epochs really ended in different weights.
    assert not same_weights(recorder.weights[0], recorder.weights[2])


def test_the_control_with_a_window_of_the_whole_run_selects_the_ramp_peak(from_the_start):
    result, out, summary, recorder = from_the_start
    assert summary["best_epoch"] == 1 and result.best_metric == 50.0
    assert summary["selection"]["start_epoch"] == 0 and summary["selection"]["window_epochs"] == 4
    assert same_weights(saved_weights(out / "best_model.keras"), recorder.weights[0])


def test_the_success_value_is_the_window_best_minus_the_input_psnr(windowed, from_the_start):
    _, _, summary, _ = windowed
    assert summary["success_value"] == pytest.approx(40.0 - summary["input_psnr"], abs=1e-4)
    control = from_the_start[2]
    assert control["success_value"] == pytest.approx(50.0 - control["input_psnr"], abs=1e-4)
    assert summary["input_psnr"] == control["input_psnr"], "one validation set, one input PSNR"


@pytest.mark.parametrize("offset, expected", [(-0.5, True), (0.5, False)], ids=["clears", "misses"])
def test_the_gain_is_the_configs_not_a_constant(offset, expected, windowed, world, folders, tmp_path_factory):
    # The input PSNR of the fixed validation set is a function of the data and the noise, not of the run, so it is
    # read from the window's run and the gain is set either side of the scripted 40 dB.
    base = windowed[2]
    gain = round(40.0 - base["input_psnr"] + offset, 6)
    result, _, summary, _ = run_scripted(
        Windowed, folders, tmp_path_factory, "gain", metric_args={"success_gain_db": gain})
    assert summary["success_threshold"] == gain
    assert summary["success"] is expected and result.success is expected


# ---------------------------------------------------------------------
# The short-run table: curriculum, selection start and warmup by epochs
# ---------------------------------------------------------------------

#: epochs -> (curriculum_epochs, selection start, window epochs, warmup_epochs): ``max(1, round(0.8 e))``, that minus 1,
#: ``e - start`` and ``max(1, round(0.1 e))``. Python rounds half to even, so 5 epochs warm up 1 epoch, as legacy.
SHORT_RUNS = {1: (1, 0, 1, 1), 2: (2, 1, 1, 1), 3: (2, 1, 2, 1), 5: (4, 3, 2, 1), 10: (8, 7, 3, 1),
              30: (24, 23, 7, 3), 100: (80, 79, 21, 10)}


@pytest.mark.parametrize("epochs", sorted(SHORT_RUNS))
def test_the_resolved_curriculum_selection_start_and_warmup_follow_the_epochs(epochs, world, folders, tmp_path):
    trainer = ImageDenoisingTrainer(config_of(tmp_path / "run", folders, epochs=epochs))
    resolved, _ = trainer._resolve(trainer.describe_data(trainer.config), "bfcnn", "tiny")
    curriculum, start, window, warmup = SHORT_RUNS[epochs]
    assert resolved.data.curriculum_epochs == curriculum
    assert trainer.selection_start_epoch(resolved) == start
    assert resolved.epochs - trainer.selection_start_epoch(resolved) == window
    assert resolved.optim_args.warmup_epochs == warmup
    assert not (tmp_path / "run").exists(), "resolution claims nothing"


def test_the_curriculum_follows_an_explicit_epochs_not_the_recipes_hundred(world, folders, tmp_path):
    trainer = ImageDenoisingTrainer(config_of(tmp_path / "run", folders, epochs=30))
    resolved, context = trainer._resolve(trainer.describe_data(trainer.config), "bfcnn", "tiny")
    assert context.defaults["epochs"] == 100 and context.epochs == 30
    assert resolved.data.curriculum_epochs == 24, "round(0.8 * 30), not round(0.8 * 100) = 80"


def test_a_smoke_run_has_a_curriculum_of_one_epoch(world, folders, tmp_path):
    trainer = ImageDenoisingTrainer(
        validate(ImageDenoisingConfig, {**mapping(tmp_path / "run", folders), "debug_args": {"smoke": True}, "epochs": "auto"}))
    resolved, _ = trainer._resolve(trainer.describe_data(trainer.config), "bfcnn", "tiny")
    assert resolved.epochs == 1 and resolved.data.curriculum_epochs == 1
    assert trainer.selection_start_epoch(resolved) == 0


def test_an_explicit_warmup_and_curriculum_win_over_the_derived_ones(world, folders, tmp_path):
    asked = mapping(tmp_path / "run", folders, epochs=10, optim_args={"warmup_epochs": 4})
    asked["data"]["curriculum_epochs"] = 5
    trainer = ImageDenoisingTrainer(validate(ImageDenoisingConfig, asked))
    resolved, _ = trainer._resolve(trainer.describe_data(trainer.config), "bfcnn", "tiny")
    assert resolved.data.curriculum_epochs == 5 and resolved.optim_args.warmup_epochs == 4


def test_a_constant_schedule_has_no_warmup_whatever_the_epochs(world, folders, tmp_path):
    asked = mapping(tmp_path / "run", folders, epochs=10, optim_args={"schedule": "constant"})
    trainer = ImageDenoisingTrainer(validate(ImageDenoisingConfig, asked))
    resolved, _ = trainer._resolve(trainer.describe_data(trainer.config), "bfcnn", "tiny")
    assert resolved.optim_args.warmup_epochs == 0


@pytest.mark.parametrize("epochs, window", [(1, 1), (3, 2), (5, 2)])
def test_a_short_real_run_holds_the_table(epochs, window, world, folders, tmp_path):
    result = train_image_denoising(**mapping(tmp_path / "run", folders, epochs=epochs))
    summary = read_json(result.summary_path)
    curriculum, start, expected_window, _ = SHORT_RUNS[epochs]
    assert (summary["curriculum"]["epochs"], summary["selection"]["start_epoch"]) == (curriculum, start)
    assert summary["selection"]["window_epochs"] == window == expected_window
    assert start + 1 <= summary["best_epoch"] <= epochs, "the best epoch is inside the window"
    assert (Path(result.run_dir) / "best_model.keras").is_file()
    assert len(read_json(Path(result.run_dir) / "training_history.json")["val_psnr"]) == epochs


# ---------------------------------------------------------------------
# What is refused before the run directory is claimed
# ---------------------------------------------------------------------


def test_a_curriculum_longer_than_the_run_is_refused_by_key(world, folders, tmp_path):
    asked = mapping(tmp_path / "run", folders, epochs=3)
    asked["data"]["curriculum_epochs"] = 5
    with pytest.raises(ConfigValidationError) as raised:
        train_image_denoising(**asked)
    assert "data.curriculum_epochs" in str(raised.value) and "3 epochs" in str(raised.value)
    assert not (tmp_path / "run").exists()


def test_a_curriculum_longer_than_the_smoke_epoch_is_refused_after_the_epochs_resolve(world, folders, tmp_path):
    asked = mapping(tmp_path / "run", folders, debug_args={"smoke": True})
    del asked["epochs"]
    asked["data"]["curriculum_epochs"] = 3
    with pytest.raises(ConfigValidationError) as raised:
        train_image_denoising(**asked)
    assert "data.curriculum_epochs" in str(raised.value)
    assert not (tmp_path / "run").exists()


def test_a_conformal_level_is_refused_by_key(world, folders, tmp_path):
    with pytest.raises(ConfigValidationError) as raised:
        train_image_denoising(**mapping(tmp_path / "run", folders, analysis_args={"conformal_alpha": 0.1}))
    assert "analysis_args.conformal_alpha" in str(raised.value)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("name", ["mae", "charbonnier"])
@pytest.mark.parametrize("family", BIAS_FREE)
def test_a_bias_free_family_refuses_an_l1_type_loss_before_the_claim(family, name, world, folders, tmp_path):
    asked = mapping(tmp_path / "run", folders, family)
    asked["loss_args"] = {"name": name}
    with pytest.raises(ConfigError) as raised:
        train_image_denoising(**asked)
    assert "loss_args.name" in str(raised.value) and "bias-free" in str(raised.value)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("name", ["mae", "charbonnier"])
@pytest.mark.parametrize("family", ["convunext_biased", "scunet"])
def test_a_biased_family_takes_any_loss(family, name, world, folders, tmp_path):
    asked = mapping(tmp_path / "run", folders, family)
    asked["loss_args"] = {"name": name}
    trainer = ImageDenoisingTrainer(validate(ImageDenoisingConfig, asked))
    trainer._resolve(trainer.describe_data(trainer.config), family, "tiny")


def test_a_patch_size_off_a_u_nets_stride_is_refused_and_a_stride_free_family_takes_it(world, folders, tmp_path):
    odd = {"patch_size": 36}
    for family in ("bfunet", "bfconvunext", "convunext_biased"):
        asked = mapping(tmp_path / f"{family}_run", folders, family)
        asked["transform_args"] = {**asked["transform_args"], **odd}
        with pytest.raises(ConfigError) as raised:
            train_image_denoising(**asked)
        assert "transform_args.patch_size" in str(raised.value) and "multiple of 8" in str(raised.value)
        assert not (tmp_path / f"{family}_run").exists()
    for family in ("bfcnn", "scunet"):
        asked = mapping(tmp_path / f"{family}_run", folders, family)
        asked["transform_args"] = {**asked["transform_args"], **odd}
        trainer = ImageDenoisingTrainer(validate(ImageDenoisingConfig, asked))
        trainer._resolve(trainer.describe_data(trainer.config), family, "tiny")


@pytest.mark.parametrize(
    "key, value, text",
    [
        ("transform_args", {"patch_size": 8}, "patch_size"),
        ("transform_args", {"patches_per_image": 0}, "patches_per_image"),
        ("transform_args", {"channels": 2}, "channels"),
        ("transform_args", {"augment": "rot90"}, "augment"),
        ("metric_args", {"success_gain_db": -1.0}, "success_gain_db"),
        ("metric_args", {"test_sigmas_255": []}, "test_sigmas_255"),
        ("metric_args", {"test_sigmas_255": [15, 15]}, "test_sigmas_255"),
        ("metric_args", {"test_sigmas_255": [0, 25]}, "test_sigmas_255"),
        ("loss_args", {"name": "huber"}, "loss_args.name"),
    ],
)
def test_a_group_value_out_of_range_is_named(key, value, text, world, folders, tmp_path):
    with pytest.raises(ConfigValidationError) as raised:
        train_image_denoising(**mapping(tmp_path / "run", folders, **{key: value}))
    assert text in str(raised.value)
    assert not (tmp_path / "run").exists()


def test_the_monitor_must_be_a_metric_the_run_reports(world, folders, tmp_path):
    with pytest.raises(ConfigError) as raised:
        train_image_denoising(**mapping(tmp_path / "run", folders, metric_args={"watch_metric": "val_accuracy"}))
    assert "val_psnr" in str(raised.value) and "val_ssim" in str(raised.value)
    assert not (tmp_path / "run").exists()


# ---------------------------------------------------------------------
# val_psnr is a function of the weights alone
# ---------------------------------------------------------------------


def build_compiled(config, family="bfcnn"):
    """The trainer, its data and its compiled, untrained model, through the task's own hooks."""
    trainer = ImageDenoisingTrainer(config)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, "tiny")
    data = trainer.build_data(resolved, context)
    model = trainer.build_model(resolved, context, data)
    loss, metrics = trainer.build_loss_and_metrics(resolved, data)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return trainer, resolved, context, data, model


def test_val_psnr_is_equal_across_evaluations_epochs_and_the_seed_of_the_training_stream(world, folders, tmp_path):
    keras.utils.set_random_seed(3)
    _, resolved, context, data, model = build_compiled(config_of(tmp_path / "a", folders))
    first = classification_run.evaluate_dict(model, data.select_ds)
    second = classification_run.evaluate_dict(model, data.select_ds)
    assert first == second, "two evaluations of the same weights are the same numbers"
    # A trainer whose TRAINING seed differs builds another training stream and the same validation set.
    other_config = validate(ImageDenoisingConfig, mapping(tmp_path / "b", folders, seed=99))
    other = ImageDenoisingTrainer(other_config)
    other_resolved, other_context = other._resolve(other.describe_data(other_config), "bfcnn", "tiny")
    other_data = other.build_data(other_resolved, other_context)
    assert classification_run.evaluate_dict(model, other_data.select_ds) == first
    x_first = next(iter(data.train_ds.take(1)))[0].numpy()
    x_other = next(iter(other_data.train_ds.take(1)))[0].numpy()
    assert not np.array_equal(x_first, x_other), "liveness: the two training streams differ"


def test_the_epoch_0_ratio_is_the_loss_over_the_identity_loss_and_a_non_finite_loss_is_refused(world, folders, tmp_path):
    trainer, resolved, _, data, model = build_compiled(config_of(tmp_path / "a", folders))
    baseline, ratio, warned = trainer.initial_loss_check(resolved, model, data)
    assert warned is None
    assert ratio == pytest.approx(baseline["loss"] / trainer._identity["loss"], rel=1e-9)
    broken = keras.Model(model.input, keras.layers.Lambda(lambda x: x * float("nan"))(model.output))
    broken.compile(optimizer="adam", loss=dr.denoising_loss("mse"), metrics=dr.denoising_metrics())
    with pytest.raises(RuntimeError, match="diverges before training"):
        trainer.initial_loss_check(resolved, broken, data)


# ---------------------------------------------------------------------
# An untrained model is below its own input and exits 1 with the numbers
# ---------------------------------------------------------------------


def test_a_model_that_learned_nothing_is_below_its_input_and_the_run_exits_1_with_the_numbers(
    world, folders, tmp_path
):
    asked = mapping(tmp_path / "run", folders, "bfcnn")
    asked["optim_args"] = {"lr": 1e-12}
    code = cli.main(["image_denoising", *flatten("", asked, [])])
    assert code == 1
    summary = read_json(tmp_path / "run" / "results_summary.json")
    history = read_json(tmp_path / "run" / "training_history.json")
    best = max(history["val_psnr"][summary["selection"]["start_epoch"]:])
    assert best < summary["input_psnr"], "an untrained bias-free denoiser is below its own input"
    assert summary["success_value"] == pytest.approx(best - summary["input_psnr"], abs=1e-4)
    assert summary["success_value"] < 0 and summary["success_threshold"] == 1.0
    assert summary["success"] is False
    log = (tmp_path / "run" / "run.log").read_text()
    assert "Missed success_threshold 1 on val_psnr" in log


# ---------------------------------------------------------------------
# checkpoint= warm start
# ---------------------------------------------------------------------


def test_a_checkpoint_warm_starts_the_model_and_the_validation_set_is_the_same(world, folders, tmp_path):
    first = train_image_denoising(**mapping(tmp_path / "first", folders, "bfcnn"))
    first_summary = read_json(first.summary_path)
    final_model = Path(first.run_dir) / "final_model.keras"
    warm = train_image_denoising(**mapping(tmp_path / "warm", folders, "bfcnn", epochs=1, checkpoint=str(final_model)))
    cold = train_image_denoising(**mapping(tmp_path / "cold", folders, "bfcnn", epochs=1))
    warm_summary, cold_summary = read_json(warm.summary_path), read_json(cold.summary_path)
    # The first run's last epoch scored these weights on this very validation set: the warm start's epoch 0 is that score.
    last = read_json(Path(first.run_dir) / "training_history.json")["val_loss"][-1]
    assert warm_summary["initial_loss_sanity_eval"]["loss"] == pytest.approx(last, rel=1e-5)
    assert cold_summary["initial_loss_sanity_eval"]["loss"] != pytest.approx(last, rel=1e-3)
    assert warm_summary["input_psnr"] == first_summary["input_psnr"] == cold_summary["input_psnr"]


# ---------------------------------------------------------------------
# The identity loss of a set with no noise
# ---------------------------------------------------------------------


def test_the_identity_of_a_set_with_no_noise_has_loss_zero_and_the_psnr_cap():
    import tensorflow as tf

    clean = np.random.default_rng(0).uniform(0, 1, (6, 32, 32, 3)).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((clean, clean)).batch(4)
    scores = dr.identity_scores(dataset, "mse")
    assert scores["loss"] == 0.0 and scores["mae"] == 0.0
    assert scores["psnr"] == pytest.approx(70.0, abs=1e-3), "10 log10(1 / keras.config.epsilon()), the metric's cap"
    assert dr.identity_scores(dataset, "charbonnier")["loss"] == pytest.approx(dr.CHARBONNIER_EPSILON, rel=1e-4)


def test_a_validation_set_with_no_noise_is_refused_after_the_claim_with_exit_2(world, folders, tmp_path, capsys):
    # sigma in U(0, 1e-30): every noisy patch equals its clean one in float32, so the identity's loss is exactly 0.
    asked = mapping(tmp_path / "run", folders, "bfcnn")
    asked["data"] = {**asked["data"], "noise_sigma_max": 1e-30, "noise_sigma_max_start": 1e-30}
    code = cli.main(["image_denoising", *flatten("", asked, [])])
    assert code == 2
    assert (tmp_path / "run").is_dir(), "the refusal needs the data, so it comes after the claim"
    message = capsys.readouterr().err
    assert "no noise to remove" in message and "data.noise_sigma_max" in message and "'out' is now used" in message


def test_the_identity_scores_are_float32_and_independent_of_the_dtype_policy():
    import tensorflow as tf

    clean = np.random.default_rng(1).uniform(0, 1, (4, 16, 16, 3)).astype(np.float32)
    noisy = clean + np.float32(1e-3)
    dataset = tf.data.Dataset.from_tensor_slices((noisy, clean)).batch(4)
    plain = dr.identity_scores(dataset, "mse")
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        mixed = dr.identity_scores(dataset, "mse")
    finally:
        keras.mixed_precision.set_global_policy("float32")
    assert mixed == plain, "a half-precision copy of the input would move the PSNR"
    assert plain["psnr"] == pytest.approx(60.0, abs=0.05)
