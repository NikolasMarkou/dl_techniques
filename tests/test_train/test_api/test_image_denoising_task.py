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
* the bias-free contract (step 9, D-015, D-025): the static scan and the numeric probes run for the three bias-free
  families and never for the other two; a planted LayerNorm and a planted constant offset are refused before ``fit``
  with exit 2; the summary holds real ``init``, ``best`` and ``final`` readings; a model whose break appears only
  through training fails at the end with exit 1 and keeps its models; and the probes do not move the run (same seed,
  probes on and off, equal weights to the bit).

* analysis, figures and held-out cells (step 10, D-014, D-026): the task's analysis is data-free (no calibration, no
  ECE) where the control through ``analysis_config_for`` on the same model produces an ECE; the held-out cells are per
  dataset and sigma, hand recomputed, with the unclipped input PSNR at sigma_255 15, 25, 50 reading 24.61, 20.17,
  14.15 dB, the best and the final model on the same noisy images, no key averaging datasets, an image past the memory
  rule named and skipped, the figure files following ``analysis_args.figures``, ``data.test_dirs`` refused when it leaks
  into train or validation, and the pure helpers moved out of ``train.bfunet`` unchanged.

The tests write only under ``tmp_path``. Every real run sets ``max_train_files`` and ``max_val_files`` to the number of
files: the default of 10,000 repeats a small folder to fill an epoch (the DIV2K rule), which is 5,000 steps here.
"""

import dataclasses
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
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
from train.bfunet import common as bunet_common
from train.bfunet import eval_psnr_vs_noise as ev
from train.common import classification_run
from train.common import denoising_data as dd
from train.common import denoising_homogeneity as dh
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
    # The analysis of a denoiser is data-free (D-014): weights and spectral, read back with the data-free reader, and the
    # summary has no calibration number.
    assert summary["analyzer"]["status"] == "ok" and summary["analyzer"]["analyzers"] == ["weights", "spectral"]
    assert "failed_analyses" not in summary["analyzer"] and "ece" not in summary
    assert (run_dir / "model_analysis" / "analysis_results.json").is_file()
    assert not (run_dir / "model_analysis" / "summary_dashboard.png").exists(), "the empty library dashboard is removed"


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


def test_the_watch_metric_help_names_every_metric_the_task_accepts():
    """The accepted set is the training names and their ``val_`` twins (NOTE 8 of the iteration-1 review): the help lists all eight."""
    from train.api.trainer import _monitor_choices
    from train.common.denoising_run import DENOISING_METRIC_NAMES

    accepted = set(_monitor_choices(list(DENOISING_METRIC_NAMES)))
    assert accepted == {"loss", "psnr", "mae", "ssim", "val_loss", "val_psnr", "val_mae", "val_ssim"}
    help_text = next(f for f in dataclasses.fields(dargs.DenoiseMetricArgs) if f.name == "watch_metric").metadata["help"]
    words = set(re.findall(r"\b[a-z_]+\b", help_text))
    assert accepted <= words, sorted(accepted - words)


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


# ---------------------------------------------------------------------
# The bias-free contract in a run (step 9, D-015, D-025)
# ---------------------------------------------------------------------

TOLERANCE = dh.PROBE_TOLERANCE


@pytest.fixture
def spy(monkeypatch):
    """Counting wrappers around the three instruments, in call order, installed on the module the task reads."""
    calls = []

    def wrap(name):
        original = getattr(dh, name)

        def wrapper(*arguments, **keywords):
            calls.append(name)
            return original(*arguments, **keywords)

        monkeypatch.setattr(dh, name, wrapper)

    for name in ("scan_static", "init_probe_clone", "probe_numeric"):
        wrap(name)
    return calls


@pytest.mark.parametrize("family", FAMILIES)
def test_the_check_runs_for_the_three_bias_free_families_and_never_for_the_other_two(family, spy, world, folders, tmp_path):
    result = train_image_denoising(**mapping(tmp_path / "run", folders, family, epochs=1))
    report = read_json(result.summary_path)["homogeneity"]
    if family in BIAS_FREE:
        # One scan, one clone, and three readings: the init clone, the best weights and the final weights.
        assert spy == ["scan_static", "init_probe_clone", "probe_numeric", "probe_numeric", "probe_numeric"]
        assert report["status"] == "passed"
    else:
        assert spy == [], f"{family} is not bias-free, so the instrument is never called for it"
        assert report["status"] == "not_applicable" and "not bias-free" in report["reason"]
        assert set(report) == {"status", "reason"}


# -- planted models: a test-only family registered straight into the registry, so the resolver's refusal tables never
#    see them (the tables refuse the KEYS that break homogeneity; these break it from inside the factory).


@keras.saving.register_keras_serializable(package="test_denoising_step9")
class Shift(keras.layers.Layer):
    """Adds one scalar weight named ``shift`` (not a name the static scan lists) to its input."""

    def __init__(self, initial=0.0, learn=False, **kwargs):
        super().__init__(**kwargs)
        self.initial, self.learn = float(initial), bool(learn)

    def build(self, input_shape):
        self.shift = self.add_weight(
            shape=(1,), initializer=keras.initializers.Constant(self.initial), trainable=self.learn, name="shift")

    def call(self, x):
        return x + keras.ops.cast(self.shift, x.dtype)

    def get_config(self):
        return {**super().get_config(), "initial": self.initial, "learn": self.learn}


@keras.saving.register_keras_serializable(package="test_denoising_step9")
class MutedOffset(keras.layers.Layer):
    """A constant 0.5 behind a layer scale at 1e-6, named so its weight path ends ``gamma_scale/gamma`` like the real ones."""

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "gamma_scale")
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,), initializer=keras.initializers.Constant(1e-6), name="gamma")

    def call(self, x):
        return x + keras.ops.cast(self.gamma, x.dtype) * 0.5


def _planted(variant, num_classes, input_shape, tail):
    from train.api.denoising_models import create_bfcnn

    base = create_bfcnn(variant=variant, num_classes=num_classes, input_shape=input_shape)
    return keras.Model(base.input, tail(base.output), name="planted")


def create_planted_layernorm(*, variant, num_classes=None, input_shape=(None, None, 3), **model_args):
    return _planted(variant, num_classes, input_shape, keras.layers.LayerNormalization())


def create_planted_offset(*, variant, num_classes=None, input_shape=(None, None, 3), **model_args):
    return _planted(variant, num_classes, input_shape, Shift(initial=0.05, learn=False))


def create_planted_drift(*, variant, num_classes=None, input_shape=(None, None, 3), **model_args):
    return _planted(variant, num_classes, input_shape, Shift(initial=0.0, learn=True))


def create_planted_muted(*, variant, num_classes=None, input_shape=(None, None, 3), **model_args):
    return _planted(variant, num_classes, input_shape, MutedOffset())


def create_planted_still(*, variant, num_classes=None, input_shape=(None, None, 3), **model_args):
    """The control: the same wrapper with a shift that starts at 0 and never learns, so it is homogeneous."""
    return _planted(variant, num_classes, input_shape, Shift(initial=0.0, learn=False))


def resolve_planted(context, optim_args, model_args):
    return dict(model_args)


PLANTED = {
    "planted_layernorm": create_planted_layernorm,
    "planted_offset": create_planted_offset,
    "planted_drift": create_planted_drift,
    "planted_muted": create_planted_muted,
    "planted_still": create_planted_still,
}


@pytest.fixture
def planted(world, monkeypatch):
    template = registry.MODELS["bfcnn"]
    for family, factory in PLANTED.items():
        monkeypatch.setitem(registry.MODELS, family, registry.ModelSpec(
            family=family, factory=f"{__name__}:{factory.__name__}", variants_attr=template.variants_attr,
            tasks=template.tasks, defaults=template.defaults, resolve_model_args=f"{__name__}:resolve_planted",
            properties=("bias_free",),
        ))


def nothing_was_trained(out):
    """The run directory of a refusal after the claim holds no history, no model and no summary."""
    for name in ("training_log.csv", "training_history.json", "best_model.keras", "final_model.keras",
                 "results_summary.json"):
        assert not (out / name).exists(), name


def test_a_planted_layernorm_is_refused_statically_before_fit_with_exit_2(planted, folders, tmp_path, capsys):
    out = tmp_path / "run"
    code = cli.main(["image_denoising", *flatten("", mapping(out, folders, "planted_layernorm"), [])])
    assert code == 2
    message = capsys.readouterr().err
    assert "layernorm" in message and "LayerNormalization" in message and "declared bias-free" in message
    # The model cannot be built before the claim without initialising TensorFlow before `setup_gpu` (D-025), so this
    # refusal comes after it: `out` is used, says so, and holds no training.
    assert "'out' is now used" in message and out.is_dir()
    nothing_was_trained(out)


def test_a_planted_constant_offset_passes_the_scan_and_is_refused_by_the_init_probe_with_the_numbers(
    planted, folders, tmp_path, capsys
):
    out = tmp_path / "run"
    code = cli.main(["image_denoising", *flatten("", mapping(out, folders, "planted_offset"), [])])
    assert code == 2
    message = capsys.readouterr().err
    assert "numeric probe at its initial weights" in message and "per alpha" in message
    assert f"above the tolerance {TOLERANCE:g}" in message
    assert "offender" not in message, "the scan is blind to a weight named shift; the numeric probe is what refuses"
    nothing_was_trained(out)


def test_a_break_muted_by_a_layer_scale_is_invisible_at_init_and_refused_on_the_clone(planted, folders, tmp_path, capsys):
    # The constant sits behind a layer scale of 1e-6, as a real layer scale starts: the model as built reads clean, the
    # clone (scale 1) reads the offset. Probing the model itself would start this run.
    inputs = np.random.default_rng(0).uniform(0, 1, (4, SIZE, SIZE, 3)).astype("float32")
    model = create_planted_muted(variant="tiny")
    assert dh.scan_static(model)["passed"] is True and dh.scan_static(model)["layer_scales"] == ["gamma_scale/gamma"]
    assert dh.probe_numeric(model, inputs)["passed"] is True, "liveness: as built, the break is invisible"
    assert dh.probe_numeric(dh.init_probe_clone(model), inputs)["passed"] is False, "the clone shows it"
    out = tmp_path / "run"
    code = cli.main(["image_denoising", *flatten("", mapping(out, folders, "planted_muted"), [])])
    assert code == 2
    assert "numeric probe at its initial weights" in capsys.readouterr().err
    nothing_was_trained(out)


def test_the_control_with_a_shift_that_starts_at_zero_and_never_learns_is_not_refused(planted, folders, tmp_path):
    # The control of the two refusals above: the registered-test-family path is not what refuses them.
    result = train_image_denoising(**mapping(tmp_path / "run", folders, "planted_still", epochs=1))
    report = read_json(result.summary_path)["homogeneity"]
    assert report["status"] == "passed" and report["numeric"]["final"]["max_rel"] < TOLERANCE / 10


# -- real readings --------------------------------------------------------------------------------------------------


def numeric_of(summary):
    return summary["homogeneity"]["numeric"]


@pytest.mark.parametrize("finished", BIAS_FREE, indirect=True)
def test_a_two_epoch_model_has_real_init_best_and_final_readings(finished, world, folders):
    family, asked, result, summary, _ = finished
    report = summary["homogeneity"]
    assert report["status"] == "passed"
    assert report["static"] == {"passed": True, "counts": {}, "offenders": []}
    readings = report["numeric"]
    assert set(readings) == {"init", "best", "final"}
    for name, reading in readings.items():
        assert reading["passed"] is True and reading["tf32"] is False, name
        assert reading["tolerance"] == TOLERANCE and reading["alphas"] == [0.3, 1.3, 3.7, 6.1], name
        assert reading["patches"] == min(dh.PROBE_PATCHES, BATCH) and len(reading["per_alpha"]) == 4, name
        # A real reading: finite, above float zero (a power-of-two alpha or a placeholder reads exactly 0), and under
        # the tolerance with room to spare (the clean side measured at most 1e-5).
        assert 0.0 < reading["max_rel"] < TOLERANCE / 10, (name, reading["max_rel"])
        assert reading["max_rel"] == max(reading["per_alpha"]), name
    # The best weights of a 2-epoch run ARE the final weights (the window is the last epoch), the init clone is not
    # the trained model: the three readings are taken at two different sets of weights.
    assert summary["selection"]["final_is_best"] is True
    assert readings["init"]["max_rel"] != readings["final"]["max_rel"]
    # The best reading is the probe of the saved best weights on the same patches, taken again here from the file.
    config = config_of(Path(result.run_dir).parent / "reread", folders, family)
    trainer = ImageDenoisingTrainer(config)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, "tiny")
    inputs = dh.probe_batch(trainer.build_data(resolved, context).select_ds)
    again = dh.probe_numeric(keras.models.load_model(Path(result.run_dir) / "best_model.keras", compile=False), inputs)
    assert again["max_rel"] == pytest.approx(readings["best"]["max_rel"], rel=1e-3, abs=1e-9)
    assert readings["best"]["max_rel"] == pytest.approx(readings["final"]["max_rel"], rel=1e-3, abs=1e-9)


def running_variances(path):
    model = keras.models.load_model(path, compile=False)
    return [np.array(w) for w in model.weights if w.path.rpartition("/")[2] in dh._RUNNING_VAR_NAMES]


@pytest.mark.parametrize("finished", ["bfconvunext"], indirect=True)
def test_batchnorm_running_variance_moved_by_training_leaves_the_trained_reading_clean(finished):
    # The edge case: a batch norm's running variance starts at 1 and training moves it; the probe reads the trained
    # weights as they are (no clone), and a bias-free batch norm stays homogeneous whatever its statistics.
    _, _, result, summary, _ = finished
    variances = running_variances(Path(result.run_dir) / "final_model.keras")
    assert variances, "the family holds batch norms with a running variance"
    assert any(not np.allclose(v, 1.0) for v in variances), "liveness: training moved at least one running variance"
    assert numeric_of(summary)["final"]["passed"] is True


@pytest.mark.parametrize("finished", ["bfunet"], indirect=True)
def test_non_additive_noise_leaves_a_bias_free_network_homogeneous_and_the_summary_says_the_identity_does_not_hold(
    finished,
):
    # The fixture's bfunet run trains on clipped MULTIPLICATIVE noise: homogeneity is a property of the network, so
    # it passes, and `score_identity` (a statement about the noise) is what says the residual is not the score.
    _, _, _, summary, _ = finished
    assert summary["noise"]["kind"] == "multiplicative" and summary["noise"]["clip"] is True
    assert summary["homogeneity"]["status"] == "passed"
    assert summary["score_identity"]["holds"] is False
    assert any("multiplicative" in note and "homogeneity" in note for note in summary["notes"])


def test_the_best_and_the_final_reading_are_taken_at_different_weights_when_they_are_different_weights(windowed):
    # The scripted run of the selection tests: best epoch 3 of 4, so the best and the final weights differ.
    _, _, summary, _ = windowed
    assert summary["selection"]["final_is_best"] is False
    readings = numeric_of(summary)
    assert readings["best"]["max_rel"] != readings["final"]["max_rel"]
    assert all(reading["passed"] is True for reading in readings.values())


# -- a break that appears only through training ----------------------------------------------------------------------


def test_a_break_that_appears_only_after_training_fails_at_the_end_with_exit_1_and_keeps_the_run(planted, folders, tmp_path):
    out = tmp_path / "run"
    asked = mapping(out, folders, "planted_drift", optim_args={"lr": 1e-2})
    code = cli.main(["image_denoising", *flatten("", asked, [])])
    assert code == 1, "a failed check after training is a failed run (exit 1), not a refusal (exit 2)"
    summary = read_json(out / "results_summary.json")
    readings = numeric_of(summary)
    # Born clean: the scan and the init probe passed (the shift starts at 0), so the run started.
    assert summary["homogeneity"]["static"]["passed"] is True and readings["init"]["passed"] is True
    assert readings["init"]["max_rel"] < TOLERANCE / 10
    # Broken by training: the learned offset is read at the best and the final weights.
    assert readings["best"]["passed"] is False and readings["final"]["passed"] is False
    assert readings["best"]["max_rel"] > 2 * TOLERANCE and readings["final"]["max_rel"] > 2 * TOLERANCE
    assert summary["homogeneity"]["status"] == "failed"
    assert summary["success"] is False
    assert summary["success_blocked_by"] == ["homogeneity_best", "homogeneity_final"]
    # The training is kept: models, history, log.
    for name in ("best_model.keras", "final_model.keras", "training_log.csv", "training_history.json"):
        assert (out / name).is_file(), name
    assert len(read_json(out / "training_history.json")["loss"]) == EPOCHS
    log = (out / "run.log").read_text()
    assert "not degree-1 homogeneous" in log and "homogeneity_best" in log


# -- the diagnostic does not move the run ----------------------------------------------------------------------------


def inert_probes(monkeypatch):
    """The three instruments replaced by stand-ins that read nothing, so a run with them is the run without a probe."""
    clean = {"passed": True, "counts": {}, "offenders": [], "layer_scales": []}
    monkeypatch.setattr(dh, "check_before_fit", lambda model, inputs, precision="float32": {
        "static": clean, "init": dh.skipped_reading(precision)})
    monkeypatch.setattr(dh, "read_weights", lambda model, inputs, precision="float32": dh.skipped_reading(precision))


@pytest.mark.parametrize("family", ["bfcnn", "bfconvunext"])
def test_the_same_seed_with_the_probes_on_and_off_gives_equal_weights_to_the_bit(family, world, folders, tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        inert_probes(patch)
        off = train_image_denoising(**mapping(tmp_path / "off", folders, family))
    assert read_json(off.summary_path)["homogeneity"]["status"] == "passed_static_only", "liveness: the stand-ins ran"
    on = train_image_denoising(**mapping(tmp_path / "on", folders, family))
    assert read_json(on.summary_path)["homogeneity"]["status"] == "passed"
    for name in ("final_model.keras", "best_model.keras"):
        assert same_weights(saved_weights(Path(on.run_dir) / name), saved_weights(Path(off.run_dir) / name)), name
    assert read_json(Path(on.run_dir) / "training_history.json") == read_json(Path(off.run_dir) / "training_history.json")


def random_snapshot(model):
    """Every random state a diagnostic could move: the model's generators, Keras' global one, Python's and NumPy's."""
    global_generator = keras.src.backend.common.global_state.get_global_attribute("global_seed_generator")
    return {
        "generators": [np.array(g.state.numpy()) for g in dh._seed_generators(model)],
        "global": None if global_generator is None else np.array(global_generator.state.numpy()),
        "python": random.getstate(),
        "numpy_keys": np.random.get_state()[1].copy(),
        "numpy_pos": np.random.get_state()[2],
    }


def assert_snapshots_equal(left, right):
    assert len(left["generators"]) == len(right["generators"])
    assert all(np.array_equal(a, b) for a, b in zip(left["generators"], right["generators"]))
    assert (left["global"] is None) == (right["global"] is None)
    if left["global"] is not None:
        assert np.array_equal(left["global"], right["global"])
    assert left["python"] == right["python"]
    assert np.array_equal(left["numpy_keys"], right["numpy_keys"]) and left["numpy_pos"] == right["numpy_pos"]


@pytest.mark.parametrize("family", BIAS_FREE)
def test_the_checks_before_and_after_fit_leave_every_random_state_where_they_found_it(family, world, folders, tmp_path):
    keras.utils.set_random_seed(7)
    config = config_of(tmp_path / "a", folders, family)
    trainer = ImageDenoisingTrainer(config)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, "tiny")
    data = trainer.build_data(resolved, context)
    model = trainer.build_model(resolved, context, data)
    keras.random.normal((1,))                         # make sure Keras' global generator exists and has moved
    random.random()
    np.random.random()
    before = random_snapshot(model)
    trainer._check_before_fit(model, data, "float32")
    assert_snapshots_equal(before, random_snapshot(model))
    dh.read_weights(model, dh.probe_batch(data.select_ds))
    assert_snapshots_equal(before, random_snapshot(model))


def test_a_half_precision_run_scans_but_reads_nothing_numerically_and_says_so(world, folders, tmp_path, spy):
    # The tolerance is a float32 number (D-019, D-025): under mixed_float16 a clean model reads above it from the
    # rounding of its compute alone, so the numeric readings are skipped and the report says so; the scan still runs.
    # A mixed_float16 run on the CPU is slow, so this goes as far as the pre-fit check, which `build_model` runs.
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        config = config_of(tmp_path / "a", folders, "bfcnn", precision="mixed_float16")
        trainer = ImageDenoisingTrainer(config)
        resolved, context = trainer._resolve(trainer.describe_data(config), "bfcnn", "tiny")
        data = trainer.build_data(resolved, context)
        trainer.build_model(resolved, context, data)
    finally:
        keras.mixed_precision.set_global_policy("float32")
    assert spy == ["scan_static"], "the scan ran; no clone was built and no reading was taken"
    assert trainer._homogeneity["init"]["status"] == "skipped" and "float32 only" in trainer._homogeneity["init"]["reason"]
    report = dh.homogeneity_report(
        trainer._homogeneity["static"], trainer._homogeneity["init"],
        dh.skipped_reading("mixed_float16"), dh.skipped_reading("mixed_float16"))
    assert report["status"] == "passed_static_only" and dh.failed_checks(report) == ()


# ---------------------------------------------------------------------
# Step 10: held-out cells, analysis, figures, data.test_dirs (D-014, D-026)
# ---------------------------------------------------------------------

SIGMAS = (15.0, 25.0, 50.0)
#: ``-20 log10(sigma_255 / 255)``: what an unclipped additive noisy image scores against its clean one.
INPUT_PSNR = {"15": 24.61, "25": 20.17, "50": 14.15}


def write_sized(root: Path, sizes, seed: int, value=None) -> Path:
    """Write one smooth RGB PNG per ``(height, width)`` (or a constant image of ``value`` when it is given)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index, (height, width) in enumerate(sizes):
        if value is not None:
            pixels = np.full((height, width, 3), round(value * 255), np.uint8)
        else:
            rows, columns = np.mgrid[0:height, 0:width]
            tint = rng.uniform(0.25, 0.75, (1, 1, 3))
            wave = 0.2 * np.sin(columns / 9.0 + index)[..., None] * np.cos(rows / 7.0)[..., None]
            pixels = (np.clip(tint + wave, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(root / f"h{index:02d}.png")
    return root


def files_of(root: Path):
    return sorted(str(path) for path in root.glob("*.png"))


@pytest.fixture(scope="module")
def held_sets(tmp_path_factory):
    """Two held-out sets of images whose sides are not multiples of 8 (``bfunet``'s stride), of different sizes."""
    root = tmp_path_factory.mktemp("held_sets")
    return {
        "alpha": write_sized(root / "alpha", [(50, 70), (45, 38), (64, 64)], 10),
        "beta": write_sized(root / "beta", [(33, 47), (40, 40)], 11),
    }


def held_request(out, folders, held_sets, family="bfunet", epochs=EPOCHS, **analysis):
    """A run that scores both sets with additive unclipped noise (so the input PSNR is the textbook one)."""
    request = mapping(out, folders, family, epochs=epochs)
    request["data"].update(noise_kind="additive", noise_clip=False)
    request["data"].pop("noise_composite_ratio", None)
    request["data"]["test_dirs"] = [f"alpha={held_sets['alpha']}", str(held_sets["beta"])]
    request["analysis_args"] = {"figures": False, "model_analysis": False, **analysis}
    return request


@pytest.fixture(scope="module")
def held_run(world, folders, held_sets, tmp_path_factory):
    """One real ``bfunet`` run with the analysis, the figures and both held-out sets on."""
    out = tmp_path_factory.mktemp("run_held") / "run"
    asked = held_request(out, folders, held_sets, figures=True, model_analysis=True)
    result = train_image_denoising(**asked)
    return asked, result, read_json(result.summary_path)


# -- the cells -----------------------------------------------------------------------------------------------------


def hand_cell(model_path, files, dataset_index, sigma_255, size_multiple):
    """The held-out cell of one set at one sigma, recomputed with numpy: pad, ``model.predict``, crop, PSNR."""
    model = keras.models.load_model(model_path, compile=False)
    noisy_psnr, restored_psnr = [], []
    for image_index, path in enumerate(files):
        clean = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        noisy = dd.add_noise(
            tf.constant(clean), tf.constant(sigma_255 / 255.0, tf.float32),
            dr.held_out_noise_seed(dr.HELD_OUT_SEED, dataset_index, image_index, sigma_255),
            kind="additive", composite_ratio=0.5, clip=False,
        ).numpy()
        height, width = clean.shape[:2]
        padded = np.pad(noisy, ((0, -height % size_multiple), (0, -width % size_multiple), (0, 0)), mode="reflect")
        out = model.predict(padded[None], verbose=0)[0][:height, :width]
        for target, image in ((noisy_psnr, noisy), (restored_psnr, out)):
            target.append(10.0 * math.log10(1.0 / np.mean((image.astype(np.float64) - clean.astype(np.float64)) ** 2)))
    return float(np.mean(noisy_psnr)), float(np.mean(restored_psnr))


def test_the_held_out_cells_are_per_dataset_and_sigma_and_recomputed_by_hand(held_run, held_sets):
    _, result, summary = held_run
    held = summary["held_out"]
    assert list(held) == ["alpha", "beta"], "one row per set, named as the entry says (a bare directory by its name)"
    run_dir = Path(result.run_dir)
    for index, name in enumerate(held):
        files = files_of(held_sets[name])
        assert list(held[name]) == ["15", "25", "50"]
        for sigma in SIGMAS:
            cell = held[name][dr.sigma_key(sigma)]
            assert set(cell) == {"input_psnr", "best_psnr", "final_psnr", "n"} and cell["n"] == len(files)
            hand_input, hand_final = hand_cell(run_dir / "final_model.keras", files, index, sigma, 8)
            _, hand_best = hand_cell(run_dir / "best_model.keras", files, index, sigma, 8)
            assert cell["input_psnr"] == pytest.approx(hand_input, abs=1e-4)
            assert cell["final_psnr"] == pytest.approx(hand_final, abs=5e-3)
            assert cell["best_psnr"] == pytest.approx(hand_best, abs=5e-3)
            # The input of an unclipped additive run reads the textbook number within the sampling error of a few
            # small images (about 0.04 dB each); stated bound 0.3 dB.
            assert cell["input_psnr"] == pytest.approx(INPUT_PSNR[dr.sigma_key(sigma)], abs=0.3)
    # A 2-epoch run's best epoch is its last: the two models are the same weights, so their cells agree; the
    # test that they are scored on the same noise when they differ is `test_the_two_models_see_the_same_noisy_images`.
    assert summary["selection"]["final_is_best"] is True
    assert all(cell["best_psnr"] == cell["final_psnr"] for per_sigma in held.values() for cell in per_sigma.values())


def test_the_held_out_block_names_its_protocol(held_run, held_sets):
    _, _, summary = held_run
    info = summary["held_out_info"]
    assert info["sigmas_255"] == [15.0, 25.0, 50.0] and info["seed"] == dr.HELD_OUT_SEED
    assert info["noise"] == {"kind": "additive", "composite_ratio": None, "clip": False}
    assert info["size_multiple"] == 8 and info["channels"] == 3 and info["skipped_images"] == {}
    assert info["memory_rule"] == {"gb_per_megapixel": dr.HELD_OUT_GB_PER_MEGAPIXEL, "budget_gb": dr.HELD_OUT_MEMORY_BUDGET_GB}
    assert any("held_out[set][sigma]" in note and "no key averages sets" in note for note in summary["notes"])


def test_no_key_of_the_held_out_blocks_averages_datasets(held_run):
    _, _, summary = held_run

    def keys(value):
        if isinstance(value, dict):
            for key, inner in value.items():
                yield key
                yield from keys(inner)

    names = [key.lower() for block in ("held_out", "held_out_info") for key in keys(summary[block])]
    assert not [name for name in names if any(word in name for word in ("mean", "average", "overall", "total", "pooled"))]
    # And the control for the scan: the keys it read are the real ones.
    assert {"alpha", "beta", "input_psnr", "best_psnr", "final_psnr", "n"} <= set(names)


def test_a_run_without_test_dirs_says_held_out_is_skipped_with_the_reason(finished):
    _, _, _, summary, _ = finished
    assert summary["held_out"] == {
        "status": "skipped", "reason": "data.test_dirs is not set: no held-out image was scored",
        "sigmas_255": [15.0, 25.0, 50.0],
    }
    assert "held_out_info" not in summary
    assert any("`held_out` is skipped" in note for note in summary["notes"])


# -- the scoring functions, on fixtures with known answers ----------------------------------------------------------


def spec_for(size_multiple=1, **changes):
    values = dict(
        sigmas_255=SIGMAS, noise_kind="additive", composite_ratio=0.5, clip=False, channels=3,
        size_multiple=size_multiple,
    )
    values.update(changes)
    return dr.HeldOutSpec(**values)


def test_psnr_of_a_known_fixture_is_hand_computed(tmp_path):
    files = files_of(write_sized(tmp_path / "flat", [(40, 40), (48, 24)], 0, value=0.5))
    clean = np.asarray(Image.open(files[0]), dtype=np.float64) / 255.0
    level = float(clean[0, 0, 0])
    # A denoiser that returns black: the MSE is level^2, the PSNR -20 log10(level).
    # (The decoder returns float32, so the level of the image is 128 / 255 to within 6e-8; the PSNR to 6e-7 dB.)
    black = dr.score_held_out(lambda noisy: np.zeros_like(noisy), {"flat": files}, spec_for())
    np.testing.assert_allclose(black.output_psnr["flat"]["25"], -20.0 * math.log10(level), atol=1e-5)
    # A denoiser that is off by exactly 0.1 everywhere: MSE 0.01, PSNR 20 dB (float32 rounding is below 1e-4 dB).
    off = dr.score_held_out(
        lambda noisy: np.full_like(noisy, level + 0.1), {"flat": files}, spec_for())
    np.testing.assert_allclose(off.output_psnr["flat"]["25"], 20.0, atol=1e-4)
    cells = dr.pair_held_out(black, off)
    assert cells["flat"]["25"]["best_psnr"] == pytest.approx(-20.0 * math.log10(level), abs=1e-5)
    assert cells["flat"]["25"]["final_psnr"] == pytest.approx(20.0, abs=1e-4) and cells["flat"]["25"]["n"] == 2
    # The identity restores nothing: its PSNR is the input's, cell for cell.
    same = dr.score_held_out(lambda noisy: noisy, {"flat": files}, spec_for())
    np.testing.assert_array_equal(same.output_psnr["flat"]["25"], same.input_psnr["flat"]["25"])


def test_the_helper_psnr_matches_the_formula_on_a_batch():
    clean = np.zeros((2, 4, 4, 1))
    denoised = np.stack([np.full((4, 4, 1), 0.1), np.full((4, 4, 1), 0.5)])
    np.testing.assert_allclose(dr.psnr_per_image(denoised, clean), [20.0, 10.0 * math.log10(4.0)], atol=1e-9)
    assert dr.psnr_per_image(clean, clean)[0] == pytest.approx(120.0)  # the 1e-12 floor of the MSE


def test_the_unclipped_input_psnr_reads_the_textbook_numbers_and_clip_moves_it(tmp_path):
    files = files_of(write_sized(tmp_path / "gray", [(128, 128)] * 4, 0, value=0.5))
    dark = files_of(write_sized(tmp_path / "dark", [(128, 128)] * 4, 0, value=0.1))
    scored = dr.score_held_out(lambda noisy: noisy, {"gray": files}, spec_for())
    for key, expected in INPUT_PSNR.items():
        # 4 images of 49,152 samples: the PSNR of one is -10 log10(mean n^2), about 0.03 dB of sampling error; the bound
        # is 0.1 dB, and the values the issue states (24.61, 20.17, 14.15) are -20 log10(sigma_255 / 255).
        assert float(np.mean(scored.input_psnr["gray"][key])) == pytest.approx(expected, abs=0.1), key
    # The control: on a dark image clipping at 0 removes the negative half of the noise, so the clipped run reads a
    # different (higher) input PSNR, and the run's `clip` is what the held-out noise obeys.
    free = dr.score_held_out(lambda noisy: noisy, {"dark": dark}, spec_for())
    clipped = dr.score_held_out(lambda noisy: noisy, {"dark": dark}, spec_for(clip=True))
    assert np.mean(clipped.input_psnr["dark"]["50"]) > np.mean(free.input_psnr["dark"]["50"]) + 1.0


@pytest.mark.parametrize("kind", ["multiplicative", "composite"])
def test_the_held_out_noise_follows_the_runs_noise_kind(tmp_path, kind):
    files = files_of(write_sized(tmp_path / "gray", [(96, 96)] * 2, 0, value=0.5))
    additive = dr.score_held_out(lambda noisy: noisy, {"gray": files}, spec_for())
    other = dr.score_held_out(lambda noisy: noisy, {"gray": files}, spec_for(noise_kind=kind))
    # On a 0.5 image multiplicative noise has half the additive amplitude (6.0 dB more input PSNR) and composite noise
    # (ratio 0.5) has variance sigma^2 (0.25 + 0.25), half the additive power (3.0 dB more).
    gain = float(np.mean(other.input_psnr["gray"]["25"] - additive.input_psnr["gray"]["25"]))
    assert gain == pytest.approx({"multiplicative": 6.0, "composite": 3.0}[kind], abs=0.3)


def test_the_two_models_see_the_same_noisy_images(tmp_path, monkeypatch):
    files_a = files_of(write_sized(tmp_path / "a", [(40, 40), (33, 47)], 1))
    files_b = files_of(write_sized(tmp_path / "b", [(36, 36)], 2))
    seen = []
    real = dr.add_noise

    def spy(clean, sigma, seed, **kwargs):
        seen.append((tuple(int(v) for v in seed.numpy()), round(float(sigma), 6), hash(clean.numpy().tobytes())))
        return real(clean, sigma, seed, **kwargs)

    monkeypatch.setattr(dr, "add_noise", spy)
    sets = {"a": files_a, "b": files_b}
    first = dr.score_held_out(lambda noisy: noisy * 0.9, sets, spec_for())
    second = dr.score_held_out(lambda noisy: noisy * 0.5, sets, spec_for())
    half = len(seen) // 2
    assert len(seen) == 2 * 3 * 3 and seen[:half] == seen[half:], "every image at every sigma gets one seed, per model"
    assert len({entry[0] for entry in seen[:half]}) == half, "and the seeds of the cells are all different"
    cells = dr.pair_held_out(first, second)
    assert cells["a"]["25"]["best_psnr"] != cells["a"]["25"]["final_psnr"], "different models, different outputs"
    assert cells["a"]["25"]["input_psnr"] == pytest.approx(float(np.mean(first.input_psnr["a"]["25"])))
    # The control: a second scoring on other noise is refused, not averaged into a difference that is part noise.
    other_noise = dr.score_held_out(lambda noisy: noisy * 0.5, sets, spec_for(seed=dr.HELD_OUT_SEED + 1))
    with pytest.raises(ValueError, match="different noisy inputs"):
        dr.pair_held_out(first, other_noise)
    # Nor is a scoring of other images.
    fewer = dr.score_held_out(lambda noisy: noisy, {"a": files_a}, spec_for())
    with pytest.raises(ValueError, match="same images"):
        dr.pair_held_out(first, fewer)


def test_the_seed_is_a_constant_of_the_image_place_never_of_the_run(tmp_path):
    first = dr.held_out_noise_seed(dr.HELD_OUT_SEED, 2, 7, 25.0)
    assert [int(v) for v in first.numpy()] == [dr.HELD_OUT_SEED, (2 * 10 ** 6 + 7) * 10 ** 6 + 25000]
    assert dr.HELD_OUT_SEED != 42, "not the default `seed` of a run"
    for bad in ((1000, 0, 25.0), (0, 10 ** 6, 25.0), (0, 0, 0.0), (0, 0, 256.0), (0, -1, 25.0)):
        with pytest.raises(ValueError, match="out of range"):
            dr.held_out_noise_seed(dr.HELD_OUT_SEED, *bad)


def test_no_cell_averages_over_datasets_and_each_set_equals_its_own_scoring(tmp_path):
    files_a = files_of(write_sized(tmp_path / "a", [(40, 40), (33, 47)], 3))
    files_b = files_of(write_sized(tmp_path / "b", [(36, 36), (64, 48), (30, 30)], 4))
    both = dr.pair_held_out(*(dr.score_held_out(fn, {"a": files_a, "b": files_b}, spec_for())
                              for fn in (lambda noisy: noisy * 0.9, lambda noisy: noisy * 0.8)))
    alone = dr.pair_held_out(*(dr.score_held_out(fn, {"a": files_a}, spec_for())
                               for fn in (lambda noisy: noisy * 0.9, lambda noisy: noisy * 0.8)))
    assert list(both) == ["a", "b"], "no pooled row"
    assert both["a"] == alone["a"], "a set's cell does not depend on the other sets"
    assert both["a"]["25"]["n"] == 2 and both["b"]["25"]["n"] == 3, "each cell counts its own images"
    for key in ("15", "25", "50"):
        assert both["a"][key]["final_psnr"] != both["b"][key]["final_psnr"]
        pooled_input = (2 * both["a"][key]["input_psnr"] + 3 * both["b"][key]["input_psnr"]) / 5
        assert both["a"][key]["input_psnr"] != pytest.approx(pooled_input, abs=1e-6)


def test_an_image_past_the_memory_rule_is_named_and_skipped_in_both_passes(tmp_path):
    small = write_sized(tmp_path / "set", [(40, 40)], 5)
    big = write_sized(tmp_path / "big", [(100, 60)], 6)
    files = files_of(small) + files_of(big)
    # 40 x 40 is 0.00256 GB at 1.6 GB per Mpx; 100 x 60 padded to a multiple of 8 (104 x 64) is 0.0106 GB.
    spec = spec_for(size_multiple=8, budget_gb=0.005)
    assert dr.whole_image_peak_gb(40, 40, 8) == pytest.approx(0.00256) and dr.whole_image_peak_gb(100, 60, 8) == pytest.approx(0.0106496)
    first = dr.score_held_out(lambda noisy: noisy, {"set": files}, spec)
    second = dr.score_held_out(lambda noisy: noisy, {"set": files}, spec)
    assert [row["path"] for row in first.skipped["set"]] == [files_of(big)[0]]
    row = first.skipped["set"][0]
    assert (row["height"], row["width"], row["budget_gb"]) == (100, 60, 0.005) and row["estimate_gb"] == 0.01
    cells = dr.pair_held_out(first, second)
    assert cells["set"]["25"]["n"] == 1, "the skipped image is not in the mean"
    # With nothing left to score, the cell says so instead of a number.
    nothing = dr.score_held_out(lambda noisy: noisy, {"set": files_of(big)}, spec)
    assert dr.pair_held_out(nothing, nothing)["set"]["25"] == {
        "input_psnr": None, "best_psnr": None, "final_psnr": None, "n": 0}
    # The default rule: the largest DIV2K validation image (2040 x 2040, padded to 2048) is inside it, a 6 Mpx image is not.
    assert dr.whole_image_peak_gb(2040, 2040, 16) < dr.HELD_OUT_MEMORY_BUDGET_GB < dr.whole_image_peak_gb(2448, 2448, 1)
    assert dr.whole_image_size(files[0]) == (40, 40)


def test_the_eager_whole_image_pass_equals_predict(tmp_path):
    model = keras.Sequential([keras.layers.Input((None, None, 3)), keras.layers.Conv2D(3, 3, padding="same")])
    noisy = np.random.default_rng(0).uniform(0, 1, (37, 45, 3)).astype("float32")
    eager = dr.denoise_whole_image(model, noisy, 8, eager=True)
    legacy = dr.denoise_whole_image(model, noisy, 8)
    assert eager.shape == legacy.shape == (37, 45, 3)
    np.testing.assert_allclose(eager, legacy, atol=1e-5)


# -- the task's run ---------------------------------------------------------------------------------------------------


def test_the_two_models_of_a_real_run_see_the_same_noisy_images(world, folders, held_sets, tmp_path, monkeypatch):
    """A run whose best and final weights differ: the cells pair on the noise (`pair_held_out` would refuse otherwise)."""
    seen = []
    real = dr.add_noise

    def spy(clean, sigma, seed, **kwargs):
        seen.append((tuple(int(v) for v in seed.numpy()), round(float(sigma), 6)))
        return real(clean, sigma, seed, **kwargs)

    monkeypatch.setattr(dr, "add_noise", spy)
    # The final model is made unlike the best one (its weights halved after the shared finalisation has saved and
    # reloaded them), because a 2-epoch run's best epoch is its last; the cells must still pair on the noise.
    finalize = classification_run.finalize_last_epoch

    def finalize_then_change(model, *args, **kwargs):
        returned = finalize(model, *args, **kwargs)
        model.set_weights([0.5 * weight for weight in model.get_weights()])
        return returned

    monkeypatch.setattr(classification_run, "finalize_last_epoch", finalize_then_change)
    result = train_image_denoising(**held_request(tmp_path / "run", folders, held_sets, family="bfcnn"))
    summary = read_json(result.summary_path)
    images = len(files_of(held_sets["alpha"])) + len(files_of(held_sets["beta"]))
    half = images * len(SIGMAS)
    assert len(seen) == 2 * half and seen[:half] == seen[half:]
    cell = summary["held_out"]["alpha"]["25"]
    assert cell["best_psnr"] != cell["final_psnr"], "the control: the two models differ, the noise they saw did not"
    assert all(cell["input_psnr"] is not None for per_sigma in summary["held_out"].values() for cell in per_sigma.values())


def test_the_task_analysis_has_no_calibration_where_the_control_through_analysis_config_for_has_an_ece(
    world, folders, tmp_path
):
    from dl_techniques.models.vision.bias_free_denoisers.bfcnn import create_bfcnn_variant
    from train.common import evaluation

    config = config_of(tmp_path / "unused", folders, "bfcnn")
    model = create_bfcnn_variant("tiny", input_shape=(None, None, 3))
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (8, SIZE, SIZE, 3)).astype("float32")
    labels = rng.integers(0, 3, (8, SIZE, SIZE))
    history = keras.callbacks.History()
    history.history = {"loss": [1.0, 0.5], "val_loss": [1.1, 0.6]}
    task = ImageDenoisingTrainer(config).run_analysis(config, model, (x, labels), history, "task", str(tmp_path / "task"))
    assert task is not None and task.calibration_metrics == {} and not task.analysis_failures
    assert task.weight_stats and task.spectral_summary_per_model
    assert not (tmp_path / "task" / "model_analysis" / "summary_dashboard.png").exists()
    # The control, on the same model and data: the rank rule's configuration reads the image as class probabilities and
    # produces an ECE (a number about nothing a denoiser is).
    control = evaluation.run_model_analysis(
        model, (x, labels), history, "control", str(tmp_path / "control"),
        config=evaluation.analysis_config_for(model, sample=x[:1], random_state=0),
    )
    assert isinstance(control.calibration_metrics["control"]["ece"], float)
    assert config.analysis_args.model_analysis is True


def test_the_analysis_block_of_a_run_is_data_free_with_no_ece(held_run):
    _, result, summary = held_run
    assert summary["analyzer"]["status"] == "ok" and summary["analyzer"]["analyzers"] == ["weights", "spectral"]
    assert "ece" not in summary and "ece" not in summary["visualizations"]
    results = read_json(Path(result.run_dir) / "model_analysis" / "analysis_results.json")
    assert results["calibration_metrics"] == {} and results["model_metrics"] == {}
    assert results["weight_stats"] and results["spectral_summary_per_model"]


def test_the_figures_switch_on_writes_the_triptychs_the_sigma_panel_and_the_whiteness(held_run):
    _, result, summary = held_run
    visualizations = Path(result.run_dir) / "visualizations"
    names = sorted(path.name for path in visualizations.iterdir())
    assert names == [
        "epoch_001_triptych.png", "epoch_002_triptych.png", "psnr_vs_sigma.png", "residual_whiteness.png",
        "training_dashboard.png",
    ]
    assert summary["visualizations"] == {
        "files": ["psnr_vs_sigma.png", "residual_whiteness.png"], "failed": [], "skipped": False, "not_drawn": {},
    }
    for name in names:
        assert (visualizations / name).stat().st_size > 2000, name


def test_the_figures_switch_off_writes_none_of_the_tasks_figures_for_every_family(finished):
    _, _, finished_result, finished_summary, _ = finished
    left = [path.name for path in (Path(finished_result.run_dir) / "visualizations").glob("*")]
    assert not [name for name in left if "triptych" in name or name in ("psnr_vs_sigma.png", "residual_whiteness.png")]
    assert finished_summary["visualizations"] == {"files": [], "failed": [], "skipped": True}


def test_a_run_with_figures_on_and_no_test_dirs_draws_the_whiteness_and_names_the_missing_psnr_figure(
    world, folders, tmp_path
):
    request = mapping(tmp_path / "run", folders, "bfcnn", epochs=1)
    request["analysis_args"] = {"figures": True, "model_analysis": False}
    result = train_image_denoising(**request)
    visualizations = read_json(result.summary_path)["visualizations"]
    assert visualizations["files"] == ["residual_whiteness.png"] and visualizations["failed"] == []
    assert visualizations["not_drawn"] == {
        "psnr_vs_sigma.png": "no held-out set was scored (data.test_dirs is not set)"}
    assert (Path(result.run_dir) / "visualizations" / "epoch_001_triptych.png").is_file()


# -- the figure functions -------------------------------------------------------------------------------------------


def test_white_noise_reads_inside_its_band_and_a_correlated_residual_does_not():
    rng = np.random.default_rng(0)
    white = rng.standard_normal((8, 32, 32, 3))
    read = dr.residual_autocorrelation(white, 8)
    band = 1.96 / math.sqrt(32 * 32 * 24)
    assert np.max(np.abs(read["horizontal"])) < 2.5 * band and np.max(np.abs(read["vertical"])) < 2.5 * band
    assert list(read["lags"]) == list(range(1, 9)) and int(read["planes"]) == 24
    smooth = np.cumsum(white, axis=2) / 4.0  # a horizontal random walk: strongly correlated along the width
    structured = dr.residual_autocorrelation(smooth, 8)
    assert structured["horizontal"][0] > 0.8 and structured["horizontal"][0] > 10 * abs(structured["vertical"][0])
    for bad in (0, 32):
        with pytest.raises(ValueError, match="max_lag"):
            dr.residual_autocorrelation(white, bad)
    with pytest.raises(ValueError, match="N, H, W, C"):
        dr.residual_autocorrelation(white[0], 4)


def test_the_end_figures_are_independent_and_a_failing_one_is_listed(tmp_path, monkeypatch):
    cells = {name: {"15": {"input_psnr": 24.6, "best_psnr": 30.0, "final_psnr": 29.8, "n": 2},
                    "50": {"input_psnr": 14.1, "best_psnr": 25.0, "final_psnr": 24.9, "n": 2}}
             for name in ("a", "b", "c", "d", "e")}
    residual = np.random.default_rng(1).standard_normal((4, 24, 24, 3))
    record = dr.write_end_figures(tmp_path, cells=cells, residual=residual, title="t")
    assert record == {"files": ["psnr_vs_sigma.png", "residual_whiteness.png"], "failed": [], "skipped": False, "not_drawn": {}}
    with pytest.raises(ValueError, match="no dataset"):
        dr.plot_psnr_vs_sigma({}, tmp_path / "x.png")
    monkeypatch.setattr(dr, "plot_psnr_vs_sigma", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    failed = dr.write_end_figures(tmp_path / "again", cells=cells, residual=residual)
    assert failed["failed"] == ["psnr_vs_sigma.png"] and failed["files"] == ["residual_whiteness.png"]


def test_the_triptych_is_four_columns_of_a_fixed_batch_and_reports_each_rows_psnr(tmp_path):
    rng = np.random.default_rng(2)
    clean = rng.uniform(0.2, 0.8, (3, 16, 16, 3)).astype("float32")
    noisy = clean + 0.1
    scores = dr.plot_triptych(noisy, clean, clean, tmp_path / "t.png", "title")
    assert scores == pytest.approx([120.0, 120.0, 120.0]), "the denoised column is the clean image: the 1e-12 floor"
    scores = dr.plot_triptych(noisy, noisy, clean, tmp_path / "u.png")
    assert scores == pytest.approx([20.0] * 3, abs=1e-4)
    assert (tmp_path / "t.png").stat().st_size > 1000
    with pytest.raises(ValueError, match="three equal"):
        dr.plot_triptych(noisy, clean[:2], clean, tmp_path / "v.png")


def test_the_triptych_callback_draws_on_its_cadence_never_stops_a_run_and_lists_failures(tmp_path):
    rng = np.random.default_rng(3)
    clean = rng.uniform(0.2, 0.8, (2, 16, 16, 3)).astype("float32")
    model = keras.Sequential([keras.layers.Input((16, 16, 3)), keras.layers.Conv2D(3, 1)])
    callback = dr.DenoisingTriptychCallback(clean + 0.1, clean, tmp_path / "figs", "run")
    callback.set_model(model)
    callback.set_params({"epochs": 40})  # a figure every 2 epochs
    for epoch in range(5):
        callback.on_epoch_end(epoch, {"val_psnr": 20.0 + epoch})
    callback.on_train_end()
    assert callback.written == ["epoch_001_triptych.png", "epoch_002_triptych.png", "epoch_004_triptych.png",
                                "epoch_005_triptych.png"] and callback.failed == {}
    assert sorted(path.name for path in (tmp_path / "figs").iterdir()) == sorted(callback.written)
    class Broken:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("the model cannot predict")

    broken = dr.DenoisingTriptychCallback(clean + 0.1, clean, tmp_path / "broken")
    broken.set_model(Broken())
    broken.on_epoch_end(0, {})
    assert list(broken.failed) == ["epoch_001_triptych.png"] and broken.written == []
    with pytest.raises(ValueError, match="equal, non-zero"):
        dr.DenoisingTriptychCallback(clean, clean[:1], tmp_path)


# -- data.test_dirs ---------------------------------------------------------------------------------------------------


def test_a_test_dirs_entry_is_a_directory_or_name_equals_directory():
    assert dargs.split_test_dir("/data/kodak24") == ("kodak24", "/data/kodak24")
    assert dargs.split_test_dir("/data/CBSD68/original/") == ("original", "/data/CBSD68/original/")
    assert dargs.split_test_dir("cbsd68=/data/CBSD68/original") == ("cbsd68", "/data/CBSD68/original")
    assert dargs.split_test_dir("/data/a=b/set") == ("set", "/data/a=b/set"), "an '=' in a parent's name is not a name"
    assert dargs.split_test_dir("a=b=c") == ("a", "b=c")


def data_with(tmp_path, **changes):
    request = {"train_dirs": [str(tmp_path / "train")], "val_dirs": [str(tmp_path / "val")], **changes}
    return validate(dargs.DenoisingData, request)


@pytest.mark.parametrize(
    "test_dirs, expected",
    [
        ([], "expected at least one directory or null"),
        (["  "], "expected non-empty entries"),
        (["x=/a", "x=/b"], "two held-out sets are named 'x'"),
        (["status=/a"], "not be one of ['status', 'reason', 'sigmas_255']"),
        (["bad name=/a"], "must start with a letter or digit"),
        (["/a/set", "/b/set"], "two held-out sets are named 'set'"),
        (["a=/data/x", "b=/data/x/inner"], "the same directory or one lies inside the other"),
        (["a=%(train)s/inner"], "lies inside or around, train_dirs entry"),
        (["a=%(val)s"], "lies inside or around, val_dirs entry"),
        (["a="], "names no directory"),
    ],
)
def test_test_dirs_refusals_name_the_key_and_the_reason(tmp_path, test_dirs, expected):
    entries = [entry % {"train": str(tmp_path / "train"), "val": str(tmp_path / "val")} for entry in test_dirs]
    with pytest.raises(ConfigValidationError) as raised:
        data_with(tmp_path, test_dirs=entries)
    message = "\n".join(raised.value.errors)
    assert "data.test_dirs" in message or "'test_dirs'" in message
    assert expected in message


def test_test_dirs_defaults_to_nothing_and_a_yaml_file_anchors_only_the_directory(tmp_path):
    assert data_with(tmp_path).test_dirs is None
    (tmp_path / "data.yaml").write_text(
        "train_dirs: [train]\nval_dirs: [val]\ntest_dirs: [held/kodak, 'cb=held/cbsd', '/abs/one']\n")
    mapping_ = dargs.read_denoising_data_file(tmp_path / "data.yaml")
    assert mapping_["test_dirs"] == [
        str(tmp_path / "held/kodak"), f"cb={tmp_path / 'held/cbsd'}", "/abs/one"]
    assert mapping_["train_dirs"] == [str(tmp_path / "train")]


def test_held_out_sets_lists_without_a_tensorflow_op_and_names_the_failing_directory(tmp_path, held_sets):
    data = data_with(tmp_path, test_dirs=[f"alpha={held_sets['alpha']}", str(held_sets["beta"])])
    sets = dargs.held_out_sets(data)
    assert list(sets) == ["alpha", "beta"] and [len(files) for files in sets.values()] == [3, 2]
    assert dargs.held_out_sets(data_with(tmp_path)) == {}
    with pytest.raises(ConfigError, match=r"data\.test_dirs.*is not a directory"):
        dargs.held_out_sets(data_with(tmp_path, test_dirs=[str(tmp_path / "absent")]))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigError, match=r"data\.test_dirs.*holds no images"):
        dargs.held_out_sets(data_with(tmp_path, test_dirs=[str(empty)]))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "broken.png").write_bytes(b"this is not an image")
    with pytest.raises(ConfigError, match=r"data\.test_dirs.*broken\.png"):
        dargs.held_out_sets(data_with(tmp_path, test_dirs=[str(bad)]), channels=3)
    assert dargs.held_out_sets(data_with(tmp_path, test_dirs=[str(bad)])) == {"bad": [str(bad / "broken.png")]}


def test_a_held_out_file_that_is_also_a_train_file_is_refused_before_the_claim(world, folders, held_sets, tmp_path, capsys):
    leaked = tmp_path / "leaky"
    leaked.mkdir()
    (leaked / "linked.png").symlink_to(next(iter(sorted(folders[0].glob("*.png")))))
    out = tmp_path / "run"
    request = held_request(out, folders, held_sets)
    request["data"]["test_dirs"] = [f"leaky={leaked}"]
    code = cli.main(["image_denoising", *flatten("", request, [])])
    assert code == 2
    message = capsys.readouterr().err
    assert "data.test_dirs" in message and "also in data.train_dirs or data.val_dirs" in message
    assert "i00.png" in message, "the file is named by its resolved path, the train file the link leads to"
    assert not out.exists(), "refused before the run directory is claimed"
    # The control: the same request with a clean directory starts (it stops at build_data, after config.json).
    request["data"]["test_dirs"] = [f"alpha={held_sets['alpha']}"]
    config = validate(ImageDenoisingConfig, request)
    trainer = ImageDenoisingTrainer(config)
    assert trainer.describe_data(config).dataset == "folder"


def test_an_undecodable_held_out_file_is_named_before_fit(world, folders, held_sets, tmp_path, capsys):
    broken = tmp_path / "broken_set"
    broken.mkdir()
    (broken / "bad.png").write_bytes(b"not an image")
    out = tmp_path / "run"
    request = held_request(out, folders, held_sets)
    request["data"]["test_dirs"] = [f"broken={broken}"]
    code = cli.main(["image_denoising", *flatten("", request, [])])
    assert code == 2
    message = capsys.readouterr().err
    assert "data.test_dirs" in message and "bad.png" in message
    nothing_was_trained(out)


# -- the helpers that moved out of train.bfunet (D-017) -------------------------------------------------------------


def test_the_moved_helpers_have_one_implementation_and_the_legacy_names_still_resolve():
    for name, legacy in (("add_awgn", "add_awgn"), ("psnr_per_image", "psnr_per_image"),
                         ("predict_denoised", "_predict_denoised"), ("denoise_whole_image", "_denoise_full")):
        assert getattr(ev, legacy) is getattr(dr, name), name
    assert ev.MAX_VAL is dr.MAX_VAL and dr.MAX_VAL == 1.0
    for name in ("expected_noise_variance", "expected_input_psnr"):
        assert getattr(bunet_common, name) is getattr(dr, name), name
    # This module cannot import `train.bfunet` (layering), so the pixel domain is stated twice and held equal here.
    assert (dr.DATA_MIN, dr.DATA_MAX) == (bunet_common.DATA_MIN, bunet_common.DATA_MAX) == (0.0, 1.0)


def test_the_moved_helpers_return_the_legacy_numbers():
    # Hand values: E[sigma^2] of U(0, 0.25) is 0.25^2 / 3; the mean input PSNR is -20/ln(10) (ln 0.25 - 1) = 20.727 dB
    # (the number the data tests read as "about 20.73").
    assert dr.expected_noise_variance(0.0, 0.25) == pytest.approx(0.25 ** 2 / 3)
    assert dr.expected_noise_variance(0.25, 0.0) == pytest.approx(0.25 ** 2 / 3), "bounds in either order"
    assert dr.expected_input_psnr(0.0, 0.25) == pytest.approx(20.727089, abs=1e-5)
    assert dr.expected_input_psnr(0.1, 0.1) == pytest.approx(20.0, abs=1e-9) and math.isnan(dr.expected_input_psnr(0.0, 0.0))
    rng = np.random.RandomState(5)
    clean = np.full((2, 8, 8, 3), 0.5, np.float32)
    noisy = dr.add_awgn(clean, 25 / 255, False, rng)
    again = dr.add_awgn(clean, 25 / 255, False, np.random.RandomState(5))
    np.testing.assert_array_equal(noisy, again)
    assert noisy.dtype == np.float32 and float(np.std(noisy - clean)) == pytest.approx(25 / 255, rel=0.05)
    assert float(dr.add_awgn(clean, 5.0, True, np.random.RandomState(1)).max()) <= 1.0
    # Reflect-pad, one forward pass, crop: the identity "model" returns its (padded then cropped) input.
    identity = keras.Sequential([keras.layers.Input((None, None, 3)), keras.layers.Identity()])
    image = np.random.default_rng(4).uniform(0, 1, (21, 30, 3)).astype("float32")
    np.testing.assert_array_equal(ev._denoise_full(identity, image, 16), image)
    np.testing.assert_array_equal(dr.denoise_whole_image(identity, image, 16, eager=True), image)
