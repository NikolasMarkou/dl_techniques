"""End-to-end guard for ``common.train()`` of the bfunet ConvUNeXt trainer.

Until this file existed nothing ran ``train()`` through ``fit`` to disk (the only calls in
the suite were two refusal cases), so a defect in any artifact the run writes was
invisible to a green suite. The fixture below runs the REAL ``common.train`` once, on
generated PNGs, into ``tmp_path`` (a conftest guard fails on any write to the repo-root
``results/``), and the tests read what landed on disk.

These are CHARACTERIZATION pins: they assert what the trainer writes today, so later
steps of the hardening plan change the run knowing exactly which pin they move. Every
expected literal below is typed here (a name, a count, the ``"[0,1]"`` stamp), never
imported from the code under test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from train.bfunet import common  # noqa: E402
from train.bfunet.train_convunext_denoiser import (  # noqa: E402
    TrainingConfig,
    build_model,
    verify_bias_free,
)

PATCH = 16
IMAGE_SIZE = 32          # larger than PATCH so random_crop has room
N_TRAIN, N_VAL = 6, 4
EPOCHS = 3
STEPS_PER_EPOCH = 3
VALIDATION_STEPS = 2
EXPERIMENT = "bfunet_e2e"
FIXTURE_BUDGET_SECONDS = 90.0

# Keras ignores an unfinished-epoch advisory here for the same reason the self-iterate
# fit test does: the tiny generated corpus is smaller than ``steps_per_epoch`` batches.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Your input ran out of data:UserWarning"),
]


def _write_pngs(directory: Path, count: int, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        pixels = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        (directory / f"img_{index}.png").write_bytes(tf.io.encode_png(pixels).numpy())


def _tiny_config(root: Path, name: str, **overrides) -> TrainingConfig:
    """A tiny ConvUNeXt config whose data and output all live under ``root``."""
    fields = dict(
        train_image_dirs=[str(root / "train")],
        val_image_dirs=[str(root / "val")],
        variant="tiny",
        depth=2,                # shallower than the variant: the XLA compile dominates
        blocks_per_level=1,     # the fixture wall time, and the pins do not need depth
        patch_size=PATCH,
        batch_size=2,
        patches_per_image=2,
        patch_shuffle_buffer=8,
        dataset_shuffle_buffer=8,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        viz_freq=1,
        viz_samples=2,
        seed=0,
        output_dir=str(root / "out"),
        experiment_name=name,
    )
    fields.update(overrides)
    return TrainingConfig(**fields)


def run_train(config: TrainingConfig) -> keras.Model:
    """Run the shared ``common.train`` exactly as ``train_convunext_denoiser.train`` does."""
    return common.train(
        config,
        build_model,
        verify_bias_free,
        model_label="ConvUNeXt",
        results_dir_prefix="convunext_denoiser",
    )


@pytest.fixture(scope="module")
def e2e(tmp_path_factory) -> SimpleNamespace:
    """One tiny ``train()`` run to disk; the wall time of the whole call is recorded."""
    root = tmp_path_factory.mktemp("bfunet_e2e")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    config = _tiny_config(root, EXPERIMENT)
    started = time.time()
    run_train(config)
    seconds = time.time() - started
    run_dir = root / "out" / EXPERIMENT
    return SimpleNamespace(
        root=root, config=config, run_dir=run_dir, seconds=seconds,
        final=keras.models.load_model(run_dir / "final_model.keras"),
    )


def _csv_rows(run_dir: Path) -> List[Dict[str, str]]:
    with open(run_dir / "training_log.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_fixture_fits_the_time_budget(e2e) -> None:
    """The run is cheap enough to be a guard: it is recorded, and bounded, in seconds."""
    print(f"\nbfunet e2e fixture wall time: {e2e.seconds:.1f} s")
    assert e2e.seconds < FIXTURE_BUDGET_SECONDS


@pytest.mark.parametrize(
    "relative",
    [
        "config.json",
        "training_log.csv",
        "training_history.json",
        "best_model.keras",
        "final_model.keras",
        "visualizations/training_dashboard.png",
        "visualizations/epoch_000_denoise_grid.png",
        "visualizations/epoch_001_denoise_grid.png",
        "visualizations/epoch_002_denoise_grid.png",
        "visualizations/epoch_003_denoise_grid.png",
    ],
)
def test_the_run_writes_this_artifact_and_it_is_not_empty(e2e, relative) -> None:
    assert (e2e.run_dir / relative).stat().st_size > 0, relative


def test_the_figures_are_real_png_files(e2e) -> None:
    for name in ("training_dashboard.png", "epoch_003_denoise_grid.png"):
        header = (e2e.run_dir / "visualizations" / name).read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", name


def test_the_run_touches_nothing_outside_its_own_directory(e2e) -> None:
    assert sorted(p.name for p in (e2e.root / "out").iterdir()) == [EXPERIMENT]


def test_the_csv_has_one_row_per_epoch(e2e) -> None:
    assert len(_csv_rows(e2e.run_dir)) == 3


def test_the_csv_carries_the_lr_and_the_validation_psnr_columns(e2e) -> None:
    columns = set(_csv_rows(e2e.run_dir)[0])
    assert {"epoch", "loss", "val_loss", "lr", "psnr_metric", "val_psnr_metric"} <= columns


def test_the_csv_lr_column_is_populated_and_positive(e2e) -> None:
    for row in _csv_rows(e2e.run_dir):
        assert float(row["lr"]) > 0.0


def test_the_history_json_records_every_epoch(e2e) -> None:
    history = json.loads((e2e.run_dir / "training_history.json").read_text())
    assert len(history["loss"]) == 3


def test_config_json_stamps_the_unit_pixel_domain(e2e) -> None:
    config = json.loads((e2e.run_dir / "config.json").read_text())
    assert config["data_range"] == "[0,1]"


def test_config_json_records_the_experiment_and_the_tiny_variant(e2e) -> None:
    config = json.loads((e2e.run_dir / "config.json").read_text())
    assert config["experiment_name"] == "bfunet_e2e"
    assert config["variant"] == "tiny"


def test_final_model_reloads_and_denoises_a_patch(e2e) -> None:
    """The reloaded file is a working bias-free denoiser: right shape, finite output."""
    patch = np.random.default_rng(0).uniform(0.0, 1.0, size=(1, PATCH, PATCH, 3))
    out = np.asarray(e2e.final.predict(patch.astype("float32"), verbose=0))
    assert out.shape == (1, PATCH, PATCH, 3)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------
# iter-1/step-3 (plan-2026-09-19T131351-b8d39688/D-002): run-directory order
# ---------------------------------------------------------------------

# The repo root, derived from this file's location (tests/test_train/test_bfunet/<file>),
# never from the trainer's own helper: the test must fail if the helper is not used.
REPO_ROOT = Path(__file__).resolve().parents[3]


def _snapshot(run_dir: Path) -> Dict[str, str]:
    """``{relative path: sha256}`` of every file under ``run_dir``."""
    return {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def test_a_reused_experiment_name_is_refused_and_the_first_run_is_byte_identical(e2e) -> None:
    """A second run under the same name used to merge into the first directory."""
    before = _snapshot(e2e.run_dir)
    assert "config.json" in before and "training_log.csv" in before

    with pytest.raises(FileExistsError) as raised:
        run_train(_tiny_config(e2e.root, EXPERIMENT))

    text = str(raised.value)
    assert str(e2e.run_dir) in text and "--experiment-name" in text, text
    assert _snapshot(e2e.run_dir) == before, "the refusal wrote or deleted something"


class _Probe(Exception):
    """Raised by the spy to stop ``train()`` right where it creates the run directory."""


def test_a_relative_output_dir_is_anchored_at_the_repo_root_not_the_cwd(
        monkeypatch, tmp_path_factory) -> None:
    """The wiring, not the helper: ``train()`` hands the repo-root path to ``prepare_run_dir``."""
    root = tmp_path_factory.mktemp("bfunet_relative")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    elsewhere = tmp_path_factory.mktemp("bfunet_elsewhere")
    monkeypatch.chdir(elsewhere)
    seen: Dict[str, object] = {}

    def spy(config, output_dir=None, **kwargs):
        seen["output_dir"] = output_dir
        raise _Probe

    monkeypatch.setattr(common, "prepare_run_dir", spy)
    config = _tiny_config(root, "relative_probe", output_dir="relative_probe_root")
    with pytest.raises(_Probe):
        run_train(config)

    assert seen["output_dir"] is not None, "prepare_run_dir was called without the resolved path"
    assert Path(seen["output_dir"]) == REPO_ROOT / "relative_probe_root" / "relative_probe"
    assert list(elsewhere.iterdir()) == [], "something was created in the working directory"
    assert not (REPO_ROOT / "relative_probe_root").exists()


@pytest.fixture(scope="module")
def unset_steps_run(tmp_path_factory) -> SimpleNamespace:
    """One-epoch run with ``steps_per_epoch`` left unset (the floor rule resolves it)."""
    root = tmp_path_factory.mktemp("bfunet_unset_steps")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    # max_train_files caps the worklist at the 6 generated images (the default of 10000
    # wraps them around to 10000 paths, i.e. 10000 steps of a tiny model).
    config = _tiny_config(
        root, "unset_steps", epochs=1, steps_per_epoch=None, max_train_files=N_TRAIN)
    run_train(config)
    return SimpleNamespace(config=config, run_dir=root / "out" / "unset_steps")


def test_config_json_records_the_resolved_steps_per_epoch(unset_steps_run) -> None:
    """6 images x 2 patches // batch 2 = 6 is below the floor of 100, so 100 is what ran."""
    written = json.loads((unset_steps_run.run_dir / "config.json").read_text())
    assert written["steps_per_epoch"] == 100


def test_the_write_back_does_not_mutate_the_callers_config(unset_steps_run) -> None:
    assert unset_steps_run.config.steps_per_epoch is None


def test_config_json_keeps_an_explicit_steps_per_epoch(e2e) -> None:
    written = json.loads((e2e.run_dir / "config.json").read_text())
    assert written["steps_per_epoch"] == 3


@pytest.fixture(scope="module")
def pool_run(tmp_path_factory) -> SimpleNamespace:
    """Self-iterate run: the finite pool (4 patches, batch 2) decides ``steps_per_epoch``."""
    root = tmp_path_factory.mktemp("bfunet_pool")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    config = _tiny_config(
        root, "pool_steps", epochs=1, steps_per_epoch=None,
        self_iterate=True, self_iterate_pool_size=4,
    )
    run_train(config)
    return SimpleNamespace(config=config, run_dir=root / "out" / "pool_steps")


def test_config_json_records_the_pool_steps_per_epoch_under_self_iterate(pool_run) -> None:
    """4 pooled patches // batch 2 = 2 steps, not the 100 of the streaming floor."""
    written = json.loads((pool_run.run_dir / "config.json").read_text())
    assert written["steps_per_epoch"] == 2
    assert pool_run.config.steps_per_epoch is None


# ---------------------------------------------------------------------
# iter-1/step-4 (plan-2026-09-19T131351-b8d39688/D-007): run.log and the epoch line
# ---------------------------------------------------------------------

EPOCH_LINE = re.compile(r"Epoch (\d+)/(\d+) - (.*) - time ([0-9.]+)s\s*$")
# The metric columns the trainer compiles, typed here (never read from the code under
# test). Every one must exist in the CSV or the comparison below would be vacuous.
LINE_METRICS = (
    "loss", "mae", "psnr_metric", "ssim_metric",
    "val_loss", "val_mae", "val_psnr_metric", "val_ssim_metric",
)


def _epoch_lines(run_dir: Path):
    """``[(epoch, total, {key: value text}, seconds)]`` for every epoch line of run.log."""
    found = []
    for line in (run_dir / "run.log").read_text().splitlines():
        match = EPOCH_LINE.search(line)
        if match is None:
            continue
        pairs = {}
        for part in match.group(3).split(" - "):
            key, value = part.rsplit(" ", 1)
            pairs[key] = value
        found.append((int(match.group(1)), int(match.group(2)), pairs, float(match.group(4))))
    return found


def test_the_run_writes_a_run_log(e2e) -> None:
    assert (e2e.run_dir / "run.log").stat().st_size > 0


def test_run_log_holds_exactly_one_epoch_line_per_epoch(e2e) -> None:
    lines = _epoch_lines(e2e.run_dir)
    assert [n for n, _, _, _ in lines] == [1, 2, 3]
    assert all(total == 3 for _, total, _, _ in lines)


def test_every_metric_the_line_needs_is_a_csv_column(e2e) -> None:
    """Anti-vacuity: the comparison below iterates a typed list, so pin that it is real."""
    assert set(LINE_METRICS) <= set(_csv_rows(e2e.run_dir)[0])


def test_each_epoch_line_equals_its_csv_row_to_four_decimals(e2e) -> None:
    """Expected texts are computed here from the CSV, not read from the callback's keys."""
    lines = _epoch_lines(e2e.run_dir)
    rows = _csv_rows(e2e.run_dir)
    assert len(lines) == len(rows) == 3
    for (epoch, _, values, _), row in zip(lines, rows):
        expected = {key: f"{float(row[key]):.4f}" for key in LINE_METRICS}
        expected["lr"] = f"{float(row['lr']):.6g}"
        assert values == expected, epoch


def test_the_epoch_line_seconds_are_positive_and_fit_inside_the_fit(e2e) -> None:
    lines = _epoch_lines(e2e.run_dir)
    assert len(lines) == 3
    assert all(seconds > 0.0 for _, _, _, seconds in lines)
    assert sum(seconds for _, _, _, seconds in lines) < e2e.seconds


def test_run_log_covers_the_whole_run_through_the_final_save(e2e) -> None:
    """The handler wraps the training body, not just the fit: the save at the end is in it."""
    text = (e2e.run_dir / "run.log").read_text()
    assert "Training completed in" in text
    assert "Saved final (last-epoch) model" in text


def test_the_run_log_handler_is_detached_when_train_returns(e2e) -> None:
    """A later message in the same process must not land in a finished run's log."""
    before = (e2e.run_dir / "run.log").read_bytes()
    logging.getLogger("dl").info("after the run: this must not reach run.log")
    assert (e2e.run_dir / "run.log").read_bytes() == before
    attached = [
        h for h in logging.getLogger("dl").handlers
        if isinstance(h, logging.FileHandler) and str(e2e.run_dir) in str(h.baseFilename)
    ]
    assert attached == []


class _LogsEditor(keras.callbacks.Callback):
    """Stands in for any callback that edits the epoch ``logs`` after the stock ones."""

    def on_epoch_end(self, epoch, logs=None):
        logs["psnr_metric"] = 12.3457


@pytest.fixture(scope="module")
def edited_run(tmp_path_factory) -> SimpleNamespace:
    """One-epoch run whose callback list gains a logs-editing callback after the stock ones."""
    root = tmp_path_factory.mktemp("bfunet_edited_logs")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    original = common.create_common_callbacks

    def with_editor(*args, **kwargs):
        callbacks, results_dir = original(*args, **kwargs)
        callbacks.append(_LogsEditor())
        return callbacks, results_dir

    config = _tiny_config(root, "edited_logs", epochs=1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "create_common_callbacks", with_editor)
        run_train(config)
    return SimpleNamespace(run_dir=root / "out" / "edited_logs")


def test_the_epoch_line_is_printed_after_every_callback_that_edits_the_logs(edited_run) -> None:
    """A line printed before the editor ran would show the compiled value, not 12.3457."""
    (line,) = _epoch_lines(edited_run.run_dir)
    assert line[2]["psnr_metric"] == "12.3457"
