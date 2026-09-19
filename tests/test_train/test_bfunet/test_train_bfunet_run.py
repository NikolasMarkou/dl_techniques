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


# ---------------------------------------------------------------------
# Learning rate: the CSV and the dashboard carry the rate the epoch STARTED at
# ---------------------------------------------------------------------

# Peak rate of the default config and the warmup start of ``learning_rate_schedule_builder``,
# as literals: the tests must not read the constants the fix touched. The tiny config resolves
# ``warmup_epochs`` to 1 (10 percent of 3 epochs, floor 1), so epoch 1 trains from the warmup
# start, epoch 2 begins at the peak, and epoch 3 begins 3 of 9 cosine steps into the decay
# (the warmup wrapper hands the cosine ``step - warmup_steps``): 1e-3 * (0.99 * 0.75 + 0.01).
WARMUP_START_LR = 1e-8
PEAK_LR = 1e-3
EPOCH_3_START_LR = 7.525e-4
LR_PANEL_TITLE = "Learning rate per epoch"


def _lr_column(run_dir: Path) -> List[float]:
    return [float(row["lr"]) for row in _csv_rows(run_dir)]


@pytest.fixture(scope="module")
def lr_run(tmp_path_factory) -> SimpleNamespace:
    """A tiny run whose dashboard histories and LR-panel figures are recorded as they are drawn.

    ``render_training_dashboard`` is wrapped to copy the history it is handed; ``plt.savefig``
    is wrapped to read the LR panel's y-limits and annotation texts off the live figure before
    it is closed. The finished run is then rebuilt with ``build_dashboard_from_dir`` under the
    same wrappers, so the rebuilt panel is observed the same way.
    """
    import matplotlib.pyplot as plt

    root = tmp_path_factory.mktemp("bfunet_lr")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    histories: List[Dict] = []
    panels: List[Dict] = []
    real_render = common.render_training_dashboard
    real_savefig = plt.savefig

    def spy_render(history, out_path, *args, **kwargs):
        histories.append({k: (None if v is None else list(v)) for k, v in history.items()})
        return real_render(history, out_path, *args, **kwargs)

    def spy_savefig(path, *args, **kwargs):
        for ax in plt.gcf().axes:
            if ax.get_title() == LR_PANEL_TITLE:
                panels.append({
                    "ylim": ax.get_ylim(),
                    "texts": [t.get_text() for t in ax.texts],
                })
        return real_savefig(path, *args, **kwargs)

    config = _tiny_config(root, "lr_run")
    run_dir = root / "out" / "lr_run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "render_training_dashboard", spy_render)
        patch.setattr(plt, "savefig", spy_savefig)
        run_train(config)
        trained_histories, trained_panels = list(histories), list(panels)
        histories.clear()
        panels.clear()
        common.build_dashboard_from_dir(str(run_dir))
    return SimpleNamespace(
        run_dir=run_dir,
        histories=trained_histories,
        panels=trained_panels,
        rebuilt_history=histories[-1],
        rebuilt_panel=panels[-1],
    )


def test_the_csv_lr_is_the_rate_each_epoch_started_at(lr_run) -> None:
    """Epoch 1 trains from the warmup start; the end-of-epoch read would show 1e-3 there."""
    lrs = _lr_column(lr_run.run_dir)
    assert lrs == pytest.approx([WARMUP_START_LR, PEAK_LR, EPOCH_3_START_LR], rel=1e-3)


def test_the_local_lr_logger_is_gone_so_there_is_one_lr_semantics(lr_run) -> None:
    assert not hasattr(common, "LRLoggerCallback")
    assert not hasattr(common, "_read_current_lr")


def test_the_epoch_zero_baseline_point_carries_no_learning_rate(lr_run) -> None:
    """Nothing trained at epoch 0; the optimizer's warmup start is not a rate any epoch used."""
    baseline = lr_run.histories[0]
    assert baseline["epoch"] == [0]
    assert np.isnan(baseline["lr"][0])


def test_the_dashboard_lr_series_is_the_csv_lr_column(lr_run) -> None:
    """The last live redraw holds the baseline nan then the CSV rates, epoch for epoch."""
    final = lr_run.histories[-1]
    assert final["epoch"] == [0, 1, 2, 3]
    assert np.isnan(final["lr"][0])
    assert final["lr"][1:] == pytest.approx(_lr_column(lr_run.run_dir), rel=1e-6)


def test_the_live_dashboard_lr_axis_is_not_squeezed_by_the_warmup_start(lr_run) -> None:
    """Peak 1e-3 and a 1e-8 first epoch: the axis floor is 1e-6, not ten decades down."""
    low, high = lr_run.panels[-1]["ylim"]
    assert low >= 1e-6 * (1.0 - 1e-9)
    assert low < EPOCH_3_START_LR      # the real schedule stays on the axis
    assert high >= PEAK_LR


def test_the_clipped_warmup_point_is_annotated_with_its_value(lr_run) -> None:
    texts = lr_run.panels[-1]["texts"]
    assert any("epoch 1" in t and "1e-08" in t and "below axis" in t for t in texts), texts


def test_a_rebuilt_dashboard_reproduces_the_csv_lr_column(lr_run) -> None:
    """``--dashboard`` on a finished run reconstructs the LR panel from config.json."""
    rebuilt = lr_run.rebuilt_history["lr"]
    assert rebuilt is not None
    assert rebuilt == pytest.approx(_lr_column(lr_run.run_dir), rel=1e-6)


def test_a_rebuilt_dashboard_clips_its_lr_axis_the_same_way(lr_run) -> None:
    low, _ = lr_run.rebuilt_panel["ylim"]
    assert low >= 1e-6 * (1.0 - 1e-9)
    assert any("1e-08" in t for t in lr_run.rebuilt_panel["texts"])


# ---------------------------------------------------------------------
# iter-1/step-6 (plan-2026-09-19T131351-b8d39688/D-003): the cosine floor is not reached
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def schedule_e100(tmp_path_factory):
    """The learning-rate schedule the REAL ``train()`` hands the optimizer, E=100 W=10 spe=400.

    ``optimizer_builder`` is replaced by a spy that keeps its schedule argument and raises,
    so ``train()`` stops right after building the schedule and no epoch is fitted. The
    schedule is then evaluated at plain integer steps (a pure function, no GPU needed).
    """
    root = tmp_path_factory.mktemp("bfunet_schedule")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    seen: Dict[str, object] = {}

    def spy(config, learning_rate, *args, **kwargs):
        seen["schedule"] = learning_rate
        raise _Probe

    config = _tiny_config(
        root, "schedule_probe", epochs=100, warmup_epochs=10, steps_per_epoch=400,
        learning_rate=1e-3)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "optimizer_builder", spy)
        with pytest.raises(_Probe):
            run_train(config)
    return seen["schedule"]


def _rate(schedule, step: int) -> float:
    return float(keras.ops.convert_to_numpy(schedule(step)))


# DECISION plan-2026-09-19T131351-b8d39688/D-003: this pins a KNOWN imperfection on purpose.
# The default schedule is WarmupSchedule(CosineDecay(decay_steps=E*spe)), which feeds the
# cosine `step - warmup_steps`, so the decay is cut at (E-W)/E of its horizon and the last
# step of E=100 W=10 spe=400 runs at 3.42e-5, not at the 1e-5 floor (alpha 0.01 x 1e-3).
# The published bfunet runs used exactly this schedule, so the number is pinned to keep
# them reproducible. Do NOT "fix" it by changing alpha or the warmup feed in train() to
# make this test green, and do NOT loosen the tolerance. To change the schedule
# deliberately: decide it in decisions.md first (the named candidate is an opt-in
# decay-to-end schedule with decay_steps = (E-W)*spe, which ends at 1.0e-5), keep the
# default as it is unless that decision says otherwise, then update the literal below
# together with the README cosine-floor note.
def test_the_default_schedule_ends_at_3_42e_minus_5_not_at_the_1e_minus_5_floor(
        schedule_e100) -> None:
    last_step = 100 * 400 - 1
    assert _rate(schedule_e100, last_step) == pytest.approx(3.42e-5, rel=0.01)


def test_the_default_schedule_warms_up_to_the_configured_rate_over_the_warmup_steps(
        schedule_e100) -> None:
    """The warmup feed: 1e-8 at step 0, exactly the peak 1e-3 where the 10 warmup epochs end."""
    assert _rate(schedule_e100, 0) == pytest.approx(1e-8, rel=1e-3)
    assert _rate(schedule_e100, 10 * 400) == pytest.approx(1e-3, rel=1e-6)


# ---------------------------------------------------------------------
# iter-1/step-7 (plan-2026-09-19T131351-b8d39688/D-005, D-011): the final model is the LAST
# epoch, divergence is recorded, a partial warm start is refused
# ---------------------------------------------------------------------

class _WeightRecorder(keras.callbacks.Callback):
    """Copies every weight tensor at the end of each epoch that actually runs."""

    def __init__(self) -> None:
        super().__init__()
        self.weights: List[List[np.ndarray]] = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([np.array(w) for w in self.model.get_weights()])


class _ScriptedValLoss(keras.callbacks.Callback):
    """Overwrites the epoch ``val_loss`` so the best epoch is deterministic, not a noise draw.

    Placed BEFORE EarlyStopping / ModelCheckpoint / CSVLogger, which all read ``logs``.
    """

    def __init__(self, values: List[float]) -> None:
        super().__init__()
        self.values = values

    def on_epoch_end(self, epoch, logs=None):
        logs["val_loss"] = self.values[epoch]


class _NanLossAtEpoch(keras.callbacks.Callback):
    """A NaN epoch loss followed by the stop TerminateOnNaN issues after a NaN batch.

    A real divergence is not reproducible on demand, so this injects its two observable
    effects: the NaN in the epoch ``logs`` (which History and CSVLogger record) and
    ``stop_training``. It runs BEFORE CSVLogger, so the NaN reaches every record.
    """

    def __init__(self, epoch_index: int) -> None:
        super().__init__()
        self.epoch_index = epoch_index

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epoch_index:
            logs["loss"] = float("nan")
            self.model.stop_training = True


def _run_with_callbacks(config: TrainingConfig, front=(), back=()) -> keras.Model:
    """``run_train`` with test callbacks put in front of / behind the stock callback list."""
    original = common.create_common_callbacks

    def patched(*args, **kwargs):
        callbacks, results_dir = original(*args, **kwargs)
        return [*front, *callbacks, *back], results_dir

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "create_common_callbacks", patched)
        return run_train(config)


def _weights_of(path: Path) -> List[np.ndarray]:
    return [np.array(w) for w in keras.models.load_model(path).get_weights()]


def _same_tensors(left: List[np.ndarray], right: List[np.ndarray]) -> bool:
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


@pytest.fixture(scope="module")
def early_stop_run(tmp_path_factory) -> SimpleNamespace:
    """Patience 1, scripted val_loss [1.0, 0.5, 0.9, 0.8]: best is epoch 2, the run stops after 3."""
    root = tmp_path_factory.mktemp("bfunet_early_stop")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    recorder = _WeightRecorder()
    config = _tiny_config(root, "early_stop", epochs=4, early_stopping_patience=1)
    _run_with_callbacks(
        config, front=[_ScriptedValLoss([1.0, 0.5, 0.9, 0.8])], back=[recorder])
    run_dir = root / "out" / "early_stop"
    return SimpleNamespace(
        run_dir=run_dir,
        recorded=recorder.weights,
        best=_weights_of(run_dir / "best_model.keras"),
        final=_weights_of(run_dir / "final_model.keras"),
    )


def test_the_early_stopping_scenario_stops_after_the_epoch_following_the_best(early_stop_run) -> None:
    """Anti-vacuity: 3 epochs ran of 4, the best (epoch 2) is not the last."""
    assert len(early_stop_run.recorded) == 3
    assert [float(r["val_loss"]) for r in _csv_rows(early_stop_run.run_dir)] == [1.0, 0.5, 0.9]


def test_best_model_holds_the_best_epochs_weights(early_stop_run) -> None:
    assert _same_tensors(early_stop_run.best, early_stop_run.recorded[1])
    assert not _same_tensors(early_stop_run.best, early_stop_run.recorded[2])


def test_final_model_holds_the_last_epochs_weights_not_the_best_ones(early_stop_run) -> None:
    """Keras EarlyStopping(restore_best_weights=True) used to make this the best epoch's."""
    assert _same_tensors(early_stop_run.final, early_stop_run.recorded[-1])
    assert not _same_tensors(early_stop_run.final, early_stop_run.best)


@pytest.fixture(scope="module")
def default_run(tmp_path_factory) -> SimpleNamespace:
    """Default path (no early stopping), 2 epochs, with the last epoch's weights recorded."""
    root = tmp_path_factory.mktemp("bfunet_default_final")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    recorder = _WeightRecorder()
    config = _tiny_config(root, "default_final", epochs=2)
    _run_with_callbacks(config, back=[recorder])
    run_dir = root / "out" / "default_final"
    return SimpleNamespace(
        run_dir=run_dir, recorded=recorder.weights,
        final=_weights_of(run_dir / "final_model.keras"),
    )


def test_the_default_path_final_model_is_the_last_epoch(default_run) -> None:
    assert len(default_run.recorded) == 2
    assert _same_tensors(default_run.final, default_run.recorded[-1])


@pytest.fixture(scope="module")
def diverged_run(tmp_path_factory) -> SimpleNamespace:
    """A run whose epoch 2 loss is NaN, driven through the trainer's own ``main()``.

    ``main()`` builds its config from argv (data directories are hard-coded to the HDD),
    so ``train`` is replaced by a stub that runs the REAL ``common.train`` on the tiny
    generated config instead; ``main()``'s own try/except and its logging are what is under
    test. The NaN is injected as described on ``_NanLossAtEpoch``.
    """
    import sys

    import train.bfunet.train_convunext_denoiser as trainer

    root = tmp_path_factory.mktemp("bfunet_diverged")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    config = _tiny_config(root, "diverged", epochs=3)
    records: List[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.INFO)
    logging.getLogger("dl").addHandler(handler)
    raised = None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer, "setup_gpu", lambda gpu_id=None: None)
        patch.setattr(
            trainer, "train",
            lambda _main_config: _run_with_callbacks(config, front=[_NanLossAtEpoch(1)]),
        )
        patch.setattr(sys, "argv", ["train_convunext_denoiser"])
        try:
            trainer.main()
        except RuntimeError as error:
            raised = error
        finally:
            logging.getLogger("dl").removeHandler(handler)
    return SimpleNamespace(
        run_dir=root / "out" / "diverged", raised=raised,
        messages=[(r.levelno, r.getMessage()) for r in records],
    )


def test_a_nan_loss_raises_runtime_error_out_of_main(diverged_run) -> None:
    assert isinstance(diverged_run.raised, RuntimeError)
    assert "diverged" in str(diverged_run.raised)


def test_a_diverged_run_writes_no_final_model(diverged_run) -> None:
    assert not (diverged_run.run_dir / "final_model.keras").exists()
    assert (diverged_run.run_dir / "training_history.json").stat().st_size > 0


def test_main_logs_a_failure_and_never_reports_success_for_a_diverged_run(diverged_run) -> None:
    texts = [text for _, text in diverged_run.messages]
    assert any(
        level == logging.ERROR and text.startswith("Training failed") and "diverged" in text
        for level, text in diverged_run.messages
    ), texts
    assert not any("Training completed successfully" in text for text in texts), texts


def _strict_json(path: Path) -> Dict:
    def refuse(token):
        raise AssertionError(f"non-strict JSON token {token} in {path}")

    return json.loads(path.read_text(), parse_constant=refuse)


def test_a_diverged_run_records_a_strict_diverged_summary(diverged_run) -> None:
    summary = _strict_json(diverged_run.run_dir / "results_summary.json")
    assert summary["status"] == "diverged"
    assert summary["experiment_name"] == "diverged"
    assert summary["epochs_requested"] == 3
    assert summary["epochs_run"] == 2
    assert summary["stopped_early"] is None
    assert summary["best_epoch"] is None
    assert summary["non_finite_metrics"] == ["loss"]
    assert summary["history"]["loss"][1] is None       # the NaN, written as null


# --- init_from --------------------------------------------------------------

def test_init_from_a_checkpoint_of_another_variant_is_refused_naming_the_layers(e2e) -> None:
    """tiny (32 filters) -> small (48): the same layer names with different shapes."""
    config = _tiny_config(
        e2e.root, "init_variant_mismatch", variant="small",
        init_from=str(e2e.run_dir / "final_model.keras"))
    with pytest.raises(ValueError) as raised:
        run_train(config)
    text = str(raised.value)
    assert "shape" in text and "encoder_level_0_convnext_v1_block_0" in text, text
    assert "bottleneck_convnext_v1_block_0" in text, text
    assert not (e2e.root / "out" / "init_variant_mismatch" / "final_model.keras").exists()


def test_a_missing_in_source_layer_only_warns_and_is_counted(e2e) -> None:
    """blocks_per_level 1 -> 2 adds 5 layers the checkpoint has never seen; same shapes elsewhere."""
    config = _tiny_config(
        e2e.root, "init_missing", blocks_per_level=2, epochs=1,
        init_from=str(e2e.run_dir / "final_model.keras"))
    run_train(config)
    warnings = [
        line for line in (e2e.root / "out" / "init_missing" / "run.log").read_text().splitlines()
        if "WARNING" in line and "init_from" in line and "missing_in_source" in line
    ]
    assert len(warnings) == 1, warnings
    assert "5 layer(s)" in warnings[0] and "encoder_level_0_convnext_v1_block_1" in warnings[0]
    assert (e2e.root / "out" / "init_missing" / "final_model.keras").stat().st_size > 0
