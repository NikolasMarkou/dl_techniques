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
N_TEST_IMAGES = 3        # per generated held-out set; fewer than TEST_NUM_SAMPLES on purpose
TEST_NUM_SAMPLES = 5     # crops per held-out set, so the crop list wraps around the images

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
        # The held-out evaluation is OFF unless a fixture asks for it: it reloads the best
        # model and predicts on two sets, which the fixtures that do not read it need not pay.
        test_eval=False,
        test_num_samples=TEST_NUM_SAMPLES,
        # Likewise the end-of-run analyzer (about 17 s at tiny/128): only the fixtures that
        # read ``model_analysis/`` or the ``analyzer`` block turn it on.
        model_analysis=False,
        output_dir=str(root / "out"),
        experiment_name=name,
    )
    fields.update(overrides)
    return TrainingConfig(**fields)


@pytest.fixture(scope="module", autouse=True)
def held_out_sets(tmp_path_factory) -> Dict[str, str]:
    """Two tiny generated stand-ins for Kodak24 and CBSD68, wired in for the whole module.

    ``common.TEST_DATASETS`` names two directories on the data disk; no test may read them.
    """
    root = tmp_path_factory.mktemp("bfunet_held_out")
    _write_pngs(root / "kodak24", N_TEST_IMAGES, seed=11)
    _write_pngs(root / "cbsd68", N_TEST_IMAGES, seed=12)
    sets = {"kodak24": str(root / "kodak24"), "cbsd68": str(root / "cbsd68")}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "TEST_DATASETS", sets)
        yield sets


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
    config = _tiny_config(root, EXPERIMENT, test_eval=True)
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
    config = _tiny_config(
        root, "early_stop", epochs=4, early_stopping_patience=1, test_eval=True)
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
    config = _tiny_config(root, "default_final", epochs=2, test_eval=True)
    # Scripted val_loss (a falling pair) so step 8's ``final_is_best`` has a True case with a
    # literal expectation; it changes no weight, only which epoch ModelCheckpoint calls best.
    # ``validate_model_loading`` is forced to report a FAILED round trip here (a corrupt
    # reload cannot be produced on demand), so ``model_loading_validated`` has a False case
    # beside the True one every other fixture records.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "validate_model_loading", lambda *args, **kwargs: False)
        _run_with_callbacks(config, front=[_ScriptedValLoss([0.9, 0.5])], back=[recorder])
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
    block = _strict_json(e2e.root / "out" / "init_missing" / "results_summary.json")["init_from"]
    assert block["path"] == str(e2e.run_dir / "final_model.keras")
    assert block["loaded"] > 0 and block["missing_in_source"] == 5 and block["shape_mismatch"] == 0
    warnings = [
        line for line in (e2e.root / "out" / "init_missing" / "run.log").read_text().splitlines()
        if "WARNING" in line and "init_from" in line and "missing_in_source" in line
    ]
    assert len(warnings) == 1, warnings
    assert "5 layer(s)" in warnings[0] and "encoder_level_0_convnext_v1_block_1" in warnings[0]
    assert (e2e.root / "out" / "init_missing" / "final_model.keras").stat().st_size > 0


# ---------------------------------------------------------------------
# iter-1/step-8 (plan-2026-09-19T131351-b8d39688): results_summary.json for every run
# ---------------------------------------------------------------------

# Typed here from the plan, never read from the trainer: the keys a finished run promises.
OK_KEYS = (
    "status", "run_dir", "experiment_name", "variant", "params", "learning_rate",
    "warmup_epochs", "steps_per_epoch", "lr_first_epoch", "lr_last_epoch", "lr_last_step",
    "noise", "epochs_requested", "epochs_run", "stopped_early", "best_epoch",
    "best_epoch_csv_index", "final_is_best", "best_val_metrics", "final_val_metrics",
    "epoch_times", "fit_wall_seconds", "model_loading_validated", "init_from", "gpu_name",
    "cuda_visible_devices", "notes", "test_eval",
)
# What a diverged run shares with it (no best epoch can be named, nothing was saved).
DIVERGED_KEYS = (
    "status", "run_dir", "experiment_name", "variant", "params", "learning_rate",
    "warmup_epochs", "steps_per_epoch", "lr_first_epoch", "lr_last_epoch", "lr_last_step",
    "noise", "epochs_requested", "epochs_run", "stopped_early", "best_epoch", "epoch_times",
    "fit_wall_seconds", "init_from", "gpu_name", "cuda_visible_devices", "notes",
    "non_finite_metrics", "history",
)
NOISE_KEYS = ("start", "end", "schedule", "sigma_min")
OK_FIXTURES = ("e2e", "early_stop_run", "default_run")


@pytest.fixture(params=OK_FIXTURES)
def ok_run(request) -> SimpleNamespace:
    """Each finished-run fixture already in this module, with its summary parsed strictly."""
    run = request.getfixturevalue(request.param)
    return SimpleNamespace(
        name=request.param, run_dir=run.run_dir,
        summary=_strict_json(run.run_dir / "results_summary.json"),
        rows=_csv_rows(run.run_dir),
    )


def test_a_finished_run_writes_a_strict_json_summary_with_every_promised_key(ok_run) -> None:
    missing = [key for key in OK_KEYS if key not in ok_run.summary]
    assert not missing, missing
    assert ok_run.summary["status"] == "ok"


def test_the_summary_names_the_real_run_directory_and_experiment(ok_run) -> None:
    assert Path(ok_run.summary["run_dir"]).is_dir()
    assert Path(ok_run.summary["run_dir"]).resolve() == ok_run.run_dir.resolve()
    assert ok_run.summary["experiment_name"] == ok_run.run_dir.name


def test_best_epoch_is_the_one_based_argmin_of_the_csv_val_loss(ok_run) -> None:
    val_loss = [float(row["val_loss"]) for row in ok_run.rows]
    expected = min(range(len(val_loss)), key=val_loss.__getitem__) + 1
    assert ok_run.summary["best_epoch"] == expected
    assert ok_run.summary["best_epoch_csv_index"] == expected - 1
    assert int(ok_run.rows[ok_run.summary["best_epoch_csv_index"]]["epoch"]) == expected - 1


def test_final_is_best_agrees_with_the_epochs_and_the_best_and_final_metrics(ok_run) -> None:
    summary = ok_run.summary
    assert summary["final_is_best"] == (summary["best_epoch"] == summary["epochs_run"])
    assert (summary["best_val_metrics"] == summary["final_val_metrics"]) == summary["final_is_best"]


def test_best_and_final_val_metrics_are_the_csv_rows_of_those_epochs(ok_run) -> None:
    csv_val = sorted(key for key in ok_run.rows[0] if key.startswith("val_"))
    assert sorted(ok_run.summary["best_val_metrics"]) == csv_val
    assert sorted(ok_run.summary["final_val_metrics"]) == csv_val
    for block, index in (
            ("best_val_metrics", ok_run.summary["best_epoch"] - 1),
            ("final_val_metrics", len(ok_run.rows) - 1)):
        for key in csv_val:
            assert ok_run.summary[block][key] == pytest.approx(
                float(ok_run.rows[index][key]), rel=1e-5), (block, key)


def test_epoch_times_are_positive_one_per_epoch_and_equal_the_run_log_seconds(ok_run) -> None:
    times = ok_run.summary["epoch_times"]
    assert len(times) == len(ok_run.rows) == ok_run.summary["epochs_run"]
    assert all(isinstance(t, float) and t > 0.0 for t in times), times
    printed = [seconds for *_, seconds in _epoch_lines(ok_run.run_dir)]
    assert len(printed) == len(times)
    assert all(abs(t - p) <= 0.05 + 1e-9 for t, p in zip(times, printed)), (times, printed)
    assert sum(times) < ok_run.summary["fit_wall_seconds"]


def test_the_summary_records_the_epoch_budget_and_whether_the_run_stopped_early(ok_run) -> None:
    summary = ok_run.summary
    literal = {"e2e": (3, 3, False), "early_stop_run": (4, 3, True), "default_run": (2, 2, False)}
    assert (summary["epochs_requested"], summary["epochs_run"], summary["stopped_early"]) == \
        literal[ok_run.name]


def test_the_scripted_scenarios_have_the_literal_best_epoch_and_final_is_best(ok_run) -> None:
    literal = {"early_stop_run": (2, 1, False), "default_run": (2, 1, True)}
    if ok_run.name in literal:
        best, index, final = literal[ok_run.name]
        assert (ok_run.summary["best_epoch"], ok_run.summary["best_epoch_csv_index"],
                ok_run.summary["final_is_best"]) == (best, index, final)


def test_the_summary_records_the_schedule_the_run_used(ok_run) -> None:
    summary = ok_run.summary
    lrs = [float(row["lr"]) for row in ok_run.rows]
    assert summary["learning_rate"] == 1e-3 and summary["warmup_epochs"] == 1
    assert summary["steps_per_epoch"] == 3
    assert summary["lr_first_epoch"] == pytest.approx(lrs[0], rel=1e-6) == pytest.approx(1e-8)
    assert summary["lr_last_epoch"] == pytest.approx(lrs[-1], rel=1e-6)
    # The last STEP of the run is after the last epoch's first step: the rate is still falling.
    assert 0.0 < summary["lr_last_step"] < summary["lr_last_epoch"]


def test_the_summary_records_the_noise_block(ok_run) -> None:
    noise = ok_run.summary["noise"]
    assert all(key in noise for key in NOISE_KEYS), noise
    assert (noise["start"], noise["end"], noise["schedule"], noise["sigma_min"]) == \
        (0.025, 0.25, "linear", 0.0)


def test_the_summary_records_the_model_and_the_finished_run_facts(ok_run) -> None:
    summary = ok_run.summary
    assert summary["variant"] == "tiny"
    assert isinstance(summary["params"], int) and summary["params"] > 1000
    assert summary["init_from"] is None
    assert summary["test_eval"]["status"] == "ok"      # the fixtures evaluate on generated sets
    assert summary["model_loading_validated"] is (ok_run.name != "default_run")
    assert summary["fit_wall_seconds"] > 0.0
    assert summary["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES")
    assert summary["gpu_name"] is None or isinstance(summary["gpu_name"], str)


def test_the_notes_say_how_to_read_the_epoch_index_and_the_lr_column(ok_run) -> None:
    notes = ok_run.summary["notes"]
    assert isinstance(notes, list) and notes and all(isinstance(n, str) and n for n in notes)
    joined = " ".join(notes)
    assert "1-based" in joined and "0-based" in joined and "START of the epoch" in joined


def test_a_diverged_summary_carries_the_shared_keys_and_null_for_the_non_finite(diverged_run) -> None:
    summary = _strict_json(diverged_run.run_dir / "results_summary.json")
    missing = [key for key in DIVERGED_KEYS if key not in summary]
    assert not missing, missing
    assert summary["stopped_early"] is None and summary["best_epoch"] is None
    assert summary["history"]["loss"] == [summary["history"]["loss"][0], None]
    assert isinstance(summary["history"]["loss"][0], float)


def test_a_diverged_summary_names_the_real_directory_and_times_each_epoch_it_ran(diverged_run) -> None:
    summary = _strict_json(diverged_run.run_dir / "results_summary.json")
    assert Path(summary["run_dir"]).resolve() == diverged_run.run_dir.resolve()
    assert summary["epochs_run"] == 2
    assert len(summary["epoch_times"]) == 2 and all(t > 0.0 for t in summary["epoch_times"])
    assert summary["fit_wall_seconds"] > 0.0
    assert summary["init_from"] is None and summary["variant"] == "tiny"
    assert all(key in summary["noise"] for key in NOISE_KEYS)
    assert summary["lr_first_epoch"] == pytest.approx(1e-8)
    assert summary["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES")


# ---------------------------------------------------------------------
# iter-1/step-9 (plan-2026-09-19T131351-b8d39688/D-009): held-out test evaluation
# ---------------------------------------------------------------------

# Typed here from the plan, never read from the trainer.
TEST_SIGMAS_255 = (15, 25, 50)
TEST_SEED = 42
CELL_KEYS = ("input_psnr", "best_psnr", "final_psnr", "gain_db", "final_gain_db", "n_patches")


def _held_out_crops(directory: str):
    """``[(sigma_255, clean, noisy)]`` the instrument must have used on ``directory``.

    The crops are drawn by the SCRIPT's sampler (that sampler is the instrument the trainer
    promises to reproduce), but the seed, the fresh per-dataset generator, the path order and
    every number computed from them are re-derived here, so a trainer that seeds differently
    or shares one generator across sets no longer matches.
    """
    from train.bfunet import eval_psnr_vs_noise as ev

    paths = sorted(str(p) for p in Path(directory).glob("*.png"))
    config = ev.EvalConfig(models={}, datasets={}, num_samples=TEST_NUM_SAMPLES, patch_size=PATCH)
    rng = np.random.RandomState(TEST_SEED)
    clean = ev.sample_clean_patches(config, paths, rng)
    return [
        (sigma, clean, ev.add_awgn(clean, sigma / 255.0, True, rng))
        for sigma in TEST_SIGMAS_255
    ]


def _mean_psnr(image: np.ndarray, clean: np.ndarray) -> float:
    """Mean over the crops of ``20 * log10(1 / rmse)`` (the peak is 1.0 on the unit domain)."""
    rmse = np.sqrt(np.mean((image.astype(np.float64) - clean.astype(np.float64)) ** 2,
                           axis=(1, 2, 3)))
    return float(np.mean(20.0 * np.log10(1.0 / rmse)))


def _block(run_dir: Path) -> Dict:
    return _strict_json(run_dir / "results_summary.json")["test_eval"]


def test_the_test_eval_block_has_every_dataset_sigma_and_field(e2e, held_out_sets) -> None:
    block = _block(e2e.run_dir)
    assert block["status"] == "ok"
    assert (block["seed"], block["patch_size"], block["num_samples"]) == (42, 16, 5)
    assert block["sigmas_255"] == [15, 25, 50]
    assert sorted(block["datasets"]) == ["cbsd68", "kodak24"]
    for name, entry in block["datasets"].items():
        assert entry["status"] == "ok", name
        assert entry["directory"] == held_out_sets[name]
        assert sorted(entry["sigmas"]) == ["15", "25", "50"], name
        for sigma, cell in entry["sigmas"].items():
            assert sorted(cell) == sorted(CELL_KEYS), (name, sigma)
            assert cell["n_patches"] == 5
            assert all(np.isfinite(cell[key]) for key in CELL_KEYS), (name, sigma)


def test_the_input_psnr_is_recomputed_from_the_same_crops(e2e, held_out_sets) -> None:
    """The noisy-input baseline: 20*log10(1/rmse) of the noisy crops against the clean ones."""
    block = _block(e2e.run_dir)
    for name, directory in held_out_sets.items():
        for sigma, clean, noisy in _held_out_crops(directory):
            recorded = block["datasets"][name]["sigmas"][str(sigma)]["input_psnr"]
            assert recorded == pytest.approx(_mean_psnr(noisy, clean), abs=1e-4), (name, sigma)


def test_the_two_sets_get_different_crops_so_the_reseed_is_visible(held_out_sets) -> None:
    """Anti-vacuity for the re-seed: a generator shared across the sets would change the
    second set's crops, and that has to show up in the number recomputed above."""
    crops = {name: _held_out_crops(directory) for name, directory in held_out_sets.items()}
    assert not np.array_equal(crops["kodak24"][0][1], crops["cbsd68"][0][1])


@pytest.mark.parametrize("run_name", ["e2e", "early_stop_run", "default_run"])
def test_the_gains_are_the_psnr_minus_the_noisy_input(request, run_name) -> None:
    block = _block(request.getfixturevalue(run_name).run_dir)
    for name, entry in block["datasets"].items():
        for sigma, cell in entry["sigmas"].items():
            assert cell["gain_db"] == pytest.approx(cell["best_psnr"] - cell["input_psnr"], abs=1e-9)
            assert cell["final_gain_db"] == pytest.approx(
                cell["final_psnr"] - cell["input_psnr"], abs=1e-9), (name, sigma)


def test_best_and_final_are_the_same_number_when_the_final_epoch_is_the_best(default_run) -> None:
    """``default_run`` scripts val_loss [0.9, 0.5]: epoch 2 is best AND last, so the file the
    best number is read from and the model in memory hold the same weights."""
    summary = _strict_json(default_run.run_dir / "results_summary.json")
    assert summary["final_is_best"] is True
    for entry in summary["test_eval"]["datasets"].values():
        for cell in entry["sigmas"].values():
            assert cell["best_psnr"] == cell["final_psnr"]
            assert cell["gain_db"] == cell["final_gain_db"]


def test_best_and_final_differ_when_the_best_epoch_is_not_the_last(early_stop_run) -> None:
    """Anti-vacuity for the equality above, and for a best/final swap: scripted val_loss
    [1.0, 0.5, 0.9] makes the best epoch 2 of 3, so the two files hold different weights."""
    summary = _strict_json(early_stop_run.run_dir / "results_summary.json")
    assert summary["final_is_best"] is False
    cells = [c for e in summary["test_eval"]["datasets"].values() for c in e["sigmas"].values()]
    assert any(c["best_psnr"] != c["final_psnr"] for c in cells)


@pytest.mark.parametrize("kind", ["best", "final"])
def test_each_number_is_that_checkpoints_own_psnr_computed_independently(
        early_stop_run, held_out_sets, kind) -> None:
    """The model behind ``best_psnr`` is ``best_model.keras`` and behind ``final_psnr`` the
    last epoch, measured here by loading the file and applying the formula directly (nothing
    from the script's predict or PSNR helpers). Uses the run whose best is not its last."""
    model = keras.models.load_model(early_stop_run.run_dir / f"{kind}_model.keras", compile=False)
    block = _block(early_stop_run.run_dir)
    for name, directory in held_out_sets.items():
        for sigma, clean, noisy in _held_out_crops(directory):
            recorded = block["datasets"][name]["sigmas"][str(sigma)][f"{kind}_psnr"]
            direct = _mean_psnr(np.asarray(model.predict(noisy, batch_size=16, verbose=0)), clean)
            assert recorded == pytest.approx(direct, abs=0.01), (kind, name, sigma)


def test_the_trainers_numbers_reproduce_the_eval_script_with_the_same_arguments(
        e2e, held_out_sets, tmp_path) -> None:
    """One paired ``eval_psnr_vs_noise`` run over the run's own two checkpoints, same seed,
    patch size, sample count and sigmas, gives the numbers the trainer wrote (0.01 dB)."""
    from train.bfunet import eval_psnr_vs_noise as ev

    out = ev.run_evaluation(ev.EvalConfig(
        models={"best": str(e2e.run_dir / "best_model.keras"),
                "final": str(e2e.run_dir / "final_model.keras")},
        datasets={name: [directory] for name, directory in held_out_sets.items()},
        sigmas_255=list(TEST_SIGMAS_255), num_samples=TEST_NUM_SAMPLES, patch_size=PATCH,
        seed=TEST_SEED, output_dir=str(tmp_path), experiment_name="paired_eval",
    ))
    rows = json.loads((out / "psnr_vs_noise.json").read_text())
    block = _block(e2e.run_dir)
    assert len(rows) == 2 * 2 * 3
    for row in rows:
        cell = block["datasets"][row["dataset"]]["sigmas"][str(int(row["sigma_255"]))]
        assert cell[f"{row['model']}_psnr"] == pytest.approx(row["psnr_mean"], abs=0.01), row
        assert cell["input_psnr"] == pytest.approx(row["input_psnr_mean"], abs=1e-6), row
        assert cell["n_patches"] == row["n"]


# --- the helper itself: skip, disable, error -----------------------------------------------

def _capture_dl_log():
    records: List[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.INFO)
    logging.getLogger("dl").addHandler(handler)
    return records, handler


def test_a_missing_directory_skips_that_set_with_a_warning_and_the_rest_still_runs(
        e2e, held_out_sets, tmp_path) -> None:
    gone = str(tmp_path / "no_such_set")
    records, handler = _capture_dl_log()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(common, "TEST_DATASETS", {"kodak24": held_out_sets["kodak24"], "cbsd68": gone})
            block = common._run_test_eval(e2e.config, e2e.final, e2e.run_dir)
    finally:
        logging.getLogger("dl").removeHandler(handler)
    assert block["status"] == "ok"
    assert block["datasets"]["kodak24"]["status"] == "ok"
    skipped = block["datasets"]["cbsd68"]
    assert skipped["status"] == "skipped" and gone in skipped["reason"], skipped
    assert any(r.levelno == logging.WARNING and gone in r.getMessage()
               and "cbsd68" in r.getMessage() for r in records), [r.getMessage() for r in records]


def test_when_every_directory_is_missing_the_whole_block_is_a_skip(e2e, tmp_path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "TEST_DATASETS", {"kodak24": str(tmp_path / "a"), "cbsd68": str(tmp_path / "b")})
        block = common._run_test_eval(e2e.config, e2e.final, e2e.run_dir)
    assert block["status"] == "skipped" and block["reason"]
    assert all(entry["status"] == "skipped" for entry in block["datasets"].values())


def test_a_disabled_test_eval_records_a_skip_and_reads_nothing(e2e, tmp_path) -> None:
    from dataclasses import replace

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "TEST_DATASETS", {"kodak24": str(tmp_path / "never_read")})
        block = common._run_test_eval(replace(e2e.config, test_eval=False), e2e.final, e2e.run_dir)
    assert block == {"status": "skipped", "reason": "disabled"}


def test_a_disabled_run_writes_the_skip_into_its_summary(unset_steps_run) -> None:
    """The fixture leaves ``test_eval`` off (the ``_tiny_config`` default): a real run, not a helper call."""
    summary = _strict_json(unset_steps_run.run_dir / "results_summary.json")
    assert summary["status"] == "ok"
    assert summary["test_eval"] == {"status": "skipped", "reason": "disabled"}


def test_an_error_in_the_test_eval_is_recorded_and_logged_and_never_raised(e2e) -> None:
    """A finished run must survive its own post-run evaluation failing, and say so."""
    from train.bfunet import eval_psnr_vs_noise as ev

    def boom(path):
        raise RuntimeError("scripted load failure")

    records, handler = _capture_dl_log()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ev, "load_denoiser", boom)
            block = common._run_test_eval(e2e.config, e2e.final, e2e.run_dir)
    finally:
        logging.getLogger("dl").removeHandler(handler)
    assert block["status"] == "error"
    assert "RuntimeError" in block["message"] and "scripted load failure" in block["message"]
    assert any(r.levelno == logging.ERROR and "scripted load failure" in r.getMessage()
               for r in records), [r.getMessage() for r in records]


def test_a_model_that_returns_its_input_gains_exactly_zero(e2e) -> None:
    """The baseline and the model share one noisy batch: identity in, identity out."""
    identity = keras.Sequential([keras.Input(shape=(PATCH, PATCH, 3)), keras.layers.Identity()])
    block = common._run_test_eval(e2e.config, identity, e2e.run_dir)
    assert block["status"] == "ok"
    for entry in block["datasets"].values():
        for cell in entry["sigmas"].values():
            assert cell["final_gain_db"] == 0.0
            assert cell["final_psnr"] == cell["input_psnr"]


# --- the flags -----------------------------------------------------------------------------

def _config_main_builds(argv: List[str]) -> TrainingConfig:
    """The config ``main()`` hands to ``train`` for ``argv`` (training stubbed)."""
    import sys

    import train.bfunet.train_convunext_denoiser as trainer

    seen: List[TrainingConfig] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer, "setup_gpu", lambda gpu_id=None: None)
        patch.setattr(trainer, "train", lambda main_config: seen.append(main_config))
        patch.setattr(sys, "argv", ["train_convunext_denoiser", *argv])
        trainer.main()
    (config,) = seen
    return config


@pytest.mark.parametrize("smoke", [False, True], ids=["full", "smoke"])
def test_the_cli_flags_reach_the_config(smoke) -> None:
    extra = ["--smoke"] if smoke else []
    default = _config_main_builds(extra)
    assert (default.test_eval, default.test_num_samples) == (True, 100)
    off = _config_main_builds([*extra, "--no-test-eval", "--test-num-samples", "7"])
    assert (off.test_eval, off.test_num_samples) == (False, 7)
    on = _config_main_builds([*extra, "--test-eval", "--test-num-samples", "3"])
    assert (on.test_eval, on.test_num_samples) == (True, 3)


def test_a_zero_sample_count_is_refused_at_the_command_line() -> None:
    with pytest.raises(ValueError, match="test_num_samples"):
        _config_main_builds(["--test-num-samples", "0"])


# --- step-6 fixes (plan-2026-09-19T224205-49c8bf80, D-015) --------------------------------------
# The end-of-run model_analysis/, the `analyzer` and `visualizations` summary blocks, the model
# summary out of run.log, and one held-out pass when the best epoch is the last one.

MODEL_TABLE_MARKERS = ("Layer (type)", "Total params", "┏", "┃")


@pytest.fixture(scope="module")
def analysis_run(tmp_path_factory) -> SimpleNamespace:
    """A tiny run with the end-of-run analyzer ON (the trainer default; ``_tiny_config`` turns it off)."""
    root = tmp_path_factory.mktemp("bfunet_analysis")
    _write_pngs(root / "train", N_TRAIN, seed=1)
    _write_pngs(root / "val", N_VAL, seed=2)
    config = _tiny_config(root, "analysis_run", epochs=2, model_analysis=True, test_eval=True)
    run_train(config)
    run_dir = root / "out" / "analysis_run"
    return SimpleNamespace(run_dir=run_dir, summary=_strict_json(run_dir / "results_summary.json"))


def test_the_end_of_run_analysis_writes_model_analysis_and_an_ok_analyzer_block(analysis_run) -> None:
    results = analysis_run.run_dir / "model_analysis" / "analysis_results.json"
    assert results.is_file() and results.stat().st_size > 0
    block = analysis_run.summary["analyzer"]
    assert block["status"] == "ok", block
    assert block["analyzers"] == ["weights", "spectral"]
    assert block["error"] is None and block["seconds"] > 0
    assert block["path"] == str(results)


def test_the_visualizations_block_lists_the_files_that_are_on_disk(analysis_run) -> None:
    block = analysis_run.summary["visualizations"]
    on_disk = sorted(p.name for p in (analysis_run.run_dir / "visualizations").iterdir())
    assert block["files"] == on_disk
    assert "training_dashboard.png" in block["files"]
    assert [f"epoch_{e:03d}_denoise_grid.png" for e in (0, 1, 2)] == [
        f for f in block["files"] if f.endswith("_denoise_grid.png")]
    assert block["failed"] == {} and block["seconds"] > 0


def test_a_run_with_the_analysis_off_records_a_skip_and_writes_no_directory(e2e) -> None:
    summary = _strict_json(e2e.run_dir / "results_summary.json")
    assert summary["analyzer"] == {
        "status": "skipped", "loss": None, "accuracy": None, "error": None, "path": None}
    assert not (e2e.run_dir / "model_analysis").exists()
    assert "training_dashboard.png" in summary["visualizations"]["files"]


def test_the_model_table_is_in_model_summary_txt_and_not_in_run_log(e2e) -> None:
    log = (e2e.run_dir / "run.log").read_text()
    assert [m for m in MODEL_TABLE_MARKERS if m in log] == []
    table = (e2e.run_dir / "model_summary.txt").read_text()
    assert "Layer (type)" in table and "Total params" in table
    assert re.search(r"Model summary \([\d,]+ parameters\) written to .*model_summary\.txt", log)


def test_an_analyzer_that_raises_or_writes_nothing_never_fails_the_run(e2e, tmp_path) -> None:
    sample = np.zeros((1, PATCH, PATCH, 3), dtype="float32")
    history = SimpleNamespace(history={})
    config = _tiny_config(tmp_path, "analysis_probe", model_analysis=True)

    def boom(*args, **kwargs):
        raise RuntimeError("scripted analyzer failure")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "run_model_analysis", boom)
        raised = common._run_end_of_run_analysis(e2e.final, sample, history, config, tmp_path)
        patch.setattr(common, "run_model_analysis", lambda *args, **kwargs: None)
        wrote_nothing = common._run_end_of_run_analysis(e2e.final, sample, history, config, tmp_path)
    assert raised["status"] == "error" and "scripted analyzer failure" in raised["error"]
    assert wrote_nothing["status"] == "missing" and wrote_nothing["analyzers"] == []


def _evaluated_model_sets(run, final_is_best: bool) -> List[List[str]]:
    """The model names each ``evaluate_dataset`` call received, per dataset, for one ``_run_test_eval``."""
    from dataclasses import replace

    from train.bfunet import eval_psnr_vs_noise as ev

    calls: List[List[str]] = []
    real = ev.evaluate_dataset

    def spy(models, *args, **kwargs):
        calls.append(sorted(models))
        return real(models, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ev, "evaluate_dataset", spy)
        block = common._run_test_eval(
            replace(run.config, test_eval=True), run.final, run.run_dir, final_is_best=final_is_best)
    assert block["status"] == "ok"
    return calls, block


def test_the_held_out_pass_runs_once_per_set_when_best_is_last_and_twice_otherwise(e2e) -> None:
    once, reused = _evaluated_model_sets(e2e, final_is_best=True)
    twice, scored = _evaluated_model_sets(e2e, final_is_best=False)
    assert once == [["best"], ["best"]] and reused["final_reused_best"] is True
    assert twice == [["best", "final"], ["best", "final"]] and scored["final_reused_best"] is False
    for entry in reused["datasets"].values():
        for cell in entry["sigmas"].values():
            assert (cell["final_psnr"], cell["final_gain_db"]) == (cell["best_psnr"], cell["gain_db"])


def test_a_finished_run_flags_the_reuse_exactly_when_the_final_epoch_is_the_best(default_run) -> None:
    summary = _strict_json(default_run.run_dir / "results_summary.json")
    assert summary["final_is_best"] is True
    assert summary["test_eval"]["final_reused_best"] is True
