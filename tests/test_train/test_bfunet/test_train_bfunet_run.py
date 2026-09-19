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
import json
import os
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
