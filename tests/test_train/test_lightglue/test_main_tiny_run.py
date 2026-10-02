"""main() end to end with explicit tiny flags (no preset exists), run directory in tmp_path."""

import json
from pathlib import Path

import keras
import numpy as np
import pytest

import train.lightglue.train_lightglue as train_lightglue

from .conftest import write_images


@pytest.fixture(scope="module")
def run(tmp_path_factory, superpoint_path):
    root = tmp_path_factory.mktemp("main_run")
    images = write_images(root, 10)
    out = root / "out"
    code = train_lightglue.main([
        "--superpoint-checkpoint", superpoint_path, "--coco-dir", str(images),
        "--val-images", "4", "--batch-size", "2", "--max-keypoints", "32",
        "--num-layers", "2", "--descriptor-dim", "32", "--num-heads", "2",
        "--epochs", "2", "--steps-per-epoch", "2", "--validation-steps", "1",
        "--warmup-steps", "1", "--seed", "7",
        "--output-dir", str(out), "--experiment-name", "tiny",
    ])
    return {"code": code, "dir": out / "tiny", "images": images, "out": out}


def test_exit_code_and_artifacts(run):
    assert run["code"] == 0
    names = {p.name for p in run["dir"].iterdir()}
    for name in ("config.json", "run.log", "training_log.csv", "training_history.json",
                 "results_summary.json", "lightglue.keras", "best_model.keras"):
        assert name in names, (name, sorted(names))


def test_summary_is_strict_json_with_finite_measured_values(run):
    summary = json.loads((run["dir"] / "results_summary.json").read_text())
    assert summary["model"] == "lightglue" and summary["epochs_run"] == 2
    for key in ("loss", "val_loss", "precision", "recall", "val_precision"):
        assert np.isfinite(summary["final"][key]), key
    assert summary["final"]["loss"] > 0.0
    assert summary["params"]["lightglue_trainable"] > 0 and summary["params"]["superpoint_frozen"] > 0
    stats = summary["label_statistics"]
    assert set(stats) == {"positive", "dustbin", "ignored", "keypoints_per_image"}
    assert summary["lightglue_reload_verified"] is True
    assert summary["best_checkpoint"]["monitor"] == "val_loss"


def test_saved_lightglue_reloads_with_the_trained_weights(run):
    reloaded = keras.saving.load_model(str(run["dir"] / "lightglue.keras"), compile=False)
    best = keras.saving.load_model(str(run["dir"] / "best_model.keras"), compile=False)
    assert type(reloaded).__name__ == type(best).__name__ == "LightGlue"
    assert reloaded.input_dim == 32 and reloaded.num_layers == 2


def test_config_json_records_the_flags(run):
    config = json.loads((run["dir"] / "config.json").read_text())
    assert config["max_keypoints"] == 32 and config["seed"] == 7 and config["val_images"] == 4


def test_a_reused_experiment_name_is_refused(run, superpoint_path):
    with pytest.raises(FileExistsError, match="already holds a run"):
        train_lightglue.main([
            "--superpoint-checkpoint", superpoint_path, "--coco-dir", str(run["images"]),
            "--output-dir", str(run["out"]), "--experiment-name", "tiny"])


def test_image_size_flag_is_checked_against_the_checkpoint(run, superpoint_path, tmp_path):
    code = train_lightglue.main([
        "--superpoint-checkpoint", superpoint_path, "--coco-dir", str(run["images"]),
        "--image-size", "128", "--output-dir", str(tmp_path), "--experiment-name", "size"])
    assert code == 2 and not (tmp_path / "size").exists()
