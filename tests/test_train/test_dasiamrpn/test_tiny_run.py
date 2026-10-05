"""Tiny synthetic smoke run for the DaSiamRPN trainer (offline, deterministic)."""

import numpy as np

import train.dasiamrpn.train_dasiamrpn as train_dasiamrpn


def test_tiny_run_trains_and_checkpoints(tmp_path) -> None:
    config = train_dasiamrpn.DaSiamRPNTrainingConfig(
        variant="otb",
        data_source="synthetic",
        num_synthetic_train=4,
        num_synthetic_val=2,
        batch_size=1,
        epochs=1,
        seed=0,
        augment=False,
        output_dir=str(tmp_path),
        experiment_name="tiny",
    )
    result = train_dasiamrpn.train_dasiamrpn(config, gpu_id=None)
    assert np.isfinite(result["best_val_loss"])
    assert result["epochs_run"] == 1
    assert list(tmp_path.rglob("best_model.keras")), "no best checkpoint written"
    assert list(tmp_path.rglob("config.json")), "no config.json written"
