"""Tiny synthetic smoke run for the SiamFC trainer (offline, deterministic)."""

import numpy as np

import train.siamfc.train_siamfc as train_siamfc


def test_tiny_run_trains_and_checkpoints(tmp_path) -> None:
    config = train_siamfc.SiamFCTrainingConfig(
        data_source="synthetic",
        num_synthetic_train=8,
        num_synthetic_val=4,
        batch_size=2,
        epochs=1,
        seed=0,
        augment=False,
        output_dir=str(tmp_path),
        experiment_name="tiny",
    )
    result = train_siamfc.train_siamfc(config, gpu_id=None)
    assert np.isfinite(result["best_val_loss"])
    assert result["epochs_run"] == 1
    assert list(tmp_path.rglob("best_model.keras")), "no best checkpoint written"
    assert list(tmp_path.rglob("config.json")), "no config.json written"
