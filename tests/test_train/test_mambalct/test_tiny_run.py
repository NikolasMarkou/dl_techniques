"""Tiny synthetic smoke run for the MambaLCT trainer (offline, deterministic)."""

import numpy as np

import train.mambalct.train_mambalct as train_mambalct


def test_tiny_run_trains_and_checkpoints(tmp_path) -> None:
    config = train_mambalct.MambaLCTTrainingConfig(
        data_source="synthetic",
        num_synthetic_train=8,
        num_synthetic_val=4,
        batch_size=2,
        epochs=1,
        seed=0,
        augment=False,
        clip_length=1,
        template_size=32,
        search_size=64,
        allow_custom_geometry=True,
        stage_dims=[16, 32, 512],
        stage_depths=[1, 1, 1],
        num_heads=[2, 2, 8],
        output_dir=str(tmp_path),
        experiment_name="tiny",
    )
    result = train_mambalct.train_mambalct(config, gpu_id=None)
    assert np.isfinite(result["best_val_loss"])
    assert result["epochs_run"] == 1
    assert list(tmp_path.rglob("best_model.keras")), "no best checkpoint written"
    assert list(tmp_path.rglob("config.json")), "no config.json written"
