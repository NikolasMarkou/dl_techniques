"""Tiny synthetic smoke runs for the DeepSORT trainer (both loss modes)."""

import numpy as np
import pytest

import train.deepsort.train_deepsort as train_deepsort


def _tiny_config(tmp_path, loss_mode, name):
    return train_deepsort.DeepSortTrainingConfig(
        data_source="synthetic",
        num_synthetic_ids=6,
        shots_per_id=4,
        p_ids=3,
        k_shots=2,
        validation_id_fraction=0.5,  # 3 val IDs: PK sampling needs >= p_ids
        epochs=1,
        steps_per_epoch=2,
        val_steps=1,
        loss_mode=loss_mode,
        seed=0,
        augment=False,
        output_dir=str(tmp_path),
        experiment_name=name,
    )


@pytest.mark.parametrize("loss_mode", ["cosine-softmax", "triplet"])
def test_tiny_run_trains_and_checkpoints(tmp_path, loss_mode) -> None:
    result = train_deepsort.train_deepsort(
        _tiny_config(tmp_path, loss_mode, f"tiny-{loss_mode}"), gpu_id=None
    )
    assert np.isfinite(result["best_val_loss"])
    assert result["epochs_run"] == 1
    assert 0.0 <= result["rank1"] <= 1.0
    assert list(tmp_path.rglob("best_model.keras")), "no best checkpoint written"
    assert list(tmp_path.rglob("config.json")), "no config.json written"
