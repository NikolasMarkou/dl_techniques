"""The CLIP trainer validates and keeps the CLIP of its lowest-``val_loss`` epoch.

Until plan-2026-10-14T062616-81938e81 step 11.2 (F-9 B5, D-237) ``main`` had no
validation split and saved only ``final_model.keras``, the last epoch: a run
that overfit kept its worst model. ``--val-fraction`` (default 0.1) now holds
out a seeded split, ``fit`` gets it as ``validation_data`` through the
contrastive bridge, and ``train.common.callbacks.BestSubmodelCheckpoint``
writes the wrapped CLIP (never the wrapper, D-036) to ``best_model.keras``
whenever ``val_loss`` improves.
"""

import json

import keras
import numpy as np
import pytest

import train.clip.train_clip as train_clip
from train.common.callbacks import BestSubmodelCheckpoint

from tests.layer_naming import PublicAutoName

from .test_the_trainer_entry_point_trains_end_to_end import _TINY

_BATCH = 8


class _Recording(BestSubmodelCheckpoint):
    """Snapshot the sub-model's weights at every epoch end."""

    snapshots = []

    def on_epoch_end(self, epoch, logs=None):
        type(self).snapshots.append(
            [np.array(w) for w in getattr(self.model, self.attribute).weights])
        super().on_epoch_end(epoch, logs)


def test_the_best_file_is_the_clip_of_the_lowest_val_loss_epoch(tmp_path, monkeypatch):
    _Recording.snapshots = []
    monkeypatch.setattr(train_clip, "BestSubmodelCheckpoint", _Recording)
    run_dir = train_clip.main([
        *_TINY, "--synthetic", "--synthetic-samples", str(3 * _BATCH),
        "--val-fraction", "0.34", "--batch-size", str(_BATCH), "--epochs", "5",
        "--output-dir", str(tmp_path / "run")])

    history = json.loads((run_dir / "training_history.json").read_text())
    val_losses = history["val_loss"]
    assert len(val_losses) == 5 and all(np.isfinite(val_losses))
    best_epoch = int(np.argmin(val_losses))
    # Anti-vacuity: at this seed the lowest val_loss is not the last epoch
    # (measured [3.47, 3.35, 3.17, 2.67, 2.82] on the RTX 4070), so the best
    # file and final_model.keras must differ; a callback that saved every
    # epoch would hold the last epoch's weights and fail below.
    assert best_epoch < len(val_losses) - 1, val_losses

    best = keras.models.load_model(run_dir / "best_model.keras")
    assert type(best).__name__ == "CLIP"
    saved = [np.array(w) for w in best.weights]
    expected = _Recording.snapshots[best_epoch]
    assert len(saved) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(saved, expected))
    last = _Recording.snapshots[-1]
    assert not all(np.array_equal(a, b) for a, b in zip(saved, last))


def test_no_validation_split_saves_no_best_model(tmp_path):
    run_dir = train_clip.main([
        *_TINY, "--synthetic", "--synthetic-samples", str(_BATCH),
        "--val-fraction", "0", "--batch-size", str(_BATCH), "--epochs", "1",
        "--output-dir", str(tmp_path / "run")])
    assert (run_dir / "final_model.keras").is_file()
    assert not (run_dir / "best_model.keras").exists()
    assert "val_loss" not in json.loads(
        (run_dir / "training_history.json").read_text())


def test_the_split_is_seeded_disjoint_and_complete():
    train, val = train_clip.split_validation(20, 0.25, seed=3)
    assert len(val) == 5 and len(train) == 15
    assert set(train).isdisjoint(val) and set(train) | set(val) == set(range(20))
    again = train_clip.split_validation(20, 0.25, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip((train, val), again))
    assert len(train_clip.split_validation(20, 0.0, seed=3)[1]) == 0


@pytest.mark.parametrize("count, fraction", [(20, -0.1), (20, 1.0), (4, 0.05), (2, 0.9)])
def test_a_split_that_leaves_a_side_empty_is_refused(count, fraction):
    with pytest.raises(ValueError, match="val-fraction"):
        train_clip.split_validation(count, fraction, seed=0)


class _Holder(PublicAutoName, keras.Model):
    def __init__(self):
        super().__init__()
        self.inner = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])


def test_the_callback_saves_the_sub_model_at_each_improvement_only(tmp_path):
    """Driven by hand with ``val_loss`` 3, 1, 2, nan: the file holds the
    epoch-1 weights, written from the sub-model, not the holder."""
    holder = _Holder()
    path = tmp_path / "best.keras"
    callback = BestSubmodelCheckpoint(str(path), "inner", monitor="val_loss")
    callback.set_model(holder)
    kernel = holder.inner.layers[0].kernel
    seen = []
    for epoch, value in enumerate([3.0, 1.0, 2.0, float("nan")]):
        kernel.assign(np.full((2, 1), float(epoch), "float32"))
        seen.append(np.array(kernel))
        callback.on_epoch_end(epoch, {"val_loss": value})
    assert callback.best == 1.0 and callback.best_epoch == 1
    restored = keras.models.load_model(path)
    assert isinstance(restored, keras.Sequential)
    np.testing.assert_array_equal(np.array(restored.layers[0].kernel), seen[1])
