"""Under ``validation_freq > 1`` the CSV log keeps its ``val_`` columns.

``train.common.callbacks.EpochCSVLogger`` (D-236) froze the header at epoch
0's keys so that a fit without validation gets no ``val_`` column. With
``validation_freq=2`` epoch 0 is not validated either, so the frozen header
had no ``val_`` column and every validated epoch's ``val_loss`` was dropped
(plan-2026-10-14T062616-81938e81 step 11.4(b), D-241). The log now has one
header for the whole run: when a key first appears after epoch 0 the file is
rewritten once with the wider header, and an epoch without validation leaves
its ``val_`` cells empty (never the stock logger's ``NA``).
"""

import csv

import keras
import numpy as np
import pytest

from train.common.callbacks import EpochCSVLogger


def _fit(path, epochs, validation_freq):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(1)])
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    x = np.random.default_rng(0).standard_normal((16, 3)).astype("float32")
    y = x.sum(axis=1, keepdims=True)
    model.fit(x, y, epochs=epochs, batch_size=8, verbose=0,
              validation_data=(x, y), validation_freq=validation_freq,
              callbacks=[EpochCSVLogger(str(path))])


def _read(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def test_validation_every_second_epoch_keeps_the_val_columns(tmp_path):
    path = tmp_path / "log.csv"
    _fit(path, epochs=4, validation_freq=2)
    header, rows = _read(path)
    assert header == ["epoch", "loss", "mae", "val_loss", "val_mae"]
    assert [row[0] for row in rows] == ["0", "1", "2", "3"]
    for row in rows:
        validated = int(row[0]) % 2 == 1  # epochs 2 and 4, one-indexed
        assert all(np.isfinite(float(cell)) for cell in row[1:3])
        if validated:
            assert all(np.isfinite(float(cell)) for cell in row[3:])
        else:
            assert row[3:] == ["", ""]


def test_validation_on_listed_epochs_only(tmp_path):
    """``validation_freq=[3]``: only the third epoch (index 2) is validated."""
    path = tmp_path / "log.csv"
    _fit(path, epochs=4, validation_freq=[3])
    header, rows = _read(path)
    assert header == ["epoch", "loss", "mae", "val_loss", "val_mae"]
    assert [row[3:] == ["", ""] for row in rows] == [True, True, False, True]


def test_every_row_is_on_disk_at_its_epoch_end(tmp_path):
    """Driven by hand: the row is written before the next epoch, and the one
    rewrite keeps the earlier rows with empty cells for the new key."""
    path = tmp_path / "log.csv"
    logger = EpochCSVLogger(str(path))
    logger.on_train_begin()
    logger.on_epoch_end(0, {"loss": 3.0})
    assert _read(path) == (["epoch", "loss"], [["0", "3.0"]])
    logger.on_epoch_end(1, {"loss": 2.0, "val_loss": 2.5})
    assert _read(path) == (["epoch", "loss", "val_loss"],
                           [["0", "3.0", ""], ["1", "2.0", "2.5"]])
    logger.on_epoch_end(2, {"loss": 1.0})
    logger.on_train_end()
    assert _read(path) == (["epoch", "loss", "val_loss"],
                           [["0", "3.0", ""], ["1", "2.0", "2.5"],
                            ["2", "1.0", ""]])


def test_append_is_refused(tmp_path):
    """A rewrite cannot widen a header an earlier run wrote; no caller appends."""
    with pytest.raises(ValueError, match="append"):
        EpochCSVLogger(str(tmp_path / "log.csv"), append=True)
