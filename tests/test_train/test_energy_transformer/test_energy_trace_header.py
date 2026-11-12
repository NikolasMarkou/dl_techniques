"""``EnergyTraceCallback`` writes ONE header over a resumed run.

The callback appends one row per epoch to a CSV and writes the header row on its first epoch of
EVERY process (``_header_written`` is a per-instance flag). A resumed run is a new process with a new
callback on the same file, so the CSV got a second header row in the middle of the data and no CSV
reader that took the first row as the header could read the rows after it. ``skip_header_if_file_exists``
(default ``False``, so both legacy trainers keep their behaviour) is the opt-in a resume takes: when the
file already holds something, the new callback appends rows and writes no header.

The epoch column is the epoch Keras passes, which is absolute (``fit(initial_epoch=2)`` calls the
first ``on_epoch_end`` with 2), so the rows of a resumed run continue the numbering.

Writes only under ``tmp_path``.
"""

import csv
import inspect
from pathlib import Path
from typing import List

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.energy_transformer import create_energy_transformer_mim
from train.energy_transformer.common import EnergyTraceCallback

N_PATCHES = 16  # a 16 x 16 image in 4 x 4 patches


@pytest.fixture(scope="module")
def stream() -> "tf.data.Dataset":
    rng = np.random.default_rng(0)
    image = rng.standard_normal((8, 16, 16, 3)).astype("float32")
    mask = rng.random((8, N_PATCHES)) < 0.4
    return tf.data.Dataset.from_tensor_slices(
        ((image, mask), rng.standard_normal((8, N_PATCHES, 48)).astype("float32"), mask.astype("float32"))
    ).batch(4)


def fresh_model() -> keras.Model:
    keras.utils.set_random_seed(0)
    model = create_energy_transformer_mim("tiny", (16, 16, 3), 4, num_steps=2)
    model.compile(optimizer="adam", loss="mse")
    return model


def run_epochs(stream, csv_path: Path, initial_epoch: int, epochs: int, **callback_kwargs) -> None:
    """One process: a new model object, a new callback on the same file, ``epochs - initial_epoch`` epochs."""
    probe_inputs, _, _ = next(iter(stream))
    callback = EnergyTraceCallback(probe_inputs=probe_inputs, csv_path=str(csv_path), **callback_kwargs)
    fresh_model().fit(stream, initial_epoch=initial_epoch, epochs=epochs, callbacks=[callback], verbose=0)


def rows_of(csv_path: Path) -> List[List[str]]:
    with open(csv_path, newline="") as handle:
        return list(csv.reader(handle))


def headers_in(rows: List[List[str]]) -> int:
    return sum(1 for row in rows if row and row[0] == "epoch")


def test_a_resumed_process_writes_no_second_header_and_continues_the_epoch_numbers(stream, tmp_path):
    path = tmp_path / "energy_trace.csv"
    run_epochs(stream, path, 0, 2)
    run_epochs(stream, path, 2, 4, skip_header_if_file_exists=True)
    rows = rows_of(path)
    assert headers_in(rows) == 1 and rows[0][0] == "epoch"
    assert [row[0] for row in rows[1:]] == ["0", "1", "2", "3"]
    assert all(len(row) == len(rows[0]) for row in rows)  # every row has the header's columns
    assert rows[0][1:] == [f"step_{i}" for i in range(len(rows[0]) - 1)]


def test_the_default_keeps_the_old_behaviour_a_header_in_every_process(stream, tmp_path):
    path = tmp_path / "energy_trace.csv"
    run_epochs(stream, path, 0, 2)
    run_epochs(stream, path, 2, 4)
    rows = rows_of(path)
    assert headers_in(rows) == 2, "the legacy callers get what they always got"
    assert [row[0] for row in rows] == ["epoch", "0", "1", "epoch", "2", "3"]


def test_the_flag_is_off_by_default_and_a_keyword():
    parameter = inspect.signature(EnergyTraceCallback.__init__).parameters["skip_header_if_file_exists"]
    assert parameter.default is False


def test_a_first_process_with_the_flag_on_still_writes_the_header(stream, tmp_path):
    path = tmp_path / "nested" / "energy_trace.csv"  # neither the file nor its directory exists
    run_epochs(stream, path, 0, 2, skip_header_if_file_exists=True)
    rows = rows_of(path)
    assert headers_in(rows) == 1 and [row[0] for row in rows] == ["epoch", "0", "1"]


def test_an_empty_file_is_not_a_header_the_flag_still_writes_one(stream, tmp_path):
    path = tmp_path / "energy_trace.csv"
    path.write_text("")  # a kill between the create and the first write
    run_epochs(stream, path, 0, 1, skip_header_if_file_exists=True)
    rows = rows_of(path)
    assert headers_in(rows) == 1 and [row[0] for row in rows] == ["epoch", "0"]

