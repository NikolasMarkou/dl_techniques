"""A fit without validation writes no ``val_`` column to ``training_log.csv``.

Keras 3.8 ``CSVLogger.on_epoch_end`` appends ``val_<key>`` for every key when
epoch 0's logs hold no ``val_`` key (meant for ``validation_freq > 1``), so
every trainer that fits without validation wrote a ``val_`` column per metric,
every cell ``NA`` (F-9 B4). Since plan-2026-10-14T062616-81938e81 step 11.2
(D-236) the shared ``train.common.callbacks.EpochCSVLogger`` freezes the header
at epoch 0's keys; ``create_callbacks`` and every trainer that writes the log
itself use it, and a census keeps ``keras.callbacks.CSVLogger(`` out of
``src/train``.
"""

import ast
import csv
from pathlib import Path

import keras
import numpy as np
import pytest

import train.common.callbacks as callbacks_module

_TRAIN = Path(callbacks_module.__file__).resolve().parents[1]


def _fit(logger, validation: bool, epochs: int = 2):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(1)])
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    x = np.random.default_rng(0).standard_normal((16, 3)).astype("float32")
    y = x.sum(axis=1, keepdims=True)
    model.fit(x, y, epochs=epochs, batch_size=8, verbose=0, callbacks=[logger],
              validation_data=(x, y) if validation else None)


def _read(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def _shared_logger(tmp_path):
    callbacks, run_dir = callbacks_module.create_callbacks(
        "tiny", run_dir=str(tmp_path / "run"), monitor="val_loss",
        include_analyzer=False)
    loggers = [c for c in callbacks if isinstance(c, keras.callbacks.CSVLogger)]
    assert len(loggers) == 1
    return loggers[0], Path(run_dir) / "training_log.csv"


def test_no_validation_no_val_column_and_no_na(tmp_path):
    logger, path = _shared_logger(tmp_path)
    _fit(logger, validation=False)
    header, rows = _read(path)
    assert header == ["epoch", "loss", "mae"]
    assert len(rows) == 2 and all("NA" not in row for row in rows)


def test_with_validation_the_val_columns_are_written(tmp_path):
    logger, path = _shared_logger(tmp_path)
    _fit(logger, validation=True)
    header, rows = _read(path)
    assert header == ["epoch", "loss", "mae", "val_loss", "val_mae"]
    assert len(rows) == 2
    for row in rows:
        assert all(np.isfinite(float(cell)) for cell in row)


def test_the_stock_logger_writes_the_phantom_columns(tmp_path):
    """Control: the behaviour the shared logger removes is real in this Keras."""
    path = tmp_path / "stock.csv"
    _fit(keras.callbacks.CSVLogger(str(path)), validation=False)
    header, rows = _read(path)
    assert header == ["epoch", "loss", "mae", "val_loss", "val_mae"]
    assert all(row[3:] == ["NA", "NA"] for row in rows)


def _stock_csv_logger_calls(source):
    """Line numbers of calls that build ``keras.callbacks.CSVLogger`` (the
    attribute chain, ``callbacks.CSVLogger`` or a bare imported name)."""
    tree = ast.parse(source)
    imported = {alias.asname or alias.name
                for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("keras")
                for alias in node.names if alias.name == "CSVLogger"}
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "CSVLogger":
            lines.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id in imported:
            lines.append(node.lineno)
    return sorted(lines)


def test_the_census_sees_every_construction_form():
    """Positive control: the three forms are found; a subclass call, an
    isinstance check and a mention are not."""
    source = (
        "import keras\n"
        "from keras.callbacks import CSVLogger\n"
        "a = keras.callbacks.CSVLogger('x.csv')\n"
        "b = CSVLogger('x.csv')\n"
        "c = callbacks.CSVLogger('x.csv')\n"
        "d = EpochCSVLogger('x.csv')\n"
        "e = isinstance(d, keras.callbacks.CSVLogger)\n"
        "# keras.callbacks.CSVLogger('x.csv')\n"
    )
    assert _stock_csv_logger_calls(source) == [3, 4, 5]


def test_no_trainer_builds_the_stock_logger():
    offenders = []
    for path in sorted(_TRAIN.rglob("*.py")):
        relative = path.relative_to(_TRAIN)
        if relative.parts[0] == "common":
            continue
        offenders += [f"{relative}:{line}" for line in
                      _stock_csv_logger_calls(path.read_text(encoding="utf-8"))]
    assert offenders == [], (
        "keras.callbacks.CSVLogger writes a val_ column per metric, all NA, "
        "when the fit does not validate; use "
        "train.common.callbacks.EpochCSVLogger:\n" + "\n".join(offenders))


def test_the_shared_logger_is_built_once_in_common():
    common = [line for line in _stock_csv_logger_calls(
        Path(callbacks_module.__file__).read_text(encoding="utf-8"))]
    assert common == [], "create_callbacks builds the stock logger"
