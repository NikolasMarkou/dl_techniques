"""The TensorFlow-free half of the forecasting data: a csv as a series array, and the split by time.

``train.common`` imports TensorFlow (``train.common.__init__`` loads ``gpu``), and the checks of a forecasting
run that can be made from the file alone (does it exist, are the named columns there and numeric, is the time
column sorted, how many rows is each segment) have to run BEFORE the GPU is claimed. So they live here, with
pandas and numpy only, and ``train.common.forecast_data`` imports them: one implementation, no copy in the
pre-claim check and the loader.

pandas and numpy are imported INSIDE the functions that read a csv, never at module scope: ``train.api.cli``
imports the forecasting task's argument module (and so this one) when it registers the tasks, and ``python -m
train help``, the help of every command and ``import train.api`` load no heavy library
(``test_cli_help_allocates_nothing`` reads ``sys.modules`` for it).

Interface contract
------------------
``read_csv_series(path, value_cols, time_col)``
    :param path: the csv file; ``None`` or a missing file is a refusal.
    :param value_cols: the value columns, or ``None`` for every column except the time column and an unnamed
        index column.
    :param time_col: the timestamp column, or ``None`` (row order is time).
    :return: ``(values, names, sha256)``: ``(rows, channels)`` float64, the value column names, and the sha256 of
        the file bytes.
    :raises ConfigError: a missing file or column, an unreadable or empty csv, a non-numeric or non-finite value
        column, a time column that is not strictly increasing.

``split_bounds(rows, select_fraction, test_fraction)``
    :return: the ``(lo, hi)`` row ranges of the train, select and test segments, consecutive and disjoint.

``check_segments(segments, rows, width, source)``
    :param segments: the ``(lo, hi)`` ranges of :func:`split_bounds`.
    :param rows: the rows of the whole series or csv; ``width``: ``lookback + horizon``; ``source``: how the message
        names the series (``"a synthetic series"``, ``"the csv"``).
    :raises ConfigError: for the first segment (train, select, test) shorter than ``width``; the one sentence of that
        refusal, said by the pre-claim check and by the loader alike.

``refuse(message)``
    :raises ConfigError: always; the one spelling of "this configuration is refused" of the forecasting data.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, NoReturn, Optional, Sequence, Tuple

from train.api.errors import ConfigError

if TYPE_CHECKING:  # annotations only; the runtime imports are inside the functions (module docstring)
    import numpy as np
    import pandas as pd


def refuse(message: str) -> NoReturn:
    """Raise :class:`ConfigError` with ``message``."""
    raise ConfigError(message)


def split_bounds(rows: int, select_fraction: float, test_fraction: float) -> Tuple[Tuple[int, int], ...]:
    """The consecutive ``(lo, hi)`` row ranges of the train, select and test segments of ``rows`` rows."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-006
    # The split is BY TIME and the segments are disjoint: train first, then select, then test. Do NOT split
    # windows at random (a window straddling a boundary puts a validation row inside a training window) and
    # do NOT let a window of one segment reach into the next: every window of a segment lies inside it.
    n_select, n_test = round(select_fraction * rows), round(test_fraction * rows)
    train_hi = rows - n_select - n_test
    return (0, train_hi), (train_hi, train_hi + n_select), (train_hi + n_select, rows)


def check_segments(segments: Sequence[Tuple[int, int]], rows: int, width: int, source: str) -> None:
    """Refuse the first of the train, select and test ``segments`` that is shorter than one window of ``width`` rows."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-037
    # The pre-claim check (train.api.forecast_args) and the loader (train.common.forecast_data) both call THIS function.
    # Do NOT spell the sentence, or loop over the segments, in either of them again: that is how the two sentences
    # could drift apart (a test holds both callers to this one function and the sentence to its pinned text).
    for name, (lo, hi) in zip(("train", "select", "test"), segments):
        if hi - lo < width:
            refuse(
                f"{source} of {rows} rows is too short: its {name} segment has {hi - lo} rows and one window needs "
                f"lookback + horizon = {width}; give it more rows or lower data.lookback, data.horizon or the fractions")


def read_csv_series(
        path: Optional[str], value_cols: Optional[Sequence[str]], time_col: Optional[str],
) -> "Tuple[np.ndarray, List[str], str]":
    """``(values, names, sha256)``: a csv as ``(rows, channels)`` float64, its value columns and its byte hash."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-036
    # This module imports pandas and numpy only, and only here, inside the function. Do NOT import train.common,
    # tensorflow or keras here (and do NOT copy these checks into a second reader for the pre-claim step):
    # train.common loads TensorFlow, which must not be loaded before setup_gpu has run, and two readers drift
    # apart. Do NOT hoist the numpy and pandas imports to module scope either: the help path imports this module
    # (through forecast_args) and must stay free of them, which a module-scope import broke in step 12.
    import numpy as np
    import pandas as pd

    if path is None:
        refuse("data.path is required for data.format 'csv'")
    file = Path(path)
    if not file.is_file():
        refuse(f"data.path {path} does not exist")
    raw = file.read_bytes()
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ConfigError(f"data.path {path} is not a readable csv: {exc}") from exc
    if len(frame) == 0:
        refuse(f"data.path {path} has no rows")
    for key, name in (("data.time_col", time_col),) + tuple(("data.value_cols", c) for c in value_cols or ()):
        if name is not None and name not in frame.columns:
            refuse(f"{key}: column {name!r} is missing from {file.name}, which has {list(frame.columns)}")
    # D-027: no implicit index or time column is a value column
    names = list(value_cols) if value_cols else [
        c for c in frame.columns if c != time_col and not re.fullmatch(r"Unnamed: \d+", str(c))]
    if not names:
        refuse(f"{file.name} has no value column: every column is the time column or an unnamed index")
    for name in names:
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            refuse(f"column {name!r} is not numeric (dtype {frame[name].dtype}): one text or decimal-comma "
                   "cell makes a whole column non-numeric; name the numeric columns in data.value_cols")
    values = frame[names].to_numpy(np.float64)
    bad = ~np.isfinite(values)
    if bad.any():
        row, col = (int(v[0]) for v in np.nonzero(bad))
        refuse(f"column {names[col]!r} holds {int(bad[:, col].sum())} non-finite cell(s), the first at row {row}")
    if time_col is not None:
        _check_time_column(frame[time_col], time_col)
    return values, names, hashlib.sha256(raw).hexdigest()


def _check_time_column(column: "pd.Series", name: str) -> None:
    import pandas as pd

    if column.dtype == object:
        try:
            column = pd.to_datetime(column)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"data.time_col {name!r} cannot be parsed as timestamps: {exc}") from exc
    if column.isna().any() or not (column.is_monotonic_increasing and column.is_unique):
        refuse(f"data.time_col {name!r} must be strictly increasing with no empty cell: sort the file by time")
