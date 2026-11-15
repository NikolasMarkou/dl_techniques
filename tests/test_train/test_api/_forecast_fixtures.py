"""What the forecasting tests share: the registry ``world``, the one series csv writer and its fixture, the numpy oracle of the standardization.

``world``
    A module-scope fixture: an EMPTY registry holding every built-in family and task, restored afterwards.

``write_series_csv(path, rows=CSV_ROWS, columns=1, time_col="t", values=None, cells=None)``
    :param columns: an int (the columns are named ``v0``, ``v1``, ...) or the column names; every column is a random walk
        with a daily season (the same numbers for the same ``rows`` and column count).
    :param time_col: the name of a dated column written first (hourly stamps), or ``None`` for no such column.
    :param values: ``{column: array}``; replaces a column's numbers, or adds a column, or (for ``time_col``) replaces the stamps.
    :param cells: ``{(row, column): text}``; replaces single cells with the raw text (``"nan"``, ``"abc"``, an out-of-order stamp).
    :return: ``path``. Every csv of the forecasting tests is written by this function, so a csv fact (the dates, the walk) has one source.

``csv_file``
    A module-scope fixture: ``write_series_csv`` of :data:`CSV_ROWS` rows, one column ``v0`` and the dated column ``t``.

``standardize_np(window, lookback, scale_floor)``
    The test oracle of ``forecast_data.standardize_tf``: ``(x, y)`` float32 from ``(..., lookback + horizon, channels)``, the
    context's first-value anchor, offset and scale taken from ``forecast_data.context_stats_np`` (the function the Predictor
    standardizes with) and applied to the whole window. It lives here, not in ``train.common``, because nothing in production
    standardizes a context and its target together in numpy.

The module imports numpy and pandas at module scope and ``train.common`` inside the function that needs it, so the child process
of ``test_forecast_resume.py`` can import it after it has set up TensorFlow.
"""

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytest

from train.api import recipes, registry
from train.api.tasks import register_builtin_tasks

#: 2,720 rows give a select and a test segment of 408 rows each at the data defaults of ``test_forecast_task`` (lookback 32,
#: horizon 8, batch 16): 47 windows with non-overlapping targets (stride 8, width 40), so ``n_select`` and ``n_test`` left unset
#: are 32 (47 rounded down to the batch), and a window that wrongly counts only the lookback would make a 48th that leaves the
#: segment (the csv test of ``test_forecast_task`` reads every window, none is left unused).
CSV_ROWS = 2720


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def write_series_csv(
        path: Path,
        rows: int = CSV_ROWS,
        columns: Union[int, Sequence[str]] = 1,
        time_col: Optional[str] = "t",
        values: Optional[Mapping[str, Any]] = None,
        cells: Optional[Mapping[Tuple[int, str], str]] = None,
) -> Path:
    """Write a csv of ``rows`` rows and return ``path``; see the module contract."""
    names = [f"v{i}" for i in range(columns)] if isinstance(columns, int) else list(columns)
    rng = np.random.default_rng(7)
    seasons = np.sin(2 * np.pi * np.arange(rows) / 24.0)[:, None]
    walk = np.cumsum(rng.normal(size=(rows, len(names))), axis=0) * 0.2 + 3.0 * seasons
    frame = pd.DataFrame(walk, columns=names)
    if time_col is not None:
        frame.insert(0, time_col, pd.date_range("2020-01-01", periods=rows, freq="h").strftime("%Y-%m-%d %H:%M:%S"))
    for column, numbers in (values or {}).items():
        frame[column] = numbers
    for (row, column), text in (cells or {}).items():
        frame[column] = frame[column].astype(object)
        frame.at[row, column] = text
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def csv_file(tmp_path_factory) -> Path:
    """The one csv a module reads (written once)."""
    return write_series_csv(tmp_path_factory.mktemp("series_csv") / "series.csv")


def standardize_np(window: np.ndarray, lookback: int, scale_floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """The numpy oracle of ``standardize_tf``; see the module contract."""
    from train.common.forecast_data import context_stats_np

    window = np.asarray(window, np.float64)
    anchor, offset, scale = context_stats_np(window[..., :lookback, :], scale_floor)
    standardized = (window - anchor - offset) / scale
    return standardized[..., :lookback, :].astype(np.float32), standardized[..., lookback:, :].astype(np.float32)
