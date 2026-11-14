"""Predict with a trained forecasting run: :class:`ForecastPredictor`, the data lister and JSON writer of ``predict``, and the
export row of the ``time_series_forecasting`` task.

``train.api.predictor.load_model`` returns a :class:`ForecastPredictor` when the run's ``task.json`` says
``time_series_forecasting`` (schema in :mod:`train.api.predictor`). A forecaster takes a window of the series in the series'
OWN units and returns the forecast in the same units: the window is standardized by its own CONTEXT statistics (the numpy twin
of the training function, :func:`train.common.forecast_data.context_stats_np`, so the Predictor, the bar and the export read one
rule), the model runs, the family's decode row (:data:`train.common.forecast_run.DECODE`) turns its output into a point and,
for a quantile head, a quantile cube, and both are de-standardized with the SAME statistics.

Interface contract
------------------
``ForecastPredictor(model, record)``
    :param model: the loaded Keras model; its input is ``(batch, lookback, channels)``.
    :param record: a forecasting record that passed :func:`train.api.predictor.check_record`.
    :raises ConfigError: for a record of another task, and for a record the model does not match: the decode row, the head,
        the quantile levels, the median index, or the shape of the forecast of one window (the horizon and the channels).

``ForecastPredictor.predict(x)``
    :param x: one window ``(L, C)`` or several ``(N, L, C)`` as a ``numpy`` array, a list or tuple of windows (each an array or
        a csv path), or a csv path (``str`` or ``Path``): the LAST ``lookback`` rows of the record's ``value_cols`` (every
        column of a run that named none; their count must be the model's channels). ``L`` is the record's lookback and ``C``
        its channels; every value must be a finite number.
    :return: ``{"point", "quantiles", "quantile_levels"}``: ``point`` is ``(N, horizon, channels)`` float64 in the series'
        own units (the median for a quantile head); ``quantiles`` is ``(N, horizon, channels, Q)`` and ``quantile_levels``
        the record's list for a quantile head, and both are ``None`` for a point head. A window is forecast from its own
        statistics only, so a window shifted by a constant gives the forecast shifted by it and a window scaled by a constant
        the forecast scaled.
    :raises ConfigError: for a window of another length or channel count, a non-finite or non-numeric value, an empty list or
        array, an input of another kind, and a csv that is missing, has too few rows, lacks a column or holds a non-numeric or
        non-finite one.

``list_data(data)`` / ``read_data(file)``
    The command's input: ``list_data`` returns ``[data]`` for a csv, a ``.npy`` or a ``.npz`` file and refuses a directory
    and any other suffix by name; ``read_data`` returns the csv path as text, or the array of the ``.npy`` or of the one array
    of the ``.npz``.

``write_predictions(predictor, file, source, out)``
    Writes ``{"task", "model", "data", "lookback", "horizon", "channels", "quantile_levels", "point", "quantiles"}`` as one JSON
    file into the new path ``out`` (nothing is overwritten). :raises ConfigError: for any error of ``predict`` and for a
    forecast that is not finite (JSON has no NaN).

``export_pending(record, *_)`` / ``export_note(record)``
    The export row of this task until its served function lands: every cell is refused by name, before anything is written.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only; ``numpy``, ``keras`` and ``train.common`` are imported inside
functions, so ``python -m train help`` allocates nothing.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Tuple

from .errors import ConfigError
from .exporting import check_free
from .predictor import FORECASTING, check_record
from .run_dir import TASK_FILE

#: Windows run through the model at a time.
CHUNK = 32

#: What ``data=`` may name for a forecasting model.
DATA_SUFFIXES = (".csv", ".npy", ".npz")

_DATA_HELP = "give a csv file, a .npy or a .npz of windows"

#: The sentence of every export cell of this task until the served function of step 16 exists.
EXPORT_PENDING = (
    f"export of a {FORECASTING} model is not available yet: the file it writes has to take RAW windows and return a "
    "forecast in the series' units, and its served function is not built; predict works on the run directory"
)


def _refuse_window(message: str) -> NoReturn:
    raise ConfigError(f"Invalid window: {message}")


def _is_path(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


def _csv_window(path: Any, record: Mapping[str, Any]) -> Any:
    """Return the last ``lookback`` rows of the record's value columns of the csv ``path``, ``(L, C)`` float64."""
    from .forecast_series import read_csv_series

    try:
        values, names, _ = read_csv_series(os.fspath(path), record["value_cols"], None)
    except ConfigError as exc:
        raise ConfigError(f"Invalid value for 'data': the csv cannot give the model's window: {exc}") from exc
    if record["value_cols"] is None and values.shape[1] != record["channels"]:
        raise ConfigError(
            f"Invalid value for 'data': {path} has {values.shape[1]} value columns {names} but the model takes "
            f"{record['channels']} channel(s)"
        )
    if len(values) < record["lookback"]:
        raise ConfigError(
            f"Invalid value for 'data': the csv has {len(values)} rows, fewer than the lookback {record['lookback']}"
        )
    return values[-record["lookback"]:]


def _window(item: Any, index: int, record: Mapping[str, Any]) -> Any:
    """Return one item as a finite float64 ``(L, C)`` window, or raise naming ``index``."""
    import numpy as np

    window = _csv_window(item, record) if _is_path(item) else item
    try:
        window = np.asarray(window, np.float64)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid window: the values must be numbers, window {index} could not be read as such ({exc})") from exc
    shape = (record["lookback"], record["channels"])
    if window.shape != shape:
        _refuse_window(
            f"each window must be {shape} (lookback, channels) but window {index} is {tuple(window.shape)}"
        )
    bad = np.argwhere(~np.isfinite(window))
    if len(bad):
        _refuse_window(
            f"window {index} holds a non-finite value (NaN or infinity) at row {int(bad[0][0])}, column {int(bad[0][1])}"
        )
    return window


def _windows_of(x: Any, record: Mapping[str, Any]) -> Any:
    """Return the windows of ``x`` as one float64 array ``(N, L, C)``; see :meth:`ForecastPredictor.predict`."""
    import numpy as np

    if _is_path(x):
        items: List[Any] = [x]
    elif isinstance(x, np.ndarray) and x.ndim in (2, 3):
        if x.size == 0:
            _refuse_window(f"the array of windows is empty (shape {tuple(x.shape)})")
        items = [x] if x.ndim == 2 else list(x)
    elif isinstance(x, np.ndarray):
        _refuse_window(
            f"expected an array (L, C) or (N, L, C), a list of windows or a csv path, got an array of shape {tuple(x.shape)}"
        )
    elif isinstance(x, (list, tuple)):
        if not x:
            _refuse_window("the list of windows is empty")
        items = list(x)
    else:
        _refuse_window(f"expected an array, a list of windows or a csv path, got {type(x).__name__}")
    return np.stack([_window(item, index, record) for index, item in enumerate(items)])


class ForecastPredictor:
    """A loaded forecaster and the standardization it was trained with; see the module contract.

    :param model: the Keras model.
    :param record: a forecasting record that passed :func:`train.api.predictor.check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        from train.common import forecast_run

        check_record(dict(record), "<given>")
        if record["task"] != FORECASTING:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor serves {FORECASTING!r} only"
            )
        self.model = model
        self.record = dict(record)
        self.decoder = forecast_run.decoder_for(record["family"], model)
        self._check_against_the_model()

    def _decode(self, standardized: Any) -> Tuple[Any, Optional[Any]]:
        """Return ``(point, cube)`` numpy float64 of one batch of standardized windows, in standardized units."""
        import keras

        # DECISION plan-2026-11-13T105600-3a8319ef/D-009
        # The model's output is turned into a point and a cube by the family's ONE decode row (the one the compiled metric and
        # the post-hoc report read), bound in `__init__`. Do NOT read `predict_forecast` here (timestral and mdn have none, and
        # its shapes differ per family) and do NOT slice the output by hand (tirex and a quantile xlstm have no channel axis,
        # nbeats returns a tuple whose second element is a residual, not a forecast).
        point, cube = self.decoder(self.model(standardized, training=False))
        to_numpy = lambda tensor: keras.ops.convert_to_numpy(tensor).astype("float64")
        return to_numpy(point), None if cube is None else to_numpy(cube)

    def _check_against_the_model(self) -> None:
        """Raise unless the record's decode row, head, levels, median and forecast shape are the model's."""
        import numpy as np

        record, decoder = self.record, self.decoder
        wrong = f"{TASK_FILE} names"
        tail = "; the file and the model are not from one run"
        if record["decode"] != decoder.row:
            raise ConfigError(f"{wrong} the decode row {record['decode']!r} but the model is {decoder.row!r}{tail}")
        if (record["head"] == "quantile") != (decoder.levels is not None):
            raise ConfigError(
                f"{wrong} a {record['head']} head but the model's {decoder.row} head is "
                f"{'a quantile head' if decoder.levels is not None else 'a point head'}{tail}"
            )
        if decoder.levels is not None:
            levels = [float(q) for q in decoder.levels]
            if len(levels) != len(record["quantile_levels"]) or not np.allclose(levels, record["quantile_levels"], rtol=0, atol=1e-9):
                raise ConfigError(f"{wrong} the quantile levels {record['quantile_levels']} but the model has {levels}{tail}")
            if decoder.median_index != record["point_index"]:
                raise ConfigError(
                    f"{wrong} the point at level index {record['point_index']} but the model's median is level index "
                    f"{decoder.median_index}{tail}"
                )
        point, cube = self._decode(np.zeros((1, record["lookback"], record["channels"]), np.float32))
        wanted = (1, record["horizon"], record["channels"])
        if tuple(point.shape) != wanted or (cube is not None and tuple(cube.shape) != (*wanted, len(decoder.levels))):
            raise ConfigError(
                f"The model forecasts {tuple(point.shape[1:])} for one window but {wrong} horizon {record['horizon']} "
                f"and {record['channels']} channel(s){tail}"
            )

    def predict(self, x: Any) -> Dict[str, Any]:
        """Return ``{"point", "quantiles", "quantile_levels"}`` for ``x``; see the module contract."""
        import numpy as np

        from train.common import forecast_data

        windows = _windows_of(x, self.record)
        # DECISION plan-2026-11-13T105600-3a8319ef/D-010
        # The window's statistics come from its CONTEXT alone and the forecast is de-standardized with the same numbers, by the
        # numpy twin of the training function (D-035: one rule for the bar, the Predictor and the export). Do NOT standardize
        # with the statistics of the training data (a window of another level or spread would be forecast at the training
        # level), do NOT skip the standardization "because the model normalizes" (every family has its own switch forced
        # off, D-023), and do NOT hand back the standardized forecast.
        mean, scale = forecast_data.context_stats_np(windows, self.record["scale_floor"])
        standardized = ((windows - mean) / scale).astype(np.float32)
        points: List[Any] = []
        cubes: List[Any] = []
        for start in range(0, len(standardized), CHUNK):
            point, cube = self._decode(standardized[start:start + CHUNK])
            points.append(point)
            if cube is not None:
                cubes.append(cube)
        quantiles = np.concatenate(cubes) * scale[..., None] + mean[..., None] if cubes else None
        return {
            "point": np.concatenate(points) * scale + mean,
            "quantiles": quantiles,
            "quantile_levels": self.record["quantile_levels"],
        }


# ---------------------------------------------------------------------
# The command line's input and output
# ---------------------------------------------------------------------


def list_data(data: Path) -> List[Path]:
    """Return ``[data]`` for a csv, ``.npy`` or ``.npz`` file; refuse a directory and any other suffix by name."""
    if data.is_dir():
        raise ConfigError(f"Invalid value for 'data': {data} is a directory, not a file: {_DATA_HELP}")
    if data.suffix.lower() not in DATA_SUFFIXES:
        raise ConfigError(f"Invalid value for 'data': {data} has the suffix {data.suffix!r}: {_DATA_HELP}")
    return [data]


def read_data(file: Path) -> Any:
    """Return what ``predict`` takes of ``file``: the csv path as text, or the numpy array of the file."""
    import numpy as np

    if file.suffix.lower() == ".csv":
        return str(file)
    try:
        loaded = np.load(file, allow_pickle=False)
        if file.suffix.lower() == ".npy":
            return loaded
        with loaded:
            arrays = sorted(loaded.files)
            if len(arrays) != 1:
                raise ConfigError(f"Invalid value for 'data': {file} holds {len(arrays)} arrays {arrays}: give one array of windows")
            return loaded[arrays[0]]
    except (OSError, ValueError, EOFError) as exc:
        raise ConfigError(f"Invalid value for 'data': {file} is not a numpy file that can be read ({exc})") from exc


def write_predictions(predictor: ForecastPredictor, file: Path, source: Any, out: Path) -> None:
    """Write the forecast of ``source`` (read from ``file``) as one JSON file into the new path ``out``."""
    import numpy as np

    check_free(out, "out", "predict")
    answer = predictor.predict(source)
    for name in ("point", "quantiles"):
        if answer[name] is not None and not np.isfinite(answer[name]).all():
            raise ConfigError(f"The model forecast a non-finite {name} for {file}: nothing is written (JSON has no NaN)")
    record = predictor.record
    document = {
        "task": FORECASTING,
        "model": record["model"],
        "data": str(file),
        "lookback": record["lookback"],
        "horizon": record["horizon"],
        "channels": record["channels"],
        "quantile_levels": answer["quantile_levels"],
        "point": answer["point"].tolist(),
        "quantiles": None if answer["quantiles"] is None else answer["quantiles"].tolist(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------
# Export (until step 16)
# ---------------------------------------------------------------------


def export_pending(record: Mapping[str, Any], *_: Any) -> NoReturn:
    """The ``shape_of`` and ``check`` of the export row: every cell is refused by name, before anything is written."""
    raise ConfigError(f"Invalid value for 'format': {EXPORT_PENDING}")


def export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence logged beside an export (never reached while every cell is refused)."""
    return EXPORT_PENDING
