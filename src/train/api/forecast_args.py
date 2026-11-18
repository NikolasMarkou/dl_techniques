"""The forecasting arguments of ``python -m train``: the data group, the transform, metric and loss groups.

The ``time_series_forecasting`` task trains a forecaster on windows of a series: a synthetic bank (the generator's
patterns, ``realizations`` draws of each) or one user csv. Every series is split BY TIME, the first slice trains, the
next selects the best epoch and the last is scored once after ``fit`` (``train.common.forecast_data``). ``TimeSeriesData``
names the source, the window and how the three segments are cut; ``ForecastTransformArgs``, ``ForecastMetricArgs`` and
``ForecastLossArgs`` are the groups of the task config (``train.api.tasks.time_series_forecasting``) that differ from
the shared ones: there is no image and no class, so ``TransformArgs`` and ``MetricArgs`` cannot serve. Every option that
cannot apply is REFUSED, never ignored.

Interface contract
------------------
``TASK`` / ``SYNTHETIC`` / ``BUILTIN`` / ``CSV`` / ``DATA_FORMATS``
    ``"time_series_forecasting"``, the one dataset name (the generator bank; it is the recipe key the family recipes
    overlay), the two sources ``builtin`` and ``csv`` and the tuple of both (the ``Literal`` of ``data.format``); the loader
    ``train.common.forecast_data`` imports them, it does not restate them.

``TimeSeriesData``
    Frozen :class:`~train.api.config.StrictConfig`. Every field maps to the keyword of the same name of
    ``train.common.forecast_data.load_forecast_batches`` (a test holds the two equal); ``format`` ``builtin`` takes
    ``dataset`` (only ``synthetic``), ``realizations``, ``series_length`` and ``patterns``; ``csv``
    takes ``path``, ``value_cols`` and ``time_col``; a key of the OTHER source that is not at its default is refused.
    ``lookback`` and ``horizon`` are ``"auto"`` until the family's recipe fills them (``resolve_auto``): the recipe
    is the one place that holds those two numbers. ``problems(prefix)`` names every option out of range.

``check_forecast_data(data)``
    :param data: a config whose ``lookback`` and ``horizon`` are resolved (``lookback`` at least 2, D-040).
    :raises ConfigError: every problem of ``data.problems``, a csv that does not exist, lacks a named column, holds a
        non-numeric or non-finite value column or an unsorted time column, and a segment (train, select or test) of the
        series or of the csv shorter than ``lookback + horizon``. pandas and numpy only (imported when a csv is read, not
        when this module loads): no TensorFlow is imported, so it runs before ``setup_gpu`` and before the run
        directory is claimed.

``forecast_facts(data)``
    :return: ``DataFacts(SYNTHETIC, None, None)``: no class count and no image size; the dataset name is the recipe key
        for both sources.

``load_forecast_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one (a relative
        ``path`` is anchored to the file).
    :return: the validated config.
    :raises ConfigError: for a file that cannot be read or a value that is not a mapping, or any invalid key (every
        problem at once; a file's lines name the file).

``ForecastTransformArgs``
    ``scale_floor``: the flatness threshold of a context, relative to its mean absolute value (``train.common.forecast_data``).
    A context whose std is not above it is flat and keeps the series' own unit; the default is 1e-6.

``ForecastMetricArgs``
    ``watch_metric`` (:data:`WATCH_METRIC`, the only value accepted; auto is the recipe's) and ``watch_mode``
    (:data:`WATCH_MODE`, likewise): the bar is a statement about the MINIMUM ``val_mae`` of the selection window, so the
    saved checkpoint must be that epoch. The recipes import the two constants.

``ForecastLossArgs``
    ``name``: ``mae``, ``mse``, ``mase`` or ``null`` (the family's own loss); :data:`LOSS_NAMES` is the one tuple, the
    ``Literal`` of the field and what ``train.common.forecast_run`` names in its refusal.

Where the numbers live
----------------------
The defaults of this module are the DATA defaults measured in the plan (four realizations of 10,000 points, noise 0.1,
15 percent select and test, scale floor 1e-6). The epochs, the batch size, ``lookback``, ``horizon`` and the optimizer
belong to the family recipes (``train.api.forecast_models``), never to a second copy here.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (``forecast_series`` reads a csv with pandas and numpy,
imported inside its functions, so loading this module loads neither); nothing in this module imports ``train.common`` or
keras, so ``python -m train time_series_forecasting help`` stays free of TensorFlow, numpy and pandas, and
``check_forecast_data`` stays free of TensorFlow. The loaders that need TensorFlow live in ``train.common.forecast_data``,
which the task imports.
"""

import dataclasses
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from .args import AutoContext, _is_auto, _recipe_values, _watch_values, _with_values
from .config import AUTO, Auto, StrictConfig, join_path
from .data_args import DataFacts, load_data_group, read_data_file
from .errors import ConfigError, ConfigValidationError
from .forecast_series import check_segments, read_csv_series, refuse, split_bounds

# ---------------------------------------------------------------------

TASK = "time_series_forecasting"
SYNTHETIC = "synthetic"
BUILTIN, CSV = "builtin", "csv"
DATA_FORMATS: Tuple[str, ...] = (BUILTIN, CSV)

#: The keys that name a csv; a relative ``path`` in a data file is anchored to the file.
CSV_KEYS: Tuple[str, ...] = ("path", "value_cols", "time_col")

#: The keys of the synthetic bank; held at their defaults for a csv.
SYNTHETIC_KEYS: Tuple[str, ...] = ("realizations", "series_length", "patterns")

#: The smallest lookback of ANY family: one context value has no spread to standardize by (D-040). A family may need more
#: (``forecast_models.FamilyRow.min_lookback``).
MIN_LOOKBACK = 2

#: The one metric and direction the bar can describe; the family recipes carry the same metric (a test holds them equal).
WATCH_METRIC = "val_mae"
WATCH_MODE = "min"

#: The losses of ``loss_args.name``; ``train.common.forecast_run`` imports the tuple.
LOSS_NAMES: Tuple[str, ...] = ("mae", "mse", "mase")

_Format = Literal[DATA_FORMATS]  # type: ignore[valid-type]
_LossName = Literal[LOSS_NAMES]  # type: ignore[valid-type]


@dataclasses.dataclass(frozen=True)
class TimeSeriesData(StrictConfig):
    """The series of a forecasting run: the source, the window and the split by time; see the module contract."""

    format: _Format = dataclasses.field(
        default=BUILTIN,
        metadata={"help": "where the series come from: builtin (the synthetic bank of the generator's patterns) or csv "
                          "(one user file, data.path); the group data is required, so give one of its keys: data.format=builtin "
                          "alone is the synthetic bank"},
    )
    dataset: str = dataclasses.field(
        default=SYNTHETIC,
        metadata={"help": "synthetic, the only value: the generator bank (builtin); a csv is named by data.path"},
    )
    path: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "csv only: the file, one row per time step; the rows must already be in time order"},
    )
    value_cols: Optional[List[str]] = dataclasses.field(
        default=None,
        metadata={"help": "csv only: the value columns, every one numeric and finite; they become the channels; null is "
                          "every column except the time column and an unnamed index column"},
    )
    time_col: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "csv only: the timestamp column, which must be strictly increasing; null means row order is time"},
    )
    lookback: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={"help": "points of context the model sees (at least 2; the family's own minimum is checked before the run); auto is the family recipe's"},
    )
    horizon: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={"help": "points ahead the model forecasts (at least 1); auto is the family recipe's"},
    )
    # DECISION plan-2026-11-13T105600-3a8319ef/D-030
    # Four realizations of every pattern. Measured: the standard error of the baseline mean on the selection
    # windows is 0.0179, 0.0107 and 0.0086 for 2, 4 and 8, against a margin of about 0.0084 at 1 percent, so 2 is a noisy
    # bar; 8 doubles the bank build (5.4 s) and held the all-zero-context series of D-035 at seed 42. Memorization did
    # not move with it (no train-versus-select gap at any of the three). Do NOT raise it to buy a quieter bar without
    # measuring the build time, and do NOT lower it to 2 (the bar then flips on seed noise).
    realizations: int = dataclasses.field(
        default=4,
        metadata={"help": "builtin only: independent draws of every generator pattern in the bank; more make the "
                          "selection windows a steadier bar and the bank slower to build (about 0.7 s per draw of all patterns)"},
    )
    series_length: int = dataclasses.field(
        default=10_000, metadata={"help": "builtin only: points per synthetic series (the split cuts it by time)"}
    )
    patterns: Optional[List[str]] = dataclasses.field(
        default=None,
        metadata={"help": "builtin only: the generator pattern names to use; null is every pattern; an unknown name is refused"},
    )
    select_fraction: float = dataclasses.field(
        default=0.15,
        metadata={"help": "share of every series, after the train slice, that selects the best epoch, in (0, 1); with "
                          "test_fraction it must leave a train slice"},
    )
    test_fraction: float = dataclasses.field(
        default=0.15, metadata={"help": "share of every series, the last slice, scored once after fit, in (0, 1)"}
    )
    n_select: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "fixed selection windows, rounded down to a whole number of batches; null is 8 per series "
                          "(builtin) or every non-overlapping window of the segment (csv)"},
    )
    n_test: Optional[int] = dataclasses.field(
        default=None, metadata={"help": "fixed test windows, rounded down to a whole number of batches; null as n_select"}
    )
    steps_per_epoch: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "batches in an epoch, at least 1; null is the number of train windows over the batch size, at "
                          "most 500 (the legacy epoch)"},
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "cap on the train windows an epoch is sized from when steps_per_epoch is null; the selection "
                          "and test windows are never capped; null is no cap"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range and every key that cannot apply to this source, at its dotted path."""
        lines: List[str] = []

        def bad(name: str, why: str) -> None:
            lines.append(f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(self, name)!r}")

        if self.dataset != SYNTHETIC:
            bad("dataset", f"expected {SYNTHETIC!r}, the only dataset (a csv is named by data.path)")
        # DECISION plan-2026-11-13T105600-3a8319ef/D-040
        # A lookback of 1 is refused by name, before the claim: one context value has no spread, so every window would be "flat"
        # and every number of the run would be in the series' own unit (measured: a lookback-1 run scored 1758.20 against 1758.20
        # of naive-last, exit 1). Do NOT lower this to 1 "for a persistence model": naive-last is a baseline the run reports.
        if not _is_auto(self.lookback) and self.lookback < MIN_LOOKBACK:
            bad("lookback", f"expected an int of at least {MIN_LOOKBACK} (one value has no spread to standardize by) or auto")
        if not _is_auto(self.horizon) and self.horizon < 1:
            bad("horizon", "expected an int of at least 1 or auto")
        for name in ("realizations", "series_length", "n_select", "n_test", "steps_per_epoch", "max_samples"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                bad(name, "expected an int of at least 1 (or null where null is allowed)")
        for name in ("select_fraction", "test_fraction"):
            if not 0.0 < getattr(self, name) < 1.0:
                bad(name, "expected a fraction strictly between 0 and 1")
        total = self.select_fraction + self.test_fraction
        if 0.0 < self.select_fraction < 1.0 and 0.0 < self.test_fraction < 1.0 and total >= 1.0:
            lines.append(
                f"Invalid value for {join_path(prefix, 'select_fraction')!r}: with data.test_fraction it must leave a train "
                f"segment, their sum is {total!r} and must be below 1"
            )
        for name in ("patterns", "value_cols"):
            if getattr(self, name) is not None and not getattr(self, name):
                bad(name, "expected a non-empty list or null")
        if self.format == CSV:
            for field in dataclasses.fields(self):
                if field.name in SYNTHETIC_KEYS and getattr(self, field.name) != field.default:
                    bad(field.name, "a key of the synthetic bank, which data.format 'csv' does not read; leave it at its default")
        else:
            for name in CSV_KEYS:
                if getattr(self, name) is not None:
                    bad(name, "a csv key, which data.format 'builtin' does not read; set data.format to 'csv' or remove it")
        return lines

    def resolve_auto(self, context: AutoContext) -> "TimeSeriesData":
        """Fill ``lookback`` and ``horizon`` from the family's recipe, then run the checks that need no TensorFlow."""
        found = _recipe_values(self, context, "data")
        for name in ("lookback", "horizon"):
            if _is_auto(getattr(self, name)) and name not in found:
                raise ConfigError(
                    f"Invalid value for 'data.{name}': the {context.family!r} recipe has no {name} for {context.dataset!r}; "
                    f"set data.{name}"
                )
        resolved = _with_values(self, "data", found)
        check_forecast_data(resolved)
        return resolved


# ---------------------------------------------------------------------
# What is known before the data is loaded
# ---------------------------------------------------------------------


def check_forecast_data(data: TimeSeriesData) -> None:
    """Refuse a data config that cannot make one window, without TensorFlow; see the module contract."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-006
    # DECISION plan-2026-11-13T105600-3a8319ef/D-036
    # Every check that needs no tensor runs BEFORE the run directory is claimed and the GPU is set up: the csv is read
    # with train.api.forecast_series (pandas and numpy only), the same reader the loader uses. Do NOT import
    # train.common or keras here (train.common loads TensorFlow, which must not be initialised before setup_gpu), and
    # do NOT write a second csv reader: two readers drift apart.
    lines = data.problems("data")
    if lines:
        raise ConfigValidationError(lines)
    for name in ("lookback", "horizon"):
        if _is_auto(getattr(data, name)):
            refuse(f"data.{name} is 'auto': resolve it from the family's recipe before the data is checked")
    width = data.lookback + data.horizon
    if data.format == "csv":
        values, _, _ = read_csv_series(data.path, data.value_cols, data.time_col)
        rows, source = len(values), "the csv"
    else:
        rows, source = data.series_length, "a synthetic series"
    check_segments(split_bounds(rows, data.select_fraction, data.test_fraction), rows, width, source)


def forecast_facts(data: TimeSeriesData) -> DataFacts:
    """Return the facts of a data config before it is loaded; see the module contract."""
    return DataFacts(dataset=SYNTHETIC, num_classes=None, image_size=None)


# ---------------------------------------------------------------------
# The config from a mapping or a file
# ---------------------------------------------------------------------


def read_forecast_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``, a relative csv ``path`` anchored to the file."""
    return read_data_file(path, ("path",))


def load_forecast_data_config(data: Any) -> TimeSeriesData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    return load_data_group(TimeSeriesData, data, read_forecast_data_file)


# ---------------------------------------------------------------------
# The other groups of the task config
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ForecastTransformArgs(StrictConfig):
    """The context standardization of the forecasting data; read by the forecasting task."""

    scale_floor: float = dataclasses.field(
        default=1e-6,
        metadata={
            "help": "flatness threshold, relative to the mean absolute value of a window's context: a context whose std is not "
            "above scale_floor * mean |context| is flat and is not divided (its scale is 1, the series' own unit); any other "
            "context is divided by its std. A csv is refused when more than 10 percent of the windows of a segment have a spread "
            "above 0 that is not above this threshold (float32 cannot resolve it: remove the level, or difference the column). The "
            "threshold never falls below float32's smallest normal number, 1.18e-38, which this field cannot lower (a column that small: "
            "multiply it by a power of ten, or set the stretch to 0), and at 0 a series whose level hides its spread is admitted, but its "
            "exported files carry the float32 rounding of the level. The default is about eight float32 steps of relative resolution; "
            "at least 0"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a negative ``scale_floor``."""
        if self.scale_floor < 0:
            return [f"Invalid value for {join_path(prefix, 'scale_floor')!r}: expected a number of at least 0, got {self.scale_floor!r}"]
        return []


@dataclasses.dataclass(frozen=True)
class ForecastMetricArgs(StrictConfig):
    """The metric that selects the checkpoint; read by the forecasting task."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "metric that picks the best epoch: val_mae on the fixed selection windows, the only value accepted "
            "(the success bar is about the minimum val_mae, so the saved checkpoint must be that epoch); auto is val_mae"
        },
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "min, the only value accepted (val_mae is an error: the best epoch is its minimum); auto is min"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a ``watch_metric`` other than ``val_mae`` and a ``watch_mode`` other than ``min``."""
        lines: List[str] = []
        for name, only in (("watch_metric", WATCH_METRIC), ("watch_mode", WATCH_MODE)):
            value = getattr(self, name)
            if not _is_auto(value) and value != only:
                lines.append(
                    f"Invalid value for {join_path(prefix, name)!r}: expected auto or {only!r}, got {value!r}; the success "
                    f"bar is a statement about the minimum val_mae of the selection window, so the saved best checkpoint "
                    f"must be that epoch ({WATCH_METRIC!r} with mode {WATCH_MODE!r}); another metric or the other direction "
                    "would save weights the verdict does not describe"
                )
        return lines

    def resolve_auto(self, context: AutoContext) -> "ForecastMetricArgs":
        """Fill ``watch_metric`` from the recipe and ``watch_mode`` from its name."""
        return _with_values(self, "metric_args", _watch_values(self, context, "metric_args"))


@dataclasses.dataclass(frozen=True)
class ForecastLossArgs(StrictConfig):
    """The loss of a forecasting run; read by the forecasting task."""

    name: Optional[_LossName] = dataclasses.field(
        default=None,
        metadata={
            "help": "mae, mse or mase (the batch-wise approximation MASELoss) for a point head; null is the family's own "
            "loss (MAE for a point head, the pinball loss for a quantile head, the likelihood for mdn); a name for a "
            "quantile head, timestral or mdn is refused"
        },
    )
