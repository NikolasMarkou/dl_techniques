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
        column (a blank line is a missing cell, refused with its row), a time column that is not strictly increasing.

``split_bounds(rows, select_fraction, test_fraction)``
    :return: the ``(lo, hi)`` row ranges of the train, select and test segments, consecutive and disjoint.

``check_segments(segments, rows, width, source)``
    :param segments: the ``(lo, hi)`` ranges of :func:`split_bounds`.
    :param rows: the rows of the whole series or csv; ``width``: ``lookback + horizon``; ``source``: how the message
        names the series (``"a synthetic series"``, ``"the csv"``).
    :raises ConfigError: for the first segment (train, select, test) shorter than ``width``; the one sentence of that
        refusal, said by the pre-claim check and by the loader alike.

``window_count(name, requested, default, batch_size)``
    :param name: ``"select"`` or ``"test"``; ``requested``: ``data.n_select`` or ``data.n_test`` (``None`` for the default);
        ``default``: the count the split has when none is asked.
    :return: the windows of the split, the request (or the default) rounded DOWN to a whole number of batches.
    :raises ConfigError: when that is fewer than one batch.

``csv_window_count(name, segment, requested, lookback, horizon, batch_size, fraction)``
    :param segment: the ``(lo, hi)`` rows of the select or test segment of a csv; windows are cut with non-overlapping targets
        (stride ``horizon``).
    :return: the windows of the split (``window_count`` of what the segment holds).
    :raises ConfigError: for a segment that holds fewer than one batch of such windows, and for a request of more than it holds.

``check_split_sizes(*, values, n_series, lookback, horizon, batch_size, select_fraction, test_fraction, n_select, n_test)``
    The arithmetic of the selection and test windows that needs no tensor, run before the run directory is claimed and by the
    loader (one implementation of every sentence). :param values: the csv's ``(rows, channels)`` array, or ``None`` for the
    synthetic bank; ``n_series``: the series of the bank (``None`` when ``data.patterns`` is null, because the generator
    that names them loads TensorFlow: the default count is then known only to the loader). :raises ConfigError: for a split
    of fewer than one batch, a request of more windows than a csv segment holds, and a csv whose select segment is constant in
    every value column (``check_select_moves``).

``check_select_moves(values, segment)``
    :raises ConfigError: when every value column is constant over the select ``segment``: every window is flat, so naive-last
        and seasonal-naive score exactly 0 and the bar (strictly below them) can be neither met nor judged.

``context_spread(context, scale_floor)`` / ``flat_contexts(contexts, scale_floor, *, resolution=False)`` / ``flat_windows(bank, positions, lookback, scale_floor, *, resolution=False)`` / ``segment_windows(n_series, segment, width, stride)``
    The ONE flatness rule, as numpy: a context is flat when its std about its first value is not above ``max(FLAT_ABSOLUTE, scale_floor * mean|x|)``.
    ``context_spread`` returns ``(std, flat)`` of ``(..., lookback, channels)`` contexts, each ``(..., 1, channels)``; ``flat_contexts`` is the bool
    ``(n,)`` of contexts flat in SOME channel (a constant column makes every window flat), the one answer ``predict``'s ``flat`` and the loader's
    counts give; ``resolution=True`` ignores a channel with std 0 (an exactly constant window is genuine, D-047). ``flat_windows`` is
    ``flat_contexts`` of the windows at ``positions`` (``(n, 2)`` ``(series, start)`` into ``bank`` ``(series, rows, channels)``);
    ``segment_windows`` is every window of every series whose targets are ``stride`` apart inside a ``segment``. The tf and keras-ops twins in
    ``train.common.forecast_data`` write the same comparison with ``FLAT_ABSOLUTE``, a test holds them equal at the threshold and one float32
    ulp either side, and the float32 export and the float64 rule can decide differently only within two float32 steps of the threshold.

``check_csv_level(values, names, *, lookback, horizon, select_fraction, test_fraction, scale_floor)``
    :param values: the csv's ``(rows, channels)`` array; ``names`` its value columns; ``lookback`` and ``horizon`` the RESOLVED window;
        the two fractions the split by time; ``scale_floor`` the flatness threshold of ``transform_args``.
    :raises ConfigError: for the first column whose nonzero max |x| is outside ``FLOAT32_RANGE`` (the exported function runs in
        float32), and for the first segment (train, select, test) of which more than ``FLAT_SHARE`` of the windows at stride
        ``horizon`` are RESOLUTION-flat (std above 0, not above the threshold: the level is too large for float32 to resolve the
        spread, so the run would mix units). The sentence names the segment, the count, the first window's rows and the one remedy
        (subtract the level); an exactly constant window is genuine and never refused (D-047). Called by the pre-claim check and the loader.

``refuse(message)``
    :raises ConfigError: always; the one spelling of "this configuration is refused" of the forecasting data.
"""

import hashlib
import io
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, NoReturn, Optional, Sequence, Tuple

from train.api.errors import ConfigError

if TYPE_CHECKING:  # annotations only; the runtime imports are inside the functions (module docstring)
    import numpy as np
    import pandas as pd


#: Fixed select and test windows per series of the synthetic bank, when ``n_select`` or ``n_test`` is not given.
SELECT_WINDOWS_PER_SERIES = 8
#: A csv is refused when more than this share of the windows of a segment have an unresolved spread (see ``check_csv_level``).
FLAT_SHARE = 0.1
#: A context whose std is not above this is flat whatever ``scale_floor`` says: the float32 export squares deviations and cannot resolve less (D-047).
FLAT_ABSOLUTE = 1e-15
#: The nonzero max |x| a csv column may have: the exported function squares deviations in float32 (1e-24 to 1e30 stays inside it).
FLOAT32_RANGE = (1e-12, 1e15)


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


def window_count(name: str, requested: Optional[int], default: int, batch_size: int) -> int:
    """The windows of a split: the request (or ``default``) rounded down to a multiple of the batch, at least one batch."""
    wanted = default if requested is None else requested
    count = wanted // batch_size * batch_size
    if count < 1:
        refuse(f"data.n_{name} = {wanted} is fewer than one batch ({batch_size}): a split needs a whole batch")
    return count


def csv_window_count(
        name: str, segment: Tuple[int, int], requested: Optional[int], lookback: int, horizon: int, batch_size: int,
        fraction: float,
) -> int:
    """The windows of a csv split: non-overlapping targets (stride ``horizon``), a whole number of batches."""
    lo, hi = segment
    available = (hi - lo - (lookback + horizon)) // horizon + 1
    if available < batch_size:
        needed = lookback + horizon * batch_size
        refuse(
            f"the {name} segment ({hi - lo} rows) holds {available} windows with non-overlapping targets, fewer than "
            f"one batch of {batch_size}: it needs at least {needed} rows (lookback + horizon * batch_size), so the "
            f"file needs about {math.ceil(needed / fraction)} rows at data.{name}_fraction = {fraction}")
    count = window_count(name, requested, available, batch_size)
    if count > available:
        refuse(f"data.n_{name} = {requested} asks for more windows than the {name} segment holds ({available})")
    return count


def check_select_moves(values: "np.ndarray", segment: Tuple[int, int]) -> None:
    """Refuse a csv whose select ``segment`` holds one value per value column (see the module contract)."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-042
    # A select segment that is constant in EVERY column makes every window flat: the context is flat (scale 1), the target equals the
    # last context value, naive-last and seasonal-naive score exactly 0 and the bar can be neither met nor judged. It was found only
    # in `build_data`, after the run directory was claimed (review NOTE 12); it is refused here, before the claim, and by the
    # loader through the same function. A segment that is flat only in SOME windows is not seen here (the baselines are
    # numbers of standardized windows, which need TensorFlow): `build_data` still refuses a zero baseline, after the claim.
    lo, hi = segment
    if bool((values[lo:hi] == values[lo]).all()):
        refuse(
            f"the select segment of the csv (rows [{lo}, {hi})) is constant in every value column: every selection window is flat, "
            "so naive-last and seasonal-naive score exactly 0 and the bar (strictly below them) can be neither met nor judged; "
            "give a csv whose rows after the train segment (data.select_fraction of them) vary")


def check_split_sizes(
        *, values: "Optional[np.ndarray]", n_series: Optional[int], lookback: int, horizon: int, batch_size: int,
        select_fraction: float, test_fraction: float, n_select: Optional[int], n_test: Optional[int],
) -> None:
    """Refuse the selection and test counts that cannot make one batch each, without a tensor (see the module contract)."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-042
    # Every refusal that needs only the config values and the csv's rows runs here, BEFORE the claim, and the loader calls the
    # same functions: `n_select` below one batch, a `data.patterns` subset whose 8 windows per series are fewer than one batch
    # (both used to end with "'out' is now used"), a csv segment with fewer than one batch of windows, and a constant select
    # segment. Do NOT copy the arithmetic into a second place: the sentences would drift. With `data.patterns` null the series
    # count needs the generator, which loads TensorFlow, so only an EXPLICIT n_select or n_test is checked then.
    if values is None:
        for name, requested in (("select", n_select), ("test", n_test)):
            if requested is not None or n_series is not None:
                default = SELECT_WINDOWS_PER_SERIES * (n_series or 0)
                window_count(name, requested, default, batch_size)
        return
    segments = split_bounds(len(values), select_fraction, test_fraction)
    check_select_moves(values, segments[1])
    for name, segment, requested, fraction in (
            ("select", segments[1], n_select, select_fraction), ("test", segments[2], n_test, test_fraction)):
        csv_window_count(name, segment, requested, lookback, horizon, batch_size, fraction)


def read_csv_series(
        path: Optional[str], value_cols: Optional[Sequence[str]], time_col: Optional[str],
) -> "Tuple[np.ndarray, List[str], str]":
    """``(values, names, sha256)``: a csv as ``(rows, channels)`` float64, its value columns and its byte hash."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-036
    # This module imports pandas and numpy only, and only here, inside the function. Do NOT import train.common,
    # tensorflow or keras here (and do NOT copy these checks into a second reader for the pre-claim step):
    # train.common loads TensorFlow, which must not be loaded before setup_gpu has run, and two readers drift
    # apart. Do NOT hoist the numpy and pandas imports to module scope either: the help path imports this module
    # (through forecast_args) and must stay free of them, which a module-scope import once broke (the help of every command then loaded numpy).
    import numpy as np
    import pandas as pd

    if path is None:
        refuse("data.path is required for data.format 'csv'")
    file = Path(path)
    if not file.is_file():
        refuse(f"data.path {path} does not exist")
    raw = file.read_bytes()
    try:
        # skip_blank_lines=False: a blank line is a MISSING row, refused below with its row number. The default drops it and the series
        # after the gap is shifted by one step with no word said (D-048); pandas' own to_csv never writes one.
        frame = pd.read_csv(io.BytesIO(raw), skip_blank_lines=False)
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


def context_spread(context: "np.ndarray", scale_floor: float) -> "Tuple[np.ndarray, np.ndarray]":
    """``(std, flat)`` of the contexts ``(..., lookback, channels)``: the std about the first value and ``std <= max(FLAT_ABSOLUTE, scale_floor * mean|x|)``."""
    import numpy as np

    context = np.asarray(context, np.float64)
    std = (context - context[..., :1, :]).std(axis=-2, keepdims=True)
    return std, std <= np.maximum(FLAT_ABSOLUTE, scale_floor * np.abs(context).mean(axis=-2, keepdims=True))


def flat_contexts(contexts: "np.ndarray", scale_floor: float, *, resolution: bool = False) -> "np.ndarray":
    """The bool ``(n,)`` of contexts ``(n, lookback, channels)`` flat in some channel (the ONE answer to "is this window flat"); ``resolution=True`` ignores a channel with std 0."""
    import numpy as np

    std, flat = context_spread(contexts, scale_floor)
    return (flat & (std > 0) if resolution else flat).reshape(len(contexts), -1).any(axis=1)


def flat_windows(bank: "np.ndarray", positions: "np.ndarray", lookback: int, scale_floor: float, *, resolution: bool = False) -> "np.ndarray":
    """The bool ``(n,)`` of the windows at ``positions`` whose context is flat (:func:`flat_contexts`), in chunks."""
    import numpy as np

    flat = np.zeros(len(positions), bool)
    for lo in range(0, len(positions), 4096):
        series, start = positions[lo:lo + 4096].T
        flat[lo:lo + 4096] = flat_contexts(bank[series[:, None], start[:, None] + np.arange(lookback)], scale_floor, resolution=resolution)
    return flat


def segment_windows(n_series: int, segment: Tuple[int, int], width: int, stride: int) -> "np.ndarray":
    """``(n, 2)`` ``(series, start)`` of every window of ``width`` rows, ``stride`` apart, inside ``segment`` of each series."""
    import numpy as np

    lo, hi = segment
    starts = lo + stride * np.arange(max(0, (hi - lo - width) // stride + 1))
    return np.stack([np.repeat(np.arange(n_series), len(starts)), np.tile(starts, n_series)], axis=1)


def check_csv_level(
        values: "np.ndarray", names: Sequence[str], *, lookback: int, horizon: int, select_fraction: float, test_fraction: float,
        scale_floor: float,
) -> None:
    """Refuse a column outside the float32 range, then a segment with too many resolution-flat windows (see the module contract)."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-045
    # The csv check counts the quantity the standardization switches on: the windows (stride = horizon, at the resolved lookback and
    # horizon) whose CONTEXT is flat. D-040 compared the whole column's std with 1e-6 of its level, which lets through a series
    # whose slow drift is large and whose windows are flat (the review's random walk plus a daily sine, shifted by 1e7: all 48
    # selection windows flat, trained in raw units, one val_mae of mixed units) and refuses a series whose windows are not. Do NOT
    # go back to the column statistic, and do NOT refuse on ONE flat window (10 percent is where a stretch becomes a share). The
    # float32 range is checked first: a std over 1e200 overflows float64 and 1e40 trains but its keras file returns NaN.
    # DECISION plan-2026-11-13T105600-3a8319ef/D-047
    # Only RESOLUTION-flat windows (std above 0 but not above the threshold: the level is too large for float32 to resolve the spread)
    # are refused; an EXACTLY constant one (std 0: a held rate, zeros, an outage) is genuine, so do NOT refuse it. Do NOT advise
    # differencing (a shifted series differenced is as shifted) or cutting rows (no gap handling), and do NOT blame `format`.
    import numpy as np

    low, high = FLOAT32_RANGE
    for name, peak in zip(names, np.abs(values).max(axis=0)):
        if peak != 0 and not low <= peak <= high:
            span = ", ".join(f"{x:.0e}".replace("e+", "e") for x in FLOAT32_RANGE)
            refuse(
                f"column {name!r} has max |x| {peak:.3g}, outside [{span}], the float32 range of the exported function (it "
                "squares deviations in float32): rescale the column to a typical size of about 1 before training")
    bank = values[None]
    for segment, bounds in zip(("train", "select", "test"), split_bounds(len(values), select_fraction, test_fraction)):
        positions = segment_windows(1, bounds, lookback + horizon, horizon)
        flat = flat_windows(bank, positions, lookback, scale_floor, resolution=True)
        if flat.sum() > FLAT_SHARE * flat.size:
            first = int(positions[flat.argmax(), 1])
            refuse(
                f"{int(flat.sum())} of {flat.size} windows of the {segment} segment have an unresolved spread (std above 0 but not above "
                f"max({FLAT_ABSOLUTE:g}, transform_args.scale_floor {scale_floor:g} * mean |x|); the first at rows {first} to "
                f"{first + lookback - 1}, counting from 0), more than {FLAT_SHARE:.0%}: the level of the column is too large against its "
                "spread for float32, so such a window is trained and scored in the series' raw units and the run would mix two units. "
                "Subtract the level (the series mean, or a trend) from the column first and add it back to the forecast")


def _check_time_column(column: "pd.Series", name: str) -> None:
    import pandas as pd

    if column.dtype == object:
        try:
            column = pd.to_datetime(column)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"data.time_col {name!r} cannot be parsed as timestamps: {exc}") from exc
    if column.isna().any() or not (column.is_monotonic_increasing and column.is_unique):
        refuse(f"data.time_col {name!r} must be strictly increasing with no empty cell: sort the file by time")
