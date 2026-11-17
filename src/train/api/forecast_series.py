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
        column (a blank line INSIDE the series is a missing cell, refused with its row; blank lines before the header and after the last row shift
        nothing and are ignored; a text cell is named with its row), a time column that is not strictly increasing.

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

``served_values(contexts)`` / ``served_flat(contexts, scale_floor)`` / ``context_spread(context, scale_floor)`` / ``flat_contexts(contexts, scale_floor, *, resolution=False)`` / ``flat_windows(bank, positions, lookback, scale_floor, *, resolution=False)`` / ``segment_windows(n_series, segment, width, stride)``
    The ONE flatness rule, as numpy: a context is flat when its std about its first value is not above ``max(FLOAT32_TINY, scale_floor * mean|x|)``,
    ``FLOAT32_TINY`` the smallest normal float32 number (D-049: there is no larger absolute term; the std is ``span * std(deviation / span)``, so no
    raw deviation is squared and float32 resolves it down to that number). ``context_spread`` returns ``(std, flat)`` of ``(..., lookback, channels)``
    contexts, each ``(..., 1, channels)``; ``served_flat`` is its ``flat`` on the FLOAT32 value of the context (``served_values``); ``flat_contexts``
    is the bool ``(n,)`` of contexts flat in SOME channel (a constant column makes every window flat), the one answer ``predict``'s ``flat`` and the
    loader's counts give, decided on the FLOAT32 value of the context, as the exported file is handed raw float32 windows (D-048) and as the training
    function ``standardize_tf`` decides (D-049); ``resolution=True`` is the csv check's question about the data as written, in float64, and ignores a
    channel with std 0 (an exactly constant window is genuine, D-047). ``flat_windows`` is ``flat_contexts`` of the windows at ``positions``
    (``(n, 2)`` ``(series, start)`` into ``bank`` ``(series, rows, channels)``); ``segment_windows`` is every window of every series whose targets
    are ``stride`` apart inside a ``segment``. The tf and keras-ops twins in ``train.common.forecast_data`` write the same comparison with
    ``FLOAT32_TINY``, a test holds them equal at the threshold and one float32 ulp either side, and the float32 export and ``flat_contexts`` (same
    values, float32 against float64 arithmetic) disagree on at most 1 window of 4,000 within 3e-5 of the threshold (none beyond 1e-4). A float32 value
    below ``FLOAT32_TINY`` (a denormal) reads 0 in ``served_values``, as TensorFlow's cast gives it and its arithmetic reads it (D-050): training, scoring,
    the count, the flag and ``predict``'s scale decide on that, and so do the keras, saved_model and tflite files; onnxruntime keeps a denormal and decides
    differently within 1.3 times ``FLOAT32_TINY`` of it (measured, D-050).

``check_csv_level(values, names, *, lookback, horizon, select_fraction, test_fraction, scale_floor)``
    :param values: the csv's ``(rows, channels)`` array; ``names`` its value columns; ``lookback`` and ``horizon`` the RESOLVED window;
        the two fractions the split by time; ``scale_floor`` the flatness threshold of ``transform_args``.
    :raises ConfigError: for the first column whose nonzero max |x| is outside ``FLOAT32_RANGE`` (the range this task is held to; D-049
        measured the files finite and within 3e-6 of a window std of ``predict`` far outside it), and for the first segment (train, select, test) of which
        more than ``FLAT_SHARE`` of the windows at stride ``horizon`` are RESOLUTION-flat (std above 0, not above the threshold: the level is too
        large for float32 to resolve the spread, so the run would mix units). The sentence names the segment, the count, the first window's rows and the
        remedies, chosen by the term the first such window FAILS: the relative one (std at most ``scale_floor * mean|x|``) says to subtract the level or
        take a first or seasonal difference, and only where just ``FLOAT32_TINY`` binds (a window spread under 1.18e-38: a column too small) it says
        to multiply the column by a power of ten or set the stretch to exactly 0, and that ``scale_floor`` cannot lower that term;
        an exactly constant window is genuine and never refused (D-047). Called by the pre-claim check and the loader.

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
# DECISION plan-2026-11-13T105600-3a8319ef/D-049
# There is NO absolute spread guard above this one number. D-047 (1e-15) and D-048 (1e-18) kept an absolute floor on the std because the exported
# function squared RAW deviations and TensorFlow's float32 CPU kernels flush a square below 1.2e-38 to 0; that floor was the one place where the
# training function (float64), `predict`, the loader's count and the three float32 files could disagree, and passes 3, 4 and 5 each found a new
# disagreement next to it. The spread is now computed WITHOUT squaring a raw deviation (`span * std(deviation / span)`, `span` the largest
# deviation about the first value: the same number in exact arithmetic), so float32 resolves it down to its smallest NORMAL number, and the
# absolute term is the representability limit and nothing else: a window whose spread is not above 1.18e-38 is flat at any `scale_floor`,
# because below it the runtimes disagree on what a denormal is (TensorFlow flushes it to 0, numpy and onnxruntime keep it): without this term
# 200 of 200 windows at a spread of 1e-40 were flat in TensorFlow and standardized by numpy (scratch_review6/r1_proto.log). Do NOT put a larger
# absolute number back "for margin" (it flattens windows float32 resolves, the pass-4 W2 defect) and do NOT square raw deviations in a
# float32 twin (the pass-5 W1 defect: a window near 1e-18 was flat in the file and standardized by `predict`).
#: The smallest normal float32 number (``2 ** -126``): the one absolute term of the flat rule. A context whose spread is not above it is flat
#: whatever ``scale_floor`` says (D-049).
FLOAT32_TINY = 1.1754943508222875e-38
#: The magnitude below which TensorFlow's float64 to float32 cast gives 0: float32's smallest normal number less half of one denormal step
#: (``FLOAT32_TINY * (1 - 2**-25)``, exact). ``served_values`` flushes below it (D-050).
FLOAT32_FLUSH = FLOAT32_TINY * (1.0 - 2.0 ** -25)
#: The nonzero max |x| a csv column may have: the range this task trains and serves in. It is no longer a float32 limit (D-049: with the spread
#: computed without squaring raw deviations, the files were measured finite and within 3e-6 of a window std of ``predict`` at a window spread of
#: 1e-30 and of 1e35 at lookback 32); it is kept because a flat window is forecast in the series' raw units, which means nothing for a unit far from 1.
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
    # DECISION plan-2026-11-13T105600-3a8319ef/D-049
    # A blank line INSIDE the series is a missing row (pandas' default drops it and the series after the gap is shifted by one step with no word said,
    # D-048); a blank line before the header or after the last row shifts nothing, so those are dropped here and the rest is read with
    # `skip_blank_lines=False`. Do NOT hand pandas the whole file with `skip_blank_lines=False`: the first blank line is then the header (review pass 5
    # NOTE 1: "has no value column") and a final newline pair is a NaN row. The sha256 is of the bytes as written.
    body = re.sub(rb"\A(\xef\xbb\xbf)?(?:[ \t]*\r?\n)+", rb"\1", raw).rstrip()   # a UTF-8 byte-order mark stays (pandas decodes it) and does not hide the blank lines (D-050)
    try:
        frame = pd.read_csv(io.BytesIO(body), skip_blank_lines=False)
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
            cells = frame[name]
            odd = cells.notna() & pd.to_numeric(cells, errors="coerce").isna()   # the cells that are text and not a number
            where = ""
            if odd.any():
                row = int(np.argmax(odd.to_numpy()))
                cell = cells.iloc[row]
                where = (f": the first cell that is not a number is {cell!r} at row {row}"
                         + (" (a blank cell: spaces only)" if str(cell).strip() == "" else ""))
            refuse(f"column {name!r} is not numeric (dtype {cells.dtype}){where}; one text or decimal-comma "
                   "cell makes a whole column non-numeric; name the numeric columns in data.value_cols")
    values = frame[names].to_numpy(np.float64)
    bad = ~np.isfinite(values)
    if bad.any():
        row, col = (int(v[0]) for v in np.nonzero(bad))
        refuse(f"column {names[col]!r} holds {int(bad[:, col].sum())} non-finite cell(s), the first at row {row} "
               "(a blank line, an empty cell, NaN or infinity)")
    if time_col is not None:
        _check_time_column(frame[time_col], time_col)
    return values, names, hashlib.sha256(raw).hexdigest()


def context_spread(context: "np.ndarray", scale_floor: float) -> "Tuple[np.ndarray, np.ndarray]":
    """``(std, flat)`` of the contexts ``(..., lookback, channels)``: the std about the first value, ``span * std(deviation / span)``, and
    ``std <= max(FLOAT32_TINY, scale_floor * mean|x|)``; ``span`` is the largest deviation, so no raw deviation is squared (D-049)."""
    import numpy as np

    context = np.asarray(context, np.float64)
    shifted = context - context[..., :1, :]
    span = np.abs(shifted).max(axis=-2, keepdims=True)
    std = span * (shifted / np.maximum(span, FLOAT32_TINY)).std(axis=-2, keepdims=True)
    return std, std <= np.maximum(FLOAT32_TINY, scale_floor * np.abs(context).mean(axis=-2, keepdims=True))


def served_values(contexts: "np.ndarray") -> "np.ndarray":
    """The float32 value of every cell of ``contexts``, as float64: what the exported file is handed (raw float32 windows, D-048).

    A value that float32 would hold as a denormal (below its smallest normal number) is 0, with its sign, the way TensorFlow's cast does:
    :func:`context_stats_np` and ``standardize_tf`` decide on THIS value, so training, scoring, the count, the flag and ``predict``'s scale
    read one window (D-050).
    """
    # DECISION plan-2026-11-13T105600-3a8319ef/D-050
    # TensorFlow's float64 to float32 cast (which `standardize_tf` decides on) FLUSHES a denormal to 0, and numpy's cast keeps it: at a context std
    # of 1.0001 times 1.18e-38, 470 to 841 of 1,000 windows were scored flat by training and counted standardized by the summary (review pass 6 W1;
    # on an accepted csv the summary said 0 of 72 select windows flat and the scoring treated 38). The flush is the same here. Do NOT cast with numpy
    # alone and do NOT flush below `FLOAT32_TINY` itself: the hardware decides the underflow on the value rounded to 24 bits with an unbounded exponent,
    # so a float64 value between `FLOAT32_TINY * (1 - 2**-25)` and `FLOAT32_TINY` is KEPT (as the smallest normal number) and everything under
    # `FLOAT32_TINY * (1 - 2**-25)` is 0, signed (swept at ulp scale against `tf.cast`, scratch_review6b/t1_cast_sweep.log; a test holds it bit for bit).
    import numpy as np

    values = np.asarray(contexts, np.float64)
    values = np.where(np.abs(values) < FLOAT32_FLUSH, np.copysign(0.0, values), values)
    return values.astype(np.float32).astype(np.float64)


def served_flat(contexts: "np.ndarray", scale_floor: float) -> "np.ndarray":
    """The flat decision per channel, bool ``(..., 1, channels)``, taken on the float32 value of ``contexts`` (the one decision, D-049)."""
    return context_spread(served_values(contexts), scale_floor)[1]


def flat_contexts(contexts: "np.ndarray", scale_floor: float, *, resolution: bool = False) -> "np.ndarray":
    """The bool ``(n,)`` of contexts ``(n, lookback, channels)`` flat in some channel (the ONE answer to "is this window flat").

    Decided on the FLOAT32 value of the context (:func:`served_flat`; a denormal reads 0 there, D-050), as the exported file decides it, and as ``standardize_tf`` (training
    and scoring) and ``predict`` decide it. ``resolution=True`` is the csv check's question instead, about the data as written: float64,
    ignoring a channel with std 0.
    """
    # DECISION plan-2026-11-13T105600-3a8319ef/D-048
    # The file is handed raw float32 windows and `predict` float64 ones: the float64 rule and the float32 function disagreed on 228 of 400
    # windows at 1e-5 of the threshold and on 142 at -0.1 percent (a band of about 1 percent, 80,000 float32 steps; the "two float32 steps"
    # of D-047 was measured on windows built float32-exact). The decision is taken here on the float32 value, once, and `predict` (the flag and
    # the scale) both go through it, so the flag, the scale and the file read the same numbers. Do NOT decide on the float64 window again, and
    # do NOT cast for `resolution=True`: at 1e12 a float32 cell is 65536 apart, a spread of 5.6 rounds to one value (std 0, "exactly
    # constant") and the csv check, which is about the spread the file cannot hold, would accept the series it exists to refuse.
    # DECISION plan-2026-11-13T105600-3a8319ef/D-049
    # D-048 left `standardize_tf` (training and the scoring of `val_mae`, the baselines and the bar) on the float64 window, so the summary's
    # `flat_windows` described windows that the scoring did not treat as flat (3 against 0 on an accepted csv, review pass 5 W2). It decides on
    # the float32 value too now: the count, the flag, the scale and the scoring are ONE decision. Do NOT give the training function its own.
    import numpy as np

    if resolution:
        std, flat = context_spread(contexts, scale_floor)
        flat = flat & (std > 0)
    else:
        flat = served_flat(contexts, scale_floor)
    return flat.reshape(len(contexts), -1).any(axis=1)


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
    # are refused; an EXACTLY constant one (std 0: a held rate, zeros, an outage) is genuine, so do NOT refuse it. Do NOT cut rows
    # (no gap handling) and do NOT blame `format`.
    # DECISION plan-2026-11-13T105600-3a8319ef/D-048
    # D-047 said differencing does not help ("the difference of a shifted series is as shifted") and that is FALSE: a first or a 24-lag difference
    # removes a constant level exactly (measured: series shifted by 1e7, 1e9, 1e12 are refused, their differences are accepted and equal the unshifted one
    # to the float64 ulp of the shift, scratch_review4/p5_difference.log). Name it, with its price: the model then forecasts differences, not values. Do NOT
    # call the level "too large" where only the float32 floor binds (a window spread at or below FLOAT32_TINY, a column that is too SMALL): there the
    # remedy is a power of ten or an exact 0, and `scale_floor` cannot lower that term.
    # DECISION plan-2026-11-13T105600-3a8319ef/D-049
    # The sentence follows the term the first resolution-flat window FAILS, not the larger of the two terms: a window at 0.9e-12 with a std of 5e-19
    # fails the RELATIVE term (5.6e-7 of its level) and was told to multiply by a power of ten, which is refused again at every power because scaling
    # keeps std over level (review pass 5 W3; the level remedy, subtract the mean and rescale, is accepted). Do NOT choose the branch by which term is
    # larger. The range check above is the range this task trains in, no longer a float32 limit.
    import numpy as np

    low, high = FLOAT32_RANGE
    for name, peak in zip(names, np.abs(values).max(axis=0)):
        if peak != 0 and not low <= peak <= high:
            span = ", ".join(f"{x:.0e}".replace("e+", "e") for x in FLOAT32_RANGE)
            refuse(
                f"column {name!r} has max |x| {peak:.3g}, outside [{span}], the range this task trains and serves in (the exported files "
                "read float32 windows, and a flat window is forecast in the series' raw units, which means nothing for a unit far from "
                "1): rescale the column to a typical size of about 1 before training")
    bank = values[None]
    for segment, bounds in zip(("train", "select", "test"), split_bounds(len(values), select_fraction, test_fraction)):
        positions = segment_windows(1, bounds, lookback + horizon, horizon)
        flat = flat_windows(bank, positions, lookback, scale_floor, resolution=True)
        if flat.sum() > FLAT_SHARE * flat.size:
            first = int(positions[flat.argmax(), 1])
            window = bank[0, first:first + lookback]
            std, hidden = context_spread(window, scale_floor)
            hidden = (hidden & (std > 0))[0]                       # the resolution-flat channels of the first such window
            # the sentence follows the term the window FAILS: the relative one (std at most scale_floor * mean |x|: the level hides the spread,
            # and multiplying by a power of ten keeps std over level) or only the float32 floor (std above the relative term, not above FLOAT32_TINY)
            relative = bool((std[0][hidden] <= scale_floor * np.abs(window).mean(axis=0)[hidden]).any())
            refuse(
                f"{int(flat.sum())} of {flat.size} windows of the {segment} segment have an unresolved spread (std above 0 but not above "
                f"max({FLOAT32_TINY:.3g}, transform_args.scale_floor {scale_floor:g} * mean |x|); the first at rows {first} to "
                f"{first + lookback - 1}, counting from 0), more than {FLAT_SHARE:.0%}: "
                + (
                    "the level of the column is too large against its spread for float32, so such a window is trained and scored in the "
                    "series' raw units and the run would mix two units. Subtract the level (the series mean, or a trend) from the column "
                    "first and add it back to the forecast, or give the model the first or the seasonal difference of the column (the "
                    "forecast is then of differences, not values); rescale the result to about 1 if its peak is then below 1e-12"
                    if relative else
                    f"the values of the column are too small for float32, whose smallest normal number {FLOAT32_TINY:.3g} is not below the "
                    "spread of such a window, so it is trained and scored in the series' raw units and the run would mix two units. Multiply "
                    "the column by a power of ten (change its unit; its peak must stay at most 1e15) so that a window spread is well above it, "
                    "or set the stretch that small to exactly 0 (an exactly constant stretch is never refused); "
                    "transform_args.scale_floor cannot lower this term"))


def _check_time_column(column: "pd.Series", name: str) -> None:
    import pandas as pd

    if column.dtype == object:
        try:
            column = pd.to_datetime(column)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"data.time_col {name!r} cannot be parsed as timestamps: {exc}") from exc
    if column.isna().any() or not (column.is_monotonic_increasing and column.is_unique):
        refuse(f"data.time_col {name!r} must be strictly increasing with no empty cell: sort the file by time")
