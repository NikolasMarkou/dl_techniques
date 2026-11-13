"""Data for the ``time_series_forecasting`` task: a stateless series bank, fixed select and test windows.

The legacy forecasting trainers draw every window from one shared stateful ``RandomState`` (a stream is not a
function of its index and a resumed ``fit`` replays it), standardize a window with its HORIZON included (a
leak) and validate on a fresh noise draw. This module builds the same three pipelines as pure functions: a bank
of series (the synthetic generator, one fresh instance per ``(pattern, realization)``, or one user csv) is split
by TIME, and a window is a function of its key, never of a hidden state.

This module imports TensorFlow; ``train.api`` imports it inside functions only. The csv reader, the column checks
and ``split_bounds`` live in ``train.api.forecast_series`` (pandas and numpy only), which a check made before
the GPU is claimed can import without TensorFlow (D-036); they are imported here, not copied.

Interface contract
------------------
``load_forecast_batches(*, format, dataset, path, value_cols, time_col, lookback, horizon, realizations,
series_length, noise_level, patterns, select_fraction, test_fraction, n_select, n_test, steps_per_epoch,
max_samples, batch_size, seed, stream_seed, scale_floor)``
    Keyword only; the config owns the defaults. ``format`` is ``builtin`` (the synthetic bank; ``dataset`` must
    be ``synthetic``; ``realizations``, ``series_length``, ``noise_level`` and ``patterns`` apply) or ``csv``
    (``path``, ``value_cols``, ``time_col`` apply; the four synthetic arguments are ignored).
    :param seed: the split seed: keys the bank and the select and test windows. Never changes on a resume.
    :param stream_seed: keys the train windows only; ``None`` means ``seed``.
    :param scale_floor: the relative floor of the context scale, see :func:`context_stats_np`.
    :return: a :class:`ForecastBatches`. Every element of every dataset is ``(x, y)``: ``x`` the context
        ``(batch, lookback, channels)`` and ``y`` the target ``(batch, horizon, channels)``, float32, standardized
        by the CONTEXT's statistics; the batch axis is static.
    :raises ConfigError: for every refusal listed in the plan's Invariant 9 that this module can see: an unknown
        format, dataset or pattern, a lookback, horizon, fraction, count or batch that cannot work, a series or
        csv segment too short for one window, a select or test split with fewer windows than one batch, a csv
        that does not exist, has no rows, lacks a named column, holds a non-numeric or non-finite cell in a value
        column or a time column that is not strictly increasing, and a bank in which every series is non-finite.
        Raised before any dataset is built; none hangs.

``ForecastBatches.with_stream(completed_steps)``
    :return: the same object on the train stream of ``derive_stream_seed(seed, completed_steps)``: the same bank,
        split, fingerprint, select and test windows; a different train stream, even for ``completed_steps = 0``.

``standardize_tf(window, lookback, scale_floor)`` and ``standardize_np(window, lookback, scale_floor)``
    Twins: ``window`` is ``(..., lookback + horizon, channels)``; returns ``(x, y)`` float32, computed in float64.
    The scale is ``max(std, scale_floor * mean|x|)`` of the context, or ``UNIT_SCALE`` when that is not above
    ``NO_SCALE`` (a flat zero context): finite and bounded for every input.

``train_window_at(stream_seed, counter, n_series, span)``
    :return: ``(series, start)`` int64 scalars, a pure function of ``(stream_seed, counter)``.
"""

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.time_series.generator import TimeSeriesGenerator, TimeSeriesGeneratorConfig
from dl_techniques.utils.logger import logger
from train.api.forecast_series import check_segments, read_csv_series, refuse, split_bounds
from train.common.pretrain_data import SELECT_SALT, TEST_SALT, derive_stream_seed

BUILTIN, CSV, SYNTHETIC = "builtin", "csv", "synthetic"

#: A context scale not above this is "no scale information" (an exactly flat zero context): see ``context_stats_np``.
NO_SCALE = 1e-30
#: The scale of such a context: the series' own unit, so the target stays what it is.
UNIT_SCALE = 1.0
#: Fixed select and test windows per series of the synthetic bank, when ``n_select`` or ``n_test`` is not given.
SELECT_WINDOWS_PER_SERIES = 8
#: The legacy epoch: the default ``steps_per_epoch`` never exceeds it.
EPOCH_STEPS = 500


# ---------------------------------------------------------------------
# standardization: the CONTEXT's statistics, applied to context and target
# ---------------------------------------------------------------------


def context_stats_np(context: np.ndarray, scale_floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and scale of ``(..., lookback, channels)`` over the time axis, as ``(..., 1, channels)`` float64.

    The scale is the POPULATION standard deviation floored at ``scale_floor * mean|x|``: relative to the context's
    own magnitude, so a series in micro-units is not floored wholesale. A scale not above ``NO_SCALE`` (a context
    that is exactly flat AND zero, which carries no scale information) is ``UNIT_SCALE``: the target stays in the
    series' own unit instead of being divided by a number near zero.
    """
    # DECISION plan-2026-11-13T105600-3a8319ef/D-005
    # The statistics come from the CONTEXT only, and the TARGET is divided by the same numbers. Do NOT
    # take them over the whole window: the legacy processor did and a zero predictor then beat naive-last
    # on MSE (the horizon was in the statistics). Do NOT floor in absolute units (D-027): the floor is
    # relative to mean|x| so a csv in micro-units keeps its scale.
    context = np.asarray(context, np.float64)
    mean = context.mean(axis=-2, keepdims=True)
    std = context.std(axis=-2, keepdims=True)
    # DECISION plan-2026-11-13T105600-3a8319ef/D-027
    # The floor binds 0.03 to 0.7 percent of the synthetic bank's windows, all of them flat `crypto_prices`
    # tails. Do NOT make it an absolute number: a csv in micro-units would be floored wholesale.
    # DECISION plan-2026-11-13T105600-3a8319ef/D-035
    # A flat zero context has std 0 and mean|x| 0, so BOTH terms are 0. An absolute floor (1e-8, as this had
    # before) divided a moving target by 1e-8: sparse_events#4 at R=8 seed 42 reached 1.08e9. Do NOT bring an
    # absolute floor back, and do NOT clip or re-key the window instead: the Predictor and the export call
    # this same function, so the rule has to be a function of the context alone. Such a window keeps its raw unit.
    scale = np.maximum(std, scale_floor * np.abs(context).mean(axis=-2, keepdims=True))
    scale = np.where(scale > NO_SCALE, scale, UNIT_SCALE)
    return mean, scale


def standardize_np(window: np.ndarray, lookback: int, scale_floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """The numpy twin of :func:`standardize_tf`: ``(x, y)`` float32 from ``(..., lookback + horizon, channels)``."""
    window = np.asarray(window, np.float64)
    context = window[..., :lookback, :]
    mean, scale = context_stats_np(context, scale_floor)
    return ((context - mean) / scale).astype(np.float32), ((window[..., lookback:, :] - mean) / scale).astype(np.float32)


def standardize_tf(window: "tf.Tensor", lookback: int, scale_floor: float) -> Tuple["tf.Tensor", "tf.Tensor"]:
    """Context and target of a window, standardized by the context's mean and floored population std.

    The twin of :func:`standardize_np`, rule for rule: the relative floor, then ``UNIT_SCALE`` for a flat zero context.
    """
    window = tf.cast(window, tf.float64)
    context = window[..., :lookback, :]
    mean = tf.reduce_mean(context, axis=-2, keepdims=True)
    std = tf.sqrt(tf.reduce_mean(tf.square(context - mean), axis=-2, keepdims=True))
    floor = scale_floor * tf.reduce_mean(tf.abs(context), axis=-2, keepdims=True)
    scale = tf.maximum(std, floor)
    scale = tf.where(scale > NO_SCALE, scale, tf.constant(UNIT_SCALE, tf.float64))
    return tf.cast((context - mean) / scale, tf.float32), tf.cast((window[..., lookback:, :] - mean) / scale, tf.float32)


# ---------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------


def _bank_seed(seed: int, pattern: str, realization: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{pattern}:{realization}".encode()).digest()[:4], "big")


def build_bank(
        *, patterns: Sequence[str], seed: int, realizations: int, series_length: int, noise_level: float,
) -> Tuple[np.ndarray, List[Tuple[str, int]], List[Tuple[str, int]]]:
    """``(series, owners, dropped)``: finite series ``(n, series_length)`` float64 and who made them.

    One fresh generator per ``(pattern, realization)``, seeded from a sha256 of ``(seed, pattern, realization)``:
    the bank is a pure function of its arguments, in any pattern order and in any interpreter.
    """
    # DECISION plan-2026-11-13T105600-3a8319ef/D-004
    # A fresh generator per (pattern, realization) with a derived seed (about 10 ms each). Do NOT share one
    # generator: its RandomState is stateful, so a series would depend on how many draws came before it and
    # a resume or a second interpreter would build another bank. Do NOT key the bank on the stream seed: it
    # moves on every resume, and the select windows would then score the same weights on other series.
    rows: List[np.ndarray] = []
    owners: List[Tuple[str, int]] = []
    dropped: List[Tuple[str, int]] = []
    for pattern in patterns:
        for realization in range(realizations):
            generator = TimeSeriesGenerator(TimeSeriesGeneratorConfig(
                n_samples=series_length, random_seed=_bank_seed(seed, pattern, realization),
                default_noise_level=noise_level))
            series = np.asarray(generator.generate_task_data(pattern), np.float64).reshape(-1)
            if series.shape[0] != series_length:
                refuse(f"pattern {pattern!r} gave {series.shape[0]} points, not data.series_length = {series_length}")
            if np.isfinite(series).all():
                rows.append(series)
                owners.append((pattern, realization))
            else:
                dropped.append((pattern, realization))
    return (np.stack(rows) if rows else np.zeros((0, series_length))), owners, dropped


def _fixed_positions(seed: int, salt: str, count: int, n_series: int, lo: int, hi: int, width: int) -> np.ndarray:
    """``(count, 2)`` int64 ``(series, start)``: row ``i`` is a function of ``(seed, salt, i)`` alone."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-004
    # Select and test windows are keyed on the SPLIT seed, the salt and the index: the same weights score
    # the same windows after any kill, in any interpreter. Do NOT key them on the stream seed (it moves on a
    # resume) and do NOT draw them from a generator with a hidden state.
    positions = np.empty((count, 2), np.int64)
    span = hi - lo - width + 1
    for i in range(count):
        digest = hashlib.sha256(f"{seed}:{salt}:{i}".encode()).digest()
        positions[i] = (i % n_series, lo + int.from_bytes(digest[8:16], "big") % span)
    return positions


def train_window_at(stream_seed: Any, counter: Any, n_series: int, span: int) -> Tuple["tf.Tensor", "tf.Tensor"]:
    """``(series, start)`` of train window number ``counter`` of the stream ``stream_seed``: a pure function.

    ``span`` is the number of window starts inside the train segment, so every window lies inside it.
    """
    key = tf.stack([tf.cast(stream_seed, tf.int64), tf.cast(counter, tf.int64)])
    u = tf.random.stateless_uniform([2], seed=key, dtype=tf.float64)
    series = tf.minimum(tf.cast(tf.floor(u[0] * n_series), tf.int64), n_series - 1)
    start = tf.minimum(tf.cast(tf.floor(u[1] * span), tf.int64), span - 1)
    return series, start


# ---------------------------------------------------------------------
# the batches
# ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ForecastBatches:
    """What the forecasting task consumes: the fields the run sequence reads, plus the keys behind them.

    :param train_ds: infinite, batched ``(x, y)``; the task passes ``steps_per_epoch`` to ``fit``. Window ``i``
        is keyed on ``(stream_seed, i)`` and lies inside the train segment.
    :param select_ds, test_ds: finite, one pass of ``n_select // batch_size`` and ``n_test // batch_size``
        batches (the remainder is dropped, the batch axis is static); window ``i`` is keyed on ``(seed, salt, i)``.
    :param steps_per_epoch, n_train, n_select, n_test: batches in an epoch, ``steps_per_epoch * batch_size``,
        and the fixed windows of the two splits (multiples of the batch).
    :param input_shape, num_classes: ``(lookback, channels)`` and ``None``.
    :param selection_split, test_split: plain-language provenance of the two splits.
    :param fingerprint: sha256 of the source (the csv bytes or the bank parameters) and of every window key; it
        does not contain the stream seed, so a resume accepts it.
    :param seed, stream_seed, batch_size, lookback, horizon: the keys.
    :param bank: ``(n_series, rows, channels)`` float64; ``segments`` the ``(lo, hi)`` rows of each split;
        ``select_windows``, ``test_windows`` the ``(series, start)`` of every fixed window.
    :param dropped: ``(pattern, realization)`` of the non-finite series the bank left out.
    :param first_batch_digest: sha256 of the first train batch, from the pure map (no fit iterator is touched).
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: "tf.data.Dataset"
    steps_per_epoch: int
    n_train: int
    n_select: int
    n_test: int
    input_shape: Tuple[int, int]
    num_classes: Optional[int]
    selection_split: str
    test_split: str
    fingerprint: str
    seed: int
    stream_seed: int
    batch_size: int
    lookback: int
    horizon: int
    bank: np.ndarray = field(repr=False)
    segments: Dict[str, Tuple[int, int]]
    select_windows: np.ndarray = field(repr=False)
    test_windows: np.ndarray = field(repr=False)
    n_series: int
    dropped: Tuple[Tuple[str, int], ...]
    first_batch_digest: str
    train_builder: Callable[[int], Tuple["tf.data.Dataset", str]] = field(repr=False)

    def with_stream(self, completed_steps: int) -> "ForecastBatches":
        """This split on the train stream of ``derive_stream_seed(seed, completed_steps)``."""
        # DECISION plan-2026-11-13T105600-3a8319ef/D-004
        # Only the train stream moves on a resume. Do NOT rebuild the split, the bank or the select and test
        # windows from the new seed (selection rows would leak into train and val_mae would change under
        # the same weights), and do NOT keep the old stream (a fresh iterator would replay epoch 0).
        stream_seed = derive_stream_seed(self.seed, completed_steps)
        train_ds, digest = self.train_builder(stream_seed)
        return dataclasses.replace(self, stream_seed=stream_seed, train_ds=train_ds, first_batch_digest=digest)


def _require_positive(name: str, value: Optional[int]) -> None:
    if value is not None and value < 1:
        refuse(f"data.{name} must be at least 1, got {value}")


def _check_request(request: Dict[str, Any]) -> None:
    if request["format"] not in (BUILTIN, CSV):
        refuse(f"data.format must be '{BUILTIN}' or '{CSV}', got {request['format']!r}")
    if request["format"] == BUILTIN:
        if request["dataset"] != SYNTHETIC:
            refuse(f"data.dataset must be '{SYNTHETIC}' for data.format '{BUILTIN}', got {request['dataset']!r}")
        _require_positive("realizations", request["realizations"])
        _require_positive("series_length", request["series_length"])
    for name in ("lookback", "horizon", "batch_size", "steps_per_epoch", "max_samples"):
        _require_positive(name, request[name])
    for name in ("select_fraction", "test_fraction"):
        if not 0.0 < request[name] < 1.0:
            refuse(f"data.{name} must be in (0, 1), got {request[name]}")
    if request["select_fraction"] + request["test_fraction"] >= 1.0:
        refuse("data.select_fraction + data.test_fraction must leave a train segment (their sum must be below 1)")
    if request["scale_floor"] < 0:
        refuse(f"transform_args.scale_floor must not be negative, got {request['scale_floor']}")


def _count(name: str, requested: Optional[int], default: int, batch_size: int) -> int:
    """Windows of a split: the request (or ``default``) rounded down to a multiple of the batch, at least one batch."""
    wanted = default if requested is None else requested
    n = wanted // batch_size * batch_size
    if n < 1:
        refuse(f"data.n_{name} = {wanted} is fewer than one batch ({batch_size}): a split needs a whole batch")
    return n


def _strided_positions(name: str, segment: Tuple[int, int], requested: Optional[int], request: Dict[str, Any]) -> np.ndarray:
    """csv windows of one segment: targets do not overlap (stride = horizon), the first window starts at ``lo``."""
    lo, hi = segment
    width, horizon, batch_size = request["lookback"] + request["horizon"], request["horizon"], request["batch_size"]
    available = (hi - lo - width) // horizon + 1
    if available < batch_size:
        needed = request["lookback"] + horizon * batch_size
        fraction = request[f"{name}_fraction"]
        refuse(
            f"the {name} segment ({hi - lo} rows) holds {available} windows with non-overlapping targets, fewer than "
            f"one batch of {batch_size}: it needs at least {needed} rows (lookback + horizon * batch_size), so the "
            f"file needs about {math.ceil(needed / fraction)} rows at data.{name}_fraction = {fraction}")
    count = _count(name, requested, available, batch_size)
    if count > available:
        refuse(f"data.n_{name} = {requested} asks for more windows than the {name} segment holds ({available})")
    return np.stack([np.zeros(count, np.int64), lo + horizon * np.arange(count, dtype=np.int64)], axis=1)


def load_forecast_batches(
        *, format: str, dataset: Optional[str], path: Optional[str], value_cols: Optional[Sequence[str]],
        time_col: Optional[str], lookback: int, horizon: int, realizations: Optional[int],
        series_length: Optional[int], noise_level: Optional[float], patterns: Optional[Sequence[str]],
        select_fraction: float, test_fraction: float, n_select: Optional[int], n_test: Optional[int],
        steps_per_epoch: Optional[int], max_samples: Optional[int], batch_size: int, seed: int,
        stream_seed: Optional[int], scale_floor: float,
) -> ForecastBatches:
    """The three forecasting pipelines of a synthetic bank or a csv; see the module contract."""
    request = dict(locals())
    _check_request(request)
    width = lookback + horizon
    source: Dict[str, Any]
    dropped: List[Tuple[str, int]] = []
    if format == BUILTIN:
        names = sorted(TimeSeriesGenerator(TimeSeriesGeneratorConfig(n_samples=1)).get_task_names())
        chosen = names if patterns is None else list(patterns)
        unknown = [p for p in chosen if p not in names]
        if unknown:
            refuse(f"data.patterns: unknown pattern {unknown[0]!r}; the generator has {len(names)}, e.g. {names[:3]}")
        segments = split_bounds(series_length, select_fraction, test_fraction)
        check_segments(segments, series_length, width, "a synthetic series")
        series, owners, dropped = build_bank(
            patterns=chosen, seed=seed, realizations=realizations, series_length=series_length, noise_level=noise_level)
        if not owners:
            refuse(f"every series of the bank was non-finite ({len(dropped)} dropped); change data.patterns or the seed")
        if dropped:
            logger.warning(f"forecast bank: {len(dropped)} non-finite series dropped: {dropped[:5]}")
        bank = series[:, :, None]
        source = dict(patterns=chosen, realizations=realizations, series_length=series_length, noise_level=noise_level)
        default = len(bank) * SELECT_WINDOWS_PER_SERIES
        windows = [
            _fixed_positions(seed, salt, _count(name, requested, default, batch_size), len(bank), lo, hi, width)
            for (name, requested, salt), (lo, hi) in zip(
                (("select", n_select, SELECT_SALT), ("test", n_test, TEST_SALT)), segments[1:])]
        where = "random starts inside the segment of every series, keyed on (seed, salt, index)"
    else:
        values, columns, digest = read_csv_series(path, value_cols, time_col)
        segments = split_bounds(len(values), select_fraction, test_fraction)
        check_segments(segments, len(values), width, "the csv")
        bank = values[None]
        source = dict(sha256=digest, columns=columns, time_col=time_col)
        windows = [_strided_positions(name, segment, requested, request)
                   for name, segment, requested in (("select", segments[1], n_select), ("test", segments[2], n_test))]
        where = "non-overlapping targets (stride = horizon) inside the segment"
    select_windows, test_windows = windows
    n_series, rows, channels = bank.shape
    train_hi = segments[0][1]
    span = train_hi - width + 1

    # DECISION plan-2026-11-13T105600-3a8319ef/D-004
    # Train windows are drawn inside the TRAIN segment only, from a counter keyed on (stream_seed, counter),
    # by a stateless map. Do NOT use from_generator or a shared RandomState (a resume would replay it and a
    # parallel map would reorder it), and do NOT draw starts over the whole series (a train window would
    # contain select or test rows).
    bank_t = tf.constant(bank)

    def fetch(series: "tf.Tensor", start: "tf.Tensor") -> Tuple["tf.Tensor", "tf.Tensor"]:
        zero = tf.constant(0, tf.int64)
        window = tf.reshape(tf.slice(bank_t, tf.stack([series, start, zero]), [1, width, channels]), [width, channels])
        return standardize_tf(window, lookback, scale_floor)

    def batched(ds: "tf.data.Dataset") -> "tf.data.Dataset":
        return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    def fixed(positions: np.ndarray) -> "tf.data.Dataset":
        ds = tf.data.Dataset.from_tensor_slices((positions[:, 0], positions[:, 1]))
        return batched(ds.map(fetch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True))

    def train_map(stream: int) -> Callable[[Any], Tuple["tf.Tensor", "tf.Tensor"]]:
        return lambda counter: fetch(*train_window_at(stream, counter, n_series, span))

    first_stream = seed if stream_seed is None else stream_seed

    def train_stream(stream: int) -> Tuple["tf.data.Dataset", str]:
        counters = tf.data.Dataset.from_tensors(0).repeat().enumerate().map(lambda i, _: i)
        train_ds = batched(counters.map(train_map(stream), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True))
        x, y = next(iter(tf.data.Dataset.range(batch_size).map(train_map(stream)).batch(batch_size)))
        digest = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
        logger.info(f"forecast train stream: stream seed {stream}, first train batch sha256 {digest}")
        return train_ds, digest

    train_ds, first_digest = train_stream(first_stream)
    available = n_series * span if max_samples is None else min(n_series * span, max_samples)
    steps = steps_per_epoch or min(EPOCH_STEPS, max(1, available // batch_size))
    fingerprint = hashlib.sha256(json.dumps(dict(
        format=format, source=source, lookback=lookback, horizon=horizon, select_fraction=select_fraction,
        test_fraction=test_fraction, n_select=len(select_windows), n_test=len(test_windows),
        scale_floor=scale_floor, seed=seed, rows=rows, channels=channels, n_series=n_series,
    ), sort_keys=True).encode()).hexdigest()
    lo, hi = segments[1]
    return ForecastBatches(
        train_ds=train_ds, select_ds=fixed(select_windows), test_ds=fixed(test_windows), steps_per_epoch=steps,
        n_train=steps * batch_size, n_select=len(select_windows), n_test=len(test_windows),
        input_shape=(lookback, channels), num_classes=None,
        selection_split=(f"rows [{lo}, {hi}) of {n_series} series, the time slice after the train segment "
                         f"([0, {train_hi})); {len(select_windows)} fixed windows, {where}"),
        test_split=(f"rows [{segments[2][0]}, {rows}) of {n_series} series, the last time slice; "
                    f"{len(test_windows)} fixed windows, {where}"),
        fingerprint=fingerprint, seed=seed, stream_seed=first_stream, batch_size=batch_size, lookback=lookback,
        horizon=horizon, bank=bank, segments=dict(zip(("train", "select", "test"), segments)),
        select_windows=select_windows, test_windows=test_windows, n_series=n_series, dropped=tuple(dropped),
        first_batch_digest=first_digest, train_builder=train_stream)
