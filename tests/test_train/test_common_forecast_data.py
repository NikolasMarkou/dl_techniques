"""The forecast data library: a stateless bank, fixed select and test windows, context-only standardization.

``train.common.forecast_data`` replaces the legacy time-series
pipeline, which draws every window from one shared stateful ``RandomState`` (a stream is not a function of its
index and a resume replays), standardizes a window with the HORIZON included (a leak: a zero predictor beat
naive-last on MSE) and validates on a fresh noise draw. What is pinned here, at tiny sizes on the CPU:

* the bank is a pure function of ``(seed, pattern, realization)``: equal in a second build, in another pattern
  order and in a fresh interpreter; a non-finite series is dropped and counted; a bank with nothing left raises;
* every series splits by TIME into train, select and test, consecutive and disjoint; every train window of
  every stream seed lies in the train segment, every select and test window in its own segment, and no row of
  one segment is in a window of another;
* the select and test windows are a function of ``(seed, salt, index)``: equal across two passes, across stream
  seeds and across an interpreter restart; a legacy-style stateful draw DISAGREES with itself, so the
  comparison can see the failure it guards; the train windows are a function of ``(stream_seed, counter)``;
* the standardization uses the CONTEXT only (the leak instrument: replacing the target leaves the standardized
  context unchanged), is centred on the context's FIRST value and scaled by the population std, the tf function, the
  numpy twin and the float32 keras-ops function agree to float32 rounding of the standardized values at levels 0, 50,
  1e3, 1e5 and 1e6 over a std of about 8, and a FLAT context (std not above ``scale_floor * mean|x|``, at zero or at any
  level) leaves the target in its own unit instead of dividing it by a tiny number (D-035, D-040);
* cardinalities: ``train_ds`` is infinite, ``select_ds`` and ``test_ds`` state ``n // batch`` over a static
  batch, ``n`` a multiple of the batch;
* a csv: segments by fraction, windows inside their segment, a strictly increasing time column, finite numeric
  named columns, a fingerprint that moves with the file bytes and with a window key;
* every refusal the data layer can make is a named ``ConfigError`` raised within 30 seconds;
* the csv reader and ``split_bounds`` live in ``train.api.forecast_series``, which loads neither TensorFlow nor Keras
  (D-036), so a check can run before the GPU is claimed;
* a resume (``with_stream``) keeps the split and the select and test streams and changes the train stream.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import dl_techniques
from train.api import forecast_series as fs
from train.api.errors import ConfigError
from train.common import forecast_data as fd
from train.common.forecast_data import (
    build_bank,
    context_stats_np,
    load_forecast_batches,
    split_bounds,
    standardize_tf,
    train_window_at,
)
from train.common.pretrain_data import derive_stream_seed

from .test_api._forecast_fixtures import standardize_np, write_series_csv

LOOKBACK, HORIZON, BATCH, SEED = 32, 8, 16, 7
WIDTH = LOOKBACK + HORIZON
PATTERNS = ["linear_trend_strong", "daily_seasonality", "random_walk"]
REALIZATIONS, ROWS = 2, 600
SELECT_FRACTION = TEST_FRACTION = 0.2

KW = dict(
    format="builtin", dataset="synthetic", path=None, value_cols=None, time_col=None,
    lookback=LOOKBACK, horizon=HORIZON, realizations=REALIZATIONS, series_length=ROWS, noise_level=0.1,
    patterns=PATTERNS, select_fraction=SELECT_FRACTION, test_fraction=TEST_FRACTION, n_select=None, n_test=None,
    steps_per_epoch=4, max_samples=None, batch_size=BATCH, seed=SEED, stream_seed=None, scale_floor=1e-6,
)
CSV_ROWS = 1000


def load(**over):
    return load_forecast_batches(**{**KW, **over})


def write_csv(tmp_path, rows=CSV_ROWS, name="series.csv", **edits):
    """A csv of a timestamp column ``ts`` and three value columns (the shared writer); ``edits`` maps a column to its new values."""
    return str(write_series_csv(tmp_path / name, rows, ("a", "b", "c"), "ts", values=edits))


def csv_kwargs(path, **over):
    return dict(format="csv", dataset=None, path=path, realizations=None, series_length=None, noise_level=None,
                patterns=None, **{"time_col": "ts", **over})


def digest(ds):
    """sha256 over one pass of ``ds``: every element, in order."""
    h = hashlib.sha256()
    for x, y in ds:
        h.update(x.numpy().tobytes())
        h.update(y.numpy().tobytes())
    return h.hexdigest()


def passes_agree(ds):
    return digest(ds) == digest(ds)


def train_digest(batches, steps=3):
    return digest(batches.train_ds.take(steps))


@pytest.fixture
def within_30_seconds():
    """Fail a test that does not return in 30 s: a too-short series must raise, never hang."""
    def hang(signum, frame):
        raise TimeoutError("no error within 30 seconds: the call hangs")

    previous = signal.signal(signal.SIGALRM, hang)
    signal.setitimer(signal.ITIMER_REAL, 30)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------

_DIGEST_SOURCE = """
import hashlib
import json

from train.common.forecast_data import load_forecast_batches


def digest(kwargs_json):
    b = load_forecast_batches(**json.loads(kwargs_json))
    h = hashlib.sha256()
    h.update(b.bank.tobytes())
    for ds in (b.select_ds, b.test_ds, b.train_ds.take(3)):
        for x, y in ds:
            h.update(x.numpy().tobytes())
            h.update(y.numpy().tobytes())
    return h.hexdigest()
"""


class TestTheBankIsAPureFunction:
    def test_two_builds_and_another_pattern_order_give_the_same_series(self):
        args = dict(seed=SEED, realizations=2, series_length=300, noise_level=0.1)
        first, owners, dropped = build_bank(patterns=PATTERNS, **args)
        second, _, _ = build_bank(patterns=PATTERNS, **args)
        reverse, reverse_owners, _ = build_bank(patterns=PATTERNS[::-1], **args)
        np.testing.assert_array_equal(first, second)
        assert first.shape == (len(PATTERNS) * 2, 300)
        assert owners == [(p, r) for p in PATTERNS for r in range(2)] and dropped == []
        for row, owner in zip(reverse, reverse_owners):
            np.testing.assert_array_equal(row, first[owners.index(owner)])

    def test_the_seed_the_pattern_and_the_realization_each_move_the_series(self):
        a, _, _ = build_bank(patterns=["random_walk"], seed=1, realizations=2, series_length=300, noise_level=0.1)
        b, _, _ = build_bank(patterns=["random_walk"], seed=2, realizations=2, series_length=300, noise_level=0.1)
        assert not np.array_equal(a[0], a[1]) and not np.array_equal(a[0], b[0])

    def test_a_fresh_interpreter_builds_the_same_bank_and_the_same_windows(self, tmp_path):
        kwargs = json.dumps({**KW, "stream_seed": 11})
        namespace = {}
        exec(_DIGEST_SOURCE, namespace)
        here = namespace["digest"](kwargs)
        src = str(Path(dl_techniques.__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=src, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3", MPLBACKEND="Agg")
        env.pop("PYTHONHASHSEED", None)
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST_SOURCE + f"\nprint('DIGEST', digest({kwargs!r}))"],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        assert [l for l in out.stdout.splitlines() if l.startswith("DIGEST ")] == [f"DIGEST {here}"]

    def test_a_non_finite_series_is_dropped_and_counted(self, monkeypatch):
        real = fd.TimeSeriesGenerator.generate_task_data

        def poisoned(self, name):
            series = real(self, name)
            return np.full_like(series, np.nan) if name == "random_walk" else series

        monkeypatch.setattr(fd.TimeSeriesGenerator, "generate_task_data", poisoned)
        series, owners, dropped = build_bank(
            patterns=PATTERNS, seed=SEED, realizations=2, series_length=300, noise_level=0.1)
        assert np.isfinite(series).all() and series.shape == (4, 300)
        assert dropped == [("random_walk", 0), ("random_walk", 1)]
        assert all(pattern != "random_walk" for pattern, _ in owners)
        assert load().dropped == (("random_walk", 0), ("random_walk", 1))

    @pytest.mark.usefixtures("within_30_seconds")
    def test_a_bank_with_nothing_left_is_a_named_error(self, monkeypatch):
        monkeypatch.setattr(
            fd.TimeSeriesGenerator, "generate_task_data", lambda self, name: np.full(ROWS, np.inf))
        with pytest.raises(ConfigError, match="every series"):
            load()


# ---------------------------------------------------------------------
# the split by time
# ---------------------------------------------------------------------


class TestTheSplitIsByTime:
    @pytest.mark.parametrize("rows", [600, 1000, 1277, 10000])
    def test_segments_are_consecutive_disjoint_and_follow_the_fractions(self, rows):
        train, select, test = split_bounds(rows, 0.15, 0.15)
        n_select, n_test = round(0.15 * rows), round(0.15 * rows)
        assert train == (0, rows - n_select - n_test)
        assert select == (train[1], train[1] + n_select)
        assert test == (select[1], rows)
        assert train[0] < train[1] and select[0] < select[1] and test[0] < test[1]

    def test_the_bank_carries_its_segments(self):
        b = load()
        assert b.bank.shape == (len(PATTERNS) * REALIZATIONS, ROWS, 1)
        assert b.segments == dict(zip(("train", "select", "test"), split_bounds(ROWS, 0.2, 0.2)))

    @pytest.mark.parametrize("stream_seed", [None, 1, 99, 2 ** 31 + 5])
    def test_every_train_window_lies_in_the_train_segment(self, stream_seed):
        b = load(stream_seed=stream_seed)
        n_train = ROWS - 2 * round(0.2 * ROWS)
        span = n_train - WIDTH + 1
        series, starts = [], []
        for counter in range(400):
            s, start = train_window_at(b.stream_seed, tf.constant(counter, tf.int64), len(b.bank), span)
            series.append(int(s))
            starts.append(int(start))
        assert 0 <= min(series) and max(series) < len(b.bank)
        assert min(starts) >= 0 and max(starts) + WIDTH <= n_train
        assert len(set(starts)) > 50 and len(set(series)) == len(b.bank)

    def test_the_train_elements_are_the_standardized_bank_slices_at_those_positions(self):
        b = load(stream_seed=5)
        span = ROWS - 2 * round(0.2 * ROWS) - WIDTH + 1
        x, y = next(iter(b.train_ds))
        for counter in range(BATCH):
            s, start = (int(v) for v in train_window_at(5, tf.constant(counter, tf.int64), len(b.bank), span))
            ex, ey = standardize_np(b.bank[s, start:start + WIDTH], LOOKBACK, KW["scale_floor"])
            np.testing.assert_allclose(x[counter].numpy(), ex, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(y[counter].numpy(), ey, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("which, segment", [("select_windows", 1), ("test_windows", 2)])
    def test_select_and_test_windows_lie_in_their_own_segment(self, which, segment):
        b = load()
        lo, hi = split_bounds(ROWS, 0.2, 0.2)[segment]
        windows = getattr(b, which)
        assert windows.shape[1] == 2 and windows.dtype == np.int64
        assert windows[:, 1].min() >= lo and windows[:, 1].max() + WIDTH <= hi
        assert 0 <= windows[:, 0].min() and windows[:, 0].max() < len(b.bank)
        assert set(windows[:, 0]) == set(range(len(b.bank)))

    def test_no_row_of_one_segment_is_in_a_window_of_another(self):
        b = load()
        train_hi = split_bounds(ROWS, 0.2, 0.2)[0][1]
        select_rows = {int(s) * ROWS + r for s, st in b.select_windows for r in range(st, st + WIDTH)}
        test_rows = {int(s) * ROWS + r for s, st in b.test_windows for r in range(st, st + WIDTH)}
        assert not select_rows & test_rows
        assert min(r % ROWS for r in select_rows | test_rows) >= train_hi
        assert max(b.select_windows[:, 1]) + WIDTH <= min(b.test_windows[:, 1])


# ---------------------------------------------------------------------
# stateless windows
# ---------------------------------------------------------------------


class TestTheWindowsAreStateless:
    def test_the_select_and_test_elements_equal_the_standardized_bank_slices(self):
        b = load()
        for ds, windows in ((b.select_ds, b.select_windows), (b.test_ds, b.test_windows)):
            xs = np.concatenate([x.numpy() for x, _ in ds])
            ys = np.concatenate([y.numpy() for _, y in ds])
            for i, (s, start) in enumerate(windows[: len(xs)]):
                ex, ey = standardize_np(b.bank[s, start:start + WIDTH], LOOKBACK, KW["scale_floor"])
                np.testing.assert_allclose(xs[i], ex, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(ys[i], ey, rtol=1e-5, atol=1e-6)

    def test_two_passes_of_select_and_test_are_equal(self):
        b = load()
        assert passes_agree(b.select_ds) and passes_agree(b.test_ds)

    def test_a_stateful_draw_disagrees_with_itself_so_the_comparison_can_see_it(self):
        state = np.random.RandomState(0)

        def draw(_):
            return state.standard_normal((WIDTH, 1)).astype("float32")

        ds = tf.data.Dataset.range(8).map(
            lambda i: (tf.numpy_function(draw, [i], tf.float32), tf.zeros((1,)))).batch(4)
        assert not passes_agree(ds)

    def test_select_and_test_do_not_move_with_the_stream_seed(self):
        fresh, other, third = load(), load(stream_seed=123), load(stream_seed=2 ** 30)
        for b in (other, third):
            np.testing.assert_array_equal(fresh.bank, b.bank)
            np.testing.assert_array_equal(fresh.select_windows, b.select_windows)
            np.testing.assert_array_equal(fresh.test_windows, b.test_windows)
            assert digest(fresh.select_ds) == digest(b.select_ds)
            assert digest(fresh.test_ds) == digest(b.test_ds)

    def test_select_and_test_move_with_the_split_seed_and_with_the_salt(self):
        a, b = load(), load(seed=SEED + 1)
        assert not np.array_equal(a.select_windows, b.select_windows)
        select_offsets = a.select_windows[:, 1] - a.segments["select"][0]
        test_offsets = a.test_windows[:, 1] - a.segments["test"][0]
        assert not np.array_equal(select_offsets, test_offsets)
        assert digest(a.select_ds) != digest(a.test_ds)

    def test_train_windows_differ_across_stream_seeds_and_counters(self):
        digests = {train_digest(load(stream_seed=s)) for s in (None, 1, 2, 3)}
        assert len(digests) == 4
        x, _ = next(iter(load(stream_seed=1).train_ds))
        assert len({row.tobytes() for row in x.numpy()}) == BATCH

    def test_a_train_stream_is_a_function_of_the_stream_seed_and_the_counter(self):
        assert train_digest(load(stream_seed=4)) == train_digest(load(stream_seed=4))
        assert train_digest(load(stream_seed=4), steps=5) != train_digest(load(stream_seed=4), steps=3)
        long = load(stream_seed=4).train_ds.take(5)
        short = load(stream_seed=4).train_ds.take(3)
        assert digest(long.take(3)) == digest(short)


# ---------------------------------------------------------------------
# standardization
# ---------------------------------------------------------------------


FLOOR = 1e-6
#: The levels of the level probe: a window of std near 8 sitting at each of them (D-040).
LEVELS = (0.0, 50.0, 1e3, 1e5, 1e6)


def oracle(window, lookback, floor):
    """The D-040 rule in ``longdouble`` on the RAW values (no first-value centring, so it does not share the code's trick).

    Centre on the context mean, scale by the population std, and a context is FLAT (scale 1) when ``std <= floor * mean|x|``.
    """
    window = np.asarray(window, np.longdouble)
    context = window[..., :lookback, :]
    mean = context.mean(axis=-2, keepdims=True)
    std = np.sqrt(((context - mean) ** 2).mean(axis=-2, keepdims=True))
    flat = std <= floor * np.abs(context).mean(axis=-2, keepdims=True)
    scale = np.where(flat, np.longdouble(1.0), std)
    return ((context - mean) / scale).astype(np.float64), ((window[..., lookback:, :] - mean) / scale).astype(np.float64)


def old_oracle(window, lookback, floor=1e-3):
    """The superseded rule of D-027 and D-035 (D-040 replaced it), kept ONLY to show that a window it did not floor is standardized the same way."""
    window = np.asarray(window, np.float64)
    context = window[..., :lookback, :]
    mean = context.mean(axis=-2, keepdims=True)
    scale = np.maximum(context.std(axis=-2, keepdims=True), floor * np.abs(context).mean(axis=-2, keepdims=True))
    scale = np.where(scale > 1e-30, scale, 1.0)
    return ((context - mean) / scale).astype(np.float32), ((window[..., lookback:, :] - mean) / scale).astype(np.float32)


def windows(n=12, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, WIDTH, channels)).cumsum(axis=1)
    return base * rng.uniform(0.1, 50.0, (n, 1, channels)) + rng.uniform(-1e3, 1e3, (n, 1, channels))


def level_windows(level, n=6, seed=3):
    """``(n, WIDTH, 1)`` windows of std about 8 at ``level``, each value a float32 number held as float64 (what a user's file holds)."""
    rng = np.random.default_rng(seed)
    t = np.arange(WIDTH)[None, :, None]
    base = 8.0 * np.sin(2 * np.pi * t / 12.0 + rng.uniform(0, 6, (n, 1, 1))) + 0.3 * rng.normal(size=(n, WIDTH, 1)).cumsum(axis=1)
    return np.asarray(level + base, np.float32).astype(np.float64)


def ops_standardize(window32, lookback, floor):
    """What the exported function does to a float32 window: ``(x - anchor - offset) / scale`` with ``context_stats_ops`` (float32)."""
    context = window32[..., :lookback, :]
    anchor, offset, scale = (np.asarray(t, np.float64) for t in fd.context_stats_ops(np.asarray(context, np.float32), floor))
    return ((context - anchor - offset) / scale).astype(np.float32), (anchor, offset, scale)


class TestTheStandardizationNeverSeesTheFuture:
    def test_replacing_the_target_leaves_the_standardized_context_unchanged(self):
        w = windows()
        shuffled = w.copy()
        shuffled[:, LOOKBACK:] = np.random.default_rng(9).standard_normal(shuffled[:, LOOKBACK:].shape) * 1e4
        x0, y0 = standardize_np(w, LOOKBACK, FLOOR)
        x1, y1 = standardize_np(shuffled, LOOKBACK, FLOOR)
        np.testing.assert_array_equal(x0, x1)
        assert not np.array_equal(y0, y1)
        tx0, _ = standardize_tf(tf.constant(w), LOOKBACK, FLOOR)
        tx1, _ = standardize_tf(tf.constant(shuffled), LOOKBACK, FLOOR)
        np.testing.assert_array_equal(tx0.numpy(), tx1.numpy())

    def test_the_context_has_zero_mean_and_unit_population_std(self):
        """Population, not sample (ddof 0): a sample std makes the standardized std sqrt(31/32), a 1.6 percent miss."""
        x, _ = standardize_np(windows(), LOOKBACK, FLOOR)
        np.testing.assert_allclose(x.mean(axis=-2), 0.0, atol=1e-5)
        np.testing.assert_allclose(x.std(axis=-2), 1.0, rtol=1e-5)
        tx, _ = standardize_tf(tf.constant(windows()), LOOKBACK, FLOOR)
        np.testing.assert_allclose(tx.numpy().std(axis=-2), 1.0, rtol=1e-5)

    def test_the_target_uses_the_context_statistics(self):
        w = windows()
        _, y = standardize_np(w, LOOKBACK, FLOOR)
        _, expected = oracle(w, LOOKBACK, FLOOR)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
        _, ty = standardize_tf(tf.constant(w), LOOKBACK, FLOOR)
        np.testing.assert_allclose(ty.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_the_tf_function_equals_its_numpy_twin_in_float32(self):
        w = windows(n=64, channels=3, seed=4)
        nx, ny = standardize_np(w, LOOKBACK, FLOOR)
        tx, ty = standardize_tf(tf.constant(w), LOOKBACK, FLOOR)
        assert nx.dtype == ny.dtype == np.float32 and tx.dtype == ty.dtype == tf.float32
        np.testing.assert_allclose(tx.numpy(), nx, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ty.numpy(), ny, rtol=1e-6, atol=1e-6)

    def test_a_window_that_the_old_floor_did_not_touch_is_standardized_as_before_and_so_are_the_baselines_and_the_bar(self):
        """D-040 changes only what a FLAT or a high-level window gets: for a normal window the standardized values, the
        naive-last and seasonal-naive MAE and the bar threshold are the ones of the superseded rule."""
        from train.common import forecast_run as fr

        w = np.concatenate([level_windows(0.0, n=32, seed=2), level_windows(50.0, n=32, seed=3), 1e-4 * level_windows(50.0, n=32, seed=4)])
        x_new, y_new = standardize_np(w, LOOKBACK, FLOOR)
        x_old, y_old = old_oracle(w, LOOKBACK)
        np.testing.assert_allclose(x_new, x_old, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y_new, y_old, rtol=1e-5, atol=1e-5)
        new, old = fr.baseline_maes(x_new, y_new), fr.baseline_maes(x_old, y_old)
        for key in ("naive_last", "seasonal_naive", "context_mean"):
            np.testing.assert_allclose(new[key], old[key], rtol=1e-5, err_msg=key)
        np.testing.assert_allclose(min(new["naive_last"], new["seasonal_naive"]), min(old["naive_last"], old["seasonal_naive"]), rtol=1e-5)

    def test_the_stats_are_the_first_value_the_offset_from_it_and_the_population_std(self):
        context = np.array([[[1.0], [2.0], [3.0], [4.0]]])
        anchor, offset, scale = context_stats_np(context, FLOOR)
        np.testing.assert_allclose(anchor, [[[1.0]]])
        np.testing.assert_allclose(offset, [[[1.5]]])
        np.testing.assert_allclose(scale, [[[np.sqrt(1.25)]]])
        anchor, offset, scale = (np.asarray(t, np.float64) for t in fd.context_stats_ops(np.asarray(context, np.float32), FLOOR))
        np.testing.assert_allclose([anchor.item(), offset.item(), scale.item()], [1.0, 1.5, np.sqrt(1.25)], rtol=1e-6)


class TestTheStandardizationIsShiftInvariantToTheFloat32LimitOfTheLevel:
    """D-040: centred on the FIRST context value, so the float32 function and the float64 twins agree at any level.

    What holds, measured at std about 8: the three twins give the standardized values of the oracle to float32 rounding of
    those values (not of the level) at every level of ``LEVELS`` (numpy and tf 5.9e-8, the float32 function at most 1.8e-7,
    at 0, 50, 1e3, 1e5 and 1e6 alike; the bound here is 1e-6); the numpy and tf functions given the SAME float64 window at
    another level give the same standardized values to 1e-6 (float64 resolves a level of 1e6 to 1e-10); the float32 FILE of a
    window cannot hold it better than one float32 step of the level (0.0625 at 1e6, 0.8 percent of this std) because the window
    itself was rounded to that grid: that, not the rule, is what limits the forecast of a float32 window at 1e6.
    """

    @pytest.mark.parametrize("level", LEVELS)
    def test_the_three_twins_equal_the_oracle_to_float32_rounding_of_the_standardized_values(self, level):
        w = level_windows(level)
        ex, ey = oracle(w, LOOKBACK, FLOOR)
        nx, ny = standardize_np(w, LOOKBACK, FLOOR)
        tx, ty = (t.numpy() for t in standardize_tf(tf.constant(w), LOOKBACK, FLOOR))
        for got, want in ((nx, ex), (ny, ey), (tx, ex), (ty, ey)):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=f"level {level:g}")
        ox, _ = ops_standardize(w.astype(np.float32), LOOKBACK, FLOOR)
        np.testing.assert_allclose(ox, ex, rtol=1e-6, atol=1e-6, err_msg=f"the float32 function at level {level:g}")

    @pytest.mark.parametrize("level", LEVELS[1:])
    def test_the_same_float64_window_at_another_level_gives_the_same_standardized_values(self, level):
        base = level_windows(0.0, seed=5)
        shifted = base + level
        for fn in (lambda a: standardize_np(a, LOOKBACK, FLOOR),
                   lambda a: tuple(t.numpy() for t in standardize_tf(tf.constant(a), LOOKBACK, FLOOR))):
            (x0, y0), (x1, y1) = fn(base), fn(shifted)
            np.testing.assert_allclose(x1, x0, rtol=1e-6, atol=1e-6, err_msg=f"level {level:g}")
            np.testing.assert_allclose(y1, y0, rtol=1e-6, atol=1e-6, err_msg=f"level {level:g}")

    @pytest.mark.parametrize("level", LEVELS[1:])
    def test_a_float32_window_at_a_level_is_standardized_to_within_the_step_of_its_own_grid(self, level):
        """At 1e6 the grid is 0.0625: the standardized values move by at most 3 grid steps over the std, and no more."""
        base = level_windows(0.0, seed=5)
        shifted = np.asarray(base + level, np.float32).astype(np.float64)
        x0, _ = standardize_np(base, LOOKBACK, FLOOR)
        x1, _ = standardize_np(shifted, LOOKBACK, FLOOR)
        bound = 3.0 * float(np.spacing(np.float32(level + 20.0))) / float(base[:, :LOOKBACK].std(axis=1).min())
        assert float(np.abs(x1 - x0).max()) <= bound + 1e-6, (level, float(np.abs(x1 - x0).max()), bound)

    @pytest.mark.parametrize("level", [1e3, 1e5, 1e6])
    def test_the_standardization_of_the_exported_function_does_not_depend_on_the_level(self, level):
        """The ops twin, run in float32, gives the standardized window of the unshifted one (rounding of the window grid only)."""
        base = np.asarray(level_windows(0.0, seed=6), np.float32).astype(np.float64)
        w = np.asarray(base + level, np.float32).astype(np.float64)
        want, _ = oracle(w, LOOKBACK, FLOOR)
        got, _ = ops_standardize(w.astype(np.float32), LOOKBACK, FLOOR)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


class TestAFlatContextHasNoScaleToDivideBy:
    """D-035 (generalised by D-040): a flat context, at zero or at any level, leaves the target in the series' own unit."""

    @staticmethod
    def both(w, lookback=LOOKBACK, floor=FLOOR):
        nx, ny = standardize_np(w, lookback, floor)
        tx, ty = standardize_tf(tf.constant(w), lookback, floor)
        return (nx, ny), (tx.numpy(), ty.numpy())

    def test_a_zero_context_with_a_spike_target_gives_the_raw_spike_in_both_twins_and_the_function(self):
        w = np.zeros((2, WIDTH, 1))
        w[0, LOOKBACK + 3, 0] = 10.84
        w[1, LOOKBACK:, 0] = np.linspace(-2.0, 5.0, HORIZON)
        for x, y in self.both(w):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            np.testing.assert_array_equal(x, 0.0)
            np.testing.assert_array_equal(y, w[:, LOOKBACK:].astype(np.float32))
            assert np.abs(y).max() < 11.0
        anchor, offset, scale = fd.context_stats_ops(np.zeros((2, LOOKBACK, 1), np.float32), FLOOR)
        assert (np.asarray(scale) == fd.UNIT_SCALE).all() and (np.asarray(anchor) == 0).all() and (np.asarray(offset) == 0).all()

    @pytest.mark.parametrize("level", [5.0, -3.0, 1e-6, 1e6])
    def test_a_constant_non_zero_context_with_a_moving_target_gives_the_raw_deviation_as_the_target(self, level):
        """The mean of the context is the level, the scale is 1: the target is ``y - level``, in the series' own unit."""
        w = np.full((3, WIDTH, 1), np.float64(np.float32(level)))
        w[:, LOOKBACK:] += np.arange(1, HORIZON + 1)[None, :, None] * np.array([1.0, -2.0, 0.5])[:, None, None]
        expected = (w[:, LOOKBACK:] - w[:, :1]).astype(np.float32)
        for x, y in self.both(w):
            np.testing.assert_array_equal(x, 0.0)
            np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-5)
        anchor, offset, scale = (np.asarray(t, np.float64) for t in fd.context_stats_ops(np.asarray(w[:, :LOOKBACK], np.float32), FLOOR))
        np.testing.assert_array_equal(scale, fd.UNIT_SCALE)
        np.testing.assert_allclose(anchor + offset, w[:, :1], rtol=1e-6)

    def test_the_tf_function_equals_the_numpy_twin_on_flat_and_ordinary_windows_together(self):
        w = windows(n=6, channels=2, seed=5)
        w[1, :LOOKBACK, 0] = 0.0
        w[1, LOOKBACK + 2, 0] = 7.5
        w[4, :LOOKBACK, 1] = 0.0
        w[4, LOOKBACK:, 1] = 123.0
        w[2, :LOOKBACK, 0] = 40.0
        (nx, ny), (tx, ty) = self.both(w)
        np.testing.assert_allclose(tx, nx, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ty, ny, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(ny[1, :, 0], w[1, LOOKBACK:, 0].astype(np.float32))
        np.testing.assert_array_equal(ny[4, :, 1], w[4, LOOKBACK:, 1].astype(np.float32))
        _, expected = oracle(w, LOOKBACK, FLOOR)
        np.testing.assert_allclose(ny, expected, rtol=1e-5, atol=1e-5)

    def test_the_flat_threshold_is_the_std_against_the_floor_times_the_mean_absolute_value(self):
        """Alternating ``level +- s``: the population std is ``s``; at level 1e5 and floor 1e-6 the threshold is 0.1.

        A flat context has scale 1, NOT the threshold: the superseded rule gave it ``floor * mean|x|`` (0.1 here), so the standardized
        context of the flat case is ``+- 0.09`` and not ``+- 0.9``."""
        level = 1e5
        for spread, flat in ((0.09, True), (0.11, False)):
            context = level + spread * np.tile([1.0, -1.0], LOOKBACK // 2)[None, :, None]
            w = np.concatenate([context, context[:, :HORIZON] + 3.0], axis=1)
            for x, y in self.both(w, floor=1e-6):
                if flat:
                    np.testing.assert_allclose(x, context - level, atol=1e-6)         # the scale is 1: the raw deviation
                    np.testing.assert_allclose(y, 3.0 + context[:, :HORIZON] - level, atol=1e-6)
                else:
                    np.testing.assert_allclose(np.abs(x), 1.0, rtol=1e-5)             # the scale is the std
                    np.testing.assert_allclose(y, (3.0 + context[:, :HORIZON] - level) / spread, rtol=1e-4)

    def test_a_zero_scale_floor_on_a_flat_context_is_the_unit_scale_too(self):
        w = np.full((1, WIDTH, 1), 4.0)
        w[:, LOOKBACK:] = 9.0
        for x, y in self.both(w, floor=0.0):
            np.testing.assert_array_equal(x, 0.0)
            np.testing.assert_allclose(y, 5.0)

    def test_the_bank_at_eight_realizations_and_seed_42_has_no_window_standardized_past_a_thousand(self):
        """The measured toxic bank: ``sparse_events#4`` gave 1.08e9; every window of every segment is scanned."""
        names = sorted(fd.TimeSeriesGenerator(fd.TimeSeriesGeneratorConfig(n_samples=1)).get_task_names())
        series_length, realizations = 10_000, 8
        bank, owners, _ = build_bank(
            patterns=names, seed=42, realizations=realizations, series_length=series_length, noise_level=0.1)
        assert bank.shape == (len(owners), series_length) and ("sparse_events", 4) in owners
        worst, where = 0.0, None
        for lo, hi in split_bounds(series_length, 0.15, 0.15):
            for index, row in enumerate(bank):
                frames = np.lib.stride_tricks.sliding_window_view(row[lo:hi], 168 + 24)[:, :, None]
                with np.errstate(all="ignore"):
                    _, y = standardize_np(frames, 168, FLOOR)
                # a NaN (0 / 0) or an inf (spike / 0) must count as the worst value, not vanish from a max
                top = float(np.abs(y).max()) if np.isfinite(y).all() else float("inf")
                if top > worst:
                    worst, where = top, (owners[index], lo)
        assert np.isfinite(worst) and worst < 1e3, (worst, where)


# ---------------------------------------------------------------------
# cardinalities
# ---------------------------------------------------------------------


class TestCardinalities:
    def test_train_is_infinite_and_select_and_test_state_n_over_batch(self):
        b = load()
        n = len(PATTERNS) * REALIZATIONS * fd.SELECT_WINDOWS_PER_SERIES // BATCH * BATCH
        assert int(b.train_ds.cardinality()) == int(tf.data.INFINITE_CARDINALITY)
        assert b.n_select == b.n_test == n and n % BATCH == 0 and n > 0
        for ds, count in ((b.select_ds, b.n_select), (b.test_ds, b.n_test)):
            assert int(ds.cardinality()) == count // BATCH
            assert sum(1 for _ in ds) == count // BATCH
        for ds in (b.train_ds, b.select_ds, b.test_ds):
            x_spec, y_spec = ds.element_spec
            assert x_spec.shape == (BATCH, LOOKBACK, 1) and y_spec.shape == (BATCH, HORIZON, 1)
            assert x_spec.dtype == y_spec.dtype == tf.float32

    def test_n_select_and_n_test_round_down_to_a_multiple_of_the_batch(self):
        b = load(n_select=50, n_test=70)
        assert (b.n_select, b.n_test) == (48, 64)
        assert int(b.select_ds.cardinality()) == 3 and int(b.test_ds.cardinality()) == 4
        assert len(b.select_windows) == 48 and len(b.test_windows) == 64

    def test_the_facts_the_sequence_reads(self):
        b = load()
        n_series = len(PATTERNS) * REALIZATIONS
        assert (b.input_shape, b.num_classes) == ((LOOKBACK, 1), None)
        assert b.steps_per_epoch == 4 and b.n_train == 4 * BATCH
        assert isinstance(b.selection_split, str) and isinstance(b.test_split, str)
        assert len(b.fingerprint) == 64 and b.n_series == n_series

    def test_the_default_steps_per_epoch_is_the_legacy_epoch_capped_by_the_windows(self):
        n_train = ROWS - 2 * round(0.2 * ROWS)
        windows_total = len(PATTERNS) * REALIZATIONS * (n_train - WIDTH + 1)
        b = load(steps_per_epoch=None)
        assert b.steps_per_epoch == min(500, max(1, windows_total // BATCH))
        assert b.n_train == b.steps_per_epoch * BATCH

    def test_max_samples_caps_the_train_epoch_and_never_select_or_test(self):
        full, smoke = load(steps_per_epoch=None), load(steps_per_epoch=None, max_samples=64)
        assert smoke.steps_per_epoch == 4 and smoke.n_train == 64
        assert (smoke.n_select, smoke.n_test) == (full.n_select, full.n_test)
        np.testing.assert_array_equal(smoke.select_windows, full.select_windows)


# ---------------------------------------------------------------------
# csv
# ---------------------------------------------------------------------


class TestCsv:
    def test_the_segments_the_windows_and_the_default_columns(self, tmp_path):
        path = write_csv(tmp_path)
        b = load(**csv_kwargs(path))
        assert b.bank.shape == (1, CSV_ROWS, 3) and b.input_shape == (LOOKBACK, 3)
        assert b.segments == dict(zip(("train", "select", "test"), split_bounds(CSV_ROWS, 0.2, 0.2)))
        for which, segment in (("select_windows", "select"), ("test_windows", "test")):
            lo, hi = b.segments[segment]
            w = getattr(b, which)
            assert (w[:, 0] == 0).all() and w[:, 1].min() == lo and w[:, 1].max() + WIDTH <= hi
            assert (np.diff(w[:, 1]) == HORIZON).all()
        train_hi = b.segments["train"][1]
        assert b.select_windows[:, 1].min() >= train_hi

    def test_the_value_columns_follow_the_order_given(self, tmp_path):
        path = write_csv(tmp_path)
        b = load(**csv_kwargs(path, value_cols=["c", "a"]))
        frame = pd.read_csv(path)
        np.testing.assert_array_equal(b.bank[0], frame[["c", "a"]].to_numpy(np.float64))

    def test_a_column_whose_spread_is_below_the_floor_of_its_level_is_refused_by_the_loader_too(self, tmp_path):
        """D-040: the same ``check_csv_level`` the pre-claim check calls; a floor of 0 or a resolvable level loads."""
        t = np.arange(CSV_ROWS)
        path = write_csv(tmp_path, a=1e12 + 8.0 * np.sin(t / 7.0))
        with pytest.raises(ConfigError, match="column 'a' varies by .* the level must be removed before training"):
            load(**csv_kwargs(path))
        assert load(**csv_kwargs(path, scale_floor=0.0)).bank.shape == (1, CSV_ROWS, 3), "a floor of 0 resolves nothing and refuses nothing"
        near = write_csv(tmp_path, name="near.csv", a=1e5 + 8.0 * np.sin(t / 7.0))
        assert load(**csv_kwargs(near)).bank.shape == (1, CSV_ROWS, 3), "a spread of 8e-5 of the level is above the floor of 1e-6"

    def test_an_index_column_and_the_time_column_are_not_value_columns(self, tmp_path):
        frame = pd.read_csv(write_csv(tmp_path))
        path = tmp_path / "indexed.csv"
        frame.to_csv(path, index=True)
        b = load(**csv_kwargs(str(path)))
        np.testing.assert_allclose(b.bank[0], frame[["a", "b", "c"]].to_numpy(np.float64), rtol=1e-12)

    def test_no_row_of_one_segment_is_in_a_window_of_another(self, tmp_path):
        b = load(**csv_kwargs(write_csv(tmp_path)))
        train_hi, (sel_lo, sel_hi), (test_lo, test_hi) = (
            b.segments["train"][1], b.segments["select"], b.segments["test"])
        select_rows = {r for _, st in b.select_windows for r in range(st, st + WIDTH)}
        test_rows = {r for _, st in b.test_windows for r in range(st, st + WIDTH)}
        assert min(select_rows) == sel_lo and max(select_rows) < sel_hi
        assert min(test_rows) == test_lo and max(test_rows) < test_hi
        assert not select_rows & test_rows
        assert max(select_rows) < min(test_rows) and sel_lo == train_hi
        span = train_hi - WIDTH + 1
        starts = [int(train_window_at(3, tf.constant(c, tf.int64), 1, span)[1]) for c in range(300)]
        assert max(starts) + WIDTH <= train_hi

    def test_the_fingerprint_moves_with_the_file_bytes_and_each_window_key(self, tmp_path):
        path = write_csv(tmp_path)
        base = load(**csv_kwargs(path)).fingerprint
        assert load(**csv_kwargs(path)).fingerprint == base
        assert load(**csv_kwargs(path, stream_seed=9, steps_per_epoch=7)).fingerprint == base
        edited = write_csv(tmp_path, name="edited.csv", a=np.arange(CSV_ROWS) * 0.5)
        assert load(**csv_kwargs(edited)).fingerprint != base
        for change in (dict(lookback=LOOKBACK + 1), dict(horizon=HORIZON + 1), dict(seed=SEED + 1),
                       dict(select_fraction=0.25), dict(scale_floor=1e-2), dict(value_cols=["a", "b"]),
                       dict(batch_size=5)):
            assert load(**csv_kwargs(path, **change)).fingerprint != base, change

    def test_the_builtin_fingerprint_moves_with_the_bank_and_each_window_key(self):
        base = load().fingerprint
        assert load(stream_seed=9, steps_per_epoch=7).fingerprint == base
        for change in (dict(lookback=LOOKBACK + 1), dict(horizon=HORIZON + 1), dict(seed=SEED + 1),
                       dict(realizations=3), dict(series_length=ROWS + 50), dict(noise_level=0.2),
                       dict(patterns=PATTERNS[:2]), dict(scale_floor=1e-2), dict(test_fraction=0.25),
                       dict(n_select=32)):
            assert load(**change).fingerprint != base, change


# ---------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------


_TF_FREE_SOURCE = """
import sys

from train.api.forecast_series import read_csv_series, split_bounds

values, names, digest = read_csv_series(sys.argv[1], ["a", "b"], "ts")
segments = split_bounds(len(values), 0.2, 0.2)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("tensorflow", "keras", "tf_keras"))
common = sorted(m for m in sys.modules if m == "train.common" or m.startswith("train.common."))
print("RESULT", values.shape, names, segments, loaded, common)
"""


class TestTheCsvReaderAndTheSplitLoadNoTensorFlow:
    """D-036: ``train.api.forecast_series`` is what a pre-claim check imports; TF must not come with it."""

    def test_a_fresh_interpreter_reads_a_csv_and_splits_it_with_tensorflow_and_keras_out_of_sys_modules(self, tmp_path):
        path = write_csv(tmp_path, rows=200)
        src = str(Path(dl_techniques.__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=src, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3", MPLBACKEND="Agg")
        out = subprocess.run(
            [sys.executable, "-c", _TF_FREE_SOURCE, str(path)], capture_output=True, text=True, env=env,
            cwd=str(tmp_path), timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        (line,) = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        assert line == "RESULT (200, 2) ['a', 'b'] ((0, 120), (120, 160), (160, 200)) [] []", line

    def test_the_loader_uses_the_same_functions_not_copies(self):
        from train.api import forecast_series as fs
        assert fd.read_csv_series is fs.read_csv_series and fd.split_bounds is fs.split_bounds
        assert fd.check_segments is fs.check_segments


@pytest.mark.usefixtures("within_30_seconds")
class TestEveryRefusalIsANamedErrorWithinThirtySeconds:
    @pytest.mark.parametrize("over, match", [
        (dict(format="parquet"), "data.format"),
        (dict(dataset="electricity"), "data.dataset"),
        (dict(lookback=0), "data.lookback"),
        (dict(horizon=0), "data.horizon"),
        (dict(select_fraction=0.5, test_fraction=0.5), "train segment"),
        (dict(select_fraction=0.0), "select_fraction"),
        (dict(series_length=WIDTH + 10), "too short"),
        (dict(patterns=["no_such_pattern"]), "unknown pattern"),
        (dict(n_select=BATCH - 1), "fewer than one batch"),
        (dict(n_test=0), "'data.n_test': expected an int of at least 1"),   # the config's own sentence: one rule (D-044)
        (dict(realizations=0), "realizations"),
        (dict(batch_size=0), "batch_size"),
        (dict(steps_per_epoch=0), "steps_per_epoch"),
    ])
    def test_a_builtin_request(self, over, match):
        with pytest.raises(ConfigError, match=match):
            load(**over)

    def test_a_csv_too_short_for_one_window_per_segment_never_hangs(self, tmp_path):
        with pytest.raises(ConfigError) as err:
            load(**csv_kwargs(write_csv(tmp_path, rows=100)))
        # The refusal is the shared segment check's one sentence (D-037), said before any window is placed: the
        # claim is asserted on its text, so a loader that skips the check and fails later on another sentence
        # (the select segment holds fewer windows than a batch) fails an assertion here, not an unrelated error.
        message = str(err.value)
        assert message.startswith("the csv of 100 rows is too short: its "), message
        assert f"one window needs lookback + horizon = {WIDTH}" in message, message

    def test_a_csv_whose_select_segment_holds_fewer_windows_than_a_batch_states_the_rows_needed(self, tmp_path):
        with pytest.raises(ConfigError, match=r"needs at least \d+ rows") as err:
            load(**csv_kwargs(write_csv(tmp_path, rows=400)))
        assert "select" in str(err.value)

    def test_a_csv_whose_select_segment_is_constant_is_refused_by_the_loader_with_the_sentence_the_check_before_the_claim_says(self, tmp_path):
        """D-042 (review NOTE 12): every selection window is flat, so naive-last and seasonal-naive would score exactly 0. The loader and
        ``forecast_series.check_split_sizes`` (before the claim) call ONE function; a column that varies elsewhere does not matter."""
        lo, hi = split_bounds(CSV_ROWS, SELECT_FRACTION, TEST_FRACTION)[1]
        values = np.random.default_rng(1).standard_normal(CSV_ROWS)
        values[lo:hi] = 4.0
        path = write_csv(tmp_path, a=values)
        with pytest.raises(ConfigError, match=rf"the select segment of the csv \(rows \[{lo}, {hi}\)\) is constant in every value column"):
            load(**csv_kwargs(path, value_cols=["a"]))
        load(**csv_kwargs(path, value_cols=["a", "b"]))  # a second column that moves keeps the baselines above 0

    def test_the_loader_reads_the_window_counts_through_the_functions_the_check_before_the_claim_calls(self, tmp_path, monkeypatch):
        calls = []
        for name in ("csv_window_count", "window_count", "check_select_moves"):
            monkeypatch.setattr(fd, name, (lambda real, name: lambda *a, **k: (calls.append(name), real(*a, **k))[1])(getattr(fd, name), name))
        load(**csv_kwargs(write_csv(tmp_path)))
        load()
        assert sorted(set(calls)) == ["check_select_moves", "csv_window_count", "window_count"], calls
        assert fd.SELECT_WINDOWS_PER_SERIES == fs.SELECT_WINDOWS_PER_SERIES, "one definition, imported"

    def test_an_unsorted_time_column(self, tmp_path):
        ts = pd.date_range("2024-01-01", periods=CSV_ROWS, freq="h").strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        ts[[10, 11]] = ts[[11, 10]]
        with pytest.raises(ConfigError, match="strictly increasing"):
            load(**csv_kwargs(write_csv(tmp_path, ts=ts)))

    def test_a_duplicate_timestamp(self, tmp_path):
        ts = pd.date_range("2024-01-01", periods=CSV_ROWS, freq="h").strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        ts[20] = ts[19]
        with pytest.raises(ConfigError, match="strictly increasing"):
            load(**csv_kwargs(write_csv(tmp_path, ts=ts)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_cell(self, tmp_path, bad):
        values = np.random.default_rng(0).standard_normal(CSV_ROWS)
        values[500] = bad
        with pytest.raises(ConfigError, match="non-finite"):
            load(**csv_kwargs(write_csv(tmp_path, a=values)))

    def test_a_named_column_that_is_missing(self, tmp_path):
        with pytest.raises(ConfigError, match="column 'nope' is missing"):
            load(**csv_kwargs(write_csv(tmp_path), value_cols=["a", "nope"]))

    def test_a_missing_time_column(self, tmp_path):
        with pytest.raises(ConfigError, match="column 'when' is missing"):
            load(**csv_kwargs(write_csv(tmp_path), time_col="when"))

    def test_a_column_with_one_text_cell_is_not_silently_dropped(self, tmp_path):
        values = np.random.default_rng(0).standard_normal(CSV_ROWS).astype(object)
        values[5] = "x"
        path = write_csv(tmp_path, a=values)
        with pytest.raises(ConfigError, match="'a' is not numeric"):
            load(**csv_kwargs(path, value_cols=["a", "b"]))
        with pytest.raises(ConfigError, match="'a' is not numeric"):
            load(**csv_kwargs(path))

    def test_a_missing_file_and_an_empty_one(self, tmp_path):
        with pytest.raises(ConfigError, match="does not exist"):
            load(**csv_kwargs(str(tmp_path / "nothing.csv")))
        header_only = tmp_path / "empty.csv"
        header_only.write_text("ts,a,b\n")
        with pytest.raises(ConfigError, match="no rows"):
            load(**csv_kwargs(str(header_only)))

    def test_a_csv_with_no_path_and_a_csv_with_no_value_column(self, tmp_path):
        with pytest.raises(ConfigError, match="data.path"):
            load(**csv_kwargs(None))
        only_time = tmp_path / "time.csv"
        only_time.write_text("ts\n" + "\n".join(f"2024-01-01 {h:02d}:00:00" for h in range(24)) + "\n")
        with pytest.raises(ConfigError, match="no value column"):
            load(**csv_kwargs(str(only_time)))

    def test_a_dtype_other_than_a_number_is_refused_in_a_default_selection_too(self, tmp_path):
        labels = np.array(["u", "v"] * (CSV_ROWS // 2))
        with pytest.raises(ConfigError, match="'label' is not numeric"):
            load(**csv_kwargs(write_csv(tmp_path, label=labels)))


# ---------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------


class TestAResumeIsANewTrainStreamOnTheSameSplit:
    def test_with_stream_keeps_the_split_and_the_fixed_streams_and_changes_the_train_stream(self):
        fresh = load()
        resumed = fresh.with_stream(100)
        other = fresh.with_stream(200)
        assert resumed.stream_seed == derive_stream_seed(SEED, 100) != fresh.stream_seed == SEED
        for b in (resumed, other):
            np.testing.assert_array_equal(fresh.bank, b.bank)
            np.testing.assert_array_equal(fresh.select_windows, b.select_windows)
            np.testing.assert_array_equal(fresh.test_windows, b.test_windows)
            assert digest(fresh.select_ds) == digest(b.select_ds) and digest(fresh.test_ds) == digest(b.test_ds)
            assert b.fingerprint == fresh.fingerprint and b.n_train == fresh.n_train
        assert len({train_digest(fresh), train_digest(resumed), train_digest(other)}) == 3
        assert len({fresh.first_batch_digest, resumed.first_batch_digest, other.first_batch_digest}) == 3

    def test_a_resume_at_step_zero_is_still_not_the_fresh_stream(self):
        fresh = load()
        assert train_digest(fresh.with_stream(0)) != train_digest(fresh)

    def test_the_resumed_stream_equals_a_fresh_load_on_that_stream_seed(self):
        resumed = load().with_stream(100)
        direct = load(stream_seed=derive_stream_seed(SEED, 100))
        assert train_digest(resumed) == train_digest(direct)
        assert resumed.first_batch_digest == direct.first_batch_digest

    def test_the_logged_digest_is_the_hash_of_the_first_batch_of_the_real_stream(self, monkeypatch):
        lines = []
        monkeypatch.setattr(fd.logger, "info", lambda message, *a, **k: lines.append(str(message)))
        b = load(stream_seed=21)
        x, y = next(iter(b.train_ds))
        expected = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
        assert b.first_batch_digest == expected
        assert any(f"stream seed {b.stream_seed}" in line and expected in line for line in lines)


# ---------------------------------------------------------------------
# the flat target of a quantile loss
# ---------------------------------------------------------------------


class TestFlattenTargets:
    """tirex and a quantile xlstm train on a ``(batch, horizon)`` target; the windows themselves do not change."""

    def test_every_split_gives_the_channel_zero_target_and_the_same_context(self):
        plain, flat = load(), fd.flatten_targets(load())
        for name in ("select_ds", "test_ds"):
            for (x, y), (fx, fy) in zip(getattr(plain, name), getattr(flat, name)):
                np.testing.assert_array_equal(x.numpy(), fx.numpy())
                assert fy.shape == (BATCH, HORIZON) and y.shape == (BATCH, HORIZON, 1)
                np.testing.assert_array_equal(y.numpy()[..., 0], fy.numpy())
        (x, y), (fx, fy) = next(iter(plain.train_ds)), next(iter(flat.train_ds))
        np.testing.assert_array_equal(y.numpy()[..., 0], fy.numpy())

    def test_the_counts_the_fingerprint_and_the_lengths_are_unchanged(self):
        plain, flat = load(), fd.flatten_targets(load())
        assert (flat.fingerprint, flat.n_train, flat.n_select, flat.n_test, flat.steps_per_epoch) == (
            plain.fingerprint, plain.n_train, plain.n_select, plain.n_test, plain.steps_per_epoch)
        assert int(flat.select_ds.cardinality()) == int(plain.select_ds.cardinality())
        assert int(flat.train_ds.cardinality()) == int(plain.train_ds.cardinality()) == tf.data.INFINITE_CARDINALITY

    def test_a_resume_keeps_the_target_flat_and_moves_only_the_train_stream(self):
        flat = fd.flatten_targets(load())
        resumed = flat.with_stream(100)
        assert next(iter(resumed.train_ds))[1].shape == (BATCH, HORIZON)
        assert resumed.stream_seed == derive_stream_seed(SEED, 100)
        plain_resumed = load().with_stream(100)
        np.testing.assert_array_equal(
            next(iter(plain_resumed.train_ds))[1].numpy()[..., 0], next(iter(resumed.train_ds))[1].numpy())
        assert digest(resumed.select_ds) == digest(flat.select_ds)
