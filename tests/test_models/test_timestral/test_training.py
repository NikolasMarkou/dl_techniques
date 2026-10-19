"""The training forward (``call``) and objective (``compute_loss``) under stock ``fit``.

Until 2026-10-19 (plan-2026-10-14T062616-81938e81 D-169) ``TimestralModel.call`` raised
``NotImplementedError``, so ``fit`` had nothing to train, and the masks used
``keras.ops.tril``, which raises "pred must not be a Python bool" once traced, so no
graph or XLA forward could run either.

The objective is the reference's (Mistral AI's announcement: a pinball loss over the 21
quantiles), scored per horizon patch in the frame the forecaster decodes it in. The
oracle below rebuilds that frame from its definition in NumPy (two-pass mean and
Bessel std over every observed value up to the end of the previous patch), independent
of ``scaler.compute_causal_statistics``.
"""

import numpy as np
import pytest

import keras

from dl_techniques.models.time_series.timestral import create_timestral_model
from dl_techniques.models.time_series.timestral.forecaster import TimestralForecaster

PS = 8


def _tiny(prediction_length=20, **overrides):
    kwargs = dict(prediction_length=prediction_length, patch_size=PS,
                  quantile_levels=(0.1, 0.5, 0.9), d_model=32, num_heads=4, mlp_hidden_dim=64,
                  block_schedule=("time", "time_variate"), block_size=4, context_length=64)
    kwargs.update(overrides)
    return create_timestral_model("5M", **kwargs)


def _seasonal_batch(n=8, context=60, horizon=20, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(context + horizon)
    full = np.stack([a * np.sin(2 * np.pi * (t + s) / 12) + o for s, a, o in
                     zip(rng.randint(0, 12, n), rng.uniform(1, 5, n), rng.uniform(-10, 10, n))])
    full = full.astype("float32")[..., None]
    return full[:, :context], full[:, context:]


def _frame(window, observed, patch, minimum_scale):
    """Mean and std of every observed value in [0, (patch + 1) * PS), two-pass, float64."""
    values = window[:(patch + 1) * PS][observed[:(patch + 1) * PS]].astype(np.float64)
    if values.size == 0:
        return 0.0, minimum_scale
    mean = values.mean()
    var = ((values - mean) ** 2).sum() / max(values.size - 1, 1)
    return mean, max(np.sqrt(var), minimum_scale)


def _numpy_loss(model, x, y, y_pred, anchor_only=False):
    """The D-169 objective from its definition: (B, T, V), (B, H, V), (B, H, V, Q).

    ``anchor_only`` is the control: every horizon patch scored in the last context
    patch's frame.
    """
    c = model.config
    tau = np.asarray(c.quantile_knots)
    total, count = 0.0, 0.0
    for i in range(x.shape[0]):
        for v in range(x.shape[2]):
            ctx = x[i, :, v].astype(np.float64)
            pad = -len(ctx) % PS
            ctx_obs = np.concatenate([np.zeros(pad, bool), np.isfinite(ctx)])
            ctx = np.concatenate([np.zeros(pad), np.nan_to_num(ctx)])
            tgt = y[i, :, v].astype(np.float64)
            window = np.concatenate([ctx, np.nan_to_num(tgt)])
            obs = np.concatenate([ctx_obs, np.isfinite(tgt)])
            last = len(ctx) // PS - 1
            loc0, scale0 = _frame(window, obs, last, c.minimum_scale)
            for step in range(len(tgt)):
                if not np.isfinite(tgt[step]):
                    continue
                patch = last if anchor_only else last + step // PS
                loc, scale = _frame(window, obs, patch, c.minimum_scale)
                q = np.arcsinh((y_pred[i, step, v].astype(np.float64) - loc0) / scale0)
                z = np.arcsinh((tgt[step] - loc) / scale)
                err = z - q
                total += np.maximum(tau * err, (tau - 1) * err).mean()
                count += 1
    return total / count


def test_call_is_predict_block_on_the_packed_rows():
    """One batch item per series, its variates its rows; left-padded to whole patches."""
    keras.utils.set_random_seed(0)
    model = _tiny()
    rng = np.random.RandomState(1)
    x = np.cumsum(rng.randn(2, 60, 3), axis=1).astype("float32")
    x[1, 10:14, 2] = np.nan
    got = keras.ops.convert_to_numpy(model(x))
    assert got.shape == (2, 20, 3, 3)
    rows = np.transpose(x, (0, 2, 1)).reshape(6, 60)
    observed = np.isfinite(rows)
    rows = np.pad(np.where(observed, rows, 0.0), ((0, 0), (4, 0)))
    observed = np.pad(observed, ((0, 0), (4, 0)))
    want = keras.ops.convert_to_numpy(model.predict_block(
        rows, observed, np.repeat(np.arange(2, dtype="int32"), 3), 3))[:, :20]
    np.testing.assert_array_equal(np.transpose(got, (0, 2, 1, 3)).reshape(6, 20, 3), want)
    univariate = keras.ops.convert_to_numpy(model(x[..., 0]))
    assert univariate.shape == (2, 20, 3)


def test_call_needs_a_static_batch():
    model = _tiny()
    with pytest.raises(ValueError, match="static batch size"):
        model._pack(keras.KerasTensor((None, 64, 1)))


def test_compute_loss_is_the_per_patch_frame_pinball():
    keras.utils.set_random_seed(0)
    model = _tiny()
    x, y = _seasonal_batch(n=3)
    x = np.concatenate([x, np.cumsum(np.random.RandomState(2).randn(3, 60, 1), axis=1)
                        .astype("float32")], axis=-1)
    y = np.concatenate([y, np.random.RandomState(3).randn(3, 20, 1).astype("float32")], axis=-1)
    x[0, 5:9, 1] = np.nan
    y[2, 3, 0] = np.nan
    y_pred = keras.ops.convert_to_numpy(model(x))
    got = float(keras.ops.convert_to_numpy(model.compute_loss(x, y, y_pred)))
    want = _numpy_loss(model, x, y, y_pred)
    # float32 work dtype: a few eps32 per stage of a ~10-stage chain.
    assert abs(got - want) <= 1e-5 * (1 + abs(want)), (got, want)
    anchor_only = _numpy_loss(model, x, y, y_pred, anchor_only=True)
    assert abs(anchor_only - want) > 1e3 * 1e-5 * (1 + abs(want)), "the frames must matter here"


def test_compute_loss_weights_series_by_sample_weight():
    keras.utils.set_random_seed(0)
    model = _tiny()
    x, y = _seasonal_batch(n=2)
    y_pred = model(x)
    first = float(model.compute_loss(x[:1], y[:1], y_pred[:1]))
    weighted = float(model.compute_loss(x, y, y_pred, sample_weight=np.array([1.0, 0.0])))
    assert weighted == pytest.approx(first, rel=1e-6)


@pytest.mark.parametrize("jit_compile", [False, True])
def test_stock_fit_memorises_one_batch(jit_compile):
    """Training bar (D-122 item 9): one fixed batch, beaten against its naive forecast.

    The forecaster (refine frame, as at inference) on the memorised batch must reach a
    fifth of the naive last-value error; the untrained model's loss must fall tenfold.
    """
    keras.utils.set_random_seed(0)
    model = _tiny()
    x, y = _seasonal_batch()
    model.compile(optimizer=keras.optimizers.Adam(3e-3), jit_compile=jit_compile)
    history = model.fit(x, y, batch_size=8, epochs=200, shuffle=False, verbose=0)
    losses = history.history["loss"]
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
    forecast = TimestralForecaster(model).forecast(
        x[..., 0], 20, symmetric_averaging=False, clip_nonnegative=False)
    mae = float(np.abs(forecast.median[:, 0] - y[..., 0]).mean())
    naive = float(np.abs(y - x[:, -1:]).mean())
    assert mae < 0.2 * naive, (mae, naive)
