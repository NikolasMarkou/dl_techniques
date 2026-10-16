"""The depth metrics the trainer compiles read the validity mask, not nothing.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-100). Measured at 561e02165:
a ``train_depth_anything`` run on MegaDepth logged ``abs_rel``, ``sq_rel``,
``rmse``, ``rmse_log`` and ``delta_1.25`` as exactly 0.0 in every train and
validation column while the loss moved. The trainer wrapped each metric in
``_DepthOnlyMetricWrapper``, which passed ``y_true[..., :1]`` on; the metrics
(since 72cad0cba) unpack ``[depth, mask]`` from ``y_true`` themselves, so the
mask they read was empty and no pixel counted.
"""

import numpy as np

from tests.numerics import reassociation_atol
from dl_techniques.metrics.depth_metrics import AbsRelMetric
from train.depth_anything.train_depth_anything import build_depth_metrics


def _batch():
    rng = np.random.default_rng(0)
    # MegaDepth's convention: depth normalized to [-1, 1], plus a mask with
    # one invalid pixel per row.
    depth = rng.uniform(-0.5, 0.9, size=(2, 4, 4, 1)).astype("float32")
    mask = np.ones_like(depth)
    mask[:, :, 0] = 0.0
    y_true = np.concatenate([depth, mask], axis=-1)
    y_pred = np.clip(depth + 0.1, -1.0, 1.0).astype("float32")
    return depth, mask, y_true, y_pred


def test_every_compiled_metric_reads_the_valid_pixels():
    depth, mask, y_true, y_pred = _batch()
    metrics = build_depth_metrics()
    assert [m.name for m in metrics] == [
        "abs_rel", "sq_rel", "rmse", "rmse_log", "delta_1.25"
    ]
    # Every prediction is off by 0.1 and most ratios are inside 1.25, so each
    # metric is strictly positive once a valid pixel is counted.
    for metric in metrics:
        metric.update_state(y_true, y_pred)
        assert float(metric.result()) > 0.0, metric.name


def test_abs_rel_is_the_masked_mean_relative_error():
    depth, mask, y_true, y_pred = _batch()
    abs_rel = build_depth_metrics()[0]
    abs_rel.update_state(y_true, y_pred)
    # The metric shifts [-1, 1] to [0, 1] and drops ground truth below
    # min_depth (0.05) after the shift.
    dt = (depth.astype(np.float64) + 1.0) / 2.0
    dp = (y_pred.astype(np.float64) + 1.0) / 2.0
    valid = (dt > AbsRelMetric().min_depth) * mask
    expected = float(np.sum(np.abs(dp - dt) / dt * valid) / np.sum(valid))
    # Two float32 sums over the pixels, compared with a float64 reference.
    atol = reassociation_atol([depth.size], 2, abs(expected))
    np.testing.assert_allclose(
        float(abs_rel.result()), expected, rtol=0.0, atol=atol
    )
