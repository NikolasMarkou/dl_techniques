"""The depth metrics the trainer compiles read the validity mask, not nothing.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-100). Measured at 561e02165:
a ``train_depth_anything`` run on MegaDepth logged ``abs_rel``, ``sq_rel``,
``rmse``, ``rmse_log`` and ``delta_1.25`` as exactly 0.0 in every train and
validation column while the loss moved. The trainer wrapped each metric in
``_DepthOnlyMetricWrapper``, which passed ``y_true[..., :1]`` on; the metrics
(since 72cad0cba) unpack ``[depth, mask]`` from ``y_true`` themselves, so the
mask they read was empty and no pixel counted.

Step 13.5 (D-108): the target is per-map normalized disparity in ``[0, 1]`` and
the loss leaves the prediction's scale and shift free, so each metric scores
the prediction after per-image least-squares scale-and-shift alignment.
"""

import numpy as np

from tests.numerics import reassociation_atol
from dl_techniques.metrics.depth_metrics import AbsRelMetric
from train.depth_anything.train_depth_anything import build_depth_metrics


def _batch():
    rng = np.random.default_rng(0)
    # MegaDepth's convention: disparity normalized to [0, 1], plus a mask with
    # one invalid column.
    target = rng.uniform(0.1, 0.9, size=(2, 4, 4, 1)).astype("float32")
    mask = np.ones_like(target)
    mask[:, :, 0] = 0.0
    y_true = np.concatenate([target, mask], axis=-1)
    # An affine copy of the target (which alignment removes) plus an error it
    # cannot remove.
    wobble = 0.05 * rng.standard_normal(target.shape)
    y_pred = (3.0 * target + 2.0 + wobble).astype("float32")
    return target, mask, y_true, y_pred


def _aligned_reference(target, mask, y_pred):
    """Per-image float64 least-squares fit of ``s * pred + t`` to the target."""
    out = np.empty(y_pred.shape, dtype=np.float64)
    for i in range(y_pred.shape[0]):
        valid = mask[i] > 0
        p = y_pred[i][valid].astype(np.float64)
        a = np.stack([p, np.ones_like(p)], axis=1)
        (s, t), *_ = np.linalg.lstsq(a, target[i][valid].astype(np.float64), rcond=None)
        out[i] = s * y_pred[i] + t
    return out


def test_every_compiled_metric_reads_the_valid_pixels():
    target, mask, y_true, y_pred = _batch()
    metrics = build_depth_metrics()
    assert [m.name for m in metrics] == [
        "abs_rel", "sq_rel", "rmse", "rmse_log", "delta_1.25"
    ]
    # The non-affine error survives alignment, and most ratios are inside
    # 1.25, so each metric is strictly positive once a valid pixel is counted.
    for metric in metrics:
        metric.update_state(y_true, y_pred)
        assert float(metric.result()) > 0.0, metric.name


def test_abs_rel_is_the_masked_mean_relative_error_of_the_aligned_prediction():
    target, mask, y_true, y_pred = _batch()
    abs_rel = build_depth_metrics()[0]
    abs_rel.update_state(y_true, y_pred)
    dt = target.astype(np.float64)
    dp = _aligned_reference(target, mask, y_pred)
    # The metric drops ground truth below min_depth (0.05).
    valid = (dt > AbsRelMetric().min_depth) * mask
    expected = float(np.sum(np.abs(dp - dt) / dt * valid) / np.sum(valid))
    # The float32 alignment solves a 2x2 system from five sums over the pixels
    # (cancellation in the determinant), then two more sums: allow 64 sums'
    # worth of float32 rounding against the float64 reference.
    atol = reassociation_atol([target.size], 64, abs(expected))
    np.testing.assert_allclose(
        float(abs_rel.result()), expected, rtol=0.0, atol=atol
    )
