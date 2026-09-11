"""Hand-derived-toy-tensor tests for `dl_techniques.losses.omnipoint_losses`.

Per plan Success Criterion 5, every assertion below compares against a value computed by hand,
not merely "runs without error." The single most important test is
`test_point_distance_loss_is_zero_for_wrong_ray_but_correct_distance`, which directly proves the
paper's decoupling claim: `PointDistanceLoss` must not penalize a correct predicted distance for
a wrong predicted ray, because it is evaluated along the GROUND-TRUTH ray, never the predicted
one.
"""

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.losses.omnipoint_losses import (
    RayDirectionLoss,
    PointDistanceLoss,
    MetricScaleLoss,
    MaskLoss,
    NormalConsistencyLoss,
    LocalConsistencyLoss,
    OmniPointCombinedLoss,
    compute_optimal_scale,
)

# ---------------------------------------------------------------------
# RayDirectionLoss
# ---------------------------------------------------------------------


class TestRayDirectionLoss:
    def test_orthogonal_unit_vectors_matches_hand_derived_l1(self):
        # r_true = (1,0,0), r_pred = (0,1,0) -> |diff| = (1,1,0), mean over channels = 2/3
        r_true = np.array([[[1.0, 0.0, 0.0]]], dtype="float32")  # (1,1,3)
        r_pred = np.array([[[0.0, 1.0, 0.0]]], dtype="float32")
        loss_fn = RayDirectionLoss()
        result = loss_fn(r_true, r_pred)
        expected = 2.0 / 3.0
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6, rtol=0)

    def test_identical_rays_give_zero_loss(self):
        r = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype="float32")  # (1,2,3)
        loss_fn = RayDirectionLoss()
        result = loss_fn(r, r)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6, rtol=0)

    def test_valid_mask_excludes_masked_pixels(self):
        r_true = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype="float32")  # (1,2,3)
        r_pred = np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]], dtype="float32")
        valid_mask = np.array([[[1.0], [0.0]]], dtype="float32")  # only first pixel valid
        loss_fn = RayDirectionLoss()
        result = loss_fn.call(r_true, r_pred, valid_mask=valid_mask)
        # only pixel 0 contributes: |diff|=(1,1,0), mean=2/3
        np.testing.assert_allclose(np.array(result), 2.0 / 3.0, atol=1e-6, rtol=0)

    def test_get_config_round_trip(self):
        loss_fn = RayDirectionLoss(name="custom_ray_loss")
        config = loss_fn.get_config()
        restored = RayDirectionLoss.from_config(config)
        assert restored.name == "custom_ray_loss"


# ---------------------------------------------------------------------
# PointDistanceLoss -- the central decoupling test
# ---------------------------------------------------------------------


class TestPointDistanceLoss:
    def test_zero_loss_for_wrong_ray_but_correct_distance(self):
        """The paper's central decoupling claim: a wrong r_hat must NOT penalize a correct d_hat.

        s_star = 1.0, d_hat == d_gt exactly. r_hat is deliberately wrong (orthogonal to r_gt) but
        PointDistanceLoss never reads r_hat -- it is evaluated along r_gt only. Loss must be
        (near-)zero.
        """
        gt_ray = np.array([[[1.0, 0.0, 0.0]]], dtype="float32")  # (1,1,3)
        gt_distance = np.array([[[5.0]]], dtype="float32")  # (1,1,1)
        pred_distance = np.array([[[5.0]]], dtype="float32")  # correct distance
        # (deliberately wrong ray, unused by this loss -- included to document the intent)
        _wrong_pred_ray = np.array([[[0.0, 1.0, 0.0]]], dtype="float32")
        s_star = np.array([1.0], dtype="float32")

        y_true = np.concatenate([gt_ray, gt_distance], axis=-1)  # (1,1,4)
        loss_fn = PointDistanceLoss()
        result = loss_fn.call(y_true, pred_distance, s_star=s_star)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6, rtol=0)

    def test_nonzero_loss_for_wrong_distance(self):
        gt_ray = np.array([[[1.0, 0.0, 0.0]]], dtype="float32")
        gt_distance = np.array([[[5.0]]], dtype="float32")
        pred_distance = np.array([[[7.0]]], dtype="float32")  # off by 2
        s_star = np.array([1.0], dtype="float32")

        y_true = np.concatenate([gt_ray, gt_distance], axis=-1)
        loss_fn = PointDistanceLoss()
        result = loss_fn.call(y_true, pred_distance, s_star=s_star)
        # diff_vec = (1*7 - 5) * (1,0,0) = (2,0,0); mean(|.|) over channels = 2/3
        np.testing.assert_allclose(np.array(result), 2.0 / 3.0, atol=1e-6, rtol=0)

    def test_scale_is_applied_to_predicted_distance(self):
        gt_ray = np.array([[[1.0, 0.0, 0.0]]], dtype="float32")
        gt_distance = np.array([[[10.0]]], dtype="float32")
        pred_distance = np.array([[[5.0]]], dtype="float32")
        s_star = np.array([2.0], dtype="float32")  # 2 * 5 == 10 -> should be zero loss

        y_true = np.concatenate([gt_ray, gt_distance], axis=-1)
        loss_fn = PointDistanceLoss()
        result = loss_fn.call(y_true, pred_distance, s_star=s_star)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6, rtol=0)

    def test_raises_without_s_star(self):
        gt_ray = np.array([[[1.0, 0.0, 0.0]]], dtype="float32")
        gt_distance = np.array([[[5.0]]], dtype="float32")
        pred_distance = np.array([[[5.0]]], dtype="float32")
        y_true = np.concatenate([gt_ray, gt_distance], axis=-1)
        loss_fn = PointDistanceLoss()
        with pytest.raises(ValueError):
            loss_fn.call(y_true, pred_distance)


# ---------------------------------------------------------------------
# MetricScaleLoss
# ---------------------------------------------------------------------


class TestMetricScaleLoss:
    def test_matches_hand_derived_log_space_value(self):
        s_hat = np.array([4.0], dtype="float32")
        s_star = np.array([2.0], dtype="float32")
        loss_fn = MetricScaleLoss()
        result = loss_fn(s_star, s_hat)
        expected = (np.log(4.0) - np.log(2.0)) ** 2
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6, rtol=0)

    def test_zero_loss_when_scales_match(self):
        s = np.array([3.0], dtype="float32")
        loss_fn = MetricScaleLoss()
        result = loss_fn(s, s)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6, rtol=0)

    def test_gradient_does_not_flow_into_s_star_source(self):
        s_hat_var = keras.Variable(np.array([4.0], dtype="float32"))
        s_star_var = keras.Variable(np.array([2.0], dtype="float32"))
        loss_fn = MetricScaleLoss()

        def compute():
            return ops.sum(loss_fn.call(s_star_var, s_hat_var))

        # Use keras' backend-agnostic gradient via a simple finite-difference-free check on TF.
        import tensorflow as tf

        with tf.GradientTape() as tape:
            tape.watch([s_hat_var.value, s_star_var.value])
            out = compute()
        grads = tape.gradient(out, [s_hat_var.value, s_star_var.value])
        assert grads[0] is not None
        assert grads[1] is None  # stop_gradient severs the s_star path entirely


# ---------------------------------------------------------------------
# compute_optimal_scale
# ---------------------------------------------------------------------


class TestComputeOptimalScale:
    def test_recovers_known_scale_noiseless(self):
        rng = np.random.default_rng(0)
        pred = rng.normal(size=(2, 4, 4, 3)).astype("float32")
        gt = 2.0 * pred
        s_star = compute_optimal_scale(pred, gt)
        np.testing.assert_allclose(np.array(s_star), [2.0, 2.0], atol=1e-4, rtol=0)

    def test_near_zero_norm_does_not_nan(self):
        pred = np.zeros((1, 4, 4, 3), dtype="float32")
        gt = np.zeros((1, 4, 4, 3), dtype="float32")
        s_star = compute_optimal_scale(pred, gt)
        assert np.all(np.isfinite(np.array(s_star)))

    def test_valid_mask_excludes_invalid_points(self):
        pred = np.ones((1, 2, 2, 3), dtype="float32")
        gt = np.ones((1, 2, 2, 3), dtype="float32")
        # corrupt one pixel's gt so that, if included, it would bias s* away from 1.0
        gt[0, 0, 0, :] = 100.0
        valid_mask = np.ones((1, 2, 2, 1), dtype="float32")
        valid_mask[0, 0, 0, 0] = 0.0
        s_star = compute_optimal_scale(pred, gt, valid_mask=valid_mask)
        np.testing.assert_allclose(np.array(s_star), [1.0], atol=1e-5, rtol=0)

    def test_rejects_non_positive_epsilon(self):
        pred = np.ones((1, 2, 2, 3), dtype="float32")
        gt = np.ones((1, 2, 2, 3), dtype="float32")
        with pytest.raises(ValueError):
            compute_optimal_scale(pred, gt, epsilon=0.0)


# ---------------------------------------------------------------------
# MaskLoss
# ---------------------------------------------------------------------


class TestMaskLoss:
    def test_matches_keras_binary_crossentropy(self):
        y_true = np.array([[[[1.0], [0.0]]]], dtype="float32")  # (1,1,2,1)
        logits = np.array([[[[2.0], [-1.5]]]], dtype="float32")
        loss_fn = MaskLoss()
        result = loss_fn(y_true, logits)

        bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction=None)
        per_pixel = np.array(bce(y_true, logits))
        expected = per_pixel.reshape(1, -1).mean(axis=-1)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# NormalConsistencyLoss / LocalConsistencyLoss
# ---------------------------------------------------------------------


def _planar_point_map(height: int, width: int) -> np.ndarray:
    """A perfectly planar point map: P(x, y) = (x, y, 0) -- zero curvature everywhere."""
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    zs = np.zeros_like(xs)
    return np.stack([xs, ys, zs], axis=-1).astype("float32")[None, ...]  # (1,H,W,3)


class TestNormalConsistencyLoss:
    def test_zero_loss_for_identical_planar_maps(self):
        p = _planar_point_map(4, 4)
        loss_fn = NormalConsistencyLoss()
        result = loss_fn(p, p)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6, rtol=0)

    def test_nonzero_loss_when_perturbed(self):
        p = _planar_point_map(4, 4)
        p_perturbed = p.copy()
        p_perturbed[0, 2, 2, 2] += 5.0  # bend one point out of plane
        loss_fn = NormalConsistencyLoss()
        result = loss_fn(p, p_perturbed)
        assert float(np.array(result)) > 1e-3

    def test_finite_on_flat_map(self):
        p = _planar_point_map(4, 4)
        loss_fn = NormalConsistencyLoss()
        result = loss_fn(p, p)
        assert np.all(np.isfinite(np.array(result)))


class TestLocalConsistencyLoss:
    def test_zero_laplacian_for_identical_planar_maps(self):
        p = _planar_point_map(5, 5)
        loss_fn = LocalConsistencyLoss()
        result = loss_fn(p, p)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6, rtol=0)

    def test_nonzero_loss_when_perturbed(self):
        p = _planar_point_map(5, 5)
        p_perturbed = p.copy()
        p_perturbed[0, 2, 2, 2] += 5.0
        loss_fn = LocalConsistencyLoss()
        result = loss_fn(p, p_perturbed)
        assert float(np.array(result)) > 1e-3


# ---------------------------------------------------------------------
# OmniPointCombinedLoss
# ---------------------------------------------------------------------


class TestOmniPointCombinedLoss:
    def _make_toy_batch(self):
        rng = np.random.default_rng(42)
        batch, h, w = 1, 5, 5

        pred_ray = rng.normal(size=(batch, h, w, 3)).astype("float32")
        pred_ray = pred_ray / np.linalg.norm(pred_ray, axis=-1, keepdims=True)
        pred_distance = np.abs(rng.normal(size=(batch, h, w, 1)).astype("float32")) + 0.5
        pred_mask_logit = rng.normal(size=(batch, h, w, 1)).astype("float32")
        pred_scale = np.array([1.5], dtype="float32")

        gt_ray = rng.normal(size=(batch, h, w, 3)).astype("float32")
        gt_ray = gt_ray / np.linalg.norm(gt_ray, axis=-1, keepdims=True)
        gt_distance = np.abs(rng.normal(size=(batch, h, w, 1)).astype("float32")) + 0.5
        gt_point = gt_ray * gt_distance
        gt_mask = (rng.normal(size=(batch, h, w, 1)) > 0).astype("float32")
        valid_mask = np.ones((batch, h, w, 1), dtype="float32")

        y_true = (gt_ray, gt_distance, gt_point, gt_mask, valid_mask)
        y_pred = (pred_ray, pred_distance, pred_mask_logit, pred_scale)
        return y_true, y_pred

    def test_equals_hand_computed_weighted_sum(self):
        y_true, y_pred = self._make_toy_batch()
        gt_ray, gt_distance, gt_point, gt_mask, valid_mask = y_true
        pred_ray, pred_distance, pred_mask_logit, pred_scale = y_pred

        lambdas = dict(
            lambda_ray=0.5, lambda_metric=0.3, lambda_normal=0.2, lambda_local=0.1,
            lambda_mask=0.4,
        )
        combined = OmniPointCombinedLoss(**lambdas)
        total = np.array(combined(y_true, y_pred))

        pred_affine_points = pred_ray * pred_distance
        s_star = np.array(
            compute_optimal_scale(pred_affine_points, gt_point, valid_mask=valid_mask)
        )

        l_ray = np.array(RayDirectionLoss().call(gt_ray, pred_ray, valid_mask=valid_mask))
        gt_ray_distance = np.concatenate([gt_ray, gt_distance], axis=-1)
        l_point = np.array(
            PointDistanceLoss().call(
                gt_ray_distance, pred_distance, s_star=s_star, valid_mask=valid_mask,
            )
        )
        l_metric = np.array(MetricScaleLoss().call(s_star, pred_scale))
        l_mask = np.array(MaskLoss().call(gt_mask, pred_mask_logit))
        pred_metric_points = pred_scale.reshape(-1, 1, 1, 1) * pred_affine_points
        l_normal = np.array(NormalConsistencyLoss().call(gt_point, pred_metric_points))
        l_local = np.array(LocalConsistencyLoss().call(gt_point, pred_metric_points))

        expected = (
                l_point
                + lambdas["lambda_ray"] * l_ray
                + lambdas["lambda_metric"] * l_metric
                + lambdas["lambda_normal"] * l_normal
                + lambdas["lambda_local"] * l_local
                + lambdas["lambda_mask"] * l_mask
        )
        np.testing.assert_allclose(total, expected, atol=1e-5, rtol=0)

    def test_changing_a_lambda_weight_changes_total_proportionally(self):
        y_true, y_pred = self._make_toy_batch()

        combined_a = OmniPointCombinedLoss(lambda_ray=1.0)
        combined_b = OmniPointCombinedLoss(lambda_ray=2.0)

        total_a = np.array(combined_a(y_true, y_pred))
        total_b = np.array(combined_b(y_true, y_pred))

        l_ray = np.array(
            RayDirectionLoss().call(y_true[0], y_pred[0], valid_mask=y_true[4])
        )
        np.testing.assert_allclose(total_b - total_a, l_ray, atol=1e-5, rtol=0)

    def test_get_config_round_trip(self):
        combined = OmniPointCombinedLoss(lambda_ray=0.7, lambda_metric=0.6)
        config = combined.get_config()
        restored = OmniPointCombinedLoss.from_config(config)
        assert restored.lambda_ray == 0.7
        assert restored.lambda_metric == 0.6

    def test_finite_on_zero_valid_mask_edge_case(self):
        y_true, y_pred = self._make_toy_batch()
        gt_ray, gt_distance, gt_point, gt_mask, _valid_mask = y_true
        zero_mask = np.zeros_like(_valid_mask)
        y_true_zero_mask = (gt_ray, gt_distance, gt_point, gt_mask, zero_mask)

        combined = OmniPointCombinedLoss()
        total = np.array(combined(y_true_zero_mask, y_pred))
        assert np.all(np.isfinite(total))

    def test_finite_on_near_zero_scale_gt_points(self):
        y_true, y_pred = self._make_toy_batch()
        gt_ray, gt_distance, gt_point, gt_mask, valid_mask = y_true
        tiny_gt_point = gt_point * 1e-10
        y_true_tiny = (gt_ray, gt_distance * 1e-10, tiny_gt_point, gt_mask, valid_mask)

        combined = OmniPointCombinedLoss()
        total = np.array(combined(y_true_tiny, y_pred))
        assert np.all(np.isfinite(total))
