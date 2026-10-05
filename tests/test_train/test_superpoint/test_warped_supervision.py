"""Guards for warped-view detector supervision (plan-2026-10-05_warped-detector).

The joint trainer supervises the detector head on BOTH views: the warped label
is the clean keypoints warped through H and re-encoded. These tests pin:

- ``_warped_grid_label``: identity H reproduces the clean label; empty /
  fully-warped-out inputs yield all-dustbin; values stay in range.
- ``_warp_points_tf`` / ``_grid_labels_tf``: the in-graph ports agree with the
  numpy references (``warp_points`` / ``keypoints_to_grid_labels``), including
  padding, out-of-bounds and nearest-to-center tie-breaks.
- ``reprojection_matches``: the viz repeatability helper (perfect shift ->
  1.0, empty -> 0.0, strict threshold boundary).
- ``SuperPointBackboneCheckpoint``: the best file reloads as a ``SuperPoint``
  backbone (the wrapper does not -- D-001).
- ``SuperPointJointModel.train_step``: the detector term is the mean of BOTH
  views (a revert to clean-only supervision goes RED).
"""

import keras
import numpy as np
import pytest

from dl_techniques.datasets.synthetic_shapes import (
    DEFAULT_CELL,
    keypoints_to_grid_labels,
)
from dl_techniques.losses.superpoint_loss import (
    SuperPointDescriptorLoss,
    SuperPointDetectorLoss,
)
from dl_techniques.models.vision.keypoints.superpoint import create_superpoint
from dl_techniques.models.vision.keypoints.superpoint.model import SuperPoint
from dl_techniques.utils.homography import sample_homography, warp_points
from train.common.keypoint_viz import reprojection_matches
from train.superpoint.train_superpoint import (
    SuperPointBackboneCheckpoint,
    SuperPointConfig,
    SuperPointJointModel,
    _grid_labels_tf,
    _warp_points_tf,
    _warped_grid_label,
    create_dataset,
)


H = W = 64
CELL = DEFAULT_CELL  # 8
HC = H // CELL
DUSTBIN = CELL * CELL  # 64


def _random_points(rng, n, h=H, w=W):
    return rng.uniform(0, min(h, w), size=(n, 2)).astype(np.float32)


# ---------------------------------------------------------------------
# _warped_grid_label
# ---------------------------------------------------------------------


class TestWarpedGridLabel:

    def test_identity_reproduces_clean_label(self):
        rng = np.random.default_rng(0)
        kps = _random_points(rng, 30)
        clean = keypoints_to_grid_labels(kps, H, W, cell=CELL)
        warped = _warped_grid_label(kps, np.eye(3, dtype=np.float32), H, W, CELL)
        assert np.array_equal(warped, clean)

    def test_empty_is_all_dustbin(self):
        out = _warped_grid_label(
            np.zeros((0, 2), dtype=np.float32), np.eye(3, dtype=np.float32), H, W, CELL
        )
        assert out.shape == (HC, HC)
        assert (out == DUSTBIN).all()

    def test_fully_warped_out_is_all_dustbin(self):
        rng = np.random.default_rng(1)
        kps = _random_points(rng, 20)
        far = np.array([[1, 0, 10_000], [0, 1, 10_000], [0, 0, 1]], dtype=np.float32)
        out = _warped_grid_label(kps, far, H, W, CELL)
        assert (out == DUSTBIN).all()

    def test_values_in_range(self):
        rng = np.random.default_rng(2)
        kps = _random_points(rng, 50)
        h_mat = sample_homography((H, W), seed=3)
        out = _warped_grid_label(kps, h_mat, H, W, CELL)
        assert out.shape == (HC, HC)
        assert out.dtype == np.int32
        assert set(np.unique(out)).issubset(set(range(DUSTBIN + 1)))


# ---------------------------------------------------------------------
# tf ports vs numpy references
# ---------------------------------------------------------------------


class TestTfParity:

    def test_warp_points_identity_and_translation(self):
        rng = np.random.default_rng(4)
        kps = _random_points(rng, 10)
        for h_mat in (
            np.eye(3, dtype=np.float32),
            np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], dtype=np.float32),
        ):
            ref = warp_points(kps, h_mat)
            got, ok = _warp_points_tf(
                keras.ops.convert_to_tensor(kps), keras.ops.convert_to_tensor(h_mat)
            )
            got, ok = keras.ops.convert_to_numpy(got), keras.ops.convert_to_numpy(ok)
            assert ok.all()
            assert np.allclose(got, ref, atol=1e-4)

    def test_warp_points_flags_horizon(self):
        # A homography collapsing w -> 0 must flag invalid, never NaN.
        kps = np.array([[10.0, 10.0], [30.0, 20.0]], dtype=np.float32)
        h_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        got, ok = _warp_points_tf(
            keras.ops.convert_to_tensor(kps), keras.ops.convert_to_tensor(h_mat)
        )
        got, ok = keras.ops.convert_to_numpy(got), keras.ops.convert_to_numpy(ok)
        assert not ok.any()
        assert np.isfinite(got).all()

    def test_grid_labels_parity_with_padding_and_ties(self):
        rng = np.random.default_rng(5)
        real = _random_points(rng, 6)
        # A nearest-to-center tie in cell (1, 2): both at dist 0.5, first wins.
        tie = np.array(
            [[2 * CELL + 3.0, 1 * CELL + 3.0], [2 * CELL + 4.0, 1 * CELL + 4.0]],
            dtype=np.float32,
        )
        oor = np.array([[-5.0, -5.0], [1000.0, 1000.0]], dtype=np.float32)
        pts = np.concatenate([real, tie, oor], axis=0)  # K = 10, all real
        ref = keypoints_to_grid_labels(pts, H, W, cell=CELL)

        padded = np.full((12, 2), -1.0, dtype=np.float32)
        padded[: len(pts)] = pts
        got = _grid_labels_tf(
            keras.ops.convert_to_tensor(padded),
            keras.ops.convert_to_tensor(np.int32(len(pts))),
            keras.ops.convert_to_tensor(np.ones(len(padded), dtype=bool)),
            H, W, CELL,
        )
        assert np.array_equal(keras.ops.convert_to_numpy(got), ref)

    def test_grid_labels_count_truncates(self):
        rng = np.random.default_rng(6)
        pts = _random_points(rng, 8)
        padded = np.full((8, 2), -1.0, dtype=np.float32)
        padded[:] = pts
        # Only the first 3 points participate.
        ref = keypoints_to_grid_labels(pts[:3], H, W, cell=CELL)
        got = _grid_labels_tf(
            keras.ops.convert_to_tensor(padded),
            keras.ops.convert_to_tensor(np.int32(3)),
            keras.ops.convert_to_tensor(np.ones(8, dtype=bool)),
            H, W, CELL,
        )
        assert np.array_equal(keras.ops.convert_to_numpy(got), ref)


# ---------------------------------------------------------------------
# reprojection_matches
# ---------------------------------------------------------------------


class TestReprojectionMatches:

    def test_perfect_shift_scores_one(self):
        src = np.array([[10.0, 10.0], [20.0, 30.0]], dtype=np.float32)
        dst = src + np.array([1.0, 0.5], dtype=np.float32)
        matches, scores, rep = reprojection_matches(
            src, np.ones(2, dtype=bool), dst, thresh=3.0)
        assert rep == pytest.approx(1.0)
        assert list(matches) == [0, 1]
        assert (scores > 0.5).all()

    def test_empty_dst_is_zero(self):
        src = np.array([[10.0, 10.0]], dtype=np.float32)
        _, _, rep = reprojection_matches(
            src, np.ones(1, dtype=bool), np.zeros((0, 2), dtype=np.float32))
        assert rep == 0.0

    def test_no_valid_src_is_zero(self):
        src = np.array([[10.0, 10.0]], dtype=np.float32)
        matches, _, rep = reprojection_matches(
            src, np.zeros(1, dtype=bool), src)
        assert rep == 0.0
        assert list(matches) == [-1]

    def test_threshold_is_strict(self):
        src = np.array([[0.0, 0.0]], dtype=np.float32)
        dst = np.array([[3.0, 0.0]], dtype=np.float32)  # dist == thresh
        matches, _, rep = reprojection_matches(
            src, np.ones(1, dtype=bool), dst, thresh=3.0)
        assert rep == 0.0
        assert list(matches) == [-1]

    def test_warp_out_excluded_from_denominator(self):
        src = np.array([[10.0, 10.0], [50.0, 50.0]], dtype=np.float32)
        valid = np.array([True, False])
        dst = np.array([[10.2, 10.1]], dtype=np.float32)
        _, _, rep = reprojection_matches(src, valid, dst, thresh=3.0)
        assert rep == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Backbone checkpoint reloads (D-001)
# ---------------------------------------------------------------------


class TestBackboneCheckpoint:

    def test_best_file_reloads_as_superpoint(self, tmp_path):
        backbone = create_superpoint("tiny", input_shape=(H, W, 1))
        backbone.build((None, H, W, 1))
        model = SuperPointJointModel(superpoint=backbone)
        path = str(tmp_path / "best_model.keras")
        cb = SuperPointBackboneCheckpoint(path, monitor="loss")
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_epoch_end(0, {"loss": 5.0})
        cb.on_epoch_end(1, {"loss": 6.0})  # worse: no overwrite
        assert cb.best == pytest.approx(5.0)
        assert cb.best_epoch == 0
        reloaded = keras.saving.load_model(path, compile=False)
        assert isinstance(reloaded, SuperPoint)
        assert np.array_equal(
            np.asarray(reloaded.get_weights()[0]), np.asarray(backbone.get_weights()[0]))


# ---------------------------------------------------------------------
# Dual-view detector supervision
# ---------------------------------------------------------------------


class TestDualViewDetectorLoss:

    def test_train_step_detector_term_is_mean_of_both_views(self):
        config = SuperPointConfig(
            input_size=H, batch_size=2, variant="tiny",
            steps_per_epoch=2, epochs=1,
        )
        ds = create_dataset(config)
        x, y = next(iter(ds.take(1)))
        assert "warped_keypoints" in y and "homography" in y

        backbone = create_superpoint("tiny", input_shape=(H, W, 1))
        backbone.build((None, H, W, 1))
        model = SuperPointJointModel(superpoint=backbone)
        model.compile(optimizer=keras.optimizers.SGD(1e-3), jit_compile=False)

        # Expected value from the pre-step weights (deterministic: no dropout).
        o1 = backbone(x, training=True)
        o2 = backbone(y["warped_image"], training=True)
        fn = SuperPointDetectorLoss()
        expected = (
            float(keras.ops.mean(fn(y["keypoints"], o1["keypoints"])))
            + float(keras.ops.mean(fn(y["warped_keypoints"], o2["keypoints"])))
        ) / 2.0

        out = model.train_step((x, y))
        assert float(out["detector_loss"]) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------
# Joint test_step execution proof (frozen-override record)
# ---------------------------------------------------------------------


class TestJointTestStep:

    def test_test_step_reports_all_losses_and_matches_manual(self):
        """Execution proof for the frozen ``SuperPointJointModel.test_step``.

        The override mirrors ``train_step`` without gradients: stock
        ``test_step`` cannot run here since ``compile()`` takes no ``loss=``
        by design and the objective needs both views. Eager at the
        trainer's own ``jit_compile=False`` — XLA is unreachable
        (measured: ``ResizeBicubic`` has no ``XLA_GPU_JIT`` kernel), so a
        jit variant would pin a falsehood. A revert to a loss-less or
        single-view ``test_step`` goes RED on the value asserts.
        """
        config = SuperPointConfig(
            input_size=H, batch_size=2, variant="tiny",
            steps_per_epoch=2, epochs=1,
        )
        ds = create_dataset(config)
        x, y = next(iter(ds.take(1)))

        backbone = create_superpoint("tiny", input_shape=(H, W, 1))
        backbone.build((None, H, W, 1))
        model = SuperPointJointModel(superpoint=backbone)
        model.compile(optimizer=keras.optimizers.SGD(1e-3), jit_compile=False)

        # Expected values from the pre-step weights (deterministic: no dropout).
        o1 = backbone(x, training=False)
        o2 = backbone(y["warped_image"], training=False)
        det_fn = SuperPointDetectorLoss()
        expected_det = (
            float(keras.ops.mean(det_fn(y["keypoints"], o1["keypoints"])))
            + float(keras.ops.mean(
                det_fn(y["warped_keypoints"], o2["keypoints"])))
        ) / 2.0
        desc_fn = SuperPointDescriptorLoss()
        expected_desc = float(keras.ops.mean(desc_fn.compute(
            model._coarse_descriptors(o1["descriptors"]),
            model._coarse_descriptors(o2["descriptors"]),
            y["correspondence"],
        )))

        out = model.test_step((x, y))
        assert set(out) == {"loss", "detector_loss", "descriptor_loss"}
        assert all(np.isfinite(float(out[k])) for k in out)
        assert float(out["detector_loss"]) == pytest.approx(expected_det, abs=1e-4)
        assert float(out["descriptor_loss"]) == pytest.approx(expected_desc, abs=1e-4)
