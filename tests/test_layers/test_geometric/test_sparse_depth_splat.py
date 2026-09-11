import pytest
import numpy as np
import keras

from dl_techniques.layers.geometric.sparse_depth_splat import SparseDepthSplat


class TestSparseDepthSplat:
    """Behavioral test suite for SparseDepthSplat.

    Scatter-free densification of an already grid-aligned sparse depth map
    (see the module docstring's input-contract / D-003 build-time finding).
    """

    def test_single_point_densifies_to_a_gaussian_bump(self):
        h, w = 16, 16
        depth = np.zeros((1, h, w, 1), dtype="float32")
        mask = np.zeros((1, h, w, 1), dtype="float32")
        depth[0, 8, 8, 0] = 5.0
        mask[0, 8, 8, 0] = 1.0

        layer = SparseDepthSplat(kernel_size=(9, 9), sigma=2.0)
        dense_depth, dense_confidence = layer((depth, mask))
        dense_depth_np = keras.ops.convert_to_numpy(dense_depth)
        dense_confidence_np = keras.ops.convert_to_numpy(dense_confidence)

        # The center pixel (the sparse sample's own location) should recover
        # close to the original depth value (normalized splat: value/weight).
        assert dense_depth_np[0, 8, 8, 0] == pytest.approx(5.0, abs=1e-3)
        # Confidence decays away from the point, and is strictly positive at
        # the center.
        assert dense_confidence_np[0, 8, 8, 0] > 0.0
        assert dense_confidence_np[0, 0, 0, 0] < dense_confidence_np[0, 8, 8, 0]
        # A neighboring pixel within the Gaussian's support gets a nonzero,
        # smaller-than-center bump.
        assert 0.0 < dense_depth_np[0, 9, 8, 0] <= dense_depth_np[0, 8, 8, 0]

    def test_all_zero_input_produces_all_zero_output_no_nan(self):
        h, w = 12, 12
        depth = np.zeros((2, h, w, 1), dtype="float32")
        mask = np.zeros((2, h, w, 1), dtype="float32")

        layer = SparseDepthSplat()
        dense_depth, dense_confidence = layer((depth, mask))
        dense_depth_np = keras.ops.convert_to_numpy(dense_depth)
        dense_confidence_np = keras.ops.convert_to_numpy(dense_confidence)

        assert np.isfinite(dense_depth_np).all()
        assert np.isfinite(dense_confidence_np).all()
        np.testing.assert_allclose(dense_depth_np, 0.0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(dense_confidence_np, 0.0, atol=1e-7, rtol=0)

    def test_output_confidence_is_finite_and_bounded_in_0_1(self):
        h, w = 16, 16
        rng = np.random.default_rng(0)
        mask = (rng.uniform(size=(2, h, w, 1)) < 0.05).astype("float32")
        depth = mask * rng.uniform(1.0, 10.0, size=(2, h, w, 1)).astype("float32")

        layer = SparseDepthSplat()
        dense_depth, dense_confidence = layer((depth, mask))
        dense_depth_np = keras.ops.convert_to_numpy(dense_depth)
        dense_confidence_np = keras.ops.convert_to_numpy(dense_confidence)

        assert np.isfinite(dense_depth_np).all()
        assert np.isfinite(dense_confidence_np).all()
        assert dense_confidence_np.min() >= 0.0
        assert dense_confidence_np.max() <= 1.0 + 1e-6

    def test_get_config_from_config_round_trip(self):
        layer = SparseDepthSplat(kernel_size=(5, 5), sigma=1.5, epsilon=1e-6)
        config = layer.get_config()
        restored = SparseDepthSplat.from_config(config)

        assert restored.kernel_size == (5, 5)
        assert restored.sigma == pytest.approx(1.5)
        assert restored.epsilon == pytest.approx(1e-6)

        depth = np.zeros((1, 10, 10, 1), dtype="float32")
        mask = np.zeros((1, 10, 10, 1), dtype="float32")
        depth[0, 5, 5, 0] = 3.0
        mask[0, 5, 5, 0] = 1.0

        out1 = layer((depth, mask))
        out2 = restored((depth, mask))
        assert out1[0].shape == out2[0].shape
        assert out1[1].shape == out2[1].shape

    def test_composes_inside_a_functional_model(self):
        depth_input = keras.Input(shape=(16, 16, 1))
        mask_input = keras.Input(shape=(16, 16, 1))
        dense_depth, dense_confidence = SparseDepthSplat()((depth_input, mask_input))
        model = keras.Model([depth_input, mask_input], [dense_depth, dense_confidence])

        depth_np = np.zeros((2, 16, 16, 1), dtype="float32")
        mask_np = np.zeros((2, 16, 16, 1), dtype="float32")
        depth_np[:, 8, 8, 0] = 4.0
        mask_np[:, 8, 8, 0] = 1.0

        out_depth, out_conf = model([depth_np, mask_np])
        assert out_depth.shape == (2, 16, 16, 1)
        assert out_conf.shape == (2, 16, 16, 1)
        assert np.isfinite(keras.ops.convert_to_numpy(out_depth)).all()
        assert np.isfinite(keras.ops.convert_to_numpy(out_conf)).all()
