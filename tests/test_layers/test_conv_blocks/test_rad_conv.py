"""Tests for :class:`RADConv2D` (Region-Aware Deformable Convolution)."""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.conv_blocks.rad_conv import RADConv2D, _box_average


class TestRADConv2DShapes:
    def test_output_shape_matches_filters_and_spatial_size(self):
        layer = RADConv2D(filters=16, kernel_size=3)
        x = keras.random.normal((2, 10, 12, 8))
        y = layer(x)
        assert y.shape == (2, 10, 12, 16)

    def test_grouped_output_shape(self):
        layer = RADConv2D(filters=16, kernel_size=3, groups=4)
        x = keras.random.normal((2, 6, 6, 8))
        y = layer(x)
        assert y.shape == (2, 6, 6, 16)

    def test_1x1_kernel(self):
        layer = RADConv2D(filters=8, kernel_size=1)
        x = keras.random.normal((1, 5, 5, 4))
        y = layer(x)
        assert y.shape == (1, 5, 5, 8)

    def test_compute_output_shape(self):
        layer = RADConv2D(filters=16, kernel_size=3)
        assert layer.compute_output_shape((None, 10, 12, 8)) == (None, 10, 12, 16)


class TestRADConv2DValidation:
    def test_rejects_non_positive_filters(self):
        with pytest.raises(ValueError):
            RADConv2D(filters=0)

    def test_rejects_filters_not_divisible_by_groups(self):
        with pytest.raises(ValueError):
            RADConv2D(filters=10, groups=3)

    def test_rejects_input_channels_not_divisible_by_groups(self):
        layer = RADConv2D(filters=9, groups=3)
        with pytest.raises(ValueError):
            layer.build((None, 4, 4, 8))


class TestRADConv2DExactIntegration:
    """Verifies the summed-area-table box average against a brute-force
    numpy oracle, at hand-computed (not layer-predicted) offsets -- pins
    down the integral-image math independently of the offset/weight convs.
    """

    def test_box_average_matches_brute_force_numpy_integral(self):
        rng = np.random.default_rng(0)
        h, w, c = 9, 11, 3
        img = rng.normal(size=(1, h, w, c)).astype("float32")

        cumulative = keras.ops.cumsum(keras.ops.cumsum(img, axis=1), axis=2)
        sat = keras.ops.pad(cumulative, [[0, 0], [1, 0], [1, 0], [0, 0]])

        # A handful of hand-picked fractional boxes.
        boxes = [
            (0.0, 2.0, 0.0, 2.0),
            (1.5, 4.5, 2.5, 6.0),
            (0.0, h, 0.0, w),  # full image
            (3.25, 3.75, 5.1, 5.9),  # sub-pixel box
        ]
        t = np.array([[b[0] for b in boxes]], dtype="float32")
        b_ = np.array([[b[1] for b in boxes]], dtype="float32")
        l = np.array([[b[2] for b in boxes]], dtype="float32")
        r = np.array([[b[3] for b in boxes]], dtype="float32")

        got = np.array(_box_average(sat, t, b_, l, r))[0]  # (4, C)

        for i, (t0, b0, l0, r0) in enumerate(boxes):
            expected = self._brute_force_box_average(img[0], t0, b0, l0, r0)
            np.testing.assert_allclose(got[i], expected, atol=1e-4, rtol=1e-5)

    @staticmethod
    def _brute_force_box_average(img: np.ndarray, t: float, b: float, l: float, r: float) -> np.ndarray:
        """Reference box average: exact area-weighted overlap sum, treating
        each source pixel ``(i, j)`` as occupying the continuous cell
        ``[i, i+1) x [j, j+1)`` -- the same convention the layer uses.
        """
        h, w, c = img.shape

        def overlaps(lo: float, hi: float, n: int) -> np.ndarray:
            idx = np.arange(n, dtype="float64")
            cell_lo = idx
            cell_hi = idx + 1.0
            return np.clip(np.minimum(hi, cell_hi) - np.maximum(lo, cell_lo), 0.0, None)

        wy = overlaps(t, b, h)  # (H,)
        wx = overlaps(l, r, w)  # (W,)
        weights = np.outer(wy, wx)  # (H, W)
        area = weights.sum()
        weighted = (img.astype("float64") * weights[:, :, None]).sum(axis=(0, 1))
        return (weighted / area).astype("float32")

    def test_degenerate_zero_area_region_does_not_nan(self):
        img = np.ones((1, 4, 4, 2), dtype="float32")
        cumulative = keras.ops.cumsum(keras.ops.cumsum(img, axis=1), axis=2)
        sat = keras.ops.pad(cumulative, [[0, 0], [1, 0], [1, 0], [0, 0]])
        t = np.array([[2.0]], dtype="float32")
        b_ = np.array([[2.0]], dtype="float32")
        l = np.array([[2.0]], dtype="float32")
        r = np.array([[2.0]], dtype="float32")
        got = np.array(_box_average(sat, t, b_, l, r))
        assert np.all(np.isfinite(got))


class TestRADConv2DGradients:
    def test_gradients_flow_to_offsets_weights_and_kernel(self):
        layer = RADConv2D(filters=6, kernel_size=3, groups=2)
        x = tf.Variable(keras.random.normal((2, 6, 6, 4)))
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_sum(tf.square(y))
        variables = layer.trainable_variables
        grads = tape.gradient(loss, variables)
        assert len(grads) == len(variables)
        for var, grad in zip(variables, grads):
            assert grad is not None, f"no gradient for {var.name}"
            assert float(tf.reduce_sum(tf.abs(grad))) > 0.0, f"zero gradient for {var.name}"

    def test_gradients_flow_to_input(self):
        layer = RADConv2D(filters=4, kernel_size=3)
        x = tf.Variable(keras.random.normal((1, 5, 5, 3)))
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_sum(tf.square(y))
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert float(tf.reduce_sum(tf.abs(grad))) > 0.0


class TestRADConv2DSerialization:
    def test_get_config_round_trip(self):
        layer = RADConv2D(
            filters=10,
            kernel_size=(3, 5),
            groups=2,
            use_bias=True,
            offset_bias_init=0.5,
        )
        config = layer.get_config()
        restored = RADConv2D.from_config(config)
        assert restored.filters == 10
        assert restored.kernel_size == (3, 5)
        assert restored.groups == 2
        assert restored.offset_bias_init == 0.5

    def test_save_and_load_round_trip_produces_identical_output(self, tmp_path):
        inputs = keras.Input(shape=(8, 8, 6))
        outputs = RADConv2D(filters=6, kernel_size=3, groups=2)(inputs)
        model = keras.Model(inputs, outputs)

        x = keras.random.normal((2, 8, 8, 6))
        y_before = model(x, training=False)

        save_path = tmp_path / "rad_conv_model.keras"
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x, training=False)

        np.testing.assert_allclose(
            np.array(y_before), np.array(y_after), atol=1e-6, rtol=0
        )
