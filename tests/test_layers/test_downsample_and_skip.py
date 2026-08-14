"""Tests for :class:`DownsampleAndSkip`.

The junction returns TWO rank-4 tensors, so shape-only assertions are structurally
blind to the two defects that matter most: a swapped output tuple and a wrong pooling
op. Every assertion here is therefore written against VALUES, not shapes, except where
a shape is the quantity under test.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.downsample_and_skip import DownsampleAndSkip


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def sample_input() -> np.ndarray:
    """A (2, 8, 8, 3) float32 batch with no structure assumed."""
    rng = np.random.default_rng(1234)
    return rng.normal(size=(2, 8, 8, 3)).astype("float32")


@pytest.fixture
def block_input() -> np.ndarray:
    """A (1, 4, 4, 1) batch whose 2x2 blocks have DIFFERENT max and mean.

    Each 2x2 block holds ``[[0, 1], [2, 3]] + 10 * block_index`` so
    ``max = 3 + 10 * i`` while ``mean = 1.5 + 10 * i``. The two pooling ops are
    therefore separated by a full 1.5 at every output position.
    """
    x = np.zeros((1, 4, 4, 1), dtype="float32")
    for bi, (r, c) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        x[0, r:r + 2, c:c + 2, 0] = np.array(
            [[0.0, 1.0], [2.0, 3.0]], dtype="float32"
        ) + 10.0 * bi
    return x


# ---------------------------------------------------------------------
# OFF path (raw skip + pooling)
# ---------------------------------------------------------------------


class TestPoolingPath:
    """``use_laplacian_pyramid=False``."""

    def test_off_path_skip_is_the_input_tensor(self, sample_input):
        """RED-proof target for a swapped output tuple.

        Compares the skip to the INPUT elementwise. A shape assertion cannot see the
        swap (skip is (2,8,8,3), downsampled is (2,4,4,3) -- and at a 1x1 input they
        would even share a shape), and a "skip is bigger" assertion is a proxy, not
        the contract.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5)
        )
        skip, downsampled = layer(keras.ops.convert_to_tensor(sample_input))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(skip), sample_input, atol=0.0, rtol=0.0
        )
        assert keras.ops.convert_to_numpy(downsampled).shape == (2, 4, 4, 3)

    def test_off_path_downsampled_is_not_the_input(self, sample_input):
        """Dead-component probe target: ``return inputs, inputs`` must not pass."""
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5)
        )
        _, downsampled = layer(keras.ops.convert_to_tensor(sample_input))
        out = keras.ops.convert_to_numpy(downsampled)

        assert out.shape != sample_input.shape
        assert out.shape == (2, 4, 4, 3)

    def test_max_pool_type_selects_max_pooling(self, block_input):
        """The default path pools by MAX, asserted on the pooled VALUES."""
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="max",
        )
        _, downsampled = layer(keras.ops.convert_to_tensor(block_input))

        expected = np.array([[3.0, 13.0], [23.0, 33.0]], dtype="float32")
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(downsampled)[0, :, :, 0], expected,
            atol=1e-6,
        )

    def test_average_pool_type_selects_average_pooling(self, block_input):
        """RED-proof target for a hardcoded ``'max'``.

        Asserted on the pooled VALUES (mean 1.5+10i vs max 3+10i), not on the
        sub-layer's class name -- a class-name assertion cannot see a pooling op that
        is constructed correctly but never applied.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="average",
        )
        _, downsampled = layer(keras.ops.convert_to_tensor(block_input))

        expected = np.array([[1.5, 11.5], [21.5, 31.5]], dtype="float32")
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(downsampled)[0, :, :, 0], expected,
            atol=1e-6,
        )

    def test_average_pooling_is_linear_hence_homogeneous(self, sample_input):
        """``f(a*x) == a*f(x)``: the property the average path exists to provide."""
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="average",
        )
        alpha = 3.7
        _, base = layer(keras.ops.convert_to_tensor(sample_input))
        _, scaled = layer(keras.ops.convert_to_tensor(alpha * sample_input))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(scaled),
            alpha * keras.ops.convert_to_numpy(base),
            atol=1e-5,
        )

    def test_off_path_constructs_no_pyramid_sublayer(self):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5)
        )
        assert layer.pyramid is None
        assert layer.pool is not None

    def test_invalid_pool_type_raises(self):
        with pytest.raises(ValueError, match="pool_type"):
            DownsampleAndSkip(
                use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
                pool_type="median",
            )

    def test_invalid_laplacian_kernel_size_raises(self):
        with pytest.raises(ValueError, match="laplacian_kernel_size"):
            DownsampleAndSkip(
                use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5, 5)
            )


# ---------------------------------------------------------------------
# ON path (Laplacian pyramid split)
# ---------------------------------------------------------------------


class TestLaplacianPath:
    """``use_laplacian_pyramid=True``."""

    def test_on_path_returns_high_band_first(self, sample_input):
        """The skip is the FULL-resolution high band; the second output is the low band.

        Value-checked against the underlying pyramid's own ``split`` so this cannot be
        satisfied by returning the input unchanged, nor by swapping the tuple.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5)
        )
        x = keras.ops.convert_to_tensor(sample_input)
        skip, downsampled = layer(x)

        ref_low, ref_high = layer.pyramid.split(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(skip),
            keras.ops.convert_to_numpy(ref_high),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(downsampled),
            keras.ops.convert_to_numpy(ref_low),
            atol=1e-6,
        )

    def test_on_path_skip_is_not_the_input(self, sample_input):
        """The high band is a RESIDUAL, so it must differ from the input in value."""
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5)
        )
        skip, _ = layer(keras.ops.convert_to_tensor(sample_input))

        assert keras.ops.convert_to_numpy(skip).shape == sample_input.shape
        assert not np.allclose(
            keras.ops.convert_to_numpy(skip), sample_input, atol=1e-4
        )

    def test_on_path_shapes(self, sample_input):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5)
        )
        skip, downsampled = layer(keras.ops.convert_to_tensor(sample_input))
        assert keras.ops.convert_to_numpy(skip).shape == (2, 8, 8, 3)
        assert keras.ops.convert_to_numpy(downsampled).shape == (2, 4, 4, 3)

    def test_on_path_pool_type_is_inert(self, sample_input):
        """``pool_type`` does not apply to the pyramid path -- both give the same split."""
        x = keras.ops.convert_to_tensor(sample_input)
        a = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5), pool_type="max"
        )(x)
        b = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5),
            pool_type="average",
        )(x)
        for lhs, rhs in zip(a, b):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(lhs),
                keras.ops.convert_to_numpy(rhs),
                atol=1e-6,
            )

    def test_on_path_kernel_size_changes_the_split(self, sample_input):
        """``laplacian_kernel_size`` is LIVE, not decorative.

        This is what makes ``test_config_round_trip_preserves_kernel_size`` mean
        something: a dropped kernel size would silently build a different operator.
        """
        x = keras.ops.convert_to_tensor(sample_input)
        _, low_a = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(3, 3)
        )(x)
        _, low_b = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(7, 7)
        )(x)

        assert not np.allclose(
            keras.ops.convert_to_numpy(low_a),
            keras.ops.convert_to_numpy(low_b),
            atol=1e-5,
        )

    def test_on_path_is_homogeneous(self, sample_input):
        """Bias-free by construction: ``f(a*x) == a*f(x)`` on BOTH bands."""
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5)
        )
        alpha = 2.5
        base = layer(keras.ops.convert_to_tensor(sample_input))
        scaled = layer(keras.ops.convert_to_tensor(alpha * sample_input))
        for lhs, rhs in zip(base, scaled):
            np.testing.assert_allclose(
                alpha * keras.ops.convert_to_numpy(lhs),
                keras.ops.convert_to_numpy(rhs),
                atol=1e-4,
            )

    def test_on_path_constructs_no_pool_sublayer(self):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5)
        )
        assert layer.pool is None
        assert layer.pyramid is not None


# ---------------------------------------------------------------------
# naming, shapes, config
# ---------------------------------------------------------------------


class TestNamingAndShape:

    def test_pool_sublayer_name_is_derived_from_the_wrapper_name(self):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            name="bottleneck_downsample",
        )
        assert layer.name == "bottleneck_downsample"
        assert layer.pool.name == "bottleneck_downsample_pool"

    def test_pyramid_sublayer_name_is_derived_from_the_wrapper_name(self):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=True, laplacian_kernel_size=(5, 5),
            name="encoder_downsample_0",
        )
        assert layer.name == "encoder_downsample_0"
        assert layer.pyramid.name == "encoder_downsample_0_pyramid"

    @pytest.mark.parametrize("use_pyramid", [False, True])
    def test_compute_output_shape_matches_the_call(self, sample_input, use_pyramid):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=use_pyramid, laplacian_kernel_size=(5, 5)
        )
        skip, downsampled = layer(keras.ops.convert_to_tensor(sample_input))
        skip_shape, down_shape = layer.compute_output_shape(sample_input.shape)

        assert tuple(skip_shape) == keras.ops.convert_to_numpy(skip).shape
        assert tuple(down_shape) == keras.ops.convert_to_numpy(downsampled).shape


class TestConfig:

    @pytest.mark.parametrize("use_pyramid", [False, True])
    def test_config_round_trip_preserves_kernel_size(self, use_pyramid):
        """RED-proof target for a ``laplacian_kernel_size`` dropped from get_config."""
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=use_pyramid,
            laplacian_kernel_size=(7, 7),
            pool_type="average",
        )
        config = layer.get_config()
        assert "laplacian_kernel_size" in config

        restored = DownsampleAndSkip.from_config(config)
        assert tuple(restored.laplacian_kernel_size) == (7, 7)
        if use_pyramid:
            assert tuple(restored.pyramid.blur_kernel_size) == (7, 7)

    def test_config_round_trip_preserves_pool_type_and_branch(self):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False,
            laplacian_kernel_size=(5, 5),
            pool_type="average",
        )
        restored = DownsampleAndSkip.from_config(layer.get_config())

        assert restored.pool_type == "average"
        assert restored.use_laplacian_pyramid is False
        assert isinstance(restored.pool, keras.layers.AveragePooling2D)

    def test_registered_serializable(self):
        assert (
            keras.saving.get_registered_name(DownsampleAndSkip)
            == "dl_techniques.layers>DownsampleAndSkip"
        )


# ---------------------------------------------------------------------
# functional-model serialization round trip
# ---------------------------------------------------------------------


class TestModelRoundTrip:

    @pytest.mark.parametrize("use_pyramid", [False, True])
    def test_keras_round_trip_in_a_functional_model(self, sample_input, use_pyramid):
        """A multi-output custom Layer must survive `.keras` save/load by VALUE."""
        inputs = keras.Input(shape=(8, 8, 3))
        skip, down = DownsampleAndSkip(
            use_laplacian_pyramid=use_pyramid,
            laplacian_kernel_size=(5, 5),
            pool_type="average",
            name="bottleneck_downsample",
        )(inputs)
        merged = keras.layers.Concatenate(name="merge")(
            [keras.layers.AveragePooling2D(pool_size=(2, 2))(skip), down]
        )
        model = keras.Model(inputs=inputs, outputs=merged)

        before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "junction.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        after = keras.ops.convert_to_numpy(reloaded(sample_input, training=False))
        np.testing.assert_allclose(before, after, atol=1e-6)

        restored_layer = reloaded.get_layer("bottleneck_downsample")
        assert restored_layer.use_laplacian_pyramid is use_pyramid
        assert restored_layer.pool_type == "average"
        assert tuple(restored_layer.laplacian_kernel_size) == (5, 5)

    def test_gradients_flow_through_both_outputs(self, sample_input):
        import tensorflow as tf

        inputs = keras.Input(shape=(8, 8, 3))
        skip, down = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5)
        )(inputs)
        head = keras.layers.Conv2D(4, 3, padding="same", name="head")
        out = keras.layers.Concatenate()(
            [head(keras.layers.AveragePooling2D(pool_size=(2, 2))(skip)), head(down)]
        )
        model = keras.Model(inputs=inputs, outputs=out)

        x = tf.convert_to_tensor(sample_input)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True)))
        grads = tape.gradient(loss, model.trainable_variables)

        assert len(grads) > 0
        assert all(g is not None for g in grads)
        assert any(float(tf.reduce_max(tf.abs(g))) > 0.0 for g in grads)
