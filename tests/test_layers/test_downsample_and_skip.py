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
# OFF path, learned variant (strided conv)
# ---------------------------------------------------------------------


class TestStridedConvPath:
    """``pool_type='strided_conv'`` -- the only branch that carries weights."""

    def test_strided_conv_downsample_has_learnable_weights(self, sample_input):
        """RED-proof target for aliasing ``'strided_conv'`` to ``'max'``.

        Both pooling branches are WEIGHTLESS, so a nonzero trainable-weight count is
        the one assertion an alias cannot survive. Shape is useless here: a
        ``Conv2D(k=2, s=2)`` and a ``MaxPooling2D(2)`` produce the SAME output shape
        on an even-sized input, so a shape-only test stays green under the alias.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="strided_conv", name="junction",
        )
        skip, downsampled = layer(keras.ops.convert_to_tensor(sample_input))

        assert len(layer.trainable_weights) > 0, (
            "the strided-conv branch must carry learnable weights; a pooling branch "
            "has zero"
        )
        assert isinstance(layer.conv, keras.layers.Conv2D)
        assert layer.pool is None
        assert layer.conv.name == "junction_conv"
        assert keras.ops.convert_to_numpy(downsampled).shape == (2, 4, 4, 3)

    def test_strided_conv_skip_is_the_input_tensor(self, sample_input):
        """Same contract as the pooling branch: the skip is the RAW input."""
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="strided_conv",
        )
        skip, _ = layer(keras.ops.convert_to_tensor(sample_input))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(skip), sample_input, atol=0.0, rtol=0.0
        )

    def test_strided_conv_preserves_the_channel_count(self):
        """Channel-PRESERVING by design (decisions.md D-013).

        The deleted ``ConvUNextModel._downsample`` widened channels here. Pinned at a
        channel count (7) that is not the default anything, so a hardcoded width
        cannot pass by coincidence.
        """
        rng = np.random.default_rng(7)
        x = rng.normal(size=(1, 8, 8, 7)).astype("float32")
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="strided_conv",
        )
        skip, downsampled = layer(keras.ops.convert_to_tensor(x))

        assert keras.ops.convert_to_numpy(downsampled).shape == (1, 4, 4, 7)
        assert keras.ops.convert_to_numpy(skip).shape == (1, 8, 8, 7)
        assert layer.conv.filters == 7

    def test_strided_conv_use_bias_true_creates_a_bias_vector(self, sample_input):
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="strided_conv", use_bias=True,
        )
        layer(keras.ops.convert_to_tensor(sample_input))

        assert layer.conv.use_bias is True
        assert layer.conv.bias is not None
        assert len([w for w in layer.weights if "bias" in w.path]) == 1

    def test_strided_conv_use_bias_false_is_degree_one_homogeneous(self, sample_input):
        """``use_bias=False`` is what makes this branch legal on the bias-free arm.

        Asserts the PROPERTY (``f(a*x) == a*f(x)``), not the flag -- a flag assertion
        cannot see a conv that carries an additive offset by another route.

        Tolerance note: this is a MATMUL, so on an Ampere+ GPU it runs in TF32
        (10-bit mantissa) and the measured relative error is ~5e-4, not ~1e-7. The
        tolerance is set for TF32 rather than for float32; it still discriminates by a
        wide margin, because a ``use_bias=True`` conv breaks the property by
        ``(1-alpha)*bias`` -- an absolute offset of 2.7 at the control test's settings,
        three orders of magnitude above this tolerance.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="strided_conv", use_bias=False,
        )
        alpha = 3.7
        _, base = layer(keras.ops.convert_to_tensor(sample_input))
        _, scaled = layer(keras.ops.convert_to_tensor(alpha * sample_input))

        assert layer.conv.bias is None
        assert not [w for w in layer.weights if "bias" in w.path]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(scaled),
            alpha * keras.ops.convert_to_numpy(base),
            rtol=5e-3, atol=5e-3,
        )

    def test_strided_conv_use_bias_true_is_not_homogeneous(self, sample_input):
        """CONTROL for the test above: with a bias the property must FAIL.

        Without this control, a homogeneity assertion that passes for a trivial
        reason (e.g. a bias initialized to zero and never trained) would look like
        evidence. The bias is set to a nonzero constant first.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type="strided_conv", use_bias=True,
        )
        layer(keras.ops.convert_to_tensor(sample_input))
        layer.conv.bias.assign(keras.ops.ones_like(layer.conv.bias))

        alpha = 3.7
        _, base = layer(keras.ops.convert_to_tensor(sample_input))
        _, scaled = layer(keras.ops.convert_to_tensor(alpha * sample_input))

        assert not np.allclose(
            keras.ops.convert_to_numpy(scaled),
            alpha * keras.ops.convert_to_numpy(base),
            rtol=5e-3, atol=5e-3,
        )

    def test_strided_conv_config_round_trip_at_non_default_values(self, sample_input):
        """Built at NON-DEFAULT knob values on purpose.

        A round trip at defaults is not a ``get_config`` instrument: the constructor
        default silently repairs a dropped key. ``use_bias`` here is ``False`` (default
        ``True``) and the initializer/regularizer are non-default too.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False,
            laplacian_kernel_size=(7, 7),
            pool_type="strided_conv",
            use_bias=False,
            kernel_initializer="orthogonal",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            name="junction",
        )
        config = layer.get_config()
        assert config["use_bias"] is False
        assert config["pool_type"] == "strided_conv"

        restored = DownsampleAndSkip.from_config(config)
        restored(keras.ops.convert_to_tensor(sample_input))

        assert restored.use_bias is False
        assert restored.conv.use_bias is False
        assert restored.conv.bias is None
        assert isinstance(restored.kernel_initializer, keras.initializers.Orthogonal)
        assert isinstance(restored.kernel_regularizer, keras.regularizers.L2)

    @pytest.mark.parametrize("pool_type", ["max", "average"])
    def test_use_bias_is_visible_to_a_bias_sweep_on_weightless_branches(
            self, pool_type):
        """``use_bias`` is INERT on the pooling branches but still OBSERVABLE.

        Every bias-free trainer in this repo audits compliance by walking
        ``model._flatten_layers()`` and flagging any layer whose ``use_bias`` is
        truthy (e.g. ``src/train/bfunet/train_unet_denoiser.py:198``). This wrapper
        now HAS that attribute, so leaving it at the ``True`` default on a bias-free
        model makes every junction report as a bias offender even though the pooling
        branch owns no weights at all. Bias-free callers must pass ``use_bias=False``
        explicitly; this test states why, so nobody "simplifies" it away.
        """
        layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type=pool_type, use_bias=False,
        )
        assert layer.weights == []
        assert getattr(layer, "use_bias", False) is False

        default_layer = DownsampleAndSkip(
            use_laplacian_pyramid=False, laplacian_kernel_size=(5, 5),
            pool_type=pool_type,
        )
        assert getattr(default_layer, "use_bias", False) is True, (
            "the constructor default is True and IS seen by a bias sweep")

    def test_strided_conv_keras_round_trip_preserves_weight_values(self, sample_input):
        """The learned kernel must survive `.keras` save/load BY VALUE.

        A weight-count or config assertion cannot see the recorded defect class where
        a sub-layer created inside ``build`` is restored with FRESH weights.
        """
        inputs = keras.Input(shape=(8, 8, 3))
        skip, down = DownsampleAndSkip(
            use_laplacian_pyramid=False,
            laplacian_kernel_size=(5, 5),
            pool_type="strided_conv",
            use_bias=False,
            name="junction",
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=down)

        before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "strided.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        after = keras.ops.convert_to_numpy(reloaded(sample_input, training=False))
        np.testing.assert_allclose(before, after, atol=1e-6)

        restored_layer = reloaded.get_layer("junction")
        assert restored_layer.pool_type == "strided_conv"
        assert restored_layer.use_bias is False
        assert len(restored_layer.trainable_weights) == 1


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
