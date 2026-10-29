"""Tests for :class:`dl_techniques.layers.fusion.hierarchical_fusion.HierarchicalScaleFusion`.

The layer's value is in how its parts compose (top-down cascade, sub-pixel upsample,
weighted merge, residual refine), and shape, parameter-count and serialization
assertions are all invariant under a broken composition. So most tests here assert a
VALUE: a round trip on weights, an impulse landing in the right slot of a non-square
grid, a coarse-level perturbation reaching the far corner, a knob changing the weights
or the output. Every "nothing moved" assertion has a "something moved" twin.
"""

import os
from unittest import mock

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.fusion.hierarchical_fusion import (
    HierarchicalScaleFusion,
    check_pyramid_shapes,
)
from dl_techniques.layers.pooling.pixel_unshuffle import PixelShuffle2D
from tests.test_models.knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)

# ---------------------------------------------------------------------

# Non-square on purpose, and odd at 6x10 -> 3x5 -> 2x3, so the crop path and a
# transposed stride are both exercised.
PYRAMID = [(12, 20, 8), (6, 10, 16), (3, 5, 32), (2, 3, 32)]
FILTERS = 8


def _inputs(batch=2, seed=0, pyramid=PYRAMID, dtype="float32"):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((batch, *shape)).astype(dtype) for shape in pyramid]


def _make(**overrides):
    config = dict(filters=FILTERS, num_levels=len(PYRAMID), name="fusion")
    config.update(overrides)
    return HierarchicalScaleFusion(**config)


def _model(pyramid=PYRAMID, **overrides):
    inputs = [keras.Input(shape) for shape in pyramid]
    layer = _make(num_levels=len(pyramid), **overrides)
    return keras.Model(inputs, layer(inputs)), layer


def _built_model_builder(**overrides):
    def build():
        model, _ = _model(**overrides)
        return model
    return build


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _relative(weights):
    """Weight paths with the root segment stripped, so two instances compare equal."""
    return sorted(w.path.split("/", 1)[-1] for w in weights)


# ---------------------------------------------------------------------
# construction and validation
# ---------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"filters": 0},
            {"filters": -4},
            {"num_levels": 1},
            {"epsilon": 0.0},
            {"upsample": "nearest"},
        ],
    )
    def test_bad_arguments_raise(self, overrides):
        with pytest.raises(ValueError):
            _make(**overrides)

    def test_wrong_level_count_raises_at_build(self):
        with pytest.raises(ValueError, match="Expected 4 feature maps"):
            _make().build([(None, *s) for s in PYRAMID[:3]])

    def test_wrong_level_count_raises_at_call(self):
        layer = _make()
        with pytest.raises(ValueError, match="Expected 4 feature maps"):
            layer(_inputs()[:3])

    def test_broken_halving_raises_at_build(self):
        shapes = [(None, 12, 20, 8), (None, 7, 10, 16), (None, 4, 5, 32), (None, 2, 3, 32)]
        with pytest.raises(ValueError, match="expected ceil"):
            _make().build(shapes)

    def test_broken_halving_raises_at_call_not_only_at_build(self):
        """A layer built on a dynamic pyramid must still refuse a bad static one."""
        layer = _make()
        layer.build([(None, None, None, s[-1]) for s in PYRAMID])
        bad = _inputs()
        bad[1] = bad[1][:, :5]
        with pytest.raises(ValueError, match="expected ceil"):
            layer(bad)

    def test_the_shape_helper_accepts_unknown_sizes(self):
        check_pyramid_shapes([(None, None, None, 3)] * 3, 3)

    def test_compute_output_shape_works_unbuilt(self):
        layer = _make()
        assert not layer.built
        out = layer.compute_output_shape([(None, *s) for s in PYRAMID])
        assert out == (None, 12, 20, FILTERS)


# ---------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------


class TestForward:
    @pytest.mark.parametrize("upsample", ["subpixel", "bilinear"])
    @pytest.mark.parametrize("weighted_fusion", [True, False])
    def test_output_shape_matches_compute_output_shape(self, upsample, weighted_fusion):
        layer = _make(upsample=upsample, weighted_fusion=weighted_fusion)
        y = layer(_inputs(), training=False)
        assert tuple(y.shape) == layer.compute_output_shape([(2, *s) for s in PYRAMID])
        assert tuple(y.shape) == (2, 12, 20, FILTERS)
        assert np.all(np.isfinite(_np(y)))

    def test_a_one_by_one_coarsest_level(self):
        pyramid = [(4, 4, 8), (2, 2, 8), (1, 1, 8)]
        layer = _make(num_levels=3)
        y = layer(_inputs(pyramid=pyramid), training=False)
        assert tuple(y.shape) == (2, 4, 4, FILTERS)
        assert np.all(np.isfinite(_np(y)))

    def test_traces_with_unknown_spatial_sizes(self):
        layer = _make()
        specs = [tf.TensorSpec([None, None, None, s[-1]], tf.float32) for s in PYRAMID]

        @tf.function(input_signature=[specs])
        def traced(xs):
            return layer(xs, training=False)

        xs = _inputs()
        eager = _np(layer(xs, training=False))
        np.testing.assert_allclose(_np(traced(xs)), eager, atol=1e-5, rtol=0)

    def test_jit_compile_matches_eager(self):
        layer = _make()
        xs = _inputs()
        eager = _np(layer(xs, training=False))
        compiled = tf.function(lambda a: layer(a, training=False), jit_compile=True)
        out = _np(compiled(xs))
        assert np.all(np.isfinite(out))
        # Measured on CPU, float32, this pyramid: XLA reassociation stays well under 1e-4
        # against an output absmax of order 1; the bound is above that and far below any
        # composition defect (a dropped level moves the output by order 1).
        np.testing.assert_allclose(out, eager, atol=1e-4, rtol=0)

    def test_mixed_float16_forward_is_finite_and_tracks_float32(self, mixed_float16_policy):
        layer = _make()
        y = layer(_inputs(), training=False)
        assert y.dtype == "float16"
        assert np.all(np.isfinite(_np(y).astype("float32")))

    def test_float64_forward(self, float64_policy):
        xs = _inputs(dtype="float64")
        assert xs[0].dtype == np.float64
        layer = _make()
        y = layer(xs, training=False)
        assert y.dtype == "float64"
        assert np.all(np.isfinite(_np(y)))


# ---------------------------------------------------------------------
# serialization and build
# ---------------------------------------------------------------------


class TestSerialization:
    @pytest.mark.parametrize("upsample", ["subpixel", "bilinear"])
    @pytest.mark.parametrize("weighted_fusion", [True, False])
    def test_keras_round_trip_restores_values(self, tmp_path, upsample, weighted_fusion):
        model, layer = _model(upsample=upsample, weighted_fusion=weighted_fusion)
        xs = _inputs()
        # One training step moves the BN moving averages off their initial values, so a
        # restore that loaded nothing cannot pass for an untrained model.
        model(xs, training=True)
        original = _np(model(xs, training=False))
        saved = [_np(w).copy() for w in layer.weights]
        assert saved, "the layer has no weights to compare"

        path = os.path.join(str(tmp_path), "fusion.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        # Before any forward pass on `loaded`, so a build()-only load path cannot hide
        # behind weights that a first call would create fresh.
        loaded_layer = [l for l in loaded.layers if isinstance(l, HierarchicalScaleFusion)][0]
        assert len(loaded_layer.weights) == len(saved)
        for before, after in zip(saved, loaded_layer.weights):
            np.testing.assert_allclose(before, _np(after), atol=0.0, rtol=0)

        np.testing.assert_allclose(
            original, _np(loaded(xs, training=False)), atol=1e-6, rtol=0,
            err_msg="HierarchicalScaleFusion values differ after a .keras round trip",
        )

    def test_get_config_covers_every_constructor_argument(self):
        layer = _make(
            upsample="bilinear", weighted_fusion=False, normalization_type="layer_norm",
            activation_type="relu", epsilon=1e-3,
            kernel_regularizer=keras.regularizers.L2(1e-4),
        )
        clone = HierarchicalScaleFusion.from_config(layer.get_config())
        assert clone.get_config() == layer.get_config()
        for key in ("num_levels", "upsample", "weighted_fusion", "normalization_type",
                    "activation_type", "epsilon"):
            assert getattr(clone, key) == getattr(layer, key)

    def test_the_initializer_is_stored_as_a_spec_not_an_instance(self):
        layer = _make(kernel_initializer=keras.initializers.HeNormal())
        assert isinstance(layer.kernel_initializer, (str, dict))

    def test_each_conv_draws_its_own_kernel(self):
        """Two convs of one shape must not replay one sample (shared-initializer trap)."""
        layer = _make()
        layer(_inputs(), training=False)
        a = _np(layer.refines[0].conv.kernel)
        b = _np(layer.refines[1].conv.kernel)
        assert a.shape == b.shape
        assert not np.allclose(a, b)


class TestBuild:
    @pytest.mark.parametrize("upsample", ["subpixel", "bilinear"])
    @pytest.mark.parametrize("weighted_fusion", [True, False])
    def test_explicit_build_matches_lazy_build(self, upsample, weighted_fusion):
        explicit = _make(upsample=upsample, weighted_fusion=weighted_fusion)
        explicit.build([(None, *s) for s in PYRAMID])
        lazy = _make(upsample=upsample, weighted_fusion=weighted_fusion)
        lazy(_inputs(), training=False)
        assert _relative(explicit.weights) == _relative(lazy.weights)
        assert explicit.count_params() == lazy.count_params()

    def test_bilinear_builds_no_up_conv_weights(self):
        """Anti-vacuity sibling of the parity test: parity passes if BOTH over-build."""
        layer = _make(upsample="bilinear")
        layer(_inputs(), training=False)
        assert not [w for w in layer.weights if "up_conv" in w.path]
        subpixel = _make()
        subpixel(_inputs(), training=False)
        assert [w for w in subpixel.weights if "up_conv" in w.path]

    def test_plain_sum_builds_no_fusion_weight(self):
        off = _make(weighted_fusion=False)
        off(_inputs(), training=False)
        assert not [w for w in off.weights if "fusion_logits" in w.path]
        on = _make()
        on(_inputs(), training=False)
        assert [w for w in on.weights if "fusion_logits" in w.path]

    def test_building_through_a_parent_keeps_the_fusion_weight_live(self):
        class _Parent(keras.layers.Layer):
            def __init__(self, child, **kwargs):
                super().__init__(**kwargs)
                self.child = child

            def call(self, inputs):
                return self.child(inputs)

        child = _make()
        _Parent(child)([keras.Input(s) for s in PYRAMID])
        logits = _np(child.fusion_logits)
        np.testing.assert_allclose(logits, np.ones((len(PYRAMID) - 1, 2)), atol=0.0, rtol=0)


# ---------------------------------------------------------------------
# every part is live
# ---------------------------------------------------------------------


class TestLiveness:
    @pytest.mark.parametrize("upsample", ["subpixel", "bilinear"])
    @pytest.mark.parametrize("weighted_fusion", [True, False])
    def test_gradients_reach_every_trainable_weight(self, upsample, weighted_fusion):
        layer = _make(upsample=upsample, weighted_fusion=weighted_fusion)
        xs = _inputs()
        with tf.GradientTape() as tape:
            loss = keras.ops.mean(keras.ops.square(layer(xs, training=True)))
        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) > 0
        for var, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {var.path}"
            assert np.any(_np(grad) != 0.0), f"all-zero gradient for {var.path}"

    def test_the_fusion_weights_change_the_output(self):
        layer = _make()
        xs = _inputs()
        before = _np(layer(xs, training=False))
        layer.fusion_logits.assign(np.tile([[3.0, 0.5]], (len(PYRAMID) - 1, 1)))
        after = _np(layer(xs, training=False))
        assert np.max(np.abs(before - after)) > 1e-3

    def test_every_input_level_reaches_the_output(self):
        layer = _make()
        xs = _inputs()
        base = _np(layer(xs, training=False))
        for level in range(len(PYRAMID)):
            bumped = [x.copy() for x in xs]
            bumped[level] = bumped[level] + np.random.default_rng(7).standard_normal(
                bumped[level].shape
            ).astype("float32")
            delta = np.max(np.abs(base - _np(layer(bumped, training=False))))
            assert delta > 1e-3, f"level {level} does not reach the output"


class TestKnobs:
    def test_filters_and_num_levels_change_the_weight_shapes(self):
        assert_structural_knob_changes_weights(
            {f: (lambda f=f: _built_model_builder(filters=f)()) for f in (4, 8, 16)},
            knob="filters",
        )
        pyramids = {n: PYRAMID[:n] for n in (2, 3, 4)}
        builders = {
            n: (lambda n=n: _model(pyramid=pyramids[n])[0]) for n in pyramids
        }
        assert_structural_knob_changes_weights(builders, knob="num_levels")

    def test_upsample_weighting_and_normalization_change_the_weight_shapes(self):
        assert_structural_knob_changes_weights(
            {u: (lambda u=u: _built_model_builder(upsample=u)()) for u in ("subpixel", "bilinear")},
            knob="upsample",
        )
        assert_structural_knob_changes_weights(
            {w: (lambda w=w: _built_model_builder(weighted_fusion=w)()) for w in (True, False)},
            knob="weighted_fusion",
        )
        assert_structural_knob_changes_weights(
            {n: (lambda n=n: _built_model_builder(normalization_type=n)())
             for n in ("batch_norm", "rms_norm")},
            knob="normalization_type",
        )

    def test_activation_and_epsilon_change_the_output_at_identical_weights(self):
        xs = _inputs()
        assert_value_knob_changes_output(
            {a: (lambda a=a: _built_model_builder(activation_type=a)()) for a in ("gelu", "relu")},
            xs, knob="activation_type",
        )
        assert_value_knob_changes_output(
            {e: (lambda e=e: _built_model_builder(epsilon=e)()) for e in (1e-4, 1.0)},
            xs, knob="epsilon",
        )


# ---------------------------------------------------------------------
# composition and orientation
# ---------------------------------------------------------------------


class TestOrientation:
    def test_pixel_shuffle_places_channel_blocks_where_the_old_helper_did(self):
        """The upsample step was a local helper before it was ``PixelShuffle2D``.

        ``output[b, h*s + i, w*s + j, c] = x[b, h, w, (i*s + j)*C + c]``, checked on a
        non-square grid so a transposed stride cannot pass.
        """
        s, c_out = 2, 3
        x = np.random.default_rng(0).standard_normal((2, 3, 5, s * s * c_out)).astype("float32")
        expected = np.zeros((2, 3 * s, 5 * s, c_out), "float32")
        for i in range(s):
            for j in range(s):
                expected[:, i::s, j::s, :] = x[..., (i * s + j) * c_out:(i * s + j + 1) * c_out]
        np.testing.assert_allclose(
            _np(PixelShuffle2D(s)(x)), expected, atol=0.0, rtol=0
        )

    def test_the_subpixel_step_routes_each_conv_channel_to_its_own_offset(self):
        """A delta in the coarse map lands at the offset its output channel names."""
        layer = _make(filters=1, num_levels=2)
        layer.build([(1, 5, 9, 1), (1, 3, 5, 1)])
        kernel = np.zeros((3, 3, 1, 4), "float32")
        kernel[1, 1, 0, :] = [1.0, 2.0, 3.0, 4.0]  # channel k -> offset (k // 2, k % 2)
        layer.up_convs[0].kernel.assign(kernel)

        y = np.zeros((1, 3, 5, 1), "float32")
        y[0, 1, 2, 0] = 1.0
        out = _np(layer._upsample(0, keras.ops.convert_to_tensor(y), 5, 9))

        assert out.shape == (1, 5, 9, 1)  # 6x10 cropped to the odd 5x9 target
        expected = np.zeros((1, 5, 9, 1), "float32")
        for k, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            expected[0, 2 + k // 2, 4 + k % 2, 0] = value
        np.testing.assert_allclose(out, expected, atol=0.0, rtol=0)

    def test_a_coarse_perturbation_reaches_the_far_corner_and_a_fine_one_does_not(self):
        layer = _make()
        xs = _inputs()
        base = _np(layer(xs, training=False))

        coarse = [x.copy() for x in xs]
        coarse[-1][:, 1, 2, :] += 5.0
        coarse_delta = np.abs(base - _np(layer(coarse, training=False)))[:, -1, -1, :].max()
        assert coarse_delta > 1e-3

        fine = [x.copy() for x in xs]
        fine[0][:, 0, 0, :] += 5.0
        fine_delta = np.abs(base - _np(layer(fine, training=False)))[:, -1, -1, :].max()
        # The finest level only reaches its own neighbourhood through one 3x3 refine.
        assert fine_delta == 0.0


# ---------------------------------------------------------------------
# the guards above can fail
# ---------------------------------------------------------------------


class TestGuardsCanFail:
    def test_a_dropped_top_down_signal_is_caught_by_the_far_corner_probe(self):
        layer = _make()
        xs = _inputs()
        base = _np(layer(xs, training=False))
        coarse = [x.copy() for x in xs]
        coarse[-1][:, 1, 2, :] += 5.0

        with mock.patch.object(
            HierarchicalScaleFusion, "_upsample",
            lambda self, i, y, h, w: keras.ops.zeros(
                (keras.ops.shape(y)[0], h, w, FILTERS), y.dtype
            ),
        ):
            broken = _np(layer(coarse, training=False))
            broken_base = _np(layer(xs, training=False))
        # With the coarse signal cut, the coarsest level cannot move the far corner.
        assert np.abs(broken_base - broken)[:, -1, -1, :].max() == 0.0
        # ... while the intact layer does move it, so the probe above is not vacuous.
        assert np.abs(base - _np(layer(coarse, training=False)))[:, -1, -1, :].max() > 1e-3

    def test_a_transposed_shuffle_is_caught_by_the_placement_test(self):
        def transposed(self, inputs, training=None):
            r = self.block_size
            shape = keras.ops.shape(inputs)
            b, h, w, c = shape[0], shape[1], shape[2], shape[3]
            x = keras.ops.reshape(inputs, (b, h, w, r, r, c // (r * r)))
            x = keras.ops.transpose(x, (0, 1, 4, 2, 3, 5))  # j before i: wrong
            return keras.ops.reshape(x, (b, h * r, w * r, c // (r * r)))

        with mock.patch.object(PixelShuffle2D, "call", transposed):
            with pytest.raises(AssertionError):
                TestOrientation().test_the_subpixel_step_routes_each_conv_channel_to_its_own_offset()
