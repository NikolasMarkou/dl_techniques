"""
Test suite for ReparamLargeKernelConv (FastViT / MobileCLIP2 MCi).

Covers initialization + stored config, constructor validation, forward shape over
multiple spatial sizes (including NON-square ones, with the stride-2 halving
asserted explicitly), `compute_output_shape` pre- and post-build, training-mode
behaviour, gradient flow, a `.keras` VALUE round trip, and the mandated
behavioural pins:

1. `test_small_kernel_branch_is_live`      -- pin 1
2. `test_groups_divisibility_raises`       -- pin 2 (the LKC half; the
   FastVitPatchEmbed half lives in `test_patch_embed.py`)
3. `test_se_uses_reference_ratio_and_bias` -- pin 3
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.fastvit.reparam_large_kernel_conv import (
    ReparamLargeKernelConv,
)



def _branch_bn(branch):
    """The single BatchNormalization inside a Conv-BN branch ``Sequential``.

    Indexed BY TYPE, not by position: under the reference padding convention the
    branch starts with an explicit ``ZeroPadding2D``, so ``layers[1]`` is the
    convolution, not the norm.
    """
    norms = [
        l for l in branch.layers
        if isinstance(l, keras.layers.BatchNormalization)
    ]
    assert len(norms) == 1, f"expected one BatchNormalization, got {len(norms)}"
    return norms[0]

class TestReparamLargeKernelConv:
    """Comprehensive test suite for ReparamLargeKernelConv."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        """The FastVitPatchEmbed-shaped configuration (depthwise, /2)."""
        return {
            'out_channels': 32,
            'kernel_size': 7,
            'stride': 2,
            'group_size': 1,
            'small_kernel': 3,
        }

    @pytest.fixture
    def sample_input(self):
        """Deterministic rank-4 sample input, (B, H, W, C) = (2, 16, 16, 16)."""
        rng = np.random.default_rng(1234)
        return rng.normal(size=(2, 16, 16, 16)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)

        assert layer.out_channels == 32
        assert layer.kernel_size == 7
        assert layer.stride == 2
        assert layer.group_size == 1
        assert layer.small_kernel == 3
        assert layer.use_se is False
        assert layer.activation is None
        assert not layer.built

    def test_dense_path_creates_branches_in_init(self):
        """group_size=0 is knowable up front, so the branches exist immediately."""
        layer = ReparamLargeKernelConv(
            out_channels=8, kernel_size=7, stride=2, group_size=0, small_kernel=3)
        assert layer.groups == 1
        assert layer.large_conv is not None
        assert layer.small_conv is not None

    def test_grouped_path_defers_branch_creation_to_build(self, basic_config):
        """group_size>0 needs in_channels, so branches are created in build()."""
        layer = ReparamLargeKernelConv(**basic_config)
        assert layer.groups is None
        assert layer.large_conv is None
        assert layer.small_conv is None

        layer.build((None, 16, 16, 16))

        assert layer.groups == 16
        assert layer.large_conv is not None
        assert layer.small_conv is not None

    def test_small_kernel_none_omits_branch(self):
        layer = ReparamLargeKernelConv(
            out_channels=8, kernel_size=7, stride=2, group_size=0)
        assert layer.small_kernel is None
        assert layer.small_conv is None

        layer.build((None, 8, 8, 8))
        assert layer.small_conv is None

    def test_config_completeness(self, basic_config):
        layer = ReparamLargeKernelConv(**basic_config, use_se=True,
                                       activation='gelu')
        config = layer.get_config()

        for key in (
                'out_channels', 'kernel_size', 'stride', 'group_size',
                'small_kernel', 'use_se', 'activation',
                'kernel_initializer', 'kernel_regularizer',
        ):
            assert key in config, f"missing '{key}' in get_config()"

        rebuilt = ReparamLargeKernelConv.from_config(config)
        assert rebuilt.out_channels == layer.out_channels
        assert rebuilt.kernel_size == layer.kernel_size
        assert rebuilt.stride == layer.stride
        assert rebuilt.group_size == layer.group_size
        assert rebuilt.small_kernel == layer.small_kernel
        assert rebuilt.use_se == layer.use_se
        assert rebuilt.activation is not None

    def test_config_round_trip_with_no_activation(self, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)
        rebuilt = ReparamLargeKernelConv.from_config(layer.get_config())
        assert rebuilt.activation is None

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("kwargs,match", [
        ({'out_channels': 0}, "out_channels must be positive"),
        ({'out_channels': -8}, "out_channels must be positive"),
        ({'kernel_size': 0}, "kernel_size must be positive"),
        ({'kernel_size': -7}, "kernel_size must be positive"),
        ({'stride': 0}, "stride must be positive"),
        ({'stride': -2}, "stride must be positive"),
        ({'group_size': -1}, "group_size must be non-negative"),
        ({'small_kernel': 0}, "small_kernel must be positive"),
        ({'small_kernel': 9}, "small_kernel must not exceed kernel_size"),
    ])
    def test_invalid_config_raises(self, basic_config, kwargs, match):
        config = dict(basic_config)
        config.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ReparamLargeKernelConv(**config)

    def test_small_kernel_equal_to_kernel_size_is_allowed(self):
        layer = ReparamLargeKernelConv(
            out_channels=8, kernel_size=3, stride=1, group_size=0, small_kernel=3)
        assert layer.small_kernel == 3

    def test_invalid_input_rank(self, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            layer.build((None, 16, 16))

    def test_undefined_channels_raises(self, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)
        with pytest.raises(ValueError, match="Input channels dimension"):
            layer.build((None, 16, 16, None))

    def test_group_size_not_dividing_in_channels_raises(self):
        layer = ReparamLargeKernelConv(
            out_channels=16, kernel_size=3, stride=1, group_size=5)
        with pytest.raises(ValueError, match="group_size must divide the input"):
            layer.build((None, 8, 8, 16))

    # ------------------------------------------------------------------
    # PIN 2 (LKC half): grouped-conv divisibility
    # ------------------------------------------------------------------

    def test_groups_divisibility_raises(self):
        """PIN 2: a depthwise LKC cannot map in_channels -> a non-multiple.

        ``group_size=1`` resolves to ``groups = in_channels``; a grouped
        convolution partitions the OUTPUT channel axis too, so ``out_channels``
        must be a multiple of ``in_channels``.

        MEASURED: Keras' own ``Conv2D`` also rejects this, with "The number of
        filters must be evenly divisible by the number of groups. Received:
        groups=16, filters=24." So asserting merely that a ``ValueError`` naming
        both numbers is raised would be VACUOUS — it passes with this layer's
        check deleted. What this layer adds, and what is pinned here, is a
        DIAGNOSABLE message stated in the caller's own vocabulary
        (``group_size``, ``in_channels``, ``out_channels``), raised at this
        layer's boundary rather than several frames down inside a sub-layer.
        """
        layer = ReparamLargeKernelConv(
            out_channels=24, kernel_size=7, stride=2, group_size=1, small_kernel=3)

        with pytest.raises(ValueError) as excinfo:
            layer.build((None, 16, 16, 16))

        message = str(excinfo.value)
        assert "in_channels=16" in message, (
            f"the divisibility error must name the input channel count in this "
            f"layer's own vocabulary; got: {message}")
        assert "out_channels=24" in message, (
            f"the divisibility error must name the output channel count in this "
            f"layer's own vocabulary; got: {message}")
        assert "group_size=1" in message, (
            f"the divisibility error must name the offending group_size; "
            f"got: {message}")
        assert "must divide both" in message, (
            f"expected this layer's own grouped-conv diagnostic, got what looks "
            f"like a lower-level Keras error: {message}")

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("height,width", [(16, 16), (16, 8), (7, 5)])
    def test_forward_shape_and_stride_halving(self, height, width):
        """Stride 2 must halve BOTH spatial dims (ceil), including non-square."""
        layer = ReparamLargeKernelConv(
            out_channels=32, kernel_size=7, stride=2, group_size=1, small_kernel=3)
        x = np.random.default_rng(0).normal(
            size=(2, height, width, 16)).astype('float32')

        y = layer(x, training=False)

        expected_h = (height + 1) // 2
        expected_w = (width + 1) // 2
        assert tuple(y.shape) == (2, expected_h, expected_w, 32)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_stride_one_preserves_spatial_dims(self):
        layer = ReparamLargeKernelConv(
            out_channels=16, kernel_size=7, stride=1, group_size=1, small_kernel=3)
        x = np.random.default_rng(1).normal(size=(2, 9, 5, 16)).astype('float32')

        y = layer(x, training=False)
        assert tuple(y.shape) == (2, 9, 5, 16)

    def test_large_kernel_survives_small_feature_map(self):
        """A 7x7 kernel under padding='same' must survive a 4x4 map."""
        layer = ReparamLargeKernelConv(
            out_channels=16, kernel_size=7, stride=1, group_size=1, small_kernel=3)
        x = np.random.default_rng(2).normal(size=(2, 4, 4, 16)).astype('float32')

        y = layer(x, training=False)
        assert tuple(y.shape) == (2, 4, 4, 16)

    def test_compute_output_shape_pre_build(self):
        layer = ReparamLargeKernelConv(
            out_channels=32, kernel_size=7, stride=2, group_size=1, small_kernel=3)
        assert not layer.built
        assert layer.compute_output_shape((None, 16, 8, 16)) == (None, 8, 4, 32)
        assert not layer.built

    def test_compute_output_shape_matches_forward(self):
        layer = ReparamLargeKernelConv(
            out_channels=32, kernel_size=7, stride=2, group_size=1, small_kernel=3)
        input_shape = (2, 16, 8, 16)
        predicted = layer.compute_output_shape(input_shape)

        x = np.random.default_rng(3).normal(size=input_shape).astype('float32')
        y = layer(x, training=False)

        assert tuple(y.shape) == tuple(predicted)
        assert layer.built
        assert layer.compute_output_shape(input_shape) == tuple(predicted)

    # ------------------------------------------------------------------
    # PIN 1: the small-kernel branch is live
    # ------------------------------------------------------------------

    def test_small_kernel_branch_is_live(self, sample_input):
        """PIN 1: adding the small branch changes the output.

        The large branch's weights are TRANSPLANTED from the ``small_kernel=None``
        layer into the ``small_kernel=3`` layer, so the ONLY difference between
        the two forward passes is the presence of the extra branch. Comparing two
        independently-initialized layers would prove nothing (both sides move).
        """
        config = dict(out_channels=32, kernel_size=7, stride=2, group_size=1)

        without_small = ReparamLargeKernelConv(**config, small_kernel=None)
        with_small = ReparamLargeKernelConv(**config, small_kernel=3)

        without_small.build(sample_input.shape)
        with_small.build(sample_input.shape)

        # Transplant: identical large branch on both sides.
        with_small.large_conv.set_weights(without_small.large_conv.get_weights())

        for a, b in zip(without_small.large_conv.weights,
                        with_small.large_conv.weights):
            np.testing.assert_allclose(
                ops.convert_to_numpy(a), ops.convert_to_numpy(b),
                atol=0, rtol=0,
                err_msg="the large-branch transplant did not take; the comparison "
                        "below would be between two different large branches")

        assert with_small.small_conv is not None
        assert without_small.small_conv is None

        y_without = ops.convert_to_numpy(without_small(sample_input, training=False))
        y_with = ops.convert_to_numpy(with_small(sample_input, training=False))

        max_abs_delta = float(np.abs(y_with - y_without).max())
        assert max_abs_delta > 1e-6, (
            f"the small-kernel branch is DEAD: with identical large-branch "
            f"weights, small_kernel=3 and small_kernel=None produce the same "
            f"output (max|delta| = {max_abs_delta})"
        )

    # ------------------------------------------------------------------
    # PIN 3: Squeeze-and-Excitation ratio + bias
    # ------------------------------------------------------------------

    def test_se_uses_reference_ratio_and_bias(self):
        """PIN 3: the SE block must use rd_ratio=0.25 WITH biases.

        Asserted from the BUILT weight shapes, not the constructor kwargs: the
        bottleneck width pins the ratio (0.25 of 32 is 8; timm's SqueezeExcite
        default of 1/16 would give 2), and the presence of two (bottleneck,) /
        (channels,) rank-1 weights pins ``use_bias=True``.
        """
        out_channels = 32
        layer = ReparamLargeKernelConv(
            out_channels=out_channels, kernel_size=7, stride=2, group_size=1,
            small_kernel=3, use_se=True)
        layer.build((None, 16, 16, 16))

        assert layer.se is not None
        assert layer.se.reduction_ratio == pytest.approx(0.25)

        shapes = [tuple(w.shape) for w in layer.se.weights]
        expected_bottleneck = 8  # round(32 * 0.25)

        assert (1, 1, out_channels, expected_bottleneck) in shapes, (
            f"SE reduce kernel must be (1,1,{out_channels},{expected_bottleneck}) "
            f"for reduction_ratio=0.25; got {shapes}"
        )
        assert (1, 1, expected_bottleneck, out_channels) in shapes, (
            f"SE restore kernel must be (1,1,{expected_bottleneck},"
            f"{out_channels}); got {shapes}"
        )

        rank1_shapes = sorted(s for s in shapes if len(s) == 1)
        assert rank1_shapes == [(expected_bottleneck,), (out_channels,)], (
            f"SE must own TWO bias vectors (use_bias=True); rank-1 weight shapes "
            f"were {rank1_shapes}"
        )
        assert len(shapes) == 4, (
            f"SE must own exactly 2 kernels + 2 biases, got {len(shapes)} weights: "
            f"{shapes}"
        )

    def test_se_absent_when_disabled(self, basic_config):
        layer = ReparamLargeKernelConv(**basic_config, use_se=False)
        assert layer.se is None

    def test_se_changes_output(self, sample_input):
        """Control for PIN 3: SE is actually applied, not just constructed."""
        config = dict(out_channels=32, kernel_size=7, stride=2, group_size=1,
                      small_kernel=3)
        without_se = ReparamLargeKernelConv(**config, use_se=False)
        with_se = ReparamLargeKernelConv(**config, use_se=True)

        without_se.build(sample_input.shape)
        with_se.build(sample_input.shape)
        with_se.large_conv.set_weights(without_se.large_conv.get_weights())
        with_se.small_conv.set_weights(without_se.small_conv.get_weights())

        y_without = ops.convert_to_numpy(without_se(sample_input, training=False))
        y_with = ops.convert_to_numpy(with_se(sample_input, training=False))

        assert float(np.abs(y_with - y_without).max()) > 1e-6

    # ------------------------------------------------------------------
    # activation
    # ------------------------------------------------------------------

    def test_activation_none_is_identity_tail(self, sample_input):
        """activation=None must leave the summed branches untouched."""
        config = dict(out_channels=32, kernel_size=7, stride=2, group_size=1,
                      small_kernel=3)
        plain = ReparamLargeKernelConv(**config, activation=None)
        gelu = ReparamLargeKernelConv(**config, activation='gelu')

        plain.build(sample_input.shape)
        gelu.build(sample_input.shape)
        gelu.large_conv.set_weights(plain.large_conv.get_weights())
        gelu.small_conv.set_weights(plain.small_conv.get_weights())

        y_plain = ops.convert_to_numpy(plain(sample_input, training=False))
        y_gelu = ops.convert_to_numpy(gelu(sample_input, training=False))

        np.testing.assert_allclose(
            y_gelu, ops.convert_to_numpy(keras.activations.gelu(y_plain)),
            atol=1e-6, rtol=0,
            err_msg="the trailing activation is not applied as act(sum)")

    # ------------------------------------------------------------------
    # training mode
    # ------------------------------------------------------------------

    def test_inference_is_deterministic(self, sample_input, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)
        a = ops.convert_to_numpy(layer(sample_input, training=False))
        b = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    def test_training_true_updates_batchnorm_statistics(
            self, sample_input, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)
        layer.build(sample_input.shape)
        bn = _branch_bn(layer.large_conv)
        before = ops.convert_to_numpy(bn.moving_mean).copy()

        layer(sample_input, training=True)

        after = ops.convert_to_numpy(bn.moving_mean)
        assert not np.allclose(before, after), (
            "training=True must reach the large branch's BatchNormalization")

    def test_frozen_layer_matches_inference_under_training_true(
            self, sample_input, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)
        layer.build(sample_input.shape)
        layer.trainable = False

        a = ops.convert_to_numpy(layer(sample_input, training=True))
        b = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    def test_batchnorm_epsilon_is_the_reference_value(self, basic_config):
        layer = ReparamLargeKernelConv(**basic_config)
        layer.build((None, 16, 16, 16))

        assert _branch_bn(layer.large_conv).epsilon == pytest.approx(1e-5)
        assert _branch_bn(layer.small_conv).epsilon == pytest.approx(1e-5)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def test_gradients_flow_to_every_trainable_weight(self, sample_input):
        layer = ReparamLargeKernelConv(
            out_channels=32, kernel_size=7, stride=2, group_size=1,
            small_kernel=3, use_se=True, activation='gelu')
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = ops.mean(ops.square(y))

        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) > 0
        for var, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {var.path}"
            assert np.any(ops.convert_to_numpy(grad) != 0.0), (
                f"all-zero gradient for {var.path}")

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def test_serialization_cycle_value_identity(self, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = ReparamLargeKernelConv(
            out_channels=32, kernel_size=7, stride=2, group_size=1,
            small_kernel=3, use_se=True, activation='gelu')(inputs)
        model = keras.Model(inputs, outputs)

        original = ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'lkc.keras')
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            original, restored, atol=1e-6, rtol=0,
            err_msg="ReparamLargeKernelConv values differ after a .keras round trip")
