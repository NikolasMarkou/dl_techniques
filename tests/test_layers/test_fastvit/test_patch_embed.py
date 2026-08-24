"""
Test suite for FastVitPatchEmbed (FastViT / MobileCLIP2 MCi).

Covers initialization + stored config, constructor validation, forward shape over
multiple spatial sizes (including NON-square ones), `compute_output_shape` pre-
and post-build, training-mode behaviour, gradient flow, a `.keras` VALUE round
trip, and the mandated behavioural pins:

2. `test_groups_divisibility_raises`       -- pin 2
4. `test_patch_embed_halves_spatial_dims`  -- pin 4
5. `test_lkc_use_act_wiring`               -- pin 5
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.fastvit.patch_embed import FastVitPatchEmbed
from dl_techniques.layers.fastvit.reparam_large_kernel_conv import (
    ReparamLargeKernelConv,
)
from dl_techniques.layers.mobile_one_block import MobileOneBlock



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

class TestFastVitPatchEmbed:
    """Comprehensive test suite for FastVitPatchEmbed."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        return {'embed_dim': 32}

    @pytest.fixture
    def sample_input(self):
        """Deterministic rank-4 sample input, (B, H, W, C) = (2, 16, 16, 16)."""
        rng = np.random.default_rng(4321)
        return rng.normal(size=(2, 16, 16, 16)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        layer = FastVitPatchEmbed(**basic_config)

        assert layer.embed_dim == 32
        assert layer.patch_size == 7
        assert layer.stride == 2
        assert layer.use_se is False
        assert layer.lkc_use_act is False
        assert not layer.built

    def test_sub_layers_created_in_init(self, basic_config):
        layer = FastVitPatchEmbed(**basic_config)

        assert isinstance(layer.proj_lkc, ReparamLargeKernelConv)
        assert isinstance(layer.proj_mobileone, MobileOneBlock)

        # Reference wiring of the large-kernel conv.
        assert layer.proj_lkc.out_channels == 32
        assert layer.proj_lkc.kernel_size == 7
        assert layer.proj_lkc.stride == 2
        assert layer.proj_lkc.group_size == 1      # timm: DEPTHWISE
        assert layer.proj_lkc.small_kernel == 3
        assert layer.proj_lkc.activation is None   # lkc_use_act=False

        # Reference wiring of the trailing pointwise MobileOne block.
        assert layer.proj_mobileone.out_channels == 32
        assert layer.proj_mobileone.kernel_size == 1
        assert layer.proj_mobileone.stride == 1
        assert layer.proj_mobileone.use_se is False

    def test_lkc_use_act_true_forwards_activation(self, basic_config):
        layer = FastVitPatchEmbed(**basic_config, lkc_use_act=True)
        assert layer.proj_lkc.activation is not None

    def test_use_se_forwarded_to_lkc(self, basic_config):
        layer = FastVitPatchEmbed(**basic_config, use_se=True)
        assert layer.proj_lkc.use_se is True
        assert layer.proj_lkc.se is not None
        # The trailing MobileOne block never uses SE at this call site.
        assert layer.proj_mobileone.use_se is False

    def test_config_completeness(self, basic_config):
        layer = FastVitPatchEmbed(
            **basic_config, patch_size=5, stride=4, use_se=True,
            lkc_use_act=True, activation='relu')
        config = layer.get_config()

        for key in (
                'embed_dim', 'patch_size', 'stride', 'use_se', 'lkc_use_act',
                'activation', 'kernel_initializer', 'kernel_regularizer',
        ):
            assert key in config, f"missing '{key}' in get_config()"

        rebuilt = FastVitPatchEmbed.from_config(config)
        assert rebuilt.embed_dim == layer.embed_dim
        assert rebuilt.patch_size == layer.patch_size
        assert rebuilt.stride == layer.stride
        assert rebuilt.use_se == layer.use_se
        assert rebuilt.lkc_use_act == layer.lkc_use_act

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("kwargs,match", [
        ({'embed_dim': 0}, "embed_dim must be positive"),
        ({'embed_dim': -32}, "embed_dim must be positive"),
        ({'embed_dim': 32, 'patch_size': 0}, "patch_size must be positive"),
        ({'embed_dim': 32, 'patch_size': -7}, "patch_size must be positive"),
        ({'embed_dim': 32, 'stride': 0}, "stride must be positive"),
        ({'embed_dim': 32, 'stride': -2}, "stride must be positive"),
    ])
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FastVitPatchEmbed(**kwargs)

    def test_invalid_input_rank(self, basic_config):
        layer = FastVitPatchEmbed(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            layer.build((None, 16, 16))

    def test_undefined_channels_raises(self, basic_config):
        layer = FastVitPatchEmbed(**basic_config)
        with pytest.raises(ValueError, match="Input channels dimension"):
            layer.build((None, 16, 16, None))

    # ------------------------------------------------------------------
    # PIN 2: grouped-conv divisibility
    # ------------------------------------------------------------------

    def test_groups_divisibility_raises(self):
        """PIN 2: embed_dim % in_channels != 0 must raise, naming both numbers.

        ``group_size=1`` on the large-kernel conv means ``groups = in_channels``,
        so the block can only widen by an exact multiple. This must be a LOUD
        raise, never a silent reshape.

        MEASURED: both the inner :class:`ReparamLargeKernelConv` and Keras'
        ``Conv2D`` also reject this configuration, so "a ValueError mentioning 48
        and 32" is satisfied three layers deep and is NOT evidence that this
        layer checks anything. The pin therefore requires the message to be
        stated in ``FastVitPatchEmbed``'s own vocabulary (``embed_dim``, and the
        word "multiple"), which only this layer's check produces.
        """
        layer = FastVitPatchEmbed(embed_dim=48)

        with pytest.raises(ValueError) as excinfo:
            layer.build((None, 16, 16, 32))

        message = str(excinfo.value)
        assert "embed_dim=48" in message, (
            f"the divisibility error must name embed_dim in this layer's own "
            f"vocabulary; got what looks like an inner layer's error: {message}")
        assert "in_channels=32" in message, (
            f"the divisibility error must name in_channels; got: {message}")
        assert "multiple" in message, (
            f"expected FastVitPatchEmbed's own diagnostic naming the depthwise "
            f"multiple requirement; got: {message}")

    def test_groups_divisibility_raises_on_call(self):
        """The same violation must surface through a normal forward call."""
        layer = FastVitPatchEmbed(embed_dim=48)
        x = np.zeros((2, 16, 16, 32), dtype='float32')
        with pytest.raises(ValueError, match="exact multiple"):
            layer(x, training=False)

    @pytest.mark.parametrize("in_channels,embed_dim", [
        (16, 16), (16, 32), (32, 64), (64, 128), (3, 48),
    ])
    def test_legal_doubling_geometry_builds(self, in_channels, embed_dim):
        layer = FastVitPatchEmbed(embed_dim=embed_dim)
        x = np.random.default_rng(5).normal(
            size=(2, 16, 16, in_channels)).astype('float32')
        y = layer(x, training=False)
        assert tuple(y.shape) == (2, 8, 8, embed_dim)

    # ------------------------------------------------------------------
    # PIN 4: spatial geometry
    # ------------------------------------------------------------------

    def test_patch_embed_halves_spatial_dims(self):
        """PIN 4 (square arm): (B,64,64,C) -> (B,32,32,embed_dim).

        Split from the non-square arm deliberately. A single test asserting both
        would short-circuit on whichever assertion comes first, leaving the other
        arm unproven against its own injection.
        """
        embed_dim = 64
        layer = FastVitPatchEmbed(embed_dim=embed_dim)
        x = np.random.default_rng(6).normal(size=(2, 64, 64, 32)).astype('float32')

        y = layer(x, training=False)

        assert tuple(y.shape) == (2, 32, 32, embed_dim), (
            f"(B,64,64,C) must map to (B,32,32,{embed_dim}), "
            f"got {tuple(y.shape)}")

    def test_patch_embed_halves_spatial_dims_non_square(self):
        """PIN 4 (non-square arm): (B,64,32,C) -> (B,32,16,embed_dim).

        A square-only assertion cannot distinguish the correct reduction from a
        transposed or per-axis-asymmetric one.
        """
        embed_dim = 64
        layer = FastVitPatchEmbed(embed_dim=embed_dim)
        x = np.random.default_rng(7).normal(size=(2, 64, 32, 32)).astype('float32')

        y = layer(x, training=False)

        assert tuple(y.shape) == (2, 32, 16, embed_dim), (
            f"(B,64,32,C) must map to (B,32,16,{embed_dim}), "
            f"got {tuple(y.shape)}")

    @pytest.mark.parametrize("height,width", [(16, 16), (16, 8), (7, 5)])
    def test_forward_shape(self, height, width):
        layer = FastVitPatchEmbed(embed_dim=32)
        x = np.random.default_rng(8).normal(
            size=(2, height, width, 16)).astype('float32')

        y = layer(x, training=False)

        assert tuple(y.shape) == (2, (height + 1) // 2, (width + 1) // 2, 32)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape_pre_build(self):
        layer = FastVitPatchEmbed(embed_dim=64)
        assert not layer.built
        assert layer.compute_output_shape((None, 64, 32, 32)) == (None, 32, 16, 64)
        assert not layer.built

    def test_compute_output_shape_matches_forward(self):
        layer = FastVitPatchEmbed(embed_dim=32)
        input_shape = (2, 16, 8, 16)
        predicted = layer.compute_output_shape(input_shape)

        x = np.random.default_rng(9).normal(size=input_shape).astype('float32')
        y = layer(x, training=False)

        assert tuple(y.shape) == tuple(predicted)
        assert layer.built
        assert layer.compute_output_shape(input_shape) == tuple(predicted)

    # ------------------------------------------------------------------
    # PIN 5: lkc_use_act wiring
    # ------------------------------------------------------------------

    def test_lkc_use_act_wiring(self, sample_input):
        """PIN 5: lkc_use_act must actually gate the LKC's activation.

        Both layers have IDENTICAL structure and transplanted weights, so the
        only difference is whether the activation is applied inside the
        large-kernel conv. Ignoring the flag makes the two outputs equal.
        """
        without_act = FastVitPatchEmbed(embed_dim=32, lkc_use_act=False)
        with_act = FastVitPatchEmbed(embed_dim=32, lkc_use_act=True)

        without_act.build(sample_input.shape)
        with_act.build(sample_input.shape)

        assert len(without_act.weights) == len(with_act.weights), (
            "lkc_use_act must not change the weight structure; the transplant "
            "below assumes identical layouts")
        with_act.set_weights(without_act.get_weights())

        assert without_act.proj_lkc.activation is None
        assert with_act.proj_lkc.activation is not None

        y_without = ops.convert_to_numpy(without_act(sample_input, training=False))
        y_with = ops.convert_to_numpy(with_act(sample_input, training=False))

        max_abs_delta = float(np.abs(y_with - y_without).max())
        assert max_abs_delta > 1e-6, (
            f"lkc_use_act is IGNORED: with transplanted identical weights, "
            f"lkc_use_act=True and lkc_use_act=False produce the same output "
            f"(max|delta| = {max_abs_delta})"
        )

    # ------------------------------------------------------------------
    # training mode
    # ------------------------------------------------------------------

    def test_inference_is_deterministic(self, sample_input, basic_config):
        layer = FastVitPatchEmbed(**basic_config)
        a = ops.convert_to_numpy(layer(sample_input, training=False))
        b = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    def test_training_true_updates_batchnorm_statistics(
            self, sample_input, basic_config):
        layer = FastVitPatchEmbed(**basic_config)
        layer.build(sample_input.shape)
        bn = _branch_bn(layer.proj_lkc.large_conv)
        before = ops.convert_to_numpy(bn.moving_mean).copy()

        layer(sample_input, training=True)

        after = ops.convert_to_numpy(bn.moving_mean)
        assert not np.allclose(before, after), (
            "training=True must reach the large-kernel branch's BatchNormalization")

    def test_frozen_layer_matches_inference_under_training_true(
            self, sample_input, basic_config):
        layer = FastVitPatchEmbed(**basic_config)
        layer.build(sample_input.shape)
        layer.trainable = False

        a = ops.convert_to_numpy(layer(sample_input, training=True))
        b = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def test_gradients_flow_to_every_trainable_weight(self, sample_input):
        layer = FastVitPatchEmbed(
            embed_dim=32, use_se=True, lkc_use_act=True)
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
        outputs = FastVitPatchEmbed(
            embed_dim=32, use_se=True, lkc_use_act=True)(inputs)
        model = keras.Model(inputs, outputs)

        original = ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'patch_embed.keras')
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            original, restored, atol=1e-6, rtol=0,
            err_msg="FastVitPatchEmbed values differ after a .keras round trip")
