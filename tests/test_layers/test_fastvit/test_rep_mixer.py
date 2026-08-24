"""
Test suite for FastVitRepMixer and FastVitRepMixerBlock (FastViT / MobileCLIP2 MCi).

Covers initialization + stored config, constructor validation, forward shape over
multiple spatial sizes (including NON-square ones) and channel widths,
`compute_output_shape` pre- and post-build, training-mode behaviour, gradient flow,
a `.keras` VALUE round trip, and the five mandated behavioural pins:

1. `test_gamma_zero_is_exact_identity`
2. `test_gamma_nonzero_is_not_identity`   (both arms — an identity-only assertion
   is also satisfied by a completely dead component)
3. `test_layer_scale_constraint_is_none_and_gamma_may_go_negative`
4. `test_norm_branch_is_a_bare_batchnorm`
5. `test_drop_path_wired`
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.fastvit.rep_mixer import (
    FastVitRepMixer,
    FastVitRepMixerBlock,
)


# =====================================================================
# FastVitRepMixer
# =====================================================================


class TestFastVitRepMixer:
    """Comprehensive test suite for the FastVitRepMixer token mixer."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        """Standard configuration used across the suite."""
        return {'dim': 16}

    @pytest.fixture
    def sample_input(self):
        """Deterministic rank-4 sample input, (B, H, W, C) = (2, 8, 8, 16)."""
        rng = np.random.default_rng(1234)
        return rng.normal(size=(2, 8, 8, 16)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        layer = FastVitRepMixer(**basic_config)

        assert layer.dim == 16
        assert layer.kernel_size == 3
        assert layer.layer_scale_init_value == pytest.approx(1e-5)
        assert not layer.built

    def test_sub_layers_created_in_init(self, basic_config):
        layer = FastVitRepMixer(**basic_config)

        assert layer.norm is not None
        assert layer.mixer is not None
        assert layer.layer_scale is not None
        assert layer.norm.num_conv_branches == 0
        assert layer.norm.use_scale_branch is False
        assert layer.norm.use_act is False
        assert layer.mixer.num_conv_branches == 1
        assert layer.mixer.use_scale_branch is True
        assert layer.mixer.use_act is False
        # group_size=1 is timm's DEPTHWISE setting.
        assert layer.norm.group_size == 1
        assert layer.mixer.group_size == 1

    def test_layer_scale_none_disables_layer_scale(self):
        layer = FastVitRepMixer(dim=8, layer_scale_init_value=None)
        assert layer.layer_scale is None
        assert layer.layer_scale_init_value is None

        x = np.random.default_rng(7).normal(size=(2, 4, 4, 8)).astype('float32')
        y = layer(x, training=False)
        assert tuple(y.shape) == (2, 4, 4, 8)

    def test_config_completeness(self, basic_config):
        layer = FastVitRepMixer(**basic_config, kernel_size=5,
                                layer_scale_init_value=0.25)
        config = layer.get_config()

        for key in (
                'dim', 'kernel_size', 'layer_scale_init_value',
                'kernel_initializer', 'kernel_regularizer',
        ):
            assert key in config, f"missing '{key}' in get_config()"

        rebuilt = FastVitRepMixer.from_config(config)
        assert rebuilt.dim == layer.dim
        assert rebuilt.kernel_size == layer.kernel_size
        assert rebuilt.layer_scale_init_value == layer.layer_scale_init_value

    def test_config_round_trip_with_none_layer_scale(self):
        layer = FastVitRepMixer(dim=8, layer_scale_init_value=None)
        rebuilt = FastVitRepMixer.from_config(layer.get_config())
        assert rebuilt.layer_scale is None

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("kwargs,match", [
        ({'dim': 0}, "dim must be positive"),
        ({'dim': -4}, "dim must be positive"),
        ({'dim': 8, 'kernel_size': 0}, "kernel_size must be positive"),
        ({'dim': 8, 'kernel_size': -3}, "kernel_size must be positive"),
        ({'dim': 8, 'kernel_size': 4}, "kernel_size must be odd"),
        ({'dim': 8, 'layer_scale_init_value': 'small'},
         "layer_scale_init_value must be a real number or None"),
    ])
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FastVitRepMixer(**kwargs)

    def test_invalid_input_rank(self, basic_config):
        layer = FastVitRepMixer(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            layer.build((None, 8, 16))

    def test_invalid_input_channels(self, basic_config):
        layer = FastVitRepMixer(**basic_config)
        with pytest.raises(ValueError, match="Input channel count must equal dim"):
            layer.build((None, 8, 8, 32))

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("height,width", [(8, 8), (12, 5), (4, 4), (3, 7)])
    @pytest.mark.parametrize("dim", [8, 32])
    def test_forward_shape(self, height, width, dim):
        layer = FastVitRepMixer(dim=dim)
        x = np.random.default_rng(0).normal(
            size=(2, height, width, dim)).astype('float32')

        y = layer(x, training=False)

        assert tuple(y.shape) == (2, height, width, dim)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape_pre_build(self):
        layer = FastVitRepMixer(dim=16)
        assert not layer.built
        assert layer.compute_output_shape((None, 12, 5, 16)) == (None, 12, 5, 16)
        assert not layer.built

    def test_compute_output_shape_matches_forward(self, basic_config):
        layer = FastVitRepMixer(**basic_config)
        input_shape = (2, 12, 5, 16)
        predicted = layer.compute_output_shape(input_shape)

        x = np.random.default_rng(3).normal(size=input_shape).astype('float32')
        y = layer(x, training=False)

        assert tuple(y.shape) == tuple(predicted)
        assert layer.built
        assert layer.compute_output_shape(input_shape) == tuple(predicted)

    # ------------------------------------------------------------------
    # behavioural pins
    # ------------------------------------------------------------------

    def test_gamma_zero_is_exact_identity(self, sample_input):
        """PIN 1: at ``layer_scale_init_value=0.0`` the mixer is the EXACT identity.

        Half of a pair. On its own this assertion is also satisfied by a layer
        whose ``call`` simply returns its input, which is why
        :meth:`test_gamma_nonzero_is_not_identity` exists.
        """
        layer = FastVitRepMixer(dim=16, layer_scale_init_value=0.0)
        y = ops.convert_to_numpy(layer(sample_input, training=False))

        np.testing.assert_allclose(
            y, sample_input, atol=1e-6, rtol=0,
            err_msg="gamma=0 must make FastVitRepMixer the exact identity"
        )

    def test_gamma_nonzero_is_not_identity(self, sample_input):
        """PIN 2: at the SHIPPED 1e-5 init the mixer must NOT be the identity.

        Uses the shipped default rather than an inflated one: the measured
        maximum absolute deviation at 1e-5 is ~1.3e-4, two orders of magnitude
        above the 1e-6 tolerance, so no weakening of the assertion is needed.
        This is the arm that a dead component (e.g. ``call`` returning ``inputs``)
        fails.
        """
        layer = FastVitRepMixer(dim=16, layer_scale_init_value=1e-5)
        y = ops.convert_to_numpy(layer(sample_input, training=False))

        max_abs_delta = float(np.abs(y - sample_input).max())
        assert max_abs_delta > 1e-6, (
            f"FastVitRepMixer with a nonzero LayerScale gamma must change its "
            f"input; max|y - x| = {max_abs_delta} (the residual branch is dead)"
        )

    def test_layer_scale_constraint_is_none_and_gamma_may_go_negative(self):
        """PIN 3: LayerScale gamma must be free to become negative.

        ``LearnableMultiplier`` defaults to ``constraint='non_neg'``. MEASURED:
        the constraint is enforced by the OPTIMIZER (``apply_gradients``), not by
        ``Variable.assign`` — a plain negative ``assign`` reads back negative under
        BOTH settings and is therefore a vacuous check. This pin instead takes a
        real SGD step whose gradient drives gamma negative.
        """
        layer = FastVitRepMixer(dim=8, layer_scale_init_value=1e-5)
        layer.build((None, 4, 4, 8))

        assert layer.layer_scale.constraint is None, (
            "LayerScale must be built with constraint=None; the "
            "LearnableMultiplier default 'non_neg' clamps gamma at zero"
        )

        gamma = layer.layer_scale.gamma
        optimizer = keras.optimizers.SGD(learning_rate=1.0)
        optimizer.build([gamma])
        # A positive gradient with lr=1.0 moves gamma from +1e-5 to about -1.0.
        grad = ops.convert_to_tensor(np.ones(gamma.shape, dtype='float32'))
        optimizer.apply_gradients([(grad, gamma)])

        gamma_value = ops.convert_to_numpy(gamma)
        assert np.all(gamma_value < 0.0), (
            f"LayerScale gamma was clamped at zero by a weight constraint "
            f"after an optimizer step; got {gamma_value}"
        )

    def test_norm_branch_is_a_bare_batchnorm(self):
        """PIN 4: the ``norm`` MobileOneBlock reduces to exactly one BatchNorm.

        Reads the BUILT sub-layer state, not the constructor kwargs: zero ``k x k``
        conv branches, no 1x1 scale branch, only the identity BatchNormalization,
        and exactly its four weights (gamma, beta, moving_mean, moving_variance).
        """
        dim = 16
        layer = FastVitRepMixer(dim=dim)
        layer.build((None, 8, 8, dim))

        assert len(layer.norm.conv_branches) == 0, (
            f"norm must have ZERO conv branches, got "
            f"{len(layer.norm.conv_branches)}"
        )
        assert layer.norm.scale_branch is None, (
            "norm must have NO 1x1 scale branch"
        )
        assert layer.norm.skip_branch is not None, (
            "norm's identity BatchNormalization branch must exist"
        )

        weight_shapes = [tuple(w.shape) for w in layer.norm.weights]
        assert weight_shapes == [(dim,)] * 4, (
            f"norm must own exactly a BatchNormalization's four (dim,) weights, "
            f"got {weight_shapes}"
        )

        # Control: the mixer is NOT degenerate.
        assert len(layer.mixer.conv_branches) == 1
        assert layer.mixer.scale_branch is not None

    # ------------------------------------------------------------------
    # training mode
    # ------------------------------------------------------------------

    def test_inference_is_deterministic(self, sample_input):
        layer = FastVitRepMixer(dim=16)
        a = ops.convert_to_numpy(layer(sample_input, training=False))
        b = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    def test_training_true_updates_batchnorm_statistics(self, sample_input):
        """training=True must reach the BatchNormalizations (moving stats move)."""
        layer = FastVitRepMixer(dim=16)
        layer.build(sample_input.shape)
        before = ops.convert_to_numpy(layer.norm.skip_branch.moving_mean).copy()

        layer(sample_input, training=True)

        after = ops.convert_to_numpy(layer.norm.skip_branch.moving_mean)
        assert not np.allclose(before, after), (
            "training=True must reach the identity BatchNormalization"
        )

    def test_frozen_layer_matches_inference_under_training_true(self, sample_input):
        layer = FastVitRepMixer(dim=16)
        layer.build(sample_input.shape)
        layer.trainable = False

        a = ops.convert_to_numpy(layer(sample_input, training=True))
        b = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def test_gradients_flow_to_every_trainable_weight(self, sample_input):
        layer = FastVitRepMixer(dim=16, layer_scale_init_value=0.5)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = ops.mean(ops.square(y))

        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) > 0
        for var, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {var.path}"
            assert np.any(ops.convert_to_numpy(grad) != 0.0), (
                f"all-zero gradient for {var.path}"
            )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def test_serialization_cycle_value_identity(self, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FastVitRepMixer(dim=16, layer_scale_init_value=0.3)(inputs)
        model = keras.Model(inputs, outputs)

        original = ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'rep_mixer.keras')
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            original, restored, atol=1e-6, rtol=0,
            err_msg="FastVitRepMixer values differ after a .keras round trip"
        )


# =====================================================================
# FastVitRepMixerBlock
# =====================================================================


class TestFastVitRepMixerBlock:
    """Comprehensive test suite for the FastVitRepMixerBlock."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        return {'dim': 16, 'mlp_ratio': 2.0}

    @pytest.fixture
    def sample_input(self):
        rng = np.random.default_rng(4321)
        return rng.normal(size=(2, 8, 8, 16)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        block = FastVitRepMixerBlock(**basic_config)

        assert block.dim == 16
        assert block.kernel_size == 3
        assert block.mlp_ratio == pytest.approx(2.0)
        assert block.hidden_dim == 32
        assert block.dropout_rate == 0.0
        assert block.drop_path_rate == 0.0
        assert block.layer_scale_init_value == pytest.approx(1e-5)
        assert not block.built

    def test_sub_layers_created_in_init(self, basic_config):
        block = FastVitRepMixerBlock(**basic_config, drop_path_rate=0.1)

        assert isinstance(block.token_mixer, FastVitRepMixer)
        assert block.mlp is not None
        assert block.mlp.hidden_dim == 32
        assert block.layer_scale is not None
        # Created UNCONDITIONALLY, including at rate 0.0.
        assert block.drop_path is not None
        assert block.drop_path.drop_path_rate == pytest.approx(0.1)

    def test_drop_path_layer_exists_at_rate_zero(self, basic_config):
        block = FastVitRepMixerBlock(**basic_config, drop_path_rate=0.0)
        assert block.drop_path is not None
        assert block.drop_path.drop_path_rate == 0.0

    def test_layer_scale_none_disables_both_layer_scales(self):
        block = FastVitRepMixerBlock(dim=8, layer_scale_init_value=None)
        assert block.layer_scale is None
        assert block.token_mixer.layer_scale is None

    def test_config_completeness(self, basic_config):
        block = FastVitRepMixerBlock(
            **basic_config, kernel_size=5, dropout_rate=0.1,
            drop_path_rate=0.2, layer_scale_init_value=0.25)
        config = block.get_config()

        for key in (
                'dim', 'kernel_size', 'mlp_ratio', 'dropout_rate',
                'drop_path_rate', 'layer_scale_init_value', 'activation',
                'kernel_initializer', 'kernel_regularizer',
        ):
            assert key in config, f"missing '{key}' in get_config()"

        rebuilt = FastVitRepMixerBlock.from_config(config)
        assert rebuilt.dim == block.dim
        assert rebuilt.kernel_size == block.kernel_size
        assert rebuilt.mlp_ratio == block.mlp_ratio
        assert rebuilt.dropout_rate == block.dropout_rate
        assert rebuilt.drop_path_rate == block.drop_path_rate
        assert rebuilt.hidden_dim == block.hidden_dim

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("kwargs,match", [
        ({'dim': 0}, "dim must be positive"),
        ({'dim': 8, 'kernel_size': 4}, "kernel_size must be odd"),
        ({'dim': 8, 'kernel_size': 0}, "kernel_size must be positive"),
        ({'dim': 8, 'mlp_ratio': 0.0}, "mlp_ratio must be positive"),
        ({'dim': 8, 'mlp_ratio': -1.0}, "mlp_ratio must be positive"),
        ({'dim': 8, 'mlp_ratio': 0.01}, "zero-width bottleneck"),
        ({'dim': 8, 'dropout_rate': 1.0}, r"dropout_rate must be in \[0, 1\)"),
        ({'dim': 8, 'dropout_rate': -0.1}, r"dropout_rate must be in \[0, 1\)"),
        ({'dim': 8, 'layer_scale_init_value': 'small'},
         "layer_scale_init_value must be a real number or None"),
    ])
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FastVitRepMixerBlock(**kwargs)

    @pytest.mark.parametrize("rate", [1.0, 1.5])
    def test_invalid_drop_path_rate_surfaces_from_stochastic_depth(self, rate):
        """StochasticDepth's own [0, 1) validation must NOT be swallowed."""
        with pytest.raises(ValueError, match=r"drop_path_rate must be in \[0, 1\)"):
            FastVitRepMixerBlock(dim=8, drop_path_rate=rate)

    def test_invalid_input_rank(self, basic_config):
        block = FastVitRepMixerBlock(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            block.build((None, 8, 16))

    def test_invalid_input_channels(self, basic_config):
        block = FastVitRepMixerBlock(**basic_config)
        with pytest.raises(ValueError, match="Input channel count must equal dim"):
            block.build((None, 8, 8, 32))

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("height,width", [(8, 8), (12, 5), (4, 4), (3, 7)])
    @pytest.mark.parametrize("dim", [8, 32])
    def test_forward_shape(self, height, width, dim):
        block = FastVitRepMixerBlock(dim=dim, mlp_ratio=2.0)
        x = np.random.default_rng(11).normal(
            size=(2, height, width, dim)).astype('float32')

        y = block(x, training=False)

        assert tuple(y.shape) == (2, height, width, dim)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape_pre_build(self):
        block = FastVitRepMixerBlock(dim=16)
        assert not block.built
        assert block.compute_output_shape((None, 12, 5, 16)) == (None, 12, 5, 16)
        assert not block.built

    def test_compute_output_shape_matches_forward(self, basic_config):
        block = FastVitRepMixerBlock(**basic_config)
        input_shape = (2, 12, 5, 16)
        predicted = block.compute_output_shape(input_shape)

        x = np.random.default_rng(13).normal(size=input_shape).astype('float32')
        y = block(x, training=False)

        assert tuple(y.shape) == tuple(predicted)
        assert block.built
        assert block.compute_output_shape(input_shape) == tuple(predicted)

    # ------------------------------------------------------------------
    # behavioural pins
    # ------------------------------------------------------------------

    def test_drop_path_wired(self, sample_input):
        """PIN 5: stochastic depth must actually guard the ConvMlp branch.

        The block is frozen (``trainable=False``) so its BatchNormalizations run
        in inference mode under ``training=True``; the ONLY remaining source of
        nondeterminism is the drop path. Two arms:
        * ``training=True`` twice at rate 0.5 must DIFFER (the branch is dropped
          per sample), and
        * ``training=False`` must be deterministic AND equal to an identically
          weighted block built at ``drop_path_rate=0.0``.
        """
        # StochasticDepth draws ONE Bernoulli per SAMPLE, so with the fixture's
        # batch of 2 two consecutive draws coincide with probability 1/4 — a 25%
        # flake that showed up only once another module's RNG consumption shifted
        # the global state. A 16-sample batch drops that to 2**-16.
        sample_input = np.repeat(sample_input, 8, axis=0)
        block = FastVitRepMixerBlock(dim=16, mlp_ratio=2.0,
                                     layer_scale_init_value=0.5,
                                     drop_path_rate=0.5)
        block.build(sample_input.shape)
        block.trainable = False

        keras.utils.set_random_seed(0)
        train_a = ops.convert_to_numpy(block(sample_input, training=True))
        train_b = ops.convert_to_numpy(block(sample_input, training=True))
        assert not np.allclose(train_a, train_b, atol=1e-6, rtol=0), (
            "drop_path_rate=0.5 at training=True must make repeated calls "
            "differ; the StochasticDepth layer is not wired into the branch"
        )

        eval_a = ops.convert_to_numpy(block(sample_input, training=False))
        eval_b = ops.convert_to_numpy(block(sample_input, training=False))
        np.testing.assert_allclose(
            eval_a, eval_b, atol=1e-6, rtol=0,
            err_msg="training=False must be deterministic"
        )

        reference = FastVitRepMixerBlock(dim=16, mlp_ratio=2.0,
                                         layer_scale_init_value=0.5,
                                         drop_path_rate=0.0)
        reference.build(sample_input.shape)
        reference.set_weights(block.get_weights())
        reference.trainable = False
        no_drop = ops.convert_to_numpy(reference(sample_input, training=False))

        np.testing.assert_allclose(
            eval_a, no_drop, atol=1e-6, rtol=0,
            err_msg="at training=False the drop path must be a pure identity"
        )

    def test_block_layer_scale_constraint_is_none(self):
        """PIN 3 (block arm): both LayerScale gammas must be unconstrained."""
        block = FastVitRepMixerBlock(dim=8)
        block.build((None, 4, 4, 8))

        assert block.layer_scale.constraint is None
        assert block.token_mixer.layer_scale.constraint is None

    # ------------------------------------------------------------------
    # training mode
    # ------------------------------------------------------------------

    def test_training_modes_differ_with_dropout(self, sample_input):
        block = FastVitRepMixerBlock(dim=16, mlp_ratio=2.0, dropout_rate=0.5,
                                     layer_scale_init_value=0.5)
        keras.utils.set_random_seed(42)
        train_out = ops.convert_to_numpy(block(sample_input, training=True))
        eval_out = ops.convert_to_numpy(block(sample_input, training=False))

        assert not np.allclose(train_out, eval_out), (
            "dropout_rate=0.5 must make training and inference outputs differ"
        )

    def test_inference_is_deterministic(self, sample_input):
        block = FastVitRepMixerBlock(dim=16, mlp_ratio=2.0, dropout_rate=0.3,
                                     drop_path_rate=0.3)
        a = ops.convert_to_numpy(block(sample_input, training=False))
        b = ops.convert_to_numpy(block(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def test_gradients_flow_to_every_trainable_weight(self, sample_input):
        block = FastVitRepMixerBlock(dim=16, mlp_ratio=2.0,
                                     layer_scale_init_value=0.5)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = block(x, training=True)
            loss = ops.mean(ops.square(y))

        grads = tape.gradient(loss, block.trainable_variables)

        assert len(block.trainable_variables) > 0
        for var, grad in zip(block.trainable_variables, grads):
            assert grad is not None, f"no gradient for {var.path}"
            assert np.any(ops.convert_to_numpy(grad) != 0.0), (
                f"all-zero gradient for {var.path}"
            )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def test_serialization_cycle_value_identity(self, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FastVitRepMixerBlock(
            dim=16, mlp_ratio=2.0, dropout_rate=0.1, drop_path_rate=0.2,
            layer_scale_init_value=0.3)(inputs)
        model = keras.Model(inputs, outputs)

        original = ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'rep_mixer_block.keras')
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            original, restored, atol=1e-6, rtol=0,
            err_msg="FastVitRepMixerBlock values differ after a .keras round trip"
        )
