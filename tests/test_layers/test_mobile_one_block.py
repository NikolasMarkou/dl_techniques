"""
Comprehensive Test Suite for MobileOneBlock Layer

This test suite follows the Modern Keras 3 testing guidelines and covers:
- Basic functionality and initialization
- Forward pass and building
- Critical serialization cycle testing
- Reparameterization functionality
- Edge cases and error conditions
- Various configurations
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf

# Import the layer being tested
from dl_techniques.layers.mobile_one_block import MobileOneBlock


class TestMobileOneBlock:
    """Comprehensive test suite for MobileOneBlock layer."""

    @pytest.fixture
    def basic_config(self):
        """Standard configuration for testing."""
        return {
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 'same',
            'activation': 'gelu'
        }

    @pytest.fixture
    def complex_config(self):
        """Complex configuration with all features enabled."""
        return {
            'out_channels': 128,
            'kernel_size': 3,
            'stride': 2,
            'padding': 'same',
            'use_se': True,
            'num_conv_branches': 2,
            'activation': 'relu',
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'zeros'
        }

    @pytest.fixture
    def sample_input(self):
        """Sample 4D input tensor for testing."""
        return keras.random.normal(shape=(4, 32, 32, 16))

    @pytest.fixture
    def large_input(self):
        """Larger input for stride testing."""
        return keras.random.normal(shape=(2, 64, 64, 32))

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    def test_initialization_basic(self, basic_config):
        """Test basic layer initialization."""
        layer = MobileOneBlock(**basic_config)

        # Check configuration stored correctly
        assert layer.out_channels == 64
        assert layer.kernel_size == 3
        assert layer.stride == 1
        assert layer.padding == 'same'
        assert not layer.use_se
        assert layer.num_conv_branches == 1

        # Check layer is not built yet
        assert not layer.built
        assert layer.inference_mode == False

        # Check sub-layers created
        assert len(layer.conv_branches) == 1
        assert layer.scale_branch is not None  # kernel_size > 1
        assert layer.skip_branch is None  # Not built yet
        assert layer.se_block is None  # use_se=False

    def test_initialization_complex(self, complex_config):
        """Test initialization with complex configuration."""
        layer = MobileOneBlock(**complex_config)

        # Check all parameters
        assert layer.out_channels == 128
        assert layer.use_se == True
        assert layer.num_conv_branches == 2
        assert len(layer.conv_branches) == 2
        assert layer.se_block is not None  # SE enabled

    def test_forward_pass_basic(self, basic_config, sample_input):
        """Test basic forward pass and building."""
        layer = MobileOneBlock(**basic_config)

        # Forward pass triggers building
        output = layer(sample_input)

        # Check layer is now built
        assert layer.built

        # Check output shape
        expected_shape = (4, 32, 32, 64)  # same padding, stride=1
        assert output.shape == expected_shape

        # Check sub-layers are built
        assert all(branch.built for branch in layer.conv_branches)
        if layer.scale_branch:
            assert layer.scale_branch.built
        if layer.skip_branch:
            assert layer.skip_branch.built

    def test_forward_pass_with_stride(self, sample_input):
        """Test forward pass with stride > 1."""
        layer = MobileOneBlock(
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding='same'
        )

        output = layer(sample_input)

        # Check downsampled output
        expected_shape = (4, 16, 16, 32)  # stride=2 halves spatial dims
        assert output.shape == expected_shape

    def test_forward_pass_with_se(self, sample_input):
        """Test forward pass with Squeeze-and-Excitation."""
        layer = MobileOneBlock(
            out_channels=64,
            kernel_size=3,
            use_se=True
        )

        output = layer(sample_input)

        assert output.shape == (4, 32, 32, 64)
        assert layer.se_block is not None
        assert layer.se_block.built

    def test_skip_connection_creation(self):
        """Test skip connection is created when appropriate."""
        # Case 1: Same channels, stride=1 -> skip connection
        layer1 = MobileOneBlock(out_channels=16, kernel_size=3, stride=1)
        input1 = keras.random.normal(shape=(2, 32, 32, 16))

        _ = layer1(input1)  # Build the layer
        assert layer1.skip_branch is not None

        # Case 2: Different channels -> no skip connection
        layer2 = MobileOneBlock(out_channels=32, kernel_size=3, stride=1)
        input2 = keras.random.normal(shape=(2, 32, 32, 16))

        _ = layer2(input2)  # Build the layer
        # Skip branch should be None because in_channels != out_channels
        assert layer2.skip_branch is None

        # Case 3: Same channels, stride > 1 -> no skip connection
        layer3 = MobileOneBlock(out_channels=16, kernel_size=3, stride=2)
        input3 = keras.random.normal(shape=(2, 32, 32, 16))

        _ = layer3(input3)  # Build the layer
        assert layer3.skip_branch is None

    # ========================================================================
    # Serialization Tests (CRITICAL)
    # ========================================================================

    def test_serialization_cycle_basic(self, basic_config, sample_input):
        """CRITICAL TEST: Full serialization cycle with basic config."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MobileOneBlock(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization (basic config)"
            )

    def test_serialization_cycle_complex(self, complex_config, sample_input):
        """CRITICAL TEST: Full serialization cycle with complex config."""
        # Create model with complex configuration
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MobileOneBlock(**complex_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_complex_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization (complex config)"
            )

    def test_config_completeness(self, complex_config):
        """Test that get_config contains all __init__ parameters."""
        layer = MobileOneBlock(**complex_config)
        config = layer.get_config()

        # Check all important config parameters are present
        required_keys = [
            'out_channels', 'kernel_size', 'stride', 'padding',
            'use_se', 'num_conv_branches', 'activation',
            'kernel_initializer', 'bias_initializer'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check values match
        assert config['out_channels'] == complex_config['out_channels']
        assert config['use_se'] == complex_config['use_se']
        assert config['num_conv_branches'] == complex_config['num_conv_branches']

    # ========================================================================
    # Training and Gradient Tests
    # ========================================================================

    def test_gradients_flow(self, basic_config, sample_input):
        """Test gradient computation through the layer."""
        layer = MobileOneBlock(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Check gradients exist for all trainable variables
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0
        assert all(g is not None for g in gradients), "Some gradients are None"

        # Check gradients have reasonable magnitudes
        for grad in gradients:
            grad_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(grad)))
            assert grad_norm > 0, "Gradient norm should be positive"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = MobileOneBlock(**basic_config)

        output = layer(sample_input, training=training)

        # Should complete successfully for all training modes
        assert output.shape == (4, 32, 32, 64)

    def test_training_vs_inference_mode_differences(self, sample_input):
        """Test that dropout in SE block behaves differently in training vs inference."""
        layer = MobileOneBlock(
            out_channels=64,
            kernel_size=3,
            use_se=True
        )

        # Multiple forward passes in training mode might give different results due to dropout
        output1_train = layer(sample_input, training=True)
        output2_train = layer(sample_input, training=True)

        # Inference mode should be deterministic
        output1_infer = layer(sample_input, training=False)
        output2_infer = layer(sample_input, training=False)

        # Inference outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1_infer),
            keras.ops.convert_to_numpy(output2_infer),
            rtol=1e-6, atol=1e-6,
            err_msg="Inference mode should be deterministic"
        )

    # ========================================================================
    # Edge Cases and Error Conditions
    # ========================================================================

    def test_invalid_parameters(self):
        """Test error conditions for invalid parameters."""
        # Invalid out_channels
        with pytest.raises(ValueError, match="out_channels must be positive"):
            MobileOneBlock(out_channels=0, kernel_size=3)

        with pytest.raises(ValueError, match="out_channels must be positive"):
            MobileOneBlock(out_channels=-1, kernel_size=3)

        # Invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            MobileOneBlock(out_channels=64, kernel_size=0)

        # Invalid stride
        with pytest.raises(ValueError, match="stride must be positive"):
            MobileOneBlock(out_channels=64, kernel_size=3, stride=0)

        # Invalid num_conv_branches. NOTE: 0 is now a legal (degenerate) value —
        # only negative counts are rejected.
        with pytest.raises(ValueError, match="num_conv_branches must be non-negative"):
            MobileOneBlock(out_channels=64, kernel_size=3, num_conv_branches=-1)

        # Invalid padding
        with pytest.raises(ValueError, match="padding must be 'same' or 'valid'"):
            MobileOneBlock(out_channels=64, kernel_size=3, padding='invalid')

    def test_compute_output_shape(self, basic_config):
        """Test output shape computation."""
        layer = MobileOneBlock(**basic_config)

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 32, 32, 64)  # same padding, stride=1
        assert output_shape == expected_shape

    def test_compute_output_shape_with_stride(self):
        """Test output shape computation with stride."""
        layer = MobileOneBlock(out_channels=64, kernel_size=3, stride=2, padding='same')

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 16, 16, 64)  # stride=2 halves dimensions
        assert output_shape == expected_shape

    def test_compute_output_shape_valid_padding(self):
        """Test output shape computation with valid padding."""
        layer = MobileOneBlock(out_channels=64, kernel_size=3, stride=1, padding='valid')

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 30, 30, 64)  # valid padding reduces by kernel_size-1
        assert output_shape == expected_shape

    # ========================================================================
    # Configuration Variations
    # ========================================================================

    @pytest.mark.parametrize("kernel_size", [1, 3, 5])
    def test_different_kernel_sizes(self, sample_input, kernel_size):
        """Test different kernel sizes."""
        layer = MobileOneBlock(out_channels=32, kernel_size=kernel_size)

        output = layer(sample_input)
        assert output.shape == (4, 32, 32, 32)

        # Check scale branch creation
        if kernel_size > 1:
            assert layer.scale_branch is not None
        else:
            assert layer.scale_branch is None

    @pytest.mark.parametrize("activation", ['relu', 'gelu', 'swish'])
    def test_different_activations(self, sample_input, activation):
        """Test different activation functions."""
        layer = MobileOneBlock(out_channels=64, kernel_size=3, activation=activation)

        output = layer(sample_input)
        assert output.shape == (4, 32, 32, 64)

    @pytest.mark.parametrize("num_branches", [1, 2, 3])
    def test_different_branch_counts(self, sample_input, num_branches):
        """Test different numbers of conv branches."""
        layer = MobileOneBlock(
            out_channels=64,
            kernel_size=3,
            num_conv_branches=num_branches
        )

        output = layer(sample_input)
        assert output.shape == (4, 32, 32, 64)
        assert len(layer.conv_branches) == num_branches

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_in_sequential_model(self, sample_input):
        """Test MobileOneBlock in a Sequential model."""
        model = keras.Sequential([
            keras.layers.Input(shape=(32, 32, 16)),
            MobileOneBlock(out_channels=64, kernel_size=3),
            MobileOneBlock(out_channels=128, kernel_size=3, stride=2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile and test forward pass
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        output = model(sample_input)
        assert output.shape == (4, 10)

    def test_multiple_blocks_serialization(self, sample_input):
        """Test serialization of model with multiple MobileOne blocks."""
        inputs = keras.Input(shape=(32, 32, 16))
        x = MobileOneBlock(out_channels=32, kernel_size=3)(inputs)
        x = MobileOneBlock(out_channels=64, kernel_size=3, stride=2, use_se=True)(x)
        x = MobileOneBlock(out_channels=128, kernel_size=3, num_conv_branches=2)(x)
        outputs = keras.layers.GlobalAveragePooling2D()(x)

        model = keras.Model(inputs, outputs)
        original_pred = model(sample_input)

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'multi_block_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Multi-block model serialization failed"
            )



class TestMobileOneBlockFastVitExtensions:
    """Pins for the ADDITIVE FastViT kwargs.

    Every kwarg added for the FastViT (timm) port must default to this layer's
    historical behaviour, because ``models/fastvlm/`` (via
    ``layers/repmixer_block.py::ConvolutionalStem``) is a live consumer. The first
    test in this class is the guard for that property; the rest pin each new axis.
    """

    @pytest.fixture
    def sample_input(self):
        """Fixed 4D input, drawn once from a pinned seed."""
        rng = np.random.default_rng(1234)
        return keras.ops.convert_to_tensor(
            rng.standard_normal((2, 16, 16, 32)).astype('float32')
        )

    @staticmethod
    def _seeded_block(seed: int, sample_input, **kwargs) -> MobileOneBlock:
        """Construct + build a block with deterministically seeded weights."""
        keras.utils.set_random_seed(seed)
        layer = MobileOneBlock(**kwargs)
        layer(sample_input, training=False)
        return layer

    # ------------------------------------------------------------------
    # 1. THE defaults-unchanged guard (plan Assumption A-2)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        'cfg_name, cfg',
        [
            # Literal fastvlm ConvolutionalStem block 1 shape (stride 2, SE on).
            ('fastvlm_stem_block_1',
             dict(out_channels=64, kernel_size=3, stride=2, use_se=True,
                  activation='gelu')),
            # stride 1 with out == in, so the identity skip branch is live too.
            ('stride1_identity_skip',
             dict(out_channels=32, kernel_size=3, stride=1, use_se=True,
                  activation='gelu')),
            # Multi-branch, no SE.
            ('two_branches_no_se',
             dict(out_channels=32, kernel_size=3, stride=1, use_se=False,
                  num_conv_branches=2, activation='gelu')),
        ],
    )
    def test_defaults_unchanged_value_identity(self, cfg_name, cfg, sample_input):
        """Default-kwargs output must equal the TRANSCRIBED PRE-CHANGE formula.

        The oracle below is a transcription of the OLD ``call()`` body:

            x = sum(conv_branch_i(inputs))
            x = x + scale_branch(inputs)        # iff kernel_size > 1
            x = x + skip_branch(inputs)         # iff stride == 1 and out == in
            x = gelu(x)                         # activation ALWAYS applied
            x = se_block(x)                     # SE strictly AFTER the activation

        It is deliberately NOT a second call into the new code: a
        default-vs-default comparison cannot be moved by any injection.
        """
        layer = self._seeded_block(0, sample_input, **cfg)

        # --- transcribed pre-change oracle -------------------------------
        expected = None
        for branch in layer.conv_branches:
            out = branch(sample_input, training=False)
            expected = out if expected is None else expected + out
        assert expected is not None, "pre-change code always had >= 1 conv branch"
        if layer.scale_branch is not None:
            expected = expected + layer.scale_branch(sample_input, training=False)
        if layer.skip_branch is not None:
            expected = expected + layer.skip_branch(sample_input, training=False)
        expected = keras.activations.gelu(expected)
        if layer.se_block is not None:
            expected = layer.se_block(expected, training=False)
        # -----------------------------------------------------------------

        actual = layer(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(actual),
            keras.ops.convert_to_numpy(expected),
            atol=1e-6, rtol=0,
            err_msg=(
                f"[{cfg_name}] default-kwargs output moved away from the "
                "pre-change formula: the extension is NOT additive"
            ),
        )

    def test_defaults_are_the_historical_values(self):
        """Every new kwarg must default to the historical behaviour."""
        layer = MobileOneBlock(out_channels=8, kernel_size=3)
        assert layer.group_size == 0
        assert layer.groups == 1
        assert layer.use_act is True
        assert layer.use_scale_branch is True
        assert layer.se_reduction_ratio == 0.25
        assert layer.se_use_bias is False
        assert layer.se_position == 'post_act'
        assert layer.padding_mode == 'keras_same'
        assert layer.norm_epsilon == pytest.approx(1e-3)

    # ------------------------------------------------------------------
    # 2. num_conv_branches=0 reduces to exactly one BatchNormalization
    # ------------------------------------------------------------------

    def test_num_conv_branches_zero_reduces_to_batchnorm(self, sample_input):
        """The degenerate FastViT RepMixer ``norm`` block IS a bare BatchNorm."""
        channels = sample_input.shape[-1]
        layer = self._seeded_block(
            7, sample_input,
            out_channels=channels, kernel_size=3, stride=1,
            num_conv_branches=0, use_scale_branch=False, group_size=1,
            use_act=False,
        )

        assert len(layer.conv_branches) == 0, "no k x k branch may be created"
        assert layer.scale_branch is None, "no scale branch may be created"
        assert layer.skip_branch is not None, "the identity BatchNorm must survive"

        # Non-degenerate BN weights, so the comparison is not satisfied by the
        # initializer defaults (gamma=1, beta=0, mean=0, var=1).
        rng = np.random.default_rng(99)
        bn_weights = [
            rng.uniform(0.5, 1.5, size=(channels,)).astype('float32'),   # gamma
            rng.standard_normal(channels).astype('float32'),             # beta
            rng.standard_normal(channels).astype('float32'),             # moving mean
            rng.uniform(0.5, 2.0, size=(channels,)).astype('float32'),   # moving var
        ]
        layer.skip_branch.set_weights(bn_weights)

        reference_bn = keras.layers.BatchNormalization()
        reference_bn.build(sample_input.shape)
        reference_bn.set_weights(bn_weights)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer(sample_input, training=False)),
            keras.ops.convert_to_numpy(reference_bn(sample_input, training=False)),
            atol=1e-6, rtol=0,
            err_msg="degenerate block is not value-identical to a bare BatchNorm",
        )

    def test_no_branch_at_all_raises(self, sample_input):
        """A configuration with zero live branches must fail loudly, not silently."""
        layer = MobileOneBlock(
            out_channels=64, kernel_size=3, stride=2,
            num_conv_branches=0, use_scale_branch=False,
        )
        with pytest.raises(ValueError, match="no active branch"):
            layer(sample_input, training=False)

    # ------------------------------------------------------------------
    # 3. group_size wiring, asserted on the BUILT kernel shape
    # ------------------------------------------------------------------

    def test_group_size_one_is_depthwise(self, sample_input):
        """``group_size=1`` must produce a DEPTHWISE kernel: (k, k, 1, C)."""
        channels = sample_input.shape[-1]
        layer = self._seeded_block(
            3, sample_input,
            out_channels=channels, kernel_size=3, group_size=1,
        )

        assert layer.groups == channels, (
            f"groups must resolve to in_channels for group_size=1, got {layer.groups}"
        )
        kernel = layer.conv_branches[0].layers[0].kernel
        assert tuple(kernel.shape) == (3, 3, 1, channels), (
            f"kxk branch kernel is not depthwise: {tuple(kernel.shape)}"
        )
        # timm applies the same group count to the 1x1 scale branch.
        scale_kernel = layer.scale_branch.layers[0].kernel
        assert tuple(scale_kernel.shape) == (1, 1, 1, channels), (
            f"scale branch kernel is not depthwise: {tuple(scale_kernel.shape)}"
        )

    def test_group_size_zero_is_dense_control(self, sample_input):
        """Control for the pin above: the default stays a dense convolution."""
        channels = sample_input.shape[-1]
        layer = self._seeded_block(
            3, sample_input, out_channels=channels, kernel_size=3,
        )
        assert layer.groups == 1
        kernel = layer.conv_branches[0].layers[0].kernel
        assert tuple(kernel.shape) == (3, 3, channels, channels)

    def test_group_size_intermediate(self, sample_input):
        """``group_size=k`` gives ``groups = in_channels // k``."""
        channels = sample_input.shape[-1]  # 32
        layer = self._seeded_block(
            3, sample_input, out_channels=channels, kernel_size=3, group_size=8,
        )
        assert layer.groups == channels // 8  # 4
        kernel = layer.conv_branches[0].layers[0].kernel
        assert tuple(kernel.shape) == (3, 3, channels // 4, channels)

    # ------------------------------------------------------------------
    # 4. se_position
    # ------------------------------------------------------------------

    def test_se_position_changes_output(self, sample_input):
        """``'pre_act'`` (timm) and ``'post_act'`` (historical) must differ."""
        cfg = dict(out_channels=32, kernel_size=3, stride=1, use_se=True,
                   activation='gelu')
        post = self._seeded_block(11, sample_input, se_position='post_act', **cfg)
        pre = self._seeded_block(11, sample_input, se_position='pre_act', **cfg)

        # Same seed => same weights; otherwise the comparison below is meaningless.
        for w_post, w_pre in zip(post.weights, pre.weights):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(w_post),
                keras.ops.convert_to_numpy(w_pre),
                atol=0, rtol=0,
                err_msg="seeding failed: the two blocks do not share weights",
            )

        out_post = keras.ops.convert_to_numpy(post(sample_input, training=False))
        out_pre = keras.ops.convert_to_numpy(pre(sample_input, training=False))

        assert np.all(np.isfinite(out_post)) and np.all(np.isfinite(out_pre))
        max_delta = float(np.max(np.abs(out_post - out_pre)))
        assert max_delta > 1e-4, (
            f"se_position is not wired: pre_act and post_act agree to {max_delta}"
        )

    def test_se_position_pre_act_matches_timm_formula(self, sample_input):
        """``'pre_act'`` is exactly ``act(se(sum_of_branches))``."""
        layer = self._seeded_block(
            13, sample_input,
            out_channels=32, kernel_size=3, stride=1, use_se=True,
            activation='gelu', se_position='pre_act',
        )

        summed = None
        for branch in layer.conv_branches:
            out = branch(sample_input, training=False)
            summed = out if summed is None else summed + out
        summed = summed + layer.scale_branch(sample_input, training=False)
        summed = summed + layer.skip_branch(sample_input, training=False)
        expected = keras.activations.gelu(
            layer.se_block(summed, training=False)
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer(sample_input, training=False)),
            keras.ops.convert_to_numpy(expected),
            atol=1e-6, rtol=0,
            err_msg="pre_act does not implement act(se(x))",
        )

    # ------------------------------------------------------------------
    # 5. use_act / use_scale_branch
    # ------------------------------------------------------------------

    def test_use_act_false_skips_activation(self, sample_input):
        """``use_act=False`` must return the raw branch sum."""
        cfg = dict(out_channels=32, kernel_size=3, stride=1, activation='gelu')
        layer = self._seeded_block(17, sample_input, use_act=False, **cfg)

        expected = None
        for branch in layer.conv_branches:
            out = branch(sample_input, training=False)
            expected = out if expected is None else expected + out
        expected = expected + layer.scale_branch(sample_input, training=False)
        expected = expected + layer.skip_branch(sample_input, training=False)

        actual = layer(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(actual),
            keras.ops.convert_to_numpy(expected),
            atol=1e-6, rtol=0,
            err_msg="use_act=False still applied the activation",
        )

        # And it genuinely differs from the activated default.
        activated = self._seeded_block(17, sample_input, use_act=True, **cfg)
        delta = np.max(np.abs(
            keras.ops.convert_to_numpy(actual)
            - keras.ops.convert_to_numpy(activated(sample_input, training=False))
        ))
        assert delta > 1e-4, "use_act had no observable effect"

    def test_use_scale_branch_false_omits_branch(self, sample_input):
        """``use_scale_branch=False`` drops the branch, its weights and its value."""
        cfg = dict(out_channels=32, kernel_size=3, stride=1, activation='gelu')
        with_scale = self._seeded_block(19, sample_input, **cfg)
        without_scale = self._seeded_block(
            19, sample_input, use_scale_branch=False, **cfg)

        assert with_scale.scale_branch is not None
        assert without_scale.scale_branch is None

        # A Conv2D(no bias) + BatchNormalization contributes 1 + 4 = 5 weights.
        assert len(without_scale.weights) == len(with_scale.weights) - 5, (
            f"weight count did not drop by the scale branch's 5 weights: "
            f"{len(with_scale.weights)} -> {len(without_scale.weights)}"
        )

        delta = np.max(np.abs(
            keras.ops.convert_to_numpy(with_scale(sample_input, training=False))
            - keras.ops.convert_to_numpy(without_scale(sample_input, training=False))
        ))
        assert delta > 1e-4, "omitting the scale branch did not change the output"

    def test_se_reduction_ratio_and_bias_are_forwarded(self, sample_input):
        """SE knobs reach the SqueezeExcitation sub-layer."""
        layer = MobileOneBlock(
            out_channels=32, kernel_size=3, use_se=True,
            se_reduction_ratio=1.0 / 16.0, se_use_bias=True,
        )
        assert layer.se_block.reduction_ratio == pytest.approx(1.0 / 16.0)
        assert layer.se_block.use_bias is True

    # ------------------------------------------------------------------
    # 6. Validation
    # ------------------------------------------------------------------

    def test_invalid_se_position_raises(self):
        with pytest.raises(ValueError, match="se_position must be 'post_act' or 'pre_act'"):
            MobileOneBlock(out_channels=32, kernel_size=3, se_position='mid_act')

    def test_invalid_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size must be non-negative"):
            MobileOneBlock(out_channels=32, kernel_size=3, group_size=-1)

    def test_group_size_indivisible_raises(self, sample_input):
        """32 input channels cannot be split into groups of 3."""
        layer = MobileOneBlock(out_channels=32, kernel_size=3, group_size=3)
        with pytest.raises(
            ValueError,
            match=r"group_size must divide the input channels.*in_channels=32.*group_size=3",
        ):
            layer(sample_input, training=False)

    def test_out_channels_indivisible_by_groups_raises(self, sample_input):
        """32 in-channels at group_size=1 gives 32 groups, which 48 out-channels
        cannot be split across."""
        layer = MobileOneBlock(out_channels=48, kernel_size=3, group_size=1)
        with pytest.raises(
            ValueError,
            match=r"resolved groups=32 must divide both in_channels=32 and out_channels=48",
        ):
            layer(sample_input, training=False)

    # ------------------------------------------------------------------
    # 7. Serialization of a NON-default configuration
    # ------------------------------------------------------------------

    def test_serialization_cycle_non_default_config(self, sample_input):
        """Round trip the degenerate FastViT config BY VALUE."""
        channels = sample_input.shape[-1]
        keras.utils.set_random_seed(23)
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MobileOneBlock(
            out_channels=channels, kernel_size=3, stride=1,
            group_size=1, use_act=False, use_scale_branch=False,
            num_conv_branches=0, use_se=True, se_position='pre_act',
            se_reduction_ratio=1.0 / 16.0, se_use_bias=True,
        )(inputs)
        model = keras.Model(inputs, outputs)
        original_pred = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'fastvit_config.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_pred),
            keras.ops.convert_to_numpy(loaded_pred),
            atol=1e-6, rtol=0,
            err_msg="non-default FastViT config did not survive the .keras round trip",
        )

    def test_serialization_cycle_grouped_config(self, sample_input):
        """Grouped conv branches are constructed in build(); prove that round trips."""
        channels = sample_input.shape[-1]
        keras.utils.set_random_seed(29)
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MobileOneBlock(
            out_channels=channels, kernel_size=3, stride=1, group_size=1,
            use_act=False, num_conv_branches=2,
        )(inputs)
        model = keras.Model(inputs, outputs)
        original_pred = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'grouped_config.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_pred),
            keras.ops.convert_to_numpy(loaded_pred),
            atol=1e-6, rtol=0,
            err_msg="grouped config did not survive the .keras round trip",
        )

    def test_config_contains_new_keys(self):
        layer = MobileOneBlock(
            out_channels=32, kernel_size=3, group_size=1, use_act=False,
            use_scale_branch=False, num_conv_branches=0,
            se_reduction_ratio=0.5, se_use_bias=True, se_position='pre_act',
        )
        config = layer.get_config()
        assert config['group_size'] == 1
        assert config['use_act'] is False
        assert config['use_scale_branch'] is False
        assert config['num_conv_branches'] == 0
        assert config['se_reduction_ratio'] == 0.5
        assert config['se_use_bias'] is True
        assert config['se_position'] == 'pre_act'

        reference = MobileOneBlock(
            out_channels=32, kernel_size=3, stride=2,
            padding_mode='reference', norm_epsilon=1e-5,
        ).get_config()
        assert reference['padding_mode'] == 'reference'
        assert reference['norm_epsilon'] == pytest.approx(1e-5)


class TestMobileOneBlockNormEpsilonAndPadding:
    """Pins for the two REFERENCE-FIDELITY axes: BN epsilon and padding grid.

    Both were shipped wrong and survived 400 green tests, because a normalization
    epsilon and a padding convention are invisible to every shape assertion. The
    default of each kwarg is Keras' own behaviour so ``models/fastvlm/`` is
    untouched; the FastViT port opts in.
    """

    @staticmethod
    def _batch_norms(layer):
        """Every BatchNormalization reachable from a built block."""
        return [
            l for l in layer._flatten_layers()
            if isinstance(l, keras.layers.BatchNormalization)
        ]

    @staticmethod
    def _conv_of(branch):
        """The single Conv2D inside a branch ``Sequential``."""
        convs = [l for l in branch.layers if isinstance(l, keras.layers.Conv2D)]
        assert len(convs) == 1, f"expected one Conv2D, got {len(convs)}"
        return convs[0]

    # ------------------------------------------------------------------
    # BN epsilon
    # ------------------------------------------------------------------

    def test_mobile_one_block_default_epsilon_unchanged(self):
        """At DEFAULT kwargs every BN keeps Keras' 1e-3 (fastvlm is untouched).

        This is the counterpart of the fidelity pin: it is the assertion that
        would fire if someone "fixed" the default to the reference value and
        silently moved every ``models/fastvlm/`` numeric.
        """
        layer = MobileOneBlock(out_channels=32, kernel_size=3, stride=1)
        layer.build((None, 8, 8, 32))

        norms = self._batch_norms(layer)
        # 1 conv branch + 1 scale branch + 1 identity skip.
        assert len(norms) == 3, f"expected 3 BatchNormalizations, got {len(norms)}"
        for bn in norms:
            assert bn.epsilon == pytest.approx(1e-3), (
                f"{bn.name} epsilon is {bn.epsilon}, not Keras' default 1e-3 — "
                "the DEFAULT behaviour of the shared block has moved"
            )

    def test_norm_epsilon_reaches_every_batch_norm(self):
        """``norm_epsilon`` must reach the conv, scale AND identity-skip norms.

        The identity-skip norm is the one that matters most: in
        ``FastVitRepMixer.norm`` it is the block's ENTIRE content.
        """
        layer = MobileOneBlock(
            out_channels=32, kernel_size=3, stride=1, num_conv_branches=2,
            norm_epsilon=1e-5,
        )
        layer.build((None, 8, 8, 32))

        norms = self._batch_norms(layer)
        assert len(norms) == 4, f"expected 4 BatchNormalizations, got {len(norms)}"
        names = {bn.name for bn in norms}
        assert 'skip_branch_bn' in names, (
            f"the identity-skip norm was not created; got {sorted(names)}")
        assert 'scale_branch_bn' in names
        for bn in norms:
            assert bn.epsilon == pytest.approx(1e-5), (
                f"{bn.name} kept epsilon {bn.epsilon}; norm_epsilon did not reach it"
            )

    # ------------------------------------------------------------------
    # Padding grid
    # ------------------------------------------------------------------

    @staticmethod
    def _dirac_block(padding_mode, live_branch):
        """A stride-2 block whose ONLY live branch carries a Dirac kernel.

        Both branches are always CONSTRUCTED (so the structural shape is
        identical between the two arms); the dead one is zeroed. The output is
        therefore exactly the set of input pixels the live branch samples.
        """
        block = MobileOneBlock(
            out_channels=1, kernel_size=3, stride=2, use_act=False,
            num_conv_branches=1, padding_mode=padding_mode,
        )
        block.build((None, 4, 4, 1))
        for branch, k, tag in (
                (block.conv_branches[0], 3, 'kxk'),
                (block.scale_branch, 1, 'scale'),
        ):
            conv = TestMobileOneBlockNormEpsilonAndPadding._conv_of(branch)
            w = np.zeros(tuple(conv.kernel.shape), dtype='float32')
            if tag == live_branch:
                w[k // 2, k // 2, 0, 0] = 1.0
            conv.set_weights([w])
        x = np.arange(16, dtype='float32').reshape(1, 4, 4, 1)
        # BatchNormalization at inference with fresh statistics is
        # x / sqrt(1 + eps); undo it so the output is the raw sampled index.
        scale = float(np.sqrt(1.0 + block.norm_epsilon))
        return np.array(
            keras.ops.convert_to_numpy(block(x, training=False))
        )[0, :, :, 0] * scale

    def test_strided_branches_sample_the_same_grid(self):
        """Under ``'reference'`` the k x k and 1 x 1 branches read the SAME pixels.

        MEASURED with Keras' own ``padding='same'`` at stride 2 on
        ``arange(16).reshape(1, 4, 4, 1)``: the 1x1 branch returns
        ``[[0, 2], [8, 10]]`` while the 3x3 branch returns ``[[5, 7], [13, 15]]``
        — a one-pixel offset, because Keras puts the extra pad at the
        bottom/right while PyTorch's ``padding=k//2`` is symmetric. Summing two
        branches that read different pixels is not the reference's function AND
        makes the block unfusible.
        """
        kxk = self._dirac_block('reference', 'kxk')
        scale = self._dirac_block('reference', 'scale')

        np.testing.assert_allclose(
            kxk, scale, atol=1e-5, rtol=0,
            err_msg=(
                f"strided branches sample different grids under 'reference' "
                f"padding: kxk={kxk.tolist()} vs 1x1={scale.tolist()}"
            ),
        )
        # Pin the IDENTITY of the grid, not just that the two agree: a
        # both-branches-broken change could agree on the wrong pixels.
        np.testing.assert_allclose(
            kxk, np.array([[0.0, 2.0], [8.0, 10.0]]), atol=1e-5, rtol=0,
            err_msg=f"reference grid is not i*stride: {kxk.tolist()}",
        )

    def test_keras_same_strided_branches_disagree(self):
        """The DEFAULT mode still has the (disclosed) misalignment.

        This is the control for the test above: it proves the probe is
        discriminating, and it pins that the default path was NOT silently
        changed under ``models/fastvlm/``.
        """
        kxk = self._dirac_block('keras_same', 'kxk')
        scale = self._dirac_block('keras_same', 'scale')
        np.testing.assert_allclose(
            kxk, np.array([[5.0, 7.0], [13.0, 15.0]]), atol=1e-5, rtol=0)
        np.testing.assert_allclose(
            scale, np.array([[0.0, 2.0], [8.0, 10.0]]), atol=1e-5, rtol=0)

    def test_reference_padding_matches_symmetric_manual(self):
        """A ``'reference'`` branch equals ZeroPadding2D(k//2) + a valid conv.

        The oracle is built OUTSIDE the layer from raw Keras layers, so it cannot
        be moved by any change to ``MobileOneBlock``.
        """
        rng = np.random.default_rng(20260814)
        x = rng.standard_normal((2, 9, 9, 4)).astype('float32')

        block = MobileOneBlock(
            out_channels=6, kernel_size=3, stride=2, use_act=False,
            num_conv_branches=1, padding_mode='reference',
        )
        block(x, training=False)
        conv = self._conv_of(block.conv_branches[0])
        kernel = np.array(keras.ops.convert_to_numpy(conv.kernel))

        manual_pad = keras.layers.ZeroPadding2D(padding=3 // 2)
        manual_conv = keras.layers.Conv2D(
            filters=6, kernel_size=3, strides=2, padding='valid', use_bias=False)
        manual_conv.build((None, 11, 11, 4))
        manual_conv.set_weights([kernel])
        expected = keras.ops.convert_to_numpy(manual_conv(manual_pad(x)))

        # Run the branch up to and including its convolution, whatever the
        # branch's internal layer order happens to be.
        actual = x
        for sub in block.conv_branches[0].layers:
            actual = sub(actual)
            if isinstance(sub, keras.layers.Conv2D):
                break
        actual = keras.ops.convert_to_numpy(actual)

        np.testing.assert_allclose(
            actual, expected, atol=0, rtol=0,
            err_msg="'reference' padding is not the symmetric k//2 convolution",
        )
        assert expected.shape == (2, 5, 5, 6), (
            f"symmetric padding changed the geometry: {expected.shape}")

    def test_default_padding_mode_unchanged(self):
        """The DEFAULT block still equals a plain ``padding='same'`` convolution.

        A fastvlm-shaped config (stride-2 k3, the ``ConvolutionalStem`` shape),
        compared against a hand-built ``Conv2D(padding='same') + BatchNormalization``
        with Keras' own defaults — an oracle that lives entirely outside the
        layer, so it goes RED if either new default moves.
        """
        rng = np.random.default_rng(31337)
        x = rng.standard_normal((2, 16, 16, 8)).astype('float32')

        block = MobileOneBlock(
            out_channels=8, kernel_size=3, stride=2, use_act=False,
            num_conv_branches=1, use_scale_branch=False,
        )
        block(x, training=False)
        branch = block.conv_branches[0]
        assert not any(
            isinstance(l, keras.layers.ZeroPadding2D) for l in branch.layers
        ), "the DEFAULT mode must not insert an explicit padding layer"
        conv = self._conv_of(branch)
        assert conv.padding == 'same'

        manual_conv = keras.layers.Conv2D(
            filters=8, kernel_size=3, strides=2, padding='same', use_bias=False)
        manual_conv.build((None, 16, 16, 8))
        manual_conv.set_weights([np.array(keras.ops.convert_to_numpy(conv.kernel))])
        manual_bn = keras.layers.BatchNormalization()
        manual_bn.build((None, 8, 8, 8))
        expected = keras.ops.convert_to_numpy(
            manual_bn(manual_conv(x), training=False))

        actual = keras.ops.convert_to_numpy(block(x, training=False))

        np.testing.assert_allclose(
            actual, expected, atol=1e-6, rtol=0,
            err_msg=(
                "the DEFAULT MobileOneBlock no longer equals Keras' own "
                "Conv2D(padding='same') + BatchNormalization(): a default moved "
                "and models/fastvlm/ has been silently changed"
            ),
        )

    def test_reference_mode_geometry_is_unchanged_for_odd_kernels(self):
        """Symmetric padding must not move the 256px feature-map ladder.

        For an ODD kernel, ``floor((H - 1) / s) + 1 == ceil(H / s)``, so the
        reference convention gives exactly the geometry ``'same'`` gives. This is
        MEASURED here rather than assumed, because it is the property that lets
        the mode be applied uniformly (stride 1 included).
        """
        for size in (256, 128, 64, 32, 16, 9, 7, 4):
            for stride in (1, 2):
                for k in (1, 3, 7):
                    block = MobileOneBlock(
                        out_channels=2, kernel_size=k, stride=stride,
                        use_act=False, padding_mode='reference')
                    got = block.compute_output_shape((None, size, size, 2))
                    expected = -(-size // stride)  # ceil
                    assert got[1] == expected and got[2] == expected, (
                        f"size={size} stride={stride} k={k}: "
                        f"got {got[1:3]}, expected {expected}")

    def test_reference_mode_stride_one_odd_kernel_is_value_identical(self):
        """At stride 1 with an odd kernel the two modes agree ELEMENTWISE.

        This is the measurement that justifies applying ``'reference'`` uniformly
        at every FastViT site instead of only where ``stride > 1``.
        """
        rng = np.random.default_rng(99)
        x = rng.standard_normal((2, 11, 11, 6)).astype('float32')

        blocks = {}
        for mode in ('keras_same', 'reference'):
            keras.utils.set_random_seed(7)
            b = MobileOneBlock(
                out_channels=6, kernel_size=3, stride=1, use_act=False,
                num_conv_branches=1, padding_mode=mode)
            b(x, training=False)
            blocks[mode] = b

        # Transplant the kernels so the ONLY difference is the padding path.
        src = self._conv_of(blocks['keras_same'].conv_branches[0])
        dst = self._conv_of(blocks['reference'].conv_branches[0])
        dst.set_weights([np.array(keras.ops.convert_to_numpy(src.kernel))])
        src_scale = self._conv_of(blocks['keras_same'].scale_branch)
        dst_scale = self._conv_of(blocks['reference'].scale_branch)
        dst_scale.set_weights(
            [np.array(keras.ops.convert_to_numpy(src_scale.kernel))])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(blocks['reference'](x, training=False)),
            keras.ops.convert_to_numpy(blocks['keras_same'](x, training=False)),
            atol=0, rtol=0,
            err_msg=(
                "stride-1 odd-kernel 'reference' padding is NOT identical to "
                "'same'; applying the mode uniformly would change behaviour"
            ),
        )

    def test_invalid_padding_mode_raises(self):
        with pytest.raises(ValueError, match='padding_mode'):
            MobileOneBlock(out_channels=8, kernel_size=3, padding_mode='torch')

    def test_invalid_norm_epsilon_raises(self):
        with pytest.raises(ValueError, match='norm_epsilon'):
            MobileOneBlock(out_channels=8, kernel_size=3, norm_epsilon=0.0)

    def test_reference_config_round_trips_by_value(self):
        """A reference-padding, reference-epsilon block survives ``.keras``."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal((2, 16, 16, 8)).astype('float32')

        keras.utils.set_random_seed(11)
        inputs = keras.Input(shape=(16, 16, 8))
        outputs = MobileOneBlock(
            out_channels=16, kernel_size=3, stride=2,
            padding_mode='reference', norm_epsilon=1e-5,
        )(inputs)
        model = keras.Model(inputs, outputs)
        original = keras.ops.convert_to_numpy(model(x, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'reference_mode.keras')
            model.save(filepath)
            loaded = keras.models.load_model(filepath)
            restored = keras.ops.convert_to_numpy(loaded(x, training=False))

        np.testing.assert_allclose(
            original, restored, atol=1e-6, rtol=0,
            err_msg="reference-mode config did not survive the .keras round trip",
        )
        inner = [
            l for l in loaded._flatten_layers()
            if isinstance(l, MobileOneBlock)
        ][0]
        assert inner.padding_mode == 'reference'
        assert inner.norm_epsilon == pytest.approx(1e-5)
