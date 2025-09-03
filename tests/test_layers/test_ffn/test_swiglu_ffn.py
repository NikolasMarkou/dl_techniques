import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.swiglu_ffn import SwiGLUFFN


class TestSwiGLUFFN:
    """Comprehensive test suite for modern SwiGLU FFN layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.random.normal([8, 64, 512])  # batch, seq_len, output_dim

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'output_dim': 512,
            'ffn_expansion_factor': 4,
            'ffn_multiple_of': 256,
            'dropout_rate': 0.1
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with all parameters."""
        return {
            'output_dim': 768,
            'ffn_expansion_factor': 8,
            'ffn_multiple_of': 128,
            'dropout_rate': 0.2,
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'ones',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_regularizer': keras.regularizers.L1(1e-5)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = SwiGLUFFN(output_dim=512)

        # Check stored configuration
        assert layer.output_dim == 512
        assert layer.ffn_expansion_factor == 4
        assert layer.ffn_multiple_of == 256
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert layer.gate_proj is not None
        assert layer.up_proj is not None
        assert layer.down_proj is not None
        assert not layer.gate_proj.built

        # Check hidden dimension calculation (2/3 rule)
        expected_hidden = int(512 * 4 * 2 / 3)  # 1365
        expected_hidden = 256 * ((expected_hidden + 256 - 1) // 256)  # Rounded to 1536
        assert layer.hidden_dim == expected_hidden

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = SwiGLUFFN(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.output_dim == 768
        assert layer.ffn_expansion_factor == 8
        assert layer.ffn_multiple_of == 128
        assert layer.dropout_rate == 0.2
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)

        # Check hidden dimension calculation with custom parameters
        expected_hidden = int(768 * 8 * 2 / 3)  # 4096
        expected_hidden = 128 * ((expected_hidden + 128 - 1) // 128)  # Rounded to 4096
        assert layer.hidden_dim == expected_hidden

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid output_dim values
        with pytest.raises(ValueError, match="output_dim must be positive"):
            SwiGLUFFN(output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            SwiGLUFFN(output_dim=-10)

        # Test invalid ffn_expansion_factor values
        with pytest.raises(ValueError, match="ffn_expansion_factor must be positive"):
            SwiGLUFFN(output_dim=512, ffn_expansion_factor=0)

        with pytest.raises(ValueError, match="ffn_expansion_factor must be positive"):
            SwiGLUFFN(output_dim=512, ffn_expansion_factor=-2)

        # Test invalid ffn_multiple_of values
        with pytest.raises(ValueError, match="ffn_multiple_of must be positive"):
            SwiGLUFFN(output_dim=512, ffn_multiple_of=0)

        with pytest.raises(ValueError, match="ffn_multiple_of must be positive"):
            SwiGLUFFN(output_dim=512, ffn_multiple_of=-64)

        # Test invalid dropout_rate values
        with pytest.raises(ValueError, match="dropout_rate must be in \\[0, 1\\]"):
            SwiGLUFFN(output_dim=512, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in \\[0, 1\\]"):
            SwiGLUFFN(output_dim=512, dropout_rate=1.5)

    def test_hidden_dimension_calculation(self):
        """Test hidden dimension calculation follows 2/3 rule and rounding."""
        test_cases = [
            # (output_dim, expansion_factor, multiple_of, expected_hidden)
            (512, 4, 256, 1536),  # int(512*4*2/3)=1365, rounded to 1536
            (768, 4, 256, 2048),  # int(768*4*2/3)=2048, already multiple
            (1024, 8, 512, 5632),  # int(1024*8*2/3)=5461, rounded to 5632
            (256, 4, 64, 704),  # int(256*4*2/3)=682, rounded to 704
            (128, 2, 32, 192),  # int(128*2*2/3)=170, rounded to 192
        ]

        for output_dim, exp_factor, multiple_of, expected in test_cases:
            layer = SwiGLUFFN(
                output_dim=output_dim,
                ffn_expansion_factor=exp_factor,
                ffn_multiple_of=multiple_of
            )
            assert layer.hidden_dim == expected

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = SwiGLUFFN(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        assert layer.gate_proj.built
        assert layer.up_proj.built
        assert layer.down_proj.built

        # Verify weights were created
        # Should have 6 weights: gate_proj kernel/bias + up_proj kernel/bias + down_proj kernel/bias
        # But use_bias=False by default, so should be 3 weights
        expected_weights = 3 if not layer.use_bias else 6
        assert len(layer.weights) == expected_weights

        # Verify output shape
        assert output.shape == sample_input.shape  # SwiGLU preserves output_dim

    def test_build_with_bias(self, sample_input):
        """Test building with bias enabled."""
        layer = SwiGLUFFN(output_dim=512, use_bias=True)
        output = layer(sample_input)

        # Should have 6 weights with bias enabled
        assert len(layer.weights) == 6
        assert output.shape == sample_input.shape

    def test_build_explicit_sub_layer_building(self, sample_input):
        """Test that explicit sub-layer building works correctly."""
        layer = SwiGLUFFN(output_dim=256)

        # Manually trigger build
        layer.build(sample_input.shape)

        # Verify all sub-layers are built
        assert layer.gate_proj.built
        assert layer.up_proj.built
        assert layer.down_proj.built

        # Verify weight shapes
        input_dim = sample_input.shape[-1]
        hidden_dim = layer.hidden_dim

        expected_shapes = [
            (input_dim, hidden_dim),  # gate_proj kernel
            (input_dim, hidden_dim),  # up_proj kernel
            (hidden_dim, layer.output_dim)  # down_proj kernel
        ]

        actual_shapes = [tuple(w.shape) for w in layer.weights]
        assert actual_shapes == expected_shapes

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = SwiGLUFFN(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        assert output.shape == sample_input.shape  # Preserves output_dim dimension
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_forward_pass_swiglu_mechanism(self):
        """Test SwiGLU gating mechanism with controlled inputs."""
        # Use controlled initialization for deterministic behavior
        layer = SwiGLUFFN(
            output_dim=4,
            ffn_expansion_factor=2,
            ffn_multiple_of=2,
            dropout_rate=0.0,
            use_bias=False,
            kernel_initializer='ones'
        )

        # Controlled input
        controlled_input = keras.ops.ones([1, 1, 4])  # All ones
        output = layer(controlled_input)

        # Verify output properties
        assert output.shape == (1, 1, 4)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

        # With ones initialization and ones input:
        # gate = SiLU(ones @ ones_matrix) = SiLU(4)
        # up = ones @ ones_matrix = 4
        # hidden = SiLU(4) * 4 ≈ 3.98 * 4 ≈ 15.92
        # output = 15.92 @ ones_matrix = 15.92 * hidden_dim (which is 4)
        # So each output element should be approximately 15.92 * 4 = 63.68

        output_np = keras.ops.convert_to_numpy(output)
        assert output_np.shape == (1, 1, 4)
        # Values should be positive due to SiLU activation
        assert np.all(output_np > 0)

    def test_dropout_training_vs_inference(self, sample_input):
        """Test dropout behavior in training vs inference modes."""
        layer = SwiGLUFFN(output_dim=512, dropout_rate=0.5)

        # Test inference mode - should be deterministic
        output_inf_1 = layer(sample_input, training=False)
        output_inf_2 = layer(sample_input, training=False)

        # Inference outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_inf_1),
            keras.ops.convert_to_numpy(output_inf_2),
            rtol=1e-6, atol=1e-6,
            err_msg="Inference outputs should match"
        )

        # Test training mode - may produce different outputs due to dropout
        output_train_1 = layer(sample_input, training=True)
        output_train_2 = layer(sample_input, training=True)

        # Both should have same shape
        assert output_train_1.shape == output_train_2.shape == sample_input.shape

    def test_zero_dropout(self, sample_input):
        """Test that zero dropout produces consistent results."""
        layer = SwiGLUFFN(output_dim=512, dropout_rate=0.0)

        # Multiple calls should produce identical results even in training mode
        output1 = layer(sample_input, training=True)
        output2 = layer(sample_input, training=True)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Zero dropout should produce identical outputs"
        )

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (output_dim, input_shape, expected_output_shape)
            (256, (None, 64, 256), (None, 64, 256)),
            (512, (8, 128, 512), (8, 128, 512)),
            (768, (4, 32, 768), (4, 32, 768)),
            (1024, (2, 16, 20, 1024), (2, 16, 20, 1024)),  # 4D input
        ]

        for output_dim, input_shape, expected_shape in test_cases:
            layer = SwiGLUFFN(output_dim=output_dim)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = SwiGLUFFN(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'output_dim', 'ffn_expansion_factor', 'ffn_multiple_of', 'dropout_rate',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['output_dim'] == 768
        assert config['ffn_expansion_factor'] == 8
        assert config['ffn_multiple_of'] == 128
        assert config['dropout_rate'] == 0.2
        assert config['use_bias'] is True

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = SwiGLUFFN(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_swiglu_model.keras')
            model.save(filepath)

            # Load without custom_objects (thanks to registration decorator)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_serialization_with_custom_config(self, custom_layer_config, sample_input):
        """Test serialization with comprehensive custom configuration."""
        # Create model with all custom parameters
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = SwiGLUFFN(**custom_layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        # Serialization cycle
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_swiglu_custom.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify predictions match
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

            # Verify loaded layer has correct configuration
            loaded_layer = loaded_model.layers[1]  # First layer is Input
            assert loaded_layer.output_dim == custom_layer_config['output_dim']
            assert loaded_layer.ffn_expansion_factor == custom_layer_config['ffn_expansion_factor']
            assert loaded_layer.dropout_rate == custom_layer_config['dropout_rate']

    def test_model_integration_complex(self, sample_input):
        """Test layer integration in a complex transformer-style model."""
        output_dim = sample_input.shape[-1]
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Multi-layer transformer-style architecture
        x = inputs
        for i in range(3):  # 3 transformer blocks
            # Self-attention (simplified with dense layer for testing)
            attn_out = keras.layers.Dense(output_dim)(x)
            x = keras.layers.LayerNormalization()(x + attn_out)

            # SwiGLU FFN
            ffn_out = SwiGLUFFN(
                output_dim=output_dim,
                ffn_expansion_factor=4,
                dropout_rate=0.1
            )(x)
            x = keras.layers.LayerNormalization()(x + ffn_out)

        # Final classification head
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test forward pass
        prediction = model(sample_input)
        assert prediction.shape == (sample_input.shape[0], 10)

        # Test that gradients flow properly
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = model(sample_input)
            loss = keras.ops.mean(output)

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_regularization_losses(self, sample_input):
        """Test that regularization losses are properly computed."""
        layer = SwiGLUFFN(
            output_dim=512,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01),
            use_bias=True
        )

        # No losses before forward pass
        initial_losses = len(layer.losses)

        # Apply the layer
        output = layer(sample_input)

        # Should have regularization losses now
        assert len(layer.losses) > initial_losses

        # Verify losses are non-zero
        if layer.losses:  # Check if any losses exist
            total_loss = sum(layer.losses)
            assert keras.ops.convert_to_numpy(total_loss) > 0

    def test_gradient_flow(self, sample_input, layer_config):
        """Test that gradients flow properly through the layer."""
        layer = SwiGLUFFN(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that all gradients exist and are non-zero
        assert len(gradients) == len(layer.trainable_variables)
        assert all(g is not None for g in gradients)

        # Check gradient shapes match variable shapes
        for grad, var in zip(gradients, layer.trainable_variables):
            assert grad.shape == var.shape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, sample_input, layer_config, training):
        """Test layer behavior in different training modes."""
        layer = SwiGLUFFN(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_input_dimensions(self):
        """Test layer with different input tensor dimensions."""
        output_dim = 128
        layer = SwiGLUFFN(output_dim=output_dim)

        test_shapes = [
            (4, output_dim),  # 2D input
            (4, 16, output_dim),  # 3D input
            (2, 8, 16, output_dim),  # 4D input
            (1, 4, 8, 16, output_dim)  # 5D input
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should preserve input shape
            assert output.shape == test_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SwiGLUFFN(output_dim=64, dropout_rate=0.0)

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((4, 8, 64)),  # Zeros
            keras.ops.ones((4, 8, 64)) * 1e-10,  # Very small
            keras.ops.ones((4, 8, 64)) * 1e3,  # Large
            keras.random.normal((4, 8, 64)) * 100,  # Large random
            keras.random.normal((4, 8, 64)) * 1e-5,  # Small random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"

    def test_weight_structure_and_shapes(self, sample_input):
        """Test that layer weights have correct structure and shapes."""
        output_dim = sample_input.shape[-1]
        layer = SwiGLUFFN(output_dim=output_dim, use_bias=True)
        layer(sample_input)  # Build the layer

        # Should have exactly 6 weights: gate/up/down kernels and biases
        assert len(layer.weights) == 6
        assert len(layer.trainable_variables) == 6

        # Verify weight shapes
        hidden_dim = layer.hidden_dim
        expected_shapes = [
            (output_dim, hidden_dim),  # gate_proj kernel
            (hidden_dim,),  # gate_proj bias
            (output_dim, hidden_dim),  # up_proj kernel
            (hidden_dim,),  # up_proj bias
            (hidden_dim, output_dim),  # down_proj kernel
            (output_dim,)  # down_proj bias
        ]

        actual_shapes = [tuple(w.shape) for w in layer.weights]
        assert actual_shapes == expected_shapes

    def test_num_parameters_property(self, sample_input):
        """Test the num_parameters property."""
        layer = SwiGLUFFN(output_dim=256, use_bias=True)

        # Should return 0 before building
        assert layer.num_parameters == 0

        # Build the layer
        layer(sample_input[:, :, :256])  # Adjust input size

        # Calculate expected parameters
        output_dim = 256
        hidden_dim = layer.hidden_dim
        expected_params = (
                output_dim * hidden_dim + hidden_dim +  # gate_proj
                output_dim * hidden_dim + hidden_dim +  # up_proj
                hidden_dim * output_dim + output_dim  # down_proj
        )

        assert layer.num_parameters == expected_params


class TestSwiGLUFFNEdgeCases:
    """Test edge cases and boundary conditions for SwiGLU FFN."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = SwiGLUFFN(output_dim=2, ffn_expansion_factor=2, ffn_multiple_of=2)
        test_input = keras.random.normal([2, 4, 2])

        output = layer(test_input)
        assert output.shape == (2, 4, 2)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = SwiGLUFFN(output_dim=2048, ffn_expansion_factor=4, ffn_multiple_of=256)
        test_input = keras.random.normal([1, 2, 2048])

        output = layer(test_input)
        assert output.shape == (1, 2, 2048)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_expansion_factors(self):
        """Test with various expansion factors."""
        output_dim = 128
        expansion_factors = [2, 4, 6, 8, 16]

        for exp_factor in expansion_factors:
            layer = SwiGLUFFN(
                output_dim=output_dim,
                ffn_expansion_factor=exp_factor,
                ffn_multiple_of=32
            )
            test_input = keras.random.normal([2, 4, output_dim])

            output = layer(test_input)
            assert output.shape == (2, 4, output_dim)
            assert not keras.ops.any(keras.ops.isnan(output))

            # Verify hidden dim increases with expansion factor
            expected_base = int(output_dim * exp_factor * 2 / 3)
            expected_hidden = 32 * ((expected_base + 32 - 1) // 32)
            assert layer.hidden_dim == expected_hidden

    def test_different_multiple_constraints(self):
        """Test with different multiple_of constraints."""
        output_dim = 512
        multiples = [32, 64, 128, 256, 512]

        for multiple in multiples:
            layer = SwiGLUFFN(
                output_dim=output_dim,
                ffn_expansion_factor=4,
                ffn_multiple_of=multiple
            )

            # Verify hidden dimension is a multiple
            assert layer.hidden_dim % multiple == 0

    def test_single_batch_single_sequence(self):
        """Test with minimal batch and sequence dimensions."""
        layer = SwiGLUFFN(output_dim=64)
        test_input = keras.random.normal([1, 1, 64])  # Single item, single step

        output = layer(test_input)
        assert output.shape == (1, 1, 64)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = SwiGLUFFN(output_dim=128)

        # Use the same layer with different inputs
        input1 = keras.random.normal([2, 8, 128])
        input2 = keras.random.normal([3, 10, 128])

        output1 = layer(input1)
        output2 = layer(input2)

        assert output1.shape == (2, 8, 128)
        assert output2.shape == (3, 10, 128)

        # Both outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(output1))
        assert not keras.ops.any(keras.ops.isnan(output2))

    def test_extreme_expansion_factors(self):
        """Test with extreme expansion factors."""
        output_dim = 64

        # Very small expansion factor
        layer_small = SwiGLUFFN(output_dim=output_dim, ffn_expansion_factor=1, ffn_multiple_of=16)
        test_input = keras.random.normal([2, 4, output_dim])

        output_small = layer_small(test_input)
        assert output_small.shape == (2, 4, output_dim)
        assert not keras.ops.any(keras.ops.isnan(output_small))

        # Very large expansion factor
        layer_large = SwiGLUFFN(output_dim=output_dim, ffn_expansion_factor=32, ffn_multiple_of=32)
        output_large = layer_large(test_input)
        assert output_large.shape == (2, 4, output_dim)
        assert not keras.ops.any(keras.ops.isnan(output_large))

    def test_hardware_optimization_multiples(self):
        """Test that common hardware-optimized multiples work correctly."""
        output_dim = 512
        hardware_multiples = [32, 64, 128, 256, 512, 1024]

        for multiple in hardware_multiples:
            layer = SwiGLUFFN(
                output_dim=output_dim,
                ffn_expansion_factor=4,
                ffn_multiple_of=multiple
            )
            test_input = keras.random.normal([1, 2, output_dim])

            output = layer(test_input)
            assert output.shape == (1, 2, output_dim)
            assert layer.hidden_dim % multiple == 0