"""
Comprehensive test suite for LogicFFN layer.

This module provides thorough testing of the LogicFFN layer including:
- Initialization and parameter validation
- Build process and layer architecture
- Forward pass and output shapes
- Serialization and model persistence
- Training dynamics and gradient flow
- Integration with other components
- Edge cases and numerical stability
"""

import pytest
import numpy as np
import keras
from keras import ops, random
import tempfile
import os
from typing import Tuple, Dict, Any
# FIX: Import tensorflow for GradientTape, as this test suite runs in a TF environment
import tensorflow as tf

# Import the layer to test
from dl_techniques.layers.ffn.logic_ffn import LogicFFN


class TestLogicFFNInitialization:
    """Test suite for LogicFFN initialization and parameter validation."""

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = LogicFFN(output_dim=768, logic_dim=256)

        # Check default values
        assert layer.output_dim == 768
        assert layer.logic_dim == 256
        assert layer.use_bias is True
        assert layer.temperature == 1.0
        assert layer.num_logic_ops == 3
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

        # Check that sublayers are None before build
        assert layer.logic_projection is None
        assert layer.gate_projection is None
        assert layer.output_projection is None

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        layer = LogicFFN(
            output_dim=512,
            logic_dim=128,
            use_bias=False,
            kernel_initializer='he_normal',
            bias_initializer='ones',
            kernel_regularizer='l2',
            bias_regularizer='l1',
            temperature=1.5
        )

        # Check custom values
        assert layer.output_dim == 512
        assert layer.logic_dim == 128
        assert layer.use_bias is False
        assert layer.temperature == 1.5
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative output_dim
        with pytest.raises(ValueError, match="output_dim must be positive"):
            LogicFFN(output_dim=-10, logic_dim=256)

        # Test zero logic_dim
        with pytest.raises(ValueError, match="logic_dim must be positive"):
            LogicFFN(output_dim=768, logic_dim=0)

        # Test negative temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            LogicFFN(output_dim=768, logic_dim=256, temperature=-0.5)


class TestLogicFFNBuild:
    """Test suite for LogicFFN build process."""

    @pytest.fixture
    def sample_input_shape(self):
        """Sample input shape for testing."""
        return (None, 128, 768)

    def test_build_process(self, sample_input_shape):
        """Test that the layer builds properly."""
        layer = LogicFFN(output_dim=512, logic_dim=256)
        layer.build(sample_input_shape)

        # Check that layer is marked as built
        assert layer.built is True
        assert layer._build_input_shape == sample_input_shape

        # Check that sublayers were created
        assert layer.logic_projection is not None
        assert layer.gate_projection is not None
        assert layer.output_projection is not None

        # Check sublayer configurations
        assert layer.logic_projection.units == 256 * 2  # Two operands
        assert layer.gate_projection.units == 3  # Three logic operations
        assert layer.output_projection.units == 512  # Output dimension

        # Check that sublayers are built
        assert layer.logic_projection.built is True
        assert layer.gate_projection.built is True
        assert layer.output_projection.built is True

    def test_build_invalid_input_shape(self):
        """Test build with invalid input shapes."""
        layer = LogicFFN(output_dim=768, logic_dim=256)

        # Test 1D input (too few dimensions)
        with pytest.raises(ValueError, match="Input must be at least 2D"):
            layer.build((768,))

        # Test input with None feature dimension
        with pytest.raises(ValueError, match="Input feature dimension must be specified"):
            layer.build((None, 128, None))

    def test_build_idempotent(self, sample_input_shape):
        """Test that multiple build calls are idempotent."""
        layer = LogicFFN(output_dim=768, logic_dim=256)

        # Build once
        layer.build(sample_input_shape)
        first_logic_projection = layer.logic_projection

        # Build again
        layer.build(sample_input_shape)

        # Should be the same objects
        assert layer.logic_projection is first_logic_projection
        assert layer.built is True


class TestLogicFFNForwardPass:
    """Test suite for LogicFFN forward pass and output shapes."""



    @pytest.fixture
    def built_layer(self):
        """Create a built layer for testing."""
        layer = LogicFFN(output_dim=512, logic_dim=256, temperature=1.2)
        layer.build((None, 64, 768))
        return layer

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return keras.random.normal((2, 64, 768))

    def test_output_shapes(self, built_layer, sample_input):
        """Test that output shapes are computed correctly."""
        # Test forward pass
        output = built_layer(sample_input)

        # Check output shape
        expected_shape = (2, 64, 512)  # (batch, seq, output_dim)
        assert output.shape == expected_shape

        # Test compute_output_shape method
        computed_shape = built_layer.compute_output_shape(sample_input.shape)
        assert computed_shape == expected_shape

    def test_forward_pass_values(self, built_layer, sample_input):
        """Test forward pass produces valid values."""
        output = built_layer(sample_input)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # FIX: The following check is removed because np.allclose fails on
        # tensors with different shapes, which is expected for this layer.
        # The fact that the output shape is different already proves the layer
        # is doing something.
        # assert not np.allclose(output.numpy(), sample_input.numpy())

    def test_training_mode(self, built_layer, sample_input):
        """Test forward pass in training vs inference mode."""
        # Training mode
        output_train = built_layer(sample_input, training=True)

        # Inference mode
        output_infer = built_layer(sample_input, training=False)

        # Outputs should be the same (no dropout in this layer)
        assert np.allclose(output_train.numpy(), output_infer.numpy())

    def test_different_input_sizes(self):
        """Test layer with different input sizes."""
        layer = LogicFFN(output_dim=256, logic_dim=128)

        # Test different sequence lengths and batch sizes
        test_shapes = [
            (1, 32, 512),    # Small batch, short sequence
            (4, 128, 768),   # Medium batch, medium sequence
            (8, 256, 1024),  # Large batch, long sequence
        ]

        for batch_size, seq_len, input_dim in test_shapes:
            layer = LogicFFN(output_dim=256, logic_dim=128)
            sample_input = keras.random.normal((batch_size, seq_len, input_dim))

            output = layer(sample_input)
            expected_shape = (batch_size, seq_len, 256)
            assert output.shape == expected_shape


class TestLogicFFNLogicOperations:
    """Test suite for the logic operations within LogicFFN."""

    def test_logic_operations_properties(self):
        """Test that logic operations have expected mathematical properties."""
        layer = LogicFFN(output_dim=512, logic_dim=256)
        layer.build((None, 64, 768))

        # Create controlled inputs to test logic operations
        # Use different magnitudes to test soft logic behavior
        test_input_high = ops.ones((1, 1, 768)) * 5.0   # Should map to high values after sigmoid
        test_input_low = ops.ones((1, 1, 768)) * -5.0   # Should map to low values after sigmoid
        test_input_medium = keras.random.normal((1, 1, 768)) # Random medium values

        # Test with different input patterns
        outputs_high = layer(test_input_high)
        outputs_low = layer(test_input_low)
        outputs_medium = layer(test_input_medium)

        # All should produce valid outputs
        assert not np.any(np.isnan(outputs_high.numpy()))
        assert not np.any(np.isnan(outputs_low.numpy()))
        assert not np.any(np.isnan(outputs_medium.numpy()))

        # Outputs should be different for different inputs
        assert not np.allclose(outputs_high.numpy(), outputs_low.numpy())

    def test_temperature_effect(self):
        """Test that temperature parameter affects gating behavior."""
        input_tensor = keras.random.normal((2, 32, 768))

        # Create layers with different temperatures
        layer_low_temp = LogicFFN(output_dim=512, logic_dim=256, temperature=0.1)
        layer_high_temp = LogicFFN(output_dim=512, logic_dim=256, temperature=10.0)

        output_low_temp = layer_low_temp(input_tensor)
        output_high_temp = layer_high_temp(input_tensor)

        # Outputs should be different due to different gating
        assert not np.allclose(output_low_temp.numpy(), output_high_temp.numpy(), rtol=1e-3)


class TestLogicFFNSerialization:
    """Test suite for LogicFFN serialization and deserialization."""

    def test_get_config(self):
        """Test configuration serialization."""
        original_layer = LogicFFN(
            output_dim=768,
            logic_dim=256,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer='l2',
            temperature=1.5
        )

        config = original_layer.get_config()

        # Check that all parameters are serialized
        assert config['output_dim'] == 768
        assert config['logic_dim'] == 256
        assert config['use_bias'] is False
        assert config['temperature'] == 1.5

        # Check serialized initializers and regularizers
        assert config['kernel_initializer']['class_name'] == 'HeNormal'
        assert config['kernel_regularizer']['class_name'] == 'L2'

    def test_from_config(self):
        """Test layer recreation from configuration."""
        original_layer = LogicFFN(
            output_dim=512,
            logic_dim=128,
            temperature=2.0
        )
        original_layer.build((None, 64, 768))

        # Get configuration
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate layer
        recreated_layer = LogicFFN.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check that configuration matches
        assert recreated_layer.output_dim == original_layer.output_dim
        assert recreated_layer.logic_dim == original_layer.logic_dim
        assert recreated_layer.temperature == original_layer.temperature
        assert recreated_layer.built == original_layer.built

    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip."""
        # Create and build original layer
        original_layer = LogicFFN(output_dim=256, logic_dim=128)
        sample_input = keras.random.normal((2, 32, 512))
        original_output = original_layer(sample_input)

        # Serialize and deserialize
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        recreated_layer = LogicFFN.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Copy weights to ensure identical computation
        for orig_weight, new_weight in zip(original_layer.weights, recreated_layer.weights):
            new_weight.assign(orig_weight)

        # Test that outputs match
        recreated_output = recreated_layer(sample_input)
        assert np.allclose(original_output.numpy(), recreated_output.numpy(), atol=1e-6)


class TestLogicFFNModelIntegration:
    """Test suite for LogicFFN integration in models."""

    def test_simple_model_integration(self):
        """Test LogicFFN in a simple sequential model."""
        inputs = keras.Input(shape=(64, 512))
        x = LogicFFN(output_dim=256, logic_dim=128)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = LogicFFN(output_dim=128, logic_dim=64)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test forward pass
        sample_input = keras.random.normal((4, 64, 512))
        prediction = model(sample_input)

        assert prediction.shape == (4, 10)
        assert not np.any(np.isnan(prediction.numpy()))

    def test_model_save_load(self):
        """Test saving and loading models with LogicFFN."""
        # Create model with LogicFFN
        inputs = keras.Input(shape=(32, 256))
        x = LogicFFN(output_dim=128, logic_dim=64, name='logic_ffn')(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Generate prediction before saving
        test_input = keras.random.normal((2, 32, 256))
        original_prediction = model.predict(test_input, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'model_with_logic_ffn.keras')
            model.save(model_path)

            # Load model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'LogicFFN': LogicFFN}
            )

            # Test prediction with loaded model
            loaded_prediction = loaded_model.predict(test_input, verbose=0)

            # Predictions should match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check that custom layer is preserved
            logic_layer = loaded_model.get_layer('logic_ffn')
            assert isinstance(logic_layer, LogicFFN)
            assert logic_layer.output_dim == 128
            assert logic_layer.logic_dim == 64


class TestLogicFFNTrainingDynamics:
    """Test suite for LogicFFN training and gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""
        layer = LogicFFN(output_dim=256, logic_dim=128)
        inputs = keras.random.normal((2, 32, 512))

        # Test gradient computation
        # FIX: Use a persistent tape and watch the input tensor directly,
        # not a keras.Variable wrapper.
        with tf.GradientTape(persistent=True) as tape:
            # We must explicitly watch non-Variable tensors to compute gradients.
            tape.watch(inputs)
            outputs = layer(inputs)
            loss = ops.mean(ops.square(outputs))

        # Compute gradients for both inputs and trainable weights
        input_grads = tape.gradient(loss, inputs)
        layer_grads = tape.gradient(loss, layer.trainable_weights)
        # Release tape resources
        del tape

        # Check that gradients exist and are not None
        assert input_grads is not None
        assert all(g is not None for g in layer_grads)

        # Check that gradients have non-zero values
        assert not np.allclose(input_grads.numpy(), 0)
        assert all(not np.allclose(g.numpy(), 0) for g in layer_grads)

    def test_training_step(self):
        """Test a complete training step with LogicFFN."""
        # Create simple model
        inputs = keras.Input(shape=(16, 128))
        x = LogicFFN(output_dim=64, logic_dim=32)(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(3, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='categorical_crossentropy'
        )

        # Create training data
        x_train = keras.random.normal((16, 16, 128))
        # FIX: use keras.random.randint for integer labels.
        # keras.random.uniform is for floats.
        y_train = keras.utils.to_categorical(
            keras.random.randint((16,), 0, 3, dtype='int32'), num_classes=3
        )

        # Get initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)

        # Train for a few steps
        model.fit(x_train, y_train, epochs=5, batch_size=8, verbose=0)

        # Get final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)

        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss + 0.1


class TestLogicFFNEdgeCases:
    """Test suite for LogicFFN edge cases and robustness."""

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = LogicFFN(output_dim=128, logic_dim=64)

        # Test with different input magnitudes
        test_cases = [
            ops.zeros((2, 16, 256)),  # All zeros
            ops.ones((2, 16, 256)) * 1e-10,  # Very small values
            ops.ones((2, 16, 256)) * 1e10,   # Very large values
            keras.random.normal((2, 16, 256)) * 1e5,  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected"

    def test_different_batch_sizes(self):
        """Test layer with various batch sizes."""
        layer = LogicFFN(output_dim=256, logic_dim=128)

        batch_sizes = [1, 2, 8, 16, 32]
        seq_len, input_dim = 64, 512

        for batch_size in batch_sizes:
            test_input = keras.random.normal((batch_size, seq_len, input_dim))
            output = layer(test_input)

            expected_shape = (batch_size, seq_len, 256)
            assert output.shape == expected_shape
            assert not np.any(np.isnan(output.numpy()))

    def test_temperature_extremes(self):
        """Test layer behavior with extreme temperature values."""
        # FIX: Use keras.random.normal, not ops.random.normal
        input_tensor = keras.random.normal((2, 16, 128))

        # Test very low temperature (sharp gating)
        layer_sharp = LogicFFN(output_dim=64, logic_dim=32, temperature=1e-3)
        output_sharp = layer_sharp(input_tensor)
        assert not np.any(np.isnan(output_sharp.numpy()))

        # Test very high temperature (uniform gating)
        layer_smooth = LogicFFN(output_dim=64, logic_dim=32, temperature=1e3)
        output_smooth = layer_smooth(input_tensor)
        assert not np.any(np.isnan(output_smooth.numpy()))


# Integration tests with pytest fixtures
@pytest.fixture
def logic_ffn_layer():
    """Fixture providing a LogicFFN layer for testing."""
    return LogicFFN(output_dim=512, logic_dim=256, temperature=1.0)


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    return {
        'inputs': keras.random.normal((4, 32, 768)),
        'targets': keras.random.uniform((4, 10), 0, 1)
    }


def test_layer_with_transformer_integration(logic_ffn_layer):
    """Test integration with transformer-like architecture."""
    # This would test integration with the transformer layer
    # if the transformer layer supported 'logic' as an ffn_type

    # Create a transformer-like block manually
    inputs = keras.Input(shape=(64, 768))

    # Attention block (simplified)
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=12, key_dim=64
    )(inputs, inputs)
    attention_output = keras.layers.LayerNormalization()(attention_output + inputs)

    # Logic FFN block - create one that outputs 768 to match input
    logic_ffn_matching = LogicFFN(output_dim=768, logic_dim=256)
    ffn_output = logic_ffn_matching(attention_output)
    output = keras.layers.LayerNormalization()(ffn_output + attention_output)

    model = keras.Model(inputs, output)

    # Test forward pass
    test_input = keras.random.normal((2, 64, 768))
    result = model(test_input)

    assert result.shape == (2, 64, 768)  # Should match input shape
    assert not np.any(np.isnan(result.numpy()))


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])