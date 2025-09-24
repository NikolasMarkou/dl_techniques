import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict
import tensorflow as tf

from dl_techniques.layers.ffn.geglu_ffn import GeGLUFFN


class TestGeGLUFFN:
    """Comprehensive test suite for GeGLUFFN layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_dim': 256,
            'output_dim': 128,
            'activation': 'gelu',
            'dropout_rate': 0.1,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros'
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input for testing."""
        return keras.random.normal(shape=(4, 64))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input for testing (sequence data)."""
        return keras.random.normal(shape=(2, 32, 64))

    @pytest.fixture
    def sample_input_4d(self) -> keras.KerasTensor:
        """Sample 4D input for testing (image-like data)."""
        return keras.random.normal(shape=(2, 8, 8, 64))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization and sub-layer creation."""
        layer = GeGLUFFN(**layer_config)

        # Check configuration storage
        assert layer.hidden_dim == layer_config['hidden_dim']
        assert layer.output_dim == layer_config['output_dim']
        assert layer.dropout_rate == layer_config['dropout_rate']
        assert layer.use_bias == layer_config['use_bias']

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers were created in __init__
        assert hasattr(layer, 'input_proj')
        assert hasattr(layer, 'output_proj')
        assert hasattr(layer, 'dropout')

        # Verify sub-layer types
        assert isinstance(layer.input_proj, keras.layers.Dense)
        assert isinstance(layer.output_proj, keras.layers.Dense)
        assert isinstance(layer.dropout, keras.layers.Dropout)

        # Check sub-layer configurations
        # GeGLU projects to hidden_dim * 2 for splitting
        assert layer.input_proj.units == layer_config['hidden_dim'] * 2
        assert layer.output_proj.units == layer_config['output_dim']
        assert layer.dropout.rate == layer_config['dropout_rate']

        # Verify activation is stored correctly
        assert layer.activation is not None

    @pytest.mark.parametrize("sample_input", [
        "sample_input_2d", "sample_input_3d", "sample_input_4d"
    ])
    def test_forward_pass(self, layer_config: Dict[str, Any],
                          sample_input: str, request) -> None:
        """Test forward pass and building for different input shapes."""
        inputs = request.getfixturevalue(sample_input)
        layer = GeGLUFFN(**layer_config)

        # Forward pass should trigger building
        output = layer(inputs)

        # Check that layer is now built
        assert layer.built

        # Check output shape
        expected_shape = list(inputs.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)

        # Check that sub-layers are built
        assert layer.input_proj.built
        assert layer.output_proj.built

        # Verify output is not NaN or infinite
        output_numpy = keras.ops.convert_to_numpy(output)
        assert not np.isnan(output_numpy).any()
        assert not np.isinf(output_numpy).any()

    def test_serialization_cycle(self, layer_config: Dict[str, Any],
                                 sample_input_3d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization cycle with prediction comparison."""
        # 1. Create model with custom layer
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = GeGLUFFN(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # 2. Get prediction from original model
        original_prediction = model(sample_input_3d)

        # 3. Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d)

            # 4. Verify identical outputs (CRITICAL)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = GeGLUFFN(**layer_config)
        config = layer.get_config()

        # Check all required parameters are present
        required_keys = {
            'hidden_dim', 'output_dim', 'activation', 'dropout_rate',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify values match (for non-serialized ones)
        assert config['hidden_dim'] == layer_config['hidden_dim']
        assert config['output_dim'] == layer_config['output_dim']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['use_bias'] == layer_config['use_bias']

    def test_gradients_flow(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor) -> None:
        """Test gradient computation through the layer."""
        layer = GeGLUFFN(**layer_config)

        # Test input gradients
        with tf.GradientTape() as tape:
            tape.watch(sample_input_3d)
            output = layer(sample_input_3d)
            loss = keras.ops.mean(keras.ops.square(output))

        # Check gradients with respect to input
        input_gradients = tape.gradient(loss, sample_input_3d)
        assert input_gradients is not None
        assert input_gradients.shape == sample_input_3d.shape

        # Test weight gradients with a separate tape
        with tf.GradientTape() as weight_tape:
            output = layer(sample_input_3d)
            loss = keras.ops.mean(keras.ops.square(output))

        # Check gradients with respect to layer weights
        weight_gradients = weight_tape.gradient(loss, layer.trainable_variables)

        # Verify all gradients exist and are non-zero
        assert len(weight_gradients) > 0
        for grad in weight_gradients:
            assert grad is not None
            grad_numpy = keras.ops.convert_to_numpy(grad)
            assert not np.allclose(grad_numpy, 0), "Gradient is zero - possible dead neurons"

    @pytest.mark.parametrize("training_mode", [True, False, None])
    def test_training_modes(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor,
                            training_mode: bool) -> None:
        """Test behavior in different training modes."""
        layer = GeGLUFFN(**layer_config)

        output = layer(sample_input_3d, training=training_mode)

        # Check output shape is consistent
        expected_shape = list(sample_input_3d.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)

        # For deterministic testing of dropout behavior
        if layer_config['dropout_rate'] > 0:
            # Multiple forward passes with dropout should differ in training
            if training_mode is True:
                output2 = layer(sample_input_3d, training=True)
                # With dropout, outputs should potentially differ
                # (though they might be identical by chance)
                assert output.shape == output2.shape
            elif training_mode is False:
                # In evaluation mode, outputs should be deterministic
                output2 = layer(sample_input_3d, training=False)
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(output),
                    keras.ops.convert_to_numpy(output2),
                    rtol=1e-6, atol=1e-6,
                    err_msg="Evaluation mode outputs should be deterministic"
                )

    def test_edge_cases(self) -> None:
        """Test error conditions and edge cases."""

        # Test invalid hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            GeGLUFFN(hidden_dim=0, output_dim=32)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            GeGLUFFN(hidden_dim=-1, output_dim=32)

        # Test invalid output_dim
        with pytest.raises(ValueError, match="output_dim must be positive"):
            GeGLUFFN(hidden_dim=32, output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            GeGLUFFN(hidden_dim=32, output_dim=-1)

        # Test invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            GeGLUFFN(hidden_dim=32, output_dim=16, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            GeGLUFFN(hidden_dim=32, output_dim=16, dropout_rate=1.1)

        # Test undefined input shape
        layer = GeGLUFFN(hidden_dim=32, output_dim=16)
        with pytest.raises(ValueError, match="Last dimension of input must be defined"):
            layer.build((None, None))

    def test_different_activations(self, sample_input_3d: keras.KerasTensor) -> None:
        """Test layer with different activation functions."""
        activations_to_test = ['gelu', 'swish', 'relu', 'tanh', 'sigmoid']

        for activation in activations_to_test:
            layer = GeGLUFFN(
                hidden_dim=64,
                output_dim=32,
                activation=activation
            )

            output = layer(sample_input_3d)

            # Check output shape
            assert output.shape == (2, 32, 32)

            # Verify output is finite
            output_numpy = keras.ops.convert_to_numpy(output)
            assert np.isfinite(output_numpy).all(), f"Non-finite output with {activation}"

    def test_no_bias_configuration(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test layer without bias terms."""
        layer = GeGLUFFN(
            hidden_dim=128,
            output_dim=64,
            use_bias=False
        )

        output = layer(sample_input_2d)
        assert output.shape == (4, 64)

        # Check that Dense layers have no bias
        assert layer.input_proj.use_bias is False
        assert layer.output_proj.use_bias is False

    def test_regularization(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test layer with kernel and bias regularization."""
        layer = GeGLUFFN(
            hidden_dim=64,
            output_dim=32,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.001)
        )

        output = layer(sample_input_2d)
        assert output.shape == (4, 32)

        # Check that regularization losses are added
        assert len(layer.losses) > 0

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape method."""
        layer = GeGLUFFN(**layer_config)

        # Test different input shapes
        test_shapes = [
            (None, 64),  # 2D
            (None, 32, 64),  # 3D
            (None, 8, 8, 64),  # 4D
            (None, 4, 4, 4, 64)  # 5D
        ]

        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)

            # Last dimension should be output_dim
            assert output_shape[-1] == layer_config['output_dim']

            # All other dimensions should be preserved
            assert output_shape[:-1] == input_shape[:-1]

    def test_multiple_instantiation(self, layer_config: Dict[str, Any],
                                    sample_input_2d: keras.KerasTensor) -> None:
        """Test that multiple layer instances work independently."""
        layer1 = GeGLUFFN(**layer_config, name='geglu1')
        layer2 = GeGLUFFN(**layer_config, name='geglu2')

        # Both layers should work independently
        output1 = layer1(sample_input_2d)
        output2 = layer2(sample_input_2d)

        assert output1.shape == output2.shape

        # Outputs should be different (different weight initialization)
        output1_numpy = keras.ops.convert_to_numpy(output1)
        output2_numpy = keras.ops.convert_to_numpy(output2)
        assert not np.allclose(output1_numpy, output2_numpy, rtol=1e-3)

    def test_gating_mechanism(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test that the GeGLU gating mechanism is working correctly."""
        # Create layer with specific parameters for easier interpretation
        layer = GeGLUFFN(
            hidden_dim=32,
            output_dim=16,
            activation='gelu',
            dropout_rate=0.0  # No dropout for cleaner testing
        )

        output = layer(sample_input_2d)

        # Access intermediate values by calling sub-layers directly
        gate_and_value = layer.input_proj(sample_input_2d)

        # Split into gate and value (GeGLU specific behavior)
        gate, value = keras.ops.split(gate_and_value, 2, axis=-1)

        # Apply activation to gate (GELU in this case)
        activated_gate = layer.activation(gate)

        # Manual gating: activated_gate * value
        manual_gated = activated_gate * value
        manual_output = layer.output_proj(manual_gated)

        # Compare with layer output
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(manual_output),
            rtol=1e-5, atol=1e-5,
            err_msg="GeGLU gating mechanism not working correctly"
        )

    def test_split_consistency(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test that the split operation works correctly."""
        layer = GeGLUFFN(hidden_dim=64, output_dim=32, dropout_rate=0.0)

        # Get projected values
        gate_and_value = layer.input_proj(sample_input_2d)

        # Check that projection has correct size (2 * hidden_dim)
        assert gate_and_value.shape[-1] == 128  # 64 * 2

        # Split and check dimensions
        gate, value = keras.ops.split(gate_and_value, 2, axis=-1)

        assert gate.shape[-1] == 64  # hidden_dim
        assert value.shape[-1] == 64  # hidden_dim
        assert gate.shape[:-1] == sample_input_2d.shape[:-1]  # Batch dimensions preserved
        assert value.shape[:-1] == sample_input_2d.shape[:-1]  # Batch dimensions preserved

    def test_with_transformer_architecture(self, sample_input_3d: keras.KerasTensor) -> None:
        """Test integration in a mini-transformer architecture."""
        embed_dim = sample_input_3d.shape[-1]  # 64

        # Create a mini transformer block with GeGLU FFN
        inputs = keras.Input(shape=(32, embed_dim))

        # Simple self-attention (using built-in MultiHeadAttention)
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=embed_dim // 8
        )(inputs, inputs)

        # Add & Norm
        x = keras.layers.Add()([inputs, attention_output])
        x = keras.layers.LayerNormalization()(x)

        # GeGLU FFN
        ffn_output = GeGLUFFN(
            hidden_dim=256,  # 4x expansion
            output_dim=embed_dim,
            activation='gelu',
            dropout_rate=0.1
        )(x)

        # Add & Norm
        x = keras.layers.Add()([x, ffn_output])
        outputs = keras.layers.LayerNormalization()(x)

        # Create model
        model = keras.Model(inputs, outputs)

        # Test forward pass
        output = model(sample_input_3d)
        assert output.shape == sample_input_3d.shape

        # Test that model can be compiled and used
        model.compile(optimizer='adam', loss='mse')

        # Create dummy target and test a training step
        dummy_target = keras.random.normal(sample_input_3d.shape)
        loss = model.train_on_batch(sample_input_3d, dummy_target)

        # Extract scalar value if it's a numpy array
        if isinstance(loss, np.ndarray):
            loss = float(loss.item())

        assert isinstance(loss, (int, float))
        assert not np.isnan(loss)

    def test_custom_initializers(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test layer with custom initializers."""
        layer = GeGLUFFN(
            hidden_dim=64,
            output_dim=32,
            kernel_initializer='he_normal',
            bias_initializer='ones'
        )

        output = layer(sample_input_2d)
        assert output.shape == (4, 32)

        # Check that custom initializers were applied
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)

    def test_zero_dropout_deterministic(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test that zero dropout gives deterministic results."""
        layer = GeGLUFFN(
            hidden_dim=64,
            output_dim=32,
            dropout_rate=0.0
        )

        # Multiple forward passes should give identical results
        output1 = layer(sample_input_2d, training=True)
        output2 = layer(sample_input_2d, training=True)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Zero dropout should give deterministic results"
        )

    def test_activation_serialization(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test that custom activations are properly serialized."""

        # Test with callable activation
        def custom_activation(x):
            return keras.ops.tanh(x) * 0.5

        layer = GeGLUFFN(
            hidden_dim=32,
            output_dim=16,
            activation=custom_activation
        )

        # This should work without throwing errors
        output = layer(sample_input_2d)
        assert output.shape == (4, 16)

        # Get config should handle the custom activation
        config = layer.get_config()
        assert 'activation' in config


# Additional integration tests
class TestGeGLUFFNIntegration:
    """Integration tests for GeGLUFFN in realistic scenarios."""

    def test_classification_model(self) -> None:
        """Test GeGLU FFN in a classification model."""
        # Create a simple classification model with GeGLU FFN
        inputs = keras.Input(shape=(784,))  # MNIST-like input

        x = GeGLUFFN(
            hidden_dim=512,
            output_dim=256,
            activation='gelu',
            dropout_rate=0.3
        )(inputs)

        x = GeGLUFFN(
            hidden_dim=128,
            output_dim=64,
            activation='swish',
            dropout_rate=0.2
        )(x)

        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test with dummy data
        x_dummy = np.random.randn(32, 784).astype(np.float32)
        y_dummy = np.random.randint(0, 10, 32)

        # Test training
        history = model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
        assert 'loss' in history.history

        # Extract scalar loss value properly
        loss_value = history.history['loss'][0]
        if isinstance(loss_value, np.ndarray):
            loss_value = float(loss_value.item())

        assert not np.isnan(loss_value)

        # Test prediction
        predictions = model.predict(x_dummy[:5], verbose=0)
        assert predictions.shape == (5, 10)
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Softmax sums to 1

    def test_sequence_model(self) -> None:
        """Test GeGLU FFN in a sequence processing model."""
        seq_len, embed_dim = 50, 128

        inputs = keras.Input(shape=(seq_len, embed_dim))

        # LSTM followed by GeGLU FFN
        x = keras.layers.LSTM(64, return_sequences=True)(inputs)

        x = GeGLUFFN(
            hidden_dim=256,
            output_dim=64,
            activation='gelu',
            dropout_rate=0.1
        )(x)

        # Global pooling and classification
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test with dummy sequence data
        x_dummy = np.random.randn(16, seq_len, embed_dim).astype(np.float32)
        y_dummy = np.random.randint(0, 5, 16)

        # Test training
        loss = model.train_on_batch(x_dummy, y_dummy)

        # Extract scalar value if it's a numpy array
        if isinstance(loss, np.ndarray):
            loss = float(loss.item())

        assert not np.isnan(loss)

        # Test prediction
        predictions = model.predict(x_dummy[:3], verbose=0)
        assert predictions.shape == (3, 5)

    def test_language_model_ffn(self) -> None:
        """Test GeGLU FFN in a transformer language model setup."""
        vocab_size, embed_dim, seq_len = 1000, 256, 32

        # Simple transformer-like model
        inputs = keras.Input(shape=(seq_len,))

        # Token embedding
        x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)

        # Positional embedding (simple learned) - fix shape broadcasting
        positions = keras.ops.arange(seq_len)
        positions = keras.ops.expand_dims(positions, axis=0)  # (1, seq_len)
        pos_embed = keras.layers.Embedding(seq_len, embed_dim)(positions)
        pos_embed = keras.ops.squeeze(pos_embed, axis=0)  # (seq_len, embed_dim)
        pos_embed = keras.ops.expand_dims(pos_embed, axis=0)  # (1, seq_len, embed_dim)

        x = keras.layers.Add()([x, pos_embed])

        # Self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=embed_dim // 8
        )(x, x)
        x = keras.layers.Add()([x, attn_output])
        x = keras.layers.LayerNormalization()(x)

        # GeGLU FFN block
        ffn_output = GeGLUFFN(
            hidden_dim=embed_dim * 4,  # Typical 4x expansion
            output_dim=embed_dim,
            activation='gelu',
            dropout_rate=0.1
        )(x)
        x = keras.layers.Add()([x, ffn_output])
        x = keras.layers.LayerNormalization()(x)

        # Output head
        outputs = keras.layers.Dense(vocab_size, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test with dummy data
        x_dummy = np.random.randint(0, vocab_size, (8, seq_len))
        y_dummy = np.random.randint(0, vocab_size, (8, seq_len, 1))

        # Test training
        loss = model.train_on_batch(x_dummy, y_dummy)

        # Extract scalar value if it's a numpy array
        if isinstance(loss, np.ndarray):
            loss = float(loss.item())

        assert not np.isnan(loss)

        # Test prediction
        predictions = model.predict(x_dummy[:2], verbose=0)
        assert predictions.shape == (2, seq_len, vocab_size)

    def test_vision_transformer_ffn(self) -> None:
        """Test GeGLU FFN in a vision_heads transformer setup."""
        img_size, patch_size, embed_dim = 32, 4, 192
        num_patches = (img_size // patch_size) ** 2  # 8*8 = 64

        # Simple ViT-like model
        inputs = keras.Input(shape=(img_size, img_size, 3))

        # Patch embedding: Conv2D to extract patches
        # Conv2D with kernel_size=patch_size, strides=patch_size
        patches = keras.layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid'
        )(inputs)  # Shape: (batch, 8, 8, 192)

        # Reshape to sequence of patches
        patches = keras.layers.Reshape((num_patches, embed_dim))(patches)  # (batch, 64, 192)

        # Class token - use a simpler approach with Embedding layer
        # Create a single dummy input for class token embedding
        class_token_input = keras.layers.Lambda(lambda x: keras.ops.zeros((keras.ops.shape(x)[0], 1), dtype='int32'))(
            patches)
        class_token = keras.layers.Embedding(1, embed_dim)(class_token_input)  # Shape: (batch, 1, 192)

        x = keras.layers.Concatenate(axis=1)([class_token, patches])  # (batch, 65, 192)

        # Self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=embed_dim // 8
        )(x, x)
        x = keras.layers.Add()([x, attn_output])
        x = keras.layers.LayerNormalization()(x)

        # GeGLU FFN
        ffn_output = GeGLUFFN(
            hidden_dim=embed_dim * 4,
            output_dim=embed_dim,
            activation='gelu',
            dropout_rate=0.1
        )(x)
        x = keras.layers.Add()([x, ffn_output])
        x = keras.layers.LayerNormalization()(x)

        # Classification head (use class token)
        cls_token = keras.layers.Lambda(lambda x: x[:, 0])(x)  # Extract class token
        outputs = keras.layers.Dense(10, activation='softmax')(cls_token)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test with dummy image data
        x_dummy = np.random.randn(4, img_size, img_size, 3).astype(np.float32)
        y_dummy = np.random.randint(0, 10, 4)

        # Test training
        loss = model.train_on_batch(x_dummy, y_dummy)

        # Extract scalar value if it's a numpy array
        if isinstance(loss, np.ndarray):
            loss = float(loss.item())

        assert not np.isnan(loss)

        # Test prediction
        predictions = model.predict(x_dummy[:2], verbose=0)
        assert predictions.shape == (2, 10)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])