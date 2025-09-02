"""
Comprehensive test suite for RepMixer and ConvolutionalStem layers.

This test suite follows modern Keras 3 best practices and tests all critical
functionality including serialization, configuration, and edge cases.
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Any, Dict, Tuple

import keras
import tensorflow as tf

from dl_techniques.layers.repmixer_block import RepMixerBlock, ConvolutionalStem


class TestRepMixerBlock:
    """Comprehensive test suite for RepMixerBlock layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for RepMixerBlock testing."""
        return {
            'dim': 64,
            'kernel_size': 3,
            'expansion_ratio': 4.0,
            'dropout_rate': 0.1,
            'activation': 'gelu',
            'use_layer_norm': True
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample 4D input tensor for testing."""
        return keras.random.normal(shape=(4, 32, 32, 64))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test RepMixerBlock initialization."""
        layer = RepMixerBlock(**layer_config)

        # Check configuration storage
        assert layer.dim == layer_config['dim']
        assert layer.kernel_size == layer_config['kernel_size']
        assert layer.expansion_ratio == layer_config['expansion_ratio']
        assert layer.dropout_rate == layer_config['dropout_rate']
        assert layer.use_layer_norm == layer_config['use_layer_norm']

        # Check sub-layer creation
        assert layer.norm1 is not None
        assert layer.norm2 is not None
        assert layer.token_mixer is not None
        assert layer.channel_mixer is not None

        # Layer should not be built initially
        assert not layer.built

    def test_forward_pass(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor) -> None:
        """Test RepMixerBlock forward pass and building."""
        layer = RepMixerBlock(**layer_config)

        # Forward pass should build the layer
        output = layer(sample_input)

        # Check layer is now built
        assert layer.built

        # Check output shape preservation
        expected_shape = sample_input.shape
        assert output.shape == expected_shape

        # Check output is different from input (layer should transform)
        assert not np.allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(sample_input),
            rtol=1e-6, atol=1e-6
        )

    def test_serialization_cycle(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization and loading cycle."""
        # Create model with RepMixerBlock
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = RepMixerBlock(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_repmixer_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = RepMixerBlock(**layer_config)
        config = layer.get_config()

        # Check all required config parameters are present
        required_keys = {
            'dim', 'kernel_size', 'expansion_ratio', 'dropout_rate',
            'activation', 'use_layer_norm', 'kernel_initializer',
            'bias_initializer', 'kernel_regularizer', 'bias_regularizer'
        }

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check specific values match
        assert config['dim'] == layer_config['dim']
        assert config['kernel_size'] == layer_config['kernel_size']
        assert config['expansion_ratio'] == layer_config['expansion_ratio']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['use_layer_norm'] == layer_config['use_layer_norm']

    def test_gradients_flow(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor) -> None:
        """Test gradient computation through the layer."""
        layer = RepMixerBlock(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Check gradients exist and are non-None
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0, "Layer should have trainable variables"
        assert all(g is not None for g in gradients), "All gradients should be non-None"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor, training: bool) -> None:
        """Test behavior in different training modes."""
        layer = RepMixerBlock(**layer_config)

        output = layer(sample_input, training=training)

        # Output shape should be consistent regardless of training mode
        assert output.shape == sample_input.shape

    @pytest.mark.parametrize("use_layer_norm", [True, False])
    def test_normalization_types(self, sample_input: keras.KerasTensor, use_layer_norm: bool) -> None:
        """Test both LayerNorm and BatchNorm configurations."""
        layer = RepMixerBlock(dim=64, use_layer_norm=use_layer_norm)
        output = layer(sample_input)

        assert output.shape == sample_input.shape

        if use_layer_norm:
            assert isinstance(layer.norm1, keras.layers.LayerNormalization)
            assert isinstance(layer.norm2, keras.layers.LayerNormalization)
        else:
            assert isinstance(layer.norm1, keras.layers.BatchNormalization)
            assert isinstance(layer.norm2, keras.layers.BatchNormalization)

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """Test output shape computation."""
        layer = RepMixerBlock(**layer_config)
        input_shape = (None, 32, 32, 64)

        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == input_shape, "Output shape should equal input shape"

    def test_edge_cases_and_validation(self) -> None:
        """Test error conditions and input validation."""
        # Test invalid dim
        with pytest.raises(ValueError, match="dim must be positive"):
            RepMixerBlock(dim=0)

        with pytest.raises(ValueError, match="dim must be positive"):
            RepMixerBlock(dim=-5)

        # Test invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be positive and odd"):
            RepMixerBlock(dim=64, kernel_size=0)

        with pytest.raises(ValueError, match="kernel_size must be positive and odd"):
            RepMixerBlock(dim=64, kernel_size=2)  # Even kernel size

        # Test invalid expansion_ratio
        with pytest.raises(ValueError, match="expansion_ratio must be positive"):
            RepMixerBlock(dim=64, expansion_ratio=0)

        with pytest.raises(ValueError, match="expansion_ratio must be positive"):
            RepMixerBlock(dim=64, expansion_ratio=-1.5)

        # Test invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            RepMixerBlock(dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            RepMixerBlock(dim=64, dropout_rate=1.5)

    def test_different_input_sizes(self, layer_config: Dict[str, Any]) -> None:
        """Test with different input spatial dimensions."""
        layer = RepMixerBlock(**layer_config)

        # Test different spatial sizes
        test_shapes = [
            (2, 16, 16, 64),
            (1, 64, 64, 64),
            (4, 8, 8, 64)
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape=shape)
            output = layer(test_input)
            assert output.shape == test_input.shape

    def test_wrong_input_dimensions(self, layer_config: Dict[str, Any]) -> None:
        """Test error handling for wrong input dimensions."""
        layer = RepMixerBlock(**layer_config)

        # Wrong number of dimensions
        wrong_input = keras.random.normal(shape=(4, 64))  # 2D instead of 4D

        with pytest.raises(ValueError, match="Expected 4D input"):
            layer.build(wrong_input.shape)

    def test_channel_mismatch(self) -> None:
        """Test error handling for channel dimension mismatch."""
        layer = RepMixerBlock(dim=64)
        wrong_input = keras.random.normal(shape=(4, 32, 32, 128))  # Wrong channels

        with pytest.raises(ValueError, match="Input channels.*must match dim"):
            layer.build(wrong_input.shape)


class TestConvolutionalStem:
    """Comprehensive test suite for ConvolutionalStem layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for ConvolutionalStem testing."""
        return {
            'out_channels': 64,
            'use_se': False,
            'activation': 'gelu',
            'kernel_initializer': 'he_normal'
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample RGB image input for testing."""
        return keras.random.normal(shape=(4, 224, 224, 3))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test ConvolutionalStem initialization."""
        layer = ConvolutionalStem(**layer_config)

        # Check configuration storage
        assert layer.out_channels == layer_config['out_channels']
        assert layer.use_se == layer_config['use_se']
        assert layer.activation == layer_config['activation']

        # Check blocks creation
        assert len(layer.blocks) == 3
        assert all(block is not None for block in layer.blocks)

        # Layer should not be built initially
        assert not layer.built

    def test_forward_pass(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor) -> None:
        """Test ConvolutionalStem forward pass and building."""
        layer = ConvolutionalStem(**layer_config)

        # Forward pass should build the layer
        output = layer(sample_input)

        # Check layer is now built
        assert layer.built

        # Check expected spatial downsampling (4x reduction)
        expected_height = sample_input.shape[1] // 4
        expected_width = sample_input.shape[2] // 4
        expected_channels = layer_config['out_channels']

        assert output.shape[0] == sample_input.shape[0]  # Batch size preserved
        assert output.shape[1] == expected_height
        assert output.shape[2] == expected_width
        assert output.shape[3] == expected_channels

    def test_serialization_cycle(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization and loading cycle."""
        # Create model with ConvolutionalStem
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = ConvolutionalStem(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_convstem_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = ConvolutionalStem(**layer_config)
        config = layer.get_config()

        # Check all required config parameters are present
        required_keys = {
            'out_channels', 'use_se', 'activation',
            'kernel_initializer', 'kernel_regularizer'
        }

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check specific values match
        assert config['out_channels'] == layer_config['out_channels']
        assert config['use_se'] == layer_config['use_se']
        assert config['activation'] == layer_config['activation']

    def test_gradients_flow(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor) -> None:
        """Test gradient computation through the layer."""
        layer = ConvolutionalStem(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Check gradients exist and are non-None
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0, "Layer should have trainable variables"
        assert all(g is not None for g in gradients), "All gradients should be non-None"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor, training: bool) -> None:
        """Test behavior in different training modes."""
        layer = ConvolutionalStem(**layer_config)

        output = layer(sample_input, training=training)

        # Check consistent output shape regardless of training mode
        expected_shape = (
            sample_input.shape[0],
            sample_input.shape[1] // 4,
            sample_input.shape[2] // 4,
            layer_config['out_channels']
        )
        assert output.shape == expected_shape

    @pytest.mark.parametrize("use_se", [True, False])
    def test_se_configuration(self, sample_input: keras.KerasTensor, use_se: bool) -> None:
        """Test with and without Squeeze-and-Excitation."""
        layer = ConvolutionalStem(out_channels=64, use_se=use_se)
        output = layer(sample_input)

        # Should work regardless of SE configuration
        expected_shape = (sample_input.shape[0], 56, 56, 64)  # 224/4 = 56
        assert output.shape == expected_shape

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """Test output shape computation."""
        layer = ConvolutionalStem(**layer_config)
        input_shape = (None, 224, 224, 3)

        output_shape = layer.compute_output_shape(input_shape)
        expected_shape = (None, 56, 56, layer_config['out_channels'])
        assert output_shape == expected_shape

    def test_edge_cases_and_validation(self) -> None:
        """Test error conditions and input validation."""
        # Test invalid out_channels
        with pytest.raises(ValueError, match="out_channels must be positive"):
            ConvolutionalStem(out_channels=0)

        with pytest.raises(ValueError, match="out_channels must be positive"):
            ConvolutionalStem(out_channels=-32)

    def test_different_input_sizes(self, layer_config: Dict[str, Any]) -> None:
        """Test with different input spatial dimensions."""
        layer = ConvolutionalStem(**layer_config)

        # Test different spatial sizes (all should work with 4x downsampling)
        test_cases = [
            ((2, 112, 112, 3), (2, 28, 28, 64)),   # 112/4 = 28
            ((1, 256, 256, 3), (1, 64, 64, 64)),   # 256/4 = 64
            ((4, 64, 64, 3), (4, 16, 16, 64))      # 64/4 = 16
        ]

        for input_shape, expected_output_shape in test_cases:
            test_input = keras.random.normal(shape=input_shape)
            output = layer(test_input)
            assert output.shape == expected_output_shape


class TestLayerIntegration:
    """Test integration patterns and combined usage."""

    def test_repmixer_stack(self) -> None:
        """Test stacking multiple RepMixer blocks."""
        inputs = keras.Input(shape=(56, 56, 128))
        x = inputs

        # Stack multiple RepMixer blocks
        for i in range(3):
            x = RepMixerBlock(
                dim=128,
                expansion_ratio=4.0,
                dropout_rate=0.1,
                name=f'repmixer_{i}'
            )(x)

        outputs = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1000)(outputs)

        model = keras.Model(inputs, outputs)

        # Test forward pass
        test_input = keras.random.normal(shape=(2, 56, 56, 128))
        output = model(test_input)

        assert output.shape == (2, 1000)

    def test_stem_to_repmixer_pipeline(self) -> None:
        """Test ConvolutionalStem followed by RepMixer blocks."""
        inputs = keras.Input(shape=(224, 224, 3))

        # Convolutional stem
        x = ConvolutionalStem(out_channels=96)(inputs)

        # RepMixer blocks
        x = RepMixerBlock(dim=96, expansion_ratio=4.0)(x)
        x = RepMixerBlock(dim=96, expansion_ratio=4.0)(x)

        # Final classification
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Test compilation and forward pass
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        test_input = keras.random.normal(shape=(2, 224, 224, 3))
        output = model(test_input)

        assert output.shape == (2, 10)

        # Test serialization of combined model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'combined_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model(test_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Combined model predictions should match after serialization"
            )


# Additional utility function for debugging
def debug_layer_serialization(layer_class, layer_config: Dict[str, Any], sample_input: keras.KerasTensor) -> None:
    """Debug helper for layer serialization issues."""
    try:
        # Test basic functionality
        layer = layer_class(**layer_config)
        output = layer(sample_input)
        print(f"✅ Forward pass successful: {output.shape}")

        # Test configuration
        config = layer.get_config()
        print(f"✅ Configuration keys: {list(config.keys())}")

        # Test serialization
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = layer_class(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, 'debug_test.keras'))
            loaded = keras.models.load_model(os.path.join(tmpdir, 'debug_test.keras'))
            print("✅ Serialization test passed")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    # Run basic smoke tests if executed directly
    print("Running basic smoke tests...")

    # Test RepMixerBlock
    print("\nTesting RepMixerBlock...")
    sample_input = keras.random.normal(shape=(2, 32, 32, 64))
    debug_layer_serialization(RepMixerBlock, {'dim': 64}, sample_input)

    # Test ConvolutionalStem
    print("\nTesting ConvolutionalStem...")
    sample_input = keras.random.normal(shape=(2, 224, 224, 3))
    debug_layer_serialization(ConvolutionalStem, {'out_channels': 64}, sample_input)

    print("\n✅ All smoke tests passed!")