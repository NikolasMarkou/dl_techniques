"""
Comprehensive test suite for MLFCLayer following Modern Keras 3 testing patterns.

This module provides thorough testing coverage for the MLFCLayer including:
- Initialization and configuration validation
- Forward pass functionality
- Serialization cycle robustness
- Gradient flow verification
- Training mode behavior
- Edge case error handling
- Multi-iteration functionality
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Any, Dict, List, Tuple

import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.mlfc_layer import MLFCLayer


class TestMLFCLayer:
    """Comprehensive test suite for MLFCLayer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'channels_list': [32, 64, 128, 256],
            'num_iterations': 1,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros'
        }

    @pytest.fixture
    def multi_iteration_config(self) -> Dict[str, Any]:
        """Configuration with multiple iterations for testing."""
        return {
            'channels_list': [64, 128, 256, 512],
            'num_iterations': 3,
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    @pytest.fixture
    def sample_inputs(self) -> List[keras.KerasTensor]:
        """Sample input tensors for testing (4 levels with 2x downsampling)."""
        # Level 1: highest resolution (batch=2, h=32, w=32, c=32)
        x1 = keras.random.normal(shape=(2, 32, 32, 32))
        # Level 2: half resolution (batch=2, h=16, w=16, c=64)
        x2 = keras.random.normal(shape=(2, 16, 16, 64))
        # Level 3: quarter resolution (batch=2, h=8, w=8, c=128)
        x3 = keras.random.normal(shape=(2, 8, 8, 128))
        # Level 4: eighth resolution (batch=2, h=4, w=4, c=256)
        x4 = keras.random.normal(shape=(2, 4, 4, 256))

        return [x1, x2, x3, x4]

    @pytest.fixture
    def sample_input_shapes(self) -> List[Tuple[int, ...]]:
        """Sample input shapes for building."""
        return [
            (None, 32, 32, 32),   # Level 1
            (None, 16, 16, 64),   # Level 2
            (None, 8, 8, 128),    # Level 3
            (None, 4, 4, 256)     # Level 4
        ]

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization and attribute setting."""
        layer = MLFCLayer(**layer_config)

        # Check configuration attributes
        assert layer.channels_list == [32, 64, 128, 256]
        assert layer.num_iterations == 1
        assert layer.total_channels == 32 + 64 + 128 + 256
        assert not layer.built

        # Check that sub-layers were created (total number in flat list)
        expected_layers_per_iter = 4
        total_expected = layer.num_iterations * expected_layers_per_iter
        assert len(layer.compilation_convs) == total_expected
        assert len(layer.squeeze_excitations) == 4  # 4 levels
        assert layer.activation is not None

        # Check sub-layer types
        assert isinstance(layer.compilation_convs[0], keras.layers.Conv2D)
        assert isinstance(layer.batch_norms[0], keras.layers.BatchNormalization)
        from dl_techniques.layers.squeeze_excitation import SqueezeExcitation # Import for isinstance
        assert isinstance(layer.squeeze_excitations[0], SqueezeExcitation)
        assert isinstance(layer.activation, keras.layers.LeakyReLU)

    def test_initialization_multi_iteration(self, multi_iteration_config: Dict[str, Any]) -> None:
        """Test layer initialization with multiple iterations."""
        layer = MLFCLayer(**multi_iteration_config)

        assert layer.num_iterations == 3

        # Check total number of layers in flat lists
        expected_total = layer.num_iterations * 4
        assert len(layer.compilation_convs) == expected_total
        assert len(layer.merge_convs) == expected_total
        assert len(layer.batch_norms) == expected_total


    def test_forward_pass(
        self,
        layer_config: Dict[str, Any],
        sample_inputs: List[keras.KerasTensor]
    ) -> None:
        """Test forward pass and automatic building."""
        layer = MLFCLayer(**layer_config)

        # Forward pass should trigger building
        outputs = layer(sample_inputs)

        assert layer.built
        assert isinstance(outputs, list)
        assert len(outputs) == 4

        # Check output shapes match input shapes
        for i, (input_tensor, output_tensor) in enumerate(zip(sample_inputs, outputs)):
            assert ops.shape(output_tensor)[0] == ops.shape(input_tensor)[0]  # Batch size
            assert ops.shape(output_tensor)[1] == ops.shape(input_tensor)[1]  # Height
            assert ops.shape(output_tensor)[2] == ops.shape(input_tensor)[2]  # Width
            assert ops.shape(output_tensor)[3] == ops.shape(input_tensor)[3]  # Channels

    def test_forward_pass_multi_iteration(
        self,
        multi_iteration_config: Dict[str, Any],
    ) -> None:
        """Test forward pass with multiple iterations."""
        # Adjust sample inputs to match multi_iteration_config channels
        adjusted_inputs = [
            keras.random.normal(shape=(2, 32, 32, 64)),   # Level 1
            keras.random.normal(shape=(2, 16, 16, 128)),  # Level 2
            keras.random.normal(shape=(2, 8, 8, 256)),    # Level 3
            keras.random.normal(shape=(2, 4, 4, 512))     # Level 4
        ]

        layer = MLFCLayer(**multi_iteration_config)
        outputs = layer(adjusted_inputs)

        assert layer.built
        assert len(outputs) == 4

        # Verify shapes are preserved
        for input_tensor, output_tensor in zip(adjusted_inputs, outputs):
            np.testing.assert_array_equal(
                ops.convert_to_numpy(ops.shape(output_tensor)),
                ops.convert_to_numpy(ops.shape(input_tensor))
            )

    def test_serialization_cycle(
        self,
        layer_config: Dict[str, Any],
        sample_inputs: List[keras.KerasTensor]
    ) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        input_layers = [
            keras.Input(shape=tensor.shape[1:]) for tensor in sample_inputs
        ]
        layer_outputs = MLFCLayer(**layer_config)(input_layers)
        model = keras.Model(inputs=input_layers, outputs=layer_outputs)

        # Get original prediction
        original_preds = model(sample_inputs)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_mlfc_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_preds = loaded_model(sample_inputs)

            # Verify identical predictions for all outputs
            assert len(loaded_preds) == len(original_preds)
            for orig_pred, loaded_pred in zip(original_preds, loaded_preds):
                np.testing.assert_allclose(
                    ops.convert_to_numpy(orig_pred),
                    ops.convert_to_numpy(loaded_pred),
                    rtol=1e-5, atol=1e-5,
                    err_msg="Predictions differ after serialization"
                )

    def test_serialization_cycle_multi_iteration(
        self,
        multi_iteration_config: Dict[str, Any]
    ) -> None:
        """Test serialization with multiple iterations."""
        # Create appropriate inputs for multi_iteration_config
        sample_inputs = [
            keras.random.normal(shape=(2, 32, 32, 64)),
            keras.random.normal(shape=(2, 16, 16, 128)),
            keras.random.normal(shape=(2, 8, 8, 256)),
            keras.random.normal(shape=(2, 4, 4, 512))
        ]

        input_layers = [keras.Input(shape=tensor.shape[1:]) for tensor in sample_inputs]
        layer_outputs = MLFCLayer(**multi_iteration_config)(input_layers)
        model = keras.Model(inputs=input_layers, outputs=layer_outputs)

        original_preds = model(sample_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_mlfc_multi_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_preds = loaded_model(sample_inputs)

            for orig_pred, loaded_pred in zip(original_preds, loaded_preds):
                np.testing.assert_allclose(
                    ops.convert_to_numpy(orig_pred),
                    ops.convert_to_numpy(loaded_pred),
                    rtol=1e-5, atol=1e-5,
                    err_msg="Multi-iteration predictions differ after serialization"
                )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = MLFCLayer(**layer_config)
        config = layer.get_config()

        # Check all configuration parameters are present
        expected_keys = [
            'channels_list', 'num_iterations', 'kernel_initializer',
            'bias_initializer', 'kernel_regularizer', 'bias_regularizer'
        ]

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify values match initialization
        assert config['channels_list'] == layer_config['channels_list']
        assert config['num_iterations'] == layer_config['num_iterations']

    def test_config_completeness_with_regularizers(self) -> None:
        """Test config completeness with regularizers."""
        config_with_reg = {
            'channels_list': [32, 64, 128, 256],
            'num_iterations': 2,
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_regularizer': keras.regularizers.L1(1e-5)
        }

        layer = MLFCLayer(**config_with_reg)
        saved_config = layer.get_config()

        # Regularizers should be serialized
        assert saved_config['kernel_regularizer'] is not None
        assert saved_config['bias_regularizer'] is not None

    def test_gradients_flow(
        self,
        layer_config: Dict[str, Any],
        sample_inputs: List[keras.KerasTensor]
    ) -> None:
        """Test gradient computation through the layer."""
        layer = MLFCLayer(**layer_config)

        with tf.GradientTape() as tape:
            # Watch input tensors
            for inp in sample_inputs:
                tape.watch(inp)

            outputs = layer(sample_inputs)
            # Compute a simple loss over all outputs
            total_loss = sum(ops.mean(ops.square(output)) for output in outputs)

        # Get gradients with respect to layer's trainable variables
        gradients = tape.gradient(total_loss, layer.trainable_variables)

        assert len(gradients) > 0, "No gradients computed"
        assert all(g is not None for g in gradients), "Some gradients are None"

        # Check that gradients have reasonable magnitudes
        for grad in gradients:
            assert not ops.any(ops.isnan(grad)), "Gradient contains NaN"
            assert not ops.any(ops.isinf(grad)), "Gradient contains Inf"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
        self,
        layer_config: Dict[str, Any],
        sample_inputs: List[keras.KerasTensor],
        training: bool
    ) -> None:
        """Test behavior in different training modes."""
        layer = MLFCLayer(**layer_config)

        outputs = layer(sample_inputs, training=training)

        assert isinstance(outputs, list)
        assert len(outputs) == 4

        # Check that outputs have correct shapes regardless of training mode
        for inp, out in zip(sample_inputs, outputs):
            assert ops.shape(out)[0] == ops.shape(inp)[0]  # Batch size preserved
            assert len(ops.shape(out)) == 4  # 4D tensor (batch, height, width, channels)

    def test_compute_output_shape(
        self,
        layer_config: Dict[str, Any],
        sample_input_shapes: List[Tuple[int, ...]]
    ) -> None:
        """Test output shape computation."""
        layer = MLFCLayer(**layer_config)

        output_shapes = layer.compute_output_shape(sample_input_shapes)

        assert isinstance(output_shapes, list)
        assert len(output_shapes) == 4

        # Output shapes should match input shapes
        for input_shape, output_shape in zip(sample_input_shapes, output_shapes):
            assert output_shape == input_shape

    def test_build_method(
        self,
        layer_config: Dict[str, Any],
        sample_input_shapes: List[Tuple[int, ...]]
    ) -> None:
        """Test explicit building of the layer."""
        layer = MLFCLayer(**layer_config)

        assert not layer.built

        # Build the layer
        layer.build(sample_input_shapes)

        assert layer.built

        # Check that sub-layers are built using the flat index
        for iter_idx in range(layer.num_iterations):
            for level_idx in range(4):
                idx = iter_idx * 4 + level_idx
                assert layer.compilation_convs[idx].built
                assert layer.merge_convs[idx].built
                assert layer.batch_norms[idx].built
                assert layer.merge_batch_norms[idx].built

        for se_layer in layer.squeeze_excitations:
            assert se_layer.built

    def test_edge_cases(self) -> None:
        """Test error conditions and edge cases."""

        # Test invalid channels_list length
        with pytest.raises(ValueError, match="channels_list must have exactly 4 elements"):
            MLFCLayer(channels_list=[32, 64, 128])  # Only 3 elements

        with pytest.raises(ValueError, match="channels_list must have exactly 4 elements"):
            MLFCLayer(channels_list=[32, 64, 128, 256, 512])  # 5 elements

        # Test negative channel counts
        with pytest.raises(ValueError, match="All channel counts must be positive"):
            MLFCLayer(channels_list=[32, -64, 128, 256])

        with pytest.raises(ValueError, match="All channel counts must be positive"):
            MLFCLayer(channels_list=[0, 64, 128, 256])

        # Test invalid num_iterations
        with pytest.raises(ValueError, match="num_iterations must be positive"):
            MLFCLayer(channels_list=[32, 64, 128, 256], num_iterations=0)

        with pytest.raises(ValueError, match="num_iterations must be positive"):
            MLFCLayer(channels_list=[32, 64, 128, 256], num_iterations=-1)

    def test_call_edge_cases(self, layer_config: Dict[str, Any]) -> None:
        """Test call method error conditions."""
        layer = MLFCLayer(**layer_config)

        # Test wrong number of inputs - this will fail in build first
        wrong_inputs = [keras.random.normal(shape=(2, 32, 32, 32))]  # Only 1 input

        # The error will actually come from build() since Keras builds before calling
        with pytest.raises(ValueError, match="input_shape must be a list of 4 shapes"):
            layer(wrong_inputs)

        # Test with 5 inputs - also fails in build
        too_many_inputs = [
            keras.random.normal(shape=(2, 32, 32, 32)) for _ in range(5)
        ]

        with pytest.raises(ValueError, match="input_shape must be a list of 4 shapes"):
            layer(too_many_inputs)

    def test_call_validation_after_build(
        self,
        layer_config: Dict[str, Any],
        sample_inputs: List[keras.KerasTensor]
    ) -> None:
        """Test call method validation on a pre-built layer."""
        layer = MLFCLayer(**layer_config)

        # Build the layer first with correct inputs
        _ = layer(sample_inputs)
        assert layer.built

        # Now test validation on the built layer by manually calling with wrong inputs
        # This bypasses the build step and tests our call validation directly
        wrong_inputs = [keras.random.normal(shape=(2, 32, 32, 32))]  # Only 1 input

        with pytest.raises(ValueError, match="Expected 4 input tensors, got 1"):
            layer.call(wrong_inputs)

        # Test with 5 inputs
        too_many_inputs = [
            keras.random.normal(shape=(2, 32, 32, 32)) for _ in range(5)
        ]

        with pytest.raises(ValueError, match="Expected 4 input tensors, got 5"):
            layer.call(too_many_inputs)

    def test_build_edge_cases(self, layer_config: Dict[str, Any]) -> None:
        """Test build method error conditions."""
        layer = MLFCLayer(**layer_config)

        # Test invalid input_shape type
        with pytest.raises(ValueError, match="input_shape must be a list of 4 shapes"):
            layer.build((None, 32, 32, 32))  # Tuple instead of list

        # Test wrong number of shapes
        with pytest.raises(ValueError, match="input_shape must be a list of 4 shapes"):
            layer.build([(None, 32, 32, 32), (None, 16, 16, 64)])  # Only 2 shapes

    def test_layer_weights(
        self,
        layer_config: Dict[str, Any],
        sample_inputs: List[keras.KerasTensor]
    ) -> None:
        """Test that layer creates appropriate weights."""
        layer = MLFCLayer(**layer_config)

        # Build layer
        _ = layer(sample_inputs)

        # Check that layer has trainable variables
        trainable_vars = layer.trainable_variables
        assert len(trainable_vars) > 0

        # Expected number of weight tensors:
        # - For each iteration and level: compilation_conv + merge_conv (2 per level per iteration)
        # - For each iteration and level: 2 batch_norms with gamma/beta (4 per level per iteration)
        # - Squeeze excitation layers (2 per level)
        # Total expected for 1 iteration: 4 levels * (2 + 4) + 4 levels * 2 = 32
        expected_min_weights = 4 * 6 + 4 * 2  # 32 weights minimum
        assert len(trainable_vars) >= expected_min_weights

    def test_different_input_sizes(self, layer_config: Dict[str, Any]) -> None:
        """Test layer with different spatial input sizes."""
        layer = MLFCLayer(**layer_config)

        # Create inputs with different sizes but same channel structure
        different_inputs = [
            keras.random.normal(shape=(1, 64, 64, 32)),   # Larger Level 1
            keras.random.normal(shape=(1, 32, 32, 64)),   # Larger Level 2
            keras.random.normal(shape=(1, 16, 16, 128)),  # Larger Level 3
            keras.random.normal(shape=(1, 8, 8, 256))     # Larger Level 4
        ]

        outputs = layer(different_inputs)

        assert len(outputs) == 4

        # Outputs should maintain input spatial dimensions
        for inp, out in zip(different_inputs, outputs):
            assert ops.shape(out)[1] == ops.shape(inp)[1]  # Height
            assert ops.shape(out)[2] == ops.shape(inp)[2]  # Width
            assert ops.shape(out)[3] == ops.shape(inp)[3]  # Channels


# Additional test fixtures for specific scenarios
@pytest.fixture
def minimal_inputs():
    """Minimal valid inputs for quick testing."""
    return [
        keras.random.normal(shape=(1, 4, 4, 32)),
        keras.random.normal(shape=(1, 2, 2, 64)),
        keras.random.normal(shape=(1, 1, 1, 128)),
        keras.random.normal(shape=(1, 1, 1, 256))
    ]


def test_minimal_case(minimal_inputs):
    """Test with minimal input sizes."""
    layer = MLFCLayer(channels_list=[32, 64, 128, 256])

    outputs = layer(minimal_inputs)

    assert len(outputs) == 4
    for inp, out in zip(minimal_inputs, outputs):
        assert ops.shape(out)[0] == ops.shape(inp)[0]  # Batch preserved
