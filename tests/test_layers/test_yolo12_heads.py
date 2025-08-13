"""
Comprehensive tests for YOLOv12 task-specific heads following modern Keras 3 patterns.

These tests verify initialization, forward pass, serialization, gradient flow,
and integration of detection, segmentation, and classification heads.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any, List, Tuple
import tensorflow as tf

# Assuming the file is in dl_techniques/layers/
# Adjust the import path if your structure is different.
from dl_techniques.layers.yolo12_heads import (
    YOLOv12DetectionHead,
    YOLOv12SegmentationHead,
    YOLOv12ClassificationHead
)


class TestYOLOv12DetectionHead:
    """Comprehensive test suite for YOLOv12DetectionHead following modern patterns."""

    @pytest.fixture
    def sample_features(self) -> List[keras.KerasTensor]:
        """Create sample multi-scale feature maps for testing."""
        return [
            keras.random.normal([2, 32, 32, 256]),  # P3: H/8, W/8
            keras.random.normal([2, 16, 16, 512]),  # P4: H/16, W/16
            keras.random.normal([2, 8, 8, 1024])    # P5: H/32, W/32
        ]

    @pytest.fixture
    def input_shapes(self) -> List[Tuple[int, ...]]:
        """Standard input shapes for multi-scale features."""
        return [
            (None, 32, 32, 256),  # P3
            (None, 16, 16, 512),  # P4
            (None, 8, 8, 1024)    # P5
        ]

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 80,
            'reg_max': 16
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom configuration with all parameters."""
        return {
            'num_classes': 5,
            'reg_max': 8,
            'bbox_channels': [32, 64, 96],
            'cls_channels': [64, 96, 128],
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = YOLOv12DetectionHead()

        # Check stored configuration
        assert layer.num_classes == 80
        assert layer.reg_max == 16
        assert layer.bbox_channels is None
        assert layer.cls_channels is None
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert len(layer.bbox_branches) == 3
        assert len(layer.cls_branches) == 3
        for i in range(3):
            assert layer.bbox_branches[i] is not None
            assert layer.cls_branches[i] is not None
            assert not layer.bbox_branches[i].built
            assert not layer.cls_branches[i].built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = YOLOv12DetectionHead(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.num_classes == 5
        assert layer.reg_max == 8
        assert layer.bbox_channels == [32, 64, 96]
        assert layer.cls_channels == [64, 96, 128]
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid num_classes values
        with pytest.raises(ValueError, match="num_classes must be positive"):
            YOLOv12DetectionHead(num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            YOLOv12DetectionHead(num_classes=-5)

        # Test invalid reg_max values
        with pytest.raises(ValueError, match="reg_max must be positive"):
            YOLOv12DetectionHead(num_classes=80, reg_max=0)

        with pytest.raises(ValueError, match="reg_max must be positive"):
            YOLOv12DetectionHead(num_classes=80, reg_max=-10)

        # Test invalid bbox_channels length
        with pytest.raises(ValueError, match="bbox_channels must have length 3"):
            YOLOv12DetectionHead(bbox_channels=[32, 64])

        with pytest.raises(ValueError, match="All bbox_channels must be positive"):
            YOLOv12DetectionHead(bbox_channels=[32, 0, 64])

        # Test invalid cls_channels length
        with pytest.raises(ValueError, match="cls_channels must have length 3"):
            YOLOv12DetectionHead(cls_channels=[64, 96, 128, 256])

        with pytest.raises(ValueError, match="All cls_channels must be positive"):
            YOLOv12DetectionHead(cls_channels=[64, -32, 128])

    def test_build_process(self, input_shapes, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = YOLOv12DetectionHead(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Build the layer explicitly
        layer.build(input_shapes)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        for i in range(3):
            assert layer.bbox_branches[i].built
            assert layer.cls_branches[i].built

        # Verify correct number of sub-layers in each branch
        for i in range(3):
            assert len(layer.bbox_branches[i].layers) == 3  # 2 ConvBlocks + 1 Conv2D
            assert len(layer.cls_branches[i].layers) == 5  # 4 ConvBlocks + 1 Conv2D

    def test_build_input_validation(self, layer_config):
        """Test build method input validation."""
        layer = YOLOv12DetectionHead(**layer_config)

        # Test with non-list input
        with pytest.raises(ValueError, match="DetectionHead expects a list of input shapes"):
            layer.build((None, 32, 32, 256))

        # Test with wrong number of inputs
        with pytest.raises(ValueError, match="DetectionHead expects exactly 3 input feature maps"):
            layer.build([(None, 32, 32, 256), (None, 16, 16, 512)])

    def test_forward_pass_basic(self, sample_features, input_shapes, layer_config):
        """Test basic forward pass functionality."""
        layer = YOLOv12DetectionHead(**layer_config)
        layer.build(input_shapes)

        output = layer(sample_features)

        # Calculate expected total anchors: 32*32 + 16*16 + 8*8 = 1024 + 256 + 64 = 1344
        # Expected output channels: 4*reg_max + num_classes = 4*16 + 80 = 144
        expected_shape = (2, 1344, 144)
        assert output.shape == expected_shape

        # Basic sanity checks
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_forward_pass_input_validation(self, layer_config):
        """Test forward pass input validation."""
        layer = YOLOv12DetectionHead(**layer_config)

        # Test with wrong number of inputs
        with pytest.raises(ValueError, match="DetectionHead expects exactly 3 input feature maps"):
            layer([keras.random.normal([2, 32, 32, 256])])  # Only 1 input

    def test_different_configurations(self, sample_features, input_shapes):
        """Test layer with different configuration parameters."""
        configs = [
            {'num_classes': 1, 'reg_max': 8},
            {'num_classes': 20, 'reg_max': 32},
            {'num_classes': 5, 'reg_max': 16, 'bbox_channels': [16, 32, 48], 'cls_channels': [32, 64, 96]}
        ]

        for config in configs:
            layer = YOLOv12DetectionHead(**config)
            layer.build(input_shapes)
            output = layer(sample_features)

            # Calculate expected output channels
            expected_channels = 4 * config['reg_max'] + config['num_classes']
            assert output.shape == (2, 1344, expected_channels)

    def test_compute_output_shape(self, input_shapes, layer_config):
        """Test output shape computation."""
        layer = YOLOv12DetectionHead(**layer_config)

        computed_shape = layer.compute_output_shape(input_shapes)
        expected_shape = (None, 1344, 144)  # batch_size=None from input_shapes[0][0]
        assert computed_shape == expected_shape

        # Test with dynamic shapes
        dynamic_shapes = [(None, None, None, 256), (None, None, None, 512), (None, None, None, 1024)]
        computed_shape = layer.compute_output_shape(dynamic_shapes)
        expected_shape = (None, None, 144)
        assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = YOLOv12DetectionHead(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'num_classes', 'reg_max', 'bbox_channels', 'cls_channels',
            'kernel_initializer', 'kernel_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['num_classes'] == 5
        assert config['reg_max'] == 8
        assert config['bbox_channels'] == [32, 64, 96]
        assert config['cls_channels'] == [64, 96, 128]

    def test_serialization_cycle(self, layer_config, sample_features):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        p3_input = keras.Input(shape=(32, 32, 256), name='p3')
        p4_input = keras.Input(shape=(16, 16, 512), name='p4')
        p5_input = keras.Input(shape=(8, 8, 1024), name='p5')
        inputs = [p3_input, p4_input, p5_input]

        outputs = YOLOv12DetectionHead(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_features)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_features)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_gradient_flow(self, sample_features, input_shapes, layer_config):
        """Test that gradients flow properly through the layer."""
        layer = YOLOv12DetectionHead(**layer_config)
        layer.build(input_shapes)

        with tf.GradientTape() as tape:
            # Tensors need to be watched manually when not created by a Variable
            tape.watch(sample_features)
            output = layer(sample_features)
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
    def test_training_modes(self, sample_features, input_shapes, layer_config, training):
        """Test layer behavior in different training modes."""
        layer = YOLOv12DetectionHead(**layer_config)
        layer.build(input_shapes)
        output = layer(sample_features, training=training)

        # Basic output validation
        expected_shape = (2, 1344, 144)
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))


class TestYOLOv12SegmentationHead:
    """Comprehensive test suite for YOLOv12SegmentationHead following modern patterns."""

    @pytest.fixture
    def sample_features(self) -> List[keras.KerasTensor]:
        """Create sample multi-scale feature maps for testing."""
        return [
            keras.random.normal([2, 32, 32, 256]),  # P3: H/8, W/8
            keras.random.normal([2, 16, 16, 512]),  # P4: H/16, W/16
            keras.random.normal([2, 8, 8, 1024])    # P5: H/32, W/32
        ]

    @pytest.fixture
    def input_shapes(self) -> List[Tuple[int, ...]]:
        """Standard input shapes for multi-scale features."""
        return [
            (None, 32, 32, 256),  # P3
            (None, 16, 16, 512),  # P4
            (None, 8, 8, 1024)    # P5
        ]

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 1,
            'intermediate_filters': [128, 64, 32, 16]
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom configuration with all parameters."""
        return {
            'num_classes': 2,
            'intermediate_filters': [256, 128, 64],
            'target_size': (512, 512),
            'use_attention': True,
            'dropout_rate': 0.2,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = YOLOv12SegmentationHead()

        # Check stored configuration
        assert layer.num_classes == 1
        assert layer.intermediate_filters == [128, 64, 32, 16]
        assert layer.target_size is None
        assert layer.use_attention is True
        assert layer.dropout_rate == 0.1
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert len(layer.upconv_blocks) == 4
        assert len(layer.skip_convs) == 2  # Created in __init__ for i < 2
        assert len(layer.attention_blocks) == 2  # Created in __init__ for i < 2
        assert all(not l.built for l in layer.upconv_blocks)
        assert all(not l.built for l in layer.skip_convs)
        assert all(not l.built for l in layer.attention_blocks)
        assert layer.final_conv is not None and not layer.final_conv.built
        assert layer.dropout is not None

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = YOLOv12SegmentationHead(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.num_classes == 2
        assert layer.intermediate_filters == [256, 128, 64]
        assert layer.target_size == (512, 512)
        assert layer.use_attention is True
        assert layer.dropout_rate == 0.2
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid num_classes values
        with pytest.raises(ValueError, match="num_classes must be positive"):
            YOLOv12SegmentationHead(num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            YOLOv12SegmentationHead(num_classes=-1)

        # Test invalid intermediate_filters
        with pytest.raises(ValueError, match="intermediate_filters must have at least 2 elements"):
            YOLOv12SegmentationHead(intermediate_filters=[64])

        # Test invalid dropout_rate values
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            YOLOv12SegmentationHead(dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            YOLOv12SegmentationHead(dropout_rate=1.5)

        # Test invalid target_size
        with pytest.raises(ValueError, match="target_size must be a tuple/list of 2 integers"):
            YOLOv12SegmentationHead(target_size=(256,))

        with pytest.raises(ValueError, match="target_size values must be positive"):
            YOLOv12SegmentationHead(target_size=(256, -128))

    def test_target_size_computation(self, input_shapes):
        """Test automatic target size computation."""
        layer = YOLOv12SegmentationHead(num_classes=1)
        layer.build(input_shapes)

        # P3 has shape (None, 32, 32, 256), so target should be (32*8, 32*8) = (256, 256)
        assert layer._computed_target_size == (256, 256)

    def test_build_process(self, input_shapes, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = YOLOv12SegmentationHead(**layer_config)

        # Layer should not be built initially
        assert not layer.built
        assert len(layer.skip_convs) == 2  # Created in __init__
        assert len(layer.attention_blocks) == 2 # Created in __init__

        # Build the layer explicitly
        layer.build(input_shapes)

        # Check that layer is now built
        assert layer.built

        # Check that sub-layers are built
        assert all(block.built for block in layer.upconv_blocks)
        assert all(block.built for block in layer.skip_convs)
        assert all(block.built for block in layer.attention_blocks)
        assert layer.final_conv.built

    def test_forward_pass_basic(self, sample_features, input_shapes, layer_config):
        """Test basic forward pass functionality."""
        layer = YOLOv12SegmentationHead(**layer_config)
        layer.build(input_shapes)

        output = layer(sample_features)

        # Expected output shape: (batch_size, target_height, target_width, num_classes)
        expected_shape = (2, 256, 256, 1)  # Auto-computed from P3: 32*8 = 256
        assert output.shape == expected_shape

        # Basic sanity checks
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_forward_pass_custom_target_size(self, sample_features, input_shapes):
        """Test forward pass with custom target size."""
        layer = YOLOv12SegmentationHead(
            num_classes=2,
            target_size=(512, 512),
            intermediate_filters=[128, 64, 32]
        )
        layer.build(input_shapes)

        output = layer(sample_features)
        expected_shape = (2, 512, 512, 2)
        assert output.shape == expected_shape

    def test_serialization_cycle(self, layer_config, sample_features):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        p3_input = keras.Input(shape=(32, 32, 256), name='p3')
        p4_input = keras.Input(shape=(16, 16, 512), name='p4')
        p5_input = keras.Input(shape=(8, 8, 1024), name='p5')
        inputs = [p3_input, p4_input, p5_input]

        outputs = YOLOv12SegmentationHead(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_features)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_features)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, sample_features, input_shapes, layer_config, training):
        """Test layer behavior in different training modes."""
        layer = YOLOv12SegmentationHead(**layer_config)
        layer.build(input_shapes)
        output = layer(sample_features, training=training)

        # Basic output validation
        expected_shape = (2, 256, 256, 1)
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))


class TestYOLOv12ClassificationHead:
    """Comprehensive test suite for YOLOv12ClassificationHead following modern patterns."""

    @pytest.fixture
    def sample_features(self) -> List[keras.KerasTensor]:
        """Create sample multi-scale feature maps for testing."""
        return [
            keras.random.normal([2, 32, 32, 256]),  # P3: H/8, W/8
            keras.random.normal([2, 16, 16, 512]),  # P4: H/16, W/16
            keras.random.normal([2, 8, 8, 1024])    # P5: H/32, W/32
        ]

    @pytest.fixture
    def input_shapes(self) -> List[Tuple[int, ...]]:
        """Standard input shapes for multi-scale features."""
        return [
            (None, 32, 32, 256),  # P3
            (None, 16, 16, 512),  # P4
            (None, 8, 8, 1024)    # P5
        ]

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 2,
            'hidden_dims': [512, 256]
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom configuration with all parameters."""
        return {
            'num_classes': 5,
            'hidden_dims': [1024, 512, 256],
            'pooling_types': ['avg', 'max'],
            'use_attention': True,
            'dropout_rate': 0.4,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = YOLOv12ClassificationHead()

        # Check stored configuration
        assert layer.num_classes == 1
        assert layer.hidden_dims == [512, 256]
        assert layer.pooling_types == ["avg", "max"]
        assert layer.use_attention is True
        assert layer.dropout_rate == 0.3
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert len(layer.pooling_layers) == 2  # avg + max
        assert len(layer.dense_layers) == 2   # For hidden_dims
        assert len(layer.dropout_layers) == 2
        assert layer.final_dense is not None

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = YOLOv12ClassificationHead(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.num_classes == 5
        assert layer.hidden_dims == [1024, 512, 256]
        assert layer.pooling_types == ['avg', 'max']
        assert layer.use_attention is True
        assert layer.dropout_rate == 0.4
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid num_classes values
        with pytest.raises(ValueError, match="num_classes must be positive"):
            YOLOv12ClassificationHead(num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            YOLOv12ClassificationHead(num_classes=-1)

        # Test empty hidden_dims
        with pytest.raises(ValueError, match="hidden_dims cannot be empty"):
            YOLOv12ClassificationHead(hidden_dims=[])

        with pytest.raises(ValueError, match="All hidden_dims must be positive"):
            YOLOv12ClassificationHead(hidden_dims=[512, 0, 256])

        # Test empty pooling_types
        with pytest.raises(ValueError, match="pooling_types cannot be empty"):
            YOLOv12ClassificationHead(pooling_types=[])

        # Test invalid pooling types
        with pytest.raises(ValueError, match="Invalid pooling type: invalid"):
            YOLOv12ClassificationHead(pooling_types=["avg", "invalid"])

        # Test invalid dropout_rate values
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            YOLOv12ClassificationHead(dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            YOLOv12ClassificationHead(dropout_rate=1.5)

    def test_build_process(self, input_shapes, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = YOLOv12ClassificationHead(**layer_config)

        # Layer should not be built initially
        assert not layer.built
        assert layer.attention_pooling is None  # Created in build()

        # Build the layer explicitly
        layer.build(input_shapes)

        # Check that layer is now built
        assert layer.built

        # Check that sub-layers are built
        assert all(hasattr(pool, 'built') and pool.built for pool in layer.pooling_layers)
        assert all(dense.built for dense in layer.dense_layers)
        assert layer.final_dense.built
        if layer.use_attention:
            assert layer.attention_pooling is not None
            assert layer.attention_pooling.built

    def test_forward_pass_basic(self, sample_features, input_shapes, layer_config):
        """Test basic forward pass functionality."""
        layer = YOLOv12ClassificationHead(**layer_config)
        layer.build(input_shapes)

        output = layer(sample_features)

        # Expected output shape: (batch_size, num_classes)
        expected_shape = (2, 2)
        assert output.shape == expected_shape

        # Basic sanity checks
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_different_pooling_configurations(self, sample_features, input_shapes):
        """Test layer with different pooling configurations."""
        pooling_configs = [
            ['avg'],
            ['max'],
            ['avg', 'max']
        ]

        for pooling_types in pooling_configs:
            layer = YOLOv12ClassificationHead(
                num_classes=3,
                pooling_types=pooling_types
            )
            layer.build(input_shapes)
            output = layer(sample_features)

            assert output.shape == (2, 3)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_attention_mechanism(self, sample_features, input_shapes):
        """Test attention mechanism functionality."""
        # Test with attention
        layer_with_attention = YOLOv12ClassificationHead(use_attention=True)
        layer_with_attention.build(input_shapes)
        output_with = layer_with_attention(sample_features)

        # Test without attention
        layer_without_attention = YOLOv12ClassificationHead(use_attention=False)
        layer_without_attention.build(input_shapes)
        output_without = layer_without_attention(sample_features)

        # Both should have same shape but different values
        assert output_with.shape == output_without.shape == (2, 1)
        # Outputs should be different due to attention mechanism
        with_numpy = keras.ops.convert_to_numpy(output_with)
        without_numpy = keras.ops.convert_to_numpy(output_without)
        assert not np.allclose(with_numpy, without_numpy, atol=1e-6)

    def test_serialization_cycle(self, layer_config, sample_features):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        p3_input = keras.Input(shape=(32, 32, 256), name='p3')
        p4_input = keras.Input(shape=(16, 16, 512), name='p4')
        p5_input = keras.Input(shape=(8, 8, 1024), name='p5')
        inputs = [p3_input, p4_input, p5_input]

        outputs = YOLOv12ClassificationHead(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_features)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_features)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, sample_features, input_shapes, layer_config, training):
        """Test layer behavior in different training modes."""
        layer = YOLOv12ClassificationHead(**layer_config)
        layer.build(input_shapes)
        output = layer(sample_features, training=training)

        # Basic output validation
        expected_shape = (2, 2)
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))


class TestHeadsIntegration:
    """Integration tests for all heads together following modern patterns."""

    @pytest.fixture
    def sample_features(self) -> List[keras.KerasTensor]:
        """Create sample multi-scale feature maps for testing."""
        return [
            keras.random.normal([2, 32, 32, 256]),  # P3: H/8, W/8
            keras.random.normal([2, 16, 16, 512]),  # P4: H/16, W/16
            keras.random.normal([2, 8, 8, 1024])    # P5: H/32, W/32
        ]

    @pytest.fixture
    def input_shapes(self) -> List[Tuple[int, ...]]:
        """Standard input shapes for multi-scale features."""
        return [
            (None, 32, 32, 256),  # P3
            (None, 16, 16, 512),  # P4
            (None, 8, 8, 1024)    # P5
        ]

    def test_all_heads_integration(self, sample_features, input_shapes):
        """Test that all heads can process the same feature maps."""
        # Create all heads with compatible configurations
        detection_head = YOLOv12DetectionHead(num_classes=1, reg_max=16)
        segmentation_head = YOLOv12SegmentationHead(
            num_classes=1,
            target_size=(256, 256),
            intermediate_filters=[128, 64, 32, 16]
        )
        classification_head = YOLOv12ClassificationHead(num_classes=1)

        # Build all heads
        detection_head.build(input_shapes)
        segmentation_head.build(input_shapes)
        classification_head.build(input_shapes)

        # Forward pass through all heads
        detection_out = detection_head(sample_features)
        segmentation_out = segmentation_head(sample_features)
        classification_out = classification_head(sample_features)

        # Check all outputs have expected shapes
        assert detection_out.shape == (2, 1344, 65)  # 1344 anchors, 65 channels (4*16+1)
        assert segmentation_out.shape == (2, 256, 256, 1)  # Full resolution mask
        assert classification_out.shape == (2, 1)  # Single class logits

        # All outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(detection_out))
        assert not keras.ops.any(keras.ops.isnan(segmentation_out))
        assert not keras.ops.any(keras.ops.isnan(classification_out))

    def test_multitask_model_integration(self, sample_features):
        """Test integration in a complete multitask model."""
        # Create inputs
        p3_input = keras.Input(shape=(32, 32, 256), name='p3')
        p4_input = keras.Input(shape=(16, 16, 512), name='p4')
        p5_input = keras.Input(shape=(8, 8, 1024), name='p5')
        inputs = [p3_input, p4_input, p5_input]

        # Create multitask outputs
        detection_out = YOLOv12DetectionHead(num_classes=80, reg_max=16, name='detection')(inputs)
        segmentation_out = YOLOv12SegmentationHead(num_classes=1, name='segmentation')(inputs)
        classification_out = YOLOv12ClassificationHead(num_classes=2, name='classification')(inputs)

        # Create model
        model = keras.Model(
            inputs=inputs,
            outputs={
                'detection': detection_out,
                'segmentation': segmentation_out,
                'classification': classification_out
            }
        )

        # Compile model
        model.compile(
            optimizer='adam',
            loss={
                'detection': 'mse',
                'segmentation': 'binary_crossentropy',
                'classification': 'sparse_categorical_crossentropy'
            }
        )

        # Test forward pass
        outputs = model(sample_features)

        # Verify outputs
        assert outputs['detection'].shape == (2, 1344, 144)
        assert outputs['segmentation'].shape == (2, 256, 256, 1)
        assert outputs['classification'].shape == (2, 2)

    def test_different_input_sizes(self):
        """Test heads with different patch sizes."""
        # Test with smaller patches (128x128 input -> 16x16, 8x8, 4x4 features)
        input_shapes_small = [
            (None, 16, 16, 256),  # P3: 128/8 = 16
            (None, 8, 8, 512),    # P4: 128/16 = 8
            (None, 4, 4, 1024)    # P5: 128/32 = 4
        ]

        features_small = [
            keras.random.normal([1, 16, 16, 256]),
            keras.random.normal([1, 8, 8, 512]),
            keras.random.normal([1, 4, 4, 1024])
        ]

        # Test segmentation head with explicit target size
        seg_head = YOLOv12SegmentationHead(
            num_classes=1,
            target_size=(128, 128),
            intermediate_filters=[64, 32, 16]
        )
        seg_head.build(input_shapes_small)
        output = seg_head(features_small)
        assert output.shape == (1, 128, 128, 1)

        # Test detection head
        det_head = YOLOv12DetectionHead(num_classes=5, reg_max=8)
        det_head.build(input_shapes_small)
        output = det_head(features_small)
        # Total anchors: 16*16 + 8*8 + 4*4 = 256 + 64 + 16 = 336
        # Output channels: 4*8 + 5 = 37
        assert output.shape == (1, 336, 37)

        # Test classification head
        cls_head = YOLOv12ClassificationHead(num_classes=3)
        cls_head.build(input_shapes_small)
        output = cls_head(features_small)
        assert output.shape == (1, 3)

    def test_gradient_flow_integration(self, sample_features, input_shapes):
        """Test gradient flow through all heads in integrated model."""
        # Create simple multitask model
        inputs_layer = [
            keras.Input(shape=shape[1:], name=f'input_{i}')
            for i, shape in enumerate(input_shapes)
        ]

        detection_out = YOLOv12DetectionHead(num_classes=2)(inputs_layer)
        segmentation_out = YOLOv12SegmentationHead(num_classes=1)(inputs_layer)
        classification_out = YOLOv12ClassificationHead(num_classes=2)(inputs_layer)

        model = keras.Model(inputs_layer, [detection_out, segmentation_out, classification_out])

        with tf.GradientTape() as tape:
            tape.watch(sample_features)
            det_out, seg_out, cls_out = model(sample_features)

            # Compute combined loss
            det_loss = keras.ops.mean(keras.ops.square(det_out))
            seg_loss = keras.ops.mean(keras.ops.square(seg_out))
            cls_loss = keras.ops.mean(keras.ops.square(cls_out))
            total_loss = det_loss + seg_loss + cls_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Check that all gradients exist
        assert len(gradients) == len(model.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_numerical_stability(self, input_shapes):
        """Test numerical stability with extreme input values."""
        # Create heads
        detection_head = YOLOv12DetectionHead(num_classes=1, reg_max=8)
        segmentation_head = YOLOv12SegmentationHead(num_classes=1)
        classification_head = YOLOv12ClassificationHead(num_classes=1)

        # Build heads
        detection_head.build(input_shapes)
        segmentation_head.build(input_shapes)
        classification_head.build(input_shapes)

        # Test different input value ranges
        test_cases = [
            # Very small values
            [keras.ops.ones((1, 32, 32, 256)) * 1e-10,
             keras.ops.ones((1, 16, 16, 512)) * 1e-10,
             keras.ops.ones((1, 8, 8, 1024)) * 1e-10],
            # Large values
            [keras.ops.ones((1, 32, 32, 256)) * 100,
             keras.ops.ones((1, 16, 16, 512)) * 100,
             keras.ops.ones((1, 8, 8, 1024)) * 100],
            # Zero values
            [keras.ops.zeros((1, 32, 32, 256)),
             keras.ops.zeros((1, 16, 16, 512)),
             keras.ops.zeros((1, 8, 8, 1024))]
        ]

        for features in test_cases:
            # Test all heads
            det_out = detection_head(features)
            seg_out = segmentation_head(features)
            cls_out = classification_head(features)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(det_out)), "Detection head NaN detected"
            assert not keras.ops.any(keras.ops.isinf(det_out)), "Detection head Inf detected"

            assert not keras.ops.any(keras.ops.isnan(seg_out)), "Segmentation head NaN detected"
            assert not keras.ops.any(keras.ops.isinf(seg_out)), "Segmentation head Inf detected"

            assert not keras.ops.any(keras.ops.isnan(cls_out)), "Classification head NaN detected"
            assert not keras.ops.any(keras.ops.isinf(cls_out)), "Classification head Inf detected"