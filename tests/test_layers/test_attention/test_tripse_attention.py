"""
Comprehensive test suite for TripSE attention layers.

Tests all four variants (TripSE1, TripSE2, TripSE3, TripSE4) and the base
TripletAttentionBranch component following Keras 3 best practices.
"""

import tempfile
import os
import pytest
import numpy as np
import tensorflow as tf
import keras

from dl_techniques.layers.attention.tripse_attention import (
    TripletAttentionBranch,
    TripSE1,
    TripSE2,
    TripSE3,
    TripSE4,
)


class TestTripletAttentionBranch:
    """Test suite for TripletAttentionBranch component."""

    @pytest.fixture
    def default_config(self):
        """Default configuration for TripletAttentionBranch."""
        return {
            'kernel_size': 7,
            'permute_pattern': (0, 1, 2),
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
        }

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor (batch=2, height=32, width=32, channels=64)."""
        return np.random.randn(2, 32, 32, 64).astype(np.float32)

    def test_instantiation(self, default_config):
        """Test layer can be instantiated with valid config."""
        layer = TripletAttentionBranch(**default_config)
        assert layer.kernel_size == default_config['kernel_size']
        assert layer.permute_pattern == default_config['permute_pattern']
        assert layer.use_bias == default_config['use_bias']

    @pytest.mark.parametrize("permute_pattern", [
        (0, 1, 2),  # H-W branch (no rotation)
        (0, 2, 1),  # C-W branch
        (2, 1, 0),  # H-C branch
    ])
    def test_permute_patterns(self, permute_pattern, sample_input):
        """Test all permutation patterns work correctly."""
        layer = TripletAttentionBranch(permute_pattern=permute_pattern)
        output = layer(sample_input)
        # Output shape should match input shape regardless of internal rotation
        assert output.shape == sample_input.shape

    def test_forward_pass_shape(self, default_config, sample_input):
        """Test forward pass produces correct output shape."""
        layer = TripletAttentionBranch(**default_config)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_build_creates_weights(self, default_config, sample_input):
        """Test that build() creates expected weights."""
        layer = TripletAttentionBranch(**default_config)
        layer(sample_input)  # Triggers build

        # Check conv layer weights
        assert layer.conv.kernel is not None
        assert layer.conv.kernel.shape[-1] == 1  # Single output channel

        # Check batch norm weights
        assert layer.batch_norm.gamma is not None
        assert layer.batch_norm.beta is not None

    def test_training_vs_inference(self, default_config, sample_input):
        """Test layer behaves correctly in training vs inference mode."""
        layer = TripletAttentionBranch(**default_config)

        # Get outputs in both modes
        train_output = layer(sample_input, training=True)
        infer_output = layer(sample_input, training=False)

        # Shapes should match
        assert train_output.shape == infer_output.shape

    def test_serialization_cycle(self, default_config, sample_input):
        """Test full save/load cycle preserves functionality."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = TripletAttentionBranch(**default_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-5, atol=1e-5,
            err_msg="Outputs should match after serialization"
        )

    def test_get_config_complete(self, default_config):
        """Test get_config returns all constructor arguments."""
        layer = TripletAttentionBranch(**default_config)
        config = layer.get_config()

        assert 'kernel_size' in config
        assert 'permute_pattern' in config
        assert 'use_bias' in config
        assert 'kernel_initializer' in config
        assert 'kernel_regularizer' in config

    def test_from_config_reconstruction(self, default_config, sample_input):
        """Test layer can be reconstructed from config."""
        original = TripletAttentionBranch(**default_config)
        original(sample_input)

        config = original.get_config()
        reconstructed = TripletAttentionBranch.from_config(config)

        assert reconstructed.kernel_size == original.kernel_size
        assert reconstructed.permute_pattern == original.permute_pattern
        assert reconstructed.use_bias == original.use_bias

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_variable_batch_size(self, default_config, batch_size):
        """Test layer handles various batch sizes."""
        layer = TripletAttentionBranch(**default_config)
        test_input = np.random.randn(batch_size, 32, 32, 64).astype(np.float32)
        output = layer(test_input)
        assert output.shape == (batch_size, 32, 32, 64)

    @pytest.mark.parametrize("spatial_size", [16, 32])
    def test_variable_spatial_size(self, default_config, spatial_size):
        """Test layer handles various spatial dimensions."""
        layer = TripletAttentionBranch(**default_config)
        test_input = np.random.randn(2, spatial_size, spatial_size, 64).astype(np.float32)
        output = layer(test_input)
        assert output.shape == (2, spatial_size, spatial_size, 64)

    @pytest.mark.parametrize("channels", [16, 32])
    def test_variable_channels(self, default_config, channels):
        """Test layer handles various channel counts."""
        layer = TripletAttentionBranch(**default_config)
        test_input = np.random.randn(2, 32, 32, channels).astype(np.float32)
        output = layer(test_input)
        assert output.shape == (2, 32, 32, channels)

    def test_with_regularization(self, sample_input):
        """Test layer with kernel regularization."""
        layer = TripletAttentionBranch(
            kernel_regularizer=keras.regularizers.L2(0.01)
        )
        layer(sample_input)
        assert len(layer.losses) > 0


class TestTripSE1:
    """Test suite for TripSE1 layer."""

    @pytest.fixture
    def default_config(self):
        return {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
        }

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(2, 32, 32, 64).astype(np.float32)

    def test_instantiation(self, default_config):
        layer = TripSE1(**default_config)
        assert layer.reduction_ratio == default_config['reduction_ratio']
        assert layer.kernel_size == default_config['kernel_size']

    def test_forward_pass_shape(self, default_config, sample_input):
        layer = TripSE1(**default_config)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_build_creates_sublayers(self, default_config, sample_input):
        """Test that build() creates all expected sub-layers."""
        layer = TripSE1(**default_config)
        layer(sample_input)  # Triggers build

        assert layer.branch_hw is not None
        assert layer.branch_cw is not None
        assert layer.branch_hc is not None
        assert layer.se_block is not None

        assert layer.branch_hw.built
        assert layer.branch_cw.built
        assert layer.branch_hc.built
        assert layer.se_block.built

    def test_training_vs_inference(self, default_config, sample_input):
        layer = TripSE1(**default_config)
        train_output = layer(sample_input, training=True)
        infer_output = layer(sample_input, training=False)
        assert train_output.shape == infer_output.shape

    def test_serialization_cycle(self, default_config, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = TripSE1(**default_config)(inputs)
        model = keras.Model(inputs, outputs)
        original_output = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-5, atol=1e-5
        )

    def test_get_config_complete(self, default_config):
        layer = TripSE1(**default_config)
        config = layer.get_config()
        assert 'reduction_ratio' in config
        assert 'kernel_size' in config

    def test_from_config_reconstruction(self, default_config, sample_input):
        original = TripSE1(**default_config)
        original(sample_input)
        config = original.get_config()
        reconstructed = TripSE1.from_config(config)
        assert reconstructed.reduction_ratio == original.reduction_ratio

    @pytest.mark.parametrize("reduction_ratio", [0.0625, 0.25])
    def test_different_reduction_ratios(self, reduction_ratio, sample_input):
        layer = TripSE1(reduction_ratio=reduction_ratio)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_with_regularization(self, sample_input):
        layer = TripSE1(kernel_regularizer=keras.regularizers.L2(0.01))
        layer(sample_input)
        assert len(layer.losses) > 0


class TestTripSE2:
    """Test suite for TripSE2 layer."""

    @pytest.fixture
    def default_config(self):
        return {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
        }

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(2, 32, 32, 64).astype(np.float32)

    def test_forward_pass_shape(self, default_config, sample_input):
        layer = TripSE2(**default_config)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_build_creates_sublayers(self, default_config, sample_input):
        """Test that build() creates all expected sub-layers in lists."""
        layer = TripSE2(**default_config)
        layer(sample_input)  # Triggers build

        # Verify list containers exist and are populated
        assert len(layer.se_layers) == 3
        assert len(layer.conv_layers) == 3
        assert len(layer.bn_layers) == 3

        # Verify individual components are built
        for i in range(3):
            assert layer.se_layers[i].built
            assert layer.conv_layers[i].built
            assert layer.bn_layers[i].built

    def test_serialization_cycle(self, default_config, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = TripSE2(**default_config)(inputs)
        model = keras.Model(inputs, outputs)
        original_output = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-5, atol=1e-5
        )

    def test_get_config_complete(self, default_config):
        layer = TripSE2(**default_config)
        config = layer.get_config()
        assert 'reduction_ratio' in config
        assert 'kernel_size' in config


class TestTripSE3:
    """Test suite for TripSE3 layer."""

    @pytest.fixture
    def default_config(self):
        return {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
        }

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(2, 32, 32, 64).astype(np.float32)

    def test_forward_pass_shape(self, default_config, sample_input):
        layer = TripSE3(**default_config)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_build_creates_sublayers(self, default_config, sample_input):
        """Test that build() creates all expected sub-layers in lists."""
        layer = TripSE3(**default_config)
        layer(sample_input)  # Triggers build

        # Verify list containers
        assert len(layer.se_layers) == 3
        assert len(layer.conv_layers) == 3
        assert len(layer.bn_layers) == 3

        # Verify built status
        for i in range(3):
            assert layer.se_layers[i].built
            assert layer.conv_layers[i].built
            assert layer.bn_layers[i].built

    def test_serialization_cycle(self, default_config, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = TripSE3(**default_config)(inputs)
        model = keras.Model(inputs, outputs)
        original_output = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-5, atol=1e-5
        )


class TestTripSE4:
    """Test suite for TripSE4 layer."""

    @pytest.fixture
    def default_config(self):
        return {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
        }

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(2, 32, 32, 64).astype(np.float32)

    def test_forward_pass_shape(self, default_config, sample_input):
        layer = TripSE4(**default_config)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_build_creates_sublayers(self, default_config, sample_input):
        """Test that build() creates all expected sub-layers."""
        layer = TripSE4(**default_config)
        layer(sample_input)  # Triggers build

        # Verify containers and single components
        assert len(layer.se_logit_layers) == 3
        assert len(layer.conv_layers) == 3
        assert len(layer.bn_layers) == 3
        assert layer.final_se is not None

        # Verify built status
        for i in range(3):
            assert layer.se_logit_layers[i].built
            assert layer.conv_layers[i].built
            assert layer.bn_layers[i].built
        assert layer.final_se.built

    def test_serialization_cycle(self, default_config, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = TripSE4(**default_config)(inputs)
        model = keras.Model(inputs, outputs)
        original_output = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-5, atol=1e-5
        )


class TestTripSEComparison:
    """Comparative tests across all TripSE variants."""

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor."""
        return np.random.randn(2, 32, 32, 64).astype(np.float32)

    @pytest.fixture
    def common_config(self):
        """Common configuration for all variants."""
        return {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
        }

    @pytest.mark.parametrize("layer_class", [TripSE1, TripSE2, TripSE3, TripSE4])
    def test_all_variants_same_output_shape(self, layer_class, common_config, sample_input):
        layer = layer_class(**common_config)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_variants_produce_different_outputs(self, common_config, sample_input):
        """Test that different variants produce different outputs."""
        np.random.seed(42)
        
        # Create models
        layer1 = TripSE1(**common_config)
        layer2 = TripSE2(**common_config)
        layer3 = TripSE3(**common_config)
        layer4 = TripSE4(**common_config)

        # Build them with the same input to initialize weights
        # Note: Weights will be initialized randomly, so strict comparison 
        # is tricky, but structurally they should compute different things.
        
        out1 = layer1(sample_input)
        out2 = layer2(sample_input)
        out3 = layer3(sample_input)
        out4 = layer4(sample_input)

        # Calculate mean absolute differences
        diff_1_2 = np.mean(np.abs(out1 - out2))
        diff_1_3 = np.mean(np.abs(out1 - out3))
        diff_1_4 = np.mean(np.abs(out1 - out4))

        assert (diff_1_2 > 1e-6 or diff_1_3 > 1e-6 or diff_1_4 > 1e-6)

    @pytest.mark.parametrize("layer_class", [TripSE1, TripSE2, TripSE3, TripSE4])
    def test_gradients_flow(self, layer_class, common_config, sample_input):
        """Test that gradients flow through all variants using TF backend."""
        layer = layer_class(**common_config)
        
        # Ensure inputs are tensors for gradient tape
        x_tensor = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            output = layer(x_tensor, training=True)
            loss = keras.ops.mean(output)

        gradients = tape.gradient(loss, layer.trainable_weights)

        # All gradients should be non-None and have values
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        
        # Check that at least some gradients are non-zero (structural connectivity)
        has_grad_flow = any(tf.reduce_sum(tf.abs(g)) > 0 for g in gradients)
        assert has_grad_flow, f"No gradient flow detected for {layer_class.__name__}"


class TestTripSEIntegration:
    """Integration tests with full models."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for integration tests."""
        x = np.random.randn(4, 32, 32, 64).astype(np.float32)
        y = np.random.randint(0, 10, size=(4,))
        return x, y

    @pytest.mark.parametrize("layer_class", [TripSE1, TripSE2, TripSE3, TripSE4])
    def test_in_cnn_model(self, layer_class, sample_data):
        """Test TripSE layers in a full CNN model."""
        x_train, y_train = sample_data

        inputs = keras.Input(shape=(32, 32, 64))
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = layer_class(reduction_ratio=0.0625)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        history = model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)
        assert 'loss' in history.history

    @pytest.mark.parametrize("layer_class", [TripSE1, TripSE2, TripSE3, TripSE4])
    def test_multiple_instances_in_model(self, layer_class, sample_data):
        """Test multiple TripSE instances in the same model."""
        x_train, y_train = sample_data

        inputs = keras.Input(shape=(32, 32, 64))
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = layer_class(reduction_ratio=0.0625, name='tripse_1')(x)
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layer_class(reduction_ratio=0.0625, name='tripse_2')(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        history = model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)
        assert 'loss' in history.history

    @pytest.mark.parametrize("layer_class", [TripSE1, TripSE2, TripSE3, TripSE4])
    def test_model_save_load(self, layer_class, sample_data):
        """Test full model with TripSE can be saved and loaded."""
        x_train, y_train = sample_data

        inputs = keras.Input(shape=(32, 32, 64))
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = layer_class(reduction_ratio=0.0625)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.fit(x_train, y_train, batch_size=2, epochs=1, verbose=0)

        original_pred = model.predict(x_train[:2], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'full_model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_pred = loaded_model.predict(x_train[:2], verbose=0)

        np.testing.assert_allclose(
            original_pred,
            loaded_pred,
            rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after model save/load"
        )

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
