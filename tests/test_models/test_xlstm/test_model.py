"""
Comprehensive tests for xLSTM Model.

This module provides thorough testing for the complete xLSTM model, which is a
keras.Model subclass that stacks sLSTM and mLSTM blocks for sequence modeling.

Tests cover:
- Model initialization and configuration
- Forward pass and shape validation
- Serialization and deserialization (CRITICAL)
- Compilation and training
- Prediction capabilities
- Different architectural configurations
- Edge cases and error handling
- Integration with Keras training loop
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any, List


from dl_techniques.models.xlstm.model import xLSTM


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def small_vocab_size() -> int:
    """Small vocabulary for fast testing."""
    return 100


@pytest.fixture
def sample_vocab_size() -> int:
    """Standard vocabulary size for tests."""
    return 1000


@pytest.fixture
def sample_embed_dim() -> int:
    """Standard embedding dimension."""
    return 64


@pytest.fixture
def sample_num_layers() -> int:
    """Standard number of layers."""
    return 4


@pytest.fixture
def sample_batch_size() -> int:
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def sample_sequence_length() -> int:
    """Standard sequence length for tests."""
    return 16


# =============================================================================
# xLSTM Model Tests
# =============================================================================

class TestXLSTMModel:
    """Comprehensive test suite for xLSTM Model."""

    @pytest.fixture
    def minimal_config(self, small_vocab_size) -> Dict[str, Any]:
        """Minimal configuration for quick tests."""
        return {
            'vocab_size': small_vocab_size,
            'embed_dim': 32,
            'num_layers': 2,
            'mlstm_ratio': 0.5,
        }

    @pytest.fixture
    def standard_config(self, sample_vocab_size, sample_embed_dim, sample_num_layers) -> Dict[str, Any]:
        """Standard configuration for comprehensive tests."""
        return {
            'vocab_size': sample_vocab_size,
            'embed_dim': sample_embed_dim,
            'num_layers': sample_num_layers,
            'mlstm_ratio': 0.5,
            'mlstm_num_heads': 4,
            'mlstm_expansion_factor': 2,
            'slstm_forget_gate': 'sigmoid',
            'ffn_type': 'swiglu',
            'ffn_expansion_factor': 2,
            'normalization_type': 'layer_norm',
            'dropout_rate': 0.1,
            'embedding_dropout_rate': 0.1,
        }

    @pytest.fixture
    def all_mlstm_config(self, sample_vocab_size, sample_embed_dim) -> Dict[str, Any]:
        """Configuration with all mLSTM blocks."""
        return {
            'vocab_size': sample_vocab_size,
            'embed_dim': sample_embed_dim,
            'num_layers': 4,
            'mlstm_ratio': 1.0,  # All mLSTM
            'mlstm_num_heads': 4,
        }

    @pytest.fixture
    def all_slstm_config(self, sample_vocab_size, sample_embed_dim) -> Dict[str, Any]:
        """Configuration with all sLSTM blocks."""
        return {
            'vocab_size': sample_vocab_size,
            'embed_dim': sample_embed_dim,
            'num_layers': 4,
            'mlstm_ratio': 0.0,  # All sLSTM
            'ffn_type': 'swiglu',
        }

    @pytest.fixture
    def sample_token_input(self, sample_batch_size, sample_sequence_length, sample_vocab_size) -> keras.KerasTensor:
        """Sample token input for model."""
        return keras.random.randint(
            shape=(sample_batch_size, sample_sequence_length),
            minval=0,
            maxval=sample_vocab_size
        )

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization_minimal(self, minimal_config):
        """Test model initialization with minimal configuration."""
        

        model = xLSTM(**minimal_config)

        assert model.vocab_size == minimal_config['vocab_size']
        assert model.embed_dim == minimal_config['embed_dim']
        assert model.num_layers == minimal_config['num_layers']
        assert model.mlstm_ratio == minimal_config['mlstm_ratio']

        # Check sub-layers were created
        assert model.embedding is not None
        assert len(model.blocks) == minimal_config['num_layers']
        assert model.final_norm is not None
        assert model.output_head is not None

    def test_initialization_standard(self, standard_config):
        """Test model initialization with standard configuration."""
        

        model = xLSTM(**standard_config)

        # Verify all configuration parameters
        assert model.vocab_size == standard_config['vocab_size']
        assert model.embed_dim == standard_config['embed_dim']
        assert model.num_layers == standard_config['num_layers']
        assert model.mlstm_num_heads == standard_config['mlstm_num_heads']
        assert model.dropout_rate == standard_config['dropout_rate']

        # Check embedding dropout was created
        assert model.embedding_dropout is not None

    def test_initialization_all_mlstm(self, all_mlstm_config):
        """Test model with all mLSTM blocks."""
        
        from dl_techniques.layers.time_series.xlstm_blocks import mLSTMBlock

        model = xLSTM(**all_mlstm_config)

        # All blocks should be mLSTM
        assert len(model.blocks) == all_mlstm_config['num_layers']
        assert all(isinstance(block, mLSTMBlock) for block in model.blocks)

    def test_initialization_all_slstm(self, all_slstm_config):
        """Test model with all sLSTM blocks."""
        
        from dl_techniques.layers.time_series.xlstm_blocks import sLSTMBlock

        model = xLSTM(**all_slstm_config)

        # All blocks should be sLSTM
        assert len(model.blocks) == all_slstm_config['num_layers']
        assert all(isinstance(block, sLSTMBlock) for block in model.blocks)

    def test_block_distribution(self, standard_config):
        """Test correct distribution of mLSTM and sLSTM blocks."""
        
        from dl_techniques.layers.time_series.xlstm_blocks import mLSTMBlock, sLSTMBlock

        model = xLSTM(**standard_config)

        num_mlstm = int(standard_config['num_layers'] * standard_config['mlstm_ratio'])
        num_slstm = standard_config['num_layers'] - num_mlstm

        mlstm_count = sum(isinstance(block, mLSTMBlock) for block in model.blocks)
        slstm_count = sum(isinstance(block, sLSTMBlock) for block in model.blocks)

        assert mlstm_count == num_mlstm
        assert slstm_count == num_slstm

    def test_initialization_no_embedding_dropout(self, minimal_config):
        """Test model without embedding dropout."""
        

        config = minimal_config.copy()
        config['embedding_dropout_rate'] = 0.0

        model = xLSTM(**config)

        assert model.embedding_dropout is None

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_invalid_vocab_size(self, minimal_config):
        """Test error handling for invalid vocab_size."""
        

        config = minimal_config.copy()
        config['vocab_size'] = 0

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            xLSTM(**config)

        config['vocab_size'] = -100
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            xLSTM(**config)

    def test_invalid_embed_dim(self, minimal_config):
        """Test error handling for invalid embed_dim."""
        

        config = minimal_config.copy()
        config['embed_dim'] = 0

        with pytest.raises(ValueError, match="embed_dim must be positive"):
            xLSTM(**config)

    def test_invalid_num_layers(self, minimal_config):
        """Test error handling for invalid num_layers."""
        

        config = minimal_config.copy()
        config['num_layers'] = 0

        with pytest.raises(ValueError, match="num_layers must be positive"):
            xLSTM(**config)

    def test_invalid_mlstm_ratio(self, minimal_config):
        """Test error handling for invalid mlstm_ratio."""
        

        config = minimal_config.copy()

        # Test ratio > 1
        config['mlstm_ratio'] = 1.5
        with pytest.raises(ValueError, match="mlstm_ratio must be in"):
            xLSTM(**config)

        # Test ratio < 0
        config['mlstm_ratio'] = -0.5
        with pytest.raises(ValueError, match="mlstm_ratio must be in"):
            xLSTM(**config)

    # -------------------------------------------------------------------------
    # Forward Pass Tests
    # -------------------------------------------------------------------------

    def test_forward_pass_inference(self, minimal_config, sample_batch_size):
        """Test forward pass in inference mode."""
        

        model = xLSTM(**minimal_config)

        seq_len = 10
        sample_input = keras.random.randint(
            shape=(sample_batch_size, seq_len),
            minval=0,
            maxval=minimal_config['vocab_size']
        )

        output = model(sample_input, training=False)

        # Verify output shape: (batch_size, seq_len, vocab_size)
        assert output.shape == (
            sample_batch_size,
            seq_len,
            minimal_config['vocab_size']
        )

    def test_forward_pass_training(self, minimal_config, sample_batch_size):
        """Test forward pass in training mode."""
        

        model = xLSTM(**minimal_config)

        seq_len = 10
        sample_input = keras.random.randint(
            shape=(sample_batch_size, seq_len),
            minval=0,
            maxval=minimal_config['vocab_size']
        )

        output = model(sample_input, training=True)

        # Verify output shape
        assert output.shape == (
            sample_batch_size,
            seq_len,
            minimal_config['vocab_size']
        )

    def test_different_sequence_lengths(self, minimal_config, sample_batch_size):
        """Test model with different sequence lengths."""
        

        model = xLSTM(**minimal_config)

        for seq_len in [5, 10, 20, 50]:
            sample_input = keras.random.randint(
                shape=(sample_batch_size, seq_len),
                minval=0,
                maxval=minimal_config['vocab_size']
            )

            output = model(sample_input, training=False)

            assert output.shape == (
                sample_batch_size,
                seq_len,
                minimal_config['vocab_size']
            )

    def test_different_batch_sizes(self, minimal_config):
        """Test model with different batch sizes."""
        

        model = xLSTM(**minimal_config)
        seq_len = 10

        for batch_size in [1, 2, 4, 8]:
            sample_input = keras.random.randint(
                shape=(batch_size, seq_len),
                minval=0,
                maxval=minimal_config['vocab_size']
            )

            output = model(sample_input, training=False)

            assert output.shape == (
                batch_size,
                seq_len,
                minimal_config['vocab_size']
            )

    # -------------------------------------------------------------------------
    # Serialization Tests (CRITICAL)
    # -------------------------------------------------------------------------

    def test_serialization_cycle_minimal(self, minimal_config, sample_batch_size):
        """CRITICAL TEST: Full serialization cycle with minimal config."""
        

        # Create model
        model = xLSTM(**minimal_config)

        # Sample input
        seq_len = 10
        sample_input = keras.random.randint(
            shape=(sample_batch_size, seq_len),
            minval=0,
            maxval=minimal_config['vocab_size']
        )

        # Get original prediction
        original_pred = model(sample_input, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_xlstm.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input, training=False)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_standard(self, standard_config, sample_batch_size):
        """CRITICAL TEST: Full serialization cycle with standard config."""
        

        # Create model
        model = xLSTM(**standard_config)

        # Sample input
        seq_len = 12
        sample_input = keras.random.randint(
            shape=(sample_batch_size, seq_len),
            minval=0,
            maxval=standard_config['vocab_size']
        )

        # Get original prediction
        original_pred = model(sample_input, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_xlstm_standard.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input, training=False)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_after_training(self, minimal_config, sample_batch_size):
        """Test serialization after training (weights loaded correctly)."""
        

        # Create and compile model
        model = xLSTM(**minimal_config)
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Generate dummy training data
        seq_len = 10
        num_samples = 32
        x_train = np.random.randint(
            0, minimal_config['vocab_size'],
            (num_samples, seq_len)
        )
        y_train = np.random.randint(
            0, minimal_config['vocab_size'],
            (num_samples, seq_len)
        )

        # Train for a few steps
        model.fit(x_train, y_train, epochs=2, batch_size=8, verbose=0)

        # Get prediction after training
        sample_input = keras.random.randint(
            shape=(sample_batch_size, seq_len),
            minval=0,
            maxval=minimal_config['vocab_size']
        )
        trained_pred = model(sample_input, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'trained_xlstm.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input, training=False)

            # Verify identical predictions (weights preserved)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(trained_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Trained weights not preserved after serialization"
            )

    # -------------------------------------------------------------------------
    # Configuration Tests
    # -------------------------------------------------------------------------

    def test_config_completeness_minimal(self, minimal_config):
        """Test that get_config contains all __init__ parameters."""
        

        model = xLSTM(**minimal_config)
        config = model.get_config()

        # Check all provided config parameters are present
        for key in minimal_config:
            assert key in config, f"Missing {key} in get_config()"

    def test_config_completeness_standard(self, standard_config):
        """Test config completeness with standard configuration."""
        

        model = xLSTM(**standard_config)
        config = model.get_config()

        # Check all config parameters are present
        expected_keys = [
            'vocab_size', 'embed_dim', 'num_layers', 'mlstm_ratio',
            'mlstm_num_heads', 'mlstm_expansion_factor', 'slstm_forget_gate',
            'ffn_type', 'ffn_expansion_factor', 'normalization_type',
            'dropout_rate', 'embedding_dropout_rate',
            'kernel_initializer', 'recurrent_initializer', 'bias_initializer',
            'kernel_regularizer', 'recurrent_regularizer', 'bias_regularizer'
        ]

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_from_config(self, standard_config):
        """Test reconstruction from config."""
        

        # Create original model
        model = xLSTM(**standard_config)

        # Get config
        config = model.get_config()

        # Reconstruct from config
        reconstructed_model = xLSTM.from_config(config)

        # Verify same configuration
        assert reconstructed_model.vocab_size == model.vocab_size
        assert reconstructed_model.embed_dim == model.embed_dim
        assert reconstructed_model.num_layers == model.num_layers
        assert reconstructed_model.mlstm_ratio == model.mlstm_ratio

    # -------------------------------------------------------------------------
    # Compilation Tests
    # -------------------------------------------------------------------------

    def test_compile_basic(self, minimal_config):
        """Test model compilation with basic settings."""
        

        model = xLSTM(**minimal_config)

        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        assert model.optimizer is not None
        assert model.loss is not None

    def test_compile_with_custom_optimizer(self, minimal_config):
        """Test compilation with custom optimizer."""
        

        model = xLSTM(**minimal_config)

        optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        assert isinstance(model.optimizer, keras.optimizers.AdamW)

    # -------------------------------------------------------------------------
    # Training Tests
    # -------------------------------------------------------------------------

    def test_fit_basic(self, minimal_config):
        """Test basic model training."""
        

        model = xLSTM(**minimal_config)
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Generate dummy data
        seq_len = 10
        num_samples = 32
        x_train = np.random.randint(
            0, minimal_config['vocab_size'],
            (num_samples, seq_len)
        )
        y_train = np.random.randint(
            0, minimal_config['vocab_size'],
            (num_samples, seq_len)
        )

        # Train
        history = model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=8,
            verbose=0
        )

        assert 'loss' in history.history
        assert len(history.history['loss']) == 2

    def test_fit_with_validation(self, minimal_config):
        """Test training with validation data."""
        

        model = xLSTM(**minimal_config)
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Generate dummy data
        seq_len = 10
        num_samples = 32
        x_train = np.random.randint(
            0, minimal_config['vocab_size'],
            (num_samples, seq_len)
        )
        y_train = np.random.randint(
            0, minimal_config['vocab_size'],
            (num_samples, seq_len)
        )

        x_val = np.random.randint(
            0, minimal_config['vocab_size'],
            (16, seq_len)
        )
        y_val = np.random.randint(
            0, minimal_config['vocab_size'],
            (16, seq_len)
        )

        # Train with validation
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=8,
            verbose=0
        )

        assert 'val_loss' in history.history
        assert len(history.history['val_loss']) == 2

    def test_gradients_flow(self, minimal_config, sample_batch_size):
        """Test that gradients flow through the model."""
        

        model = xLSTM(**minimal_config)

        seq_len = 10
        sample_input = keras.random.randint(
            shape=(sample_batch_size, seq_len),
            minval=0,
            maxval=minimal_config['vocab_size']
        )

        with tf.GradientTape() as tape:
            output = model(sample_input, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    # -------------------------------------------------------------------------
    # Prediction Tests
    # -------------------------------------------------------------------------

    def test_predict_basic(self, minimal_config):
        """Test model prediction."""
        

        model = xLSTM(**minimal_config)

        seq_len = 10
        num_samples = 16
        x_test = np.random.randint(
            0, minimal_config['vocab_size'],
            (num_samples, seq_len)
        )

        predictions = model.predict(x_test, verbose=0)

        # Verify prediction shape
        assert predictions.shape == (
            num_samples,
            seq_len,
            minimal_config['vocab_size']
        )

    def test_predict_single_sample(self, minimal_config):
        """Test prediction on single sample."""
        

        model = xLSTM(**minimal_config)

        seq_len = 10
        x_test = np.random.randint(
            0, minimal_config['vocab_size'],
            (1, seq_len)
        )

        predictions = model.predict(x_test, verbose=0)

        assert predictions.shape == (1, seq_len, minimal_config['vocab_size'])

    # -------------------------------------------------------------------------
    # Architecture Configuration Tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mlstm_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_different_mlstm_ratios(self, minimal_config, mlstm_ratio):
        """Test model with different mLSTM/sLSTM ratios."""
        

        config = minimal_config.copy()
        config['mlstm_ratio'] = mlstm_ratio

        model = xLSTM(**config)

        # Test forward pass
        seq_len = 10
        batch_size = 4
        sample_input = keras.random.randint(
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config['vocab_size']
        )

        output = model(sample_input, training=False)
        assert output.shape == (batch_size, seq_len, config['vocab_size'])

    @pytest.mark.parametrize("num_layers", [1, 2, 4, 6, 8])
    def test_different_num_layers(self, minimal_config, num_layers):
        """Test model with different numbers of layers."""
        

        config = minimal_config.copy()
        config['num_layers'] = num_layers

        model = xLSTM(**config)

        assert len(model.blocks) == num_layers

        # Test forward pass
        seq_len = 10
        batch_size = 4
        sample_input = keras.random.randint(
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config['vocab_size']
        )

        output = model(sample_input, training=False)
        assert output.shape == (batch_size, seq_len, config['vocab_size'])

    @pytest.mark.parametrize("embed_dim", [32, 64, 128, 256])
    def test_different_embed_dims(self, minimal_config, embed_dim):
        """Test model with different embedding dimensions."""
        

        config = minimal_config.copy()
        config['embed_dim'] = embed_dim

        model = xLSTM(**config)

        assert model.embed_dim == embed_dim

        # Test forward pass
        seq_len = 10
        batch_size = 4
        sample_input = keras.random.randint(
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config['vocab_size']
        )

        output = model(sample_input, training=False)
        assert output.shape == (batch_size, seq_len, config['vocab_size'])

    @pytest.mark.parametrize("normalization_type", ['layer_norm', 'rms_norm'])
    def test_different_normalization_types(self, minimal_config, normalization_type):
        """Test model with different normalization types."""
        

        config = minimal_config.copy()
        config['normalization_type'] = normalization_type

        model = xLSTM(**config)

        # Test forward pass
        seq_len = 10
        batch_size = 4
        sample_input = keras.random.randint(
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config['vocab_size']
        )

        output = model(sample_input, training=False)
        assert output.shape == (batch_size, seq_len, config['vocab_size'])

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_single_layer_model(self, minimal_config):
        """Test model with single layer."""
        

        config = minimal_config.copy()
        config['num_layers'] = 1
        config['mlstm_ratio'] = 0.0  # Single sLSTM layer

        model = xLSTM(**config)

        assert len(model.blocks) == 1

        # Test forward pass
        seq_len = 10
        batch_size = 4
        sample_input = keras.random.randint(
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config['vocab_size']
        )

        output = model(sample_input, training=False)
        assert output.shape == (batch_size, seq_len, config['vocab_size'])

    def test_very_small_vocab(self, minimal_config):
        """Test model with very small vocabulary."""
        

        config = minimal_config.copy()
        config['vocab_size'] = 10

        model = xLSTM(**config)

        # Test forward pass
        seq_len = 5
        batch_size = 2
        sample_input = keras.random.randint(
            shape=(batch_size, seq_len),
            minval=0,
            maxval=10
        )

        output = model(sample_input, training=False)
        assert output.shape == (batch_size, seq_len, 10)

    def test_no_dropout(self, minimal_config):
        """Test model with no dropout."""
        

        config = minimal_config.copy()
        config['dropout_rate'] = 0.0
        config['embedding_dropout_rate'] = 0.0

        model = xLSTM(**config)

        assert model.dropout_rate == 0.0
        assert model.embedding_dropout is None

        # Test forward pass
        seq_len = 10
        batch_size = 4
        sample_input = keras.random.randint(
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config['vocab_size']
        )

        output = model(sample_input, training=False)
        assert output.shape == (batch_size, seq_len, config['vocab_size'])


# =============================================================================
# Integration Tests
# =============================================================================

class TestXLSTMIntegration:
    """Integration tests for xLSTM model."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        

        # Create model
        model = xLSTM(
            vocab_size=100,
            embed_dim=32,
            num_layers=2,
            mlstm_ratio=0.5,
        )

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Generate data
        seq_len = 10
        num_samples = 64
        x_train = np.random.randint(0, 100, (num_samples, seq_len))
        y_train = np.random.randint(0, 100, (num_samples, seq_len))

        x_val = np.random.randint(0, 100, (16, seq_len))
        y_val = np.random.randint(0, 100, (16, seq_len))

        # Train
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=16,
            verbose=0
        )

        # Evaluate
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)

        assert loss is not None
        assert accuracy is not None

        # Predict
        x_test = np.random.randint(0, 100, (8, seq_len))
        predictions = model.predict(x_test, verbose=0)

        assert predictions.shape == (8, seq_len, 100)

    def test_save_load_and_continue_training(self):
        """Test saving, loading, and continuing training."""
        

        # Create and train initial model
        model = xLSTM(
            vocab_size=100,
            embed_dim=32,
            num_layers=2,
            mlstm_ratio=0.5,
        )

        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        # Initial training
        seq_len = 10
        x_train = np.random.randint(0, 100, (32, seq_len))
        y_train = np.random.randint(0, 100, (32, seq_len))

        model.fit(x_train, y_train, epochs=2, verbose=0)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'checkpoint.keras')
            model.save(filepath)

            # Load model
            loaded_model = keras.models.load_model(filepath)

            # Continue training
            loaded_model.compile(
                optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            )

            history = loaded_model.fit(x_train, y_train, epochs=2, verbose=0)

            assert 'loss' in history.history
            assert len(history.history['loss']) == 2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])