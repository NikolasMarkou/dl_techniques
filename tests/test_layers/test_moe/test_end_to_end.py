"""
Comprehensive test suite for the MixtureOfExperts layer.

This module provides extensive testing of the MoE layer, covering initialization,
forward passes, serialization, configuration management, and gradient flow.
Follows modern Keras 3 testing patterns with particular emphasis on the critical
serialization cycle test.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any


from dl_techniques.layers.moe.layer import MixtureOfExperts, create_ffn_moe
from dl_techniques.layers.moe.config import MoEConfig, ExpertConfig, GatingConfig


# Integration test with real training scenario
class TestMoETrainingIntegration:

    def test_end_to_end_training_1d(self):
        """Test complete end-to-end MoE training."""
        # Create a simple classification model with MoE
        num_classes = 5
        input_dim = 128

        inputs = keras.Input(shape=(input_dim,))

        # Add a standard dense layer first
        x = keras.layers.Dense(256, activation='relu')(inputs)

        # MoE layer
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 512, 'output_dim': 256}
            ),
            gating_config=GatingConfig(top_k=2, aux_loss_weight=0.01)
        )
        x = MixtureOfExperts(config=moe_config)(x)

        # Classification head
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate synthetic data
        x_train = keras.random.normal(shape=(64, input_dim))
        y_train = keras.random.randint(shape=(64,), minval=0, maxval=num_classes, dtype='int32')

        x_val = keras.random.normal(shape=(32, input_dim))
        y_val = keras.random.randint(shape=(32,), minval=0, maxval=num_classes, dtype='int32')

        # Train for a few epochs
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=16,
            verbose=0
        )

        # Training should complete successfully
        assert len(history.history['loss']) == 3
        assert all(np.isfinite(loss) for loss in history.history['loss'])

        # Model should be able to predict
        predictions = model.predict(x_val, verbose=0)
        assert predictions.shape == (32, num_classes)
        assert np.allclose(np.sum(predictions, axis=1), 1.0, rtol=1e-5)  # Softmax outputs

    def test_end_to_end_training_2d(self):
        """Test complete end-to-end MoE training."""
        # Create a simple classification model with MoE
        num_classes = 5
        input_dim = 128
        input_features = 64

        inputs = keras.Input(shape=(input_dim, input_features))

        x = inputs

        # MoE layer
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 512, 'output_dim': 256}
            ),
            gating_config=GatingConfig(top_k=2, aux_loss_weight=0.01)
        )
        x = MixtureOfExperts(config=moe_config)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Classification head
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate synthetic data
        x_train = keras.random.normal(shape=(64, input_dim, input_features), dtype='float32')
        y_train = keras.random.randint(shape=(64,), minval=0, maxval=num_classes, dtype='int32')

        x_val = keras.random.normal(shape=(32, input_dim, input_features), dtype='float32')
        y_val = keras.random.randint(shape=(32,), minval=0, maxval=num_classes, dtype='int32')

        # Train for a few epochs
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=16,
            verbose=0
        )

        # Training should complete successfully
        assert len(history.history['loss']) == 3
        assert all(np.isfinite(loss) for loss in history.history['loss'])

        # Model should be able to predict
        predictions = model.predict(x_val, verbose=0)
        assert predictions.shape == (32, num_classes)
        assert np.allclose(np.sum(predictions, axis=1), 1.0, rtol=1e-5)  # Softmax outputs

    def test_end_to_end_training_3d(self):
        """Test complete end-to-end MoE training."""
        # Create a simple classification model with MoE
        num_classes = 5
        input_dim = 128
        input_spatial = 32
        input_features = 64

        inputs = keras.Input(shape=(input_dim, input_spatial, input_features))

        x = inputs

        # MoE layer
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 512, 'output_dim': 256}
            ),
            gating_config=GatingConfig(top_k=2, aux_loss_weight=0.01)
        )
        x = MixtureOfExperts(config=moe_config)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)

        # Classification head
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate synthetic data
        x_train = keras.random.normal(shape=(64, input_dim, input_spatial, input_features), dtype='float32')
        y_train = keras.random.randint(shape=(64,), minval=0, maxval=num_classes, dtype='int32')

        x_val = keras.random.normal(shape=(32, input_dim, input_spatial, input_features), dtype='float32')
        y_val = keras.random.randint(shape=(32,), minval=0, maxval=num_classes, dtype='int32')

        # Train for a few epochs
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=16,
            verbose=0
        )

        # Training should complete successfully
        assert len(history.history['loss']) == 3
        assert all(np.isfinite(loss) for loss in history.history['loss'])

        # Model should be able to predict
        predictions = model.predict(x_val, verbose=0)
        assert predictions.shape == (32, num_classes)
        assert np.allclose(np.sum(predictions, axis=1), 1.0, rtol=1e-5)  # Softmax outputs


# Run tests with: pytest test_mixture_of_experts.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])