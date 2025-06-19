import os
import keras
import pytest
import tempfile
import numpy as np
import tensorflow as tf

from dl_techniques.layers.orthoblock import (
    OrthoBlock
)


# Setup fixtures for common test data
@pytest.fixture
def sample_data_2d():
    """Create sample 2D data for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.normal(size=(4, 10)).astype(np.float32)


@pytest.fixture
def sample_data_4d():
    """Create sample 4D data for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.normal(size=(4, 8, 8, 3)).astype(np.float32)


class TestOrthoCenterBlock:
    """Tests for the complete OrthoCenterBlock."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        block = OrthoBlock(
            units=32,
            activation='relu',
            ortho_reg_factor=0.1
        )
        assert block.units == 32
        assert block.ortho_reg_factor == 0.1

        block = OrthoBlock(
            units=64,
            activation=None,
            use_bias=False,
        )
        assert block.units == 64
        assert block.use_bias is False

    def test_orthogonal_regularization(self, sample_data_2d):
        """Test that orthogonal regularization is applied."""
        block = OrthoBlock(
            units=16,
            ortho_reg_factor=0.1
        )

        # Create a model with the block to access regularization losses
        inputs = keras.Input(shape=(sample_data_2d.shape[1],))
        outputs = block(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Forward pass to ensure regularization is applied
        _ = model(tf.convert_to_tensor(sample_data_2d))

        # Check regularization losses exist
        assert len(model.losses) > 0

    def test_gradient_flow(self, sample_data_2d):
        """Test gradient flow through the block."""
        block = OrthoBlock(units=16)

        with tf.GradientTape() as tape:
            inputs = tf.Variable(sample_data_2d)
            outputs = block(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients for all trainable variables
        trainable_vars = block.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Check that all gradients exist and are finite
        for grad in gradients:
            assert grad is not None
            assert not np.any(np.isnan(grad.numpy()))
            assert not np.any(np.isinf(grad.numpy()))

    def test_model_integration(self, sample_data_2d):
        """Test that the block can be used in a real model that trains."""
        # Create a simple classification problem
        num_classes = 3
        y_dummy = np.random.randint(0, num_classes, size=(sample_data_2d.shape[0],))

        # Create model with our block
        block = OrthoBlock(
            units=16,
            activation='relu',
            ortho_reg_factor=0.1
        )

        inputs = keras.Input(shape=(sample_data_2d.shape[1],))
        x = block(inputs)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train for a couple of epochs
        history = model.fit(
            sample_data_2d, y_dummy,
            epochs=2,
            verbose=0
        )

        # Check that training occurred (loss should decrease)
        assert history.history['loss'][0] >= history.history['loss'][1]

    def test_serialization_and_save_load(self, sample_data_2d):
        """Test serialization and model saving/loading."""
        block = OrthoBlock(
            units=16,
            activation='relu',
            ortho_reg_factor=0.1
        )

        # Create and compile model
        inputs = keras.Input(shape=(sample_data_2d.shape[1],))
        x = block(inputs)
        outputs = keras.layers.Dense(3, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy'
        )

        # Get predictions before saving
        predictions = model.predict(sample_data_2d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            # Custom objects dict for loading
            custom_objects = {
                'OrthoCenterBlock': OrthoBlock
            }

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )

            # Check predictions match
            loaded_predictions = loaded_model.predict(sample_data_2d)
            assert np.allclose(predictions, loaded_predictions, rtol=1e-5)

    def test_scheduled_regularization(self, sample_data_2d):
        """Test the regularization scheduler."""
        block = OrthoBlock(
            units=16,
            ortho_reg_factor=0.1
        )

        inputs = keras.Input(shape=(sample_data_2d.shape[1],))
        x = block(inputs)
        outputs = keras.layers.Dense(3)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Create dummy labels
        y_dummy = np.random.randint(0, 3, size=(sample_data_2d.shape[0],))

        # Create a simple scheduler to test callback mechanism
        class TestScheduler(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.called = False

            def on_epoch_begin(self, epoch, logs=None):
                self.called = True
                # In real code, you'd dynamically adjust the regularization factor

        scheduler = TestScheduler()

        # Compile and fit for one epoch
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.fit(
            sample_data_2d, y_dummy,
            epochs=1,
            callbacks=[scheduler],
            verbose=0
        )

        # Verify the scheduler was called
        assert scheduler.called


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__])