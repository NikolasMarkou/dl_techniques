import os
import keras
import pytest
import tempfile
import numpy as np
import tensorflow as tf

from dl_techniques.layers.orthoblock import (
    OrthonomalRegularizer,
    CenteringNormalization,
    LogitNormalization,
    OrthoCenterBlock
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


class TestOrthonomalRegularizer:
    """Tests for the OrthonomalRegularizer class."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        reg = OrthonomalRegularizer(factor=0.1)
        assert reg.factor == 0.1

        reg = OrthonomalRegularizer(factor=0.5)
        assert reg.factor == 0.5

    def test_regularization_effect(self):
        """Test that regularization penalizes non-orthogonal weights."""
        reg = OrthonomalRegularizer(factor=0.1)

        # Create a perfect orthonormal matrix (identity)
        weights = np.eye(5, dtype=np.float32)
        loss = reg(tf.convert_to_tensor(weights))
        # Loss should be very close to zero for orthonormal weights
        assert abs(loss.numpy()) < 1e-5

        # Create a non-orthonormal matrix (ones)
        weights = np.ones((5, 5), dtype=np.float32)
        loss = reg(tf.convert_to_tensor(weights))
        # Loss should be substantial for non-orthonormal weights
        assert loss.numpy() > 1.0

    def test_serialization(self):
        """Test serialization and deserialization."""
        reg = OrthonomalRegularizer(factor=0.1)
        config = reg.get_config()

        # Recreate from config
        reg_new = OrthonomalRegularizer.from_config(config)
        assert reg_new.factor == reg.factor


class TestCenteringNormalization:
    """Tests for the CenteringNormalization class."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        norm = CenteringNormalization(axis=-1)
        assert norm.axis == [-1]

        norm = CenteringNormalization(axis=[1, 2])
        assert norm.axis == [1, 2]

    def test_centering_effect_2d(self, sample_data_2d):
        """Test that centering produces zero mean across feature dimension."""
        norm = CenteringNormalization(axis=-1)
        outputs = norm(tf.convert_to_tensor(sample_data_2d))

        # Check outputs have zero mean along feature dimension
        means = tf.reduce_mean(outputs, axis=-1)
        assert np.allclose(means.numpy(), 0.0, atol=1e-6)

        # Check shape is preserved
        assert outputs.shape == sample_data_2d.shape

        # Check variance is preserved (not scaled)
        input_var = np.var(sample_data_2d, axis=-1)
        output_var = np.var(outputs.numpy(), axis=-1)
        assert np.allclose(input_var, output_var, rtol=1e-5)

    def test_centering_effect_4d(self, sample_data_4d):
        """Test centering on 4D data (like CNN feature maps)."""
        norm = CenteringNormalization(axis=-1)
        outputs = norm(tf.convert_to_tensor(sample_data_4d))

        # Check outputs have zero mean across channels
        means = tf.reduce_mean(outputs, axis=-1)
        assert np.allclose(means.numpy(), 0.0, atol=1e-6)

        # Check shape is preserved
        assert outputs.shape == sample_data_4d.shape

    def test_serialization(self):
        """Test serialization and deserialization."""
        norm = CenteringNormalization(axis=-1, epsilon=1e-5)

        # Build the layer
        norm.build((None, 10))

        config = norm.get_config()
        build_config = norm.get_build_config()

        # Recreate from config
        norm_new = CenteringNormalization.from_config(config)
        norm_new.build_from_config(build_config)

        assert norm_new.axis == norm.axis
        assert norm_new.epsilon == norm.epsilon


class TestLogitNormalization:
    """Tests for the LogitNormalization class."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        norm = LogitNormalization(axis=-1)
        assert norm.axis == -1

        norm = LogitNormalization(axis=1, epsilon=1e-5)
        assert norm.axis == 1
        assert norm.epsilon == 1e-5

    def test_normalization_effect(self, sample_data_2d):
        """Test that logit normalization produces unit norm vectors."""
        norm = LogitNormalization(axis=-1)
        outputs = norm(tf.convert_to_tensor(sample_data_2d))

        # Check outputs have unit norm along feature dimension
        square_sum = tf.reduce_sum(tf.square(outputs), axis=-1)
        norms = tf.sqrt(square_sum)
        assert np.allclose(norms.numpy(), 1.0, atol=1e-6)

        # Check shape is preserved
        assert outputs.shape == sample_data_2d.shape

    def test_direction_preservation(self, sample_data_2d):
        """Test that logit normalization preserves direction but not magnitude."""
        norm = LogitNormalization(axis=-1)
        outputs = norm(tf.convert_to_tensor(sample_data_2d))

        # Normalize the input data manually for comparison
        norms = np.sqrt(np.sum(sample_data_2d ** 2, axis=-1, keepdims=True))
        expected = sample_data_2d / (norms + 1e-7)

        # Check direction is preserved (cosine similarity should be close to 1)
        similarity = np.sum(outputs.numpy() * expected, axis=-1)
        assert np.allclose(similarity, 1.0, atol=1e-5)

    def test_serialization(self):
        """Test serialization and deserialization."""
        norm = LogitNormalization(axis=-1, epsilon=1e-5)

        # Build the layer
        norm.build((None, 10))

        config = norm.get_config()
        build_config = norm.get_build_config()

        # Recreate from config
        norm_new = LogitNormalization.from_config(config)
        norm_new.build_from_config(build_config)

        assert norm_new.axis == norm.axis
        assert norm_new.epsilon == norm.epsilon


class TestOrthoCenterBlock:
    """Tests for the complete OrthoCenterBlock."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        block = OrthoCenterBlock(
            units=32,
            activation='relu',
            ortho_reg_factor=0.1
        )
        assert block.units == 32
        assert block.ortho_reg_factor == 0.1

        block = OrthoCenterBlock(
            units=64,
            activation=None,
            use_bias=False,
        )
        assert block.units == 64
        assert block.use_bias is False

    def test_sublayers_initialization(self, sample_data_2d):
        """Test that all sublayers are properly initialized after build."""
        block = OrthoCenterBlock(units=16)

        # Forward pass to build
        outputs = block(tf.convert_to_tensor(sample_data_2d))

        # Check all sublayers exist
        assert block.dense is not None
        assert block.centering is not None
        assert block.logit_norm is not None
        assert block.constrained_scale is not None

        # Check output shape
        assert outputs.shape == (sample_data_2d.shape[0], 16)

    def test_orthogonal_regularization(self, sample_data_2d):
        """Test that orthogonal regularization is applied."""
        block = OrthoCenterBlock(
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

    def test_full_pipeline(self, sample_data_2d):
        """Test the complete transformation pipeline."""
        # Create an input with known properties
        centered_input = sample_data_2d - np.mean(sample_data_2d, axis=-1, keepdims=True)

        # Create block with identity weights for testing
        block = OrthoCenterBlock(
            units=sample_data_2d.shape[1],  # Same dimensions for easier testing
            activation=None,
            use_bias=False,
            ortho_reg_factor=0.0,  # No regularization for this test
            kernel_initializer=keras.initializers.Identity(),
            scale_initial_value=1.0  # All features fully active
        )

        # Forward pass
        outputs = block(tf.convert_to_tensor(sample_data_2d))

        # With identity weights, centering should make mean zero
        output_means = tf.reduce_mean(outputs, axis=-1)
        assert np.allclose(output_means.numpy(), 0.0, atol=1e-5)

        # With identity weights and scale=1, all vectors should be unit norm
        output_norms = tf.sqrt(tf.reduce_sum(tf.square(outputs), axis=-1))
        assert np.allclose(output_norms.numpy(), 1.0, atol=1e-5)

    def test_gradient_flow(self, sample_data_2d):
        """Test gradient flow through the block."""
        block = OrthoCenterBlock(units=16)

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
        block = OrthoCenterBlock(
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
        block = OrthoCenterBlock(
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
                'OrthoCenterBlock': OrthoCenterBlock,
                'CenteringNormalization': CenteringNormalization,
                'LogitNormalization': LogitNormalization,
                'OrthonomalRegularizer': OrthonomalRegularizer
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
        block = OrthoCenterBlock(
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