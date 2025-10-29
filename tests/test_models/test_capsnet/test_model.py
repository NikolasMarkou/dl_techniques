"""
Comprehensive test suite for the CapsNet model.

This module contains all tests for the CapsNet model implementation,
covering initialization, forward pass, training, serialization,
model integration, metrics, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple

from dl_techniques.models.capsnet.model import CapsNet, CapsuleAccuracy, create_capsnet
from dl_techniques.layers.capsules import PrimaryCapsule, RoutingCapsule
from dl_techniques.losses.capsule_margin_loss import capsule_margin_loss
from dl_techniques.utils.tensors import length


class TestCapsuleAccuracy:
    """Test suite for CapsuleAccuracy metric."""

    @pytest.fixture
    def metric(self):
        """Create CapsuleAccuracy metric instance."""
        return CapsuleAccuracy()

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions (capsule lengths)."""
        batch_size = 8
        num_classes = 10
        return tf.random.uniform([batch_size, num_classes], 0, 1)

    @pytest.fixture
    def sample_labels(self):
        """Create sample one-hot labels."""
        batch_size = 8
        num_classes = 10
        labels = tf.random.uniform([batch_size], 0, num_classes, dtype=tf.int32)
        return tf.one_hot(labels, num_classes)

    def test_metric_initialization(self, metric):
        """Test metric initialization."""
        assert metric.name == "capsule_accuracy"
        assert metric.total.numpy() == 0.0
        assert metric.count.numpy() == 0.0

    def test_metric_update_with_lengths(self, metric, sample_predictions, sample_labels):
        """Test metric update with capsule lengths."""
        metric.update_state(sample_labels, sample_predictions)

        result = metric.result()
        assert 0.0 <= result.numpy() <= 1.0
        assert metric.count.numpy() == sample_labels.shape[0]

    def test_metric_update_with_dict(self, metric, sample_predictions, sample_labels):
        """Test metric update with dictionary containing lengths."""
        pred_dict = {"length": sample_predictions, "other": tf.zeros_like(sample_predictions)}
        metric.update_state(sample_labels, pred_dict)

        result = metric.result()
        assert 0.0 <= result.numpy() <= 1.0

    def test_metric_reset(self, metric, sample_predictions, sample_labels):
        """Test metric reset functionality."""
        metric.update_state(sample_labels, sample_predictions)
        initial_result = metric.result()

        metric.reset_state()
        assert metric.total.numpy() == 0.0
        assert metric.count.numpy() == 0.0
        assert metric.result().numpy() == 0.0

    def test_metric_accuracy_calculation(self, metric):
        """Test accuracy calculation with known values."""
        # Create perfect predictions
        labels = tf.one_hot([0, 1, 2], 3)
        predictions = tf.constant([[0.9, 0.1, 0.1],
                                  [0.1, 0.9, 0.1],
                                  [0.1, 0.1, 0.9]])

        metric.update_state(labels, predictions)
        result = metric.result()
        assert result.numpy() == 1.0  # Perfect accuracy

        # Reset and test with wrong predictions
        metric.reset_state()
        wrong_predictions = tf.constant([[0.1, 0.9, 0.1],  # Wrong
                                        [0.9, 0.1, 0.1],   # Wrong
                                        [0.1, 0.1, 0.9]])  # Correct

        metric.update_state(labels, wrong_predictions)
        result = metric.result()
        assert abs(result.numpy() - 1/3) < 1e-6  # 1 out of 3 correct


class TestCapsNet:
    """Test suite for CapsNet model implementation."""

    @pytest.fixture
    def mnist_input_shape(self) -> Tuple[int, int, int]:
        """Create MNIST input shape."""
        return (28, 28, 1)

    @pytest.fixture
    def cifar_input_shape(self) -> Tuple[int, int, int]:
        """Create CIFAR-10 input shape."""
        return (32, 32, 3)

    @pytest.fixture
    def num_classes(self) -> int:
        """Number of classes for testing."""
        return 10

    @pytest.fixture
    def sample_mnist_data(self, mnist_input_shape):
        """Create sample MNIST-like data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(mnist_input_shape), 0, 1)

    @pytest.fixture
    def sample_cifar_data(self, cifar_input_shape):
        """Create sample CIFAR-like data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(cifar_input_shape), 0, 1)

    @pytest.fixture
    def sample_labels(self, num_classes):
        """Create sample one-hot labels."""
        batch_size = 4
        labels = tf.random.uniform([batch_size], 0, num_classes, dtype=tf.int32)
        return tf.one_hot(labels, num_classes)

    def test_initialization_defaults(self, num_classes):
        """Test initialization with default parameters."""
        capsnet = CapsNet(num_classes=num_classes)

        assert capsnet.num_classes == num_classes
        assert capsnet.routing_iterations == 3
        assert capsnet.conv_filters == [256, 256]
        assert capsnet.primary_capsules == 32
        assert capsnet.primary_capsule_dim == 8
        assert capsnet.digit_capsule_dim == 16
        assert capsnet.reconstruction is True
        assert capsnet.use_batch_norm is True
        assert capsnet.positive_margin == 0.9
        assert capsnet.negative_margin == 0.1
        assert capsnet.downweight == 0.5
        assert capsnet.reconstruction_weight == 0.01
        assert capsnet._layers_built is False

    def test_initialization_custom(self, num_classes, mnist_input_shape):
        """Test initialization with custom parameters."""
        capsnet = CapsNet(
            num_classes=num_classes,
            routing_iterations=5,
            conv_filters=[128, 256],
            primary_capsules=16,
            primary_capsule_dim=4,
            digit_capsule_dim=8,
            reconstruction=False,
            input_shape=mnist_input_shape,
            decoder_architecture=[256, 512],
            use_batch_norm=False,
            positive_margin=0.95,
            negative_margin=0.05,
            downweight=0.3,
            reconstruction_weight=0.005,
            name="custom_capsnet"
        )

        assert capsnet.num_classes == num_classes
        assert capsnet.routing_iterations == 5
        assert capsnet.conv_filters == [128, 256]
        assert capsnet.primary_capsules == 16
        assert capsnet.primary_capsule_dim == 4
        assert capsnet.digit_capsule_dim == 8
        assert capsnet.reconstruction is False
        assert capsnet._input_shape == mnist_input_shape
        assert capsnet.use_batch_norm is False
        assert capsnet.positive_margin == 0.95
        assert capsnet.negative_margin == 0.05
        assert capsnet.downweight == 0.3
        assert capsnet.reconstruction_weight == 0.005
        assert capsnet.name == "custom_capsnet"
        # Model layers are built only when first called, not during initialization
        assert capsnet._layers_built is False

    def test_initialization_with_regularization(self, num_classes):
        """Test initialization with regularization."""
        capsnet = CapsNet(
            num_classes=num_classes,
            kernel_regularizer="l2"
        )

        assert capsnet.kernel_regularizer is not None
        assert isinstance(capsnet.kernel_regularizer, keras.regularizers.L2)

    def test_parameter_validation(self):
        """Test parameter validation with invalid inputs."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            CapsNet(num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            CapsNet(num_classes=-5)

        with pytest.raises(ValueError, match="routing_iterations must be positive"):
            CapsNet(num_classes=10, routing_iterations=0)

        with pytest.raises(ValueError, match="primary_capsules must be positive"):
            CapsNet(num_classes=10, primary_capsules=-1)

        with pytest.raises(ValueError, match="primary_capsule_dim must be positive"):
            CapsNet(num_classes=10, primary_capsule_dim=0)

        with pytest.raises(ValueError, match="digit_capsule_dim must be positive"):
            CapsNet(num_classes=10, digit_capsule_dim=-2)

    def test_build_process(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test model building process."""
        capsnet = CapsNet(num_classes=num_classes)

        # Build by calling with data
        outputs = capsnet(sample_mnist_data)

        assert capsnet._layers_built is True
        assert capsnet.built is True
        assert len(capsnet.conv_layers) == len(capsnet.conv_filters)
        assert capsnet.primary_caps is not None
        assert capsnet.digit_caps is not None
        assert isinstance(capsnet.primary_caps, PrimaryCapsule)
        assert isinstance(capsnet.digit_caps, RoutingCapsule)

    def test_build_with_reconstruction(self, num_classes, mnist_input_shape):
        """Test building with reconstruction enabled."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=True
        )

        # Build the model by calling it with sample data
        sample_data = tf.random.uniform([1] + list(mnist_input_shape), 0, 1)
        _ = capsnet(sample_data)

        assert capsnet.decoder is not None
        assert isinstance(capsnet.decoder, keras.Sequential)

    def test_build_without_reconstruction(self, num_classes, mnist_input_shape):
        """Test building without reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        assert capsnet.decoder is None

    def test_forward_pass_without_reconstruction(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test forward pass without reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        outputs = capsnet(sample_mnist_data)

        # Check output structure
        assert isinstance(outputs, dict)
        assert "digit_caps" in outputs
        assert "length" in outputs
        assert "reconstructed" not in outputs

        # Check output shapes
        batch_size = sample_mnist_data.shape[0]
        assert outputs["digit_caps"].shape == (batch_size, num_classes, capsnet.digit_capsule_dim)
        assert outputs["length"].shape == (batch_size, num_classes)

        # Check for valid values
        assert not np.any(np.isnan(outputs["digit_caps"].numpy()))
        assert not np.any(np.isnan(outputs["length"].numpy()))

        # Check length values are positive (they represent magnitudes)
        assert np.all(outputs["length"].numpy() >= 0)

    def test_forward_pass_with_reconstruction(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test forward pass with reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=True
        )

        outputs = capsnet(sample_mnist_data, mask=sample_labels)

        # Check output structure
        assert isinstance(outputs, dict)
        assert "digit_caps" in outputs
        assert "length" in outputs
        assert "reconstructed" in outputs

        # Check reconstruction shape
        assert outputs["reconstructed"].shape == sample_mnist_data.shape

        # Check reconstruction values are in valid range (sigmoid output)
        recon_values = outputs["reconstructed"].numpy()
        assert np.all(recon_values >= 0)
        assert np.all(recon_values <= 1)

    def test_reconstruction_without_mask(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test reconstruction using predicted classes (no mask provided)."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=True
        )

        # Call without mask - should use predicted classes
        outputs = capsnet(sample_mnist_data)

        assert "reconstructed" in outputs
        assert outputs["reconstructed"].shape == sample_mnist_data.shape

    def test_different_input_shapes(self, num_classes):
        """Test with different input shapes."""
        test_shapes = [
            (28, 28, 1),  # MNIST
            (32, 32, 3),  # CIFAR-10
            (64, 64, 3),  # Larger images
        ]

        for shape in test_shapes:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=shape,
                reconstruction=False  # Faster testing
            )

            batch_data = tf.random.uniform([2] + list(shape), 0, 1)
            outputs = capsnet(batch_data)

            assert outputs["digit_caps"].shape == (2, num_classes, capsnet.digit_capsule_dim)
            assert outputs["length"].shape == (2, num_classes)

    def test_different_capsule_configurations(self, mnist_input_shape):
        """Test with different capsule configurations."""
        configs = [
            {"num_classes": 5, "primary_capsules": 16, "primary_capsule_dim": 4, "digit_capsule_dim": 8},
            {"num_classes": 20, "primary_capsules": 64, "primary_capsule_dim": 16, "digit_capsule_dim": 32},
        ]

        for config in configs:
            capsnet = CapsNet(
                input_shape=mnist_input_shape,
                reconstruction=False,
                **config
            )

            batch_data = tf.random.uniform([2] + list(mnist_input_shape), 0, 1)
            outputs = capsnet(batch_data)

            assert outputs["digit_caps"].shape == (2, config["num_classes"], config["digit_capsule_dim"])
            assert outputs["length"].shape == (2, config["num_classes"])

    def test_different_conv_filters(self, num_classes, mnist_input_shape):
        """Test with different convolutional filter configurations."""
        filter_configs = [
            [128],
            [256, 128],
            [64, 128, 256],
        ]

        for filters in filter_configs:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                conv_filters=filters,
                reconstruction=False
            )

            batch_data = tf.random.uniform([2] + list(mnist_input_shape), 0, 1)
            outputs = capsnet(batch_data)

            assert outputs["length"].shape == (2, num_classes)

    def test_train_step(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test custom training step."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )
        capsnet.compile(optimizer="adam", metrics=[CapsuleAccuracy()])

        # Test training step
        metrics = capsnet.train_step((sample_mnist_data, sample_labels))

        # Check returned metrics
        assert "loss" in metrics
        assert "margin_loss" in metrics
        assert "reconstruction_loss" in metrics
        # Metrics may be nested under 'compile_metrics'
        if "compile_metrics" in metrics:
            assert "capsule_accuracy" in metrics["compile_metrics"]
        else:
            assert "capsule_accuracy" in metrics

        # Check metric values are scalars and valid
        for metric_name, metric_value in metrics.items():
            if metric_name == "compile_metrics":
                # Check nested metrics
                for nested_name, nested_value in metric_value.items():
                    assert nested_value.shape == ()
                    assert not np.isnan(nested_value.numpy())
            else:
                assert metric_value.shape == ()
                assert not np.isnan(metric_value.numpy())
                if "loss" in metric_name:
                    assert metric_value.numpy() >= 0

    def test_test_step(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test custom test step."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )
        capsnet.compile(optimizer="adam", metrics=[CapsuleAccuracy()])

        # Test evaluation step
        metrics = capsnet.test_step((sample_mnist_data, sample_labels))

        # Check returned metrics
        assert "loss" in metrics
        assert "margin_loss" in metrics
        assert "reconstruction_loss" in metrics
        # Metrics may be nested under 'compile_metrics'
        if "compile_metrics" in metrics:
            assert "capsule_accuracy" in metrics["compile_metrics"]
        else:
            assert "capsule_accuracy" in metrics

    def test_train_step_without_reconstruction(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test training step without reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )
        capsnet.compile(optimizer="adam")

        metrics = capsnet.train_step((sample_mnist_data, sample_labels))

        # Reconstruction loss should be 0
        assert metrics["reconstruction_loss"].numpy() == 0.0

    def test_different_margin_parameters(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test with different margin loss parameters."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            positive_margin=0.95,
            negative_margin=0.05,
            downweight=0.3,
            reconstruction=False
        )
        capsnet.compile(optimizer="adam")

        metrics = capsnet.train_step((sample_mnist_data, sample_labels))
        assert "margin_loss" in metrics

    def test_serialization(self, num_classes, mnist_input_shape):
        """Test serialization and deserialization."""
        original_capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            routing_iterations=5,
            conv_filters=[128, 256],
            primary_capsules=16,
            primary_capsule_dim=4,
            digit_capsule_dim=8,
            reconstruction=False,
            use_batch_norm=False,
            positive_margin=0.95,
            negative_margin=0.05,
            downweight=0.3,
            reconstruction_weight=0.005
        )

        # Get configs
        config = original_capsnet.get_config()

        # Recreate the model
        recreated_capsnet = CapsNet.from_config(config)

        # Check configuration matches
        assert recreated_capsnet.num_classes == original_capsnet.num_classes
        assert recreated_capsnet.routing_iterations == original_capsnet.routing_iterations
        assert recreated_capsnet.conv_filters == original_capsnet.conv_filters
        assert recreated_capsnet.primary_capsules == original_capsnet.primary_capsules
        assert recreated_capsnet.primary_capsule_dim == original_capsnet.primary_capsule_dim
        assert recreated_capsnet.digit_capsule_dim == original_capsnet.digit_capsule_dim
        assert recreated_capsnet.reconstruction == original_capsnet.reconstruction
        assert recreated_capsnet.use_batch_norm == original_capsnet.use_batch_norm
        assert recreated_capsnet.positive_margin == original_capsnet.positive_margin
        assert recreated_capsnet.negative_margin == original_capsnet.negative_margin
        assert recreated_capsnet.downweight == original_capsnet.downweight
        assert recreated_capsnet.reconstruction_weight == original_capsnet.reconstruction_weight

    def test_model_save_load(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test saving and loading a CapsNet model."""
        # Create and compile model
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False,  # Simpler for testing
            name="test_capsnet"
        )
        capsnet.compile(optimizer="adam", metrics=[CapsuleAccuracy()])

        # Train for one step to initialize all variables
        capsnet.train_step((sample_mnist_data, sample_labels))

        # Generate prediction before saving
        original_outputs = capsnet(sample_mnist_data, training=False)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "capsnet_model.keras")

            # Save the model
            capsnet.save(model_path)

            # Load the model
            loaded_capsnet = keras.models.load_model(
                model_path,
                custom_objects={
                    "CapsNet": CapsNet,
                    "PrimaryCapsule": PrimaryCapsule,
                    "RoutingCapsule": RoutingCapsule,
                    "CapsuleAccuracy": CapsuleAccuracy,
                    "capsule_margin_loss": capsule_margin_loss,
                    "length": length
                }
            )

            # Generate prediction with loaded model
            loaded_outputs = loaded_capsnet(sample_mnist_data, training=False)

            # Check output shapes match
            for key in original_outputs.keys():
                assert original_outputs[key].shape == loaded_outputs[key].shape

    def test_training_integration(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test training integration with fit() method."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            conv_filters=[64, 128],  # Smaller for faster testing
            reconstruction=False
        )
        capsnet.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=[CapsuleAccuracy()]
        )

        # Create dataset with repeated data
        expanded_data = tf.tile(sample_mnist_data, [2, 1, 1, 1])  # 8 samples
        expanded_labels = tf.tile(sample_labels, [2, 1])  # 8 labels

        dataset = tf.data.Dataset.from_tensor_slices((expanded_data, expanded_labels))
        dataset = dataset.batch(4)

        # Train for a few steps
        history = capsnet.fit(dataset, epochs=2, verbose=0)

        # Check that training metrics are recorded
        assert "loss" in history.history
        assert "margin_loss" in history.history
        assert "reconstruction_loss" in history.history
        assert "capsule_accuracy" in history.history
        assert len(history.history["loss"]) == 2  # 2 epochs

    def test_gradient_flow(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test gradient flow through the model."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )
        capsnet.compile(optimizer="adam")

        with tf.GradientTape() as tape:
            outputs = capsnet(sample_mnist_data, training=True)
            # Compute margin loss manually
            margin_loss = tf.reduce_mean(capsule_margin_loss(
                outputs["length"],
                sample_labels,
                capsnet.downweight,
                capsnet.positive_margin,
                capsnet.negative_margin
            ))

        # Get gradients
        grads = tape.gradient(margin_loss, capsnet.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check some gradients have non-zero values
        has_nonzero_grad = any(np.any(g.numpy() != 0) for g in grads if g is not None)
        assert has_nonzero_grad

    def test_create_capsnet_factory(self, num_classes, mnist_input_shape):
        """Test the create_capsnet factory function."""
        capsnet = create_capsnet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            optimizer="adam",
            learning_rate=0.001
        )

        assert isinstance(capsnet, CapsNet)
        assert capsnet.num_classes == num_classes
        assert capsnet._input_shape == mnist_input_shape
        # Build the model by calling it
        sample_data = tf.random.uniform([1] + list(mnist_input_shape), 0, 1)
        _ = capsnet(sample_data)
        assert capsnet.built is True
        assert capsnet.optimizer is not None
        assert len(capsnet.metrics) > 0  # Should have CapsuleAccuracy

    def test_create_capsnet_with_custom_optimizer(self, num_classes, mnist_input_shape):
        """Test create_capsnet with custom optimizer."""
        custom_optimizer = keras.optimizers.Adam(learning_rate=0.002)

        capsnet = create_capsnet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            optimizer=custom_optimizer
        )

        assert capsnet.optimizer == custom_optimizer

    def test_numerical_stability(self, num_classes):
        """Test model stability with extreme input values."""
        # Use larger input size to avoid negative output dimensions
        input_shape = (64, 64, 1)
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=input_shape,
            reconstruction=False
        )

        test_cases = [
            tf.zeros((2,) + input_shape),  # All zeros
            tf.ones((2,) + input_shape),   # All ones
            tf.random.uniform((2,) + input_shape, 0, 1e-6),  # Very small values
            tf.random.uniform((2,) + input_shape, 1-1e-6, 1),  # Very close to 1
        ]

        for i, test_input in enumerate(test_cases):
            outputs = capsnet(test_input)

            # Check for NaN/Inf values
            for key, tensor in outputs.items():
                assert not np.any(np.isnan(tensor.numpy())), f"NaN in {key} for test case {i}"
                assert not np.any(np.isinf(tensor.numpy())), f"Inf in {key} for test case {i}"

    def test_batch_size_independence(self, num_classes, mnist_input_shape):
        """Test that model works with different batch sizes."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            test_data = tf.random.uniform((batch_size,) + mnist_input_shape, 0, 1)
            outputs = capsnet(test_data)

            assert outputs["digit_caps"].shape[0] == batch_size
            assert outputs["length"].shape[0] == batch_size

    def test_routing_iterations(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test with different routing iterations."""
        routing_iterations = [1, 3, 5, 10]

        for iterations in routing_iterations:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                routing_iterations=iterations,
                reconstruction=False
            )

            outputs = capsnet(sample_mnist_data)

            # Should work with any number of iterations
            assert outputs["length"].shape == (sample_mnist_data.shape[0], num_classes)

    def test_without_batch_norm(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test model without batch normalization."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            use_batch_norm=False,
            reconstruction=False
        )

        outputs = capsnet(sample_mnist_data)
        assert outputs["length"].shape == (sample_mnist_data.shape[0], num_classes)

    def test_reconstruction_weight_effect(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test effect of different reconstruction weights."""
        weights = [0.0, 0.001, 0.01, 0.1]

        for weight in weights:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                reconstruction=True,
                reconstruction_weight=weight
            )
            capsnet.compile(optimizer="adam")

            metrics = capsnet.train_step((sample_mnist_data, sample_labels))

            # Total loss should include reconstruction component proportional to weight
            assert "loss" in metrics
            assert "reconstruction_loss" in metrics

    def test_invalid_input_shape_error(self, num_classes):
        """Test that invalid input shapes raise appropriate errors."""
        capsnet = CapsNet(num_classes=num_classes)

        # Test with wrong number of dimensions
        invalid_input = tf.random.uniform([4, 28, 28])  # Missing channel dimension

        with pytest.raises(ValueError, match="Expected 4D input"):
            capsnet(invalid_input)

    def test_model_summary(self, num_classes, mnist_input_shape):
        """Test model summary functionality."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )

        # Should not raise an error
        capsnet.summary()

    def test_save_model_method(self, num_classes, mnist_input_shape):
        """Test the save_model method."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")

            # Should not raise an error
            capsnet.save_model(model_path)

            # File should exist
            assert os.path.exists(model_path)

    def test_load_model_method(self, num_classes, mnist_input_shape):
        """Test the load_model class method."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")

            # Save model
            capsnet.save_model(model_path)

            # Load model using class method
            loaded_capsnet = CapsNet.load_model(model_path)

            assert isinstance(loaded_capsnet, CapsNet)
            assert loaded_capsnet.num_classes == num_classes

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])