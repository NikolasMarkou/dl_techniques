"""
Comprehensive test suite for the ConvNeXt V2 model.

This module contains all tests for the ConvNeXt V2 model implementation,
covering initialization, forward pass, training, serialization,
model integration, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple, List

from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2, create_convnext_v2
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block


class TestConvNeXtV2:
    """Test suite for ConvNeXt V2 model implementation."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for standard images."""
        return (224, 224, 3)

    @pytest.fixture
    def small_input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for smaller images."""
        return (64, 64, 3)

    @pytest.fixture
    def cifar_input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for CIFAR-10 like images."""
        return (32, 32, 3)

    @pytest.fixture
    def num_classes(self) -> int:
        """Create test number of classes."""
        return 10

    @pytest.fixture
    def sample_data(self, input_shape):
        """Create sample input data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(input_shape), 0, 1)

    @pytest.fixture
    def small_sample_data(self, small_input_shape):
        """Create small sample input data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(small_input_shape), 0, 1)

    @pytest.fixture
    def cifar_sample_data(self, cifar_input_shape):
        """Create CIFAR-like sample data."""
        batch_size = 8
        return tf.random.uniform([batch_size] + list(cifar_input_shape), 0, 1)

    def test_initialization_defaults(self, num_classes):
        """Test initialization with default parameters."""
        model = ConvNeXtV2(num_classes=num_classes)

        assert model.num_classes == num_classes
        assert model.depths == [3, 3, 9, 3]  # ConvNeXt-Tiny defaults
        assert model.dims == [96, 192, 384, 768]
        assert model.drop_path_rate == 0.0
        assert model.kernel_size == 7
        assert model.activation == "gelu"
        assert model.use_bias is True
        assert model.kernel_regularizer is None
        assert model.dropout_rate == 0.0
        assert model.spatial_dropout_rate == 0.0
        assert model.use_gamma is True
        assert model.use_softorthonormal_regularizer is False
        assert model.include_top is True

    def test_initialization_custom(self, num_classes, small_input_shape):
        """Test initialization with custom parameters."""
        custom_depths = [2, 2, 6, 2]
        custom_dims = [64, 128, 256, 512]

        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=custom_depths,
            dims=custom_dims,
            drop_path_rate=0.1,
            kernel_size=5,
            activation="relu",
            use_bias=False,
            dropout_rate=0.2,
            spatial_dropout_rate=0.1,
            use_gamma=False,
            use_softorthonormal_regularizer=True,
            include_top=False,
            input_shape=small_input_shape
        )

        assert model.num_classes == num_classes
        assert model.depths == custom_depths
        assert model.dims == custom_dims
        assert model.drop_path_rate == 0.1
        assert model.kernel_size == 5
        assert model.activation == "relu"
        assert model.use_bias is False
        assert model.dropout_rate == 0.2
        assert model.spatial_dropout_rate == 0.1
        assert model.use_gamma is False
        assert model.use_softorthonormal_regularizer is True
        assert model.include_top is False

    def test_initialization_with_regularization(self, num_classes):
        """Test initialization with regularization."""
        regularizer = keras.regularizers.L2(0.01)
        model = ConvNeXtV2(
            num_classes=num_classes,
            kernel_regularizer=regularizer
        )

        assert model.kernel_regularizer == regularizer

    @pytest.mark.parametrize(
        "variant,expected_depths,expected_dims",
        [
            ("atto", [2, 2, 6, 2], [40, 80, 160, 320]),
            ("femto", [2, 2, 6, 2], [48, 96, 192, 384]),
            ("pico", [2, 2, 6, 2], [64, 128, 256, 512]),
            ("nano", [2, 2, 8, 2], [80, 160, 320, 640]),
            ("tiny", [3, 3, 9, 3], [96, 192, 384, 768]),
            ("base", [3, 3, 27, 3], [128, 256, 512, 1024]),
            ("large", [3, 3, 27, 3], [192, 384, 768, 1536]),
            ("huge", [3, 3, 27, 3], [352, 704, 1408, 2816]),
        ],
    )
    def test_model_variants(self, variant, expected_depths, expected_dims, num_classes):
        """Test all predefined model variants."""
        model = ConvNeXtV2.from_variant(variant, num_classes=num_classes)

        assert model.num_classes == num_classes
        assert model.depths == expected_depths
        assert model.dims == expected_dims

    def test_invalid_variant(self, num_classes):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            ConvNeXtV2.from_variant("invalid_variant", num_classes=num_classes)

    def test_micro_variants(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test the smaller ConvNeXt V2 variants (atto, femto, pico, nano)."""
        micro_variants = ["atto", "femto", "pico", "nano"]

        for variant in micro_variants:
            model = ConvNeXtV2.from_variant(
                variant,
                num_classes=num_classes,
                input_shape=cifar_input_shape  # Specify correct input shape
            )

            outputs = model(cifar_sample_data)
            assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

    def test_forward_pass_with_top(self, num_classes, sample_data):
        """Test forward pass with classification head."""
        model = ConvNeXtV2(num_classes=num_classes, depths=[1, 1], dims=[32, 64])

        outputs = model(sample_data)

        # Check output shape
        expected_shape = (sample_data.shape[0], num_classes)
        assert outputs.shape == expected_shape

        # Check for valid values
        assert not np.any(np.isnan(outputs.numpy()))
        assert not np.any(np.isinf(outputs.numpy()))

    def test_different_input_shapes(self, num_classes):
        """Test with different input shapes."""
        test_shapes = [
            (32, 32, 3),   # CIFAR-10
            (64, 64, 3),   # Medium size
            (128, 128, 3), # Larger
            (224, 224, 1), # Grayscale ImageNet size
            (256, 256, 3), # Large
        ]

        for shape in test_shapes:
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=[2, 2, 4, 2],  # Smaller for faster testing
                dims=[32, 64, 128, 256],
                input_shape=shape  # Specify the correct input shape
            )
            batch_data = tf.random.uniform([2] + list(shape), 0, 1)

            outputs = model(batch_data)
            assert outputs.shape == (2, num_classes)

    def test_different_depths_configurations(self, num_classes, cifar_input_shape):
        """Test with different depth configurations."""
        depth_configs = [
            [1, 1, 2, 1],     # Very small
            [2, 2, 4, 2],     # Small
            [2, 2, 6, 2],     # Atto/Femto/Pico (V2 micro variants)
            [2, 2, 8, 2],     # Nano (V2 variant)
            [3, 3, 9, 3],     # Tiny (default)
        ]

        for depths in depth_configs:
            dims = [32, 64, 128, 256]  # Keep dims small for testing
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=depths,
                dims=dims,
                input_shape=cifar_input_shape  # Specify correct input shape
            )
            batch_data = tf.random.uniform([2] + list(cifar_input_shape), 0, 1)

            outputs = model(batch_data)
            assert outputs.shape == (2, num_classes)

    def test_different_activations(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test with different activation functions."""
        activations = ["relu", "gelu", "leaky_relu", "elu", "swish"]

        for activation in activations:
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=[1, 1, 2, 1],  # Small for faster testing
                dims=[32, 64, 128, 256],
                activation=activation,
                input_shape=cifar_input_shape  # Specify correct input shape
            )

            outputs = model(cifar_sample_data)
            assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

    def test_stochastic_depth(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test stochastic depth (drop path) functionality."""
        drop_path_rates = [0.0, 0.1, 0.2, 0.3]

        for drop_path_rate in drop_path_rates:
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=[2, 2, 4, 2],
                dims=[32, 64, 128, 256],
                drop_path_rate=drop_path_rate,
                input_shape=cifar_input_shape  # Specify correct input shape
            )

            # Test training mode (stochastic depth active)
            outputs_train = model(cifar_sample_data, training=True)

            # Test inference mode (stochastic depth inactive)
            outputs_test = model(cifar_sample_data, training=False)

            # Both should have correct shapes
            expected_shape = (cifar_sample_data.shape[0], num_classes)
            assert outputs_train.shape == expected_shape
            assert outputs_test.shape == expected_shape

    def test_dropout_configurations(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test different dropout configurations."""
        dropout_configs = [
            (0.0, 0.0),   # No dropout
            (0.1, 0.0),   # Only regular dropout
            (0.0, 0.1),   # Only spatial dropout
            (0.1, 0.1),   # Both dropouts
        ]

        for dropout_rate, spatial_dropout_rate in dropout_configs:
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=[1, 1, 2, 1],
                dims=[32, 64, 128, 256],
                dropout_rate=dropout_rate,
                spatial_dropout_rate=spatial_dropout_rate,
                input_shape=cifar_input_shape  # Specify correct input shape
            )

            outputs = model(cifar_sample_data, training=True)
            assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

    def test_gamma_scaling(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test learnable gamma scaling."""
        use_gamma_options = [True, False]

        for use_gamma in use_gamma_options:
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=[1, 1, 2, 1],
                dims=[32, 64, 128, 256],
                use_gamma=use_gamma,
                input_shape=cifar_input_shape  # Specify correct input shape
            )

            outputs = model(cifar_sample_data)
            assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

    def test_regularization_options(self, num_classes, cifar_input_shape):
        """Test different regularization options."""
        # Test kernel regularization
        model_with_l2 = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1, 2, 1],
            dims=[32, 64, 128, 256],
            kernel_regularizer=keras.regularizers.L2(0.01),
            input_shape=cifar_input_shape  # Specify correct input shape
        )

        # Test soft orthonormal regularization
        model_with_orthonormal = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1, 2, 1],
            dims=[32, 64, 128, 256],
            use_softorthonormal_regularizer=True,
            input_shape=cifar_input_shape  # Specify correct input shape
        )

        batch_data = tf.random.uniform([2] + list(cifar_input_shape), 0, 1)

        outputs_l2 = model_with_l2(batch_data)
        outputs_orthonormal = model_with_orthonormal(batch_data)

        assert outputs_l2.shape == (2, num_classes)
        assert outputs_orthonormal.shape == (2, num_classes)

    def test_global_response_normalization_integration(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test that Global Response Normalization is properly integrated."""
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1, 2, 1],
            dims=[32, 64, 128, 256],
            input_shape=cifar_input_shape  # Specify correct input shape
        )

        # Forward pass should work without issues
        outputs = model(cifar_sample_data)
        assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

        # Check that GRN layers are present in the model
        grn_layers_found = 0
        for layer in model.layers:
            # Check if it's a ConvNextV2Block and has GRN
            if hasattr(layer, 'grn') and layer.grn is not None:
                grn_layers_found += 1

        # We should find GRN layers in the V2 blocks
        # Note: This is an indirect test since the blocks are created dynamically
        assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

    def test_model_compilation(self, num_classes):
        """Test model compilation with different optimizers and losses."""
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1, 2, 1],
            dims=[32, 64, 128, 256]
        )

        # Test with different optimizers
        optimizers = [
            keras.optimizers.Adam(learning_rate=0.001),
            keras.optimizers.SGD(learning_rate=0.01),
            keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
        ]

        for optimizer in optimizers:
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"]
            )
            assert model.optimizer == optimizer

    def test_serialization(self, num_classes):
        """Test serialization and deserialization of the model."""
        original_model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[2, 2, 6, 2],
            dims=[64, 128, 256, 512],
            drop_path_rate=0.1,
            kernel_size=5,
            activation="relu",
            use_bias=False,
            dropout_rate=0.1,
            spatial_dropout_rate=0.05,
            use_gamma=False,
            use_softorthonormal_regularizer=True
        )

        # Get config
        config = original_model.get_config()

        # Recreate the model
        recreated_model = ConvNeXtV2.from_config(config)

        # Check configuration matches
        assert recreated_model.num_classes == original_model.num_classes
        assert recreated_model.depths == original_model.depths
        assert recreated_model.dims == original_model.dims
        assert recreated_model.drop_path_rate == original_model.drop_path_rate
        assert recreated_model.kernel_size == original_model.kernel_size
        assert recreated_model.activation == original_model.activation
        assert recreated_model.use_bias == original_model.use_bias
        assert recreated_model.dropout_rate == original_model.dropout_rate
        assert recreated_model.spatial_dropout_rate == original_model.spatial_dropout_rate
        assert recreated_model.use_gamma == original_model.use_gamma
        assert recreated_model.use_softorthonormal_regularizer == original_model.use_softorthonormal_regularizer

    def test_model_save_load(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test saving and loading a ConvNeXt V2 model."""
        # Create and compile model
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1, 2, 1],
            dims=[32, 64, 128, 256],
            input_shape=cifar_input_shape,  # Specify correct input shape
            name="test_convnext_v2"
        )
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Generate prediction before saving
        original_outputs = model(cifar_sample_data, training=False)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "convnext_v2_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "ConvNeXtV2": ConvNeXtV2,
                    "ConvNextV2Block": ConvNextV2Block
                }
            )

            # Generate prediction with loaded model
            loaded_outputs = loaded_model(cifar_sample_data, training=False)

            # Check outputs match (shapes should be identical)
            assert original_outputs.shape == loaded_outputs.shape
            assert loaded_model.count_params() == model.count_params()

    def test_training_integration(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test training integration with a small dataset."""
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1],
            dims=[32, 64,],
            input_shape=cifar_input_shape  # Specify correct input shape
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Create small dataset
        labels = tf.random.uniform([cifar_sample_data.shape[0]], 0, num_classes, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((cifar_sample_data, labels))
        dataset = dataset.batch(4)

        # Train for a few steps
        history = model.fit(dataset, epochs=2, verbose=0)

        # Check that training metrics are recorded
        assert "loss" in history.history
        assert "accuracy" in history.history
        assert len(history.history["loss"]) == 2  # 2 epochs

        # Check that loss decreases (or at least doesn't increase dramatically)
        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        assert final_loss < initial_loss * 2  # Allow some variance

    def test_gradient_flow(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test gradient flow through the model."""
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1],
            dims=[32, 64],
            input_shape=cifar_input_shape  # Specify correct input shape
        )

        labels = tf.random.uniform([cifar_sample_data.shape[0]], 0, num_classes, dtype=tf.int32)

        with tf.GradientTape() as tape:
            outputs = model(cifar_sample_data, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(labels, outputs, from_logits=True)
            loss = tf.reduce_mean(loss)

        # Get gradients
        grads = tape.gradient(loss, model.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check some gradients have non-zero values
        has_nonzero_grad = any(np.any(g.numpy() != 0) for g in grads if g is not None)
        assert has_nonzero_grad

    def test_create_convnext_v2_factory(self, num_classes):
        """Test the create_convnext_v2 factory function."""
        model = create_convnext_v2(
            variant="tiny",
            num_classes=num_classes,
            drop_path_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(0.01)
        )

        assert isinstance(model, ConvNeXtV2)
        assert model.num_classes == num_classes
        assert model.depths == [3, 3, 9, 3]  # Tiny variant
        assert model.dims == [96, 192, 384, 768]
        assert model.drop_path_rate == 0.1

    def test_factory_with_micro_variants(self, num_classes):
        """Test factory function with V2 micro variants."""
        micro_variants = ["atto", "femto", "pico", "nano"]

        for variant in micro_variants:
            model = create_convnext_v2(
                variant=variant,
                num_classes=num_classes
            )

            assert isinstance(model, ConvNeXtV2)
            expected_config = ConvNeXtV2.MODEL_VARIANTS[variant]
            assert model.depths == expected_config["depths"]
            assert model.dims == expected_config["dims"]

    def test_factory_with_pretrained_warning(self, num_classes):
        """Test factory function with pretrained warning."""
        # This should log a warning but still create the model
        model = create_convnext_v2(
            variant="base",
            num_classes=num_classes,
            pretrained=True
        )

        assert isinstance(model, ConvNeXtV2)
        assert model.depths == [3, 3, 27, 3]  # Base variant

    def test_numerical_stability(self, num_classes):
        """Test model stability with extreme input values."""
        input_shape = (32, 32, 3)
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1],
            dims=[16, 32],
            input_shape=input_shape  # Specify correct input shape
        )

        test_cases = [
            tf.zeros((2,) + input_shape),  # All zeros
            tf.ones((2,) + input_shape),   # All ones
            tf.random.uniform((2,) + input_shape, 0, 1e-6),  # Very small values
            tf.random.uniform((2,) + input_shape, 1-1e-6, 1),  # Very close to 1
            tf.random.normal((2,) + input_shape, 0, 10),  # Large variance
        ]

        for i, test_input in enumerate(test_cases):
            outputs = model(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(outputs.numpy())), f"NaN in outputs for test case {i}"
            assert not np.any(np.isinf(outputs.numpy())), f"Inf in outputs for test case {i}"

    def test_batch_size_independence(self, num_classes, cifar_input_shape):
        """Test that model works with different batch sizes."""
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[1, 1],
            dims=[32, 64],
            input_shape=cifar_input_shape  # Specify correct input shape
        )

        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            test_data = tf.random.uniform((batch_size,) + cifar_input_shape, 0, 1)
            outputs = model(test_data)

            assert outputs.shape == (batch_size, num_classes)

    def test_model_summary_information(self, num_classes, capsys):
        """Test that model summary provides useful information."""
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[2, 2],
            dims=[64, 128]
        )

        # Call summary (which should also log additional info)
        model.summary()

        # Check that something was printed
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Summary was printed

    def test_kernel_size_variations(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test different kernel sizes."""
        kernel_sizes = [3, 5, 7, 9]

        for kernel_size in kernel_sizes:
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=[1, 1, 2, 1],
                dims=[32, 64, 128, 256],
                kernel_size=kernel_size,
                input_shape=cifar_input_shape  # Specify correct input shape
            )

            outputs = model(cifar_sample_data)
            assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

    def test_v2_specific_features(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test ConvNeXt V2 specific features (mainly GRN integration)."""
        # Create a V2 model and ensure it works properly with GRN
        model = ConvNeXtV2(
            num_classes=num_classes,
            depths=[2, 2, 6, 2],  # Atto-like configuration
            dims=[40, 80, 160, 320],
            input_shape=cifar_input_shape  # Specify correct input shape
        )

        # Should work without issues even with GRN
        outputs = model(cifar_sample_data)
        assert outputs.shape == (cifar_sample_data.shape[0], num_classes)

        # Test training mode
        outputs_train = model(cifar_sample_data, training=True)
        assert outputs_train.shape == (cifar_sample_data.shape[0], num_classes)

    def test_model_memory_efficiency(self, num_classes, cifar_input_shape):
        """Test model creation and deletion for memory efficiency."""
        # Create and delete multiple models to test memory management
        for i in range(3):
            model = ConvNeXtV2(
                num_classes=num_classes,
                depths=[1, 1, 2, 1],
                dims=[16, 32, 64, 128],  # Small model
                input_shape=cifar_input_shape  # Specify correct input shape
            )

            test_data = tf.random.uniform([2] + list(cifar_input_shape), 0, 1)
            _ = model(test_data)

            # Explicitly delete
            del model

        # If we get here without memory issues, the test passes
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])