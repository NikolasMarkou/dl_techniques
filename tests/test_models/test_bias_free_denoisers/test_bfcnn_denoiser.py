"""
Comprehensive test suite for Bias-Free CNN Denoiser Model.

Tests cover initialization, validation, architecture verification, forward pass,
serialization, scaling invariance property, and variant configurations.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
import tensorflow as tf
from typing import Tuple, Dict, Any

# Import the new model functions
from dl_techniques.models.bias_free_denoisers.bfcnn import (
    create_bfcnn_denoiser,
    create_bfcnn_variant,
    BFCNN_CONFIGS
)
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock

from .conftest import HOMOGENEITY_RTOL, HOMOGENEITY_SCALES, fit_one_step
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)


class TestBFCNNDenoiser:
    """Test suite for Bias-Free CNN Denoiser implementation."""

    @pytest.fixture
    def grayscale_input_shape(self) -> Tuple[int, int, int]:
        """Standard grayscale image shape for testing."""
        return (64, 64, 1)

    @pytest.fixture
    def rgb_input_shape(self) -> Tuple[int, int, int]:
        """Standard RGB image shape for testing."""
        return (128, 128, 3)

    @pytest.fixture
    def variable_input_shape(self) -> Tuple[int, int, int]:
        """Variable size input shape for testing."""
        return (None, None, 1)

    @pytest.fixture
    def test_image_grayscale(self) -> np.ndarray:
        """Create test grayscale image data."""
        return np.random.rand(2, 64, 64, 1).astype(np.float32)

    @pytest.fixture
    def test_image_rgb(self) -> np.ndarray:
        """Create test RGB image data."""
        return np.random.rand(2, 128, 128, 3).astype(np.float32)

    @pytest.fixture
    def default_model_config(self) -> Dict[str, Any]:
        """Default configuration for model creation."""
        return {
            'num_blocks': 4,
            'filters': 32,
            'kernel_size': 3,
            'activation': 'relu',
            'final_activation': 'linear'
        }

    # ================================================================
    # Initialization Tests
    # ================================================================

    def test_initialization_defaults(self, grayscale_input_shape):
        """Test initialization with default parameters."""
        model = create_bfcnn_denoiser(input_shape=grayscale_input_shape)

        # Check model properties
        assert model.name == 'bfcnn_denoiser'
        assert len(model.layers) > 0
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

    def test_initialization_custom_parameters(self, rgb_input_shape):
        """Test initialization with custom parameters."""
        custom_config = {
            'num_blocks': 6,
            'filters': 128,
            'initial_kernel_size': 7,
            'kernel_size': 5,
            'activation': 'relu',  # Changed from 'leaky_relu' to standard activation
            'final_activation': 'linear',  # Changed from 'tanh' to maintain scaling invariance
            'kernel_initializer': 'he_normal',
            'model_name': 'custom_bfcnn'
        }

        model = create_bfcnn_denoiser(
            input_shape=rgb_input_shape,
            **custom_config
        )

        # Check custom values are applied
        assert model.name == 'custom_bfcnn'
        assert model.input_shape == (None,) + rgb_input_shape
        assert model.output_shape == (None,) + rgb_input_shape

        # Should have initial conv + 6 residual blocks + final conv
        # Exact layer count depends on BiasFreeResidualBlock internal structure
        assert len(model.layers) >= 8  # At minimum

    def test_initialization_variable_input_size(self, variable_input_shape):
        """Test initialization with variable input size."""
        model = create_bfcnn_denoiser(
            input_shape=variable_input_shape,
            num_blocks=3,
            filters=64
        )

        assert model.input_shape == (None, None, None, 1)
        assert model.output_shape == (None, None, None, 1)

        # Test with different sized inputs
        small_input = np.random.rand(1, 32, 32, 1).astype(np.float32)
        large_input = np.random.rand(1, 256, 256, 1).astype(np.float32)

        small_output = model(small_input)
        large_output = model(large_input)

        assert small_output.shape == (1, 32, 32, 1)
        assert large_output.shape == (1, 256, 256, 1)

    # ================================================================
    # Input Validation Tests
    # ================================================================

    def test_invalid_input_shape_type(self):
        """Test that invalid input_shape type raises TypeError."""
        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfcnn_denoiser(input_shape=[64, 64, 1])

        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfcnn_denoiser(input_shape=(64, 64))

        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfcnn_denoiser(input_shape=(64, 64, 1, 1))

    def test_invalid_num_blocks(self, grayscale_input_shape):
        """Test that negative num_blocks raises ValueError."""
        with pytest.raises(ValueError, match="num_blocks must be non-negative"):
            create_bfcnn_denoiser(
                input_shape=grayscale_input_shape,
                num_blocks=-1
            )

    def test_invalid_filters(self, grayscale_input_shape):
        """Test that non-positive filters raises ValueError."""
        with pytest.raises(ValueError, match="filters must be positive"):
            create_bfcnn_denoiser(
                input_shape=grayscale_input_shape,
                filters=0
            )

        with pytest.raises(ValueError, match="filters must be positive"):
            create_bfcnn_denoiser(
                input_shape=grayscale_input_shape,
                filters=-10
            )

    def test_zero_blocks_allowed(self, grayscale_input_shape):
        """Test that zero num_blocks is allowed (no residual blocks)."""
        model = create_bfcnn_denoiser(
            input_shape=grayscale_input_shape,
            num_blocks=0,
            filters=32
        )

        # Should still work with just initial and final conv
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    # ================================================================
    # Architecture Verification Tests
    # ================================================================

    def test_model_architecture_structure(self, grayscale_input_shape):
        """Test that model has expected architectural structure."""
        num_blocks = 3
        filters = 64

        model = create_bfcnn_denoiser(
            input_shape=grayscale_input_shape,
            num_blocks=num_blocks,
            filters=filters
        )

        # Check layer names exist
        layer_names = [layer.name for layer in model.layers]

        # Should have stem layer
        assert any('stem' in name for name in layer_names)

        # Should have residual blocks
        for i in range(num_blocks):
            assert any(f'residual_block_{i}' in name for name in layer_names)

        # Should have final conv
        assert any('final_conv' in name for name in layer_names)

    def test_output_channels_match_input(self):
        """Test that output channels match input channels."""
        test_cases = [
            (64, 64, 1),   # Grayscale
            (128, 128, 3), # RGB
            (32, 32, 4),   # RGBA
        ]

        for input_shape in test_cases:
            model = create_bfcnn_denoiser(
                input_shape=input_shape,
                num_blocks=2,
                filters=32
            )

            input_channels = input_shape[2]
            assert model.output_shape[-1] == input_channels

    # ================================================================
    # Forward Pass Tests
    # ================================================================

    def test_forward_pass_grayscale(self, test_image_grayscale):
        """Test forward pass with grayscale images."""
        model = create_bfcnn_denoiser(
            input_shape=(64, 64, 1),
            num_blocks=3,
            filters=32
        )

        output = model(test_image_grayscale)

        # Check output properties
        assert output.shape == test_image_grayscale.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_rgb(self, test_image_rgb):
        """Test forward pass with RGB images."""
        model = create_bfcnn_denoiser(
            input_shape=(128, 128, 3),
            num_blocks=4,
            filters=64
        )

        output = model(test_image_rgb)

        # Check output properties
        assert output.shape == test_image_rgb.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_scaling_invariance_property(self, test_image_grayscale, homogeneity_probe):
        """A TRAINED, factory-built BFCNN is degree-1 homogeneous.

        Reaches the model through ``create_bfcnn_denoiser`` at its defaults -- NOT through
        ``src/train/bfunet/train_bfcnn_denoiser.py``, which is where the ``'batchnorm'`` ->
        ``BiasFreeBatchNorm`` remap used to live and which therefore masked this defect
        from every trainer-side test.

        The ``fit_one_step`` call is the entire point: on an untrained model stock
        ``BatchNormalization`` has ``moving_mean == 0`` and is exactly homogeneous, so the
        pre-2026-08-15 version of this test passed against a model that did not have the
        property. See ``conftest.py`` for the tolerance derivation and the measured
        pre-fix / post-fix numbers.
        """
        model = create_bfcnn_denoiser(
            input_shape=(64, 64, 1),
            num_blocks=2,
            filters=8,
            final_activation='linear',  # required: any other activation is not homogeneous
        )
        fit_one_step(model)

        for c in HOMOGENEITY_SCALES:
            err = homogeneity_probe(model, test_image_grayscale, c)
            assert err < HOMOGENEITY_RTOL, (
                f"homogeneity violated at c={c}: relative error {err:.3e} exceeds "
                f"{HOMOGENEITY_RTOL:.0e}. A trained bias-free denoiser must satisfy "
                "f(c*x) = c*f(x) to float32 round-off"
            )

        moving_means = [w for w in model.weights if 'moving_mean' in w.path]
        assert not moving_means, (
            "the factory-built BFCNN still contains "
            f"{len(moving_means)} moving_mean variable(s) -- it is using stock "
            "BatchNormalization, whose moving_mean subtraction is an additive constant "
            "that breaks f(c*x) = c*f(x)"
        )

    def test_different_batch_sizes(self, grayscale_input_shape):
        """Test model with different batch sizes."""
        model = create_bfcnn_denoiser(
            input_shape=grayscale_input_shape,
            num_blocks=2,
            filters=32
        )

        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            test_input = np.random.rand(batch_size, 64, 64, 1).astype(np.float32)
            output = model(test_input)

            assert output.shape == (batch_size, 64, 64, 1)
            assert not np.any(np.isnan(output.numpy()))

    # ================================================================
    # Numerical Stability Tests
    # ================================================================

    def test_numerical_stability(self, grayscale_input_shape):
        """Test model stability with extreme input values."""
        model = create_bfcnn_denoiser(
            input_shape=grayscale_input_shape,
            num_blocks=3,
            filters=32
        )

        # Test with different input magnitudes
        test_cases = [
            np.zeros((1, 64, 64, 1), dtype=np.float32),  # Zeros
            np.ones((1, 64, 64, 1), dtype=np.float32) * 1e-10,  # Very small
            np.ones((1, 64, 64, 1), dtype=np.float32) * 1e5,    # Very large
            np.random.normal(0, 100, (1, 64, 64, 1)).astype(np.float32)  # High variance
        ]

        for test_input in test_cases:
            output = model(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    # ================================================================
    # Serialization Tests
    # ================================================================

    def test_model_serialization(self, test_image_grayscale):
        """Test saving and loading the model."""
        original_model = create_bfcnn_denoiser(
            input_shape=(64, 64, 1),
            num_blocks=3,
            filters=32,
            model_name='serialization_test'
        )

        # Generate prediction before saving
        original_prediction = original_model.predict(test_image_grayscale)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "bfcnn_model.keras")

            # Save the model
            original_model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'BiasFreeConv2D': BiasFreeConv2D,
                    'BiasFreeResidualBlock': BiasFreeResidualBlock
                }
            )

            # Test that loaded model produces same results
            loaded_prediction = loaded_model.predict(test_image_grayscale)

            # Verify model name and predictions match
            assert loaded_model.name == 'serialization_test'
            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6,
                atol=1e-8
            )

    # ================================================================
    # Training Integration Tests
    # ================================================================

    def test_model_compilation(self, grayscale_input_shape):
        """Test that model can be compiled with different optimizers and losses."""
        model = create_bfcnn_denoiser(
            input_shape=grayscale_input_shape,
            num_blocks=2,
            filters=32
        )

        # Test different compilation configurations
        compilation_configs = [
            {'optimizer': 'adam', 'loss': 'mse'},
            {'optimizer': 'rmsprop', 'loss': 'mae'},
            {'optimizer': keras.optimizers.Adam(learning_rate=0.001), 'loss': 'huber'},
        ]

        for config in compilation_configs:
            model.compile(**config)
            assert model.optimizer is not None
            assert hasattr(model, 'loss')

    def test_gradient_flow(self, test_image_grayscale):
        """Test gradient flow through the model."""
        model = create_bfcnn_denoiser(
            input_shape=(64, 64, 1),
            num_blocks=2,
            filters=32
        )

        model.compile(optimizer='adam', loss='mse')

        # Create target (for denoising, target could be clean image)
        target = test_image_grayscale * 0.8  # Simulated clean image

        # Test that gradients can be computed
        with tf.GradientTape() as tape:
            predictions = model(test_image_grayscale, training=True)
            loss = keras.losses.mean_squared_error(target, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in gradients)

        # Check gradients have non-zero values
        assert all(tf.reduce_any(tf.not_equal(g, 0.0)) for g in gradients)

    # ================================================================
    # Variant Configuration Tests
    # ================================================================

    def test_variant_configs_exist(self):
        """Test that all expected variant configurations exist."""
        expected_variants = ['tiny', 'small', 'base', 'large', 'xlarge']

        for variant in expected_variants:
            assert variant in BFCNN_CONFIGS
            config = BFCNN_CONFIGS[variant]
            assert 'num_blocks' in config
            assert 'filters' in config
            assert 'description' in config

    def test_create_bfcnn_variant_function(self, grayscale_input_shape):
        """Test the create_bfcnn_variant function with all variants."""
        variants = ['tiny', 'small', 'base', 'large', 'xlarge']

        for variant in variants:
            model = create_bfcnn_variant(variant, grayscale_input_shape)

            # Check model properties
            assert model.name == f'bfcnn_{variant}'
            assert model.input_shape == (None,) + grayscale_input_shape
            assert model.output_shape == (None,) + grayscale_input_shape

            # Test forward pass
            test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
            output = model(test_input)
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_variant_invalid_name(self, grayscale_input_shape):
        """Test that invalid variant name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            create_bfcnn_variant('invalid', grayscale_input_shape)

    def test_variant_parameter_override(self, grayscale_input_shape):
        """Test that variant parameters can be overridden."""
        # Override the base variant's filters
        model = create_bfcnn_variant(
            'base',
            grayscale_input_shape,
            filters=128,  # Override default
            model_name='custom_base'  # Override default name
        )

        assert model.name == 'custom_base'

        # Test forward pass still works
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_all_variants_consistency(self):
        """Test that all variants work with same input."""
        input_shape = (64, 64, 1)
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        variants = ['tiny', 'small', 'base', 'large', 'xlarge']

        models = []
        for variant in variants:
            model = create_bfcnn_variant(variant, input_shape)
            models.append(model)

        for model in models:
            output = model(test_input)

            # All should produce valid outputs
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    def test_variant_complexity_ordering(self):
        """Test that variants have expected complexity ordering."""
        input_shape = (64, 64, 1)
        variants = ['tiny', 'small', 'base', 'large', 'xlarge']

        model_complexities = []
        for variant in variants:
            model = create_bfcnn_variant(variant, input_shape)
            # Use number of parameters as complexity metric
            complexity = model.count_params()
            model_complexities.append(complexity)

        # Each subsequent variant should have more parameters
        for i in range(1, len(model_complexities)):
            assert model_complexities[i] > model_complexities[i-1], \
                f"Variant {variants[i]} should have more parameters than {variants[i-1]}"

    # ================================================================
    # Edge Cases and Robustness Tests
    # ================================================================

    def test_single_pixel_image(self):
        """Test model with minimal image size."""
        model = create_bfcnn_denoiser(
            input_shape=(1, 1, 1),
            num_blocks=1,
            filters=16,
            kernel_size=1  # Must use 1x1 kernels for 1x1 images
        )

        test_input = np.random.rand(1, 1, 1, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_large_filter_count(self, grayscale_input_shape):
        """Test model with large number of filters."""
        model = create_bfcnn_denoiser(
            input_shape=grayscale_input_shape,
            num_blocks=2,
            filters=512  # Large filter count
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_many_residual_blocks(self, grayscale_input_shape):
        """Test model with many residual blocks."""
        model = create_bfcnn_denoiser(
            input_shape=grayscale_input_shape,
            num_blocks=20,  # Many blocks
            filters=32
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    # ================================================================
    # Performance and Memory Tests
    # ================================================================

    def test_memory_efficiency(self, grayscale_input_shape):
        """Test that model creation doesn't cause memory issues."""
        # Create multiple models to test memory management
        models = []

        for i in range(5):
            model = create_bfcnn_denoiser(
                input_shape=grayscale_input_shape,
                num_blocks=3,
                filters=32,
                model_name=f'memory_test_{i}'
            )
            models.append(model)

        # All models should be created successfully
        assert len(models) == 5

        # Test they all work
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        for model in models:
            output = model(test_input)
            assert output.shape == test_input.shape


# ================================================================
# Parameterized Tests for Multiple Configurations
# ================================================================

class TestBFCNNParameterized:
    """Parameterized tests for different model configurations."""

    @pytest.mark.parametrize("input_shape,num_blocks,filters", [
        ((32, 32, 1), 2, 16),
        ((64, 64, 3), 4, 32),
        ((128, 128, 1), 8, 64),
        ((256, 256, 3), 6, 128),
    ])
    def test_various_configurations(self, input_shape, num_blocks, filters):
        """Each configuration builds and forwards without NaN.

        This is a build smoke only; the shape is the input's shape by
        construction and says nothing about whether ``num_blocks``/``filters``
        reached the graph. That claim is made by
        :meth:`test_num_blocks_and_filters_change_the_parameterisation`.
        """
        model = create_bfcnn_denoiser(
            input_shape=input_shape,
            num_blocks=num_blocks,
            filters=filters
        )

        # Test forward pass
        batch_size = 1
        test_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
        output = model(test_input)

        assert output.shape == (batch_size,) + input_shape
        assert not np.any(np.isnan(output.numpy()))

    def test_num_blocks_and_filters_change_the_parameterisation(self):
        """``num_blocks`` and ``filters`` must reach the parameterisation.

        Structural knobs: they change the weight shapes, so an output-difference
        check would be satisfied by the different random draw alone. The weight
        signature is the discriminating fact.
        """
        blocks = {
            n: (lambda n=n: create_bfcnn_denoiser(
                input_shape=(32, 32, 1), num_blocks=n, filters=16))
            for n in (2, 4, 8)
        }
        sigs = assert_structural_knob_changes_weights(blocks, knob="num_blocks")
        # Stronger than "different": more blocks means strictly more weights.
        assert len(sigs[2]) < len(sigs[4]) < len(sigs[8])

        widths = {
            f: (lambda f=f: create_bfcnn_denoiser(
                input_shape=(32, 32, 1), num_blocks=2, filters=f))
            for f in (16, 32, 64)
        }
        wsigs = assert_structural_knob_changes_weights(widths, knob="filters")
        # Same layer count, strictly wider tensors.
        assert len(wsigs[16]) == len(wsigs[32]) == len(wsigs[64])
        assert wsigs[16][0][-1] * 2 == wsigs[32][0][-1] == wsigs[64][0][-1] // 2

    def test_different_activations(self):
        """The activation must reach the forward pass.

        A value knob: every activation gives the same weight shapes, so under a
        fixed seed the models hold bit-identical weights and any output
        difference is the activation and nothing else.
        """
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        builders = {
            a: (lambda a=a: create_bfcnn_denoiser(
                input_shape=(64, 64, 1), num_blocks=2, filters=32, activation=a))
            for a in ('relu', 'elu', 'swish', 'gelu')
        }
        deltas = assert_value_knob_changes_output(
            builders, test_input, knob="activation",
        )
        assert min(deltas.values()) > 1e-5

    def test_different_kernel_sizes(self):
        """``kernel_size`` must reach the convolution kernels.

        Structural: the kernel's spatial extent is a weight dimension, so the
        signature carries the claim directly.
        """
        builders = {
            k: (lambda k=k: create_bfcnn_denoiser(
                input_shape=(64, 64, 1), num_blocks=2, filters=32, kernel_size=k))
            for k in (1, 3, 5, 7)
        }
        sigs = assert_structural_knob_changes_weights(builders, knob="kernel_size")
        for k, sig in sigs.items():
            # bfcnn.py fixes the stem at `initial_kernel_size=5` (line 152) and
            # the output projection at 1x1 (line 177); `kernel_size` governs the
            # residual blocks only, so those two extents are always present.
            spatial = {s[:2] for s in sig if len(s) == 4}
            assert spatial == {(5, 5), (k, k), (1, 1)}, (
                f"kernel_size={k} produced conv extents {sorted(spatial)}"
            )

    @pytest.mark.parametrize("variant", ["tiny", "small", "base", "large", "xlarge"])
    def test_all_variants(self, variant):
        """Test all predefined model variants."""
        input_shape = (64, 64, 1)
        model = create_bfcnn_variant(variant, input_shape)

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))
        assert model.name == f'bfcnn_{variant}'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])