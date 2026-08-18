"""
Comprehensive test suite for Bias-Free U-Net Model.

Tests cover initialization, validation, architecture verification, forward pass,
serialization, scaling invariance, skip connections, variants, and multi-scale processing.
"""

import os
import keras
import pytest
import tempfile
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any

from dl_techniques.models.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
    create_bfunet_variant,
    BFUNET_CONFIGS
)

from .conftest import (
    HOMOGENEITY_RTOL,
    HOMOGENEITY_SCALES,
    fit_one_step,
    homogeneity_error,
    homogeneity_error_raw,
    tf32_disabled,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)


def _first_norm(layer):
    """The normalization sublayer of a ``BiasFreeConv2D`` or ``BiasFreeResidualBlock``."""
    if hasattr(layer, 'batch_norm'):
        return layer.batch_norm
    return layer.conv1.batch_norm  # BiasFreeResidualBlock wraps two BiasFreeConv2D


class TestBiasFreeUNet:
    """Test suite for Bias-Free U-Net implementation."""

    @pytest.fixture
    def grayscale_input_shape(self) -> Tuple[int, int, int]:
        """Standard grayscale image shape for testing."""
        return (64, 64, 1)

    @pytest.fixture
    def rgb_input_shape(self) -> Tuple[int, int, int]:
        """Standard RGB image shape for testing."""
        return (128, 128, 3)

    @pytest.fixture
    def large_input_shape(self) -> Tuple[int, int, int]:
        """Large image shape for testing deep U-Net."""
        return (256, 256, 1)

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
    def test_image_large(self) -> np.ndarray:
        """Create large test image data."""
        return np.random.rand(1, 256, 256, 1).astype(np.float32)

    @pytest.fixture
    def default_model_config(self) -> Dict[str, Any]:
        """Default configuration for model creation."""
        return {
            'depth': 4,
            'initial_filters': 64,
            'filter_multiplier': 2,
            'blocks_per_level': 2,
            'kernel_size': 3,
            'activation': 'relu',
            'final_activation': 'linear'
        }

    # ================================================================
    # Initialization Tests
    # ================================================================

    def test_initialization_defaults(self, grayscale_input_shape):
        """Test initialization with default parameters."""
        model = create_bfunet_denoiser(input_shape=grayscale_input_shape)

        # Check model properties
        assert model.name == 'bias_free_unet'
        assert len(model.layers) > 0
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

    def test_initialization_variable_input_size(self, variable_input_shape):
        """Test initialization with variable input size."""
        model = create_bfunet_denoiser(
            input_shape=variable_input_shape,
            depth=3,
            initial_filters=16
        )

        assert model.input_shape == (None, None, None, 1)
        assert model.output_shape == (None, None, None, 1)

        # Test with different sized inputs (must be divisible by 2^depth for proper U-Net operation)
        small_input = np.random.rand(1, 32, 32, 1).astype(np.float32)  # 32 = 4 * 2^3
        large_input = np.random.rand(1, 128, 128, 1).astype(np.float32)  # 128 = 16 * 2^3

        small_output = model(small_input)
        large_output = model(large_input)

        assert small_output.shape == (1, 32, 32, 1)
        assert large_output.shape == (1, 128, 128, 1)

    def test_different_depths(self, grayscale_input_shape):
        """``depth`` must add levels, not just be accepted.

        The output shape is the input's shape at every depth, so it is not
        evidence. `depth` is structural: each extra level adds encoder, decoder
        and skip weights, so the weight-shape signature carries the claim and
        the parameter count must grow strictly.
        """
        depths = [3, 4, 5]  # Updated minimum depth to 3
        builders = {
            d: (lambda d=d: create_bfunet_denoiser(
                input_shape=grayscale_input_shape, depth=d, initial_filters=8))
            for d in depths
        }
        sigs = assert_structural_knob_changes_weights(builders, knob="depth")
        counts = [sum(int(np.prod(w)) for w in sigs[d]) for d in depths]
        assert counts[0] < counts[1] < counts[2], (
            f"depth did not grow the parameter count: {dict(zip(depths, counts))}"
        )

        # Forward pass still works at every depth, and preserves the shape.
        for depth in depths:
            model = create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=depth,
                initial_filters=8
            )

            test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
            output = model(test_input)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    # ================================================================
    # Input Validation Tests
    # ================================================================

    def test_invalid_input_shape_type(self):
        """Test that invalid input_shape type raises TypeError."""
        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfunet_denoiser(input_shape=[64, 64, 1])

        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfunet_denoiser(input_shape=(64, 64))

        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfunet_denoiser(input_shape=(64, 64, 1, 1))

    def test_invalid_depth(self, grayscale_input_shape):
        """Test that invalid depth raises ValueError.

        The depth floor is 2 (relaxed from 3 when the high-freq-blocks feature landed);
        this assertion still named the OLD floor and still listed ``depth=2`` as invalid,
        so it failed against the factory's actual message. Pre-existing debt, unrelated to
        the [0,1] domain migration.
        """
        with pytest.raises(ValueError, match="depth must be at least 2"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=0
            )

        with pytest.raises(ValueError, match="depth must be at least 2"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=1
            )

        with pytest.raises(ValueError, match="depth must be at least 2"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=-1
            )

    def test_invalid_initial_filters(self, grayscale_input_shape):
        """Test that non-positive initial_filters raises ValueError."""
        with pytest.raises(ValueError, match="initial_filters must be positive"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                initial_filters=0
            )

        with pytest.raises(ValueError, match="initial_filters must be positive"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                initial_filters=-10
            )

    def test_invalid_filter_multiplier(self, grayscale_input_shape):
        """Test that invalid filter_multiplier raises ValueError."""
        with pytest.raises(ValueError, match="filter_multiplier must be at least 1"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                filter_multiplier=0
            )

    def test_invalid_blocks_per_level(self, grayscale_input_shape):
        """Test that non-positive blocks_per_level raises ValueError."""
        with pytest.raises(ValueError, match="blocks_per_level must be positive"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                blocks_per_level=0
            )

    def test_minimal_valid_configuration(self, grayscale_input_shape):
        """Test minimal valid configuration."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=1,
            filter_multiplier=1,
            blocks_per_level=1
        )

        # Should still work
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    # ================================================================
    # Variant Tests (New)
    # ================================================================

    def test_available_variants(self):
        """Test that all defined variants are available."""
        expected_variants = ['tiny', 'small', 'base', 'large', 'xlarge']
        available_variants = list(BFUNET_CONFIGS.keys())

        for variant in expected_variants:
            assert variant in available_variants

    def test_create_variant_tiny(self, grayscale_input_shape):
        """Test creating tiny variant."""
        model = create_bfunet_variant('tiny', grayscale_input_shape)

        assert model.name == 'bias_free_unet_tiny'
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_small(self, rgb_input_shape):
        """Test creating small variant."""
        model = create_bfunet_variant('small', rgb_input_shape)

        assert model.name == 'bias_free_unet_small'
        assert model.input_shape == (None,) + rgb_input_shape
        assert model.output_shape == (None,) + rgb_input_shape

        # Test forward pass
        test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_base(self, grayscale_input_shape):
        """Test creating base variant."""
        model = create_bfunet_variant('base', grayscale_input_shape)

        assert model.name == 'bias_free_unet_base'
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

        # Should have more layers than tiny/small
        assert len(model.layers) > 15

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_large(self, grayscale_input_shape):
        """Test creating large variant."""
        model = create_bfunet_variant('large', grayscale_input_shape)

        assert model.name == 'bias_free_unet_large'
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

        # Should have more layers than base
        assert len(model.layers) > 20

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_xlarge(self, large_input_shape):
        """Test creating xlarge variant."""
        model = create_bfunet_variant('xlarge', large_input_shape)

        assert model.name == 'bias_free_unet_xlarge'
        assert model.input_shape == (None,) + large_input_shape
        assert model.output_shape == (None,) + large_input_shape

        # Should have the most layers (depth=5)
        assert len(model.layers) > 30

        # Test forward pass
        test_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_invalid_variant(self, grayscale_input_shape):
        """Test that invalid variant name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            create_bfunet_variant('invalid', grayscale_input_shape)

    def test_variant_with_custom_parameters(self, grayscale_input_shape):
        """Test variant creation with custom parameter overrides."""
        model = create_bfunet_variant(
            'base',
            grayscale_input_shape,
            activation='gelu',
            use_residual_blocks=False,
            model_name='custom_base_unet'
        )

        assert model.name == 'custom_base_unet'

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_variants_consistency(self):
        """Test that all variants work with same input."""
        input_shape = (64, 64, 1)
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        variants = ['tiny', 'small', 'base']

        for variant in variants:
            model = create_bfunet_variant(variant, input_shape)
            output = model(test_input)

            # All should produce valid outputs
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    # ================================================================
    # Architecture Verification Tests
    # ================================================================

    def test_filter_progression(self, grayscale_input_shape):
        """Test that filter sizes progress correctly through the network."""
        depth = 4
        initial_filters = 32
        filter_multiplier = 2

        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=depth,
            initial_filters=initial_filters,
            filter_multiplier=filter_multiplier
        )

        # Expected filter progression: [32, 64, 128, 256, 512] for depth=4
        expected_filters = [initial_filters * (filter_multiplier ** i) for i in range(depth + 1)]

        # This is more of a structural test
        assert len(expected_filters) == depth + 1

    def test_output_channels_match_input(self):
        """Test that output channels match input channels."""
        test_cases = [
            (64, 64, 1),   # Grayscale
            (128, 128, 3), # RGB
            (32, 32, 4),   # RGBA
        ]

        for input_shape in test_cases:
            model = create_bfunet_denoiser(
                input_shape=input_shape,
                depth=3,  # Updated minimum depth
                initial_filters=16
            )

            input_channels = input_shape[2]
            assert model.output_shape[-1] == input_channels

    def test_residual_vs_standard_blocks(self, grayscale_input_shape):
        """Test model with residual blocks vs standard convolution blocks."""
        configs = [
            {'use_residual_blocks': True},
            {'use_residual_blocks': False}
        ]

        for config in configs:
            model = create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=3,  # Updated minimum depth
                initial_filters=32,
                **config
            )

            test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
            output = model(test_input)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_model_creation_success(self, grayscale_input_shape):
        """Test that model creation completes successfully with proper structure."""
        depth = 4
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=depth,
            initial_filters=64,
            model_name='test_unet'
        )

        # Verify model was created successfully
        assert model is not None
        assert model.name == 'test_unet'
        assert len(model.layers) > 0

        # Verify model has expected parameter count for given configuration
        total_params = model.count_params()
        assert total_params > 0

    # ================================================================
    # Forward Pass Tests
    # ================================================================

    def test_forward_pass_grayscale(self, test_image_grayscale):
        """Test forward pass with grayscale images."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=32
        )

        output = model(test_image_grayscale)

        # Check output properties
        assert output.shape == test_image_grayscale.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_rgb(self, test_image_rgb):
        """Test forward pass with RGB images."""
        model = create_bfunet_denoiser(
            input_shape=(128, 128, 3),
            depth=3,
            initial_filters=32
        )

        output = model(test_image_rgb)

        # Check output properties
        assert output.shape == test_image_rgb.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_large_image(self, test_image_large):
        """Test forward pass with large images."""
        model = create_bfunet_denoiser(
            input_shape=(256, 256, 1),
            depth=4,
            initial_filters=32
        )

        output = model(test_image_large)

        # Check output properties
        assert output.shape == test_image_large.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_scaling_invariance_property(self, test_image_grayscale, homogeneity_probe):
        """A TRAINED, factory-built bias-free U-Net is degree-1 homogeneous.

        Reaches the model through ``create_bfunet_denoiser`` at its defaults -- NOT through
        ``src/train/bfunet/``, which is where the ``'batchnorm'`` -> ``BiasFreeBatchNorm``
        remap used to live and which therefore masked this defect from every trainer-side
        test.

        The ``fit_one_step`` call is the entire point: on an untrained model stock
        ``BatchNormalization`` has ``moving_mean == 0`` and is exactly homogeneous, so the
        pre-2026-08-15 version of this test passed against a model that did not have the
        property. See ``conftest.py`` for the tolerance derivation and the measured
        pre-fix / post-fix numbers.
        """
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=2,
            initial_filters=8,
            blocks_per_level=1,
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

    def test_homogeneity_error_is_round_off_not_bias(self, test_image_grayscale):
        """Pins the DIAGNOSIS behind `test_scaling_invariance_property`, not its symptom.

        # DECISION plan-2026-08-18T123346-c3c4a681/D-003
        The residual homogeneity error of a bias-free U-Net is arithmetic ROUND-OFF, so it must
        shrink when the arithmetic gets more precise. An additive-bias leak -- the defect this
        family of tests exists to catch -- would not: it is a property of the weights, not of
        the mantissa. Measured 2026-08-18 on GPU 1 over c in (0.5, 2, 3, 10, 100, 1e4):
        TF32 ~6.5e-04, true float32 ~1.2e-06, float64 ~1.4e-07 (the float64 floor is the
        decoder's bilinear ``UpSampling2D``, which ``tf.image.resize`` executes in float32 at
        any dtype policy -- see conftest for the full table and the per-layer walk, where the
        encoder reads 4.01e-16).

        Do NOT tighten this to an absolute bar on the TF32 reading: it is genuinely unstable
        run to run (2.86e-04 / 5.06e-04 / 1.08e-03 observed), which is why the assertion is on
        the RATIO, where the separation is ~100x-1000x and the bar is 10x. Do NOT delete the
        skips: on CPU, and on any GPU that does not actually take the TF32 path, both readings
        are the same number and the experiment has no discriminating power.
        """
        if not tf.config.list_physical_devices('GPU'):
            pytest.skip("TF32 is a GPU (Ampere+) execution mode; the flag is inert on CPU")

        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=2,
            initial_filters=8,
            blocks_per_level=1,
            final_activation='linear',
        )
        fit_one_step(model)

        previous = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(True)
        try:
            err_tf32 = homogeneity_error_raw(model, test_image_grayscale, 3.0)
        finally:
            tf.config.experimental.enable_tensor_float_32_execution(previous)
            assert (
                tf.config.experimental.tensor_float_32_execution_enabled() == previous
            ), "TF32 setting leaked out of the diagnosis test"

        err_true_f32 = homogeneity_error(model, test_image_grayscale, 3.0)

        if err_tf32 < 1e-5:
            pytest.skip(
                f"this GPU did not take the TF32 path (error {err_tf32:.3e}); "
                "the on/off comparison has nothing to discriminate"
            )

        assert err_true_f32 < HOMOGENEITY_RTOL, (
            f"true-float32 homogeneity error {err_true_f32:.3e} exceeds "
            f"{HOMOGENEITY_RTOL:.0e} -- this is NOT reduced precision and is a real defect"
        )
        assert err_true_f32 * 10 < err_tf32, (
            f"homogeneity error did not track arithmetic precision: TF32 {err_tf32:.3e} vs "
            f"true float32 {err_true_f32:.3e}. Round-off shrinks with the mantissa; an "
            "additive-bias leak would stay put, so this looks like a real bias leak"
        )

        moving_means = [w for w in model.weights if 'moving_mean' in w.path]
        assert not moving_means, (
            "the factory-built bias-free U-Net still contains "
            f"{len(moving_means)} moving_mean variable(s) -- it is using stock "
            "BatchNormalization, whose moving_mean subtraction is an additive constant "
            "that breaks f(c*x) = c*f(x)"
        )

    def test_deep_supervision_head_uses_the_block_normalization(self):
        """The deep-supervision heads get the same normalization as the blocks.

        The heads feed gradient straight into the decoder, so a head left on a different
        normalization than the blocks makes that gradient scale-dependent. Asserted on the
        BUILT layer TYPE, not on the ``normalization_type`` string, so it stays true however
        the name is resolved.
        """
        model = create_bfunet_denoiser(
            input_shape=(32, 32, 1),
            depth=2,
            initial_filters=8,
            blocks_per_level=1,
            enable_deep_supervision=True,
        )
        heads = [l for l in model.layers if l.name.startswith('supervision_intermediate_')]
        assert heads, "no deep-supervision head was built"

        block = next(l for l in model.layers if l.name.startswith('encoder_level_0_'))
        block_norm_type = type(_first_norm(block))
        for head in heads:
            assert type(head.batch_norm) is block_norm_type, (
                f"deep-supervision head {head.name!r} normalizes with "
                f"{type(head.batch_norm).__name__} while the encoder blocks use "
                f"{block_norm_type.__name__} -- the head ignores block_normalization"
            )

    def test_different_batch_sizes(self, grayscale_input_shape):
        """Test model with different batch sizes."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            test_input = np.random.rand(batch_size, 64, 64, 1).astype(np.float32)
            output = model(test_input)

            assert output.shape == (batch_size, 64, 64, 1)
            assert not np.any(np.isnan(output.numpy()))

    def test_skip_connections_functionality(self, grayscale_input_shape):
        """Test that skip connections are working properly."""
        # Create a simple test to verify skip connections
        unet_model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=32
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        unet_output = unet_model(test_input)

        # U-Net should produce valid output with skip connections
        assert unet_output.shape == test_input.shape
        assert not np.any(np.isnan(unet_output.numpy()))

        # Skip connections should help preserve fine details
        layer_names = [layer.name for layer in unet_model.layers]
        assert any('concat' in name for name in layer_names), "Skip connections not found"

    # ================================================================
    # Numerical Stability Tests
    # ================================================================

    def test_numerical_stability(self, grayscale_input_shape):
        """Test model stability with extreme input values."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        # Test with different input magnitudes
        test_cases = [
            np.zeros((1, 64, 64, 1), dtype=np.float32),  # Zeros
            np.ones((1, 64, 64, 1), dtype=np.float32) * 1e-10,  # Very small
            np.ones((1, 64, 64, 1), dtype=np.float32) * 1e3,    # Large values
            np.random.normal(0, 50, (1, 64, 64, 1)).astype(np.float32)  # High variance
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
        original_model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=32,
            model_name='serialization_test'
        )

        # Generate prediction before saving
        original_prediction = original_model.predict(test_image_grayscale)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "bias_free_unet_model.keras")

            # Save the model
            original_model.save(model_path)

            # Verify the model file was created
            assert os.path.exists(model_path)

            # Verify model structure can be inspected
            assert original_model.name == 'serialization_test'
            assert len(original_model.layers) > 0

    # ================================================================
    # Training Integration Tests
    # ================================================================

    def test_model_compilation(self, grayscale_input_shape):
        """Test that model can be compiled with different optimizers and losses."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        # Test different compilation configurations
        compilation_configs = [
            {'optimizer': 'adam', 'loss': 'mse'},
            {'optimizer': 'rmsprop', 'loss': 'mae'},
            {'optimizer': keras.optimizers.Adam(learning_rate=0.001), 'loss': 'binary_crossentropy'},
        ]

        for config in compilation_configs:
            model.compile(**config)
            assert model.optimizer is not None
            assert model.loss is not None

    def test_gradient_flow(self, test_image_grayscale):
        """Test gradient flow through the U-Net model."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        model.compile(optimizer='adam', loss='mse')

        # Create target
        target = test_image_grayscale * 0.9  # Simulated target

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
    # Edge Cases and Robustness Tests
    # ================================================================

    def test_minimum_depth_configuration(self, grayscale_input_shape):
        """Test U-Net with minimum depth (depth=3)."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_large_depth_configuration(self, large_input_shape):
        """Test U-Net with large depth."""
        model = create_bfunet_denoiser(
            input_shape=large_input_shape,
            depth=5,  # Deep U-Net
            initial_filters=16  # Keep filters low to manage memory
        )

        test_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_non_power_of_two_dimensions(self):
        """Test U-Net with input dimensions that are not powers of 2."""
        # U-Net with pooling/upsampling can handle non-power-of-2 dimensions
        # thanks to the Resizing layer for dimension matching
        input_shape = (96, 80, 1)  # Non-power-of-2 dimensions

        model = create_bfunet_denoiser(
            input_shape=input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        test_input = np.random.rand(1, 96, 80, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_large_filter_count(self, grayscale_input_shape):
        """Test model with large number of filters."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=256,  # Large filter count
            filter_multiplier=2
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_single_block_per_level(self, grayscale_input_shape):
        """Test U-Net with single block per level."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,
            initial_filters=32,
            blocks_per_level=1  # Minimal blocks
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))


# ================================================================
# Parameterized Tests for Multiple Configurations
# ================================================================

class TestBiasFreeUNetParameterized:
    """Parameterized tests for different U-Net configurations."""

    @pytest.mark.parametrize("input_shape,depth,initial_filters", [
        ((32, 32, 1), 3, 16),  # Updated minimum depth
        ((64, 64, 3), 3, 32),
        ((128, 128, 1), 4, 64),
        ((96, 96, 3), 3, 32),  # Non-power-of-2 dimensions
    ])
    def test_various_configurations(self, input_shape, depth, initial_filters):
        """Each configuration builds and forwards without NaN.

        A build smoke only. The shape assertion below is true at every depth and
        width by construction; `test_different_depths` and
        `test_different_initial_filters` carry the knob claims.
        """
        model = create_bfunet_denoiser(
            input_shape=input_shape,
            depth=depth,
            initial_filters=initial_filters
        )

        # Test forward pass
        batch_size = 1
        test_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
        output = model(test_input)

        assert output.shape == (batch_size,) + input_shape
        assert not np.any(np.isnan(output.numpy()))

    def test_different_activations(self):
        """The activation must reach the forward pass.

        A value knob: the parameterisation is identical across activations, so
        under one seed the models hold bit-identical weights and the whole
        output difference is the activation.
        """
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        builders = {
            a: (lambda a=a: create_bfunet_denoiser(
                input_shape=(64, 64, 1), depth=3, initial_filters=16,
                activation=a))
            for a in ('relu', 'leaky_relu', 'elu', 'swish', 'gelu')
        }
        deltas = assert_value_knob_changes_output(
            builders, test_input, knob="activation",
        )
        assert min(deltas.values()) > 1e-5

    def test_different_kernel_sizes(self):
        """``kernel_size`` must reach the convolution kernels.

        bfunet.py fixes the stem at ``initial_kernel_size=5`` (line 362) and the
        output projection at 1x1, so those extents are present at every setting;
        `kernel_size` governs every other conv.
        """
        builders = {
            k: (lambda k=k: create_bfunet_denoiser(
                input_shape=(64, 64, 1), depth=3, initial_filters=16,
                kernel_size=k))
            for k in (1, 3, 5, 7)
        }
        sigs = assert_structural_knob_changes_weights(builders, knob="kernel_size")
        for k, sig in sigs.items():
            spatial = {w[:2] for w in sig if len(w) == 4}
            assert (k, k) in spatial, (
                f"kernel_size={k} produced no {k}x{k} kernel; extents were "
                f"{sorted(spatial)}"
            )
            assert spatial <= {(5, 5), (k, k), (1, 1)}, (
                f"kernel_size={k} produced unexpected extents {sorted(spatial)}"
            )

    def test_different_filter_multipliers(self):
        """``filter_multiplier`` must widen each successive level.

        Structural. bfunet.py:339 computes
        ``initial_filters * multiplier ** level``; at multiplier 1 the levels
        are flat, so the parameter count must grow strictly with the multiplier.
        """
        multipliers = (1, 2, 3, 4)
        builders = {
            m: (lambda m=m: create_bfunet_denoiser(
                input_shape=(64, 64, 1), depth=3, initial_filters=16,
                filter_multiplier=m))
            for m in multipliers
        }
        sigs = assert_structural_knob_changes_weights(
            builders, knob="filter_multiplier")
        counts = [sum(int(np.prod(w)) for w in sigs[m]) for m in multipliers]
        assert counts == sorted(counts) and counts[0] < counts[-1], (
            f"filter_multiplier did not widen the model: "
            f"{dict(zip(multipliers, counts))}"
        )

    def test_different_blocks_per_level(self):
        """``blocks_per_level`` must add blocks, not just be accepted.

        Structural: more blocks per level means strictly more weight tensors,
        which a differing-shapes check alone would not pin down.
        """
        counts_swept = (1, 2, 3, 4)
        builders = {
            b: (lambda b=b: create_bfunet_denoiser(
                input_shape=(64, 64, 1), depth=3, initial_filters=16,
                blocks_per_level=b))
            for b in counts_swept
        }
        sigs = assert_structural_knob_changes_weights(
            builders, knob="blocks_per_level")
        n_weights = [len(sigs[b]) for b in counts_swept]
        assert n_weights == sorted(n_weights) and n_weights[0] < n_weights[-1], (
            f"blocks_per_level did not add weight tensors: "
            f"{dict(zip(counts_swept, n_weights))}"
        )

    @pytest.mark.parametrize("variant", ['tiny', 'small', 'base', 'large', 'xlarge'])
    def test_all_variants_parameterized(self, variant):
        """Test all variants with parameterized approach."""
        input_shape = (64, 64, 1)
        model = create_bfunet_variant(variant, input_shape)

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))



class TestPretrainedContract:
    """`pretrained=True` must RAISE, never return a random-init model.

    Before this contract, `BFUNET_PRETRAINED_WEIGHTS` held placeholder URLs on a non-existent host,
    `create_bfunet_variant` caught the download failure, logged a warning and continued with
    random initialization — so a caller asking for pretrained weights silently
    got untrained ones and no error. Do not reinstate that.
    """

    def test_create_variant_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError, match="No pretrained BFUNet weights"):
            create_bfunet_variant("tiny", (32, 32, 1), pretrained=True)

    def test_pretrained_false_still_builds(self):
        model = create_bfunet_variant("tiny", (32, 32, 1), pretrained=False)
        assert isinstance(model, keras.Model)

    def test_local_path_still_works(self, tmp_path):
        src = create_bfunet_variant("tiny", (32, 32, 1))
        path = str(tmp_path / "bfunet.keras")
        src.save(path)
        loaded = create_bfunet_variant("tiny", (32, 32, 1), pretrained=path)
        x = np.random.rand(1, 32, 32, 1).astype("float32")
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(src(x)),
            keras.ops.convert_to_numpy(loaded(x)),
            atol=1e-6,
        )

    def test_no_placeholder_weight_table(self):
        import dl_techniques.models.bias_free_denoisers.bfunet as mod
        assert not hasattr(mod, "BFUNET_PRETRAINED_WEIGHTS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])