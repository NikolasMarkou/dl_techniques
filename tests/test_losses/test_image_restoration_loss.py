"""
Comprehensive Test Suite for DarkIR Loss Functions.

Tests cover functionality, edge cases, serialization, and numerical correctness
for all loss functions in the darkir_losses_refined module.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.image_restoration_loss import (
    CharbonnierLoss,
    FrequencyLoss,
    EdgeLoss,
    VGGLoss,
    EnhanceLoss,
    DarkIRCompositeLoss
)


class TestCharbonnierLoss:
    """Test suite for CharbonnierLoss."""

    def test_initialization(self) -> None:
        """Test loss initialization with default and custom parameters."""
        # Default initialization
        loss = CharbonnierLoss()
        assert loss.epsilon == 1e-3
        assert loss.reduction == "sum_over_batch_size"

        # Custom initialization
        loss_custom = CharbonnierLoss(epsilon=1e-2, reduction="sum", name="custom_charb")
        assert loss_custom.epsilon == 1e-2
        assert loss_custom.reduction == "sum"
        assert loss_custom.name == "custom_charb"

    def test_forward_pass(self) -> None:
        """Test forward pass with synthetic data."""
        # Use reduction='none' to check per-sample shape
        loss_fn = CharbonnierLoss(reduction='none')

        # Create synthetic data
        y_true = keras.random.normal((4, 32, 32, 3), seed=42)
        y_pred = keras.random.normal((4, 32, 32, 3), seed=43)

        # Compute loss
        loss_value = loss_fn(y_true, y_pred)

        # Verify output shape and properties
        assert keras.ops.shape(loss_value) == (4,)
        assert keras.ops.all(loss_value >= 0.0)  # Loss should be non-negative

    def test_identical_inputs(self) -> None:
        """Test that identical inputs produce near-zero loss."""
        loss_fn = CharbonnierLoss(epsilon=1e-6)

        y_true = keras.random.normal((2, 16, 16, 3), seed=42)
        y_pred = y_true  # Identical

        loss_value = loss_fn(y_true, y_pred)

        # Loss should be very small (epsilon-dependent)
        assert keras.ops.all(loss_value < 1e-5)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow correctly through the loss."""
        import tensorflow as tf

        loss_fn = CharbonnierLoss()

        y_true = keras.random.normal((2, 8, 8, 3), seed=42)
        y_pred_var = tf.Variable(keras.random.normal((2, 8, 8, 3), seed=43))

        with tf.GradientTape() as tape:
            loss = loss_fn(y_true, y_pred_var)
            total_loss = keras.ops.sum(loss)

        gradients = tape.gradient(total_loss, y_pred_var)

        # Gradients should exist and be non-zero
        assert gradients is not None
        assert keras.ops.any(keras.ops.abs(gradients) > 0)

    def test_serialization(self) -> None:
        """Test loss serialization and deserialization."""
        loss = CharbonnierLoss(epsilon=1e-4, reduction="none")

        # Serialize
        config = loss.get_config()

        # Deserialize
        loss_restored = CharbonnierLoss.from_config(config)

        # Verify parameters
        assert loss_restored.epsilon == loss.epsilon
        assert loss_restored.reduction == loss.reduction

    def test_different_epsilons(self) -> None:
        """Test behavior with different epsilon values."""
        y_true = keras.ops.ones((2, 8, 8, 3))
        y_pred = keras.ops.zeros((2, 8, 8, 3))

        loss_small = CharbonnierLoss(epsilon=1e-6)
        loss_large = CharbonnierLoss(epsilon=1e-1)

        val_small = keras.ops.mean(loss_small(y_true, y_pred))
        val_large = keras.ops.mean(loss_large(y_true, y_pred))

        # Larger epsilon should produce slightly larger loss for same error
        assert val_large >= val_small


class TestFrequencyLoss:
    """Test suite for FrequencyLoss."""

    def test_initialization(self) -> None:
        """Test loss initialization."""
        loss = FrequencyLoss()
        assert loss.loss_weight == 1.0
        assert loss.norm == 'l1'

        loss_l2 = FrequencyLoss(loss_weight=0.5, norm='l2')
        assert loss_l2.loss_weight == 0.5
        assert loss_l2.norm == 'l2'

    def test_forward_pass(self) -> None:
        """Test forward pass with synthetic data."""
        # Use reduction='none' to check per-sample shape
        loss_fn = FrequencyLoss(norm='l1', reduction='none')

        y_true = keras.random.uniform((2, 32, 32, 3), seed=42)
        y_pred = keras.random.uniform((2, 32, 32, 3), seed=43)

        loss_value = loss_fn(y_true, y_pred)

        # Verify shape and properties
        assert keras.ops.shape(loss_value) == (2,)
        assert keras.ops.all(loss_value >= 0.0)

    def test_l1_vs_l2_norm(self) -> None:
        """Test that L1 and L2 norms produce different losses."""
        y_true = keras.random.uniform((2, 16, 16, 3), seed=42)
        y_pred = keras.random.uniform((2, 16, 16, 3), seed=43)

        loss_l1 = FrequencyLoss(norm='l1')
        loss_l2 = FrequencyLoss(norm='l2')

        val_l1 = keras.ops.mean(loss_l1(y_true, y_pred))
        val_l2 = keras.ops.mean(loss_l2(y_true, y_pred))

        # L1 and L2 should produce different values
        assert not keras.ops.isclose(val_l1, val_l2, rtol=1e-3)

    def test_identical_inputs(self) -> None:
        """Test that identical inputs produce zero loss."""
        loss_fn = FrequencyLoss()

        y_true = keras.random.uniform((2, 16, 16, 3), seed=42)
        y_pred = y_true

        loss_value = loss_fn(y_true, y_pred)

        # Should be very close to zero
        assert keras.ops.all(loss_value < 1e-5)

    def test_loss_weight_scaling(self) -> None:
        """Test that loss_weight correctly scales the output."""
        y_true = keras.random.uniform((2, 16, 16, 3), seed=42)
        y_pred = keras.random.uniform((2, 16, 16, 3), seed=43)

        loss_1 = FrequencyLoss(loss_weight=1.0)
        loss_2 = FrequencyLoss(loss_weight=2.0)

        val_1 = loss_1(y_true, y_pred)
        val_2 = loss_2(y_true, y_pred)

        # val_2 should be approximately 2x val_1
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(val_2),
            keras.ops.convert_to_numpy(val_1) * 2.0,
            rtol=1e-5, atol=1e-5,
            err_msg="Loss weight scaling failed"
        )

    def test_serialization(self) -> None:
        """Test loss serialization."""
        loss = FrequencyLoss(loss_weight=0.7, norm='l2')

        config = loss.get_config()
        loss_restored = FrequencyLoss.from_config(config)

        assert loss_restored.loss_weight == loss.loss_weight
        assert loss_restored.norm == loss.norm

    def test_multichannel(self) -> None:
        """Test with different channel counts."""
        loss_fn = FrequencyLoss()

        # Single channel
        y_true_1ch = keras.random.uniform((2, 16, 16, 1), seed=42)
        y_pred_1ch = keras.random.uniform((2, 16, 16, 1), seed=43)
        loss_1ch = loss_fn(y_true_1ch, y_pred_1ch)

        # Three channels
        y_true_3ch = keras.random.uniform((2, 16, 16, 3), seed=42)
        y_pred_3ch = keras.random.uniform((2, 16, 16, 3), seed=43)
        loss_3ch = loss_fn(y_true_3ch, y_pred_3ch)

        # Both should produce valid losses
        assert keras.ops.all(loss_1ch >= 0.0)
        assert keras.ops.all(loss_3ch >= 0.0)


class TestEdgeLoss:
    """Test suite for EdgeLoss."""

    def test_initialization(self) -> None:
        """Test loss initialization."""
        loss = EdgeLoss()
        assert loss.loss_weight == 1.0
        assert loss.channels == 3
        # Kernel should be initialized in __init__
        assert loss.kernel is not None
        assert tuple(keras.ops.shape(loss.kernel)) == (5, 5, 3, 1)

        loss_custom = EdgeLoss(loss_weight=0.5, channels=1)
        assert loss_custom.loss_weight == 0.5
        assert loss_custom.channels == 1
        assert tuple(keras.ops.shape(loss_custom.kernel)) == (5, 5, 1, 1)

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        # Use reduction='none' for shape check
        loss_fn = EdgeLoss(channels=3, reduction='none')

        y_true = keras.random.uniform((2, 32, 32, 3), seed=42)
        y_pred = keras.random.uniform((2, 32, 32, 3), seed=43)

        loss_value = loss_fn(y_true, y_pred)

        assert keras.ops.shape(loss_value) == (2,)
        assert keras.ops.all(loss_value >= 0.0)

    def test_identical_inputs(self) -> None:
        """Test with identical inputs."""
        loss_fn = EdgeLoss()

        y_true = keras.random.uniform((2, 32, 32, 3), seed=42)
        y_pred = y_true

        loss_value = loss_fn(y_true, y_pred)

        # Should be very close to zero
        assert keras.ops.all(loss_value < 1e-6)

    def test_edge_sensitivity(self) -> None:
        """Test that loss is sensitive to edges."""
        loss_fn = EdgeLoss()

        # Smooth gradient
        y_true = keras.ops.linspace(0.0, 1.0, 32 * 32 * 3)
        y_true = keras.ops.reshape(y_true, (1, 32, 32, 3))

        # Sharp edge (step function)
        y_pred = keras.ops.concatenate([
            keras.ops.zeros((1, 32, 16, 3)),
            keras.ops.ones((1, 32, 16, 3))
        ], axis=2)

        loss_value = loss_fn(y_true, y_pred)

        # Loss should be significant due to edge difference
        assert keras.ops.all(loss_value > 0.001)

    def test_serialization(self) -> None:
        """Test loss serialization."""
        loss = EdgeLoss(loss_weight=0.8, channels=1)

        config = loss.get_config()
        loss_restored = EdgeLoss.from_config(config)

        assert loss_restored.loss_weight == loss.loss_weight
        assert loss_restored.channels == loss.channels

    def test_different_channels(self) -> None:
        """Test with different channel counts."""
        # Single channel
        loss_1ch = EdgeLoss(channels=1)
        y_true_1ch = keras.random.uniform((2, 32, 32, 1), seed=42)
        y_pred_1ch = keras.random.uniform((2, 32, 32, 1), seed=43)
        val_1ch = loss_1ch(y_true_1ch, y_pred_1ch)

        # RGB
        loss_3ch = EdgeLoss(channels=3)
        y_true_3ch = keras.random.uniform((2, 32, 32, 3), seed=42)
        y_pred_3ch = keras.random.uniform((2, 32, 32, 3), seed=43)
        val_3ch = loss_3ch(y_true_3ch, y_pred_3ch)

        assert keras.ops.all(val_1ch >= 0.0)
        assert keras.ops.all(val_3ch >= 0.0)


class TestVGGLoss:
    """Test suite for VGGLoss."""

    def test_initialization(self) -> None:
        """Test loss initialization."""
        loss = VGGLoss()
        assert loss.loss_weight == 1.0
        # Model should be initialized in __init__
        assert loss.vgg_model is not None

        loss_custom = VGGLoss(loss_weight=0.5)
        assert loss_custom.loss_weight == 0.5

    def test_forward_pass(self) -> None:
        """Test forward pass with proper input size."""
        # Use reduction='none' for shape check
        loss_fn = VGGLoss(reduction='none')

        # VGG requires larger input (min ~32x32, but use 64x64 for stability)
        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        loss_value = loss_fn(y_true, y_pred)

        assert keras.ops.shape(loss_value) == (2,)
        assert keras.ops.all(loss_value >= 0.0)

    def test_identical_inputs(self) -> None:
        """Test with identical inputs."""
        loss_fn = VGGLoss()

        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = y_true

        loss_value = loss_fn(y_true, y_pred)

        # Should be very close to zero
        assert keras.ops.all(loss_value < 1e-5)

    def test_perceptual_sensitivity(self) -> None:
        """Test that loss captures perceptual differences."""
        loss_fn = VGGLoss()

        # Original image
        y_true = keras.random.uniform((1, 64, 64, 3), seed=42)

        # Small noise (pixel-level different but perceptually similar)
        y_pred_small_noise = y_true + keras.random.normal((1, 64, 64, 3), seed=43) * 0.01
        y_pred_small_noise = keras.ops.clip(y_pred_small_noise, 0.0, 1.0)

        # Large structural change
        y_pred_large_noise = keras.random.uniform((1, 64, 64, 3), seed=44)

        loss_small = loss_fn(y_true, y_pred_small_noise)
        loss_large = loss_fn(y_true, y_pred_large_noise)

        # Large structural change should have higher perceptual loss
        assert keras.ops.all(loss_large > loss_small)

    def test_serialization(self) -> None:
        """Test loss serialization."""
        loss = VGGLoss(loss_weight=0.7)

        config = loss.get_config()
        loss_restored = VGGLoss.from_config(config)

        assert loss_restored.loss_weight == loss.loss_weight

    def test_layer_weights(self) -> None:
        """Test that layer weights are properly initialized."""
        loss = VGGLoss()

        assert len(loss.layer_weights) == 5
        assert loss.layer_weights == [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]


class TestEnhanceLoss:
    """Test suite for EnhanceLoss."""

    def test_initialization(self) -> None:
        """Test loss initialization."""
        loss = EnhanceLoss()
        assert loss.loss_weight == 1.0
        assert loss.vgg_weight == 0.01
        assert loss.scale_factor == 8

        loss_custom = EnhanceLoss(loss_weight=0.5, vgg_weight=0.02, scale_factor=4)
        assert loss_custom.loss_weight == 0.5
        assert loss_custom.vgg_weight == 0.02
        assert loss_custom.scale_factor == 4

    def test_forward_pass_different_resolutions(self) -> None:
        """Test with high-res ground truth and low-res prediction."""
        loss_fn = EnhanceLoss()

        # High-resolution ground truth
        y_true = keras.random.uniform((2, 128, 128, 3), seed=42)

        # Low-resolution prediction (e.g., from intermediate layer)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        loss_value = loss_fn(y_true, y_pred)

        # Should handle resolution mismatch
        assert loss_value.shape == ()  # Scalar
        assert loss_value >= 0.0

    def test_forward_pass_same_resolution(self) -> None:
        """Test with matching resolutions."""
        loss_fn = EnhanceLoss()

        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert loss_value >= 0.0

    def test_identical_inputs(self) -> None:
        """Test with identical inputs."""
        loss_fn = EnhanceLoss()

        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = y_true

        loss_value = loss_fn(y_true, y_pred)

        # Should be very close to zero
        assert loss_value < 1e-4

    def test_reduction_modes(self) -> None:
        """Test different reduction modes."""
        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        loss_mean = EnhanceLoss(reduction="sum_over_batch_size")
        loss_sum = EnhanceLoss(reduction="sum")
        loss_none = EnhanceLoss(reduction="none")

        val_mean = loss_mean(y_true, y_pred)
        val_sum = loss_sum(y_true, y_pred)
        val_none = loss_none(y_true, y_pred)

        # Check shapes
        assert val_mean.shape == ()
        assert val_sum.shape == ()
        assert val_none.shape == (2,)

    def test_serialization(self) -> None:
        """Test loss serialization."""
        loss = EnhanceLoss(loss_weight=0.8, vgg_weight=0.03, scale_factor=2)

        config = loss.get_config()
        loss_restored = EnhanceLoss.from_config(config)

        assert loss_restored.loss_weight == loss.loss_weight
        assert loss_restored.vgg_weight == loss.vgg_weight
        assert loss_restored.scale_factor == loss.scale_factor


class TestDarkIRCompositeLoss:
    """Test suite for DarkIRCompositeLoss."""

    def test_initialization(self) -> None:
        """Test loss initialization."""
        loss = DarkIRCompositeLoss()
        assert loss.charbonnier_weight == 1.0
        assert loss.ssim_weight == 0.2
        assert loss.perceptual_weight == 0.0
        assert loss.vgg_loss is None  # Not created when weight is 0

        loss_with_vgg = DarkIRCompositeLoss(
            charbonnier_weight=0.8,
            ssim_weight=0.1,
            perceptual_weight=0.1
        )
        assert loss_with_vgg.charbonnier_weight == 0.8
        assert loss_with_vgg.ssim_weight == 0.1
        assert loss_with_vgg.perceptual_weight == 0.1
        assert loss_with_vgg.vgg_loss is not None

    def test_forward_pass_without_perceptual(self) -> None:
        """Test forward pass without perceptual loss."""
        loss_fn = DarkIRCompositeLoss(
            charbonnier_weight=1.0,
            ssim_weight=0.2,
            perceptual_weight=0.0
        )

        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert loss_value >= 0.0

    def test_forward_pass_with_perceptual(self) -> None:
        """Test forward pass with perceptual loss."""
        loss_fn = DarkIRCompositeLoss(
            charbonnier_weight=1.0,
            ssim_weight=0.2,
            perceptual_weight=0.1
        )

        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert loss_value >= 0.0

    def test_identical_inputs(self) -> None:
        """Test with identical inputs."""
        loss_fn = DarkIRCompositeLoss()

        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = y_true

        loss_value = loss_fn(y_true, y_pred)

        # Should be very close to zero (Charbonnier epsilon is 1e-3, so loss is exactly 1e-3)
        assert loss_value < 1.1e-3

    def test_component_weights(self) -> None:
        """Test that component weights affect final loss."""
        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        # Only Charbonnier
        loss_charb_only = DarkIRCompositeLoss(
            charbonnier_weight=1.0,
            ssim_weight=0.0,
            perceptual_weight=0.0
        )

        # Only SSIM
        loss_ssim_only = DarkIRCompositeLoss(
            charbonnier_weight=0.0,
            ssim_weight=1.0,
            perceptual_weight=0.0
        )

        # Balanced
        loss_balanced = DarkIRCompositeLoss(
            charbonnier_weight=0.5,
            ssim_weight=0.5,
            perceptual_weight=0.0
        )

        val_charb = loss_charb_only(y_true, y_pred)
        val_ssim = loss_ssim_only(y_true, y_pred)
        val_balanced = loss_balanced(y_true, y_pred)

        # All should be positive
        assert val_charb >= 0.0
        assert val_ssim >= 0.0
        assert val_balanced >= 0.0

        # Balanced should be different from individual components
        assert not keras.ops.isclose(val_balanced, val_charb, rtol=0.1)
        assert not keras.ops.isclose(val_balanced, val_ssim, rtol=0.1)

    def test_serialization(self) -> None:
        """Test loss serialization."""
        loss = DarkIRCompositeLoss(
            charbonnier_weight=0.7,
            ssim_weight=0.2,
            perceptual_weight=0.1,
            reduction="sum"
        )

        config = loss.get_config()
        loss_restored = DarkIRCompositeLoss.from_config(config)

        assert loss_restored.charbonnier_weight == loss.charbonnier_weight
        assert loss_restored.ssim_weight == loss.ssim_weight
        assert loss_restored.perceptual_weight == loss.perceptual_weight
        assert loss_restored.reduction == loss.reduction

    def test_ssim_component(self) -> None:
        """Test SSIM component behavior."""
        loss_fn = DarkIRCompositeLoss(
            charbonnier_weight=0.0,
            ssim_weight=1.0,
            perceptual_weight=0.0
        )

        # Identical images should have SSIM=1, thus (1-SSIM)=0
        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = y_true

        loss_identical = loss_fn(y_true, y_pred)

        # Different images should have lower SSIM, thus higher (1-SSIM)
        y_pred_diff = keras.random.uniform((2, 64, 64, 3), seed=43)
        loss_different = loss_fn(y_true, y_pred_diff)

        # Different images should have higher loss
        assert loss_different > loss_identical


class TestIntegration:
    """Integration tests for all loss functions."""

    def test_all_losses_with_same_input(self) -> None:
        """Test that all losses work with the same input."""
        y_true = keras.random.uniform((2, 64, 64, 3), seed=42)
        y_pred = keras.random.uniform((2, 64, 64, 3), seed=43)

        losses = [
            CharbonnierLoss(),
            FrequencyLoss(),
            EdgeLoss(channels=3),
            VGGLoss(),
            EnhanceLoss(),
            DarkIRCompositeLoss()
        ]

        for loss_fn in losses:
            loss_value = loss_fn(y_true, y_pred)
            assert keras.ops.all(keras.ops.isfinite(loss_value)), f"{loss_fn.name} produced non-finite values"
            assert keras.ops.all(loss_value >= 0.0), f"{loss_fn.name} produced negative values"

    def test_model_compilation(self) -> None:
        """Test that losses can be used in model compilation."""
        # Simple model
        inputs = keras.Input(shape=(32, 32, 3))
        outputs = keras.layers.Conv2D(3, 3, padding='same')(inputs)
        model = keras.Model(inputs, outputs)

        # Test compilation with each loss
        losses_to_test = [
            CharbonnierLoss(),
            DarkIRCompositeLoss(perceptual_weight=0.0)  # Skip VGG for speed
        ]

        for loss_fn in losses_to_test:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=loss_fn
            )

            # Test single training step
            x = keras.random.uniform((4, 32, 32, 3), seed=42)
            y = keras.random.uniform((4, 32, 32, 3), seed=43)

            history = model.fit(x, y, epochs=1, verbose=0, batch_size=2)

            # Should complete without errors
            assert 'loss' in history.history
            assert len(history.history['loss']) == 1

    def test_save_and_load_with_custom_losses(self) -> None:
        """Test saving and loading models with custom losses."""
        import tempfile
        import os

        # Create simple model
        inputs = keras.Input(shape=(16, 16, 3))
        outputs = keras.layers.Conv2D(3, 3, padding='same')(inputs)
        model = keras.Model(inputs, outputs)

        # Compile with custom loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=CharbonnierLoss(epsilon=1e-4)
        )

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.keras')
            model.save(model_path)

            # Load model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'CharbonnierLoss': CharbonnierLoss
                }
            )

            # Test that loaded model works
            x = keras.random.uniform((2, 16, 16, 3), seed=42)
            pred_original = model.predict(x, verbose=0)
            pred_loaded = loaded_model.predict(x, verbose=0)

            # Predictions should be identical
            np.testing.assert_allclose(
                pred_original,
                pred_loaded,
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model predictions don't match"
            )


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])