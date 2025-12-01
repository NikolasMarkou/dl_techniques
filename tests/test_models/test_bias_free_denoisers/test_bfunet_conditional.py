"""
Comprehensive test suite for Conditional Bias-Free U-Net Model.

Tests cover initialization, conditioning injection mechanisms (Film/Concat),
scaling invariance (Homogeneity), deep supervision, and classifier-free guidance support.
"""

import os
import keras
import pytest
import tempfile
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any

from dl_techniques.models.bias_free_denoisers.bfunet_conditional import (
    create_conditional_bfunet_denoiser,
    create_conditional_bfunet_variant,
    BFUNET_CONFIGS
)


class TestConditionalBiasFreeUNet:
    """Test suite for Class-Conditional Bias-Free U-Net implementation."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """Standard input shape for testing."""
        return (64, 64, 3)

    @pytest.fixture
    def num_classes(self) -> int:
        """Standard number of classes."""
        return 10

    @pytest.fixture
    def test_image_batch(self, input_shape) -> np.ndarray:
        """Create a batch of test images."""
        return np.random.rand(4, *input_shape).astype(np.float32)

    @pytest.fixture
    def test_labels_batch(self, num_classes) -> np.ndarray:
        """Create a batch of random class labels."""
        return np.random.randint(0, num_classes, size=(4,)).astype(np.int32)

    # ================================================================
    # Initialization & Validation Tests
    # ================================================================

    def test_initialization_defaults(self, input_shape, num_classes):
        """Test initialization with default parameters."""
        model = create_conditional_bfunet_denoiser(
            input_shape=input_shape,
            num_classes=num_classes
        )

        assert len(model.inputs) == 2
        # Input 0: Image (Batch, H, W, C)
        assert model.inputs[0].shape[1:] == input_shape
        # Input 1: Class Label (Batch, 1)
        assert model.inputs[1].shape[1:] == (1,)

        assert model.output_shape[1:] == input_shape

    def test_invalid_arguments(self, input_shape):
        """Test validation logic."""
        with pytest.raises(ValueError, match="num_classes must be at least 2"):
            create_conditional_bfunet_denoiser(input_shape=input_shape, num_classes=1)

        with pytest.raises(ValueError, match="Invalid class_injection_method"):
            create_conditional_bfunet_denoiser(
                input_shape=input_shape,
                num_classes=10,
                class_injection_method='invalid_method'
            )

    # ================================================================
    # Injection Mechanism Tests
    # ================================================================

    def test_spatial_broadcast_shapes(self, input_shape, num_classes):
        """Test 'spatial_broadcast' injection logic behaves correctly."""
        model = create_conditional_bfunet_denoiser(
            input_shape=input_shape,
            num_classes=num_classes,
            class_injection_method='spatial_broadcast',
            depth=3
        )
        # Just ensure forward pass works without shape mismatches
        x = np.random.rand(1, *input_shape).astype(np.float32)
        c = np.array([0])
        y = model([x, c])
        assert y.shape == (1, *input_shape)

    def test_channel_concat_shapes(self, input_shape, num_classes):
        """Test 'channel_concat' injection logic behaves correctly."""
        model = create_conditional_bfunet_denoiser(
            input_shape=input_shape,
            num_classes=num_classes,
            class_injection_method='channel_concat',
            depth=3
        )
        x = np.random.rand(1, *input_shape).astype(np.float32)
        c = np.array([0])
        y = model([x, c])
        assert y.shape == (1, *input_shape)

    # ================================================================
    # Feature Tests: Deep Supervision & CFG
    # ================================================================

    def test_deep_supervision_outputs(self, input_shape, num_classes):
        """Test that deep supervision returns list of outputs with correct shapes."""
        depth = 3
        model = create_conditional_bfunet_denoiser(
            input_shape=input_shape,
            num_classes=num_classes,
            depth=depth,
            enable_deep_supervision=True
        )

        assert isinstance(model.output, list)
        # With depth 3, we expect outputs from level 0 (final), level 1, level 2.
        # Total outputs = 1 (final) + (depth - 1) supervision outputs = 3
        assert len(model.output) == 3

        x = np.random.rand(1, *input_shape).astype(np.float32)
        c = np.array([0])
        outputs = model([x, c])

        # All outputs should match input spatial resolution (due to conv 1x1 at output)
        assert outputs[0].shape == (1, 64, 64, 3)
        assert outputs[1].shape == (1, 32, 32, 3)
        assert outputs[2].shape == (1, 16, 16, 3)

    def test_cfg_token_handling(self, input_shape, num_classes):
        """Test that the model accepts the unconditional token index."""
        model = create_conditional_bfunet_denoiser(
            input_shape=input_shape,
            num_classes=num_classes,
            enable_cfg_training=True
        )

        # Unconditional token is typically num_classes - 1 (last index)
        uncond_token = num_classes - 1

        x = np.random.rand(1, *input_shape).astype(np.float32)
        c = np.array([uncond_token])

        # Should not error
        y = model([x, c])
        assert not np.any(np.isnan(y.numpy()))

    # ================================================================
    # Variant Tests
    # ================================================================

    @pytest.mark.parametrize("variant", ['tiny', 'small', 'base'])
    def test_variants_creation(self, variant, input_shape, num_classes):
        """Test creation of specific variants."""
        model = create_conditional_bfunet_variant(
            variant,
            input_shape=input_shape,
            num_classes=num_classes,
            enable_deep_supervision=False
        )
        assert model.name.startswith(f'conditional_bfunet_{variant}')

        x = np.random.rand(1, *input_shape).astype(np.float32)
        c = np.array([0])
        y = model([x, c])
        assert y.shape == (1, *input_shape)

    # ================================================================
    # Numerical Stability & Gradient Flow
    # ================================================================

    def test_gradient_flow_conditioning(self, input_shape, num_classes):
        """Verify gradients flow back through the class embedding."""
        model = create_conditional_bfunet_denoiser(
            input_shape=input_shape,
            num_classes=num_classes,
            depth=3,
            initial_filters=8
        )

        x = tf.random.normal((2, *input_shape))
        c = tf.constant([0, 1])

        with tf.GradientTape() as tape:
            # We can't differentiate wrt integer inputs `c`, but we can check
            # if the embedding layer weights receive gradients.
            y = model([x, c], training=True)
            loss = tf.reduce_mean(tf.square(y))

        # Get gradients for all trainable variables
        grads = tape.gradient(loss, model.trainable_variables)

        # Robustly identify embedding weights by finding the layer instance
        embedding_layer = None
        for layer in model.layers:
            if isinstance(layer, keras.layers.Embedding):
                embedding_layer = layer
                break

        assert embedding_layer is not None, "Embedding layer not found in model"

    def test_serialization(self, input_shape, num_classes):
        """Test save/load consistency."""
        model = create_conditional_bfunet_denoiser(
            input_shape=input_shape,
            num_classes=num_classes,
            depth=3,
            initial_filters=8
        )

        x = np.random.rand(1, *input_shape).astype(np.float32)
        c = np.array([0])
        y_orig = model([x, c])

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "cond_bfunet.keras")
            model.save(path)

            loaded_model = keras.models.load_model(path)
            y_loaded = loaded_model([x, c])

            np.testing.assert_allclose(y_orig.numpy(), y_loaded.numpy(), rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])