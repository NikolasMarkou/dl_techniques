"""
Comprehensive test suite for Unified Conditional Bias-Free U-Net.

Tests cover multi-modal conditioning (Dense, Discrete, Hybrid),
various injection mechanisms (FiLM, Concat, Mult), and theoretical compliance.
"""

import os
import keras
import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple

from dl_techniques.models.bias_free_denoisers.bfunet_conditional_unified import (
    create_unified_conditional_bfunet,
    create_depth_estimation_bfunet,
    create_class_conditional_bfunet,
    create_semantic_depth_bfunet
)


class TestUnifiedConditionalBFUNet:
    """Test suite for Unified Conditional BFU-Net."""

    @pytest.fixture
    def target_shape(self) -> Tuple[int, int, int]:
        return (64, 64, 1)  # Depth map or Grayscale

    @pytest.fixture
    def dense_shape(self) -> Tuple[int, int, int]:
        return (64, 64, 3)  # RGB Image

    @pytest.fixture
    def num_classes(self) -> int:
        return 5

    # ================================================================
    # Modality Configuration Tests
    # ================================================================

    def test_dense_only_configuration(self, target_shape, dense_shape):
        """Test initialization and forward pass with Dense conditioning only."""
        model = create_unified_conditional_bfunet(
            target_shape=target_shape,
            dense_conditioning_shape=dense_shape,
            num_classes=None,
            depth=3,
            initial_filters=16
        )

        assert len(model.inputs) == 2
        # Input 0: Noisy Target
        assert model.inputs[0].shape[1:] == target_shape
        # Input 1: Dense Condition
        assert model.inputs[1].shape[1:] == dense_shape

        x_target = np.random.rand(1, *target_shape).astype(np.float32)
        x_dense = np.random.rand(1, *dense_shape).astype(np.float32)

        y = model([x_target, x_dense])
        assert y.shape == (1, *target_shape)

    def test_discrete_only_configuration(self, target_shape, num_classes):
        """Test initialization and forward pass with Discrete conditioning only."""
        model = create_unified_conditional_bfunet(
            target_shape=target_shape,
            dense_conditioning_shape=None,
            num_classes=num_classes,
            depth=3,
            initial_filters=16
        )

        assert len(model.inputs) == 2
        # Input 1: Class Label
        assert model.inputs[1].dtype == 'int32'

        x_target = np.random.rand(1, *target_shape).astype(np.float32)
        c_label = np.array([0])

        y = model([x_target, c_label])
        assert y.shape == (1, *target_shape)

    def test_hybrid_configuration(self, target_shape, dense_shape, num_classes):
        """Test initialization and forward pass with Hybrid conditioning."""
        model = create_unified_conditional_bfunet(
            target_shape=target_shape,
            dense_conditioning_shape=dense_shape,
            num_classes=num_classes,
            depth=3,
            initial_filters=16
        )

        assert len(model.inputs) == 3
        # Order: [Target, Dense, Discrete]

        x_target = np.random.rand(1, *target_shape).astype(np.float32)
        x_dense = np.random.rand(1, *dense_shape).astype(np.float32)
        c_label = np.array([0])

        y = model([x_target, x_dense, c_label])
        assert y.shape == (1, *target_shape)

    def test_missing_conditions_error(self, target_shape):
        """Ensure error is raised if no conditioning is provided."""
        with pytest.raises(ValueError, match="At least one conditioning modality"):
            create_unified_conditional_bfunet(
                target_shape=target_shape,
                dense_conditioning_shape=None,
                num_classes=None
            )

    # ================================================================
    # Injection Method Tests (Theoretical Compliance)
    # ================================================================

    @pytest.mark.parametrize("method", ['film', 'multiplication', 'concatenation'])
    def test_dense_injection_methods(self, target_shape, dense_shape, method):
        """Test all dense injection methods."""
        model = create_unified_conditional_bfunet(
            target_shape=target_shape,
            dense_conditioning_shape=dense_shape,
            dense_injection_method=method,
            depth=3,
            initial_filters=8
        )

        x_target = np.random.rand(1, *target_shape).astype(np.float32)
        x_dense = np.random.rand(1, *dense_shape).astype(np.float32)

        y = model([x_target, x_dense])
        assert not np.any(np.isnan(y.numpy()))

    @pytest.mark.parametrize("method", ['spatial_broadcast', 'channel_concat'])
    def test_discrete_injection_methods(self, target_shape, num_classes, method):
        """Test all discrete injection methods."""
        model = create_unified_conditional_bfunet(
            target_shape=target_shape,
            num_classes=num_classes,
            discrete_injection_method=method,
            depth=3,
            initial_filters=8
        )

        x_target = np.random.rand(1, *target_shape).astype(np.float32)
        c_label = np.array([0])

        y = model([x_target, c_label])
        assert not np.any(np.isnan(y.numpy()))

    # ================================================================
    # Convenience Wrapper Tests
    # ================================================================

    def test_create_depth_estimation_bfunet(self):
        """Test wrapper for depth estimation."""
        model = create_depth_estimation_bfunet(
            depth_shape=(32, 32, 1),
            rgb_shape=(32, 32, 3),
            depth=3
        )
        assert model.name == 'depth_estimation_bfunet'
        assert len(model.inputs) == 2

    def test_create_class_conditional_bfunet(self):
        """Test wrapper for class conditional generation."""
        model = create_class_conditional_bfunet(
            image_shape=(32, 32, 3),
            num_classes=10,
            depth=3
        )
        assert model.name == 'class_conditional_bfunet'
        assert len(model.inputs) == 2

    def test_create_semantic_depth_bfunet(self):
        """Test wrapper for semantic depth estimation (Hybrid)."""
        model = create_semantic_depth_bfunet(
            depth_shape=(32, 32, 1),
            rgb_shape=(32, 32, 3),
            num_classes=5,
            depth=3
        )
        assert model.name == 'semantic_depth_bfunet'
        assert len(model.inputs) == 3

    # ================================================================
    # Deep Supervision
    # ================================================================

    def test_unified_deep_supervision(self, target_shape, dense_shape):
        """Test deep supervision in unified model."""
        model = create_unified_conditional_bfunet(
            target_shape=target_shape,
            dense_conditioning_shape=dense_shape,
            depth=3,
            enable_deep_supervision=True
        )

        x_target = np.random.rand(1, *target_shape).astype(np.float32)
        x_dense = np.random.rand(1, *dense_shape).astype(np.float32)

        outputs = model([x_target, x_dense])

        assert isinstance(outputs, list)
        assert len(outputs) == 3
        # Check shapes (decreasing resolution is handled internally, but outputs are upsampled?
        # Actually in BFUNet implementation logic:
        # Supervision outputs are typically 1x1 convs at the decoder level resolution.
        # Level 0: 64x64
        # Level 1: 32x32
        # Level 2: 16x16
        assert outputs[0].shape == (1, 64, 64, 1)
        assert outputs[1].shape == (1, 32, 32, 1)
        assert outputs[2].shape == (1, 16, 16, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])