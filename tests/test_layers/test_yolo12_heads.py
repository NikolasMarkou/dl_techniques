"""
Basic tests for YOLOv12 task-specific heads.

Simple pytest cases to verify basic functionality of detection, segmentation,
and classification heads.
"""

import pytest
import tensorflow as tf
import keras
import numpy as np

from dl_techniques.layers.yolo12_heads import (
    YOLOv12DetectionHead,
    YOLOv12SegmentationHead,
    YOLOv12ClassificationHead
)


class TestYOLOv12DetectionHead:
    """Basic tests for YOLOv12DetectionHead."""

    def test_detection_head_creation(self):
        """Test that detection head can be created."""
        head = YOLOv12DetectionHead(num_classes=1, reg_max=16)
        assert head.num_classes == 1
        assert head.reg_max == 16

    def test_detection_head_build_and_call(self):
        """Test detection head build and forward pass."""
        # Create head
        head = YOLOv12DetectionHead(num_classes=1, reg_max=16)

        # Create test input shapes (P3, P4, P5)
        input_shapes = [
            (None, 32, 32, 256),  # P3: H/8
            (None, 16, 16, 512),  # P4: H/16
            (None, 8, 8, 1024)  # P5: H/32
        ]

        # Build the head
        head.build(input_shapes)
        assert head.built

        # Create test inputs
        batch_size = 2
        p3 = tf.random.normal((batch_size, 32, 32, 256))
        p4 = tf.random.normal((batch_size, 16, 16, 512))
        p5 = tf.random.normal((batch_size, 8, 8, 1024))
        inputs = [p3, p4, p5]

        # Forward pass
        output = head(inputs)

        # Check output shape
        # Total anchors = 32*32 + 16*16 + 8*8 = 1024 + 256 + 64 = 1344
        # Output channels = 4*reg_max + num_classes = 4*16 + 1 = 65
        expected_shape = (batch_size, 1344, 65)
        assert output.shape == expected_shape

    def test_detection_head_serialization(self):
        """Test detection head serialization."""
        head = YOLOv12DetectionHead(num_classes=2, reg_max=8)
        config = head.get_config()

        # Check config has expected keys
        assert config['num_classes'] == 2
        assert config['reg_max'] == 8

        # Test from_config
        new_head = YOLOv12DetectionHead.from_config(config)
        assert new_head.num_classes == 2
        assert new_head.reg_max == 8


class TestYOLOv12SegmentationHead:
    """Basic tests for YOLOv12SegmentationHead."""

    def test_segmentation_head_creation(self):
        """Test that segmentation head can be created."""
        head = YOLOv12SegmentationHead(num_classes=1)
        assert head.num_classes == 1

    def test_segmentation_head_auto_target_size(self):
        """Test segmentation head with auto-computed target size."""
        head = YOLOv12SegmentationHead(num_classes=1)

        input_shapes = [
            (None, 32, 32, 256),  # P3: H/8, so target should be 32*8=256
            (None, 16, 16, 512),  # P4: H/16
            (None, 8, 8, 1024)  # P5: H/32
        ]

        head.build(input_shapes)

        # Check computed target size
        assert head._computed_target_size == (256, 256)

    def test_segmentation_head_serialization(self):
        """Test segmentation head serialization."""
        head = YOLOv12SegmentationHead(
            num_classes=1,
            intermediate_filters=[64, 32],
            target_size=(128, 128)
        )
        config = head.get_config()

        assert config['num_classes'] == 1
        assert config['intermediate_filters'] == [64, 32]
        assert config['target_size'] == (128, 128)


class TestYOLOv12ClassificationHead:
    """Basic tests for YOLOv12ClassificationHead."""

    def test_classification_head_creation(self):
        """Test that classification head can be created."""
        head = YOLOv12ClassificationHead(num_classes=1)
        assert head.num_classes == 1

    def test_classification_head_serialization(self):
        """Test classification head serialization."""
        head = YOLOv12ClassificationHead(
            num_classes=2,
            hidden_dims=[256],
            pooling_types=["avg"],
            dropout_rate=0.5
        )
        config = head.get_config()

        assert config['num_classes'] == 2
        assert config['hidden_dims'] == [256]
        assert config['pooling_types'] == ["avg"]
        assert config['dropout_rate'] == 0.5


class TestHeadsIntegration:
    """Basic integration tests for all heads together."""

    def test_all_heads_with_same_features(self):
        """Test that all heads can process the same feature maps."""
        # Create all heads
        detection_head = YOLOv12DetectionHead(num_classes=1, reg_max=16)
        segmentation_head = YOLOv12SegmentationHead(
            num_classes=1,
            target_size=(256, 256),
            intermediate_filters=[128, 64, 32, 16]
        )
        classification_head = YOLOv12ClassificationHead(num_classes=1)

        # Common input shapes
        input_shapes = [
            (None, 32, 32, 256),
            (None, 16, 16, 512),
            (None, 8, 8, 1024)
        ]

        # Build all heads
        detection_head.build(input_shapes)
        segmentation_head.build(input_shapes)
        classification_head.build(input_shapes)

        # Create test inputs
        batch_size = 2
        features = [
            tf.random.normal((batch_size, 32, 32, 256)),
            tf.random.normal((batch_size, 16, 16, 512)),
            tf.random.normal((batch_size, 8, 8, 1024))
        ]

        # Forward pass through all heads
        detection_out = detection_head(features)
        segmentation_out = segmentation_head(features)
        classification_out = classification_head(features)

        # Check all outputs have expected shapes
        assert detection_out.shape == (batch_size, 1344, 65)  # 1344 anchors, 65 channels
        assert segmentation_out.shape == (batch_size, 256, 256, 1)  # Full resolution mask
        assert classification_out.shape == (batch_size, 1)  # Single class probability

    def test_heads_with_different_input_sizes(self):
        """Test heads with different patch sizes."""
        # Test with smaller patches (128x128)
        input_shapes_small = [
            (None, 16, 16, 256),  # P3: 128/8 = 16
            (None, 8, 8, 512),  # P4: 128/16 = 8
            (None, 4, 4, 1024)  # P5: 128/32 = 4
        ]

        seg_head = YOLOv12SegmentationHead(
            num_classes=1,
            target_size=(128, 128),
            intermediate_filters=[64, 32, 16]
        )

        seg_head.build(input_shapes_small)

        # Test forward pass
        batch_size = 1
        features_small = [
            tf.random.normal((batch_size, 16, 16, 256)),
            tf.random.normal((batch_size, 8, 8, 512)),
            tf.random.normal((batch_size, 4, 4, 1024))
        ]

        output = seg_head(features_small)
        assert output.shape == (batch_size, 128, 128, 1)
