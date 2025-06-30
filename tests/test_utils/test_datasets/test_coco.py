"""
Comprehensive test suite for COCO Dataset Preprocessor.

Tests cover configuration classes, dummy dataset generation, dataset builder
functionality, preprocessing, augmentation, and integration scenarios.

File: tests/test_utils/test_datasets/test_coco.py
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
from typing import Dict, Tuple, Any

# Import the classes and functions to test
from dl_techniques.utils.datasets.coco import (
    AugmentationConfig,
    DatasetConfig,
    COCODatasetBuilder,
    create_dummy_coco_dataset,
    create_coco_dataset,
    COCO_CLASSES,
    COCO_NUM_CLASSES,
    INVALID_BBOX_VALUE
)


class TestAugmentationConfig:
    """Test AugmentationConfig dataclass."""

    def test_default_initialization(self):
        """Test default initialization values."""
        config = AugmentationConfig()

        assert config.brightness_delta == 0.1
        assert config.contrast_delta == 0.1
        assert config.saturation_delta == 0.1
        assert config.hue_delta == 0.05
        assert config.horizontal_flip_prob == 0.5
        assert config.vertical_flip_prob == 0.0
        assert config.rotation_degrees == 0.0
        assert config.mixup_alpha == 0.0
        assert config.cutmix_alpha == 0.0
        assert config.mosaic_prob == 0.0

    def test_custom_initialization(self):
        """Test custom initialization values."""
        config = AugmentationConfig(
            brightness_delta=0.2,
            horizontal_flip_prob=0.8,
            rotation_degrees=15.0
        )

        assert config.brightness_delta == 0.2
        assert config.horizontal_flip_prob == 0.8
        assert config.rotation_degrees == 15.0
        # Check defaults are preserved
        assert config.contrast_delta == 0.1
        assert config.vertical_flip_prob == 0.0


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""

    def test_from_class_names(self):
        """Test creation from class names."""
        class_names = ['cat', 'dog', 'bird']
        config = DatasetConfig.from_class_names(class_names)

        assert config.class_names == class_names
        assert config.num_classes == 3
        assert isinstance(config.class_mapping, dict)

    def test_coco_default(self):
        """Test COCO default configuration."""
        config = DatasetConfig.coco_default()

        assert config.class_names == COCO_CLASSES
        assert config.num_classes == COCO_NUM_CLASSES
        assert config.num_classes == 80

    def test_num_classes_property(self):
        """Test num_classes property calculation."""
        config = DatasetConfig(class_names=['a', 'b', 'c', 'd'])
        assert config.num_classes == 4


class TestCreateDummyCOCODataset:
    """Test dummy COCO dataset creation."""

    @pytest.mark.parametrize("img_size,num_samples", [
        (320, 10),
        (640, 5),
        (512, 20)
    ])
    def test_dummy_dataset_shapes(self, img_size: int, num_samples: int):
        """Test dummy dataset produces correct COCO-style format."""
        dataset = create_dummy_coco_dataset(
            num_samples=num_samples,
            img_size=img_size,
            max_boxes=15
        )

        # Test one example
        for example in dataset.take(1):
            # Check image
            assert example['image'].shape == (img_size, img_size, 3)
            assert example['image'].dtype == tf.uint8

            # Check objects structure
            assert 'objects' in example
            objects = example['objects']
            assert 'bbox' in objects
            assert 'label' in objects
            assert 'segmentation' in objects

            # Bounding boxes should be [N, 4] where N varies
            bboxes = objects['bbox']
            assert bboxes.shape[-1] == 4  # [ymin, xmin, ymax, xmax]
            assert bboxes.dtype == tf.float32

            # Labels should match number of boxes
            labels = objects['label']
            assert labels.shape[0] == bboxes.shape[0]
            assert labels.dtype == tf.int64

    def test_dummy_dataset_coordinate_ranges(self):
        """Test that dummy dataset generates valid normalized coordinates."""
        dataset = create_dummy_coco_dataset(num_samples=5, img_size=416)

        for example in dataset.take(5):
            objects = example['objects']
            bboxes = objects['bbox']
            labels = objects['label']

            if tf.shape(bboxes)[0] > 0:  # If there are boxes
                # Check coordinate ranges [0, 1]
                assert tf.reduce_min(bboxes) >= 0.0
                assert tf.reduce_max(bboxes) <= 1.0

                # Check coordinate order: ymin < ymax, xmin < xmax
                ymin, xmin, ymax, xmax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
                assert tf.reduce_all(ymax > ymin)
                assert tf.reduce_all(xmax > xmin)

                # Class IDs should be non-negative integers
                assert tf.reduce_min(labels) >= 0

    def test_dummy_dataset_count(self):
        """Test that dummy dataset produces correct number of samples."""
        num_samples = 25
        dataset = create_dummy_coco_dataset(num_samples=num_samples, img_size=224)

        count = 0
        for _ in dataset:
            count += 1

        assert count == num_samples

    def test_dummy_dataset_coco_compatibility(self):
        """Test that dummy dataset format is compatible with COCO filter."""
        dataset = create_dummy_coco_dataset(num_samples=3, img_size=320)

        # This should work without errors if format is correct
        builder = COCODatasetBuilder(img_size=320, batch_size=1, augment_data=False)

        # Test that filter function works
        for example in dataset.take(1):
            result = builder._filter_valid_examples(example)
            assert isinstance(result, tf.Tensor)
            assert result.dtype == tf.bool


class TestCOCODatasetBuilder:
    """Test COCODatasetBuilder class."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic configuration for testing."""
        return {
            'img_size': 416,
            'batch_size': 4,
            'max_boxes_per_image': 10,
            'use_detection': True,
            'use_segmentation': False,
            'augment_data': False  # Disable for consistent testing
        }

    @pytest.fixture
    def builder(self, basic_config: Dict[str, Any]) -> COCODatasetBuilder:
        """Create a basic builder instance."""
        return COCODatasetBuilder(**basic_config)

    def test_initialization_defaults(self):
        """Test builder initialization with defaults."""
        builder = COCODatasetBuilder()

        assert builder.img_size == 640
        assert builder.batch_size == 32
        assert builder.max_boxes_per_image == 100
        assert builder.use_detection is True
        assert builder.use_segmentation is False
        assert builder.augment_data is True
        assert isinstance(builder.augmentation_config, AugmentationConfig)
        assert isinstance(builder.dataset_config, DatasetConfig)

    def test_initialization_custom_config(self):
        """Test builder initialization with custom configurations."""
        aug_config = AugmentationConfig(brightness_delta=0.3)
        dataset_config = DatasetConfig.from_class_names(['cat', 'dog'])

        builder = COCODatasetBuilder(
            img_size=512,
            batch_size=16,
            augmentation_config=aug_config,
            dataset_config=dataset_config
        )

        assert builder.img_size == 512
        assert builder.batch_size == 16
        assert builder.augmentation_config.brightness_delta == 0.3
        assert builder.dataset_config.num_classes == 2

    @pytest.mark.parametrize("invalid_config,expected_error", [
        ({'img_size': 0}, ValueError),
        ({'img_size': -100}, ValueError),
        ({'batch_size': 0}, ValueError),
        ({'batch_size': -5}, ValueError),
        ({'max_boxes_per_image': 0}, ValueError),
        ({'use_detection': False, 'use_segmentation': False}, ValueError),
    ])
    def test_validation_errors(self, basic_config: Dict[str, Any],
                             invalid_config: Dict[str, Any],
                             expected_error: Exception):
        """Test that invalid configurations raise appropriate errors."""
        config = {**basic_config, **invalid_config}

        with pytest.raises(expected_error):
            COCODatasetBuilder(**config)

    def test_validate_bbox(self, builder: COCODatasetBuilder):
        """Test bounding box validation."""
        # Valid bboxes (normalized format: [ymin, xmin, ymax, xmax])
        valid_bboxes = tf.constant([
            [0.1, 0.1, 0.5, 0.5],  # Valid box
            [0.0, 0.0, 0.2, 0.2],  # Edge case: starts at origin
            [0.8, 0.8, 1.0, 1.0],  # Edge case: ends at edge
        ], dtype=tf.float32)

        valid_mask = builder._validate_bbox(valid_bboxes)
        assert tf.reduce_all(valid_mask)

        # Invalid bboxes
        invalid_bboxes = tf.constant([
            [-0.1, 0.1, 0.5, 0.5],  # Negative coordinate
            [0.1, 0.1, 1.1, 0.5],   # Coordinate > 1.0
            [0.5, 0.5, 0.1, 0.1],   # x2 < x1, y2 < y1
            [0.1, 0.1, 0.101, 0.101],  # Too small area
        ], dtype=tf.float32)

        invalid_mask = builder._validate_bbox(invalid_bboxes)
        assert not tf.reduce_any(invalid_mask)

    def test_preprocess_detection_targets(self, builder: COCODatasetBuilder):
        """Test detection target preprocessing."""
        # Mock objects dictionary
        mock_objects = {
            'bbox': tf.constant([
                [0.1, 0.1, 0.4, 0.4],  # [ymin, xmin, ymax, xmax]
                [0.5, 0.5, 0.8, 0.8],
            ], dtype=tf.float32),
            'label': tf.constant([1, 5], dtype=tf.int64)
        }

        image_shape = tf.constant([416, 416, 3])
        targets = builder._preprocess_detection_targets(mock_objects, image_shape)

        # Check shape
        assert targets.shape == (builder.max_boxes_per_image, 5)

        # Check first two boxes are valid (non-sentinel)
        assert targets[0, 0] == 1.0  # class_id
        assert targets[1, 0] == 5.0  # class_id

        # Check coordinates are converted to [x1, y1, x2, y2] format
        assert targets[0, 1] == 0.1  # x1 (was xmin)
        assert targets[0, 2] == 0.1  # y1 (was ymin)
        assert targets[0, 3] == 0.4  # x2 (was xmax)
        assert targets[0, 4] == 0.4  # y2 (was ymax)

        # Check padding uses sentinel values
        assert tf.reduce_all(targets[2:, :] == INVALID_BBOX_VALUE)

    def test_color_augmentation_disabled(self, builder: COCODatasetBuilder):
        """Test that color augmentation is disabled when augment_data=False."""
        builder.augment_data = False

        original_image = tf.random.uniform((416, 416, 3), dtype=tf.float32)
        augmented_image = builder._apply_color_augmentation(original_image)

        # Should be identical when disabled
        assert tf.reduce_all(tf.equal(original_image, augmented_image))

    def test_color_augmentation_enabled(self):
        """Test that color augmentation works when enabled."""
        aug_config = AugmentationConfig(brightness_delta=0.5)
        builder = COCODatasetBuilder(
            augment_data=True,
            augmentation_config=aug_config
        )

        # Set random seed for reproducibility
        tf.random.set_seed(42)

        original_image = tf.ones((416, 416, 3), dtype=tf.float32) * 0.5
        augmented_image = builder._apply_color_augmentation(original_image)

        # Values should be clipped to [0, 1]
        assert tf.reduce_min(augmented_image) >= 0.0
        assert tf.reduce_max(augmented_image) <= 1.0

    def test_horizontal_flip_augmentation(self, builder: COCODatasetBuilder):
        """Test horizontal flip augmentation."""
        builder.augment_data = True
        builder.augmentation_config.horizontal_flip_prob = 1.0  # Always flip

        # Create test image and targets
        image = tf.random.uniform((416, 416, 3), dtype=tf.float32)
        targets = {
            'detection': tf.constant([
                [1.0, 0.1, 0.2, 0.3, 0.4],  # [class, x1, y1, x2, y2]
                [2.0, 0.6, 0.7, 0.8, 0.9],
            ], dtype=tf.float32)
        }

        flipped_image, flipped_targets = builder._apply_horizontal_flip(image, targets)

        # Check that x coordinates are flipped: new_x = 1.0 - old_x
        original_x1, original_x2 = 0.1, 0.3
        expected_x1, expected_x2 = 1.0 - original_x2, 1.0 - original_x1

        assert tf.abs(flipped_targets['detection'][0, 1] - expected_x1) < 1e-6
        assert tf.abs(flipped_targets['detection'][0, 3] - expected_x2) < 1e-6

        # Y coordinates should remain unchanged
        assert flipped_targets['detection'][0, 2] == 0.2
        assert flipped_targets['detection'][0, 4] == 0.4

    @patch('dl_techniques.utils.datasets.coco.tfds.load')
    def test_load_tfds_dataset_success(self, mock_tfds_load, builder: COCODatasetBuilder):
        """Test successful loading of TFDS dataset."""
        # Mock tfds.load to return mock datasets
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_tfds_load.side_effect = [mock_train_ds, mock_val_ds]

        train_ds, val_ds = builder._load_tfds_dataset()

        assert train_ds == mock_train_ds
        assert val_ds == mock_val_ds
        assert mock_tfds_load.call_count == 2

    @patch('dl_techniques.utils.datasets.coco.tfds.load')
    @patch('dl_techniques.utils.datasets.coco.create_dummy_coco_dataset')
    def test_load_tfds_dataset_fallback(self, mock_dummy, mock_tfds_load,
                                      builder: COCODatasetBuilder):
        """Test fallback to dummy dataset when TFDS fails."""
        # Make tfds.load raise an exception
        mock_tfds_load.side_effect = Exception("TFDS not available")

        # Mock dummy dataset
        mock_dummy_ds = MagicMock()
        mock_dummy.return_value = mock_dummy_ds
        mock_dummy_ds.take.return_value = MagicMock()
        mock_dummy_ds.skip.return_value = MagicMock()

        train_ds, val_ds = builder._load_tfds_dataset()

        # Should call dummy dataset creation
        mock_dummy.assert_called_once()
        mock_dummy_ds.take.assert_called_once_with(800)
        mock_dummy_ds.skip.assert_called_once_with(800)

    def test_get_dataset_info(self, builder: COCODatasetBuilder):
        """Test dataset info retrieval."""
        info = builder.get_dataset_info()

        # Check required keys are present
        required_keys = [
            'img_size', 'batch_size', 'max_boxes_per_image',
            'use_detection', 'use_segmentation', 'augment_data',
            'invalid_bbox_value', 'num_classes', 'augmentation_config',
            'pipeline_optimizations'
        ]

        for key in required_keys:
            assert key in info

        # Check specific values
        assert info['img_size'] == builder.img_size
        assert info['batch_size'] == builder.batch_size
        assert info['invalid_bbox_value'] == INVALID_BBOX_VALUE
        assert isinstance(info['pipeline_optimizations'], list)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def small_builder(self) -> COCODatasetBuilder:
        """Create a builder for integration testing."""
        return COCODatasetBuilder(
            img_size=224,
            batch_size=2,
            max_boxes_per_image=5,
            use_detection=True,
            use_segmentation=False,
            augment_data=False
        )

    @patch('dl_techniques.utils.datasets.coco.tfds.load')
    def test_create_datasets_integration(self, mock_tfds_load,
                                       small_builder: COCODatasetBuilder):
        """Test end-to-end dataset creation."""
        # Use dummy dataset for testing
        mock_tfds_load.side_effect = Exception("Use dummy")

        train_ds, val_ds = small_builder.create_datasets()

        # Test training dataset
        for images, targets in train_ds.take(1):
            # Check batch shapes
            assert images.shape == (2, 224, 224, 3)  # [batch, H, W, C]
            assert 'detection' in targets
            assert targets['detection'].shape == (2, 5, 5)  # [batch, max_boxes, 5]

            # Check data types
            assert images.dtype == tf.float32
            assert targets['detection'].dtype == tf.float32

            # Check value ranges
            assert tf.reduce_min(images) >= 0.0
            assert tf.reduce_max(images) <= 1.0

            # Check that some boxes might be valid (not all padding)
            detection_targets = targets['detection']
            non_padding_mask = tf.not_equal(detection_targets, INVALID_BBOX_VALUE)
            assert tf.reduce_any(non_padding_mask)  # At least some non-padding values

        # Test validation dataset
        for images, targets in val_ds.take(1):
            assert images.shape == (2, 224, 224, 3)
            assert targets['detection'].shape == (2, 5, 5)

    def test_convenience_function(self):
        """Test the convenience function."""
        with patch('dl_techniques.utils.datasets.coco.tfds.load') as mock_tfds:
            mock_tfds.side_effect = Exception("Use dummy")

            train_ds, val_ds = create_coco_dataset(
                img_size=320,
                batch_size=4,
                use_detection=True,
                augment_data=False
            )

            # Verify datasets are created
            assert train_ds is not None
            assert val_ds is not None

            # Test one batch
            for images, targets in train_ds.take(1):
                assert images.shape[0] == 4  # batch_size
                assert images.shape[1:] == (320, 320, 3)  # img_size
                assert 'detection' in targets

                # Check that we have some valid detections (not all padding)
                detection_targets = targets['detection']
                non_padding_mask = tf.not_equal(detection_targets, INVALID_BBOX_VALUE)
                assert tf.reduce_any(non_padding_mask)
                break

    def test_augmentation_pipeline(self):
        """Test that augmentation pipeline works without errors."""
        aug_config = AugmentationConfig(
            brightness_delta=0.1,
            contrast_delta=0.1,
            horizontal_flip_prob=0.5
        )

        builder = COCODatasetBuilder(
            img_size=256,
            batch_size=2,
            max_boxes_per_image=3,
            augment_data=True,
            augmentation_config=aug_config
        )

        with patch('dl_techniques.utils.datasets.coco.tfds.load') as mock_tfds:
            mock_tfds.side_effect = Exception("Use dummy")

            train_ds, _ = builder.create_datasets()

            # Process a few batches to ensure no errors
            count = 0
            for images, targets in train_ds.take(3):
                assert images.shape == (2, 256, 256, 3)
                assert targets['detection'].shape == (2, 3, 5)

                # Verify some augmentation might have occurred
                # (Images should still be in valid range after augmentation)
                assert tf.reduce_min(images) >= 0.0
                assert tf.reduce_max(images) <= 1.0

                # Check for valid detections
                detection_targets = targets['detection']
                non_padding_mask = tf.not_equal(detection_targets, INVALID_BBOX_VALUE)
                # Should have at least some valid detections
                assert tf.reduce_any(non_padding_mask)

                count += 1

            assert count == 3


class TestConstants:
    """Test module constants and configurations."""

    def test_coco_classes_count(self):
        """Test COCO classes constant."""
        assert len(COCO_CLASSES) == 80
        assert COCO_NUM_CLASSES == 80

    def test_invalid_bbox_value(self):
        """Test invalid bbox value constant."""
        assert INVALID_BBOX_VALUE == -1.0
        assert isinstance(INVALID_BBOX_VALUE, float)

    def test_coco_classes_content(self):
        """Test that COCO classes contain expected entries."""
        assert 'person' in COCO_CLASSES
        assert 'car' in COCO_CLASSES
        assert 'dog' in COCO_CLASSES
        assert 'cat' in COCO_CLASSES
        # Check it's the first class (common in COCO)
        assert COCO_CLASSES[0] == 'person'


# Pytest configuration and markers
@pytest.mark.parametrize("use_detection,use_segmentation", [
    (True, False),
    (False, True),
    (True, True),
])
def test_task_configurations(use_detection: bool, use_segmentation: bool):
    """Test different task configuration combinations."""
    builder = COCODatasetBuilder(
        img_size=320,
        batch_size=2,
        use_detection=use_detection,
        use_segmentation=use_segmentation,
        augment_data=False
    )

    assert builder.use_detection == use_detection
    assert builder.use_segmentation == use_segmentation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])