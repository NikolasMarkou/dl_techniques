"""
COCO Dataset Preprocessor for YOLOv12 Pre-training

This module provides utilities to load, preprocess, and create TensorFlow datasets
from the COCO 2017 dataset for pre-training YOLOv12 multi-task models.

The preprocessor handles:
- Loading COCO via tensorflow_datasets
- Converting COCO annotations to YOLOv12 format
- Creating detection and segmentation targets
- Proper data augmentation for pre-training
- Batching with proper padding for variable-length sequences

FIXED VERSION: Now supports configurable segmentation classes for COCO pretraining
(80 classes) vs crack detection fine-tuning (1 class).

Fixed Issues:
- Relaxed filtering to only require object presence (not both detection and segmentation)
- Improved dummy dataset generation to pass filters
- More robust error handling

Usage:
    ```python
    from dl_techniques.utils.datasets.coco import COCODatasetBuilder

    # For COCO pretraining (80 segmentation classes)
    builder = COCODatasetBuilder(
        img_size=640,
        batch_size=16,
        segmentation_classes=80,
        cache_dir="/path/to/cache"
    )

    # For crack detection (1 segmentation class)
    builder = COCODatasetBuilder(
        img_size=640,
        batch_size=16,
        segmentation_classes=1,
        cache_dir="/path/to/cache"
    )

    train_ds, val_ds = builder.create_datasets()
    ```
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# COCO 2017 Classes (80 classes)
# ---------------------------------------------------------------------

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

COCO_NUM_CLASSES = len(COCO_CLASSES)

# Sentinel value for padding detection targets (avoids confusion with valid zeros)
INVALID_BBOX_VALUE = -1.0

# ---------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation parameters."""
    brightness_delta: float = 0.1
    contrast_delta: float = 0.1
    saturation_delta: float = 0.1
    hue_delta: float = 0.05
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0  # Usually disabled for detection
    rotation_degrees: float = 0.0    # Disabled by default due to complexity
    mixup_alpha: float = 0.0         # Disabled by default
    cutmix_alpha: float = 0.0        # Disabled by default
    mosaic_prob: float = 0.0         # Disabled by default

# ---------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for dataset class mapping and filtering."""
    class_names: List[str]
    class_mapping: Dict[int, int] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.class_names)

    @classmethod
    def from_class_names(cls, class_names: List[str]) -> 'DatasetConfig':
        """Create config from list of class names."""
        return cls(class_names=class_names)

    @classmethod
    def coco_default(cls) -> 'DatasetConfig':
        """Create default COCO configuration."""
        return cls(class_names=COCO_CLASSES)

# ---------------------------------------------------------------------

def create_dummy_coco_dataset(
    num_samples: int,
    img_size: int,
    num_classes: int = 80,
    max_boxes: int = 20,
    min_boxes: int = 1,
    segmentation_classes: int = 80  # NEW: Configurable segmentation classes
) -> tf.data.Dataset:
    """
    Create a dummy COCO-style dataset for testing and development.

    FIXED: Now creates dummy data that passes the relaxed filter.

    Args:
        num_samples: Number of samples to generate.
        img_size: Input image size.
        num_classes: Number of object classes.
        max_boxes: Maximum number of boxes per image.
        min_boxes: Minimum number of boxes per image.
        segmentation_classes: Number of segmentation classes.

    Returns:
        TensorFlow dataset with COCO-style dictionary format.
        Compatible with the COCODatasetBuilder pipeline.
    """
    def generator():
        for _ in range(num_samples):
            # Generate dummy image
            img = np.random.uniform(0, 255, (img_size, img_size, 3)).astype(np.uint8)

            # Generate random number of boxes (ensure at least 1 for filter)
            num_boxes = np.random.randint(max(min_boxes, 1), max_boxes + 1)

            # Create bounding boxes in COCO format [ymin, xmin, ymax, xmax]
            bboxes = []
            labels = []

            for i in range(num_boxes):
                # Random class
                cls_id = np.random.randint(0, num_classes)

                # Random box coordinates (normalized, ensure valid boxes)
                xmin = np.random.uniform(0, 0.8)
                ymin = np.random.uniform(0, 0.8)
                xmax = np.random.uniform(xmin + 0.05, min(xmin + 0.4, 1.0))
                ymax = np.random.uniform(ymin + 0.05, min(ymin + 0.4, 1.0))

                # COCO format: [ymin, xmin, ymax, xmax]
                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(cls_id)

            # Convert to numpy arrays
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            # Generate dummy segmentation mask based on segmentation_classes
            if segmentation_classes == 1:
                # Binary segmentation - generate random binary mask
                seg_mask = np.random.randint(0, 2, (img_size, img_size, 1), dtype=np.float32)
            else:
                # Multi-class segmentation - generate random multi-class mask
                seg_mask = np.random.randint(0, segmentation_classes, (img_size, img_size), dtype=np.int32)
                seg_mask = np.eye(segmentation_classes)[seg_mask].astype(np.float32)  # Convert to one-hot

            # FIXED: Create dummy segmentation data that indicates presence
            # This ensures the dummy data passes the filter
            dummy_segmentation = tf.constant(['dummy_segmentation'] * num_boxes, dtype=tf.string)

            # Create COCO-style example dictionary
            example = {
                'image': img,
                'objects': {
                    'bbox': bboxes,
                    'label': labels,
                    # FIXED: Add non-empty segmentation field for filter compatibility
                    'segmentation': dummy_segmentation
                },
                # Add the segmentation mask directly for easier processing
                '_segmentation_mask': seg_mask
            }

            yield example

    # Define output signature for COCO-style format
    output_signature = {
        'image': tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.uint8),
        'objects': {
            'bbox': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'segmentation': tf.TensorSpec(shape=(None,), dtype=tf.string)
        },
        '_segmentation_mask': tf.TensorSpec(
            shape=(img_size, img_size, segmentation_classes) if segmentation_classes > 1
                  else (img_size, img_size, 1),
            dtype=tf.float32
        )
    }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset

# ---------------------------------------------------------------------

class COCODatasetBuilder:
    """
    Enhanced COCO Dataset Builder with complete implementation.

    Features:
    - Proper tfds integration with correct splits
    - Flexible class configuration
    - Complete augmentation pipeline
    - Robust error handling and validation
    - Memory-efficient processing
    - Configurable segmentation classes

    FIXED VERSION: Now supports configurable segmentation classes and relaxed filtering.
    """

    def __init__(
        self,
        img_size: int = 640,
        batch_size: int = 32,
        max_boxes_per_image: int = 100,
        cache_dir: Optional[str] = None,
        use_detection: bool = True,
        use_segmentation: bool = False,
        augment_data: bool = True,
        augmentation_config: Optional[AugmentationConfig] = None,
        min_bbox_area: float = 4.0,
        class_names: Optional[List[str]] = None,
        dataset_config: Optional[DatasetConfig] = None,
        data_dir: Optional[str] = None,
        # NEW: Configurable segmentation classes
        segmentation_classes: int = 80,
        # NEW: Memory management options
        shuffle_buffer_size: int = 100,
        limit_train_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize COCO dataset builder.

        Args:
            img_size: Target image size for resizing.
            batch_size: Batch size for training.
            max_boxes_per_image: Maximum number of boxes per image.
            cache_dir: Directory for caching processed data.
            use_detection: Enable detection task.
            use_segmentation: Enable segmentation task.
            augment_data: Enable data augmentation.
            augmentation_config: Custom augmentation configuration.
            min_bbox_area: Minimum bounding box area in pixels.
            class_names: Custom class names (overrides dataset_config).
            dataset_config: Custom dataset configuration.
            data_dir: Directory where COCO data is stored.
            segmentation_classes: Number of segmentation classes (80 for COCO, 1 for binary).
            shuffle_buffer_size: Size of shuffle buffer (reduce if out of memory).
            limit_train_samples: Limit training samples for memory efficiency.
            **kwargs: Additional configuration options.
        """
        # Core configuration
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_boxes_per_image = max_boxes_per_image
        self.cache_dir = cache_dir
        self.use_detection = use_detection
        self.use_segmentation = use_segmentation
        self.augment_data = augment_data
        self.min_bbox_area = min_bbox_area
        self.data_dir = data_dir

        # NEW: Segmentation configuration
        self.segmentation_classes = segmentation_classes

        # NEW: Memory management
        self.shuffle_buffer_size = shuffle_buffer_size
        self.limit_train_samples = limit_train_samples

        # NEW: Track if using dummy data
        self.using_dummy_data = False

        # Configure class setup
        if dataset_config is not None:
            self.dataset_config = dataset_config
        elif class_names is not None:
            self.dataset_config = DatasetConfig.from_class_names(class_names)
        else:
            self.dataset_config = DatasetConfig.coco_default()

        # Configure augmentations
        self.augmentation_config = augmentation_config or AugmentationConfig()

        # Validate configuration
        self._validate_configuration()

        logger.info(f"COCODatasetBuilder initialized:")
        logger.info(f"  - Image size: {self.img_size}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Detection classes: {self.dataset_config.num_classes}")
        logger.info(f"  - Segmentation classes: {self.segmentation_classes}")
        logger.info(f"  - Detection: {self.use_detection}")
        logger.info(f"  - Segmentation: {self.use_segmentation}")
        logger.info(f"  - Augmentation: {self.augment_data}")
        logger.info(f"  - Shuffle buffer size: {self.shuffle_buffer_size}")
        if self.limit_train_samples:
            logger.info(f"  - Train sample limit: {self.limit_train_samples}")
        else:
            logger.info(f"  - Train sample limit: None (full dataset)")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        if not self.use_detection and not self.use_segmentation:
            raise ValueError("At least one task (detection or segmentation) must be enabled")

        if self.img_size <= 0:
            raise ValueError("img_size must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.max_boxes_per_image <= 0:
            raise ValueError("max_boxes_per_image must be positive")

        if self.segmentation_classes <= 0:
            raise ValueError("segmentation_classes must be positive")

        # Warn about non-standard class count
        if (self.dataset_config.num_classes != 80 and
            self.dataset_config.class_names == COCO_CLASSES):
            logger.warning(
                f"Using {self.dataset_config.num_classes} classes but COCO has 80. "
                f"Ensure your configuration is correct."
            )

        logger.info("✅ Configuration validation passed")

    def _load_tfds_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load COCO dataset using tensorflow_datasets with proper splits.

        Returns:
            Tuple of (train_dataset, validation_dataset).
        """
        try:
            load_kwargs = {
                'as_supervised': False,
                'with_info': False,
                'shuffle_files': True,
            }

            if self.data_dir:
                load_kwargs['data_dir'] = self.data_dir

            # Load predefined splits
            train_ds = tfds.load('coco/2017', split='train', **load_kwargs)
            val_ds = tfds.load('coco/2017', split='validation', **load_kwargs)

            logger.info("✅ Successfully loaded COCO dataset from tfds")
            self.using_dummy_data = False
            return train_ds, val_ds

        except Exception as e:
            logger.error(f"❌ Failed to load COCO dataset: {e}")
            logger.info("🔄 Creating dummy dataset for testing...")

            # Set flag to indicate we're using dummy data
            self.using_dummy_data = True

            # Create larger dummy dataset for proper training simulation
            # Generate enough samples for multiple epochs
            samples_needed = max(10000, self.batch_size * 100)  # At least 100 batches worth

            dummy_ds = create_dummy_coco_dataset(
                num_samples=samples_needed,
                img_size=self.img_size,
                segmentation_classes=self.segmentation_classes
            )

            # Split 80/20 for train/val
            train_size = int(samples_needed * 0.8)
            train_ds = dummy_ds.take(train_size)
            val_ds = dummy_ds.skip(train_size).take(int(samples_needed * 0.2))

            logger.info(f"Created dummy dataset with {train_size} training and {int(samples_needed * 0.2)} validation samples")
            return train_ds, val_ds

    def _filter_valid_examples(self, example: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Filter examples that have valid annotations for enabled tasks.

        FIXED: Now only requires the presence of objects/labels, not both detection and segmentation.
        Since we create synthetic segmentation masks from bounding boxes, we only need objects.

        Args:
            example: COCO example from tfds.

        Returns:
            Boolean tensor indicating if example is valid.
        """
        try:
            objects = example['objects']

            # FIXED: Simplified filter - only check for presence of objects/labels
            # Since we create synthetic segmentation from detection, we only need objects
            labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))
            has_objects = tf.greater(tf.shape(labels)[0], 0)

            # Optional: Additional validation for detection if enabled
            if self.use_detection:
                bboxes = objects.get('bbox', tf.zeros((0, 4)))
                has_valid_bboxes = tf.greater(tf.shape(bboxes)[0], 0)
                has_objects = tf.logical_and(has_objects, has_valid_bboxes)

            return has_objects

        except Exception as e:
            logger.debug(f"Error filtering example: {e}")
            return tf.constant(False, dtype=tf.bool)

    def _validate_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """
        Validate bounding boxes for correctness.

        Args:
            bbox: Bounding boxes [N, 4] in format [ymin, xmin, ymax, xmax] (normalized).

        Returns:
            Boolean mask indicating valid bounding boxes.
        """
        ymin, xmin, ymax, xmax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

        # Check coordinate bounds
        valid_coords = (
            (ymin >= 0.0) & (xmin >= 0.0) &
            (ymax <= 1.0) & (xmax <= 1.0)
        )

        # Check coordinate order
        valid_order = (ymax > ymin) & (xmax > xmin)

        # Check minimum area requirement
        height = (ymax - ymin) * tf.cast(self.img_size, tf.float32)
        width = (xmax - xmin) * tf.cast(self.img_size, tf.float32)
        area = height * width
        valid_area = area >= self.min_bbox_area

        return valid_coords & valid_order & valid_area

    def _preprocess_detection_targets(
        self,
        objects: Dict[str, tf.Tensor],
        image_shape: tf.Tensor
    ) -> tf.Tensor:
        """
        Preprocess detection targets from COCO format.

        Args:
            objects: Objects dictionary from COCO example.
            image_shape: Original image shape [height, width, channels].

        Returns:
            Detection targets [max_boxes, 5] in format [class_id, x1, y1, x2, y2].
        """
        try:
            # Get bounding boxes and labels
            bboxes = objects.get('bbox', tf.zeros((0, 4)))  # [ymin, xmin, ymax, xmax]
            labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))

            # Validate bboxes
            valid_mask = self._validate_bbox(bboxes)
            bboxes = tf.boolean_mask(bboxes, valid_mask)
            labels = tf.boolean_mask(labels, valid_mask)

            # Convert to [x1, y1, x2, y2] format
            ymin, xmin, ymax, xmax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            converted_bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)

            # Create detection targets [class_id, x1, y1, x2, y2]
            num_objects = tf.shape(labels)[0]
            detection_targets = tf.concat([
                tf.cast(tf.expand_dims(labels, 1), tf.float32),
                converted_bboxes
            ], axis=1)

            # Truncate if we have too many boxes, then pad to exact size
            current_boxes = tf.shape(detection_targets)[0]

            # Truncate to max_boxes if necessary
            detection_targets = detection_targets[:self.max_boxes_per_image]

            # Calculate padding needed (ensure non-negative)
            actual_boxes = tf.minimum(current_boxes, self.max_boxes_per_image)
            pad_size = self.max_boxes_per_image - actual_boxes

            # Use sentinel values for padding
            padding = tf.fill((pad_size, 5), INVALID_BBOX_VALUE)
            detection_targets = tf.concat([detection_targets, padding], axis=0)

            # Ensure exact size (should be guaranteed now, but safety check)
            detection_targets = detection_targets[:self.max_boxes_per_image]

            return detection_targets

        except Exception as e:
            logger.debug(f"Error preprocessing detection targets: {e}")
            # Return all invalid boxes as fallback
            return tf.fill((self.max_boxes_per_image, 5), INVALID_BBOX_VALUE)

    def _preprocess_segmentation_targets(
            self,
            objects: Dict[str, tf.Tensor],
            image_shape: tf.Tensor,
            example: Optional[Dict[str, tf.Tensor]] = None
    ) -> tf.Tensor:
        """
        Preprocess segmentation targets.

        FIXED VERSION: Uses a robust, vectorized method to create masks from bounding boxes,
        avoiding fragile loops and graph-mode errors with `tf.tensor_scatter_nd_update`.
        This version correctly handles the background class to prevent class ID collisions.

        Args:
            objects: Objects dictionary from COCO example.
            image_shape: Original image shape [height, width, channels].
            example: Full example dictionary (for dummy data).

        Returns:
            Segmentation masks [img_size, img_size, segmentation_classes].
        """
        try:
            # Check if we have dummy data with a pre-generated mask. This logic is unchanged.
            if example is not None and '_segmentation_mask' in example:
                mask = example['_segmentation_mask']
                # Resize to target size if needed
                if tf.shape(mask)[0] != self.img_size or tf.shape(mask)[1] != self.img_size:
                    mask = tf.image.resize(
                        mask,
                        [self.img_size, self.img_size],
                        method='nearest'
                    )
                return mask

            # Get bounding boxes and labels for real data.
            bboxes = objects.get('bbox', tf.zeros((0, 4)))  # [ymin, xmin, ymax, xmax]
            labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))

            # Initialize an integer mask for class indices. Class 0 is reserved for the background.
            final_mask = tf.zeros((self.img_size, self.img_size), dtype=tf.int32)

            # Only proceed if there are any bounding boxes to process.
            # This 'if' is safe for graph mode as it checks the tensor shape.
            if tf.shape(bboxes)[0] > 0:
                # Convert normalized bbox coordinates [0, 1] to absolute pixel coordinates.
                bboxes_px = bboxes * tf.cast(tf.stack(
                    [self.img_size, self.img_size, self.img_size, self.img_size]
                ), dtype=tf.float32)
                ymin, xmin, ymax, xmax = tf.unstack(bboxes_px, axis=1)

                # Create coordinate grids for vectorized operations.
                # y_coords shape: (img_size, 1), x_coords shape: (1, img_size)
                # This allows broadcasting to efficiently create boolean masks for each box.
                y_coords = tf.cast(tf.range(self.img_size)[:, tf.newaxis], tf.float32)
                x_coords = tf.cast(tf.range(self.img_size)[tf.newaxis, :], tf.float32)

                # Iterate through boxes from back to front.
                # This ensures smaller objects (often listed later) are drawn on top of larger ones.
                # Using tf.range makes the loop compatible with graph execution.
                for i in tf.range(tf.shape(bboxes)[0] - 1, -1, -1):
                    # BUG FIX #2: Add 1 to the class ID to avoid collision with background (class 0).
                    # Now, 'person' (original ID 0) becomes 1, and so on.
                    box_class_id = tf.cast(labels[i], tf.int32) + 1

                    # Get pixel coordinates for the current box.
                    by, bx, bh, bw = ymin[i], xmin[i], ymax[i], xmax[i]

                    # BUG FIX #1: Create a boolean mask for the current box using vectorized comparison.
                    # This is the graph-safe replacement for the fragile Python 'if' statement.
                    box_mask = (y_coords >= by) & (y_coords < bh) & (x_coords >= bx) & (x_coords < bw)

                    # Use tf.where to "paint" the class ID onto the final_mask.
                    # This is a robust, graph-safe conditional assignment.
                    # Where box_mask is True, use the new class_id; otherwise, keep the existing value.
                    final_mask = tf.where(box_mask, box_class_id, final_mask)

            # Convert the final integer mask into the format expected by the loss function.
            if self.segmentation_classes == 1:
                # For binary segmentation, any non-zero pixel is considered part of the mask.
                # Add a channel dimension at the end.
                return tf.cast(final_mask > 0, tf.float32)[..., tf.newaxis]
            else:
                # For multi-class, one-hot encode the integer mask.
                # The depth must be num_classes + 1 to account for our background class (0).
                # Our mask now has values from 0 (background) to 80.
                one_hot_mask = tf.one_hot(final_mask, depth=self.segmentation_classes + 1, dtype=tf.float32)

                # Slice off the background channel (channel 0) to match the model's output.
                # The model predicts 80 channels, not 81.
                return one_hot_mask[:, :, 1:]

        except Exception as e:
            # This error handling logic is unchanged and remains robust.
            logger.debug(f"Error preprocessing segmentation targets: {e}")
            # Return appropriate fallback shape
            if self.segmentation_classes == 1:
                return tf.zeros((self.img_size, self.img_size, 1), dtype=tf.float32)
            else:
                return tf.zeros((self.img_size, self.img_size, self.segmentation_classes), dtype=tf.float32)

    def _apply_color_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply color-based augmentations using tf.image.random_* functions."""
        if not self.augment_data:
            return image

        config = self.augmentation_config

        # Use tf.image.random_* functions for cleaner, more idiomatic code
        if config.brightness_delta > 0:
            image = tf.image.random_brightness(image, max_delta=config.brightness_delta)

        if config.contrast_delta > 0:
            # random_contrast takes a range [lower, upper]
            image = tf.image.random_contrast(
                image,
                lower=1.0 - config.contrast_delta,
                upper=1.0 + config.contrast_delta
            )

        if config.saturation_delta > 0:
            image = tf.image.random_saturation(
                image,
                lower=1.0 - config.saturation_delta,
                upper=1.0 + config.saturation_delta
            )

        if config.hue_delta > 0:
            image = tf.image.random_hue(image, max_delta=config.hue_delta)

        # Ensure values stay in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def _apply_horizontal_flip(
        self,
        image: tf.Tensor,
        targets: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Apply horizontal flip augmentation."""
        if (self.augment_data and
            tf.random.uniform([]) < self.augmentation_config.horizontal_flip_prob):

            # Flip image
            image = tf.image.flip_left_right(image)

            # Flip detection boxes if present
            if 'detection' in targets:
                bboxes = targets['detection']
                # Convert x coordinates: new_x = 1.0 - old_x
                class_ids = bboxes[:, 0:1]
                x1, y1, x2, y2 = bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]

                # Flip x coordinates
                new_x1 = 1.0 - x2
                new_x2 = 1.0 - x1

                flipped_bboxes = tf.stack([new_x1, y1, new_x2, y2], axis=1)
                targets['detection'] = tf.concat([class_ids, flipped_bboxes], axis=1)

            # Flip segmentation masks if present
            if 'segmentation' in targets:
                targets['segmentation'] = tf.image.flip_left_right(targets['segmentation'])

        return image, targets

    def _apply_vertical_flip(
        self,
        image: tf.Tensor,
        targets: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Apply vertical flip augmentation."""
        if (self.augment_data and
            tf.random.uniform([]) < self.augmentation_config.vertical_flip_prob):

            # Flip image
            image = tf.image.flip_up_down(image)

            # Flip detection boxes if present
            if 'detection' in targets:
                bboxes = targets['detection']
                # Convert y coordinates: new_y = 1.0 - old_y
                class_ids = bboxes[:, 0:1]
                x1, y1, x2, y2 = bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]

                # Flip y coordinates
                new_y1 = 1.0 - y2
                new_y2 = 1.0 - y1

                flipped_bboxes = tf.stack([x1, new_y1, x2, new_y2], axis=1)
                targets['detection'] = tf.concat([class_ids, flipped_bboxes], axis=1)

            # Flip segmentation masks if present
            if 'segmentation' in targets:
                targets['segmentation'] = tf.image.flip_up_down(targets['segmentation'])

        return image, targets

    def _apply_geometric_augmentations(
        self,
        image: tf.Tensor,
        targets: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Apply geometric augmentations."""
        if not self.augment_data:
            return image, targets

        # Apply horizontal flip
        image, targets = self._apply_horizontal_flip(image, targets)

        # Apply vertical flip
        image, targets = self._apply_vertical_flip(image, targets)

        # Note: Rotation is disabled by default due to complexity
        # Would require careful implementation of bbox coordinate transformation

        return image, targets

    def _preprocess_example(
        self,
        example: Dict[str, tf.Tensor],
        is_training: bool = True
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Preprocess a single example.

        Args:
            example: COCO example from tfds.
            is_training: Whether this is for training (enables augmentation).

        Returns:
            Tuple of (processed_image, targets_dict).
        """
        # Get image and resize
        image = example['image']
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [self.img_size, self.img_size])

        # Initialize targets dictionary
        targets = {}

        # Process detection targets
        if self.use_detection:
            targets['detection'] = self._preprocess_detection_targets(
                example['objects'], tf.shape(image)
            )

        # Process segmentation targets
        if self.use_segmentation:
            targets['segmentation'] = self._preprocess_segmentation_targets(
                example['objects'], tf.shape(image), example
            )

        # Apply augmentations during training
        if is_training:
            image = self._apply_color_augmentation(image)
            image, targets = self._apply_geometric_augmentations(image, targets)

        return image, targets

    def _process_dataset(
        self,
        dataset: tf.data.Dataset,
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Process dataset with preprocessing and efficient batching using padded_batch.

        Args:
            dataset: Input dataset.
            is_training: Whether this is training dataset.

        Returns:
            Processed and batched dataset.
        """
        # Apply filtering first to reduce data volume
        if is_training and self.limit_train_samples is not None:
            # Take a subset for memory efficiency during development
            dataset = dataset.take(self.limit_train_samples)
            logger.info(f"Limited training dataset to {self.limit_train_samples} samples for memory efficiency")

        # Preprocess examples
        dataset = dataset.map(
            lambda x: self._preprocess_example(x, is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if is_training:
            # IMPORTANT: The order of `shuffle`, `cache`, and `repeat` is critical
            # for both correctness and performance.

            # 1. Shuffle the entire dataset. This is the most important step for ensuring
            #    the model sees a different order of examples in each epoch.
            logger.info(f"Shuffling training dataset with buffer size: {self.shuffle_buffer_size}")
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

            # 2. Cache the result of the mapping and shuffling. Caching after shuffling
            #    is often a good tradeoff, as shuffling can be I/O intensive.
            #    The first epoch will be slow as it builds the cache. Subsequent epochs
            #    will be much faster as they read from the pre-shuffled, pre-processed cache.
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"coco_train_{self.img_size}_{self.segmentation_classes}")
                dataset = dataset.cache(cache_path)
                logger.info(f"Training dataset will be cached at: {cache_path}")

            # 3. Repeat the dataset for multiple epochs. Since the dataset is already
            #    shuffled, Keras will just pull from this infinite stream of data.
            #    A re-shuffling will happen when the original dataset is fully iterated through.
            dataset = dataset.repeat()
        else:
            # For validation, no need to shuffle.
            # Repeat is good practice to prevent OutOfRange errors if validation_steps is large.
            dataset = dataset.repeat()

        # Define shapes and padding values for efficient padded_batch
        image_shape = [self.img_size, self.img_size, 3]
        targets_shapes = {}
        padding_values = (0.0, {})  # 0.0 for images, dict for targets

        if self.use_detection:
            targets_shapes['detection'] = [self.max_boxes_per_image, 5]
            padding_values[1]['detection'] = INVALID_BBOX_VALUE

        if self.use_segmentation:
            targets_shapes['segmentation'] = [self.img_size, self.img_size, self.segmentation_classes]
            padding_values[1]['segmentation'] = 0.0

        # Use efficient padded_batch instead of batch + map
        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=(image_shape, targets_shapes),
            padding_values=padding_values,
            drop_remainder=is_training
        )

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create training and validation datasets.

        Returns:
            Tuple of (train_dataset, validation_dataset).
        """
        try:
            logger.info("🏗️ Creating COCO datasets...")

            # Load datasets with proper splits
            train_raw, val_raw = self._load_tfds_dataset()

            # Filter valid examples
            logger.info("🔍 Filtering valid examples...")
            train_raw = train_raw.filter(self._filter_valid_examples)
            val_raw = val_raw.filter(self._filter_valid_examples)

            # Process datasets
            logger.info("🏋️ Processing training dataset...")
            train_ds = self._process_dataset(train_raw, is_training=True)

            logger.info("🧪 Processing validation dataset...")
            val_ds = self._process_dataset(val_raw, is_training=False)

            logger.info("✅ Datasets created successfully!")
            return train_ds, val_ds

        except Exception as e:
            logger.error(f"❌ Failed to create datasets: {e}")
            raise

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'max_boxes_per_image': self.max_boxes_per_image,
            'use_detection': self.use_detection,
            'use_segmentation': self.use_segmentation,
            'segmentation_classes': self.segmentation_classes,  # NEW
            'using_dummy_data': self.using_dummy_data,  # NEW
            'augment_data': self.augment_data,
            'min_bbox_area': self.min_bbox_area,
            'cache_dir': self.cache_dir,
            'invalid_bbox_value': INVALID_BBOX_VALUE,  # Important for loss calculation
            'num_classes': self.dataset_config.num_classes,
            'class_names': self.dataset_config.class_names[:5] + ['...']
                          if len(self.dataset_config.class_names) > 5
                          else self.dataset_config.class_names,
            'total_classes': len(self.dataset_config.class_names),
            'augmentation_config': {
                'brightness_delta': self.augmentation_config.brightness_delta,
                'contrast_delta': self.augmentation_config.contrast_delta,
                'saturation_delta': self.augmentation_config.saturation_delta,
                'hue_delta': self.augmentation_config.hue_delta,
                'horizontal_flip_prob': self.augmentation_config.horizontal_flip_prob,
                'vertical_flip_prob': self.augmentation_config.vertical_flip_prob,
                'rotation_degrees': self.augmentation_config.rotation_degrees,
            },
            'pipeline_optimizations': [
                'Uses tf.data.padded_batch for efficient batching',
                'Sentinel value padding prevents false positives',
                'tf.image.random_* functions for optimized augmentation',
                'Proper train/validation splits from tfds',
                'Dataset repeats for multiple epochs',
                f'Fallback to dummy dataset: {"Yes" if self.using_dummy_data else "No"}',
                f'Configurable segmentation classes: {self.segmentation_classes}',
                'FIXED: Relaxed filtering only requires object presence',
                f'Memory-efficient shuffle buffer: {self.shuffle_buffer_size}',
                f'Sample limit: {self.limit_train_samples or "None"}'
            ]
        }

# ---------------------------------------------------------------------

def create_coco_dataset(
    img_size: int = 640,
    batch_size: int = 32,
    use_detection: bool = True,
    use_segmentation: bool = False,
    segmentation_classes: int = 80,  # NEW: Configurable segmentation classes
    augment_data: bool = True,
    class_names: Optional[List[str]] = None,
    shuffle_buffer_size: int = 100,  # NEW: Memory management
    limit_train_samples: Optional[int] = None,  # NEW: Memory management
    **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience function to create COCO datasets.

    Args:
        img_size: Target image size.
        batch_size: Batch size for training.
        use_detection: Enable detection task.
        use_segmentation: Enable segmentation task.
        segmentation_classes: Number of segmentation classes (80 for COCO, 1 for binary).
        augment_data: Enable data augmentation.
        class_names: Custom class names (optional).
        shuffle_buffer_size: Size of shuffle buffer (reduce if out of memory).
        limit_train_samples: Limit training samples for memory efficiency.
        **kwargs: Additional configuration options.

    Returns:
        Tuple of (train_dataset, validation_dataset).
    """
    logger.info("🚀 Creating COCO dataset with enhanced preprocessor")

    builder = COCODatasetBuilder(
        img_size=img_size,
        batch_size=batch_size,
        use_detection=use_detection,
        use_segmentation=use_segmentation,
        segmentation_classes=segmentation_classes,
        augment_data=augment_data,
        class_names=class_names,
        shuffle_buffer_size=shuffle_buffer_size,
        limit_train_samples=limit_train_samples,
        **kwargs
    )

    train_ds, val_ds = builder.create_datasets()

    # Log dataset information
    info = builder.get_dataset_info()
    logger.info("📋 Dataset configuration:")
    for key, value in info.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")

    return train_ds, val_ds

# ---------------------------------------------------------------------


def create_coco_pretraining_dataset(
    img_size: int = 640,
    batch_size: int = 32,
    **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create COCO dataset specifically configured for pretraining.

    Args:
        img_size: Target image size.
        batch_size: Batch size for training.
        **kwargs: Additional configuration options.

    Returns:
        Tuple of (train_dataset, validation_dataset).
    """
    return create_coco_dataset(
        img_size=img_size,
        batch_size=batch_size,
        use_detection=True,
        use_segmentation=True,
        segmentation_classes=80,  # COCO has 80 classes for segmentation
        augment_data=True,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_crack_dataset_placeholder(
    img_size: int = 640,
    batch_size: int = 32,
    **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create dataset configured for crack detection fine-tuning.

    Args:
        img_size: Target image size.
        batch_size: Batch size for training.
        **kwargs: Additional configuration options.

    Returns:
        Tuple of (train_dataset, validation_dataset).
    """
    # This would use your actual crack detection dataset
    # For now, create dummy dataset with binary segmentation
    return create_coco_dataset(
        img_size=img_size,
        batch_size=batch_size,
        use_detection=True,
        use_segmentation=True,
        segmentation_classes=1,  # Binary crack segmentation
        augment_data=True,
        **kwargs
    )

# ---------------------------------------------------------------------
