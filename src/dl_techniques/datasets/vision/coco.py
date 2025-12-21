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
    segmentation_classes: int = 80
) -> tf.data.Dataset:
    """
    Create a dummy COCO-style dataset for testing and development.

    Args:
        num_samples: Number of samples to generate.
        img_size: Input image size.
        num_classes: Number of object classes.
        max_boxes: Maximum number of boxes per image.
        min_boxes: Minimum number of boxes per image.
        segmentation_classes: Number of segmentation classes.

    Returns:
        TensorFlow dataset with COCO-style dictionary format.
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
            # NOTE: We must generate the exact shape expected by the output signature
            if segmentation_classes == 1:
                # Binary segmentation (H, W, 1) float32
                seg_mask = np.random.randint(0, 2, (img_size, img_size, 1)).astype(np.float32)
            else:
                # Multi-class segmentation (H, W, Classes) float32
                # Generate indices first
                seg_indices = np.random.randint(0, segmentation_classes, (img_size, img_size))
                # Convert to one-hot and ensure float32
                seg_mask = np.eye(segmentation_classes)[seg_indices].astype(np.float32)

            # Create dummy segmentation data that indicates presence
            dummy_segmentation = tf.constant(['dummy_segmentation'] * num_boxes, dtype=tf.string)

            example = {
                'image': img,
                'objects': {
                    'bbox': bboxes,
                    'label': labels,
                    'segmentation': dummy_segmentation
                },
                '_segmentation_mask': seg_mask
            }

            yield example

    # Define output signature matching the generator outputs exactly
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
        segmentation_classes: int = 80,
        shuffle_buffer_size: int = 1000,  # Default increased for better mixing
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
            shuffle_buffer_size: Size of shuffle buffer.
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

        # Segmentation configuration
        self.segmentation_classes = segmentation_classes

        # Memory management
        self.shuffle_buffer_size = shuffle_buffer_size
        self.limit_train_samples = limit_train_samples

        # Track if using dummy data
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
            logger.error(f"Failed to load COCO dataset: {e}")
            logger.info("Creating dummy dataset for testing...")

            self.using_dummy_data = True
            # Generate enough samples for multiple epochs
            samples_needed = max(10000, self.batch_size * 100)

            dummy_ds = create_dummy_coco_dataset(
                num_samples=samples_needed,
                img_size=self.img_size,
                segmentation_classes=self.segmentation_classes
            )

            # Split 80/20 for train/val
            train_size = int(samples_needed * 0.8)
            train_ds = dummy_ds.take(train_size)
            val_ds = dummy_ds.skip(train_size).take(int(samples_needed * 0.2))

            logger.info(f"Created dummy dataset with {train_size} training samples")
            return train_ds, val_ds

    def _filter_valid_examples(self, example: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Filter examples that have valid annotations for enabled tasks.

        Args:
            example: COCO example from tfds.

        Returns:
            Boolean tensor indicating if example is valid.
        """
        try:
            objects = example['objects']
            labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))
            has_objects = tf.greater(tf.shape(labels)[0], 0)

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
            bbox: Bounding boxes [N, 4] in format [ymin, xmin, ymax, xmax].

        Returns:
            Boolean mask indicating valid bounding boxes.
        """
        # Ensure float32 for calculations
        bbox = tf.cast(bbox, tf.float32)
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
            image_shape: Original image shape.

        Returns:
            Detection targets [max_boxes, 5] in format [class_id, x1, y1, x2, y2].
        """
        try:
            bboxes = objects.get('bbox', tf.zeros((0, 4)))  # [ymin, xmin, ymax, xmax]
            labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))

            # Validate bboxes
            valid_mask = self._validate_bbox(bboxes)
            bboxes = tf.boolean_mask(bboxes, valid_mask)
            labels = tf.boolean_mask(labels, valid_mask)

            # Convert to [x1, y1, x2, y2] format
            ymin, xmin, ymax, xmax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            converted_bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)

            # Create detection targets
            detection_targets = tf.concat([
                tf.cast(tf.expand_dims(labels, 1), tf.float32),
                converted_bboxes
            ], axis=1)

            # Truncate and pad
            current_boxes = tf.shape(detection_targets)[0]
            detection_targets = detection_targets[:self.max_boxes_per_image]

            actual_boxes = tf.minimum(current_boxes, self.max_boxes_per_image)
            pad_size = self.max_boxes_per_image - actual_boxes
            pad_size = tf.maximum(pad_size, 0)

            padding = tf.fill((pad_size, 5), INVALID_BBOX_VALUE)
            detection_targets = tf.concat([detection_targets, padding], axis=0)

            # Ensure exact static shape for graph mode if possible
            detection_targets = tf.ensure_shape(
                detection_targets, [self.max_boxes_per_image, 5]
            )

            return detection_targets

        except Exception as e:
            logger.debug(f"Error preprocessing detection targets: {e}")
            return tf.fill((self.max_boxes_per_image, 5), INVALID_BBOX_VALUE)

    def _preprocess_segmentation_targets(
            self,
            objects: Dict[str, tf.Tensor],
            image_shape: tf.Tensor,
            example: Optional[Dict[str, tf.Tensor]] = None
    ) -> tf.Tensor:
        """
        Preprocess segmentation targets using vectorized mask generation.

        Args:
            objects: Objects dictionary from COCO example.
            image_shape: Original image shape.
            example: Full example dictionary (for dummy data).

        Returns:
            Segmentation masks [img_size, img_size, channels].
        """
        try:
            # Handle dummy data path
            if example is not None and '_segmentation_mask' in example:
                mask = example['_segmentation_mask']
                if tf.shape(mask)[0] != self.img_size or tf.shape(mask)[1] != self.img_size:
                    mask = tf.image.resize(
                        mask,
                        [self.img_size, self.img_size],
                        method='nearest'
                    )
                return mask

            # Real data path
            bboxes = objects.get('bbox', tf.zeros((0, 4)))
            labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))

            # Initialize canvas (class 0 is background)
            final_mask = tf.zeros((self.img_size, self.img_size), dtype=tf.int32)

            if tf.shape(bboxes)[0] > 0:
                # Convert normalized coordinates to pixels
                bboxes_px = bboxes * tf.cast(tf.stack(
                    [self.img_size, self.img_size, self.img_size, self.img_size]
                ), dtype=tf.float32)
                ymin, xmin, ymax, xmax = tf.unstack(bboxes_px, axis=1)

                # Create coordinate grids
                y_coords = tf.cast(tf.range(self.img_size)[:, tf.newaxis], tf.float32)
                x_coords = tf.cast(tf.range(self.img_size)[tf.newaxis, :], tf.float32)

                # Iterate boxes backwards so later boxes overwrite earlier ones
                # Using tf.range makes this a graph-compatible loop
                for i in tf.range(tf.shape(bboxes)[0] - 1, -1, -1):
                    # Shift class ID by 1 to reserve 0 for background
                    box_class_id = tf.cast(labels[i], tf.int32) + 1
                    by, bx, bh, bw = ymin[i], xmin[i], ymax[i], xmax[i]

                    box_mask = (y_coords >= by) & (y_coords < bh) & \
                               (x_coords >= bx) & (x_coords < bw)

                    # Update mask only where box_mask is True
                    final_mask = tf.where(box_mask, box_class_id, final_mask)

            # Format output based on task type
            if self.segmentation_classes == 1:
                # Binary: Any non-zero class is foreground
                return tf.cast(final_mask > 0, tf.float32)[..., tf.newaxis]
            else:
                # Multi-class: One-hot encode
                # depth = classes + 1 (for background)
                one_hot = tf.one_hot(
                    final_mask,
                    depth=self.segmentation_classes + 1,
                    dtype=tf.float32
                )
                # Remove background channel to match model output (usually softmax across classes)
                return one_hot[:, :, 1:]

        except Exception as e:
            logger.debug(f"Error preprocessing segmentation targets: {e}")
            channels = 1 if self.segmentation_classes == 1 else self.segmentation_classes
            return tf.zeros((self.img_size, self.img_size, channels), dtype=tf.float32)

    def _apply_color_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply color-based augmentations."""
        if not self.augment_data:
            return image

        config = self.augmentation_config

        # Random brightness
        if config.brightness_delta > 0:
            image = tf.image.random_brightness(image, max_delta=config.brightness_delta)

        # Random contrast
        if config.contrast_delta > 0:
            image = tf.image.random_contrast(
                image,
                lower=1.0 - config.contrast_delta,
                upper=1.0 + config.contrast_delta
            )

        # Random saturation
        if config.saturation_delta > 0:
            image = tf.image.random_saturation(
                image,
                lower=1.0 - config.saturation_delta,
                upper=1.0 + config.saturation_delta
            )

        # Random hue
        if config.hue_delta > 0:
            image = tf.image.random_hue(image, max_delta=config.hue_delta)

        return tf.clip_by_value(image, 0.0, 1.0)

    def _apply_horizontal_flip(
        self,
        image: tf.Tensor,
        targets: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Apply horizontal flip augmentation using tf.cond."""
        if not self.augment_data:
            return image, targets

        do_flip = tf.random.uniform([]) < self.augmentation_config.horizontal_flip_prob

        def flip_fn():
            new_image = tf.image.flip_left_right(image)
            new_targets = targets.copy()

            if self.use_detection:
                bboxes = new_targets['detection']
                class_ids = bboxes[:, 0:1]
                x1, y1, x2, y2 = bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]
                # Flip x: new_x = 1 - old_x
                new_x1 = 1.0 - x2
                new_x2 = 1.0 - x1
                flipped_bboxes = tf.stack([new_x1, y1, new_x2, y2], axis=1)
                new_targets['detection'] = tf.concat([class_ids, flipped_bboxes], axis=1)

            if self.use_segmentation:
                new_targets['segmentation'] = tf.image.flip_left_right(new_targets['segmentation'])

            return new_image, new_targets

        def no_flip_fn():
            return image, targets

        return tf.cond(do_flip, flip_fn, no_flip_fn)

    def _apply_vertical_flip(
        self,
        image: tf.Tensor,
        targets: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Apply vertical flip augmentation using tf.cond."""
        if not self.augment_data:
            return image, targets

        do_flip = tf.random.uniform([]) < self.augmentation_config.vertical_flip_prob

        def flip_fn():
            new_image = tf.image.flip_up_down(image)
            new_targets = targets.copy()

            if self.use_detection:
                bboxes = new_targets['detection']
                class_ids = bboxes[:, 0:1]
                x1, y1, x2, y2 = bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]
                # Flip y: new_y = 1 - old_y
                new_y1 = 1.0 - y2
                new_y2 = 1.0 - y1
                flipped_bboxes = tf.stack([x1, new_y1, x2, new_y2], axis=1)
                new_targets['detection'] = tf.concat([class_ids, flipped_bboxes], axis=1)

            if self.use_segmentation:
                new_targets['segmentation'] = tf.image.flip_up_down(new_targets['segmentation'])

            return new_image, new_targets

        def no_flip_fn():
            return image, targets

        return tf.cond(do_flip, flip_fn, no_flip_fn)

    def _preprocess_example(
        self,
        example: Dict[str, tf.Tensor],
        is_training: bool = True
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Preprocess a single example. Resizes, generates targets, and augments.
        """
        # Get image and resize
        image = example['image']
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [self.img_size, self.img_size])

        targets = {}

        if self.use_detection:
            targets['detection'] = self._preprocess_detection_targets(
                example['objects'], tf.shape(image)
            )

        if self.use_segmentation:
            targets['segmentation'] = self._preprocess_segmentation_targets(
                example['objects'], tf.shape(image), example
            )

        if is_training:
            image = self._apply_color_augmentation(image)
            image, targets = self._apply_horizontal_flip(image, targets)
            image, targets = self._apply_vertical_flip(image, targets)

        return image, targets

    def _process_dataset(
        self,
        dataset: tf.data.Dataset,
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Process dataset with optimized pipeline order for efficiency and correctness.
        """
        # 1. Filter and limit first (works on raw data, low cost)
        if is_training and self.limit_train_samples is not None:
            dataset = dataset.take(self.limit_train_samples)
            logger.info(f"Limited training dataset to {self.limit_train_samples} samples")

        # 2. Caching Strategy
        # We cache the raw/filtered data BEFORE augmentation to avoid freezing random augmentations.
        # However, since TFDS reads from disk are reasonably fast and raw images are compressed,
        # caching decoded images (float32) uses massive RAM.
        # We optionally cache here if specifically requested, but it's often better to stream.
        if self.cache_dir and is_training:
            cache_path = os.path.join(self.cache_dir, "coco_filtered_raw")
            # dataset = dataset.cache(cache_path) # Uncomment if disk I/O is the bottleneck
            pass

        # 3. Shuffle (CRITICAL: Shuffle before mapping/augmentation)
        # Shuffling here involves raw records (small), not decoded images (huge).
        if is_training:
            logger.info(f"Shuffling dataset with buffer size: {self.shuffle_buffer_size}")
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # 4. Repeat (Must happen before map to allow unique augmentations per epoch)
        if is_training:
            dataset = dataset.repeat()

        # 5. Map (Preprocess + Augment)
        # Now applied dynamically to the shuffled, repeated stream
        dataset = dataset.map(
            lambda x: self._preprocess_example(x, is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 6. Batching
        image_shape = [self.img_size, self.img_size, 3]
        targets_shapes = {}
        padding_values = (0.0, {})

        if self.use_detection:
            targets_shapes['detection'] = [self.max_boxes_per_image, 5]
            padding_values[1]['detection'] = INVALID_BBOX_VALUE

        if self.use_segmentation:
            targets_shapes['segmentation'] = [
                self.img_size,
                self.img_size,
                1 if self.segmentation_classes == 1 else self.segmentation_classes
            ]
            padding_values[1]['segmentation'] = 0.0

        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=(image_shape, targets_shapes),
            padding_values=padding_values,
            drop_remainder=is_training
        )

        # 7. Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create training and validation datasets."""
        try:
            logger.info("Creating COCO datasets...")

            train_raw, val_raw = self._load_tfds_dataset()

            logger.info("Filtering valid examples...")
            train_raw = train_raw.filter(self._filter_valid_examples)
            val_raw = val_raw.filter(self._filter_valid_examples)

            logger.info("Processing training dataset...")
            train_ds = self._process_dataset(train_raw, is_training=True)

            logger.info("Processing validation dataset...")
            val_ds = self._process_dataset(val_raw, is_training=False)

            return train_ds, val_ds

        except Exception as e:
            logger.error(f"Failed to create datasets: {e}")
            raise

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'max_boxes_per_image': self.max_boxes_per_image,
            'use_detection': self.use_detection,
            'use_segmentation': self.use_segmentation,
            'segmentation_classes': self.segmentation_classes,
            'using_dummy_data': self.using_dummy_data,
            'augment_data': self.augment_data,
            'num_classes': self.dataset_config.num_classes,
            'pipeline_order': 'Shuffle -> Repeat -> Map(Augment) -> Batch',
        }

# ---------------------------------------------------------------------


def create_coco_dataset(
    img_size: int = 640,
    batch_size: int = 32,
    use_detection: bool = True,
    use_segmentation: bool = False,
    segmentation_classes: int = 80,
    augment_data: bool = True,
    class_names: Optional[List[str]] = None,
    shuffle_buffer_size: int = 1000,
    limit_train_samples: Optional[int] = None,
    **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience function to create COCO datasets.
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


def create_coco_pretraining_dataset(
    img_size: int = 640,
    batch_size: int = 32,
    **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create COCO dataset specifically configured for pretraining.
    """
    return create_coco_dataset(
        img_size=img_size,
        batch_size=batch_size,
        use_detection=True,
        use_segmentation=True,
        segmentation_classes=80,
        augment_data=True,
        **kwargs
    )

# ---------------------------------------------------------------------