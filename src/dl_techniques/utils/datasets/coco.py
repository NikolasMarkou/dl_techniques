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
    from coco_preprocessor import COCODatasetBuilder

    builder = COCODatasetBuilder(
        img_size=640,
        batch_size=16,
        cache_dir="/path/to/cache"
    )

    train_ds, val_ds = builder.create_datasets()
    ```

File: src/dl_techniques/utils/datasets/coco_yolo12.py
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

from dl_techniques.utils.logger import logger

# COCO 2017 has 80 object classes
COCO_NUM_CLASSES = 80



# ---------------------------------------------------------------------

def create_dummy_coco_dataset(
    num_samples: int,
    img_size: int,
    num_classes: int = 80,
    max_boxes: int = 20,
    min_boxes: int = 1
) -> tf.data.Dataset:
    """
    Create a dummy COCO-style dataset for object detection training.

    Args:
        num_samples: Number of samples to generate.
        img_size: Input image size.
        num_classes: Number of object classes.
        max_boxes: Maximum number of boxes per image.
        min_boxes: Minimum number of boxes per image.

    Returns:
        TensorFlow dataset with (image, labels) pairs.
        Labels format: (class_id, x1, y1, x2, y2) in absolute coordinates.
    """

    def generator():
        for _ in range(num_samples):
            # Generate dummy image
            img = np.random.rand(img_size, img_size, 3).astype(np.float32)

            # Generate random number of boxes
            num_boxes = np.random.randint(min_boxes, max_boxes + 1)

            # Initialize labels array
            labels = np.zeros((max_boxes, 5), dtype=np.float32)

            for i in range(num_boxes):
                # Random class
                cls_id = np.random.randint(0, num_classes)

                # Random box coordinates (ensure valid boxes)
                x1 = np.random.uniform(0, img_size * 0.8)
                y1 = np.random.uniform(0, img_size * 0.8)
                x2 = np.random.uniform(x1 + 20, min(x1 + 200, img_size))
                y2 = np.random.uniform(y1 + 20, min(y1 + 200, img_size))

                labels[i] = [cls_id, x1, y1, x2, y2]

            yield img, labels

    output_signature = (
        tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(max_boxes, 5), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset

# ---------------------------------------------------------------------

class COCODatasetBuilder:
    """
    COCO Dataset Builder for YOLOv12 Pre-training.

    This class handles loading and preprocessing the COCO 2017 dataset
    for use with YOLOv12 multi-task models during pre-training.

    Args:
        img_size: Target image size (will be resized to img_size x img_size).
        batch_size: Batch size for training.
        num_classes: Number of COCO classes (default: 80).
        max_boxes_per_image: Maximum number of bounding boxes per image.
        cache_dir: Directory to cache the dataset (optional).
        use_detection: Whether to include detection targets.
        use_segmentation: Whether to include segmentation targets.
        augment_data: Whether to apply data augmentation.
    """
    def __init__(
            self,
            img_size: int = 640,
            batch_size: int = 16,
            num_classes: int = 80,
            max_boxes_per_image: int = 100,
            cache_dir: Optional[str] = None,
            use_detection: bool = True,
            use_segmentation: bool = True,
            augment_data: bool = True
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.max_boxes_per_image = max_boxes_per_image
        self.cache_dir = cache_dir
        self.use_detection = use_detection
        self.use_segmentation = use_segmentation
        self.augment_data = augment_data

        # Validate parameters
        if num_classes != self.COCO_NUM_CLASSES:
            logger.warning(f"COCO has {self.COCO_NUM_CLASSES} classes, but num_classes={num_classes}")

        logger.info(f"COCODatasetBuilder initialized:")
        logger.info(f"  Image size: {img_size}x{img_size}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Detection: {use_detection}, Segmentation: {use_segmentation}")
        logger.info(f"  Data augmentation: {augment_data}")

    def _load_coco_dataset(self, split: str) -> tf.data.Dataset:
        """
        Load COCO dataset using tensorflow_datasets.

        Args:
            split: Dataset split ('train', 'validation').

        Returns:
            Raw COCO dataset.
        """
        try:
            logger.info(f"Loading COCO 2017 {split} split...")

            # Configure download directory if specified
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                data_dir = self.cache_dir
            else:
                data_dir = None

            # Load dataset
            dataset, info = tfds.load(
                'coco/2017',
                split=split,
                with_info=True,
                shuffle_files=True,
                data_dir=data_dir,
                as_supervised=False
            )

            logger.info(f"✓ COCO {split} dataset loaded successfully")
            logger.info(f"  Number of examples: {info.splits[split].num_examples}")
            logger.info(f"  Features: {list(info.features.keys())}")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load COCO dataset: {e}")
            logger.error("Make sure tensorflow_datasets is installed: pip install tensorflow-datasets")
            raise

    def _preprocess_detection_targets(self, objects: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Convert COCO detection annotations to YOLOv12 format.

        Args:
            objects: COCO objects dictionary containing bboxes and labels.

        Returns:
            Detection targets tensor with shape [num_boxes, 5] where each row is
            [class_id, x1, y1, x2, y2] in normalized coordinates.
        """
        # Extract bounding boxes and labels
        bboxes = objects['bbox']  # [num_boxes, 4] in [y_min, x_min, y_max, x_max] format
        labels = tf.cast(objects['label'], tf.float32)  # [num_boxes]

        # COCO bboxes are in [y_min, x_min, y_max, x_max] format (normalized)
        # Convert to [x1, y1, x2, y2] format for YOLOv12
        x1 = bboxes[:, 1]  # x_min
        y1 = bboxes[:, 0]  # y_min
        x2 = bboxes[:, 3]  # x_max
        y2 = bboxes[:, 2]  # y_max

        # Stack into final format: [class_id, x1, y1, x2, y2]
        detection_targets = tf.stack([labels, x1, y1, x2, y2], axis=1)

        return detection_targets

    def _preprocess_segmentation_targets(
            self,
            objects: Dict[str, tf.Tensor],
            image_shape: tf.Tensor
    ) -> tf.Tensor:
        """
        Convert COCO segmentation masks to binary segmentation target.

        Args:
            objects: COCO objects dictionary containing segmentation masks.
            image_shape: Original image shape.

        Returns:
            Binary segmentation mask with shape [img_size, img_size, 1].
        """
        # Get instance masks - shape: [num_instances, height, width]
        masks = objects['mask']

        # Convert to float32
        masks = tf.cast(masks, tf.float32)

        # Combine all instance masks into a single binary mask (union)
        # Any pixel that belongs to any object becomes 1
        combined_mask = tf.reduce_max(masks, axis=0)  # [height, width]

        # Resize to target size
        combined_mask = tf.expand_dims(combined_mask, axis=-1)  # [height, width, 1]
        combined_mask = tf.image.resize(
            combined_mask,
            [self.img_size, self.img_size],
            method='nearest'
        )

        # Ensure binary values
        combined_mask = tf.cast(combined_mask > 0.5, tf.float32)

        return combined_mask

    def _apply_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply data augmentation to training images.

        Args:
            image: Input image tensor.

        Returns:
            Augmented image tensor.
        """
        if not self.augment_data:
            return image

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Random saturation
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

        # Random hue (small change)
        image = tf.image.random_hue(image, max_delta=0.05)

        # Ensure values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def _preprocess_example(self, element: Dict[str, Any]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Preprocess a single COCO example for YOLOv12 training.

        Args:
            element: Raw COCO example dictionary.

        Returns:
            Tuple of (preprocessed_image, targets_dict).
        """
        # Extract and preprocess image
        image = element['image']
        original_shape = tf.shape(image)

        # Resize image to target size
        image = tf.image.resize(image, [self.img_size, self.img_size])
        image = tf.cast(image, tf.float32) / 255.0

        # Apply augmentation if enabled
        image = self._apply_augmentation(image)

        # Prepare targets dictionary
        targets = {}

        # Process detection targets if enabled
        if self.use_detection:
            detection_targets = self._preprocess_detection_targets(element['objects'])
            targets['detection'] = detection_targets

        # Process segmentation targets if enabled
        if self.use_segmentation:
            segmentation_targets = self._preprocess_segmentation_targets(
                element['objects'],
                original_shape
            )
            targets['segmentation'] = segmentation_targets

        return image, targets

    def _filter_valid_examples(self, element: Dict[str, Any]) -> bool:
        """
        Filter out examples that don't have valid annotations.

        Args:
            element: COCO example dictionary.

        Returns:
            True if example should be kept, False otherwise.
        """
        # Check if there are any objects
        num_objects = tf.shape(element['objects']['bbox'])[0]

        # Keep examples with at least one object
        return num_objects > 0

    def _create_padded_batch_shapes(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Define padded batch shapes for variable-length sequences.

        Returns:
            Tuple of (image_shape, targets_shapes_dict).
        """
        image_shape = [self.img_size, self.img_size, 3]

        targets_shapes = {}

        if self.use_detection:
            # Pad detection targets to max_boxes_per_image
            targets_shapes['detection'] = [self.max_boxes_per_image, 5]

        if self.use_segmentation:
            # Segmentation targets have fixed shape
            targets_shapes['segmentation'] = [self.img_size, self.img_size, 1]

        return image_shape, targets_shapes

    def create_datasets(
            self,
            train_split: str = 'train',
            val_split: str = 'validation'
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create training and validation datasets.

        Args:
            train_split: Training split name.
            val_split: Validation split name.

        Returns:
            Tuple of (train_dataset, validation_dataset).
        """
        logger.info("Creating COCO datasets for YOLOv12 pre-training...")

        # Load raw datasets
        train_raw = self._load_coco_dataset(train_split)
        val_raw = self._load_coco_dataset(val_split)

        # Filter valid examples
        train_raw = train_raw.filter(self._filter_valid_examples)
        val_raw = val_raw.filter(self._filter_valid_examples)

        # Apply preprocessing
        train_ds = train_raw.map(
            self._preprocess_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        val_ds = val_raw.map(
            self._preprocess_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Get padded batch shapes
        image_shape, targets_shapes = self._create_padded_batch_shapes()

        # Create padded batches
        train_ds = train_ds.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=(image_shape, targets_shapes),
            padding_values=(0.0, {
                k: 0.0 for k in targets_shapes.keys()
            }),
            drop_remainder=True
        )

        val_ds = val_ds.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=(image_shape, targets_shapes),
            padding_values=(0.0, {
                k: 0.0 for k in targets_shapes.keys()
            }),
            drop_remainder=False
        )

        # Optimize performance
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        # Cache if directory specified
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
            train_cache = str(cache_path / "train_cache")
            val_cache = str(cache_path / "val_cache")

            train_ds = train_ds.cache(train_cache)
            val_ds = val_ds.cache(val_cache)

            logger.info(f"Datasets cached to {self.cache_dir}")

        logger.info("✓ COCO datasets created successfully")
        return train_ds, val_ds

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset configuration.

        Returns:
            Dictionary containing dataset information.
        """
        return {
            'dataset_name': 'COCO 2017',
            'num_classes': self.num_classes,
            'image_size': f"{self.img_size}x{self.img_size}",
            'batch_size': self.batch_size,
            'max_boxes_per_image': self.max_boxes_per_image,
            'tasks': {
                'detection': self.use_detection,
                'segmentation': self.use_segmentation
            },
            'augmentation': self.augment_data
        }


def create_coco_dataset(
        img_size: int = 640,
        batch_size: int = 16,
        cache_dir: Optional[str] = None,
        **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience function to create COCO datasets.

    Args:
        img_size: Target image size.
        batch_size: Batch size.
        cache_dir: Cache directory (optional).
        **kwargs: Additional arguments for COCODatasetBuilder.

    Returns:
        Tuple of (train_dataset, validation_dataset).
    """
    builder = COCODatasetBuilder(
        img_size=img_size,
        batch_size=batch_size,
        cache_dir=cache_dir,
        **kwargs
    )

    return builder.create_datasets()


# Example usage
if __name__ == "__main__":
    # Test the dataset builder
    builder = COCODatasetBuilder(
        img_size=640,
        batch_size=4,  # Small batch for testing
        use_detection=True,
        use_segmentation=True,
        augment_data=True
    )

    print("Dataset info:", builder.get_dataset_info())

    try:
        train_ds, val_ds = builder.create_datasets()

        # Test by taking one batch
        sample_batch = next(iter(train_ds))
        images, targets = sample_batch

        print(f"✓ Sample batch loaded successfully")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets keys: {list(targets.keys())}")

        if 'detection' in targets:
            print(f"  Detection targets shape: {targets['detection'].shape}")
        if 'segmentation' in targets:
            print(f"  Segmentation targets shape: {targets['segmentation'].shape}")

    except Exception as e:
        print(f"✗ Error testing dataset: {e}")