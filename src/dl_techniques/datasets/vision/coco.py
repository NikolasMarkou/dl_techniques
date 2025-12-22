"""
COCO Dataset Preprocessor for YOLOv12 Pre-training.

This module provides a production-grade pipeline to load, preprocess, and create
TensorFlow datasets from the COCO 2017 dataset for pre-training YOLOv12 architectures.

It handles the complexity of loading COCO data via ``tensorflow_datasets``, validating
annotations, performing data augmentation, and formatting targets for object detection
and segmentation models.

Usage:
    .. code-block:: python

        from dl_techniques.utils.datasets.coco import COCODatasetBuilder

        # Initialize builder
        builder = COCODatasetBuilder(
            img_size=640,
            batch_size=32,
            use_detection=True,
            use_segmentation=True,
            segmentation_classes=80
        )

        # Create datasets
        train_ds, val_ds = builder.create_datasets()

"""

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
# Constants
# ---------------------------------------------------------------------

COCO_CLASSES: List[str] = [
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

COCO_NUM_CLASSES: int = len(COCO_CLASSES)
INVALID_BBOX_VALUE: float = -1.0


# ---------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation parameters.

    :param brightness_delta: Maximum delta for random brightness adjustment.
    :type brightness_delta: float
    :param contrast_delta: Factor for random contrast adjustment.
    :type contrast_delta: float
    :param saturation_delta: Factor for random saturation adjustment.
    :type saturation_delta: float
    :param hue_delta: Maximum delta for random hue adjustment.
    :type hue_delta: float
    :param horizontal_flip_prob: Probability of applying horizontal flip.
    :type horizontal_flip_prob: float
    :param vertical_flip_prob: Probability of applying vertical flip.
    :type vertical_flip_prob: float
    """
    brightness_delta: float = 0.1
    contrast_delta: float = 0.1
    saturation_delta: float = 0.1
    hue_delta: float = 0.05
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0


@dataclass
class DatasetConfig:
    """
    Configuration for dataset class mapping and filtering.

    :param class_names: List of class names to include.
    :type class_names: List[str]
    :param class_mapping: Optional mapping from source ID to target ID.
    :type class_mapping: Dict[int, int]
    """
    class_names: List[str]
    class_mapping: Dict[int, int] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        """
        Get number of classes.

        :return: Count of classes.
        :rtype: int
        """
        return len(self.class_names)

    @classmethod
    def from_class_names(cls, class_names: List[str]) -> 'DatasetConfig':
        """
        Create config from list of class names.

        :param class_names: List of class strings.
        :return: DatasetConfig instance.
        """
        return cls(class_names=class_names)

    @classmethod
    def coco_default(cls) -> 'DatasetConfig':
        """
        Create default COCO configuration.

        :return: DatasetConfig instance with 80 COCO classes.
        """
        return cls(class_names=COCO_CLASSES)


# ---------------------------------------------------------------------
# Dummy Data Generation
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

    :param num_samples: Number of samples to generate.
    :param img_size: Input image size (square).
    :param num_classes: Number of object classes.
    :param max_boxes: Maximum number of boxes per image.
    :param min_boxes: Minimum number of boxes per image.
    :param segmentation_classes: Number of segmentation classes.
    :return: TensorFlow dataset yielding dictionary examples.
    """
    def generator():
        for _ in range(num_samples):
            # Generate dummy image
            img = np.random.uniform(0, 255, (img_size, img_size, 3)).astype(np.uint8)

            # Generate random number of boxes
            num_boxes = np.random.randint(max(min_boxes, 1), max_boxes + 1)

            # Create bounding boxes in COCO format [ymin, xmin, ymax, xmax]
            bboxes = []
            labels = []

            for _ in range(num_boxes):
                cls_id = np.random.randint(0, num_classes)
                # Random box coordinates (normalized)
                ymin = np.random.uniform(0, 0.8)
                xmin = np.random.uniform(0, 0.8)
                ymax = np.random.uniform(ymin + 0.05, min(ymin + 0.4, 1.0))
                xmax = np.random.uniform(xmin + 0.05, min(xmin + 0.4, 1.0))
                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(cls_id)

            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            # Generate dummy segmentation mask indices
            seg_indices = np.random.randint(0, segmentation_classes, (img_size, img_size, 1))
            seg_mask = seg_indices.astype(np.float32)

            # Dummy segmentation polygons (required by signature, though unused)
            dummy_segmentation = tf.constant(['dummy_poly'] * num_boxes, dtype=tf.string)

            yield {
                'image': img,
                'objects': {
                    'bbox': bboxes,
                    'label': labels,
                    'segmentation': dummy_segmentation
                },
                '_segmentation_mask': seg_mask
            }

    output_signature = {
        'image': tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.uint8),
        'objects': {
            'bbox': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'segmentation': tf.TensorSpec(shape=(None,), dtype=tf.string)
        },
        '_segmentation_mask': tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32)
    }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset


# ---------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------

class COCODatasetBuilder:
    """
    Enhanced COCO Dataset Builder with robust preprocessing pipeline.

    This class orchestrates the loading, filtering, augmentation, and formatting
    of COCO data for deep learning models.

    :param img_size: Target image size for resizing.
    :param batch_size: Batch size for training.
    :param max_boxes_per_image: Maximum number of boxes per image (for padding).
    :param cache_dir: Directory for caching processed data.
    :param use_detection: Enable detection task (generate bbox targets).
    :param use_segmentation: Enable segmentation task (generate mask targets).
    :param augment_data: Enable data augmentation.
    :param augmentation_config: Custom augmentation configuration.
    :param min_bbox_area: Minimum bounding box area in pixels to consider valid.
    :param class_names: Custom class names (overrides dataset_config).
    :param dataset_config: Custom dataset configuration.
    :param data_dir: Directory where COCO data is stored.
    :param segmentation_classes: Number of segmentation classes.
    :param shuffle_buffer_size: Size of shuffle buffer.
    :param limit_train_samples: Limit training samples for memory efficiency.
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
        shuffle_buffer_size: int = 10000,
        limit_train_samples: Optional[int] = None,
        **kwargs: Any
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_boxes_per_image = max_boxes_per_image
        self.cache_dir = cache_dir
        self.use_detection = use_detection
        self.use_segmentation = use_segmentation
        self.augment_data = augment_data
        self.min_bbox_area = min_bbox_area
        self.data_dir = data_dir
        self.segmentation_classes = segmentation_classes
        self.shuffle_buffer_size = shuffle_buffer_size
        self.limit_train_samples = limit_train_samples
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

        self._validate_configuration()

        logger.info(f"COCODatasetBuilder initialized:")
        logger.info(f"  - Image size: {self.img_size}")
        logger.info(f"  - Detection: {self.use_detection}")
        logger.info(f"  - Segmentation: {self.use_segmentation}")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        if not self.use_detection and not self.use_segmentation:
            raise ValueError("At least one task (detection or segmentation) must be enabled")
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _load_tfds_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load COCO dataset using tensorflow_datasets with proper splits.

        :return: Tuple of (train_dataset, validation_dataset).
        """
        try:
            load_kwargs = {
                'as_supervised': False,
                'with_info': False,
                'shuffle_files': True,
            }
            if self.data_dir:
                load_kwargs['data_dir'] = self.data_dir

            logger.info("Loading COCO 2017 dataset...")
            train_ds = tfds.load('coco/2017', split='train', **load_kwargs)
            val_ds = tfds.load('coco/2017', split='validation', **load_kwargs)

            self.using_dummy_data = False
            return train_ds, val_ds

        except Exception as e:
            logger.error(f"Failed to load COCO dataset: {e}")
            logger.info("Falling back to DUMMY dataset for development...")

            self.using_dummy_data = True
            # Generate minimal sufficient samples
            samples_needed = max(1000, self.batch_size * 50)

            dummy_ds = create_dummy_coco_dataset(
                num_samples=samples_needed,
                img_size=self.img_size,
                segmentation_classes=self.segmentation_classes
            )

            train_size = int(samples_needed * 0.8)
            train_ds = dummy_ds.take(train_size)
            val_ds = dummy_ds.skip(train_size)

            return train_ds, val_ds

    def _filter_valid_examples(self, example: Dict[str, Any]) -> tf.Tensor:
        """
        Filter examples ensuring they have valid objects/boxes.
        """
        objects = example.get('objects', {})
        labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))
        has_objects = tf.shape(labels)[0] > 0

        if self.use_detection:
            bboxes = objects.get('bbox', tf.zeros((0, 4)))
            has_valid_bboxes = tf.shape(bboxes)[0] > 0
            has_objects = tf.logical_and(has_objects, has_valid_bboxes)

        return has_objects

    def _validate_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """
        Validate bounding boxes for correctness.

        :param bbox: Bounding boxes [N, 4] in format [ymin, xmin, ymax, xmax].
        :return: Boolean mask indicating valid bounding boxes.
        """
        bbox = tf.cast(bbox, tf.float32)
        ymin, xmin, ymax, xmax = tf.unstack(bbox, axis=-1)

        # Check bounds (0-1)
        valid_coords = (ymin >= 0.0) & (xmin >= 0.0) & (ymax <= 1.0) & (xmax <= 1.0)
        # Check ordering
        valid_order = (ymax > ymin) & (xmax > xmin)
        # Check area
        height = (ymax - ymin) * float(self.img_size)
        width = (xmax - xmin) * float(self.img_size)
        valid_area = (height * width) >= self.min_bbox_area

        return valid_coords & valid_order & valid_area

    def _preprocess_detection_targets(
        self,
        objects: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """
        Preprocess detection targets from COCO format.

        Converts [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax] and pads.

        :param objects: Objects dictionary from COCO example.
        :return: Targets [max_boxes, 5] (class_id, x1, y1, x2, y2).
        """
        bboxes = objects.get('bbox', tf.zeros((0, 4), dtype=tf.float32))
        labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))

        # Filter invalid boxes
        valid_mask = self._validate_bbox(bboxes)
        bboxes = tf.boolean_mask(bboxes, valid_mask)
        labels = tf.boolean_mask(labels, valid_mask)

        # Convert coordinates
        ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=-1)
        converted_bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)

        # Concat labels
        targets = tf.concat([
            tf.cast(labels[:, tf.newaxis], tf.float32),
            converted_bboxes
        ], axis=1)

        # Pad/Truncate
        num_boxes = tf.shape(targets)[0]
        targets = targets[:self.max_boxes_per_image]

        pad_size = tf.maximum(self.max_boxes_per_image - num_boxes, 0)
        padding = tf.fill([pad_size, 5], INVALID_BBOX_VALUE)

        final_targets = tf.concat([targets, padding], axis=0)
        final_targets = tf.ensure_shape(final_targets, [self.max_boxes_per_image, 5])

        return final_targets

    def _preprocess_segmentation_targets(
        self,
        objects: Dict[str, tf.Tensor],
        example: Optional[Dict[str, Any]] = None
    ) -> tf.Tensor:
        """
        Preprocess segmentation targets using box-supervised rasterization.

        :param objects: Objects dictionary.
        :param example: Full example (for dummy data bypass).
        :return: Segmentation masks [H, W, Channels].
        """
        # 1. Dummy Data / Pre-computed Path
        if example is not None and '_segmentation_mask' in example:
            mask = example['_segmentation_mask']
            if tf.shape(mask)[0] != self.img_size:
                mask = tf.image.resize(mask, [self.img_size, self.img_size], method='nearest')
            # Handle class dimension
            if self.segmentation_classes > 1 and tf.shape(mask)[-1] == 1:
                mask = tf.squeeze(mask, axis=-1)
                mask = tf.one_hot(tf.cast(mask, tf.int32), self.segmentation_classes)
            return tf.cast(mask, tf.float32)

        # 2. Rasterization Path
        bboxes = objects.get('bbox', tf.zeros((0, 4), dtype=tf.float32))
        labels = objects.get('label', tf.zeros((0,), dtype=tf.int64))

        valid_mask = self._validate_bbox(bboxes)
        bboxes = tf.boolean_mask(bboxes, valid_mask)
        labels = tf.boolean_mask(labels, valid_mask)

        # Prepare Canvas
        canvas = tf.zeros((self.img_size, self.img_size), dtype=tf.int32)

        # Coordinate grids
        y_grid = tf.cast(tf.range(self.img_size)[:, tf.newaxis], tf.float32)
        x_grid = tf.cast(tf.range(self.img_size)[tf.newaxis, :], tf.float32)

        # Pixel coordinates
        scale = tf.cast(self.img_size, tf.float32)
        bboxes_px = bboxes * scale

        # While loop for graph compatibility (Python for-loop fails on Tensors)
        num_boxes = tf.shape(bboxes)[0]

        def cond(i, _):
            return i >= 0

        def body(i, current_canvas):
            # Extract box
            box = bboxes_px[i]
            by, bx, bh, bw = box[0], box[1], box[2], box[3]
            cls_id = tf.cast(labels[i], tf.int32) + 1  # 0 is background

            # Create box mask
            in_box = (y_grid >= by) & (y_grid < bh) & (x_grid >= bx) & (x_grid < bw)

            # Update canvas (later boxes overwrite)
            new_canvas = tf.where(in_box, cls_id, current_canvas)
            return i - 1, new_canvas

        # Iterate backwards to match typical painter's algorithm if needed,
        # or just ensure consistent ordering.
        _, final_canvas = tf.while_loop(
            cond,
            body,
            loop_vars=[num_boxes - 1, canvas],
            maximum_iterations=self.max_boxes_per_image
        )

        # Format output
        if self.segmentation_classes == 1:
            output = tf.cast(final_canvas > 0, tf.float32)[..., tf.newaxis]
        else:
            # One-hot encoding (depth = classes + background)
            one_hot = tf.one_hot(final_canvas, self.segmentation_classes + 1)
            output = one_hot[..., 1:]  # discard background channel

        return output

    def _apply_augmentations(
        self,
        image: tf.Tensor,
        targets: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Apply random photometric and geometric augmentations."""
        if not self.augment_data:
            return image, targets

        cfg = self.augmentation_config

        # Photometric
        if cfg.brightness_delta > 0:
            image = tf.image.random_brightness(image, cfg.brightness_delta)
        if cfg.contrast_delta > 0:
            image = tf.image.random_contrast(image, 1.0 - cfg.contrast_delta, 1.0 + cfg.contrast_delta)
        if cfg.saturation_delta > 0:
            image = tf.image.random_saturation(image, 1.0 - cfg.saturation_delta, 1.0 + cfg.saturation_delta)
        if cfg.hue_delta > 0:
            image = tf.image.random_hue(image, cfg.hue_delta)

        image = tf.clip_by_value(image, 0.0, 1.0)

        # Geometric: Horizontal Flip
        do_h_flip = tf.random.uniform([]) < cfg.horizontal_flip_prob

        def h_flip_fn():
            new_img = tf.image.flip_left_right(image)
            new_targets = targets.copy()

            if self.use_detection:
                dets = new_targets['detection']
                cls_ids = dets[:, 0:1]
                x1, y1, x2, y2 = tf.unstack(dets[:, 1:], axis=-1)

                # Flip x coords: new_x = 1 - old_x
                nx1 = 1.0 - x2
                nx2 = 1.0 - x1

                # Reassemble, masking invalid boxes
                valid_mask = tf.cast(tf.not_equal(dets[:, 0], INVALID_BBOX_VALUE), tf.float32)
                flipped_boxes = tf.stack([nx1, y1, nx2, y2], axis=1)

                final_boxes = tf.where(valid_mask[:, tf.newaxis] > 0, flipped_boxes, dets[:, 1:])
                new_targets['detection'] = tf.concat([cls_ids, final_boxes], axis=1)

            if self.use_segmentation:
                new_targets['segmentation'] = tf.image.flip_left_right(new_targets['segmentation'])

            return new_img, new_targets

        image, targets = tf.cond(do_h_flip, h_flip_fn, lambda: (image, targets))

        # Geometric: Vertical Flip
        do_v_flip = tf.random.uniform([]) < cfg.vertical_flip_prob

        def v_flip_fn():
            new_img = tf.image.flip_up_down(image)
            new_targets = targets.copy()

            if self.use_detection:
                dets = new_targets['detection']
                cls_ids = dets[:, 0:1]
                x1, y1, x2, y2 = tf.unstack(dets[:, 1:], axis=-1)

                # Flip y coords
                ny1 = 1.0 - y2
                ny2 = 1.0 - y1

                valid_mask = tf.cast(tf.not_equal(dets[:, 0], INVALID_BBOX_VALUE), tf.float32)
                flipped_boxes = tf.stack([x1, ny1, x2, ny2], axis=1)

                final_boxes = tf.where(valid_mask[:, tf.newaxis] > 0, flipped_boxes, dets[:, 1:])
                new_targets['detection'] = tf.concat([cls_ids, final_boxes], axis=1)

            if self.use_segmentation:
                new_targets['segmentation'] = tf.image.flip_up_down(new_targets['segmentation'])

            return new_img, new_targets

        image, targets = tf.cond(do_v_flip, v_flip_fn, lambda: (image, targets))

        return image, targets

    def _process_example(
        self,
        example: Dict[str, Any],
        is_training: bool
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Map function to process a single example."""
        # Load and resize image
        image = tf.cast(example['image'], tf.float32) / 255.0
        image = tf.image.resize(image, [self.img_size, self.img_size])

        # Generate targets
        targets = {}
        if self.use_detection:
            targets['detection'] = self._preprocess_detection_targets(example['objects'])

        if self.use_segmentation:
            targets['segmentation'] = self._preprocess_segmentation_targets(example['objects'], example)

        # Apply augmentations
        if is_training:
            image, targets = self._apply_augmentations(image, targets)

        return image, targets

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create the final optimized training and validation datasets.

        :return: Tuple of (train_ds, val_ds).
        """
        logger.info("Creating datasets...")
        train_raw, val_raw = self._load_tfds_dataset()

        # Filter invalid samples first
        train_raw = train_raw.filter(self._filter_valid_examples)
        val_raw = val_raw.filter(self._filter_valid_examples)

        def process_pipeline(ds: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
            # 1. Limit samples (optional)
            if is_training and self.limit_train_samples:
                ds = ds.take(self.limit_train_samples)

            # 2. Shuffle (before map)
            if is_training:
                ds = ds.shuffle(self.shuffle_buffer_size)
                ds = ds.repeat()

            # 3. Map (preprocessing + augmentation)
            ds = ds.map(
                lambda x: self._process_example(x, is_training),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # 4. Batch with Padding
            pad_shapes = ([self.img_size, self.img_size, 3], {})
            pad_values = (0.0, {})

            if self.use_detection:
                pad_shapes[1]['detection'] = [self.max_boxes_per_image, 5]
                pad_values[1]['detection'] = INVALID_BBOX_VALUE

            if self.use_segmentation:
                c = self.segmentation_classes if self.segmentation_classes > 1 else 1
                pad_shapes[1]['segmentation'] = [self.img_size, self.img_size, c]
                pad_values[1]['segmentation'] = 0.0

            ds = ds.padded_batch(
                self.batch_size,
                padded_shapes=pad_shapes,
                padding_values=pad_values,
                drop_remainder=is_training
            )

            # 5. Prefetch
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = process_pipeline(train_raw, is_training=True)
        val_ds = process_pipeline(val_raw, is_training=False)

        return train_ds, val_ds

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset configuration information.

        :return: Dictionary of configuration parameters.
        """
        return {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'max_boxes': self.max_boxes_per_image,
            'classes': self.dataset_config.num_classes,
            'segmentation_classes': self.segmentation_classes,
            'using_dummy_data': self.using_dummy_data
        }


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_coco_dataset(
    img_size: int = 640,
    batch_size: int = 32,
    use_detection: bool = True,
    use_segmentation: bool = False,
    segmentation_classes: int = 80,
    augment_data: bool = True,
    class_names: Optional[List[str]] = None,
    **kwargs: Any
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Factory function to create a ready-to-use COCO dataset pipeline.

    :param img_size: Target image size (square).
    :param batch_size: Training batch size.
    :param use_detection: Enable object detection targets.
    :param use_segmentation: Enable segmentation mask targets.
    :param segmentation_classes: Number of segmentation classes.
    :param augment_data: Enable/disable data augmentation.
    :param class_names: Optional list of class names to filter/use.
    :param kwargs: Additional arguments passed to COCODatasetBuilder.
    :return: Tuple of (train_dataset, validation_dataset).
    """
    builder = COCODatasetBuilder(
        img_size=img_size,
        batch_size=batch_size,
        use_detection=use_detection,
        use_segmentation=use_segmentation,
        segmentation_classes=segmentation_classes,
        augment_data=augment_data,
        class_names=class_names,
        **kwargs
    )
    return builder.create_datasets()

# ---------------------------------------------------------------------
