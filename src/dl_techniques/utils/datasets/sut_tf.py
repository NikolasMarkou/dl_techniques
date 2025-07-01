"""
TensorFlow-Native SUT-Crack Dataset Patch-based Loader - Optimized Version.

This module provides a highly optimized, TensorFlow-native implementation of the
SUT-Crack dataset loader with dramatic performance improvements while maintaining
the same functionality and API as the original loader.

Expected Directory Structure:
/path/to/root/
├── segmentation/
│   ├── Ground Truth/      (Contains binary .png mask images)
│   └── Original Image/    (Contains full-resolution .jpg source images)
├── object_detection/
│   └── (Contains .xml annotations for images with cracks)
└── classification/
    ├── With Crack/        (Contains .jpg images with cracks)
    └── Without Crack/     (Contains .jpg images without cracks)

Key Optimizations:
- Full TensorFlow operations (no numpy/python loops)
- Vectorized patch sampling and bounding box operations
- Efficient tf.data pipeline with caching and prefetching
- Parallel image loading and processing
- TensorFlow-native augmentation
- Memory-efficient batch processing
"""

import os
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Bounding box representation with enhanced functionality."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_name: str = "Crack"

    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError(f"Invalid bounding box: {self}")

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_tensor(self) -> tf.Tensor:
        """Convert to TensorFlow tensor [xmin, ymin, xmax, ymax]."""
        return tf.constant([self.xmin, self.ymin, self.xmax, self.ymax], dtype=tf.float32)

    def to_normalized_tensor(self, image_width: int, image_height: int) -> tf.Tensor:
        """Convert to normalized tensor coordinates."""
        return tf.constant([
            self.xmin / image_width,
            self.ymin / image_height,
            self.xmax / image_width,
            self.ymax / image_height
        ], dtype=tf.float32)


@dataclass
class ImageAnnotation:
    """Complete annotation for a single image with validation."""
    image_path: str
    mask_path: Optional[str]
    width: int
    height: int
    bboxes: List[BoundingBox]
    has_crack: bool

    def __post_init__(self):
        """Validate annotation data."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid image dimensions: {self.width}x{self.height}")

        # Validate bounding boxes are within image bounds
        for bbox in self.bboxes:
            if (bbox.xmin < 0 or bbox.ymin < 0 or
                bbox.xmax > self.width or bbox.ymax > self.height):
                logger.warning(f"Bounding box {bbox} extends beyond image bounds "
                              f"({self.width}x{self.height}) in {self.image_path}")

    def to_tensor_dict(self) -> Dict[str, tf.Tensor]:
        """Convert annotation to TensorFlow tensors."""
        # Convert bboxes to tensor
        if self.bboxes:
            bbox_tensor = tf.stack([bbox.to_tensor() for bbox in self.bboxes])
        else:
            bbox_tensor = tf.zeros((0, 4), dtype=tf.float32)

        return {
            'image_path': tf.constant(self.image_path, dtype=tf.string),
            'mask_path': tf.constant(self.mask_path or '', dtype=tf.string),
            'width': tf.constant(self.width, dtype=tf.int32),
            'height': tf.constant(self.height, dtype=tf.int32),
            'bboxes': bbox_tensor,
            'has_crack': tf.constant(self.has_crack, dtype=tf.bool),
            'num_bboxes': tf.constant(len(self.bboxes), dtype=tf.int32)
        }

    @classmethod
    def from_xml(cls, xml_path: str, image_path: str, mask_path: Optional[str]) -> 'ImageAnnotation':
        """Create annotation from a Pascal VOC XML file."""
        # Keep the same XML parsing logic as original
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            if size is None:
                raise ValueError(f"No size element found in XML: {xml_path}")

            width_elem = size.find('width')
            height_elem = size.find('height')
            if width_elem is None or height_elem is None:
                raise ValueError(f"Missing width/height in XML: {xml_path}")

            width = int(width_elem.text)
            height = int(height_elem.text)

            bboxes = []
            for obj in root.findall('object'):
                class_name_elem = obj.find('name')
                bbox_elem = obj.find('bndbox')

                if class_name_elem is None or bbox_elem is None:
                    logger.warning(f"Incomplete object annotation in {xml_path}")
                    continue

                class_name = class_name_elem.text

                # Extract bbox coordinates with validation
                required_coords = ['xmin', 'ymin', 'xmax', 'ymax']
                coords = {}
                for coord in required_coords:
                    coord_elem = bbox_elem.find(coord)
                    if coord_elem is None:
                        logger.warning(f"Missing {coord} in bbox for {xml_path}")
                        break
                    coords[coord] = float(coord_elem.text)
                else:
                    # All coordinates found
                    try:
                        bbox = BoundingBox(
                            xmin=coords['xmin'],
                            ymin=coords['ymin'],
                            xmax=coords['xmax'],
                            ymax=coords['ymax'],
                            class_name=class_name
                        )
                        bboxes.append(bbox)
                    except ValueError as e:
                        logger.warning(f"Invalid bounding box in {xml_path}: {e}")

            has_crack = len(bboxes) > 0

            return cls(
                image_path=image_path,
                mask_path=mask_path,
                width=width,
                height=height,
                bboxes=bboxes,
                has_crack=has_crack
            )
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML {xml_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing XML {xml_path}: {e}")


class TensorFlowNativePatchSampler:
    """TensorFlow-native patch sampler with vectorized operations."""

    def __init__(
        self,
        patch_size: int = 256,
        positive_ratio: float = 0.7,
        min_crack_area_ratio: float = 0.01,
        overlap_threshold: float = 0.3,
        spatial_jitter: float = 0.1,
        enable_augmentation: bool = True
    ):
        """Initialize TensorFlow-native patch sampler."""
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.min_crack_area_ratio = min_crack_area_ratio
        self.overlap_threshold = overlap_threshold
        self.spatial_jitter = spatial_jitter
        self.enable_augmentation = enable_augmentation

        # Pre-compute constants as tensors
        self.patch_size_tensor = tf.constant(patch_size, dtype=tf.int32)
        self.patch_size_f = tf.constant(patch_size, dtype=tf.float32)
        self.half_patch = tf.constant(patch_size // 2, dtype=tf.float32)

    @tf.function
    def _load_and_decode_image(self, image_path: tf.Tensor) -> tf.Tensor:
        """Load and decode image using TensorFlow operations."""
        raw_image = tf.io.read_file(image_path)

        # Try JPEG first, then fallback to general image decoding
        try:
            image = tf.io.decode_jpeg(raw_image, channels=3)
        except:
            image = tf.io.decode_image(raw_image, channels=3, expand_animations=False)

        # Ensure 3 channels
        image = tf.ensure_shape(image, [None, None, 3])
        return tf.cast(image, tf.float32)

    @tf.function
    def _load_and_decode_mask(self, mask_path: tf.Tensor) -> tf.Tensor:
        """Load and decode mask using TensorFlow operations."""
        # Handle empty mask path
        def load_mask():
            raw_mask = tf.io.read_file(mask_path)
            try:
                mask = tf.io.decode_png(raw_mask, channels=1)
            except:
                mask = tf.io.decode_image(raw_mask, channels=1, expand_animations=False)

            mask = tf.squeeze(mask, axis=-1)
            # Convert to binary (0 or 1)
            mask = tf.cast(mask > 127, tf.float32)
            return mask

        def empty_mask():
            return tf.zeros([1, 1], dtype=tf.float32)

        return tf.cond(
            tf.greater(tf.strings.length(mask_path), 0),
            load_mask,
            empty_mask
        )

    @tf.function
    def _generate_grid_centers(
        self,
        bboxes: tf.Tensor,
        image_width: tf.Tensor,
        image_height: tf.Tensor
    ) -> tf.Tensor:
        """Generate grid of potential patch centers around bounding boxes."""
        # Initialize variables
        grid_spacing = tf.cast(self.patch_size_tensor // 4, tf.float32)
        image_width_f = tf.cast(image_width, tf.float32)
        image_height_f = tf.cast(image_height, tf.float32)

        # Ensure patch centers are always within valid bounds
        min_center_x = self.half_patch
        max_center_x = image_width_f - self.half_patch
        min_center_y = self.half_patch
        max_center_y = image_height_f - self.half_patch

        # Define empty result
        def return_empty():
            return tf.zeros([0, 2], dtype=tf.float32)

        def generate_centers():
            # Vectorized calculation of grid bounds for all bboxes
            x_start = tf.maximum(min_center_x, tf.maximum(min_center_x, bboxes[:, 0] - self.half_patch))
            x_end = tf.minimum(max_center_x, tf.minimum(max_center_x, bboxes[:, 2] + self.half_patch))
            y_start = tf.maximum(min_center_y, tf.maximum(min_center_y, bboxes[:, 1] - self.half_patch))
            y_end = tf.minimum(max_center_y, tf.minimum(max_center_y, bboxes[:, 3] + self.half_patch))

            # Create a fixed grid that we'll filter
            # Determine the overall bounds
            global_x_start = tf.reduce_min(x_start)
            global_x_end = tf.reduce_max(x_end)
            global_y_start = tf.reduce_min(y_start)
            global_y_end = tf.reduce_max(y_end)

            # Ensure bounds are valid
            global_x_start = tf.maximum(global_x_start, min_center_x)
            global_x_end = tf.minimum(global_x_end, max_center_x)
            global_y_start = tf.maximum(global_y_start, min_center_y)
            global_y_end = tf.minimum(global_y_end, max_center_y)

            def create_grid():
                x_coords = tf.range(global_x_start, global_x_end, grid_spacing)
                y_coords = tf.range(global_y_start, global_y_end, grid_spacing)
                x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
                all_grid_points = tf.stack([tf.reshape(x_grid, [-1]), tf.reshape(y_grid, [-1])], axis=1)

                # --- High-Performance Vectorized Filtering ---
                points_expanded = tf.expand_dims(all_grid_points, 1)
                x_start_exp = tf.expand_dims(x_start, 0)
                x_end_exp = tf.expand_dims(x_end, 0)
                y_start_exp = tf.expand_dims(y_start, 0)
                y_end_exp = tf.expand_dims(y_end, 0)

                in_x_range = (points_expanded[:, :, 0] >= x_start_exp) & (points_expanded[:, :, 0] <= x_end_exp)
                in_y_range = (points_expanded[:, :, 1] >= y_start_exp) & (points_expanded[:, :, 1] <= y_end_exp)

                in_any_bbox_region = tf.reduce_any(in_x_range & in_y_range, axis=1)
                valid_centers = tf.boolean_mask(all_grid_points, in_any_bbox_region)
                return valid_centers

            # Check if valid region exists after bounds enforcement
            return tf.cond(
                tf.logical_or(
                    tf.greater_equal(global_x_start, global_x_end),
                    tf.greater_equal(global_y_start, global_y_end)
                ),
                return_empty,
                create_grid
            )

        # Handle all edge cases with tf.cond for proper graph execution
        # Note: We use tf.cond instead of Python if statements because this function
        # is decorated with @tf.function, requiring TensorFlow control flow operations
        return tf.cond(
            tf.equal(tf.shape(bboxes)[0], 0),  # No bboxes case
            return_empty,
            lambda: tf.cond(
                tf.logical_or(  # Handle images too small for patches
                    tf.less_equal(max_center_x, min_center_x),
                    tf.less_equal(max_center_y, min_center_y)
                ),
                return_empty,
                generate_centers
            )
        )

    @tf.function
    def _sample_patch_centers(
        self,
        bboxes: tf.Tensor,
        image_width: tf.Tensor,
        image_height: tf.Tensor,
        num_positive: tf.Tensor,
        num_negative: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample patch centers for positive and negative patches."""
        image_width_f = tf.cast(image_width, tf.float32)
        image_height_f = tf.cast(image_height, tf.float32)

        # Generate positive patch centers
        def sample_positive():
            # Initialize all variables at the beginning
            min_center_x = self.half_patch
            max_center_x = image_width_f - self.half_patch
            min_center_y = self.half_patch
            max_center_y = image_height_f - self.half_patch

            # Initialize default values
            grid_centers = tf.zeros([0, 2], dtype=tf.float32)
            bbox_centers = tf.zeros([0, 2], dtype=tf.float32)
            random_centers = tf.zeros([0, 2], dtype=tf.float32)
            all_positive_candidates = tf.zeros([0, 2], dtype=tf.float32)

            # Check if image is too small - return early if so
            def return_empty_positive():
                return tf.zeros([0, 2], dtype=tf.float32)

            def process_positive_sampling():
                # Generate grid centers around bboxes
                grid_pts = self._generate_grid_centers(bboxes, image_width, image_height)

                # Add some random centers near bboxes
                bbox_ctrs = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0  # [cx, cy]

                # Expand sampling region
                bbox_sizes = tf.reduce_max(bboxes[:, 2:] - bboxes[:, :2], axis=1, keepdims=True)
                expansion = tf.maximum(bbox_sizes, self.half_patch)

                # Sample random centers near bboxes
                num_bbox_samples = tf.minimum(num_positive * 2, tf.shape(bbox_ctrs)[0] * 4)

                # Repeat bbox centers and expansions to match num_bbox_samples
                num_bboxes = tf.shape(bbox_ctrs)[0]
                repeat_factor = tf.cast(tf.math.ceil(tf.cast(num_bbox_samples, tf.float32) / tf.cast(num_bboxes, tf.float32)), tf.int32)

                repeated_centers = tf.tile(bbox_ctrs, [repeat_factor, 1])[:num_bbox_samples]
                repeated_expansions = tf.tile(expansion, [repeat_factor, 1])[:num_bbox_samples]

                # Add random jitter
                jitter = tf.random.uniform([num_bbox_samples, 2], -1.0, 1.0) * repeated_expansions
                rand_ctrs = repeated_centers + jitter

                # Combine grid and random centers
                candidates = tf.concat([grid_pts, rand_ctrs], axis=0)

                # Clip to valid region with proper bounds
                candidates_clipped = tf.clip_by_value(
                    candidates,
                    [min_center_x, min_center_y],
                    [max_center_x, max_center_y]
                )

                # Sample required number
                num_candidates = tf.shape(candidates_clipped)[0]
                return tf.cond(
                    tf.greater(num_candidates, num_positive),
                    lambda: tf.gather(candidates_clipped, tf.random.shuffle(tf.range(num_candidates))[:num_positive]),
                    lambda: candidates_clipped
                )

            # Use tf.cond to handle the branching properly
            return tf.cond(
                tf.logical_or(
                    tf.less_equal(max_center_x, min_center_x),
                    tf.less_equal(max_center_y, min_center_y)
                ),
                return_empty_positive,
                process_positive_sampling
            )

        def no_positive():
            return tf.zeros([0, 2], dtype=tf.float32)

        positive_centers = tf.cond(
            tf.logical_and(tf.greater(num_positive, 0), tf.greater(tf.shape(bboxes)[0], 0)),
            sample_positive,
            no_positive
        )

        # Generate negative patch centers (avoid bbox regions)
        def sample_negative():
            # Initialize all variables at the beginning
            min_center_x = self.half_patch
            max_center_x = image_width_f - self.half_patch
            min_center_y = self.half_patch
            max_center_y = image_height_f - self.half_patch

            # Initialize default values
            avoidance_zones = tf.zeros([0, 4], dtype=tf.float32)
            max_attempts = num_negative * 5
            candidate_centers = tf.zeros([0, 2], dtype=tf.float32)
            valid_mask = tf.zeros([0], dtype=tf.bool)

            # Check if image is too small - return early if so
            def return_empty_negative():
                return tf.zeros([0, 2], dtype=tf.float32)

            def process_negative_sampling():
                # Create avoidance zones around bboxes
                def create_avoidance_zones():
                    buffer = self.half_patch
                    zones = tf.stack([
                        tf.maximum(0.0, bboxes[:, 0] - buffer),
                        tf.maximum(0.0, bboxes[:, 1] - buffer),
                        tf.minimum(image_width_f, bboxes[:, 2] + buffer),
                        tf.minimum(image_height_f, bboxes[:, 3] + buffer)
                    ], axis=1)
                    return zones

                def no_avoidance_zones():
                    return tf.zeros([0, 4], dtype=tf.float32)

                zones = tf.cond(
                    tf.greater(tf.shape(bboxes)[0], 0),
                    create_avoidance_zones,
                    no_avoidance_zones
                )

                # Sample random centers and filter out those in avoidance zones
                attempts = num_negative * 5
                candidates = tf.random.uniform(
                    [attempts, 2],
                    [min_center_x, min_center_y],
                    [max_center_x, max_center_y]
                )

                # Check if candidates are in avoidance zones
                def check_avoidance():
                    # Vectorized intersection check
                    candidates_expanded = tf.expand_dims(candidates, 1)  # [N, 1, 2]
                    zones_expanded = tf.expand_dims(zones, 0)  # [1, M, 4]

                    # Check if candidates are inside any zone
                    in_zone = tf.logical_and(
                        tf.logical_and(
                            candidates_expanded[:, :, 0] >= zones_expanded[:, :, 0],
                            candidates_expanded[:, :, 0] <= zones_expanded[:, :, 2]
                        ),
                        tf.logical_and(
                            candidates_expanded[:, :, 1] >= zones_expanded[:, :, 1],
                            candidates_expanded[:, :, 1] <= zones_expanded[:, :, 3]
                        )
                    )

                    # A candidate is invalid if it's in ANY zone
                    invalid_mask = tf.reduce_any(in_zone, axis=1)
                    valid_mask_result = tf.logical_not(invalid_mask)
                    return valid_mask_result

                def all_valid():
                    return tf.ones([attempts], dtype=tf.bool)

                mask = tf.cond(
                    tf.greater(tf.shape(zones)[0], 0),
                    check_avoidance,
                    all_valid
                )

                # Select valid candidates
                valid_centers = tf.boolean_mask(candidates, mask)
                return valid_centers[:num_negative]

            # Use tf.cond to handle the branching properly
            return tf.cond(
                tf.logical_or(
                    tf.less_equal(max_center_x, min_center_x),
                    tf.less_equal(max_center_y, min_center_y)
                ),
                return_empty_negative,
                process_negative_sampling
            )

        def no_negative():
            return tf.zeros([0, 2], dtype=tf.float32)

        negative_centers = tf.cond(
            tf.greater(num_negative, 0),
            sample_negative,
            no_negative
        )

        return positive_centers, negative_centers

    @tf.function
    def _extract_patches_vectorized(
        self,
        image: tf.Tensor,
        mask: tf.Tensor,
        centers: tf.Tensor,
        bboxes: tf.Tensor,
        max_boxes_per_patch: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Extract patches and their corresponding data vectorized."""
        num_patches = tf.shape(centers)[0]

        # Initialize variables in all branches
        half_patch = tf.cast(self.patch_size // 2, tf.int32)
        centers_int = tf.cast(centers, tf.int32)

        def extract_patches():
            """Extract patches when centers are available."""
            # Use tf.map_fn to process patches in parallel instead of Python loops
            def extract_single_patch(i):
                center = centers_int[i]

                # Get image dimensions first
                image_height = tf.shape(image)[0]
                image_width = tf.shape(image)[1]

                # Ensure center is within valid bounds for patch extraction
                # Valid center range: [half_patch, image_size - half_patch]
                center_x = tf.clip_by_value(center[0], half_patch, image_width - half_patch)
                center_y = tf.clip_by_value(center[1], half_patch, image_height - half_patch)

                # Calculate desired patch bounds
                y1_desired = center_y - half_patch
                y2_desired = center_y + half_patch
                x1_desired = center_x - half_patch
                x2_desired = center_x + half_patch

                # Calculate what we can actually extract from the image (should be exact now)
                y1_actual = tf.maximum(0, y1_desired)
                y2_actual = tf.minimum(image_height, y2_desired)
                x1_actual = tf.maximum(0, x1_desired)
                x2_actual = tf.minimum(image_width, x2_desired)

                # Additional safety check - ensure slice parameters are valid
                y1_actual = tf.clip_by_value(y1_actual, 0, image_height - 1)
                x1_actual = tf.clip_by_value(x1_actual, 0, image_width - 1)
                slice_height = tf.clip_by_value(y2_actual - y1_actual, 1, image_height - y1_actual)
                slice_width = tf.clip_by_value(x2_actual - x1_actual, 1, image_width - x1_actual)

                # Calculate padding needed
                pad_top = tf.maximum(0, -y1_desired)
                pad_bottom = tf.maximum(0, y2_desired - image_height)
                pad_left = tf.maximum(0, -x1_desired)
                pad_right = tf.maximum(0, x2_desired - image_width)

                # Extract the available region from the image with safe parameters
                img_patch_raw = tf.slice(
                    image,
                    [y1_actual, x1_actual, 0],
                    [slice_height, slice_width, 3]
                )

                # Now pad to get exactly the right size
                img_patch = tf.pad(
                    img_patch_raw,
                    [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    mode='REFLECT'
                )

                # This should now be exactly the right shape
                img_patch = tf.ensure_shape(img_patch, [self.patch_size, self.patch_size, 3])

                # Extract mask patch with same logic
                def extract_mask():
                    mask_resized = mask
                    # Resize mask to image size if needed
                    mask_shape = tf.shape(mask)
                    image_shape = tf.shape(image)

                    mask_resized = tf.cond(
                        tf.logical_or(
                            tf.not_equal(mask_shape[0], image_shape[0]),
                            tf.not_equal(mask_shape[1], image_shape[1])
                        ),
                        lambda: tf.squeeze(tf.image.resize(
                            tf.expand_dims(mask, -1),
                            [image_shape[0], image_shape[1]],
                            method='nearest'
                        ), -1),
                        lambda: mask
                    )

                    # Extract mask patch with same safe logic
                    mask_slice_height = tf.clip_by_value(slice_height, 1, tf.shape(mask_resized)[0] - y1_actual)
                    mask_slice_width = tf.clip_by_value(slice_width, 1, tf.shape(mask_resized)[1] - x1_actual)

                    mask_patch_raw = tf.slice(
                        mask_resized,
                        [y1_actual, x1_actual],
                        [mask_slice_height, mask_slice_width]
                    )

                    # Pad mask patch
                    mask_patch_padded = tf.pad(
                        mask_patch_raw,
                        [[pad_top, pad_bottom], [pad_left, pad_right]],
                        mode='CONSTANT',
                        constant_values=0
                    )

                    return mask_patch_padded

                def empty_mask():
                    return tf.zeros([self.patch_size, self.patch_size], dtype=tf.float32)

                mask_patch = tf.cond(
                    tf.greater(tf.reduce_prod(tf.shape(mask)), 1),
                    extract_mask,
                    empty_mask
                )
                mask_patch = tf.ensure_shape(mask_patch, [self.patch_size, self.patch_size])

                # Process bboxes for this patch (use desired coordinates for proper scaling)
                patch_bbox = tf.cast([x1_desired, y1_desired, x2_desired, y2_desired], tf.float32)

                # Find overlapping bboxes using vectorized operations
                def process_bboxes():
                    # Check intersection with all bboxes at once
                    intersects = tf.logical_and(
                        tf.logical_and(bboxes[:, 2] > patch_bbox[0], bboxes[:, 0] < patch_bbox[2]),
                        tf.logical_and(bboxes[:, 3] > patch_bbox[1], bboxes[:, 1] < patch_bbox[3])
                    )

                    # Get intersecting bboxes
                    overlapping_bboxes = tf.boolean_mask(bboxes, intersects)

                    # Adjust coordinates to patch space and normalize
                    def adjust_bboxes():
                        # Adjust bbox coordinates to patch space (accounting for the desired patch bounds)
                        adjusted_x_min = tf.clip_by_value(
                            (overlapping_bboxes[:, 0] - patch_bbox[0]) / self.patch_size_f,
                            0.0, 1.0
                        )
                        adjusted_y_min = tf.clip_by_value(
                            (overlapping_bboxes[:, 1] - patch_bbox[1]) / self.patch_size_f,
                            0.0, 1.0
                        )
                        adjusted_x_max = tf.clip_by_value(
                            (overlapping_bboxes[:, 2] - patch_bbox[0]) / self.patch_size_f,
                            0.0, 1.0
                        )
                        adjusted_y_max = tf.clip_by_value(
                            (overlapping_bboxes[:, 3] - patch_bbox[1]) / self.patch_size_f,
                            0.0, 1.0
                        )

                        adjusted = tf.stack([
                            tf.zeros(tf.shape(overlapping_bboxes)[0]),  # class_id
                            adjusted_x_min,
                            adjusted_y_min,
                            adjusted_x_max,
                            adjusted_y_max
                        ], axis=1)

                        # Pad or truncate to max_boxes (10)
                        max_boxes = max_boxes_per_patch
                        num_boxes = tf.shape(adjusted)[0]

                        adjusted = tf.cond(
                            tf.greater(num_boxes, max_boxes),
                            lambda: adjusted[:max_boxes],
                            lambda: tf.pad(adjusted, [[0, max_boxes - num_boxes], [0, 0]])
                        )

                        return adjusted, 1  # label = 1 (has crack)

                    def no_bboxes():
                        return tf.zeros([max_boxes_per_patch, 5], dtype=tf.float32), 0 # CORRECT

                    return tf.cond(
                        tf.greater(tf.shape(overlapping_bboxes)[0], 0),
                        adjust_bboxes,
                        no_bboxes
                    )

                def no_bboxes_available():
                    return tf.zeros([max_boxes_per_patch, 5], dtype=tf.float32), 0 # CORRECT

                bbox_tensor, label = tf.cond(
                    tf.greater(tf.shape(bboxes)[0], 0),
                    process_bboxes,
                    no_bboxes_available
                )

                return img_patch, mask_patch, label, bbox_tensor

            # Process all patches using tf.map_fn
            patches_data = tf.map_fn(
                extract_single_patch,
                tf.range(num_patches),
                fn_output_signature=(
                    tf.TensorSpec([self.patch_size, self.patch_size, 3], tf.float32),
                    tf.TensorSpec([self.patch_size, self.patch_size], tf.float32),
                    tf.TensorSpec([], tf.int32),
                    tf.TensorSpec([max_boxes_per_patch, 5], tf.float32)
                ),
                parallel_iterations=10
            )

            image_patches, mask_patches, labels, bbox_patches = patches_data
            return image_patches, mask_patches, labels, bbox_patches

        def return_empty():
            """Return empty tensors when no centers."""
            return (
                tf.zeros([0, self.patch_size, self.patch_size, 3], dtype=tf.float32),
                tf.zeros([0, self.patch_size, self.patch_size], dtype=tf.float32),
                tf.zeros([0], dtype=tf.int32),
                tf.zeros([0, max_boxes_per_patch, 5], dtype=tf.float32)  # CORRECT
            )

        # Use tf.cond to handle the branching properly
        return tf.cond(
            tf.greater(num_patches, 0),
            extract_patches,
            return_empty
        )

    @tf.function
    def _apply_augmentation_tf(self, image_patches: tf.Tensor) -> tf.Tensor:
        """Apply TensorFlow-native augmentation to patches."""
        if not self.enable_augmentation:
            return image_patches

        # Random brightness
        image_patches = tf.image.random_brightness(image_patches, 0.1)

        # Random contrast
        image_patches = tf.image.random_contrast(image_patches, 0.8, 1.2)

        # Random saturation
        image_patches = tf.image.random_saturation(image_patches, 0.8, 1.2)

        # Random flip
        image_patches = tf.image.random_flip_left_right(image_patches)
        image_patches = tf.image.random_flip_up_down(image_patches)

        return image_patches


class OptimizedSUTDataset:
    """Highly optimized TensorFlow-native dataset for SUT-Crack patch-based learning."""

    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        patches_per_image: int = 16,
        validation_split: float = 0.0,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize optimized SUT-Crack dataset.

        Args:
            data_dir: Root dataset directory.
            patch_size: Size of square patches.
            patches_per_image: Number of patches per crack image.
            validation_split: Fraction of data for validation.
            seed: Random seed for reproducibility.
            cache_dir: Directory for caching preprocessed data.
            **kwargs: Additional parameters.
        """
        self.data_dir = Path(data_dir).resolve()
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.validation_split = validation_split
        self.seed = seed
        self.cache_dir = cache_dir

        # Configuration parameters
        self.max_boxes_per_patch = kwargs.get('max_boxes_per_patch', 10)
        self.include_segmentation = kwargs.get('include_segmentation', True)
        self.positive_ratio = kwargs.get('positive_ratio', 0.7)
        self.min_crack_area_ratio = kwargs.get('min_crack_area_ratio', 0.01)
        self.enable_augmentation = kwargs.get('enable_augmentation', True)

        # Directory paths
        self.image_dir = self.data_dir / "segmentation" / "Original Image"
        self.mask_dir = self.data_dir / "segmentation" / "Ground Truth"
        self.detection_dir = self.data_dir / "object_detection"
        self.no_crack_dir = self.data_dir / "classification" / "Without Crack"

        # Validate directories
        self._validate_dirs()

        # Initialize TensorFlow-native sampler
        self.sampler = TensorFlowNativePatchSampler(
            patch_size=patch_size,
            positive_ratio=self.positive_ratio,
            min_crack_area_ratio=self.min_crack_area_ratio,
            enable_augmentation=self.enable_augmentation,
            overlap_threshold=kwargs.get('overlap_threshold', 0.3)
        )

        # Load annotations (same as original)
        self.annotations = self._load_annotations()
        logger.info(f"Loaded {len(self.annotations)} total image annotations.")

        if not self.annotations:
            raise ValueError("No valid annotations could be loaded.")

        # Create data splits if needed
        if validation_split > 0:
            self._create_data_splits()

    def _validate_dirs(self):
        """Validate required directories exist."""
        required_dirs = [self.image_dir, self.detection_dir, self.no_crack_dir]
        if self.include_segmentation:
            required_dirs.append(self.mask_dir)

        missing_dirs = [d for d in required_dirs if not d.exists()]
        if missing_dirs:
            raise FileNotFoundError(f"Required directories not found: {missing_dirs}")

        logger.info("Directory structure validated successfully")

    def _load_annotations(self) -> List[ImageAnnotation]:
        """Load all annotations - same as original implementation."""
        annotations = []

        # Load crack images from XML annotations
        logger.info("Loading annotations for images WITH cracks...")
        xml_files = list(self.detection_dir.glob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML annotation files")

        xml_success_count = 0
        for xml_file in xml_files:
            try:
                # Parse XML to get filename
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename_elem = root.find('filename')

                if filename_elem is None:
                    logger.warning(f"No filename found in {xml_file.name}")
                    continue

                filename_in_xml = filename_elem.text
                image_path = self.image_dir / filename_in_xml

                # Check if image exists
                if not image_path.exists():
                    logger.warning(f"Image '{image_path.name}' referenced in "
                                 f"'{xml_file.name}' not found. Skipping.")
                    continue

                # Determine mask path
                mask_path = None
                if self.include_segmentation:
                    # Try different extensions for mask
                    mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
                    base_name = Path(filename_in_xml).stem

                    for ext in mask_extensions:
                        potential_mask = self.mask_dir / f"{base_name}{ext}"
                        if potential_mask.exists():
                            mask_path = str(potential_mask)
                            break

                # Create annotation
                annotation = ImageAnnotation.from_xml(
                    str(xml_file), str(image_path), mask_path
                )

                # Validate annotation
                if annotation.width > 0 and annotation.height > 0:
                    annotations.append(annotation)
                    xml_success_count += 1
                else:
                    logger.warning(f"Invalid dimensions in {xml_file.name}")

            except Exception as e:
                logger.error(f"Failed to process XML {xml_file.name}: {e}")

        logger.info(f"Successfully loaded {xml_success_count} crack image annotations")

        # Load no-crack images
        logger.info("Loading annotations for images WITHOUT cracks...")
        no_crack_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        no_crack_images = []

        for pattern in no_crack_extensions:
            no_crack_images.extend(self.no_crack_dir.glob(pattern))

        logger.info(f"Found {len(no_crack_images)} potential no-crack images")

        no_crack_success_count = 0
        for image_path in no_crack_images:
            try:
                # Use TensorFlow to get image dimensions efficiently
                raw_image = tf.io.read_file(str(image_path))
                try:
                    shape = tf.image.extract_jpeg_shape(raw_image)
                    width, height = int(shape[1]), int(shape[0])
                except:
                    # Fallback for non-JPEG
                    img = tf.io.decode_image(raw_image, expand_animations=False)
                    width, height = int(img.shape[1]), int(img.shape[0])

                if width > 0 and height > 0:
                    annotation = ImageAnnotation(
                        image_path=str(image_path),
                        mask_path=None,
                        width=width,
                        height=height,
                        bboxes=[],
                        has_crack=False
                    )
                    annotations.append(annotation)
                    no_crack_success_count += 1

            except Exception as e:
                logger.error(f"Failed to process no-crack image {image_path.name}: {e}")

        logger.info(f"Successfully loaded {no_crack_success_count} no-crack image annotations")

        # Final validation
        valid_annotations = [ann for ann in annotations if Path(ann.image_path).exists()]

        logger.info(f"Final dataset: {len(valid_annotations)} valid annotations "
                   f"({sum(1 for a in valid_annotations if a.has_crack)} with cracks, "
                   f"{sum(1 for a in valid_annotations if not a.has_crack)} without cracks)")

        return valid_annotations

    def _create_data_splits(self):
        """Create train/validation splits."""
        tf.random.set_seed(self.seed)

        # Separate crack and no-crack annotations
        crack_annotations = [ann for ann in self.annotations if ann.has_crack]
        no_crack_annotations = [ann for ann in self.annotations if not ann.has_crack]

        # Split each category separately
        def split_annotations(annotations, split_ratio):
            n_val = int(len(annotations) * split_ratio)
            indices = tf.random.shuffle(tf.range(len(annotations)))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            return ([annotations[i] for i in train_indices.numpy()],
                    [annotations[i] for i in val_indices.numpy()])

        crack_train, crack_val = split_annotations(crack_annotations, self.validation_split)
        no_crack_train, no_crack_val = split_annotations(no_crack_annotations, self.validation_split)

        # Store splits
        self.train_annotations = crack_train + no_crack_train
        self.val_annotations = crack_val + no_crack_val

        # Shuffle combined annotations
        tf.random.shuffle(self.train_annotations)
        tf.random.shuffle(self.val_annotations)

        logger.info(f"Data splits created - Train: {len(self.train_annotations)}, "
                   f"Validation: {len(self.val_annotations)}")

    def create_tf_dataset(
            self,
            batch_size: int = 32,
            shuffle: bool = True,
            repeat: bool = False,
            prefetch_buffer: int = tf.data.AUTOTUNE,
            subset: str = "train",
            cache: bool = True,
            num_parallel_calls: int = tf.data.AUTOTUNE
    ) -> tf.data.Dataset:
        """
        Create an optimized TensorFlow dataset with high-performance ordering.

        The order of operations is crucial for performance:
        1. Source: Create dataset from annotations.
        2. Map: Apply parallel processing to generate patches.
        3. Cache: Store the processed patches in memory/on disk. This avoids
           re-running the expensive mapping step on every epoch.
        4. Shuffle: Shuffle the in-memory cached data.
        5. Repeat: Repeat the shuffled data for multiple epochs.
        6. Batch: Group items into batches.
        7. Prefetch: Prepare the next batches while the current one is being processed.

        Args:
            batch_size: Batch size for training.
            shuffle: Whether to shuffle data.
            repeat: Whether to repeat dataset. It's recommended to handle this in the
                    training script for clarity (e.g., train_ds.repeat()).
            prefetch_buffer: Prefetch buffer size.
            subset: Which subset to use ('train', 'validation', 'all').
            cache: Whether to cache the dataset.
            num_parallel_calls: Number of parallel calls for processing.

        Returns:
            An optimized and high-performance TensorFlow dataset.
        """
        # Select appropriate annotations
        if subset == "train" and hasattr(self, 'train_annotations'):
            annotations = self.train_annotations
        elif subset == "validation" and hasattr(self, 'val_annotations'):
            annotations = self.val_annotations
        else:
            annotations = self.annotations

        logger.info(f"Creating '{subset}' dataset with {len(annotations)} annotations")

        if not annotations:
            logger.warning(f"No annotations found for subset '{subset}'. Returning an empty dataset.")
            # Return an empty dataset with the correct structure to avoid errors downstream
            empty_images = tf.zeros([0, self.patch_size, self.patch_size, 3], dtype=tf.float32)
            empty_targets = {
                'detection': tf.zeros([0, self.max_boxes_per_patch, 5], dtype=tf.float32),
                'segmentation': tf.zeros([0, self.patch_size, self.patch_size, 1], dtype=tf.float32),
                'classification': tf.zeros([0], dtype=tf.int32)
            }
            return tf.data.Dataset.from_tensor_slices((empty_images, empty_targets)).batch(batch_size)

        # Convert annotations to tensor format
        annotation_tensors = [ann.to_tensor_dict() for ann in annotations]

        # Step 1: Create dataset from the source annotations
        dataset = tf.data.Dataset.from_generator(
            lambda: annotation_tensors,
            output_signature={
                'image_path': tf.TensorSpec([], tf.string),
                'mask_path': tf.TensorSpec([], tf.string),
                'width': tf.TensorSpec([], tf.int32),
                'height': tf.TensorSpec([], tf.int32),
                'bboxes': tf.TensorSpec([None, 4], tf.float32),
                'has_crack': tf.TensorSpec([], tf.bool),
                'num_bboxes': tf.TensorSpec([], tf.int32)
            }
        )

        # Step 2: Map the processing function in parallel
        dataset = dataset.map(
            self._process_single_annotation_tf,
            num_parallel_calls=num_parallel_calls,
            deterministic=False
        )

        # Flatten the dataset (each annotation produces multiple patches)
        dataset = dataset.flat_map(lambda x: x)

        # Step 3: Cache the dataset after the expensive map operations
        if cache:
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"{subset}_cache")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                dataset = dataset.cache(cache_path)
                logger.info(f"Dataset caching enabled for '{subset}' subset at: {cache_path}")
            else:
                dataset = dataset.cache()
                logger.info(f"In-memory dataset caching enabled for '{subset}' subset.")

        # Step 4: Shuffle the now-cached dataset
        if shuffle:
            # Robust buffer size calculation to prevent errors with small/empty datasets
            num_patches = len(annotations) * self.patches_per_image
            if num_patches > 0:
                buffer_size = max(1, min(1000, num_patches))
                logger.info(f"Shuffling dataset with buffer size: {buffer_size}")
                dataset = dataset.shuffle(
                    buffer_size=buffer_size,
                    seed=self.seed,
                    reshuffle_each_iteration=True
                )
            else:
                logger.warning("Skipping shuffle for dataset with 0 annotations.")

        # Step 5: Repeat the dataset (if specified)
        # Note: It's often better to call .repeat() in the training script itself
        # for clarity, especially for the training set.
        if repeat:
            dataset = dataset.repeat()

        # Step 6: Batch the data
        # drop_remainder=True can sometimes improve performance on GPU but is False by default
        dataset = dataset.batch(batch_size, drop_remainder=False)

        # Step 7: Prefetch to overlap data preparation and model execution
        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    # In sut_optimized.py, completely replace the _process_single_annotation_tf method:

    @tf.function
    def _process_single_annotation_tf(self, annotation_dict: Dict[str, tf.Tensor]) -> tf.data.Dataset:
        """Process a single annotation to generate patches using TensorFlow operations."""
        # Load image and mask
        image = self.sampler._load_and_decode_image(annotation_dict['image_path'])
        mask = self.sampler._load_and_decode_mask(annotation_dict['mask_path'])

        # Handle mask orientation mismatch - check for 90-degree rotation
        image_shape = tf.shape(image)
        mask_shape = tf.shape(mask)

        # Check if mask is rotated 90 degrees (width and height swapped)
        is_rotated = tf.logical_and(
            tf.equal(image_shape[0], mask_shape[1]),
            tf.equal(image_shape[1], mask_shape[0])
        )

        mask = tf.cond(
            is_rotated,
            lambda: tf.transpose(mask),
            lambda: mask
        )

        # Determine number of patches
        num_patches = tf.cond(
            annotation_dict['has_crack'],
            lambda: tf.constant(self.patches_per_image, dtype=tf.int32),
            lambda: tf.constant(max(1, self.patches_per_image // 4), dtype=tf.int32)
        )

        # Calculate positive/negative split
        num_positive = tf.cond(
            annotation_dict['has_crack'],
            lambda: tf.cast(tf.cast(num_patches, tf.float32) * self.positive_ratio, tf.int32),
            lambda: tf.constant(0, dtype=tf.int32)
        )
        num_negative = num_patches - num_positive

        # Sample patch centers
        positive_centers, negative_centers = self.sampler._sample_patch_centers(
            annotation_dict['bboxes'],
            annotation_dict['width'],
            annotation_dict['height'],
            num_positive,
            num_negative
        )

        # Combine centers
        all_centers = tf.concat([positive_centers, negative_centers], axis=0)

        # Extract patches
        if tf.greater(tf.shape(all_centers)[0], 0):
            image_patches, mask_patches, labels, bbox_patches = self.sampler._extract_patches_vectorized(
                image, mask, all_centers, annotation_dict['bboxes'], self.max_boxes_per_patch
            )

            # Apply augmentation
            image_patches = self.sampler._apply_augmentation_tf(image_patches)

            # Normalize images
            image_patches = image_patches / 255.0

            # CORRECTED: Create proper format for multi-task training
            # The dataset should return (inputs, targets) where:
            # - inputs: the image patches (single tensor)
            # - targets: dictionary matching the model output structure

            targets = {
                'detection': bbox_patches,
                'segmentation': tf.expand_dims(mask_patches, -1),
                'classification': labels
            }

            # Create dataset with correct structure
            outputs = tf.data.Dataset.from_tensor_slices((image_patches, targets))

        else:
            # Empty dataset if no valid patches
            empty_images = tf.zeros([0, self.patch_size, self.patch_size, 3], dtype=tf.float32)
            empty_targets = {
                'detection': tf.zeros([0, self.max_boxes_per_patch, 5], dtype=tf.float32),
                'segmentation': tf.zeros([0, self.patch_size, self.patch_size, 1], dtype=tf.float32),
                'classification': tf.zeros([0], dtype=tf.int32)
            }
            outputs = tf.data.Dataset.from_tensor_slices((empty_images, empty_targets))

        return outputs

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics - same as original."""
        total_images = len(self.annotations)
        with_cracks = sum(1 for ann in self.annotations if ann.has_crack)
        without_cracks = total_images - with_cracks
        total_bboxes = sum(len(ann.bboxes) for ann in self.annotations)

        # Calculate expected patches per epoch
        patches_with_cracks = with_cracks * self.patches_per_image
        patches_without_cracks = without_cracks * max(1, self.patches_per_image // 4)
        total_patches = patches_with_cracks + patches_without_cracks

        info = {
            'total_images': total_images,
            'images_with_cracks': with_cracks,
            'images_without_cracks': without_cracks,
            'total_bboxes': total_bboxes,
            'avg_bboxes_per_crack_image': total_bboxes / max(1, with_cracks),
            'patch_size': self.patch_size,
            'patches_per_crack_image': self.patches_per_image,
            'patches_per_no_crack_image': max(1, self.patches_per_image // 4),
            'total_patches_per_epoch': total_patches,
            'crack_patch_ratio': patches_with_cracks / max(1, total_patches)
        }

        # Add split information if available
        if hasattr(self, 'train_annotations'):
            train_with_cracks = sum(1 for ann in self.train_annotations if ann.has_crack)
            val_with_cracks = sum(1 for ann in self.val_annotations if ann.has_crack)

            info.update({
                'train_images': len(self.train_annotations),
                'train_with_cracks': train_with_cracks,
                'val_images': len(self.val_annotations),
                'val_with_cracks': val_with_cracks,
            })

        return info


# Backward compatibility aliases
SUTDataset = OptimizedSUTDataset
SUTCrackPatchSampler = TensorFlowNativePatchSampler


def create_sut_crack_dataset(data_dir: str, **kwargs) -> tf.data.Dataset:
    """Convenience function to create optimized SUT-Crack dataset."""
    dataset = OptimizedSUTDataset(data_dir=data_dir, **kwargs)
    info = dataset.get_dataset_info()
    logger.info(f"Dataset statistics: {info}")
    return dataset.create_tf_dataset(batch_size=kwargs.get('batch_size', 32))

# ---------------------------------------------------------------------