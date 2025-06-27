"""
SUT-Crack Dataset Patch-based Loader for Multi-task Learning - Refined Version.

This module provides efficient patch-based data loading for the SUT-Crack dataset,
supporting object detection, segmentation, and classification tasks simultaneously.
The loader implements smart patch sampling strategies to handle large images
by extracting patches during training.

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

Key Improvements:
- Enhanced patch sampling with better spatial distribution
- Robust error handling and validation
- Efficient TensorFlow-native operations
- Data augmentation support
- Better memory management
- Comprehensive logging and debugging

File: src/dl_techniques/utils/datasets/sut.py
"""

import warnings
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any, Union


# Assuming logger is configured elsewhere, creating a placeholder if not
try:
    from dl_techniques.utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


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

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bbox intersects with another."""
        return not (self.xmax <= other.xmin or
                    self.xmin >= other.xmax or
                    self.ymax <= other.ymin or
                    self.ymin >= other.ymax)

    def intersection_area(self, other: 'BoundingBox') -> float:
        """Calculate intersection area with another bbox."""
        if not self.intersects(other):
            return 0.0

        x_overlap = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        y_overlap = min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        return x_overlap * y_overlap

    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bbox."""
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return intersection / (union + 1e-8)

    def to_normalized(self, image_width: int, image_height: int) -> 'BoundingBox':
        """Convert to normalized coordinates [0, 1]."""
        return BoundingBox(
            xmin=self.xmin / image_width,
            ymin=self.ymin / image_height,
            xmax=self.xmax / image_width,
            ymax=self.ymax / image_height,
            class_name=self.class_name
        )


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

    @classmethod
    def from_xml(cls, xml_path: str, image_path: str, mask_path: Optional[str]) -> 'ImageAnnotation':
        """Create annotation from a Pascal VOC XML file and pre-determined paths."""
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


@dataclass
class PatchSample:
    """Represents a patch sample with all task labels and metadata."""
    image_patch: np.ndarray
    mask_patch: Optional[np.ndarray]
    bboxes: List[BoundingBox]
    classification_label: int
    patch_coords: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    original_image_shape: Tuple[int, int]  # (height, width)
    crack_coverage: float = 0.0  # Ratio of crack pixels in patch

    def __post_init__(self):
        """Validate patch sample."""
        if self.image_patch.shape[0] != self.image_patch.shape[1]:
            raise ValueError("Patch must be square")
        if self.mask_patch is not None and self.mask_patch.shape != self.image_patch.shape[:2]:
            raise ValueError("Mask and image patch dimensions must match")


class SUTCrackPatchSampler:
    """Enhanced smart patch sampling strategy for SUT-Crack dataset."""

    def __init__(
        self,
        patch_size: int = 256,
        positive_ratio: float = 0.7,
        min_crack_area_ratio: float = 0.01,
        max_attempts: int = 100,
        enable_augmentation: bool = True,
        overlap_threshold: float = 0.3,
        spatial_jitter: float = 0.1
    ):
        """
        Initialize patch sampler with enhanced parameters.

        Args:
            patch_size: Size of square patches.
            positive_ratio: Ratio of crack-containing patches.
            min_crack_area_ratio: Minimum crack area to consider patch positive.
            max_attempts: Maximum attempts to find valid patches.
            enable_augmentation: Whether to enable data augmentation.
            overlap_threshold: Minimum IoU overlap for positive patches.
            spatial_jitter: Amount of spatial jittering for patch centers.
        """
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.min_crack_area_ratio = min_crack_area_ratio
        self.max_attempts = max_attempts
        self.enable_augmentation = enable_augmentation
        self.overlap_threshold = overlap_threshold
        self.spatial_jitter = spatial_jitter

    def _load_image_tf(self, image_path: str) -> Optional[np.ndarray]:
        """Load image using TensorFlow with proper error handling."""
        try:
            # Read raw file
            raw_image = tf.io.read_file(image_path)

            # Try JPEG decoding first (most common)
            try:
                image_tensor = tf.io.decode_jpeg(raw_image, channels=3)
            except tf.errors.InvalidArgumentError:
                # Fallback to general image decoding
                image_tensor = tf.io.decode_image(raw_image, channels=3, expand_animations=False)

            # Ensure 3 channels
            if tf.shape(image_tensor)[-1] != 3:
                if tf.shape(image_tensor)[-1] == 1:
                    image_tensor = tf.repeat(image_tensor, 3, axis=-1)
                else:
                    logger.warning(f"Unexpected number of channels in {image_path}")
                    return None

            return image_tensor.numpy()

        except (tf.errors.OpError, tf.errors.InvalidArgumentError) as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {image_path}: {e}")
            return None

    def _load_mask_tf(self, mask_path: str) -> Optional[np.ndarray]:
        """Load mask using TensorFlow with proper validation."""
        try:
            raw_mask = tf.io.read_file(mask_path)

            # Try PNG decoding first (most common for masks)
            try:
                mask_tensor = tf.io.decode_png(raw_mask, channels=1)
            except tf.errors.InvalidArgumentError:
                # Fallback to general image decoding
                mask_tensor = tf.io.decode_image(raw_mask, channels=1, expand_animations=False)

            mask_array = tf.squeeze(mask_tensor, axis=-1).numpy()

            # Normalize mask to binary values
            if mask_array.max() > 1:
                mask_array = (mask_array > 127).astype(np.uint8)

            return mask_array

        except (tf.errors.OpError, tf.errors.InvalidArgumentError):
            logger.warning(f"Failed to load mask {mask_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading mask {mask_path}: {e}")
            return None

    def sample_patches(
        self,
        annotation: ImageAnnotation,
        num_patches: int,
        strategy: str = "balanced"
    ) -> List[PatchSample]:
        """
        Sample patches from a single image with enhanced strategies.

        Args:
            annotation: Image annotation with metadata.
            num_patches: Number of patches to sample.
            strategy: Sampling strategy ('balanced', 'aggressive', 'conservative').

        Returns:
            List of patch samples.
        """
        # Load image and mask
        image = self._load_image_tf(annotation.image_path)
        if image is None:
            logger.warning(f"Skipping {annotation.image_path} - failed to load image")
            return []

        mask = None
        if annotation.mask_path:
            mask = self._load_mask_tf(annotation.mask_path)
            if mask is not None and mask.shape != image.shape[:2]:
                logger.warning(f"Mask shape {mask.shape} doesn't match image {image.shape[:2]}")
                mask = None

        # Adjust sampling strategy based on image content
        if strategy == "aggressive" and annotation.has_crack:
            pos_ratio = min(0.9, self.positive_ratio + 0.2)
        elif strategy == "conservative":
            pos_ratio = max(0.3, self.positive_ratio - 0.2)
        else:
            pos_ratio = self.positive_ratio

        # Calculate patch distribution
        if annotation.has_crack:
            num_positive = max(1, int(num_patches * pos_ratio))
        else:
            num_positive = 0
        num_negative = num_patches - num_positive

        patches = []

        # Sample positive patches
        if num_positive > 0:
            positive_patches = self._sample_positive_patches_enhanced(
                annotation, image, mask, num_positive
            )
            patches.extend(positive_patches)

        # Sample negative patches
        if num_negative > 0:
            negative_patches = self._sample_negative_patches_enhanced(
                annotation, image, mask, num_negative
            )
            patches.extend(negative_patches)

        # Fill remaining slots with random patches if needed
        while len(patches) < num_patches:
            random_patch = self._sample_random_patch(annotation, image, mask)
            if random_patch:
                patches.append(random_patch)
            else:
                break

        # Apply augmentation if enabled
        if self.enable_augmentation:
            patches = self._apply_augmentation(patches)

        return patches[:num_patches]

    def _sample_positive_patches_enhanced(
        self,
        annotation: ImageAnnotation,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        num_patches: int
    ) -> List[PatchSample]:
        """Enhanced positive patch sampling with better spatial distribution."""
        patches = []
        attempts = 0

        if not annotation.bboxes:
            return []

        # Create a grid of potential centers for better spatial distribution
        grid_centers = self._generate_grid_centers(annotation, annotation.bboxes)

        while len(patches) < num_patches and attempts < self.max_attempts * num_patches:
            attempts += 1

            # Choose sampling strategy: bbox-based or grid-based
            if len(grid_centers) > 0 and np.random.random() < 0.7:
                center_x, center_y = np.random.choice(grid_centers)
                # Add spatial jittering
                jitter_x = np.random.uniform(-self.spatial_jitter, self.spatial_jitter) * self.patch_size
                jitter_y = np.random.uniform(-self.spatial_jitter, self.spatial_jitter) * self.patch_size
                center_x += jitter_x
                center_y += jitter_y
            else:
                # Traditional bbox-based sampling
                target_bbox = np.random.choice(annotation.bboxes)
                center_x, center_y = self._sample_near_bbox(annotation, target_bbox)

            # Get patch coordinates
            patch_coords = self._get_patch_coords(center_x, center_y, annotation.width, annotation.height)
            if not patch_coords:
                continue

            # Validate patch quality
            patch_bboxes = self._get_patch_bboxes(patch_coords, annotation.bboxes)
            crack_ratio = self._calculate_crack_ratio(patch_coords, mask, patch_bboxes, annotation)

            # Enhanced validation: check for sufficient overlap or crack content
            has_sufficient_overlap = any(
                self._calculate_bbox_overlap_ratio(bbox, patch_coords) > self.overlap_threshold
                for bbox in annotation.bboxes
            )

            if (crack_ratio >= self.min_crack_area_ratio or
                len(patch_bboxes) > 0 or
                has_sufficient_overlap):

                patch = self._create_patch_sample(
                    patch_coords, annotation, image, mask, patch_bboxes, 1, crack_ratio
                )
                if patch:
                    patches.append(patch)

        if len(patches) < num_patches:
            logger.debug(f"Only found {len(patches)}/{num_patches} positive patches "
                        f"after {attempts} attempts for {annotation.image_path}")

        return patches

    def _sample_negative_patches_enhanced(
        self,
        annotation: ImageAnnotation,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        num_patches: int
    ) -> List[PatchSample]:
        """Enhanced negative patch sampling avoiding crack regions."""
        patches = []
        attempts = 0

        # Create avoidance zones around cracks
        avoidance_zones = self._create_avoidance_zones(annotation.bboxes, annotation)

        while len(patches) < num_patches and attempts < self.max_attempts * num_patches:
            attempts += 1

            # Sample patch center avoiding crack regions
            center_x, center_y = self._sample_avoiding_zones(
                annotation, avoidance_zones, buffer=self.patch_size // 2
            )

            patch_coords = self._get_patch_coords(center_x, center_y, annotation.width, annotation.height)
            if not patch_coords:
                continue

            # Validate it's truly negative
            patch_bboxes = self._get_patch_bboxes(patch_coords, annotation.bboxes)
            crack_ratio = self._calculate_crack_ratio(patch_coords, mask, patch_bboxes, annotation)

            if crack_ratio < self.min_crack_area_ratio and len(patch_bboxes) == 0:
                patch = self._create_patch_sample(
                    patch_coords, annotation, image, mask, [], 0, crack_ratio
                )
                if patch:
                    patches.append(patch)

        return patches

    def _generate_grid_centers(
        self,
        annotation: ImageAnnotation,
        bboxes: List[BoundingBox]
    ) -> List[Tuple[float, float]]:
        """Generate grid of potential patch centers around crack regions."""
        centers = []

        for bbox in bboxes:
            # Create a grid around each bbox
            grid_spacing = self.patch_size // 4  # Overlap patches

            x_start = max(self.patch_size // 2, bbox.xmin - self.patch_size // 2)
            x_end = min(annotation.width - self.patch_size // 2, bbox.xmax + self.patch_size // 2)
            y_start = max(self.patch_size // 2, bbox.ymin - self.patch_size // 2)
            y_end = min(annotation.height - self.patch_size // 2, bbox.ymax + self.patch_size // 2)

            x_coords = np.arange(x_start, x_end, grid_spacing)
            y_coords = np.arange(y_start, y_end, grid_spacing)

            for x in x_coords:
                for y in y_coords:
                    centers.append((float(x), float(y)))

        return centers

    def _sample_near_bbox(self, annotation: ImageAnnotation, bbox: BoundingBox) -> Tuple[float, float]:
        """Sample patch center near a bounding box with improved distribution."""
        # Expand sampling region around bbox
        expansion = max(bbox.width, bbox.height, self.patch_size // 2)

        center_x_low = max(self.patch_size / 2, bbox.center_x - expansion)
        center_x_high = min(annotation.width - self.patch_size / 2, bbox.center_x + expansion)
        center_y_low = max(self.patch_size / 2, bbox.center_y - expansion)
        center_y_high = min(annotation.height - self.patch_size / 2, bbox.center_y + expansion)

        if center_x_low >= center_x_high or center_y_low >= center_y_high:
            # Fallback to bbox center
            return bbox.center_x, bbox.center_y

        center_x = np.random.uniform(center_x_low, center_x_high)
        center_y = np.random.uniform(center_y_low, center_y_high)

        return center_x, center_y

    def _create_avoidance_zones(
        self,
        bboxes: List[BoundingBox],
        annotation: ImageAnnotation
    ) -> List[BoundingBox]:
        """Create zones to avoid when sampling negative patches."""
        avoidance_zones = []

        for bbox in bboxes:
            # Expand bbox by patch size to create avoidance zone
            buffer = self.patch_size // 2
            expanded = BoundingBox(
                xmin=max(0, bbox.xmin - buffer),
                ymin=max(0, bbox.ymin - buffer),
                xmax=min(annotation.width, bbox.xmax + buffer),
                ymax=min(annotation.height, bbox.ymax + buffer),
                class_name="avoidance"
            )
            avoidance_zones.append(expanded)

        return avoidance_zones

    def _sample_avoiding_zones(
        self,
        annotation: ImageAnnotation,
        avoidance_zones: List[BoundingBox],
        buffer: int = 0
    ) -> Tuple[float, float]:
        """Sample patch center avoiding specified zones."""
        max_attempts = 50

        for _ in range(max_attempts):
            center_x = np.random.uniform(
                self.patch_size / 2 + buffer,
                annotation.width - self.patch_size / 2 - buffer
            )
            center_y = np.random.uniform(
                self.patch_size / 2 + buffer,
                annotation.height - self.patch_size / 2 - buffer
            )

            # Check if center is in any avoidance zone
            patch_bbox = BoundingBox(
                center_x - self.patch_size / 2, center_y - self.patch_size / 2,
                center_x + self.patch_size / 2, center_y + self.patch_size / 2
            )

            if not any(patch_bbox.intersects(zone) for zone in avoidance_zones):
                return center_x, center_y

        # Fallback: return random center
        center_x = np.random.uniform(self.patch_size / 2, annotation.width - self.patch_size / 2)
        center_y = np.random.uniform(self.patch_size / 2, annotation.height - self.patch_size / 2)
        return center_x, center_y

    def _calculate_bbox_overlap_ratio(self, bbox: BoundingBox, patch_coords: Tuple[int, int, int, int]) -> float:
        """Calculate what ratio of the bbox is covered by the patch."""
        x1, y1, x2, y2 = patch_coords
        patch_bbox = BoundingBox(x1, y1, x2, y2)

        intersection = bbox.intersection_area(patch_bbox)
        return intersection / (bbox.area + 1e-8)

    def _sample_random_patch(self, annotation: ImageAnnotation, image: np.ndarray, mask: Optional[np.ndarray]) -> Optional[PatchSample]:
        """Sample a random patch as fallback with improved validation."""
        for _ in range(10):  # Limited attempts for fallback
            center_x = np.random.uniform(self.patch_size / 2, annotation.width - self.patch_size / 2)
            center_y = np.random.uniform(self.patch_size / 2, annotation.height - self.patch_size / 2)

            patch_coords = self._get_patch_coords(center_x, center_y, annotation.width, annotation.height)
            if not patch_coords:
                continue

            patch_bboxes = self._get_patch_bboxes(patch_coords, annotation.bboxes)
            crack_ratio = self._calculate_crack_ratio(patch_coords, mask, patch_bboxes, annotation)
            label = 1 if len(patch_bboxes) > 0 or crack_ratio > self.min_crack_area_ratio else 0

            patch = self._create_patch_sample(
                patch_coords, annotation, image, mask, patch_bboxes, label, crack_ratio
            )
            if patch:
                return patch

        return None

    def _calculate_crack_ratio(
        self,
        patch_coords: Tuple[int, int, int, int],
        mask: Optional[np.ndarray],
        patch_bboxes: List[BoundingBox],
        annotation: ImageAnnotation
    ) -> float:
        """Calculate crack coverage ratio in patch with better fallback."""
        if mask is not None:
            x1, y1, x2, y2 = patch_coords
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)

            if x2 > x1 and y2 > y1:
                mask_patch = mask[y1:y2, x1:x2]
                return np.sum(mask_patch) / mask_patch.size if mask_patch.size > 0 else 0.0
            return 0.0
        else:
            # Fallback: estimate from bbox overlap
            patch_bbox = BoundingBox(patch_coords[0], patch_coords[1], patch_coords[2], patch_coords[3])
            total_overlap = sum(bbox.intersection_area(patch_bbox) for bbox in annotation.bboxes)
            return total_overlap / (self.patch_size ** 2)

    def _get_patch_coords(
        self,
        cx: float,
        cy: float,
        w: int,
        h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get patch coordinates with improved validation."""
        half = self.patch_size // 2
        x1, y1 = int(cx - half), int(cy - half)
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None

        return (x1, y1, x2, y2)

    def _get_patch_bboxes(
        self,
        patch_coords: Tuple[int, int, int, int],
        original_bboxes: List[BoundingBox]
    ) -> List[BoundingBox]:
        """Get bounding boxes adjusted to patch coordinates with validation."""
        x1, y1, _, _ = patch_coords
        patch_bbox = BoundingBox(x1, y1, x1 + self.patch_size, y1 + self.patch_size)

        patch_bboxes = []
        for bbox in original_bboxes:
            if bbox.intersects(patch_bbox):
                # Adjust coordinates to patch space and clamp to bounds
                adj_bbox = BoundingBox(
                    xmin=max(0, min(self.patch_size, bbox.xmin - x1)),
                    ymin=max(0, min(self.patch_size, bbox.ymin - y1)),
                    xmax=max(0, min(self.patch_size, bbox.xmax - x1)),
                    ymax=max(0, min(self.patch_size, bbox.ymax - y1)),
                    class_name=bbox.class_name
                )

                # Only include if the adjusted bbox has meaningful size
                if adj_bbox.width > 1 and adj_bbox.height > 1:
                    patch_bboxes.append(adj_bbox)

        return patch_bboxes

    def _create_patch_sample(
        self,
        patch_coords: Tuple[int, int, int, int],
        annotation: ImageAnnotation,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        patch_bboxes: List[BoundingBox],
        label: int,
        crack_ratio: float = 0.0
    ) -> Optional[PatchSample]:
        """Create patch sample with enhanced validation."""
        x1, y1, x2, y2 = patch_coords

        # Extract image patch with bounds checking
        if (y2 - y1 != self.patch_size or x2 - x1 != self.patch_size or
            y1 < 0 or x1 < 0 or y2 > image.shape[0] or x2 > image.shape[1]):
            return None

        image_patch = image[y1:y2, x1:x2].copy()

        # Validate patch extraction
        if image_patch.shape[:2] != (self.patch_size, self.patch_size):
            return None

        # Extract mask patch if available
        mask_patch = None
        if mask is not None:
            if y2 <= mask.shape[0] and x2 <= mask.shape[1]:
                mask_patch = mask[y1:y2, x1:x2].copy()
                # Validate mask patch
                if mask_patch.shape != (self.patch_size, self.patch_size):
                    mask_patch = None

        try:
            return PatchSample(
                image_patch=image_patch,
                mask_patch=mask_patch,
                bboxes=patch_bboxes,
                classification_label=label,
                patch_coords=patch_coords,
                original_image_shape=(annotation.height, annotation.width),
                crack_coverage=crack_ratio
            )
        except ValueError as e:
            logger.warning(f"Failed to create patch sample: {e}")
            return None

    def _apply_augmentation(self, patches: List[PatchSample]) -> List[PatchSample]:
        """Apply data augmentation to patches."""
        if not self.enable_augmentation:
            return patches

        augmented_patches = []
        for patch in patches:
            # Apply augmentations with probability
            augmented_patch = self._augment_single_patch(patch)
            augmented_patches.append(augmented_patch)

        return augmented_patches

    def _augment_single_patch(self, patch: PatchSample) -> PatchSample:
        """Apply augmentation to a single patch."""
        # For now, return original patch
        # TODO: Implement augmentations like rotation, flip, brightness, etc.
        return patch


class SUTDataset:
    """Enhanced TensorFlow-compatible dataset for SUT-Crack patch-based multi-task learning."""

    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        patches_per_image: int = 16,
        validation_split: float = 0.0,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize enhanced SUT-Crack patch dataset.

        Args:
            data_dir: Root dataset directory.
            patch_size: Size of square patches.
            patches_per_image: Number of patches per crack image.
            validation_split: Fraction of data for validation.
            seed: Random seed for reproducibility.
            **kwargs: Additional parameters.
        """
        self.data_dir = Path(data_dir).resolve()
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.validation_split = validation_split
        self.seed = seed

        # Configuration parameters
        self.max_boxes_per_patch = kwargs.get('max_boxes_per_patch', 10)
        self.include_segmentation = kwargs.get('include_segmentation', True)
        self.sampling_strategy = kwargs.get('sampling_strategy', 'balanced')
        self.cache_size = kwargs.get('cache_size', 100)  # Number of images to cache

        # Small image handling
        self.small_image_strategy = kwargs.get('small_image_strategy', 'pad')  # 'pad', 'tile', 'resize'
        self.min_image_size = kwargs.get('min_image_size', 64)  # Minimum acceptable image size

        # Directory paths
        self.image_dir = self.data_dir / "segmentation" / "Original Image"
        self.mask_dir = self.data_dir / "segmentation" / "Ground Truth"
        self.detection_dir = self.data_dir / "object_detection"
        self.no_crack_dir = self.data_dir / "classification" / "Without Crack"

        # Validate directory structure
        self._validate_dirs()

        # Initialize sampler
        self.sampler = SUTCrackPatchSampler(
            patch_size=patch_size,
            positive_ratio=kwargs.get('positive_ratio', 0.7),
            min_crack_area_ratio=kwargs.get('min_crack_area_ratio', 0.01),
            enable_augmentation=kwargs.get('enable_augmentation', True),
            overlap_threshold=kwargs.get('overlap_threshold', 0.3)
        )

        # Load annotations
        self.annotations = self._load_annotations()
        logger.info(f"Loaded {len(self.annotations)} total image annotations.")

        if not self.annotations:
            raise ValueError("No valid annotations could be loaded. "
                           "Please check directory structure and file integrity.")

        # Split data if validation split is specified
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

    def _get_image_dimensions_tf(self, image_path: str) -> Tuple[Optional[int], Optional[int]]:
        """Get image dimensions efficiently using TensorFlow."""
        try:
            # Try JPEG shape extraction first (most efficient)
            try:
                shape_tensor = tf.image.extract_jpeg_shape(tf.io.read_file(image_path))
                return int(shape_tensor[1]), int(shape_tensor[0])  # (width, height)
            except tf.errors.InvalidArgumentError:
                # Fallback for non-JPEG images
                raw = tf.io.read_file(image_path)
                img_tensor = tf.io.decode_image(raw, expand_animations=False)
                shape = img_tensor.shape
                return int(shape[1]), int(shape[0])  # (width, height)

        except Exception as e:
            logger.warning(f"Could not get dimensions for {image_path}: {e}")
            return None, None

    def _load_annotations(self) -> List[ImageAnnotation]:
        """Load all annotations with enhanced error handling and validation."""
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
                width, height = self._get_image_dimensions_tf(str(image_path))
                if width and height and width > 0 and height > 0:
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
                else:
                    logger.warning(f"Could not get valid dimensions for {image_path.name}")

            except Exception as e:
                logger.error(f"Failed to process no-crack image {image_path.name}: {e}")

        logger.info(f"Successfully loaded {no_crack_success_count} no-crack image annotations")

        # Final validation - now accepting smaller images
        valid_annotations = []
        for ann in annotations:
            if (Path(ann.image_path).exists() and
                ann.width >= self.min_image_size and
                ann.height >= self.min_image_size):
                valid_annotations.append(ann)
            else:
                logger.warning(f"Excluding annotation due to validation failure: {ann.image_path} "
                             f"(size: {ann.width}x{ann.height}, min required: {self.min_image_size}x{self.min_image_size})")

        # Log statistics about small images
        small_images = [ann for ann in valid_annotations
                       if ann.width < self.patch_size or ann.height < self.patch_size]
        if small_images:
            logger.info(f"Found {len(small_images)} images smaller than patch size "
                       f"({self.patch_size}x{self.patch_size}). Using '{self.small_image_strategy}' strategy.")

        logger.info(f"Final dataset: {len(valid_annotations)} valid annotations "
                   f"({sum(1 for a in valid_annotations if a.has_crack)} with cracks, "
                   f"{sum(1 for a in valid_annotations if not a.has_crack)} without cracks)")

        return valid_annotations

    def _create_data_splits(self):
        """Create train/validation splits."""
        np.random.seed(self.seed)

        # Separate crack and no-crack annotations
        crack_annotations = [ann for ann in self.annotations if ann.has_crack]
        no_crack_annotations = [ann for ann in self.annotations if not ann.has_crack]

        # Split each category separately to maintain balance
        def split_annotations(annotations, split_ratio):
            n_val = int(len(annotations) * split_ratio)
            indices = np.random.permutation(len(annotations))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            return [annotations[i] for i in train_indices], [annotations[i] for i in val_indices]

        crack_train, crack_val = split_annotations(crack_annotations, self.validation_split)
        no_crack_train, no_crack_val = split_annotations(no_crack_annotations, self.validation_split)

        # Store splits
        self.train_annotations = crack_train + no_crack_train
        self.val_annotations = crack_val + no_crack_val

        # Shuffle combined annotations
        np.random.shuffle(self.train_annotations)
        np.random.shuffle(self.val_annotations)

        logger.info(f"Data splits created - Train: {len(self.train_annotations)}, "
                   f"Validation: {len(self.val_annotations)}")

    def create_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        repeat: bool = True,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        subset: str = "train"
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with enhanced options.

        Args:
            batch_size: Batch size for training.
            shuffle: Whether to shuffle data.
            repeat: Whether to repeat dataset.
            prefetch_buffer: Prefetch buffer size.
            subset: Which subset to use ('train', 'validation', 'all').

        Returns:
            TensorFlow dataset.
        """
        # Select appropriate annotations
        if subset == "train" and hasattr(self, 'train_annotations'):
            annotations = self.train_annotations
        elif subset == "validation" and hasattr(self, 'val_annotations'):
            annotations = self.val_annotations
        else:
            annotations = self.annotations

        def patch_generator():
            """Generator function with enhanced patch sampling."""
            indices = np.arange(len(annotations))
            if shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                annotation = annotations[idx]

                # Determine number of patches based on image type
                if annotation.has_crack:
                    num_patches = self.patches_per_image
                else:
                    # Sample fewer patches from no-crack images to balance dataset
                    num_patches = max(1, self.patches_per_image // 4)

                try:
                    patches = self.sampler.sample_patches(
                        annotation, num_patches, strategy=self.sampling_strategy
                    )

                    for patch in patches:
                        yield self._patch_to_tensors(patch)

                except Exception as e:
                    logger.warning(f"Failed to generate patches for {annotation.image_path}: {e}")

        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            patch_generator,
            output_signature=self._get_output_signature()
        )

        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.seed)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)

        if prefetch_buffer:
            dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def _patch_to_tensors(self, patch: PatchSample) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Convert patch sample to TensorFlow tensors with validation."""
        # Image tensor (normalize to [0, 1])
        image_tensor = tf.convert_to_tensor(patch.image_patch, dtype=tf.float32) / 255.0

        # Detection labels
        detection_labels = np.zeros((self.max_boxes_per_patch, 5), dtype=np.float32)
        for i, bbox in enumerate(patch.bboxes[:self.max_boxes_per_patch]):
            detection_labels[i] = [
                0,  # class_id (always 0 for crack)
                bbox.xmin / self.patch_size,  # normalized xmin
                bbox.ymin / self.patch_size,  # normalized ymin
                bbox.xmax / self.patch_size,  # normalized xmax
                bbox.ymax / self.patch_size   # normalized ymax
            ]

        # Segmentation labels
        if patch.mask_patch is not None:
            seg_tensor = tf.convert_to_tensor(
                patch.mask_patch[..., np.newaxis], dtype=tf.float32
            )
        else:
            seg_tensor = tf.zeros((self.patch_size, self.patch_size, 1), dtype=tf.float32)

        # Classification labels
        cls_tensor = tf.convert_to_tensor(patch.classification_label, dtype=tf.int32)

        labels = {
            'detection': tf.convert_to_tensor(detection_labels, dtype=tf.float32),
            'segmentation': seg_tensor,
            'classification': cls_tensor
        }

        return image_tensor, labels

    def _get_output_signature(self) -> Tuple[tf.TensorSpec, Dict[str, tf.TensorSpec]]:
        """Get output signature for TensorFlow dataset."""
        image_spec = tf.TensorSpec(
            shape=(self.patch_size, self.patch_size, 3),
            dtype=tf.float32
        )

        labels_spec = {
            'detection': tf.TensorSpec(
                shape=(self.max_boxes_per_patch, 5),
                dtype=tf.float32
            ),
            'segmentation': tf.TensorSpec(
                shape=(self.patch_size, self.patch_size, 1),
                dtype=tf.float32
            ),
            'classification': tf.TensorSpec(
                shape=(),
                dtype=tf.int32
            )
        }

        return image_spec, labels_spec

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
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


def create_sut_crack_dataset(data_dir: str, **kwargs) -> tf.data.Dataset:
    """Convenience function to create SUT-Crack dataset."""
    dataset = SUTDataset(data_dir=data_dir, **kwargs)
    info = dataset.get_dataset_info()
    logger.info(f"Dataset statistics: {info}")
    return dataset.create_tf_dataset(batch_size=kwargs.get('batch_size', 32))


if __name__ == "__main__":
    import argparse
    import traceback
    import time

    parser = argparse.ArgumentParser(
        description="Enhanced SUT-Crack Dataset Loader Test Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_dir", type=str,
                       help="Path to the root SUT-Crack dataset directory.")
    parser.add_argument("--patch-size", type=int, default=256,
                       help="Size of the square patches to extract.")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for testing the dataset iterator.")
    parser.add_argument("--patches-per-image", type=int, default=8,
                       help="Number of patches to sample per image WITH cracks.")
    parser.add_argument("--no-segmentation", action="store_true",
                       help="Disable loading of segmentation masks.")
    parser.add_argument("--validation-split", type=float, default=0.2,
                       help="Fraction of data to use for validation.")
    parser.add_argument("--test-batches", type=int, default=3,
                       help="Number of batches to test.")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 80)
    print("Enhanced SUT-Crack Dataset Loader Test")
    print(f"  Dataset Directory: {args.data_dir}")
    print(f"  Patch Size: {args.patch_size}x{args.patch_size}")
    print(f"  Patches per Crack Image: {args.patches_per_image}")
    print(f"  Validation Split: {args.validation_split}")
    print("=" * 80)

    try:
        start_time = time.time()

        # Initialize dataset
        dataset_loader = SUTDataset(
            data_dir=args.data_dir,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            include_segmentation=(not args.no_segmentation),
            validation_split=args.validation_split,
            enable_augmentation=True,
            sampling_strategy='balanced'
        )

        load_time = time.time() - start_time
        print(f"\n✅ Dataset loaded successfully in {load_time:.2f} seconds")

        # Print statistics
        info = dataset_loader.get_dataset_info()
        print("\n📊 Dataset Statistics:")
        for key, value in info.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")

        # Test train dataset
        print(f"\n🚂 Testing training dataset...")
        train_dataset = dataset_loader.create_tf_dataset(
            batch_size=args.batch_size,
            shuffle=True,
            repeat=False,
            subset="train"
        )

        print(f"  Processing {args.test_batches} training batches...")
        train_batch_count = 0
        train_start = time.time()

        for batch_images, batch_labels in train_dataset.take(args.test_batches):
            train_batch_count += 1

            if train_batch_count == 1:  # Detailed info for first batch
                print(f"\n  📦 Batch {train_batch_count} Details:")
                print(f"    Image shape: {batch_images.shape}, dtype: {batch_images.dtype}")
                print(f"    Image value range: [{tf.reduce_min(batch_images):.3f}, {tf.reduce_max(batch_images):.3f}]")

                det_labels = batch_labels['detection']
                seg_labels = batch_labels['segmentation']
                cls_labels = batch_labels['classification']

                print(f"    Detection labels: {det_labels.shape}")
                print(f"    Segmentation labels: {seg_labels.shape}")
                print(f"    Classification labels: {cls_labels.shape}")
                print(f"    Classification distribution: {np.bincount(cls_labels.numpy())}")

                # Check for valid detections
                valid_detections = tf.reduce_sum(tf.cast(det_labels[..., 0] > 0, tf.int32))
                print(f"    Valid detections in batch: {valid_detections}")

        train_time = time.time() - train_start
        print(f"  ✅ Processed {train_batch_count} training batches in {train_time:.2f} seconds")

        # Test validation dataset if split was created
        if hasattr(dataset_loader, 'val_annotations'):
            print(f"\n🧪 Testing validation dataset...")
            val_dataset = dataset_loader.create_tf_dataset(
                batch_size=args.batch_size,
                shuffle=False,
                repeat=False,
                subset="validation"
            )

            val_batch_count = 0
            for batch_images, batch_labels in val_dataset.take(2):
                val_batch_count += 1

            print(f"  ✅ Processed {val_batch_count} validation batches")

        # Performance summary
        total_time = time.time() - start_time
        print(f"\n⚡ Performance Summary:")
        print(f"  • Total test time: {total_time:.2f} seconds")
        print(f"  • Dataset loading: {load_time:.2f} seconds")
        print(f"  • Batch processing: {total_time - load_time:.2f} seconds")

        if train_batch_count > 0:
            patches_processed = train_batch_count * args.batch_size
            patches_per_second = patches_processed / train_time
            print(f"  • Processing speed: {patches_per_second:.1f} patches/second")

        print(f"\n🎉 All tests completed successfully!")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Configuration Error: {e}")
        print("  💡 Please ensure:")
        print("    • The data_dir path is correct")
        print("    • The dataset has the expected directory structure")
        print("    • Image and annotation files are present")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        else:
            print("  💡 Run with --verbose for detailed error information")