"""
Coordinate Transformation Utilities for Patch-based Multi-task Learning.

This module provides utilities for transforming coordinates between patch space
and full image space, handling bounding box transformations, and managing
patch-based inference results aggregation.

Key Features:
    - Patch to full image coordinate transformation
    - Bounding box coordinate adjustments
    - Non-Maximum Suppression for overlapping detections
    - Segmentation patch stitching
    - Classification result aggregation

"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@dataclass
class PatchInfo:
    """Information about a patch location in the full image."""
    x1: int  # Left coordinate in full image
    y1: int  # Top coordinate in full image
    x2: int  # Right coordinate in full image
    y2: int  # Bottom coordinate in full image
    patch_size: int  # Size of the square patch
    overlap_x: int = 0  # Overlap with adjacent patches in x direction
    overlap_y: int = 0  # Overlap with adjacent patches in y direction

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class DetectionResult:
    """Single detection result with confidence."""
    bbox: List[float]  # [x1, y1, x2, y2] in full image coordinates
    confidence: float
    class_id: int = 0  # Always 0 for crack detection

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id
        }


@dataclass
class PatchPrediction:
    """Predictions from a single patch."""
    patch_info: PatchInfo
    detections: List[DetectionResult]
    segmentation_mask: Optional[np.ndarray]  # (patch_size, patch_size)
    classification_score: float  # Probability of crack presence
    classification_confidence: Optional[float] = None


class CoordinateTransformer:
    """Handles coordinate transformations between patch and full image space."""

    @staticmethod
    def patch_to_full_image_bbox(
            patch_bbox: List[float],
            patch_info: PatchInfo
    ) -> List[float]:
        """
        Transform bounding box from patch coordinates to full image coordinates.

        Args:
            patch_bbox: [x1, y1, x2, y2] in patch space (0 to patch_size).
            patch_info: Information about patch location.

        Returns:
            [x1, y1, x2, y2] in full image coordinates.
        """
        x1_patch, y1_patch, x2_patch, y2_patch = patch_bbox

        # Transform to full image coordinates
        x1_full = patch_info.x1 + x1_patch
        y1_full = patch_info.y1 + y1_patch
        x2_full = patch_info.x1 + x2_patch
        y2_full = patch_info.y1 + y2_patch

        return [x1_full, y1_full, x2_full, y2_full]

    @staticmethod
    def full_image_to_patch_bbox(
            full_bbox: List[float],
            patch_info: PatchInfo
    ) -> List[float]:
        """
        Transform bounding box from full image coordinates to patch coordinates.

        Args:
            full_bbox: [x1, y1, x2, y2] in full image space.
            patch_info: Information about patch location.

        Returns:
            [x1, y1, x2, y2] in patch coordinates, clipped to patch bounds.
        """
        x1_full, y1_full, x2_full, y2_full = full_bbox

        # Transform to patch coordinates
        x1_patch = max(0, x1_full - patch_info.x1)
        y1_patch = max(0, y1_full - patch_info.y1)
        x2_patch = min(patch_info.patch_size, x2_full - patch_info.x1)
        y2_patch = min(patch_info.patch_size, y2_full - patch_info.y1)

        return [x1_patch, y1_patch, x2_patch, y2_patch]

    @staticmethod
    def normalize_bbox(
            bbox: List[float],
            image_width: int,
            image_height: int
    ) -> List[float]:
        """Normalize bounding box coordinates to [0, 1] range."""
        x1, y1, x2, y2 = bbox
        return [
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height
        ]

    @staticmethod
    def denormalize_bbox(
            normalized_bbox: List[float],
            image_width: int,
            image_height: int
    ) -> List[float]:
        """Convert normalized bounding box back to pixel coordinates."""
        x1_norm, y1_norm, x2_norm, y2_norm = normalized_bbox
        return [
            x1_norm * image_width,
            y1_norm * image_height,
            x2_norm * image_width,
            y2_norm * image_height
        ]


class PatchGridGenerator:
    """Generates grid of patches for sliding window inference."""

    def __init__(self, patch_size: int, overlap: int = 64):
        """
        Initialize patch grid generator.

        Args:
            patch_size: Size of square patches.
            overlap: Overlap between adjacent patches in pixels.
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

    def generate_patches(
            self,
            image_width: int,
            image_height: int
    ) -> List[PatchInfo]:
        """
        Generate list of patch locations covering the full image.

        Args:
            image_width: Width of the full image.
            image_height: Height of the full image.

        Returns:
            List of PatchInfo objects covering the image.
        """
        patches = []

        # Calculate number of patches needed
        num_patches_x = max(1, (image_width - self.overlap) // self.stride)
        num_patches_y = max(1, (image_height - self.overlap) // self.stride)

        for row in range(num_patches_y):
            for col in range(num_patches_x):
                # Calculate patch coordinates
                x1 = col * self.stride
                y1 = row * self.stride
                x2 = min(x1 + self.patch_size, image_width)
                y2 = min(y1 + self.patch_size, image_height)

                # Adjust if patch extends beyond image
                if x2 - x1 < self.patch_size:
                    x1 = max(0, image_width - self.patch_size)
                    x2 = image_width

                if y2 - y1 < self.patch_size:
                    y1 = max(0, image_height - self.patch_size)
                    y2 = image_height

                # Calculate overlaps
                overlap_x = self.overlap if col > 0 else 0
                overlap_y = self.overlap if row > 0 else 0

                patch_info = PatchInfo(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    patch_size=self.patch_size,
                    overlap_x=overlap_x,
                    overlap_y=overlap_y
                )

                patches.append(patch_info)

        logger.debug(f"Generated {len(patches)} patches for image {image_width}x{image_height}")
        return patches

    def extract_patch(
            self,
            image: np.ndarray,
            patch_info: PatchInfo
    ) -> np.ndarray:
        """
        Extract a patch from the full image.

        Args:
            image: Full image array (H, W, C).
            patch_info: Patch location information.

        Returns:
            Extracted patch of size (patch_size, patch_size, C).
        """
        patch = image[patch_info.y1:patch_info.y2, patch_info.x1:patch_info.x2]

        # Ensure patch is the correct size (pad if necessary)
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            if len(patch.shape) == 3:
                padded_patch = np.zeros((self.patch_size, self.patch_size, patch.shape[2]),
                                        dtype=patch.dtype)
            else:
                padded_patch = np.zeros((self.patch_size, self.patch_size), dtype=patch.dtype)

            h, w = patch.shape[:2]
            padded_patch[:h, :w] = patch
            patch = padded_patch

        return patch


class NonMaximumSuppression:
    """Non-Maximum Suppression for overlapping detections."""

    @staticmethod
    def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_intersect = max(x1_1, x1_2)
        y1_intersect = max(y1_1, y1_2)
        x2_intersect = min(x2_1, x2_2)
        y2_intersect = min(y2_1, y2_2)

        if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
            return 0.0

        intersection = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @classmethod
    def apply_nms(
            cls,
            detections: List[DetectionResult],
            iou_threshold: float = 0.5,
            score_threshold: float = 0.1
    ) -> List[DetectionResult]:
        """
        Apply Non-Maximum Suppression to detection results.

        Args:
            detections: List of detection results.
            iou_threshold: IoU threshold for suppression.
            score_threshold: Minimum confidence score threshold.

        Returns:
            Filtered list of detections after NMS.
        """
        if not detections:
            return []

        # Filter by score threshold
        filtered_detections = [d for d in detections if d.confidence >= score_threshold]

        if not filtered_detections:
            return []

        # Sort by confidence (descending)
        sorted_detections = sorted(filtered_detections, key=lambda x: x.confidence, reverse=True)

        # Apply NMS
        keep = []
        while sorted_detections:
            # Keep the detection with highest confidence
            current = sorted_detections.pop(0)
            keep.append(current)

            # Remove detections with high IoU
            remaining = []
            for detection in sorted_detections:
                iou = cls.compute_iou(current.bbox, detection.bbox)
                if iou < iou_threshold:
                    remaining.append(detection)

            sorted_detections = remaining

        return keep


class SegmentationStitcher:
    """Stitches segmentation patches into full image masks."""

    def __init__(self, blend_mode: str = "average"):
        """
        Initialize segmentation stitcher.

        Args:
            blend_mode: How to blend overlapping regions ('average', 'max', 'vote').
        """
        self.blend_mode = blend_mode

    def stitch_patches(
            self,
            patch_predictions: List[PatchPrediction],
            image_width: int,
            image_height: int
    ) -> np.ndarray:
        """
        Stitch segmentation patches into full image mask.

        Args:
            patch_predictions: List of patch predictions with masks.
            image_width: Width of full image.
            image_height: Height of full image.

        Returns:
            Full image segmentation mask (H, W).
        """
        # Initialize output mask and weight map
        full_mask = np.zeros((image_height, image_width), dtype=np.float32)
        weight_map = np.zeros((image_height, image_width), dtype=np.float32)

        for pred in patch_predictions:
            if pred.segmentation_mask is None:
                continue

            patch_info = pred.patch_info
            mask_patch = pred.segmentation_mask.astype(np.float32)

            # Create weight matrix (reduce weight at borders for smooth blending)
            if self.blend_mode == "average":
                weights = self._create_patch_weights(patch_info.patch_size)
            else:
                weights = np.ones((patch_info.patch_size, patch_info.patch_size), dtype=np.float32)

            # Extract the region of the full mask
            y1, y2 = patch_info.y1, patch_info.y2
            x1, x2 = patch_info.x1, patch_info.x2

            # Handle edge cases where patch might be smaller
            patch_h, patch_w = mask_patch.shape[:2]
            weights = weights[:patch_h, :patch_w]

            if self.blend_mode == "average":
                full_mask[y1:y2, x1:x2] += mask_patch * weights
                weight_map[y1:y2, x1:x2] += weights
            elif self.blend_mode == "max":
                full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask_patch)
                weight_map[y1:y2, x1:x2] = np.maximum(weight_map[y1:y2, x1:x2], weights)
            elif self.blend_mode == "vote":
                # Binary voting
                binary_mask = (mask_patch > 0.5).astype(np.float32)
                full_mask[y1:y2, x1:x2] += binary_mask
                weight_map[y1:y2, x1:x2] += 1.0

        # Normalize by weights
        valid_mask = weight_map > 0
        full_mask[valid_mask] /= weight_map[valid_mask]

        if self.blend_mode == "vote":
            # Convert vote counts to probabilities
            full_mask = full_mask / np.maximum(weight_map, 1)

        return full_mask

    def _create_patch_weights(self, patch_size: int) -> np.ndarray:
        """Create weight matrix with reduced weights at patch borders."""
        weights = np.ones((patch_size, patch_size), dtype=np.float32)

        # Create border fade
        fade_width = min(16, patch_size // 8)  # Fade over 16 pixels or 1/8 of patch

        if fade_width > 0:
            # Create fade mask
            for i in range(fade_width):
                weight = (i + 1) / fade_width

                # Apply fade to borders
                weights[i, :] *= weight  # top
                weights[-(i + 1), :] *= weight  # bottom
                weights[:, i] *= weight  # left
                weights[:, -(i + 1)] *= weight  # right

        return weights


class ClassificationAggregator:
    """Aggregates patch-level classification results."""

    @staticmethod
    def aggregate_scores(
            patch_predictions: List[PatchPrediction],
            method: str = "average"
    ) -> Tuple[float, float]:
        """
        Aggregate patch classification scores into image-level prediction.

        Args:
            patch_predictions: List of patch predictions.
            method: Aggregation method ('average', 'max', 'vote', 'weighted').

        Returns:
            Tuple of (aggregated_score, confidence).
        """
        if not patch_predictions:
            return 0.0, 0.0

        scores = [pred.classification_score for pred in patch_predictions]

        if method == "average":
            agg_score = np.mean(scores)
            confidence = 1.0 - np.std(scores)  # Lower std = higher confidence
        elif method == "max":
            agg_score = np.max(scores)
            confidence = agg_score
        elif method == "vote":
            binary_votes = [1 if score > 0.5 else 0 for score in scores]
            agg_score = np.mean(binary_votes)
            confidence = abs(agg_score - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
        elif method == "weighted":
            # Weight by detection confidence if available
            weights = []
            for pred in patch_predictions:
                if pred.detections:
                    max_det_conf = max(det.confidence for det in pred.detections)
                    weights.append(max_det_conf)
                else:
                    weights.append(pred.classification_score)

            if sum(weights) > 0:
                weights = np.array(weights) / sum(weights)
                agg_score = np.average(scores, weights=weights)
                confidence = np.average(weights)
            else:
                agg_score = np.mean(scores)
                confidence = 0.5
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return float(agg_score), float(max(0.0, min(1.0, confidence)))


class ResultAggregator:
    """Main class for aggregating all multi-task results."""

    def __init__(
            self,
            nms_iou_threshold: float = 0.5,
            nms_score_threshold: float = 0.1,
            segmentation_blend_mode: str = "average",
            classification_method: str = "weighted"
    ):
        """
        Initialize result aggregator.

        Args:
            nms_iou_threshold: IoU threshold for NMS.
            nms_score_threshold: Score threshold for detections.
            segmentation_blend_mode: Blending mode for segmentation.
            classification_method: Classification aggregation method.
        """
        self.nms = NonMaximumSuppression()
        self.stitcher = SegmentationStitcher(blend_mode=segmentation_blend_mode)
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_threshold = nms_score_threshold
        self.classification_method = classification_method

    def aggregate_predictions(
            self,
            patch_predictions: List[PatchPrediction],
            image_width: int,
            image_height: int
    ) -> Dict[str, Any]:
        """
        Aggregate all patch predictions into final results.

        Args:
            patch_predictions: List of patch predictions.
            image_width: Width of full image.
            image_height: Height of full image.

        Returns:
            Dictionary with aggregated results for all tasks.
        """
        # Aggregate detections
        all_detections = []
        for pred in patch_predictions:
            for detection in pred.detections:
                # Transform bbox to full image coordinates
                full_bbox = CoordinateTransformer.patch_to_full_image_bbox(
                    detection.bbox, pred.patch_info
                )
                all_detections.append(DetectionResult(
                    bbox=full_bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id
                ))

        # Apply NMS to detections
        final_detections = self.nms.apply_nms(
            all_detections,
            iou_threshold=self.nms_iou_threshold,
            score_threshold=self.nms_score_threshold
        )

        # Aggregate segmentation
        segmentation_mask = self.stitcher.stitch_patches(
            patch_predictions, image_width, image_height
        )

        # Aggregate classification
        classification_score, classification_confidence = ClassificationAggregator.aggregate_scores(
            patch_predictions, method=self.classification_method
        )

        return {
            'detections': [det.to_dict() for det in final_detections],
            'segmentation': segmentation_mask,
            'classification': {
                'score': classification_score,
                'confidence': classification_confidence,
                'prediction': int(classification_score > 0.5)
            },
            'num_patches': len(patch_predictions),
            'stats': {
                'total_patch_detections': len(all_detections),
                'final_detections_after_nms': len(final_detections),
                'avg_patch_classification_score': np.mean([p.classification_score for p in patch_predictions]),
                'segmentation_coverage': np.sum(segmentation_mask > 0.5) / (image_width * image_height)
            }
        }


# Utility functions for easy usage
def create_patch_grid(
        image_width: int,
        image_height: int,
        patch_size: int = 256,
        overlap: int = 64
) -> List[PatchInfo]:
    """Convenience function to create patch grid."""
    generator = PatchGridGenerator(patch_size=patch_size, overlap=overlap)
    return generator.generate_patches(image_width, image_height)


def aggregate_patch_results(
        patch_predictions: List[PatchPrediction],
        image_width: int,
        image_height: int,
        **kwargs
) -> Dict[str, Any]:
    """Convenience function to aggregate patch results."""
    aggregator = ResultAggregator(**kwargs)
    return aggregator.aggregate_predictions(patch_predictions, image_width, image_height)
