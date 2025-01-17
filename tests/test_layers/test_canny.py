"""
Test suite for the Canny Edge Detector implementation.

This module contains test cases to verify the functionality of the
CannyEdgeDetector class using pytest.
"""

import pytest
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from dl_techniques.layers.canny import Canny


@pytest.fixture
def default_detector() -> Canny:
    """
    Fixture providing a CannyEdgeDetector instance with default parameters.

    Returns:
        CannyEdgeDetector: Default configured detector
    """
    return Canny()


@pytest.fixture
def custom_detector() -> Canny:
    """
    Fixture providing a CannyEdgeDetector instance with custom parameters.

    Returns:
        CannyEdgeDetector: Custom configured detector
    """
    return Canny(
        sigma=1.0,
        threshold_min=30,
        threshold_max=90,
        tracking_connection=7,
        tracking_iterations=4
    )


def create_test_image(size: Tuple[int, int] = (64, 64)) -> tf.Tensor:
    """
    Create a simple test image with a vertical line.

    Args:
        size: Tuple of (height, width) for the image size

    Returns:
        tf.Tensor: Test image as a tensor
    """
    image = np.zeros(size, dtype=np.float32)
    center = size[1] // 2
    image[:, center - 1:center + 2] = 255.0
    return tf.convert_to_tensor(image[None, :, :, None])


def test_initialization():
    """Test that the detector initializes with different parameters."""
    # Test default initialization
    detector1 = Canny()
    assert detector1.threshold == (50, 80)

    # Test custom initialization
    detector2 = Canny(
        sigma=1.2,
        threshold_min=40,
        threshold_max=100,
        tracking_connection=3,
        tracking_iterations=5
    )
    assert detector2.threshold == (40, 100)

    # Test invalid sigma
    with pytest.raises(ValueError):
        Canny(sigma=0.5)


def test_basic_edge_detection(default_detector: Canny):
    """Test edge detection on a simple vertical line image."""
    # Create test image with vertical line
    test_image = create_test_image()

    # Detect edges
    edges = default_detector(test_image)

    # Verify output properties
    assert edges.shape == test_image.shape
    assert tf.reduce_max(edges) <= 1.0
    assert tf.reduce_min(edges) >= 0.0

    # Verify edge detection (should find vertical edges)
    center_slice = edges[0, :, 32, 0]
    edge_positions = tf.where(center_slice > 0.5)
    assert len(edge_positions) >= 1


def test_threshold_sensitivity(custom_detector: Canny):
    """Test edge detection sensitivity to threshold changes."""
    test_image = create_test_image()

    # Detect edges with different thresholds
    detector1 = Canny(threshold_min=20, threshold_max=60)
    detector2 = Canny(threshold_min=80, threshold_max=120)

    edges1 = detector1(test_image)
    edges2 = detector2(test_image)

    # Higher threshold should detect fewer edges
    assert tf.reduce_sum(edges1) >= tf.reduce_sum(edges2)


def test_noise_handling():
    """Test edge detection with noisy input."""
    # Create noisy test image
    size = (64, 64)
    noise = np.random.normal(0, 25, size).astype(np.float32)
    image = create_test_image(size)
    noisy_image = tf.clip_by_value(image + noise[None, :, :, None], 0, 255)

    # Test with different sigma values
    detector1 = Canny(sigma=0.8)  # Less smoothing
    detector2 = Canny(sigma=2.0)  # More smoothing

    edges1 = detector1(noisy_image)
    edges2 = detector2(noisy_image)

    # More smoothing should result in fewer edges
    assert tf.reduce_sum(edges1) >= tf.reduce_sum(edges2)


def test_output_consistency(default_detector: Canny):
    """Test consistency of edge detection results."""
    test_image = create_test_image()

    # Run edge detection multiple times
    results: List[tf.Tensor] = []
    for _ in range(3):
        edges = default_detector(test_image)
        results.append(edges)

    # Verify results are consistent
    for i in range(1, len(results)):
        assert tf.reduce_all(tf.equal(results[0], results[i]))


def create_circle_image(size: Tuple[int, int] = (128, 128),
                        radius: int = 40) -> tf.Tensor:
    """
    Create a test image with a circle.

    Args:
        size: Tuple of (height, width) for the image size
        radius: Radius of the circle

    Returns:
        tf.Tensor: Test image with a circle
    """
    y, x = np.ogrid[-size[0] // 2:size[0] // 2, -size[1] // 2:size[1] // 2]
    mask = x * x + y * y <= radius * radius
    image = np.zeros(size, dtype=np.float32)
    image[mask] = 255.0
    return tf.convert_to_tensor(image[None, :, :, None])


def test_multi_scale_detection():
    """
    Test edge detection at multiple scales.
    Verifies that the detector can handle images of different sizes
    and maintains reasonable edge detection characteristics across scales.

    The test accounts for non-linear relationships between image scaling
    and edge detection by:
    1. Adjusting sigma proportionally to scale
    2. Using scale-appropriate thresholds
    3. Comparing edge structure rather than raw density
    """
    base_size = (64, 64)
    scales = [0.5, 1.0, 2.0]
    base_sigma = 1.0

    # Create base image and get reference edges
    base_image = create_circle_image(base_size)
    base_detector = Canny(
        sigma=base_sigma,
        threshold_min=50,
        threshold_max=80
    )
    base_edges = base_detector(base_image)

    for scale in scales:
        # Adjust detector parameters based on scale
        scaled_sigma = base_sigma * scale
        scaled_detector = Canny(
            sigma=max(0.8, scaled_sigma),  # Ensure minimum sigma
            threshold_min=int(50 * np.sqrt(scale)),  # Scale thresholds non-linearly
            threshold_max=int(80 * np.sqrt(scale))
        )

        # Resize image
        new_size = tuple(int(s * scale) for s in base_size)
        scaled_image = tf.image.resize(
            base_image,
            new_size,
            method='bilinear'
        )

        # Detect edges at new scale
        scaled_edges = scaled_detector(scaled_image)

        # Resize edge map back to base size for comparison
        normalized_edges = tf.image.resize(
            scaled_edges,
            base_size,
            method='nearest'
        )

        # Compare structural similarity rather than raw density
        # Allow for greater variance at extreme scales
        tolerance = 0.3 if scale in [0.5, 2.0] else 0.2

        # Calculate Intersection over Union (IoU) of edge regions
        intersection = tf.reduce_sum(tf.minimum(normalized_edges, base_edges))
        union = tf.reduce_sum(tf.maximum(normalized_edges, base_edges))
        iou = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero

        # IoU should be above threshold
        min_iou = 0.5 if scale in [0.5, 2.0] else 0.6
        assert iou > min_iou, f"Scale {scale}: IoU {iou} below threshold {min_iou}"

        # Check edge structure continuity
        edge_diff = tf.abs(normalized_edges - base_edges)
        mean_diff = tf.reduce_mean(edge_diff)
        assert mean_diff < tolerance, f"Scale {scale}: Mean difference {mean_diff} exceeds tolerance {tolerance}"


def test_gradient_direction_accuracy():
    """
    Test accuracy of gradient direction computation.
    Verifies that the detector correctly identifies edge orientations
    using test patterns with known gradient directions.
    """
    size = (64, 64)
    center = size[0] // 2
    padding = 2  # Width of the line

    def create_line_pattern(angle_deg: float) -> tf.Tensor:
        """Create a test pattern with a line at specified angle."""
        image = np.zeros(size, dtype=np.float32)

        if angle_deg == 0:  # Vertical line
            image[:, center - padding:center + padding] = 255.0
        elif angle_deg == 90:  # Horizontal line
            image[center - padding:center + padding, :] = 255.0
        elif angle_deg == 45:  # Diagonal line
            for i in range(size[0]):
                for j in range(max(0, i - padding), min(size[0], i + padding + 1)):
                    image[i, j] = 255.0

        # Add slight blur to avoid aliasing
        image = tf.convert_to_tensor(image[None, :, :, None])
        return image

    # Test angles
    angles = [0, 45, 90]
    detector = Canny(
        sigma=1.0,
        threshold_min=20,
        threshold_max=60
    )

    for angle in angles:
        # Create test pattern for this angle
        image = create_line_pattern(angle)
        edges = detector(image)

        # Create region of interest mask (strip along expected edge)
        roi_mask = np.zeros(size, dtype=np.float32)
        if angle == 0:  # Vertical
            roi_mask[:, center - 4:center + 4] = 1.0
        elif angle == 90:  # Horizontal
            roi_mask[center - 4:center + 4, :] = 1.0
        elif angle == 45:  # Diagonal
            for i in range(size[0]):
                for j in range(max(0, i - 4), min(size[0], i + 5)):
                    roi_mask[i, j] = 1.0

        roi_mask = tf.convert_to_tensor(roi_mask)
        non_roi_mask = 1.0 - roi_mask

        # Calculate edge responses in ROI and non-ROI regions
        edges_in_roi = edges[0, :, :, 0] * roi_mask
        edges_outside_roi = edges[0, :, :, 0] * non_roi_mask

        # Calculate metrics
        roi_sum = tf.reduce_sum(edges_in_roi)
        non_roi_sum = tf.reduce_sum(edges_outside_roi)
        roi_area = tf.reduce_sum(roi_mask)
        non_roi_area = tf.reduce_sum(non_roi_mask)

        # Calculate edge density in both regions
        roi_density = roi_sum / (roi_area + 1e-6)
        non_roi_density = non_roi_sum / (non_roi_area + 1e-6)

        # Assertions
        assert roi_sum > 0, f"No edges detected along {angle} degree line"
        assert roi_density > non_roi_density * 3, (
            f"Edge density along {angle} degree line not significantly "
            f"higher than background (ROI density: {roi_density}, "
            f"Non-ROI density: {non_roi_density})"
        )

        # Verify edge continuity in ROI
        edge_points = tf.where(edges_in_roi > 0)
        if len(edge_points) > 0:
            # Calculate gaps between consecutive edge points
            sorted_points = tf.sort(edge_points[:, 1 if angle == 0 else 0])
            gaps = sorted_points[1:] - sorted_points[:-1]
            max_gap = tf.reduce_max(gaps)

            assert max_gap <= 3, (
                f"Large gap ({max_gap}) found in edge detection "
                f"for {angle} degree line"
            )


def test_batch_processing():
    """
    Test processing of batched inputs.
    Verifies that the detector can handle batched inputs correctly
    and produces consistent results regardless of batch size.
    """
    batch_sizes = [1, 2, 4, 8]
    image = create_circle_image()
    detector = Canny()

    # Create reference result
    reference = detector(image)

    for batch_size in batch_sizes:
        # Create batch by repeating the image
        batch = tf.repeat(image, batch_size, axis=0)
        batch_result = detector(batch)

        # Verify batch output shape
        assert batch_result.shape[0] == batch_size

        # Verify consistent results across batch
        for i in range(batch_size):
            batch_slice = batch_result[i:i + 1]
            assert tf.reduce_all(tf.equal(batch_slice, reference))

if __name__ == "__main__":
    pytest.main([__file__])