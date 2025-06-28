"""
Comprehensive test suite for bounding box IoU utility functions.

This module tests all IoU calculation variants, NMS functionality,
and edge cases to ensure robust performance in object detection tasks.
"""

import pytest
import numpy as np
import keras
from keras import ops
from typing import Tuple, Union

from dl_techniques.utils.bounding_box import (
    bbox_iou,
    bbox_nms,
    _calculate_intersection_area,
    _convert_to_corner_format)


class TestBboxIoU:
    """Test suite for bounding box IoU calculations."""

    @pytest.fixture
    def perfect_overlap_boxes_xyxy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create identical boxes in xyxy format for perfect overlap test."""
        box1 = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
        box2 = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
        return box1, box2

    @pytest.fixture
    def perfect_overlap_boxes_xywh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create identical boxes in xywh format for perfect overlap test."""
        # Center at (15, 15) with width=10, height=10
        box1 = np.array([[15.0, 15.0, 10.0, 10.0]], dtype=np.float32)
        box2 = np.array([[15.0, 15.0, 10.0, 10.0]], dtype=np.float32)
        return box1, box2

    @pytest.fixture
    def no_overlap_boxes_xyxy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create non-overlapping boxes in xyxy format."""
        box1 = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        box2 = np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
        return box1, box2

    @pytest.fixture
    def partial_overlap_boxes_xyxy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create partially overlapping boxes in xyxy format."""
        box1 = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        box2 = np.array([[5.0, 5.0, 15.0, 15.0]], dtype=np.float32)
        return box1, box2

    @pytest.fixture
    def batch_boxes_xyxy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create batch of boxes for testing broadcasting."""
        box1 = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0],
            [20.0, 20.0, 30.0, 30.0]
        ], dtype=np.float32)
        box2 = np.array([
            [5.0, 5.0, 15.0, 15.0],
            [10.0, 10.0, 20.0, 20.0],
            [25.0, 25.0, 35.0, 35.0]
        ], dtype=np.float32)
        return box1, box2

    def test_perfect_overlap_standard_iou_xyxy(self, perfect_overlap_boxes_xyxy):
        """Test standard IoU with perfect overlap in xyxy format."""
        box1, box2 = perfect_overlap_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        iou = bbox_iou(box1_tensor, box2_tensor, xywh=False)

        assert iou.shape == (1,)
        assert np.allclose(iou.numpy(), 1.0, rtol=1e-5)

    def test_perfect_overlap_standard_iou_xywh(self, perfect_overlap_boxes_xywh):
        """Test standard IoU with perfect overlap in xywh format."""
        box1, box2 = perfect_overlap_boxes_xywh
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        iou = bbox_iou(box1_tensor, box2_tensor, xywh=True)

        assert iou.shape == (1,)
        assert np.allclose(iou.numpy(), 1.0, rtol=1e-5)

    def test_no_overlap_standard_iou(self, no_overlap_boxes_xyxy):
        """Test standard IoU with no overlap."""
        box1, box2 = no_overlap_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        iou = bbox_iou(box1_tensor, box2_tensor, xywh=False)

        assert iou.shape == (1,)
        assert np.allclose(iou.numpy(), 0.0, atol=1e-6)

    def test_partial_overlap_standard_iou(self, partial_overlap_boxes_xyxy):
        """Test standard IoU with partial overlap."""
        box1, box2 = partial_overlap_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        iou = bbox_iou(box1_tensor, box2_tensor, xywh=False)

        # Expected IoU calculation:
        # Intersection area: 5x5 = 25
        # Union area: 100 + 100 - 25 = 175
        # IoU = 25/175 = 1/7 ≈ 0.1429
        expected_iou = 1.0 / 7.0

        assert iou.shape == (1,)
        assert np.allclose(iou.numpy(), expected_iou, rtol=1e-4)

    def test_batch_processing(self, batch_boxes_xyxy):
        """Test IoU calculation with batch of boxes."""
        box1, box2 = batch_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        iou = bbox_iou(box1_tensor, box2_tensor, xywh=False)

        assert iou.shape == (3,)

        # First pair: partial overlap (same as partial_overlap test)
        assert np.allclose(iou.numpy()[0], 1.0 / 7.0, rtol=1e-4)

        # Second pair: perfect overlap
        assert np.allclose(iou.numpy()[1], 1.0, rtol=1e-5)

        # Third pair: partial overlap
        # Box1: [20,20,30,30], Box2: [25,25,35,35]
        # Intersection: 5x5=25, Union: 100+100-25=175, IoU=25/175=1/7
        assert np.allclose(iou.numpy()[2], 1.0 / 7.0, rtol=1e-4)

    def test_giou_no_overlap(self, no_overlap_boxes_xyxy):
        """Test GIoU with non-overlapping boxes."""
        box1, box2 = no_overlap_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        giou = bbox_iou(box1_tensor, box2_tensor, xywh=False, GIoU=True)

        # GIoU should be negative for non-overlapping boxes
        assert giou.shape == (1,)
        assert giou.numpy()[0] < 0.0

        # Calculate expected GIoU:
        # IoU = 0, Enclosing box area = 30*30 = 900, Union = 200
        # GIoU = 0 - (900-200)/900 = -700/900 ≈ -0.7778
        expected_giou = -700.0 / 900.0
        assert np.allclose(giou.numpy(), expected_giou, rtol=1e-4)

    def test_diou_calculation(self, partial_overlap_boxes_xyxy):
        """Test DIoU calculation."""
        box1, box2 = partial_overlap_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        diou = bbox_iou(box1_tensor, box2_tensor, xywh=False, DIoU=True)
        standard_iou = bbox_iou(box1_tensor, box2_tensor, xywh=False)

        assert diou.shape == (1,)
        # DIoU should be less than standard IoU due to center distance penalty
        assert diou.numpy()[0] < standard_iou.numpy()[0]

    def test_ciou_calculation(self, partial_overlap_boxes_xyxy):
        """Test CIoU calculation."""
        box1, box2 = partial_overlap_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        ciou = bbox_iou(box1_tensor, box2_tensor, xywh=False, CIoU=True)
        diou = bbox_iou(box1_tensor, box2_tensor, xywh=False, DIoU=True)

        assert ciou.shape == (1,)
        # CIoU should be less than DIoU due to aspect ratio penalty
        # (in this case, boxes have same aspect ratio, so difference might be small)
        assert ciou.numpy()[0] <= diou.numpy()[0]

    def test_multiple_iou_variants_error(self, perfect_overlap_boxes_xyxy):
        """Test that selecting multiple IoU variants raises ValueError."""
        box1, box2 = perfect_overlap_boxes_xyxy
        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        with pytest.raises(ValueError, match="Only one of GIoU, DIoU, or CIoU can be True"):
            bbox_iou(box1_tensor, box2_tensor, GIoU=True, DIoU=True)

        with pytest.raises(ValueError, match="Only one of GIoU, DIoU, or CIoU can be True"):
            bbox_iou(box1_tensor, box2_tensor, CIoU=True, GIoU=True)

        with pytest.raises(ValueError, match="Only one of GIoU, DIoU, or CIoU can be True"):
            bbox_iou(box1_tensor, box2_tensor, CIoU=True, DIoU=True, GIoU=True)

    def test_format_conversion_consistency(self):
        """Test that xyxy and xywh formats give same results when representing same boxes."""
        # Same box in different formats
        box_xyxy = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
        box_xywh = np.array([[15.0, 15.0, 10.0, 10.0]], dtype=np.float32)  # center=(15,15), size=(10,10)

        box_xyxy_tensor = ops.convert_to_tensor(box_xyxy)
        box_xywh_tensor = ops.convert_to_tensor(box_xywh)

        iou_xyxy = bbox_iou(box_xyxy_tensor, box_xyxy_tensor, xywh=False)
        iou_xywh = bbox_iou(box_xywh_tensor, box_xywh_tensor, xywh=True)

        assert np.allclose(iou_xyxy.numpy(), iou_xywh.numpy(), rtol=1e-5)

    def test_zero_area_boxes(self):
        """Test handling of zero-area boxes."""
        # Zero area boxes (degenerate)
        box1 = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
        box2 = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)

        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        iou = bbox_iou(box1_tensor, box2_tensor, xywh=False)

        # IoU should be 0 when one box has zero area
        assert iou.shape == (1,)
        assert np.allclose(iou.numpy(), 0.0, atol=1e-6)

    def test_broadcasting_different_shapes(self):
        """Test broadcasting with different input shapes."""
        # Single box vs multiple boxes
        box1 = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)  # (1, 4)
        box2 = np.array([
            [5.0, 5.0, 15.0, 15.0],
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0]
        ], dtype=np.float32)  # (3, 4)

        box1_tensor = ops.convert_to_tensor(box1)
        box2_tensor = ops.convert_to_tensor(box2)

        iou = bbox_iou(box1_tensor, box2_tensor, xywh=False)

        assert iou.shape == (3,)

        # First: partial overlap
        assert np.allclose(iou.numpy()[0], 1.0 / 7.0, rtol=1e-4)
        # Second: perfect overlap
        assert np.allclose(iou.numpy()[1], 1.0, rtol=1e-5)
        # Third: no overlap
        assert np.allclose(iou.numpy()[2], 0.0, atol=1e-6)


class TestBboxNMS:
    """Test suite for Non-Maximum Suppression functionality."""

    @pytest.fixture
    def sample_boxes_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sample boxes and scores for NMS testing."""
        boxes = np.array([
            [10.0, 10.0, 20.0, 20.0],  # High overlap with box 1
            [12.0, 12.0, 22.0, 22.0],  # High overlap with box 0
            [50.0, 50.0, 60.0, 60.0],  # No overlap
            [51.0, 51.0, 61.0, 61.0],  # High overlap with box 2
            [100.0, 100.0, 110.0, 110.0]  # No overlap
        ], dtype=np.float32)

        scores = np.array([0.9, 0.8, 0.95, 0.7, 0.6], dtype=np.float32)
        return boxes, scores

    def test_basic_nms_functionality(self, sample_boxes_scores):
        """Test basic NMS functionality."""
        boxes, scores = sample_boxes_scores
        boxes_tensor = ops.convert_to_tensor(boxes)
        scores_tensor = ops.convert_to_tensor(scores)

        selected_boxes, selected_scores, selected_indices = bbox_nms(
            boxes_tensor, scores_tensor, iou_threshold=0.5, score_threshold=0.0
        )

        # Should select fewer boxes than input
        assert selected_boxes.shape[0] <= boxes.shape[0]
        assert selected_scores.shape[0] == selected_boxes.shape[0]
        assert selected_indices.shape[0] == selected_boxes.shape[0]

        # Scores should be in descending order
        selected_scores_np = selected_scores.numpy()
        assert np.all(selected_scores_np[:-1] >= selected_scores_np[1:])

    def test_nms_score_threshold(self, sample_boxes_scores):
        """Test NMS with score threshold filtering."""
        boxes, scores = sample_boxes_scores
        boxes_tensor = ops.convert_to_tensor(boxes)
        scores_tensor = ops.convert_to_tensor(scores)

        # High score threshold should filter out low-scoring boxes
        selected_boxes, selected_scores, selected_indices = bbox_nms(
            boxes_tensor, scores_tensor,
            iou_threshold=0.5, score_threshold=0.8
        )

        # Should only keep boxes with scores >= 0.8
        assert np.all(selected_scores.numpy() >= 0.8)

    def test_nms_max_outputs(self, sample_boxes_scores):
        """Test NMS with max outputs limit."""
        boxes, scores = sample_boxes_scores
        boxes_tensor = ops.convert_to_tensor(boxes)
        scores_tensor = ops.convert_to_tensor(scores)

        max_outputs = 2
        selected_boxes, selected_scores, selected_indices = bbox_nms(
            boxes_tensor, scores_tensor,
            iou_threshold=0.1, score_threshold=0.0, max_outputs=max_outputs
        )

        # Should not exceed max_outputs
        assert selected_boxes.shape[0] <= max_outputs

    def test_nms_no_boxes_pass_threshold(self):
        """Test NMS when no boxes pass the score threshold."""
        boxes = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
        scores = np.array([0.1], dtype=np.float32)

        boxes_tensor = ops.convert_to_tensor(boxes)
        scores_tensor = ops.convert_to_tensor(scores)

        selected_boxes, selected_scores, selected_indices = bbox_nms(
            boxes_tensor, scores_tensor,
            iou_threshold=0.5, score_threshold=0.5  # Higher than any score
        )

        # Should return empty tensors
        assert selected_boxes.shape == (0, 4)
        assert selected_scores.shape == (0,)
        assert selected_indices.shape == (0,)

    def test_nms_xywh_format(self):
        """Test NMS with xywh format boxes."""
        # Convert sample boxes to xywh format
        boxes_xyxy = np.array([
            [10.0, 10.0, 20.0, 20.0],
            [12.0, 12.0, 22.0, 22.0]
        ], dtype=np.float32)

        # Convert to xywh: center_x, center_y, width, height
        boxes_xywh = np.array([
            [15.0, 15.0, 10.0, 10.0],  # (10,10,20,20) -> center=(15,15), size=(10,10)
            [17.0, 17.0, 10.0, 10.0]  # (12,12,22,22) -> center=(17,17), size=(10,10)
        ], dtype=np.float32)

        scores = np.array([0.9, 0.8], dtype=np.float32)

        boxes_tensor = ops.convert_to_tensor(boxes_xywh)
        scores_tensor = ops.convert_to_tensor(scores)

        selected_boxes, selected_scores, selected_indices = bbox_nms(
            boxes_tensor, scores_tensor,
            iou_threshold=0.5, xywh=True
        )

        # Should work without errors
        assert selected_boxes.shape[1] == 4
        assert len(selected_scores.shape) == 1


class TestHelperFunctions:
    """Test suite for helper functions used in IoU calculations."""

    def test_convert_to_corner_format_xywh(self):
        """Test conversion from xywh to corner format."""
        # Center format: (15, 15, 10, 10) should convert to (10, 10, 20, 20)
        box1 = ops.convert_to_tensor([[15.0, 15.0, 10.0, 10.0]])
        box2 = ops.convert_to_tensor([[15.0, 15.0, 10.0, 10.0]])

        result = _convert_to_corner_format(box1, box2, xywh=True)
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2 = result

        # Check conversion
        assert np.allclose(b1_x1.numpy(), [[10.0]])
        assert np.allclose(b1_y1.numpy(), [[10.0]])
        assert np.allclose(b1_x2.numpy(), [[20.0]])
        assert np.allclose(b1_y2.numpy(), [[20.0]])

    def test_calculate_intersection_area(self):
        """Test intersection area calculation."""

        # Create overlapping boxes
        b1_x1 = ops.convert_to_tensor([[0.0]])
        b1_y1 = ops.convert_to_tensor([[0.0]])
        b1_x2 = ops.convert_to_tensor([[10.0]])
        b1_y2 = ops.convert_to_tensor([[10.0]])

        b2_x1 = ops.convert_to_tensor([[5.0]])
        b2_y1 = ops.convert_to_tensor([[5.0]])
        b2_x2 = ops.convert_to_tensor([[15.0]])
        b2_y2 = ops.convert_to_tensor([[15.0]])

        intersection = _calculate_intersection_area(
            b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
        )

        # Expected intersection: 5x5 = 25
        assert np.allclose(intersection.numpy(), [[25.0]])

    def test_calculate_intersection_area_no_overlap(self):
        """Test intersection area calculation with no overlap."""

        # Create non-overlapping boxes
        b1_x1 = ops.convert_to_tensor([[0.0]])
        b1_y1 = ops.convert_to_tensor([[0.0]])
        b1_x2 = ops.convert_to_tensor([[10.0]])
        b1_y2 = ops.convert_to_tensor([[10.0]])

        b2_x1 = ops.convert_to_tensor([[20.0]])
        b2_y1 = ops.convert_to_tensor([[20.0]])
        b2_x2 = ops.convert_to_tensor([[30.0]])
        b2_y2 = ops.convert_to_tensor([[30.0]])

        intersection = _calculate_intersection_area(
            b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
        )

        # Expected intersection: 0
        assert np.allclose(intersection.numpy(), [[0.0]])


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_empty_boxes(self):
        """Test behavior with empty box tensors."""
        empty_boxes = ops.convert_to_tensor(np.zeros((0, 4), dtype=np.float32))

        # This should handle gracefully or raise appropriate error
        try:
            iou = bbox_iou(empty_boxes, empty_boxes, xywh=False)
            assert iou.shape == (0,)
        except Exception as e:
            # If it raises an exception, it should be informative
            assert "empty" in str(e).lower() or "shape" in str(e).lower()

    def test_mismatched_shapes(self):
        """Test error handling for mismatched input shapes."""
        box1 = ops.convert_to_tensor([[0.0, 0.0, 10.0, 10.0]])  # (1, 4)
        box2 = ops.convert_to_tensor([0.0, 0.0, 10.0])  # (3,) - wrong shape

        with pytest.raises(Exception):  # Should raise some kind of error
            bbox_iou(box1, box2, xywh=False)

    def test_negative_coordinates(self):
        """Test handling of negative coordinates."""
        box1 = ops.convert_to_tensor([[-10.0, -10.0, 0.0, 0.0]])
        box2 = ops.convert_to_tensor([[-5.0, -5.0, 5.0, 5.0]])

        iou = bbox_iou(box1, box2, xywh=False)

        # Should handle negative coordinates properly
        assert not np.any(np.isnan(iou.numpy()))
        assert not np.any(np.isinf(iou.numpy()))
        assert 0.0 <= iou.numpy()[0] <= 1.0

    def test_very_large_coordinates(self):
        """Test handling of very large coordinates."""
        large_val = 1e6
        box1 = ops.convert_to_tensor([[0.0, 0.0, large_val, large_val]])
        box2 = ops.convert_to_tensor([[0.0, 0.0, large_val, large_val]])

        iou = bbox_iou(box1, box2, xywh=False)

        # Should handle large coordinates properly
        assert not np.any(np.isnan(iou.numpy()))
        assert not np.any(np.isinf(iou.numpy()))
        assert np.allclose(iou.numpy(), 1.0, rtol=1e-4)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])