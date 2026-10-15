"""Every vision head's ``compute_output_shape`` equals the shape ``call`` returns.

Guard for plan-2026-10-14T062616-81938e81 step 7 (decisions D-027, D-028). At
``84438bee6`` the DPT ``DepthEstimationHead`` reported ``'confidence'`` as
``(B, sH, sW, output_channels)`` while ``call`` returned the last decoder
stage's features, ``(B, sH, sW, dpt_dims[-1])``: measured 1 against 32 channels
at the defaults. Every other head agreed. The check is eager ``call`` against
``compute_output_shape`` on a non-square input, per head and per mode.

``SegmentationHead``'s ``upsampling_factor`` is also pinned to mean what it
says: the output is ``upsampling_factor`` times the input. It built
``int(sqrt(factor))`` doublings, which is the factor only at 2, 4, 16 and 32
(factor 8 gave 4x, factor 1 gave 2x), consistently in ``call`` and
``compute_output_shape``, so a shape-agreement check alone could not see it.
"""

from typing import Any

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision import create_multi_task_head
from dl_techniques.layers.heads.vision.factory import (
    ClassificationHead,
    DepthEstimationHead,
    DetectionHead,
    EnhancementHead,
    InstanceSegmentationHead,
    OpticalFlowHead,
    SegmentationHead,
    SurfaceNormalsHead,
)
from dl_techniques.layers.heads.vision import create_vision_head
from dl_techniques.layers.heads.vision.task_types import VisionTaskType

HIDDEN_DIM = 16
SPATIAL = (2, 6, 10, HIDDEN_DIM)
SEQUENCE = (2, 12, HIDDEN_DIM)

_SUBJECTS = (
    ("classification", lambda: ClassificationHead(num_classes=5, hidden_dim=HIDDEN_DIM), SPATIAL),
    ("classification_cls_token", lambda: ClassificationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, pooling_type="cls_token"), SEQUENCE),
    ("segmentation", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_skip_connections=False), SPATIAL),
    ("segmentation_x16", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_skip_connections=False,
        upsampling_factor=16), SPATIAL),
    ("segmentation_x8", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_skip_connections=False,
        upsampling_factor=8), SPATIAL),
    ("segmentation_x1", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_skip_connections=False,
        upsampling_factor=1), SPATIAL),
    ("segmentation_sequence", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_skip_connections=False,
        input_format="sequence", patch_grid_size=(3, 4)), SEQUENCE),
    ("instance_segmentation", lambda: InstanceSegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, num_instances=4), SPATIAL),
    ("detection", lambda: DetectionHead(num_classes=5, hidden_dim=HIDDEN_DIM), SPATIAL),
    ("detection_without_regression", lambda: DetectionHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_regression=False), SPATIAL),
    ("detection_sequence", lambda: DetectionHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, input_format="sequence",
        patch_grid_size=(3, 4)), SEQUENCE),
    ("depth_progressive", lambda: DepthEstimationHead(hidden_dim=HIDDEN_DIM), SPATIAL),
    ("depth_dpt_defaults", lambda: DepthEstimationHead(decoder_style="dpt"), SPATIAL),
    ("depth_dpt_narrow", lambda: DepthEstimationHead(
        decoder_style="dpt", dpt_dims=[16, 8],
        dpt_upsample_factor=4, output_channels=2), SPATIAL),
    ("enhancement", lambda: EnhancementHead(hidden_dim=HIDDEN_DIM), SPATIAL),
    ("enhancement_x2", lambda: EnhancementHead(hidden_dim=HIDDEN_DIM, scale_factor=2), SPATIAL),
    ("surface_normals", lambda: SurfaceNormalsHead(hidden_dim=HIDDEN_DIM), SPATIAL),
    ("optical_flow", lambda: OpticalFlowHead(hidden_dim=HIDDEN_DIM), SPATIAL),
    ("keypoint_detection", lambda: create_vision_head(
        "keypoint_detection", num_classes=65, hidden_dim=HIDDEN_DIM), SPATIAL),
    ("multi_task", lambda: create_multi_task_head(
        [VisionTaskType.DETECTION, "depth_estimation", "surface_normals",
         "optical_flow", "denoising"],
        detection={"num_classes": 3, "hidden_dim": HIDDEN_DIM},
        depth_estimation={"hidden_dim": HIDDEN_DIM},
        surface_normals={"hidden_dim": HIDDEN_DIM},
        optical_flow={"hidden_dim": HIDDEN_DIM},
        denoising={"hidden_dim": HIDDEN_DIM},
        use_task_specific_attention=False), SPATIAL),
)


def _shape_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _shape_tree(v) for k, v in value.items()}
    return tuple(value.shape) if hasattr(value, "shape") else tuple(value)


@pytest.mark.parametrize("label,builder,shape", _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_compute_output_shape_matches_the_eager_call(label, builder, shape) -> None:
    head = builder()
    x = np.random.default_rng(5).normal(size=shape).astype("float32")
    returned = _shape_tree(head(x))
    reported = _shape_tree(head.compute_output_shape(shape))
    assert reported == returned, (
        f"{label}: compute_output_shape reports {reported}, call returns {returned}"
    )


def test_the_dpt_confidence_is_the_last_decoder_stage() -> None:
    """Pin WHAT 'confidence' is in DPT mode, not only that the shapes agree."""
    head = DepthEstimationHead(decoder_style="dpt", dpt_dims=[16, 8],
                               dpt_upsample_factor=4)
    out = head(np.random.default_rng(6).normal(size=SPATIAL).astype("float32"))
    assert tuple(out["confidence"].shape) == (2, 24, 40, 8)
    assert tuple(out["depth"].shape) == (2, 24, 40, 1)


@pytest.mark.parametrize("factor", [1, 2, 4, 8, 16])
def test_the_segmentation_output_is_upsampling_factor_times_the_input(factor) -> None:
    head = SegmentationHead(num_classes=3, hidden_dim=HIDDEN_DIM,
                            use_skip_connections=False, upsampling_factor=factor)
    out = head(np.random.default_rng(7).normal(size=SPATIAL).astype("float32"))
    assert tuple(out.shape) == (2, 6 * factor, 10 * factor, 3)
    assert len(head.upsample_layers) == factor.bit_length() - 1


@pytest.mark.parametrize("factor", [0, 3, 6, 12, 4.0, True])
def test_a_factor_that_is_not_a_power_of_two_is_refused(factor) -> None:
    with pytest.raises(ValueError, match="upsampling_factor"):
        SegmentationHead(num_classes=3, upsampling_factor=factor)
