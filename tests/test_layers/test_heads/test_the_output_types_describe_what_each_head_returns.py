"""``VisionTaskType.get_output_types`` names exactly the keys each head returns.

Guard for plan-2026-10-14T062616-81938e81 step 11.1 (review semantic item,
decision D-052). Measured at ``9343b0f87``: ``get_output_types`` documented
keys no head returned for 9 of the 10 dispatched task types. Keypoint
detection documented ``keypoints``/``visibility``/``scores`` while its
``DetectionHead`` returned ``classifications``/``regressions`` (with 9 anchors
of box regressions by default), detection documented
``bboxes``/``classes``/``scores``, segmentation ``masks``/``logits`` for a head
that returns a bare tensor, and the 27 task types with no head fell back to a
generic ``{'output': 'float32[...]'}`` entry for a head that does not exist.

Now the table holds the ten dispatched task types and nothing else. A head
that returns a bare tensor (segmentation) is described by the one key
``'output'``; every other key must be a key of the head's output dict.
Keypoint detection builds a real keypoint score map: one logit vector per
location (``num_anchors=1``) and no box regression, the SuperPoint detector
head's 64 cell positions plus the dustbin when ``num_classes=65``.
"""

from typing import Any, Dict

import numpy as np
import pytest

from dl_techniques.layers.heads.vision import (
    TaskConfiguration,
    VisionTaskType,
    create_vision_head,
)

HIDDEN_DIM = 16
SPATIAL = (2, 6, 10, HIDDEN_DIM)

DISPATCHED: Dict[VisionTaskType, Dict[str, Any]] = {
    VisionTaskType.DETECTION: {"num_classes": 3},
    VisionTaskType.SEGMENTATION: {"num_classes": 3, "use_skip_connections": False},
    VisionTaskType.INSTANCE_SEGMENTATION: {"num_classes": 3, "num_instances": 4},
    VisionTaskType.CLASSIFICATION: {"num_classes": 3},
    VisionTaskType.DEPTH_ESTIMATION: {},
    VisionTaskType.SURFACE_NORMALS: {},
    VisionTaskType.OPTICAL_FLOW: {},
    VisionTaskType.KEYPOINT_DETECTION: {"num_classes": 65},
    VisionTaskType.DENOISING: {},
    VisionTaskType.SUPER_RESOLUTION: {},
}


def _returned_keys(out) -> set:
    return set(out) if isinstance(out, dict) else {"output"}


@pytest.mark.parametrize("task", list(DISPATCHED), ids=lambda t: t.value)
def test_the_documented_keys_are_the_returned_keys(task) -> None:
    head = create_vision_head(task, hidden_dim=HIDDEN_DIM, **DISPATCHED[task])
    out = head(np.random.default_rng(3).normal(size=SPATIAL).astype("float32"))
    documented = VisionTaskType.get_output_types(task)
    assert set(documented) == _returned_keys(out), (
        f"{task.value}: documented {sorted(documented)}, returned "
        f"{sorted(_returned_keys(out))}")


@pytest.mark.parametrize("task", [t for t in VisionTaskType if t not in DISPATCHED],
                         ids=lambda t: t.value)
def test_a_task_with_no_head_has_no_output_types(task) -> None:
    with pytest.raises(ValueError, match=task.value):
        VisionTaskType.get_output_types(task)


def test_the_keypoint_head_is_a_score_map_not_a_box_detector() -> None:
    head = create_vision_head("keypoint_detection", num_classes=65,
                              hidden_dim=HIDDEN_DIM)
    out = head(np.zeros(SPATIAL, dtype="float32"))
    assert sorted(out) == ["classifications"]
    assert tuple(out["classifications"].shape) == (2, 6, 10, 65)
    assert head.num_anchors == 1 and head.use_regression is False


def test_the_output_specifications_of_a_configuration_follow_the_table() -> None:
    config = TaskConfiguration([VisionTaskType.DETECTION, VisionTaskType.OPTICAL_FLOW])
    specs = config.get_output_specifications()
    assert set(specs[VisionTaskType.OPTICAL_FLOW]) == {"flow"}
    assert set(specs[VisionTaskType.DETECTION]) == {"classifications", "regressions"}
