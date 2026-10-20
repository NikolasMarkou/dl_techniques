"""YOLOv12 refuses a task it has no head for, in the model and in the loss.

``YOLOv12MultiTask`` and ``YOLOv12MultiTaskLoss`` take any ``VisionTaskType``
through ``parse_task_list``, but only detection, segmentation and
classification have a YOLOv12 head and a YOLOv12 sub-loss. Measured at
d520ec1dc over the 34 other members: mixed with DETECTION, the model built a
detection-only network and the loss a detection-only loss, the extra task
dropped in silence (34 of 34 each); alone, the model raised Keras' opaque
"`outputs` argument cannot be empty" and the loss built an empty loss set
(34 of 34 each).

The supported set is written out here, not imported, so the module constant
is checked against an independent oracle.
"""

from __future__ import annotations

import re

import pytest

from dl_techniques.layers.heads.vision.task_types import VisionTaskType as T
from dl_techniques.losses.yolo12_multitask_loss import YOLOv12MultiTaskLoss
from dl_techniques.models.vision.yolo12.multitask import YOLOv12MultiTask

SUPPORTED = {T.DETECTION, T.SEGMENTATION, T.CLASSIFICATION}
UNSUPPORTED = [t for t in T if t not in SUPPORTED]


def test_the_unsupported_population_is_not_empty():
    # Anti-vacuity: 37 members today, 3 with a YOLOv12 head.
    assert len(UNSUPPORTED) >= 30, len(UNSUPPORTED)


def test_the_module_constant_names_exactly_the_three_heads():
    from dl_techniques.layers.yolo12.yolo12_heads import YOLOV12_HEAD_TASKS

    assert set(YOLOV12_HEAD_TASKS) == SUPPORTED


def _build_model(tasks):
    return YOLOv12MultiTask(
        num_classes=3, task_config=tasks, input_shape=(64, 64, 3), scale="n",
    )


def _build_loss(tasks):
    return YOLOv12MultiTaskLoss(tasks=tasks, num_classes=3, input_shape=(64, 64))


@pytest.mark.parametrize("build", [_build_model, _build_loss], ids=["model", "loss"])
@pytest.mark.parametrize("mixed", [False, True], ids=["alone", "mixed_with_detection"])
@pytest.mark.parametrize("task", UNSUPPORTED, ids=[t.value for t in UNSUPPORTED])
def test_an_unsupported_task_is_refused_by_name(build, mixed, task):
    tasks = [T.DETECTION, task] if mixed else task
    with pytest.raises(ValueError, match=re.escape(task.value)) as info:
        build(tasks)
    assert "create_vision_head" in str(info.value)


@pytest.mark.parametrize("build", [_build_model, _build_loss], ids=["model", "loss"])
def test_the_three_supported_tasks_still_build(build):
    # Liveness: the refusal must not catch the tasks YOLOv12 does build.
    built = build([T.DETECTION, T.SEGMENTATION, T.CLASSIFICATION])
    assert built is not None
