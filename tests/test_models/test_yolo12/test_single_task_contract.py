"""RED proof for C-7: a single-task YOLOv12 emits a tensor, not a dict.

`multitask.py:316` set `outputs = task_outputs` (a dict) unconditionally, while
`multitask.py:33-34` promised "…or single tensor for single tasks" and
`README.md:264` stated outright "The output will be a single tensor, not a
dictionary". Following README Example 1 therefore raised
`AttributeError: 'dict' object has no attribute 'shape'` on
`model.predict(images).shape`.

The package's wider coverage gap is real and out of scope here (6 tests total,
none touching segmentation or classification, none touching any scale but
"n"). What this module adds is exactly what the single-task contract needs to
be testable: one arm per task, and the multi-task control that must STAY a dict.
"""

from __future__ import annotations

import numpy as np
import pytest

from dl_techniques.models.yolo12.multitask import create_yolov12_multitask
from dl_techniques.layers.heads.vision.task_types import VisionTaskType

INPUT_SHAPE = (64, 64, 3)


def _images(n=2):
    return np.random.default_rng(0).random((n,) + INPUT_SHAPE).astype("float32")


class TestSingleTaskReturnsATensor:
    def test_readme_example_1_works_verbatim(self):
        model = create_yolov12_multitask(
            num_detection_classes=4, tasks="detection",
            input_shape=INPUT_SHAPE, scale="n",
        )
        out = model.predict(_images(), verbose=0)
        assert not isinstance(out, dict), type(out)
        assert out.shape[0] == 2

    @pytest.mark.parametrize("task,kwargs", [
        ("detection", dict(num_detection_classes=4)),
        ("segmentation", dict(num_segmentation_classes=3)),
        ("classification", dict(num_classification_classes=5)),
    ])
    def test_every_single_task_config_emits_one_tensor(self, task, kwargs):
        model = create_yolov12_multitask(
            tasks=task, input_shape=INPUT_SHAPE, scale="n", **kwargs,
        )
        out = model(_images(), training=False)
        assert not isinstance(out, dict), f"{task} returned {type(out)}"
        assert hasattr(out, "shape")
        assert np.all(np.isfinite(np.asarray(out)))

    def test_classification_head_width_follows_its_own_class_count(self):
        """Anti-vacuity for the parametrized arm: assert the tensor is the
        RIGHT one, not merely a tensor."""
        model = create_yolov12_multitask(
            tasks="classification", num_classification_classes=7,
            input_shape=INPUT_SHAPE, scale="n",
        )
        out = model(_images(), training=False)
        assert tuple(out.shape) == (2, 7), out.shape


class TestMultiTaskStaysADict:
    def test_two_tasks_still_return_a_named_dict(self):
        model = create_yolov12_multitask(
            num_detection_classes=4, num_segmentation_classes=3,
            tasks=[VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION],
            input_shape=INPUT_SHAPE, scale="n",
        )
        out = model(_images(), training=False)
        assert isinstance(out, dict), type(out)
        assert set(out) == {"detection", "segmentation"}
