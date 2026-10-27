"""Vision head dispatch table and factory functions.

`VISION_HEAD_DISPATCH` is the one table from task type to head class and
per-task defaults, read by every helper below. `create_vision_head`, 
`create_enhancement_head` build heads through it.

`create_multi_task_head` lives in ``multitask.py`` because it imports
`MultiTaskHead`, which itself imports `create_vision_head` from here —
putting both in one module would create an import cycle.
"""

from typing import Any, Dict, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import lookup_head
from .base import BaseVisionHead
from .classification import ClassificationHead
from .detection import DetectionHead
from .depth import DepthEstimationHead
from .dense_vector import SurfaceNormalsHead, OpticalFlowHead
from .enhancement import EnhancementHead
from .instance_segmentation import InstanceSegmentationHead
from .segmentation import SegmentationHead
from .task_types import VisionTaskType


#: The one dispatch table of the vision heads: each task type that has a
#: head, mapped to ``(head class, per-task defaults)``. A default yields to a
#: caller's value for the same key. :func:`create_vision_head`,
#: :class:`MultiTaskHead`, :func:`create_enhancement_head` and
#: :class:`HeadConfiguration` all read it, so they agree on which task types
#: exist; :meth:`VisionTaskType.get_output_types` documents each one's keys
#: (pinned against the heads by a test, since ``task_types.py`` cannot
#: import this module).
VISION_HEAD_DISPATCH: Dict[
    VisionTaskType, Tuple[type, Dict[str, Any]]] = {
    VisionTaskType.DETECTION: (DetectionHead, {}),
    VisionTaskType.SEGMENTATION: (SegmentationHead, {}),
    VisionTaskType.INSTANCE_SEGMENTATION: (InstanceSegmentationHead, {}),
    VisionTaskType.CLASSIFICATION: (ClassificationHead, {}),
    VisionTaskType.DEPTH_ESTIMATION: (DepthEstimationHead, {}),
    # DECISION plan-2026-10-14T062616-81938e81/D-051
    # Normals and flow have heads of their own. Do NOT route them back
    # through DepthEstimationHead(output_channels=3/2): its sigmoid range map
    # squeezed every component into [0.1, 100], so no normal was unit length
    # and no flow was negative. See decisions.md D-051.
    VisionTaskType.SURFACE_NORMALS: (SurfaceNormalsHead, {}),
    VisionTaskType.OPTICAL_FLOW: (OpticalFlowHead, {}),
    # DECISION plan-2026-10-14T062616-81938e81/D-052
    # A keypoint head is a score map: one logit vector per location and no
    # box regression (SuperPoint's detector: 64 cell positions + dustbin).
    # Do NOT fall back to the DetectionHead defaults (9 anchors, 4-dim box
    # regressions): that is a box detector under a keypoint name, and
    # get_output_types documented keys it never returned. See decisions.md
    # D-052.
    VisionTaskType.KEYPOINT_DETECTION: (
        DetectionHead, {'num_anchors': 1, 'use_regression': False}),
    VisionTaskType.DENOISING: (EnhancementHead, {}),
    VisionTaskType.SUPER_RESOLUTION: (EnhancementHead, {'scale_factor': 2}),
}

#: The two task types :func:`create_enhancement_head` serves.
ENHANCEMENT_TASK_TYPES: Tuple[VisionTaskType, ...] = (
    VisionTaskType.DENOISING, VisionTaskType.SUPER_RESOLUTION)


def create_vision_head(
        task_type: Union[VisionTaskType, str],
        **kwargs: Any
) -> BaseVisionHead:
    """
    Build the vision head that serves a task type.

    Ten of the 37 ``VisionTaskType`` members reach a head, through
    :data:`VISION_HEAD_DISPATCH`. The other 27 raise ``ValueError``. There is
    no fallback head and never has been: a task with no implementation fails
    loudly rather than returning a plausible wrong one.

    Seven members get a class of their own. Three reuse another task's class
    with different defaults; a caller's value for a defaulted key wins.

    **Architecture Overview:**

    .. code-block:: text

        task_type: VisionTaskType, or a str that from_string parses
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          one of these 10     the other 27
                 │                 │
                 ▼                 ▼
           head instance     ValueError raised

        DETECTION              DetectionHead
        SEGMENTATION           SegmentationHead
        INSTANCE_SEGMENTATION  InstanceSegmentationHead
        CLASSIFICATION         ClassificationHead
        DEPTH_ESTIMATION       DepthEstimationHead
        SURFACE_NORMALS        SurfaceNormalsHead
        OPTICAL_FLOW           OpticalFlowHead
        KEYPOINT_DETECTION     DetectionHead, num_anchors 1, no regression
        DENOISING              EnhancementHead
        SUPER_RESOLUTION       EnhancementHead, scale_factor 2

    The ``VisionTaskType`` docstring in ``vision/task_types.py`` names all 27
    members that raise. :data:`VISION_HEAD_DISPATCH` is the arbiter if the
    two disagree.

    :param task_type: Task the head is for, as an enum member or its string
        value.
    :type task_type: Union[VisionTaskType, str]
    :param kwargs: Forwarded to the chosen head class, over the task's
        defaults.
    :return: Configured vision head.
    :rtype: BaseVisionHead
    :raises ValueError: If the task type is one of the other 27.
    """
    _, (head_class, defaults) = lookup_head(
        task_type, VISION_HEAD_DISPATCH, VisionTaskType, 'vision')
    return head_class(**{**defaults, **kwargs})


def create_enhancement_head(
        task_type: Union[VisionTaskType, str],
        **kwargs: Any
) -> EnhancementHead:
    """
    Build an :class:`EnhancementHead` for a restoration task.

    Accepts the two enhancement task types, as enum members or their string
    values; super-resolution gets ``scale_factor=2`` unless the caller passed
    one, denoising the head's own same-resolution default. Every other task
    type raises: this function builds enhancement heads only.

    The class itself lives at module scope. This function does not define it.

    :param task_type: ``VisionTaskType.DENOISING`` or
        ``VisionTaskType.SUPER_RESOLUTION``, or its string value.
    :type task_type: Union[VisionTaskType, str]
    :param kwargs: Forwarded to :class:`EnhancementHead`.
    :return: Configured enhancement head.
    :rtype: EnhancementHead
    :raises ValueError: If the task type is not an enhancement task.
    """
    task_type = VisionTaskType.coerce(task_type)
    if task_type not in ENHANCEMENT_TASK_TYPES:
        raise ValueError(
            f"create_enhancement_head builds "
            f"{[t.value for t in ENHANCEMENT_TASK_TYPES]} only, got "
            f"{task_type.value!r}; use create_vision_head for other tasks."
        )
    return create_vision_head(task_type, **kwargs)
