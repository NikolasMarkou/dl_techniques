"""Public API of the vision heads sub-package.

Eighteen names are exported, from two modules. ``vision/factory.py``
provides the ten head layers, three factory functions
(``create_vision_head``, ``create_multi_task_head``,
``create_enhancement_head``) and ``HeadConfiguration``.
``vision/task_types.py`` provides ``VisionTaskType``,
``TaskConfiguration``, ``CommonTaskConfigurations`` and ``parse_task_list``.

``BaseVisionHead`` is exported but is not a usable head. It defines no
``call``. It builds ``attention`` when ``use_attention`` is set and ``ffn``
when ``use_ffn`` is set, and nothing else. Each head normalizes and drops
inside its own ``ConvBlock`` or ``DenseBlock``. Use one of the nine task
heads instead.
``get_task_suggestions`` and ``validate_task_combination`` are NOT exported;
import them from ``.task_types`` when you need them. Nor are
``VISION_HEAD_DISPATCH``, ``merge_head_config_overrides`` and
``build_model_vision_head`` (the one-call head builder every vision model
uses); import them from ``.factory``.

Import from this package rather than from the submodule::

    from dl_techniques.layers.heads.vision import create_vision_head

The wider facade ``dl_techniques.layers.heads.create_head('vision', ...)``
calls ``create_vision_head`` for you.

``create_vision_head`` serves 10 of the 37 ``VisionTaskType`` members and
raises ``ValueError`` for the other 27. The table in ``VisionTaskType`` lists
both groups.
"""

from .factory import (
    BaseVisionHead,
    DetectionHead,
    SegmentationHead,
    DepthEstimationHead,
    SurfaceNormalsHead,
    OpticalFlowHead,
    ClassificationHead,
    InstanceSegmentationHead,
    MultiTaskHead,
    EnhancementHead,
    create_vision_head,
    create_multi_task_head,
    create_enhancement_head,
    HeadConfiguration,
)
from .task_types import (
    VisionTaskType,
    TaskConfiguration,
    CommonTaskConfigurations,
    parse_task_list,
)

__all__ = [
    "BaseVisionHead",
    "DetectionHead",
    "SegmentationHead",
    "DepthEstimationHead",
    "SurfaceNormalsHead",
    "OpticalFlowHead",
    "ClassificationHead",
    "InstanceSegmentationHead",
    "MultiTaskHead",
    "EnhancementHead",
    "create_vision_head",
    "create_multi_task_head",
    "create_enhancement_head",
    "HeadConfiguration",
    "VisionTaskType",
    "TaskConfiguration",
    "CommonTaskConfigurations",
    "parse_task_list",
]
