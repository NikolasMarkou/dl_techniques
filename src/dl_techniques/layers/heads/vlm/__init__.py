"""Public API of the VLM heads sub-package.

Eleven names are exported, from two modules. ``vlm/factory.py`` provides the
six head layers and two factory functions (``create_vlm_head``,
``create_multi_task_vlm_head``). ``vlm/task_types.py`` provides
``VLMTaskType``, ``VLMTaskConfig`` and ``VLMTaskConfiguration``.

``BaseVLMHead`` is exported but is not a usable head. It defines no ``call``,
only the shared fusion, normalization and optional feed-forward stages its
subclasses build on. ``create_vlm_head`` refuses to return it. Use one of the
four task heads instead.

Import from this package rather than from the submodule::

    from dl_techniques.layers.heads.vlm import create_vlm_head

The wider facade ``dl_techniques.layers.heads.create_head('vlm', ...)`` calls
``create_vlm_head`` for you.

``create_vlm_head`` serves 4 of the 47 ``VLMTaskType`` members:
``IMAGE_CAPTIONING``, ``VISUAL_QUESTION_ANSWERING``, ``VISUAL_GROUNDING`` and
``IMAGE_TEXT_MATCHING``. The other 43 raise ``ValueError``. The table in
``VLMTaskType`` lists them.

``MultiTaskVLMHead`` fans one input dict to every sub-head. ``VQAHead`` reads
``question_features`` while the other three read ``text_features``, so a
wrapper that mixes VQA with another task needs both keys in that dict.
"""

from .task_types import VLMTaskType, VLMTaskConfig, VLMTaskConfiguration
from .factory import (
    create_vlm_head, create_multi_task_vlm_head,
    BaseVLMHead, ImageCaptioningHead, VQAHead,
    VisualGroundingHead, ImageTextMatchingHead, MultiTaskVLMHead
)

__all__ = [
    "VLMTaskType",
    "VLMTaskConfig",
    "VLMTaskConfiguration",
    "create_vlm_head",
    "create_multi_task_vlm_head",
    "BaseVLMHead",
    "ImageCaptioningHead",
    "VQAHead",
    "VisualGroundingHead",
    "ImageTextMatchingHead",
    "MultiTaskVLMHead",
]
