"""Task type to head class."""

from typing import Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import lookup_head
from .task_types import VLMTaskType
from .image_captioning import ImageCaptioningHead
from .vqa import VQAHead
from .visual_grounding import VisualGroundingHead
from .image_text_matching import ImageTextMatchingHead


# DECISION plan-2026-10-14T062616-81938e81/D-059
# One table, read by get_head_class and therefore by create_vlm_head and
# MultiTaskVLMHead. Do NOT add a row that maps a task type to BaseVLMHead as a
# placeholder (DENSE_CAPTIONING and VISUAL_DIALOGUE were, only to be rejected
# a line later), and do NOT restore a default: a task type with no head
# raises. See decisions.md D-059 (and D-052, the vision table).
VLM_HEAD_DISPATCH: Dict[VLMTaskType, type] = {
    VLMTaskType.IMAGE_CAPTIONING: ImageCaptioningHead,
    VLMTaskType.VISUAL_QUESTION_ANSWERING: VQAHead,
    VLMTaskType.VISUAL_GROUNDING: VisualGroundingHead,
    VLMTaskType.IMAGE_TEXT_MATCHING: ImageTextMatchingHead,
}


def get_head_class(task_type: Union[VLMTaskType, str]) -> type:
    """
    Return the head class for a VLM task type, or raise.

    The class comes from :data:`VLM_HEAD_DISPATCH`. Four task types have a
    head: image captioning, visual question answering, visual grounding, and
    image-text matching. Every other member of ``VLMTaskType`` raises. There
    is no silent fallback, matching the vision and NLP dispatchers.

    To add support for a task type, implement its head and add it to the
    table. Do not restore a default.

    This used to be ``head_mapping.get(task_type, BaseVLMHead)``, so 41 of the
    47 members quietly returned ``BaseVLMHead``, which has no ``call()``: the
    object constructed fine and died with ``NotImplementedError``, naming the
    base class rather than the task, the moment it was used.

    :param task_type: The VLM task type, as a member or its string value.
    :type task_type: Union[VLMTaskType, str]
    :return: The head class for that task type.
    :rtype: type
    :raises ValueError: If no head is implemented for ``task_type``, or the
        string names no task type.
    """
    _, head_class = lookup_head(task_type, VLM_HEAD_DISPATCH, VLMTaskType, 'VLM')
    return head_class
