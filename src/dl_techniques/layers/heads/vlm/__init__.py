"""Public API of the VLM heads sub-package.

The heads take feature tensors, not images or token ids. Nothing here knows
which foundation model produced them, so the same head works on CLIP, BLIP or
Flamingo features. Every feature width comes from the input shapes handed to
``build``; no head takes a declared ``vision_dim`` or ``text_dim``, and each
head owns exactly the sub-layers its ``call`` runs.

Eleven names are exported. Each head layer lives in its own module
(``base``, ``image_captioning``, ``vqa``, ``visual_grounding``,
``image_text_matching``, ``multitask``); ``factory.py`` holds the two
factory functions (``create_vlm_head``, ``create_multi_task_vlm_head``),
``dispatch.py`` the task-type table and ``get_head_class``, and ``utils.py``
the shared private helpers. ``task_types.py`` provides ``VLMTaskType``,
``VLMTaskConfig`` and ``VLMTaskConfiguration``.

Classes
-------
* ``BaseVLMHead`` -- shared construction for the fusion stack. Two heads
  inherit from it. It has no ``call`` and refuses to be constructed on its own.
* ``ImageCaptioningHead`` -- a transformer decoder over text, cross-attending
  to vision. Does NOT inherit from ``BaseVLMHead``.
* ``VQAHead`` -- pools each modality, concatenates, classifies. Does NOT
  inherit from ``BaseVLMHead`` either.
* ``VisualGroundingHead`` -- scores image regions against a text query and
  regresses a box from the top-scoring one.
* ``ImageTextMatchingHead`` -- a contrastive branch and a fine-grained fusion
  branch, run in parallel.
* ``MultiTaskVLMHead`` -- several heads behind one layer.

Padding masks ride in the input dict next to the features they describe:
``vision_mask`` for ``vision_features``, ``text_mask`` for ``text_features``
and ``question_mask`` for ``question_features``. Each is a ``(batch, length)``
keep predicate, 1 for a real token and 0 for padding, the convention every
attention layer in the package uses. A head given no mask treats every
position as real.

Example::

    >>> from dl_techniques.layers.heads.vlm import (
    ...     create_vlm_head, VLMTaskConfig, VLMTaskType)
    >>> head = create_vlm_head(VLMTaskConfig(
    ...     name='vqa', task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
    ...     num_classes=100))
    >>> out = head({'vision_features': v, 'question_features': q})
    >>> sorted(out)
    ['answer_logits']

``BaseVLMHead`` is exported but is not a usable head. It defines no ``call``,
only the shared fusion layer its two subclasses build on, and constructing it
directly raises ``TypeError``. Use one of the four task heads instead.

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

No model or trainer in the repository constructs a VLM head today.
"""

from .task_types import VLMTaskType, VLMTaskConfig, VLMTaskConfiguration
from .factory import create_vlm_head, create_multi_task_vlm_head
from .base import BaseVLMHead
from .image_captioning import ImageCaptioningHead
from .vqa import VQAHead
from .visual_grounding import VisualGroundingHead
from .image_text_matching import ImageTextMatchingHead
from .multitask import MultiTaskVLMHead

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
