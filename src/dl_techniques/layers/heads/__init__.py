"""Task-head layers, organized by domain (``nlp``, ``vision``, ``vlm``).

This package consolidates the formerly-separate ``nlp_heads``, ``vision_heads``,
and ``vlm_heads`` packages into a single import surface. It re-exports the full
public API of each domain sub-package, plus a :func:`create_head` dispatch
facade that routes to the per-domain single-head factories.

Domains
-------
- :mod:`~dl_techniques.layers.heads.nlp` — NLP task heads (classification, token
  classification, QA, multiple-choice, generation, similarity, multi-task). NLP
  sequence pooling reuses the shared ``SequencePooling`` layer for cls/mean/max
  (the learnable ``attention`` strategy stays inline — see D-002).
- :mod:`~dl_techniques.layers.heads.vision` — vision task heads (detection,
  segmentation, depth, classification, instance segmentation, enhancement,
  multi-task) + the task-type vocabulary.
- :mod:`~dl_techniques.layers.heads.vlm` — vision-language model heads
  (captioning, VQA, visual grounding, image-text matching, multi-task).

Facade
------
- :func:`create_head` — ``create_head(domain, *args, **kwargs)`` dispatches to
  the domain's native ``create_*_head`` factory (thin shim, no signature
  unification — see D-004).

Task-type vocabulary
--------------------
For a single import surface over all task-type enums/configs, use
:mod:`dl_techniques.layers.heads.task_types`.

Example
-------
>>> from dl_techniques.layers.heads import create_head, VisionTaskType
>>> head = create_head('vision', VisionTaskType.CLASSIFICATION, num_classes=10)
"""

# =========================================================================
# NLP heads
# =========================================================================
from .nlp import (
    NLPTaskType,
    NLPTaskConfig,
    create_nlp_head,
    create_multi_task_nlp_head,
    QuestionAnsweringHead,
    MultipleChoiceHead,
    MultiTaskNLPHead,
    TextClassificationHead,
    TokenClassificationHead,
    TextGenerationHead,
    TextSimilarityHead,
)

# =========================================================================
# Vision heads
# =========================================================================
from .vision import (
    BaseVisionHead,
    DetectionHead,
    SegmentationHead,
    DepthEstimationHead,
    ClassificationHead,
    InstanceSegmentationHead,
    MultiTaskHead,
    EnhancementHead,
    create_vision_head,
    create_multi_task_head,
    create_enhancement_head,
    HeadConfiguration,
    VisionTaskType,
    TaskType,  # back-compat alias for VisionTaskType (D-003)
    TaskConfiguration,
    CommonTaskConfigurations,
    parse_task_list,
)

# =========================================================================
# VLM heads
# =========================================================================
from .vlm import (
    VLMTaskType,
    VLMTaskConfig,
    VLMTaskConfiguration,
    create_vlm_head,
    create_multi_task_vlm_head,
    BaseVLMHead,
    ImageCaptioningHead,
    VQAHead,
    VisualGroundingHead,
    ImageTextMatchingHead,
    MultiTaskVLMHead,
)

# =========================================================================
# Dispatch facade
# =========================================================================
from .factory import create_head

__all__ = [
    # --- NLP ---
    "NLPTaskType",
    "NLPTaskConfig",
    "create_nlp_head",
    "create_multi_task_nlp_head",
    "QuestionAnsweringHead",
    "MultipleChoiceHead",
    "MultiTaskNLPHead",
    "TextClassificationHead",
    "TokenClassificationHead",
    "TextGenerationHead",
    "TextSimilarityHead",
    # --- Vision ---
    "BaseVisionHead",
    "DetectionHead",
    "SegmentationHead",
    "DepthEstimationHead",
    "ClassificationHead",
    "InstanceSegmentationHead",
    "MultiTaskHead",
    "EnhancementHead",
    "create_vision_head",
    "create_multi_task_head",
    "create_enhancement_head",
    "HeadConfiguration",
    "VisionTaskType",
    "TaskType",
    "TaskConfiguration",
    "CommonTaskConfigurations",
    "parse_task_list",
    # --- VLM ---
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
    # --- Facade ---
    "create_head",
]
