"""Task-head layers, grouped by domain (``nlp``, ``vision``, ``vlm``, ``time_series``).

This package merges the former ``nlp_heads``, ``vision_heads`` and
``vlm_heads`` packages into one import surface. It re-exports the full public
API of all four domain sub-packages except the eight NLP in-model helpers
(``TASK_HEAD_NAME``, ``build_model_task_head`` and the rest), which a language
model imports from :mod:`~dl_techniques.layers.heads.nlp` itself. It also
exports :func:`create_head`, a dispatcher that routes to the per-domain
single-head factory.

Domains
-------
- :mod:`~dl_techniques.layers.heads.nlp` — NLP task heads: classification,
  token classification, question answering, multiple choice, generation,
  similarity and multi-task. Sequence pooling for the ``cls``, ``mean``,
  ``max`` and ``last`` strategies reuses the shared ``SequencePooling`` layer.
  The learnable ``attention`` strategy stays inline in ``nlp/base.py``,
  because it uses a different mechanism and a different weight set.
- :mod:`~dl_techniques.layers.heads.vision` — vision task heads: detection,
  segmentation, depth, classification, instance segmentation, enhancement and
  multi-task, plus the task-type vocabulary.
- :mod:`~dl_techniques.layers.heads.vlm` — vision-language heads: captioning,
  VQA, visual grounding, image-text matching and multi-task.
- :mod:`~dl_techniques.layers.heads.time_series` — forecasting heads: fixed and
  variable horizon quantile heads, Gaussian and negative-binomial likelihood
  heads, and the Timestral quantile head. No factory: the forecasting models
  build these directly.

Facade
------
- :func:`create_head` — ``create_head(domain, *args, **kwargs)`` calls the
  domain's own ``create_*_head`` factory. It forwards every argument unchanged
  and does not unify the three signatures.

Task-type vocabulary
--------------------
:mod:`dl_techniques.layers.heads.task_types` gives one import surface over
every task-type enum and config class.

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
    BaseNLPHead,
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
    VisionTaskType,
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
# Time-series heads
# =========================================================================
from .time_series import (
    QuantileHead,
    QuantileSequenceHead,
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
    TimestralQuantileHead,
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
    "BaseNLPHead",
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
    # --- Time series ---
    "QuantileHead",
    "QuantileSequenceHead",
    "GaussianLikelihoodHead",
    "NegativeBinomialLikelihoodHead",
    "TimestralQuantileHead",
    # --- Facade ---
    "create_head",
]
