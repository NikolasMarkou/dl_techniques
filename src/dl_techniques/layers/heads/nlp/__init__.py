"""Public API of the NLP heads sub-package.

Twelve of the names are re-exported from ``nlp/factory.py``. Two of those,
``NLPTaskType`` and ``NLPTaskConfig``, are defined in ``nlp/task_types.py``
and pass through ``factory.py`` on the way out. The other five come from
``nlp/model_head.py``.

Seventeen names are exported: the two task-description types, the two factory
functions ``create_nlp_head`` and ``create_multi_task_nlp_head``, eight head
layers (``BaseNLPHead`` included, as ``BaseVisionHead`` and ``BaseVLMHead``
are exported by their domains), and the five in-model head helpers
``validate_task_head_knobs``, ``build_model_task_head``,
``task_config_to_dict``, ``finalize_head_outputs`` and
``head_attention_mask``. ``get_head_class`` is NOT exported; import it from
``.factory``, and the dispatch table ``NLP_HEAD_DISPATCH`` from
``.task_types``, when you need them.

Import from this package rather than from the submodule::

    from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskType

The wider facade ``dl_techniques.layers.heads.create_head('nlp', ...)`` calls
``create_nlp_head`` for you.
"""

from .factory import (
    NLPTaskType, NLPTaskConfig, create_nlp_head, create_multi_task_nlp_head,
    BaseNLPHead, QuestionAnsweringHead, MultipleChoiceHead, MultiTaskNLPHead,
    TextClassificationHead, TokenClassificationHead, TextGenerationHead, TextSimilarityHead
)
from .model_head import (
    validate_task_head_knobs, build_model_task_head, task_config_to_dict,
    finalize_head_outputs, head_attention_mask,
)

__all__ = [
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
    "validate_task_head_knobs",
    "build_model_task_head",
    "task_config_to_dict",
    "finalize_head_outputs",
    "head_attention_mask",
]