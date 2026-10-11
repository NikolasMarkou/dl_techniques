"""Public API of the NLP heads sub-package.

Eleven of the names are re-exported from ``nlp/factory.py``. Two of those,
``NLPTaskType`` and ``NLPTaskConfig``, are defined in ``nlp/task_types.py``
and pass through ``factory.py`` on the way out. The other three come from
``nlp/model_head.py``.

Fourteen names are exported: the two task-description types, the two factory
functions ``create_nlp_head`` and ``create_multi_task_nlp_head``, seven head
layers, and the three in-model head helpers ``build_model_task_head``,
``task_config_to_dict`` and ``finalize_head_outputs``. ``BaseNLPHead``,
``get_head_class`` and ``NLPHeadConfiguration`` are NOT exported; import them
from ``.factory`` when you need them.

Import from this package rather than from the submodule::

    from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskType

The wider facade ``dl_techniques.layers.heads.create_head('nlp', ...)`` calls
``create_nlp_head`` for you.
"""

from .factory import (
    NLPTaskType, NLPTaskConfig, create_nlp_head, create_multi_task_nlp_head,
    QuestionAnsweringHead, MultipleChoiceHead, MultiTaskNLPHead,
    TextClassificationHead, TokenClassificationHead, TextGenerationHead, TextSimilarityHead
)
from .model_head import (
    build_model_task_head, task_config_to_dict, finalize_head_outputs
)

__all__ = [
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
    "build_model_task_head",
    "task_config_to_dict",
    "finalize_head_outputs",
]