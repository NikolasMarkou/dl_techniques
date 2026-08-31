"""Public API of the NLP heads sub-package.

Everything here is re-exported from ``nlp/factory.py``. Two of the names,
``NLPTaskType`` and ``NLPTaskConfig``, are defined in ``nlp/task_types.py``
and pass through ``factory.py`` on the way out.

Eleven names are exported: the two task-description types, the two factory
functions ``create_nlp_head`` and ``create_multi_task_nlp_head``, and seven
head layers. ``BaseNLPHead``, ``get_head_class`` and ``NLPHeadConfiguration``
are NOT exported; import them from ``.factory`` when you need them.

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
]