"""Task type to head class."""

from typing import Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import lookup_head
from .task_types import NLPTaskType, NLP_HEAD_DISPATCH
from .text_classification import TextClassificationHead
from .token_classification import TokenClassificationHead
from .question_answering import QuestionAnsweringHead
from .text_similarity import TextSimilarityHead
from .text_generation import TextGenerationHead
from .multiple_choice import MultipleChoiceHead


#: Head class name, as :data:`NLP_HEAD_DISPATCH` writes it, to the class.
_HEAD_CLASSES: Dict[str, type] = {
    cls.__name__: cls for cls in (
        TextClassificationHead,
        TokenClassificationHead,
        QuestionAnsweringHead,
        TextSimilarityHead,
        TextGenerationHead,
        MultipleChoiceHead,
    )
}

def get_head_class(task_type: Union[NLPTaskType, str]) -> type:
    """
    Look up the head class for a task type.

    The class comes from :data:`~.task_types.NLP_HEAD_DISPATCH`, the one
    table from task type to head (D-064). It covers 24 of the 37 members of
    :class:`NLPTaskType`. **The other 13 raise ``ValueError``.** That is a
    guard, not a gap in the table. This function used to end with
    ``head_mapping.get(task_type, TextClassificationHead)``. Asking for
    ``MACHINE_TRANSLATION`` then returned a text classifier that built, trained
    and produced plausible nonsense. Nothing ever failed. A task with no head
    is a caller error and is now reported as one.

    To add support for one of the 13, implement or choose a head and add its
    row to the table. Do not restore the fallback.

    **Architecture Overview:**

    .. code-block:: text

        task_type (member or string value)
            │
            ▼
        ┌──────────────────────┐        ┌────────────┐
        │ in NLP_HEAD_DISPATCH?│─ no ──►│ ValueError │
        └──────────┬───────────┘        └────────────┘
                   │ yes
                   ▼
        ┌──────────────────────────────────────┐
        │ one of the six leaf head classes:    │
        │ TextClassificationHead  (10 tasks)   │
        │ TokenClassificationHead  (4 tasks)   │
        │ TextGenerationHead       (4 tasks)   │
        │ TextSimilarityHead       (3 tasks)   │
        │ QuestionAnsweringHead    (2 tasks)   │
        │ MultipleChoiceHead       (1 task)    │
        └──────────────────────────────────────┘

    :param task_type: The NLP task type, as a member or its string value.
    :type task_type: Union[NLPTaskType, str]
    :return: The head class for that task type.
    :rtype: type
    :raises ValueError: If no head is implemented for ``task_type``, or the
        string names no task type. The message lists every supported task
        type.
    """
    _, head_name = lookup_head(task_type, NLP_HEAD_DISPATCH, NLPTaskType, 'NLP')
    return _HEAD_CLASSES[head_name]
