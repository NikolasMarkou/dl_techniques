"""Public API of the NLP heads sub-package.

The task-description types come from ``nlp/task_types.py``, the two factory
functions from ``nlp/factory.py``, each head layer from its own module
(``base``, ``text_classification``, ``token_classification``,
``question_answering``, ``text_similarity``, ``text_generation``,
``multiple_choice``, ``multitask``) and the in-model helpers from
``nlp/model_head.py``.

Twenty names are exported: the two task-description types, the two factory
functions ``create_nlp_head`` and ``create_multi_task_nlp_head``, eight head
layers (``BaseNLPHead`` included, as ``BaseVisionHead`` and ``BaseVLMHead``
are exported by their domains), and the eight in-model head helpers
``TASK_HEAD_NAME``, ``validate_task_head_knobs``, ``build_model_task_head``,
``task_config_to_dict``, ``finalize_head_outputs``, ``head_attention_mask``,
``head_output_shape`` and ``ids_shape_of``. A model that owns a head imports
all of them from here, never from ``.model_head`` (pinned by
``tests/test_models/test_the_language_models_import_one_head_surface.py``).
``get_head_class`` is NOT exported; import it from ``.dispatch``, and the
dispatch table ``NLP_HEAD_DISPATCH`` from ``.task_types``, when you need them.

Import from this package rather than from the submodule::

    from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskType

The wider facade ``dl_techniques.layers.heads.create_head('nlp', ...)`` calls
``create_nlp_head`` for you.
"""

from .task_types import NLPTaskType, NLPTaskConfig
from .factory import create_nlp_head, create_multi_task_nlp_head
from .base import BaseNLPHead
from .text_classification import TextClassificationHead
from .token_classification import TokenClassificationHead
from .question_answering import QuestionAnsweringHead
from .text_similarity import TextSimilarityHead
from .text_generation import TextGenerationHead
from .multiple_choice import MultipleChoiceHead
from .multitask import MultiTaskNLPHead
from .model_head import (
    TASK_HEAD_NAME, validate_task_head_knobs, build_model_task_head,
    task_config_to_dict, finalize_head_outputs, head_attention_mask,
    head_output_shape, ids_shape_of,
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
    "TASK_HEAD_NAME",
    "head_output_shape",
    "ids_shape_of",
]