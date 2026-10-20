"""Public API of the NLP heads sub-package.

Twelve of the names are re-exported from ``nlp/factory.py``. Two of those,
``NLPTaskType`` and ``NLPTaskConfig``, are defined in ``nlp/task_types.py``
and pass through ``factory.py`` on the way out. The other eight come from
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
``get_head_class`` is NOT exported; import it from ``.factory``, and the
dispatch table ``NLP_HEAD_DISPATCH`` from ``.task_types``, when you need them.

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