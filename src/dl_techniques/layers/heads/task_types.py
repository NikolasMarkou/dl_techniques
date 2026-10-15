"""One import surface for every ``heads`` task-type name.

This module re-exports the task-type enums, their config classes and their
helpers. They come from the three domain sub-packages: ``nlp``, ``vision`` and
``vlm``. Import them from one place::

    from dl_techniques.layers.heads.task_types import (
        NLPTaskType, VisionTaskType, VLMTaskType,
    )

There is nothing here but imports and ``__all__``. Each name keeps the
behaviour it has in its own sub-package.

The three domain enums have distinct names (``NLPTaskType``,
``VisionTaskType``, ``VLMTaskType``), so no name collides. None of them has an
alias.
"""

from .nlp.task_types import (
    NLPTaskType,
    NLPTaskConfig,
)
from .vision.task_types import (
    VisionTaskType,
    TaskConfiguration,
    CommonTaskConfigurations,
    parse_task_list,
)
from .vlm.task_types import (
    VLMTaskType,
    VLMTaskConfig,
    VLMTaskConfiguration,
)

__all__ = [
    # --- NLP ---
    "NLPTaskType",
    "NLPTaskConfig",
    # --- Vision ---
    "VisionTaskType",
    "TaskConfiguration",
    "CommonTaskConfigurations",
    "parse_task_list",
    # --- VLM ---
    "VLMTaskType",
    "VLMTaskConfig",
    "VLMTaskConfiguration",
]
