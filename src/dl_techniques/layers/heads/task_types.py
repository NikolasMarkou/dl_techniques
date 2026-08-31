"""One import surface for every ``heads`` task-type name.

This module re-exports the task-type enums, their config classes and their
helpers from the three domain sub-packages (``nlp``, ``vision``, ``vlm``)::

    from dl_techniques.layers.heads.task_types import (
        NLPTaskType, VisionTaskType, VLMTaskType,
    )

There is nothing here but imports and ``__all__``. Each name keeps the
behaviour it has in its own sub-package.

The three domain enums have distinct names (``NLPTaskType``,
``VisionTaskType``, ``VLMTaskType``), so no name collides. ``TaskType`` is a
back-compat alias for the vision enum: ``TaskType is VisionTaskType`` is
``True``. There is no matching alias for the NLP or VLM enum. The anchor that
records the alias is D-003 in
``dl_techniques/layers/heads/vision/task_types.py``.
"""

from .nlp.task_types import (
    NLPTaskType,
    NLPTaskConfig,
)
from .vision.task_types import (
    VisionTaskType,
    TaskType,
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
    # Back-compat alias: TaskType is VisionTaskType. The anchor is D-003 in
    # vision/task_types.py.
    "TaskType",
    "TaskConfiguration",
    "CommonTaskConfigurations",
    "parse_task_list",
    # --- VLM ---
    "VLMTaskType",
    "VLMTaskConfig",
    "VLMTaskConfiguration",
]
