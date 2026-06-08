"""Aggregated task-type vocabulary for the merged ``heads`` package.

This module re-exports the task-type enums plus their configuration / helper
symbols from the three domain sub-packages (``nlp``, ``vision``, ``vlm``) so
that callers have a single import surface for the full head task-type
vocabulary::

    from dl_techniques.layers.heads.task_types import (
        NLPTaskType, VisionTaskType, VLMTaskType,
    )

The domain enums are namespaced (``NLPTaskType`` / ``VisionTaskType`` /
``VLMTaskType``) and globally unique. The generic ``TaskType`` alias is the
vision enum (``TaskType is VisionTaskType``), kept for back-compat (see D-003).
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
    "TaskType",  # back-compat alias for VisionTaskType (D-003)
    "TaskConfiguration",
    "CommonTaskConfigurations",
    "parse_task_list",
    # --- VLM ---
    "VLMTaskType",
    "VLMTaskConfig",
    "VLMTaskConfiguration",
]
