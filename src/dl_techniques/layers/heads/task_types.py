"""One import surface for every ``heads`` task-type name.

This module re-exports the task-type enums, their config classes and their
helpers. They come from the four domain sub-packages: ``nlp``, ``vision``,
``vision_language`` and ``time_series``. Import them from one place::

    from dl_techniques.layers.heads.task_types import (
        NLPTaskType, VisionTaskType, VLMTaskType, TimeSeriesTaskType,
    )

There is nothing here but imports and ``__all__``. Each name keeps the
behaviour it has in its own sub-package.

The four domain enums have distinct names (``NLPTaskType``,
``VisionTaskType``, ``VLMTaskType``, ``TimeSeriesTaskType``), so no name
collides. None of them has an alias.
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
from .vision_language.task_types import (
    VLMTaskType,
    VLMTaskConfig,
    VLMTaskConfiguration,
)
from .time_series.task_types import TimeSeriesTaskType

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
    # --- Time series ---
    "TimeSeriesTaskType",
]
