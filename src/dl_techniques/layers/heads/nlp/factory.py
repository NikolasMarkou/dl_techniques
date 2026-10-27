"""Factory functions for the NLP heads.

``create_nlp_head`` builds one head from a task config; ``create_multi_task_nlp_head``
builds a ``MultiTaskNLPHead``. The classes live in their own modules (``base``,
``text_classification``, ...) and the table from task type to class in ``dispatch``."""

from typing import Dict, Sequence, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import key_by_name
from .task_types import NLPTaskConfig
from .base import BaseNLPHead
from .dispatch import get_head_class
from .multitask import MultiTaskNLPHead


def create_nlp_head(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        input_dim: int,
        **kwargs: Any
) -> BaseNLPHead:
    """
    Build one NLP head from a task config.

    The task type in the config picks the class through
    :func:`get_head_class`. A plain dict is converted to an
    :class:`NLPTaskConfig` first, and may give ``task_type`` as its string
    value. Extra keyword arguments go to the head's constructor unchanged, so
    anything :class:`BaseNLPHead` accepts can be set here.

    **Architecture Overview:**

    .. code-block:: text

        task_config (NLPTaskConfig or dict)
                │
                ▼
        ┌──────────────────────────┐
        │ NLPTaskConfig.coerce     │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ get_head_class(task_type)│
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ head_class(task_config,  │
        │   input_dim, **kwargs)   │
        └──────────────────────────┘

    :param task_config: An :class:`NLPTaskConfig`, or a dict of its fields.
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
    :param input_dim: Width of the features the foundation model emits.
    :type input_dim: int
    :param kwargs: Extra keyword arguments for the head constructor.
    :return: A configured head for that task.
    :rtype: BaseNLPHead
    :raises ValueError: If the task type has no implemented head, or a field
        or argument is invalid.
    :raises TypeError: If ``task_config`` is neither form.
    """
    task_config = NLPTaskConfig.coerce(task_config)
    head_class = get_head_class(task_config.task_type)
    return head_class(task_config=task_config, input_dim=input_dim, **kwargs)

def create_multi_task_nlp_head(
        task_configs: Union[Sequence[Union[NLPTaskConfig, Dict[str, Any]]],
                            Dict[str, Union[NLPTaskConfig, Dict[str, Any]]]],
        input_dim: int,
        **kwargs: Any
) -> MultiTaskNLPHead:
    """
    Build a :class:`MultiTaskNLPHead` from several task configs.

    A list or a tuple (the same semantics) is turned into a dict keyed by each
    config's ``name``; two configs with one name are refused, because keying
    would keep only the last. Pass a dict directly to choose the keys
    yourself. Either form may hold dicts of config fields. An empty sequence
    or dict is refused, and so is any other type.

    :param task_configs: A list or tuple of configs, or a dict of task name
        to config.
    :type task_configs: Union[Sequence[...], Dict[str, ...]]
    :param input_dim: Width of the shared features. Becomes the head's
        ``shared_input_dim``.
    :type input_dim: int
    :param kwargs: Extra keyword arguments for :class:`MultiTaskNLPHead`.
    :return: The configured multi-task head.
    :rtype: MultiTaskNLPHead
    :raises ValueError: If ``task_configs`` is not a list, tuple or dict, the
        sequence repeats a name, the set is empty, or any task type has no
        implemented head.
    """
    if isinstance(task_configs, (list, tuple)):
        task_configs = [NLPTaskConfig.coerce(c) for c in task_configs]
    task_configs = key_by_name(
        task_configs, lambda c: c.name, 'create_multi_task_nlp_head')

    return MultiTaskNLPHead(
        task_configs=task_configs,
        shared_input_dim=input_dim,
        **kwargs
    )
