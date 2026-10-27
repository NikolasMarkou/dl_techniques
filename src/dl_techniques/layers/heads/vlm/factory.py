"""Factory functions for the VLM heads.

``create_vlm_head`` builds one head from a ``VLMTaskConfig``; ``create_multi_task_vlm_head``
builds a ``MultiTaskVLMHead``. The classes live in their own modules and the table from task
type to class in ``dispatch``."""

import keras
from typing import Dict, Union, Any, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import key_by_name
from .task_types import VLMTaskConfig
from .dispatch import get_head_class
from .multitask import MultiTaskVLMHead


def create_vlm_head(
    task_config: Union[VLMTaskConfig, Dict[str, Any]], **kwargs: Any
) -> keras.layers.Layer:
    """
    Build one VLM head from a task configuration.

    The task type in ``task_config`` picks the class. An unsupported task type
    raises rather than returning a default head.

    :param task_config: VLMTaskConfig object, or a dict of its fields. A dict
        may give ``task_type`` as its string value, as ``VLMTaskConfig``
        itself accepts.
    :type task_config: Union[VLMTaskConfig, Dict[str, Any]]
    :param kwargs: Additional configuration for the head, such as
        ``fusion_strategy`` or ``num_layers``. Which keywords are accepted
        depends on the head class the task type selects; any other raises
        ``ValueError``.
    :return: A configured VLM head for the given task.
    :rtype: keras.layers.Layer
    :raises ValueError: If no head is implemented for the task type.
    """
    if isinstance(task_config, dict):
        task_config = VLMTaskConfig(**task_config)

    head_class = get_head_class(task_config.task_type)
    return head_class(task_config=task_config, **kwargs)


def create_multi_task_vlm_head(
    task_configs: Union[
        Sequence[Union[VLMTaskConfig, Dict[str, Any]]],
        Dict[str, Union[VLMTaskConfig, Dict[str, Any]]],
    ],
    **kwargs: Any,
) -> MultiTaskVLMHead:
    """
    Build a :class:`MultiTaskVLMHead` from task configurations.

    A list or a tuple (the same semantics) is keyed by each configuration's
    ``name``, and two configurations with one name are refused (keying would
    keep only the last). A dict is used as it is. Either form may hold
    ``VLMTaskConfig`` objects or dicts of their fields (``task_type`` may be
    its string value), as ``create_vlm_head`` accepts. An empty sequence or
    dict is refused, and so is any other type.

    :param task_configs: A list or tuple of configs, or a dict of task name to
        config; each config a ``VLMTaskConfig`` or a dict of its fields.
    :type task_configs: Union[Sequence[...], Dict[str, ...]]
    :param kwargs: Shared per-head configuration, plus this wrapper's own Keras
        base kwargs. No head takes a declared ``vision_dim`` / ``text_dim``,
        so passing either raises (no head accepts it).

        Kwarg routing is three-tiered:

        1. **Wrapper-owned** (``task_config``) -- supplied to every head from
           ``task_configs``; passing it raises.
        2. **``task_specific_kwargs``** -- a dict keyed by task name, applied to
           that head only. STRICT: an unknown task name raises.
        3. **Everything else** -- shared, and routed BEST-EFFORT to each head
           whose constructor accepts it. A key no head accepts raises, so a
           misspelling raises too. But a key that is real for SOME head, and
           merely meant for a DIFFERENT one, is applied wherever it fits. Only
           a ``logger.info`` records where it landed. Use
           ``task_specific_kwargs`` when you mean exactly one head.

        The heads also disagree on their input key. ``VQAHead`` reads
        ``question_features`` while the other four read ``text_features``, and
        this wrapper fans one input dict to all of them. A wrapper mixing VQA
        with any other task raises ``KeyError`` unless the caller supplies both
        keys. Duplicating the same tensor under both is fine.
    :return: Configured multi-task VLM head instance.
    :rtype: MultiTaskVLMHead
    :raises ValueError: If ``task_configs`` is not a list, tuple or dict, is
        empty, or a sequence repeats a name, or a keyword is wrapper-owned, is
        accepted by no head, or names a task absent from ``task_configs``.
    :raises TypeError: If an entry is neither a ``VLMTaskConfig`` nor a dict.
    """
    if isinstance(task_configs, (list, tuple)):
        task_configs = [VLMTaskConfig.coerce(c) for c in task_configs]
    task_configs = key_by_name(
        task_configs, lambda c: c.name, 'create_multi_task_vlm_head')

    return MultiTaskVLMHead(task_configs=task_configs, **kwargs)
