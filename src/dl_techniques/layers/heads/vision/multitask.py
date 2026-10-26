"""Multi-task vision head: several heads behind one layer."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .dispatch import (
    _as_vision_task_type,
    VISION_HEAD_DISPATCH,
    _dispatched_task_list,
    create_vision_head,
)
from .task_types import TaskConfiguration, VisionTaskType


@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class MultiTaskHead(keras.layers.Layer):
    """
    Several vision heads behind one layer.

    Each entry in ``task_configs`` becomes one head. The heads are
    independent: they share the input, not any weights. This class does not
    inherit from :class:`BaseVisionHead`, so it builds no common layers of
    its own.

    A dict input is read per task name. A task name missing from the dict
    falls back to the ``'shared'`` entry; a task with neither raises
    ``ValueError`` naming it. A single tensor is sent to every head
    unchanged.

    Each head is built by :func:`create_vision_head`, so this layer serves
    exactly the ten task types that function dispatches, with the same
    per-task defaults, and refuses the other 27 naming the entry.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, H, W, C), or a dict keyed by task name
        with a 'shared' fallback for any name it omits
                                   │
                      ┌────────────┴────────────┐
                      ▼            ▼            ▼
                 ┌───────────┐┌───────────┐┌───────────┐
                 │ head[t1]  ││ head[t2]  ││ head[tN]  │
                 └─────┬─────┘└─────┬─────┘└─────┬─────┘
                       ▼            ▼            ▼
                  outputs[t1]  outputs[t2]  outputs[tN]

        Each head is what create_vision_head builds for its
        task_type: the ten dispatched task types, see that function.

        Any other task type raises ValueError naming the entry.

    Every head returns a dict, so ``call`` returns a dict of dicts: task name
    to that head's own output keys.

    Input shape:
        ``(batch, height, width, channels)``, or a dict of such shapes keyed
        by task name.

    Output shape:
        Dict mapping each task name to that head's output shapes.

    :param task_configs: Dict mapping a task name to that head's config. Each
        config carries a ``'task_type'`` key naming one of the ten dispatched
        task types, as a :class:`VisionTaskType` member or its string value.
        Keras' attribute tracking stores the outer dict and each config as
        copies, so a later change to the caller's dicts reaches neither the
        heads nor ``get_config``.
        ``get_config`` stores the string value, so the head survives a
        ``.keras`` round trip.
    :type task_configs: Dict[str, Dict[str, Any]]
    :param shared_backbone_dim: Default ``hidden_dim`` for any task config
        that does not set one.
    :type shared_backbone_dim: int
    :param use_task_specific_attention: Value forced onto every task head's
        ``use_attention``. A per-task ``use_attention`` is overwritten.
    :type use_task_specific_attention: bool
    :param kwargs: Additional arguments for the base Layer class.

    :ivar task_heads: Dict mapping each task name to its head instance. Each
        head is named ``'<task name>_head'`` unless its config sets ``name``.
    :vartype task_heads: Dict[str, keras.layers.Layer]

    :raises ValueError: From ``__init__``, when ``task_configs`` is empty;
        from ``_create_task_heads``, when a config names a
        task type with no head; from ``build``, ``call`` and
        ``compute_output_shape``, when a dict input has neither the task's
        name nor ``'shared'``.
    """

    def __init__(
            self,
            task_configs: Dict[str, Dict[str, Any]],
            shared_backbone_dim: int = 256,
            use_task_specific_attention: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration and build one head per task.

        :param task_configs: Dict mapping a task name to that head's config.
            Each config carries a ``'task_type'`` key.
        :type task_configs: Dict[str, Dict[str, Any]]
        :param shared_backbone_dim: Default ``hidden_dim`` for any task
            config that does not set one.
        :type shared_backbone_dim: int
        :param use_task_specific_attention: Value forced onto every task
            head's ``use_attention``.
        :type use_task_specific_attention: bool
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_configs`` is empty, or a config names a
            task type with no head.
        """
        if not task_configs:
            raise ValueError(
                "MultiTaskHead needs at least one task in task_configs; an "
                "empty wrapper returns {} for any input."
            )
        super().__init__(**kwargs)

        # Not an alias of the caller's dict: Keras' attribute tracking stores
        # a dict assigned to a layer as a TrackedDict copy, its nested dicts
        # too (measured 2026-10-15, so a review's aliasing claim was refuted;
        # pinned by test_the_multi_task_head_builds_every_dispatched_task.py).
        self.task_configs = task_configs
        self.shared_backbone_dim = shared_backbone_dim
        self.use_task_specific_attention = use_task_specific_attention

        self._create_task_heads()

    def _create_task_heads(self) -> None:
        """
        Build one head per entry in ``task_configs``.

        Each entry names a task type and carries that head's keyword
        arguments. ``hidden_dim`` defaults to ``shared_backbone_dim`` and
        ``use_attention`` is forced to ``use_task_specific_attention``, so a
        per-task ``use_attention`` in the caller dict is overwritten.

        Every head comes from :func:`create_vision_head`: the ten task types
        it dispatches are accepted, with its per-task defaults. Any other
        raises.

        :return: None.
        :rtype: None
        :raises ValueError: If a config names a task type with no head.
        """

        self.task_heads = {}

        for task_name, config in self.task_configs.items():
            # Copy before pop: popping 'task_type' from the stored config
            # stripped it as a side effect and broke get_config() round trips
            # and repeated construction.
            config = dict(config)
            task_type = _as_vision_task_type(config.pop('task_type'))
            if task_type not in VISION_HEAD_DISPATCH:
                raise ValueError(
                    f"MultiTaskHead task {task_name!r}: task type "
                    f"{task_type.value!r} has no vision head. "
                    f"{_dispatched_task_list()}"
                )

            # Add shared configuration
            config['hidden_dim'] = config.get('hidden_dim', self.shared_backbone_dim)
            config['use_attention'] = self.use_task_specific_attention
            # Named after the task, unless the caller names it: an unnamed
            # head takes a process-global Keras suffix ('detection_head_2'),
            # which a .keras reload re-enters, so every weight path under it
            # moved across a round trip.
            config.setdefault('name', f'{task_name}_head')

            self.task_heads[task_name] = create_vision_head(task_type, **config)

    @staticmethod
    def _task_input(inputs: Dict[str, Any], task_name: str) -> Any:
        """
        The entry of a dict input (or input shape) that serves one task.

        :param inputs: Dict keyed by task name, optionally with ``'shared'``.
        :type inputs: Dict[str, Any]
        :param task_name: The task whose entry is wanted.
        :type task_name: str
        :return: ``inputs[task_name]``, else ``inputs['shared']``.
        :rtype: Any
        :raises ValueError: If ``inputs`` has neither key.
        """
        if task_name in inputs:
            return inputs[task_name]
        if 'shared' in inputs:
            return inputs['shared']
        raise ValueError(
            f"MultiTaskHead got a dict input with no entry for task "
            f"{task_name!r} and no 'shared' entry; keys: {sorted(inputs)}"
        )

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build every task head on its own input shape.

        A dict ``input_shape`` gives per-task shapes, falling back to its
        ``'shared'`` entry. A single shape is used for every task.

        :param input_shape: One shared feature-map shape, or a dict of
            per-task shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        :return: None.
        :rtype: None
        """
        for task_name, task_head in self.task_heads.items():
            if isinstance(input_shape, dict):
                task_input_shape = self._task_input(input_shape, task_name)
            else:
                task_input_shape = input_shape
            task_head.build(task_input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, Dict[str, keras.KerasTensor]]:
        """
        Run every task head and collect the outputs.

        A dict input is read per task name, falling back to its ``'shared'``
        entry. A single tensor is sent to every head unchanged.

        :param inputs: One feature map, or a dict of per-task feature maps.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Keras training flag, forwarded to each head.
        :type training: Optional[bool]
        :return: Dict mapping each task name to that head's output dict.
        :rtype: Dict[str, Dict[str, keras.KerasTensor]]
        """

        outputs = {}

        # Handle different input formats
        if isinstance(inputs, dict):
            # Task-specific inputs
            for task_name, task_head in self.task_heads.items():
                task_input = self._task_input(inputs, task_name)
                outputs[task_name] = task_head(task_input, training=training)
        else:
            # Shared input for all tasks
            for task_name, task_head in self.task_heads.items():
                outputs[task_name] = task_head(inputs, training=training)

        return outputs

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
    ) -> Dict[str, Any]:
        """
        Report each task head's output shape.

        Delegates to every head's own ``compute_output_shape``. A dict
        ``input_shape`` gives per-task shapes, falling back to its
        ``'shared'`` entry. A single shape is used for every task.

        :param input_shape: One shared feature-map shape, or a dict of
            per-task shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        :return: Dict mapping each task name to that head's output shape.
        :rtype: Dict[str, Any]
        """
        output_shapes = {}
        for task_name, task_head in self.task_heads.items():
            if isinstance(input_shape, dict):
                task_input_shape = self._task_input(input_shape, task_name)
            else:
                task_input_shape = input_shape
            output_shapes[task_name] = task_head.compute_output_shape(task_input_shape)
        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        ``task_configs`` is the copy taken at construction, except that a
        :class:`VisionTaskType` ``task_type`` becomes its string value.
        Nothing here mutates the caller's dicts, so a round trip rebuilds the
        same heads.

        :return: Config dict carrying ``task_configs``,
            ``shared_backbone_dim`` and ``use_task_specific_attention``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        # DECISION plan-2026-10-14T062616-81938e81/D-027
        # Store each task_type as its string value. Do NOT store the caller's
        # VisionTaskType member: an Enum is not JSON-serializable, so saving
        # any model holding this head raised TypeError. _create_task_heads
        # parses the string back. See decisions.md D-027.
        task_configs = {}
        for task_name, task_config in self.task_configs.items():
            task_config = dict(task_config)
            task_type = task_config['task_type']
            if isinstance(task_type, VisionTaskType):
                task_config['task_type'] = task_type.value
            task_configs[task_name] = task_config
        config.update({
            'task_configs': task_configs,
            'shared_backbone_dim': self.shared_backbone_dim,
            'use_task_specific_attention': self.use_task_specific_attention
        })
        return config


def create_multi_task_head(
        task_configuration: Union[
            TaskConfiguration, Sequence[Union[VisionTaskType, str]], Dict[str, Dict]],
        **kwargs: Any
) -> MultiTaskHead:
    """
    Build a :class:`MultiTaskHead` from a task configuration.

    A :class:`TaskConfiguration` contributes its enabled tasks. A list or a
    tuple (the same semantics) contributes its entries, each a
    :class:`VisionTaskType` member or its string value; a task named twice is
    refused, because the heads are keyed by task and one would be dropped. A
    dict is used as the task-config mapping unchanged. In the first two cases
    each task's own options are taken from the ``kwargs`` entry named after
    the task's value (``detection={'num_classes': 80}``), and that entry is
    removed from ``kwargs``.

    The remaining ``kwargs`` go to :class:`MultiTaskHead`.

    :param task_configuration: A :class:`TaskConfiguration`, a list or tuple
        of :class:`VisionTaskType` members or strings, or a dict mapping task
        names to config dicts.
    :type task_configuration: Union[TaskConfiguration, Sequence[Union[VisionTaskType, str]], Dict[str, Dict]]
    :param kwargs: Per-task option dicts, keyed by task value (sequence and
        :class:`TaskConfiguration` forms only), plus any argument for
        :class:`MultiTaskHead`.
    :return: Configured multi-task head.
    :rtype: MultiTaskHead
    :raises ValueError: If ``task_configuration`` is none of the accepted
        types, a sequence entry is not a task, a sequence names a task twice,
        or (from :class:`MultiTaskHead`) the task set is empty.
    """

    if isinstance(task_configuration, TaskConfiguration):
        tasks = task_configuration.get_enabled_tasks()
    elif isinstance(task_configuration, (list, tuple)):
        tasks = [VisionTaskType.coerce(task) for task in task_configuration]
        values = [task.value for task in tasks]
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise ValueError(
                f"create_multi_task_head: task(s) {repeated} appear more than "
                "once; the heads are keyed by task, so all but one would be "
                "dropped. Name each task once."
            )
    elif isinstance(task_configuration, dict):
        tasks = None
        task_configs = task_configuration
    else:
        raise ValueError(
            "create_multi_task_head: task_configuration must be a "
            "TaskConfiguration or one of: list, tuple, dict; got "
            f"{type(task_configuration).__name__}."
        )

    if tasks is not None:
        # pop, not get: a per-task dict left in kwargs would also reach
        # MultiTaskHead(**kwargs), which refuses it as an unknown keyword, so
        # these forms could never take a per-task option.
        task_configs = {
            task.value: {'task_type': task, **kwargs.pop(task.value, {})}
            for task in tasks
        }

    return MultiTaskHead(task_configs=task_configs, **kwargs)
