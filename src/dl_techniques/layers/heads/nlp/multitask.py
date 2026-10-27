"""``MultiTaskNLPHead``: several NLP heads behind one layer."""

import keras
from typing import Dict, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import positive_int
from .task_types import NLPTaskConfig
from dl_techniques.utils.keras_registration import register_dl_technique
from .dispatch import get_head_class


@register_dl_technique("dl_techniques.layers.heads.nlp.multitask")
class MultiTaskNLPHead(keras.layers.Layer):
    """
    Several task heads behind one layer.

    One head is built per entry in ``task_configs``, using
    :func:`get_head_class` to pick the class. This layer does not inherit from
    :class:`BaseNLPHead`; it owns the sub-heads and routes to them.

    ``call`` takes an optional keyword-only ``task_name``. Given one, it runs
    that head alone and returns that head's own output dict. Given ``None``,
    it runs every head and returns a dict of those dicts, keyed by task name.
    ``compute_output_shape`` and the symbolic call follow ``task_name`` too.

    With ``use_task_specific_projections`` set, each task first goes through
    its own ``Dense`` projection, so the heads can work at different widths.

    The wrapper and every sub-head hold their own copies of the task configs
    (D-065): editing a config after construction changes nothing here. The
    caller's outer dict was never aliased (Keras stores a dict assigned to a
    layer as a ``TrackedDict`` copy, measured; review NLP-5 refuted for it).

    **Architecture Overview:**

    .. code-block:: text

        inputs (shared features), task_name
                        │
                 ┌──────┴──────────────────┐
           task_name given            task_name None
                 ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ known task?      │  │ loop every task  │
        │ else ValueError  │  │ in task_heads    │
        └────────┬─────────┘  └────────┬─────────┘
                 ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ task_projections │  │ task_projections │
        │ [name] (optional)│  │ [name] (optional)│
        └────────┬─────────┘  └────────┬─────────┘
                 ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ task_heads[name] │  │ task_heads[name] │
        └────────┬─────────┘  └────────┬─────────┘
                 ▼                     ▼
          that head's dict     {task: head dict}
                               one entry per task

    Input shape:
        Whatever the sub-heads accept. A tensor is passed to every head
        unchanged. A dict is passed through with its ``'hidden_states'``
        entry replaced when a projection applies.

    Output shape:
        The selected head's output dict, or a dict of every head's output dict
        keyed by task name.

    :param task_configs: Task name to :class:`NLPTaskConfig`, or to a dict of
        its fields (the form ``get_config`` writes). One head is built per
        entry. Must not be empty.
    :type task_configs: Dict[str, Union[NLPTaskConfig, Dict[str, Any]]]
    :param shared_input_dim: Width of the shared features coming in.
    :type shared_input_dim: int
    :param use_task_specific_projections: Whether each task gets its own
        Dense projection before its head.
    :type use_task_specific_projections: bool
    :param kwargs: Additional arguments for the base Layer class.

    :ivar task_heads: Task name to head layer. A dict of layers round-trips
        through ``.keras`` by value (measured; Keras tracks it by key).
    :vartype task_heads: Dict[str, BaseNLPHead]
    :ivar task_projections: Task name to projection layer. Empty when
        ``use_task_specific_projections`` is false.
    :vartype task_projections: Dict[str, keras.layers.Dense]

    :raises ValueError: If ``task_configs`` is empty or not a dict, from
        :func:`get_head_class` when a task type has no head, and from ``call``
        when ``task_name`` is not a known task.
    """

    def __init__(
            self,
            task_configs: Dict[str, Union[NLPTaskConfig, Dict[str, Any]]],
            shared_input_dim: int,
            use_task_specific_projections: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Build one head, and optionally one projection, per task.

        :param task_configs: Task name to :class:`NLPTaskConfig` or a dict of
            its fields.
        :type task_configs: Dict[str, Union[NLPTaskConfig, Dict[str, Any]]]
        :param shared_input_dim: Width of the shared features coming in.
        :type shared_input_dim: int
        :param use_task_specific_projections: Whether each task gets its own
            Dense projection before its head.
        :type use_task_specific_projections: bool
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_configs`` is empty or not a dict, or a
            task type has no implemented head.
        """
        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError(
                "MultiTaskNLPHead: task_configs must be a non-empty dict of task "
                f"name to NLPTaskConfig, got {task_configs!r}."
            )
        shared_input_dim = positive_int(
            shared_input_dim, 'shared_input_dim', 'MultiTaskNLPHead')
        super().__init__(**kwargs)

        # DECISION plan-2026-10-14T062616-81938e81/D-065
        # Copy every config (NLPTaskConfig.coerce). Do NOT store the caller's
        # objects: setting num_classes=7 on one after construction left the
        # built classifier at 3 units while get_config reported 7, so a rebuild
        # disagreed with the saved weights (review NLP-5, inner configs). See
        # decisions.md D-065.
        configs = {name: NLPTaskConfig.coerce(cfg) for name, cfg in task_configs.items()}
        self.task_configs = configs
        self.shared_input_dim = shared_input_dim
        self.use_task_specific_projections = use_task_specific_projections

        # Heads and projections, created in __init__ per the package rule.
        self.task_heads = {}
        self.task_projections = {}

        for task_name, task_config in configs.items():
            # Create appropriate head
            head_class = get_head_class(task_config.task_type)

            # Set input dimension
            if use_task_specific_projections:
                projection_dim = task_config.hidden_size or shared_input_dim
                self.task_projections[task_name] = keras.layers.Dense(
                    projection_dim,
                    name=f"{task_name}_projection"
                )
                input_dim = projection_dim
            else:
                input_dim = shared_input_dim

            # Create head
            self.task_heads[task_name] = head_class(
                task_config=task_config,
                input_dim=input_dim
            )

    def _head_input_shape(self, task_name: str, input_shape: Union[Tuple, Dict]) -> Union[Tuple, Dict]:
        """
        Return the input shape a task's head sees, after its projection.

        :param task_name: A configured task name.
        :type task_name: str
        :param input_shape: Shape of the shared input, or a dict of shapes.
        :type input_shape: Union[Tuple, Dict]
        :return: The same shape, with the feature width replaced by the
            projection's when projections are on.
        :rtype: Union[Tuple, Dict]
        """
        if not self.use_task_specific_projections:
            return input_shape
        width = self.task_configs[task_name].hidden_size or self.shared_input_dim
        if isinstance(input_shape, dict):
            hidden = tuple(input_shape['hidden_states'])
            return {**input_shape, 'hidden_states': hidden[:-1] + (width,)}
        return tuple(input_shape)[:-1] + (width,)

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build every projection, then every head.

        With projections on, each head is built for the width its own
        projection emits rather than for ``shared_input_dim``.

        :param input_shape: Shape of the shared input, or a dict of shapes
            carrying a ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        hidden_shape = (input_shape['hidden_states']
                        if isinstance(input_shape, dict) else input_shape)
        for projection in self.task_projections.values():
            projection.build(tuple(hidden_shape))

        for task_name, head in self.task_heads.items():
            head.build(self._head_input_shape(task_name, input_shape))

        super().build(input_shape)

    def _selected_heads(self, task_name: Optional[str]) -> Dict[str, keras.layers.Layer]:
        """
        Return the heads a call runs: one for ``task_name``, else all of them.

        :param task_name: A configured task name, or ``None``.
        :type task_name: Optional[str]
        :return: Task name to head.
        :rtype: Dict[str, keras.layers.Layer]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        if task_name is None:
            return dict(self.task_heads)
        if task_name not in self.task_heads:
            raise ValueError(
                f"Unknown task: {task_name!r}. Configured tasks: "
                f"{sorted(self.task_heads)}."
            )
        return {task_name: self.task_heads[task_name]}

    def _project(self, task_name: str, inputs: Any, symbolic: bool = False) -> Any:
        """
        Apply a task's projection to the shared input, if projections are on.

        :param task_name: A configured task name.
        :type task_name: str
        :param inputs: The shared input, a tensor or a dict.
        :type inputs: Any
        :param symbolic: Use the projection's ``compute_output_spec`` (for a
            symbolic call) instead of calling it.
        :type symbolic: bool
        :return: The input the task's head takes.
        :rtype: Any
        """
        if not self.use_task_specific_projections:
            return inputs
        projection = self.task_projections[task_name]
        apply = projection.compute_output_spec if symbolic else projection
        if isinstance(inputs, dict):
            return {**inputs, 'hidden_states': apply(inputs['hidden_states'])}
        return apply(inputs)

    def compute_output_shape(
            self,
            input_shape: Union[Tuple, Dict],
            *,
            task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the output shapes ``call`` returns for the same ``task_name``.

        :param input_shape: Shape of the shared input, or a dict of shapes
            carrying a ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output shapes, or task name to output shapes.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        shapes = {
            name: head.compute_output_shape(self._head_input_shape(name, input_shape))
            for name, head in self._selected_heads(task_name).items()
        }
        return shapes[task_name] if task_name is not None else shapes

    def compute_output_spec(
            self,
            inputs: Any,
            training: Optional[bool] = None,
            *,
            task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the symbolic outputs ``call`` returns for the same ``task_name``.

        Keras' default routes a symbolic call through ``compute_output_shape``
        with the input shapes only, so ``task_name`` never reached it and a
        symbolic call returned every task's outputs (the D-060 defect of the
        VLM wrapper). Each selected head reports its own spec instead.

        :param inputs: Symbolic shared inputs.
        :type inputs: Any
        :param training: Unused; accepted because ``call`` takes it.
        :type training: Optional[bool]
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output spec, or task name to output spec.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        specs = {
            name: head.compute_output_spec(self._project(name, inputs, symbolic=True))
            for name, head in self._selected_heads(task_name).items()
        }
        return specs[task_name] if task_name is not None else specs

    # DECISION plan-2026-10-14T062616-81938e81/D-065
    # task_name is keyword-only and training keeps the second positional slot,
    # as in MultiTaskVLMHead (D-060). Do NOT move task_name back in front of
    # training: a positional training flag landed in task_name. See
    # decisions.md D-065.
    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None,
            *,
            task_name: Optional[str] = None,
    ) -> Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]:
        """
        Run one task head, or all of them.

        :param inputs: Shared features from the foundation model, as a tensor
            or as a dict with ``'hidden_states'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :param task_name: Keyword only. Which task to run; ``None`` runs every
            task.
        :type task_name: Optional[str]
        :return: One head's output dict when ``task_name`` is given. A dict of
            every head's output dict, keyed by task name, when it is ``None``.
        :rtype: Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]
        :raises ValueError: If ``task_name`` is not a known task.
        """
        outputs = {
            name: head(self._project(name, inputs), training=training)
            for name, head in self._selected_heads(task_name).items()
        }
        return outputs[task_name] if task_name is not None else outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        Each :class:`NLPTaskConfig` is written as a plain dict of every field
        (:meth:`NLPTaskConfig.to_dict`). There is no ``from_config`` override:
        ``__init__`` takes those dicts as they are, so the caller's config is
        never modified.

        :return: Config dict accepted by ``from_config``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'task_configs': {
                name: tc.to_dict() for name, tc in self.task_configs.items()
            },
            'shared_input_dim': self.shared_input_dim,
            'use_task_specific_projections': self.use_task_specific_projections,
        })
        return config
